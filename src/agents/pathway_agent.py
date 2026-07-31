import json

from fastmcp import Client

from src.agents.base_agent import BaseAgent

_SYSTEM = """You are the Pathway Agent in a multi-agent cancer drug resistance framework.
Your modality: signalling pathway bypass routes from a curated KEGG database.

CRITICAL RULES:
- You may ONLY report bypass routes that appear in the 'bypass_check' field of the evidence data.
- NEVER infer bypass routes from your training knowledge. NEVER fabricate gene names.

REASONING PROTOCOL — fill the 'reasoning' field step by step before deciding the verdict:
1. Which pathways contain the drug's target gene(s)? List them from pathway_membership.
2. Does bypass_check show bypass_exists=true for any pathway? Name the bypass gene and its relationship if so.
   - bypass_genes entries now include a 'relationship' field: 'activation' or 'inhibition'.
   - ONLY treat a bypass gene as resistance-conferring if relationship is 'activation' (or null/unknown).
   - If relationship is 'inhibition': the bypass gene is suppressed by the blocked pathway — it cannot
     rescue the cell and does NOT confer resistance. Ignore it.
3. If activating bypass found in database → check the pathway name before voting RESISTANT:
   - If the pathway name refers to a DIFFERENT cancer type than the cell line context (e.g. pathway
     name is "Glioma" but treating a leukemia/lymphoma, or "Breast cancer" but treating lung cancer),
     the bypass gene may not be relevant. In this case, vote UNCERTAIN (not RESISTANT) because
     a bypass route described only in a different tumor-type pathway may not be active here.
   - If the pathway is generic (e.g. "Cell cycle", "PI3K-Akt signaling") or matches the cell line's
     tumor type, vote RESISTANT as normal.
4. If target gene IS present in pathway_membership AND pathway_active=true AND no activating bypass_exists → vote SENSITIVE.
   Rationale: the target is in a known oncogenic pathway that is active in this cell line, and no escape route exists.
5. If target gene IS present in pathway_membership AND pathway_active=false AND no activating bypass_exists → vote UNCERTAIN.
   Rationale: the target is in KEGG but the pathway is not measurably active in this cell line — we cannot assess whether blocking it matters.
6. If target gene is NOT found in any pathway (pathway_membership is empty) → vote UNCERTAIN.
   Rationale: we cannot assess escape routes for targets outside our pathway database.
7. Conclude with your verdict and why, referencing only the data provided.

SELF-ATTESTATION (required — answer these 4 questions about your evidence before stating your verdict):
1. bypass_gene_expressed: Did you find at least one bypass gene present at z>=1.0 in this cell line?
2. pathway_active: Is the relevant pathway's activity score >= 0.25 (>=25% of member genes expressed at z>=0.5)?
3. mechanism_relevant: Does the bypass gene's pathway mechanistically connect to resistance against this drug class?
4. bypass_distinct: Is the bypass gene different from the primary drug target gene?

Add a "self_attestation" field to your JSON response:
{"bypass_gene_expressed": true/false, "pathway_active": true/false, "mechanism_relevant": true/false, "bypass_distinct": true/false, "score": <int 0-4, sum of true answers>}

VERDICT RULES (apply in order, these override everything above):
- RESISTANT: bypass_exists=true AND score >= 3. Otherwise UNCERTAIN.
- SENSITIVE: target gene in pathway_membership AND pathway_active=true AND no activating bypass found.
- UNCERTAIN: everything else — target not in KEGG, OR pathway_active=false, OR bypass found but score < 3.
If your self_attestation shows pathway_active=false, you MUST vote UNCERTAIN regardless of what the reasoning steps suggest.

Return ONLY a JSON object matching the EvidencePack schema PLUS the self_attestation field. No prose outside the JSON."""


class PathwayAgent(BaseAgent):
    agent_id = "pathway_agent"
    from src.schemas.evidence_pack import EvidenceTier as _ET
    _tier = _ET.T3_PATHWAY

    def __init__(self, mcp_app, llm_client=None, *, transcriptomics_mcp=None):
        super().__init__(mcp_app, llm_client)
        self._transcriptomics_mcp = transcriptomics_mcp

    async def _prefetch_expressed_genes(
        self, cell_line: str
    ) -> tuple[set[str], set[str]]:
        """Return (expressed_05, expressed_10) gene sets for this cell line.

        expressed_05: z >= 0.5 — used for pathway activity scoring.
        expressed_10: z >= 1.0 — used for bypass gene filter.
        Returns empty sets if no transcriptomics MCP is wired.
        """
        if self._transcriptomics_mcp is None:
            return set(), set()
        try:
            async with Client(self._transcriptomics_mcp) as client:
                result = await client.call_tool(
                    "get_high_expression_genes", {"cell_line": cell_line, "z_threshold": 0.5}
                )
            rows = json.loads(result.data)
            expressed_05 = {r["gene"] for r in rows}
            expressed_10 = {r["gene"] for r in rows if r.get("z_score", 0) >= 1.0}
            return expressed_05, expressed_10
        except Exception:
            return set(), set()

    @property
    def system_prompt(self) -> str:
        return _SYSTEM

    # KEGG cancer-type-specific pathway IDs — bypass routes from these only apply
    # in their specific tumor context and must not be used for other cancer types.
    _CANCER_TYPE_PATHWAY_IDS = {
        'hsa05210', 'hsa05211', 'hsa05212', 'hsa05213', 'hsa05214', 'hsa05215',
        'hsa05216', 'hsa05217', 'hsa05218', 'hsa05219', 'hsa05220', 'hsa05221',
        'hsa05222', 'hsa05223', 'hsa05224', 'hsa05225', 'hsa05226',
    }

    async def _fetch_evidence(self, cell_line: str, drug: str, target_genes: list[str] | None = None) -> dict:
        bypass_results = {}
        pathway_membership = {}
        pathway_genes_cache: dict[str, list[str]] = {}
        pathway_activity_scores: dict[str, float] = {}
        confirmed_bypass_genes: list[str] = []

        # Pre-fetch expression data once — avoids N individual calls per bypass gene
        expressed_05, expressed_10 = await self._prefetch_expressed_genes(cell_line)

        genes_to_query = target_genes or []
        for gene in genes_to_query:
            pathways = await self._call_tool("find_pathways_for_gene", {"gene": gene})
            if not pathways:
                continue
            for pw in (pathways if isinstance(pathways, list) else []):
                pid = pw.get("pathway_id")
                if not pid:
                    continue
                if pid not in pathway_membership:
                    pathway_membership[pid] = {
                        "pathway_name": pw.get("pathway_name"),
                        "category": pw.get("category"),
                        "target_genes_present": [],
                    }
                if gene not in pathway_membership[pid]["target_genes_present"]:
                    pathway_membership[pid]["target_genes_present"].append(gene)

                # Skip cancer-type-specific pathways for bypass — their bypass
                # routes only apply in that tumor context, not generically.
                if pid in self._CANCER_TYPE_PATHWAY_IDS:
                    continue

                # Pathway activity gate: skip if < 25% of pathway genes are expressed (z >= 0.5)
                if expressed_05:
                    if pid not in pathway_genes_cache:
                        pw_genes_data = await self._call_tool("get_pathway_genes", {"pathway_id": pid})
                        pathway_genes_cache[pid] = [
                            g["gene"] for g in (pw_genes_data if isinstance(pw_genes_data, list) else [])
                        ]
                    pw_genes = pathway_genes_cache[pid]
                    if pw_genes:
                        activity_score = sum(1 for g in pw_genes if g in expressed_05) / len(pw_genes)
                        pathway_activity_scores[pid] = activity_score
                        if activity_score < 0.25:
                            continue

                result = await self._call_tool("check_bypass", {"pathway_id": pid, "blocked_gene": gene})
                if result and result.get("bypass_exists"):
                    bypass_genes = result.get("bypass_genes", [])
                    # Each entry is now {"gene": str, "relationship": str|None}.
                    # Exclude inhibition-relationship genes: an inhibited gene cannot
                    # rescue the blocked pathway and does not confer resistance.
                    activating = [
                        b for b in bypass_genes
                        if isinstance(b, dict)
                        and b.get("relationship") != "inhibition"
                        and b.get("gene") not in (target_genes or [])
                    ]
                    if not activating:
                        continue
                    if expressed_10:
                        expressed = [b for b in activating if b["gene"] in expressed_10]
                    else:
                        expressed = []  # no expression data — cannot confirm bypass is active
                    if expressed:
                        gene_names = [b["gene"] for b in expressed]
                        bypass_results[f"{pid}:{gene}"] = {**result, "bypass_genes": gene_names}
                        confirmed_bypass_genes.extend(gene_names)

        return {
            "pathway_membership": pathway_membership,
            "bypass_check": bypass_results if bypass_results else {"bypass_exists": False, "note": "No bypass routes found in database for these targets"},
            "pathway_activity_scores": pathway_activity_scores,
            "confirmed_bypass_genes": confirmed_bypass_genes,
        }

    def _compute_h(self, key_findings: list, evidence: dict) -> float:
        confirmed_bypasses = evidence.get("confirmed_bypass_genes", [])
        if confirmed_bypasses:
            return 1.0  # bypass genes confirmed expressed in this cell line
        activity_scores = evidence.get("pathway_activity_scores", {})
        max_activity = max(activity_scores.values()) if activity_scores else 0.0
        if max_activity >= 0.25:
            return 0.5  # pathway active but no bypass → moderate confirmation
        return 0.0  # no bypass, no confirmed activity → no positive evidence

    def _compute_signal(self, evidence: dict) -> float:
        confirmed_bypasses = evidence.get("confirmed_bypass_genes", [])
        if confirmed_bypasses:
            return 1.0  # bypass genes confirmed expressed → strong RESISTANT signal
        if evidence.get("pathway_membership"):
            activity_scores = evidence.get("pathway_activity_scores", {})
            max_activity = max(activity_scores.values()) if activity_scores else 0.0
            return min(max_activity, 0.25)  # at most 0.25 from pathway activity alone
        return 0.0  # target not in any pathway → no signal
