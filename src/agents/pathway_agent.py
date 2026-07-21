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
2. Does bypass_check show bypass_exists=true for any pathway? Name the bypass gene if so.
3. If bypass found in database → vote RESISTANT (the tumour has an escape route around the drug's target).
4. If target gene IS present in pathway_membership AND bypass_exists=false → vote SENSITIVE.
   Rationale: the target is embedded in a known oncogenic pathway and no escape route exists in the
   database, so blocking it should suppress signalling.
5. If target gene is NOT found in any pathway (pathway_membership is empty) → vote UNCERTAIN.
   Rationale: we cannot assess escape routes for targets outside our pathway database.
6. Conclude with your verdict and why, referencing only the data provided.

SELF-ATTESTATION (required — answer these 4 questions about your evidence before stating your verdict):
1. bypass_gene_expressed: Did you find at least one bypass gene present at z>=1.0 in this cell line?
2. pathway_active: Is the relevant pathway's activity score >= 0.4 (>=40% of member genes expressed at z>=0.5)?
3. mechanism_relevant: Does the bypass gene's pathway mechanistically connect to resistance against this drug class?
4. bypass_distinct: Is the bypass gene different from the primary drug target gene?

Add a "self_attestation" field to your JSON response:
{"bypass_gene_expressed": true/false, "pathway_active": true/false, "mechanism_relevant": true/false, "bypass_distinct": true/false, "score": <int 0-4, sum of true answers>}

Your verdict can only be RESISTANT (bypass found) if self_attestation score >= 3.
If score < 3, set verdict to UNCERTAIN even if bypass_exists is true in the data.

Return ONLY a JSON object matching the EvidencePack schema PLUS the self_attestation field. No prose outside the JSON."""


class PathwayAgent(BaseAgent):
    agent_id = "pathway_agent"

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

    async def _fetch_evidence(self, cell_line: str, drug: str, target_genes: list[str] | None = None) -> dict:
        bypass_results = {}
        pathway_membership = {}
        pathway_genes_cache: dict[str, list[str]] = {}

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

                # Pathway activity gate: skip if < 40% of pathway genes are expressed (z >= 0.5)
                # This filters out pathways not operating in this cancer context
                if expressed_05:
                    if pid not in pathway_genes_cache:
                        pw_genes_data = await self._call_tool("get_pathway_genes", {"pathway_id": pid})
                        pathway_genes_cache[pid] = [
                            g["gene"] for g in (pw_genes_data if isinstance(pw_genes_data, list) else [])
                        ]
                    pw_genes = pathway_genes_cache[pid]
                    if pw_genes:
                        activity_score = sum(1 for g in pw_genes if g in expressed_05) / len(pw_genes)
                        if activity_score < 0.4:
                            continue

                result = await self._call_tool("check_bypass", {"pathway_id": pid, "blocked_gene": gene})
                if result and result.get("bypass_exists"):
                    bypass_genes = result.get("bypass_genes", [])
                    # Filter out bypass genes that are themselves drug targets (e.g. MAP2K1↔MAP2K2 for trametinib)
                    real_bypasses = [b for b in bypass_genes if b not in (target_genes or [])]
                    if not real_bypasses:
                        continue
                    # Bypass gene must be notably upregulated (z >= 1.0) — not just expressed
                    if expressed_10:
                        expressed = [b for b in real_bypasses if b in expressed_10]
                    else:
                        expressed = real_bypasses  # fail open if no expression data
                    if expressed:
                        bypass_results[f"{pid}:{gene}"] = {**result, "bypass_genes": expressed}

        return {
            "pathway_membership": pathway_membership,
            "bypass_check": bypass_results if bypass_results else {"bypass_exists": False, "note": "No bypass routes found in database for these targets"},
        }

    def _compute_signal(self, evidence: dict) -> float:
        bypass_check = evidence.get("bypass_check", {})
        has_bypass = False
        if isinstance(bypass_check, dict):
            has_bypass = bypass_check.get("bypass_exists", False) or any(
                v.get("bypass_exists") for v in bypass_check.values() if isinstance(v, dict)
            )
        if has_bypass:
            return 1.0  # bypass found → RESISTANT signal
        # Target in pathway but no bypass → moderate SENSITIVE signal
        if evidence.get("pathway_membership"):
            return 0.6
        return 0.0  # target not in any pathway → no signal
