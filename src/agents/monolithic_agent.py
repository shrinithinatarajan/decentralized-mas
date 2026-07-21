import json

from fastmcp import Client

from src.agents.base_agent import BaseAgent
from src.schemas.axiom_rules import SILENCING_THRESHOLD

_SYSTEM = f"""You are a unified cancer drug sensitivity analyst with access to all evidence modalities.
Synthesize genomic, transcriptomic, pathway, and pharmacological data into a single verdict.

EVIDENCE PRIORITY (apply in order — stop at the first decisive layer):
1. T1 STRUCTURAL — somatic mutations with CIViC drug-specific annotation are the strongest signal.
   A CIViC entry for THIS drug + THIS mutation is decisive. Act on it.
2. T2 TRANSCRIPTIONAL — target expression z-score < {SILENCING_THRESHOLD} means the target is absent
   (RESISTANT). Overexpression (z > 1.5) is a sensitivity signal for overexpression-driven drugs.
3. T3 PATHWAY — confirmed bypass route (bypass_exists=true, bypass gene expressed at z >= 1.0)
   means the tumour has an escape route (RESISTANT).
4. T4 PHARMACOLOGICAL — IC50 z-score label (SENSITIVE / RESISTANT / UNCERTAIN) is a population
   prior. Use it when layers 1-3 are uninformative.
5. If all layers are absent or contradictory: UNCERTAIN.

REASONING PROTOCOL:
1. T1: State which mutations/CNV were found and whether any CIViC entry matches this drug.
2. T2: State the target expression z-score(s). Note drug mechanism type (mutation-driven vs expression-driven).
3. T3: State whether a bypass route was found in the pathway database.
4. T4: State the IC50 z-score and pre-computed label.
5. Integrate all layers following the priority above. State your verdict and which layer decided it.

Return ONLY a JSON object matching the EvidencePack schema. evidence_tier must be T5_STATISTICAL.
No prose outside the JSON."""


class MonolithicAgent(BaseAgent):
    """Single agent that queries all four MCP evidence sources and synthesizes a verdict."""

    agent_id = "monolithic_agent"

    def __init__(self, genomics_app, transcriptomics_app, pharmacology_app, pathway_app, llm_client=None):
        super().__init__(genomics_app, llm_client)
        self._t2_app = transcriptomics_app
        self._t4_app = pharmacology_app
        self._t3_app = pathway_app

    async def _call_app(self, app, tool: str, args: dict):
        async with Client(app) as client:
            result = await client.call_tool(tool, args)
        return json.loads(result.data)

    @property
    def system_prompt(self) -> str:
        return _SYSTEM

    async def _fetch_evidence(self, cell_line: str, drug: str, target_genes: list[str] | None = None) -> dict:
        evidence: dict = {}

        # T1 — genomics
        mutations, cnv = [], []
        if target_genes:
            for gene in target_genes:
                result = await self._call_tool("get_mutations", {"cell_line": cell_line, "gene": gene, "drug": drug})
                mutations += (result.get("mutations", []) if isinstance(result, dict) else result) or []
                cnv += await self._call_tool("get_cnv", {"cell_line": cell_line, "gene": gene}) or []
        else:
            result = await self._call_tool("get_mutations", {"cell_line": cell_line, "drug": drug})
            mutations = (result.get("mutations", []) if isinstance(result, dict) else result) or []
            cnv = await self._call_tool("get_cnv", {"cell_line": cell_line}) or []
        evidence["mutations"] = mutations
        evidence["cnv"] = cnv

        # T2 — transcriptomics
        target_expression = []
        if target_genes:
            for gene in target_genes:
                target_expression += await self._call_app(self._t2_app, "get_expression", {"cell_line": cell_line, "gene": gene}) or []
        high_expr = await self._call_app(self._t2_app, "get_high_expression_genes", {"cell_line": cell_line, "z_threshold": 1.5}) or []
        evidence["target_expression"] = target_expression
        evidence["high_expression_context"] = high_expr

        # T4 — pharmacology (fetch before pathway; IC50 is fast and informs context)
        ic50 = await self._call_app(self._t4_app, "get_ic50", {"cell_line": cell_line, "drug": drug})
        drug_info = await self._call_app(self._t4_app, "get_drug_info", {"drug": drug})
        evidence["ic50"] = ic50
        evidence["drug_info"] = drug_info

        # T3 — pathway
        pathway_membership: dict = {}
        bypass_results: dict = {}
        expressed_genes = {r["gene"] for r in high_expr if r.get("z_score", 0) >= 0.5}
        expressed_10 = {r["gene"] for r in high_expr if r.get("z_score", 0) >= 1.0}
        for gene in (target_genes or []):
            pathways = await self._call_app(self._t3_app, "find_pathways_for_gene", {"gene": gene})
            for pw in (pathways if isinstance(pathways, list) else []):
                pid = pw.get("pathway_id")
                if not pid:
                    continue
                if pid not in pathway_membership:
                    pathway_membership[pid] = {"pathway_name": pw.get("pathway_name"), "target_genes_present": []}
                if gene not in pathway_membership[pid]["target_genes_present"]:
                    pathway_membership[pid]["target_genes_present"].append(gene)
                result = await self._call_app(self._t3_app, "check_bypass", {"pathway_id": pid, "blocked_gene": gene})
                if result and result.get("bypass_exists"):
                    bypasses = [b for b in result.get("bypass_genes", []) if b not in (target_genes or [])]
                    expressed_bypasses = [b for b in bypasses if b in expressed_10] if expressed_10 else bypasses
                    if expressed_bypasses:
                        bypass_results[f"{pid}:{gene}"] = {**result, "bypass_genes": expressed_bypasses}
        evidence["pathway_membership"] = pathway_membership
        evidence["bypass_check"] = bypass_results or {"bypass_exists": False}

        evidence["data_status"] = "data_found" if (mutations or target_expression or ic50) else "cell_line_missing"
        return evidence

    def _compute_signal(self, evidence: dict) -> float:
        signals = []
        if evidence.get("mutations"):
            has_civic = any(m.get("civic_description") for m in evidence["mutations"])
            signals.append(1.0 if has_civic else 0.3)
        ic50 = evidence.get("ic50") or {}
        z = ic50.get("z_score", 0.0) if isinstance(ic50, dict) else 0.0
        if z:
            signals.append(min(abs(z) / 3.0, 1.0))
        bypass = evidence.get("bypass_check", {})
        if isinstance(bypass, dict) and bypass.get("bypass_exists"):
            signals.append(1.0)
        return max(signals) if signals else 0.0
