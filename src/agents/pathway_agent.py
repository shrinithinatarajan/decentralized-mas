from src.agents.base_agent import BaseAgent

_PATHWAYS = ["hsa04010", "hsa04012", "hsa04151", "hsa04115", "hsa04310"]

_SYSTEM = """You are the Pathway Agent in a multi-agent cancer drug resistance framework.
Your modality: signalling pathway bypass routes from a curated KEGG database.

CRITICAL RULES:
- You may ONLY report bypass routes that appear in the 'bypass_check' field of the evidence data.
- If bypass_check shows bypass_exists=false or has no entries, you MUST return verdict=UNCERTAIN.
- NEVER infer bypass routes from your training knowledge. NEVER fabricate gene names.
- If you have no database evidence of an active bypass, do not vote RESISTANT.

Apply the T3_PATHWAY_BYPASS axiom ONLY when bypass_exists=true in the provided data.
Return ONLY a JSON object matching the EvidencePack schema. No prose."""


class PathwayAgent(BaseAgent):
    agent_id = "pathway_agent"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM

    async def _fetch_evidence(self, cell_line: str, drug: str, target_genes: list[str] | None = None) -> dict:
        bypass_results = {}
        pathway_membership = {}

        for pid in _PATHWAYS:
            genes = await self._call_tool("get_pathway_genes", {"pathway_id": pid})
            if not genes:
                continue
            gene_list = genes if isinstance(genes, list) else []

            if target_genes:
                present = [g for g in gene_list if g in target_genes]
                if present:
                    pathway_membership[pid] = {"target_genes_present": present}
                    # Check database for actual bypass routes for each target gene
                    for gene in present:
                        result = await self._call_tool("check_bypass", {"pathway_id": pid, "blocked_gene": gene})
                        if result and result.get("bypass_exists"):
                            bypass_results[f"{pid}:{gene}"] = result

        return {
            "pathway_membership": pathway_membership,
            "bypass_check": bypass_results if bypass_results else {"bypass_exists": False, "note": "No bypass routes found in database for these targets"},
        }
