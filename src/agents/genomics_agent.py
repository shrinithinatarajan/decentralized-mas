from src.agents.base_agent import BaseAgent

_SYSTEM = """You are the Genomics Agent in a multi-agent cancer drug resistance framework.
Your modality: somatic mutations and copy number variation (DNA level).
Apply the T1_STRUCTURAL axiom: structural DNA alterations are the highest-priority evidence.

REASONING PROTOCOL — fill the 'reasoning' field step by step before deciding the verdict:
1. What mutations or CNV did the MCP data return for the drug's target gene(s)?
2. If a driver mutation is present in the drug target: this strongly predicts SENSITIVE.
3. If the gene is WILD-TYPE (in DB, no mutation): DO NOT default to RESISTANT.
   Wild-type status means the gene is not mutated, but the drug may still work through
   expression-level activity, pathway context, or IC50 history. Prefer UNCERTAIN unless
   other evidence (e.g. wild_type_note says the target requires mutation for activation).
4. If the cell line is not in the database at all: state no data and lean UNCERTAIN.
5. Weigh CNV: amplification of an oncogene target → sensitivity signal; deletion → resistance.
6. Conclude with your verdict and why.

Return ONLY a JSON object matching the EvidencePack schema. No prose outside the JSON."""


class GenomicsAgent(BaseAgent):
    agent_id = "genomics_agent"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM

    async def _fetch_evidence(self, cell_line: str, drug: str, target_genes: list[str] | None = None) -> dict:
        mutations, cnv, wild_type_genes = [], [], []
        if target_genes:
            for gene in target_genes:
                result = await self._call_tool("get_mutations", {"cell_line": cell_line, "gene": gene})
                if isinstance(result, dict):
                    muts = result.get("mutations", [])
                    if not muts and result.get("cell_line_in_db"):
                        wild_type_genes.append(gene)
                    mutations += muts
                else:
                    mutations += result or []
                cnv += await self._call_tool("get_cnv", {"cell_line": cell_line, "gene": gene}) or []
        else:
            result = await self._call_tool("get_mutations", {"cell_line": cell_line})
            mutations = (result.get("mutations", []) if isinstance(result, dict) else result) or []
            cnv = await self._call_tool("get_cnv", {"cell_line": cell_line}) or []
        evidence: dict = {"mutations": mutations, "cnv": cnv}
        if wild_type_genes:
            evidence["wild_type_genes"] = wild_type_genes
            evidence["wild_type_note"] = (
                f"The following target genes have no somatic point mutations in {cell_line}: "
                f"{wild_type_genes}. "
                "IMPORTANT CAVEAT: CCLE point-mutation data does NOT capture chromosomal "
                "fusions or rearrangements (e.g. BCR-ABL, EML4-ALK, MET exon 14 skipping). "
                "No point mutation found does NOT rule out fusion-driven sensitivity. "
                "Prefer UNCERTAIN unless there is positive evidence of resistance "
                "(e.g. bypass pathway mutations, target gene deletion, or known "
                "resistance alteration in this cell line)."
            )
        return evidence

    def _compute_signal(self, evidence: dict) -> float:
        mutations = evidence.get("mutations", [])
        if not mutations:
            return 0.0
        return 1.0 if any(m.get("is_driver") for m in mutations) else 0.5
