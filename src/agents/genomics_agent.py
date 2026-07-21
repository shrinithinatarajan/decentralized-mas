from src.agents.base_agent import BaseAgent

_SYSTEM = """You are the Genomics Agent in a multi-agent cancer drug resistance framework.
Your modality: somatic mutations and copy number variation (DNA level).
Apply the T1_STRUCTURAL axiom: structural DNA alterations are the highest-priority evidence.

REASONING PROTOCOL — fill the 'reasoning' field step by step before deciding the verdict:
1. What mutations or CNV did the MCP data return for the drug's target gene(s)?
2. CRITICAL — CHECK CIViC DESCRIPTIONS FIRST: If any mutation has a civic_description,
   read it carefully for the SPECIFIC DRUG being tested (not other drugs mentioned).
   - If the description mentions resistance to THIS drug → vote RESISTANT.
   - If the description mentions sensitivity to THIS drug → vote SENSITIVE.
   - CIViC is drug-specific: a mutation may confer resistance to drug A and sensitivity
     to drug B (e.g. EGFR T790M → resistant to afatinib, sensitive to osimertinib).
   - Match drug names flexibly (e.g. 'afatinib' matches 'Afatinib', 'BIBW2992').
   - If the description is ambiguous or does not mention THIS drug, proceed to step 3.
3. If a driver mutation is present but no CIViC context for this drug: vote UNCERTAIN.
   A mutation in the target gene does not tell you whether the drug binds better or worse
   without knowing the mutation's functional consequence for THIS drug. Do NOT default to
   SENSITIVE. State that CIViC evidence is absent and the functional consequence is unknown.
   Exception: CNV amplification of the target oncogene is a sensitivity signal even without
   CIViC (gene amplification → drug target overexpressed → drug likely effective).
4. If the gene is WILD-TYPE (in DB, no mutation): DO NOT default to RESISTANT.
   Wild-type status means the gene is not mutated, but the drug may still work through
   expression-level activity, pathway context, or IC50 history. Prefer UNCERTAIN unless
   other evidence (e.g. wild_type_note says the target requires mutation for activation).
5. If the cell line is not in the database at all: state no data and lean UNCERTAIN.
6. Weigh CNV carefully by status field:
   - Amplification (cnv_value > 2, status='amplification') → sensitivity signal: target overexpressed.
   - HOMOZYGOUS deletion (status='homozygous_deletion') → RESISTANT: both copies gone, drug has no target.
   - HEMIZYGOUS deletion (status='hemizygous_deletion') → UNCERTAIN: one copy remains, gene still expressed at reduced level. Do NOT vote RESISTANT on hemizygous deletion alone — the target is still present.
7. Conclude with your verdict and why, explicitly citing any CIViC description used.

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
                result = await self._call_tool("get_mutations", {"cell_line": cell_line, "gene": gene, "drug": drug})
                if isinstance(result, dict):
                    muts = result.get("mutations", [])
                    if not muts and result.get("cell_line_in_db"):
                        wild_type_genes.append(gene)
                    mutations += muts
                else:
                    mutations += result or []
                cnv += await self._call_tool("get_cnv", {"cell_line": cell_line, "gene": gene}) or []
        else:
            result = await self._call_tool("get_mutations", {"cell_line": cell_line, "drug": drug})
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
        if not target_genes:
            evidence["data_status"] = "insufficient_evidence"
        elif mutations:
            evidence["data_status"] = "data_found"
        elif wild_type_genes:
            evidence["data_status"] = "target_wild_type"
        else:
            evidence["data_status"] = "cell_line_missing"
        return evidence

    def _compute_signal(self, evidence: dict) -> float:
        mutations = evidence.get("mutations", [])
        if not mutations:
            return 0.0
        has_civic   = any(m.get("civic_description") for m in mutations)
        has_driver  = any(m.get("is_driver") for m in mutations)
        has_cnv_amp = any(c.get("cnv_value", 0) > 1 for c in evidence.get("cnv", []))
        if has_civic:
            return 1.0   # drug-specific CIViC evidence → strong, decisive T1
        if has_cnv_amp:
            return 0.75  # amplification is a sensitivity signal even without CIViC
        if has_driver:
            return 0.3   # driver mutation but no drug-specific context → agent votes UNCERTAIN
        return 0.2       # VUS / passenger
