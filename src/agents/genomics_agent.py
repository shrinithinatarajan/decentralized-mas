from src.agents.base_agent import BaseAgent

_SYSTEM = """You are the Genomics Agent in a multi-agent cancer drug resistance framework.
Your modality: somatic mutations, copy number variation (DNA level), and CRISPR gene essentiality (DepMap).
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
3. DepMap CRISPR Chronos supplement: if depmap_scores are provided, use them as supporting evidence.
   - Chronos ≤ -1.0: strong essentiality — gene is critical for cell survival → SENSITIVE signal.
   - Chronos -0.5 to -1.0: borderline — vote SENSITIVE with confidence capped at 0.55. It is weak evidence, but still a signal: when it is the only evidence available (no CIViC, no IC50 data from T4), use it. Do NOT use it to override a stronger IC50 RESISTANT verdict.
   - Chronos > -0.5: gene is non-essential in this cell line → not informative for sensitivity.
   - Only use as a tiebreaker or to support an UNCERTAIN call — do NOT override a strong CIViC verdict.
   - Example: no CIViC evidence + wild-type + Chronos = -1.2 → lean SENSITIVE with moderate confidence.
   - CRITICAL CAVEAT: Chronos measures CRISPR gene KNOCKOUT essentiality — this is NOT the same as
     pharmacological drug sensitivity. A low Chronos score means the cell cannot survive WITHOUT the
     gene (KO lethal), but a drug inhibitor may still fail to achieve sufficient target inhibition at
     therapeutic concentrations. When T4 IC50 data contradicts Chronos (e.g. Chronos ≤ -1.0 but
     z_score > 0.5 RESISTANT), prefer the IC50 verdict — it is a direct drug measurement.
     Cap confidence at 0.65 when Chronos is the sole basis for a SENSITIVE verdict.
   OpenTargets fallback: if opentargets_evidence is provided (CIViC and DepMap both thin):
   - drug_matches with phase ≥ 3 → strong clinical signal, use to support SENSITIVE or RESISTANT verdict.
   - drug_matches with phase 1–2 → weak signal, lean UNCERTAIN but note the association.
   - mechanismOfAction field may specify inhibitor/activator — use to infer direction of effect.
   - Do NOT vote based on OpenTargets alone without mechanism reasoning.
4. If a driver mutation is present but no CIViC context for this drug: vote UNCERTAIN.
   A mutation in the target gene does not tell you whether the drug binds better or worse
   without knowing the mutation's functional consequence for THIS drug. Do NOT default to
   SENSITIVE. State that CIViC evidence is absent and the functional consequence is unknown.
   Exception: CNV amplification of the target oncogene is a sensitivity signal even without
   CIViC (gene amplification → drug target overexpressed → drug likely effective).
5. If the gene is WILD-TYPE (in DB, no mutation): DO NOT default to RESISTANT.
   Wild-type status means the gene is not mutated, but the drug may still work through
   expression-level activity, pathway context, or IC50 history. Prefer UNCERTAIN unless
   other evidence (e.g. wild_type_note says the target requires mutation for activation).
6. If the cell line is not in the database at all: state no data and lean UNCERTAIN.
7. Weigh CNV carefully by status field:
   - Amplification (status='amplification') → sensitivity signal: target overexpressed. Note: cnv_value is often NULL — rely on the status field, not the numeric value.
   - HOMOZYGOUS deletion (status='homozygous_deletion') → RESISTANT: both copies gone, drug has no target.
   - HEMIZYGOUS deletion (status='hemizygous_deletion') → UNCERTAIN: one copy remains, gene still expressed at reduced level. Do NOT vote RESISTANT on hemizygous deletion alone — the target is still present.
8. Conclude with your verdict and why, explicitly citing any CIViC description or DepMap score used.

CONFIDENCE CALIBRATION — apply these caps before reporting confidence:
- If no somatic mutations found AND CNV status is 'neutral' (or no CNV data) AND DepMap Chronos > -0.5
  (or no DepMap data): evidence is genuinely weak → cap confidence at 0.55 regardless of other signals.
- Only exceed confidence 0.75 when you have a specific CIViC entry for THIS drug, a homozygous
  deletion, a clear amplification, or Chronos ≤ -1.0.

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

        # DepMap supplement: query Chronos scores when CIViC coverage is thin
        civic_covered = any(m.get("civic_description") for m in mutations)
        if target_genes and not civic_covered:
            depmap_result = await self._call_tool(
                "get_depmap_dependency", {"cell_line": cell_line, "genes": target_genes}
            )
            scores = (depmap_result or {}).get("scores", []) if isinstance(depmap_result, dict) else []
            if scores:
                evidence["depmap_scores"] = scores
                evidence["depmap_note"] = "Chronos <= -1.0 = strong essentiality (decisive); -0.5 to -1.0 = borderline (supporting only)"

            # OpenTargets fallback: query drug-gene associations when CIViC + DepMap are both thin
            depmap_decisive = any(s.get("chronos", 0) <= -1.0 for s in scores)
            if not depmap_decisive:
                ot_result = await self._call_tool(
                    "get_opentargets_evidence", {"genes": target_genes, "drug": drug}
                )
                if ot_result:
                    ot_list = ot_result if isinstance(ot_result, list) else []
                    matched = [g for g in ot_list if g.get("drug_matches")]
                    if matched:
                        evidence["opentargets_evidence"] = matched
                        evidence["opentargets_note"] = (
                            "OpenTargets found clinical drug-gene associations — "
                            "use phase and mechanismOfAction as weak supporting evidence"
                        )

        return evidence

    def _compute_h(self, key_findings: list, evidence: dict) -> float:
        mutations = evidence.get("mutations", [])
        has_civic   = any(m.get("civic_description") for m in mutations)
        has_driver  = any(m.get("is_driver") for m in mutations)
        has_cnv_amp = any(c.get("status") == "amplification" for c in evidence.get("cnv", []))
        depmap_only = (
            not mutations
            and not has_cnv_amp
            and bool(evidence.get("depmap_scores"))
        )
        if has_civic:
            return 1.0   # drug-specific causal evidence — fully verified
        if has_cnv_amp:
            return 0.8   # structural amplification — strong signal
        if has_driver:
            return 0.5   # driver mutation but no drug-specific context
        if depmap_only:
            return 0.2   # Chronos essentiality only — indirect, not drug-specific
        from src.agents.gcs import compute_h
        return compute_h(key_findings, evidence)

    def _compute_signal(self, evidence: dict) -> float:
        mutations = evidence.get("mutations", [])
        if not mutations:
            return 0.0
        has_civic   = any(m.get("civic_description") for m in mutations)
        has_driver  = any(m.get("is_driver") for m in mutations)
        has_cnv_amp = any(c.get("status") == "amplification" for c in evidence.get("cnv", []))
        if has_civic:
            return 1.0   # drug-specific CIViC evidence → strong, decisive T1
        if has_cnv_amp:
            return 0.75  # amplification is a sensitivity signal even without CIViC
        if has_driver:
            return 0.3   # driver mutation but no drug-specific context → agent votes UNCERTAIN
        return 0.2       # VUS / passenger
