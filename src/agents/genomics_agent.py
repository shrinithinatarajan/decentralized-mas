from src.agents.base_agent import BaseAgent

_SYSTEM = """You are the Genomics Agent in a multi-agent cancer drug resistance framework.
Your modality: somatic mutations, copy number variation (DNA level), CRISPR gene essentiality (DepMap),
OncoKB oncogenicity annotation, and RPPA protein expression.
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
3. CHECK OncoKB ANNOTATION: If oncokb_annotations are provided for a mutation:
   - oncogenicity='Oncogenic' or 'Likely Oncogenic' AND mutationEffect='Gain-of-function' or 'Likely Gain-of-function':
     → The mutation activates the target gene. If the drug INHIBITS this gene, vote SENSITIVE.
   - oncogenicity='Oncogenic' or 'Likely Oncogenic' AND mutationEffect='Loss-of-function' or 'Likely Loss-of-function':
     → The mutation inactivates the target. The drug may not have a functional target → lean RESISTANT or UNCERTAIN.
   - highestSensitiveLevel is set (e.g. LEVEL_1, LEVEL_2, LEVEL_3A):
     → This mutation is clinically annotated as sensitive to the drug (or drug class) → vote SENSITIVE with high confidence.
   - highestResistanceLevel is set (e.g. LEVEL_R1, LEVEL_R2):
     → Clinically annotated resistance → vote RESISTANT.
   - oncogenicity='VUS' or 'Unknown': not sufficient alone → proceed to other evidence.
   - oncogenicity='Neutral' or 'Likely Neutral': mutation is passenger, not driver → prefer UNCERTAIN.
4. DepMap CRISPR Chronos supplement: if depmap_scores are provided, use them as supporting evidence.
   - Chronos ≤ -1.0: strong essentiality — gene is critical for cell survival → SENSITIVE signal.
   - Chronos -0.5 to -1.0: borderline — vote SENSITIVE with confidence capped at 0.55. It is weak evidence, but still a signal: when it is the only evidence available (no CIViC, no IC50 data from T4), use it. Do NOT use it to override a stronger IC50 RESISTANT verdict.
   - Chronos > -0.5: gene is non-essential in this cell line → not informative for sensitivity.
   - Only use as a tiebreaker or to support an UNCERTAIN call — do NOT override a strong CIViC or OncoKB verdict.
   CRITICAL CAVEAT: Chronos measures CRISPR gene KNOCKOUT essentiality — this is NOT the same as
   pharmacological drug sensitivity. A low Chronos score means the cell cannot survive WITHOUT the
   gene (KO lethal), but a drug inhibitor may still fail to achieve sufficient target inhibition at
   therapeutic concentrations. When T4 IC50 data contradicts Chronos, prefer the IC50 verdict.
   Cap confidence at 0.65 when Chronos is the sole basis for a SENSITIVE verdict.
5. CHECK RPPA PROTEIN EXPRESSION: If rppa_expression is provided:
   - rppa_score > 0.5 for the target protein: protein is overexpressed → target is present and active
     → this SUPPORTS sensitivity for an inhibitor drug (the drug has an abundant target to inhibit).
   - rppa_score < -0.5: protein is underexpressed → drug may lack target protein to inhibit
     → lean RESISTANT or UNCERTAIN for an inhibitor.
   - rppa_score near 0 (-0.5 to 0.5): average expression, not informative on its own.
   - Use RPPA primarily when: (a) no CIViC or OncoKB decisive evidence exists, AND
     (b) target gene is wild-type. It is the strongest available evidence when both (a) and (b) hold.
   - Do NOT override a strong CIViC or OncoKB verdict with RPPA alone.
6. OpenTargets fallback: if opentargets_evidence is provided (CIViC and DepMap both thin):
   - drug_matches with phase ≥ 3 → strong clinical signal, use to support SENSITIVE or RESISTANT verdict.
   - drug_matches with phase 1–2 → weak signal, lean UNCERTAIN but note the association.
7. If a driver mutation is present but no CIViC/OncoKB context for this drug: vote UNCERTAIN.
   A mutation in the target gene does not tell you whether the drug binds better or worse
   without knowing the mutation's functional consequence for THIS drug.
   Exception: CNV amplification of the target oncogene is a sensitivity signal even without
   CIViC (gene amplification → drug target overexpressed → drug likely effective).
8. If the gene is WILD-TYPE (in DB, no mutation): DO NOT default to RESISTANT.
   Wild-type status means the gene is not mutated, but the drug may still work through
   expression-level activity, pathway context, or IC50 history. Check RPPA for protein
   expression. Prefer UNCERTAIN unless RPPA shows high expression (→ SENSITIVE).
9. If the cell line is not in the database at all: state no data and lean UNCERTAIN.
10. Weigh CNV carefully by status field:
    - Amplification (status='amplification') → sensitivity signal: target overexpressed.
    - HOMOZYGOUS deletion (status='homozygous_deletion') → RESISTANT: both copies gone.
    - HEMIZYGOUS deletion (status='hemizygous_deletion') → UNCERTAIN: one copy remains.
11. Conclude with your verdict and why, explicitly citing any CIViC description, OncoKB annotation,
    DepMap score, or RPPA expression value used.

CONFIDENCE CALIBRATION — apply these caps before reporting confidence:
- OncoKB highestSensitiveLevel=LEVEL_1 or CIViC drug-specific description: up to 0.90 confidence.
- OncoKB oncogenicity=Oncogenic + Gain-of-function (inhibitor drug): up to 0.80 confidence.
- CNV amplification OR Chronos ≤ -1.0: up to 0.75 confidence.
- RPPA overexpression (score > 0.5) with wild-type gene: cap at 0.65 (protein-only evidence).
- No CIViC/OncoKB AND no CNV AND Chronos > -0.5: cap at 0.55 regardless.

Return ONLY a JSON object matching the EvidencePack schema. No prose outside the JSON."""


class GenomicsAgent(BaseAgent):
    agent_id = "genomics_agent"
    from src.schemas.evidence_pack import EvidenceTier as _ET
    _tier = _ET.T1_STRUCTURAL

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

        # OncoKB: annotate each mutation with oncogenicity + drug sensitivity level
        if target_genes and mutations:
            oncokb_annotations = []
            for mut in mutations:
                alteration = mut.get("mutation", "")
                if not alteration:
                    continue
                gene = mut.get("gene", "")
                if gene not in target_genes:
                    continue
                ann = await self._call_tool(
                    "get_oncokb_annotation",
                    {"gene": gene, "alteration": alteration, "drug": drug},
                )
                if isinstance(ann, dict) and not ann.get("error"):
                    oncokb_annotations.append({**ann, "mutation": alteration})
                elif isinstance(ann, dict) and "ONCOKB_TOKEN not set" in (ann.get("error") or ""):
                    break  # no token — skip all
            if oncokb_annotations:
                evidence["oncokb_annotations"] = oncokb_annotations

        # RPPA protein expression: always fetch for target genes (helps wild-type cases)
        if target_genes:
            rppa_result = await self._call_tool(
                "get_rppa_expression", {"cell_line": cell_line, "genes": target_genes}
            )
            rppa_data = rppa_result if isinstance(rppa_result, dict) else {}
            rppa_entries = [
                r for r in rppa_data.get("rppa", [])
                if r.get("entries")  # only include genes with actual measurements
            ]
            if rppa_entries:
                evidence["rppa_expression"] = rppa_entries

            # OpenTargets fallback: only when CIViC + DepMap + OncoKB are all thin
            depmap_decisive = any(
                s.get("chronos", 0) <= -1.0
                for s in evidence.get("depmap_scores", [])
            )
            oncokb_decisive = any(
                a.get("highestSensitiveLevel") or a.get("highestResistanceLevel")
                for a in evidence.get("oncokb_annotations", [])
            )
            if not civic_covered and not depmap_decisive and not oncokb_decisive:
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
        has_civic = any(m.get("civic_description") for m in mutations)
        has_driver = any(m.get("is_driver") for m in mutations)
        has_cnv_amp = any(c.get("status") == "amplification" for c in evidence.get("cnv", []))

        # OncoKB decisive signal
        oncokb_anns = evidence.get("oncokb_annotations", [])
        has_oncokb_sensitive = any(a.get("highestSensitiveLevel") for a in oncokb_anns)
        has_oncokb_resistant = any(a.get("highestResistanceLevel") for a in oncokb_anns)
        has_oncokb_oncogenic = any(
            a.get("oncogenicity") in ("Oncogenic", "Likely Oncogenic") for a in oncokb_anns
        )

        # RPPA decisive signal
        rppa_entries = evidence.get("rppa_expression", [])
        rppa_high = any(
            (e.get("total_protein_score") or 0) > 0.5 for e in rppa_entries
        )
        rppa_low = all(
            (e.get("total_protein_score") or 0) < -0.5
            for e in rppa_entries if e.get("total_protein_score") is not None
        ) and rppa_entries

        depmap_only = (
            not mutations and not has_cnv_amp and bool(evidence.get("depmap_scores"))
        )

        if has_civic:
            return 1.0
        if has_oncokb_sensitive or has_oncokb_resistant:
            return 0.9   # OncoKB drug-level annotation
        if has_cnv_amp:
            return 0.8
        if has_oncokb_oncogenic:
            return 0.6   # oncogenic mutation but no specific drug level
        if has_driver:
            return 0.5
        if rppa_high or rppa_low:
            return 0.4   # protein expression only
        if depmap_only:
            return 0.2
        from src.agents.gcs import compute_h
        return compute_h(key_findings, evidence)

    def _compute_signal(self, evidence: dict) -> float:
        mutations = evidence.get("mutations", [])
        has_civic = any(m.get("civic_description") for m in mutations)
        has_cnv_amp = any(c.get("status") == "amplification" for c in evidence.get("cnv", []))

        oncokb_anns = evidence.get("oncokb_annotations", [])
        has_oncokb_decisive = any(
            a.get("highestSensitiveLevel") or a.get("highestResistanceLevel")
            for a in oncokb_anns
        )
        has_oncokb_oncogenic = any(
            a.get("oncogenicity") in ("Oncogenic", "Likely Oncogenic") for a in oncokb_anns
        )

        rppa_entries = evidence.get("rppa_expression", [])
        rppa_score = max(
            (abs(e.get("total_protein_score") or 0) for e in rppa_entries),
            default=0.0,
        )

        if has_civic:
            return 1.0
        if has_oncokb_decisive:
            return 0.9
        if has_cnv_amp:
            return 0.75
        if has_oncokb_oncogenic:
            return 0.5
        if mutations and any(m.get("is_driver") for m in mutations):
            return 0.3
        if rppa_score > 0.5:
            return min(rppa_score / 2.0, 0.4)  # RPPA alone: at most 0.4
        return 0.2
