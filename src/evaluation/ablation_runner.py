import random
from collections import Counter
from enum import Enum

from src.protocols.axiom_resolver import ResolutionResult
from src.protocols.debate_engine import ConsensusResult, DebateEngine, agent_snapshot
from src.schemas.axiom_rules import AXIOM_HIERARCHY, CONFIDENCE_DECAY_PER_FLIP, EVIDENCE_TO_AXIOM_TIER
from src.schemas.evidence_pack import EvidencePack, Verdict


class AblationVariant(str, Enum):
    NO_DEBATE = "no_debate"
    NO_AXIOMS = "no_axioms"
    RANDOM_AXIOM_ORDER = "random_axiom_order"


class NoDebateEngine:
    """Majority-vote aggregator — no debate rounds."""

    async def run(self, packs: list[EvidencePack], agents=None, **kwargs) -> ConsensusResult:
        decisive = [p for p in packs if p.verdict != Verdict.UNCERTAIN]
        if not decisive:
            return ConsensusResult(
                final_verdict=Verdict.UNCERTAIN,
                final_confidence=sum(p.confidence for p in packs) / len(packs),
                cell_line=packs[0].cell_line,
                drug=packs[0].drug,
                winning_agent=max(packs, key=lambda p: p.confidence).agent_id,
                rounds_taken=0,
                forced=False,
                dissenting_agents=[],
                resolution_method="MAJORITY_VOTE",
                r1_agents=agent_snapshot(packs),
            )

        counts: Counter = Counter(p.verdict for p in decisive)
        # group confidence totals per verdict for tie-breaking
        conf_sum: dict = {}
        for p in decisive:
            conf_sum[p.verdict] = conf_sum.get(p.verdict, 0.0) + p.confidence

        majority = max(counts, key=lambda v: (counts[v], conf_sum[v]))
        winners = [p for p in decisive if p.verdict == majority]
        best = max(winners, key=lambda p: p.confidence)
        avg_conf = sum(p.confidence for p in winners) / len(winners)
        contributing = [p.agent_id for p in sorted(winners, key=lambda p: p.confidence, reverse=True)]

        return ConsensusResult(
            final_verdict=majority,
            final_confidence=avg_conf,
            cell_line=packs[0].cell_line,
            drug=packs[0].drug,
            winning_agent=best.agent_id,
            rounds_taken=0,
            forced=False,
            dissenting_agents=[],
            resolution_method="MAJORITY_VOTE",
            r1_agents=agent_snapshot(packs),
            contributing_agents=contributing,
        )


class ConfidenceOnlyResolver:
    """Resolves conflicts purely by confidence — ignores axiom hierarchy."""

    def resolve(self, packs: list[EvidencePack], peer_endorsements: dict | None = None, **kwargs) -> ResolutionResult:
        winner = max(packs, key=lambda p: p.confidence)
        adjusted = [
            p if p.agent_id == winner.agent_id
            else p.model_copy(update={"confidence": p.confidence * (1 - CONFIDENCE_DECAY_PER_FLIP)})
            for p in packs
        ]
        return ResolutionResult(
            verdict=winner.verdict,
            winning_agent=winner.agent_id,
            axiom_applied="CONFIDENCE_ONLY",
            adjusted_packs=adjusted,
        )


class RandomAxiomResolver:
    """Resolves conflicts using a randomly shuffled axiom hierarchy."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def resolve(self, packs: list[EvidencePack], peer_endorsements: dict | None = None, **kwargs) -> ResolutionResult:
        tiers = list(AXIOM_HIERARCHY.keys())
        values = list(AXIOM_HIERARCHY.values())
        self._rng.shuffle(values)
        shuffled = dict(zip(tiers, values))

        winner = max(
            packs,
            key=lambda p: (shuffled[EVIDENCE_TO_AXIOM_TIER[p.evidence_tier]], p.confidence),
        )
        adjusted = [
            p if p.agent_id == winner.agent_id
            else p.model_copy(update={"confidence": p.confidence * (1 - CONFIDENCE_DECAY_PER_FLIP)})
            for p in packs
        ]
        return ResolutionResult(
            verdict=winner.verdict,
            winning_agent=winner.agent_id,
            axiom_applied="RANDOM_AXIOM",
            adjusted_packs=adjusted,
        )


def make_engine(variant: AblationVariant) -> DebateEngine | NoDebateEngine:
    if variant == AblationVariant.NO_DEBATE:
        return NoDebateEngine()
    if variant == AblationVariant.NO_AXIOMS:
        return DebateEngine(resolver=ConfidenceOnlyResolver())
    if variant == AblationVariant.RANDOM_AXIOM_ORDER:
        return DebateEngine(resolver=RandomAxiomResolver())
    raise ValueError(f"Unknown variant: {variant}")


# ---------------------------------------------------------------------------
# No-MCP variant: agents receive evidence as natural-language prose, not JSON
# ---------------------------------------------------------------------------

def evidence_to_prose(evidence: dict) -> str:
    """Convert a structured MCP evidence dict to natural-language paragraphs."""
    parts = []

    # Genomics
    mutations = evidence.get("mutations") or []
    if mutations:
        mut_strs = [f"{m.get('gene','?')} {m.get('mutation','?')} ({m.get('mutation_type','?')})" for m in mutations[:5]]
        parts.append("Mutations identified: " + "; ".join(mut_strs) + ".")
        for m in mutations[:3]:
            desc = m.get("civic_description") or ""
            if desc:
                parts.append(f"CIViC for {m.get('gene')} {m.get('mutation')}: {desc[:400]}")
            okb = m.get("oncokb_annotations") or {}
            if okb:
                parts.append(f"OncoKB for {m.get('gene')} {m.get('mutation')}: oncogenicity={okb.get('oncogenicity')}, effect={okb.get('mutationEffect')}, sensitiveLevel={okb.get('highestSensitiveLevel')}, resistanceLevel={okb.get('highestResistanceLevel')}.")

    cnv = evidence.get("cnv") or []
    if cnv:
        cnv_strs = [f"{c.get('gene')} CNV={c.get('cnv_value',0):.2f} ({c.get('status','?')})" for c in cnv[:5]]
        parts.append("Copy number variation: " + "; ".join(cnv_strs) + ".")

    rppa = evidence.get("rppa_expression") or []
    if rppa:
        rppa_strs = [f"{r.get('gene')} RPPA={r['entries'][0]['rppa_score']:.2f}" for r in rppa[:3] if r.get("entries")]
        if rppa_strs:
            parts.append("Protein expression (RPPA): " + "; ".join(rppa_strs) + ".")

    # Transcriptomics
    target_expr = evidence.get("target_expression") or []
    if target_expr:
        expr_strs = [f"{e.get('gene')} z={e.get('z_score',0):.2f} TPM={e.get('tpm',0):.1f}" for e in target_expr[:5]]
        parts.append("Target gene expression (CCLE): " + "; ".join(expr_strs) + ".")

    high_expr = evidence.get("high_expression_context") or []
    if high_expr:
        top = [f"{e['gene']} z={e.get('z_score',0):.2f}" for e in high_expr[:5]]
        parts.append("Highly expressed genes in this cell line: " + "; ".join(top) + ".")

    # Pharmacology
    ic50 = evidence.get("ic50") or {}
    if ic50 and isinstance(ic50, dict):
        parts.append(f"IC50 data: ln_ic50={ic50.get('ln_ic50','N/A')}, z_score={ic50.get('z_score','N/A')}, AUC={ic50.get('auc','N/A')}.")

    drug_info = evidence.get("drug_info") or {}
    if drug_info and isinstance(drug_info, dict):
        parts.append(f"Drug info: targets={drug_info.get('target_genes','?')}, MOA={drug_info.get('moa','?')}, class={drug_info.get('drug_class','?')}.")

    # Pathway
    pw_mem = evidence.get("pathway_membership") or {}
    if pw_mem:
        pw_strs = [f"{v.get('pathway_name', pid)}" for pid, v in list(pw_mem.items())[:5]]
        parts.append("Target gene pathway membership (KEGG): " + "; ".join(pw_strs) + ".")

    pw_scores = evidence.get("pathway_activity_scores") or {}
    if pw_scores:
        active = {pid: s for pid, s in pw_scores.items() if s >= 0.25}
        pids = list(pw_mem.keys()) if pw_mem else []
        active_target = {pid: s for pid, s in active.items() if pid in pids}
        if active_target:
            parts.append("Active pathways (score >= 0.25) containing the target: " +
                         "; ".join(f"{pw_mem.get(p,{}).get('pathway_name',p)} ({s:.2f})" for p, s in list(active_target.items())[:5]) + ".")

    bypass = evidence.get("bypass_check") or {}
    confirmed = evidence.get("confirmed_bypass_genes") or []
    if confirmed:
        parts.append(f"Confirmed bypass genes expressed in this cell line: {', '.join(list(set(confirmed))[:5])}.")
    elif bypass and isinstance(bypass, dict):
        any_bypass = any(v.get("bypass_exists") for v in bypass.values() if isinstance(v, dict))
        if any_bypass:
            parts.append("Bypass routes found for the blocked gene(s) in the relevant pathway(s).")

    return "\n".join(parts) if parts else "No relevant data found for this case."


class NaturalLanguageAgent:
    """Wraps any agent to receive evidence as prose paragraphs instead of structured JSON."""

    def __init__(self, agent) -> None:
        self._agent = agent
        self.agent_id = agent.agent_id

    async def analyze(
        self,
        cell_line: str,
        drug: str,
        target_genes: list[str] | None = None,
        *,
        run_logger=None,
        case_id: str | None = None,
    ):
        import json as _json
        from src.agents.base_agent import _SCHEMA_EXAMPLE
        from src.agents.gcs import compute_gcs
        from src.schemas.evidence_pack import Verdict

        self._agent._run_logger = run_logger
        self._agent._case_id = case_id
        evidence = await self._agent._fetch_evidence(cell_line, drug, target_genes)
        prose = evidence_to_prose(evidence)

        prompt = (
            f"Cell line: {cell_line}\nDrug: {drug}\n\n"
            f"Evidence summary (natural language — no structured data provided):\n{prose}\n\n"
            f"Respond with ONLY a JSON object in this exact format (no prose, no markdown):\n{_SCHEMA_EXAMPLE}"
        )
        raw = await self._agent.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system=self._agent.system_prompt,
            run_logger=run_logger,
            agent_id=self.agent_id,
            case_id=case_id,
        )
        pack = self._agent._parse_pack(raw, cell_line, drug)
        pack.confidence = compute_gcs(
            self._agent._compute_h(pack.key_findings, evidence),
            evidence, pack.evidence_tier, self._agent._compute_signal(evidence),
        )
        if pack.verdict == Verdict.UNCERTAIN:
            pack.confidence = min(pack.confidence, 0.5)
        pack = pack.model_copy(update={"data_status": evidence.get("data_status"), "raw_evidence": evidence})
        return pack

    async def critique(self, *args, **kwargs):
        return await self._agent.critique(*args, **kwargs)
