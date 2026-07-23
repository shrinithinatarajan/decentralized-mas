from dataclasses import dataclass

from src.schemas.axiom_rules import (
    AXIOM_HIERARCHY,
    CONFIDENCE_DECAY_PER_FLIP,
    EVIDENCE_TO_AXIOM_TIER,
    T1_OVERRIDE_CONFIDENCE_THRESHOLD,
)
from src.schemas.drug_mechanism import is_targeted
from src.schemas.evidence_pack import EvidencePack


@dataclass
class ResolutionResult:
    verdict: object          # Verdict enum
    winning_agent: str
    axiom_applied: str
    adjusted_packs: list[EvidencePack]


class AxiomResolver:
    def resolve(self, packs: list[EvidencePack], peer_endorsements: dict | None = None) -> ResolutionResult:
        from src.schemas.evidence_pack import EvidenceTier, Verdict

        decisive = [p for p in packs if p.verdict != Verdict.UNCERTAIN and p.confidence > 0]
        candidates = decisive if decisive else packs

        # Check if T1 agent is decisive but below the confidence threshold for hard override
        t1_packs = [p for p in candidates if p.evidence_tier == EvidenceTier.T1_STRUCTURAL]
        lower_packs = [p for p in candidates if p.evidence_tier != EvidenceTier.T1_STRUCTURAL]

        use_confidence_weighted = False
        if t1_packs and lower_packs:
            t1 = max(t1_packs, key=lambda p: p.confidence)
            if t1.confidence < T1_OVERRIDE_CONFIDENCE_THRESHOLD:
                # T1 evidence is weak — check if lower tiers unanimously disagree
                lower_verdicts = {p.verdict for p in lower_packs}
                if len(lower_verdicts) == 1 and t1.verdict not in lower_verdicts:
                    # Unanimous lower-tier consensus contradicts weak T1 → use weighted vote
                    use_confidence_weighted = True

        # For non-targeted drugs (cytotoxics, broad epigenetics, metabolic), T1 structural
        # mutation evidence is not mechanistically relevant. Demote T1 to T3 priority so
        # T4 IC50 data can win tiebreaks on these drugs.
        drug = packs[0].drug if packs else ""
        def _effective_priority(p: EvidencePack) -> int:
            from src.schemas.evidence_pack import EvidenceTier
            base = AXIOM_HIERARCHY[EVIDENCE_TO_AXIOM_TIER[p.evidence_tier]]
            if not is_targeted(drug) and p.evidence_tier == EvidenceTier.T1_STRUCTURAL:
                return 1  # demote below T4 (priority 2); IC50 evidence wins on non-targeted drugs
            return base

        if use_confidence_weighted:
            # Confidence-weighted vote: max confidence per verdict, pick highest
            from collections import defaultdict
            scores: dict = defaultdict(float)
            for p in candidates:
                scores[p.verdict] = max(scores[p.verdict], p.confidence)
            winning_verdict = max(scores, key=lambda v: scores[v])
            winner = max(
                [p for p in candidates if p.verdict == winning_verdict],
                key=lambda p: p.confidence,
            )
            winning_axiom_str = "T5_STATISTICAL_CONSENSUS"  # consensus overrode weak T1
        else:
            winner = max(
                candidates,
                key=lambda p: (
                    (peer_endorsements or {}).get(p.agent_id, 0.0),
                    _effective_priority(p),
                    p.confidence,
                ),
            )
            winning_axiom_str = EVIDENCE_TO_AXIOM_TIER[winner.evidence_tier].value

        adjusted: list[EvidencePack] = []
        for p in packs:
            if p.agent_id == winner.agent_id:
                adjusted.append(p)
            else:
                new_conf = (
                    p.confidence * (1 - CONFIDENCE_DECAY_PER_FLIP)
                    if p.verdict != winner.verdict
                    else p.confidence
                )
                adjusted.append(p.model_copy(update={"confidence": new_conf}))

        return ResolutionResult(
            verdict=winner.verdict,
            winning_agent=winner.agent_id,
            axiom_applied=winning_axiom_str,
            adjusted_packs=adjusted,
        )
