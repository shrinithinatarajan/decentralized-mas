from dataclasses import dataclass

from src.schemas.axiom_rules import AXIOM_HIERARCHY, CONFIDENCE_DECAY_PER_FLIP, EVIDENCE_TO_AXIOM_TIER
from src.schemas.evidence_pack import EvidencePack


@dataclass
class ResolutionResult:
    verdict: object          # Verdict enum
    winning_agent: str
    axiom_applied: str
    adjusted_packs: list[EvidencePack]


class AxiomResolver:
    def resolve(self, packs: list[EvidencePack]) -> ResolutionResult:
        # pick winner: highest axiom tier, break ties by confidence
        winner = max(
            packs,
            key=lambda p: (
                AXIOM_HIERARCHY[EVIDENCE_TO_AXIOM_TIER[p.evidence_tier]],
                p.confidence,
            ),
        )
        winning_axiom = EVIDENCE_TO_AXIOM_TIER[winner.evidence_tier]

        adjusted: list[EvidencePack] = []
        for p in packs:
            if p.agent_id == winner.agent_id:
                adjusted.append(p)
            else:
                # apply confidence decay to agents whose verdict differs from winner
                new_conf = (
                    p.confidence * (1 - CONFIDENCE_DECAY_PER_FLIP)
                    if p.verdict != winner.verdict
                    else p.confidence
                )
                adjusted.append(p.model_copy(update={"confidence": new_conf}))

        return ResolutionResult(
            verdict=winner.verdict,
            winning_agent=winner.agent_id,
            axiom_applied=winning_axiom.value,
            adjusted_packs=adjusted,
        )
