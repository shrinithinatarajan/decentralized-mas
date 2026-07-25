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

    def resolve(self, packs: list[EvidencePack], peer_endorsements: dict | None = None) -> ResolutionResult:
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

    def resolve(self, packs: list[EvidencePack], peer_endorsements: dict | None = None) -> ResolutionResult:
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
