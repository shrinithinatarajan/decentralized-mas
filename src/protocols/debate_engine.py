from dataclasses import dataclass, field

from src.protocols.axiom_resolver import AxiomResolver
from src.protocols.conflict_detector import ConflictDetector
from src.schemas.axiom_rules import MAX_DEBATE_ROUNDS
from src.schemas.evidence_pack import EvidencePack, Verdict


@dataclass
class ConsensusResult:
    final_verdict: Verdict
    final_confidence: float
    cell_line: str
    drug: str
    winning_agent: str
    rounds_taken: int
    forced: bool
    dissenting_agents: list[str]
    trace: list[dict] = field(default_factory=list)


class DebateEngine:
    def __init__(self) -> None:
        self._detector = ConflictDetector()
        self._resolver = AxiomResolver()

    def run(self, packs: list[EvidencePack]) -> ConsensusResult:
        cell_line = packs[0].cell_line
        drug = packs[0].drug
        current = list(packs)
        trace: list[dict] = []
        dissenting: set[str] = set()
        forced = False

        for round_num in range(1, MAX_DEBATE_ROUNDS + 1):
            if not self._detector.has_conflict(current):
                break

            resolution = self._resolver.resolve(current)
            trace.append({
                "round": round_num,
                "axiom_applied": resolution.axiom_applied,
                "winning_agent": resolution.winning_agent,
                "verdict": resolution.verdict.value,
            })

            # record agents overridden (verdict differed from winner)
            for p in current:
                if p.agent_id != resolution.winning_agent and p.verdict != resolution.verdict:
                    dissenting.add(p.agent_id)

            current = self._update_packs(current, resolution)

            if not self._detector.has_conflict(current):
                break
        else:
            forced = True

        # final resolution on whatever state we're in
        final = self._resolver.resolve(current)
        avg_conf = sum(p.confidence for p in final.adjusted_packs) / len(final.adjusted_packs)

        return ConsensusResult(
            final_verdict=final.verdict,
            final_confidence=avg_conf,
            cell_line=cell_line,
            drug=drug,
            winning_agent=final.winning_agent,
            rounds_taken=len(trace),
            forced=forced,
            dissenting_agents=sorted(dissenting),
            trace=trace,
        )

    def _update_packs(self, packs: list[EvidencePack], resolution) -> list[EvidencePack]:
        updated = []
        for p in resolution.adjusted_packs:
            if p.agent_id != resolution.winning_agent and p.verdict != resolution.verdict:
                updated.append(p.model_copy(update={"verdict": resolution.verdict}))
            else:
                updated.append(p)
        return updated
