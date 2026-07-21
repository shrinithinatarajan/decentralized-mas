import asyncio
from collections import Counter
from dataclasses import dataclass, field

from src.protocols.axiom_resolver import AxiomResolver
from src.schemas.evidence_pack import EvidencePack, Verdict

@dataclass
class ConsensusResult:
    final_verdict: Verdict
    final_confidence: float
    cell_line: str
    drug: str
    winning_agent: str
    rounds_taken: int
    forced: bool           # True = resolver fired as tiebreaker; False = genuine consensus
    dissenting_agents: list[str]
    resolution_method: str = ""  # "CONSENSUS_R1", "CONSENSUS_R2", "RESOLVER_TIEBREAK"
    trace: list[dict] = field(default_factory=list)


def _check_consensus(packs: list[EvidencePack]) -> Verdict | None:
    """Return the verdict if a strict majority of decisive agents agree, else None.

    Requires at least 2 decisive (non-UNCERTAIN) agents — a single decisive agent
    always escalates to Round 2 peer critique rather than winning by default.

    T3 self-attestation gate: pathway agent can only be decisive if its self_attestation
    score >= 3. Replaces the blunt T1+T2 quorum rule with a grounded evidence check.

    T1 veto: if a decisive T1 agent disagrees with the majority, route to resolver.
    """
    from src.schemas.evidence_pack import EvidenceTier
    decisive = [p for p in packs if p.verdict != Verdict.UNCERTAIN]
    if not decisive:
        return None

    # T3 self-attestation gate: demote pathway agent if checklist score < 3
    filtered: list[EvidencePack] = []
    for p in decisive:
        if p.evidence_tier == EvidenceTier.T3_PATHWAY:
            score = (p.self_attestation or {}).get("score", 0)
            if score < 3:
                continue
        filtered.append(p)
    decisive = filtered

    if not decisive:
        return None

    counts = Counter(p.verdict for p in decisive)
    majority_verdict, n = counts.most_common(1)[0]
    if n <= len(decisive) / 2:
        return None
    t1_decisive = [p for p in decisive if p.evidence_tier == EvidenceTier.T1_STRUCTURAL]
    if t1_decisive and all(p.verdict != majority_verdict for p in t1_decisive):
        return None
    return majority_verdict



class DebateEngine:
    def __init__(self, resolver=None) -> None:
        self._resolver = resolver if resolver is not None else AxiomResolver()

    async def run(
        self,
        packs: list[EvidencePack],
        agents: list | None = None,
        *,
        run_logger=None,
        case_id: str | None = None,
    ) -> ConsensusResult:
        """Run the debate.

        Flow:
          1. Check consensus on round-1 packs  → done if >=3 agree.
          2. If agents provided and no consensus: run round-2 cross-modal revision.
          3. Check consensus on revised packs  → done if >=3 agree.
          4. Still no consensus: axiom resolver fires as tiebreaker (forced=True).
        """
        cell_line = packs[0].cell_line
        drug      = packs[0].drug
        trace: list[dict] = []

        # --- Round 1 consensus check ---
        consensus_verdict = _check_consensus(packs)
        if consensus_verdict is not None:
            result = self._build_result(
                packs, consensus_verdict, cell_line, drug,
                rounds_taken=1, forced=False,
                resolution_method="CONSENSUS_R1", trace=trace,
            )
            if run_logger:
                run_logger.log_resolution(
                    case_id=case_id, resolution_method=result.resolution_method,
                    winning_agent=result.winning_agent, verdict=result.final_verdict.value,
                    forced=result.forced,
                )
            return result

        trace.append(self._snapshot(packs, round_num=1, note="no_consensus"))
        if run_logger:
            run_logger.log_debate_round(case_id=case_id, round_num=1, note="no_consensus", snapshot=trace[-1])

        # --- Round 2: peer critique (Karpathy council style — verdicts locked, peers scored) ---
        peer_endorsements: dict[str, float] = {}
        if agents:
            critiqued, peer_endorsements = await self._run_critique_round(
                packs, agents, run_logger=run_logger, case_id=case_id
            )
            trace.append(self._snapshot(critiqued, round_num=2, note="post_critique"))
            if run_logger:
                run_logger.log_debate_round(case_id=case_id, round_num=2, note="post_critique", snapshot=trace[-1])
            packs = critiqued  # updated confidence, peer_scores attached

        # --- Resolver tiebreak (last resort) ---
        resolution = self._resolver.resolve(packs, peer_endorsements=peer_endorsements)
        avg_conf = sum(p.confidence for p in resolution.adjusted_packs) / len(resolution.adjusted_packs)
        dissenting = [
            p.agent_id for p in packs
            if p.agent_id != resolution.winning_agent and p.verdict != resolution.verdict
        ]
        trace.append({
            "round": len(trace) + 1,
            "note": "resolver_tiebreak",
            "axiom_applied": resolution.axiom_applied,
            "winning_agent": resolution.winning_agent,
            "verdict": resolution.verdict.value,
        })
        if run_logger:
            run_logger.log_resolution(
                case_id=case_id, resolution_method="RESOLVER_TIEBREAK",
                winning_agent=resolution.winning_agent, verdict=resolution.verdict.value, forced=True,
            )
        return ConsensusResult(
            final_verdict=resolution.verdict,
            final_confidence=avg_conf,
            cell_line=cell_line,
            drug=drug,
            winning_agent=resolution.winning_agent,
            rounds_taken=len(trace),
            forced=True,
            dissenting_agents=sorted(dissenting),
            resolution_method="RESOLVER_TIEBREAK",
            trace=trace,
        )

    async def _run_critique_round(
        self, packs: list[EvidencePack], agents: list, *, run_logger=None, case_id: str | None = None
    ) -> tuple[list[EvidencePack], dict[str, float]]:
        """Each agent scores peers' reasoning quality; verdicts stay locked."""
        agent_map = {a.agent_id: a for a in agents}
        # Build ordered peer list per agent (determines anonymization labels A, B, C...)
        peer_order = {p.agent_id: [q for q in packs if q.agent_id != p.agent_id] for p in packs}

        tasks = []
        for pack in packs:
            agent = agent_map.get(pack.agent_id)
            if agent is None:
                async def _passthrough(p=pack): return p
                tasks.append(_passthrough())
            else:
                tasks.append(agent.critique(
                    pack, peer_order[pack.agent_id], run_logger=run_logger, case_id=case_id
                ))

        critiqued = list(await asyncio.gather(*tasks))

        # Aggregate endorsement scores: for each agent, sum of scores given TO it by peers
        endorsements: dict[str, float] = {p.agent_id: 0.0 for p in packs}
        for reviewer in critiqued:
            peers = peer_order[reviewer.agent_id]
            for i, peer_pack in enumerate(peers):
                label = chr(ord('A') + i)
                endorsements[peer_pack.agent_id] += reviewer.peer_scores.get(label, 3.0)

        return critiqued, endorsements

    def _build_result(
        self,
        packs: list[EvidencePack],
        verdict: Verdict,
        cell_line: str,
        drug: str,
        rounds_taken: int,
        forced: bool,
        resolution_method: str,
        trace: list[dict],
    ) -> ConsensusResult:
        agreeing   = [p for p in packs if p.verdict == verdict]
        dissenting = [p.agent_id for p in packs
                      if p.verdict != verdict and p.verdict != Verdict.UNCERTAIN]
        avg_conf   = sum(p.confidence for p in agreeing) / len(agreeing)
        winning    = max(agreeing, key=lambda p: p.confidence).agent_id
        return ConsensusResult(
            final_verdict=verdict,
            final_confidence=avg_conf,
            cell_line=cell_line,
            drug=drug,
            winning_agent=winning,
            rounds_taken=rounds_taken,
            forced=forced,
            dissenting_agents=sorted(dissenting),
            resolution_method=resolution_method,
            trace=trace,
        )

    @staticmethod
    def _snapshot(packs: list[EvidencePack], round_num: int, note: str) -> dict:
        return {
            "round": round_num,
            "note": note,
            "verdicts": {
                p.agent_id: {"verdict": p.verdict.value, "confidence": round(p.confidence, 3)}
                for p in packs
            },
        }
