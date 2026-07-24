import asyncio
from collections import Counter
from dataclasses import dataclass, field

from src.protocols.axiom_resolver import AxiomResolver
from src.schemas.axiom_rules import AXIOM_HIERARCHY, EVIDENCE_TO_AXIOM_TIER
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
    r1_agents: list[dict] = field(default_factory=list)  # per-agent R1 snapshot (always populated)


# Threshold: if mean peer endorsement falls below this, the agent is eligible to revise its verdict
_REVISION_ENDORSEMENT_THRESHOLD = 0.35
# Confidence penalty applied when an agent revises under peer pressure
_REVISION_CONFIDENCE_PENALTY = 0.70


def _compute_peer_majority_verdicts(packs: list[EvidencePack]) -> dict[str, Verdict | None]:
    """For each agent, return the strict majority verdict of all OTHER decisive agents, or None."""
    result: dict[str, Verdict | None] = {}
    for p in packs:
        others_decisive = [q for q in packs if q.agent_id != p.agent_id and q.verdict != Verdict.UNCERTAIN]
        if not others_decisive:
            result[p.agent_id] = None
            continue
        counts = Counter(q.verdict for q in others_decisive)
        majority_verdict, n = counts.most_common(1)[0]
        result[p.agent_id] = majority_verdict if n > len(others_decisive) / 2 else None
    return result


def _apply_verdict_revision(
    packs: list[EvidencePack],
    endorsements: dict[str, float],
    trace: list[dict],
    skip_ids: set[str] | None = None,
) -> list[EvidencePack]:
    """Revise an agent's verdict when peers collectively reject its reasoning.

    Flip conditions (ALL must hold):
      1. Agent is decisive (not UNCERTAIN) — abstainers don't flip.
      2. Mean peer endorsement < threshold — peers found the reasoning unconvincing.
      3. A strict majority of OTHER decisive agents endorse a different verdict.
    On flip: verdict adopts peer majority; confidence penalised to signal revised-under-pressure.
    """
    n_peers = len(packs) - 1
    if n_peers <= 0:
        return packs

    peer_majorities = _compute_peer_majority_verdicts(packs)
    revised: list[EvidencePack] = []
    revisions: list[dict] = []

    for p in packs:
        if skip_ids and p.agent_id in skip_ids:
            revised.append(p)
            continue
        mean_end = endorsements.get(p.agent_id, 0.0) / n_peers
        peer_maj = peer_majorities.get(p.agent_id)
        should_flip = (
            p.verdict != Verdict.UNCERTAIN
            and mean_end < _REVISION_ENDORSEMENT_THRESHOLD
            and peer_maj is not None
            and peer_maj != p.verdict
        )
        if should_flip:
            new_conf = round(p.confidence * _REVISION_CONFIDENCE_PENALTY, 4)
            revisions.append({
                "agent": p.agent_id,
                "from": p.verdict.value,
                "to": peer_maj.value,  # type: ignore[union-attr]
                "mean_endorsement": round(mean_end, 3),
                "confidence_before": p.confidence,
                "confidence_after": new_conf,
            })
            revised.append(p.model_copy(update={"verdict": peer_maj, "confidence": new_conf}))
        else:
            revised.append(p)

    if revisions:
        trace.append({"round": "R2_VERDICT_REVISION", "revisions": revisions})

    return revised


def _check_consensus(packs: list[EvidencePack]) -> Verdict | None:
    """Return the verdict if a strict majority of decisive agents agree, else None.

    T3 self-attestation gate: pathway agent only decisive if self_attestation score >= 3.
    T1 veto: if all decisive T1 agents disagree with the majority, route to resolver.
    """
    from src.schemas.evidence_pack import EvidenceTier
    decisive = [p for p in packs if p.verdict != Verdict.UNCERTAIN]
    if not decisive:
        return None

    # T3 self-attestation gate: demote pathway agent if checklist score < 3
    filtered: list[EvidencePack] = []
    for p in decisive:
        if p.evidence_tier == EvidenceTier.T3_PATHWAY:
            sa = p.self_attestation or {}
            score = sa.get("score") or 0  # treat absent/None as 0 → fail gate
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
        r1_agents = self._r1_agent_snapshot(packs)
        consensus_verdict = _check_consensus(packs)
        if consensus_verdict is not None:
            result = self._build_result(
                packs, consensus_verdict, cell_line, drug,
                rounds_taken=1, forced=False,
                resolution_method="CONSENSUS_R1", trace=trace,
                r1_agents=r1_agents,
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

        # --- Round 2: peer critique — only when there is genuine conflict (≥1 SENSITIVE and ≥1 RESISTANT) ---
        r1_decisive_verdicts = [p.verdict for p in packs if p.verdict != Verdict.UNCERTAIN]
        has_conflict = Verdict.SENSITIVE in r1_decisive_verdicts and Verdict.RESISTANT in r1_decisive_verdicts

        peer_endorsements: dict[str, float] = {}
        critique_revised: set[str] = set()
        if agents and has_conflict:
            pre_critique_verdicts = {p.agent_id: p.verdict for p in packs}
            critiqued, peer_endorsements = await self._run_critique_round(
                packs, agents, run_logger=run_logger, case_id=case_id
            )
            critique_revised = {p.agent_id for p in critiqued if p.verdict != pre_critique_verdicts[p.agent_id]}
            trace.append(self._snapshot(critiqued, round_num=2, note="post_critique"))
            if run_logger:
                run_logger.log_debate_round(case_id=case_id, round_num=2, note="post_critique", snapshot=trace[-1])
            packs = critiqued

        # --- R2 verdict revision: skip agents already revised by critique() to prevent cascade ---
        if agents and has_conflict and peer_endorsements:
            packs = _apply_verdict_revision(packs, peer_endorsements, trace, skip_ids=critique_revised)
            if run_logger:
                revision_entry = next((e for e in reversed(trace) if e.get("round") == "R2_VERDICT_REVISION"), None)
                if revision_entry:
                    run_logger.log_debate_round(case_id=case_id, round_num=3, note="verdict_revision", snapshot=revision_entry)

        # --- Round 2 consensus check (Bug A fix) ---
        consensus_verdict = _check_consensus(packs)
        if consensus_verdict is not None:
            result = self._build_result(
                packs, consensus_verdict, cell_line, drug,
                rounds_taken=2, forced=False,
                resolution_method="CONSENSUS_R2", trace=trace,
                r1_agents=r1_agents,
            )
            if run_logger:
                run_logger.log_resolution(
                    case_id=case_id, resolution_method="CONSENSUS_R2",
                    winning_agent=result.winning_agent,
                    verdict=result.final_verdict.value, forced=False,
                )
            return result

        # --- Resolver tiebreak (last resort) ---
        resolution = self._resolver.resolve(packs, peer_endorsements=peer_endorsements)
        # Use only agents agreeing with winner for confidence (Bug D fix)
        agreeing = [p for p in resolution.adjusted_packs if p.verdict == resolution.verdict]
        if not agreeing:
            winning_pack = next((p for p in resolution.adjusted_packs if p.agent_id == resolution.winning_agent), None)
            agreeing = [winning_pack] if winning_pack else resolution.adjusted_packs
        avg_conf = sum(p.confidence for p in agreeing) / len(agreeing)
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
            r1_agents=r1_agents,
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
                endorsements[peer_pack.agent_id] += reviewer.peer_scores.get(label, 0.0)  # 0.0 not 3.0: absent reviewer contributes nothing (Bug B/F fix)

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
        r1_agents: list[dict] | None = None,
    ) -> ConsensusResult:
        agreeing   = [p for p in packs if p.verdict == verdict]
        dissenting = [p.agent_id for p in packs
                      if p.verdict != verdict and p.verdict != Verdict.UNCERTAIN]
        avg_conf   = sum(p.confidence for p in agreeing) / len(agreeing)
        winning    = max(
            agreeing,
            key=lambda p: (AXIOM_HIERARCHY[EVIDENCE_TO_AXIOM_TIER[p.evidence_tier]], p.confidence),
        ).agent_id  # axiom tier first, confidence as tiebreak (Bug H fix)
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
            r1_agents=r1_agents or [],
        )

    @staticmethod
    def _snapshot(packs: list[EvidencePack], round_num: int, note: str) -> dict:
        return {
            "round": round_num,
            "note": note,
            "verdicts": {
                p.agent_id: {
                    "verdict": p.verdict.value,
                    "confidence": round(p.confidence, 3),
                    "evidence_tier": p.evidence_tier.value if p.evidence_tier else None,
                    "data_status": p.data_status,
                    "key_findings": [f.model_dump() for f in p.key_findings],
                    "reasoning": p.reasoning,
                    "self_attestation": p.self_attestation,
                }
                for p in packs
            },
        }

    @staticmethod
    def _r1_agent_snapshot(packs: list[EvidencePack]) -> list[dict]:
        """Compact per-agent R1 record always stored regardless of resolution path."""
        return agent_snapshot(packs)


def agent_snapshot(packs: list[EvidencePack]) -> list[dict]:
    """Compact per-agent record (verdict/confidence/reasoning/data_status/self_attestation).

    Shared by DebateEngine's r1_agents and NoDebateEngine so every engine variant
    logs the same per-agent trace fields regardless of aggregation method.
    """
    return [
        {
            "agent_id": p.agent_id,
            "verdict": p.verdict.value,
            "confidence": round(p.confidence, 3),
            "evidence_tier": p.evidence_tier.value if p.evidence_tier else None,
            "data_status": p.data_status,
            "key_findings": [f.model_dump() for f in p.key_findings],
            "reasoning": p.reasoning,
            "self_attestation": p.self_attestation,
        }
        for p in packs
    ]
