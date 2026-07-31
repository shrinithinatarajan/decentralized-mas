import asyncio
from collections import Counter
from dataclasses import dataclass, field

from src.protocols.axiom_resolver import AxiomResolver
from src.schemas.axiom_rules import AXIOM_HIERARCHY, EVIDENCE_TO_AXIOM_TIER
from src.schemas.evidence_pack import EvidencePack, Verdict

@dataclass
class AxiomChallenge:
    """A directed challenge from a higher-tier decisive agent to a lower-tier decisive agent.

    Emitted when both agents are decisive but disagree. The challenged agent must provide
    a specific mechanistic rebuttal from its own modality, or concede.
    """
    challenger_id: str
    challenger_tier: str
    challenged_id: str
    challenged_tier: str
    challenger_verdict: str
    challenged_verdict: str
    argument: str            # mechanistic argument citing challenger's own key findings


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
    # Agreeing agents ranked by confidence (highest first) — who actually contributed
    # the strongest evidence, as distinct from winning_agent's tier-first attribution.
    contributing_agents: list[str] = field(default_factory=list)


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
    maintain_ids: set[str] | None = None,
) -> list[EvidencePack]:
    """Revise an agent's verdict when peers collectively reject its reasoning.

    Flip conditions (ALL must hold):
      1. Agent is decisive (not UNCERTAIN) — abstainers don't flip.
      2. Mean peer endorsement < threshold — peers found the reasoning unconvincing.
      3. A strict majority of OTHER decisive agents endorse a different verdict.
      4. Agent is not in maintain_ids (successfully rebutted an AxiomChallenge).
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
        if maintain_ids and p.agent_id in maintain_ids:
            # Agent successfully rebutted an AxiomChallenge — honour its verdict
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
    """Return the verdict only when ALL decisive agents agree on the same verdict.

    Consensus requires unanimity among decisive agents — any single dissenting
    decisive agent routes to debate. UNCERTAIN agents do not count as dissent.

    T3 quality gates still apply before an agent is counted as decisive:
      RESISTANT: self_attestation score >= 3
      SENSITIVE: pathway_active = True
    """
    from src.schemas.evidence_pack import EvidenceTier
    decisive = [p for p in packs if p.verdict != Verdict.UNCERTAIN]
    if not decisive:
        return None

    # T3 quality gates
    filtered: list[EvidencePack] = []
    for p in decisive:
        if p.evidence_tier == EvidenceTier.T3_PATHWAY:
            sa = p.self_attestation or {}
            score = sa.get("score") or 0
            if p.verdict == Verdict.RESISTANT and score < 3:
                continue
            if p.verdict == Verdict.SENSITIVE and not sa.get("pathway_active", False):
                continue
        filtered.append(p)
    decisive = filtered

    if not decisive:
        return None

    verdicts = {p.verdict for p in decisive}
    if len(verdicts) != 1:
        return None  # any dissent → debate
    return decisive[0].verdict



class DebateEngine:
    def __init__(self, resolver=None) -> None:
        self._resolver = resolver if resolver is not None else AxiomResolver()

    async def run(
        self,
        packs: list[EvidencePack],
        agents: list | None = None,
        target_genes: list[str] | None = None,
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

        # --- Round 2: AxiomChallenge emission + evidence-gated critique ---
        # Only when there is genuine conflict (≥1 SENSITIVE and ≥1 RESISTANT among decisive agents)
        r1_decisive_verdicts = [p.verdict for p in packs if p.verdict != Verdict.UNCERTAIN]
        has_conflict = Verdict.SENSITIVE in r1_decisive_verdicts and Verdict.RESISTANT in r1_decisive_verdicts

        peer_endorsements: dict[str, float] = {}
        critique_revised: set[str] = set()
        challenge_maintained_ids: set[str] = set()
        challenges: list[AxiomChallenge] = []
        if agents and has_conflict:
            # Emit AxiomChallenges: higher-tier decisive agent → lower-tier decisive agent with opposing verdict
            challenges = self._emit_challenges(packs)
            if challenges:
                trace.append({
                    "round": "AXIOM_CHALLENGES",
                    "challenges": [
                        {
                            "from": ch.challenger_id, "from_tier": ch.challenger_tier,
                            "to": ch.challenged_id, "to_tier": ch.challenged_tier,
                            "challenger_verdict": ch.challenger_verdict,
                            "challenged_verdict": ch.challenged_verdict,
                            "argument": ch.argument,
                        }
                        for ch in challenges
                    ],
                })

            pre_critique_verdicts = {p.agent_id: p.verdict for p in packs}
            challenger_ids = {ch.challenger_id for ch in challenges}
            critiqued, peer_endorsements, challenge_maintained_ids = await self._run_critique_round(
                packs, agents, challenges=challenges, challenger_ids=challenger_ids,
                run_logger=run_logger, case_id=case_id
            )
            critique_revised = {p.agent_id for p in critiqued if p.verdict != pre_critique_verdicts[p.agent_id]}
            trace.append(self._snapshot(critiqued, round_num=2, note="post_critique", prev_verdicts=pre_critique_verdicts))
            if run_logger:
                run_logger.log_debate_round(case_id=case_id, round_num=2, note="post_critique", snapshot=trace[-1])
            packs = critiqued

        # --- R2 verdict revision: skip agents revised by critique() or who rebutted a challenge ---
        if agents and has_conflict and peer_endorsements:
            packs = _apply_verdict_revision(
                packs, peer_endorsements, trace,
                skip_ids=critique_revised,
                maintain_ids=challenge_maintained_ids,
            )
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
        resolution = self._resolver.resolve(
            packs, peer_endorsements=peer_endorsements,
            challenge_maintained_ids=challenge_maintained_ids,
            target_genes=target_genes,
        )
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
            "peer_endorsements": {k: round(v, 3) for k, v in peer_endorsements.items()} if peer_endorsements else {},
        })
        # B2: Distinguish genuine S-vs-R conflict from unanimous-under-quorum.
        from src.schemas.evidence_pack import Verdict as _Verdict
        decisive_verdicts = {p.verdict for p in packs if p.verdict != _Verdict.UNCERTAIN and p.confidence > 0}
        is_genuine_conflict = (
            _Verdict.SENSITIVE in decisive_verdicts and _Verdict.RESISTANT in decisive_verdicts
        )
        resolution_method = "RESOLVER_TIEBREAK" if is_genuine_conflict else "RESOLVER_PRIORITY"
        is_forced = is_genuine_conflict

        if run_logger:
            run_logger.log_resolution(
                case_id=case_id, resolution_method=resolution_method,
                winning_agent=resolution.winning_agent, verdict=resolution.verdict.value, forced=is_forced,
            )
        return ConsensusResult(
            final_verdict=resolution.verdict,
            final_confidence=avg_conf,
            cell_line=cell_line,
            drug=drug,
            winning_agent=resolution.winning_agent,
            rounds_taken=len(trace),
            forced=is_forced,
            dissenting_agents=sorted(dissenting),
            resolution_method=resolution_method,
            trace=trace,
            r1_agents=r1_agents,
        )

    def _emit_challenges(self, packs: list[EvidencePack]) -> list[AxiomChallenge]:
        """Identify decisive inter-tier conflicts and emit AxiomChallenges from higher to lower tier.

        T3 quality gate: pathway agent may not issue a challenge unless self_attestation score >= 3.
        Same gate as _check_consensus — if T3 cannot contribute to consensus, it cannot challenge.
        """
        from src.schemas.evidence_pack import EvidenceTier
        challenges: list[AxiomChallenge] = []
        decisive = [p for p in packs if p.verdict != Verdict.UNCERTAIN]
        for p1 in decisive:
            for p2 in decisive:
                if p1.agent_id == p2.agent_id or p1.verdict == p2.verdict:
                    continue
                p1_pri = AXIOM_HIERARCHY[EVIDENCE_TO_AXIOM_TIER[p1.evidence_tier]]
                p2_pri = AXIOM_HIERARCHY[EVIDENCE_TO_AXIOM_TIER[p2.evidence_tier]]
                if p1_pri <= p2_pri:
                    continue
                # T3 quality gate: pathway agent needs self_attestation score >= 3 to challenge
                if p1.evidence_tier == EvidenceTier.T3_PATHWAY:
                    sa_score = (p1.self_attestation or {}).get("score") or 0
                    if sa_score < 3:
                        continue
                # p1 outranks p2 and disagrees — emit challenge
                findings_str = "; ".join(
                    f"{f.biomarker}={f.value}: {f.interpretation}"
                    for f in p1.key_findings
                ) or "no specific findings cited"
                challenges.append(AxiomChallenge(
                    challenger_id=p1.agent_id,
                    challenger_tier=p1.evidence_tier.value,
                    challenged_id=p2.agent_id,
                    challenged_tier=p2.evidence_tier.value,
                    challenger_verdict=p1.verdict.value,
                    challenged_verdict=p2.verdict.value,
                    argument=(
                        f"My {p1.evidence_tier.value} evidence (confidence {p1.confidence:.2f}) "
                        f"supports {p1.verdict.value}. Specific findings: {findings_str}. "
                        f"Your {p2.evidence_tier.value} verdict of {p2.verdict.value} must be "
                        f"mechanistically reconciled with this higher-tier evidence or conceded."
                    ),
                ))
        return challenges

    async def _run_critique_round(
        self, packs: list[EvidencePack], agents: list,
        challenges: list[AxiomChallenge] | None = None,
        challenger_ids: set[str] | None = None,
        *, run_logger=None, case_id: str | None = None
    ) -> tuple[list[EvidencePack], dict[str, float], set[str]]:
        """Evidence-gated peer review with optional AxiomChallenge responses.

        Returns (critiqued_packs, endorsements, challenge_maintained_ids):
          challenge_maintained_ids — agents that successfully rebutted a challenge;
                                     excluded from _apply_verdict_revision (they already defended).
        """
        agent_map = {a.agent_id: a for a in agents}
        # Build challenger-to-challenged mapping so challengers don't review challenged peer's evidence
        challenger_to_challenged: dict[str, set[str]] = {}
        for ch in (challenges or []):
            challenger_to_challenged.setdefault(ch.challenger_id, set()).add(ch.challenged_id)
        # Challengers cannot review the evidence of peers they challenged — prevents paradoxical self-flip
        peer_order = {
            p.agent_id: [
                q for q in packs
                if q.agent_id != p.agent_id
                and q.agent_id not in challenger_to_challenged.get(p.agent_id, set())
            ]
            for p in packs
        }
        # Index challenges by challenged agent
        challenge_map: dict[str, AxiomChallenge] = {}
        for ch in (challenges or []):
            # One challenge per challenged agent (highest-priority challenger wins if multiple)
            existing = challenge_map.get(ch.challenged_id)
            if existing is None:
                challenge_map[ch.challenged_id] = ch
            else:
                if (AXIOM_HIERARCHY[EVIDENCE_TO_AXIOM_TIER.get(
                    next((p.evidence_tier for p in packs if p.agent_id == ch.challenger_id), None),
                    list(EVIDENCE_TO_AXIOM_TIER.values())[0]
                )] >
                    AXIOM_HIERARCHY[EVIDENCE_TO_AXIOM_TIER.get(
                    next((p.evidence_tier for p in packs if p.agent_id == existing.challenger_id), None),
                    list(EVIDENCE_TO_AXIOM_TIER.values())[0]
                )]):
                    challenge_map[ch.challenged_id] = ch

        tasks = []
        for pack in packs:
            agent = agent_map.get(pack.agent_id)
            ch = challenge_map.get(pack.agent_id)
            if agent is None:
                async def _passthrough(p=pack): return (p, None)
                tasks.append(_passthrough())
            else:
                tasks.append(agent.critique(
                    pack, peer_order[pack.agent_id],
                    challenge=ch,
                    is_challenger=(pack.agent_id in (challenger_ids or set())),
                    challenged_ids={ch2.challenged_id for ch2 in (challenges or []) if ch2.challenger_id == pack.agent_id},
                    run_logger=run_logger, case_id=case_id,
                ))

        results = list(await asyncio.gather(*tasks))
        critiqued = [r[0] for r in results]
        challenge_outcomes = {pack.agent_id: r[1] for pack, r in zip(packs, results)}

        # challenge_maintained_ids: agents that successfully rebutted — skip verdict revision for them
        challenge_maintained_ids: set[str] = {
            aid for aid, maintained in challenge_outcomes.items()
            if maintained is True
        }

        # Aggregate endorsement scores: sum of mechanistic relevance scores given TO each agent
        endorsements: dict[str, float] = {p.agent_id: 0.0 for p in packs}
        for reviewer in critiqued:
            peers = peer_order[reviewer.agent_id]
            for i, peer_pack in enumerate(peers):
                label = chr(ord("A") + i)
                endorsements[peer_pack.agent_id] += reviewer.peer_scores.get(label, 0.0)

        return critiqued, endorsements, challenge_maintained_ids

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
        contributing = [
            p.agent_id for p in sorted(agreeing, key=lambda p: p.confidence, reverse=True)
        ]
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
            contributing_agents=contributing,
        )

    @staticmethod
    def _snapshot(packs: list[EvidencePack], round_num: int, note: str, prev_verdicts: dict | None = None) -> dict:
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
                    "peer_scores_given": p.peer_scores or {},
                    "verdict_changed_from": (
                        prev_verdicts[p.agent_id].value
                        if prev_verdicts and p.agent_id in prev_verdicts and prev_verdicts[p.agent_id] != p.verdict
                        else None
                    ),
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
