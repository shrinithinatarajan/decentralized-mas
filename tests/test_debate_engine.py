import asyncio
from src.schemas.evidence_pack import EvidencePack, Verdict, EvidenceTier, Finding
from src.protocols.debate_engine import DebateEngine, ConsensusResult, _check_consensus


def _run(coro):
    return asyncio.run(coro)


def _pack(
    agent_id: str,
    verdict: str,
    tier: str,
    confidence: float = 0.8,
) -> EvidencePack:
    return EvidencePack(
        agent_id=agent_id,
        cell_line="A375",
        # Dabrafenib is in is_targeted()'s allowlist (unlike Vemurafenib — a
        # separate, tracked bug, see Phase 4 in to-do.md) so these tests
        # exercise axiom-tier priority cleanly, without T1 being incorrectly
        # demoted by the drug-targeting check.
        drug="Dabrafenib",
        verdict=Verdict(verdict),
        confidence=confidence,
        evidence_tier=EvidenceTier(tier),
        key_findings=[
            Finding(
                biomarker="BRAF_V600E",
                value="mutant",
                interpretation="driver",
                data_source="test",
            )
        ],
    )


def test_quorum_floor_blocks_single_decisive_agent_auto_win():
    """Bug C: 1 decisive agent among 4 must not trivially 'win' consensus —
    decisive count must exceed half of ALL agents, not just half of the
    decisive subset (which is 0.5 when there's only one decisive agent)."""
    packs = [
        _pack("genomics_agent",        "SENSITIVE", "T1_STRUCTURAL"),
        _pack("transcriptomics_agent", "UNCERTAIN", "T2_TRANSCRIPTIONAL"),
        _pack("pharmacology_agent",    "UNCERTAIN", "T4_PHARMACOLOGICAL"),
        _pack("pathway_agent",         "UNCERTAIN", "T3_PATHWAY"),
    ]
    assert _check_consensus(packs) is None


def test_quorum_floor_blocks_correlated_two_of_four_agreement():
    """Case a7: 2 decisive agents unanimous, 2 silently abstained. 2 is not
    a majority of 4 agents — this must route to the resolver, not resolve
    as if unanimous agreement among a shrunken decisive set were consensus."""
    packs = [
        _pack("genomics_agent",        "UNCERTAIN", "T1_STRUCTURAL"),
        _pack("transcriptomics_agent", "UNCERTAIN", "T2_TRANSCRIPTIONAL"),
        _pack("pharmacology_agent",    "RESISTANT", "T4_PHARMACOLOGICAL"),
        _pack("pathway_agent",         "RESISTANT", "T3_PATHWAY"),
    ]
    packs[3].self_attestation = {"score": 4}
    assert _check_consensus(packs) is None


def test_quorum_floor_allows_three_of_four_majority():
    """3 of 4 decisive and agreeing still clears the relative floor (3 > 2)."""
    packs = [
        _pack("genomics_agent",        "SENSITIVE", "T1_STRUCTURAL"),
        _pack("transcriptomics_agent", "SENSITIVE", "T2_TRANSCRIPTIONAL"),
        _pack("pharmacology_agent",    "SENSITIVE", "T4_PHARMACOLOGICAL"),
        _pack("pathway_agent",         "UNCERTAIN", "T3_PATHWAY"),
    ]
    assert _check_consensus(packs) == Verdict.SENSITIVE


def test_quorum_floor_forces_resolver_end_to_end():
    """Integration: a7-shaped case must resolve via RESOLVER_TIEBREAK, not
    CONSENSUS_R1, once the quorum floor is enforced."""
    packs = [
        _pack("genomics_agent",        "UNCERTAIN", "T1_STRUCTURAL"),
        _pack("transcriptomics_agent", "UNCERTAIN", "T2_TRANSCRIPTIONAL"),
        _pack("pharmacology_agent",    "RESISTANT", "T4_PHARMACOLOGICAL"),
        _pack("pathway_agent",         "RESISTANT", "T3_PATHWAY"),
    ]
    packs[3].self_attestation = {"score": 4}
    engine = DebateEngine()
    result = _run(engine.run(packs))
    assert result.resolution_method == "RESOLVER_TIEBREAK"
    assert result.forced is True


def test_contributing_agents_ranked_by_confidence_not_tier():
    """winning_agent stays tier-first (axiom hierarchy), but contributing_agents
    must reflect who actually had the highest-confidence evidence — a T1 agent
    that merely agrees at low confidence isn't who "did the work"."""
    packs = [
        _pack("genomics_agent",     "SENSITIVE", "T1_STRUCTURAL",      confidence=0.55),
        _pack("pharmacology_agent", "SENSITIVE", "T4_PHARMACOLOGICAL", confidence=0.92),
        _pack("pathway_agent",      "SENSITIVE", "T3_PATHWAY",         confidence=0.88),
    ]
    engine = DebateEngine()
    result = _run(engine.run(packs))
    assert result.winning_agent == "genomics_agent"  # tier-first, unchanged
    assert result.contributing_agents[0] == "pharmacology_agent"  # highest confidence
    assert result.contributing_agents[1] == "pathway_agent"
    assert result.contributing_agents[2] == "genomics_agent"


def test_r1_agents_snapshot_includes_self_attestation():
    """r1_agents must carry self_attestation so downstream trials can log it —
    reasoning/data_status were already captured, self_attestation was silently dropped."""
    packs = [
        _pack("genomics_agent",        "SENSITIVE", "T1_STRUCTURAL"),
        _pack("transcriptomics_agent", "SENSITIVE", "T2_TRANSCRIPTIONAL"),
        _pack("pharmacology_agent",    "SENSITIVE", "T4_PHARMACOLOGICAL"),
    ]
    packs[0].self_attestation = {"score": 4, "checklist": {"a": True}}
    engine = DebateEngine()
    result = _run(engine.run(packs))
    genomics_entry = next(a for a in result.r1_agents if a["agent_id"] == "genomics_agent")
    assert genomics_entry["self_attestation"] == {"score": 4, "checklist": {"a": True}}


def test_three_of_four_agree_gives_consensus_r1():
    """>=3 agents agreeing in round 1 -> CONSENSUS_R1, resolver never fires."""
    packs = [
        _pack("genomics_agent",        "SENSITIVE", "T1_STRUCTURAL"),
        _pack("transcriptomics_agent", "SENSITIVE", "T2_TRANSCRIPTIONAL"),
        _pack("pharmacology_agent",    "SENSITIVE", "T4_PHARMACOLOGICAL"),
        _pack("pathway_agent",         "RESISTANT", "T3_PATHWAY"),
    ]
    engine = DebateEngine()
    result = _run(engine.run(packs))
    assert isinstance(result, ConsensusResult)
    assert result.final_verdict == Verdict.SENSITIVE
    assert result.resolution_method == "CONSENSUS_R1"
    assert result.forced is False
    assert result.rounds_taken == 1


def test_four_of_four_agree_gives_consensus_r1():
    """All agents agreeing -> CONSENSUS_R1."""
    packs = [
        _pack("genomics_agent",        "RESISTANT", "T1_STRUCTURAL"),
        _pack("transcriptomics_agent", "RESISTANT", "T2_TRANSCRIPTIONAL"),
        _pack("pharmacology_agent",    "RESISTANT", "T4_PHARMACOLOGICAL"),
        _pack("pathway_agent",         "RESISTANT", "T3_PATHWAY"),
    ]
    engine = DebateEngine()
    result = _run(engine.run(packs))
    assert result.final_verdict == Verdict.RESISTANT
    assert result.resolution_method == "CONSENSUS_R1"
    assert result.forced is False


def test_two_two_split_falls_to_resolver_without_agents():
    """2-2 split with no agents for round 2 -> resolver tiebreak."""
    packs = [
        _pack("genomics_agent",        "SENSITIVE", "T1_STRUCTURAL",      confidence=0.9),
        _pack("transcriptomics_agent", "SENSITIVE", "T2_TRANSCRIPTIONAL", confidence=0.7),
        _pack("pharmacology_agent",    "RESISTANT", "T4_PHARMACOLOGICAL", confidence=0.8),
        _pack("pathway_agent",         "RESISTANT", "T3_PATHWAY",         confidence=0.6),
    ]
    # T3 self-attestation gate treats a missing score as 0 (fails the >=3 gate) —
    # set it so pathway_agent counts as decisive and this is a genuine 2-2 split.
    packs[3].self_attestation = {"score": 4}
    engine = DebateEngine()
    result = _run(engine.run(packs))  # no agents passed -> round 2 skipped
    assert result.resolution_method == "RESOLVER_TIEBREAK"
    assert result.forced is True


def test_t1_high_confidence_wins_tiebreak():
    """In a 2-2 tiebreak, T1 with confidence >= threshold overrides lower tiers."""
    packs = [
        _pack("genomics_agent",        "SENSITIVE", "T1_STRUCTURAL",      confidence=0.9),
        _pack("transcriptomics_agent", "SENSITIVE", "T2_TRANSCRIPTIONAL", confidence=0.65),
        _pack("pharmacology_agent",    "RESISTANT", "T4_PHARMACOLOGICAL", confidence=0.8),
        _pack("pathway_agent",         "RESISTANT", "T3_PATHWAY",         confidence=0.75),
    ]
    packs[3].self_attestation = {"score": 4}
    engine = DebateEngine()
    result = _run(engine.run(packs))
    assert result.resolution_method == "RESOLVER_TIEBREAK"
    assert result.final_verdict == Verdict.SENSITIVE


def test_dissenting_agents_logged_in_tiebreak():
    """Agents overridden by resolver appear in dissenting_agents."""
    packs = [
        _pack("genomics_agent",        "SENSITIVE", "T1_STRUCTURAL",      confidence=0.9),
        _pack("transcriptomics_agent", "RESISTANT", "T2_TRANSCRIPTIONAL", confidence=0.8),
        _pack("pharmacology_agent",    "RESISTANT", "T4_PHARMACOLOGICAL", confidence=0.7),
        _pack("pathway_agent",         "RESISTANT", "T3_PATHWAY",         confidence=0.6),
    ]
    engine = DebateEngine()
    result = _run(engine.run(packs))
    assert len(result.dissenting_agents) >= 1


def test_trace_populated_for_contested_case():
    """Trace has at least one entry when case is contested (2-2 split)."""
    packs = [
        _pack("genomics_agent",        "SENSITIVE", "T1_STRUCTURAL",      confidence=0.9),
        _pack("transcriptomics_agent", "SENSITIVE", "T2_TRANSCRIPTIONAL", confidence=0.7),
        _pack("pharmacology_agent",    "RESISTANT", "T4_PHARMACOLOGICAL", confidence=0.8),
        _pack("pathway_agent",         "RESISTANT", "T3_PATHWAY",         confidence=0.6),
    ]
    packs[3].self_attestation = {"score": 4}
    engine = DebateEngine()
    result = _run(engine.run(packs))
    assert len(result.trace) >= 1
    assert "round" in result.trace[0]


def test_final_confidence_is_positive():
    packs = [
        _pack("genomics_agent",        "SENSITIVE", "T1_STRUCTURAL",      confidence=0.9),
        _pack("pharmacology_agent",    "SENSITIVE", "T4_PHARMACOLOGICAL", confidence=0.6),
        _pack("pathway_agent",         "SENSITIVE", "T3_PATHWAY",         confidence=0.7),
        _pack("transcriptomics_agent", "SENSITIVE", "T2_TRANSCRIPTIONAL", confidence=0.8),
    ]
    engine = DebateEngine()
    result = _run(engine.run(packs))
    assert 0.0 < result.final_confidence <= 1.0


def test_t1_veto_blocks_consensus_when_t1_disagrees_with_majority():
    """3 lower-tier agents agree but T1 (genomics) dissents -> resolver, not CONSENSUS_R1.

    A confirmed structural finding is an axiological override, not a minority vote.
    The resolver then picks T1's verdict via the axiom hierarchy.
    """
    packs = [
        _pack("genomics_agent",        "RESISTANT", "T1_STRUCTURAL",      confidence=0.9),
        _pack("transcriptomics_agent", "SENSITIVE", "T2_TRANSCRIPTIONAL", confidence=0.8),
        _pack("pharmacology_agent",    "SENSITIVE", "T4_PHARMACOLOGICAL", confidence=0.7),
        _pack("pathway_agent",         "SENSITIVE", "T3_PATHWAY",         confidence=0.6),
    ]
    engine = DebateEngine()
    result = _run(engine.run(packs))
    # Must NOT short-circuit to consensus — T1 veto forces resolver
    assert result.resolution_method == "RESOLVER_TIEBREAK"
    assert result.forced is True
    # Resolver picks T1 (highest axiom tier) -> RESISTANT
    assert result.final_verdict == Verdict.RESISTANT
    assert result.winning_agent == "genomics_agent"


def test_t1_veto_does_not_fire_when_t1_agrees_with_majority():
    """T1 veto must not trigger when T1 is part of the majority."""
    packs = [
        _pack("genomics_agent",        "SENSITIVE", "T1_STRUCTURAL"),
        _pack("transcriptomics_agent", "SENSITIVE", "T2_TRANSCRIPTIONAL"),
        _pack("pharmacology_agent",    "SENSITIVE", "T4_PHARMACOLOGICAL"),
        _pack("pathway_agent",         "RESISTANT", "T3_PATHWAY"),
    ]
    engine = DebateEngine()
    result = _run(engine.run(packs))
    assert result.resolution_method == "CONSENSUS_R1"
    assert result.final_verdict == Verdict.SENSITIVE


def test_result_records_cell_line_and_drug():
    packs = [_pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL")]
    engine = DebateEngine()
    result = _run(engine.run(packs))
    assert result.cell_line == "A375"
    assert result.drug == "Dabrafenib"
