import asyncio
from src.schemas.evidence_pack import EvidencePack, Verdict, EvidenceTier, Finding
from src.protocols.debate_engine import DebateEngine, ConsensusResult


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
        drug="Vemurafenib",
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


def test_result_records_cell_line_and_drug():
    packs = [_pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL")]
    engine = DebateEngine()
    result = _run(engine.run(packs))
    assert result.cell_line == "A375"
    assert result.drug == "Vemurafenib"
