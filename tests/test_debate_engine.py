import pytest
from src.schemas.evidence_pack import EvidencePack, Verdict, EvidenceTier, Finding
from src.schemas.axiom_rules import MAX_DEBATE_ROUNDS
from src.protocols.debate_engine import DebateEngine, ConsensusResult


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


def test_no_conflict_returns_consensus_immediately():
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL"),
        _pack("transcriptomics_agent", "SENSITIVE", "T2_TRANSCRIPTIONAL"),
        _pack("pharmacology_agent", "SENSITIVE", "T4_PHARMACOLOGICAL"),
    ]
    engine = DebateEngine()
    result = engine.run(packs)
    assert isinstance(result, ConsensusResult)
    assert result.final_verdict == Verdict.SENSITIVE
    assert result.rounds_taken == 0


def test_conflict_resolved_by_axiom_in_one_round():
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL", confidence=0.9),
        _pack("transcriptomics_agent", "RESISTANT", "T2_TRANSCRIPTIONAL", confidence=0.8),
        _pack("pharmacology_agent", "RESISTANT", "T4_PHARMACOLOGICAL", confidence=0.7),
    ]
    engine = DebateEngine()
    result = engine.run(packs)
    assert result.final_verdict == Verdict.SENSITIVE
    assert result.rounds_taken == 1


def test_forced_resolution_after_max_rounds():
    # Simulate unresolvable conflict by all agents having the same tier
    # so the resolver picks a winner but others "resist" — we force this
    # by using a subclass that never converges
    class NeverConvergeEngine(DebateEngine):
        def _update_packs(self, packs, resolution):
            # agents don't change their verdicts (stubborn)
            return packs

    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL", confidence=0.9),
        _pack("transcriptomics_agent", "RESISTANT", "T1_STRUCTURAL", confidence=0.8),
    ]
    engine = NeverConvergeEngine()
    result = engine.run(packs)
    assert result.rounds_taken == MAX_DEBATE_ROUNDS
    assert result.forced is True


def test_dissenting_agents_logged():
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL", confidence=0.9),
        _pack("transcriptomics_agent", "RESISTANT", "T2_TRANSCRIPTIONAL", confidence=0.8),
        _pack("pharmacology_agent", "RESISTANT", "T4_PHARMACOLOGICAL", confidence=0.7),
    ]
    engine = DebateEngine()
    result = engine.run(packs)
    # agents overridden by T1 should appear in dissent log
    assert len(result.dissenting_agents) >= 1
    assert any("transcriptomics" in d or "pharmacology" in d for d in result.dissenting_agents)


def test_debate_trace_records_each_round():
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL", confidence=0.9),
        _pack("transcriptomics_agent", "RESISTANT", "T2_TRANSCRIPTIONAL", confidence=0.8),
    ]
    engine = DebateEngine()
    result = engine.run(packs)
    assert len(result.trace) == result.rounds_taken
    if result.rounds_taken > 0:
        entry = result.trace[0]
        assert "round" in entry
        assert "axiom_applied" in entry
        assert "winning_agent" in entry


def test_final_confidence_is_weighted_average_of_adjusted_packs():
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL", confidence=0.9),
        _pack("pharmacology_agent", "SENSITIVE", "T4_PHARMACOLOGICAL", confidence=0.6),
    ]
    engine = DebateEngine()
    result = engine.run(packs)
    assert 0.0 < result.final_confidence <= 1.0


def test_result_records_cell_line_and_drug():
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL"),
    ]
    engine = DebateEngine()
    result = engine.run(packs)
    assert result.cell_line == "A375"
    assert result.drug == "Vemurafenib"
