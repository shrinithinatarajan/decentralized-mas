import pytest
from collections import Counter

from src.protocols.debate_engine import DebateEngine, ConsensusResult
from src.protocols.axiom_resolver import AxiomResolver, ResolutionResult
from src.schemas.evidence_pack import EvidencePack, Verdict, EvidenceTier, Finding
from src.evaluation.ablation_runner import (
    AblationVariant,
    NoDebateEngine,
    ConfidenceOnlyResolver,
    RandomAxiomResolver,
    make_engine,
)


def _pack(agent_id, verdict, tier, confidence=0.8):
    return EvidencePack(
        agent_id=agent_id,
        cell_line="A375",
        drug="Vemurafenib",
        verdict=Verdict(verdict),
        confidence=confidence,
        evidence_tier=EvidenceTier(tier),
        key_findings=[Finding(biomarker="BRAF", value="V600E", interpretation="driver", data_source="GDSC")],
    )


# --- DebateEngine: custom resolver injection ---

def test_debate_engine_accepts_custom_resolver():
    class AlwaysFirstResolver:
        def resolve(self, packs):
            return ResolutionResult(
                verdict=packs[0].verdict,
                winning_agent=packs[0].agent_id,
                axiom_applied="CUSTOM",
                adjusted_packs=packs,
            )

    engine = DebateEngine(resolver=AlwaysFirstResolver())
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL", confidence=0.5),
        _pack("transcriptomics_agent", "RESISTANT", "T2_TRANSCRIPTIONAL", confidence=0.9),
    ]
    result = engine.run(packs)
    # T2 would normally win by axiom, but custom resolver always picks first agent
    assert result.final_verdict == Verdict.SENSITIVE


# --- NoDebateEngine ---

def test_no_debate_engine_returns_consensus_result():
    engine = NoDebateEngine()
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL"),
        _pack("transcriptomics_agent", "RESISTANT", "T2_TRANSCRIPTIONAL"),
        _pack("pharmacology_agent", "SENSITIVE", "T4_PHARMACOLOGICAL"),
    ]
    result = engine.run(packs)
    assert isinstance(result, ConsensusResult)


def test_no_debate_engine_majority_sensitive():
    engine = NoDebateEngine()
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL"),
        _pack("transcriptomics_agent", "SENSITIVE", "T2_TRANSCRIPTIONAL"),
        _pack("pharmacology_agent", "RESISTANT", "T4_PHARMACOLOGICAL"),
    ]
    result = engine.run(packs)
    assert result.final_verdict == Verdict.SENSITIVE


def test_no_debate_engine_majority_resistant():
    engine = NoDebateEngine()
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL"),
        _pack("transcriptomics_agent", "RESISTANT", "T2_TRANSCRIPTIONAL"),
        _pack("pharmacology_agent", "RESISTANT", "T4_PHARMACOLOGICAL"),
    ]
    result = engine.run(packs)
    assert result.final_verdict == Verdict.RESISTANT


def test_no_debate_engine_takes_zero_rounds():
    engine = NoDebateEngine()
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL"),
        _pack("transcriptomics_agent", "RESISTANT", "T2_TRANSCRIPTIONAL"),
    ]
    result = engine.run(packs)
    assert result.rounds_taken == 0
    assert result.forced is False


def test_no_debate_engine_tie_broken_by_confidence():
    engine = NoDebateEngine()
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL", confidence=0.9),
        _pack("transcriptomics_agent", "RESISTANT", "T2_TRANSCRIPTIONAL", confidence=0.6),
    ]
    result = engine.run(packs)
    # Tied 1-1; SENSITIVE has higher total confidence
    assert result.final_verdict == Verdict.SENSITIVE


# --- ConfidenceOnlyResolver ---

def test_confidence_only_resolver_ignores_axiom_tier():
    resolver = ConfidenceOnlyResolver()
    # T4 (low tier) with very high confidence should beat T1 (high tier) with low confidence
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL", confidence=0.3),
        _pack("pharmacology_agent", "RESISTANT", "T4_PHARMACOLOGICAL", confidence=0.95),
    ]
    result = resolver.resolve(packs)
    assert result.verdict == Verdict.RESISTANT
    assert result.winning_agent == "pharmacology_agent"


def test_confidence_only_resolver_returns_resolution_result():
    resolver = ConfidenceOnlyResolver()
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL", confidence=0.8),
        _pack("pharmacology_agent", "RESISTANT", "T4_PHARMACOLOGICAL", confidence=0.7),
    ]
    result = resolver.resolve(packs)
    assert isinstance(result, ResolutionResult)


# --- RandomAxiomResolver ---

def test_random_axiom_resolver_returns_resolution_result():
    resolver = RandomAxiomResolver(seed=42)
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL"),
        _pack("pharmacology_agent", "RESISTANT", "T4_PHARMACOLOGICAL"),
    ]
    result = resolver.resolve(packs)
    assert isinstance(result, ResolutionResult)
    assert result.verdict in (Verdict.SENSITIVE, Verdict.RESISTANT)


def test_random_axiom_resolver_is_deterministic_given_same_seed():
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL"),
        _pack("pharmacology_agent", "RESISTANT", "T4_PHARMACOLOGICAL"),
    ]
    r1 = RandomAxiomResolver(seed=7).resolve(packs)
    r2 = RandomAxiomResolver(seed=7).resolve(packs)
    assert r1.winning_agent == r2.winning_agent


# --- make_engine factory ---

def test_make_engine_no_debate_returns_no_debate_engine():
    engine = make_engine(AblationVariant.NO_DEBATE)
    assert isinstance(engine, NoDebateEngine)


def test_make_engine_no_axioms_returns_debate_engine():
    engine = make_engine(AblationVariant.NO_AXIOMS)
    assert isinstance(engine, DebateEngine)


def test_make_engine_random_axiom_order_returns_debate_engine():
    engine = make_engine(AblationVariant.RANDOM_AXIOM_ORDER)
    assert isinstance(engine, DebateEngine)


def test_make_engine_no_axioms_uses_confidence_only_resolver():
    engine = make_engine(AblationVariant.NO_AXIOMS)
    assert isinstance(engine._resolver, ConfidenceOnlyResolver)


def test_make_engine_random_axiom_uses_random_resolver():
    engine = make_engine(AblationVariant.RANDOM_AXIOM_ORDER)
    assert isinstance(engine._resolver, RandomAxiomResolver)
