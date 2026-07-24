import pytest
from src.schemas.evidence_pack import EvidencePack, Verdict, EvidenceTier, Finding
from src.schemas.axiom_rules import CONFIDENCE_DECAY_PER_FLIP


# --- fixtures ---

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


# ============================================================
# ConflictDetector
# ============================================================

from src.protocols.conflict_detector import ConflictDetector


def test_no_conflict_when_all_agree():
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL"),
        _pack("transcriptomics_agent", "SENSITIVE", "T2_TRANSCRIPTIONAL"),
        _pack("pharmacology_agent", "SENSITIVE", "T4_PHARMACOLOGICAL"),
    ]
    cd = ConflictDetector()
    assert cd.has_conflict(packs) is False
    assert cd.conflicts(packs) == []


def test_conflict_detected_when_verdicts_differ():
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL"),
        _pack("transcriptomics_agent", "RESISTANT", "T2_TRANSCRIPTIONAL"),
        _pack("pharmacology_agent", "SENSITIVE", "T4_PHARMACOLOGICAL"),
    ]
    cd = ConflictDetector()
    assert cd.has_conflict(packs) is True


def test_conflicts_returns_disagreeing_agent_ids():
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL"),
        _pack("transcriptomics_agent", "RESISTANT", "T2_TRANSCRIPTIONAL"),
        _pack("pharmacology_agent", "UNCERTAIN", "T4_PHARMACOLOGICAL"),
    ]
    cd = ConflictDetector()
    conflict_ids = {c["agent_id"] for c in cd.conflicts(packs)}
    assert "transcriptomics_agent" in conflict_ids
    assert "pharmacology_agent" in conflict_ids


def test_unanimous_uncertain_is_not_a_conflict():
    packs = [
        _pack("genomics_agent", "UNCERTAIN", "T1_STRUCTURAL"),
        _pack("transcriptomics_agent", "UNCERTAIN", "T2_TRANSCRIPTIONAL"),
    ]
    cd = ConflictDetector()
    assert cd.has_conflict(packs) is False


def test_single_pack_never_conflicts():
    cd = ConflictDetector()
    assert cd.has_conflict([_pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL")]) is False


# ============================================================
# AxiomResolver
# ============================================================

from src.protocols.axiom_resolver import AxiomResolver


def test_resolver_returns_highest_tier_verdict():
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL", confidence=0.9),
        _pack("transcriptomics_agent", "RESISTANT", "T2_TRANSCRIPTIONAL", confidence=0.8),
        _pack("pharmacology_agent", "RESISTANT", "T4_PHARMACOLOGICAL", confidence=0.7),
    ]
    resolver = AxiomResolver()
    result = resolver.resolve(packs)
    # T1 is highest tier — its verdict wins
    assert result.verdict == Verdict.SENSITIVE
    assert result.winning_agent == "genomics_agent"


def test_resolver_t1_overrides_majority():
    # 3 agents say RESISTANT but T1 says SENSITIVE
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL", confidence=0.85),
        _pack("transcriptomics_agent", "RESISTANT", "T2_TRANSCRIPTIONAL", confidence=0.8),
        _pack("pharmacology_agent", "RESISTANT", "T4_PHARMACOLOGICAL", confidence=0.75),
        _pack("pathway_agent", "RESISTANT", "T3_PATHWAY", confidence=0.7),
    ]
    resolver = AxiomResolver()
    result = resolver.resolve(packs)
    assert result.verdict == Verdict.SENSITIVE
    assert result.winning_agent == "genomics_agent"


def test_resolver_applies_confidence_decay_to_overridden_agents():
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL", confidence=0.9),
        _pack("transcriptomics_agent", "RESISTANT", "T2_TRANSCRIPTIONAL", confidence=0.8),
    ]
    resolver = AxiomResolver()
    result = resolver.resolve(packs)
    # transcriptomics was overridden (flipped) — its adjusted confidence should decay
    overridden = next(a for a in result.adjusted_packs if a.agent_id == "transcriptomics_agent")
    assert overridden.confidence == pytest.approx(0.8 * (1 - CONFIDENCE_DECAY_PER_FLIP))


def test_resolver_winning_agent_confidence_unchanged():
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL", confidence=0.9),
        _pack("transcriptomics_agent", "RESISTANT", "T2_TRANSCRIPTIONAL", confidence=0.8),
    ]
    resolver = AxiomResolver()
    result = resolver.resolve(packs)
    winner = next(a for a in result.adjusted_packs if a.agent_id == "genomics_agent")
    assert winner.confidence == pytest.approx(0.9)


def test_resolver_tie_broken_by_higher_confidence():
    # Two agents at same tier — higher confidence wins
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL", confidence=0.6),
        _pack("other_agent", "RESISTANT", "T1_STRUCTURAL", confidence=0.9),
    ]
    resolver = AxiomResolver()
    result = resolver.resolve(packs)
    assert result.verdict == Verdict.RESISTANT
    assert result.winning_agent == "other_agent"


def test_resolver_axiom_tier_outranks_peer_endorsement():
    # T4 has a much higher peer endorsement than T1, but axiom tier must still
    # decide the winner — peer consensus cannot outrank the hierarchy of truth.
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL", confidence=0.9),
        _pack("pharmacology_agent", "RESISTANT", "T4_PHARMACOLOGICAL", confidence=0.85),
    ]
    resolver = AxiomResolver()
    result = resolver.resolve(packs, peer_endorsements={"genomics_agent": 0.1, "pharmacology_agent": 0.9})
    assert result.verdict == Verdict.SENSITIVE
    assert result.winning_agent == "genomics_agent"


def test_resolver_result_includes_axiom_used():
    packs = [
        _pack("genomics_agent", "SENSITIVE", "T1_STRUCTURAL"),
        _pack("pharmacology_agent", "RESISTANT", "T4_PHARMACOLOGICAL"),
    ]
    resolver = AxiomResolver()
    result = resolver.resolve(packs)
    assert "T1" in result.axiom_applied
