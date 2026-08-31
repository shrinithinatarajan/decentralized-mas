from enum import Enum


class AxiomTier(str, Enum):
    T1_STRUCTURAL = "T1_STRUCTURAL"
    T2_TRANSCRIPTIONAL_GATE = "T2_TRANSCRIPTIONAL_GATE"
    T3_PATHWAY_BYPASS = "T3_PATHWAY_BYPASS"
    T4_PHARMACOLOGICAL_PRIOR = "T4_PHARMACOLOGICAL_PRIOR"
    T5_STATISTICAL_CONSENSUS = "T5_STATISTICAL_CONSENSUS"


# Higher value = higher priority (T1 overrides all)
AXIOM_HIERARCHY: dict[AxiomTier, int] = {
    AxiomTier.T1_STRUCTURAL: 5,
    AxiomTier.T2_TRANSCRIPTIONAL_GATE: 4,
    AxiomTier.T3_PATHWAY_BYPASS: 3,
    AxiomTier.T4_PHARMACOLOGICAL_PRIOR: 2,
    AxiomTier.T5_STATISTICAL_CONSENSUS: 1,
}

SILENCING_THRESHOLD = -2.0
CONFIDENCE_DECAY_PER_FLIP = 0.15
MAX_DEBATE_ROUNDS = 3

# T1 must exceed this confidence to hard-override a unanimous lower-tier consensus.
# Below this, the resolver falls back to confidence-weighted voting across all agents.
T1_OVERRIDE_CONFIDENCE_THRESHOLD = 0.80


def _build_evidence_to_axiom_map():
    """Build a mapping from EvidenceTier to AxiomTier.

    Uses deferred import to avoid circular dependency.
    EvidenceTier and AxiomTier have different string values that need mapping
    (e.g. T2_TRANSCRIPTIONAL vs T2_TRANSCRIPTIONAL_GATE).
    """
    from src.schemas.evidence_pack import EvidenceTier
    return {
        EvidenceTier.T1_STRUCTURAL: AxiomTier.T1_STRUCTURAL,
        EvidenceTier.T2_TRANSCRIPTIONAL: AxiomTier.T2_TRANSCRIPTIONAL_GATE,
        EvidenceTier.T3_PATHWAY: AxiomTier.T3_PATHWAY_BYPASS,
        EvidenceTier.T4_PHARMACOLOGICAL: AxiomTier.T4_PHARMACOLOGICAL_PRIOR,
        EvidenceTier.T5_STATISTICAL: AxiomTier.T5_STATISTICAL_CONSENSUS,
    }


EVIDENCE_TO_AXIOM_TIER = _build_evidence_to_axiom_map()
