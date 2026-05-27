import pytest
from src.schemas.evidence_pack import EvidencePack, Finding, Verdict, EvidenceTier
from src.schemas.debate_message import DebateMessage, AxiomChallenge
from src.schemas.axiom_rules import AxiomTier, AXIOM_HIERARCHY


def test_evidence_pack_valid():
    pack = EvidencePack(
        agent_id="genomics_agent",
        cell_line="A375",
        drug="Vemurafenib",
        verdict=Verdict.SENSITIVE,
        confidence=0.85,
        evidence_tier=EvidenceTier.T1_STRUCTURAL,
        key_findings=[
            Finding(
                biomarker="BRAF_V600E",
                value="mutant",
                interpretation="Oncogenic driver present",
                data_source="GDSC_mutations",
                axiom_invoked="CENTRAL_DOGMA_DNA_PRIMACY",
            )
        ],
    )
    assert pack.verdict == Verdict.SENSITIVE
    assert pack.confidence == 0.85
    assert len(pack.key_findings) == 1


def test_evidence_pack_confidence_bounds():
    with pytest.raises(Exception):
        EvidencePack(
            agent_id="x", cell_line="A375", drug="V",
            verdict=Verdict.SENSITIVE, confidence=1.5,
            evidence_tier=EvidenceTier.T1_STRUCTURAL, key_findings=[],
        )


def test_axiom_challenge_valid():
    challenge = AxiomChallenge(
        challenger="transcriptomics_agent",
        target="genomics_agent",
        axiom=AxiomTier.T2_TRANSCRIPTIONAL_GATE,
        argument="BRAF silenced at RNA level",
        evidence={"gene": "BRAF", "z_score": -2.3},
        requested_action="REVISE_VERDICT_TO_RESISTANT",
    )
    assert challenge.axiom == AxiomTier.T2_TRANSCRIPTIONAL_GATE


def test_axiom_hierarchy_ordered():
    assert AXIOM_HIERARCHY[AxiomTier.T1_STRUCTURAL] > AXIOM_HIERARCHY[AxiomTier.T2_TRANSCRIPTIONAL_GATE]
    assert AXIOM_HIERARCHY[AxiomTier.T2_TRANSCRIPTIONAL_GATE] > AXIOM_HIERARCHY[AxiomTier.T3_PATHWAY_BYPASS]
