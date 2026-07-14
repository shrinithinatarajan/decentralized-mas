"""RED phase: tests for Grounded Confidence Score."""
import json
import pytest
from unittest.mock import AsyncMock, patch
import importlib

from src.schemas.evidence_pack import EvidenceTier, Verdict
from src.schemas.evidence_pack import Finding


# ── helpers ──────────────────────────────────────────────────────────────────

def _finding(biomarker: str) -> Finding:
    return Finding(biomarker=biomarker, value="v", interpretation="i", data_source="db")


# ── compute_h ────────────────────────────────────────────────────────────────

def test_h_all_biomarkers_found():
    from src.agents.gcs import compute_h
    findings = [_finding("BRAF"), _finding("EGFR")]
    evidence = {"mutations": [{"gene": "BRAF", "mutation": "V600E"}, {"gene": "EGFR", "mutation": "T790M"}]}
    assert compute_h(findings, evidence) == 1.0


def test_h_no_biomarkers_found():
    from src.agents.gcs import compute_h
    findings = [_finding("TP53"), _finding("KRAS")]
    evidence = {"mutations": [{"gene": "BRAF", "mutation": "V600E"}]}
    assert compute_h(findings, evidence) == 0.0


def test_h_partial_match():
    from src.agents.gcs import compute_h
    findings = [_finding("BRAF"), _finding("KRAS")]
    evidence = {"mutations": [{"gene": "BRAF", "mutation": "V600E"}]}
    assert compute_h(findings, evidence) == pytest.approx(0.5)


def test_h_empty_findings_returns_zero():
    from src.agents.gcs import compute_h
    assert compute_h([], {"mutations": [{"gene": "BRAF"}]}) == 0.0


def test_h_case_insensitive():
    from src.agents.gcs import compute_h
    findings = [_finding("braf")]
    evidence = {"mutations": [{"gene": "BRAF"}]}
    assert compute_h(findings, evidence) == 1.0


# ── compute_e ────────────────────────────────────────────────────────────────

def test_e_nonempty_list_returns_one():
    from src.agents.gcs import compute_e
    assert compute_e({"mutations": [{"gene": "BRAF"}]}) == 1.0


def test_e_nonempty_dict_value_returns_one():
    from src.agents.gcs import compute_e
    assert compute_e({"ic50": {"ln_ic50": -1.2, "label": "SENSITIVE"}}) == 1.0


def test_e_all_empty_lists_returns_zero():
    from src.agents.gcs import compute_e
    assert compute_e({"mutations": [], "cnv": []}) == 0.0


def test_e_none_values_returns_zero():
    from src.agents.gcs import compute_e
    assert compute_e({"ic50": None, "drug_info": None}) == 0.0


def test_e_empty_dict_returns_zero():
    from src.agents.gcs import compute_e
    assert compute_e({}) == 0.0


# ── tier weights ─────────────────────────────────────────────────────────────

def test_tier_weight_t1():
    from src.agents.gcs import tier_weight
    assert tier_weight(EvidenceTier.T1_STRUCTURAL) == 1.0


def test_tier_weight_t5():
    from src.agents.gcs import tier_weight
    assert tier_weight(EvidenceTier.T5_STATISTICAL) == 0.2


# ── compute_gcs ──────────────────────────────────────────────────────────────

def test_gcs_formula_all_perfect():
    """H=1, E=1, T=T1=1, S=1 → GCS=1.0"""
    from src.agents.gcs import compute_gcs
    findings = [_finding("BRAF")]
    evidence = {"mutations": [{"gene": "BRAF"}]}
    gcs = compute_gcs(findings, evidence, EvidenceTier.T1_STRUCTURAL, signal_strength=1.0)
    assert gcs == pytest.approx(1.0)


def test_gcs_empty_evidence_caps_at_060():
    """E=0 → GCS ≤ 0.60 regardless of other components."""
    from src.agents.gcs import compute_gcs
    findings = [_finding("BRAF")]
    evidence = {"mutations": [], "cnv": []}  # empty → E=0
    gcs = compute_gcs(findings, evidence, EvidenceTier.T1_STRUCTURAL, signal_strength=1.0)
    # H=1 (BRAF in evidence text?). BRAF not in empty lists → H=0. So:
    # GCS = 0*0.4 + 0*0.3 + 1.0*0.2 + 1.0*0.1 = 0.30
    assert gcs <= 0.60


def test_gcs_hallucinated_finding_lowers_score():
    """When H=0 (all fabricated), GCS is much lower than H=1."""
    from src.agents.gcs import compute_gcs
    evidence = {"mutations": [{"gene": "EGFR"}]}
    findings_real = [_finding("EGFR")]
    findings_hallucinated = [_finding("FABRICATED_GENE")]
    gcs_real = compute_gcs(findings_real, evidence, EvidenceTier.T2_TRANSCRIPTIONAL, 0.5)
    gcs_hallucinated = compute_gcs(findings_hallucinated, evidence, EvidenceTier.T2_TRANSCRIPTIONAL, 0.5)
    assert gcs_real > gcs_hallucinated


def test_gcs_weights():
    """Verify exact formula: H=0.5, E=1, T=T2=0.8, S=0.5."""
    from src.agents.gcs import compute_gcs
    findings = [_finding("BRAF"), _finding("FAKE")]
    evidence = {"expression": [{"gene": "BRAF", "z_score": 2.1}]}
    gcs = compute_gcs(findings, evidence, EvidenceTier.T2_TRANSCRIPTIONAL, signal_strength=0.5)
    expected = 0.5 * 0.40 + 1.0 * 0.30 + 0.8 * 0.20 + 0.5 * 0.10
    assert gcs == pytest.approx(expected, abs=1e-6)


def test_gcs_bounded_zero_to_one():
    from src.agents.gcs import compute_gcs
    findings = [_finding("X")]
    evidence = {"data": []}
    gcs = compute_gcs(findings, evidence, EvidenceTier.T5_STATISTICAL, signal_strength=0.0)
    assert 0.0 <= gcs <= 1.0


# ── integration: analyze() replaces LLM confidence with GCS ──────────────────

def _pack_json_with_confidence(confidence=0.99) -> str:
    return json.dumps({
        "agent_id": "genomics_agent",
        "cell_line": "A375",
        "drug": "Vemurafenib",
        "verdict": "SENSITIVE",
        "confidence": confidence,
        "evidence_tier": "T1_STRUCTURAL",
        "key_findings": [
            {"biomarker": "BRAF", "value": "V600E", "interpretation": "driver", "data_source": "genomics_db"}
        ],
        "caveats": [],
        "conflict_flags": [],
    })


@pytest.mark.asyncio
async def test_analyze_replaces_llm_confidence_with_gcs(genomics_db, monkeypatch):
    """After analyze(), pack.confidence must equal GCS, not LLM's self-reported 0.99."""
    monkeypatch.setenv("GENOMICS_DB", str(genomics_db))
    from src.mcp_servers import genomics_server
    importlib.reload(genomics_server)
    from src.agents.genomics_agent import GenomicsAgent

    with patch("src.agents.base_agent.LLMClient") as MockLLM:
        MockLLM.return_value.complete = AsyncMock(return_value=_pack_json_with_confidence(0.99))
        agent = GenomicsAgent(mcp_app=genomics_server.mcp)
        pack = await agent.analyze("A375", "Vemurafenib", target_genes=["BRAF"])

    # LLM reported 0.99 — that must have been overwritten by GCS
    assert pack.confidence != pytest.approx(0.99, abs=1e-3), \
        "LLM's self-reported confidence was NOT replaced by GCS"
    assert 0.0 <= pack.confidence <= 1.0


@pytest.mark.asyncio
async def test_analyze_gcs_higher_when_evidence_matches(genomics_db, monkeypatch):
    """GCS for BRAF finding backed by BRAF mutation data > GCS for fabricated gene."""
    monkeypatch.setenv("GENOMICS_DB", str(genomics_db))
    from src.mcp_servers import genomics_server
    importlib.reload(genomics_server)
    from src.agents.genomics_agent import GenomicsAgent

    real_pack_json = json.dumps({
        "agent_id": "genomics_agent", "cell_line": "A375", "drug": "Vemurafenib",
        "verdict": "RESISTANT", "confidence": 0.9, "evidence_tier": "T1_STRUCTURAL",
        "key_findings": [{"biomarker": "BRAF", "value": "V600E", "interpretation": "driver", "data_source": "db"}],
        "caveats": [], "conflict_flags": [],
    })
    hallucinated_pack_json = json.dumps({
        "agent_id": "genomics_agent", "cell_line": "A375", "drug": "Vemurafenib",
        "verdict": "RESISTANT", "confidence": 0.9, "evidence_tier": "T1_STRUCTURAL",
        "key_findings": [{"biomarker": "FABRICATED_GENE_XYZ", "value": "mut", "interpretation": "...", "data_source": "db"}],
        "caveats": [], "conflict_flags": [],
    })

    with patch("src.agents.base_agent.LLMClient") as MockLLM:
        MockLLM.return_value.complete = AsyncMock(return_value=real_pack_json)
        agent = GenomicsAgent(mcp_app=genomics_server.mcp)
        real_pack = await agent.analyze("A375", "Vemurafenib", target_genes=["BRAF"])

    with patch("src.agents.base_agent.LLMClient") as MockLLM:
        MockLLM.return_value.complete = AsyncMock(return_value=hallucinated_pack_json)
        agent2 = GenomicsAgent(mcp_app=genomics_server.mcp)
        hallucinated_pack = await agent2.analyze("A375", "Vemurafenib", target_genes=["BRAF"])

    assert real_pack.confidence > hallucinated_pack.confidence
