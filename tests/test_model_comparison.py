import pytest
from unittest.mock import AsyncMock

from src.protocols.debate_engine import ConsensusResult
from src.schemas.evidence_pack import EvidencePack, Verdict, EvidenceTier, Finding
from src.data.loader import Case
from src.llm.client import LLMClient
from src.evaluation.model_comparison import (
    FREE_MODELS,
    run_comparison,
    evaluate_comparison,
)


def _pack(agent_id, verdict="SENSITIVE"):
    return EvidencePack(
        agent_id=agent_id,
        cell_line="A375",
        drug="Vemurafenib",
        verdict=Verdict(verdict),
        confidence=0.8,
        evidence_tier=EvidenceTier.T1_STRUCTURAL,
        key_findings=[Finding(biomarker="BRAF", value="V600E", interpretation="driver", data_source="GDSC")],
    )


def _mock_agent(agent_id, verdict="SENSITIVE"):
    agent = AsyncMock()
    agent.analyze = AsyncMock(return_value=_pack(agent_id, verdict))
    return agent


def _factory(verdict="SENSITIVE"):
    """Returns an agent_factory that ignores the LLMClient and yields mock agents."""
    def factory(llm_client: LLMClient):
        return [
            _mock_agent("genomics_agent", verdict),
            _mock_agent("transcriptomics_agent", verdict),
            _mock_agent("pharmacology_agent", verdict),
            _mock_agent("pathway_agent", verdict),
        ]
    return factory


CASES = [
    Case(cell_line="A375", drug="Vemurafenib", label="SENSITIVE"),
    Case(cell_line="HCT116", drug="Oxaliplatin", label="RESISTANT"),
]


# --- FREE_MODELS ---

def test_free_models_contains_gemini_flash():
    assert any("gemini" in v for v in FREE_MODELS.values())


def test_free_models_contains_groq_qwen():
    assert any("qwen" in v for v in FREE_MODELS.values())


def test_free_models_contains_groq_gemma():
    assert any("gemma" in v for v in FREE_MODELS.values())


def test_free_models_has_four_entries():
    assert len(FREE_MODELS) == 4


def test_free_models_values_use_provider_prefix():
    for model_str in FREE_MODELS.values():
        assert ":" in model_str, f"{model_str!r} missing provider prefix"


# --- run_comparison ---

@pytest.mark.asyncio
async def test_run_comparison_returns_dict_keyed_by_label(tmp_path):
    model_map = {"model-a": "groq:gemma2-9b-it", "model-b": "groq:qwen-qwq-32b"}
    results = await run_comparison(model_map, CASES, _factory(), cache_db=tmp_path / "cache.db")
    assert set(results.keys()) == {"model-a", "model-b"}


@pytest.mark.asyncio
async def test_run_comparison_each_model_has_one_result_per_case(tmp_path):
    model_map = {"gemma": "groq:gemma2-9b-it"}
    results = await run_comparison(model_map, CASES, _factory(), cache_db=tmp_path / "cache.db")
    assert len(results["gemma"]) == len(CASES)


@pytest.mark.asyncio
async def test_run_comparison_results_are_consensus_results(tmp_path):
    model_map = {"gemma": "groq:gemma2-9b-it"}
    results = await run_comparison(model_map, CASES, _factory(), cache_db=tmp_path / "cache.db")
    assert all(isinstance(r, ConsensusResult) for r in results["gemma"])


@pytest.mark.asyncio
async def test_run_comparison_creates_separate_client_per_model(tmp_path):
    seen_models = []

    def tracking_factory(llm_client: LLMClient):
        seen_models.append(llm_client.model)
        return [_mock_agent("genomics_agent")]

    model_map = {"a": "groq:gemma2-9b-it", "b": "gemini:gemini-1.5-flash"}
    await run_comparison(model_map, CASES, tracking_factory, cache_db=tmp_path / "cache.db")
    assert set(seen_models) == {"groq:gemma2-9b-it", "gemini:gemini-1.5-flash"}


# --- evaluate_comparison ---

@pytest.mark.asyncio
async def test_evaluate_comparison_returns_metrics_per_model(tmp_path):
    from src.evaluation.metrics import EvaluationMetrics
    model_map = {"gemma": "groq:gemma2-9b-it", "qwen": "groq:qwen-qwq-32b"}
    comparison = await run_comparison(model_map, CASES, _factory(), cache_db=tmp_path / "cache.db")
    metrics = evaluate_comparison(comparison, CASES)
    assert set(metrics.keys()) == {"gemma", "qwen"}
    assert all(isinstance(m, EvaluationMetrics) for m in metrics.values())
