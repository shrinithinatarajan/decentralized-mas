from pathlib import Path
from typing import Callable

from src.data.loader import Case, load_cases
from src.evaluation.metrics import EvaluationMetrics, evaluate
from src.llm.client import LLMClient
from src.orchestrator import Orchestrator
from src.protocols.debate_engine import ConsensusResult

FREE_MODELS: dict[str, str] = {
    "gemini-flash": "gemini:gemini-1.5-flash",
    "qwen-32b": "groq:qwen-qwq-32b",
    "gemma-9b": "groq:gemma2-9b-it",
    "llama-70b": "groq:llama-3.3-70b-versatile",
}


async def run_comparison(
    model_map: dict[str, str],
    cases: list[Case],
    agent_factory: Callable[[LLMClient], list],
    cache_db: Path | None = None,
) -> dict[str, list[ConsensusResult]]:
    results: dict[str, list[ConsensusResult]] = {}
    for label, model_str in model_map.items():
        client = LLMClient(model=model_str, cache_db=cache_db)
        agents = agent_factory(client)
        orch = Orchestrator(agents=agents)
        results[label] = await orch.run_all(cases)
    return results


def evaluate_comparison(
    comparison_results: dict[str, list[ConsensusResult]],
    cases: list[Case],
) -> dict[str, EvaluationMetrics]:
    return {
        label: evaluate(results, cases)
        for label, results in comparison_results.items()
    }
