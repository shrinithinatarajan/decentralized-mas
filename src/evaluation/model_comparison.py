import asyncio
from pathlib import Path
from typing import Callable

from src.data.loader import Case, load_cases
from src.evaluation.metrics import EvaluationMetrics, evaluate
from src.llm.client import LLMClient, make_rate_limiter
from src.orchestrator import Orchestrator
from src.protocols.debate_engine import ConsensusResult

FREE_MODELS: dict[str, str] = {
    "llama-70b": "nim:meta/llama-3.3-70b-instruct",
    "llama-8b": "nim:meta/llama-3.1-8b-instruct",
    "mixtral-8x7b": "nim:mistralai/mixtral-8x7b-instruct-v0.1",
    "llama-70b-v31": "nim:meta/llama-3.1-70b-instruct",
    "gemma-3n-4b": "nim:google/gemma-3n-e4b-it",
    "nemotron-49b": "nim:nvidia/llama-3.3-nemotron-super-49b-v1",
    "llama-11b": "nim:meta/llama-3.2-11b-vision-instruct",
    "gemini-2.5-flash": "gemini:gemini-2.5-flash",
    "gemini-2.5-flash-lite": "gemini:gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite": "vertex:gemini-3.1-flash-lite",
    "glm-5.2": "nim:z-ai/glm-5.2",
    "minimax-m3": "nim:minimaxai/minimax-m3",
    "kimi-k2.6": "nim:moonshotai/kimi-k2.6",
}


async def run_comparison(
    model_map: dict[str, str],
    cases: list[Case],
    agent_factory: Callable[[LLMClient], list],
    cache_db: Path | None = None,
    results_path: Path | None = None,
) -> dict[str, list[ConsensusResult]]:
    import json as _json
    limiter = make_rate_limiter()
    results: dict[str, list[ConsensusResult]] = {}
    for label, model_str in model_map.items():
        print(f"\n=== Running model: {label} ({model_str}) ===", flush=True)
        client = LLMClient(model=model_str, cache_db=cache_db, rate_limiter=limiter)
        agents = agent_factory(client)
        orch = Orchestrator(agents=agents)
        traces_path = (results_path.parent / f"traces_{label}.jsonl") if results_path else None
        results[label] = await orch.run_all(cases, traces_path=traces_path)
        # save after each model so crashes don't lose completed work
        if results_path is not None:
            existing = _json.loads(results_path.read_text()) if results_path.exists() else {}
            existing[label] = vars(evaluate(results[label], cases))
            results_path.write_text(_json.dumps(existing, indent=2))
            print(f"  Saved {label} to {results_path}", flush=True)
        # cool down between models to avoid NIM rate limit accumulation
        await asyncio.sleep(30)
    return results


def evaluate_comparison(
    comparison_results: dict[str, list[ConsensusResult]],
    cases: list[Case],
    split: str | None = None,
) -> dict[str, EvaluationMetrics]:
    """Evaluate all models. Pass split='test' to restrict to held-out cases only."""
    gt = [c for c in cases if c.split == split] if split else cases
    return {
        label: evaluate(results, gt)
        for label, results in comparison_results.items()
    }
