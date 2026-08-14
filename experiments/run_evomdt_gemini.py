"""Run OpenEvoMDT (5-agent) on the 31 gold-standard cases using vertex:gemini-3.1-flash-lite.

Injects our LLMClient as the provider so OpenEvoMDT uses Vertex AI Gemini
instead of OpenRouter, with no extra auth setup.

Usage:
    PYTHONPATH=. python experiments/run_evomdt_gemini.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, TypeVar

os.environ.setdefault("VERTEX_PROJECT", "project-d3bf2d5b-3451-46fd-8f3")
os.environ.setdefault("VERTEX_LOCATION", "us-central1")

# Add OpenEvoMDT to path
sys.path.insert(0, "/tmp/OpenEvoMDT")

from evomdt.config import AppConfig
from evomdt.io import iter_dataset_tasks
from evomdt.llm import ChatMessage, LLMTrace, parse_json_content
from evomdt.pipeline import EvoMDTSystem

from src.llm.client import LLMClient, make_rate_limiter

MODEL    = "vertex:gemini-3.1-flash-lite"
BENCHMARK = Path("data/evomdt_benchmark.jsonl")
OUT      = Path("experiments/results/evomdt_raw_results.json")
CONFIG   = Path("/tmp/OpenEvoMDT/config.toml")

T = TypeVar("T")


class VertexGeminiProvider:
    """StructuredLLM implementation backed by our LLMClient (Vertex AI Gemini)."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def generate_structured(self, messages: list[ChatMessage], parser: Callable[[str], T]) -> LLMTrace:
        lm_messages = [{"role": m.role, "content": m.content} for m in messages]
        # Extract system message if present
        system = ""
        user_msgs = []
        for m in lm_messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_msgs.append(m)
        raw = await self._client.complete(messages=user_msgs, system=system)
        parsed = parser(raw)
        return LLMTrace(raw_content=raw, parsed=parsed, model_name=MODEL)


async def main() -> None:
    limiter = make_rate_limiter(calls_per_minute=30)
    client = LLMClient(
        model=MODEL,
        cache_db=Path("src/data/processed/llm_cache.db"),
        rate_limiter=limiter,
    )
    provider = VertexGeminiProvider(client)

    config = AppConfig.load(CONFIG)
    system = EvoMDTSystem(config, provider_factory=lambda _: provider)

    results: list[dict[str, Any]] = []
    gold_labels: dict[str, str] = {}
    for raw in BENCHMARK.read_text().strip().split("\n"):
        case_data = json.loads(raw)
        gold_labels[case_data["case_id"]] = case_data["reference_answer"]  # A or B

    cases = list(iter_dataset_tasks(BENCHMARK))
    print(f"Running {len(cases)} cases with {MODEL} via OpenEvoMDT...", flush=True)

    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case.case_id}", end="", flush=True)
        try:
            result = await system.run_case(case)
            final_answer = None
            if result.coordinator_decision and hasattr(result.coordinator_decision, "final_answer"):
                final_answer = result.coordinator_decision.final_answer
            elif result.agent_traces:
                for tr in reversed(result.agent_traces):
                    if tr.role == "coordinator" and hasattr(tr.parsed_output, "final_answer"):
                        final_answer = tr.parsed_output.final_answer
                        break
            gold = gold_labels.get(case.case_id)
            correct = (final_answer == gold) if final_answer and gold else None
            accuracy = result.evaluation.metrics.get("accuracy") if result.evaluation else None
            print(f" → {final_answer} (gold={gold}, {'✓' if correct else '✗'})", flush=True)
            results.append({
                "case_id": case.case_id,
                "predicted": final_answer,
                "gold": gold,
                "correct": correct,
                "accuracy": accuracy,
            })
        except Exception as e:
            print(f" → ERROR: {e}", flush=True)
            results.append({"case_id": case.case_id, "predicted": None, "gold": gold_labels.get(case.case_id), "correct": None, "error": str(e)})

    # Compute AUROC (binary A/B → SENSITIVE=1, RESISTANT=0)
    from sklearn.metrics import roc_auc_score, cohen_kappa_score, accuracy_score
    import numpy as np

    valid = [r for r in results if r["predicted"] and r["gold"]]
    y_true = np.array([1 if r["gold"] == "A" else 0 for r in valid])
    y_pred = np.array([1 if r["predicted"] == "A" else 0 for r in valid])
    auroc = float(roc_auc_score(y_true, y_pred)) if len(set(y_true)) > 1 else float("nan")
    kappa = float(cohen_kappa_score(y_true, y_pred))
    acc   = float(accuracy_score(y_true, y_pred))

    output = {
        "model": MODEL,
        "framework": "OpenEvoMDT",
        "n_cases": len(cases),
        "n_valid": len(valid),
        "auroc": round(auroc, 4),
        "kappa": round(kappa, 4),
        "accuracy": round(acc, 4),
        "per_case": results,
    }
    OUT.write_text(json.dumps(output, indent=2))
    print(f"\nAUROC={auroc:.4f}  κ={kappa:.4f}  acc={acc:.4f}  ({len(valid)}/{len(cases)} cases)")
    print(f"Results saved to {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
