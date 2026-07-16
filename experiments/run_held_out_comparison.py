"""Run all NIM models on the v2 held-out cases and compare with gemini-3.1-flash-lite.

Usage:
    NIM_API_KEY=... VERTEX_PROJECT=... LLM_CALLS_PER_MINUTE=8 \\
    PYTHONPATH=. python experiments/run_held_out_comparison.py

Outputs:
    experiments/results/held_out_v2_metrics.json
"""
import asyncio
import json
from pathlib import Path

from src.agents.genomics_agent import GenomicsAgent
from src.agents.transcriptomics_agent import TranscriptomicsAgent
from src.agents.pharmacology_agent import PharmacologyAgent
from src.agents.pathway_agent import PathwayAgent
from src.data.loader import load_cases
from src.evaluation.metrics import evaluate
from src.evaluation.model_comparison import FREE_MODELS
from src.llm.client import LLMClient, make_rate_limiter
from src.orchestrator import Orchestrator

CASES_FILE = Path("cases_held_out_v2.yaml")
RESULTS    = Path("experiments/results")
OUT_PATH   = RESULTS / "held_out_v2_metrics.json"
CACHE      = Path("src/data/processed/llm_cache.db")

NIM_MODELS = {k: v for k, v in FREE_MODELS.items() if v.startswith("nim:")}


def _mcp_apps():
    from src.mcp_servers.genomics_server import mcp as genomics_app
    from src.mcp_servers.transcriptomics_server import mcp as transcriptomics_app
    from src.mcp_servers.pharmacology_server import mcp as pharmacology_app
    from src.mcp_servers.pathway_server import mcp as pathway_app
    return genomics_app, transcriptomics_app, pharmacology_app, pathway_app


def make_agents(llm_client: LLMClient) -> list:
    genomics_app, transcriptomics_app, pharmacology_app, pathway_app = _mcp_apps()
    return [
        GenomicsAgent(genomics_app, llm_client),
        TranscriptomicsAgent(transcriptomics_app, llm_client),
        PharmacologyAgent(pharmacology_app, llm_client),
        PathwayAgent(pathway_app, llm_client),
    ]


def _print_table(all_metrics: dict) -> None:
    print("\n" + "="*65)
    print(f"  {'Model':<25}  {'AUROC':>6}  {'AUPRC':>6}  {'κ':>6}  {'n':>4}")
    print("  " + "-"*55)
    for label, m in sorted(all_metrics.items(), key=lambda x: x[1].get("auroc", 0), reverse=True):
        if "error" in m:
            print(f"  {label:<25}  ERROR: {m['error'][:30]}")
        else:
            print(f"  {label:<25}  {m['auroc']:>6.3f}  {m['auprc']:>6.3f}  {m['cohens_kappa']:>6.3f}  {m['n_evaluated']:>4}")
    print("="*65)


async def run_one(label: str) -> None:
    model_str = NIM_MODELS.get(label)
    if not model_str:
        print(f"Unknown model '{label}'. Available: {list(NIM_MODELS.keys())}")
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    cases = load_cases(CASES_FILE)

    all_metrics: dict = {}
    if OUT_PATH.exists():
        all_metrics = json.loads(OUT_PATH.read_text())
    # Always keep gemini baseline
    all_metrics.setdefault("gemini-3.1-flash-lite", {
        "auroc": 0.820, "auprc": 0.873, "cohens_kappa": 0.700, "n_evaluated": 20,
    })

    if label in all_metrics and "error" not in all_metrics[label]:
        print(f"[{label}] already done — skipping. Use --force to re-run.")
        _print_table(all_metrics)
        return

    print(f"=== {label} ({model_str}) — {len(cases)} cases ===", flush=True)
    limiter = make_rate_limiter()
    try:
        client  = LLMClient(model=model_str, cache_db=CACHE, rate_limiter=limiter)
        agents  = make_agents(client)
        orch    = Orchestrator(agents=agents)
        results = await orch.run_all(
            cases,
            traces_path=RESULTS / f"traces_held_out_v2_{label}.jsonl",
        )
        m = evaluate(results, cases)
        all_metrics[label] = {
            "auroc": round(m.auroc, 3),
            "auprc": round(m.auprc, 3),
            "cohens_kappa": round(m.cohens_kappa, 3),
            "n_evaluated": m.n_evaluated,
        }
        print(f"  AUROC={m.auroc:.3f}  AUPRC={m.auprc:.3f}  κ={m.cohens_kappa:.3f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        all_metrics[label] = {"error": str(e)}

    OUT_PATH.write_text(json.dumps(all_metrics, indent=2))
    _print_table(all_metrics)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python experiments/run_held_out_comparison.py <model_label>")
        print(f"Available: {list(NIM_MODELS.keys())}")
        sys.exit(1)
    asyncio.run(run_one(sys.argv[1]))
