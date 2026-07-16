"""Run the held-out test set (cases_held_out.yaml) and report metrics.

This is a clean evaluation: cell lines and cases were never seen during
development, prompt engineering, or any prior system tuning.

Usage:
    VERTEX_PROJECT=... LLM_CALLS_PER_MINUTE=8 \
    PYTHONPATH=. python experiments/run_held_out.py
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
from src.llm.client import LLMClient, make_rate_limiter
from src.orchestrator import Orchestrator

RESULTS = Path("experiments/results")
MODEL   = "vertex:gemini-3.1-flash-lite"
CACHE   = Path("src/data/processed/llm_cache.db")


def _mcp_apps():
    from src.mcp_servers.genomics_server import mcp as genomics_app
    from src.mcp_servers.transcriptomics_server import mcp as transcriptomics_app
    from src.mcp_servers.pharmacology_server import mcp as pharmacology_app
    from src.mcp_servers.pathway_server import mcp as pathway_app
    return genomics_app, transcriptomics_app, pharmacology_app, pathway_app


async def main(cases_file: str = "cases_held_out.yaml"):
    RESULTS.mkdir(parents=True, exist_ok=True)
    cases = load_cases(Path(cases_file))
    print(f"Loaded {len(cases)} held-out cases from {cases_file}.")

    limiter = make_rate_limiter()
    client  = LLMClient(model=MODEL, cache_db=CACHE, rate_limiter=limiter)
    genomics_app, transcriptomics_app, pharmacology_app, pathway_app = _mcp_apps()
    agents  = [
        GenomicsAgent(genomics_app, client),
        TranscriptomicsAgent(transcriptomics_app, client),
        PharmacologyAgent(pharmacology_app, client),
        PathwayAgent(pathway_app, client),
    ]
    orch = Orchestrator(agents=agents)
    traces = RESULTS / "traces_held_out.jsonl"
    results = await orch.run_all(cases, traces_path=traces)

    metrics = evaluate(results, cases)
    out = {
        "gemini-3.1-flash-lite (held-out)": {
            "auroc":         metrics.auroc,
            "auprc":         metrics.auprc,
            "spearman_rho":  metrics.spearman_rho,
            "cohens_kappa":  metrics.cohens_kappa,
            "n_evaluated":   metrics.n_evaluated,
        }
    }
    out_path = RESULTS / "held_out_metrics.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("\nHeld-out test metrics (never-seen cell lines):")
    print(f"  AUROC={metrics.auroc:.3f}  AUPRC={metrics.auprc:.3f}  "
          f"κ={metrics.cohens_kappa:.3f}  n={metrics.n_evaluated}")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    import sys
    f = sys.argv[1] if len(sys.argv) > 1 else "cases_held_out.yaml"
    asyncio.run(main(f))
