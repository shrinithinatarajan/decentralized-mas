"""Run all four free models over the GDSC2 dev cases and save results + figures.

Usage:
    export GROQ_API_KEY="gsk_..."
    export GEMINI_API_KEY="AIza..."
    python experiments/compare_models.py

Outputs:
    experiments/results/model_comparison.json
    experiments/figures/roc_comparison.pdf
    experiments/figures/ablation_comparison.pdf
"""

import asyncio
import json
from pathlib import Path

from src.agents.genomics_agent import GenomicsAgent
from src.agents.transcriptomics_agent import TranscriptomicsAgent
from src.agents.pharmacology_agent import PharmacologyAgent
from src.agents.pathway_agent import PathwayAgent
from src.data.loader import load_cases
from src.llm.client import LLMClient
from src.evaluation.model_comparison import FREE_MODELS, run_comparison, evaluate_comparison
from src.evaluation.visualize import plot_roc_curve, plot_ablation_comparison

DATA = Path("src/data/processed")
RESULTS = Path("experiments/results")
FIGURES = Path("experiments/figures")


def _mcp_apps():
    from src.mcp_servers.genomics_server import mcp as genomics_app
    from src.mcp_servers.transcriptomics_server import mcp as transcriptomics_app
    from src.mcp_servers.pharmacology_server import mcp as pharmacology_app
    from src.mcp_servers.pathway_server import mcp as pathway_app
    return genomics_app, transcriptomics_app, pharmacology_app, pathway_app


def make_agent_factory():
    genomics_app, transcriptomics_app, pharmacology_app, pathway_app = _mcp_apps()

    def factory(llm_client: LLMClient):
        return [
            GenomicsAgent(genomics_app, llm_client),
            TranscriptomicsAgent(transcriptomics_app, llm_client),
            PharmacologyAgent(pharmacology_app, llm_client),
            PathwayAgent(pathway_app, llm_client),
        ]
    return factory


async def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    cases = load_cases(Path("cases.yaml"))
    print(f"Loaded {len(cases)} cases.")

    factory = make_agent_factory()
    # skip models already saved or known to be unavailable
    SKIP = {"mixtral-8x7b", "gemma-3n-4b", "gemini-2.5-flash", "gemini-2.5-flash-lite"}  # quota limited
    existing = json.loads((RESULTS / "model_comparison.json").read_text()) if (RESULTS / "model_comparison.json").exists() else {}
    run_models = {k: v for k, v in FREE_MODELS.items() if k not in SKIP and k not in existing}
    comparison = await run_comparison(
        run_models,
        cases,
        factory,
        cache_db=DATA / "llm_cache.db",
        results_path=RESULTS / "model_comparison.json",
    )

    metrics = evaluate_comparison(comparison, cases)

    # merge with any previously saved results so partial runs accumulate
    out_path = RESULTS / "model_comparison.json"
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    existing.update({label: vars(m) for label, m in metrics.items()})
    out_path.write_text(json.dumps(existing, indent=2))
    print("Metrics:")
    for label, m in metrics.items():
        print(f"  {label}: AUROC={m.auroc:.3f}  AUPRC={m.auprc:.3f}  κ={m.cohens_kappa:.3f}")

    # plot using ALL saved results, not just current run
    from src.evaluation.metrics import EvaluationMetrics
    import math
    all_metrics = {
        label: EvaluationMetrics(**data)
        for label, data in existing.items()
        if not math.isnan(data.get("auroc", float("nan")))
    }
    fig = plot_ablation_comparison(all_metrics)
    fig.savefig(FIGURES / "model_comparison.pdf", bbox_inches="tight")
    print(f"Figure saved to {FIGURES / 'model_comparison.pdf'} ({len(all_metrics)} models)")


if __name__ == "__main__":
    asyncio.run(main())
