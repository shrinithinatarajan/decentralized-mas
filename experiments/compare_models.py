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

    cases = load_cases(Path("data/cases.yaml"))
    print(f"Loaded {len(cases)} cases.")

    factory = make_agent_factory()
    comparison = await run_comparison(
        FREE_MODELS,
        cases,
        factory,
        cache_db=DATA / "llm_cache.db",
    )

    metrics = evaluate_comparison(comparison, cases)

    # save metrics
    out = {label: vars(m) for label, m in metrics.items()}
    (RESULTS / "model_comparison.json").write_text(json.dumps(out, indent=2))
    print("Metrics:")
    for label, m in metrics.items():
        print(f"  {label}: AUROC={m.auroc:.3f}  AUPRC={m.auprc:.3f}  κ={m.cohens_kappa:.3f}")

    # ablation bar chart
    fig = plot_ablation_comparison(metrics)
    fig.savefig(FIGURES / "model_comparison.pdf", bbox_inches="tight")
    print(f"Figure saved to {FIGURES / 'model_comparison.pdf'}")


if __name__ == "__main__":
    asyncio.run(main())
