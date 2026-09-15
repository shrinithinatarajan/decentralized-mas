"""Re-run random_axiom_order with multiple seeds to get a distribution.

Runs the random_axiom_order ablation condition 7 times with different random
seeds and reports mean ± SD for AUROC, AUPRC, kappa, and Spearman rho.
This turns the single-point estimate into a distribution, supporting the
claim that the specific axiom ordering (T1→T5) adds value over any arbitrary order.

Usage:
    PYTHONPATH=. python experiments/run_random_axiom_seeds.py

Outputs:
    experiments/results/random_axiom_seeds.json
"""
import asyncio
import json
import os
import sqlite3
import numpy as np
from pathlib import Path

os.environ.setdefault("VERTEX_PROJECT", "project-d3bf2d5b-3451-46fd-8f3")

from src.agents.genomics_agent import GenomicsAgent
from src.agents.transcriptomics_agent import TranscriptomicsAgent
from src.agents.pharmacology_agent import PharmacologyAgent
from src.agents.pathway_agent import PathwayAgent
from src.data.loader import load_ctrp_cases
from src.evaluation.ablation_runner import RandomAxiomResolver
from src.evaluation.metrics import evaluate
from src.llm.client import LLMClient, make_rate_limiter
from src.orchestrator import Orchestrator, _normalize_targets
from src.protocols.debate_engine import DebateEngine

MODEL   = "vertex:gemini-3.1-flash-lite"
DATA    = Path("src/data/processed")
RESULTS = Path("experiments/results")
CASES   = Path("data/cases/cases_gold_standard.yaml")

SEEDS = [0, 1, 2, 3, 4, 5, 6]


def _mcp_apps():
    from src.mcp_servers.genomics_server import mcp as g
    from src.mcp_servers.transcriptomics_server import mcp as t
    from src.mcp_servers.pharmacology_server import mcp as p
    from src.mcp_servers.pathway_server import mcp as pw
    return g, t, p, pw


def _get_target_genes(drug: str) -> list[str] | None:
    conn = sqlite3.connect(DATA / "pharmacology.db")
    row = conn.execute("SELECT target_genes FROM drug_info WHERE drug=?", (drug,)).fetchone()
    conn.close()
    if not row or not row[0]:
        return None
    return _normalize_targets([g.strip() for g in row[0].split(",")])


async def run_seed(seed: int, apps, client) -> dict:
    g_app, t_app, p_app, pw_app = apps
    agents = [
        GenomicsAgent(g_app, client),
        TranscriptomicsAgent(t_app, client),
        PharmacologyAgent(p_app, client),
        PathwayAgent(pw_app, client, transcriptomics_mcp=t_app),
    ]
    engine = DebateEngine(resolver=RandomAxiomResolver(seed=seed))
    orch   = Orchestrator(agents=agents, engine=engine)
    cases  = load_ctrp_cases(CASES)
    results, case_list = [], []
    for case in cases:
        tg = _get_target_genes(case.drug)
        r  = await orch.run_case(case.cell_line, case.drug, target_genes=tg)
        results.append(r)
        case_list.append(case)
    m = evaluate(results, case_list)
    return {
        "seed": seed,
        "auroc": m.auroc,
        "auprc": m.auprc,
        "cohens_kappa": m.cohens_kappa,
        "spearman_rho": m.spearman_rho,
        "n_total": m.n_total,
        "n_decisive": m.n_decisive,
        "coverage": m.coverage,
    }


async def main():
    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / "random_axiom_seeds.json"
    saved = json.loads(out_path.read_text()) if out_path.exists() else {"runs": []}
    done_seeds = {r["seed"] for r in saved["runs"]}

    limiter = make_rate_limiter()
    client  = LLMClient(model=MODEL, cache_db=DATA / "llm_cache.db", rate_limiter=limiter)
    apps    = _mcp_apps()

    for seed in SEEDS:
        if seed in done_seeds:
            print(f"  seed={seed}  [cached]")
            continue
        print(f"  seed={seed}  running 100 cases...", flush=True)
        result = await run_seed(seed, apps, client)
        saved["runs"].append(result)
        out_path.write_text(json.dumps(saved, indent=2))
        print(f"  seed={seed}  AUROC={result['auroc']:.3f}  κ={result['cohens_kappa']:.3f}  cov={result['coverage']:.0%}")

    # Compute summary statistics
    runs = saved["runs"]
    for metric in ["auroc", "auprc", "cohens_kappa", "spearman_rho"]:
        vals = [r[metric] for r in runs]
        saved[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
        saved[f"{metric}_std"]  = round(float(np.std(vals)), 4)
        saved[f"{metric}_min"]  = round(float(np.min(vals)), 4)
        saved[f"{metric}_max"]  = round(float(np.max(vals)), 4)

    out_path.write_text(json.dumps(saved, indent=2))

    print("\n" + "=" * 60)
    print("RANDOM AXIOM ORDER — SEED DISTRIBUTION")
    print("=" * 60)
    print(f"{'Metric':<15} {'Mean':>7} {'SD':>7} {'Min':>7} {'Max':>7}")
    print("-" * 50)
    for metric, label in [("auroc", "AUROC"), ("auprc", "AUPRC"),
                          ("cohens_kappa", "Kappa"), ("spearman_rho", "Spearmanρ")]:
        print(f"{label:<15} {saved[f'{metric}_mean']:>7.3f} "
              f"{saved[f'{metric}_std']:>7.3f} "
              f"{saved[f'{metric}_min']:>7.3f} "
              f"{saved[f'{metric}_max']:>7.3f}")

    full_auroc = 0.9165  # from ablation_mini.json
    mean_rand  = saved["auroc_mean"]
    std_rand   = saved["auroc_std"]
    print(f"\nFull system AUROC:          {full_auroc:.3f}")
    print(f"Random axiom AUROC mean±SD: {mean_rand:.3f} ± {std_rand:.3f}")
    print(f"Gap (full − random mean):   {full_auroc - mean_rand:+.3f}")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
