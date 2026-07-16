"""Run ablation variants across all 5 CTRPv2 held-out sets and report mean ± std.

Variants:
  full_system     -- full debate + axiom resolver (baseline, already in ctrp_validation.json)
  no_debate       -- majority vote, no debate
  no_axioms       -- debate with confidence-only resolver
  random_axioms   -- debate with randomly shuffled axiom hierarchy

Usage:
    python experiments/run_ablation_ctrp.py

Outputs:
    experiments/results/ablation_ctrp.json
"""
import asyncio
import json
import math
import os
import statistics
from pathlib import Path

os.environ.setdefault("VERTEX_PROJECT", "project-d3bf2d5b-3451-46fd-8f3")

from src.agents.genomics_agent import GenomicsAgent
from src.agents.transcriptomics_agent import TranscriptomicsAgent
from src.agents.pharmacology_agent import PharmacologyAgent
from src.agents.pathway_agent import PathwayAgent
from src.data.loader import Case, load_ctrp_cases
from src.evaluation.ablation_runner import AblationVariant, make_engine
from src.evaluation.metrics import evaluate, EvaluationMetrics
from src.llm.client import LLMClient, make_rate_limiter
from src.orchestrator import Orchestrator

MODEL = "vertex:gemini-3.1-flash-lite"
DATA = Path("src/data/processed")
RESULTS = Path("experiments/results")
SET_FILES = [Path(f"data/cases/cases_held_out_ctrp_{i}.yaml") for i in range(1, 6)]

VARIANTS = {
    "no_debate":     AblationVariant.NO_DEBATE,
    "no_axioms":     AblationVariant.NO_AXIOMS,
    "random_axioms": AblationVariant.RANDOM_AXIOM_ORDER,
}


def _mcp_apps():
    from src.mcp_servers.genomics_server import mcp as g
    from src.mcp_servers.transcriptomics_server import mcp as t
    from src.mcp_servers.pharmacology_server import mcp as p
    from src.mcp_servers.pathway_server import mcp as pw
    return g, t, p, pw


def _agg(metrics: list[EvaluationMetrics], attr: str):
    vals = [getattr(m, attr) for m in metrics if not math.isnan(getattr(m, attr))]
    if not vals:
        return float("nan"), float("nan")
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


async def run_variant(variant_name: str, variant: AblationVariant,
                      client: LLMClient, apps: tuple,
                      all_output: dict, out_path: Path) -> list[EvaluationMetrics]:
    g_app, t_app, p_app, pw_app = apps
    set_metrics = []

    for i, set_path in enumerate(SET_FILES, 1):
        key = f"{variant_name}_set_{i}"
        if key in all_output:
            print(f"  [Set {i}] already done (AUROC={all_output[key]['auroc']:.3f}), skipping")
            set_metrics.append(EvaluationMetrics(**all_output[key]))
            continue

        cases = load_ctrp_cases(set_path)
        engine = make_engine(variant)
        agents = [
            GenomicsAgent(g_app, client),
            TranscriptomicsAgent(t_app, client),
            PharmacologyAgent(p_app, client),
            PathwayAgent(pw_app, client),
        ]
        orch = Orchestrator(agents=agents, engine=engine)
        traces_path = RESULTS / f"traces_{variant_name}_set_{i}.jsonl"

        print(f"  [Set {i}/5] {set_path.name} ({len(cases)} cases)", flush=True)
        results = await orch.run_all(cases, traces_path=traces_path)
        m = evaluate(results, cases)
        set_metrics.append(m)
        all_output[key] = {
            "auroc": m.auroc, "auprc": m.auprc,
            "cohens_kappa": m.cohens_kappa, "spearman_rho": m.spearman_rho,
            "n_evaluated": m.n_evaluated,
        }
        out_path.write_text(json.dumps(all_output, indent=2))
        print(f"    AUROC={m.auroc:.3f}  AUPRC={m.auprc:.3f}  κ={m.cohens_kappa:.3f}  n={m.n_evaluated}")

    return set_metrics


async def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "ablation_ctrp.json"
    all_output = json.loads(out_path.read_text()) if out_path.exists() else {}

    limiter = make_rate_limiter()
    client = LLMClient(model=MODEL, cache_db=DATA / "llm_cache.db", rate_limiter=limiter)
    apps = _mcp_apps()

    summary_rows = []

    # Load full_system baseline from ctrp_validation.json
    baseline_path = RESULTS / "ctrp_validation.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        fs_metrics = [
            EvaluationMetrics(**baseline[f"set_{i}"])
            for i in range(1, 6) if f"set_{i}" in baseline
        ]
        auroc_m, auroc_s = _agg(fs_metrics, "auroc")
        auprc_m, auprc_s = _agg(fs_metrics, "auprc")
        kappa_m, kappa_s = _agg(fs_metrics, "cohens_kappa")
        summary_rows.append(("full_system", auroc_m, auroc_s, auprc_m, auprc_s, kappa_m, kappa_s))
        print(f"full_system (baseline): AUROC={auroc_m:.3f}±{auroc_s:.3f}  AUPRC={auprc_m:.3f}±{auprc_s:.3f}  κ={kappa_m:.3f}±{kappa_s:.3f}")

    for variant_name, variant in VARIANTS.items():
        print(f"\n=== Variant: {variant_name} ===", flush=True)
        set_metrics = await run_variant(variant_name, variant, client, apps, all_output, out_path)
        auroc_m, auroc_s = _agg(set_metrics, "auroc")
        auprc_m, auprc_s = _agg(set_metrics, "auprc")
        kappa_m, kappa_s = _agg(set_metrics, "cohens_kappa")
        summary_rows.append((variant_name, auroc_m, auroc_s, auprc_m, auprc_s, kappa_m, kappa_s))
        all_output[f"{variant_name}_summary"] = {
            "auroc": {"mean": auroc_m, "std": auroc_s},
            "auprc": {"mean": auprc_m, "std": auprc_s},
            "cohens_kappa": {"mean": kappa_m, "std": kappa_s},
        }
        out_path.write_text(json.dumps(all_output, indent=2))

    print("\n" + "=" * 72)
    print(f"  {'Variant':<20} {'AUROC':>14} {'AUPRC':>14} {'kappa':>14}")
    print("=" * 72)
    for row in summary_rows:
        name, am, as_, pm, ps, km, ks = row
        print(f"  {name:<20} {am:.3f}±{as_:.3f}   {pm:.3f}±{ps:.3f}   {km:.3f}±{ks:.3f}")
    print("=" * 72)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
