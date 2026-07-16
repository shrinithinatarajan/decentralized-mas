"""Run all 5 CTRPv2 held-out sets through the debate system and report mean ± std.

Usage:
    python experiments/run_ctrp_validation.py

Outputs:
    experiments/results/ctrp_validation.json
    experiments/results/traces_ctrp_set_{N}.jsonl
"""
import asyncio
import json
import math
import statistics
from pathlib import Path

import os

os.environ.setdefault("VERTEX_PROJECT", "project-d3bf2d5b-3451-46fd-8f3")

from src.agents.genomics_agent import GenomicsAgent
from src.agents.transcriptomics_agent import TranscriptomicsAgent
from src.agents.pharmacology_agent import PharmacologyAgent
from src.agents.pathway_agent import PathwayAgent
from src.data.loader import Case, load_ctrp_cases
from src.evaluation.metrics import evaluate
from src.llm.client import LLMClient, make_rate_limiter
from src.orchestrator import Orchestrator

MODEL = "vertex:gemini-3.1-flash-lite"
DATA = Path("src/data/processed")
RESULTS = Path("experiments/results")
SET_FILES = [Path(f"data/cases/cases_held_out_ctrp_{i}.yaml") for i in range(1, 6)]


def _mcp_apps():
    from src.mcp_servers.genomics_server import mcp as g
    from src.mcp_servers.transcriptomics_server import mcp as t
    from src.mcp_servers.pharmacology_server import mcp as p
    from src.mcp_servers.pathway_server import mcp as pw
    return g, t, p, pw


async def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    limiter = make_rate_limiter()
    client = LLMClient(model=MODEL, cache_db=DATA / "llm_cache.db", rate_limiter=limiter)

    g_app, t_app, p_app, pw_app = _mcp_apps()

    out_path = RESULTS / "ctrp_validation.json"
    all_output = json.loads(out_path.read_text()) if out_path.exists() else {}
    set_metrics = []

    for i, set_path in enumerate(SET_FILES, 1):
        key = f"set_{i}"
        if key in all_output and "auroc" in all_output[key]:
            print(f"\n=== Set {i}/5: already done (AUROC={all_output[key]['auroc']:.3f}), skipping ===")
            from src.evaluation.metrics import EvaluationMetrics
            set_metrics.append(EvaluationMetrics(**all_output[key]))
            continue

        print(f"\n=== Set {i}/5: {set_path.name} ===", flush=True)
        cases = load_ctrp_cases(set_path)
        print(f"  {len(cases)} cases  ({sum(c.label=='SENSITIVE' for c in cases)}S / {sum(c.label=='RESISTANT' for c in cases)}R)")

        agents = [
            GenomicsAgent(g_app, client),
            TranscriptomicsAgent(t_app, client),
            PharmacologyAgent(p_app, client),
            PathwayAgent(pw_app, client),
        ]
        orch = Orchestrator(agents=agents)
        traces_path = RESULTS / f"traces_ctrp_set_{i}.jsonl"

        results = await orch.run_all(cases, traces_path=traces_path)
        m = evaluate(results, cases)
        set_metrics.append(m)
        all_output[key] = {
            "auroc": m.auroc, "auprc": m.auprc,
            "cohens_kappa": m.cohens_kappa, "spearman_rho": m.spearman_rho,
            "n_evaluated": m.n_evaluated,
        }
        out_path.write_text(json.dumps(all_output, indent=2))
        print(f"  AUROC={m.auroc:.3f}  AUPRC={m.auprc:.3f}  κ={m.cohens_kappa:.3f}  ρ={m.spearman_rho:.3f}  n={m.n_evaluated}")

    # aggregate
    def _agg(attr):
        vals = [getattr(m, attr) for m in set_metrics if not math.isnan(getattr(m, attr))]
        if not vals:
            return float("nan"), float("nan")
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return mean, std

    print("\n" + "=" * 60)
    print(f"  Model: {MODEL}  |  Sets: {len(set_metrics)}")
    print("=" * 60)
    summary = {}
    for attr, label in [("auroc", "AUROC"), ("auprc", "AUPRC"), ("cohens_kappa", "κ"), ("spearman_rho", "ρ")]:
        mean, std = _agg(attr)
        summary[attr] = {"mean": mean, "std": std}
        print(f"  {label:>6}: {mean:.3f} ± {std:.3f}")
    print("=" * 60)

    all_output["summary"] = summary
    out_path = RESULTS / "ctrp_validation.json"
    out_path.write_text(json.dumps(all_output, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
