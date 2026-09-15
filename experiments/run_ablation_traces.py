"""Run no_debate, no_axioms, and random_axiom_order with per-case trace saving.

Saves per-case correct/incorrect predictions needed for McNemar tests.
Does NOT re-run conditions already in ablation_mini_traces.json.

Usage:
    PYTHONPATH=. python experiments/run_ablation_traces.py

Outputs:
    experiments/results/ablation_mini_traces.json
    experiments/results/significance_tests.json  (updated with McNemar results)
"""
import asyncio
import json
import os
import sqlite3
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score  # noqa: F401 used in DeLong block

os.environ.setdefault("VERTEX_PROJECT", "project-d3bf2d5b-3451-46fd-8f3")

from src.agents.genomics_agent import GenomicsAgent
from src.agents.transcriptomics_agent import TranscriptomicsAgent
from src.agents.pharmacology_agent import PharmacologyAgent
from src.agents.pathway_agent import PathwayAgent
from src.data.loader import load_ctrp_cases
from src.evaluation.ablation_runner import AblationVariant, make_engine, RandomAxiomResolver
from src.evaluation.metrics import evaluate
from src.llm.client import LLMClient, make_rate_limiter
from src.orchestrator import Orchestrator, _normalize_targets
from src.protocols.debate_engine import DebateEngine

MODEL   = "vertex:gemini-3.1-flash-lite"
DATA    = Path("src/data/processed")
RESULTS = Path("experiments/results")
CASES   = Path("data/cases/cases_gold_standard.yaml")

# Conditions to run with trace saving
TRACE_CONDITIONS = ["no_debate", "no_axioms", "random_axiom_order"]
# Use seed=0 for random_axiom_order (matches first seed from run_random_axiom_seeds.py)
RANDOM_SEED = 0


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


def _make_engine(label: str) -> DebateEngine:
    if label == "no_debate":
        return make_engine(AblationVariant.NO_DEBATE)
    if label == "no_axioms":
        return make_engine(AblationVariant.NO_AXIOMS)
    if label == "random_axiom_order":
        return DebateEngine(resolver=RandomAxiomResolver(seed=RANDOM_SEED))
    raise ValueError(label)


async def run_condition(label: str, apps, client) -> list[dict]:
    g_app, t_app, p_app, pw_app = apps
    agents = [
        GenomicsAgent(g_app, client),
        TranscriptomicsAgent(t_app, client),
        PharmacologyAgent(p_app, client),
        PathwayAgent(pw_app, client, transcriptomics_mcp=t_app),
    ]
    engine = _make_engine(label)
    orch   = Orchestrator(agents=agents, engine=engine)
    cases  = load_ctrp_cases(CASES)
    per_case = []
    for case in cases:
        tg = _get_target_genes(case.drug)
        r  = await orch.run_case(case.cell_line, case.drug, target_genes=tg)
        correct = (r.final_verdict == case.label) if r.final_verdict != "UNCERTAIN" else False
        per_case.append({
            "cell_line": case.cell_line,
            "drug": case.drug,
            "true_label": case.label,
            "verdict": r.final_verdict,
            "confidence": r.final_confidence,
            "correct": correct,
        })
    return per_case


def mcnemar_test(correct_a, correct_b):
    correct_a = [bool(x) for x in correct_a]
    correct_b = [bool(x) for x in correct_b]
    a_only   = sum(1 for a, b in zip(correct_a, correct_b) if a and not b)
    b_only   = sum(1 for a, b in zip(correct_a, correct_b) if not a and b)
    both_r   = sum(1 for a, b in zip(correct_a, correct_b) if a and b)
    both_w   = sum(1 for a, b in zip(correct_a, correct_b) if not a and not b)
    n_disc = a_only + b_only
    if n_disc == 0:
        return {"statistic": 0.0, "p_value": 1.0, "note": "no discordant pairs",
                "contingency": {"both_right": both_r, "a_only": a_only,
                                "b_only": b_only, "both_wrong": both_w}}
    chi2 = (abs(a_only - b_only) - 1) ** 2 / n_disc
    p = float(stats.chi2.sf(chi2, df=1))
    return {
        "statistic": round(chi2, 4),
        "p_value": round(p, 4),
        "significant_p05": p < 0.05,
        "contingency": {"both_right": both_r, "a_only": a_only,
                        "b_only": b_only, "both_wrong": both_w},
    }


async def main():
    RESULTS.mkdir(exist_ok=True)
    traces_path = RESULTS / "ablation_mini_traces.json"
    saved = json.loads(traces_path.read_text()) if traces_path.exists() else {}

    limiter = make_rate_limiter()
    client  = LLMClient(model=MODEL, cache_db=DATA / "llm_cache.db", rate_limiter=limiter)
    apps    = _mcp_apps()

    # Run each condition if not cached
    for label in TRACE_CONDITIONS:
        if label in saved:
            print(f"  {label:<24} [cached, {len(saved[label])} cases]")
            continue
        print(f"  {label:<24} running 100 cases...", flush=True)
        per_case = await run_condition(label, apps, client)
        saved[label] = per_case
        traces_path.write_text(json.dumps(saved, indent=2))
        n_correct = sum(1 for r in per_case if r["correct"])
        print(f"  {label:<24} accuracy={n_correct}/{len(per_case)}")

    # Load full system per-case results for paired comparison
    fs_correct = []
    with open(RESULTS / "traces_gold_standard.jsonl") as f:
        for line in f:
            t = json.loads(line)
            fs_correct.append(t["correct"])

    # McNemar tests
    print("\n" + "=" * 60)
    print("McNEMAR TESTS: full system vs ablation conditions")
    print("=" * 60)
    mcnemar_results = {}
    for label in TRACE_CONDITIONS:
        abl_correct = [r["correct"] for r in saved[label]]
        # Align lengths (both should be 100)
        n = min(len(fs_correct), len(abl_correct))
        result = mcnemar_test(fs_correct[:n], abl_correct[:n])
        mcnemar_results[f"full_system_vs_{label}"] = result
        sig = "*" if result["significant_p05"] else ""
        print(f"  vs {label:<22} chi2={result['statistic']:.3f}  p={result['p_value']:.3f}{sig}")
        cont = result["contingency"]
        print(f"    both_right={cont['both_right']}  fs_only={cont['a_only']}  "
              f"abl_only={cont['b_only']}  both_wrong={cont['both_wrong']}")

    # DeLong (bootstrap) on AUROC using per-case confidence scores
    print("\n" + "=" * 60)
    print("DeLong AUROC TESTS: full system vs ablation conditions")
    print("=" * 60)

    # Load full system scores
    fs_y_true, fs_scores_list = [], []
    with open(RESULTS / "traces_gold_standard.jsonl") as f:
        for line in f:
            t = json.loads(line)
            fs_y_true.append(1 if t["true_label"] == "SENSITIVE" else 0)
            score = t["final_confidence"] if t["final_verdict"] == "SENSITIVE" \
                    else 1 - t["final_confidence"]
            fs_scores_list.append(score)

    import numpy as np
    rng = np.random.default_rng(42)
    N_BOOT = 10_000
    delong_results = {}
    for label in TRACE_CONDITIONS:
        abl_cases = saved[label]
        n = min(len(fs_y_true), len(abl_cases))
        yt = np.array(fs_y_true[:n])
        fs_s = np.array(fs_scores_list[:n])
        abl_s = np.array([
            r["confidence"] if r["verdict"] == "SENSITIVE" else 1 - r["confidence"]
            for r in abl_cases[:n]
        ])
        observed_delta = roc_auc_score(yt, fs_s) - roc_auc_score(yt, abl_s)
        deltas = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            try:
                d = roc_auc_score(yt[idx], fs_s[idx]) - roc_auc_score(yt[idx], abl_s[idx])
                deltas.append(d)
            except Exception:
                pass
        deltas = np.array(deltas)
        p_one_sided = float(np.mean(deltas <= 0))
        ci = [round(float(np.percentile(deltas, 2.5)), 4),
              round(float(np.percentile(deltas, 97.5)), 4)]
        result = {
            "observed_delta_auroc": round(observed_delta, 4),
            "ci_95": ci,
            "p_one_sided": round(p_one_sided, 4),
            "significant_p05": p_one_sided < 0.05,
        }
        delong_results[f"full_system_vs_{label}"] = result
        sig = "*" if result["significant_p05"] else ""
        print(f"  vs {label:<22} ΔAUROC={observed_delta:+.3f}  "
              f"CI=[{ci[0]:.3f},{ci[1]:.3f}]  p={p_one_sided:.3f}{sig}")

    # Update significance_tests.json
    sig_path = RESULTS / "significance_tests.json"
    sig_data = json.loads(sig_path.read_text()) if sig_path.exists() else {}
    sig_data["ablation_mcnemar"] = mcnemar_results
    sig_data["ablation_delong"] = delong_results
    sig_data.pop("ablation_tests_note", None)
    sig_path.write_text(json.dumps(sig_data, indent=2))
    print(f"\nUpdated {sig_path}")


if __name__ == "__main__":
    asyncio.run(main())
