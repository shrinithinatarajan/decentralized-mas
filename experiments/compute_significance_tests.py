"""Statistical significance tests and bootstrap CIs for DMAS paper.

Produces:
  experiments/results/significance_tests.json

Computed here (no re-run needed):
  - Bootstrap 95% CIs on AUROC and kappa for full system
  - DeLong test: full system vs RF baseline (per-case confidence scores available)
  - McNemar test: full system vs RF baseline (per-case binary predictions)

Requires ablation per-case trace re-run (see run_mini_ablation.py --save-traces):
  - McNemar: full system vs no_debate, no_axioms, random_axiom_order
  - DeLong: full system vs monolithic_llm (need monolithic per-case confidences)

Usage:
    PYTHONPATH=. python experiments/compute_significance_tests.py
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score

RNG_SEED = 42
N_BOOTSTRAP = 10_000
RESULTS = Path("experiments/results")


# ── helpers ────────────────────────────────────────────────────────────────

def bootstrap_ci(y_true, scores, metric_fn, n=N_BOOTSTRAP, seed=RNG_SEED):
    """Return (mean, lower, upper) bootstrap CI for a scalar metric."""
    rng = np.random.default_rng(seed)
    vals = []
    y_true = np.array(y_true)
    scores = np.array(scores)
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            vals.append(metric_fn(y_true[idx], scores[idx]))
        except Exception:
            pass
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def delong_auc_ci(y_true, scores, n=N_BOOTSTRAP, seed=RNG_SEED):
    """Bootstrap CI on AUROC (DeLong approximation via bootstrap)."""
    return bootstrap_ci(y_true, scores, roc_auc_score, n=n, seed=seed)


def delong_compare(y_true, scores_a, scores_b, n=N_BOOTSTRAP, seed=RNG_SEED):
    """Bootstrap test: is AUROC(a) > AUROC(b)? Returns (delta, p_value, ci_lower, ci_upper)."""
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    deltas = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            da = roc_auc_score(y_true[idx], scores_a[idx])
            db = roc_auc_score(y_true[idx], scores_b[idx])
            deltas.append(da - db)
        except Exception:
            pass
    deltas = np.array(deltas)
    observed = roc_auc_score(y_true, scores_a) - roc_auc_score(y_true, scores_b)
    p_value = float(np.mean(deltas <= 0))  # one-sided: P(delta <= 0)
    return {
        "observed_delta": round(observed, 4),
        "bootstrap_mean_delta": round(float(np.mean(deltas)), 4),
        "ci_95": [round(float(np.percentile(deltas, 2.5)), 4), round(float(np.percentile(deltas, 97.5)), 4)],
        "p_one_sided": round(p_value, 4),
        "significant_p05": p_value < 0.05,
    }


def mcnemar_test(correct_a, correct_b):
    """McNemar test on paired binary correct/incorrect vectors.
    Returns (statistic, p_value, contingency_table).
    """
    correct_a = np.array(correct_a, dtype=bool)
    correct_b = np.array(correct_b, dtype=bool)
    # contingency: both right, a right b wrong, a wrong b right, both wrong
    both_right  = int(np.sum( correct_a &  correct_b))
    a_only      = int(np.sum( correct_a & ~correct_b))
    b_only      = int(np.sum(~correct_a &  correct_b))
    both_wrong  = int(np.sum(~correct_a & ~correct_b))
    # McNemar with continuity correction
    n_discordant = a_only + b_only
    if n_discordant == 0:
        return {"statistic": 0.0, "p_value": 1.0, "note": "no discordant pairs",
                "contingency": {"both_right": both_right, "a_only": a_only,
                                "b_only": b_only, "both_wrong": both_wrong}}
    chi2 = (abs(a_only - b_only) - 1) ** 2 / n_discordant
    p = float(stats.chi2.sf(chi2, df=1))
    return {
        "statistic": round(chi2, 4),
        "p_value": round(p, 4),
        "significant_p05": p < 0.05,
        "contingency": {"both_right": both_right, "a_only": a_only,
                        "b_only": b_only, "both_wrong": both_wrong},
    }


def cohens_kappa(y_true, y_pred):
    from sklearn.metrics import cohen_kappa_score
    decisive = [(t, p) for t, p in zip(y_true, y_pred) if p != "UNCERTAIN"]
    if not decisive:
        return float("nan")
    t, p = zip(*decisive)
    return cohen_kappa_score(t, p)


# ── load full system traces ────────────────────────────────────────────────

def load_full_system():
    cases = []
    with open(RESULTS / "traces_gold_standard.jsonl") as f:
        for line in f:
            cases.append(json.loads(line))
    y_true      = [1 if c["true_label"] == "SENSITIVE" else 0 for c in cases]
    scores      = [c["final_confidence"] if c["final_verdict"] == "SENSITIVE"
                   else 1 - c["final_confidence"] for c in cases]
    correct     = [c["correct"] for c in cases]
    verdicts    = [c["final_verdict"] for c in cases]
    case_ids    = [c["case_id"] for c in cases]
    return y_true, scores, correct, verdicts, case_ids


def load_rf_baseline(case_ids):
    """Align RF per-case predictions to the full-system case order."""
    data = json.loads((RESULTS / "rf_baseline.json").read_text())
    # build lookup by (cell_line, drug)
    lookup = {}
    for r in data["per_case"]:
        key = (r["cell_line"], r["drug"])
        lookup[key] = r
    # parse case_id: "mini_1:N:CELL_LINE:DRUG"
    aligned_correct = []
    aligned_scores  = []
    missing = []
    for cid in case_ids:
        parts = cid.split(":")
        cell_line, drug = parts[2], parts[3]
        r = lookup.get((cell_line, drug))
        if r is None:
            missing.append(cid)
            aligned_correct.append(None)
            aligned_scores.append(None)
        else:
            aligned_correct.append(r["correct"])
            # RF probability of SENSITIVE
            aligned_scores.append(r.get("prob_sensitive", 0.5))
    if missing:
        print(f"  Warning: {len(missing)} cases not found in RF baseline — excluded from paired tests")
    return aligned_correct, aligned_scores


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("Loading full system traces...")
    y_true, fs_scores, fs_correct, fs_verdicts, case_ids = load_full_system()
    n = len(y_true)
    print(f"  {n} cases loaded")

    results = {}

    # ── 1. Bootstrap CIs: full system ──────────────────────────────────────
    print("\nComputing bootstrap CIs for full system...")
    auroc_mean, auroc_lo, auroc_hi = delong_auc_ci(y_true, fs_scores)
    print(f"  AUROC: {auroc_mean:.4f} [{auroc_lo:.4f}, {auroc_hi:.4f}]")

    # Bootstrap kappa — use actual verdicts, bootstrap over case indices
    fs_correct_arr = np.array(fs_correct, dtype=float)
    rng = np.random.default_rng(RNG_SEED)
    kappa_vals = []
    y_true_arr = np.array(y_true)
    fs_verdicts_arr = np.array(fs_verdicts)
    from sklearn.metrics import cohen_kappa_score
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        yt = y_true_arr[idx]
        preds = fs_verdicts_arr[idx]
        decisive = [(t, p) for t, p in zip(yt, preds) if p != "UNCERTAIN"]
        if len(set(t for t, _ in decisive)) < 2 or len(set(p for _, p in decisive)) < 2:
            continue
        t_arr, p_arr = zip(*decisive)
        try:
            kappa_vals.append(cohen_kappa_score(
                ["SENSITIVE" if t == 1 else "RESISTANT" for t in t_arr], p_arr))
        except Exception:
            pass
    kappa_mean = float(np.mean(kappa_vals))
    kappa_lo   = float(np.percentile(kappa_vals, 2.5))
    kappa_hi   = float(np.percentile(kappa_vals, 97.5))
    print(f"  Kappa: {kappa_mean:.4f} [{kappa_lo:.4f}, {kappa_hi:.4f}]")

    results["full_system_bootstrap"] = {
        "n_cases": n,
        "n_bootstrap": N_BOOTSTRAP,
        "auroc": {"mean": round(auroc_mean, 4), "ci_95": [round(auroc_lo, 4), round(auroc_hi, 4)]},
        "cohens_kappa": {"mean": round(kappa_mean, 4), "ci_95": [round(kappa_lo, 4), round(kappa_hi, 4)]},
    }

    # ── 2. RF baseline comparison ───────────────────────────────────────────
    print("\nLoading RF baseline...")
    rf_correct, rf_scores = load_rf_baseline(case_ids)

    # Filter to cases present in both
    paired_idx = [i for i, (c, s) in enumerate(zip(rf_correct, rf_scores))
                  if c is not None and s is not None]
    print(f"  {len(paired_idx)} paired cases for RF comparison")

    y_true_p   = [y_true[i]    for i in paired_idx]
    fs_scores_p = [fs_scores[i] for i in paired_idx]
    fs_corr_p  = [fs_correct[i] for i in paired_idx]
    rf_scores_p = [rf_scores[i] for i in paired_idx]
    rf_corr_p  = [rf_correct[i] for i in paired_idx]

    print("  DeLong (bootstrap): full system vs RF...")
    delong_rf = delong_compare(y_true_p, fs_scores_p, rf_scores_p)
    print(f"    delta AUROC = {delong_rf['observed_delta']:+.4f}, p = {delong_rf['p_one_sided']:.4f}")

    print("  McNemar: full system vs RF...")
    mcnemar_rf = mcnemar_test(fs_corr_p, rf_corr_p)
    print(f"    chi2 = {mcnemar_rf['statistic']:.4f}, p = {mcnemar_rf['p_value']:.4f}")

    results["full_system_vs_rf"] = {
        "n_paired": len(paired_idx),
        "delong_auroc": delong_rf,
        "mcnemar_accuracy": mcnemar_rf,
    }

    # ── 3. Note: ablation McNemar/DeLong needs per-case traces ─────────────
    results["ablation_tests_note"] = (
        "McNemar and DeLong tests for no_debate, no_axioms, random_axiom_order "
        "require per-case trace files. Re-run run_mini_ablation.py with "
        "--save-traces flag to generate these, then extend this script."
    )

    # ── save ───────────────────────────────────────────────────────────────
    out = RESULTS / "significance_tests.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    # ── summary for manuscript ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MANUSCRIPT NUMBERS")
    print("=" * 60)
    fs = results["full_system_bootstrap"]
    print(f"Full system AUROC: {fs['auroc']['mean']:.3f} "
          f"(95% CI {fs['auroc']['ci_95'][0]:.3f}–{fs['auroc']['ci_95'][1]:.3f})")
    print(f"Full system κ:     {fs['cohens_kappa']['mean']:.3f} "
          f"(95% CI {fs['cohens_kappa']['ci_95'][0]:.3f}–{fs['cohens_kappa']['ci_95'][1]:.3f})")
    rf = results["full_system_vs_rf"]
    print(f"vs RF  ΔAUROC: {rf['delong_auroc']['observed_delta']:+.3f} "
          f"(p={rf['delong_auroc']['p_one_sided']:.3f}, "
          f"sig={rf['delong_auroc']['significant_p05']})")
    print(f"vs RF  McNemar: chi2={rf['mcnemar_accuracy']['statistic']:.3f}, "
          f"p={rf['mcnemar_accuracy']['p_value']:.3f}, "
          f"sig={rf['mcnemar_accuracy']['significant_p05']}")


if __name__ == "__main__":
    main()
