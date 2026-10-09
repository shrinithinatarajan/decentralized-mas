"""Derive a continuous resistance probability from per-agent GCS scores.

Each agent's GCS is sign-flipped by verdict direction (RESISTANT=+1, SENSITIVE=-1)
and aggregated using the same tier weights as the Hierarchy of Truth.
The resulting raw score ∈ [-1, +1] is passed through a sigmoid to yield a
resistance probability ∈ [0, 1].

This extends the binary SENSITIVE/RESISTANT output to a calibrated continuous score
suitable for generating reliability diagrams and ECE statistics.

Outputs:
  experiments/results/continuous_scores.json  — per-case scores + calibration data
  experiments/results/calibration_plot.png    — reliability diagram

Usage:
    PYTHONPATH=. python experiments/compute_continuous_score.py
"""
import json
import math
from pathlib import Path

from sklearn.metrics import roc_auc_score, average_precision_score

TRACES = Path("experiments/results/traces_gold_standard.jsonl")
OUT    = Path("experiments/results/continuous_scores.json")
PLOT   = Path("experiments/results/calibration_plot.png")

TIER_WEIGHTS = {
    "T1_STRUCTURAL": 1.0,
    "T2_TRANSCRIPTIONAL": 0.8,
    "T3_PATHWAY": 0.6,
    "T4_PHARMACOLOGICAL": 0.4,
    "T5_STATISTICAL": 0.2,
}
SIGMOID_K = 3.0


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-SIGMOID_K * x))


def resistance_prob(r1_verdicts: dict) -> float | None:
    """Compute tier-weighted signed GCS aggregate → resistance probability."""
    weighted_sum = 0.0
    weight_total = 0.0
    for v in r1_verdicts.values():
        verdict = v.get("verdict", "UNCERTAIN")
        conf    = float(v.get("confidence", 0.0))
        tier    = v.get("evidence_tier", "T5_STATISTICAL")
        w = TIER_WEIGHTS.get(tier, 0.2)
        if verdict == "RESISTANT":
            signed = conf
        elif verdict == "SENSITIVE":
            signed = -conf
        else:
            signed = 0.0
        weighted_sum += w * signed
        weight_total += w
    if weight_total == 0:
        return None
    return sigmoid(weighted_sum / weight_total)


def compute_ece(probs: list[float], labels: list[int], n_bins: int = 10) -> float:
    bin_size = 1.0 / n_bins
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = i * bin_size, (i + 1) * bin_size
        idx = [j for j, p in enumerate(probs) if lo <= p < hi]
        if not idx:
            continue
        avg_conf = sum(probs[j] for j in idx) / len(idx)
        avg_acc  = sum(labels[j] for j in idx) / len(idx)
        ece += (len(idx) / n) * abs(avg_conf - avg_acc)
    return round(ece, 4)


def calibration_bins(probs: list[float], labels: list[int], n_bins: int = 10) -> list[dict]:
    bin_size = 1.0 / n_bins
    bins = []
    for i in range(n_bins):
        lo, hi = i * bin_size, (i + 1) * bin_size
        idx = [j for j, p in enumerate(probs) if lo <= p < hi]
        if not idx:
            bins.append({"bin_low": round(lo, 2), "bin_high": round(hi, 2), "n": 0,
                         "avg_conf": None, "fraction_positive": None})
        else:
            bins.append({
                "bin_low": round(lo, 2),
                "bin_high": round(hi, 2),
                "n": len(idx),
                "avg_conf": round(sum(probs[j] for j in idx) / len(idx), 4),
                "fraction_positive": round(sum(labels[j] for j in idx) / len(idx), 4),
            })
    return bins


def main():
    records = [json.loads(l) for l in TRACES.open() if l.strip()]
    print(f"Loaded {len(records)} traces")

    # Diagnostic: understand trace format across records
    has_trace = sum(1 for r in records if r.get("trace"))
    has_r1agents = sum(1 for r in records if r.get("r1_agents"))
    sample = records[0] if records else {}
    print(f"  Records with 'trace' field   : {has_trace}")
    print(f"  Records with 'r1_agents' field: {has_r1agents}")
    print(f"  Top-level keys in record[0]  : {list(sample.keys())}")

    scores, labels, cases = [], [], []
    skip_label = skip_verdicts = skip_prob = 0
    for r in records:
        true_label = r.get("true_label", "")
        if true_label not in ("SENSITIVE", "RESISTANT"):
            skip_label += 1
            continue

        r1_verdicts = {}
        for rnd in r.get("trace", []):
            if rnd.get("round") == 1:
                r1_verdicts = rnd.get("verdicts", {})
                break
        if not r1_verdicts and r.get("r1_agents"):
            agents = r["r1_agents"]
            if agents and isinstance(agents[0], dict):
                # handle both {agent_id: ...} dict and [{agent_id: ..., ...}] list
                if "agent_id" in agents[0]:
                    r1_verdicts = {a["agent_id"]: a for a in agents}
                else:
                    # agents might be keyed differently — inspect first item
                    print(f"  r1_agents[0] keys: {list(agents[0].keys())}")
        if not r1_verdicts:
            skip_verdicts += 1
            continue

        prob = resistance_prob(r1_verdicts)
        if prob is None:
            skip_prob += 1
            continue

        label = 1 if true_label == "RESISTANT" else 0
        scores.append(prob)
        labels.append(label)
        cases.append({
            "cell_line": r["cell_line"],
            "drug": r["drug"],
            "true_label": true_label,
            "resistance_prob": round(prob, 4),
            "final_verdict": r.get("final_verdict"),
            "correct": r.get("correct"),
        })

    print(f"  Computed scores for {len(scores)} cases")

    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    ece   = compute_ece(scores, labels)
    bins  = calibration_bins(scores, labels)

    print(f"\n  Continuous AUROC : {auroc:.4f}")
    print(f"  Continuous AUPRC : {auprc:.4f}")
    print(f"  ECE              : {ece:.4f}")

    output = {
        "n_cases": len(scores),
        "continuous_auroc": round(auroc, 4),
        "continuous_auprc": round(auprc, 4),
        "ece": ece,
        "sigmoid_k": SIGMOID_K,
        "tier_weights": TIER_WEIGHTS,
        "calibration_bins": bins,
        "cases": cases,
    }
    OUT.write_text(json.dumps(output, indent=2))
    print(f"  Saved to {OUT}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        filled = [b for b in bins if b["n"] > 0]
        x = [b["avg_conf"] for b in filled]
        y = [b["fraction_positive"] for b in filled]
        sz = [max(30, b["n"] * 8) for b in filled]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.scatter(x, y, s=sz, color="#2457A6", alpha=0.85, zorder=3)
        ax.plot(x, y, color="#2457A6", lw=1.5, label=f"DMAS  (ECE = {ece:.3f})")
        ax.fill_between(x, x, y, alpha=0.08, color="#2457A6")
        ax.set_xlabel("Mean predicted resistance probability", fontsize=12)
        ax.set_ylabel("Fraction RESISTANT", fontsize=12)
        ax.set_title("Reliability Diagram — Tier-Weighted Resistance Probability", fontsize=12)
        ax.legend(fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.05, 0.92, f"AUROC = {auroc:.3f}", transform=ax.transAxes,
                fontsize=11, color="#2457A6")
        fig.tight_layout()
        fig.savefig(PLOT, dpi=150)
        print(f"  Calibration plot saved to {PLOT}")
    except ImportError:
        print("  matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
