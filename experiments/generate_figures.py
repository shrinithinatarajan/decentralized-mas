"""Generate all paper figures from 100-case gold-standard results.

Usage:
    PYTHONPATH=. python experiments/generate_figures.py

Outputs (all saved to images/):
    resolution.png       - R1 / R2 / AxiomResolver breakdown
    winning_agent.png    - verdict-driving agent distribution
    calibration.png      - GCS confidence vs. binary outcome
    ablation-chart.png   - AUROC and kappa for all 14 conditions
    faithfulness.png     - Likert faithfulness per agent + winning agent
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRACES   = Path("experiments/results/traces_gold_standard.jsonl")
ABLATION = Path("experiments/results/ablation_mini.json")
QUAL     = Path("experiments/results/qualitative_scores.json")
OUT_DIR  = Path("images")
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})

COLOR_T1  = "#CC3B2E"
COLOR_T2  = "#9145B5"
COLOR_T3  = "#18A58D"
COLOR_T4  = "#E38B00"
COLOR_FULL = "#2457A6"
COLOR_CORRECT = "#2ECC71"
COLOR_WRONG   = "#E74C3C"


# ---------------------------------------------------------------------------
# Load traces
# ---------------------------------------------------------------------------
def load_traces() -> list[dict]:
    return [json.loads(l) for l in TRACES.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Fig 1: Resolution method distribution
# ---------------------------------------------------------------------------
def plot_resolution(traces: list[dict]) -> None:
    method_map = {
        "CONSENSUS_R1":      "Round 1 Consensus",
        "CONSENSUS_R2":      "Round 2 Consensus",
        "RESOLVER_TIEBREAK": "AxiomResolver",
        "RESOLVER_PRIORITY": "AxiomResolver",
    }
    counts: dict[str, int] = {}
    for t in traces:
        key = method_map.get(t["resolution_method"], t["resolution_method"])
        counts[key] = counts.get(key, 0) + 1

    labels = ["Round 1 Consensus", "Round 2 Consensus", "AxiomResolver"]
    values = [counts.get(l, 0) for l in labels]
    colors = [COLOR_FULL, COLOR_T2, COLOR_T4]
    n_total = sum(values)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.6,
                f"{v}\n({v/n_total*100:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of cases")
    ax.set_title("Resolution Method Distribution\n(100 gold-standard cases)", fontweight="bold")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="x", alpha=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "resolution.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved resolution.png  {counts}")


# ---------------------------------------------------------------------------
# Fig 2: Winning agent distribution
# ---------------------------------------------------------------------------
def plot_winning_agent(traces: list[dict]) -> None:
    agent_map = {
        "genomics_agent":       "T1 Genomics",
        "transcriptomics_agent": "T2 Transcriptomics",
        "pathway_agent":        "T3 Pathway",
        "pharmacology_agent":   "T4 Pharmacology",
    }
    counts: dict[str, int] = {}
    for t in traces:
        key = agent_map.get(t.get("winning_agent", ""), t.get("winning_agent", "Unknown"))
        counts[key] = counts.get(key, 0) + 1

    labels = ["T1 Genomics", "T2 Transcriptomics", "T3 Pathway", "T4 Pharmacology"]
    values = [counts.get(l, 0) for l in labels]
    colors = [COLOR_T1, COLOR_T2, COLOR_T3, COLOR_T4]
    n_total = len(traces)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.55, zorder=3)
    for bar, v in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.4,
                bar.get_y() + bar.get_height() / 2,
                f"{v}  ({v/n_total*100:.1f}%)",
                va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Number of cases")
    ax.set_title("Winning Agent Distribution\n(100 gold-standard cases)", fontweight="bold")
    ax.set_xlim(0, max(values) * 1.3)
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "winning_agent.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved winning_agent.png  {counts}")


# ---------------------------------------------------------------------------
# Fig 3: GCS calibration scatter
# ---------------------------------------------------------------------------
def plot_calibration(traces: list[dict]) -> None:
    correct_gcs, wrong_gcs = [], []
    for t in traces:
        gcs = t.get("final_confidence", t.get("final_gcs", None))
        if gcs is None:
            continue
        if t["correct"]:
            correct_gcs.append(float(gcs))
        else:
            wrong_gcs.append(float(gcs))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(correct_gcs, [1] * len(correct_gcs),
               c=COLOR_CORRECT, marker="o", s=60, alpha=0.7, label=f"Correct (n={len(correct_gcs)})", zorder=3)
    ax.scatter(wrong_gcs, [0] * len(wrong_gcs),
               c=COLOR_WRONG, marker="x", s=80, linewidths=1.8, label=f"Incorrect (n={len(wrong_gcs)})", zorder=3)

    ax.set_xlabel("Grounded Confidence Score (GCS)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Incorrect", "Correct"])
    ax.set_xlim(0, 1.05)
    ax.set_title("GCS Confidence vs. Prediction Outcome\n(100 gold-standard cases)", fontweight="bold")
    ax.legend(framealpha=0.9, loc="center left")
    ax.grid(axis="x")
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "calibration.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved calibration.png  correct={len(correct_gcs)}  wrong={len(wrong_gcs)}")


# ---------------------------------------------------------------------------
# Fig 4: Ablation chart — AUROC and kappa
# ---------------------------------------------------------------------------
def plot_ablation(ablation: dict) -> None:
    # Display order: full system first, then architectural ablations, then single-agent
    order = [
        "full_system",
        "no_debate", "no_axioms", "random_axiom_order", "monolithic_llm", "no_mcp",
        "only_genomics", "only_transcriptomics", "only_pathway", "only_pharmacology",
        "no_genomics", "no_transcriptomics", "no_pathway", "no_pharmacology",
    ]
    labels = [
        "full_system",
        "no_debate", "no_axioms", "random_axiom_order", "monolithic_llm", "no_mcp",
        "only_genomics", "only_transcriptomics", "only_pathway", "only_pharmacology",
        "no_genomics", "no_transcriptomics", "no_pathway", "no_pharmacology",
    ]
    aurocs  = [ablation[k]["auroc"]         for k in order]
    kappas  = [ablation[k]["cohens_kappa"]   for k in order]

    x       = np.arange(len(order))
    width   = 0.38
    full_auroc = ablation["full_system"]["auroc"]
    full_kappa = ablation["full_system"]["cohens_kappa"]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars_a = ax.bar(x - width / 2, aurocs, width, label="AUROC", color=COLOR_FULL, alpha=0.85, zorder=3)
    bars_k = ax.bar(x + width / 2, kappas, width, label="Cohen's κ", color=COLOR_T4, alpha=0.85, zorder=3)

    ax.axhline(full_auroc, color=COLOR_FULL, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axhline(full_kappa, color=COLOR_T4,  linestyle="--", linewidth=1.2, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title(
        f"Ablation Study — AUROC and Cohen's κ (100 gold-standard cases)\n"
        f"Dashed lines: full system (AUROC={full_auroc:.3f}, κ={full_kappa:.3f})",
        fontweight="bold",
    )
    ax.legend(loc="upper right", framealpha=0.9)
    # Annotate full_system bars
    for bar in [bars_a[0], bars_k[0]]:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color="#333")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation-chart.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ablation-chart.png")


# ---------------------------------------------------------------------------
# Fig 5: Faithfulness (Likert per agent) + winning agent distribution
# ---------------------------------------------------------------------------
def plot_faithfulness(qual: dict, traces: list[dict]) -> None:
    per_agent = qual["metric_4_biological_faithfulness"]["per_agent"]
    agent_names = {
        "genomics_agent":        "T1 Genomics",
        "transcriptomics_agent": "T2 Transcriptomics",
        "pathway_agent":         "T3 Pathway",
        "pharmacology_agent":    "T4 Pharmacology",
    }
    keys   = ["genomics_agent", "transcriptomics_agent", "pathway_agent", "pharmacology_agent"]
    labels = [agent_names[k] for k in keys]
    scores = [per_agent[k] for k in keys]
    colors = [COLOR_T1, COLOR_T2, COLOR_T3, COLOR_T4]
    overall = qual["metric_4_biological_faithfulness"]["mean_score"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars = ax.bar(labels, scores, color=colors, width=0.55, zorder=3)
    ax.axhline(overall, color="#555", linestyle="--", linewidth=1.3,
               label=f"Overall mean = {overall:.3f}")
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.04,
                f"{s:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Mean Likert Faithfulness Score (1–5)")
    ax.set_title("Faithfulness per Agent\n(100 gold-standard cases)", fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(axis="x", alpha=0)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "faithfulness.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved faithfulness.png  overall={overall:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading data...")
    traces   = load_traces()
    ablation = json.loads(ABLATION.read_text())
    qual     = json.loads(QUAL.read_text())
    print(f"  Traces: {len(traces)} cases")

    print("Generating figures...")
    plot_resolution(traces)
    plot_winning_agent(traces)
    plot_calibration(traces)
    plot_ablation(ablation)
    plot_faithfulness(qual, traces)
    print("Done. All figures saved to images/")


if __name__ == "__main__":
    main()
