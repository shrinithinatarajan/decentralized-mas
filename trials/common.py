"""Shared setup for the trials/ experiment ladder (isolated agents -> consensus -> debate).

All three experiments run on the SAME fixed 20-case dataset (trials/dataset.json, built by
trials/build_dataset.py) so results are directly comparable. Every case in that dataset was
verified to have real data in all 4 source DBs (genomics, transcriptomics, pharmacology,
pathways) — see build_dataset.py — so isolated-agent performance in Experiment 1 reflects
reasoning quality, not missing evidence.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("VERTEX_PROJECT", "project-d3bf2d5b-3451-46fd-8f3")

from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from src.agents.genomics_agent import GenomicsAgent
from src.agents.transcriptomics_agent import TranscriptomicsAgent
from src.agents.pharmacology_agent import PharmacologyAgent
from src.agents.pathway_agent import PathwayAgent
from src.data.loader import Case
from src.llm.client import LLMClient, make_rate_limiter

DATA    = Path("src/data/processed")
RESULTS = Path("trials/results")
DATASET = Path("trials/dataset.json")
MODEL   = "vertex:gemini-3.1-flash-lite"


def load_fixed_dataset() -> list[tuple[str, str, Case]]:
    """Return [(set_id, case_id, Case), ...] for the 20-case dataset shared by all 3 trials.

    Run `PYTHONPATH=. python trials/build_dataset.py` first if trials/dataset.json is missing.
    """
    if not DATASET.exists():
        raise FileNotFoundError(
            f"{DATASET} not found — run `PYTHONPATH=. python trials/build_dataset.py` first."
        )
    entries = json.loads(DATASET.read_text())
    return [
        (e["set"], e["case_id"], Case(cell_line=e["cell_line"], drug=e["drug"], label=e["label"]))
        for e in entries
    ]


def load_target_genes_map() -> dict[str, list[str]]:
    """case_id -> target_genes, precomputed by build_dataset.py (avoids re-deriving aliases)."""
    entries = json.loads(DATASET.read_text())
    return {e["case_id"]: e["target_genes"] for e in entries}


def mcp_apps():
    from src.mcp_servers.genomics_server import mcp as g
    from src.mcp_servers.transcriptomics_server import mcp as t
    from src.mcp_servers.pharmacology_server import mcp as p
    from src.mcp_servers.pathway_server import mcp as pw
    return g, t, p, pw


def make_client() -> LLMClient:
    limiter = make_rate_limiter()
    return LLMClient(model=MODEL, cache_db=DATA / "llm_cache.db", rate_limiter=limiter)


def make_agents(client: LLMClient) -> list:
    g_app, t_app, p_app, pw_app = mcp_apps()
    return [
        GenomicsAgent(g_app, client),
        TranscriptomicsAgent(t_app, client),
        PharmacologyAgent(p_app, client),
        PathwayAgent(pw_app, client, transcriptomics_mcp=t_app),
    ]


def sensitivity_score(verdict: str, confidence: float) -> float:
    """Map (verdict, confidence) to a single P(sensitive)-like score for AUROC/Spearman."""
    if verdict == "SENSITIVE":
        return confidence
    if verdict == "RESISTANT":
        return 1.0 - confidence
    return 0.5  # UNCERTAIN


def score_records(records: list[dict]) -> dict:
    """Compute accuracy / uncertain rate / AUROC / Spearman rho over a uniform record shape:
    {"true_label": ..., "verdict": ..., "confidence": ...}
    Used identically across all three trials so the numbers are comparable.
    """
    n = len(records)
    correct = sum(1 for r in records if r["verdict"] == r["true_label"])
    uncertain = sum(1 for r in records if r["verdict"] == "UNCERTAIN")
    wrong = n - correct - uncertain

    binary = [r for r in records if r["true_label"] in ("SENSITIVE", "RESISTANT") and r["verdict"] != "UNCERTAIN"]
    labels = [1 if r["true_label"] == "SENSITIVE" else 0 for r in binary]
    scores = [sensitivity_score(r["verdict"], r["confidence"]) for r in binary]

    auroc = roc_auc_score(labels, scores) if len(set(labels)) == 2 else float("nan")
    rho = spearmanr(scores, labels).statistic if len(binary) >= 2 and len(set(labels)) == 2 else float("nan")

    return {
        "n": n,
        "correct": correct,
        "wrong": wrong,
        "uncertain": uncertain,
        "accuracy_all": correct / n if n else float("nan"),
        "accuracy_definitive": correct / (correct + wrong) if (correct + wrong) else float("nan"),
        "uncertain_rate": uncertain / n if n else float("nan"),
        "auroc": auroc,
        "spearman_rho": rho,
    }


def print_summary(title: str, summary: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  n={summary['n']}  correct={summary['correct']}  wrong={summary['wrong']}  uncertain={summary['uncertain']}")
    print(f"  Accuracy (all, UNCERTAIN=wrong):  {summary['accuracy_all']:.1%}")
    print(f"  Accuracy (definitive only):       {summary['accuracy_definitive']:.1%}")
    print(f"  Uncertain rate:                   {summary['uncertain_rate']:.1%}")
    print(f"  AUROC:                            {summary['auroc']:.3f}")
    print(f"  Spearman rho:                     {summary['spearman_rho']:.3f}")
