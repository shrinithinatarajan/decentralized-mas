"""Ablation: genomics agent alone on all 100 CTRP held-out cases.

Compares AUROC / Spearman against the full 4-agent debate system
(results in experiments/results/traces_full_set_*.jsonl).

Usage:
    PYTHONPATH=. python experiments/ablation_single_agent.py
"""
import asyncio
import json
import os
import sqlite3
from pathlib import Path
from collections import Counter

os.environ.setdefault("VERTEX_PROJECT", "project-d3bf2d5b-3451-46fd-8f3")

from src.agents.genomics_agent import GenomicsAgent
from src.data.loader import load_ctrp_cases
from src.llm.client import LLMClient, make_rate_limiter
from src.orchestrator import _normalize_targets
from src.mcp_servers.genomics_server import mcp as genomics_app

DATA    = Path("src/data/processed")
RESULTS = Path("experiments/results")
MODEL   = "vertex:gemini-3.1-flash-lite"

SETS = [
    (1, Path("data/cases/cases_held_out_ctrp_1.yaml")),
    (2, Path("data/cases/cases_held_out_ctrp_2.yaml")),
    (3, Path("data/cases/cases_held_out_ctrp_3.yaml")),
    (4, Path("data/cases/cases_held_out_ctrp_4.yaml")),
    (5, Path("data/cases/cases_held_out_ctrp_5.yaml")),
]


def _get_target_genes(drug: str) -> list[str] | None:
    conn = sqlite3.connect(DATA / "pharmacology.db")
    row = conn.execute(
        "SELECT target_genes FROM drug_info WHERE drug=?", (drug,)
    ).fetchone()
    conn.close()
    if not row or not row[0]:
        return None
    raw = [g.strip() for g in row[0].split(",")]
    return _normalize_targets(raw)


def auroc(y_true: list[int], y_score: list[float]) -> float:
    paired = sorted(zip(y_score, y_true), reverse=True)
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = auc = 0
    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            auc += tp
    return auc / (n_pos * n_neg)


def spearman(y_true: list, y_score: list) -> float:
    n = len(y_true)
    def rank_list(lst):
        order = sorted(range(n), key=lambda i: lst[i])
        ranks = [0] * n
        for rank, idx in enumerate(order):
            ranks[idx] = rank + 1
        return ranks
    ry, rs = rank_list(y_true), rank_list(y_score)
    d2 = sum((a - b) ** 2 for a, b in zip(ry, rs))
    return 1 - 6 * d2 / (n * (n ** 2 - 1))


async def main():
    RESULTS.mkdir(exist_ok=True)
    limiter = make_rate_limiter()
    client  = LLMClient(model=MODEL, cache_db=DATA / "llm_cache.db", rate_limiter=limiter)
    agent   = GenomicsAgent(genomics_app, client)

    records: list[dict] = []
    out_path = RESULTS / "ablation_genomics_only.jsonl"
    out_path.unlink(missing_ok=True)

    for set_num, yaml_path in SETS:
        cases = load_ctrp_cases(yaml_path)[:6]
        print(f"\n=== Set {set_num}: {yaml_path.name} — {len(cases)} cases ===")
        for i, case in enumerate(cases, 1):
            target_genes = _get_target_genes(case.drug)
            pack = await agent.analyze(case.cell_line, case.drug, target_genes)
            pred = pack.verdict.value
            conf = pack.confidence
            correct = pred == case.label
            match = "✓" if correct else ("?" if pred == "UNCERTAIN" else "✗")
            print(f"  [{i}/20] {case.cell_line} + {case.drug}")
            print(f"         {pred:<12} conf={conf:.2f}  gt={case.label}  {match}")
            rec = {
                "set": set_num,
                "cell_line": case.cell_line,
                "drug": case.drug,
                "target_genes": target_genes,
                "true_label": case.label,
                "final_verdict": pred,
                "final_confidence": conf,
                "correct": correct,
            }
            records.append(rec)
            with out_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    # ── metrics ──────────────────────────────────────────────────────────────
    y_true, scores = [], []
    for r in records:
        if r["true_label"] not in ("SENSITIVE", "RESISTANT"):
            continue
        y = 1 if r["true_label"] == "SENSITIVE" else 0
        pred, conf = r["final_verdict"], r["final_confidence"]
        score = conf if pred == "SENSITIVE" else (1 - conf if pred == "RESISTANT" else 0.5)
        y_true.append(y)
        scores.append(score)

    correct   = sum(r["correct"] for r in records)
    uncertain = sum(r["final_verdict"] == "UNCERTAIN" for r in records)
    wrong     = len(records) - correct - uncertain
    definitive = correct + wrong

    # compare against full debate system
    debate_records: list[dict] = []
    for i in range(1, 6):
        p = RESULTS / f"traces_full_set_{i}.jsonl"
        if p.exists():
            with open(p) as f:
                for line in f:
                    debate_records.append(json.loads(line))

    d_y, d_s = [], []
    for r in debate_records:
        if r["true_label"] not in ("SENSITIVE", "RESISTANT"):
            continue
        y = 1 if r["true_label"] == "SENSITIVE" else 0
        pred, conf = r["final_verdict"], r["final_confidence"]
        score = conf if pred == "SENSITIVE" else (1 - conf if pred == "RESISTANT" else 0.5)
        d_y.append(y)
        d_s.append(score)

    print(f"\n{'='*60}")
    print(f"  ABLATION: genomics agent only  |  n=100")
    print(f"  Accuracy (UNCERTAIN=wrong): {correct}/100 = {correct}%")
    print(f"  Accuracy (definitive only): {correct}/{definitive} = {correct/definitive:.1%}")
    print(f"  UNCERTAIN rate:             {uncertain}/100 = {uncertain}%")
    print(f"  AUROC:                      {auroc(y_true, scores):.3f}")
    print(f"  Spearman ρ:                 {spearman(y_true, scores):.3f}")
    if debate_records:
        print(f"\n  FULL DEBATE (4-agent):")
        d_correct   = sum(r["correct"] for r in debate_records)
        d_uncertain = sum(r["final_verdict"] == "UNCERTAIN" for r in debate_records)
        d_wrong     = len(debate_records) - d_correct - d_uncertain
        print(f"  Accuracy (definitive only): {d_correct}/{d_correct+d_wrong} = {d_correct/(d_correct+d_wrong):.1%}")
        print(f"  UNCERTAIN rate:             {d_uncertain}/100 = {d_uncertain}%")
        print(f"  AUROC:                      {auroc(d_y, d_s):.3f}")
        print(f"  Spearman ρ:                 {spearman(d_y, d_s):.3f}")
    print(f"{'='*60}")
    print(f"  Traces: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
