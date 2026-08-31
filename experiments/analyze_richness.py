"""Data-richness stratification analysis.

For each case in the full trace files, computes a richness score from three sources:
  civic_hits   — number of CIViC entries for target genes × drug
  ic50_exists  — whether CTRP has IC50 data for this cell line + drug
  depmap_hit   — whether DepMap has a decisive Chronos signal (≤ -0.5) for any target gene

Splits cases into data-rich (score ≥ 2) and data-sparse (score < 2) and reports
AUROC, Spearman ρ, and accuracy for each tier separately.

Usage:
    PYTHONPATH=. python experiments/analyze_richness.py
"""
import json
import sqlite3
from pathlib import Path

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

DATA = Path("src/data/processed")
RESULTS = Path("experiments/results")


def _civic_hits(genes: list[str], drug: str) -> int:
    """Count CIViC predictive-evidence rows matching any target gene × drug."""
    if not genes:
        return 0
    con = sqlite3.connect(DATA / "civic.db")
    drug_norm = drug.upper()
    placeholders = ",".join("?" * len(genes))
    rows = con.execute(
        f"SELECT COUNT(*) FROM predictive_evidence "
        f"WHERE gene_norm IN ({placeholders}) AND drug_norm = ?",
        [g.upper() for g in genes] + [drug_norm],
    ).fetchone()
    con.close()
    return rows[0] if rows else 0


def _ic50_exists(cell_line: str, drug: str) -> bool:
    """Return True if CTRP has any IC50 row for this cell line + drug."""
    con = sqlite3.connect(DATA / "pharmacology.db")
    row = con.execute(
        "SELECT 1 FROM drug_response WHERE cell_line = ? AND drug = ? LIMIT 1",
        (cell_line, drug),
    ).fetchone()
    con.close()
    return row is not None


def _depmap_hit(cell_line: str, genes: list[str]) -> bool:
    """Return True if DepMap has Chronos ≤ -0.5 for any target gene in this cell line."""
    if not genes:
        return False
    norm = cell_line.upper().replace("-", "").replace(" ", "").replace("_", "")
    con = sqlite3.connect(DATA / "depmap.db")
    placeholders = ",".join("?" * len(genes))
    row = con.execute(
        f"SELECT 1 FROM chronos_scores "
        f"WHERE cell_line_norm = ? AND gene IN ({placeholders}) AND chronos <= -0.5 LIMIT 1",
        [norm] + [g.upper() for g in genes],
    ).fetchone()
    con.close()
    return row is not None


def richness_score(cell_line: str, drug: str, target_genes: list[str] | None) -> dict:
    genes = target_genes or []
    civic  = _civic_hits(genes, drug)
    ic50   = _ic50_exists(cell_line, drug)
    depmap = _depmap_hit(cell_line, genes)
    score  = (1 if civic > 0 else 0) + (1 if ic50 else 0) + (1 if depmap else 0)
    return {"civic_hits": civic, "ic50_exists": ic50, "depmap_hit": depmap, "score": score}


def auroc_and_rho(records):
    labels, scores, preds, trues = [], [], [], []
    for r in records:
        gt   = r["true_label"]
        pred = r["final_verdict"]
        conf = r["final_confidence"]
        if gt not in ("SENSITIVE", "RESISTANT") or pred == "UNCERTAIN":
            continue
        y = 1 if gt == "SENSITIVE" else 0
        s = conf if pred == "SENSITIVE" else 1 - conf
        labels.append(y); scores.append(s)
        trues.append(gt); preds.append(pred)
    if len(set(labels)) < 2:
        return dict(n=len(labels), auroc=float("nan"), rho=float("nan"), accuracy=float("nan"))
    auroc = roc_auc_score(labels, scores)
    rho   = spearmanr(scores, labels).statistic
    acc   = sum(t == p for t, p in zip(trues, preds)) / len(trues)
    return dict(n=len(labels), auroc=auroc, rho=rho, accuracy=acc)


def main():
    # Load all available full-set traces
    records = []
    for f in sorted(RESULTS.glob("traces_full_set_*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} total records from {len(list(RESULTS.glob('traces_full_set_*.jsonl')))} sets\n")

    # Compute richness for each record
    rich, sparse = [], []
    detail_rows = []
    for r in records:
        cell   = r["cell_line"]
        drug   = r["drug"]
        genes  = r.get("target_genes") or []
        rd     = richness_score(cell, drug, genes)
        r["_richness"] = rd
        tier = "rich" if rd["score"] >= 2 else "sparse"
        (rich if tier == "rich" else sparse).append(r)
        detail_rows.append((cell, drug, rd["score"], rd["civic_hits"], rd["ic50_exists"], rd["depmap_hit"], r["final_verdict"], r["true_label"], r.get("correct")))

    # Print per-case detail
    print(f"{'Cell line':<20} {'Drug':<22} {'Score'} {'CIViC':>5} {'IC50':>5} {'DepMap':>6}  {'pred':<10} {'gt':<10} ✓/✗")
    print("-" * 100)
    for row in detail_rows:
        cell, drug, score, civic, ic50, dep, pred, gt, correct = row
        tier_tag = "[RICH]  " if score >= 2 else "[SPARSE]"
        mark = "✓" if correct else ("?" if pred == "UNCERTAIN" else "✗")
        print(f"{tier_tag} {cell:<18} {drug:<22} {score}     {civic:>3}   {'Y' if ic50 else 'N':>4}   {'Y' if dep else 'N':>5}  {pred:<10} {gt:<10} {mark}")

    # Summary metrics by tier
    print()
    print("=" * 60)
    all_m   = auroc_and_rho(records)
    rich_m  = auroc_and_rho(rich)
    sparse_m = auroc_and_rho(sparse)

    print(f"\n{'Tier':<12} {'n':>4}  {'Accuracy':>9}  {'AUROC':>7}  {'Spearman ρ':>10}")
    print("-" * 50)
    print(f"{'All':<12} {all_m['n']:>4}  {all_m['accuracy']:>9.3f}  {all_m['auroc']:>7.4f}  {all_m['rho']:>10.4f}")
    print(f"{'Data-rich':<12} {rich_m['n']:>4}  {rich_m['accuracy']:>9.3f}  {rich_m['auroc']:>7.4f}  {rich_m['rho']:>10.4f}")
    print(f"{'Data-sparse':<12} {sparse_m['n']:>4}  {sparse_m['accuracy']:>9.3f}  {sparse_m['auroc']:>7.4f}  {sparse_m['rho']:>10.4f}")
    print()
    print(f"Data-rich cases  (score ≥ 2): {len(rich)}")
    print(f"Data-sparse cases (score < 2): {len(sparse)}")


if __name__ == "__main__":
    main()
