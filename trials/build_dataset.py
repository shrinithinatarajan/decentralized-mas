"""Build the fixed 20-case dataset shared by all three trials.

Unlike cases_balanced_a/b.yaml (which were curated so ONE specific tier has strong
signal per case), this sampler requires that ALL FOUR agents have real, queryable
evidence for a case:
  - genomics:        a mutation OR cnv row for the cell line + a target gene
  - transcriptomics:  an expression row (non-null z_score) for the cell line + a target gene
  - pharmacology:     a drug_response row for the cell line + drug (IC50 data)
  - pathway:          at least one target gene maps to a pathway_id in pathways.db

This makes Experiment 1 (isolated agents) a fair comparison — no agent is
handicapped by missing data going in, so differences in accuracy reflect
reasoning quality, not data coverage.

Usage:
    PYTHONPATH=. python trials/build_dataset.py
Writes: trials/dataset.json
"""
import json
import random
import sqlite3
from pathlib import Path

from src.data.loader import load_cases, load_ctrp_cases, Case
from src.orchestrator import _normalize_targets

DATA = Path("src/data/processed")
OUT  = Path("trials/dataset.json")
SEED = 42
N_PER_SET = 10
N_SETS = 2

CASE_FILES_WRAPPED = [  # load_cases: top-level "cases:" key
    "data/cases/cases.yaml",
    "data/cases/cases_held_out.yaml",
    "data/cases/cases_held_out_v2.yaml",
]
CASE_FILES_FLAT = [  # load_ctrp_cases: flat list, may include target_genes
    "data/cases/cases_balanced_a.yaml",
    "data/cases/cases_balanced_b.yaml",
    "data/cases/cases_balanced_c.yaml",
    "data/cases/cases_held_out_ctrp_1.yaml",
    "data/cases/cases_held_out_ctrp_2.yaml",
    "data/cases/cases_held_out_ctrp_3.yaml",
    "data/cases/cases_held_out_ctrp_4.yaml",
    "data/cases/cases_held_out_ctrp_5.yaml",
]


def _target_genes(case: Case) -> list[str] | None:
    case_targets = getattr(case, "target_genes", None) or (
        [g.strip() for g in case.putative_target.split(",")] if case.putative_target else None
    )
    if case_targets:
        return _normalize_targets(case_targets)
    conn = sqlite3.connect(DATA / "pharmacology.db")
    row = conn.execute("SELECT target_genes FROM drug_info WHERE drug=?", (case.drug,)).fetchone()
    conn.close()
    if not row or not row[0]:
        return None
    return _normalize_targets([g.strip() for g in row[0].split(",")])


def _has_genomics(conn_g, cell_line: str, genes: list[str]) -> bool:
    q = "SELECT 1 FROM mutations WHERE cell_line=? AND gene=? UNION SELECT 1 FROM cnv WHERE cell_line=? AND gene=? LIMIT 1"
    return any(conn_g.execute(q, (cell_line, g, cell_line, g)).fetchone() for g in genes)


def _has_transcriptomics(conn_t, cell_line: str, genes: list[str]) -> bool:
    q = "SELECT 1 FROM expression WHERE cell_line=? AND gene=? AND z_score IS NOT NULL LIMIT 1"
    return any(conn_t.execute(q, (cell_line, g)).fetchone() for g in genes)


def _has_pharmacology(conn_p, cell_line: str, drug: str) -> bool:
    row = conn_p.execute("SELECT 1 FROM drug_response WHERE cell_line=? AND drug=?", (cell_line, drug)).fetchone()
    return row is not None


def _has_pathway(conn_pw, genes: list[str]) -> bool:
    q = "SELECT 1 FROM pathway_genes WHERE gene=? LIMIT 1"
    return any(conn_pw.execute(q, (g,)).fetchone() for g in genes)


def main():
    pool: dict[tuple[str, str], Case] = {}
    for path in CASE_FILES_WRAPPED:
        for c in load_cases(Path(path)):
            pool.setdefault((c.cell_line, c.drug), c)
    for path in CASE_FILES_FLAT:
        for c in load_ctrp_cases(Path(path)):
            pool.setdefault((c.cell_line, c.drug), c)
    print(f"Candidate pool: {len(pool)} unique (cell_line, drug) cases")

    conn_g  = sqlite3.connect(DATA / "genomics.db")
    conn_t  = sqlite3.connect(DATA / "transcriptomics.db")
    conn_p  = sqlite3.connect(DATA / "pharmacology.db")
    conn_pw = sqlite3.connect(DATA / "pathways.db")

    eligible: list[dict] = []
    for case in pool.values():
        if case.label not in ("SENSITIVE", "RESISTANT"):
            continue
        genes = _target_genes(case)
        if not genes:
            continue
        if not (
            _has_genomics(conn_g, case.cell_line, genes)
            and _has_transcriptomics(conn_t, case.cell_line, genes)
            and _has_pharmacology(conn_p, case.cell_line, case.drug)
            and _has_pathway(conn_pw, genes)
        ):
            continue
        eligible.append({
            "cell_line": case.cell_line, "drug": case.drug,
            "label": case.label, "target_genes": genes,
        })

    for c in (conn_g, conn_t, conn_p, conn_pw):
        c.close()

    print(f"Eligible (all 4 modalities have data): {len(eligible)}")
    if len(eligible) < N_PER_SET * N_SETS:
        raise SystemExit(
            f"Only {len(eligible)} cases have data in all 4 DBs; need {N_PER_SET * N_SETS}. "
            "Widen the candidate pool or relax a modality check."
        )

    # Stratified sample: balance SENSITIVE/RESISTANT within each set.
    rng = random.Random(SEED)
    sensitive = [c for c in eligible if c["label"] == "SENSITIVE"]
    resistant = [c for c in eligible if c["label"] == "RESISTANT"]
    rng.shuffle(sensitive)
    rng.shuffle(resistant)

    n_total = N_PER_SET * N_SETS
    half = n_total // 2
    chosen = sensitive[:half] + resistant[:n_total - half]
    if len(chosen) < n_total:
        raise SystemExit(f"Not enough class balance: {len(sensitive)} SENSITIVE, {len(resistant)} RESISTANT eligible.")
    rng.shuffle(chosen)

    sets = [chosen[i * N_PER_SET:(i + 1) * N_PER_SET] for i in range(N_SETS)]
    dataset = []
    for set_idx, cases in enumerate(sets):
        set_id = chr(ord("a") + set_idx)
        for i, c in enumerate(cases, 1):
            dataset.append({
                "set": set_id,
                "case_id": f"{set_id}:{i}:{c['cell_line']}:{c['drug']}",
                **c,
            })

    OUT.write_text(json.dumps(dataset, indent=2))
    print(f"\nWrote {len(dataset)} cases to {OUT}")
    for set_idx in range(N_SETS):
        set_id = chr(ord("a") + set_idx)
        set_cases = [d for d in dataset if d["set"] == set_id]
        n_sens = sum(1 for d in set_cases if d["label"] == "SENSITIVE")
        print(f"  set {set_id}: {len(set_cases)} cases ({n_sens} SENSITIVE / {len(set_cases) - n_sens} RESISTANT)")


if __name__ == "__main__":
    main()
