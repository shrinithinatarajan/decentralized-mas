"""Sample multiple non-overlapping held-out validation sets from CTRPv2.

Generates N balanced sets of 20 cases (10 SENSITIVE + 10 RESISTANT) drawn
from the intersection of CTRPv2 labels and our genomics/pharmacology DBs.
Sets are non-overlapping by (cell_line, drug) pair to allow averaged metrics.

Usage:
    PYTHONPATH=. python src/data/sample_held_out_ctrp.py --n-sets 5 --seed 42
    PYTHONPATH=. python src/data/sample_held_out_ctrp.py --n-sets 5 --seed 42 --dry-run

Outputs:
    data/cases/cases_held_out_ctrp_{1..N}.yaml
"""
import argparse
import random
import re
import sqlite3
from pathlib import Path

import yaml

CTRP_DB   = Path("src/data/processed/ground_truth_ctrp.db")
PHARM_DB  = Path("src/data/processed/pharmacology.db")
GENO_DB   = Path("src/data/processed/genomics.db")
OUT_DIR   = Path("data/cases")

CASES_PER_SET  = 20   # 10 SENSITIVE + 10 RESISTANT
PER_LABEL      = CASES_PER_SET // 2


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def _load_existing_excluded() -> set[str]:
    """Cell lines already used in any existing cases YAML."""
    excluded: set[str] = set()
    for p in Path("data/cases").glob("cases*.yaml"):
        try:
            data = yaml.safe_load(p.read_text()) or []
            for case in data:
                excluded.add(case.get("cell_line", ""))
        except Exception:
            pass
    return excluded


def _build_valid_pairs() -> tuple[list, list]:
    """Return (sensitive_pairs, resistant_pairs) matched across all DBs."""
    # CTRPv2 labeled pairs
    ctrp = sqlite3.connect(CTRP_DB)
    ctrp_rows = ctrp.execute(
        "SELECT cell_line, drug, label FROM sensitivity WHERE label != 'UNCERTAIN'"
    ).fetchall()
    ctrp.close()

    # Our DB cell lines and drugs
    pharm = sqlite3.connect(PHARM_DB)
    our_cells = {r[0] for r in pharm.execute("SELECT DISTINCT cell_line FROM drug_response")}
    our_drugs  = {r[0] for r in pharm.execute("SELECT DISTINCT drug FROM drug_response")}
    pharm.close()

    geno = sqlite3.connect(GENO_DB)
    geno_cells = {r[0] for r in geno.execute("SELECT DISTINCT cell_line FROM mutations")}
    geno.close()

    our_cell_norm  = {_norm(c): c for c in our_cells}
    our_drug_norm  = {_norm(d): d for d in our_drugs}

    excluded = _load_existing_excluded()
    excluded_norm = {_norm(e) for e in excluded}

    sensitive, resistant = [], []
    for ctrp_cl, ctrp_dr, label in ctrp_rows:
        cl_key = _norm(ctrp_cl)
        dr_key = _norm(ctrp_dr)
        if cl_key not in our_cell_norm or dr_key not in our_drug_norm:
            continue
        if cl_key in excluded_norm:
            continue
        our_cl = our_cell_norm[cl_key]
        our_dr = our_drug_norm[dr_key]
        # Require cell line has genomic data
        if our_cl not in geno_cells:
            continue
        entry = {"cell_line": our_cl, "drug": our_dr, "label": label}
        if label == "SENSITIVE":
            sensitive.append(entry)
        else:
            resistant.append(entry)

    return sensitive, resistant


def sample_sets(n_sets: int, seed: int) -> list[list[dict]]:
    rng = random.Random(seed)
    sensitive, resistant = _build_valid_pairs()

    print(f"Valid pool — SENSITIVE: {len(sensitive)}, RESISTANT: {len(resistant)}")
    max_sets = min(len(sensitive), len(resistant)) // PER_LABEL
    if n_sets > max_sets:
        raise ValueError(f"Requested {n_sets} sets but only {max_sets} non-overlapping sets possible")

    rng.shuffle(sensitive)
    rng.shuffle(resistant)

    sets: list[list[dict]] = []
    used_pairs: set[tuple] = set()

    # Deduplicate pool by (cell_line, drug) to ensure non-overlap
    def _unique_pool(pool):
        seen: set[tuple] = set()
        out = []
        for e in pool:
            key = (e["cell_line"], e["drug"])
            if key not in seen:
                seen.add(key)
                out.append(e)
        return out

    sensitive = _unique_pool(sensitive)
    resistant = _unique_pool(resistant)

    s_idx = r_idx = 0
    for i in range(n_sets):
        s_cases, r_cases = [], []

        while len(s_cases) < PER_LABEL and s_idx < len(sensitive):
            e = sensitive[s_idx]; s_idx += 1
            key = (e["cell_line"], e["drug"])
            if key not in used_pairs:
                s_cases.append(e); used_pairs.add(key)

        while len(r_cases) < PER_LABEL and r_idx < len(resistant):
            e = resistant[r_idx]; r_idx += 1
            key = (e["cell_line"], e["drug"])
            if key not in used_pairs:
                r_cases.append(e); used_pairs.add(key)

        if len(s_cases) < PER_LABEL or len(r_cases) < PER_LABEL:
            print(f"Warning: could only build {i} complete sets")
            break

        combined = s_cases + r_cases
        rng.shuffle(combined)
        sets.append(combined)
        print(f"  Set {i+1}: {len(s_cases)}S + {len(r_cases)}R "
              f"— drugs: {sorted({e['drug'] for e in combined})}")

    return sets


def write_sets(sets: list[list[dict]], dry_run: bool = False) -> None:
    for i, cases in enumerate(sets, 1):
        out_path = OUT_DIR / f"cases_held_out_ctrp_{i}.yaml"
        if dry_run:
            print(f"[dry-run] Would write {out_path} ({len(cases)} cases)")
            continue
        out_path.write_text(yaml.dump(cases, default_flow_style=False, sort_keys=False))
        print(f"Written: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sets", type=int, default=5)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sets = sample_sets(args.n_sets, args.seed)
    print(f"\nSampled {len(sets)} sets of {CASES_PER_SET} cases each")
    write_sets(sets, dry_run=args.dry_run)
