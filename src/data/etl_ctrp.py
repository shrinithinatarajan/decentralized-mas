"""Download CTRPv2 drug sensitivity data and build independent ground truth labels.

CTRPv2 (Cancer Therapeutics Response Portal v2) is an independent dataset from
the Broad Institute measuring drug sensitivity across 860+ CCLE cell lines. Using
it as ground truth avoids data leakage: our pharmacology.db is derived from GDSC2
z-scores, so GDSC2-based labels trivially predict from the pharmacology agent.

Label derivation:
    AUC = area under viability curve (lower = more sensitive)
    Per drug, label the bottom SENSITIVE_QUANTILE as SENSITIVE,
    top RESISTANT_QUANTILE as RESISTANT, middle excluded (uncertain).

Usage:
    PYTHONPATH=. python src/data/etl_ctrp.py

Outputs:
    src/data/raw/ctrp/             (raw downloaded files)
    src/data/processed/ground_truth_ctrp.db

If automatic download fails, download manually from:
    https://depmap.org/portal/download/all/ → search "CTRP"
or:
    https://portals.broadinstitute.org/ctrp/
Expect three files:
    v20.meta.per_cell_line.txt
    v20.meta.per_compound.txt
    v20.data.curves_post_qc.txt
"""
import sqlite3
import urllib.request
from pathlib import Path

import pandas as pd

RAW_DIR = Path("src/data/raw/ctrp")
DB_PATH = Path("src/data/processed/ground_truth_ctrp.db")

# Binary label thresholds: per drug, lowest X% = SENSITIVE, highest X% = RESISTANT
SENSITIVE_QUANTILE = 0.25
RESISTANT_QUANTILE = 0.75

# CTRPv2 files hosted on DepMap figshare (stable IDs as of 2024)
_CTRP_URLS: dict[str, str] = {}  # files must be placed manually in RAW_DIR
_REQUIRED_FILES = [
    "v20.meta.per_cell_line.txt",
    "v20.meta.per_compound.txt",
    "v20.meta.per_experiment.txt",
    "v20.data.curves_post_qc.txt",
]


def _check_files() -> bool:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    missing = [f for f in _REQUIRED_FILES if not (RAW_DIR / f).exists()]
    if missing:
        print(f"  Missing files in {RAW_DIR}:")
        for f in missing:
            print(f"    {f}")
        return False
    return True


def _load_cell_line_map() -> dict[str, str]:
    """Return {master_ccl_id -> ccl_name} from CTRPv2 cell line metadata."""
    path = RAW_DIR / "v20.meta.per_cell_line.txt"
    df = pd.read_csv(path, sep="\t", usecols=["master_ccl_id", "ccl_name"])
    return dict(zip(df["master_ccl_id"].astype(str), df["ccl_name"]))


def _load_experiment_map() -> dict[str, str]:
    """Return {experiment_id -> master_ccl_id} from CTRPv2 experiment metadata."""
    path = RAW_DIR / "v20.meta.per_experiment.txt"
    df = pd.read_csv(path, sep="\t", usecols=["experiment_id", "master_ccl_id"])
    return dict(zip(df["experiment_id"].astype(str), df["master_ccl_id"].astype(str)))


def _load_compound_map() -> dict[str, str]:
    """Return {master_cpd_id -> cpd_name} from CTRPv2 compound metadata."""
    path = RAW_DIR / "v20.meta.per_compound.txt"
    df = pd.read_csv(path, sep="\t", usecols=["master_cpd_id", "cpd_name"])
    return dict(zip(df["master_cpd_id"].astype(str), df["cpd_name"]))


def _normalize_cell_line(name: str) -> str:
    """Normalize to uppercase, hyphens for spaces, remove trailing spaces."""
    return name.strip().upper().replace(" ", "-")


def _normalize_drug(name: str) -> str:
    return name.strip()


def build_ground_truth_db() -> None:
    print("Loading CTRPv2 AUC data...")
    curves_path = RAW_DIR / "v20.data.curves_post_qc.txt"
    ccl_map = _load_cell_line_map()
    exp_map = _load_experiment_map()
    cpd_map = _load_compound_map()

    curves = pd.read_csv(
        curves_path, sep="\t",
        usecols=["experiment_id", "master_cpd_id", "area_under_curve"]
    )
    print(f"  {len(curves):,} dose-response curves loaded")

    # Join experiment_id -> master_ccl_id -> ccl_name
    curves["master_ccl_id"] = curves["experiment_id"].astype(str).map(exp_map)
    curves["cell_line"] = (
        curves["master_ccl_id"]
        .map(ccl_map)
        .apply(lambda x: _normalize_cell_line(x) if pd.notna(x) else None)
    )
    curves["drug"] = (
        curves["master_cpd_id"].astype(str)
        .map(cpd_map)
        .apply(lambda x: _normalize_drug(x) if pd.notna(x) else None)
    )
    curves = curves.dropna(subset=["cell_line", "drug", "area_under_curve"])

    # Per drug, compute quantile thresholds and assign labels
    print("  Computing per-drug sensitivity labels...")
    labels: list[tuple] = []
    for drug, grp in curves.groupby("drug"):
        auc = grp["area_under_curve"]
        lo = auc.quantile(SENSITIVE_QUANTILE)
        hi = auc.quantile(RESISTANT_QUANTILE)
        for _, row in grp.iterrows():
            a = row["area_under_curve"]
            if a <= lo:
                lbl = "SENSITIVE"
            elif a >= hi:
                lbl = "RESISTANT"
            else:
                lbl = "UNCERTAIN"
            labels.append((row["cell_line"], drug, round(float(a), 6), lbl))

    df_out = pd.DataFrame(labels, columns=["cell_line", "drug", "auc", "label"])
    sensitive = (df_out["label"] == "SENSITIVE").sum()
    resistant = (df_out["label"] == "RESISTANT").sum()
    uncertain = (df_out["label"] == "UNCERTAIN").sum()
    print(f"  Labels: {sensitive:,} SENSITIVE, {resistant:,} RESISTANT, "
          f"{uncertain:,} UNCERTAIN")
    print(f"  Unique cell lines: {df_out['cell_line'].nunique():,}")
    print(f"  Unique drugs: {df_out['drug'].nunique():,}")

    # Write DB
    print(f"\nWriting {DB_PATH} ...")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS sensitivity")
    conn.execute("DROP TABLE IF EXISTS compound_meta")
    conn.execute("""
        CREATE TABLE sensitivity (
            cell_line TEXT NOT NULL,
            drug TEXT NOT NULL,
            auc REAL NOT NULL,
            label TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE compound_meta (
            cpd_id TEXT PRIMARY KEY,
            cpd_name TEXT NOT NULL
        )
    """)

    conn.executemany(
        "INSERT INTO sensitivity VALUES (?,?,?,?)",
        df_out[["cell_line", "drug", "auc", "label"]].itertuples(index=False, name=None)
    )
    # Also store compound name mapping for external lookups
    cpd_rows = [(k, v) for k, v in cpd_map.items()]
    conn.executemany("INSERT INTO compound_meta VALUES (?,?)", cpd_rows)

    conn.execute("CREATE INDEX idx_sens_cell_drug ON sensitivity(cell_line, drug)")
    conn.execute("CREATE INDEX idx_sens_drug ON sensitivity(drug)")
    conn.commit()
    conn.close()
    print(f"  Done — {len(df_out):,} rows in sensitivity table")


if __name__ == "__main__":
    import time
    t0 = time.time()

    print("Step 1: Checking CTRPv2 files...")
    if not _check_files():
        print(
            f"\nPlace the required files in {RAW_DIR} and re-run."
            "\nDownload CTRPv2.0_2015_ctd2_ExpandedDataset.zip from "
            "https://portals.broadinstitute.org/ctrp/"
        )
        raise SystemExit(1)
    print(f"  All files present in {RAW_DIR}")
    # Remove stale download import
    import urllib.request as _u  # noqa: F401 kept for backward compat

    print("\nStep 2: Building ground truth DB...")
    build_ground_truth_db()

    print(f"\nDone in {(time.time()-t0):.1f}s")
