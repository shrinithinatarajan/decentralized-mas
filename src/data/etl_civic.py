"""Download CIViC nightly clinical evidence and build civic.db.

Downloads nightly-ClinicalEvidenceSummaries.tsv from civicdb.org (public, no auth).
Filters to Predictive evidence (drug sensitivity/resistance) and builds a SQLite
database that the genomics MCP server queries at runtime for drug-specific context.

Usage:
    PYTHONPATH=. python src/data/etl_civic.py

Outputs:
    src/data/raw/civic/nightly-ClinicalEvidenceSummaries.tsv
    src/data/processed/civic.db
"""
import re
import sqlite3
import urllib.request
from pathlib import Path

import pandas as pd

RAW_DIR  = Path("src/data/raw/civic")
DB_PATH  = Path("src/data/processed/civic.db")
CIVIC_URL = "https://civicdb.org/downloads/nightly/nightly-ClinicalEvidenceSummaries.tsv"


def _download() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / "nightly-ClinicalEvidenceSummaries.tsv"
    print(f"  Downloading CIViC nightly ({CIVIC_URL}) ...")
    urllib.request.urlretrieve(CIVIC_URL, dest)
    print(f"  Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def _normalize_variant(v: str) -> str:
    """Strip p. prefix, spaces; uppercase for matching."""
    return re.sub(r"^p\.", "", str(v).strip()).upper()


def _normalize_drug(d: str) -> str:
    return str(d).strip().upper()


def _parse_molecular_profile(mp: str) -> list[tuple[str, str]]:
    """Return list of (gene, variant) from a molecular_profile string.

    Handles simple profiles ("EGFR T790M") and compound profiles
    ("EGFR T790M AND EGFR L858R") by splitting on AND/OR.
    Returns empty list for profiles without a clear gene+variant pair.
    """
    pairs = []
    # Split compound profiles on AND/OR
    parts = re.split(r"\s+(?:AND|OR)\s+", mp.strip())
    for part in parts:
        tokens = part.strip().split()
        if len(tokens) >= 2:
            gene = tokens[0].upper()
            variant = " ".join(tokens[1:]).upper()
            pairs.append((gene, variant))
    return pairs


def build_civic_db(tsv_path: Path) -> None:
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    print(f"  {len(df):,} total CIViC evidence rows")

    # Keep only Predictive evidence with a drug listed
    pred = df[
        (df["evidence_type"].str.strip().str.lower() == "predictive") &
        (df["therapies"].notna()) &
        (df["therapies"].str.strip() != "")
    ].copy()
    print(f"  {len(pred):,} Predictive evidence rows with therapies")

    # Expand molecular_profile → (gene, variant) rows
    expanded_rows = []
    for _, row in pred.iterrows():
        pairs = _parse_molecular_profile(str(row["molecular_profile"]))
        if not pairs:
            continue
        sig       = str(row["significance"]).strip()
        direction = str(row["evidence_direction"]).strip()
        drug      = str(row["therapies"]).strip()
        lvl       = str(row.get("evidence_level", "")).strip()
        disease   = str(row.get("disease", "")).strip()
        statement = str(row.get("evidence_statement", "")).strip()
        for gene, variant in pairs:
            variant_norm = _normalize_variant(variant)
            drug_norm    = _normalize_drug(drug)
            description  = (
                f"{gene} {variant} → {sig} to {drug} "
                f"(CIViC level {lvl}; {disease}). {statement}"
            ).strip()
            expanded_rows.append((
                gene, gene, variant, variant_norm,
                drug, drug_norm, sig, direction, lvl, disease, description
            ))

    print(f"  {len(expanded_rows):,} expanded (gene, variant, drug) evidence rows")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS predictive_evidence")
    conn.execute("""
        CREATE TABLE predictive_evidence (
            gene          TEXT NOT NULL,
            gene_norm     TEXT NOT NULL,
            variant       TEXT NOT NULL,
            variant_norm  TEXT NOT NULL,
            drug          TEXT NOT NULL,
            drug_norm     TEXT NOT NULL,
            significance  TEXT NOT NULL,
            direction     TEXT,
            evidence_level TEXT,
            disease       TEXT,
            description   TEXT
        )
    """)
    conn.executemany("INSERT INTO predictive_evidence VALUES (?,?,?,?,?,?,?,?,?,?,?)", expanded_rows)
    conn.execute("CREATE INDEX idx_civic_gene_var ON predictive_evidence(gene_norm, variant_norm)")
    conn.execute("CREATE INDEX idx_civic_drug ON predictive_evidence(drug_norm)")
    conn.commit()

    n = conn.execute("SELECT COUNT(*) FROM predictive_evidence").fetchone()[0]
    genes = conn.execute("SELECT COUNT(DISTINCT gene_norm) FROM predictive_evidence").fetchone()[0]
    drugs = conn.execute("SELECT COUNT(DISTINCT drug_norm) FROM predictive_evidence").fetchone()[0]
    conn.close()
    print(f"  civic.db: {n:,} evidence rows, {genes} genes, {drugs} drugs")


if __name__ == "__main__":
    import time
    t0 = time.time()
    tsv = _download()
    print("Building civic.db ...")
    build_civic_db(tsv)
    print(f"Done in {time.time()-t0:.1f}s")
