"""Full DB rebuild for all 940 GDSC2 cell lines.

Reads the raw CCLE CSV files and populates genomics, transcriptomics, and
pharmacolgy DBs for every cell line in GDSC2. After this runs, the system
can answer queries about any GDSC2 cell line without further patching.

Usage:
    PYTHONPATH=. python src/data/etl_full_rebuild.py

Estimated runtime: 20-40 min (dominated by mutations.csv chunked read).
"""
import sqlite3
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW   = Path("src/data/raw/ccle")
GDSC2 = Path("src/data/raw/gdsc2/ic50.csv")
DB_GEN   = Path("src/data/processed/genomics.db")
DB_TRAN  = Path("src/data/processed/transcriptomics.db")
DB_PHARM = Path("src/data/processed/pharmacology.db")


def _strip(name: str) -> str:
    """Normalise a cell line name for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _build_name_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Return (stripped_name -> ModelID, stripped_name -> CellLineName) from CCLE model.csv."""
    df = pd.read_csv(RAW / "model.csv", usecols=["ModelID", "CellLineName"])
    mid   = {_strip(row.CellLineName): row.ModelID      for row in df.itertuples() if pd.notna(row.CellLineName)}
    cname = {_strip(row.CellLineName): row.CellLineName for row in df.itertuples() if pd.notna(row.CellLineName)}
    return mid, cname


def _gdsc2_lines() -> list[str]:
    df = pd.read_csv(GDSC2, usecols=["CELL_LINE_NAME"])
    return sorted(df["CELL_LINE_NAME"].unique().tolist())


def _gdsc2_to_ccle(gdsc2_lines: list[str], mid_map: dict) -> dict[str, str]:
    """Return {gdsc2_name -> ModelID} for lines that match CCLE."""
    mapping: dict[str, str] = {}
    for name in gdsc2_lines:
        mid = mid_map.get(_strip(name))
        if mid:
            mapping[name] = mid
    return mapping


# ── Genomics ────────────────────────────────────────────────────────────────

def rebuild_genomics(gdsc_to_mid: dict[str, str]) -> None:
    print("\n=== Rebuilding genomics DB ===")
    target_ids = set(gdsc_to_mid.values())
    mid_to_gdsc = {v: k for k, v in gdsc_to_mid.items()}

    conn = sqlite3.connect(DB_GEN)
    conn.execute("DROP TABLE IF EXISTS mutations")
    conn.execute("DROP TABLE IF EXISTS cnv")
    conn.execute("""
        CREATE TABLE mutations (
            cell_line TEXT NOT NULL, gene TEXT NOT NULL,
            mutation TEXT NOT NULL, mutation_type TEXT,
            cosmic_id TEXT, is_driver INTEGER DEFAULT 0,
            civic_description TEXT)
    """)
    conn.execute("""
        CREATE TABLE cnv (
            cell_line TEXT NOT NULL, gene TEXT NOT NULL,
            cnv_value REAL, status TEXT)
    """)
    conn.commit()

    # Mutations — chunked read of 581 MB file
    print("  Loading mutations (chunked)...")
    inserted = 0
    chunk_n = 0
    for chunk in pd.read_csv(RAW / "mutations.csv", chunksize=100_000, low_memory=False):
        rows = chunk[chunk["ModelID"].isin(target_ids)]
        if rows.empty:
            chunk_n += 1
            continue
        batch = []
        for _, r in rows.iterrows():
            cl = mid_to_gdsc.get(r["ModelID"])
            if not cl:
                continue
            civic = str(r["CivicDescription"]) if pd.notna(r.get("CivicDescription")) else None
            is_drv = int(
                bool(r.get("OncogeneHighImpact"))
                or bool(r.get("TumorSuppressorHighImpact"))
                or bool(civic)  # CIViC evidence = clinically significant = driver
            )
            batch.append((
                cl, r.get("HugoSymbol", ""),
                str(r["ProteinChange"]) if pd.notna(r.get("ProteinChange")) else "",
                r.get("VariantType", ""), "", is_drv, civic,
            ))
        conn.executemany("INSERT INTO mutations VALUES (?,?,?,?,?,?,?)", batch)
        conn.commit()
        inserted += len(batch)
        chunk_n += 1
        if chunk_n % 10 == 0:
            print(f"    ...{chunk_n} chunks, {inserted:,} mutations so far")
    print(f"  Mutations done: {inserted:,} rows")

    # CNV
    print("  Loading CNV...")
    cnv_df = pd.read_csv(RAW / "cnv.csv")
    if "IsDefaultEntryForModel" in cnv_df.columns:
        cnv_df = cnv_df[cnv_df["IsDefaultEntryForModel"] == "Yes"]
    cnv_df = cnv_df[cnv_df["ModelID"].isin(target_ids)]
    meta = {"ModelID","SequencingID","ModelConditionID","IsDefaultEntryForMC",
            "IsDefaultEntryForModel","cell_line","Unnamed: 0"}
    gene_cols = [c for c in cnv_df.columns if c not in meta]

    def _status(v):
        if v < 0.5: return "homozygous_deletion"
        if v < 1.5: return "hemizygous_deletion"
        if v < 2.5: return "neutral"
        if v < 4.0: return "gain"
        return "amplification"

    batch, n = [], 0
    for _, row in cnv_df.iterrows():
        cl = mid_to_gdsc.get(row["ModelID"])
        if not cl:
            continue
        for col in gene_cols:
            try:
                val = float(row[col])
                batch.append((cl, col.split(" (")[0], val, _status(val)))
            except (ValueError, TypeError):
                pass
        if len(batch) >= 200_000:
            conn.executemany("INSERT INTO cnv VALUES (?,?,?,?)", batch)
            conn.commit()
            n += len(batch)
            batch = []
    if batch:
        conn.executemany("INSERT INTO cnv VALUES (?,?,?,?)", batch)
        conn.commit()
        n += len(batch)
    print(f"  CNV done: {n:,} rows")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_mut_cl ON mutations(cell_line)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cnv_cl ON cnv(cell_line)")
    conn.commit()
    conn.close()
    print(f"  genomics.db rebuilt for {len(gdsc_to_mid)} cell lines")


# ── Transcriptomics ─────────────────────────────────────────────────────────

META_COLS = {"ModelID", "SequencingID", "ModelConditionID", "IsDefaultEntryForMC",
             "IsDefaultEntryForModel", "cell_line", "Unnamed: 0"}


def _log1p_to_tpm(value: float) -> float:
    return (2 ** value) - 1


def rebuild_transcriptomics(gdsc_to_mid: dict[str, str]) -> None:
    print("\n=== Rebuilding transcriptomics DB ===")
    target_ids = set(gdsc_to_mid.values())
    mid_to_gdsc = {v: k for k, v in gdsc_to_mid.items()}

    conn = sqlite3.connect(DB_TRAN)
    conn.execute("DROP TABLE IF EXISTS expression")
    conn.execute("""
        CREATE TABLE expression (
            cell_line TEXT NOT NULL, gene TEXT NOT NULL,
            tpm REAL, z_score REAL, percentile REAL)
    """)
    conn.commit()

    print("  Loading expression.csv...")
    expr_df = pd.read_csv(RAW / "expression.csv")
    if "IsDefaultEntryForModel" in expr_df.columns:
        expr_df = expr_df[expr_df["IsDefaultEntryForModel"] == "Yes"]
    expr_df = expr_df[expr_df["ModelID"].isin(target_ids)].copy()
    expr_df["cell_line"] = expr_df["ModelID"].map(mid_to_gdsc)
    expr_df = expr_df.dropna(subset=["cell_line"]).set_index("cell_line")

    gene_cols = [c for c in expr_df.columns if c not in META_COLS and c != "ModelID"]
    expr_df = expr_df[gene_cols]
    expr_df.columns = [c.split(" (")[0] for c in expr_df.columns]

    print(f"  Computing z-scores across {len(expr_df)} cell lines x {len(expr_df.columns)} genes...")
    gene_means = expr_df.mean(axis=0)
    gene_stds  = expr_df.std(axis=0).replace(0, 1)
    z_scores   = (expr_df - gene_means) / gene_stds

    print("  Inserting expression rows...")
    batch, total = [], 0
    for cl in expr_df.index:
        for gene in expr_df.columns:
            try:
                raw_val = float(expr_df.loc[cl, gene])
                tpm = _log1p_to_tpm(raw_val)
                z   = float(z_scores.loc[cl, gene])
                pct = float(np.sum(expr_df[gene] <= raw_val) / len(expr_df))
                batch.append((cl, gene, tpm, z, pct))
            except (ValueError, TypeError):
                pass
            if len(batch) >= 500_000:
                conn.executemany("INSERT INTO expression VALUES (?,?,?,?,?)", batch)
                conn.commit()
                total += len(batch)
                batch = []
                print(f"    ...{total:,} rows inserted")
    if batch:
        conn.executemany("INSERT INTO expression VALUES (?,?,?,?,?)", batch)
        conn.commit()
        total += len(batch)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_expr_cl ON expression(cell_line)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_expr_gene ON expression(gene)")
    conn.commit()
    conn.close()
    print(f"  transcriptomics.db rebuilt: {total:,} rows for {len(expr_df)} cell lines")


# ── Pharmacology ─────────────────────────────────────────────────────────────

def rebuild_pharmacology() -> None:
    print("\n=== Rebuilding pharmacology DB ===")
    df = pd.read_csv(GDSC2)
    df.columns = [c.upper() for c in df.columns]

    conn = sqlite3.connect(DB_PHARM)
    conn.execute("DROP TABLE IF EXISTS drug_response")
    conn.execute("DROP TABLE IF EXISTS drug_info")
    conn.execute("""
        CREATE TABLE drug_response (
            cell_line TEXT NOT NULL, drug TEXT NOT NULL,
            ln_ic50 REAL, auc REAL, z_score REAL, label TEXT)
    """)
    conn.execute("""
        CREATE TABLE drug_info (
            drug TEXT PRIMARY KEY, target_genes TEXT,
            moa TEXT, pathway TEXT, drug_class TEXT)
    """)
    conn.commit()

    def _label(z): return "SENSITIVE" if z < -0.5 else "RESISTANT" if z > 0.5 else "UNCERTAIN"

    batch = [
        (r.CELL_LINE_NAME, r.DRUG_NAME, float(r.LN_IC50),
         float(r.AUC), float(r.Z_SCORE), _label(float(r.Z_SCORE)))
        for r in df.itertuples()
    ]
    conn.executemany("INSERT INTO drug_response VALUES (?,?,?,?,?,?)", batch)

    drug_meta = df.drop_duplicates("DRUG_NAME")
    drug_batch = [
        (r.DRUG_NAME,
         str(r.PUTATIVE_TARGET) if pd.notna(r.PUTATIVE_TARGET) else "",
         "", str(r.PATHWAY_NAME) if pd.notna(r.PATHWAY_NAME) else "", "")
        for r in drug_meta.itertuples()
    ]
    conn.executemany("INSERT OR REPLACE INTO drug_info VALUES (?,?,?,?,?)", drug_batch)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_dr_cl ON drug_response(cell_line)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dr_drug ON drug_response(drug)")
    conn.commit()
    conn.close()
    print(f"  pharmacology.db rebuilt: {len(batch):,} rows, {df['CELL_LINE_NAME'].nunique()} cell lines, {df['DRUG_NAME'].nunique()} drugs")


if __name__ == "__main__":
    import time
    t0 = time.time()

    print("Building CCLE name maps...")
    mid_map, _ = _build_name_maps()

    print("Loading GDSC2 cell lines...")
    gdsc2_lines = _gdsc2_lines()
    gdsc_to_mid = _gdsc2_to_ccle(gdsc2_lines, mid_map)
    print(f"  {len(gdsc2_lines)} GDSC2 lines → {len(gdsc_to_mid)} matched to CCLE ModelIDs")
    unmatched = [l for l in gdsc2_lines if l not in gdsc_to_mid]
    if unmatched:
        print(f"  {len(unmatched)} unmatched (no CCLE genomic data): {unmatched[:5]}...")

    rebuild_pharmacology()
    rebuild_genomics(gdsc_to_mid)
    rebuild_transcriptomics(gdsc_to_mid)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
