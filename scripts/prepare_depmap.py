"""Build depmap.db from DepMap CRISPR Chronos gene-effect data.

Manual download (one-time):
  1. Go to https://depmap.org/portal/download/all/
  2. Download CRISPRGeneEffect.csv  (CRISPR section)
  3. Download Model.csv             (Metadata section)
  4. Run:
       PYTHONPATH=. python scripts/prepare_depmap.py \\
           --crispr /path/to/CRISPRGeneEffect.csv \\
           --model  /path/to/Model.csv
"""
import argparse
import io
import json
import re
import sqlite3
from pathlib import Path

OUT_DB = Path("src/data/processed/depmap.db")


def main(crispr_path: Path, model_path: Path) -> None:
    import pandas as pd

    OUT_DB.parent.mkdir(parents=True, exist_ok=True)

    # --- sample info: map DepMap_ID -> stripped cell line name ---
    print(f"Loading {model_path}...")
    model_df = pd.read_csv(model_path)
    # Column names vary by release
    name_col = next(
        c for c in model_df.columns
        if c in ("StrippedCellLineName", "cell_line_name", "CellLineName")
    )
    id_col = next(c for c in model_df.columns if c in ("ModelID", "DepMap_ID"))
    id_to_name = dict(zip(model_df[id_col], model_df[name_col].str.upper()))
    print(f"  {len(id_to_name)} cell lines loaded")

    # --- CRISPR gene effect matrix ---
    print(f"Loading {crispr_path} (may take a minute)...")
    crispr_df = pd.read_csv(crispr_path, index_col=0)
    # Columns: "SYMBOL (ENTREZID)" → strip to symbol
    crispr_df.columns = [c.split(" ")[0] for c in crispr_df.columns]
    print(f"  Matrix: {crispr_df.shape[0]} cell lines x {crispr_df.shape[1]} genes")

    # --- Write to SQLite ---
    conn = sqlite3.connect(OUT_DB)
    conn.execute("DROP TABLE IF EXISTS chronos_scores")
    conn.execute("""
        CREATE TABLE chronos_scores (
            depmap_id      TEXT NOT NULL,
            cell_line      TEXT NOT NULL,
            cell_line_norm TEXT NOT NULL,
            gene           TEXT NOT NULL,
            chronos        REAL NOT NULL,
            PRIMARY KEY (depmap_id, gene)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_norm_gene ON chronos_scores(cell_line_norm, gene)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gene      ON chronos_scores(gene)")

    rows: list[tuple] = []
    total = 0
    for depmap_id, row in crispr_df.iterrows():
        cl_name = id_to_name.get(depmap_id, str(depmap_id).upper())
        cl_norm = re.sub(r"[\s\-_]", "", cl_name).upper()
        for gene, score in row.items():
            if pd.notna(score):
                rows.append((depmap_id, cl_name, cl_norm, gene, float(score)))
        if len(rows) >= 100_000:
            conn.executemany("INSERT OR IGNORE INTO chronos_scores VALUES (?,?,?,?,?)", rows)
            total += len(rows)
            rows.clear()
            print(f"  {total:,} rows written...", end="\r", flush=True)
    if rows:
        conn.executemany("INSERT OR IGNORE INTO chronos_scores VALUES (?,?,?,?,?)", rows)
        total += len(rows)
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM chronos_scores").fetchone()[0]
    print(f"\nStored {count:,} Chronos rows in {OUT_DB}")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build depmap.db from DepMap CSV files")
    parser.add_argument("--crispr", required=True, type=Path, help="Path to CRISPRGeneEffect.csv")
    parser.add_argument("--model",  required=True, type=Path, help="Path to Model.csv")
    args = parser.parse_args()
    main(args.crispr, args.model)
