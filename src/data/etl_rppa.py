"""ETL: Load CCLE RPPA protein expression data into depmap.db.

Source: CCLE_RPPA_20181003.csv (899 cell lines, 214 antibodies)
Cell line name format: CCLEName (e.g. DMS53_LUNG) → mapped to CellLineName via model.csv.
"""
import re
import sqlite3
import pandas as pd
from pathlib import Path


RPPA_SCHEMA = """
CREATE TABLE IF NOT EXISTS rppa_expression (
    cell_line     TEXT NOT NULL,
    cell_line_norm TEXT NOT NULL,
    antibody      TEXT NOT NULL,
    rppa_score    REAL,
    PRIMARY KEY (cell_line_norm, antibody)
)
"""

ANTIBODY_GENE_SCHEMA = """
CREATE TABLE IF NOT EXISTS rppa_antibody_genes (
    antibody  TEXT PRIMARY KEY,
    gene_norm TEXT NOT NULL
)
"""


def _norm(s: str) -> str:
    return re.sub(r"[\s\-_/]", "", str(s)).upper()


def _antibody_to_gene(antibody: str) -> str:
    """Extract a gene-like symbol from an RPPA antibody column name."""
    # Strip phospho/modification suffixes: _pS473, _Caution, etc.
    base = re.split(r"_p[A-Z]\d|_Caution|_cleavedD|\(", antibody)[0].strip()
    # Some use protein aliases: c-Met → MET, B-Raf → BRAF, HER2 → ERBB2
    alias_map = {
        "c-Met": "MET",
        "B-Raf": "BRAF",
        "C-Raf": "RAF1",
        "A-Raf": "ARAF",
        "HER2": "ERBB2",
        "HER3": "ERBB3",
        "HER4": "ERBB4",
        "Akt": "AKT1",
        "mTOR": "MTOR",
        "MEK1": "MAP2K1",
        "MEK2": "MAP2K2",
        "p21": "CDKN1A",
        "p27": "CDKN1B",
        "p53": "TP53",
        "p62": "SQSTM1",
        "p90RSK": "RPS6KA1",
        "VEGFR2": "KDR",
        "PI3K-p110-alpha": "PIK3CA",
        "PI3K-p85": "PIK3R1",
    }
    return alias_map.get(base, base.replace("-", "").replace(" ", ""))


def load_rppa(
    depmap_db: Path,
    rppa_csv: Path,
    model_csv: Path,
) -> None:
    # Build CCLEName → CellLineName map
    model = pd.read_csv(model_csv, usecols=["ModelID", "CellLineName", "CCLEName"])
    ccle_to_name = dict(zip(model["CCLEName"], model["CellLineName"]))

    df = pd.read_csv(rppa_csv, index_col=0)

    conn = sqlite3.connect(depmap_db)
    conn.execute(RPPA_SCHEMA)
    conn.execute(ANTIBODY_GENE_SCHEMA)

    # Build antibody → gene mapping
    antibody_gene_rows = [(ab, _norm(_antibody_to_gene(ab))) for ab in df.columns]
    conn.executemany(
        "INSERT OR REPLACE INTO rppa_antibody_genes VALUES (?, ?)",
        antibody_gene_rows,
    )

    # Insert expression rows
    rows = []
    for ccle_id, row in df.iterrows():
        cell_line = ccle_to_name.get(ccle_id) or ccle_id
        cell_line_norm = _norm(cell_line)
        for antibody, score in row.items():
            if pd.notna(score):
                rows.append((cell_line, cell_line_norm, str(antibody), float(score)))

    conn.executemany(
        "INSERT OR REPLACE INTO rppa_expression VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    print(f"Loaded {len(rows)} RPPA rows for {df.shape[0]} cell lines, {df.shape[1]} antibodies")


if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent
    load_rppa(
        depmap_db=base / "src/data/processed/depmap.db",
        rppa_csv=base / "src/data/raw/depmap/CCLE_RPPA_20181003.csv",
        model_csv=base / "src/data/raw/ccle/model.csv",
    )
