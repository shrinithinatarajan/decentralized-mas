import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

EXPRESSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS expression (
    cell_line TEXT NOT NULL,
    gene TEXT NOT NULL,
    tpm REAL,
    z_score REAL,
    percentile REAL
)
"""

META_COLS = {"ModelID", "SequencingID", "ModelConditionID", "IsDefaultEntryForMC",
             "IsDefaultEntryForModel", "cell_line", "Unnamed: 0"}


def _log1p_to_tpm(value: float) -> float:
    return (2 ** value) - 1


def _load_model_map(model_csv: Path) -> dict[str, str]:
    df = pd.read_csv(model_csv, usecols=["ModelID", "CellLineName"])
    return dict(zip(df["ModelID"], df["CellLineName"]))


def create_transcriptomics_db(
    db_path: Path,
    expression_csv: Path,
    cell_lines: list[str],
    model_csv: Path | None = None,
) -> None:
    model_map = _load_model_map(model_csv) if model_csv else {}
    cell_line_set = set(cell_lines)

    expr_df = pd.read_csv(expression_csv)
    if model_map:
        expr_df["cell_line"] = expr_df["ModelID"].map(model_map)
    else:
        expr_df["cell_line"] = expr_df["ModelID"]
    if "IsDefaultEntryForModel" in expr_df.columns:
        expr_df = expr_df[expr_df["IsDefaultEntryForModel"] == "Yes"]
    expr_df = expr_df[expr_df["cell_line"].isin(cell_line_set)]
    expr_df = expr_df.set_index("cell_line")

    gene_cols = [c for c in expr_df.columns if c not in META_COLS]
    expr_df = expr_df[gene_cols]
    expr_df.columns = [c.split(" (")[0] for c in expr_df.columns]

    gene_means = expr_df.mean(axis=0)
    gene_stds = expr_df.std(axis=0).replace(0, 1)
    z_scores = (expr_df - gene_means) / gene_stds

    conn = sqlite3.connect(db_path)
    conn.execute(EXPRESSION_SCHEMA)

    for cell_line in expr_df.index:
        for gene in expr_df.columns:
            tpm = _log1p_to_tpm(float(expr_df.loc[cell_line, gene]))
            z = float(z_scores.loc[cell_line, gene])
            pct = float(np.sum(expr_df[gene] <= expr_df.loc[cell_line, gene]) / len(expr_df))
            conn.execute(
                "INSERT INTO expression VALUES (?,?,?,?,?)",
                (cell_line, gene, tpm, z, pct),
            )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    from src.data.loader import load_cases
    cases = load_cases()
    cell_lines = [c.cell_line for c in cases]
    create_transcriptomics_db(
        Path("src/data/processed/transcriptomics.db"),
        Path("src/data/raw/ccle/expression.csv"),
        cell_lines,
        model_csv=Path("src/data/raw/ccle/model.csv"),
    )
    print(f"transcriptomics.db built for {len(cell_lines)} cell lines")
