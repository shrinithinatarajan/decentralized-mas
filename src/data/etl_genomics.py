import sqlite3
import pandas as pd
from pathlib import Path

MUTATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS mutations (
    cell_line TEXT NOT NULL,
    gene TEXT NOT NULL,
    mutation TEXT NOT NULL,
    mutation_type TEXT,
    cosmic_id TEXT,
    is_driver INTEGER DEFAULT 0
)
"""

CNV_SCHEMA = """
CREATE TABLE IF NOT EXISTS cnv (
    cell_line TEXT NOT NULL,
    gene TEXT NOT NULL,
    cnv_value REAL,
    status TEXT
)
"""


def _cnv_status(value: float) -> str:
    if value < 0.5:
        return "homozygous_deletion"
    if value < 1.5:
        return "hemizygous_deletion"
    if value < 2.5:
        return "neutral"
    if value < 4.0:
        return "gain"
    return "amplification"


def _load_model_map(model_csv: Path) -> dict[str, str]:
    """Returns {ModelID: CellLineName} from DepMap Model.csv."""
    df = pd.read_csv(model_csv, usecols=["ModelID", "CellLineName"])
    return dict(zip(df["ModelID"], df["CellLineName"]))


def create_genomics_db(
    db_path: Path,
    mutations_csv: Path,
    cnv_csv: Path,
    cell_lines: list[str],
    model_csv: Path | None = None,
) -> None:
    model_map = _load_model_map(model_csv) if model_csv else {}
    cell_line_set = set(cell_lines)

    conn = sqlite3.connect(db_path)
    conn.execute(MUTATIONS_SCHEMA)
    conn.execute(CNV_SCHEMA)

    mut_df = pd.read_csv(mutations_csv, low_memory=False)
    if model_map:
        mut_df["cell_line"] = mut_df["ModelID"].map(model_map)
    else:
        mut_df["cell_line"] = mut_df["ModelID"]
    mut_df = mut_df[mut_df["cell_line"].isin(cell_line_set)]

    for _, row in mut_df.iterrows():
        is_driver = int(
            bool(row.get("OncogeneHighImpact")) or bool(row.get("TumorSuppressorHighImpact"))
        )
        conn.execute(
            "INSERT INTO mutations VALUES (?,?,?,?,?,?)",
            (
                row["cell_line"],
                row.get("HugoSymbol", ""),
                str(row["ProteinChange"]) if pd.notna(row.get("ProteinChange")) else "",
                row.get("VariantType", ""),
                "",  # cosmic_id not in DepMap format
                is_driver,
            ),
        )

    # CNV: rows=cell lines (by ModelID), columns=genes as "GENE (ENTREZ_ID)"
    cnv_df = pd.read_csv(cnv_csv)
    if model_map:
        cnv_df["cell_line"] = cnv_df["ModelID"].map(model_map)
    else:
        cnv_df["cell_line"] = cnv_df["ModelID"]
    if "IsDefaultEntryForModel" in cnv_df.columns:
        cnv_df = cnv_df[cnv_df["IsDefaultEntryForModel"] == "Yes"]
    cnv_df = cnv_df[cnv_df["cell_line"].isin(cell_line_set)]

    meta_cols = {"ModelID", "SequencingID", "ModelConditionID", "IsDefaultEntryForMC",
                 "IsDefaultEntryForModel", "cell_line", "Unnamed: 0"}
    gene_cols = [c for c in cnv_df.columns if c not in meta_cols]

    for _, row in cnv_df.iterrows():
        for col in gene_cols:
            gene = col.split(" (")[0]
            value = float(row[col])
            conn.execute(
                "INSERT INTO cnv VALUES (?,?,?,?)",
                (row["cell_line"], gene, value, _cnv_status(value)),
            )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    from src.data.loader import load_cases
    cases = load_cases()
    cell_lines = [c.cell_line for c in cases]
    create_genomics_db(
        Path("src/data/processed/genomics.db"),
        Path("src/data/raw/ccle/mutations.csv"),
        Path("src/data/raw/ccle/cnv.csv"),
        cell_lines,
        model_csv=Path("src/data/raw/ccle/model.csv"),
    )
    print(f"genomics.db built for {len(cell_lines)} cell lines")
