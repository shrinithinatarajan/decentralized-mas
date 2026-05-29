import sqlite3
import pandas as pd
from pathlib import Path

DRUG_RESPONSE_SCHEMA = """
CREATE TABLE IF NOT EXISTS drug_response (
    cell_line TEXT NOT NULL,
    drug TEXT NOT NULL,
    ln_ic50 REAL,
    auc REAL,
    z_score REAL,
    label TEXT
)
"""

DRUG_INFO_SCHEMA = """
CREATE TABLE IF NOT EXISTS drug_info (
    drug TEXT PRIMARY KEY,
    target_genes TEXT,
    moa TEXT,
    pathway TEXT,
    drug_class TEXT
)
"""

SENSITIVITY_PROFILE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sensitivity_profile (
    drug TEXT,
    mutation TEXT,
    median_ic50 REAL,
    resistant_fraction REAL,
    n_cell_lines INTEGER
)
"""


def _z_label(z: float) -> str:
    if z < -0.5:
        return "SENSITIVE"
    if z > 0.5:
        return "RESISTANT"
    return "UNCERTAIN"


def _fetch_chembl_info(drug_name: str) -> dict:
    from chembl_webresource_client.new_client import new_client
    results = new_client.molecule.filter(pref_name__iexact=drug_name).only(["molecule_chembl_id"])
    if not results:
        return {}
    chembl_id = results[0]["molecule_chembl_id"]
    mechs = new_client.mechanism.filter(molecule_chembl_id=chembl_id)
    if not mechs:
        return {}
    return {
        "moa": mechs[0].get("mechanism_of_action", ""),
        "drug_class": mechs[0].get("mechanism_of_action", "").split()[0] if mechs else "",
    }


def create_pharmacology_db(
    db_path: Path,
    ic50_csv: Path,
    cell_lines: list[str],
    fetch_chembl: bool = True,
) -> None:
    df = pd.read_csv(ic50_csv)
    df.columns = [c.upper() for c in df.columns]
    df = df[df["CELL_LINE_NAME"].isin(cell_lines)]

    conn = sqlite3.connect(db_path)
    conn.execute(DRUG_RESPONSE_SCHEMA)
    conn.execute(DRUG_INFO_SCHEMA)
    conn.execute(SENSITIVITY_PROFILE_SCHEMA)

    for _, row in df.iterrows():
        conn.execute(
            "INSERT INTO drug_response VALUES (?,?,?,?,?,?)",
            (
                row["CELL_LINE_NAME"],
                row["DRUG_NAME"],
                float(row["LN_IC50"]),
                float(row["AUC"]),
                float(row["Z_SCORE"]),
                _z_label(float(row["Z_SCORE"])),
            ),
        )

    # drug_info: one row per unique drug, sourced from CSV + optional ChEMBL
    drug_meta = (
        df.groupby("DRUG_NAME")
        .first()
        .reset_index()
        [["DRUG_NAME"] + [c for c in ["PUTATIVE_TARGET", "PATHWAY_NAME"] if c in df.columns]]
    )
    for _, row in drug_meta.iterrows():
        drug = row["DRUG_NAME"]
        target = row.get("PUTATIVE_TARGET", "") if "PUTATIVE_TARGET" in row.index else ""
        pathway = row.get("PATHWAY_NAME", "") if "PATHWAY_NAME" in row.index else ""
        moa, drug_class = "", ""
        if fetch_chembl:
            info = _fetch_chembl_info(drug)
            moa = info.get("moa", "")
            drug_class = info.get("drug_class", "")
        conn.execute(
            "INSERT OR REPLACE INTO drug_info VALUES (?,?,?,?,?)",
            (drug, str(target) if pd.notna(target) else "", moa, str(pathway) if pd.notna(pathway) else "", drug_class),
        )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    from src.data.loader import load_cases
    cases = load_cases()
    cell_lines = [c.cell_line for c in cases]
    create_pharmacology_db(
        Path("src/data/processed/pharmacology.db"),
        Path("src/data/raw/gdsc2/ic50.csv"),
        cell_lines,
        fetch_chembl=True,
    )
    print(f"pharmacology.db built for {len(cell_lines)} cell lines")
