"""Patch genomics and transcriptomics DBs for cell lines whose CCLE name
differs from the cases.yaml name. Appends rows under the cases.yaml name.
"""
import sqlite3
import pandas as pd
from pathlib import Path

# cases.yaml name -> (CCLE CellLineName, CCLE ModelID)
ALIASES: dict[str, tuple[str, str]] = {
    "A375":         ("A-375",      "ACH-000219"),
    "MCF7":         ("MCF-7",      "ACH-000019"),
    "PANC-04-03":   ("Panc 04.03", "ACH-000235"),
    "EoL-1-cell":   ("EOL-1",      "ACH-000198"),
    "SUP-B8":       ("SUP-B8",     "ACH-002308"),
    "HCE-4":        ("HCE-4",      "ACH-002243"),
    "SW872":        ("SW872",      "ACH-002310"),
    "COLO-320-HSR": ("COLO-320",   "ACH-000202"),
}

RAW = Path("src/data/raw/ccle")
DB_GEN  = Path("src/data/processed/genomics.db")
DB_TRAN = Path("src/data/processed/transcriptomics.db")


def _cnv_status(value: float) -> str:
    if value < 0.5:  return "homozygous_deletion"
    if value < 1.5:  return "hemizygous_deletion"
    if value < 2.5:  return "neutral"
    if value < 4.0:  return "gain"
    return "amplification"


def patch_genomics():
    mut_df = pd.read_csv(RAW / "mutations.csv", low_memory=False)
    cnv_df = pd.read_csv(RAW / "cnv.csv")

    conn = sqlite3.connect(DB_GEN)
    inserted_mut = inserted_cnv = 0

    for cases_name, (ccle_name, model_id) in ALIASES.items():
        # Check if already present
        existing = conn.execute(
            "SELECT COUNT(*) FROM mutations WHERE cell_line=?", (cases_name,)
        ).fetchone()[0]
        if existing > 0:
            print(f"  {cases_name}: already in genomics DB ({existing} rows), skipping")
            continue

        # Mutations
        rows = mut_df[mut_df["ModelID"] == model_id]
        for _, row in rows.iterrows():
            is_driver = int(bool(row.get("OncogeneHighImpact")) or bool(row.get("TumorSuppressorHighImpact")))
            conn.execute(
                "INSERT INTO mutations VALUES (?,?,?,?,?,?)",
                (cases_name, row.get("HugoSymbol",""),
                 str(row["ProteinChange"]) if pd.notna(row.get("ProteinChange")) else "",
                 row.get("VariantType",""), "", is_driver),
            )
            inserted_mut += 1

        # CNV
        if "IsDefaultEntryForModel" in cnv_df.columns:
            cnv_rows = cnv_df[(cnv_df["ModelID"] == model_id) &
                              (cnv_df["IsDefaultEntryForModel"] == "Yes")]
        else:
            cnv_rows = cnv_df[cnv_df["ModelID"] == model_id]

        meta_cols = {"ModelID","SequencingID","ModelConditionID","IsDefaultEntryForMC",
                     "IsDefaultEntryForModel","cell_line","Unnamed: 0"}
        gene_cols = [c for c in cnv_df.columns if c not in meta_cols]

        for _, row in cnv_rows.iterrows():
            for col in gene_cols:
                try:
                    val = float(row[col])
                    gene = col.split(" (")[0]
                    conn.execute("INSERT INTO cnv VALUES (?,?,?,?)",
                                 (cases_name, gene, val, _cnv_status(val)))
                    inserted_cnv += 1
                except (ValueError, TypeError):
                    pass

        print(f"  {cases_name} ({ccle_name}): {inserted_mut} mutations, {inserted_cnv} CNV rows added")
        inserted_mut = inserted_cnv = 0

    conn.commit()
    conn.close()


def patch_transcriptomics():
    from src.data.etl_transcriptomics import META_COLS, _log1p_to_tpm
    import numpy as np

    model_df = pd.read_csv(RAW / "model.csv", usecols=["ModelID","CellLineName"])
    model_map = dict(zip(model_df["ModelID"], model_df["CellLineName"]))
    # Override CCLE names with cases.yaml names for the 4 mismatched lines
    for cases_name, (ccle_name, model_id) in ALIASES.items():
        model_map[model_id] = cases_name

    conn = sqlite3.connect(DB_TRAN)
    # Which of the 4 still need to be added?
    to_add = []
    for cases_name in ALIASES:
        existing = conn.execute(
            "SELECT COUNT(*) FROM expression WHERE cell_line=?", (cases_name,)
        ).fetchone()[0]
        if existing > 0:
            print(f"  {cases_name}: already in transcriptomics DB, skipping")
        else:
            to_add.append(cases_name)
    conn.close()

    if not to_add:
        print("  Nothing to patch in transcriptomics DB.")
        return

    expr_df = pd.read_csv(RAW / "expression.csv")
    expr_df["cell_line"] = expr_df["ModelID"].map(model_map)
    if "IsDefaultEntryForModel" in expr_df.columns:
        expr_df = expr_df[expr_df["IsDefaultEntryForModel"] == "Yes"]
    expr_df = expr_df[expr_df["cell_line"].isin(to_add)].set_index("cell_line")

    gene_cols = [c for c in expr_df.columns if c not in META_COLS]
    expr_df = expr_df[gene_cols]
    expr_df.columns = [c.split(" (")[0] for c in expr_df.columns]

    gene_means = expr_df.mean(axis=0)
    gene_stds  = expr_df.std(axis=0).replace(0, 1)
    z_scores   = (expr_df - gene_means) / gene_stds

    conn = sqlite3.connect(DB_TRAN)
    for cell_line in expr_df.index:
        inserted = 0
        for gene in expr_df.columns:
            try:
                tpm = _log1p_to_tpm(float(expr_df.loc[cell_line, gene]))
                z   = float(z_scores.loc[cell_line, gene])
                pct = float(np.sum(expr_df[gene] <= expr_df.loc[cell_line, gene]) / len(expr_df))
                conn.execute("INSERT INTO expression VALUES (?,?,?,?,?)",
                             (cell_line, gene, tpm, z, pct))
                inserted += 1
            except (ValueError, TypeError):
                pass
        print(f"  {cell_line}: {inserted} expression rows added")
    conn.commit()
    conn.close()


def discover_missing(cases_path: Path = Path("cases.yaml")) -> None:
    """Cross-reference every cell line in cases.yaml against CCLE model.csv.

    Prints any cell line that is absent from the DB under its cases.yaml name
    so it can be added to ALIASES. Run this whenever cases.yaml changes.
    """
    import yaml
    model_df = pd.read_csv(RAW / "model.csv", usecols=["ModelID", "CellLineName", "StrippedCellLineName"])
    # Build lookup: lowercase stripped name → canonical CCLE CellLineName
    ccle_names = {row["StrippedCellLineName"].lower(): row["CellLineName"]
                  for _, row in model_df.iterrows()
                  if pd.notna(row["StrippedCellLineName"])}

    with open(cases_path) as f:
        data = yaml.safe_load(f)
    case_lines = sorted({c["cell_line"] for c in data["cases"]})

    conn_gen = sqlite3.connect(DB_GEN)
    conn_tran = sqlite3.connect(DB_TRAN)

    print(f"\nCell line coverage audit ({len(case_lines)} unique lines in {cases_path}):")
    print(f"  {'Cell line':<25}  {'In genomics DB':<16}  {'In transcriptomics DB':<22}  {'CCLE match'}")
    print("  " + "-" * 85)
    for cl in case_lines:
        in_gen  = conn_gen.execute("SELECT COUNT(*) FROM mutations WHERE cell_line=?", (cl,)).fetchone()[0] > 0
        in_tran = conn_tran.execute("SELECT COUNT(*) FROM expression WHERE cell_line=?", (cl,)).fetchone()[0] > 0
        # Try to find the CCLE canonical name
        stripped = cl.lower().replace("-", "").replace(" ", "").replace("_", "")
        ccle_match = ccle_names.get(stripped, "NOT FOUND")
        flag = "" if (in_gen and in_tran) else " ⚠️"
        print(f"  {cl:<25}  {'yes' if in_gen else 'NO':<16}  {'yes' if in_tran else 'NO':<22}  {ccle_match}{flag}")

    conn_gen.close()
    conn_tran.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "discover":
        discover_missing()
    else:
        print("Patching genomics DB...")
        patch_genomics()
        print("Patching transcriptomics DB...")
        patch_transcriptomics()
        print("Done.")
