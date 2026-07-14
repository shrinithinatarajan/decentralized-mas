"""Sample a held-out test set from GDSC2 using cell lines NOT in the dev set.

Steps:
  1. Load GDSC2 ic50.csv, filter to clear labels (|z| > 1.5)
  2. Exclude cell lines already in cases.yaml
  3. Keep only cell lines present in CCLE model.csv (so we can add genomic data)
  4. Stratify by pathway -> tier and label -> sample 20 balanced cases
  5. Patch genomics, transcriptomics, and pharmacology DBs for new cell lines
  6. Write cases_held_out.yaml

Usage:
    PYTHONPATH=. python src/data/sample_held_out.py
"""
import random
import sqlite3
from pathlib import Path

import pandas as pd
import yaml

RAW        = Path("src/data/raw/ccle")
GDSC2      = Path("src/data/raw/gdsc2/ic50.csv")
DB_GEN     = Path("src/data/processed/genomics.db")
DB_TRAN    = Path("src/data/processed/transcriptomics.db")
DB_PHARM   = Path("src/data/processed/pharmacology.db")
OUT_YAML   = Path("cases_held_out.yaml")

RNG_SEED   = 42
N_CASES    = 20   # 10 SENSITIVE + 10 RESISTANT
Z_THRESH   = 1.5  # cleaner labels than the dev-set threshold of 0.5

# Map GDSC2 PATHWAY_NAME -> axiom tier (same logic as cases.yaml)
PATHWAY_TIER: dict[str, str] = {
    "EGFR signaling":              "T1_STRUCTURAL",
    "ERK MAPK signaling":          "T1_STRUCTURAL",
    "p53 pathway":                 "T1_STRUCTURAL",
    "ABL signaling":               "T1_STRUCTURAL",
    "Hormone-related":             "T2_TRANSCRIPTIONAL_GATE",
    "PI3K/MTOR signaling":         "T3_PATHWAY_BYPASS",
    "RTK signaling":               "T3_PATHWAY_BYPASS",
    "WNT signaling":               "T3_PATHWAY_BYPASS",
    "IGF1R signaling":             "T3_PATHWAY_BYPASS",
    "DNA replication":             "T4_PHARMACOLOGICAL_PRIOR",
    "Apoptosis regulation":        "T4_PHARMACOLOGICAL_PRIOR",
    "Genome integrity":            "T4_PHARMACOLOGICAL_PRIOR",
    "Cell cycle":                  "T5_STATISTICAL_CONSENSUS",
    "Mitosis":                     "T5_STATISTICAL_CONSENSUS",
    "Chromatin other":             "T5_STATISTICAL_CONSENSUS",
    "Chromatin histone methylation": "T5_STATISTICAL_CONSENSUS",
    "Chromatin histone acetylation": "T5_STATISTICAL_CONSENSUS",
    "Other, kinases":              "T5_STATISTICAL_CONSENSUS",
    "Other":                       "T5_STATISTICAL_CONSENSUS",
    "Unclassified":                "T5_STATISTICAL_CONSENSUS",
    "Protein stability and degradation": "T5_STATISTICAL_CONSENSUS",
    "Metabolism":                  "T5_STATISTICAL_CONSENSUS",
}

# Non-informative putative targets (skip these for cleaner evaluation)
SKIP_TARGETS = {
    "Broad spectrum kinase inhibitor", "Alkylating agent",
    "Topoisomerase inhibitor", "Antimetabolite", "Mitotic inhibitor",
}


def _load_dev_pairs() -> set[tuple[str, str]]:
    with open("cases.yaml") as f:
        data = yaml.safe_load(f)
    return {(c["cell_line"], c["drug"]) for c in data["cases"]}


def _load_dev_cell_lines() -> set[str]:
    with open("cases.yaml") as f:
        data = yaml.safe_load(f)
    return {c["cell_line"] for c in data["cases"]}


def _sample_candidates(dev_lines: set[str], dev_pairs: set[tuple[str, str]]) -> pd.DataFrame:
    """Return filtered, labelled GDSC2 rows excluding dev cases."""
    df = pd.read_csv(GDSC2)
    df.columns = [c.upper() for c in df.columns]

    # Exclude dev cell lines entirely for maximal independence
    df = df[~df["CELL_LINE_NAME"].isin(dev_lines)]

    # Clear labels only
    df = df[df["Z_SCORE"].abs() >= Z_THRESH].copy()
    df["label"] = df["Z_SCORE"].apply(lambda z: "SENSITIVE" if z < 0 else "RESISTANT")

    # Assign tier
    df["tier"] = df["PATHWAY_NAME"].map(PATHWAY_TIER).fillna("T5_STATISTICAL_CONSENSUS")

    # Drop non-informative targets
    df = df[~df["PUTATIVE_TARGET"].isin(SKIP_TARGETS)]
    df = df.dropna(subset=["PUTATIVE_TARGET"])
    df = df[df["PUTATIVE_TARGET"].str.strip() != ""]

    return df


def _pick_cases(candidates: pd.DataFrame, ccle_names: set[str]) -> pd.DataFrame:
    """Stratified sample: 10 SENSITIVE + 10 RESISTANT across diverse tiers."""
    rng = random.Random(RNG_SEED)

    # Only cell lines we can extend the DB for
    candidates = candidates[candidates["CELL_LINE_NAME"].isin(ccle_names)].copy()

    tier_order = [
        "T1_STRUCTURAL",
        "T3_PATHWAY_BYPASS",
        "T4_PHARMACOLOGICAL_PRIOR",
        "T2_TRANSCRIPTIONAL_GATE",
        "T5_STATISTICAL_CONSENSUS",
    ]

    picked: list[dict] = []
    used_lines: set[str] = set()

    for label in ["SENSITIVE", "RESISTANT"]:
        needed = N_CASES // 2
        pool = candidates[candidates["label"] == label].copy()
        # Prefer high |z| and diverse tiers and cell lines
        for tier in tier_order * 4:   # cycle tiers until we have enough
            if len(picked) >= needed * 2 - (needed if label == "RESISTANT" else 0):
                # recalculate per-label count
                pass
            tier_pool = pool[
                (pool["tier"] == tier) &
                (~pool["CELL_LINE_NAME"].isin(used_lines))
            ].sort_values("Z_SCORE", key=abs, ascending=False)
            if tier_pool.empty:
                continue
            row = tier_pool.iloc[0]
            picked.append({
                "cell_line": row["CELL_LINE_NAME"],
                "drug":      row["DRUG_NAME"],
                "label":     label,
                "z_score":   round(float(row["Z_SCORE"]), 3),
                "pathway":   str(row["PATHWAY_NAME"]),
                "putative_target": str(row["PUTATIVE_TARGET"]),
                "axiom_tier": row["tier"],
                "split":     "heldout",
                "notes":     f"Held-out test case; GDSC2 z={row['Z_SCORE']:.3f}",
            })
            used_lines.add(row["CELL_LINE_NAME"])
            # remove this cell line from pool to avoid duplicates
            pool = pool[pool["CELL_LINE_NAME"] != row["CELL_LINE_NAME"]]
            if sum(1 for p in picked if p["label"] == label) >= needed:
                break

    return pd.DataFrame(picked)


def _patch_genomics(new_lines: list[str], model_id_map: dict[str, str]) -> None:
    """Add mutations + CNV for new cell lines to genomics.db."""
    conn = sqlite3.connect(DB_GEN)
    to_add = [cl for cl in new_lines
              if conn.execute("SELECT COUNT(*) FROM mutations WHERE cell_line=?", (cl,)).fetchone()[0] == 0]
    if not to_add:
        print("  Genomics: all new lines already in DB.")
        conn.close()
        return

    target_ids = {model_id_map[cl] for cl in to_add if cl in model_id_map}
    print(f"  Genomics: adding {len(to_add)} cell lines ({len(target_ids)} ModelIDs)...")

    # Mutations (chunked to avoid loading 553MB at once)
    inserted_mut = 0
    for chunk in pd.read_csv(RAW / "mutations.csv", chunksize=50_000, low_memory=False):
        rows = chunk[chunk["ModelID"].isin(target_ids)]
        for _, row in rows.iterrows():
            cl = next((c for c, mid in model_id_map.items() if mid == row["ModelID"] and c in to_add), None)
            if cl is None:
                continue
            is_driver = int(bool(row.get("OncogeneHighImpact")) or bool(row.get("TumorSuppressorHighImpact")))
            conn.execute("INSERT INTO mutations VALUES (?,?,?,?,?,?)",
                (cl, row.get("HugoSymbol", ""),
                 str(row["ProteinChange"]) if pd.notna(row.get("ProteinChange")) else "",
                 row.get("VariantType", ""), "", is_driver))
            inserted_mut += 1
    print(f"    {inserted_mut} mutation rows inserted")

    # CNV
    cnv_df = pd.read_csv(RAW / "cnv.csv")
    if "IsDefaultEntryForModel" in cnv_df.columns:
        cnv_df = cnv_df[cnv_df["IsDefaultEntryForModel"] == "Yes"]
    cnv_df = cnv_df[cnv_df["ModelID"].isin(target_ids)]
    meta_cols = {"ModelID","SequencingID","ModelConditionID","IsDefaultEntryForMC",
                 "IsDefaultEntryForModel","cell_line","Unnamed: 0"}
    gene_cols = [c for c in cnv_df.columns if c not in meta_cols]
    inserted_cnv = 0
    for _, row in cnv_df.iterrows():
        cl = next((c for c, mid in model_id_map.items() if mid == row["ModelID"] and c in to_add), None)
        if cl is None:
            continue
        for col in gene_cols:
            try:
                val = float(row[col])
                gene = col.split(" (")[0]
                status = ("homozygous_deletion" if val < 0.5 else "hemizygous_deletion" if val < 1.5
                          else "neutral" if val < 2.5 else "gain" if val < 4.0 else "amplification")
                conn.execute("INSERT INTO cnv VALUES (?,?,?,?)", (cl, gene, val, status))
                inserted_cnv += 1
            except (ValueError, TypeError):
                pass
    print(f"    {inserted_cnv} CNV rows inserted")
    conn.commit()
    conn.close()


def _patch_transcriptomics(new_lines: list[str], model_id_map: dict[str, str]) -> None:
    """Add expression rows for new cell lines."""
    import numpy as np
    from src.data.etl_transcriptomics import META_COLS, _log1p_to_tpm

    conn = sqlite3.connect(DB_TRAN)
    to_add = [cl for cl in new_lines
              if conn.execute("SELECT COUNT(*) FROM expression WHERE cell_line=?", (cl,)).fetchone()[0] == 0]
    if not to_add:
        print("  Transcriptomics: all new lines already in DB.")
        conn.close()
        return

    target_ids = {model_id_map[cl] for cl in to_add if cl in model_id_map}
    print(f"  Transcriptomics: adding {len(to_add)} cell lines...")

    expr_df = pd.read_csv(RAW / "expression.csv")
    if "IsDefaultEntryForModel" in expr_df.columns:
        expr_df = expr_df[expr_df["IsDefaultEntryForModel"] == "Yes"]
    expr_df = expr_df[expr_df["ModelID"].isin(target_ids)]

    if expr_df.empty:
        print("    No expression data found for these ModelIDs in expression.csv")
        conn.close()
        return

    gene_cols = [c for c in expr_df.columns if c not in META_COLS and c != "ModelID"]
    expr_df = expr_df.set_index("ModelID")[gene_cols]
    expr_df.columns = [c.split(" (")[0] for c in expr_df.columns]

    gene_means = expr_df.mean(axis=0)
    gene_stds  = expr_df.std(axis=0).replace(0, 1)
    z_scores   = (expr_df - gene_means) / gene_stds

    inserted = 0
    for model_id in expr_df.index:
        cl = next((c for c, mid in model_id_map.items() if mid == model_id and c in to_add), None)
        if cl is None:
            continue
        for gene in expr_df.columns:
            try:
                tpm = _log1p_to_tpm(float(expr_df.loc[model_id, gene]))
                z   = float(z_scores.loc[model_id, gene])
                pct = float(np.sum(expr_df[gene] <= expr_df.loc[model_id, gene]) / len(expr_df))
                conn.execute("INSERT INTO expression VALUES (?,?,?,?,?)", (cl, gene, tpm, z, pct))
                inserted += 1
            except (ValueError, TypeError):
                pass
    print(f"    {inserted} expression rows inserted")
    conn.commit()
    conn.close()


def _patch_pharmacology(new_lines: list[str]) -> None:
    """Add drug response rows for new cell lines from GDSC2."""
    conn = sqlite3.connect(DB_PHARM)
    to_add = [cl for cl in new_lines
              if conn.execute("SELECT COUNT(*) FROM drug_response WHERE cell_line=?", (cl,)).fetchone()[0] == 0]
    if not to_add:
        print("  Pharmacology: all new lines already in DB.")
        conn.close()
        return

    print(f"  Pharmacology: adding {len(to_add)} cell lines...")
    df = pd.read_csv(GDSC2)
    df.columns = [c.upper() for c in df.columns]
    df = df[df["CELL_LINE_NAME"].isin(to_add)]

    def _label(z: float) -> str:
        return "SENSITIVE" if z < -0.5 else "RESISTANT" if z > 0.5 else "UNCERTAIN"

    inserted = 0
    for _, row in df.iterrows():
        conn.execute("INSERT INTO drug_response VALUES (?,?,?,?,?,?)",
            (row["CELL_LINE_NAME"], row["DRUG_NAME"],
             float(row["LN_IC50"]), float(row["AUC"]),
             float(row["Z_SCORE"]), _label(float(row["Z_SCORE"]))))
        inserted += 1

    # drug_info: add any missing drugs
    for drug in df["DRUG_NAME"].unique():
        exists = conn.execute("SELECT COUNT(*) FROM drug_info WHERE drug=?", (drug,)).fetchone()[0]
        if not exists:
            row = df[df["DRUG_NAME"] == drug].iloc[0]
            conn.execute("INSERT OR REPLACE INTO drug_info VALUES (?,?,?,?,?)",
                (drug, str(row.get("PUTATIVE_TARGET", "")),
                 "", str(row.get("PATHWAY_NAME", "")), ""))
    print(f"    {inserted} drug response rows inserted")
    conn.commit()
    conn.close()


def main() -> None:
    rng = random.Random(RNG_SEED)

    print("Loading CCLE model map...")
    model_df = pd.read_csv(RAW / "model.csv", usecols=["ModelID", "CellLineName", "StrippedCellLineName"])
    # ModelID -> CellLineName (CCLE canonical)
    mid_to_name = dict(zip(model_df["ModelID"], model_df["CellLineName"]))
    # CellLineName -> ModelID (for reverse lookup)
    name_to_mid = dict(zip(model_df["CellLineName"], model_df["ModelID"]))
    ccle_names = set(model_df["CellLineName"].dropna())

    dev_lines = _load_dev_cell_lines()
    dev_pairs = _load_dev_pairs()
    print(f"Dev set: {len(dev_lines)} cell lines, {len(dev_pairs)} (cell_line, drug) pairs")

    print("Sampling candidates from GDSC2...")
    candidates = _sample_candidates(dev_lines, dev_pairs)
    print(f"  {len(candidates)} candidate rows after filtering (|z|>={Z_THRESH}, not in dev)")
    print(f"  Unique cell lines in candidates: {candidates['CELL_LINE_NAME'].nunique()}")
    # Filter to lines in CCLE
    candidates = candidates[candidates["CELL_LINE_NAME"].isin(ccle_names)]
    print(f"  After CCLE filter: {candidates['CELL_LINE_NAME'].nunique()} unique cell lines")

    print("Picking stratified sample...")
    picked_df = _pick_cases(candidates, ccle_names)
    print(f"  Picked {len(picked_df)} cases:")
    print(picked_df[["cell_line","drug","label","z_score","axiom_tier"]].to_string(index=False))

    new_lines = picked_df["cell_line"].tolist()
    # Build model_id_map for new lines
    model_id_map = {cl: name_to_mid[cl] for cl in new_lines if cl in name_to_mid}

    print("\nPatching databases for new cell lines...")
    _patch_genomics(new_lines, model_id_map)
    _patch_transcriptomics(new_lines, model_id_map)
    _patch_pharmacology(new_lines)

    print(f"\nWriting {OUT_YAML}...")
    cases_out = []
    for _, row in picked_df.iterrows():
        cases_out.append({
            "cell_line":      row["cell_line"],
            "drug":           row["drug"],
            "label":          row["label"],
            "z_score":        row["z_score"],
            "pathway":        row["pathway"],
            "putative_target": row["putative_target"],
            "axiom_tier":     row["axiom_tier"],
            "split":          "heldout",
            "notes":          row["notes"],
        })
    with open(OUT_YAML, "w") as f:
        yaml.dump({"cases": cases_out}, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f"Done. Held-out test set written to {OUT_YAML}")
    label_counts = picked_df["label"].value_counts().to_dict()
    tier_counts  = picked_df["axiom_tier"].value_counts().to_dict()
    print(f"  Labels: {label_counts}")
    print(f"  Tiers:  {tier_counts}")


if __name__ == "__main__":
    main()
