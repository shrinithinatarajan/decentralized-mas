import sqlite3
import pytest
import pandas as pd
import yaml
from pathlib import Path
from src.data.loader import load_cases, get_db_connection, Case
from src.data.etl_genomics import create_genomics_db, MUTATIONS_SCHEMA, CNV_SCHEMA


def test_load_cases(tmp_path):
    cases_file = tmp_path / "cases.yaml"
    cases_file.write_text("""
cases:
  - cell_line: A375
    drug: Vemurafenib
    label: SENSITIVE
    notes: "test"
""")
    cases = load_cases(cases_file)
    assert len(cases) == 1
    assert cases[0].cell_line == "A375"
    assert cases[0].label == "SENSITIVE"


def test_get_db_connection(genomics_db):
    conn = get_db_connection(genomics_db)
    row = conn.execute("SELECT * FROM mutations WHERE cell_line='A375'").fetchone()
    assert row is not None
    conn.close()


def test_load_cases_missing_notes(tmp_path):
    cases_file = tmp_path / "cases.yaml"
    cases_file.write_text("""
cases:
  - cell_line: MCF7
    drug: Fulvestrant
    label: SENSITIVE
""")
    cases = load_cases(cases_file)
    assert cases[0].notes == ""


def test_load_cases_invalid_label(tmp_path):
    cases_file = tmp_path / "cases.yaml"
    cases_file.write_text("""
cases:
  - cell_line: A375
    drug: Vemurafenib
    label: SENSITVE
""")
    with pytest.raises(ValueError, match="Invalid label"):
        load_cases(cases_file)


def test_load_cases_extended_fields(tmp_path):
    cases_file = tmp_path / "cases.yaml"
    cases_file.write_text("""
cases:
  - cell_line: A375
    drug: Dabrafenib
    label: SENSITIVE
    z_score: -4.169
    pathway: "ERK MAPK signaling"
    putative_target: "BRAF"
    axiom_tier: T1_STRUCTURAL
    notes: "BRAF V600E melanoma"
""")
    cases = load_cases(cases_file)
    assert cases[0].z_score == -4.169
    assert cases[0].pathway == "ERK MAPK signaling"
    assert cases[0].axiom_tier == "T1_STRUCTURAL"


def _write_model_csv(tmp_path, mapping: dict) -> "Path":
    model_csv = tmp_path / "model.csv"
    rows = "ModelID,CellLineName\n" + "\n".join(f"{k},{v}" for k, v in mapping.items())
    model_csv.write_text(rows)
    return model_csv


def _write_mut_csv(tmp_path, rows: list[tuple]) -> "Path":
    """rows: (ModelID, HugoSymbol, ProteinChange, VariantType, OncogeneHighImpact, TumorSuppressorHighImpact)"""
    mut_csv = tmp_path / "mutations.csv"
    header = "ModelID,HugoSymbol,ProteinChange,VariantType,VepImpact,OncogeneHighImpact,TumorSuppressorHighImpact,IsDefaultEntryForModel\n"
    body = "\n".join(",".join(str(v) for v in r) for r in rows)
    mut_csv.write_text(header + body)
    return mut_csv


def _write_cnv_csv(tmp_path, model_genes: dict) -> "Path":
    """model_genes: {model_id: {gene: value}}"""
    cnv_csv = tmp_path / "cnv.csv"
    all_genes = sorted({g for genes in model_genes.values() for g in genes})
    lines = ["ModelID,IsDefaultEntryForModel," + ",".join(f"{g} (0)" for g in all_genes)]
    for mid, genes in model_genes.items():
        vals = ",".join(str(genes.get(g, 2.0)) for g in all_genes)
        lines.append(f"{mid},Yes,{vals}")
    cnv_csv.write_text("\n".join(lines))
    return cnv_csv


def test_create_genomics_db_schema(tmp_path):
    model_csv = _write_model_csv(tmp_path, {"ACH-000001": "A375"})
    mut_csv = _write_mut_csv(tmp_path, [
        ("ACH-000001", "BRAF", "p.V600E", "SNV", "HIGH", "True", "False", "Yes")
    ])
    cnv_csv = _write_cnv_csv(tmp_path, {"ACH-000001": {"BRAF": 2.1}})
    db_path = tmp_path / "genomics.db"

    create_genomics_db(db_path, mut_csv, cnv_csv, cell_lines=["A375"], model_csv=model_csv)

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT * FROM mutations WHERE cell_line='A375'").fetchone() is not None
    assert conn.execute("SELECT * FROM cnv WHERE cell_line='A375'").fetchone() is not None
    conn.close()


def test_create_genomics_db_driver_flag(tmp_path):
    model_csv = _write_model_csv(tmp_path, {"ACH-000001": "A375"})
    mut_csv = _write_mut_csv(tmp_path, [
        ("ACH-000001", "BRAF", "p.V600E",  "SNV", "HIGH",     "True",  "False", "Yes"),
        ("ACH-000001", "TP53", "p.R248W",  "SNV", "MODERATE", "False", "False", "Yes"),
    ])
    cnv_csv = _write_cnv_csv(tmp_path, {"ACH-000001": {"BRAF": 2.1, "TP53": 1.0}})
    db_path = tmp_path / "genomics.db"

    create_genomics_db(db_path, mut_csv, cnv_csv, cell_lines=["A375"], model_csv=model_csv)

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT is_driver FROM mutations WHERE gene='BRAF'").fetchone()[0] == 1
    assert conn.execute("SELECT is_driver FROM mutations WHERE gene='TP53'").fetchone()[0] == 0
    conn.close()


def test_create_genomics_db_cnv_status(tmp_path):
    model_csv = _write_model_csv(tmp_path, {"ACH-000001": "A375"})
    mut_csv = _write_mut_csv(tmp_path, [])
    cnv_csv = _write_cnv_csv(tmp_path, {"ACH-000001": {
        "GENE1": 0.2, "GENE2": 1.0, "GENE3": 2.1, "GENE4": 3.5, "GENE5": 6.0
    }})
    db_path = tmp_path / "genomics.db"

    create_genomics_db(db_path, mut_csv, cnv_csv, cell_lines=["A375"], model_csv=model_csv)

    conn = sqlite3.connect(db_path)
    statuses = {row[1]: row[3] for row in conn.execute("SELECT * FROM cnv").fetchall()}
    assert statuses["GENE1"] == "homozygous_deletion"
    assert statuses["GENE2"] == "hemizygous_deletion"
    assert statuses["GENE3"] == "neutral"
    assert statuses["GENE4"] == "gain"
    assert statuses["GENE5"] == "amplification"
    conn.close()


def _write_expr_csv(tmp_path, model_genes: dict) -> "Path":
    """model_genes: {model_id: {gene: value}} — same format as cnv"""
    expr_csv = tmp_path / "expression.csv"
    all_genes = sorted({g for genes in model_genes.values() for g in genes})
    lines = ["ModelID,IsDefaultEntryForModel," + ",".join(f"{g} (0)" for g in all_genes)]
    for mid, genes in model_genes.items():
        vals = ",".join(str(genes.get(g, 0.0)) for g in all_genes)
        lines.append(f"{mid},Yes,{vals}")
    expr_csv.write_text("\n".join(lines))
    return expr_csv


def test_create_transcriptomics_db_basic(tmp_path):
    from src.data.etl_transcriptomics import create_transcriptomics_db
    model_csv = _write_model_csv(tmp_path, {"ACH-000001": "A375"})
    expr_csv = _write_expr_csv(tmp_path, {"ACH-000001": {"BRAF": 4.2, "EGFR": 1.1}})
    db_path = tmp_path / "transcriptomics.db"

    create_transcriptomics_db(db_path, expr_csv, cell_lines=["A375"], model_csv=model_csv)

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT * FROM expression WHERE cell_line='A375' AND gene='BRAF'").fetchone()
    assert row is not None
    assert row[2] > 0  # tpm > 0
    conn.close()


def test_create_transcriptomics_db_strips_entrez_id(tmp_path):
    from src.data.etl_transcriptomics import create_transcriptomics_db
    model_csv = _write_model_csv(tmp_path, {"ACH-000001": "A375"})
    expr_csv = _write_expr_csv(tmp_path, {"ACH-000001": {"BRAF": 3.0, "MYC": 5.1}})
    db_path = tmp_path / "transcriptomics.db"

    create_transcriptomics_db(db_path, expr_csv, cell_lines=["A375"], model_csv=model_csv)

    conn = sqlite3.connect(db_path)
    genes = {row[1] for row in conn.execute("SELECT * FROM expression").fetchall()}
    assert "BRAF" in genes and "MYC" in genes
    assert not any("(" in g for g in genes)
    conn.close()


def test_create_transcriptomics_db_z_scores(tmp_path):
    from src.data.etl_transcriptomics import create_transcriptomics_db
    model_csv = _write_model_csv(tmp_path, {
        "ACH-000001": "LOW", "ACH-000002": "MID", "ACH-000003": "HIGH"
    })
    expr_csv = _write_expr_csv(tmp_path, {
        "ACH-000001": {"BRAF": 1.0},
        "ACH-000002": {"BRAF": 3.0},
        "ACH-000003": {"BRAF": 5.0},
    })
    db_path = tmp_path / "transcriptomics.db"

    create_transcriptomics_db(db_path, expr_csv, cell_lines=["LOW", "MID", "HIGH"], model_csv=model_csv)

    conn = sqlite3.connect(db_path)
    rows = {r[0]: r[3] for r in conn.execute("SELECT cell_line, gene, tpm, z_score FROM expression").fetchall()}
    assert rows["LOW"] < rows["MID"] < rows["HIGH"]
    conn.close()


def test_create_transcriptomics_db_filters_cell_lines(tmp_path):
    from src.data.etl_transcriptomics import create_transcriptomics_db
    model_csv = _write_model_csv(tmp_path, {"ACH-000001": "A375", "ACH-000002": "MCF7"})
    expr_csv = _write_expr_csv(tmp_path, {
        "ACH-000001": {"BRAF": 4.2},
        "ACH-000002": {"BRAF": 3.1},
    })
    db_path = tmp_path / "transcriptomics.db"

    create_transcriptomics_db(db_path, expr_csv, cell_lines=["A375"], model_csv=model_csv)

    conn = sqlite3.connect(db_path)
    assert {r[0] for r in conn.execute("SELECT cell_line FROM expression").fetchall()} == {"A375"}
    conn.close()


def test_create_pharmacology_db_basic(tmp_path):
    from src.data.etl_pharmacology import create_pharmacology_db
    db_path = tmp_path / "pharmacology.db"
    ic50_csv = tmp_path / "ic50.csv"
    ic50_csv.write_text(
        "CELL_LINE_NAME,DRUG_NAME,LN_IC50,AUC,Z_SCORE,PUTATIVE_TARGET,PATHWAY_NAME\n"
        "A375,Dabrafenib,-4.1,0.25,-4.2,BRAF,ERK MAPK signaling\n"
    )

    create_pharmacology_db(db_path, ic50_csv, cell_lines=["A375"], fetch_chembl=False)

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT * FROM drug_response WHERE cell_line='A375' AND drug='Dabrafenib'"
    ).fetchone()
    assert row is not None
    assert row[2] == pytest.approx(-4.1, rel=1e-3)  # ln_ic50
    conn.close()


def test_create_pharmacology_db_drug_info(tmp_path):
    from src.data.etl_pharmacology import create_pharmacology_db
    db_path = tmp_path / "pharmacology.db"
    ic50_csv = tmp_path / "ic50.csv"
    ic50_csv.write_text(
        "CELL_LINE_NAME,DRUG_NAME,LN_IC50,AUC,Z_SCORE,PUTATIVE_TARGET,PATHWAY_NAME\n"
        "A375,Dabrafenib,-4.1,0.25,-4.2,BRAF,ERK MAPK signaling\n"
        "MCF7,Dabrafenib,-1.0,0.5,0.2,BRAF,ERK MAPK signaling\n"
    )

    create_pharmacology_db(db_path, ic50_csv, cell_lines=["A375", "MCF7"], fetch_chembl=False)

    conn = sqlite3.connect(db_path)
    info = conn.execute("SELECT * FROM drug_info WHERE drug='Dabrafenib'").fetchone()
    assert info is not None
    assert info[1] == "BRAF"       # target_genes
    assert info[3] == "ERK MAPK signaling"  # pathway
    conn.close()


def test_create_pharmacology_db_sensitivity_label(tmp_path):
    from src.data.etl_pharmacology import create_pharmacology_db
    db_path = tmp_path / "pharmacology.db"
    ic50_csv = tmp_path / "ic50.csv"
    ic50_csv.write_text(
        "CELL_LINE_NAME,DRUG_NAME,LN_IC50,AUC,Z_SCORE,PUTATIVE_TARGET,PATHWAY_NAME\n"
        "A375,Dabrafenib,-4.1,0.25,-2.0,BRAF,ERK MAPK signaling\n"
        "MCF7,Gefitinib,1.0,0.8,1.5,EGFR,EGFR signaling\n"
    )

    create_pharmacology_db(
        db_path, ic50_csv, cell_lines=["A375", "MCF7"], fetch_chembl=False
    )

    conn = sqlite3.connect(db_path)
    sensitive = conn.execute(
        "SELECT label FROM drug_response WHERE cell_line='A375'"
    ).fetchone()[0]
    resistant = conn.execute(
        "SELECT label FROM drug_response WHERE cell_line='MCF7'"
    ).fetchone()[0]
    assert sensitive == "SENSITIVE"
    assert resistant == "RESISTANT"
    conn.close()


def test_create_pharmacology_db_filters_cell_lines(tmp_path):
    from src.data.etl_pharmacology import create_pharmacology_db
    db_path = tmp_path / "pharmacology.db"
    ic50_csv = tmp_path / "ic50.csv"
    ic50_csv.write_text(
        "CELL_LINE_NAME,DRUG_NAME,LN_IC50,AUC,Z_SCORE,PUTATIVE_TARGET,PATHWAY_NAME\n"
        "A375,Dabrafenib,-4.1,0.25,-4.2,BRAF,ERK MAPK signaling\n"
        "NCI-H1975,Gefitinib,0.5,0.7,-0.4,EGFR,EGFR signaling\n"
    )

    create_pharmacology_db(db_path, ic50_csv, cell_lines=["A375"], fetch_chembl=False)

    conn = sqlite3.connect(db_path)
    lines = {r[0] for r in conn.execute("SELECT cell_line FROM drug_response").fetchall()}
    assert lines == {"A375"}
    conn.close()


def test_parse_kegg_pathway():
    from src.data.etl_pathways import parse_kegg_pathway
    raw = "hsa04010\thsa:673\nhsa04010\thsa:5604\n"
    genes = parse_kegg_pathway(raw)
    assert "673" in genes or len(genes) == 2


def test_create_pathways_db_schema(tmp_path):
    from src.data.etl_pathways import create_pathways_db
    db_path = tmp_path / "pathways.db"
    create_pathways_db(db_path, pathway_ids=[], gene_map={})

    conn = sqlite3.connect(db_path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "pathway_genes" in tables
    assert "bypass_routes" in tables
    assert "upstream_regulators" in tables
    conn.close()


def test_create_pathways_db_injects_curated_bypasses(tmp_path):
    from src.data.etl_pathways import create_pathways_db, KNOWN_BYPASSES
    db_path = tmp_path / "pathways.db"
    # Provide genes directly via gene_list injection to avoid HTTP
    create_pathways_db(
        db_path,
        pathway_ids=[],
        gene_map={},
        extra_genes={"hsa04010": ["BRAF", "MAP2K1"]},
    )

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT * FROM pathway_genes WHERE pathway_id='hsa04010'").fetchall()
    assert len(rows) == 2
    conn.close()


def test_create_pathways_db_bypass_routes_written(tmp_path):
    from src.data.etl_pathways import create_pathways_db
    db_path = tmp_path / "pathways.db"
    create_pathways_db(
        db_path,
        pathway_ids=[],
        gene_map={},
        extra_genes={"hsa04010": ["BRAF"]},
    )

    conn = sqlite3.connect(db_path)
    bypass = conn.execute(
        "SELECT * FROM bypass_routes WHERE pathway_id='hsa04010' AND blocked_gene='BRAF'"
    ).fetchall()
    assert len(bypass) > 0
    assert bypass[0][2] == "MAP2K2"   # known bypass gene
    assert bypass[0][3] == 1          # bypass_exists
    conn.close()


def test_create_genomics_db_filters_cell_lines(tmp_path):
    model_csv = _write_model_csv(tmp_path, {"ACH-000001": "A375", "ACH-000002": "MCF7"})
    mut_csv = _write_mut_csv(tmp_path, [
        ("ACH-000001", "BRAF",   "p.V600E",  "SNV", "HIGH", "True",  "False", "Yes"),
        ("ACH-000002", "PIK3CA", "p.H1047R", "SNV", "HIGH", "True",  "False", "Yes"),
    ])
    cnv_csv = _write_cnv_csv(tmp_path, {
        "ACH-000001": {"BRAF": 2.1},
        "ACH-000002": {"BRAF": 1.0},
    })
    db_path = tmp_path / "genomics.db"

    create_genomics_db(db_path, mut_csv, cnv_csv, cell_lines=["A375"], model_csv=model_csv)

    conn = sqlite3.connect(db_path)
    assert {r[0] for r in conn.execute("SELECT cell_line FROM mutations").fetchall()} == {"A375"}
    assert "MCF7" not in {r[0] for r in conn.execute("SELECT cell_line FROM cnv").fetchall()}
    conn.close()
