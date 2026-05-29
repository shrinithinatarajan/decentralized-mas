import sqlite3
import pytest
from pathlib import Path


@pytest.fixture
def genomics_db(tmp_path):
    db = tmp_path / "genomics.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE mutations (
            cell_line TEXT, gene TEXT, mutation TEXT,
            mutation_type TEXT, cosmic_id TEXT, is_driver INTEGER DEFAULT 0
        );
        CREATE TABLE cnv (
            cell_line TEXT, gene TEXT, cnv_value REAL, status TEXT
        );
        INSERT INTO mutations VALUES ('A375','BRAF','V600E','missense','COSM476',1);
        INSERT INTO cnv VALUES ('A375','BRAF',2.0,'neutral');
    """)
    conn.commit()
    conn.close()
    return db


@pytest.fixture
def transcriptomics_db(tmp_path):
    db = tmp_path / "transcriptomics.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE expression (
            cell_line TEXT, gene TEXT, tpm REAL,
            z_score REAL, percentile REAL
        );
        INSERT INTO expression VALUES ('A375','BRAF',45.2,1.3,0.87);
        INSERT INTO expression VALUES ('A375','EGFR',2.1,-2.3,0.05);
    """)
    conn.commit()
    conn.close()
    return db


@pytest.fixture
def pharmacology_db(tmp_path):
    db = tmp_path / "pharmacology.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE drug_response (
            cell_line TEXT, drug TEXT, ln_ic50 REAL,
            auc REAL, z_score REAL, label TEXT
        );
        CREATE TABLE drug_info (
            drug TEXT PRIMARY KEY, target_genes TEXT,
            moa TEXT, pathway TEXT, drug_class TEXT
        );
        CREATE TABLE sensitivity_profile (
            drug TEXT, mutation TEXT, median_ic50 REAL,
            resistant_fraction REAL, n_cell_lines INTEGER
        );
        INSERT INTO drug_response VALUES ('A375','Vemurafenib',-1.2,0.3,-1.5,'SENSITIVE');
        INSERT INTO drug_info VALUES ('Vemurafenib','BRAF','BRAF inhibitor','MAPK','targeted');
        INSERT INTO sensitivity_profile VALUES ('Vemurafenib','V600E',0.35,0.1,120);
    """)
    conn.commit()
    conn.close()
    return db


@pytest.fixture
def pathways_db(tmp_path):
    db = tmp_path / "pathways.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE pathway_genes (
            pathway_id TEXT, gene TEXT, role TEXT, position TEXT
        );
        CREATE TABLE bypass_routes (
            pathway_id TEXT, blocked_gene TEXT,
            bypass_gene TEXT, bypass_exists INTEGER
        );
        CREATE TABLE upstream_regulators (
            gene TEXT, regulator TEXT, relationship TEXT, pathway TEXT
        );
        INSERT INTO pathway_genes VALUES ('hsa04010','BRAF','kinase','mid');
        INSERT INTO bypass_routes VALUES ('hsa04010','BRAF','MEK2',1);
        INSERT INTO upstream_regulators VALUES ('BRAF','RAS','activates','hsa04010');
    """)
    conn.commit()
    conn.close()
    return db
