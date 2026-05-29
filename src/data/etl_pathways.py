import sqlite3
import time
import requests
from pathlib import Path

KEGG_REST = "https://rest.kegg.jp"

PATHWAY_GENES_SCHEMA = """
CREATE TABLE IF NOT EXISTS pathway_genes (
    pathway_id TEXT, gene TEXT, role TEXT, position TEXT
)"""

BYPASS_ROUTES_SCHEMA = """
CREATE TABLE IF NOT EXISTS bypass_routes (
    pathway_id TEXT, blocked_gene TEXT,
    bypass_gene TEXT, bypass_exists INTEGER
)"""

UPSTREAM_REGULATORS_SCHEMA = """
CREATE TABLE IF NOT EXISTS upstream_regulators (
    gene TEXT, regulator TEXT, relationship TEXT, pathway TEXT
)"""

# Curated bypass routes — KEGG REST does not expose these directly
KNOWN_BYPASSES: dict[tuple[str, str], list[str]] = {
    ("hsa04010", "BRAF"):   ["MAP2K2"],         # MAPK: MEK2 bypass
    ("hsa04012", "EGFR"):   ["MET", "ERBB2"],   # ErbB: MET/HER2 bypass
    ("hsa04151", "PIK3CA"): ["AKT2"],           # PI3K-Akt: AKT2 bypass
    ("hsa04115", "TP53"):   ["MDM2"],           # p53: MDM2-mediated bypass
    ("hsa04310", "CTNNB1"): ["ROR2"],           # Wnt: non-canonical bypass
}


def parse_kegg_pathway(raw_text: str) -> list[str]:
    """Extract Entrez IDs from KEGG /link response (tab-separated pathway\tgene)."""
    genes = []
    for line in raw_text.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            genes.append(parts[1].replace("hsa:", ""))
    return genes


def _fetch_pathway_genes(pathway_id: str, gene_map: dict[str, str]) -> list[str]:
    try:
        resp = requests.get(f"{KEGG_REST}/link/hsa/{pathway_id}", timeout=10)
        resp.raise_for_status()
        entrez_ids = parse_kegg_pathway(resp.text)
        return [gene_map.get(eid, eid) for eid in entrez_ids]
    except Exception:
        return []


def create_pathways_db(
    db_path: Path,
    pathway_ids: list[str],
    gene_map: dict[str, str],
    extra_genes: dict[str, list[str]] | None = None,
) -> None:
    """Build pathways.db from KEGG REST. extra_genes bypasses HTTP for testing."""
    conn = sqlite3.connect(db_path)
    conn.execute(PATHWAY_GENES_SCHEMA)
    conn.execute(BYPASS_ROUTES_SCHEMA)
    conn.execute(UPSTREAM_REGULATORS_SCHEMA)

    all_pathway_genes: dict[str, list[str]] = dict(extra_genes or {})

    for pathway_id in pathway_ids:
        genes = _fetch_pathway_genes(pathway_id, gene_map)
        all_pathway_genes.setdefault(pathway_id, []).extend(genes)
        time.sleep(0.4)  # KEGG rate limit: max 3 req/s

    for pathway_id, genes in all_pathway_genes.items():
        for gene in genes:
            conn.execute(
                "INSERT INTO pathway_genes VALUES (?,?,?,?)",
                (pathway_id, gene, "", ""),
            )
        for (pid, blocked), bypass_genes in KNOWN_BYPASSES.items():
            if pid == pathway_id:
                for bg in bypass_genes:
                    conn.execute(
                        "INSERT INTO bypass_routes VALUES (?,?,?,?)",
                        (pathway_id, blocked, bg, 1),
                    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    PATHWAY_IDS = ["hsa04010", "hsa04012", "hsa04151", "hsa04115", "hsa04310"]
    create_pathways_db(
        Path("src/data/processed/pathways.db"),
        PATHWAY_IDS,
        gene_map={},
    )
    print("pathways.db built")
