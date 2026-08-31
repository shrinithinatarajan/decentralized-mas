import json
import os
import sqlite3
from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("Pathway Server")


def _db() -> Path:
    return Path(os.getenv("PATHWAYS_DB", "src/data/processed/pathways.db"))


@mcp.tool()
def get_pathway_genes(pathway_id: str) -> str:
    conn = sqlite3.connect(_db())
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT gene, role, position FROM pathway_genes WHERE pathway_id = ?",
        (pathway_id,),
    ).fetchall()
    conn.close()
    return json.dumps([dict(r) for r in rows])


@mcp.tool()
def check_bypass(pathway_id: str, blocked_gene: str) -> str:
    conn = sqlite3.connect(_db())
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT b.bypass_gene, b.bypass_exists,
               u.relationship
        FROM bypass_routes b
        LEFT JOIN upstream_regulators u
          ON b.bypass_gene = u.gene AND b.pathway_id = u.pathway
        WHERE b.pathway_id = ? AND b.blocked_gene = ?
        """,
        (pathway_id, blocked_gene),
    ).fetchall()
    conn.close()
    bypass_genes = [
        {"gene": r["bypass_gene"], "relationship": r["relationship"]}
        for r in rows if r["bypass_exists"]
    ]
    return json.dumps({
        "bypass_exists": len(bypass_genes) > 0,
        "bypass_genes": bypass_genes,
    })


@mcp.tool()
def find_pathways_for_gene(gene: str) -> str:
    """Return all pathways that contain a given gene, with pathway name and category."""
    conn = sqlite3.connect(_db())
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT pg.pathway_id, pg.role, pg.position,
               pm.name AS pathway_name, pm.category
        FROM pathway_genes pg
        LEFT JOIN pathway_meta pm ON pg.pathway_id = pm.pathway_id
        WHERE pg.gene = ?
        """,
        (gene,),
    ).fetchall()
    conn.close()
    return json.dumps([dict(r) for r in rows])


@mcp.tool()
def get_upstream_regulators(gene: str) -> str:
    conn = sqlite3.connect(_db())
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT regulator, relationship, pathway FROM upstream_regulators WHERE gene = ?",
        (gene,),
    ).fetchall()
    conn.close()
    return json.dumps([dict(r) for r in rows])


if __name__ == "__main__":
    mcp.run()
