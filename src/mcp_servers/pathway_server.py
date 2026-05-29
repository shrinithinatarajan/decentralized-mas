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
        "SELECT bypass_gene, bypass_exists FROM bypass_routes "
        "WHERE pathway_id = ? AND blocked_gene = ?",
        (pathway_id, blocked_gene),
    ).fetchall()
    conn.close()
    bypass_genes = [r["bypass_gene"] for r in rows if r["bypass_exists"]]
    return json.dumps({
        "bypass_exists": len(bypass_genes) > 0,
        "bypass_genes": bypass_genes,
    })


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
