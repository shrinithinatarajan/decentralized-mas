import json
import os
import sqlite3
from pathlib import Path
from fastmcp import FastMCP

mcp = FastMCP("Genomics Server")


def _db() -> Path:
    return Path(os.getenv("GENOMICS_DB", "src/data/processed/genomics.db"))


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db())
    conn.row_factory = sqlite3.Row
    return conn


@mcp.tool()
def get_mutations(cell_line: str, gene: str | None = None) -> str:
    """Fetch mutations for a cell line, optionally filtered by gene."""
    with _conn() as conn:
        cell_in_db = conn.execute(
            "SELECT COUNT(*) FROM mutations WHERE cell_line=?", (cell_line,)
        ).fetchone()[0] > 0
        if gene:
            rows = conn.execute(
                "SELECT * FROM mutations WHERE cell_line=? AND gene=?",
                (cell_line, gene),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM mutations WHERE cell_line=?", (cell_line,)
            ).fetchall()
    mutations = [dict(r) for r in rows]
    result: dict = {"mutations": mutations, "cell_line_in_db": cell_in_db}
    if not mutations and cell_in_db and gene:
        result["interpretation"] = (
            f"{gene} has no somatic mutations in {cell_line} — wild-type status. "
            "Absence of an oncogenic driver mutation may indicate resistance to targeted inhibitors."
        )
    return json.dumps(result)


@mcp.tool()
def get_cnv(cell_line: str, gene: str | None = None) -> str:
    """Fetch copy number variation data for a cell line."""
    with _conn() as conn:
        if gene:
            rows = conn.execute(
                "SELECT * FROM cnv WHERE cell_line=? AND gene=?",
                (cell_line, gene),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM cnv WHERE cell_line=?", (cell_line,)
            ).fetchall()
    return json.dumps([dict(r) for r in rows])


@mcp.tool()
def check_mutation_impact(gene: str, mutation: str) -> str:
    """Check if a mutation is a known cancer driver and which cell lines carry it."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM mutations WHERE gene=? AND mutation=? AND is_driver=1",
            (gene, mutation),
        ).fetchall()
    return json.dumps({
        "gene": gene,
        "mutation": mutation,
        "is_driver": len(rows) > 0,
        "cell_lines": list({r["cell_line"] for r in rows}),
    })
