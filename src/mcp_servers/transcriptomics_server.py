import json
import os
import sqlite3
from pathlib import Path
from fastmcp import FastMCP
from src.schemas.axiom_rules import SILENCING_THRESHOLD

mcp = FastMCP("Transcriptomics Server")


def _db() -> Path:
    return Path(os.getenv("TRANSCRIPTOMICS_DB", "src/data/processed/transcriptomics.db"))


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db())
    conn.row_factory = sqlite3.Row
    return conn


@mcp.tool()
def get_expression(cell_line: str, gene: str | None = None) -> str:
    """Get expression levels (TPM, z-score, percentile) for a cell line."""
    with _conn() as conn:
        if gene:
            rows = conn.execute(
                "SELECT * FROM expression WHERE cell_line=? AND gene=?",
                (cell_line, gene),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM expression WHERE cell_line=?", (cell_line,)
            ).fetchall()
    return json.dumps([dict(r) for r in rows])


@mcp.tool()
def check_silencing(cell_line: str, gene: str) -> str:
    """Check if a gene is transcriptionally silenced (z-score < SILENCING_THRESHOLD)."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT z_score FROM expression WHERE cell_line=? AND gene=?",
            (cell_line, gene),
        ).fetchone()
    if row is None:
        return json.dumps({
            "is_silenced": None,
            "z_score": None,
            "threshold_used": SILENCING_THRESHOLD,
            "reason": "no expression data",
        })
    z = float(row["z_score"])
    return json.dumps({
        "is_silenced": z < SILENCING_THRESHOLD,
        "z_score": z,
        "threshold_used": SILENCING_THRESHOLD,
    })


@mcp.tool()
def get_high_expression_genes(cell_line: str, z_threshold: float = 1.0) -> str:
    """Return genes with z-score above threshold — candidate pathway drivers."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT gene, tpm, z_score FROM expression WHERE cell_line=? AND z_score > ?",
            (cell_line, z_threshold),
        ).fetchall()
    return json.dumps([dict(r) for r in rows])


if __name__ == "__main__":
    mcp.run()
