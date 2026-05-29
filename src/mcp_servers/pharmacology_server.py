import json
import os
import sqlite3
from pathlib import Path
from fastmcp import FastMCP

mcp = FastMCP("Pharmacology Server")


def _db() -> Path:
    return Path(os.getenv("PHARMACOLOGY_DB", "src/data/processed/pharmacology.db"))


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db())
    conn.row_factory = sqlite3.Row
    return conn


@mcp.tool()
def get_ic50(cell_line: str, drug: str) -> str:
    """Get ln(IC50), AUC, z-score, and sensitivity label for a cell line + drug pair."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM drug_response WHERE cell_line=? AND drug=?",
            (cell_line, drug),
        ).fetchone()
    if row is None:
        return json.dumps({"error": "no data", "cell_line": cell_line, "drug": drug})
    return json.dumps(dict(row))


@mcp.tool()
def get_drug_info(drug: str) -> str:
    """Get target genes, mechanism of action, pathway, and drug class for a drug."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM drug_info WHERE drug=?", (drug,)
        ).fetchone()
    if row is None:
        return json.dumps({"error": "no drug info", "drug": drug})
    return json.dumps(dict(row))


@mcp.tool()
def get_sensitivity_profile(drug: str) -> str:
    """Get population-level sensitivity profile per mutation for a drug."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sensitivity_profile WHERE drug=?", (drug,)
        ).fetchall()
    return json.dumps([dict(r) for r in rows])


if __name__ == "__main__":
    mcp.run()
