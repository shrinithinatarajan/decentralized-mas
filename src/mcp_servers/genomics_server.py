import json
import os
import re
import sqlite3
from pathlib import Path
from fastmcp import FastMCP

mcp = FastMCP("Genomics Server")


def _db() -> Path:
    return Path(os.getenv("GENOMICS_DB", "src/data/processed/genomics.db"))


def _civic_db() -> Path:
    return Path(os.getenv("CIVIC_DB", "src/data/processed/civic.db"))


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db())
    conn.row_factory = sqlite3.Row
    return conn


def _norm_variant(v: str) -> str:
    return re.sub(r"^p\.", "", str(v).strip()).upper()


def _norm_drug(d: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(d).upper())


def _lookup_civic(gene: str, protein_change: str, drug: str) -> list[dict]:
    """Return CIViC predictive evidence for a (gene, variant, drug) triple."""
    civic_path = _civic_db()
    if not civic_path.exists():
        return []
    variant_norm = _norm_variant(protein_change)
    drug_norm    = _norm_drug(drug)
    try:
        conn = sqlite3.connect(civic_path)
        conn.row_factory = sqlite3.Row
        # Match gene + variant exactly, then filter by drug with fuzzy prefix
        rows = conn.execute(
            """
            SELECT * FROM predictive_evidence
            WHERE gene_norm = ?
              AND variant_norm = ?
            """,
            (gene.upper(), variant_norm),
        ).fetchall()
        conn.close()
        # Filter to rows whose drug_norm contains or is contained in the queried drug_norm
        matches = [
            dict(r) for r in rows
            if drug_norm in r["drug_norm"] or r["drug_norm"] in drug_norm
        ]
        return matches
    except Exception:
        return []


@mcp.tool()
def get_mutations(cell_line: str, gene: str | None = None, drug: str | None = None) -> str:
    """Fetch mutations for a cell line, optionally filtered by gene.

    If drug is provided, each mutation is enriched with CIViC predictive evidence
    specific to that drug, replacing or supplementing the static civic_description.
    """
    with _conn() as conn:
        cell_in_db = conn.execute(
            "SELECT COUNT(*) FROM mutations WHERE cell_line=?", (cell_line,)
        ).fetchone()[0] > 0
        if gene:
            rows = conn.execute(
                "SELECT * FROM mutations WHERE cell_line=? AND gene=? ORDER BY is_driver DESC",
                (cell_line, gene),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM mutations WHERE cell_line=? ORDER BY is_driver DESC",
                (cell_line,),
            ).fetchall()
    mutations = [dict(r) for r in rows]

    # Enrich with live CIViC lookup if drug is provided
    if drug:
        for mut in mutations:
            protein_change = mut.get("protein_change") or mut.get("mutation", "")
            civic_hits = _lookup_civic(mut.get("gene", ""), protein_change, drug)
            if civic_hits:
                # Prefer drug-matched CIViC over static column
                mut["civic_description"] = " | ".join(h["description"] for h in civic_hits)
                mut["civic_significance"] = civic_hits[0]["significance"]
                mut["civic_level"]        = civic_hits[0]["evidence_level"]

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
    """Check if a mutation is a known cancer driver and return CIViC evidence."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM mutations WHERE gene=? AND mutation=? AND is_driver=1",
            (gene, mutation),
        ).fetchall()
    civic_descriptions = list({r["civic_description"] for r in rows if r["civic_description"]})
    return json.dumps({
        "gene": gene,
        "mutation": mutation,
        "is_driver": len(rows) > 0,
        "cell_lines": list({r["cell_line"] for r in rows}),
        "civic_descriptions": civic_descriptions,
    })
