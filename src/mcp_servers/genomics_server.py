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


def _depmap_db() -> Path:
    return Path(os.getenv("DEPMAP_DB", "src/data/processed/depmap.db"))


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
        # Filter to rows whose drug_norm matches AND direction=Supports (Does Not Support inverts meaning)
        matches = [
            dict(r) for r in rows
            if (drug_norm in r["drug_norm"] or r["drug_norm"] in drug_norm)
            and r["direction"] == "Supports"
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
            # `mutation` is the only variant column production genomics.db has;
            # there is no separate `protein_change` column.
            civic_hits = _lookup_civic(mut.get("gene", ""), mut.get("mutation", ""), drug)
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
def get_depmap_dependency(cell_line: str, genes: list[str]) -> str:
    """Return DepMap CRISPR Chronos gene-effect scores for a cell line.

    Chronos ≤ -0.5 → gene is essential in this cell line → SENSITIVE signal.
    Chronos > -0.5  → gene is non-essential → not informative for sensitivity.
    Returns empty list if depmap.db is not present (run scripts/prepare_depmap.py first).
    """
    db_path = _depmap_db()
    if not db_path.exists():
        return json.dumps({"error": "depmap.db not found — run scripts/prepare_depmap.py", "scores": []})

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    results = []
    # Normalize: strip spaces, hyphens, underscores and uppercase for fuzzy match
    import re as _re
    cl_norm = _re.sub(r"[\s\-_]", "", cell_line).upper()
    for gene in genes:
        row = conn.execute(
            "SELECT depmap_id, cell_line, gene, chronos FROM chronos_scores "
            "WHERE cell_line_norm=? AND gene=?",
            (cl_norm, gene),
        ).fetchone()
        if row:
            results.append(dict(row))
    conn.close()
    return json.dumps({"scores": results, "threshold_note": "Chronos <= -0.5 = gene essential = SENSITIVE signal"})


_ENSEMBL_REST  = "https://rest.ensembl.org"
_OT_GRAPHQL    = "https://api.platform.opentargets.org/api/v4/graphql"
_JSON_HEADERS  = {"Content-Type": "application/json", "Accept": "application/json"}


def _gene_to_ensembl(gene_symbol: str) -> str | None:
    """Convert gene symbol to ENSG ID via Ensembl REST API."""
    import requests as _req
    try:
        url = f"{_ENSEMBL_REST}/xrefs/symbol/homo_sapiens/{gene_symbol}"
        r = _req.get(url, headers=_JSON_HEADERS, timeout=10)
        r.raise_for_status()
        ids = [row["id"] for row in r.json() if row.get("id", "").startswith("ENSG")]
        return ids[0] if ids else None
    except Exception:
        return None


@mcp.tool()
def get_opentargets_evidence(genes: list[str], drug: str) -> str:
    """Query OpenTargets for known drug-gene associations.

    Used as T1 fallback when CIViC has no coverage for a drug.
    Returns drug mechanism, clinical phase, and any sensitivity/resistance hints.
    """
    import requests as _req
    import re

    drug_norm = re.sub(r"[^a-z0-9]", "", drug.lower())
    results = []

    for gene in genes:
        ensembl_id = _gene_to_ensembl(gene)
        if not ensembl_id:
            results.append({"gene": gene, "error": "ensembl_id_not_found"})
            continue

        query = """
        {
          target(ensemblId: "%s") {
            knownDrugs(size: 100) {
              rows {
                drug { name prefName }
                mechanismOfAction
                phase
                drugType
                status
              }
            }
          }
        }
        """ % ensembl_id

        try:
            r = _req.post(_OT_GRAPHQL, json={"query": query}, headers=_JSON_HEADERS, timeout=15)
            r.raise_for_status()
            data = r.json()
            rows = (data.get("data") or {}).get("target", {}).get("knownDrugs", {}).get("rows", [])
        except Exception as e:
            results.append({"gene": gene, "ensembl_id": ensembl_id, "error": str(e)})
            continue

        # Find rows matching our drug (fuzzy name match)
        matched = [
            row for row in rows
            if drug_norm in re.sub(r"[^a-z0-9]", "", (row.get("drug") or {}).get("name", "").lower())
            or drug_norm in re.sub(r"[^a-z0-9]", "", (row.get("drug") or {}).get("prefName", "").lower())
        ]

        results.append({
            "gene": gene,
            "ensembl_id": ensembl_id,
            "drug_matches": matched,
            "total_known_drugs_for_gene": len(rows),
            "note": (
                "Drug found in OpenTargets with mechanism evidence" if matched
                else "Drug not found in OpenTargets for this gene — no additional evidence"
            ),
        })

    return json.dumps(results)


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


def _rppa_db() -> Path:
    return Path(os.getenv("DEPMAP_DB", "src/data/processed/depmap.db"))


@mcp.tool()
def get_rppa_expression(cell_line: str, genes: list[str]) -> str:
    """Return CCLE RPPA protein expression scores for target genes in a cell line.

    Scores are log2 protein expression (z-scored per antibody across ~900 cell lines).
    Positive = above-average expression, negative = below-average.
    Returns antibody-level entries; a gene may map to multiple antibodies (total + phospho).
    Missing = cell line not in RPPA panel (~900 cell lines covered).
    """
    db_path = _rppa_db()
    if not db_path.exists():
        return json.dumps({"error": "depmap.db not found", "results": []})

    import re as _re
    cl_norm = _re.sub(r"[\s\-_/]", "", cell_line).upper()
    gene_norms = [_re.sub(r"[\s\-_/]", "", g).upper() for g in genes]

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    results = []
    for gene_norm in gene_norms:
        antibodies = [
            r["antibody"]
            for r in conn.execute(
                "SELECT antibody FROM rppa_antibody_genes WHERE gene_norm=?", (gene_norm,)
            ).fetchall()
        ]
        if not antibodies:
            results.append({"gene": gene_norm, "antibodies": [], "note": "Gene not in RPPA panel"})
            continue
        entries = []
        for ab in antibodies:
            row = conn.execute(
                "SELECT rppa_score FROM rppa_expression WHERE cell_line_norm=? AND antibody=?",
                (cl_norm, ab),
            ).fetchone()
            if row:
                entries.append({"antibody": ab, "rppa_score": round(row["rppa_score"], 4)})
        if entries:
            total_score = next((e["rppa_score"] for e in entries if "_p" not in e["antibody"] and "_Caution" not in e["antibody"]), None)
            results.append({
                "gene": gene_norm,
                "entries": entries,
                "total_protein_score": total_score,
                "note": (
                    "rppa_score > 0.5: high protein expression; "
                    "rppa_score < -0.5: low expression; near 0: average"
                ),
            })
        else:
            results.append({"gene": gene_norm, "antibodies": antibodies, "note": f"Cell line not in RPPA panel (checked norm: {cl_norm})"})
    conn.close()
    return json.dumps({"cell_line": cell_line, "rppa": results})


@mcp.tool()
def get_oncokb_annotation(gene: str, alteration: str, drug: str, tumor_type: str = "") -> str:
    """Query OncoKB for variant oncogenicity and drug sensitivity levels.

    Requires ONCOKB_TOKEN environment variable (free academic registration at oncokb.org).
    Returns:
      - oncogenicity: Oncogenic / Likely Oncogenic / VUS / Likely Neutral / Neutral / Unknown
      - mutationEffect: Gain-of-function / Loss-of-function / likely GOF/LOF / Switch-of-function / Neutral / Unknown
      - highestSensitiveLevel: e.g. LEVEL_1, LEVEL_2, LEVEL_3A (FDA/expert-supported drug sensitivity)
      - highestResistanceLevel: e.g. LEVEL_R1, LEVEL_R2 (confirmed resistance)
    Gracefully returns empty result when token is absent or API is unavailable.
    """
    import re as _re
    token = os.getenv("ONCOKB_TOKEN", "")
    if not token:
        return json.dumps({
            "gene": gene, "alteration": alteration,
            "error": "ONCOKB_TOKEN not set — register free at oncokb.org/api-access",
            "oncogenicity": "Unknown", "mutationEffect": "Unknown",
        })

    import urllib.request, urllib.error

    # alteration should be without "p." prefix (e.g., V600E not p.V600E)
    alt_clean = _re.sub(r"^p\.", "", alteration.strip())
    params = f"hugoSymbol={urllib.parse.quote(gene)}&alteration={urllib.parse.quote(alt_clean)}"
    if tumor_type:
        params += f"&tumorType={urllib.parse.quote(tumor_type)}"

    import urllib.parse
    url = f"https://www.oncokb.org/api/v1/annotate/mutations/byProteinChange?{params}"
    try:
        req_obj = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req_obj, timeout=10) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return json.dumps({"gene": gene, "alteration": alt_clean, "error": f"OncoKB HTTP {e.code}: {e.reason}"})
    except Exception as e:
        return json.dumps({"gene": gene, "alteration": alt_clean, "error": str(e)})

    # Find drug-specific treatments
    treatments = data.get("treatments", [])
    drug_norm = _re.sub(r"[^a-z0-9]", "", drug.lower())
    matched_treatments = []
    for t in treatments:
        for d in t.get("drugs", []):
            d_norm = _re.sub(r"[^a-z0-9]", "", (d.get("drugName") or "").lower())
            if drug_norm in d_norm or d_norm in drug_norm:
                matched_treatments.append({
                    "drug": d.get("drugName"),
                    "level": t.get("level"),
                    "levelAssociatedCancerType": t.get("levelAssociatedCancerType", {}).get("name", ""),
                })

    return json.dumps({
        "gene": gene,
        "alteration": alt_clean,
        "oncogenicity": data.get("oncogenic", "Unknown"),
        "mutationEffect": (data.get("mutationEffect") or {}).get("knownEffect", "Unknown"),
        "highestSensitiveLevel": data.get("highestSensitiveLevel"),
        "highestResistanceLevel": data.get("highestResistanceLevel"),
        "drug_matched_treatments": matched_treatments,
        "allele_exist": data.get("alleleExist", False),
    })
