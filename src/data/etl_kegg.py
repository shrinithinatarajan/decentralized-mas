"""Download full KEGG human pathway database and rebuild pathways.db.

Downloads all human (hsa) signalling/cancer pathways via the KEGG REST API,
parses KGML topology to extract gene membership and bypass routes, and
populates pathways.db with pathway_genes, bypass_routes, upstream_regulators.

Usage:
    PYTHONPATH=. python src/data/etl_kegg.py

Outputs:
    src/data/raw/kegg/              (cached KGML files)
    src/data/processed/pathways.db  (rebuilt)

Note: KEGG REST API is rate-limited. The script adds delays between requests.
Estimated runtime: 20-40 min for full download.
"""
import json
import sqlite3
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

RAW_DIR = Path("src/data/raw/kegg")
DB_PATH = Path("src/data/processed/pathways.db")
KEGG_BASE = "https://rest.kegg.jp"

# Cancer/signalling pathway categories to include
INCLUDE_CATEGORIES = {
    "Signal transduction",
    "Signaling molecules and interaction",
    "Cell growth and death",
    "Cellular community - eukaryotes",
    "Cell motility",
    "Cancer: overview",
    "Cancer: specific types",
    "Immune system",
    "Endocrine system",
    "Environmental adaptation",
}


def _get(url: str, delay: float = 0.4) -> str:
    time.sleep(delay)
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode("utf-8")


def _fetch_pathway_list() -> list[dict]:
    """Return list of all human pathways with id and name."""
    dest = RAW_DIR / "pathway_list.txt"
    if dest.exists():
        text = dest.read_text()
    else:
        print("  Fetching pathway list...")
        text = _get(f"{KEGG_BASE}/list/pathway/hsa")
        dest.write_text(text)
    pathways = []
    for line in text.strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            pid = parts[0].strip()  # e.g. path:hsa04010
            name = parts[1].strip()
            pathways.append({"id": pid.replace("path:", ""), "name": name})
    return pathways


def _fetch_pathway_categories() -> dict[str, str]:
    """Return {pathway_id: category} from KEGG BRITE hierarchy."""
    dest = RAW_DIR / "pathway_brite.json"
    if dest.exists():
        return json.loads(dest.read_text())
    print("  Fetching BRITE pathway hierarchy...")
    text = _get(f"{KEGG_BASE}/get/br:hsa00001/json")
    categories: dict[str, str] = {}
    try:
        data = json.loads(text)
        def _walk(node, cat=""):
            name = node.get("name", "")
            children = node.get("children", [])
            for child in children:
                child_name = child.get("name", "")
                if child_name.startswith("hsa"):
                    pid = child_name.split(" ")[0]
                    categories[pid] = cat or name
                else:
                    _walk(child, cat or name)
        _walk(data)
    except Exception as e:
        print(f"  Warning: BRITE parse failed ({e}), using all pathways")
    dest.write_text(json.dumps(categories))
    return categories


def _fetch_kgml(pathway_id: str) -> str | None:
    """Download and cache KGML for a pathway."""
    dest = RAW_DIR / "kgml" / f"{pathway_id}.xml"
    if dest.exists():
        return dest.read_text()
    try:
        text = _get(f"{KEGG_BASE}/get/{pathway_id}/kgml")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        return text
    except Exception as e:
        print(f"    Warning: failed to fetch {pathway_id}: {e}")
        return None


def _build_entrez_to_symbol() -> dict[str, str]:
    """Map KEGG Entrez gene IDs to HGNC approved symbols using HGNC data."""
    hgnc_path = Path("src/data/raw/hgnc/hgnc_complete_set.txt")
    if not hgnc_path.exists():
        print("  Warning: HGNC data not found, gene symbols may be Entrez IDs")
        return {}
    import pandas as pd
    df = pd.read_csv(hgnc_path, sep="\t", low_memory=False,
                     usecols=["symbol", "entrez_id"])
    df = df.dropna(subset=["entrez_id"])
    df["entrez_id"] = df["entrez_id"].astype(int).astype(str)
    return dict(zip(df["entrez_id"], df["symbol"]))


def _parse_kgml(pathway_id: str, kgml: str, entrez_map: dict) -> dict:
    """Parse KGML XML and return genes, relations, and derived bypass routes."""
    try:
        root = ET.fromstring(kgml)
    except ET.ParseError:
        return {"genes": [], "relations": [], "bypass_routes": []}

    # Build entry_id -> list of gene symbols
    entry_genes: dict[str, list[str]] = {}
    for entry in root.findall("entry"):
        if entry.get("type") != "gene":
            continue
        eid = entry.get("id", "")
        names = entry.get("name", "").split()
        symbols = []
        for name in names:
            entrez = name.replace("hsa:", "")
            sym = entrez_map.get(entrez, entrez)
            symbols.append(sym)
        entry_genes[eid] = symbols

    # Gene membership
    all_genes = sorted({g for syms in entry_genes.values() for g in syms})

    # Build activation/inhibition graph: gene -> set of genes it activates
    activates: dict[str, set[str]] = {g: set() for g in all_genes}
    inhibits:  dict[str, set[str]] = {g: set() for g in all_genes}

    relations = []
    for rel in root.findall("relation"):
        if rel.get("type") not in ("PPrel", "GErel"):
            continue
        entry1 = rel.get("entry1", "")
        entry2 = rel.get("entry2", "")
        subtypes = {s.get("name") for s in rel.findall("subtype")}
        for g1 in entry_genes.get(entry1, []):
            for g2 in entry_genes.get(entry2, []):
                rel_type = "activation" if "activation" in subtypes else \
                           "inhibition" if "inhibition" in subtypes else "association"
                relations.append((g1, g2, rel_type))
                if rel_type == "activation":
                    activates[g1].add(g2)
                elif rel_type == "inhibition":
                    inhibits[g1].add(g2)

    # Derive bypass routes:
    # For each gene A (potential drug target), find genes B such that
    # B activates the same downstream targets as A but is NOT downstream of A.
    # B can substitute for A → B is a bypass gene when A is blocked.
    bypass_routes = []
    for blocked in all_genes:
        downstream_of_blocked = activates.get(blocked, set())
        if not downstream_of_blocked:
            continue
        for bypass_cand in all_genes:
            if bypass_cand == blocked:
                continue
            # bypass candidate must not be downstream of blocked gene
            if bypass_cand in downstream_of_blocked:
                continue
            # bypass candidate must activate at least one of blocked's downstream targets
            cand_activates = activates.get(bypass_cand, set())
            if cand_activates & downstream_of_blocked:
                bypass_routes.append((blocked, bypass_cand))

    return {"genes": all_genes, "relations": relations, "bypass_routes": bypass_routes}


def rebuild_pathways_db(pathways: list[dict], entrez_map: dict) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS pathway_genes")
    conn.execute("DROP TABLE IF EXISTS bypass_routes")
    conn.execute("DROP TABLE IF EXISTS upstream_regulators")
    conn.execute("DROP TABLE IF EXISTS pathway_meta")
    conn.execute("""
        CREATE TABLE pathway_meta (
            pathway_id TEXT PRIMARY KEY, name TEXT, category TEXT)
    """)
    conn.execute("""
        CREATE TABLE pathway_genes (
            pathway_id TEXT NOT NULL, gene TEXT NOT NULL,
            role TEXT, position TEXT)
    """)
    conn.execute("""
        CREATE TABLE bypass_routes (
            pathway_id TEXT NOT NULL, blocked_gene TEXT NOT NULL,
            bypass_gene TEXT NOT NULL, bypass_exists INTEGER DEFAULT 1)
    """)
    conn.execute("""
        CREATE TABLE upstream_regulators (
            gene TEXT NOT NULL, regulator TEXT NOT NULL,
            relationship TEXT, pathway TEXT)
    """)
    conn.commit()

    total_genes = total_bypass = total_regulators = 0
    for i, pw in enumerate(pathways, 1):
        pid  = pw["id"]
        name = pw["name"]
        cat  = pw.get("category", "")
        print(f"  [{i}/{len(pathways)}] {pid}: {name[:50]}", flush=True)

        kgml = _fetch_kgml(pid)
        if not kgml:
            continue

        parsed = _parse_kgml(pid, kgml, entrez_map)

        conn.execute("INSERT OR REPLACE INTO pathway_meta VALUES (?,?,?)",
                     (pid, name, cat))

        gene_rows = [(pid, g, None, None) for g in parsed["genes"]]
        conn.executemany("INSERT INTO pathway_genes VALUES (?,?,?,?)", gene_rows)
        total_genes += len(gene_rows)

        bypass_rows = [(pid, blocked, bypass, 1)
                       for blocked, bypass in parsed["bypass_routes"]]
        conn.executemany("INSERT INTO bypass_routes VALUES (?,?,?,?)", bypass_rows)
        total_bypass += len(bypass_rows)

        # Upstream regulators: for each gene, record what activates it
        for g1, g2, rel_type in parsed["relations"]:
            if rel_type in ("activation", "inhibition"):
                conn.execute("INSERT INTO upstream_regulators VALUES (?,?,?,?)",
                             (g2, g1, rel_type, pid))
                total_regulators += 1

        conn.commit()

    conn.execute("CREATE INDEX IF NOT EXISTS idx_pg_pathway ON pathway_genes(pathway_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pg_gene ON pathway_genes(gene)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_br_blocked ON bypass_routes(pathway_id, blocked_gene)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ur_gene ON upstream_regulators(gene)")
    conn.commit()
    conn.close()
    print(f"\n  pathways.db rebuilt:")
    print(f"    {len(pathways)} pathways, {total_genes:,} gene memberships")
    print(f"    {total_bypass:,} bypass routes")
    print(f"    {total_regulators:,} upstream regulator edges")


if __name__ == "__main__":
    import time as _time
    t0 = _time.time()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching KEGG pathway list...")
    all_pathways = _fetch_pathway_list()
    print(f"  {len(all_pathways)} human pathways found")

    print("Fetching BRITE category map...")
    categories = _fetch_pathway_categories()

    # Tag each pathway with its category and filter to relevant ones
    tagged = []
    for pw in all_pathways:
        cat = categories.get(pw["id"], "")
        pw["category"] = cat
        if not categories or cat in INCLUDE_CATEGORIES or not cat:
            tagged.append(pw)
    print(f"  {len(tagged)} pathways after category filter")

    print("Building Entrez -> HGNC symbol map...")
    entrez_map = _build_entrez_to_symbol()
    print(f"  {len(entrez_map):,} Entrez IDs mapped")

    print(f"\nDownloading and parsing KGML for {len(tagged)} pathways...")
    rebuild_pathways_db(tagged, entrez_map)

    print(f"\nDone in {(_time.time()-t0)/60:.1f} min")
