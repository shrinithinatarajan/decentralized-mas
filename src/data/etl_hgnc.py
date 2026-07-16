"""Download HGNC complete set and build a gene alias mapping.

Downloads the HGNC complete dataset (TSV) from genenames.org, parses
alias_symbol and prev_symbol columns, and writes a JSON mapping of
alias -> approved HGNC symbol to src/data/raw/hgnc/gene_aliases.json.

Also derives SKIP_TARGETS: GDSC2 PUTATIVE_TARGET values that are
pharmacological class descriptors rather than gene symbols.

Usage:
    PYTHONPATH=. python src/data/etl_hgnc.py

Outputs:
    src/data/raw/hgnc/hgnc_complete_set.txt  (raw download, ~9 MB)
    src/data/raw/hgnc/gene_aliases.json       (alias -> symbol mapping)
    src/data/raw/hgnc/skip_targets.json       (non-gene drug target descriptors)
"""
import json
import urllib.request
from pathlib import Path

import pandas as pd

HGNC_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
OUT_DIR  = Path("src/data/raw/hgnc")
GDSC2    = Path("src/data/raw/gdsc2/ic50.csv")


def download_hgnc() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "hgnc_complete_set.txt"
    if dest.exists():
        print(f"  {dest} already exists, skipping download")
        return dest
    print(f"  Downloading HGNC complete set from {HGNC_URL} ...")
    urllib.request.urlretrieve(HGNC_URL, dest)
    print(f"  Saved to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def build_alias_map(hgnc_path: Path) -> dict[str, str]:
    """Return {alias -> approved_symbol} from HGNC alias_symbol and prev_symbol columns."""
    df = pd.read_csv(hgnc_path, sep="\t", low_memory=False,
                     usecols=["symbol", "alias_symbol", "prev_symbol"])

    approved = set(df["symbol"].dropna().tolist())
    alias_map: dict[str, str] = {}

    for _, row in df.iterrows():
        symbol = row["symbol"]
        if not isinstance(symbol, str):
            continue
        for col in ("alias_symbol", "prev_symbol"):
            raw = row.get(col)
            if not isinstance(raw, str) or not raw.strip():
                continue
            for alias in raw.split("|"):
                alias = alias.strip()
                if alias and alias != symbol and alias not in approved:
                    # Only add if the alias is NOT itself an approved symbol
                    # (avoids overwriting a valid symbol with a different mapping)
                    alias_map[alias] = symbol

    print(f"  Built alias map: {len(alias_map):,} aliases -> {len(approved):,} approved symbols")
    return alias_map


def derive_skip_targets(alias_map: dict[str, str]) -> list[str]:
    """Return GDSC2 PUTATIVE_TARGET values that are not gene symbols.

    These are pharmacological class descriptors (e.g. 'Broad spectrum kinase
    inhibitor') rather than HGNC gene identifiers.
    """
    df = pd.read_csv(GDSC2, usecols=["PUTATIVE_TARGET"])
    targets = df["PUTATIVE_TARGET"].dropna().unique().tolist()

    # Load approved HGNC symbols
    hgnc_path = OUT_DIR / "hgnc_complete_set.txt"
    approved_df = pd.read_csv(hgnc_path, sep="\t", low_memory=False, usecols=["symbol"])
    approved = set(approved_df["symbol"].dropna().tolist())
    approved_lower = {s.lower() for s in approved}
    alias_lower = {k.lower() for k in alias_map}

    skip = []
    for t in targets:
        # A target may be a comma-separated list of genes — check each part
        parts = [p.strip() for p in str(t).split(",")]
        any_gene = any(
            p in approved or p in alias_map or p.lower() in approved_lower or p.lower() in alias_lower
            for p in parts
        )
        if not any_gene:
            skip.append(t)

    skip.sort()
    print(f"  Derived {len(skip)} SKIP_TARGETS from {len(targets)} unique PUTATIVE_TARGET values")
    return skip


if __name__ == "__main__":
    print("=== HGNC gene alias ETL ===")
    hgnc_path = download_hgnc()

    print("Building alias map...")
    alias_map = build_alias_map(hgnc_path)
    alias_out = OUT_DIR / "gene_aliases.json"
    alias_out.write_text(json.dumps(alias_map, indent=2, sort_keys=True))
    print(f"  Saved to {alias_out}")

    print("Deriving SKIP_TARGETS...")
    skip = derive_skip_targets(alias_map)
    skip_out = OUT_DIR / "skip_targets.json"
    skip_out.write_text(json.dumps(skip, indent=2))
    print(f"  Saved to {skip_out}")
    print("\nSample aliases:")
    for k, v in list(alias_map.items())[:10]:
        print(f"  {k!r} -> {v!r}")
    print("\nSample SKIP_TARGETS:")
    for t in skip[:10]:
        print(f"  {t!r}")
    print("\nDone.")
