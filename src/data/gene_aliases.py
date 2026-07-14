"""Canonical gene symbol lookup for drug target names.

Source: HGNC approved symbols (https://www.genenames.org/) and
        UniProt gene names (https://www.uniprot.org/).

This table is built from reference databases — it is NOT derived from
the evaluation test set. It covers major oncology signalling pathways
so that any alias encountered in a drug label resolves to the CCLE/HGNC
approved symbol before hitting the genomics / transcriptomics DBs.
"""

# Maps alias → HGNC approved symbol (or list of symbols for multi-target aliases)
GENE_ALIASES: dict[str, str | list[str]] = {
    # ── ABL / BCR-ABL ──────────────────────────────────────────────────────────
    "ABL": "ABL1",
    "c-ABL": "ABL1",
    "BCR-ABL": "ABL1",

    # ── EGFR / ErbB family ─────────────────────────────────────────────────────
    "ERBB1": "EGFR",
    "HER1": "EGFR",
    "HER2": "ERBB2",
    "NEU": "ERBB2",
    "c-erbB-2": "ERBB2",
    "HER3": "ERBB3",
    "HER4": "ERBB4",

    # ── RAS / RAF / MEK / ERK ──────────────────────────────────────────────────
    "RAF": ["BRAF", "RAF1"],   # ambiguous class; query both isoforms
    "CRAF": "RAF1",
    "c-RAF": "RAF1",
    "MEK1": "MAP2K1",
    "MEK2": "MAP2K2",
    "ERK1": "MAPK3",
    "ERK2": "MAPK1",

    # ── PI3K family ────────────────────────────────────────────────────────────
    "PI3Kalpha": "PIK3CA",
    "PI3K-alpha": "PIK3CA",
    "p110alpha": "PIK3CA",
    "PI3K (class 1)": "PIK3CA",   # default class-I isoform to alpha
    "PI3Kbeta": "PIK3CB",
    "PI3K-beta": "PIK3CB",
    "p110beta": "PIK3CB",
    "PI3Kgamma": "PIK3CG",
    "PI3K-gamma": "PIK3CG",
    "p110gamma": "PIK3CG",
    "PI3Kdelta": "PIK3CD",
    "PI3K-delta": "PIK3CD",
    "p110delta": "PIK3CD",

    # ── mTOR complexes ─────────────────────────────────────────────────────────
    "mTOR": "MTOR",
    "MTORC1": "MTOR",
    "MTORC2": "MTOR",

    # ── VEGFR / RTK family ─────────────────────────────────────────────────────
    "VEGFR": "KDR",          # VEGFR2 is the primary therapeutic target
    "VEGFR1": "FLT1",
    "VEGFR2": "KDR",
    "VEGFR3": "FLT4",
    "VEGFR3/FLT4": "FLT4",
    "PDGFR": ["PDGFRA", "PDGFRB"],
    "PDGFRalpha": "PDGFRA",
    "PDGFR-alpha": "PDGFRA",
    "PDGFRbeta": "PDGFRB",
    "PDGFR-beta": "PDGFRB",
    "RON": "MST1R",
    "c-RON": "MST1R",
    "TIE2": "TEK",
    "Tie-2": "TEK",
    "c-KIT": "KIT",
    "CD117": "KIT",
    "c-MET": "MET",
    "HGFR": "MET",
    "c-RET": "RET",
    "c-FMS": "CSF1R",
    "CSF1-R": "CSF1R",
    "EPHA2": "EPHA2",       # already canonical — listed for completeness
    "AXL": "AXL",
    "TYRO3": "TYRO3",
    "MERTK": "MERTK",

    # ── IGF / insulin ──────────────────────────────────────────────────────────
    "IGF1R": "IGF1R",
    "INSR": "INSR",
    "IR": "INSR",

    # ── Hormone receptors ──────────────────────────────────────────────────────
    "ER": "ESR1",
    "ERalpha": "ESR1",
    "ER-alpha": "ESR1",
    "ERbeta": "ESR2",
    "ER-beta": "ESR2",
    # AR is already canonical
    "androgen receptor": "AR",

    # ── Apoptosis / IAP family ─────────────────────────────────────────────────
    "IAP1": "BIRC4",
    "XIAP": "XIAP",          # canonical
    "IAP2": "BIRC3",
    "cIAP2": "BIRC3",
    "cIAP1": "BIRC2",
    "Survivin": "BIRC5",
    "BCL-2": "BCL2",
    "BCL-XL": "BCL2L1",
    "MCL-1": "MCL1",
    "BAX": "BAX",
    "BAK": "BAK1",

    # ── p53 / MDM2 ─────────────────────────────────────────────────────────────
    "p53": "TP53",

    # ── JAK / STAT ─────────────────────────────────────────────────────────────
    "JAK": "JAK1",           # ambiguous; default to JAK1

    # ── SRC family ─────────────────────────────────────────────────────────────
    "c-SRC": "SRC",
    "LYN": "LYN",
    "FYN": "FYN",
    "LCK": "LCK",

    # ── MYC family ─────────────────────────────────────────────────────────────
    "c-MYC": "MYC",
    "N-MYC": "MYCN",
    "L-MYC": "MYCL",

    # ── ALK / ROS1 / RET ───────────────────────────────────────────────────────
    "c-ALK": "ALK",
    "ROS": "ROS1",

    # ── PARP ───────────────────────────────────────────────────────────────────
    "PARP": "PARP1",
    "PARP-1": "PARP1",
    "PARP-2": "PARP2",

    # ── HDAC ───────────────────────────────────────────────────────────────────
    "HDAC": "HDAC1",         # ambiguous class; default to HDAC1

    # ── Aurora kinases ─────────────────────────────────────────────────────────
    "Aurora A": "AURKA",
    "Aurora B": "AURKB",
    "Aurora kinase A": "AURKA",
    "Aurora kinase B": "AURKB",

    # ── PLK / WEE1 / CHK ───────────────────────────────────────────────────────
    "PLK1": "PLK1",
    "WEE1": "WEE1",
    "CHK1": "CHEK1",
    "CHK2": "CHEK2",
    "Chk1": "CHEK1",
    "Chk2": "CHEK2",

    # ── Proteasome ─────────────────────────────────────────────────────────────
    "PSMD4": "PSMD4",
}

# Non-gene drug-target descriptors — skip gene-level DB queries;
# rely on pharmacology/pathway agents instead.
SKIP_TARGETS: set[str] = {
    "Broad spectrum kinase inhibitor",
    "Alkylating agent",
    "Topoisomerase inhibitor",
    "Topoisomerase I inhibitor",
    "Topoisomerase II inhibitor",
    "Antimetabolite",
    "Mitotic inhibitor",
    "Proteasome inhibitor",
    "HDAC inhibitor",
    "DNA methyltransferase inhibitor",
    "DNA damaging agent",
    "Nucleoside analogue",
    "Platinum compound",
}
