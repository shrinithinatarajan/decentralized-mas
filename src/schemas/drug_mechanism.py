"""Drug mechanism classification for resolver priority adjustment.

Targeted drugs have a direct genomic driver (mutation, amplification, fusion)
where T1 structural evidence is mechanistically the right tiebreak signal.

Non-targeted drugs (cytotoxics, broad epigenetics, metabolic, HSP90, proteasome,
checkpoint-agnostic) have no single genomic mutation driver — T4 pharmacological
IC50 is more informative and should not be overridden by T1 in tiebreaks.
"""

# Drugs where T1 genomic mutation/CNV IS the correct tiebreak signal.
# Criteria: drug acts through a single-gene target for which CCLE has
# mutation, CNV, or expression data that mechanistically predicts response.
TARGETED_DRUGS: frozenset[str] = frozenset({
    # ── EGFR / ERBB2 ────────────────────────────────────────────────────────
    "Afatinib", "AZD3759", "Erlotinib", "Gefitinib", "Lapatinib",
    "Osimertinib", "Sapitinib",

    # ── BRAF / MEK / ERK ────────────────────────────────────────────────────
    "Dabrafenib", "PLX-4720", "SB590885", "Refametinib", "Selumetinib",
    "SCH772984", "Trametinib", "Ulixertinib", "VX-11e",

    # ── ABL / SRC ───────────────────────────────────────────────────────────
    "Nilotinib", "Bosutinib", "Dasatinib",

    # ── IGF1R ───────────────────────────────────────────────────────────────
    "BMS-536924", "BMS-754807", "GSK1904529A", "Linsitinib", "NVP-ADW742",

    # ── RTK (single-gene amplification/fusion driver) ───────────────────────
    "AZD4547", "AZD1332", "GW441756",          # FGFR1-3 / NTRK1-3
    "Crizotinib", "Savolitinib", "Foretinib",  # MET / ALK / ROS1
    "AZD1208",                                  # PIM kinase
    "Lestaurtinib",                             # FLT3 / NTRK

    # ── p53 pathway (TP53 mutation status is the primary biomarker) ─────────
    "MIRA-1", "PRIMA-1MET", "Nutlin-3a (-)", "Serdemetan",

    # ── Hormone receptor (ESR1 / AR expression-driven) ──────────────────────
    "Bicalutamide", "Fulvestrant", "GDC0810", "Tamoxifen",

    # ── PARP inhibitors (BRCA1/2 mutation = primary biomarker) ──────────────
    "Olaparib", "Niraparib", "Rucaparib", "Talazoparib", "Veliparib",

    # ── BTK ─────────────────────────────────────────────────────────────────
    "Ibrutinib",

    # ── BCL2 family (BCL2 expression is key; T2 > T1 but genomics still informative) ──
    "Navitoclax", "Venetoclax", "ABT737", "WEHI-539",
    "MIM1", "UMI-77", "AZD5991", "TW 37", "Sabutoclax", "Obatoclax Mesylate",
    "LCL161", "AZD5582",

    # ── PI3K/AKT/mTOR (PIK3CA / PTEN mutation drives sensitivity) ───────────
    # Note: T1 is informative but weaker than in oncogene-addicted settings.
    # Kept as targeted because PIK3CA/PTEN status IS used clinically.
    "Alpelisib", "AMG-319", "AZD6482", "AZD8186", "Buparlisib",
    "CZC24832", "Dactolisib", "GNE-317", "Ipatasertib", "LJI308",
    "MK-2206", "OSI-027", "Pictilisib", "Taselisib", "Uprosertib",
    "AZD2014", "AZD8055", "AZD5363", "AT13148", "GSK2110183B",
    "Rapamycin", "Temsirolimus", "PF-4708671", "GSK2578215A",

    # ── WNT pathway (APC / CTNNB1 mutation drives sensitivity) ─────────────
    "AZ6102", "CHIR-99021", "IWP-2", "LGK974", "MN-64",
    "SB216763", "WIKI4", "Wnt-C59", "XAV939",

    # ── EGFR downstream / RTK broad (somatic mutation still relevant) ───────
    "Axitinib", "Cediranib", "Motesanib", "Sorafenib",
    "SB505124", "PD173074",

    # ── Hedgehog pathway (PTCH1/SMO mutation drives sensitivity) ─────────────
    "Vismodegib",

    # ── IDH1/IDH2 mutant-specific inhibitors ─────────────────────────────────
    "AGI-5198", "AGI-6780",

    # ── MEK inhibitors (BRAF/RAS mutation downstream) ─────────────────────────
    "PD0325901",

    # ── DNA-damage response with clear mutation biomarker ───────────────────
    # ATM/ATR/CHEK inhibitors: BRCA / ATM mutation stratifies response
    "AZD6738", "KU-55933", "VE-822", "VE821", "AZD7762", "MK-8776",
    "Mirin", "NU7441", "BIBR-1532",

    # ── CDK4/6 (RB1 loss is the resistance biomarker) ───────────────────────
    "Palbociclib", "Ribociclib", "AZD5438",

    # ── Other specific single-target kinases with genomic context ────────────
    "Doramapimod",   # p38 / stress-kinase context
    "BX795",         # TBK1 / PDK1 — PDK1 genomic context
    "AZ960",         # JAK2 (JAK2 V617F mutation)
    "Ruxolitinib",   # JAK1/2 — JAK2 V617F
    "Entospletinib", "PRT062607",  # SYK — B-cell malignancy context
    "GSK2830371",    # WIP1 — TP53 pathway context
    "EHT-1864",      # RAC1 — mutation context
    "LY2109761",     # TGFB1 — pathway-specific
    "SL0101",        # RSK / AURK
    "JNK Inhibitor VIII",  # JNK — stress context
    "GSK269962A",    # ROCK1/2 — cytoskeletal but mutation-associated
    "Bosutinib",
})


def is_targeted(drug: str) -> bool:
    """Return True if the drug has a clear genomic driver where T1 axiom priority is valid."""
    return drug in TARGETED_DRUGS
