"""Random Forest baseline on GDSC/CTRPv2 omics features.

Train on ~108K decisive CTRP pairs (excluding gold standard 31).
Features: target gene mutation status, target gene expression z-score,
          driver mutation count, key driver gene flags.
Test on 31 gold standard cases with literature labels.

Usage:
    PYTHONPATH=. python experiments/run_rf_baseline.py
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, cohen_kappa_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PHARMA_DB  = Path("src/data/processed/pharmacology.db")
GENO_DB    = Path("src/data/processed/genomics.db")
EXPR_DB    = Path("src/data/processed/transcriptomics.db")
GOLD_YAML  = Path("data/cases/cases_gold_standard.yaml")
OUT        = Path("experiments/results/rf_baseline.json")

# Key driver genes used as background features
DRIVER_GENES = [
    "EGFR", "KRAS", "BRAF", "NRAS", "PIK3CA", "PTEN", "TP53",
    "BRCA1", "BRCA2", "ERBB2", "MET", "ALK", "ROS1", "ABL1",
    "CDK4", "CDK6", "MDM2", "PARP1", "MEK1", "MEK2", "RB1",
]

# Gold standard cell-line/drug pairs to EXCLUDE from training
GOLD_CELL_LINES = {
    "NCI-H1975","BT-474","JIMT-1","HCC-827","HCC1937","A375","SK-MEL-28",
    "8505C","A2058","MCF7","MDA-MB-436","MDA-MB-231","T47D","A549","NCI-H1650",
    "NCI-H3122","HCT-116","K-562","Saos-2","SJSA-1","CAPAN-1","KMOE-2",
}
GOLD_DRUGS = {
    "Osimertinib","Lapatinib","Erlotinib","Gefitinib","Dabrafenib","PLX-4720",
    "Trametinib","Alpelisib","Pictilisib","Palbociclib","Olaparib","Dasatinib",
    "Nilotinib","Crizotinib","Nutlin-3a (-)",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_mutations() -> dict[str, set[str]]:
    """cell_line -> set of mutated gene names."""
    conn = sqlite3.connect(GENO_DB)
    rows = conn.execute("SELECT cell_line, gene FROM mutations WHERE mutation != '' OR mutation_type != ''").fetchall()
    conn.close()
    out: dict[str, set[str]] = {}
    for cl, gene in rows:
        out.setdefault(cl, set()).add(gene.upper())
    return out


def load_driver_mutations() -> dict[str, set[str]]:
    """cell_line -> set of mutated driver gene names."""
    conn = sqlite3.connect(GENO_DB)
    rows = conn.execute("SELECT cell_line, gene FROM mutations WHERE is_driver=1").fetchall()
    conn.close()
    out: dict[str, set[str]] = {}
    for cl, gene in rows:
        out.setdefault(cl, set()).add(gene.upper())
    return out


def load_expression() -> dict[tuple[str, str], float]:
    """(cell_line, gene) -> z_score."""
    conn = sqlite3.connect(EXPR_DB)
    rows = conn.execute("SELECT cell_line, gene, z_score FROM expression").fetchall()
    conn.close()
    return {(cl, g.upper()): z for cl, g, z in rows}


def load_drug_targets() -> dict[str, list[str]]:
    """drug -> list of target gene names."""
    conn = sqlite3.connect(PHARMA_DB)
    rows = conn.execute("SELECT drug, target_genes FROM drug_info").fetchall()
    conn.close()
    out: dict[str, list[str]] = {}
    for drug, tg in rows:
        if not tg:
            out[drug] = []
            continue
        genes = [g.strip().upper() for g in tg.replace("(", "").replace(")", "").split(",") if g.strip()]
        out[drug] = genes
    return out


def load_drug_response(decisive_only: bool = True) -> list[dict]:
    conn = sqlite3.connect(PHARMA_DB)
    conn.row_factory = sqlite3.Row
    if decisive_only:
        rows = conn.execute("SELECT * FROM drug_response WHERE label IN ('SENSITIVE','RESISTANT')").fetchall()
    else:
        rows = conn.execute("SELECT * FROM drug_response").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(
    cell_line: str,
    drug: str,
    mutations: dict,
    driver_muts: dict,
    expr: dict,
    drug_targets: dict,
) -> list[float]:
    targets = [g.upper() for g in drug_targets.get(drug, [])]
    cl_muts   = mutations.get(cell_line, set())
    cl_drivers = driver_muts.get(cell_line, set())

    # Target-level features
    target_mutated        = float(any(g in cl_muts for g in targets))
    target_driver_mutated = float(any(g in cl_drivers for g in targets))
    n_target_mutations    = float(sum(1 for g in targets if g in cl_muts))

    # Target expression features
    target_zscores = [expr.get((cell_line, g), 0.0) for g in targets]
    target_expr_mean = float(np.mean(target_zscores)) if target_zscores else 0.0
    target_expr_max  = float(np.max(target_zscores))  if target_zscores else 0.0

    # Background genomics
    n_driver_muts = float(len(cl_drivers))

    # Binary driver gene flags for key cancer genes
    driver_flags = [float(g in cl_muts) for g in DRIVER_GENES]

    return [
        target_mutated,
        target_driver_mutated,
        n_target_mutations,
        target_expr_mean,
        target_expr_max,
        n_driver_muts,
        *driver_flags,
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...", flush=True)
    mutations   = load_mutations()
    driver_muts = load_driver_mutations()
    expr        = load_expression()
    drug_targets = load_drug_targets()
    all_pairs   = load_drug_response(decisive_only=True)

    gold_cases = yaml.safe_load(GOLD_YAML.read_text())
    gold_index = {(c["cell_line"], c["drug"]): c["label"] for c in gold_cases}

    # Split: exclude gold standard pairs from training
    train_pairs = [
        r for r in all_pairs
        if not (r["cell_line"] in GOLD_CELL_LINES and r["drug"] in GOLD_DRUGS)
    ]
    print(f"Training pairs: {len(train_pairs)}", flush=True)

    # Build training features
    print("Building training features...", flush=True)
    X_train, y_train = [], []
    for r in train_pairs:
        feats = build_features(r["cell_line"], r["drug"], mutations, driver_muts, expr, drug_targets)
        X_train.append(feats)
        y_train.append(1 if r["label"] == "SENSITIVE" else 0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Train RF
    print("Training Random Forest...", flush=True)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # CV score on training set (sanity check)
    cv_auroc = cross_val_score(rf, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"Train 5-fold AUROC: {cv_auroc.mean():.4f} ± {cv_auroc.std():.4f}", flush=True)

    # Test on gold standard 31 cases
    print("\nGold standard predictions:", flush=True)
    results = []
    y_true, y_prob = [], []
    for case in gold_cases:
        cl, drug, label = case["cell_line"], case["drug"], case["label"]
        feats = build_features(cl, drug, mutations, driver_muts, expr, drug_targets)
        prob_sensitive = rf.predict_proba([feats])[0][1]
        pred = "SENSITIVE" if prob_sensitive >= 0.5 else "RESISTANT"
        correct = (pred == label)
        results.append({
            "cell_line": cl, "drug": drug,
            "gold_label": label,
            "rf_pred": pred,
            "prob_sensitive": round(float(prob_sensitive), 4),
            "correct": correct,
        })
        y_true.append(1 if label == "SENSITIVE" else 0)
        y_prob.append(float(prob_sensitive))
        print(f"  {'✓' if correct else '✗'}  {cl:20s} + {drug:20s}  label={label:10s} pred={pred:10s} p={prob_sensitive:.3f}")

    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    y_pred_arr = (y_prob_arr >= 0.5).astype(int)

    accuracy  = float(np.mean(y_true_arr == y_pred_arr))
    auroc     = float(roc_auc_score(y_true_arr, y_prob_arr))
    auprc     = float(average_precision_score(y_true_arr, y_prob_arr))
    kappa     = float(cohen_kappa_score(y_true_arr, y_pred_arr))

    print(f"\n{'='*60}")
    print(f"  RF Baseline on Gold Standard 31:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  AUROC:     {auroc:.4f}")
    print(f"  AUPRC:     {auprc:.4f}")
    print(f"  Cohen κ:   {kappa:.4f}")
    print(f"  Training N: {len(train_pairs):,} decisive CTRP pairs")
    print(f"")
    print(f"  Context (from ablation study):")
    print(f"  Full system AUROC:      0.819")
    print(f"  Only pharmacology:      0.863  (pure CTRP z-score lookup)")
    print(f"  Only genomics (CIViC):  0.772")

    output = {
        "model": "RandomForest",
        "training_pairs": len(train_pairs),
        "features": ["target_mutated", "target_driver_mutated", "n_target_mutations",
                     "target_expr_mean", "target_expr_max", "n_driver_muts"] + [f"driver_{g}" for g in DRIVER_GENES],
        "n_estimators": 300,
        "cv_auroc_mean": round(float(cv_auroc.mean()), 4),
        "cv_auroc_std":  round(float(cv_auroc.std()), 4),
        "gold_standard_n": len(gold_cases),
        "accuracy":  round(accuracy, 4),
        "auroc":     round(auroc, 4),
        "auprc":     round(auprc, 4),
        "cohens_kappa": round(kappa, 4),
        "per_case": results,
        "comparison": {
            "full_system": 0.819,
            "only_pharmacology_ctrp_lookup": 0.863,
            "only_genomics_civic": 0.772,
        },
    }
    OUT.write_text(json.dumps(output, indent=2))
    print(f"  Saved: {OUT}")


if __name__ == "__main__":
    main()
