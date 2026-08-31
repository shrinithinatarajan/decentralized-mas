"""Build OpenEvoMDT benchmark JSONL from gold standard cases.

Formats each case as a biomedical_qa A/B question.
Biomarkers come from our genomics DB (driver mutations) + known structural
alterations from the gold standard YAML comments.

Usage:
    PYTHONPATH=. python experiments/build_evomdt_benchmark.py
Outputs:
    data/evomdt_benchmark.jsonl
"""
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import yaml

GOLD_YAML  = Path("data/cases/cases_gold_standard.yaml")
GENO_DB    = Path("src/data/processed/genomics.db")
OUT        = Path("data/evomdt_benchmark.jsonl")

# Cancer type per cell line
CANCER_TYPES = {
    "NCI-H1975": "NSCLC (lung adenocarcinoma)",
    "HCC-827":   "NSCLC (lung adenocarcinoma)",
    "NCI-H1650": "NSCLC (lung adenocarcinoma)",
    "NCI-H3122": "NSCLC (lung adenocarcinoma)",
    "A549":      "NSCLC (lung adenocarcinoma, KRAS-mutant)",
    "A375":      "Melanoma (BRAF V600E)",
    "SK-MEL-28": "Melanoma (BRAF V600E)",
    "A2058":     "Melanoma (BRAF V600E)",
    "KMOE-2":    "Melanoma (BRAF V600E, NRAS Q61R)",
    "8505C":     "Anaplastic thyroid carcinoma (BRAF V600E)",
    "BT-474":    "Breast cancer (HER2-amplified, ER+)",
    "JIMT-1":    "Breast cancer (HER2-amplified, intrinsically resistant)",
    "MCF7":      "Breast cancer (ER+, HER2-negative)",
    "MDA-MB-231": "Triple-negative breast cancer (BRCA1/2 wild-type)",
    "MDA-MB-436": "Triple-negative breast cancer (BRCA1-mutant)",
    "HCC1937":   "Triple-negative breast cancer (BRCA1-mutant)",
    "T47D":      "Breast cancer (ER+, PIK3CA H1047R)",
    "CAPAN-1":   "Pancreatic adenocarcinoma (BRCA2-mutant, KRAS G12V)",
    "HCT-116":   "Colorectal carcinoma (KRAS G13D, microsatellite instability-high)",
    "K-562":     "Chronic myelogenous leukemia (BCR-ABL1 fusion)",
    "Saos-2":    "Osteosarcoma (TP53-null, RB1-null)",
    "SJSA-1":    "Osteosarcoma (MDM2-amplified, TP53 wild-type)",
}

# Known structural alterations not captured as point mutations in the SNV table
EXTRA_BIOMARKERS = {
    "HCC-827":   [("EGFR", "exon-19 deletion (E746-A750del)"), ("PTEN", "intact")],
    "NCI-H1650": [("EGFR", "exon-19 deletion (E746-A750del)"), ("PTEN", "deleted (homozygous loss)")],
    "NCI-H3122": [("EML4-ALK", "fusion rearrangement (variant 1)")],
    "K-562":     [("BCR-ABL1", "fusion (Philadelphia chromosome)")],
    "BT-474":    [("ERBB2", "amplification (HER2+)"), ("PIK3CA", "wild-type")],
    "JIMT-1":    [("ERBB2", "amplification (HER2+)"), ("PIK3CA", "mutated"), ("PTEN", "low expression"), ("NRG1", "high expression — bypass ligand")],
    "SJSA-1":    [("MDM2", "amplification"), ("TP53", "wild-type")],
    "Saos-2":    [("TP53", "null (homozygous deletion)"), ("RB1", "null")],
}

# Drug mechanism summaries for context
DRUG_MOA = {
    "Osimertinib":   "3rd-generation EGFR TKI; specifically designed to overcome T790M gatekeeper resistance",
    "Lapatinib":     "dual EGFR/ERBB2 (HER2) TKI",
    "Erlotinib":     "1st-generation EGFR TKI (reversible)",
    "Gefitinib":     "1st-generation EGFR TKI (reversible)",
    "Dabrafenib":    "selective BRAF V600E inhibitor",
    "PLX-4720":      "selective BRAF V600E inhibitor (tool compound, same mechanism as vemurafenib)",
    "Trametinib":    "MEK1/2 inhibitor (downstream of BRAF/RAS)",
    "Alpelisib":     "PI3K-alpha (PIK3CA) selective inhibitor",
    "Pictilisib":    "pan-PI3K class I inhibitor",
    "Palbociclib":   "CDK4/6 inhibitor; requires functional RB1",
    "Olaparib":      "PARP1/2 inhibitor; requires BRCA1/2 loss-of-function for synthetic lethality",
    "Dasatinib":     "BCR-ABL / SRC inhibitor (2nd-generation)",
    "Nilotinib":     "BCR-ABL inhibitor (2nd-generation, 20-50x more potent than imatinib)",
    "Crizotinib":    "ALK/MET/ROS1 inhibitor",
    "Nutlin-3a (-)": "MDM2 antagonist; activates p53; only effective when TP53 is wild-type",
}


def load_driver_mutations() -> dict[str, list[str]]:
    conn = sqlite3.connect(GENO_DB)
    rows = conn.execute(
        "SELECT cell_line, gene, mutation FROM mutations WHERE is_driver=1 AND mutation != ''"
    ).fetchall()
    conn.close()
    seen: dict[str, set] = defaultdict(set)
    out: dict[str, list] = defaultdict(list)
    for cl, gene, mut in rows:
        key = (gene, mut)
        if key not in seen[cl]:
            seen[cl].add(key)
            out[cl].append({"name": gene, "value": mut})
    return dict(out)


def build_case(case: dict, driver_muts: dict) -> dict:
    cl    = case["cell_line"]
    drug  = case["drug"]
    label = case["label"]  # SENSITIVE or RESISTANT

    biomarkers = list(driver_muts.get(cl, []))
    for gene, val in EXTRA_BIOMARKERS.get(cl, []):
        biomarkers.append({"name": gene, "value": val})

    cancer = CANCER_TYPES.get(cl, "cancer cell line")
    moa    = DRUG_MOA.get(drug, drug)

    biomarker_str = "; ".join(f"{b['name']} {b['value']}" for b in biomarkers[:12]) if biomarkers else "no additional driver mutations identified"

    question = (
        f"Cell line: {cl} ({cancer}). "
        f"Key genomic alterations: {biomarker_str}. "
        f"Drug: {drug} — mechanism: {moa}. "
        f"Based on the molecular profile and drug mechanism, predict whether this cell line will be: "
        f"(A) SENSITIVE to {drug}, or (B) RESISTANT to {drug}?"
    )

    return {
        "case_id": f"{cl}__{drug.replace(' ', '_').replace('(', '').replace(')', '')}",
        "domain": "biomedical_qa",
        "task_type": "generation",
        "cancer_type": cancer,
        "patient_context": {
            "age": None,
            "sex": "unknown",
            "summary": f"Cancer cell line {cl} ({cancer}) used in pre-clinical drug sensitivity testing.",
        },
        "biomarkers": biomarkers[:12],
        "preferences": ["Return only the benchmark answer label A or B."],
        "question": question,
        "options": ["A", "B"],
        "reference_answer": "A" if label == "SENSITIVE" else "B",
        "metadata": {
            "output_contract": {
                "type": "enum_choice",
                "allowed_values": ["A", "B"],
                "final_answer_only": True,
            },
            "require_provenance": False,
        },
    }


def main():
    cases       = yaml.safe_load(GOLD_YAML.read_text())
    driver_muts = load_driver_mutations()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for case in cases:
            record = build_case(case, driver_muts)
            f.write(json.dumps(record) + "\n")

    print(f"Written {len(cases)} cases to {OUT}")
    # Print one example
    cases_built = [json.loads(l) for l in OUT.read_text().strip().splitlines()]
    print("\nExample (first case):")
    ex = cases_built[0]
    print(f"  case_id: {ex['case_id']}")
    print(f"  question: {ex['question'][:200]}...")
    print(f"  reference_answer: {ex['reference_answer']}")
    print(f"  biomarkers: {ex['biomarkers'][:4]}")


if __name__ == "__main__":
    main()
