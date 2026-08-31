"""Programmatic hallucination rate: checks whether genes/mutations cited in
key_findings actually appear in the raw_evidence the agent received.

Definition: a hallucination is when an agent cites a specific gene or mutation
in key_findings that is NOT present in ANY field of its raw_evidence tool output.

This is an objective check -- no LLM judge involved.

Usage:
    PYTHONPATH=. python experiments/score_hallucination.py
"""
import json
from pathlib import Path
from collections import defaultdict

RESULTS = Path("experiments/results")


def extract_known_genes(agent_id: str, raw_evidence: dict) -> set[str]:
    """Return all gene names that appear anywhere in the agent's raw_evidence."""
    known = set()

    # T1: mutations + cnv + depmap + oncokb + rppa + wild_type
    for m in raw_evidence.get("mutations", []):
        if g := m.get("gene"):
            known.add(g)
    for c in raw_evidence.get("cnv", []):
        if g := c.get("gene"):
            known.add(g)
    for d in raw_evidence.get("depmap_scores", []):
        if g := d.get("gene"):
            known.add(g)
    for a in raw_evidence.get("oncokb_annotations", []):
        if g := a.get("gene"):
            known.add(g)
    for r in raw_evidence.get("rppa_expression", []):
        if g := r.get("gene"):
            known.add(g)
    for g in raw_evidence.get("wild_type_genes", []):
        known.add(g)

    # T2: expression entries
    for e in raw_evidence.get("expression", []):
        if g := e.get("gene"):
            known.add(g)
    for e in raw_evidence.get("target_expression", []):
        if g := e.get("gene"):
            known.add(g)

    # T3: pathway membership keys + bypass genes
    for pid, info in raw_evidence.get("pathway_membership", {}).items():
        for g in info.get("target_genes_present", []):
            known.add(g)
    for g in raw_evidence.get("confirmed_bypass_genes", []):
        known.add(g)
    for key, bcheck in raw_evidence.get("bypass_check", {}).items():
        if isinstance(bcheck, dict):
            for bg in bcheck.get("bypass_genes", []):
                if isinstance(bg, dict):
                    if g := bg.get("gene"):
                        known.add(g)
                elif isinstance(bg, str):
                    known.add(bg)

    # T4: IC50 data drug field (not a gene, skip)
    # drug_info keys
    for entry in raw_evidence.get("ic50", []) if isinstance(raw_evidence.get("ic50"), list) else []:
        pass  # T4 doesn't cite gene names in key_findings typically

    return known


def extract_known_mutations(raw_evidence: dict) -> set[tuple[str, str]]:
    """Return (gene, mutation) pairs from raw_evidence mutations."""
    pairs = set()
    for m in raw_evidence.get("mutations", []):
        g = m.get("gene", "")
        mut = m.get("mutation", "")
        if g and mut:
            pairs.add((g, mut))
    return pairs


def check_agent_hallucination(
    agent_id: str,
    key_findings: list,
    raw_evidence: dict,
    target_genes: list,
) -> tuple[bool, list[str]]:
    """Return (hallucinated, list_of_fabricated_items)."""
    if not key_findings or not raw_evidence:
        return False, []

    known_genes = extract_known_genes(agent_id, raw_evidence)
    known_mutations = extract_known_mutations(raw_evidence)

    # Target genes are always "known" -- agent legitimately knows the drug target
    for g in (target_genes or []):
        known_genes.add(g)

    fabricated = []
    for kf in key_findings:
        biomarker = kf.get("biomarker", "")
        value = kf.get("value", "")
        if not biomarker:
            continue

        # Check gene is known
        if biomarker not in known_genes:
            fabricated.append(f"{biomarker} (gene not in evidence)")
            continue

        # If value looks like a specific mutation (e.g. p.T790M, G12V), check it
        if value and value.startswith("p.") or (value and len(value) > 2 and value[0].isalpha() and value[-1].isalpha() and any(c.isdigit() for c in value)):
            # This looks like a mutation notation -- verify it's in raw_evidence
            pair = (biomarker, value)
            # Fuzzy: check if value appears as substring of any known mutation
            mutation_genes = {m.get("gene") for m in raw_evidence.get("mutations", [])}
            known_mut_values = {m.get("mutation", "") for m in raw_evidence.get("mutations", []) if m.get("gene") == biomarker}
            if biomarker in mutation_genes and value not in known_mut_values:
                # Check partial match (e.g. agent writes "L858R" vs evidence has "p.L858R")
                canon_value = value.lstrip("p.")
                if not any(canon_value in mv for mv in known_mut_values):
                    fabricated.append(f"{biomarker} {value} (mutation not in evidence)")

    return len(fabricated) > 0, fabricated


def main():
    traces_path = RESULTS / "traces_gold_standard.jsonl"
    records = []
    with open(traces_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    print(f"Loaded {len(records)} traces\n")

    agent_stats = defaultdict(lambda: {"total": 0, "hallucinated": 0, "examples": []})
    total_agents = 0
    total_hallucinated = 0

    for rec in records:
        cell_line = rec["cell_line"]
        drug = rec["drug"]
        target_genes = rec.get("target_genes") or []

        for round_data in rec.get("trace", []):
            if round_data.get("round") != 1:
                continue
            for agent_id, agent_data in round_data.get("verdicts", {}).items():
                kf = agent_data.get("key_findings") or []
                raw = agent_data.get("raw_evidence") or {}
                if not kf:
                    continue

                hallucinated, items = check_agent_hallucination(
                    agent_id, kf, raw, target_genes
                )
                total_agents += 1
                agent_stats[agent_id]["total"] += 1
                if hallucinated:
                    total_hallucinated += 1
                    agent_stats[agent_id]["hallucinated"] += 1
                    if len(agent_stats[agent_id]["examples"]) < 3:
                        agent_stats[agent_id]["examples"].append(
                            {"case": f"{cell_line}+{drug}", "fabricated": items}
                        )

    print("=" * 60)
    print(f"  Hallucination Rate: {total_hallucinated}/{total_agents} "
          f"({100*total_hallucinated/total_agents:.1f}%)")
    print("=" * 60)
    print()
    print("  Per-agent breakdown:")
    for agent_id, stats in sorted(agent_stats.items()):
        rate = stats['hallucinated'] / stats['total'] if stats['total'] else 0
        print(f"    {agent_id}: {stats['hallucinated']}/{stats['total']} "
              f"({100*rate:.1f}%)")
        for ex in stats["examples"]:
            print(f"      [{ex['case']}] {ex['fabricated']}")
    print()

    output = {
        "total_agent_outputs": total_agents,
        "hallucinated": total_hallucinated,
        "hallucination_rate": round(total_hallucinated / total_agents, 4) if total_agents else 0,
        "per_agent": {
            aid: {
                "total": s["total"],
                "hallucinated": s["hallucinated"],
                "rate": round(s["hallucinated"] / s["total"], 4) if s["total"] else 0,
            }
            for aid, s in agent_stats.items()
        },
    }
    out_path = RESULTS / "hallucination_scores.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
