"""Cross-tier leakage rate evaluation.

Distinction from hallucination (score_hallucination.py):
  Hallucination : gene cited is NOT present in raw_evidence at all
  Leakage       : gene IS present in raw_evidence, but the specific
                  mutation / value cited was NOT returned by the MCP
                  tool call — the agent injected it from parametric knowledge

Workflow:
  1. For every Round-1 finding where the gene is known in raw_evidence:
     a. Fast programmatic check: is the value string present in the evidence?
        YES → GROUNDED (no LLM call needed)
        NO  → potential leakage — send to Gemini judge
  2. Gemini judge returns GROUNDED / INFERRED / LEAKED per finding
  3. Aggregate per-agent and system-level leakage rates

Outputs:
  experiments/results/leakage_rates.json

Usage:
    PYTHONPATH=. python experiments/compute_leakage_rate.py
"""
import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("VERTEX_PROJECT", "project-d3bf2d5b-3451-46fd-8f3")

from src.llm.client import LLMClient, make_rate_limiter

TRACES = Path("experiments/results/traces_gold_standard.jsonl")
OUT    = Path("experiments/results/leakage_rates.json")
MODEL  = "vertex:gemini-3.1-flash-lite"

JUDGE_SYSTEM = """You are auditing a biomedical AI system for cross-tier leakage.

Cross-tier leakage: an agent cites a specific mutation or classification that is NOT
present in the raw database output it received from an MCP tool call, even though the
gene itself IS in the evidence. This means the agent injected detail from its training
knowledge rather than retrieving it.

This is different from hallucination, where the gene is entirely absent from evidence.

You will receive:
  1. A specific finding the agent cited (biomarker, value, interpretation, claimed source)
  2. The raw MCP tool output the agent actually received

Classify the finding as exactly one of:
  GROUNDED  — the specific value/mutation and interpretation appear directly in the raw evidence
  INFERRED  — the gene is in evidence; value is a standard biomedical interpretation (acceptable)
  LEAKED    — gene is in evidence but specific value/mutation/classification is NOT (parametric injection)

Return JSON only, no prose:
{"classification": "GROUNDED"|"INFERRED"|"LEAKED", "reason": "<one sentence>"}"""


def extract_known_genes(raw_evidence: dict) -> set[str]:
    genes: set[str] = set()
    for field_data in raw_evidence.values():
        if isinstance(field_data, list):
            for item in field_data:
                if isinstance(item, dict):
                    g = item.get("gene")
                    if g:
                        genes.add(str(g).upper())
    return genes


def value_in_evidence(value: str, raw_evidence: dict) -> bool:
    evidence_text = json.dumps(raw_evidence).lower()
    normalized = value.lower().lstrip("p.")
    return normalized in evidence_text or value.lower() in evidence_text


async def judge_finding(client: LLMClient, finding: dict, raw_evidence: dict) -> dict:
    evidence_str = json.dumps(raw_evidence, default=str)
    if len(evidence_str) > 2000:
        evidence_str = evidence_str[:2000] + "... [truncated]"

    msg = (
        f"Finding cited by agent:\n"
        f"  biomarker     : {finding.get('biomarker')}\n"
        f"  value         : {finding.get('value')}\n"
        f"  interpretation: {finding.get('interpretation')}\n"
        f"  claimed source: {finding.get('data_source')}\n\n"
        f"Raw MCP tool output the agent received:\n{evidence_str}\n\n"
        f"Classify this finding as GROUNDED, INFERRED, or LEAKED."
    )
    raw = await client.complete(
        messages=[{"role": "user", "content": msg}],
        system=JUDGE_SYSTEM,
    )
    text = raw.strip().strip("```json").strip("```").strip()
    return json.loads(text)


async def main():
    records = [json.loads(l) for l in TRACES.open() if l.strip()]
    print(f"Loaded {len(records)} traces")

    limiter = make_rate_limiter(calls_per_minute=8)
    client  = LLMClient(
        model=MODEL,
        cache_db=Path("src/data/processed/llm_cache.db"),
        rate_limiter=limiter,
    )

    per_case: list[dict] = []
    agent_stats: dict = defaultdict(lambda: {"grounded": 0, "inferred": 0, "leaked": 0, "total": 0})
    total = grounded = inferred = leaked = 0

    for i, r in enumerate(records):
        cell_line, drug = r["cell_line"], r["drug"]

        r1_verdicts: dict = {}
        for rnd in r.get("trace", []):
            if rnd.get("round") == 1:
                r1_verdicts = rnd.get("verdicts", {})
                break
        if not r1_verdicts and r.get("r1_agents"):
            r1_verdicts = {a["agent_id"]: a for a in r["r1_agents"] if "agent_id" in a}

        case_findings: list[dict] = []
        for agent_id, v in r1_verdicts.items():
            if v.get("verdict") == "UNCERTAIN":
                continue
            kf_list = v.get("key_findings") or []
            raw_ev  = v.get("raw_evidence") or {}
            if not kf_list or not raw_ev:
                continue

            known_genes = extract_known_genes(raw_ev)

            for kf in kf_list:
                biomarker = (kf.get("biomarker") or "").upper()
                value     = kf.get("value") or ""

                # Skip findings where gene is unknown — that's hallucination territory
                if biomarker not in known_genes:
                    continue

                total += 1
                agent_stats[agent_id]["total"] += 1

                if value_in_evidence(value, raw_ev):
                    classification = "GROUNDED"
                    reason = "Value present in raw evidence (string match)"
                else:
                    print(f"  [{i+1}/{len(records)}] {cell_line}+{drug} | {agent_id} | "
                          f"{biomarker} {value}", flush=True)
                    try:
                        judgment = await judge_finding(client, kf, raw_ev)
                        classification = judgment.get("classification", "LEAKED")
                        reason = judgment.get("reason", "")
                    except Exception as e:
                        print(f"    ERROR: {e}")
                        classification = "LEAKED"
                        reason = str(e)

                agent_stats[agent_id][classification.lower()] += 1
                if classification == "GROUNDED":
                    grounded += 1
                elif classification == "INFERRED":
                    inferred += 1
                else:
                    leaked += 1

                case_findings.append({
                    "agent": agent_id,
                    "biomarker": biomarker,
                    "value": value,
                    "classification": classification,
                    "reason": reason,
                })

        if case_findings:
            n_leaked = sum(1 for c in case_findings if c["classification"] == "LEAKED")
            per_case.append({
                "cell_line": cell_line,
                "drug": drug,
                "true_label": r.get("true_label"),
                "final_verdict": r.get("final_verdict"),
                "correct": r.get("correct"),
                "n_findings_audited": len(case_findings),
                "n_leaked": n_leaked,
                "leakage_rate": round(n_leaked / len(case_findings), 4),
                "findings": case_findings,
            })

    leakage_rate  = round(leaked  / total, 4) if total else 0.0
    grounded_rate = round(grounded / total, 4) if total else 0.0
    inferred_rate = round(inferred / total, 4) if total else 0.0

    output = {
        "model": MODEL,
        "n_cases": len(records),
        "total_findings_audited": total,
        "grounded": grounded,
        "inferred": inferred,
        "leaked": leaked,
        "leakage_rate": leakage_rate,
        "grounded_rate": grounded_rate,
        "inferred_rate": inferred_rate,
        "per_agent": {
            aid: {
                "total": s["total"],
                "grounded": s["grounded"],
                "inferred": s["inferred"],
                "leaked": s["leaked"],
                "leakage_rate": round(s["leaked"] / s["total"], 4) if s["total"] else 0.0,
            }
            for aid, s in agent_stats.items()
        },
        "per_case": per_case,
    }
    OUT.write_text(json.dumps(output, indent=2))

    print(f"\n{'='*60}")
    print(f"  Total findings audited : {total}")
    print(f"  Grounded : {grounded}  ({100*grounded_rate:.1f}%)")
    print(f"  Inferred : {inferred}  ({100*inferred_rate:.1f}%)")
    print(f"  Leaked   : {leaked}   ({100*leakage_rate:.1f}%)")
    print(f"  Output   : {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
