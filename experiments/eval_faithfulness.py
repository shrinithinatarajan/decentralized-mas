"""LLM-as-judge faithfulness evaluation.

For each decisive agent in round 1, an LLM judge receives:
  - The structured key_findings (retrieved evidence from MCP tools)
  - The agent's free-text reasoning

The judge decomposes reasoning into atomic factual claims and marks each as:
  supported   — directly traceable to a key_finding value/source
  inferred    — biomedical domain knowledge derived from supported facts (not a violation)
  unsupported — factual assertion with no trace in the retrieved evidence

Faithfulness = supported / (supported + unsupported)
Hallucination rate = unsupported_agents / total_decisive_agents

Usage:
    PYTHONPATH=. python experiments/eval_faithfulness.py
"""
import asyncio
import json
import os
from pathlib import Path

os.environ.setdefault("VERTEX_PROJECT", "project-d3bf2d5b-3451-46fd-8f3")

from src.llm.client import LLMClient, make_rate_limiter

TRACES = Path("experiments/results/traces_gold_standard.jsonl")
OUT    = Path("experiments/results/faithfulness_eval.json")
MODEL  = "vertex:gemini-3.1-flash-lite"

JUDGE_SYSTEM = """You are a faithfulness auditor for a multi-agent biomedical reasoning system.

Your job: given (1) the raw evidence an agent retrieved from databases, and (2) the agent's reasoning text, decompose the reasoning into atomic factual claims and classify each claim.

A "factual claim" is any assertion about the world that could be true or false based on data:
- Biomarker presence or absence ("EGFR L858R is present")
- Numerical values ("z_score is -3.85", "IC50 ln = -0.11")
- Database classifications ("CIViC classifies T790M as sensitive")
- Pathway membership ("BRAF is in the MAPK/ERK pathway")
- Expression levels ("ERBB2 TPM is 344.5")

NOT a factual claim (exclude from evaluation):
- Logical conclusions ("therefore the cell line is sensitive")
- Domain knowledge inferences ("T790M is a well-known resistance mechanism")
- Confidence statements ("I am confident that...")
- Uncertainty expressions ("this is borderline because...")

For each factual claim you identify, classify it as ONE of:
  supported   — the claim directly matches a value, biomarker, or source in the retrieved evidence
  inferred    — the claim is a standard biomedical interpretation of supported facts (acceptable)
  unsupported — the claim asserts a specific fact NOT present in the retrieved evidence

Return JSON only, no prose:
{
  "claims": [
    {"claim": "...", "type": "supported"|"inferred"|"unsupported", "evidence_ref": "which key_finding supports it, or null"}
  ],
  "faithfulness_score": <float 0-1, supported/(supported+unsupported)>,
  "has_hallucination": <bool, true if any unsupported claims exist>,
  "summary": "<one sentence>"
}"""

def format_evidence(raw_evidence: dict) -> str:
    import json as _json
    lines = ["Retrieved evidence (raw MCP tool outputs):"]
    # Exclude non-data metadata keys
    skip = {"data_status", "error"}
    for key, val in raw_evidence.items():
        if key in skip or val is None:
            continue
        # Serialize compactly; truncate very long values
        val_str = _json.dumps(val, default=str)
        if len(val_str) > 800:
            val_str = val_str[:800] + "... [truncated]"
        lines.append(f"  [{key}]: {val_str}")
    return "\n".join(lines)


def format_evidence_from_key_findings(key_findings: list[dict]) -> str:
    lines = ["Retrieved evidence (key_findings summary — raw_evidence not available):"]
    for i, kf in enumerate(key_findings, 1):
        lines.append(f"  [{i}] biomarker={kf.get('biomarker')} | value={kf.get('value')} | "
                     f"interpretation={kf.get('interpretation')} | source={kf.get('data_source')}")
    return "\n".join(lines)


async def judge_one(client: LLMClient, cell_line: str, drug: str, agent: str,
                    key_findings: list[dict], reasoning: str, raw_evidence: dict | None = None) -> dict:
    evidence_str = format_evidence(raw_evidence) if raw_evidence else format_evidence_from_key_findings(key_findings)
    user_msg = f"""Case: {cell_line} + {drug}
Agent: {agent}

{evidence_str}

Agent reasoning:
{reasoning}

Classify each factual claim in the reasoning."""

    raw = await client.complete(
        messages=[{"role": "user", "content": user_msg}],
        system=JUDGE_SYSTEM,
    )
    # Parse JSON — strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


async def main():
    records = [json.loads(l) for l in TRACES.open()]
    limiter = make_rate_limiter(calls_per_minute=8)
    client  = LLMClient(model=MODEL, cache_db=Path("src/data/processed/llm_cache.db"),
                        rate_limiter=limiter)

    results = []
    total_supported = total_unsupported = 0
    agents_with_hallucination = 0
    total_decisive_agents = 0

    for r in records:
        cell_line, drug = r["cell_line"], r["drug"]
        # get round-1 verdicts
        r1_verdicts = {}
        for rnd in r["trace"]:
            if rnd["round"] == 1:
                r1_verdicts = rnd.get("verdicts", {})
                break

        for agent, v in r1_verdicts.items():
            if v["verdict"] == "UNCERTAIN":
                continue
            kf = v.get("key_findings") or []
            reasoning = v.get("reasoning", "").strip()
            if not kf or not reasoning:
                continue

            total_decisive_agents += 1
            print(f"  judging {cell_line}+{drug} | {agent}...", flush=True)
            raw_ev = v.get("raw_evidence") or {}
            try:
                judgment = await judge_one(client, cell_line, drug, agent, kf, reasoning, raw_ev)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue

            sup   = sum(1 for c in judgment["claims"] if c["type"] == "supported")
            unsup = sum(1 for c in judgment["claims"] if c["type"] == "unsupported")
            total_supported   += sup
            total_unsupported += unsup
            if judgment.get("has_hallucination"):
                agents_with_hallucination += 1

            results.append({
                "cell_line": cell_line,
                "drug": drug,
                "agent": agent,
                "verdict": v["verdict"],
                "faithfulness_score": judgment.get("faithfulness_score"),
                "has_hallucination": judgment.get("has_hallucination"),
                "n_claims": len(judgment["claims"]),
                "n_supported": sup,
                "n_unsupported": unsup,
                "summary": judgment.get("summary"),
                "claims": judgment["claims"],
            })

    overall_faithfulness = total_supported / (total_supported + total_unsupported) if (total_supported + total_unsupported) else 1.0
    hallucination_rate = agents_with_hallucination / total_decisive_agents if total_decisive_agents else 0.0

    output = {
        "model": MODEL,
        "n_cases": len(records),
        "n_decisive_agents_evaluated": total_decisive_agents,
        "overall_faithfulness_score": round(overall_faithfulness, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "agents_with_hallucination": agents_with_hallucination,
        "total_supported_claims": total_supported,
        "total_unsupported_claims": total_unsupported,
        "per_agent_results": results,
    }

    OUT.write_text(json.dumps(output, indent=2))
    print(f"\n{'='*60}")
    print(f"  Decisive agents evaluated: {total_decisive_agents}")
    print(f"  Overall faithfulness:      {overall_faithfulness:.4f}")
    print(f"  Hallucination rate:        {hallucination_rate:.4f} ({agents_with_hallucination}/{total_decisive_agents})")
    print(f"  Total claims:              {total_supported + total_unsupported}")
    print(f"    supported:               {total_supported}")
    print(f"    unsupported:             {total_unsupported}")
    print(f"  Output: {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
