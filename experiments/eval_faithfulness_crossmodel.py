"""Cross-model faithfulness evaluation using Llama 3.3 70B via Groq as judge.

Addresses the same-model-family bias of the original eval (Gemini judging Gemini).
Uses an identical rubric and prompt to allow direct score comparison.

Usage:
    PYTHONPATH=. python experiments/eval_faithfulness_crossmodel.py

Outputs:
    experiments/results/faithfulness_eval_crossmodel.json
"""
import asyncio
import json
import os
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("VERTEX_PROJECT", "project-d3bf2d5b-3451-46fd-8f3")

from src.llm.client import LLMClient, make_rate_limiter

TRACES = Path("experiments/results/traces_gold_standard.jsonl")
OUT    = Path("experiments/results/faithfulness_eval_crossmodel.json")
JUDGE_MODEL = "groq:qwen/qwen3.8-27b"

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


def format_evidence(key_findings: list[dict], raw_evidence: dict | None) -> str:
    lines = ["Retrieved evidence (MCP tool outputs):"]
    for i, kf in enumerate(key_findings, 1):
        lines.append(f"  [{i}] biomarker={kf.get('biomarker')} | value={kf.get('value')} | "
                     f"interpretation={kf.get('interpretation')} | source={kf.get('data_source')}")
    if raw_evidence:
        lines.append("\nRaw database output (additional retrieved values):")
        import json as _json
        lines.append("  " + _json.dumps(raw_evidence, default=str)[:800])
    return "\n".join(lines)


async def judge_one(client: LLMClient, cell_line: str, drug: str, agent: str,
                    key_findings: list[dict], reasoning: str,
                    raw_evidence: dict | None = None) -> dict:
    evidence_str = format_evidence(key_findings, raw_evidence)
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
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


async def main():
    # Load already-completed cases to allow resuming
    if OUT.exists():
        existing = json.loads(OUT.read_text())
        done = {(r["cell_line"], r["drug"], r["agent"]) for r in existing.get("per_agent_results", [])}
        results = existing.get("per_agent_results", [])
        print(f"Resuming: {len(done)} already judged")
    else:
        done, results = set(), []

    records = [json.loads(l) for l in TRACES.open()]
    # Groq free tier: ~30 req/min for 70B
    limiter = make_rate_limiter(calls_per_minute=25)
    client  = LLMClient(model=JUDGE_MODEL, cache_db=Path("src/data/processed/llm_cache.db"),
                        rate_limiter=limiter)

    total_supported = sum(r["n_supported"] for r in results)
    total_unsupported = sum(r["n_unsupported"] for r in results)
    agents_with_hallucination = sum(1 for r in results if r["has_hallucination"])
    total_decisive_agents = len(results)

    for record in records:
        cell_line, drug = record["cell_line"], record["drug"]
        r1_agents = record.get("r1_agents", [])

        for agent_data in r1_agents:
            agent    = agent_data["agent_id"]
            verdict  = agent_data.get("verdict", "UNCERTAIN")
            if verdict == "UNCERTAIN":
                continue
            kf          = agent_data.get("key_findings") or []
            reasoning   = (agent_data.get("reasoning") or "").strip()
            raw_evidence = agent_data.get("raw_evidence")
            if not kf or not reasoning:
                continue
            if (cell_line, drug, agent) in done:
                continue

            total_decisive_agents += 1
            print(f"  judging {cell_line}+{drug} | {agent}...", flush=True)
            try:
                judgment = await judge_one(client, cell_line, drug, agent, kf, reasoning, raw_evidence)
            except Exception as e:
                print(f"    ERROR: {e}")
                total_decisive_agents -= 1
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
                "verdict": verdict,
                "faithfulness_score": judgment.get("faithfulness_score"),
                "has_hallucination": judgment.get("has_hallucination"),
                "n_claims": len(judgment["claims"]),
                "n_supported": sup,
                "n_unsupported": unsup,
                "summary": judgment.get("summary"),
            })
            done.add((cell_line, drug, agent))

            # Save after every case so we can resume
            _save(results, total_supported, total_unsupported,
                  agents_with_hallucination, total_decisive_agents, len(records))

    _save(results, total_supported, total_unsupported,
          agents_with_hallucination, total_decisive_agents, len(records))

    # Per-agent summary
    per_agent: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if r["faithfulness_score"] is not None:
            per_agent[r["agent"]].append(r["faithfulness_score"])

    print(f"\n{'='*60}")
    print(f"CROSS-MODEL FAITHFULNESS (judge: {JUDGE_MODEL})")
    print(f"{'='*60}")
    overall = total_supported / (total_supported + total_unsupported) if (total_supported + total_unsupported) else 1.0
    print(f"  Overall faithfulness:  {overall:.4f}")
    print(f"  Hallucination rate:    {agents_with_hallucination}/{total_decisive_agents} = {agents_with_hallucination/total_decisive_agents:.4f}")
    print(f"  Per agent:")
    for ag, scores in sorted(per_agent.items()):
        print(f"    {ag:<30} mean={sum(scores)/len(scores):.3f}  n={len(scores)}")


def _save(results, sup, unsup, halluc, total, n_cases):
    overall = sup / (sup + unsup) if (sup + unsup) else 1.0
    output = {
        "judge_model": JUDGE_MODEL,
        "n_cases": n_cases,
        "n_decisive_agents_evaluated": total,
        "overall_faithfulness_score": round(overall, 4),
        "hallucination_rate": round(halluc / total if total else 0, 4),
        "agents_with_hallucination": halluc,
        "total_supported_claims": sup,
        "total_unsupported_claims": unsup,
        "per_agent_results": results,
    }
    OUT.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
