"""Score Biological Faithfulness (Metric 4) and Hallucination Rate (Metric 5).

Reads traces_gold_standard.jsonl, sends each agent's reasoning + key_findings
to the LLM judge, and saves results to experiments/results/qualitative_scores.json.

Usage:
    PYTHONPATH=. python experiments/score_qualitative.py
    PYTHONPATH=. python experiments/score_qualitative.py --traces experiments/results/traces_gold_standard.jsonl
"""

import asyncio
import json
import os
import argparse
from pathlib import Path

os.environ.setdefault("VERTEX_PROJECT", "project-d3bf2d5b-3451-46fd-8f3")

from src.llm.client import LLMClient, make_rate_limiter

DATA = Path("src/data/processed")
RESULTS = Path("experiments/results")
MODEL = "vertex:gemini-3.1-flash-lite"

FAITHFULNESS_SYSTEM = """\
You are a biomedical expert evaluating the quality of drug-response reasoning produced
by an AI agent. You will be given the agent's structured evidence (key_findings) and
its free-text reasoning. Score the BIOLOGICAL FAITHFULNESS on a 1-5 Likert scale.

Scoring rubric:
5 — Mechanistically precise. All claims are grounded in the retrieved data.
    Biological logic is correct, drug mechanism is accurately applied, no invented facts.
4 — Mostly valid. Minor imprecision or missing nuance but no significant errors.
3 — Generally valid but with logical gaps, oversimplifications, or unclear causal chains.
2 — Significant biological errors or interpretation mistakes; some claims unsupported.
1 — Biologically invalid, contradicts the data, or contains clear hallucinations.

Respond in JSON only:
{"score": <1-5>, "justification": "<one sentence>", "issues": ["<issue1>", ...]}
issues should list specific biological errors or unsupported claims (empty list if none)."""

HALLUCINATION_SYSTEM = """\
You are a fact-checker for AI-generated biomedical reasoning. You will be given:
- key_findings: structured data actually retrieved by the agent from databases
- reasoning: the agent's free-text explanation

Your task: identify any FACTUAL CLAIMS in the reasoning that are NOT supported by
the key_findings. A hallucination is a specific stated fact (gene name with specific
value, mutation position, numeric expression level, drug response value, database
citation) that does not appear in or cannot be inferred from the key_findings.

Do NOT flag:
- General biological background knowledge (e.g., "EGFR TKIs block kinase activity")
- Logical inferences from the data
- Statements about drug mechanism not requiring database lookup

DO flag:
- Specific mutation names/positions claimed but not in key_findings
- Specific z-scores or numeric values cited but absent from key_findings
- Database entries (e.g., "CIViC Level A") claimed but not in key_findings
- Cell-line-specific facts not derivable from the key_findings

Respond in JSON only:
{"hallucinations": ["<claim1>", ...], "has_hallucination": true/false}
If no hallucinations: {"hallucinations": [], "has_hallucination": false}"""


async def score_agent(client: LLMClient, case_id: str, agent: dict) -> dict:
    agent_id = agent["agent_id"]
    key_findings = agent.get("key_findings", [])
    reasoning = agent.get("reasoning", "")
    verdict = agent.get("verdict", "UNCERTAIN")

    if not reasoning or verdict == "UNCERTAIN":
        return {
            "agent_id": agent_id,
            "verdict": verdict,
            "skipped": True,
            "reason": "no reasoning or UNCERTAIN verdict",
        }

    findings_text = json.dumps(key_findings, indent=2)

    prompt_faith = (
        f"Case: {case_id}\nAgent: {agent_id}\nVerdict: {verdict}\n\n"
        f"KEY FINDINGS (retrieved from databases):\n{findings_text}\n\n"
        f"AGENT REASONING:\n{reasoning}\n\nScore the biological faithfulness 1-5."
    )

    prompt_halluc = (
        f"Case: {case_id}\nAgent: {agent_id}\n\n"
        f"KEY FINDINGS (retrieved from databases):\n{findings_text}\n\n"
        f"AGENT REASONING:\n{reasoning}\n\nIdentify any hallucinated factual claims."
    )

    faith_raw = await client.complete(
        [{"role": "user", "content": prompt_faith}],
        system=FAITHFULNESS_SYSTEM,
    )
    halluc_raw = await client.complete(
        [{"role": "user", "content": prompt_halluc}],
        system=HALLUCINATION_SYSTEM,
    )

    def parse_json(raw: str) -> dict:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return json.loads(raw)
        except Exception:
            return {"parse_error": raw[:200]}

    faith = parse_json(faith_raw)
    halluc = parse_json(halluc_raw)

    return {
        "agent_id": agent_id,
        "verdict": verdict,
        "faithfulness_score": faith.get("score"),
        "faithfulness_justification": faith.get("justification", ""),
        "faithfulness_issues": faith.get("issues", []),
        "has_hallucination": halluc.get("has_hallucination", False),
        "hallucinations": halluc.get("hallucinations", []),
        "skipped": False,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default=str(RESULTS / "traces_gold_standard.jsonl"))
    args = parser.parse_args()

    traces_path = Path(args.traces)
    records = []
    with open(traces_path) as f:
        for line in f:
            records.append(json.loads(line.strip()))

    limiter = make_rate_limiter()
    client = LLMClient(model=MODEL, cache_db=DATA / "llm_cache.db", rate_limiter=limiter)

    all_scores = []
    for r in records:
        case_label = f"{r['cell_line']} + {r['drug']}"
        case_id = r.get("case_id", case_label)
        print(f"\n[{case_label}]")
        case_results = {"case": case_label, "true_label": r["true_label"],
                        "final_verdict": r["final_verdict"], "correct": r["correct"],
                        "agent_scores": []}

        agents = r.get("r1_agents", [])
        tasks = [score_agent(client, case_id, a) for a in agents]
        scores = await asyncio.gather(*tasks)

        for s in scores:
            if s.get("skipped"):
                print(f"  {s['agent_id']}: skipped ({s['reason']})")
            else:
                flag = " [HALLUCINATION]" if s.get("has_hallucination") else ""
                print(f"  {s['agent_id']}: faithfulness={s['faithfulness_score']}/5{flag}")
            case_results["agent_scores"].append(s)

        all_scores.append(case_results)

    # Aggregate
    faith_scores = [
        s["faithfulness_score"]
        for r in all_scores
        for s in r["agent_scores"]
        if not s.get("skipped") and s.get("faithfulness_score") is not None
    ]
    halluc_agents = [
        s
        for r in all_scores
        for s in r["agent_scores"]
        if not s.get("skipped")
    ]
    n_halluc = sum(1 for s in halluc_agents if s.get("has_hallucination"))

    mean_faith = sum(faith_scores) / len(faith_scores) if faith_scores else 0
    halluc_rate = n_halluc / len(halluc_agents) if halluc_agents else 0

    print(f"\n{'='*60}")
    print(f"  Biological Faithfulness Score: {mean_faith:.2f}/5  (target: >3.5)")
    print(f"  Hallucination Rate: {halluc_rate:.1%}  ({n_halluc}/{len(halluc_agents)} agents)  (target: <5%)")
    print(f"{'='*60}")

    # Per-agent breakdown
    per_agent: dict[str, list] = {}
    for r in all_scores:
        for s in r["agent_scores"]:
            if not s.get("skipped") and s.get("faithfulness_score"):
                per_agent.setdefault(s["agent_id"], []).append(s["faithfulness_score"])
    print("\nMean faithfulness by agent:")
    for agent, scores_list in sorted(per_agent.items()):
        print(f"  {agent}: {sum(scores_list)/len(scores_list):.2f}/5  (n={len(scores_list)})")

    output = {
        "traces_source": str(traces_path),
        "n_cases": len(all_scores),
        "n_agents_scored": len(halluc_agents),
        "metric_4_biological_faithfulness": {
            "mean_score": round(mean_faith, 3),
            "target": ">3.5",
            "n_scored": len(faith_scores),
            "per_agent": {a: round(sum(v)/len(v), 3) for a, v in per_agent.items()},
        },
        "metric_5_hallucination_rate": {
            "rate": round(halluc_rate, 4),
            "target": "<0.05",
            "n_with_hallucination": n_halluc,
            "n_total_agents": len(halluc_agents),
        },
        "case_scores": all_scores,
    }

    out_path = RESULTS / "qualitative_scores.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
