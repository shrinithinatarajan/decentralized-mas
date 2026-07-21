"""Run 3 cases and print a full debate trace to verify the council mechanism.

Usage:
    python experiments/debug_debate_trace.py
"""
import asyncio
import json
import os
from pathlib import Path

os.environ.setdefault("VERTEX_PROJECT", "project-d3bf2d5b-3451-46fd-8f3")

from src.agents.genomics_agent import GenomicsAgent
from src.agents.transcriptomics_agent import TranscriptomicsAgent
from src.agents.pharmacology_agent import PharmacologyAgent
from src.agents.pathway_agent import PathwayAgent
from src.llm.client import LLMClient, make_rate_limiter
from src.orchestrator import Orchestrator
from src.schemas.evidence_pack import Verdict

DATA = Path("src/data/processed")

# Single case: TE-15 + Dabrafenib was wrong (T3 falsely voted RESISTANT) before pathway activity + quorum fixes
CASES = [
    {"cell_line": "TE-15", "drug": "Dabrafenib", "label": "SENSITIVE", "target_genes": ["BRAF"]},
]

SEP = "=" * 70
SEP2 = "-" * 70


def _mcp_apps():
    from src.mcp_servers.genomics_server import mcp as g
    from src.mcp_servers.transcriptomics_server import mcp as t
    from src.mcp_servers.pharmacology_server import mcp as p
    from src.mcp_servers.pathway_server import mcp as pw
    return g, t, p, pw


class InstrumentedOrchestrator(Orchestrator):
    """Patches the debate engine to print verbose traces."""

    async def run_case(self, cell_line, drug, target_genes=None):
        from src.protocols.debate_engine import _check_consensus

        print(f"\n{SEP}")
        print(f"CASE: {cell_line} + {drug}")
        print(SEP)

        # --- Round 1: independent agent analysis ---
        packs = list(await asyncio.gather(*[a.analyze(cell_line, drug, target_genes) for a in self.agents]))

        print("\nROUND 1 — Independent Analysis:")
        for p in packs:
            print(f"  {p.agent_id:<30} {p.verdict.value:<12} conf={p.confidence:.2f}  tier={p.evidence_tier.value}")
            if p.reasoning:
                print(f"    reasoning: {p.reasoning[:120].strip()}...")

        consensus = _check_consensus(packs)
        if consensus:
            print(f"\n  ✓ CONSENSUS_R1 → {consensus.value}")
            result = await self.engine.run(packs, agents=self.agents)
            print(f"  Final: {result.final_verdict.value}  conf={result.final_confidence:.2f}  method={result.resolution_method}")
            return result

        print("  ✗ No R1 consensus — entering peer critique round")

        # --- Round 2: critique (verbose) ---
        agent_map = {a.agent_id: a for a in self.agents}
        peer_order = {p.agent_id: [q for q in packs if q.agent_id != p.agent_id] for p in packs}
        labels_map = {p.agent_id: {chr(ord('A') + i): peer_order[p.agent_id][i].agent_id
                                    for i in range(len(peer_order[p.agent_id]))}
                      for p in packs}

        print("\nROUND 2 — Peer Critique (verdicts locked):")
        critiqued = []
        endorsements = {p.agent_id: 0.0 for p in packs}
        for pack in packs:
            agent = agent_map.get(pack.agent_id)
            peers = peer_order[pack.agent_id]
            label_to_id = labels_map[pack.agent_id]
            c = await agent.critique(pack, peers)
            critiqued.append(c)
            print(f"\n  Reviewer: {pack.agent_id} (verdict locked: {c.verdict.value}, new conf={c.confidence:.2f})")
            if c.peer_scores:
                for label, score in sorted(c.peer_scores.items()):
                    peer_id = label_to_id.get(label, "?")
                    print(f"    Scored Peer {label} ({peer_id}): {score}/5")
                    endorsements[peer_id] += score
            else:
                print("    (no peer scores returned)")

        print("\n  Peer Endorsement Totals:")
        for agent_id, total in sorted(endorsements.items(), key=lambda x: -x[1]):
            pack = next(p for p in critiqued if p.agent_id == agent_id)
            print(f"    {agent_id:<30} total_score={total:.1f}  verdict={pack.verdict.value}")

        # --- Resolver ---
        print("\nRESOLVER:")
        decisive = [p for p in critiqued if p.verdict != Verdict.UNCERTAIN]
        print(f"  Decisive agents: {[p.agent_id for p in decisive]}")
        if not decisive:
            print("  ⚠ All agents UNCERTAIN — resolver falls back to all packs")
        resolution = self.engine._resolver.resolve(critiqued, peer_endorsements=endorsements)
        print(f"  Winner: {resolution.winning_agent}  verdict={resolution.verdict.value}  axiom={resolution.axiom_applied}")

        result = await self.engine.run(packs, agents=self.agents)
        print(f"\n  Final: {result.final_verdict.value}  conf={result.final_confidence:.2f}  method={result.resolution_method}")
        return result


async def main():
    limiter = make_rate_limiter()
    client = LLMClient(model="vertex:gemini-3.1-flash-lite", cache_db=DATA / "llm_cache.db", rate_limiter=limiter)
    g_app, t_app, p_app, pw_app = _mcp_apps()
    agents = [
        GenomicsAgent(g_app, client),
        TranscriptomicsAgent(t_app, client),
        PharmacologyAgent(p_app, client),
        PathwayAgent(pw_app, client, transcriptomics_mcp=t_app),
    ]
    orch = InstrumentedOrchestrator(agents=agents)

    for case in CASES:
        result = await orch.run_case(case["cell_line"], case["drug"], target_genes=case.get("target_genes"))
        label = case["label"]
        match = "✓" if result.final_verdict.value == label else ("?" if result.final_verdict == Verdict.UNCERTAIN else "✗")
        print(f"\n  Ground truth: {label}  {match}")
        print(SEP2)


if __name__ == "__main__":
    asyncio.run(main())
