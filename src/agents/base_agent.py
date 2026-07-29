import json
import re
from abc import ABC, abstractmethod

from fastmcp import Client

from src.agents.gcs import compute_gcs
from src.llm.client import LLMClient
from src.schemas.evidence_pack import EvidencePack, EvidenceTier, Verdict

def _truncate_evidence(evidence: dict, max_items: int = 20) -> dict:
    result = {}
    for k, v in evidence.items():
        if isinstance(v, list):
            result[k] = v[:max_items]
        elif isinstance(v, dict):
            result[k] = _truncate_evidence(v, max_items)
        else:
            result[k] = v
    return result


_SCHEMA_EXAMPLE = json.dumps({
    "agent_id": "<your_agent_id e.g. genomics_agent>",
    "cell_line": "<cell_line name>",
    "drug": "<drug name>",
    "reasoning": "<step-by-step: (1) what evidence did MCP return? (2) what does it imply for drug sensitivity? (3) are there contradictions? (4) final verdict rationale>",
    "verdict": "SENSITIVE or RESISTANT or UNCERTAIN",
    "confidence": 0.85,
    "evidence_tier": "T1_STRUCTURAL or T2_TRANSCRIPTIONAL or T3_PATHWAY or T4_PHARMACOLOGICAL or T5_STATISTICAL",
    "key_findings": [
        {"biomarker": "GENE", "value": "alteration", "interpretation": "biological effect", "data_source": "db_name"}
    ],
    "caveats": [],
    "conflict_flags": []
})


class BaseAgent(ABC):
    agent_id: str

    def __init__(self, mcp_app, llm_client: LLMClient | None = None) -> None:
        self.mcp_app = mcp_app
        self.llm = llm_client or LLMClient()

    async def analyze(
        self,
        cell_line: str,
        drug: str,
        target_genes: list[str] | None = None,
        *,
        run_logger=None,
        case_id: str | None = None,
    ) -> EvidencePack:
        self._run_logger = run_logger
        self._case_id = case_id
        evidence = await self._fetch_evidence(cell_line, drug, target_genes)
        prompt = self._build_prompt(cell_line, drug, evidence)
        raw = await self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system=self.system_prompt,
            run_logger=run_logger,
            agent_id=self.agent_id,
            case_id=case_id,
        )
        pack = self._parse_pack(raw, cell_line, drug)
        pack.confidence = compute_gcs(
            self._compute_h(pack.key_findings, evidence),
            evidence, pack.evidence_tier, self._compute_signal(evidence)
        )
        if pack.verdict == Verdict.UNCERTAIN:
            pack.confidence = min(pack.confidence, 0.5)
        pack = pack.model_copy(update={"data_status": evidence.get("data_status")})
        if run_logger:
            run_logger.log_agent_decision(
                case_id=case_id, agent_id=self.agent_id, round_num=1,
                verdict=pack.verdict.value, confidence=pack.confidence, reasoning=pack.reasoning,
            )
        return pack

    async def critique(
        self,
        original_pack: EvidencePack,
        peer_packs: list[EvidencePack],
        *,
        challenge=None,          # AxiomChallenge | None — imported lazily to avoid circular dep
        is_challenger: bool = False,      # True if this agent issued a challenge this round
        challenged_ids: set | None = None,  # IDs of peers this agent challenged (unused here; exclusion handled in engine)
        run_logger=None,
        case_id: str | None = None,
    ) -> tuple["EvidencePack", "bool | None"]:
        """Round-2 peer review with evidence-gated mechanistic scoring.

        Peer score = mechanistic relevance (0–1), not checklist completeness.
        Verdict revision requires citing a specific cross-modal finding.
        UNCERTAIN agents cannot flip to decisive without non-empty mechanistic reason.

        Returns (critiqued_pack, challenge_maintained):
          challenge_maintained — True  = successfully rebutted an AxiomChallenge
                                 False = conceded the challenge
                                 None  = no challenge received
        """
        labels = [chr(ord("A") + i) for i in range(len(peer_packs))]
        own_tier = original_pack.evidence_tier.value if hasattr(original_pack.evidence_tier, "value") else str(original_pack.evidence_tier)

        # Build cross-modal summary block
        cross_modal_lines = []
        for peer in peer_packs:
            tier_key = peer.evidence_tier.value if hasattr(peer.evidence_tier, "value") else str(peer.evidence_tier)
            if peer.key_findings:
                findings_str = "; ".join(
                    f"{f.biomarker}={f.value} ({f.interpretation})" for f in peer.key_findings
                )
            else:
                findings_str = "none reported"
            cross_modal_lines.append(f"  [{tier_key}] {peer.verdict.value} (conf={peer.confidence:.2f}): {findings_str}")
        cross_modal_block = "\n".join(cross_modal_lines)

        # Per-peer text for mechanistic relevance scoring
        peer_text = ""
        for label, peer in zip(labels, peer_packs):
            tier_key = peer.evidence_tier.value if hasattr(peer.evidence_tier, "value") else str(peer.evidence_tier)
            peer_text += (
                f"\n--- Peer {label} (tier: {tier_key}) ---\n"
                f"Verdict: {peer.verdict.value}  Confidence: {peer.confidence:.2f}\n"
                f"Reasoning: {peer.reasoning}\n"
                f"Key findings: {json.dumps([f.model_dump() for f in peer.key_findings])}\n"
            )

        peer_schema_parts = []
        for label in labels:
            peer_schema_parts.append(f'"peer_{label}_score": <float 0.0–1.0>')
        peer_schema = ", ".join(peer_schema_parts)

        # AxiomChallenge section (only when a challenge is received)
        challenge_section = ""
        challenge_schema = ""
        if challenge is not None:
            challenge_section = (
                f"\nAXIOM CHALLENGE from {challenge.challenger_tier} agent:\n"
                f"Their verdict: {challenge.challenger_verdict}\n"
                f"Their argument: {challenge.argument}\n\n"
                f"CHALLENGE RESPONSE REQUIRED (task 3 below).\n"
            )
            challenge_schema = (
                ', "challenge_rebuttal": "<specific mechanistic counter-argument citing your own evidence, or null if conceding>",'
                ' "challenge_maintained": <true = you rebutted with evidence, false = you concede>'
            )

        prompt = (
            f"Cell line: {original_pack.cell_line}\nDrug: {original_pack.drug}\n\n"
            f"YOUR ROUND-1 VERDICT: {original_pack.verdict.value}  (confidence: {original_pack.confidence:.2f})\n"
            f"Your tier: {own_tier}\n"
            f"Your reasoning: {original_pack.reasoning}\n"
            f"{challenge_section}\n"
            f"CROSS-MODAL FINDINGS FROM ALL SPECIALISTS:\n{cross_modal_block}\n\n"
            f"PEER ANALYSES TO REVIEW:\n{peer_text}\n"
            "TASKS:\n"
            "1. MECHANISTIC PEER SCORING: Score each peer 0.0–1.0 on ONE criterion only:\n"
            "   Does this peer\'s key finding have a DIRECT MECHANISTIC CONNECTION to why this drug\n"
            "   would or would not work in this specific cell line?\n"
            "   - 1.0 = names specific gene/value AND explains mechanistic link to drug action\n"
            "   - 0.5 = relevant evidence but indirect or population-level only\n"
            "   - 0.0 = generic, no mechanistic connection, or evidence is for a different drug/target\n\n"
            "2. VERDICT REVISION (strict gate — read carefully):\n"
            "   You MAY set verdict_revised=true ONLY IF all three conditions hold:\n"
            "   (a) A peer from a DIFFERENT tier cited a finding mechanistically INCONSISTENT with your conclusion.\n"
            "   (b) You can name the specific finding: [gene/value] from [tier] contradicts [your conclusion] because [mechanism].\n"
            "   (c) The contradiction is mechanistic, not just a different number or a different verdict.\n"
            "   You may NOT revise because: peers agree with each other, a peer scored well, you feel uncertain,\n"
            "   or no peers provided mechanistic evidence against your conclusion.\n"
            "   IMPORTANT - UNCERTAIN TO DECISIVE RULE: If your Round-1 verdict was UNCERTAIN, you may ONLY\n"
            "   become decisive if a peer cited a specific finding from their modality that recontextualises\n"
            "   your own data. You may NOT flip by reinterpreting data you already had in Round 1 --\n"
            "   if that data existed and you were UNCERTAIN, it was not sufficient then and nothing has\n"
            "   changed about your own evidence. A hemizygous deletion, borderline z-score, or partial\n"
            "   CNV that produced UNCERTAIN in Round 1 cannot be cited as the reason for a decisive\n"
            "   verdict in Round 2. Your revision_reason must name the PEER\'s finding (not your own)\n"
            "   and explain why it mechanistically recontextualises your modality\'s evidence.\n\n"
        )
        if challenge is not None:
            prompt += (
                "3. CHALLENGE RESPONSE: Address the AxiomChallenge above.\n"
                "   - If you have specific evidence from YOUR modality that mechanistically contradicts the challenger\'s argument,\n"
                "     set challenge_maintained=true and state the rebuttal.\n"
                "   - If you cannot provide a mechanistic counter-argument from your own evidence, set challenge_maintained=false.\n"
                "   - Peer scores do not count as a rebuttal. Only your own modality\'s evidence does.\n\n"
            )

        prompt += (
            "Respond with ONLY this JSON (no prose, no markdown):\n"
            f'{{"verdict": "<SENSITIVE|RESISTANT|UNCERTAIN>", '
            f'"confidence": <float>, '
            f'"verdict_revised": <true or false>, '
            f'"revision_reason": "<specific gene/value/mechanism that caused revision, or null>", '
            f'{peer_schema}'
            f'{challenge_schema}'
            f', "peer_review_reasoning": "<one sentence per peer on mechanistic relevance>"}}' 
        )

        raw = await self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system=self.system_prompt,
            run_logger=run_logger,
            agent_id=self.agent_id,
            case_id=case_id,
        )
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())
        data: dict = {}
        for candidate in [cleaned] + [m.group(0) for m in re.finditer(r"\{.*?\}", raw, re.DOTALL)]:
            try:
                data = json.loads(candidate)
                break
            except Exception:
                continue

        new_verdict_str = data.get("verdict", original_pack.verdict.value)
        try:
            new_verdict = Verdict(new_verdict_str)
        except Exception:
            new_verdict = original_pack.verdict
        verdict_revised = bool(data.get("verdict_revised", False)) and (new_verdict != original_pack.verdict)

        # Gate: UNCERTAIN → decisive requires a non-empty mechanistic revision_reason
        if original_pack.verdict == Verdict.UNCERTAIN and new_verdict != Verdict.UNCERTAIN:
            reason = (data.get("revision_reason") or "").strip()
            if not verdict_revised or not reason:
                new_verdict = original_pack.verdict
                verdict_revised = False

        new_conf = float(data.get("confidence", original_pack.confidence))
        new_conf = max(0.0, min(1.0, new_conf))
        if verdict_revised:
            new_conf = round(new_conf * 0.78, 4)
        if new_verdict == Verdict.UNCERTAIN:
            new_conf = min(new_conf, 0.5)

        peer_score_map = {}
        for label in labels:
            score = data.get(f"peer_{label}_score")
            peer_score_map[label] = float(score) if score is not None else 0.5

        critiqued = original_pack.model_copy(update={
            "verdict": new_verdict,
            "confidence": new_conf,
            "peer_scores": peer_score_map,
        })

        # Challenge outcome
        challenge_maintained: bool | None = None
        if challenge is not None:
            raw_maintained = data.get("challenge_maintained")
            rebuttal = (data.get("challenge_rebuttal") or "").strip()
            if raw_maintained is True and rebuttal:
                challenge_maintained = True
            else:
                # Concede: adopt challenger's verdict at reduced confidence
                challenge_maintained = False
                try:
                    conceded_verdict = Verdict(challenge.challenger_verdict)
                except Exception:
                    conceded_verdict = Verdict.UNCERTAIN
                conceded_conf = round(original_pack.confidence * 0.65, 4)
                if conceded_verdict == Verdict.UNCERTAIN:
                    conceded_conf = min(conceded_conf, 0.5)
                critiqued = critiqued.model_copy(update={
                    "verdict": conceded_verdict,
                    "confidence": conceded_conf,
                })

        if run_logger:
            rev_note = f" [REVISED from {original_pack.verdict.value}: {data.get('revision_reason', '')}]" if verdict_revised else ""
            ch_note = f" [CHALLENGE: {'maintained' if challenge_maintained else 'conceded'}]" if challenge is not None else ""
            run_logger.log_agent_decision(
                case_id=case_id, agent_id=self.agent_id, round_num=2,
                verdict=critiqued.verdict.value, confidence=critiqued.confidence,
                reasoning=str(data.get("peer_review_reasoning") or "") + rev_note + ch_note,
            )
        return critiqued, challenge_maintained


    @abstractmethod
    async def _fetch_evidence(self, cell_line: str, drug: str, target_genes: list[str] | None = None) -> dict:
        ...

    def _compute_h(self, key_findings: list, evidence: dict) -> float:
        from src.agents.gcs import compute_h
        return compute_h(key_findings, evidence)

    @abstractmethod
    def _compute_signal(self, evidence: dict) -> float:
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        ...

    def _build_prompt(self, cell_line: str, drug: str, evidence: dict) -> str:
        return (
            f"Cell line: {cell_line}\nDrug: {drug}\n\n"
            f"Evidence data:\n{json.dumps(_truncate_evidence(evidence), indent=2)}\n\n"
            f"Respond with ONLY a JSON object in this exact format (no prose, no markdown):\n{_SCHEMA_EXAMPLE}"
        )

    def _parse_pack(self, raw: str, cell_line: str = "unknown", drug: str = "unknown") -> EvidencePack:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())
        candidates = [cleaned] if cleaned else []
        for m in re.finditer(r"\{.*?\}", raw, re.DOTALL):
            candidates.append(m.group(0))
        for candidate in candidates:
            try:
                data = json.loads(candidate)
                pack = EvidencePack.model_validate(data)
                pack.reasoning = data.get("reasoning", "")
                if "self_attestation" in data:
                    pack.self_attestation = data["self_attestation"]
                elif pack.evidence_tier.value == "T3_PATHWAY":
                    # LLM may return individual boolean fields without the wrapper dict.
                    # Reconstruct score so the T3 gate in _check_consensus has something to work with.
                    sa = {
                        k: data.get(k)
                        for k in ("bypass_gene_expressed", "pathway_active", "mechanism_relevant", "bypass_distinct")
                    }
                    if any(v is not None for v in sa.values()):
                        pack.self_attestation = {**sa, "score": sum(1 for v in sa.values() if v is True)}
                return pack
            except Exception:
                continue
        return EvidencePack(
            agent_id=self.agent_id,
            cell_line=cell_line,
            drug=drug,
            verdict=Verdict.UNCERTAIN,
            confidence=0.0,
            evidence_tier=EvidenceTier.T5_STATISTICAL,
            key_findings=[],
            caveats=[f"parse_failed: {raw[:200]}"],
        )

    async def _call_tool(self, tool: str, args: dict) -> list | dict | None:
        async with Client(self.mcp_app) as client:
            result = await client.call_tool(tool, args)
        data = json.loads(result.data)
        run_logger = getattr(self, "_run_logger", None)
        if run_logger:
            run_logger.log_tool_call(
                case_id=getattr(self, "_case_id", None), agent_id=self.agent_id,
                tool=tool, args=args, result=data,
            )
        return data
