from src.schemas.evidence_pack import EvidencePack


class ConflictDetector:
    def has_conflict(self, packs: list[EvidencePack]) -> bool:
        return len(self.conflicts(packs)) > 0

    def conflicts(self, packs: list[EvidencePack]) -> list[dict]:
        if len(packs) <= 1:
            return []
        verdicts = {p.verdict for p in packs}
        if len(verdicts) == 1:
            return []
        # majority verdict
        from collections import Counter
        counts = Counter(p.verdict for p in packs)
        majority = counts.most_common(1)[0][0]
        return [
            {"agent_id": p.agent_id, "verdict": p.verdict, "evidence_tier": p.evidence_tier}
            for p in packs
            if p.verdict != majority
        ]
