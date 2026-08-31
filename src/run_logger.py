"""Per-run structured logging: one JSONL log file per run_id, shared by all agents.

Every agent message, tool call, LLM call, and debate decision for a run is
appended as a single JSON line to artifacts/run_<run_id>.log, tagged with
run_id and case_id so a single run/test-case can be reconstructed from the file.
"""
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Project-root artifacts/ dir (this file lives at <root>/src/run_logger.py).
DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "artifacts"


class RunLogger:
    def __init__(self, run_id: str | None = None, log_dir: Path | None = None) -> None:
        self.run_id = run_id or str(uuid.uuid4())
        log_dir = log_dir or DEFAULT_LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"run_{self.run_id}.log"

        self._logger = logging.getLogger(f"mas.run.{self.run_id}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        if not self._logger.handlers:
            handler = logging.FileHandler(self.log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)

    def _emit(self, event: str, **fields) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": event,
            **fields,
        }
        self._logger.info(json.dumps(record, default=str))

    def log_run_start(self, n_cases: int) -> None:
        self._emit("run_start", n_cases=n_cases)

    def log_run_end(self, n_cases: int) -> None:
        self._emit("run_end", n_cases=n_cases)

    def log_case_start(self, case_id: str, cell_line: str, drug: str) -> None:
        self._emit("case_start", case_id=case_id, cell_line=cell_line, drug=drug)

    def log_case_end(self, case_id: str, result: dict) -> None:
        self._emit("case_end", case_id=case_id, **result)

    def log_tool_call(self, case_id: str, agent_id: str, tool: str, args: dict, result) -> None:
        self._emit("tool_call", case_id=case_id, agent_id=agent_id, tool=tool, args=args, result=result)

    def log_llm_call(
        self,
        case_id: str,
        agent_id: str,
        model: str,
        messages: list[dict],
        system: str,
        response: str,
        cached: bool,
        latency_s: float,
        error: str | None = None,
    ) -> None:
        self._emit(
            "llm_call",
            case_id=case_id,
            agent_id=agent_id,
            model=model,
            n_messages=len(messages),
            response_len=len(response),
            cached=cached,
            latency_s=round(latency_s, 4),
            error=error,
        )

    def log_agent_decision(
        self, case_id: str, agent_id: str, round_num: int, verdict: str, confidence: float, reasoning: str
    ) -> None:
        self._emit(
            "agent_decision",
            case_id=case_id,
            agent_id=agent_id,
            round=round_num,
            verdict=verdict,
            confidence=round(confidence, 4),
            reasoning=reasoning,
        )

    def log_debate_round(self, case_id: str, round_num: int, note: str, snapshot: dict) -> None:
        self._emit("debate_round", case_id=case_id, round=round_num, note=note, snapshot=snapshot)

    def log_resolution(
        self, case_id: str, resolution_method: str, winning_agent: str, verdict: str, forced: bool
    ) -> None:
        self._emit(
            "resolution",
            case_id=case_id,
            resolution_method=resolution_method,
            winning_agent=winning_agent,
            verdict=verdict,
            forced=forced,
        )

    def close(self) -> None:
        for handler in list(self._logger.handlers):
            handler.close()
            self._logger.removeHandler(handler)


