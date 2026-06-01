import hashlib
import json
import sqlite3
from pathlib import Path


class ResponseCache:
    def __init__(self, db_path: Path) -> None:
        self._db = db_path
        self._init()

    def _init(self) -> None:
        conn = sqlite3.connect(self._db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                response  TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def _key(self, model: str, messages: list[dict], system: str = "") -> str:
        payload = json.dumps({"model": model, "system": system, "messages": messages}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, model: str, messages: list[dict], system: str = "") -> str | None:
        conn = sqlite3.connect(self._db)
        row = conn.execute(
            "SELECT response FROM llm_cache WHERE cache_key = ?",
            (self._key(model, messages, system),),
        ).fetchone()
        conn.close()
        return row[0] if row else None

    def set(self, model: str, messages: list[dict], response: str, system: str = "") -> None:
        conn = sqlite3.connect(self._db)
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache (cache_key, response) VALUES (?, ?)",
            (self._key(model, messages, system), response),
        )
        conn.commit()
        conn.close()
