import os
import sqlite3
import yaml
from dataclasses import dataclass
from pathlib import Path

VALID_LABELS = {"SENSITIVE", "RESISTANT", "UNCERTAIN"}


@dataclass
class Case:
    cell_line: str
    drug: str
    label: str
    notes: str = ""
    z_score: float | None = None
    pathway: str = ""
    putative_target: str = ""
    axiom_tier: str = ""
    split: str = "test"   # "dev" (tuning) or "test" (held-out evaluation)

    def __post_init__(self):
        if self.label not in VALID_LABELS:
            raise ValueError(f"Invalid label {self.label!r}; must be one of {VALID_LABELS}")


def load_cases(path: Path | None = None) -> list[Case]:
    if path is None:
        path = Path(os.getenv("CASES_FILE", "data/cases/cases.yaml"))
    with open(path) as f:
        data = yaml.safe_load(f)
    if "cases" not in data:
        raise KeyError(f"Expected top-level 'cases' key in {path}; got keys: {list(data.keys())}")
    cases = []
    for i, entry in enumerate(data["cases"]):
        try:
            cases.append(Case(**entry))
        except TypeError as e:
            raise TypeError(f"Entry {i} in {path} has unexpected fields: {e}") from e
    return cases


def load_ctrp_cases(path: Path) -> list[Case]:
    """Load flat-list CTRPv2 yaml (no 'cases:' wrapper)."""
    with open(path) as f:
        entries = yaml.safe_load(f)
    return [Case(cell_line=e["cell_line"], drug=e["drug"], label=e["label"]) for e in entries]


def get_db_connection(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn
