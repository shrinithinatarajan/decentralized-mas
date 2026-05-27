import pytest
import yaml
from pathlib import Path
from src.data.loader import load_cases, get_db_connection, Case


def test_load_cases(tmp_path):
    cases_file = tmp_path / "cases.yaml"
    cases_file.write_text("""
cases:
  - cell_line: A375
    drug: Vemurafenib
    label: SENSITIVE
    notes: "test"
""")
    cases = load_cases(cases_file)
    assert len(cases) == 1
    assert cases[0].cell_line == "A375"
    assert cases[0].label == "SENSITIVE"


def test_get_db_connection(genomics_db):
    conn = get_db_connection(genomics_db)
    row = conn.execute("SELECT * FROM mutations WHERE cell_line='A375'").fetchone()
    assert row is not None
    conn.close()


def test_load_cases_missing_notes(tmp_path):
    cases_file = tmp_path / "cases.yaml"
    cases_file.write_text("""
cases:
  - cell_line: MCF7
    drug: Fulvestrant
    label: SENSITIVE
""")
    cases = load_cases(cases_file)
    assert cases[0].notes == ""


def test_load_cases_invalid_label(tmp_path):
    cases_file = tmp_path / "cases.yaml"
    cases_file.write_text("""
cases:
  - cell_line: A375
    drug: Vemurafenib
    label: SENSITVE
""")
    with pytest.raises(ValueError, match="Invalid label"):
        load_cases(cases_file)
