"""Print a side-by-side comparison of Experiments 1, 2, 3 once all have been run.

Usage:
    PYTHONPATH=. python trials/compare.py
"""
import json

from trials.common import RESULTS


def main():
    exp1 = json.loads((RESULTS / "exp1_summary.json").read_text())
    exp2 = json.loads((RESULTS / "exp2_summary.json").read_text())["summary"]
    exp3 = json.loads((RESULTS / "exp3_summary.json").read_text())["summary"]

    best_agent_id, best_agent = max(exp1.items(), key=lambda kv: kv[1]["accuracy_definitive"])

    rows = [
        (f"Best isolated agent ({best_agent_id})", best_agent),
        ("Consensus, no debate (majority vote)", exp2),
        ("Full debate (axiom resolver)", exp3),
    ]

    print(f"\n{'=' * 78}")
    print(f"  {'Stage':<38}{'Acc(def)':>10}{'Uncertain':>11}{'AUROC':>9}{'rho':>9}")
    print(f"{'=' * 78}")
    for name, s in rows:
        print(f"  {name:<38}{s['accuracy_definitive']:>9.1%} {s['uncertain_rate']:>10.1%} {s['auroc']:>9.3f}{s['spearman_rho']:>9.3f}")
    print(f"{'=' * 78}")

    print("\n  All isolated agents:")
    for agent_id, s in sorted(exp1.items(), key=lambda kv: kv[1]["accuracy_definitive"], reverse=True):
        print(f"    {agent_id:<24}{s['accuracy_definitive']:>9.1%} {s['uncertain_rate']:>10.1%} {s['auroc']:>9.3f}{s['spearman_rho']:>9.3f}")


if __name__ == "__main__":
    main()
