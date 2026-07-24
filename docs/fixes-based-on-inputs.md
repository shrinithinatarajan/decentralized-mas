# Fixes based on run inputs — diagnosis of the debate system

Analysis of the `rework-debate` branch against the observed symptoms (resolver mispredictions,
Round 1 always winning, pharmacology agent outperforming the system). All numbers below are
measured from `experiments/results/traces_full_set_*.jsonl` (n=93) and the source DBs, not
estimated. Reproduction commands are at the bottom.

---

## 0. Headline — the baseline comparison is wrong

The claim "pharmacology alone gets ~87%, which throws the whole argument over" compares T4 on
**its own decisive subset** against the system on **all cases**. Coverage-matched over the same
93 cases, with abstention counted as wrong:

| Policy | Correct | Accuracy |
|---|---|---|
| T4-only (pharmacology alone) | 48/93 | **52%** |
| Full debate system | 58/93 | **62%** |

T4 is decisive on 61/93 cases and gets 79% of those right. On the remaining 32 it is silent.
The system already converts part of that silence into correct answers.

**The bar for the thesis is 52%, not 87%.** The multi-agent argument is not dead — it is being
measured against the wrong number. The correct framing is a coverage/accuracy tradeoff: a
pharmacological prior is accurate but narrow; the system's contribution is extending decision
coverage toward 100% without giving back the prior's accuracy.

---

## 1. "All mispredictions come from the axiom resolver"

Half-true, and the mechanism is not what it looks like.

```
CONSENSUS_R1       58 cases   84% correct
CONSENSUS_R2        9 cases   33% correct
RESOLVER_TIEBREAK  26 cases   23% correct
```

The resolver's 23% decomposes into two different problems:

- **12 of its 26 cases output `UNCERTAIN`** — an automatic loss under the current scoring.
- On the 14 cases where it commits, it scores 6/14 = **43%**, i.e. roughly chance. It is *not*
  systematically inverted.

Critically, **T4 wins the resolver 0 times out of 26.** The resolver mostly fires when T4 has
*already abstained*, so it is not hijacking good pharmacology answers — it is being handed the
hard cases where the pharmacological prior is silent. Its inputs are near-chance there: the
pathway agent's own R1 verdicts on those same 14 cases score 7/14 = 50%.

**Conclusion:** the resolver is not primarily a tiebreak-logic bug. It is an evidence-quality
problem on the hard stratum. The 12 `UNCERTAIN` outputs, however, are free points.

Resolver winning agents: `transcriptomics_agent` 10, `pathway_agent` 10, `genomics_agent` 3,
plus 3 hallucinated pathway IDs (see §2).

---

## 2. Agents hallucinate their own `agent_id`, silently disabling critique

The traces contain seven distinct pathway identities:

```
pathway_agent, pathway_agent_001, pathway_agent_01, pathway_agent_kegg,
pathway_agent_sirt2, pathway_bypass_agent, signalling_pathway_agent
```

Cause — `src/agents/base_agent.py:267`:

```python
pack = EvidencePack.model_validate(data)   # data["agent_id"] comes from the LLM's JSON
```

`_SCHEMA_EXAMPLE` tells the model `"agent_id": "<your_agent_id e.g. genomics_agent>"`, and it
improvises. Downstream, `src/protocols/debate_engine.py:270`:

```python
agent = agent_map.get(pack.agent_id)
if agent is None:
    async def _passthrough(p=pack): return p   # no critique, empty peer_scores
```

So any case with a hallucinated ID **silently skips peer review entirely** — roughly 14% of
cases. It also corrupts endorsement bookkeeping, peer label alignment (A/B/C), and
`winning_agent` attribution.

**Fix:** stamp `agent_id` from the agent class after validation; never trust it from model
output. One line, outsized impact.

---

## 3. The axiom hierarchy is effectively dead code in the resolver

`src/protocols/axiom_resolver.py:66`:

```python
winner = max(candidates, key=lambda p: (
    (peer_endorsements or {}).get(p.agent_id, 0.0),   # ← primary sort key
    _effective_priority(p),                            # ← axiom tier, tiebreak only
    p.confidence,
))
```

Endorsement is a continuous float, so exact ties essentially never occur — meaning the
Hierarchy of Truth only ever breaks ties that don't happen. The novel contribution is switched
off in the code path where it matters most.

It is also biased. The peer checklists in `base_agent.py:92` are not equally satisfiable:

- **T4 checklist:** "Did this peer report a specific IC50 z-score value?" → trivially yes.
- **T1 checklist:** "Did this peer cite a CIViC/OncoKB entry?" → usually no, coverage is sparse.

So T4 systematically scores ~4/4 while T1 scores ~1/4, and endorsement-first makes the resolver
a T4 amplifier by construction.

Secondary issue — `_effective_priority` (`axiom_resolver.py:46`) explicitly demotes T1 below T4
for non-targeted drugs, compounding the same bias.

---

## 4. The T3 quality gate is applied in consensus but not in the resolver

`_check_consensus` (`debate_engine.py:112`) demotes a pathway agent whose self-attestation score
is < 3. `AxiomResolver.resolve()` applies **no such gate**. A pathway agent that explicitly
failed its own quality check is excluded from consensus but can still win the tiebreak.

This matters because pathway is the noisiest agent — 64% coverage at 54% accuracy — and it wins
13 of 26 resolver cases.

---

## 5. Label leakage in two case files

`drug_response.label` is derived by the same `z < -0.5 / z > 0.5` rule used to build the GDSC2
labels, and `get_ic50` returns `SELECT *` — so T4 is handed the answer key directly. The
pharmacology system prompt even instructs it to read that field.

| Source | T4's precomputed label == ground truth |
|---|---|
| `data/cases/cases.yaml` (GDSC2) | **40/40 = 100%** |
| `data/cases/cases_held_out.yaml` | **20/20 = 100%** |
| `cases_held_out_ctrp_1..5` | 80–86% |

The CTRP sets are clean — `src/data/etl_ctrp.py` was written specifically to avoid this, and it
works. The GDSC2-labelled files are not.

**Actions:**
- Exclude `cases.yaml` and `cases_held_out.yaml` from all reported evaluation.
- `trials/dataset.json` is contaminated (it sampled from both pools; 16/20 leaked) and must be
  rebuilt CTRP-only before the trials ladder is run.

---

## 6. Why only Round 1 ever wins

Two conditions combine:

1. **Single decisive agent wins R1.** `_check_consensus` uses `if n <= len(decisive) / 2: return None`.
   With one decisive agent, `1 <= 0.5` is False, so it returns that agent's verdict. The
   docstring says a single decisive agent should escalate to peer critique; the code disagrees.
2. **Critique only runs on explicit SENSITIVE-vs-RESISTANT clashes.** `has_conflict` requires
   both verdicts present among decisive agents (`debate_engine.py:182`).

Measured distribution of decisive agents per case: `{0: 12, 1: 32, 2: 31, 3: 18}`. **32 cases
have exactly one decisive agent** — usually T4 — and exit at R1 with no debate. Combined with 12
all-UNCERTAIN cases, roughly half of all cases can never reach the debate machinery.

Per-agent R1 coverage explains why:

| Agent | Coverage | Accuracy on decisive | Final verdict matches its R1 |
|---|---|---|---|
| pharmacology (T4) | 66% | 79% | 75% |
| pathway (T3) | 64% | 54% | 46% |
| genomics (T1) | **16%** | 67% | 26% |
| transcriptomics (T2) | **12%** | 73% | 18% |

T1 and T2 barely participate. The prompts push hard toward `UNCERTAIN` (wild-type → UNCERTAIN,
no CIViC context → UNCERTAIN, not in DB → UNCERTAIN). Those rules were presumably added to stop
hallucination, but they have made the two highest-authority tiers nearly inert.

---

## 7. Confidence is nearly flat with respect to evidence strength

`src/agents/gcs.py:56`:

```python
return round((h * 0.40) + (e * 0.30) + (t * 0.20) + (s * 0.10), 6)
```

`s` (signal strength) carries only 10% weight, and `e` is 1.0 whenever the MCP returned
anything. So a borderline case (z = 0.51, essentially a coin flip across labs) and a decisive
one (z = 3.0) differ by roughly 0.03 in reported confidence.

Confidence is therefore a *retrieval-quality* score, not a calibrated probability of
correctness. This is the main reason AUROC looks weak — the ranking signal has been flattened —
and it also makes confidence a poor tiebreak inside the resolver.

---

## 8. On the ground-truth labels

The concern that deriving labels from z-scores was "dumb" is misplaced — it is standard practice
in this literature, and the cross-lab setup is a **strength**, not a flaw:

- Labels: CTRPv2 (Broad), derived by per-drug quantile.
- T4's data: GDSC2 (Sanger), z-scored per drug.
- The two labs agree on ~56 of 91 cases (~62%), consistent with the published GDSC/CCLE
  concordance literature (agreement is good for strong effects, poor for weak ones).

This is external validation, not leakage — provided the GDSC2-labelled case files from §5 are
excluded.

The instinct about the remaining discordant cases is correct and important: **restricting
evaluation to the concordant subset would produce a fake high accuracy.** The discordant cases
are where a pharmacological prior is structurally unable to win, so they are exactly where
mechanistic multi-agent reasoning has to earn its keep. Keep them in and stratify the reporting.

---

## The conceptual fix: conditional axiom precedence

The Hierarchy of Truth is biologically motivated but empirically inverted on this data —
pharmacology (79%) outperforms pathway (54%), yet T3 outranks T4.

Rather than abandon the hierarchy, make **axiom precedence conditional on evidential
sufficiency**: a tier exercises override authority only when its evidence clears a quality bar.
A CIViC-backed driver mutation *should* override an IC50 prior. "KEGG lists a bypass gene"
should not override a direct dose-response measurement.

This preserves the core contribution, explains every observed failure, and is defensible in the
write-up as a principled refinement rather than a retreat. The machinery for it already exists
(T3 self-attestation) — it is simply not applied where the decisions are being made.

---

## Approaches, ranked by leverage

| # | Approach | Est. gain | Cost |
|---|---|---|---|
| 1 | **Fix the evaluation frame** — coverage-matched baselines, risk-coverage / AURC, drop the leaked case files, stratify by T4-decisive vs T4-abstain | 0 pts, but reframes the result as 62% vs 52% | none |
| 2 | **Stamp `agent_id` from the class** — restores critique on ~14% of cases | structural | 1 line |
| 3 | **Eliminate the 12 `UNCERTAIN` resolver outputs** — force a calibrated commit | ~+6 pts | small |
| 4 | **Gate tier authority by evidence quality** — apply the T3 self-attestation gate inside the resolver; make axiom priority the primary sort key and endorsement a bounded tiebreak | ~+3–8 pts | moderate |
| 5 | **Raise T1/T2 coverage** (16% / 12%) — targeted re-query round: abstaining agents see peers' findings and re-interrogate their own MCP server with a specific question | largest upside on the hard stratum | high, most novel |
| 6 | **Calibrate confidence to evidence strength** — raise the weight of `s`, scale T4 confidence with \|z\|, penalise the \|z\| ∈ [0.5, 1.0] band where cross-lab labels flip | AUROC, not accuracy | moderate |
| 7 | **Withhold the precomputed `label` from T4** — force reasoning rather than echo; report both ways as a documented ablation | integrity | small |

Approach 5 is what makes this a genuine multi-agent thesis rather than "T4 with extra steps": on
the 32 cases where T4 is silent, mechanism is the only thing that can win, and T1/T2 currently
abstain on ~85% of everything.

**Suggested order:** 1 + 2 + 3 first (cheap; #1 alone changes the story from "we lose to 87%" to
"we beat 52%"), then choose between 4 and 5 depending on remaining API budget and time.

**Realistic target:** lifting the 26 resolver cases from 23% to even ~55% puts the system near
70–75%, against a 52% single-agent baseline. That is a strong, honest thesis result.

---

## Reproducing these numbers

```bash
# Resolution-method breakdown, per-agent coverage/accuracy, baseline comparison
python - <<'EOF'
import json, glob
from collections import Counter, defaultdict
recs=[]
for f in sorted(glob.glob("experiments/results/traces_full_set_*.jsonl")):
    for line in open(f):
        if line.strip(): recs.append(json.loads(line))
acc=defaultdict(lambda:[0,0])
for r in recs:
    m=r.get("resolution_method"); acc[m][0]+= r["final_verdict"]==r["true_label"]; acc[m][1]+=1
for m,(c,n) in acc.items(): print(f"{m:22} {c}/{n} = {c/n:.0%}")
t4c=t4d=0
for r in recs:
    for a in r.get("r1_agents",[]):
        if a["agent_id"]=="pharmacology_agent" and a["verdict"]!="UNCERTAIN":
            t4d+=1; t4c+= a["verdict"]==r["true_label"]
sysc=sum(r["final_verdict"]==r["true_label"] for r in recs)
print(f"T4-only: {t4c}/{len(recs)}={t4c/len(recs):.0%}   system: {sysc}/{len(recs)}={sysc/len(recs):.0%}")
EOF

# Label-leakage check per case file
python - <<'EOF'
import sqlite3, yaml
from pathlib import Path
ph = sqlite3.connect("src/data/processed/pharmacology.db")
for name, path, flat in [("cases.yaml","data/cases/cases.yaml",False),
                         ("held_out","data/cases/cases_held_out.yaml",False),
                         ("ctrp_1","data/cases/cases_held_out_ctrp_1.yaml",True)]:
    d=yaml.safe_load(Path(path).read_text()); entries = d if flat else d["cases"]
    n=agree=dec=0
    for e in entries:
        r=ph.execute("SELECT label FROM drug_response WHERE cell_line=? AND drug=?",
                     (e["cell_line"],e["drug"])).fetchone()
        n+=1
        if r and r[0]!="UNCERTAIN":
            dec+=1; agree+= r[0]==e["label"]
    print(f"{name:12} n={n} decisive={dec} T4label==GT={agree}")
EOF
```
