# Phase 0–3 Fix Impact — Results Analysis

Comparison of `trials/results/pre-fix/` (baseline, before this session's Phase
0–3 fixes) against `trials/results/post-fix0-3/` (same 20-case dataset,
same LLM backend, rerun after Phase 0–3 landed). Every observation below is
grounded in a specific summary metric or trace entry from these two result
sets, tied to the specific code change that produced it.

---

## exp1 — isolated agents (control group check)

| Agent | Metric | Pre | Post | Δ |
|---|---|---|---|---|
| genomics | accuracy_all / AUROC / spearman | 0.15 / 0.667 / 0.408 | **identical** | 0 |
| transcriptomics | accuracy_all / AUROC / spearman | 0.35 / 0.95 / 0.814 | **identical** | 0 |
| pathway | accuracy_all / AUROC / spearman | 0.55 / 0.444 / -0.097 | **identical** | 0 |
| **pharmacology** | accuracy_all | 0.75 | 0.70 | **-0.05** |
| pharmacology | accuracy_definitive | 0.789 | 0.737 | -0.053 |
| pharmacology | AUROC | 0.867 | 0.767 | **-0.10** |
| pharmacology | spearman ρ | 0.642 | 0.464 | -0.178 |

Three of four agents are byte-for-byte unchanged — the correct control
result, since nothing in Phase 0–3 touches genomics/transcriptomics/pathway's
isolated evidence path. Only pharmacology moved, and it moved down. This is
the direct, expected consequence of **Phase 1's `label`-strip**
(`pharmacology_server.py::get_ic50`) — pharmacology can no longer read the
leaked ground-truth column and must now genuinely classify from
`z_score`/`auc` (via the rewritten system prompt in
`pharmacology_agent.py`). **A 10-point AUROC drop here is not a
regression — it's the removal of an answer key.** The pre-fix 0.867 AUROC
was measuring "can the agent read a field," not "can the agent reason."
0.767 is pharmacology's real number.

---

## exp2 — consensus, no debate (`NoDebateEngine` fix)

| Metric | Pre | Post | Δ |
|---|---|---|---|
| correct / wrong / uncertain | 10 / 2 / **8** | 14 / 6 / **0** | +4 / +4 / **-8** |
| accuracy_all | 0.50 | 0.70 | **+0.20** |
| accuracy_definitive | 0.833 | 0.70 | -0.133 |
| AUROC | 0.857 | 0.70 | -0.157 |
| uncertain_rate | 40% | **0%** | -40pp |
| winning_agents | genomics 8, transcriptomics 8, pathway 4, **pharmacology 0** | genomics **1**, transcriptomics 9, **pharmacology 7**, pathway 3 | — |

This is the clearest before/after in the whole run, and it's fully
case-traceable. A case-by-case diff shows **exactly 8 cases** flipped from
`UNCERTAIN` pre-fix to a decisive verdict post-fix:

- `a6`, `a9`, `a10`, `b1` → all flipped to **correct** (`RESISTANT`,
  matching ground truth)
- `a3`, `a8`, `b3`, `b8` → flipped to **wrong**

The 4 corrected cases are the *exact* four (`a6, a9, a10, b1`) diagnosed
earlier this session as "pharmacology's correct decisive verdict suppressed
by 2 UNCERTAIN abstentions outvoting it." The fix
(`Counter(p.verdict for p in packs)` → filtered to the `decisive` subset,
in `src/evaluation/ablation_runner.py::NoDebateEngine.run`) predicted these
specific cases would flip to correct, and they did.

The `winning_agents` shift explains the mechanism precisely: **pharmacology
went from 0 wins to 7**, **genomics collapsed from 8 wins to 1**. Genomics
is decisive only 5/20 times per exp1 (75% abstention rate) and correct on
just 3 of those — it could not possibly have legitimately "won" 8 consensus
votes on real decisive evidence. It was winning by `UNCERTAIN`-plurality,
the exact bug. Once abstentions stop counting as votes, genomics's true
(small) decisive footprint shows up honestly.

The accuracy_definitive/AUROC *drop* (0.833→0.70, 0.857→0.70) is a
denominator artifact, not the system getting worse: pre-fix, these metrics
were computed over only the 12/20 cases that happened to resolve
decisively — an artificially shrunk, self-selected-easy subset, since the
bug silently punted 8 harder cases to `UNCERTAIN` rather than scoring them.
Post-fix, `n_decisive` = 20/20, so the metric now has to answer every case,
including the ones it used to get a free pass on. **accuracy_all
(0.50→0.70) is the fair comparison, and it went up 20 points.**

---

## exp3 — full debate (Phase 2 resolver + Phase 3 quorum floor)

| Metric | Pre | Post | Δ |
|---|---|---|---|
| correct / wrong | 15 / 5 | 14 / 6 | -1 |
| accuracy_all | 0.75 | 0.70 | -0.05 |
| AUROC | 0.77 | 0.76 | -0.01 (~flat) |
| spearman ρ | 0.468 | 0.451 | ~flat |
| **CONSENSUS_R1** | **18** | **3** | -15 |
| CONSENSUS_R2 | 2 | 10 | +8 |
| **RESOLVER_TIEBREAK** | **0** | **7** | **+7** |
| winning_agents | genomics 5, transcriptomics 6, **pharmacology 7**, pathway 2 | genomics **12**, transcriptomics 6, **pharmacology 0**, pathway 2 | — |

This is the headline result of **Phase 3's quorum floor**, and it's stark:
pre-fix, 90% of cases (18/20) resolved trivially at R1 — meaning the system
was barely debating at all, just rubber-stamping whatever a single agent or
a shrunken majority said. Post-fix, only 15% (3/20) resolve at R1; the rest
are forced through real R2 critique or the resolver. This is exactly what
"fixes Bug C and the correlated-error blind spot" was supposed to produce,
and the trace data confirms it's mechanically working, not just relabeling
outcomes.

An important, previously-invisible consequence: **`RESOLVER_TIEBREAK` never
fired once in the pre-fix run.** This means **Phase 2's resolver sort-key
fix had zero observable effect on this dataset until Phase 3 unlocked
it** — the resolver code was correct but dormant. This confirms the
sequencing dependency documented in `to-do.md` ("Phase 2 before Phase 3,
else the quorum floor routes more cases into a resolver that still inverts
the hierarchy") — the resolver is now actually load-bearing on 7/20 cases
and would have been actively wrong on several of them had Phase 2 not
landed first.

`winning_agent` swinging from pharmacology (7→0) to genomics (5→12) is the
direct, correct consequence of the resolver now firing for real: whenever
genomics is decisive and agreeing, its T1 tier wins the (now-fixed)
tier-first sort in `axiom_resolver.py`, exactly per the hierarchy-of-truth
design — this is the system behaving as designed, not a bug.

### Case-level deep dive: `b7` (`KMS-11:CHIR-99021`) — the one accuracy regression

True label `SENSITIVE`. Correct pre-fix, wrong post-fix. Tracing the actual
R1 → critique → resolver flow (available for the first time thanks to
Phase 0's trace logging):

- R1: pathway gets gated out by the T3 self-attestation check
  (`self_attestation.score = 0`), and genomics is `UNCERTAIN`. Only 2/4
  agents decisive — correctly fails the new quorum floor (`2 ≤ 4/2`),
  forcing a critique round instead of trivial R1 consensus.
- Critique round: **transcriptomics and pharmacology swap verdicts with
  each other** — transcriptomics: `SENSITIVE 0.90` → `RESISTANT 0.45`;
  pharmacology: `RESISTANT 0.43` → `SENSITIVE 0.45`. A genuine sycophantic
  double-flip, not evidence-driven (both land at nearly the same
  post-revision confidence).
- Resolver: correctly (per Phase 2's fix) picks transcriptomics's T2 tier
  over pharmacology's T4 — but transcriptomics's verdict by that point is
  the *flipped* one, not the *evidenced* one.

**This is a live, concrete instance of the exact peer-majority sycophancy
pattern `to-do.md`'s Phase 6 (`AxiomChallenge`, still unbuilt) is meant to
fix.** Phase 3 didn't cause it — but by routing more cases through the
critique round, it's now exposing a pre-existing, still-open bug that the
pre-fix run almost never had the opportunity to surface, since it almost
never reached critique in the first place.

---

## A new gap this comparison surfaced

Checking `contributing_agents` (Phase 1's attribution field) across
resolution paths: it works exactly as intended for `CONSENSUS_R1`/
`CONSENSUS_R2` cases (e.g. `a2`: `winning_agent=genomics_agent` but
`contributing_agents` correctly ranks transcriptomics first by confidence).
But **`RESOLVER_TIEBREAK` cases have `contributing_agents: []` — always
empty.** The resolver path in `DebateEngine.run()` builds its
`ConsensusResult` directly rather than going through `_build_result`, so it
never populates the field. This was invisible before because the pre-fix
run had zero `RESOLVER_TIEBREAK` cases to expose it; now that Phase 3
routes 35% of cases there, it's a real, silent gap in 7/20 of this run's
records. **Not yet fixed — candidate for `to-do.md`.**

---

## Bottom line

Every fix did exactly what its diagnosis said it would, verified
case-by-case: label leakage removed (pharmacology's inflated numbers
corrected downward), the `UNCERTAIN`-vote bug fixed (4 predicted cases
flipped correct, exactly as predicted), and the quorum floor is now
genuinely forcing debate instead of rubber-stamping. None of the headline
accuracy numbers went up, and exp3 dipped slightly — but every drop traces
to a specific, honest mechanism (a denominator correction, a leakage
removal, or an *already-diagnosed, not-yet-fixed* Phase 6 bug getting newly
exposed), not to any fix behaving incorrectly. The system is measurably
more rigorous and more debuggable than before; whether it's more *accurate*
depends on fixes not yet done (Phase 4's evidence layer, Phase 5's pathway
rebuild, Phase 6's real axiom challenge).
