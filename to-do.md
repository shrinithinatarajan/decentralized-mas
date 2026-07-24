# To-Do — sequenced by dependency, not by category

Generated from a brutal-review pass over the debate engine, resolver, ablation
baselines, and evidence data sources. Ordering matters: several fixes make
results *worse* if applied out of sequence (noted inline). Do not skip the
sequencing — see rationale in each phase.

---

## Phase 0 — zero-dependency, do immediately (hygiene + cheap bug fixes)

- [x] **Fix `EvaluationMetrics` / test drift.** Dataclass now has
  `n_total`/`n_decisive`/`coverage`; tests still pass `n_evaluated`. 23/174
  tests currently fail. Blocks trustworthy CI signal for every other fix
  below — do this first so regressions from later fixes are detectable.
  Done: updated `tests/test_metrics.py` and `tests/test_visualize.py` to the
  current field names/plot shape (production code was already correct — this
  was pure test-side drift). Also fixed two more masked `test_visualize.py`
  assertions (bar count, ylabel) that a `TypeError` had been hiding.
- [x] **Fix `NoDebateEngine.run()` UNCERTAIN-as-vote bug**
  (`src/evaluation/ablation_runner.py:20-27`). `Counter(p.verdict for p in
  packs)` counts UNCERTAIN as a votable verdict. Reuse `_check_consensus`'s
  decisive-only filter. Confirmed this session: suppressed 4 previously-correct
  pharmacology verdicts (cases a6, a9, a10, b1) by letting 2 abstentions
  outvote 1 correct decisive answer. Every exp2-style ablation number
  generated before this fix is invalid.
  Done: `NoDebateEngine.run()` now filters to decisive packs before voting;
  falls back to UNCERTAIN only when zero agents are decisive. Verified with
  RED tests reproducing the exact 2-abstain/1-decisive scenario.
- [x] **Fix `mutation`/`protein_change` silent fallback**
  (`genomics_server.py::get_mutations`) — `mut.get("protein_change") or
  mut.get("mutation", "")` silently masks a schema mismatch (no such column).
  Timeboxed audit: grep all MCP servers for the same `.get(x) or .get(y)`
  pattern; fix what's found, don't open-end this into a full rewrite.
  Done: audit found only this one instance across all MCP servers. Confirmed
  via `PRAGMA table_info(mutations)` that production `genomics.db` has no
  `protein_change` column at all — the fallback was silently always firing.
  Removed the dead fallback; also fixed `tests/conftest.py`'s `genomics_db`
  fixture, which had fabricated a `protein_change` column that doesn't exist
  in production (that drift is exactly why this bug went undetected).
- [x] **Log `reasoning`, `self_attestation`, `data_status` in
  `trials/*.jsonl` outputs**, not just verdict/confidence/tier. Needed to
  debug non-determinism (case a8 flipped outcome between exp1 isolated run
  and exp3 debate run with no way to tell why) and is a hard dependency for
  the reasoning-scorer work in Phase 4.
  Done: `debate_engine.py`'s snapshot helpers (`_snapshot`, and the extracted
  module-level `agent_snapshot()`) now include `self_attestation` alongside
  the already-present `reasoning`/`data_status`. `NoDebateEngine` was
  silently dropping `r1_agents` entirely (defaulted to `[]`) — now populates
  it via the same shared `agent_snapshot()` helper. `exp1/exp2/exp3` scripts
  updated to write these fields (exp1: per-agent pack fields directly; exp2/
  exp3: `result.r1_agents` and, for exp3, `result.trace`).

## Phase 1 — must land before any large re-run (invalidates current results)

- [ ] **Strip `label` from every MCP tool response server-side**
  (`pharmacology_server.py::get_ic50`/`get_drug_response` — currently
  `SELECT *`). 16/20 cases in `trials/dataset.json` have T4's own
  precomputed `drug_response.label` equal to ground truth. Pharmacology's
  78.9%/75.0% (exp1) is not yet trustworthy evidence of real reasoning.
  **Do this before validating any fix that depends on a trustworthy T4
  baseline (Phase 3, Phase 5).**
- [ ] **Separate verdict attribution from tier-priority display**
  (`debate_engine.py::_build_result`, line ~307). `winning_agent` is
  currently picked by `(AXIOM_HIERARCHY[tier], confidence)` — tier-first,
  so any agreeing T1 agent gets credited as "winner" regardless of whether
  it contributed the deciding signal (case a1: pharmacology and pathway did
  the epistemic work at higher confidence; genomics got the credit). Add a
  `contributing_agents` field based on actual decisiveness/confidence.
  **Hard dependency for Phase 4's reasoning-scorer** — building it against
  the current biased attribution field bakes the bias into every hallucination/
  faithfulness number it produces.

## Phase 2 — resolver correctness (blocks Phase 3)

- [x] **Fix `AxiomResolver.resolve()`'s sort key**
  (`axiom_resolver.py`). Currently:
  `winner = max(candidates, key=lambda p: (peer_endorsements.get(p.agent_id,
  0.0), _effective_priority(p), p.confidence))` — peer endorsement is the
  PRIMARY sort key, axiom tier is secondary. This structurally cannot
  express "T1 outranks T4," which defeats the entire hierarchy-of-truth
  premise whenever the resolver fires. **This must be fixed before Phase 3's
  quorum floor**, because raising the quorum floor routes MORE cases into
  this resolver — applying the quorum floor first makes results worse, not
  better, by increasing exposure to a resolver that still inverts the
  hierarchy.
  Done: swapped the key to `(_effective_priority(p), peer_endorsements.get(...),
  p.confidence)` — axiom tier now primary, peer endorsement only breaks ties
  within the same tier. New test `test_resolver_axiom_tier_outranks_peer_endorsement`
  reproduces the exact documented failure mode (T4 with high peer endorsement vs
  T1 with low) and only passes after the reorder.
  **Important correction to the original diagnosis**: the 9 tests that were
  failing at the start of this work (`test_protocols.py` x5,
  `test_debate_engine.py` x4) were NOT actually failing because of this
  sort-key bug — `peer_endorsements` is `None`/empty in all of them, so the
  old primary key tied at 0.0 for every agent and fell through to tier
  priority anyway. Root causes were two unrelated, pre-existing test/fixture
  bugs: (1) the shared test fixtures hardcoded `drug="Vemurafenib"`, and
  `is_targeted("Vemurafenib")` returns `False` — Vemurafenib is missing from
  the Phase 4 allowlist — which silently demoted T1's effective priority
  below T4; (2) three `test_debate_engine.py` cases didn't set
  `self_attestation` on their `pathway_agent` packs, so the (unrelated) T3
  self-attestation gate in `_check_consensus` silently dropped pathway from
  the decisive set, turning an intended 2-2 split into a 2-1 majority that
  resolved before the resolver ever ran. Fixed by swapping the fixtures'
  drug to `Dabrafenib` (already allow-listed) and adding
  `self_attestation={"score": 4}` to the affected packs — this keeps
  `is_targeted()`'s allowlist untouched (still Phase 4 scope) while letting
  these tests actually exercise what they claim to test.

## Phase 3 — consensus mechanics (depends on Phase 2)

- [ ] **Add a decisive-quorum floor to `_check_consensus`**
  (`debate_engine.py:98-130`). Require ≥2 decisive agents before R1/R2 can
  resolve by majority; below that, always route to the resolver with an
  explicit low-confidence flag. Fixes Bug C (single decisive agent
  auto-wins trivially since `n=1 > len(decisive)/2=0.5`) and the correlated-
  error blind spot (case a7: pharmacology + pathway both wrong and
  unanimous among just the two decisive agents, genomics/transcriptomics
  silently abstained, no mechanism caught the correlated error).
  **Do not apply before Phase 2 is complete.**

## Phase 4 — evidence layer (T1), do the cheap pilot before the full rebuild

- [ ] **Cheap pilot: prompt-only tweak making DepMap Chronos co-primary**
  in the genomics agent prompt (currently CIViC-gated primary, Chronos
  capped at confidence 0.55–0.65). No server-side change — run this first
  as a fast test of the hypothesis before committing to the full re-anchor
  below.
- [ ] **Full re-anchor: DepMap Chronos essentiality as T1's primary signal,
  CIViC demoted to confirmatory overlay.** 21.5M usable Chronos rows vs
  3,032 usable CIViC rows (0.11% coverage on mutations). Expected to sharply
  cut genomics's current 75% abstention rate.
  **Validate on a held-out calibration slice before trusting inside the live
  debate loop** — essentiality (a gene is required for viability) is not the
  same claim as sensitivity (a specific drug against that gene will work).
  Making genomics both more decisive (this fix) and more correctly
  prioritized (next fix) at the same time risks turning it from silently
  useless into loudly wrong if Chronos turns out to be a weaker sensitivity
  proxy than assumed. Do not skip the calibration check.
- [ ] **Replace `is_targeted()`'s hardcoded ~150-drug allowlist**
  (`drug_mechanism.py`) with a rule derived from `drug_info.target_genes`
  count / MOA string. Currently misses 38% of CTRP eval drugs (20/53),
  incorrectly demoting T1 priority for them. Validate the new rule against
  the current hand-curated list to confirm no regression on already-correct
  classifications. Pair with the Chronos re-anchor above — both increase
  T1's role in resolution, validate together, not in isolation.

## Phase 5 — pathway data (biggest lift, timebox or defer)

- [ ] **Rebuild `bypass_routes` with a real definition**, or explicitly
  retire T3 as a voting tier until rebuilt. Current table: 239,327/239,327
  rows have `bypass_exists=1`, zero rows with 0 — vacuously always-true.
  Pathway agent AUROC 0.444, Spearman −0.097 (exp1) — worse than ignoring
  the input. Real rebuild requires deriving directionality from
  `upstream_regulators` (69,537 rows) + expression threshold + essentiality
  — this is a full derived-table pipeline, not a patch. **If the timeline
  before defense is tight, retiring T3 from voting is the correct fallback
  — do not leave the vacuous table live, it is actively harming results,
  not neutral.**
  Note: a correct rebuild will likely *shrink* pathway's coverage from its
  current near-100%. That is the honest outcome (abstain instead of being
  confidently wrong) — frame this in the writeup before it's misread as
  regression.
- [ ] **Calibrate T2's silencing threshold** (currently z < −2.0, fires on
  1.39% of expression rows). Sweep threshold vs. AUROC/coverage tradeoff
  against existing GDSC/CTRP labels — this is a pure DB + correlation
  analysis, no LLM calls needed, cheapest high-value item on this list. Do
  not change the threshold without this calibration; −2.0 is currently an
  unvalidated magic number.

## Phase 6 — the actual thesis contribution (largest single lift)

- [ ] **Build the real `AxiomChallenge` exchange.** Minimal viable version:
  when a higher-tier decisive agent disagrees with a lower-tier one, emit
  one `AxiomChallenge` (tier, evidence, argument); the lower-tier agent's
  `critique()` response must address it before being eligible to flip.
  **Must replace or explicitly subordinate `_apply_verdict_revision`'s
  peer-majority flip path** (`debate_engine.py:44-95`) — if both mechanisms
  coexist, agents can still flip via majority pressure alone with no axiom
  challenge involved, meaning the sycophancy pattern the thesis critiques
  remains live in parallel with its own fix. This is a design decision, not
  an additive patch — resolve it explicitly.
  Highest-value item relative to the thesis's actual claim: everything
  above makes the current ensemble more correct; this is what makes it a
  hierarchy-of-truth system rather than an ensemble with a priority tiebreak.
  Sequence after Phase 2/3 — this touches the same revision code path.

## Phase 7 — close Aim 2 (reasoning quality / faithfulness)

- [ ] **Build `reasoning_scorer.py`** from the existing `compute_h()` logic
  in `gcs.py` (fraction of cited biomarkers present in real MCP evidence —
  this is already the hallucination-rate computation, currently folded into
  a single opaque GCS scalar and never reported standalone). Report
  hallucination rate and axiom-invocation-consistency per case.
  **Depends on Phase 1's attribution fix and Phase 0's trace logging** —
  building this against biased attribution or missing trace fields produces
  numbers that look like Aim 2 compliance without actually being valid.
- [ ] **Validate GCS weighting empirically** (`h*0.40 + e*0.30 + t*0.20 +
  s*0.10`) against actual correctness once reasoning_scorer exists — this
  weighting was chosen, never measured. Don't keep reporting it as
  calibrated confidence until it's checked.

## Phase 8 — final reportable numbers only

- [ ] **Re-run key comparisons N≥3 times, report variance.** LLM sampling
  non-determinism visibly changed an outcome this session (case a8: 2
  decisive agents in the exp1 isolated run vs apparently 1 in the exp3
  debate run, changing which consensus bug fired). 3x cost/time multiplier
  — apply this ONLY to final thesis-reportable runs, after Phases 0–6 have
  stabilized. Applying it during active debugging triples iteration cost
  for zero benefit while code is still changing.

---

## Explicit sequencing dependencies (do not violate)

- Phase 2 (resolver sort-key fix) **before** Phase 3 (quorum floor) — else
  the quorum floor routes more cases into a resolver that still inverts the
  axiom hierarchy.
- Phase 1 (label strip) **before** Phase 4/Phase 8 — else T1 re-anchor
  validation and final numbers are checked against a cheating T4 baseline.
- Phase 1 (attribution fix) **before** Phase 7 (reasoning scorer) — else
  hallucination/faithfulness numbers inherit the tier-first attribution bias.
- Phase 0 (trace logging) **before** Phase 7 — reasoning_scorer needs
  `self_attestation`/`data_status` fields that don't exist in current traces.
- Phase 6 (AxiomChallenge) **after** Phase 2/3 — touches the same
  `_apply_verdict_revision` code path; redesigning it before consensus
  mechanics are stable means redoing the design twice.
