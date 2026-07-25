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

- [x] **Strip `label` from every MCP tool response server-side**
  (`pharmacology_server.py::get_ic50`/`get_drug_response` — currently
  `SELECT *`). 16/20 cases in `trials/dataset.json` have T4's own
  precomputed `drug_response.label` equal to ground truth. Pharmacology's
  78.9%/75.0% (exp1) is not yet trustworthy evidence of real reasoning.
  **Do this before validating any fix that depends on a trustworthy T4
  baseline (Phase 3, Phase 5).**
  Done: `get_ic50` now pops `label` from the response dict before returning.
  The pharmacology agent's system prompt was directly instructing the LLM to
  read `label` (a leaked-answer field) — rewrote the reasoning protocol so
  the agent classifies SENSITIVE/RESISTANT from `z_score` itself instead of
  reading a precomputed field, since leaving the prompt unchanged would have
  had the agent looking for a field that no longer exists. New test asserts
  `"label" not in agent.system_prompt.lower()` to prevent this regressing.
- [x] **Separate verdict attribution from tier-priority display**
  (`debate_engine.py::_build_result`, line ~307). `winning_agent` is
  currently picked by `(AXIOM_HIERARCHY[tier], confidence)` — tier-first,
  so any agreeing T1 agent gets credited as "winner" regardless of whether
  it contributed the deciding signal (case a1: pharmacology and pathway did
  the epistemic work at higher confidence; genomics got the credit). Add a
  `contributing_agents` field based on actual decisiveness/confidence.
  **Hard dependency for Phase 4's reasoning-scorer** — building it against
  the current biased attribution field bakes the bias into every hallucination/
  faithfulness number it produces.
  Done: added `ConsensusResult.contributing_agents` — agreeing agents ranked
  by confidence descending. `winning_agent` is untouched (still tier-first,
  which is the correct semantic for "whose axiom won"); `contributing_agents`
  is the new field for "who actually had the strongest evidence." Populated
  in both `DebateEngine._build_result` and `NoDebateEngine` (the latter was
  about to silently default to `[]`, same class of bug as Phase 0's
  `r1_agents` fix). Logged in `exp2`/`exp3` trial outputs.

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

- [x] **Add a decisive-quorum floor to `_check_consensus`**
  (`debate_engine.py:98-130`). Require ≥2 decisive agents before R1/R2 can
  resolve by majority; below that, always route to the resolver with an
  explicit low-confidence flag. Fixes Bug C (single decisive agent
  auto-wins trivially since `n=1 > len(decisive)/2=0.5`) and the correlated-
  error blind spot (case a7: pharmacology + pathway both wrong and
  unanimous among just the two decisive agents, genomics/transcriptomics
  silently abstained, no mechanism caught the correlated error).
  **Do not apply before Phase 2 is complete.**
  **Design correction before implementing**: a literal fixed floor of
  "≥2 decisive agents" only fixes Bug C — it does not block case a7, since
  a7 already has exactly 2 decisive agents. Implemented instead as a
  *relative* floor: `len(decisive) > len(all_packs) / 2` (decisive agents
  must be a strict majority of the full 4-agent panel, not just of the
  decisive subset). This is what the to-do's own two examples actually
  require, and it's now safe post-Phase-2 — the resolver correctly applies
  the axiom hierarchy instead of a raw majority vote, so routing more
  low-quorum cases to it is a feature, not a risk.
  Done: added the relative-floor check in `_check_consensus` right after
  the T3 self-attestation gate (so gated-out pathway packs correctly count
  against quorum too). 4 new tests: direct unit tests on `_check_consensus`
  for the single-decisive-agent case (Bug C) and the 2-of-4 correlated-
  agreement case (case a7), a control test confirming 3-of-4 still resolves
  normally, and an end-to-end `DebateEngine.run()` test confirming the a7
  shape now returns `RESOLVER_TIEBREAK`/`forced=True` instead of
  `CONSENSUS_R1`.

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

## Phase 8 — close the gap between the test suite and the actual Specific Aims

Audited `tests/` against `docs/specific_aims.tex` (the authoritative Aims
document). Everything in Phases 0–7 is tested correctly at the mechanism
level, but several pieces the Aims document commits to defending have no
test coverage — in two cases, no code at all. None of this was visible from
reading the engine code; it only shows up by reading the Aims doc and the
test suite side by side. **This phase should land before Phase 9's final
numbers** — you cannot report a defensible Aim 1/Aim 2 table with 4 of the
required comparisons untested or unimplemented.

- [ ] **Test `MonolithicAgent` — Aim 1's required second baseline.**
  `src/agents/monolithic_agent.py` exists (single-LLM baseline, no MCP
  grounding, no debate) but zero files under `tests/` reference it. Aim 1's
  milestone is "framework exceeds *both* baselines" — right now only the
  Random Forest baseline has any test coverage, and even that is only
  exercised on synthetic 2-point arrays (`test_baselines.py`), never real
  GDSC2 features. Add unit tests mirroring the pattern in `test_agents.py`
  (mocked LLM response → verdict/confidence/tier), and confirm the mocked
  monolithic prompt path actually differs from the multi-agent prompts
  (no MCP tool calls, no axiom hierarchy).
- [ ] **Test the CTRPv2 held-out validation path.** Aim 1's stated milestone
  ("CTRPv2 AUROC within 0.05 of development-set AUROC") has no test
  anywhere — `test_etl.py`'s 20 tests all cover the GDSC2 dev-set ETL path
  only; nothing references `CTRP` or `held_out`. At minimum, test that
  `etl_ctrp.py`'s loader produces the expected schema and that whatever
  script computes "AUROC delta between dev and held-out set" is covered by
  a unit test on synthetic metrics (not requiring a live LLM run).
- [ ] **Implement and test the 2 missing ablations from Aim 2's six
  pre-specified list.** `AblationVariant` currently has exactly 3 members
  (`NO_DEBATE`, `NO_AXIOMS`, `RANDOM_AXIOM_ORDER`). Missing:
  - **Single agent** — collapse to one agent (no cross-tier resolution at
    all); this is the sharpest test of whether the axiom hierarchy adds
    anything over a single specialist.
  - **Free-text tools** — replace MCP-validated structured tool returns
    with free-text summaries of the same underlying data. This is the
    direct test of the thesis's "first to deploy MCP as a structural
    anti-hallucination layer" claim — without this ablation existing, that
    claim is asserted, not measured.
  Add both as new `AblationVariant` members with a minimal engine/agent
  variant each, plus tests following the existing `NoDebateEngine` pattern.
- [ ] **Promote `compute_h()` to a standalone, reported "hallucination
  rate"** — overlaps with Phase 7's `reasoning_scorer.py` item; this note
  is here to make explicit that Aim 2's <5% target is a *primary* metric in
  the Aims doc, not a nice-to-have. `test_gcs.py` tests `compute_h()`
  correctly in isolation but there is currently no test proving a
  per-dataset "hallucination rate" number can be produced at all, because
  the aggregation code doesn't exist yet.
- [ ] **Build a "biological faithfulness score" — this does not exist
  anywhere in the codebase, not even as a stub.** Aim 2's other primary
  metric (≥90% axiom-invocation consistency with established molecular
  biology) needs: (1) a rubric or expert-curated ground-truth table mapping
  valid `axiom_invoked` strings to the tiers/contexts where they're
  biologically valid, (2) a scorer comparing each case's invoked axioms
  against it, (3) tests for the scorer itself. Currently `Finding.axiom_invoked`
  is only round-tripped as a free-text schema field (`test_schemas.py`,
  `test_agents.py`) — nothing validates its content. This is the larger of
  the two missing-metric items; timebox the rubric-building step separately
  from the scorer-coding step.
- [ ] **Fix `test_create_pathways_db_bypass_routes_written`'s blind spot.**
  It only asserts one specific row has `bypass_exists=1`; it never asserts
  any row has `bypass_exists=0`. This test passes identically whether
  `bypass_routes` is a real signal or the vacuous always-true table
  documented in Phase 5 (239,327/239,327 rows =1). Add a negative-case
  assertion once Phase 5's rebuild lands, so this regression can never go
  undetected again. **Depends on Phase 5.**

## Phase 9 — final reportable numbers only

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
- Phase 8's bypass_routes test fix **depends on** Phase 5's rebuild — the
  negative-case assertion has nothing to assert against until then.
- Phase 8 (Aims coverage: baselines, ablations, faithfulness/hallucination
  metrics) **before** Phase 9 (final numbers) — Phase 9's comparisons are
  only defensible once the code and tests they report on actually exist.
