# Phase 40: Verification Gate — Context

**Gathered:** 2026-04-23
**Status:** Ready for planning

<domain>
## Phase Boundary

The last checkpoint before any new compute is committed. Re-evaluate the existing CRBs under
all v2.2 fixes (coord frame — COORD-02/02b/03/04; PE h-threading + per-param epsilon —
PE-01/02; L_cat — STAT-01; P_det zero-fill — STAT-03; eigenvalue sky radius — COORD-05) and
confirm the posterior at h=0.73 is stable. Includes the 27-value h-sweep, an anisotropy audit
binned by `|qS − π/2|` quartiles, and the STAT-04 quadrature-weight-outside-grid diagnostic
summary that decides whether Phase 41 fires.

**In scope:**
- VERIFY-01: full CPU pytest pass with v2.2-era regression tests included
- VERIFY-02: re-evaluate at h=0.73 against the v2.1 baseline; abort gate on >5% MAP shift
- VERIFY-03: 27-h sweep re-evaluated; convergence figure + `m_z_improvement.html` updated
- VERIFY-04: anisotropy audit by `|qS − π/2|` quartiles; Stage-2-trigger semantics (not blocker)
- VERIFY-05: P_det quadrature-weight-outside-grid diagnostic across all 27 h-values with
  per-h histogram; Phase 41 trigger computed from h=0.73 mean

**Out of scope:**
- New cluster compute (Phase 41 / Phase 42 are conditional and follow this gate)
- New physics changes — any change needed to recover stability is a separate phase
- Automated bisection of v2.2 fixes when the abort gate fires — halt-and-diagnose only
- Refactors or broader architecture changes
- Paper-draft regeneration — v2.0 Paper remains paused pending this gate

</domain>

<decisions>
## Implementation Decisions

### VERIFY-02: Baseline pinning and abort-gate comparison

- **D-01 (Baseline source):** The current posteriors in `simulations/h_0_*/combined_posterior.json`
  are the v2.1-era result (pre-Phase-37/38 physics changes, MAP=0.73, bias=0.0%, N=59 per memory
  `project_bias_audit.md`). Before any Phase 40 re-evaluation, copy them verbatim to
  `simulations/_archive_v2_1_baseline/`. No re-derivation under a git tag — the current files
  already represent v2.1 state because `--evaluate` has not been run since Phase 36 landed.
- **D-02 (Archive location):** `simulations/_archive_v2_1_baseline/` only. Raw posteriors stay in
  the git-ignored `simulations/` tree; nothing duplicated into `.planning/`. The aggregated
  `verify_gate_{timestamp}.md` references the archive by path.
- **D-03 (Comparison metrics — all four reported, only #1 is the abort gate):**
  1. **MAP at h=0.73** — abort if `|MAP_v2.2 − MAP_v2.1| / 0.73 ≥ 0.05` (explicit SC-2 rule)
  2. **CI width** — reported as sanity signal; not an abort criterion
  3. **bias_percent** — reported; SC-2 requires `< 1%` at h=0.73
  4. **Full 27-h posterior shape** — two-sample KS test on the `log_posterior` curves between
     archived v2.1 and post-Phase-40 v2.2 runs; flagged in the report but not an abort metric
- **D-04 (MAP definition):** `argmax(log_posterior)` over the discrete 27-h grid — already locked
  by `master_thesis_code/bayesian_inference/evaluation_report.py:253`. Do not reimplement.

### VERIFY-01: Test-suite gate

- **D-05:** `uv run pytest -m "not gpu"` must exit 0 with **test count ≥ 540** (Phase 39 baseline).
  No new failures relative to Phase 39.
- **D-06 (v2.2 regression inventory — all must appear in the collection):**
  - `test_coordinate_roundtrip.py` (Phase 36, 9 tests)
  - PE-01 h-threading test (Phase 37, `test_set_host_galaxy_parameters_uses_injected_h` or peer)
  - `test_l_cat_equivalence.py` (Phase 38, 3 tests)
  - STAT-03 P_det zero-fill symmetry test (Phase 38)
  - HPC-02 SIGTERM drain test (Phase 39, commit 815ac4a)
- **D-07 (Slow suite):** Not required for VERIFY-01 pass. Can be spot-run but not gating.

### Execution wave structure

- **D-08 (Strict 3-wave gate):**
  - **Wave 1:** VERIFY-01 — full CPU pytest pass. Must succeed before Wave 2.
  - **Wave 2:** VERIFY-02 — re-evaluate `simulations/h_0_73/` only. Check abort gate against
    D-03 #1. If `|ΔMAP| / 0.73 ≥ 0.05`, halt the phase (see D-10) and **do not proceed to W3**.
  - **Wave 3:** VERIFY-03 (remaining 26 h-values), VERIFY-04 (anisotropy audit), VERIFY-05
    (quadrature diagnostic) run in parallel — all three read from the full 27-h re-eval output,
    so W3 is "the 27-h sweep plus two post-processors".
- **D-09:** The archive step (D-01) happens **before Wave 1** — a Wave 0 preflight. If the
  archive copy fails or the archive already exists (re-run case), fail loudly with an explicit
  error, do not silently overwrite.

### Abort-gate response (VERIFY-02 trigger)

- **D-10 (Halt and diagnose):** When the abort gate fires:
  1. Stop further execution (no W3).
  2. Write `.planning/debug/abort_verify_gate_{timestamp}.md` with:
     - Baseline MAP, v2.2 MAP, signed shift in %
     - Full `log_posterior` curves side-by-side (baseline vs v2.2)
     - Candidate-cause table mapping each v2.2 physics change to a plausible shift direction
       (PE-02 epsilon → CRB scale; STAT-01 L_cat → catalog weighting; STAT-03 zero-fill → P_det
       floor; COORD fixes → source-localization shift)
     - Pointer to next-step options (rollback branch, targeted diagnostic phase)
  3. Do **not** auto-advance to Phase 41. The user drives the next step manually.
- **D-11:** Automated bisection is explicitly **out of scope** (the codebase has no per-fix
  feature flags and building them is a Phase-41-sized project on its own).

### VERIFY-04: Anisotropy rule

- **D-12 (Stage-2 trigger semantics, not blocker):** The REQUIREMENTS.md wording "`>1σ shift is
  a blocker`" is **obsolete**. ROADMAP.md SC-4 ("`Stage 2 trigger rather than an abort
  condition`") governs. REQUIREMENTS.md VERIFY-04 must be updated as part of the plan (first
  atomic commit of the audit plan).
- **D-13 (Test statistic):** Per-quartile MAP difference. For each of 4 `|qS − π/2|` quartiles:
  1. Select events in the quartile
  2. Re-combine their posteriors → `MAP_q`
  3. Compare `|MAP_q − MAP_total|` to `σ`, where `σ = (CI_upper − CI_lower) / 2` from the
     **overall h=0.73 posterior** (not per-quartile CI, which would be dominated by small N).
  4. Flag quartile if `|MAP_q − MAP_total| > σ`.
- **D-14 (Quartile definition):** Equal-count quartiles on `|qS − π/2|` across the N=59
  h=0.73 events (not across all 27×59 event-instances). Use the post-Phase-40 re-evaluation's
  event set, not the archived v2.1.
- **D-15 (Artifact):** Dedicated `.planning/debug/anisotropy_audit_{timestamp}.md` containing
  the quartile table (edges, event count per quartile, `MAP_q`, `|MAP_q − MAP_total|`, trigger
  status), a brief pass/trigger verdict, and a link back to the main `verify_gate` report.

### VERIFY-05: P_det quadrature-weight diagnostic

- **D-16 (Scope):** All 27 h-values. For every `(h, event)` pair in the sweep, log the
  per-event `quadrature_weight_outside_grid` fraction that STAT-04 already emits as a WARNING.
- **D-17 (Summary statistics):**
  - Mean and max across events **per h-value** (27 rows)
  - Mean and max across all `(h, event)` pairs (1 row)
  - Histogram of per-event fractions, aggregated across all h-values
- **D-18 (Phase 41 trigger):** `mean_{h=0.73}(quadrature_weight_outside_grid) > 0.05` triggers
  Phase 41. This is the primary-baseline scope to match SC-3's `h=0.73` focus; the all-h
  aggregate is reporting context, not the decision number.
- **D-19 (Artifact):** Dedicated `.planning/debug/pdet_quadrature_summary_{timestamp}.md`.

### Artifact layout (hybrid index + subfiles)

- **D-20:** Top-level index `.planning/debug/verify_gate_{timestamp}.md` is a short
  verdict table:

  ```
  | REQ       | Status                             | Detail |
  |-----------|------------------------------------|--------|
  | VERIFY-01 | PASS / FAIL                        | pytest command, pass count, new v2.2 tests listed |
  | VERIFY-02 | PASS / ABORT                       | baseline MAP, v2.2 MAP, shift %, link to abort_* if fired |
  | VERIFY-03 | PASS                               | convergence figure path, m_z_improvement.html updated y/n |
  | VERIFY-04 | PASS / STAGE-2-TRIGGER             | → anisotropy_audit_{ts}.md |
  | VERIFY-05 | PASS / PHASE-41-TRIGGER / REPORT   | → pdet_quadrature_summary_{ts}.md, h=0.73 mean |
  ```

  Per-VERIFY detail lives in siblings:
  - `verify_gate_{ts}.md` (index)
  - `anisotropy_audit_{ts}.md` (VERIFY-04)
  - `pdet_quadrature_summary_{ts}.md` (VERIFY-05)
  - `abort_verify_gate_{ts}.md` (only if abort fires)

  VERIFY-01/02/03 are short enough to inline in the index.
- **D-21:** `{timestamp}` is UTC ISO-8601 compact (`YYYYMMDDTHHMMSSZ`). Identical timestamp
  across the four sibling files in a single gate run.

### Claude's Discretion

- Exact MD table formatting, column widths, and histogram binning (10 bins over [0, 1] is a
  sensible default).
- Whether to emit a machine-readable JSON sidecar (`verify_gate_{ts}.json`) alongside the MD
  index — useful for CI automation but not required by any SC.
- How to surface the KS-test p-value in D-03 #4 (inline in the index or in the verify_gate
  body).
- Parallel execution mechanism for Wave 3 — the h-sweep itself is CPU-bound serial inside
  `main.py --evaluate`; VERIFY-04/05 are post-processors that read the same output directory
  so they can run sequentially in a single script after the sweep finishes.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 40 roadmap & requirements
- `.planning/ROADMAP.md` §"Phase 40: Verification Gate" (lines 125–139) — goal, routing, abort
  gate, SC-1..SC-5
- `.planning/REQUIREMENTS.md` VERIFY-01..05 (lines 49–53) — REQ definitions. **Note: VERIFY-04
  wording "blocker" is superseded by D-12; plan must update it.**
- `.planning/STATE.md` §"Key Context for v2.2 Pipeline Correctness" and §"Phase Notes" —
  current progress, Phase 37 + 38 caveats about baseline drift

### Prior-phase artifacts (the v2.2 fixes being verified)
- `.planning/phases/36-coordinate-frame-fix/36-VERIFICATION.md` — coord-frame fix verification
- `.planning/phases/36-coordinate-frame-fix/36-superset-regression.pkl` — Phase 36 regression
  pickle (obsolete for Fisher values per Phase 37 note; sky-frame values still valid)
- `.planning/phases/37-parameter-estimation-correctness/37-01-SUMMARY.md` — PE-01 h-threading
- `.planning/phases/37-parameter-estimation-correctness/37-02-SUMMARY.md` — PE-02 per-param
  epsilon (the one whose Fisher-value changes feed into VERIFY-02)
- `.planning/phases/38-statistical-correctness/38-SUMMARY.md` — STAT-01..04
- `.planning/phases/39-hpc-visualization-safe-wins/39-VERIFICATION.md` — baseline SC gate
  style / reporting layout to reuse

### Code — MAP and posterior machinery
- `master_thesis_code/bayesian_inference/evaluation_report.py:240–290` — `BaselineSnapshot`,
  `extract_baseline()`, MAP = argmax(log_posterior) over discrete h grid (locks D-04)
- `master_thesis_code/bayesian_inference/evaluation_report.py:340–400` — `compare_snapshots()`
  delta-MAP logic (can be reused for VERIFY-02 abort gate)
- `master_thesis_code/main.py:160–220` — `--evaluate` CLI entry and
  `baseline`/`current` snapshot comparison CLI
- `master_thesis_code/bayesian_inference/bayesian_statistics.py:119, 454–463, 672–717` —
  STAT-03 zero-fill sites and STAT-04 quadrature WARNING (source of VERIFY-05 data)

### Reference data locations
- `simulations/h_0_*/combined_posterior.json` — current posteriors (= v2.1 baseline per D-01)
- `simulations/h_0_*/combined_posterior_with_bh_mass.json` — with-BH variant (archive too)
- `.planning/debug/baseline.json` — **NOT the v2.1 production baseline**; it is a 5-point
  synthetic unit-test artifact. Do not confuse.
- `.planning/debug/comparison_current.md` — template for how evaluation comparisons are
  formatted in this project

### Physics-change protocol
- `CLAUDE.md` §"Math/Physics Validation Workflow" — VERIFY-02 runs physics-changed code paths
  (STAT-01, STAT-03, PE-01, PE-02, COORD-02/02b/03/04) and the routing table requires GPD for
  this REQ. The rest of Phase 40 is GSD-native (test orchestration, reporting).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `BaselineSnapshot` dataclass (`evaluation_report.py:53`) + `extract_baseline()` (line 240) +
  `compare_snapshots()` (line 340) give a ready-made v2.1-vs-v2.2 diff pipeline. Phase 40 can
  feed both the archived posteriors and the v2.2 re-eval through these functions and emit the
  shift-% directly.
- `.planning/debug/baseline.json` / `comparison_current.md` / `diagnostic_summary_current.md`
  establish an existing `.planning/debug/` report format — reuse it so the hybrid layout
  (D-20) stays consistent with prior phases.
- Phase 39's `39-VERIFICATION.md` shows the SC-by-SC verdict table style Phase 40 should
  mirror for its `verify_gate_{ts}.md` index.
- `main.py --evaluate` CLI already drives the full `--h_value`-sweep-over-event-set pipeline;
  the 27-h sweep runs as a small shell loop around it.

### Established Patterns
- MAP extraction is `argmax(log_posterior)` over the discrete h grid. Don't introduce KDEs.
- Posteriors live as per-h-value `combined_posterior.json` files, not a single 27-h matrix.
  Diagnostic code loads them with `load_posteriors()`.
- `.planning/debug/` is the established location for evaluation artifacts — not
  `.planning/phases/40-*/`. The phase directory holds PLAN/SUMMARY/VERIFICATION only.
- Timestamped filenames (ISO-8601 compact UTC) are already used in
  `master_thesis_code_20260408_*.log` — adopt that convention for D-21.
- Physics-change git convention: subject-line prefix `[PHYSICS]` — applies to VERIFY-02
  runs that execute physics-changed code even if Phase 40 itself writes no formula.

### Integration Points
- Evaluation is CPU-only in the current workflow (no GPU dependency for `--evaluate`).
  Phase 40 runs entirely on the dev machine; no cluster submission.
- STAT-04 WARNING (`quadrature_weight_outside_grid`) is emitted as a standard `_LOGGER.warning`.
  VERIFY-05 must capture these from the re-eval log (via log file or by reading the written
  per-event diagnostic field — check `bayesian_statistics.py:119` for where it lands in output).
- The convergence figure (SC-3) is produced by `plot_h0_convergence` in
  `plotting/bayesian_plots.py`. Phase 39 VIZ-02 added bootstrap HDI bands to the right panel;
  that runs automatically when the full 27-h posterior set is present.
- `docs_src/interactive/m_z_improvement.html` is regenerated by
  `master_thesis_code --generate_interactive`; Phase 40 just needs to run it after the sweep.

</code_context>

<specifics>
## Specific Ideas

- Abort-gate response should be actionable: the `abort_verify_gate_{ts}.md` candidate-cause
  table (D-10) lets the user choose a diagnostic path without re-reading five prior
  SUMMARY.md files.
- Event 2 is flagged in memory (`project_injection_todo.md` + Phase 38 note) as having 100%
  off-grid quadrature weight. VERIFY-05 should explicitly call out which events dominate the
  mean — not only report the aggregate — so the Phase 41 injection-grid extension can target
  the right events.
- Keep the verify_gate_{ts}.md index under ~150 lines. If it grows beyond that, move detail
  into more sibling files rather than padding the index.

</specifics>

<deferred>
## Deferred Ideas

- **Per-fix feature flags** for automated v2.2 bisection on abort (D-11) — non-trivial
  infra; only build if a Phase 40 abort actually fires.
- **Two-tier anisotropy rule** (>1σ trigger, >3σ blocker) — considered but rejected in
  favor of simpler Stage-2-trigger-only semantics (D-12). Revisit only if empirical Phase 40
  data shows the single-threshold rule is too permissive.
- **KDE-based MAP** — considered but rejected; locked to `argmax(log_posterior)` on the
  discrete 27-h grid (D-04).
- **Machine-readable JSON sidecar** for the verify_gate report — Claude's Discretion; add
  if the Phase 41 decision logic wants to read it programmatically, otherwise MD is enough.
- **CI width as an abort criterion** — intentionally not an abort metric (D-03); CI can shift
  legitimately under the STAT-01/03 zero-fill fixes without invalidating the MAP.
- **h-sweep cluster execution** — not needed; evaluation is CPU-cheap enough locally.
  Revisit only if 27-h wall-clock becomes a bottleneck.

</deferred>

---

*Phase: 40-verification-gate*
*Context gathered: 2026-04-23*
