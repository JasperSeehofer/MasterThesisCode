# Phase 40 PLAN-CHECK (Iteration 2)

**Final verdict: VERIFICATION PASSED**

**Date:** 2026-04-23
**Iteration:** 2
**Scope:** Re-verify 6 revision commits against 2 blockers + 5 warnings from iteration 1.

---

## Previously flagged items

### B1 (BLOCKER) — 40-06 depends_on must include 40-00

**✓ RESOLVED** (commit 6a3daa0)

Evidence: `40-06-PLAN.md:6` now reads:
```
depends_on: ["40-00", "40-01", "40-02", "40-03", "40-04", "40-05"]
```
All 6 upstream plans listed, 40-00 present.

### B2 (BLOCKER) — concrete committed Python driver for abort diagnostic

**✓ RESOLVED** (commit 7b1542f)

Evidence in `40-02-PLAN.md`:
- Line 465: `files_modified` includes `.planning/debug/verify02_write_abort_diagnostic_{ts}.py  # concrete writer driver (always created)`
- Lines 495–614: Full verbatim driver body with `argparse` (`--comparison`, `--baseline`, `--current`, `--out`), NOT a heredoc inline.
- Lines 556–564: 7-row cause table pinned at plan time: PE-02, PE-01, STAT-01, STAT-03, COORD-02/02b, COORD-03, COORD-04 (exactly 7 rows matching the spec).
- Lines 566–569: 4-option Triage list (rollback / diagnostic phase / accept / consult).
- Lines 595–597: log-posterior side-by-side table emitted as `| h | log_post_v2.1 | log_post_v2.2 | Δ |` built from baseline + current JSONs.
- Line 704: driver is committed via `FILES_TO_ADD+=(".planning/debug/verify02_write_abort_diagnostic_${TS}.py")` in the `[PHYSICS]` commit.
- Line 737: automated verify asserts exactly 7 cause rows via `grep -c '^|.* | 3[67] |'`.

Stable path: `.planning/debug/verify02_write_abort_diagnostic_${TS}.py` (timestamp-suffixed but pattern-stable, committed alongside the comparison driver for provenance).

### W1 — 40-05 borderline band verdict

**✓ RESOLVED** (commit 7c701e4)

Evidence in `40-05-PLAN.md`:
- Lines 428–429: `BORDERLINE_LOW = 0.03`, `BORDERLINE_HIGH = PHASE_41_THRESHOLD` (0.05).
- Lines 437–452: 3-way verdict `PHASE-41-TRIGGERED` (`> 0.05`), `PHASE-41-TRIGGER-BORDERLINE` (`[0.03, 0.05]`), `NOT-TRIGGERED` (`< 0.03`).
- Line 575: JSON emits `phase_41_trigger: trigger_verdict` (one of the 3 strings) plus back-compat `phase_41_trigger_bool`.
- Line 612: automated regex `(PHASE-41-TRIGGERED|PHASE-41-TRIGGER-BORDERLINE|NOT-TRIGGERED)`.
- 40-06-PLAN.md:434–441 routes BORDERLINE to `BORDERLINE-PAUSE` (halt chain, user decides), matching the intent of W1.

### W2 — 40-01 STAT-03 zero-fill test node id pinned

**✓ RESOLVED** (commit 799048e — note user cited 799042e but log shows 799048e)

Evidence in `40-01-PLAN.md`:
- Line 205: `STAT03_PRIMARY_NODE="master_thesis_code_test/test_completion_term_fix.py::TestZeroFillPdetAccessor::test_zero_fill_matches_standard_inside_grid"` — exact pin from the feedback.
- Line 206: `grep -cF "$STAT03_PRIMARY_NODE"` — literal match, no regex guessing.
- Lines 208–211: fallback regex retained only as a secondary guard.
- Comment block lines 72–74 documents the W2 pin and the fallback rationale.

### W3 — flat REQUIREMENTS wording (no nested parentheses)

**✓ RESOLVED** (commit c654abb)

Evidence in `40-04-PLAN.md`:
- Line 17 (must_haves.truths): `"...replaced with flat form '— >1σ shift is a Stage-2 trigger for Phase 42 (not a blocker)' (em-dash, single parens per W3)"`
- Line 70 (interfaces comment): target text is `"...shows no systematic trend — >1σ shift is a Stage-2 trigger for Phase 42 (not a blocker)"`
- Line 138 (Task 1 New string): `"- [ ] **VERIFY-04**: Anisotropy audit: H₀ MAP binned by \`|qS − π/2|\` quartiles shows no systematic trend — >1σ shift is a Stage-2 trigger for Phase 42 (not a blocker)"` — single em-dash + single parenthesis pair, no nesting.
- Line 155 automated verify: `grep -q "Stage-2 trigger for Phase 42 (not a blocker)"`.
- Line 600 (success_criteria): explicitly states `"(W3: em-dash + single pair of parens)"`.

### W4 — 40-03 Python driver replaces shell loop

**✓ RESOLVED** (commit c2ae5c1)

Evidence in `40-03-PLAN.md`:
- Line 126: `.planning/debug/verify03_build_summary_{ts}.py    # W4: one-shot CSV builder` in `files_modified`.
- Lines 208–283: full Python driver body (not a shell `for h in …` loop). Takes `<posteriors_root> <csv_out>` as argv, globs `h_*.json` internally, writes CSV with header `h,file_mtime_utc,log_posterior_max,n_detections`.
- Lines 287–289: single `uv run python "$DRIVER" simulations/posteriors "$CSV"` invocation — one process, no per-file shell loop.
- Line 304 automated verify: `test -f ".planning/debug/verify03_build_summary_${TS}.py"`.
- Line 321 acceptance: driver committed alongside CSV.

### W5 — 40-04 argv-based driver (no sed -i)

**✓ RESOLVED** (commit c654abb — same commit as W3, plan verified uses argv not sed)

Evidence in `40-04-PLAN.md`:
- Lines 129–139 (Task 1, REQUIREMENTS edit): uses the `Edit` tool with exact old/new strings — NOT `sed -i`. Explicit directive at line 129: `"Edit the line in place using the Edit tool (not Write — the file has many other lines)"`.
- Lines 251–253 (anisotropy driver, Task 2): `if len(sys.argv) < 2: sys.exit(...)` ; `CRB_CSV = Path(sys.argv[1])` — CRB CSV path is argv-driven, not baked into the committed driver.
- Line 434–438: invocation `uv run python ".planning/debug/anisotropy_driver_${TS}.py" "$CRB_CSV"` — path passed at runtime; committed driver stays byte-identical to what ran.
- Line 250 comment: `"# --- Inputs (CRB path from sys.argv[1] per W5 — no sed-i on committed Python) ---"`.

---

## Sanity checks (not previously flagged)

| Check | Status | Evidence |
|-------|--------|----------|
| REQ-ID coverage VERIFY-01..05 → 40-01..05 | **✓ PASS** | Each VERIFY-NN appears in exactly one plan's `requirements:` frontmatter (40-01→V01, 40-02→V02, 40-03→V03, 40-04→V04, 40-05→V05). 40-00 and 40-06 carry `requirements: []` as expected (archive + phase-close). No gaps, no overlaps. |
| Wave ordering W0→W1→W2→W3→W4 | **✓ PASS** | 40-00 wave 0 deps []; 40-01 wave 1 deps [40-00]; 40-02 wave 2 deps [40-00,40-01]; 40-03 wave 3 deps [40-02]; 40-04 wave 3 deps [40-03]; 40-05 wave 3 deps [40-03]; 40-06 wave 4 deps [40-00..05]. Strictly increasing, no cycles, no forward references. |
| `gpd_gate: true` on 40-02 | **✓ PASS** | 40-02-PLAN.md:17. |
| Acceptance criteria ↔ REQ-IDs ↔ ROADMAP SCs | **✓ PASS** | 40-01 acceptance references D-05/D-06 thresholds for SC-1; 40-02 acceptance maps to SC-2 (`|ΔMAP|/0.73 < 0.05`, `bias < 1%`); 40-03 acceptance maps to SC-3 (`|MAP - 0.73| ≤ 0.01`, ≥27 h); 40-04 maps to SC-4 (σ rule + Stage-2 trigger); 40-05 maps to SC-5 (reporting SC with Phase 41 trigger). |
| Deep-work rules | **✓ PASS** | No abstractions invented (uses existing `BaselineSnapshot`, `extract_baseline`, `generate_comparison_report`, `combine_posteriors`, `plot_h0_convergence`). Atomic commit per task (40-04 has 2: wording + audit; 40-02 has 1 `[PHYSICS]`; 40-03 has 1 `[PHYSICS]`). `[PHYSICS]` prefix correctly applied to 40-02 and 40-03 (runs physics-changed code paths) and NOT to 40-01/04/05/06 (pure ledger/diagnostic). |
| 40-03 Wave 3 dep on 40-02 (abort gate) | **✓ PASS** | 40-03 deps=[40-02]; 40-04/05 deps=[40-03]. Wave 3 cannot start if 40-02 ABORT. Matches the "W2 (40-02 abort gate) → W3 (40-03+04+05) → W4 (40-06 close)" intent. |

---

## Summary

All 2 blockers and 5 warnings from iteration 1 are resolved. 6 sanity-check dimensions pass. Plans ready for execution.

**VERIFICATION PASSED**
