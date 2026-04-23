# Phase 40: Verification Gate — Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `40-CONTEXT.md` — this log preserves the alternatives considered.

**Date:** 2026-04-23
**Phase:** 40-verification-gate
**Areas discussed:** Baseline pinning (VERIFY-02), Anisotropy rule (VERIFY-04), Wave structure + abort response, Artifact layout + VERIFY-05 scope

---

## Area Selection

| Area | Description | Selected |
|------|-------------|----------|
| Baseline pinning (VERIFY-02) | What is 'v2.1 baseline MAP'? Archive, re-derive, or trust memory? | ✓ |
| Anisotropy rule (VERIFY-04) | REQUIREMENTS.md 'blocker' vs ROADMAP.md 'Stage-2 trigger' contradiction. | ✓ |
| Wave structure + snapshot | Strict gate vs parallel; reversibility of the abort gate. | ✓ |
| Artifact layout + VERIFY-05 scope | Aggregated vs per-VERIFY reports; h=0.73 only vs all 27 h-values. | ✓ |

**User's choice:** All four areas.

---

## Area 1: Baseline pinning (VERIFY-02)

### Q1: How should we pin the v2.1 baseline for the VERIFY-02 abort gate?

| Option | Description | Selected |
|--------|-------------|----------|
| Archive current posteriors | Before any --evaluate run, copy simulations/h_0_*/combined_posterior.json to simulations/_archive_v2_1_baseline/. Effectively v2.1 because --evaluate has not been re-run since Phase 36. | ✓ |
| Re-derive under v2.1 tag | Check out the last commit before Phase 37 physics changes landed; run --evaluate fresh; archive; return to main. Bit-exact but expensive. | |
| Trust memory figure | Use MAP=0.73, bias=0.0% from project_bias_audit.md memory as the baseline number. No artifact snapshot. | |

**User's choice:** Archive current posteriors.
**Notes:** The current simulations/ state is already pre-Phase-37/38 physics-change posteriors because `--evaluate` has not been re-run since Phase 36. Archiving is cheapest and reversible.

### Q2: What should the abort-gate comparison compare?

| Option | Description | Selected |
|--------|-------------|----------|
| MAP at h=0.73 | Primary abort-gate metric per SC-2. | ✓ |
| CI width | Sanity signal (SC-3). Not an abort criterion. | ✓ |
| bias_percent | Reporting completeness. SC-2 requires < 1%. | ✓ |
| Full 27-h posterior shape | Two-sample KS on log_posterior curves. Tighter, non-SC check. | ✓ |

**User's choice:** All four (multi-select).
**Notes:** Only MAP is the abort trigger; the other three are reported alongside.

### Q3: Where should the baseline archive live?

| Option | Description | Selected |
|--------|-------------|----------|
| simulations/_archive_v2_1_baseline/ only | Inside git-ignored simulations/ tree. No duplicate in .planning/. | ✓ |
| .planning/debug/verify_baseline/ only | Committed JSON; fully reproducible. Duplicates simulations/ data. | |
| Both: raw in simulations, summary committed | Raw posteriors in simulations/; BaselineSnapshot JSON committed. | |

**User's choice:** simulations/_archive_v2_1_baseline/ only.
**Notes:** Raw posteriors stay in simulations/. The verify_gate report references them by path; no duplicate artifact under .planning/.

---

## Area 2: Anisotropy rule (VERIFY-04)

### Q1: How should we resolve the REQUIREMENTS.md vs ROADMAP.md contradiction?

| Option | Description | Selected |
|--------|-------------|----------|
| Stage-2 trigger only | Any >1σ shift triggers Phase 42. Never blocks. Matches ROADMAP SC-4 and 2026-04-21 staged-campaign philosophy. | ✓ |
| Tiered: >1σ trigger, >3σ blocker | Codifies both intents with separate thresholds. | |
| Blocker only | Any >1σ pauses the paper. Matches REQUIREMENTS.md strictly. | |

**User's choice:** Stage-2 trigger only.
**Notes:** REQUIREMENTS.md wording will be updated in the Phase 40 plan's first atomic commit to match ROADMAP.md.

### Q2: What test statistic should flag the anisotropy?

| Option | Description | Selected |
|--------|-------------|----------|
| Per-quartile MAP difference | MAP per |qS−π/2| quartile vs overall MAP; flag if |MAP_q − MAP_total| > 1σ (overall CI width). | ✓ |
| Linear trend slope across quartiles | Fit MAP_q vs quartile index; flag on slope magnitude. | |
| KS-test between equatorial / polar halves | Split at median; single p-value. | |

**User's choice:** Per-quartile MAP difference.
**Notes:** σ uses the overall h=0.73 posterior CI half-width (not per-quartile CI, which would be dominated by small N).

### Q3: Where should the anisotropy result be logged?

| Option | Description | Selected |
|--------|-------------|----------|
| Aggregated verify_gate report | Quartile table inline in .planning/debug/verify_gate_{ts}.md. | |
| Dedicated anisotropy_audit.md | Separate .planning/debug/anisotropy_audit_{ts}.md referenced from the main index. | ✓ |

**User's choice:** Dedicated anisotropy_audit.md.
**Notes:** Enables easier cross-run diffs and selective paper quoting.

---

## Area 3: Wave structure + abort-gate response

### Q1: How should Phase 40 waves be gated?

| Option | Description | Selected |
|--------|-------------|----------|
| Strict 3-wave gate | W1 VERIFY-01 → W2 VERIFY-02 + abort check → W3 VERIFY-03/04/05 parallel. Stops before 27-h sweep if abort fires. | ✓ |
| 2-wave (tests → all re-eval) | W1 VERIFY-01 → W2 VERIFY-02..05 parallel. | |
| Single wave | All VERIFY-0N together; check abort after completion. | |

**User's choice:** Strict 3-wave gate.
**Notes:** Compute-cost-optimal. Preserves the "pause before investing compute" intent of SC-2.

### Q2: What happens if the VERIFY-02 abort gate fires?

| Option | Description | Selected |
|--------|-------------|----------|
| Halt and open diagnostic phase | Stop W3. Write abort_{ts}.md with shift, candidate-cause table, per-fix toggle pointers. User decides next step. | ✓ |
| Halt and automatically bisect | Re-evaluate with each v2.2 fix toggled off. Requires per-fix feature flags the codebase lacks. | |
| Halt and flag for manual review | Just stop; surface the shift. No automated artifact. | |

**User's choice:** Halt and open diagnostic phase.
**Notes:** Automated bisection is deferred — the codebase has no per-fix feature flags and building them is a separate project.

### Q3: What defines VERIFY-01 'pass'?

| Option | Description | Selected |
|--------|-------------|----------|
| Full CPU suite green + new v2.2 tests included | pytest -m 'not gpu' exits 0; ≥540 tests (Phase 39 baseline); v2.2 regression inventory present. | ✓ |
| Full CPU + slow suite | Also run 'slow' tests. Hours slower; not physics-critical. | |
| Just 'not gpu and not slow' | Dev-fast subset; may skip v2.2 regressions marked slow. | |

**User's choice:** Full CPU suite green + new v2.2 tests included.
**Notes:** Explicit inventory: test_coordinate_roundtrip.py, PE-01 h-threading, test_l_cat_equivalence.py, STAT-03 P_det zero-fill symmetry, HPC-02 SIGTERM drain.

---

## Area 4: Artifact layout + VERIFY-05 scope

### Q1: How should the Phase 40 artifacts be organized?

| Option | Description | Selected |
|--------|-------------|----------|
| Hybrid: aggregated report + per-VERIFY subfiles | verify_gate_{ts}.md index with verdict table; detail in anisotropy_audit_{ts}.md and pdet_quadrature_summary_{ts}.md siblings. | ✓ |
| Single aggregated report | Everything in verify_gate_{ts}.md. 1000+ lines; hard to diff. | |
| Per-VERIFY files only, no index | Five files, filesystem is the index. | |

**User's choice:** Hybrid: aggregated report + per-VERIFY subfiles.

### Q2: What is the VERIFY-05 quadrature-weight diagnostic scope?

| Option | Description | Selected |
|--------|-------------|----------|
| All 27 h-values, per-h histogram + aggregate | Log per-event fraction for every (h, event); mean/max per h-value and aggregated. | ✓ |
| h=0.73 events only | Matches SC-3 sanity focus; smaller report. | |
| h=0.73 + two flanking values | Compromise (e.g., 0.65, 0.81). | |

**User's choice:** All 27 h-values, per-h histogram + aggregate.
**Notes:** Broadest signal for Phase 41 trigger decision.

### Q3: How is 'mean extrapolation weight > 5%' (Phase 41 trigger) computed?

| Option | Description | Selected |
|--------|-------------|----------|
| Mean across events at h=0.73 | Primary-baseline scope matching SC-3. | ✓ |
| Mean across all events and all h-values | Broadest; dilutes h=0.73 signal. | |
| Max per-h mean (worst-case) | Conservative; most likely to fire Phase 41. | |

**User's choice:** Mean across events at h=0.73.
**Notes:** The all-27-h reporting (Q2) is context; the trigger decision uses h=0.73 only.

---

## Claude's Discretion

- Exact MD table formatting and histogram binning defaults (suggested 10 bins over [0, 1]).
- Whether to emit a machine-readable JSON sidecar alongside the verify_gate MD index.
- How to surface the KS-test p-value (inline in index or in body).
- Wave-3 parallel execution mechanism — the 27-h sweep runs sequentially inside --evaluate;
  VERIFY-04/05 are post-processors on the output directory, so "W3 parallel" may in practice
  be "sweep → 04 → 05" in one script.

## Deferred Ideas

- Per-fix feature flags for automated v2.2 bisection on abort (non-trivial infra; only build
  if a Phase 40 abort actually fires).
- Two-tier anisotropy rule (>1σ trigger, >3σ blocker) — considered, rejected for simplicity.
- KDE-based MAP — rejected; locked to argmax(log_posterior) on the discrete 27-h grid.
- Machine-readable JSON sidecar for verify_gate — optional, decide during plan.
- CI width as an abort criterion — intentionally not an abort metric.
- h-sweep cluster execution — not needed; CPU-cheap locally.
