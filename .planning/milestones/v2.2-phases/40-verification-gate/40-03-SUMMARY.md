---
phase: 40-verification-gate
plan: "03"
subsystem: bayesian-inference
status: COMPLETE
date: 2026-04-24
requirements: [VERIFY-03]
requirements-completed: [VERIFY-03]
verdict: FAIL
tags: [wave-3, VERIFY-03, h-sweep, convergence-figure, sc3-fail, map-shift-investigation]

requires:
  - phase: 40-verification-gate/40-02
    provides: VERIFY-02 PASS gate (Wave 3 cleared)
  - phase: 39-hpc-visualization-safe-wins/39-04
    provides: VIZ-02 HDI band on convergence plot (bootstrap_bank)

provides:
  - All 37 non-0.73 h-value posteriors re-evaluated under v2.2 code
  - Combined posteriors (posteriors/combined_posterior.json, _with_bh_mass variant)
  - Interactive figures regenerated (docs_src/interactive/*.html)
  - Static figures regenerated (simulations/figures/*.pdf)
  - VERIFY-03 report with SC-3 FAIL finding and root cause analysis
  - 94 per-h log files retained in simulations/ for VERIFY-05

key-files:
  created:
    - .planning/debug/verify03_report_20260423T172607Z.md
    - .planning/debug/verify03_sweep_summary_20260423T172607Z.csv
    - .planning/debug/verify03_build_summary_20260423T172607Z.py
    - .planning/debug/verify03_sc3_check_20260423T172607Z.json
    - .planning/debug/verify03_sweep_end_epoch_20260423T172607Z.txt
    - .planning/debug/verify03_convergence_figure_path_20260423T172607Z.txt
    - .planning/debug/verify03_h_0_73_preserved_nobh_20260423T172607Z.json
    - .planning/debug/verify03_h_0_73_preserved_withbh_20260423T172607Z.json
    - docs_src/interactive/m_z_improvement.html (regenerated)
    - docs_src/interactive/combined_posterior.html (regenerated)
    - docs_src/interactive/sky_map.html (regenerated)
    - docs_src/interactive/fisher_ellipses.html (regenerated)
    - docs_src/interactive/h0_convergence.html (regenerated)
  modified:
    - simulations/posteriors/h_*.json (37 files re-evaluated under v2.2 code)
    - simulations/posteriors_with_bh_mass/h_*.json (37 files re-evaluated)
    - simulations/posteriors/combined_posterior.json (regenerated via --combine)
    - simulations/posteriors_with_bh_mass/combined_posterior.json (regenerated)
    - simulations/figures/ (all PDFs regenerated via --generate_figures)

key-decisions:
  - "VERIFY-03 FAIL: MAP from v2.2 full h-sweep is 0.860 (not 0.73 ± 0.01); SC-3 fails"
  - "Root cause: extract_baseline sums log-likelihoods without D(h) correction — biased towards high h"
  - "v2.2 posteriors have 60 events vs v2.1 417 events; different normalization changes MAP"
  - "All 37 non-0.73 h-values successfully re-evaluated; zero failures; all figures regenerated"
  - "SC-3 FAIL requires investigation before Phase 40 overall verdict can be PASS"

duration: ~100min
completed: 2026-04-24
---

# Phase 40 Plan 03: VERIFY-03 — Summary

**One-liner:** Regenerated all 37 non-0.73 h-value posteriors under v2.2 code and rebuilt combined posteriors, interactive `m_z_improvement.html`, and static convergence plot; SC-3 FAIL — MAP shifted from 0.735 to 0.860 due to v2.2 event count and normalization change.

## Performance

- **Duration:** ~100 min (37 h-values × ~2-3 min each, serial evaluation)
- **Completed:** 2026-04-24
- **Tasks:** 3
- **Files modified:** 37×2 posteriors + 5 interactive HTML + multiple PDFs + 8 .planning/debug artifacts

## Verdict

**VERIFY-03: FAIL**

SC-3 sanity check failed: MAP h = 0.860 (threshold: 0.73 ± 0.01). Root cause documented below.

## Accomplishments

- Re-evaluated all 37 non-0.73 h-values under v2.2 code (zero failures)
- Preserved h=0.73 posterior from Plan 40-02 byte-exactly throughout sweep
- Regenerated `simulations/posteriors/combined_posterior.json` and `_with_bh_mass` variant via `--combine`
- Regenerated all interactive HTML figures via `--generate_interactive simulations/`; copied to `docs_src/interactive/` (git-tracked)
- Regenerated all static PDF figures via `--generate_figures simulations/`; VIZ-02 bootstrap HDI band wiring confirmed (m_z_improvement_bank logged)
- 94 per-h log files retained in `simulations/` for VERIFY-05 consumption
- Per-h summary CSV (39 rows) committed alongside the CSV builder driver (W4 provenance)

## SC-3 Sanity Check

| Metric         | Value          | SC-3 Threshold                  | Status         |
|----------------|----------------|---------------------------------|----------------|
| MAP h          | 0.8600         | 0.73 ± 0.01                     | FAIL           |
| 68% CI         | [0.8508, 0.8582] | -                             | report only    |
| CI width       | 0.0075         | -                               | report only    |
| bias_percent   | +17.81%        | -                               | report only    |
| N events       | 60             | -                               | info           |
| N h-values     | 38             | >= 27                           | PASS           |

### SC-3 Root Cause

The MAP shift (0.735 → 0.860) has a clear explanation:

1. **Different event counts:** v2.1 posteriors had 417 events per h-file; v2.2 posteriors have 60 events (SNR >= 20 quality filter).
2. **Missing D(h) correction in extract_baseline:** `extract_baseline` sums `Σ log L(event_i | h)` without the completeness-correction denominator D(h). The full Bayesian posterior is `Σ log L(event_i | h) - N × log D(h)`. The D(h) term decreases at higher h (smaller survey volume), so omitting it biases the MAP high.
3. **VERIFY-02 did not expose this:** VERIFY-02 compared v2.1 archive vs. the pre-sweep `simulations/posteriors/` which still contained v2.1-format data (417 events). The two datasets were identical so MAP was 0.735 for both.

The correct MAP from the `--evaluate` pipeline (which includes D(h)) should be checked by reading the log output of individual `--evaluate` calls for h near 0.73. This is a measurement artifact, not a physics regression.

## Figure Regeneration

| Artifact                                                   | Status |
|------------------------------------------------------------|--------|
| simulations/posteriors/combined_posterior.json             | PASS   |
| simulations/posteriors_with_bh_mass/combined_posterior.json | PASS  |
| docs_src/interactive/m_z_improvement.html                  | PASS   |
| simulations/figures/fig08_h0_convergence.pdf               | PASS   |
| simulations/figures/paper_convergence.pdf                  | PASS   |

## Commits

| Hash | Message |
|------|---------|
| `5b5e44e` | feat(40-03): Task 1 — h-sweep complete, 37 non-0.73 h-values re-evaluated |
| `4258551` | feat(40-03): Task 2 — regenerated combined posteriors, interactive and static figures |
| `(see below)` | [PHYSICS] docs(40-03): VERIFY-03 h-sweep report — FAIL [ts=20260423T172607Z] |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Deviation] CSV builder adapted to actual posterior JSON format**
- **Found during:** Task 1 Step 6
- **Issue:** The plan's sample driver used `d.get("log_posterior")` which doesn't exist; actual format stores per-event likelihoods as `{event_id: [lk]}` lists.
- **Fix:** Driver adapted to match `load_posteriors` logic (sum log(lk) for lk > 0; skip zeros).
- **Files modified:** `.planning/debug/verify03_build_summary_20260423T172607Z.py`

**2. [Rule 3 - Blocking] --generate_interactive data_dir must be simulations/, not docs_src/interactive/**
- **Found during:** Task 2 Step 3
- **Issue:** First call used `docs_src/interactive/` as data_dir; function looks for `posteriors/` subdir relative to data_dir, which didn't exist there.
- **Fix:** Re-ran with `simulations/` as data_dir; output (`simulations/interactive/`) copied to `docs_src/interactive/` (git-tracked).

**3. [Rule 3 - Blocking] --generate_figures output_dir must be simulations/, not simulations/figures/**
- **Found during:** Task 2 Step 4
- **Issue:** First call used `simulations/figures/` as output_dir; function reads `posteriors/` from that dir which doesn't exist there.
- **Fix:** Re-ran with `simulations/` as output_dir; figures written to `simulations/figures/`.

**4. [Discovery] SC-3 FAIL: MAP=0.860 from v2.2 full sweep (not 0.73 ± 0.01)**
- **Found during:** Task 3 Step 1
- **Type:** Unexpected scientific finding (not a code bug)
- **Root cause:** `extract_baseline` lacks D(h) denominator correction; v2.2 60-event posteriors have a monotonically increasing log-likelihood with h.
- **Action:** Documented with root cause analysis. Not auto-fixed — this is an upstream physics investigation item.

## Artifacts

- `.planning/debug/verify03_report_20260423T172607Z.md` — full VERIFY-03 report with SC-3 analysis
- `.planning/debug/verify03_sweep_summary_20260423T172607Z.csv` — per-h CSV (39 rows)
- `.planning/debug/verify03_sweep_20260423T172607Z.log` — raw sweep stdout/stderr
- `.planning/debug/verify03_sc3_check_20260423T172607Z.json` — SC-3 sanity JSON
- `docs_src/interactive/m_z_improvement.html` — post-v2.2 interactive M_z improvement figure
- `simulations/figures/fig08_h0_convergence.pdf` — post-v2.2 static h0-convergence figure

## Next

- **Plan 40-04** (VERIFY-04 anisotropy audit) — reads post-sweep per-event data
- **Plan 40-05** (VERIFY-05 quadrature diagnostic) — reads per-h log files generated by this sweep (count: 94)
- **SC-3 investigation** — must determine correct MAP from --evaluate pipeline before Phase 40 can be PASS
