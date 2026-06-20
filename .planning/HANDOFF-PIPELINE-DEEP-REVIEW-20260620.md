# HANDOFF — Pipeline deep-review agenda + residual-bias close-out

**Date:** 2026-06-20  **Author:** session with Jasper
**Branch state:** `main` clean; **`physics/lcat-gray-ratio-of-sums`** (commit `816f904`) holds the
L_cat fix, **NOT merged** (awaits multi-seed cluster verification — see §2).

---

## 1. WHERE WE ARE (this session)

The residual-H0-bias chase was pushed to a real, committed result and is now **paused** pending a
cluster run. Full narrative: `docs/H0_BIAS_RESOLUTION.md` §3.17. Memory:
[[project_residual_bias_decomposition]], [[project_fisher_frame_mismatch]].

- **Fisher ecliptic/equatorial frame suspect → CLOSED (verified non-cause).** seed400 covariance is
  genuinely native-ecliptic; rotating would double-rotate. The deep-survey **independently re-confirmed**
  this (FEW differentiates in ecliptic qS/phiS, `waveform_generator.py:64` `is_ecliptic_latitude=False`).
  **Stop chasing it.**
- **Completion term → characterized, faithful to Gray (2020).** Its `dV_c/dz` volume-prior pull is
  anchored by the catalog term (figure `scripts/bias_investigation/test_31`). Two wrong-signed "fixes"
  (catalog `dd_L/dz` Jacobian; completeness-decline) were **refuted by an empirical sign-test**.
- **`[PHYSICS]` L_cat Gray Eq. A.9/A.10 fix LANDED (`816f904`):** removed spurious `p_det` from the
  catalog numerators (both channels) + ratio-of-sums aggregation; reverses the Phase-38 STAT-01
  *misreading*. **Measured: 1D MAP 0.750→0.740 (bias HALVED), 2D 0.7375→0.7350.** `/check` 568 green.
- **Residual after the fix: 1D +0.010, 2D +0.005** — at/below the ~0.017 single-seed scatter.

---

## 2. BLOCKING NEXT STEP — the keystone study (was: "multi-seed verification")

**The deep-survey's #1 finding reframes the cluster TODO into the single most valuable thing the
project can do.** The entire bias result rests on **ONE** EMRI population realization (seed400); the
author's own `test_24` docstring admits the multi-truth panel reuses the same injection set — so
+0.020/+0.0075 is statistically indistinguishable from single-catalog scatter.

**Run a multi-realization injection-recovery + coverage study** — K≥20–50 **independent population
seeds** at `h_true=0.73` on bwUniCluster. It does TRIPLE duty in one campaign:
1. **Bias verdict** = mean MAP across realizations (is there a real systematic at all?).
2. **Error bar** = spread across realizations (the cosmic-variance uncertainty the paper currently fudges).
3. **Calibration** = PP-plot / KS-uniformity of the `h_true` quantile → the coverage figure every
   GW-cosmology referee demands (currently **absent from all 600+ tests**).

The user runs the cluster jobs when back at the controlling machine (see `TODO.md` CLUSTER-VERIFY).
Minimum viable version is already wired: `for S in 500 600 700 800; do bash
cluster/submit_resimulate_phase50.sh --seed $S; done` (after `git checkout main && git pull && uv sync
--extra gpu`); scale to K≥20 for the real study. **Merge `physics/lcat-gray-ratio-of-sums` once this
runs** (the L_cat fix is correct on Gray-A.9/A.10 grounds regardless — merge even if the residual proves
to be scatter; just frame the residual accordingly).

> ⚠️ When generating the new CRBs, also do the S-effort fix: write `_coord_frame`/`_cov_frame=ecliptic`
> inside `save_cramer_rao_bound` so fresh output is self-labelled and `migrate_crb_to_ecliptic.py` is a
> guaranteed no-op — permanently kills the double-rotation trap for this campaign.

---

## 3. IN-DEPTH PIPELINE REVIEW — prioritized agenda (the user's 4 areas)

Grounded in a 4-prong deep-survey (full output:
`/tmp/.../tasks/wkux4nkut.output`, 76 KB — has per-area findings with file:line). **My corrections to
the survey's claims are flagged ⚠️.**

### P0 — correctness / freeze (cheap, unblock everything)
- **Freeze a config-of-record + fix the paper's hard contradictions.** `paper/` contradicts itself on the
  three numbers that drive every result: **SNR threshold** (`results.tex` says 15 vs `method.tex:96` says
  20), **Fisher stencil** (forward-diff vs five-point), **Ωₘ** (0.25 vs Planck). Reconcile across all
  `.tex`; add a CI check that the paper's quoted config matches `constants.py`. *(S)*
- **Make the paper compile into a reviewable draft.** Every `\includegraphics` in `results.tex` is
  **commented out**; captions carry `\todo`/`\pending`. Wire in the figures (PDFs exist, dated 2026-05-15),
  remove placeholders, fix the `SNR>15`→`SNR≥20` caption + abstract precision numbers. *(S)*
- **Redshifted-mass convention** ⚠️ **VERIFIED in production** (the prime population-side bias suspect):
  `handler.py:635` `_map_BH_masses_to_redshifted_masses` is defined but **never called**; the sim injects
  source-frame `M` into FEW; the production 2D mass filter `handler.py:379-385` divides `M_z/(1+z)`.
  Establish ONE convention end-to-end (this is the 2D channel = the paper's novelty), add a regression
  test, and run an isolated `--evaluate` to quantify its H0 shift. **/physics-change gate.** *(M)*

### P1 — HPC performance (correctness-preserving)
- **Batch the entire h-grid in ONE process.** Each h-value is a separate process re-reading the 1.4 GB
  GLADE catalog + re-spawning the pool + rebuilding BallTrees/survival-grid (~16.5s ≈ 40% fixed overhead
  × N_h; a 63-pt sweep wastes ~17 min). Loop the h-grid in-process → 1.6–2× and makes the K-realization
  sweeps affordable. Changes **no number**. *(M)*
- **Estimator + CI: pick ONE, end-to-end.** Production `--combine` reports MAP from a grid; the paper's
  ±0.013 CI appears to come from a different path. Choose posterior **median + 16/84 (or HPD)** from the
  *same* combined log-posterior on a Δh≈0.001 grid; add a grid-independence regression test. *(S)*
- **Thread `T` into the PSD** (`t_obs_years=self.T`) so the 5-yr run stops using the 4-yr
  confusion-noise level. **/physics-change.** *(S)*

### P2 — cleanup / later
- ⚠️ **`TRUE_HUBBLE_CONSTANT=0.7` is NOT a live bias** (survey claimed `galaxy.py:366` filters hosts at
  h=0.7 — but `datamodels/galaxy.py` is the **Pipeline-A synthetic catalog, not imported by `--evaluate`**;
  verified). It's a **dead-code footgun** — delete the constant + the synthetic `GalaxyCatalog`, or
  reconcile to 0.73. *(S, cleanup)*
- Simulation-side perf: Fisher FFT-reuse + on-device scalar accumulation; MC-denominator `fixed_quad`. *(M)*

### Plotting + GitHub Pages (area 3)
- **Fix the flagship H0 posterior figure** — currently two near-identical blues + a needle in whitespace;
  auto-zoom x-limits to the data CI, recolor to blue vs vermillion so the Mz-narrowing is visible. *(S)*
- **Add field-standard figures:** corner/`arviz` posteriors, sky-localization credible-area contours on a
  proper projection, the PP-plot/coverage figure (falls out of §2). *(M)*
- **Pages v2 (big bet, staged AFTER science frozen):** build-driven, provenance-stamped site — explorable
  H0 posterior (slider over N_events / Mz cut), galaxy-catalog sky explorer, scrollytelling methods
  walkthrough, per-figure "reproduce this command" links generated from `run_metadata.json`. *(L)*

---

## 4. BLIND SPOTS — "what this project would love to have" (you asked me to shine here)

1. **You may be chasing noise.** The whole bias hunt rests on one population draw; the fix (K independent
   seeds) *also* gives you the error bar and the coverage plot. This is the highest-leverage realization
   in the project right now — one campaign retires a months-long question.
2. **Zero calibration anywhere.** No PP-plot, no SBC, no coverage in 600+ tests. `gwcosmo`/`icarogw` ship
   this as standard; without it the quoted ±0.013 / 1.8% precision is an *unverified, rejectable* credible
   interval. This is the single biggest "what a referee will demand" gap.
3. **No external cross-check.** The bespoke completeness/`D(h)` likelihood (the most-debugged, most
   error-prone code) has never been validated against `gwcosmo`/`icarogw` on a shared simplified mock
   (complete catalog, P_det=1, single host, near-analytic answer). A toy cross-check anchors the method,
   localizes any residual bias instantly, and gives the methods section the validation a new pipeline owes.
4. **Estimator/CI come from two code paths, one of which self-flags as biased** — exactly the
   inconsistency that unravels under review. Unify it (P1 above).
5. **The paper isn't reviewable today** and contradicts itself on the config — review-killers wholly
   independent of the physics.
6. **"What's the actual contribution vs gwcosmo?"** — the thesis should state crisply what's novel
   (EMRI/LISA specificity; forecast vs measurement) and back it with the cross-check + coverage. Worth an
   explicit framing pass.

---

## 5. SUGGESTED SESSION ORDER (from the survey, with my corrections)

1. **(S, ~30 min) Close-out + freeze:** mark Fisher-frame dead (done in §3.17); freeze config-of-record
   (one SNR threshold, one stencil, one Ωₘ, one h_true); reconcile/delete `TRUE_HUBBLE_CONSTANT` (cleanup,
   *not* a bias).
2. **(M, /physics-change) Redshifted-mass convention** — make sim↔inference self-consistent, regression
   test, isolated `--evaluate` to quantify the shift. The prime population-side lead — settle *before*
   burning cluster time.
3. **(M) Batch the h-grid in one process** — prerequisite to make the K-realization sweeps affordable.
4. **(S/M) Estimator + golden-run lock** — median+HPD, grid-independence + end-to-end-MAP regression tests.
5. **(M, KEYSTONE) Launch the multi-realization campaign** (K≥20 seeds); while it runs, build the
   PP-plot/coverage harness so the calibration figure drops out automatically.
6. **(S, during cluster idle) Paper consistency pass** — wire figures, fix the flagship posterior, remove
   contradictions, add the paper-vs-constants CI check.
7. **(M) Simulation perf polish** (Fisher FFT-reuse, on-device accumulation) — speeds future campaigns.
8. **(L) Toy cross-check vs gwcosmo/icarogw + sensitivity sweeps** — dedicated phases once bias/coverage
   is answered.
9. **(M/L) Plotting + Pages v2** — final, public-facing, after the science is frozen.

---

## 6. PROTOCOL / POINTERS
- L_cat fix on branch `physics/lcat-gray-ratio-of-sums` (`816f904`); `main` clean. Merge after §2.
- Physics changes → `/physics-change` gate (5-point) + measure the sign empirically before committing
  (this session refuted 2 wrong-signed fixes that way).
- Working tree: `.planning/debug/{baseline.json,comparison_current.md}` were modified by an earlier
  session — NOT ours; leave unstaged. seed400 archives under `simulations/*_archive_20260619_*` and
  `*_20260620_lcat_gray_*` (FULL baseline + catalog_only + Gray-A.10 runs preserved).
- Vault debrief filed (`800e373` in professional-vault): W-CONF-12 (contaminated-CSV), 2 SCV patterns
  (p_det-denominator-only + ratio-of-sums; empirical-sign-test + primary-source verification).
- Full deep-survey output (per-area detail, file:line): the `wkux4nkut.output` task file (76 KB).
