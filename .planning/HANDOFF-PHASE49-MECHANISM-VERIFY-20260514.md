# Handoff — Verify SNR-threshold mechanism before F4 commit (2026-05-14)

## TL;DR

F1 fix (h-stable `dl_edges`) landed but the user-reported spiky posterior is
**NOT** resolved (max adjacent-bin ratio still 16×/51× in 1D/2D; MAP shifted
+0.0056 further from truth). A second mechanism — **SNR-threshold integer
crossings** of individual injections — is hypothesized but **not verified**.

**Next action: run a ~30-min local diagnostic to confirm or refute the
mechanism BEFORE committing to a 1-2 day F4 (Farr 2019 reweighting) refactor.**

Lesson behind this gate: [[scientific-computing-validation#verify-user-symptom-targeting-fixes-against-the-original-diagnostic-not-just-property-tests]] +
EXP-20 (don't accept a diagnosis without independent verification). The
F1 fix went through without that verification gate; we are not making the
same mistake on F4.

## Current state (as of 2026-05-14T15:00 CEST, pre-clear)

### Code
- F1 fix in `[PHYSICS]` commit `87ea7a8` (h-stable `dl_edges` for histogram
  p_det estimator in `master_thesis_code/bayesian_inference/simulation_detection_probability.py:404,540`).
- 523 pytest pass; mypy/ruff/format clean.

### Cluster
- F1 cluster validation completed (job `4662333`, 14 tasks × ~21 min on cpu_il).
- Pre-F1 posteriors preserved at
  `bwunicluster:/pfs/work9/workspace/scratch/st_ac147838-emri/run_production_h0p73_20260506/simulations/archive/production_h0.73_20260512_175829/`.
- Post-F1 posteriors at the same RUN_DIR's
  `posteriors{,_with_bh_mass}/` (63 files each, h-grid identical to Phase 48).

### Local
- 63 post-F1 1D + 63 post-F1 2D posteriors rsynced to
  `simulations/cluster_run_production_h0p73_20260506/posteriors{,_with_bh_mass}/`.
- Combined posteriors at `simulations/cluster_run_production_h0p73_20260506/combined_posterior{,_with_bh_mass}.json`.
- Pre-F1 verdict: `scripts/bias_investigation/outputs/phase46_merged/h3_production_sweep_verdict.json` (restored from git, the Phase 48 reference).
- Post-F1 verdict: `scripts/bias_investigation/outputs/phase46_merged/F1_post_fix_verdict_PARTIAL.json` (this session's run).

### Docs / vault
- Debug session: `.planning/debug/posterior-noisy-peak.md` — status updated to `partial-fix-landed-second-mechanism-suspected`; follow-up section appended with all numbers + F4 proposal + handoff items.
- DATA_INVENTORY.md: Phase 48 row tagged STALE; new 2026-05-14 Phase 49 F1 PARTIAL row added.
- CHANGELOG.md: Unreleased Changed section documents PARTIAL outcome.
- Vault: SCV pattern "Hyperparameter-dependent discretization → coherent noise" annotated as F1=necessary-but-not-sufficient; new SCV positive pattern "Verify user-symptom-targeting fixes against the original diagnostic" promoted from W-PRE-06 inline note; W-PRE-09 observation row filed; vault commit `d34e88a`.

### Commits unpushed
- Project: `d9d9380` (F1 PARTIAL post-mortem). On local `main`; not on `origin/main`.
- Vault: `d34e88a` (debrief addendum). On local `main`; not on `origin/main`.
- **Decide whether to push before starting the diagnostic.** Pushing is reversible; the partial verdict is real and shouldn't be hidden. Recommended: push first (one command each), then start the diagnostic.

## Action — the diagnostic script

### Hypothesis to test

For each injection at fixed `(z_inj, SNR_raw, h_inj)`:
- `SNR(h) = SNR_raw · d_L(z_inj, h_inj) / d_L(z_inj, h)` shifts smoothly with h.
- At some specific `h*`, `SNR(h*) = 20` (the detection threshold).
- At h < h*, the injection counts as "detected" in `np.histogram2d(detected_mask=True)` for its bin; at h > h*, it doesn't.
- This produces a 1/N_bin step in `p_det = detected / total` at that bin and at h = h*.
- The post-F1 spike pattern (especially the discontinuity at h=0.738→0.739) should correspond to threshold crossings of specific identifiable injections at specific h-values.

### Script outline

Save as `scripts/bias_investigation/test_29_snr_threshold_crossings.py`. CPU-only;
takes ~10 min wall.

```python
"""Verify whether residual post-F1 p_det spikes correlate with
SNR-threshold integer crossings of individual injections.

Strategy:
1. Load the post-F1 SimulationDetectionProbability instance for the
   production injection campaign.
2. Probe p_det(d_L=d_L_query, M_z=M_z_query, h) across h ∈ [0.730, 0.745]
   at Δh=0.0005 (finer than the Phase 48 grid so we resolve crossings
   precisely).  Query points chosen to live in the bin containing the
   F1 PARTIAL MAP at h=0.738.
3. Compute Δp_det / Δh for each adjacent h-pair.  Flag h-values where
   |Δp_det| > 0.01 (the "spike" scale from the post-F1 verdict).
4. For each flagged h-value h_jump:
   a. Identify the bin (i_dL, j_M) under the query point.
   b. Compute SNR(h_jump) for all 105k injections; identify the subset
      whose SNR straddles the threshold (e.g. SNR(h_jump - Δh) > 20 >
      SNR(h_jump + Δh)) AND whose d_L_target(h_jump) falls in bin (i_dL, j_M).
   c. Count how many such injections explain the magnitude of Δp_det
      (expected: at least one, with |Δp_det| ≈ flips × 1/total_in_bin).
5. Output a table: h_jump | Δp_det | injection_idx | SNR(h-Δh) | SNR(h) |
   SNR(h+Δh) | (d_L_inj(h_jump), M_z_inj).
6. If table is populated: mechanism confirmed.
7. If table is empty (no threshold-crossing injections in spike bins):
   mechanism refuted; we have a different bug to chase.

Reuses fixtures from
master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py.
"""
```

### Files and paths the script will need

- `INJECTION_DATA_DIR` — production injection campaign:
  `/home/jasper/Repositories/MasterThesisCode/master_thesis_code/data/injections/`
  (or wherever the production pipeline points; check `master_thesis_code/main.py`
  for the canonical path).
- `SimulationDetectionProbability` instance — construct with
  `snr_threshold=20.0`, default `dl_bins=60`, `mass_bins=40`,
  `h_prior_range=(0.60, 0.86)` (F1 defaults).
- Query points — pick from the F1 PARTIAL verdict JSON's MAP region:
  - `h-grid = np.arange(0.730, 0.745 + 1e-9, 0.0005)` (31 points, Δh=0.0005, 2× finer than Phase 48).
  - `d_L_query` — pick the median injection d_L at h=0.73 (probably ~1 Gpc; check via `dist_vectorized(z_arr, h=0.73)`).
  - `M_z_query` — median observer-frame M_z (from `_M_arr * (1+_z_arr)`).

### Expected output / decision tree

| Outcome | Interpretation | Next action |
|---|---|---|
| Table populated, |Δp_det| ≈ flips/total_in_bin for each spike | **Mechanism confirmed.** Proceed to F4 plan: Farr 2019 fixed-injection + analytic-reweighting refactor. | Write FILE: `.planning/PHASE-49-F4-PLAN.md`; engage `/physics-change`. |
| Table empty (no threshold crossings explain spikes) OR magnitude mismatch | **Mechanism refuted.** Hypothesis was wrong; need to identify the actual residual mechanism. | Re-open the gsd-debug session with hypothesis-refuted note; consider M-axis bin crossings, RegularGridInterpolator C0 discontinuities at the *fixed* knots, or numerical-precision issues in the per-h reweighting math. |
| Mixed (some spikes explained, others not) | **Both mechanisms contribute.** F4 addresses the threshold-crossing part; the other spikes need separate analysis. | Write F4 plan but note the orthogonal residual; expect F4 to partially close but not fully eliminate. |

### Cheaper alternative (smoke test, ~5 min)

If running the full diagnostic feels expensive, a quick smoke test:

```python
# Load the production SDP, probe p_det at fixed query across a fine h-grid,
# count "jump" h-values.  If <2 jumps in [0.730, 0.745], mechanism is unlikely
# to be SNR-threshold crossings (each crossing = 1 jump; we see ~10 spikes
# in the actual posterior over this range).
```

This won't identify *which* injections cause the spikes but will tell you whether the order of magnitude of the threshold-crossings is consistent with the observed spike count.

## Outstanding questions for the next session (resolve while running A)

1. **What's the source of the 2D σ_boot=0 in the F1 PARTIAL verdict?**
   - Every bootstrap landed on h=0.738. Is this because the posterior really is concentrated on one grid bin (rare but possible), or because of a bug in the bootstrap loop where it grabbed the same argmax 1000× (e.g. floating-point tie-breaking)?
   - Cheap to check: print the 1000 bootstrap MAP values from test_28's bootstrap loop; histogram them.
2. **Did F1 actually fix the bin-edge-drift mechanism it was designed to fix?**
   - Compare pre-F1 vs post-F1 dl_edges across two close h values (e.g. h=0.731 and h=0.732) — confirm post-F1 edges are identical.
   - Already covered by the new regression test `test_dl_edges_identical_across_two_trial_h`. Sanity-check that test passes on the current HEAD just to confirm.
3. **Is the SNR-threshold mechanism in fact dominant, or is there a third mechanism we haven't named?** The handoff hypothesis is one of several. The diagnostic in this handoff is specifically about ruling SNR-threshold in or out; if ruled out, the gsd-debug session needs to re-open.

## Long-term followups (do NOT block on these)

- Refactor `simulation_detection_probability.py` toward Farr 2019 form (F4) regardless of this session's diagnostic outcome — it's the consensus practice and the EMRI inference paper will be expected to use it.
- Caretaker sweep on the vault will eventually upgrade or demote the two SCV patterns + W-PRE-09 row (currently all `tentative`); next session doesn't need to drive that.

## How to start the next session

```bash
# 1. Verify state matches handoff
cd /home/jasper/Repositories/MasterThesisCode
git log --oneline -5            # expect d9d9380 at HEAD
git status                       # expect only .planning/debug/baseline.json
                                 # + .planning/debug/comparison_current.md modified

# 2. (Optional) push the two unpushed commits
git push origin main             # d9d9380
cd /home/jasper/Repositories/professional-vault && git push     # d34e88a

# 3. Read this handoff + the debug session follow-up
cat .planning/HANDOFF-PHASE49-MECHANISM-VERIFY-20260514.md
cat .planning/debug/posterior-noisy-peak.md | head -250

# 4. Start the diagnostic
#    Create scripts/bias_investigation/test_29_snr_threshold_crossings.py per the
#    outline above.  Reuse SimulationDetectionProbability + dist_vectorized
#    imports already wired in test_simulation_detection_probability.py.

# 5. Decision after diagnostic:
#    - confirmed → write .planning/PHASE-49-F4-PLAN.md + engage /physics-change
#    - refuted   → re-open gsd-debug session with hypothesis-refuted note
```
