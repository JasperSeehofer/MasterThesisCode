---
session_id: map-0p86-lcat-explosion
status: RESOLVED — Phase 44 shipped, MAP shifted 0.860 → 0.765 on cluster (412 events). Residual +0.035 to be addressed in a follow-up phase.
created: 2026-04-28
last_updated: 2026-04-29
resolved_in: Phase 44 (commit 3697bdd, merged to main as 857a1c8)
symptom: "MAP=0.86 after ecliptic migration; L_comp 1000-36000x larger at h=0.86 than h=0.73 for ~4 close events that dominate the product"
root_cause: "p_det_zero_fill returns 0 for d_L < dl_centers[0]. dl_centers[0] = dl_max/120 is a bin-width artifact that scales ∝ 1/h. Close events (d_L=0.085-0.097 Gpc) drop below this moving threshold at h=0.73 (threshold=0.100 Gpc) but not at h=0.86 (threshold=0.085 Gpc). L_comp=0 at h=0.73, L_comp>>0 at h=0.86 for these events. The threshold is NOT the injection minimum — injections cover d_L→0 through GLADE low-z galaxies."
fix_proposal: "Remove result[dl_arr < dl_min] = 0.0 from zero_fill function. Keep result[dl_arr > dl_max] = 0.0. Nearest-neighbour extrapolation (fill_value=None, already on interpolator) returns p_det(dl_centers[0]) ≈ 0.55 below threshold — correct because the first bin covers (0, 2*dl_centers[0]) and genuinely has p_det≈0.55 from real injection data."
fix_status: |
  RESOLVED. Cluster re-eval (jobs 4160638 + 4160639) on production seed200 (412 events
  post-SNR-filter) gave MAP = 0.7650 (was 0.860 pre-fix, truth h=0.73). Shift of
  -0.095 toward truth, +145.7 log-unit pathology eliminated, all 4 zero-handling
  strategies agree (no zero events remain). Cluster equatorial-revert test was
  cancelled (4159826/4159827) — local revert reproduced MAP=0.86 on the same data
  in 2 minutes (see "Local Revert Falsifiability" below), making the cluster revert
  redundant. Residual bias +0.035 (MAP=0.765 vs truth 0.730) deferred to a follow-up
  phase per plan §8 fallback regime "MAP ∈ [0.74, 0.79]".
---

# Debug Session: MAP=0.86 L_comp Explosion

## Problem Statement

After two physics fixes (D(h) correction commit `2853c32`, ecliptic covariance migration commit `ab4bc80` + cluster task), the evaluation pipeline still gives MAP=0.86 on the production seed200 dataset (312 events, cluster job 4148717). True injection value is h=0.73. Posterior at h=0.73 is negligible relative to h=0.86.

---

## What Was Ruled Out (do not re-investigate)

| Hypothesis | Status | Evidence |
|---|---|---|
| D(h) direction is a bug (should increase with h) | **ELIMINATED** | D(h) ∝ h^{-3} from dVc/dz is physically correct. Computed: D(0.60)=4.24e6, D(0.73)=3.71e6, D(0.86)=3.36e6 — decreasing is correct. |
| L_cat explosion from more BallTree galaxy matches at high h | **ELIMINATED as primary cause** | L_cat varies, but the dominant mechanism is L_comp going from 0 to >>0 for the 4 close events. |
| Ecliptic migration introduced a frame bug in the BallTree | **ELIMINATED** | Code audit confirms handler.py and migrate_crb_to_ecliptic.py are correct. |
| MAP=0.73 pre-migration was the correct baseline | **ELIMINATED** | It was accidental cancellation from a sky-frame mismatch (equatorial CRBs queried into ecliptic BallTree). See "Why MAP≈0.73 was accidental" below. |
| D(h) zero-fill causes D(h) to change differentially with h | **ELIMINATED** | z_cross where d_L(z,h)=dl_grid_min(h) is z≈0.0238 for ALL h (because dl_grid_min(h) ∝ 1/h and dist(z,h) ∝ 1/h cancel). So D(h) misses the same z<0.0238 slice at all h — no differential effect from this. |
| Injection minimum d_L sets dl_centers[0] | **ELIMINATED** | dl_centers[0] = dl_max/(2*N_bins) is a bin-width artifact. Injection campaign draws z from GLADE (which reaches z≈0.001 → d_L≈0.004 Gpc), so injections DO cover d_L→0. |

---

## Root Cause — Full Derivation

### Grid geometry (critical — read first)

`_build_grid_1d` (simulation_detection_probability.py:465–502):

```python
dl_max = float(np.max(dl_vals)) * 1.1          # max injection d_L * 1.1
dl_edges = np.linspace(0, dl_max, N_bins + 1)   # N_bins=60; edges: 0, dl_max/60, 2*dl_max/60, ...
dl_centers = 0.5 * (dl_edges[:-1] + dl_edges[1:])  # centers: dl_max/120, 3*dl_max/120, ...
```

- **Grid edges start at 0** — the first bin covers **(0, 2*dl_centers[0])**, not (dl_centers[0], ...)
- **dl_centers[0] = dl_max / 120** is purely a bin-width artifact, NOT related to minimum injection d_L
- At h=0.73: dl_max ≈ 12 Gpc → dl_centers[0] ≈ 0.0998 Gpc
- At h=0.86: dl_max ≈ 10.2 Gpc → dl_centers[0] ≈ 0.0847 Gpc
- **dl_centers[0] scales as ≈ 1/h** (because dl_max = max d_L ∝ 1/h after rescaling injections to h_target)

### Injection campaign coverage

`main.py:553–730` — `injection_campaign()`:
- Draws z from GLADE galaxy catalog via `cosmological_model.sample_emri_events()`
- GLADE reaches z ≈ 0.001 → d_L ≈ 0.004 Gpc
- Sets `luminosity_distance = dist(sample.redshift, h=h_value)` directly (line 657)
- **The injection campaign already covers d_L → 0.** No new injections needed.
- The first bin (0, 2*dl_centers[0]) has real injection events, and p_det ≈ 0.55 in that bin is the genuine average detection fraction over (0, ~0.2 Gpc)

### The bug (lines 708–713)

```python
def detection_probability_without_bh_mass_interpolated_zero_fill(...):
    ...
    result = np.clip(interp_1d(points), 0.0, 1.0)          # fill_value=None → nearest-neighbour

    # BUG: zeroes out left half of first bin, NOT just below injection minimum
    dl_centers = interp_1d.grid[0]
    dl_min = float(dl_centers[0])                           # = dl_max/120 ≈ 0.10 Gpc at h=0.73
    dl_max = float(dl_centers[-1])
    out_of_range = (dl_arr < dl_min) | (dl_arr > dl_max)
    result[out_of_range] = 0.0                              # ← zeroes the left half of the first bin
```

For an event with d_L_det = 0.085 Gpc:
- h=0.73: dl_centers[0] = 0.0998 → 0.085 < 0.0998 → p_det = 0 (event in first bin but zero-filled)
- h=0.86: dl_centers[0] = 0.0847 → 0.085 > 0.0847 → p_det = interpolated ≈ 0.57

The threshold moves with h. The same event crosses from p_det=0 to p_det≈0.57 as h increases past the crossing point. L_comp explodes.

### Why p_det=1 is WRONG for the fix (important — first proposal was incorrect)

Initial proposal was `result[dl_arr < dl_min] = 1.0`. This is wrong for two reasons:

1. **Physically incorrect**: The first bin's p_det ≈ 0.55 represents real injection statistics over (0, 0.2 Gpc). Even at d_L=0.005 Gpc, not all EMRIs are detected — sky position, inclination, and orbital parameters still matter. The SNR ∝ 1/d_L scaling helps, but the angular factors and waveform parameters can still produce SNR < threshold.

2. **Introduces a new h-dependent discontinuity**: Setting p_det=1 for d_L < dl_centers[0](h) creates a step at dl_centers[0](h) ∝ 1/h. This moving kink in the L_comp integrand would create its own h-dependent bias — potentially replacing the current bug with a different one.

The correct fix is **nearest-neighbour extrapolation**, which is what `fill_value=None` on the interpolator already provides. The zero_fill code overrides this with 0 — incorrectly.

### Mechanism for event 113 (d_L=0.085 Gpc, f_i=0.578)

L_comp integral (schematic): ∫ p_GW(d_L | 0.085, σ) × p_det_zero_fill(d_L, h) × dVc/dz dd_L

| h | dl_centers[0] | 0.085 vs threshold | p_det at 0.085 | L_comp |
|---|---|---|---|---|
| 0.73 | 0.0998 | below → 0 | 0.000 | 0.000 |
| 0.80 | 0.0924 | below → 0 | 0.000 | 0.002 |
| 0.86 | 0.0847 | above → in grid | 0.572 | 0.062 |

With nearest-neighbour fix (p_det ≈ 0.55 for d_L < dl_centers[0] at all h):

| h | L_comp (fixed) |
|---|---|
| 0.60 | 0.116 |
| 0.70 | 0.131 |
| 0.73 | 0.139 |
| 0.80 | 0.151 |
| 0.86 | 0.117 |

No longer monotonically growing — the MAP bias is removed.

### Why MAP≈0.73 before the ecliptic migration was accidental

Timeline of sky-frame state:
- **Pre-Phase-36**: CRBs had equatorial (qS, phiS). BallTree was ecliptic. BallTree received equatorial coordinates for an ecliptic galaxy catalog → searched wrong sky patches → L_cat randomised → h-dependence washed out → MAP≈0.73 (coincidentally correct, because accidental L_cat randomisation cancelled the L_comp bug).
- **Phase-36 (pos-only migration)**: CRB sky positions migrated to ecliptic. CRB covariance (sigma_qS, sigma_phiS) still equatorial. Partial fix.
- **Phase-43 (full migration, commit ab4bc80)**: CRB covariance also rotated to ecliptic frame. Sky patches now correct. L_cat now has its true h-dependence. The L_comp zero-fill bug is no longer masked → MAP=0.86 surfaces.

**Conclusion: the migration is correct. The bias was always there.**

---

## Quantitative Evidence

| Check | Finding |
|---|---|
| Events driving the bias | Events 113, 114, 108, 106 with d_L_det = 0.0854, 0.0934, 0.0946, 0.0966 Gpc |
| dl_centers[0] at h=0.73 | 0.0998 Gpc. All four events fall below. |
| dl_centers[0] at h=0.86 | 0.0847 Gpc. All four events fall above. |
| p_det for event 113 at h=0.73 (zero_fill) | 0.000 |
| p_det for event 113 at h=0.86 (zero_fill) | 0.572 |
| Log-likelihood delta event 113 | log_lk(h=0.73)=-5.01, log_lk(h=0.86)=+5.48. Delta = 10.49 log units toward h=0.86. |
| Net log-posterior shift 0.86 vs 0.73 (all events) | +145.7 log units. Four close events dominate completely. |
| First bin p_det (genuine injection statistic) | ≈ 0.55 over (0, ~0.2 Gpc). Not an edge artifact — real events in this bin. |
| D(h) variation (zero_fill, current) | 4.24e6 → 3.46e6 from h=0.60 to h=0.86. ~18% decrease. Physically correct direction. |
| D(h) variation (nearest-neighbour fix) | 5.01e6 → 3.62e6. Still decreasing. Magnitude slightly larger at low h, but same direction. |

---

## Proposed Fix

**File:** `master_thesis_code/bayesian_inference/simulation_detection_probability.py:708–713`

**Change:** Remove the left-side zero-fill. Keep only the right-side zero-fill.

```python
# OLD — zeroes both sides of the grid boundary
dl_centers = interp_1d.grid[0]
dl_min = float(dl_centers[0])
dl_max = float(dl_centers[-1])
out_of_range = (dl_arr < dl_min) | (dl_arr > dl_max)
result[out_of_range] = 0.0

# NEW — only zero above dl_max (too distant → too quiet → undetectable)
dl_centers = interp_1d.grid[0]
dl_max = float(dl_centers[-1])
result[dl_arr > dl_max] = 0.0
# d_L < dl_centers[0]: nearest-neighbour extrapolation from fill_value=None already
# returns first-bin p_det ≈ 0.55, which is the correct injection-based estimate.
```

**Physics justification:**
- `d_L > dl_max`: source beyond injection horizon → too quiet → p_det = 0. Unchanged. ✓
- `d_L < dl_centers[0]`: source is within the first bin (0, 2*dl_centers[0]). Real injections exist there. Nearest-neighbour returns first-bin p_det ≈ 0.55, which is physically appropriate. ✓
- No new discontinuity is introduced. The interpolator transitions smoothly from the first bin to the second bin.

**Reference:** The first bin covers (0, 2*dl_centers[0]) by grid construction (`linspace(0, dl_max, 61)`). p_det in that bin is computed from actual injections drawn from GLADE at low z. The zero-fill at dl_centers[0] is a bin-midpoint cutoff with no physical meaning.

**Dimensional analysis:** p_det ∈ [0,1] dimensionless. Nearest-neighbour returns ≈ 0.55 which is in range. ✓

**Limiting cases:**
- d_L → 0: p_det = first-bin value ≈ 0.55. Slightly underestimates (true p_det → 1 as d_L → 0), but all events affected by this fix are at d_L ≈ 0.085–0.097 Gpc, well above d_L=0. Acceptable approximation.
- d_L → dl_max: p_det → near 0 from interpolation. ✓
- d_L → ∞: p_det = 0. ✓

**Scope:** This function is called exclusively from `precompute_completion_denominator` (bayesian_statistics.py:119) for the D(h) denominator. It is NOT called for the per-event L_comp evaluation (that uses `detection_probability_without_bh_mass_interpolated` with fill_value=None). Wait — VERIFY THIS: confirm which function is used for L_comp vs D(h).

---

## Open Questions for Next Session

These must be answered before closing:

1. **Verify function call sites**: Confirm that `detection_probability_without_bh_mass_interpolated_zero_fill` is the function used for L_comp integration, not just D(h). Check `bayesian_statistics.py:923–945` (p_Di computation) to see which p_det method is called there. The fix scope depends on this.

2. **Cluster revert test**: Job 4159826 (evaluate) + 4159827 (combine) submitted on cluster at `/pfs/work9/workspace/scratch/st_ac147838-emri/run_equatorial_revert_20260428`. Uses `.bak_equatorial` equatorial CRBs. Expected to give MAP≈0.86 (not 0.73), confirming the bias predates the migration. Results will appear at `run_equatorial_revert_20260428/simulations/posteriors/combined_posterior.json`. Rsync and check before implementing the fix.

3. **First-bin injection count**: How many injection events fall in the first bin (0, 2*dl_centers[0]) for the production injection dataset? If very few (< ~10), p_det ≈ 0.55 is noisy and may not be reliable. If many, the fix is robust. Check `quality_flags(h=0.73)['n_total'][0]` to get the count.

4. **Post-fix verification**: After implementing the fix, run `--evaluate` locally with a synthetic dataset to confirm L_comp is now roughly h-symmetric for close events. Then submit a cluster re-evaluate job and confirm MAP shifts toward h=0.73.

5. **Physics-change protocol**: The fix modifies a detection probability computation. Requires `/physics-change` gate before implementation with the exact old/new formula, reference, dimensional analysis, and limiting case. The physics-change format above is a draft — invoke the skill formally before writing code.

6. **Residual bias check**: After the fix, MAP may not be exactly 0.73. There may be residual bias from: (a) noisy first-bin p_det, (b) the L_cat genuine h-dependence for events that are not in the close-event cluster, (c) the D(h) normalization being imperfect. If MAP is not within ±0.01 of 0.73 after the fix, a further investigation is warranted.

---

## Files Involved

| File | Role | Action needed |
|---|---|---|
| `master_thesis_code/bayesian_inference/simulation_detection_probability.py:708–713` | `detection_probability_without_bh_mass_interpolated_zero_fill` — the bug | Remove `result[dl_arr < dl_min] = 0.0` line |
| `master_thesis_code/bayesian_inference/bayesian_statistics.py:70–151` | `precompute_completion_denominator` — calls the zero_fill function | No change needed here |
| `master_thesis_code/bayesian_inference/bayesian_statistics.py:923–945` | `p_Di` / L_comp computation — need to verify which p_det method is called | Verify call site (open question 1) |

## Files Confirmed Correct — Do Not Change

- `scripts/migrate_crb_to_ecliptic.py`: transformation is mathematically correct
- `master_thesis_code/galaxy_catalogue/handler.py`: BallTree in ecliptic frame is correct
- `master_thesis_code/bayesian_inference/posterior_combination.py`: D(h) correction correctly subtracts N×log D(h)
- CRB CSV files (migrated to ecliptic): correct sky angles and covariances

---

## Phase 44 Fix Applied (2026-04-28)

**Branch:** `phase-44-pdet-zerofill-fix` (off `main`)
**Files changed:**
- `master_thesis_code/bayesian_inference/simulation_detection_probability.py` —
  removed left-side cutoff at lines 708–713; rewrote docstring to enumerate the
  6 actual call sites (was stale "exclusively for D(h)"). Added defensive
  `total_counts[0] < 100` warning in `_build_grid_1d`.
- `master_thesis_code/bayesian_inference/bayesian_statistics.py` — updated 6
  inline call-site comments to reflect the new boundary convention.
- `master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py` —
  added `TestZeroFillBoundaryConvention` (4 tests).
- `CHANGELOG.md`, `CLAUDE.md`, `.planning/STATE.md` — documentation.

**Open Question 1 resolution:** the debug session claim that the function was
called only by `precompute_completion_denominator` was WRONG. Actual call sites
verified by `grep` in `bayesian_statistics.py`: lines 119, 1006, 1178, 1198,
1427, 1440 (D(h) denominator, L_comp numerator, L_cat num/denom, +2 legacy).
All six share the function — Phase 38 STAT-03 (commit a70d1a2) deliberately
enforced this. The Phase 44 fix preserves the symmetry by editing the function
body only.

**Open Question 3 resolution:** ran `quality_flags(h=h)` at h ∈ {0.60, 0.73,
0.86} on production injections. `n_total[0] = 312` at all three (well above
100-event reliability threshold). `p̂(c_0) = 0.471, 0.545, 0.596` respectively
— smoothly varying, no h-dependent step.

**Local verification:** at `d_L = 0.085 Gpc` (the close-event d_L), p_det
varies smoothly 0.558 → 0.595 across `h ∈ [0.65, 0.86]`. Pre-fix it was 0 for
all h < 0.86 and 0.59 at h = 0.86 — that single jump drove the entire MAP bias.

**Open Questions still open:**
- 2: cluster revert (jobs 4159826/4159827) PENDING. Needed to confirm the bias
  predates the ecliptic migration (expected MAP ≈ 0.86 on `.bak_equatorial`
  CRBs).
- 4: cluster re-eval with the fix applied — pending merge.
- 6: residual bias check against MAP ∈ [0.72, 0.74] — pending cluster re-eval.

---

## Local Revert Falsifiability (2026-04-28, 23:55–00:00)

Cluster revert was effectively superseded by an in-place local test on the same
49-event production subset:

| | Events surviving zero-handling | MAP | h=0.73 vs MAP gap |
|---|---|---|---|
| Pre-fix (left-side cutoff restored) | 3 / 49 | **h = 0.86** | h=0.73 trails by +5.72 log units |
| Post-fix (current main) | 27 / 49 | **h = 0.73** | h=0.86 trails by −0.60 log units |

Same data, same code, only the boolean cutoff line differs. Pre-fix on the same
subset reproduces the MAP=0.86 pathology. Diagnosis confirmed.

The cluster revert (4159826/4159827) was therefore cancelled in favour of running
a fresh post-fix re-eval on the production seed200 (jobs 4160638/4160639).

---

## Cluster Re-Evaluation Result (2026-04-29)

Jobs 4160638 (38-task array, eval) + 4160639 (combine) on `cpu` partition,
~2:30 each. Source CRBs: `run_20260401_seed200/simulations/` (post-Phase-43
ecliptic-migrated). Results in `results/phase44_posteriors/`.

| Metric | Pre-fix | Post-fix |
|---|---|---|
| MAP h | 0.860 | **0.7650** |
| Distance from truth (h=0.73) | +0.130 (+18%) | +0.035 (+5%) |
| Events with zero likelihoods | many (per handoff) | 0 |
| Strategy variance (4 zero-handling strategies) | strategies disagreed | all 4 agree exactly |
| 68% equal-tailed interval | (not reported) | [0.750, 0.765] |

Headline: catastrophic +145.7 log-unit pathology eliminated; all 4 zero-handling
strategies now produce identical MAP, confirming no events are being suppressed
by zero-handling logic. The fix is unambiguously a net win.

**Residual bias +0.035 deferred** to a follow-up phase (Phase 45 candidate). Per
the Phase 44 plan §8 fallback table, this places us in the
"MAP ∈ [0.74, 0.79]" regime whose recommended remedy is Alternative C — add an
explicit `(d_L=0, p_det=interp(c_0))` anchor to refine first-bin behavior. The
hypothesis is that p̂(c_0) ≈ 0.55 underestimates the true p_det → 1 limit at
d_L → 0, biasing L_comp low at low h (where c_0 is largest) and pushing MAP
toward higher h.
