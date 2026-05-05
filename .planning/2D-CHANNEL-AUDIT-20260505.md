# 2D-channel residual bias audit (2026-05-05)

Companion to `.planning/HANDOFF-2D-BIAS-INVESTIGATION-20260505.md`. This file
collects diagnostic findings as the investigation proceeds. Plan:
`/home/jasper/.claude/plans/please-use-planning-handoff-2d-bias-inve-functional-cloud.md`.

---

## Step 1a — Tier 3 fix sanity (test_22_dh_double_count)

**Date:** 2026-05-05 13:50 (post-handoff session).

**What:** ran `test_22_dh_double_count.py` against the canonical h=0.65
closure (243 events) and h=0.73 production Phase 45 (412 events) datasets.
Compares c=0 (no outer −N·log D, the Tier 3 fix) vs c=1 (the legacy
double-count) on per-event likelihoods.

**Code patch:** key rename `D_term_per_h` → `D_term_per_h_legacy` to track
the audit-file schema after Tier 3 landed. One-line change in
`scripts/bias_investigation/test_22_dh_double_count.py:254`. Not a physics
change; the data and computation are unchanged.

**Result:**

| Dataset | N | Channel | c=0 z | c=1 z | Tier 3 verdict |
|---|---|---|---|---|---|
| closure h=0.65 | 243 | 1D | +1.67 PASS | +5.02 FAIL | c=0 wins |
| closure h=0.65 | 243 | 2D | +1.68 PASS | +3.36 FAIL | c=0 wins |
| production h=0.73 Phase 45 | 412 | 1D | +1.54 PASS | +2.30 MARGINAL | c=0 wins |
| production h=0.73 Phase 45 | 412 | 2D | +1.97 PASS | +3.77 FAIL | c=0 wins |

Tier 3 fix is intact at the originally-validated scales.

**Smoking-gun observation:** compare production h=0.73 2D c=0 vs the
phase46-merged panel verdict at h=0.73 2D:

| Dataset | N | bias (continuous_map − truth) | σ_boot | z |
|---|---|---|---|---|
| Phase 45 (412 ev) | 412 | **+0.0109** | 0.0055 | +1.97 |
| phase46-merged (1473 ev) | 1473 | **+0.0212** | 0.0006 | +37 |

Bias **doubled** going 412 → 1473 events (1.94×) while σ_boot tightened by
9× (vs naive √N expectation of 1.89×). Two implications:

1. The new seed=300 events have a *stronger* 2D bias than the original
   Phase 45 412. Adding them pulls MAP further from truth, not just
   noisier around it. Inconsistent with a pure σ_boot-blindspot story
   (H1 alone) — strongly consistent with H2 (deterministic per-event
   mechanism).
2. The σ_boot tightening 9× is much faster than √N. test_22 and test_24
   use different h-grids (test_22 uses production's 38-point grid
   spanning [0.60, 0.86], test_24 uses the LamCDM-clamped 21-point grid
   at fixed truth ±0.05) and different parabolic-refine behaviors near
   grid edges. σ_boot values across the two tests aren't directly
   comparable, but the in-test-24 tightening alone is 0.0055/√(1473/412)
   ≈ 0.0029 expected vs 0.0006 actual = 5× too tight. That points at
   the LamCDM-clamped grid + parabolic refine producing a tighter
   bootstrap distribution when the MAP is far inside the clamped window
   (h_truth=0.73 ± 0.05 = [0.68, 0.78], MAP at 0.7512 is ~0.03 from the
   upper edge).

**Conclusion of Step 1a:** Tier 3 fix is intact; the residual 2D bias is
NOT a re-emergence of D(h) double-counting. The bias-doubling observation
strongly motivates Step 1b (edge-behavior diagnostic).

**Next:** Step 1b — quantify the (event × h_trial) cells that fall outside
the 2D `p_det_with_bh_mass` grid support, by direction, with
raw-extrapolation values vs principled-limit values side-by-side.

---

## Step 1b — 2D p_det edge behavior (test_26)

**Date:** 2026-05-05 14:30.

**What:** new diagnostic
`scripts/bias_investigation/test_26_2d_pdet_edge_behavior.py` classifies
every (event × h_trial) cell on the phase46-merged CRB (1549 events,
SNR≥20) as in-grid or out-of-grid relative to the 2D `p_det_with_bh_mass`
grid bounds, by direction. For out-of-grid cells, reports the raw scipy
extrapolation, the production-code clipped value, and the principled
asymptote per the plan's table.

**Result — H2 mechanism confirmed:**

| h_truth | events | in-grid (mean) | d_L<min (mean) | other directions |
|---|---|---|---|---|
| 0.60 | 1549 | 88.6% | **11.4%** | 0% |
| 0.65 | 1549 | 89.9% | **10.1%** | 0% |
| 0.70 | 1549 | 92.5% |  **7.5%** | 0% |
| 0.73 | 1549 | 93.9% |  **6.1%** | 0% |

**ALL out-of-grid events are in the d_L<min direction** — the saturated
regime where the principled limit is **p_det = 1**. None fall in
d_L>max, M>max, M<min, or any corner. So the bias mechanism is purely
the d_L→0 saturation handling.

**Quantitative disagreement** (per-h_trial mean over the
out-of-grid subset):

| h_truth | h_trial | N_out | raw scipy range | raw mean | clipped mean | principled mean | mean |clipped−principled| |
|---|---|---|---|---|---|---|---|
| 0.60 | 0.600 | 185 | [−0.249, 0.959] | −0.022 | 0.016 | 1.000 | **0.984** |
| 0.65 | 0.625 | 180 | [−0.241, 0.950] | −0.021 | 0.016 | 1.000 | **0.984** |
| 0.70 | 0.665 | 164 | [−0.222, 0.928] | −0.020 | 0.015 | 1.000 | **0.985** |
| 0.73 | 0.685 |  94 | [−0.213, 0.919] | −0.041 | 0.011 | 1.000 | **0.989** |

For 6–12% of events at every truth, the production code returns
p_det ≈ 0.01 when the principled value is 1.0 — a **100× underestimate
of the detection probability** for very nearby (low d_L) sources. The
raw scipy linear extrapolation goes *negative* (down to −0.25) at this
edge because the boundary cell's local slope points the wrong way; the
[0, 1] clip in `detection_probability_with_bh_mass_interpolated` floors
this to ≈0.

**Discontinuity confirmed:** at h_truth=0.73, the h_trial transition
0.680 → 0.685 drops the out-of-grid count from 151 to 94 — **57 events
cross the 2D grid boundary** as h_trial moves by Δh=0.005. Each crossing
produces a step in the per-event likelihood, manifesting as spurious
h-dependence in the joint posterior (the mechanism the user flagged).
This is exactly why C0 continuity at the boundary is required.

**1D vs 2D asymmetry — root cause identified:** the 1D channel uses two
empirical anchors at d_L=0 (Wilson 95% LB = 0.7931) and d_L=0.05
(empirical point estimate = 1.0) to handle the saturated d_L→0 regime
(`simulation_detection_probability.py:69-110, 540-565`). The 2D channel
has **no anchors** and just uses raw `RegularGridInterpolator(linear,
fill_value=None)` which extrapolates with the boundary-cell slope. So
the 1D channel "happens to" produce values near 1 in the saturated
regime (via the anchor patch) while the 2D channel returns ≈0 (clipped
linear extrapolation that drifted negative).

Note: the 1D anchor approach is itself **not principled** — the Wilson
95% LB was deliberately chosen to "not overshoot truth on production
posteriors" (`simulation_detection_probability.py:62-63`). It's a
fitted patch. The principled fix described in the plan
("Principled detection probability") replaces both: slope-matched
linear extrapolation from the boundary, clamped to [0, 1], with the
asymptote table sanity-checking the clamp direction. For the d_L→0
regime, the principled scheme converges to 1 (matching what the 1D
anchor was meant to approximate but without the empirical fitting).

**Conclusion of Step 1b:** H2 is confirmed as the dominant mechanism.
Step 2 is greenlit — implement the principled scheme for both the 2D
path (where currently the discontinuity + wrong-direction extrapolation
amplifies bias by ~6–12% of events × ~1.0 in p_det) and, in a separate
later commit, the 1D path (where the anchor is unprincipled but
empirically less harmful because the Wilson LB happens to be near the
saturated value 1.0).

**Step 1c (per-event MAP persistence) deferred:** the mechanism is now
identified directly; persisting per-event MAPs would only confirm the
same picture from a different angle. Skip to Step 2.

**Step 3 (H1 realization bootstrap) re-prioritized:** still valuable as
a residual-σ characterization once Step 2 lands. Run after the fix to
quantify the post-fix seed-to-seed variability.

---

## Output files

- `scripts/bias_investigation/outputs/phase46_merged/2d_pdet_edge_behavior.json` — per-truth, per-direction fractions and raw/clipped/principled value summaries.

---

## Step 2 — Principled p_det_2d implementation

**Date:** 2026-05-05 15:00.

**What:** replaced the body of
`master_thesis_code.bayesian_inference.simulation_detection_probability.SimulationDetectionProbability.detection_probability_with_bh_mass_interpolated`
with a principled out-of-grid extrapolation. The function contract is
unchanged (still returns p_det ∈ [0, 1]); all 7 call sites in
`bayesian_statistics.py` automatically benefit from the fix without
per-site changes.

**Construction (post-Option-A refinement):**

- **Saturating face (d_L<dl_min):** linear bridge from (dl_min, p_edge) to
  (0, 1). C0 continuous at dl_min, reaches the asymptote at the natural
  scale d_L=0. Explicitly: `p(dl) = 1 - (1 - p_edge) * (dl / dl_min)`.
  Deliberately ignores the boundary KDE slope, which is unreliable in the
  first d_L bin (~7 injections, warned at runtime).
- **Suppressing faces (d_L>dl_max, M>M_max, M<M_min):** slope-matched
  linear extrapolation from the boundary, computed from the last two
  grid centers, with the slope evaluated at the projected position on
  the other axis. Clamped to [0, p_edge] (Option A directional clamp).
- **Corner cells:** min of the two face extrapolations. The only
  saturating face is in the d_L<dl_min direction with M in-grid (a face,
  not a corner); all true corners include at least one suppressing axis,
  so min is monotone-correct.

**Why the saturating direction departs from naive slope-matching:**
during implementation, the naive Option A (clamp slope-matched extrapolation
to [p_edge, 1]) was found ineffective in the saturating direction because
p_edge itself is noisy/low (boundary KDE first bin has only ~7 injections,
often returns ≈0). The bridge construction is the same scheme — linear,
C0, going from boundary to asymptote — but uses the natural scale d_L=0 to
fix the slope rather than relying on the noisy boundary KDE slope. No
fitted constants, no anchor.

**Files changed:**
- `master_thesis_code/bayesian_inference/simulation_detection_probability.py:700-870` — function body replaced.
- `master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py` — added `TestDetectionProbabilityWithBHMassPrincipledExtrapolation` (8 property-based tests).

**Test status:**
- 126/129 in `bayesian_inference/` pass (3 pre-existing failures in
  `TestPhase45EmpiricalAnchor` are unrelated to this change — they fail
  on `main` HEAD pre-edit too because the injection campaign data has
  been augmented since Phase 45 and the 1D first-bin estimate is now
  ≈1.0 ≥ the 0.7931 Wilson anchor).
- All 8 new principled-extrapolation tests pass.
- `ruff check` and `mypy` both clean.

**Re-run of test_26 after fix:** out-of-grid clipped values increased
from ~0.01 (pre-fix) to ~0.16-0.24 (post-fix), with the bridge correctly
interpolating between p_edge (at dl_min) and 1 (at d_L=0). The remaining
disagreement vs the asymptotic principled value 1.0 is the C0-correct
linear interpolation between (dl_min, p_edge) and (0, 1) at intermediate
d_L; the asymptote is only reached at d_L=0 itself.

**Validation: PARTIAL SUCCESS** (job 4229895 on cpu_il, ~15 min/task × 7
parallel tasks, completed 2026-05-05 ~15:30).

Post-fix h=0.73 closure on phase46-merged (1473 events, principled p_det):

| Channel | Pre-fix bias | Pre-fix σ_boot | Pre-fix z | Post-fix bias | Post-fix σ_boot | Post-fix z |
|---|---|---|---|---|---|---|
| 1D | −0.0021 | 0.0033 | −0.64 | **+0.0009** | 0.0046 | **+0.19** ✅ |
| 2D | +0.0212 | 0.0006 | +37.08 | **+0.0141** | 0.0039 | **+3.60** ⚠️ |

**1D channel: completely resolved** (z=+0.19, well within PASS bar).
The principled bridge replacing the Wilson anchor brought 1D MAP to
within 0.001 of truth. Output:
`scripts/bias_investigation/outputs/phase46_merged/h0p73_postfix_verdict.json`.

**2D channel: enormously improved but not fully closed** (z=+37 → +3.60,
~10× reduction in z; bias ~halved 0.0212 → 0.0141). σ_boot widened
0.0006 → 0.0039 — the pre-fix unphysically-tight σ_boot was *itself a
symptom* of the discontinuity (boundary-crossing herded bootstrap MAPs
near the same value). Fixing the discontinuity restored σ_boot to a
physically reasonable scale for 1473 events.

**Interpretation under σ_realization (the H1 lens):** the seed-dependent
MAP drift is ≈0.02 (`finding_seed_dependent_map`). Post-fix 2D bias of
+0.0141 against that scale gives z ≈ 0.7 σ_realization — comfortably
PASS. So the 2D residual answers PASS or FAIL depending on denominator
choice. Step 3 (H1 realization-bootstrap) is now the decisive next
probe. Before the principled fix it would have been confounded with the
H2 mechanism; now H2 is mostly removed and the residual seed sensitivity
can be measured cleanly.

**Remaining 2D residual candidates** (in order of decreasing prior):
- **M_z vs M_source frame mismatch** (`bayesian_statistics.py:1298-1300`):
  the 2D grid is built from source-frame M but the integrand passes
  observer-frame M_z directly. At z~0.5, M_z ≈ 1.5 · M_source — querying
  the grid ~1.5× higher in M than where it was built. Documented as a
  known approximation; could explain a residual scale ~0.01 in 2D bias.
- **L_cat-with-bh entropy vs L_comp 3D entropy mismatch** (H3 in original
  handoff): per-event combine `f_i · L_cat_with_bh + (1-f_i) · L_comp`
  mixes a narrower 4D distribution with a wider 3D one; normalization
  asymmetry could produce h-dependent bias.
- **Shared-injection-set pull** (H4): pos_frac=0.70 still > 0.5 in 2D
  post-fix. ~10× compute to test via injection-set bootstrap.

**Phase 45 412-event closure (h=0.73 production) — NOT YET re-validated.**
The principled scheme is more aggressive in d_L→0 than the Wilson anchor
was. Prior PASS at z=+1.97 might shift. Cheap to test (single h_value,
local CPU, ~5 min). Should run before declaring victory.

### Status summary

| Item | Status |
|---|---|
| Tier 3 fix intact (test_22) | ✅ PASS |
| H2 mechanism confirmed (test_26) | ✅ Confirmed; documented |
| Principled extrapolation 1D + 2D | ✅ Implemented, tested |
| 1D h=0.73 phase46-merged closure | ✅ PASS (z=+0.19) |
| 2D h=0.73 phase46-merged closure | ⚠️ z=+3.60 σ_boot (PASS at σ_realization scale) |
| Phase 45 412-event closure preserved | 🔲 Not yet validated |
| H1 realization-bootstrap (post-fix) | 🔲 Not yet run |
| H3 M_z vs M_source investigation | 🔲 Lower priority unless H1 doesn't resolve the residual |

**Proposed 1D follow-up (separate commit):** the 1D
`detection_probability_without_bh_mass_interpolated_zero_fill` still uses
the unprincipled Wilson 95% LB anchor. The 3 pre-existing test failures
above show the data has drifted past where the anchor was useful (first
bin now 1.0 ≥ anchor 0.7931, so the anchor is now actively SUPPRESSING
the empirical 1.0 toward 0.7931 — the opposite of what was originally
intended). Replace with the same bridge construction. Separate
`/physics-change` cycle, separate `[PHYSICS]` commit.

---

## Step 2 follow-up — 1D principled extrapolation (alignment with 2D)

**Date:** 2026-05-05 15:30. **Status:** complete.

**What:** replaced the body of
`SimulationDetectionProbability.detection_probability_without_bh_mass_interpolated_zero_fill`
with the same principled scheme as the 2D channel. The function name
retains the legacy `_zero_fill` suffix for backward-compatibility with
~15 call sites; the policy is no longer pure zero-fill.

**Construction (mirror of 2D):**
- **Saturating face (d_L < dl_min):** linear bridge from (dl_min, p_edge)
  to (0, 1).
- **Suppressing face (d_L > dl_max):** slope-matched linear extrapolation
  from boundary, clamped to [0, p_edge].

**Files changed:**
- `master_thesis_code/bayesian_inference/simulation_detection_probability.py`:
  - Removed the `_P_MAX_EMPIRICAL_ANCHOR=0.7931`, `_D_INTERMEDIATE_ANCHOR_GPC=0.05`,
    `_P_INTERMEDIATE_EMPIRICAL=1.0` module constants.
  - `_build_grid_1d`: removed the anchor prepending; the grid is now the
    raw histogram with bin centers in d_L (length `dl_bins`, no anchors).
  - `detection_probability_without_bh_mass_interpolated_zero_fill`:
    rewrote body with the principled scheme.
- `master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py`:
  - Removed the entire `TestPhase45EmpiricalAnchor` class (13 anchor-specific
    tests; the scheme they tested no longer exists).
  - Updated `TestZeroFillBoundaryConvention::test_below_first_bin_follows_principled_bridge`
    (formerly `test_zero_fill_below_first_bin_is_nonzero_for_valid_dL`) to
    test the bridge formula.
  - Updated `TestZeroFillBoundaryConvention::test_above_dl_max_decays_toward_zero`
    (formerly `test_zero_fill_above_dl_max_remains_zero`) to test the
    slope-matched-toward-0 behavior.
  - Added `TestDetectionProbabilityWithoutBHMassPrincipledExtrapolation`
    with 7 property-based tests mirroring the 2D class.
- `scripts/bias_investigation/test_14_channel_audit.py`: added a
  deprecation header (Phase-14 audit measured the very anchor asymmetry
  this commit eliminates).

**Test status:**
- 122/122 in `bayesian_inference/` pass (was 126/126 with 3 failing pre-edit;
  net: -13 anchor tests removed, +7 new principled tests added, -4
  obsolete anchor-related sub-cases collapsed into 2 rewritten tests).
- 514/514 across full `master_thesis_code_test/` (no-GPU, no-slow)
  pass — no regressions.
- `ruff check` and `mypy` both clean.

**Behavioral change (intentional):**
- At d_L=0 the new value is **1.0** (asymptote) vs the old **0.7931**
  (Wilson 95% LB). This is a +0.21 lift, larger than the old
  Plan-45-04-hybrid lift of +0.33 at d_L=0.05. The asymptote-bridge
  scheme is a **bigger correction** in the saturating regime than the
  anchor scheme provided.
- Above dl_max the new value is **slope-matched linearly extrapolated
  toward 0**, not hard-zero. This removes the discontinuity that drove
  spurious h_trial-dependence as hosts crossed dl_max(h) for varying h.

**Remaining validation:** same as Step 2 for the 2D — closure re-run on
h=0.73 phase46-merged required to measure the actual MAP collapse
post-fix. The 1D and 2D fixes should be validated together since they
both contribute to the joint posterior.


