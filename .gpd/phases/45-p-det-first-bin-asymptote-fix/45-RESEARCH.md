# Phase 45 Research — P_det First-Bin Asymptote Fix

**Status:** RESEARCH COMPLETE — diagnosis was performed as a bootstrap-first
kickoff (Step 0 + Step 1, see `.gpd/HANDOFF-phase45-diagnosis.md`). This
RESEARCH.md is a planning-ready summary for `gpd-planner`.

---

## 1. Problem statement

Phase 44 fixed the h-dependent zero-fill cutoff at `dl_centers[0] ∝ 1/h` in
`detection_probability_without_bh_mass_interpolated_zero_fill`
(commit `3697bdd`). Cluster re-eval on production seed200 (412 events,
truth h=0.73, SNR≥20) gave **MAP = 0.7650** vs **target ∈ [0.72, 0.74]**.
The residual **+0.0350 (3 σ_boot)** is *systematic*, not statistical.

## 2. Root cause (locked from Phase 45 kickoff diagnostics)

| Diagnostic | Result | Implication |
|---|---|---|
| Bootstrap (B=1000, T8) | 68% [0.745, 0.765] excludes 0.73; σ_boot=0.0114 | systematic |
| First-bin density (T9) | upper-skew ratio 3.22 (29 vs 9 in upper/lower thirds; weighted mean d_L=0.132 in [0, 0.20] bin) | `p̂(c_0)` averages high-d_L (low-p_det) events |
| Empirical asymptote (T10) | **16/16 detected for d_L<0.10 Gpc, p̂=1.000** [0.806, 1.000] Wilson 95% | true p_det → 1 as d_L → 0 |
| Interpolator query (T10) | `interp(0.001) = 0.7476`, `interp(0.10) = 0.5444` (linear extrapolation through bins 0,1) | **scipy `fill_value=None` does linear extrapolation, NOT nearest-neighbour** as docstring at L722–727 claims |
| Window proximity (T11) | 0/60 events touch d_L=0; **26/60 (43%) cross c_0 ≈ 0.10 Gpc** | Anchor-at-zero is *inert*; fix must live in `[0, c_0]` |

**Mechanism:** the histogram-derived first-bin estimate `p̂(c_0) ≈ 0.544`
underestimates p_det in `[0, c_0]` because the upper-skewed injection
density makes `p̂` an average over events near 0.13 Gpc, not near 0. Linear
extrapolation through bins 0,1 partly recovers the upward trend (reaching
~0.75 at d_L=0) but is still 0.25 below the empirical asymptote 1.0.
26/60 events have their 4σ integration window crossing into this region,
so L_comp is suppressed at low h, biasing MAP upward.

## 3. Phase 44 plan §8 was wrong

Plan §8 recommended *"Alternative C — explicit `(d_L=0, p_det=interp(c_0))`
anchor"*. **Two errors:**

1. Anchoring at `interp(c_0)` is a no-op: it equals `p̂(c_0)`, so the
   prepended point either preserves the existing slope or flattens to a
   constant — neither moves the interpolator toward the empirical truth.
2. The plan assumed nearest-neighbour fill below `c_0`. Empirically (T10)
   it's linear extrapolation. Even so, the value at d_L→0 reaches only
   ~0.75 vs empirical 1.0.

The docstring at `simulation_detection_probability.py:722–727` repeats the
incorrect "nearest-neighbour" claim and should be corrected as part of the
patch.

## 4. Proposed fix family (planner to choose)

All candidates lift the interpolator in `[0, c_0]`. **Constraint:** the
chosen value must be **h-independent** (computed once from the unrescaled
injection campaign, applied at every target h) — see Phase 44 regression
test `test_zero_fill_no_h_dependent_step_for_close_dL` which guards
`max_diff < 0.20` across h ∈ {0.65, …, 0.85}.

### 4a. Empirical anchor at `(0, p_max_empirical)` [simplest]

Prepend `(0.0, p_max_empirical)` to `(dl_centers, p_det_1d)` in
`_build_grid_1d` (L518–522) before constructing the
`RegularGridInterpolator`. Three sub-options for the anchor value:

- **(i)** `p_max_empirical = 1.0` (point estimate from 16/16 in `[0, 0.10]`)
- **(ii)** `p_max_empirical = 0.806` (Wilson 95% lower bound — conservative)
- **(iii)** `p_max_empirical = 0.95` (a compromise; explicitly halfway)

Effect on `interp(0.05)`: rises from current ~0.65 to ~0.78–0.95 depending
on choice.

### 4b. Sub-bin the first p_det bin

Replace `dl_edges = np.linspace(0, dl_max, 61)` with custom edges that
split `[0, 2c_0]` into 4 sub-bins. Each sub-bin gets its own histogram
p̂. **Risk:** the smallest sub-bins may have <10 injections (only 16 events
in `[0, 0.10]` total) → noisy. A Wilson-CI floor would mitigate. More
code surface than 4a.

### 4c. Hybrid (anchor + first-bin sub-binning)

Combine 4a (anchor at 0) with one intermediate point at `(c_0/2, p̂_split)`
where `p̂_split = 16/16 = 1.0` from the d_L<0.10 subset. Most physically
defensible but most code surface.

**Recommended for planning:** start with **4a-(ii)** (Wilson lower bound
0.806, conservative). It's a single-line surgical change; if the cluster
re-eval acceptance test fails, escalate to 4c.

## 5. Acceptance criteria (locked-in success metrics)

- Cluster re-eval on production seed200 (412 events, truth h=0.73)
  produces **MAP ∈ [0.72, 0.74]** AND 68% equal-tailed interval
  **contains 0.73**. Use `cluster/submit_phase44_eval.sh` as template.
- All 4 `TestZeroFillBoundaryConvention` regression tests pass.
- 557 CPU tests still pass.
- The 4 zero-handling strategies (`naive`, `exclude`, `per-event-floor`,
  `physics-floor`) continue to produce identical MAPs (Phase 44 invariant).
- Anchor value is **h-independent** — verified by checking that
  `max p_det(d_L=0.05; h) − min p_det(d_L=0.05; h)` over h ∈ {0.65, …, 0.85}
  is below 0.05 (was 0.20 budget in T_zero_fill_no_h_dependent_step).
- Docstring at `simulation_detection_probability.py:722–727` corrected to
  state "linear extrapolation" not "nearest-neighbour".

## 6. Out of scope (defer to Phase 46+)

- 4 MEDIUM physics caveats (wCDM hardcoded LCDM, hardcoded 10% σ_dL in
  Pipeline A, WMAP-era cosmology, redshift uncertainty `(1+z)^3`).
- 2D BH-mass variant `detection_probability_with_bh_mass_interpolated`
  (orthogonal; not in the production likelihood path).
- L_cat changes, sky-frame work, simulation campaign re-run.

## 7. Files in scope

| Path | Role |
|---|---|
| `master_thesis_code/bayesian_inference/simulation_detection_probability.py:466-528` | `_build_grid_1d` — anchor surface |
| `master_thesis_code/bayesian_inference/simulation_detection_probability.py:698-764` | Phase 44 fixed `_zero_fill` — docstring at L722–727 to correct |
| `master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py::TestZeroFillBoundaryConvention` | 4 regression tests; must continue to pass |
| `master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py` | new test for `(0, p_max_empirical)` anchor presence |
| `cluster/submit_phase45_eval.sh` (NEW) | clone of `submit_phase44_eval.sh` |
| `.gpd/HANDOFF-phase45-diagnosis.md` | full diagnostic context (this RESEARCH summarizes) |
| `scripts/bias_investigation/outputs/phase45/` | T8–T11 numeric outputs |

## 8. Physics-change protocol prerequisites

This is a physics change (modifies a probability function's behavior near
d_L=0). Per `CLAUDE.md` `/physics-change` protocol the plan must include:

- **Old formula:** `p_det(d_L < c_0) = linear_extrap(bins[0], bins[1])`
- **New formula:** `p_det(d_L < c_0) = linear_interp((0, p_max_empirical), (c_0, p̂(c_0)))`
- **Reference:** Step 1b empirical asymptote, T10 outputs at
  `scripts/bias_investigation/outputs/phase45/pdet_asymptote.json`
- **Dimensional analysis:** probability ∈ [0, 1] dimensionless; preserved.
- **Limiting case:** `p_det(d_L → 0) → p_max_empirical` (matches injection
  campaign asymptote at h_inj=0.73)

## 9. Risk register

| Risk | Mitigation |
|---|---|
| Anchor value mis-estimates true asymptote at high h_inj groups | Recompute `p_max_empirical` across all h_inj groups (not just 0.73) before locking |
| Fix overshoots → MAP < 0.72 | Acceptance window is [0.72, 0.74]; cluster re-eval is the gate |
| Phase 44 regression test `test_zero_fill_no_h_dependent_step_for_close_dL` fails because anchor is constant but `p̂(c_0)` is not | Anchor-at-0 doesn't change `p̂(c_0)`; the h-dependent spread comes from `p̂(c_0)` itself, not the anchor. Test budget is 0.20 (well above 0.05 expected) |
| Linear interpolation between `(0, p_max)` and `(c_0, p̂(c_0))` introduces a slope discontinuity at `c_0` | Acceptable — the discontinuity is in the slope, not the value, so the integrand is continuous |

## 10. RESEARCH COMPLETE

Diagnostics established the mechanism, ruled out the wrong fix (anchor at
`interp(c_0)`), and identified the correct fix family with a recommended
default. Planner can proceed without re-research.
