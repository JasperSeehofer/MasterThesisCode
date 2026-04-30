# Phase 45 Diagnosis Lock-In — Residual MAP Bias

**Status:** diagnosis locked. Ready to invoke `/gpd:plan-phase 45` with the
findings below. Do **not** inherit Phase 44 plan §8's "Alternative C — anchor
at `(d_L=0, p_det=interp(c_0))`" wording; that fix is **inert** on the
production posterior (Step 1c). The correct fix lives in `[0, c_0]`, not at
`d_L = 0`.

---

## Verdict tree

| Step | Output | Verdict |
|---|---|---|
| **0 — Bootstrap** | `outputs/phase45/bootstrap_summary.json` | **systematic** (interval_68 = [0.745, 0.765] excludes truth 0.73 by ~3 σ_boot=0.0114; full-sample MAP=0.7650 reproduces cached) |
| **1a — First-bin density** | `outputs/phase45/first_bin_density.json` | **upper-skew** (n_upper3 / n_lower3 = 29 / 9 = 3.22; weighted mean d_L in bin = 0.132 Gpc > midpoint 0.10) |
| **1b — p_det asymptote** | `outputs/phase45/pdet_asymptote.json` | **interpolator underestimates by ~0.45** at c_0; empirical 16/16 = 1.000 [0.806, 1.000] for d_L < 0.10 Gpc; interpolator returns 0.544 at c_0, ~0.75 at d_L → 0 (linear extrapolation, NOT nearest-neighbour as Phase 44 plan §8 assumed) |
| **1c — Window proximity** | `outputs/phase45/window_proximity.json` | **0/60 events touch d_L=0** but **26/60 (43%) integrate across c_0 ≈ 0.10 Gpc**. Anchor-at-zero is inert; the fix must lift `[0, c_0]` |

Decision-tree row that fired: **Systematic + first-bin upper-edge skew** →
"Implement a corrected first-bin treatment with empirical anchor". The exact
form differs from Phase 44 plan §8 — see "Plan-phase instruction" below.

---

## Headline numbers (cite verbatim in `/gpd:plan-phase 45`)

| Quantity | Value | Source |
|---|---|---|
| Truth h | 0.7300 | injection seed |
| Full-sample MAP | 0.7650 | reproduced from cache; `combined_posterior.json` |
| Bootstrap σ_MAP | 0.0114 | B=1000 resamples |
| 68% bootstrap interval | [0.745, 0.765] | excludes truth |
| 95% bootstrap interval | [0.745, 0.765] | (= 68% because h-grid is discrete; MAPs cluster at grid points 0.745, 0.760, 0.765) |
| Residual / σ_boot | +0.0350 / 0.0114 ≈ **3.07 σ_boot** | not statistical |
| First-bin upper-skew ratio | 29 / 9 = **3.22** | n_upper3 / n_lower3 in [0, 0.20] Gpc, h_inj=0.73 |
| Weighted mean d_L in first bin | **0.132 Gpc** (vs midpoint 0.10) | confirms upper-edge dominance |
| **Empirical p_max** at d_L < 0.10 Gpc | **1.000 [0.806, 1.000]** (Wilson 95%) | 16/16 detected, h_inj=0.73 |
| Interpolator p_det at c_0=0.10 | 0.5444 | scipy linear, fill_value=None |
| Interpolator p_det at d_L=0.001 | 0.7476 | linear extrapolation through bins 0,1 |
| Underestimate at d_L → 0 | 1.000 − 0.748 = **0.25** | bias driver |
| Underestimate at c_0 | 1.000 − 0.544 = **0.46** | bias driver |
| Events with window touching d_L=0 | **0 / 60** (local proxy) | Anchor-at-zero is inert |
| Events with window crossing c_0 | **26 / 60 (43%)** (local proxy) | This is where the bias lives |
| Min d_L/σ | 13.5 | closest event; far above the 4 needed for zero-touch |
| Median σ/d_L | 0.046 | typical Fisher precision; threshold is 0.25 |

---

## What Phase 44 plan §8's "Alternative C" got wrong

Plan §8 said: *"explicit `(d_L=0, p_det=interp(c_0))` anchor"*. Two errors:

1. **Anchoring at `interp(c_0)` is a no-op.** Linear extrapolation through
   `(c_0, p̂(c_0))` and the prepended `(0, p̂(c_0))` is constant at `p̂(c_0)`
   on `[0, c_0]` — but `interp(c_0)` is already `p̂(c_0)`, so prepending
   `(0, p̂(c_0))` either reproduces the existing extrapolation slope or
   flattens it; either way the value at production-relevant d_L's
   (0.04–0.10 Gpc) does not move toward the empirical truth (1.0).
2. **Phase 45 evidence: scipy's `fill_value=None` is *linear extrapolation*
   in `RegularGridInterpolator(method="linear")`, NOT nearest-neighbour.**
   Both the handoff §H1 description and the Phase 44 docstring at
   `simulation_detection_probability.py:722–727` say "nearest-neighbour from
   `RegularGridInterpolator(fill_value=None)` returns the first-bin estimate
   `p̂(c_0)`". Empirically (Step 1b query):
   `interp(0.001) = 0.7476 ≠ p̂(c_0) = 0.5444`. The docstring/handoff
   description is wrong; the interpolator linearly extrapolates through the
   first two bins. Plan-phase should fix the docstring as part of the patch.

---

## Plan-phase instruction (`/gpd:plan-phase 45`)

**Goal:** lift the p_det interpolator in `[0, c_0]` so that production events
with their 4σ window crossing this region see a value consistent with the
empirical asymptote, while keeping the fix h-independent and preserving all
4 `TestZeroFillBoundaryConvention` regression tests.

**Candidate fixes (rank-ordered by simplicity; pick during plan-phase):**

1. **Sub-bin the first p_det bin.** Replace `dl_edges = np.linspace(0, dl_max, 61)`
   with a finer first-bin grid (e.g. split `[0, 2c_0]` into 4 sub-bins by
   prepending `[0, c_0/2, c_0, 1.5·c_0]` to the existing edges). Each
   sub-bin gets its own histogram-derived p̂. Risk: small sub-bins may have
   <10 injections → noisy. Wilson-CI floor is needed.

2. **Empirical anchor at `(0, p_max_empirical)`.** Prepend `(0.0, 1.0)` (or
   `(0.0, 0.806)` Wilson 95% lower bound for conservatism) to
   `(dl_centers, p_det_1d)` in `_build_grid_1d` (L518–522). Linear interp
   between this anchor and the existing first bin center. Effect on
   `interp(0.05)`: rises from ~0.55 to a value between 0.81 and 1.0
   depending on chosen anchor.

3. **Hybrid:** anchor at `(0, p_max_empirical)` AND insert a `(c_0/2, p̂_split)`
   intermediate point computed from the d_L < c_0 subset specifically
   (16 injections at h_inj=0.73, all detected → p̂_split=1.0). This is the
   most physically defensible, but requires more code surface.

**Constraints (mandatory):**

- The anchor must be **h-independent**. Compute `p_max_empirical` once from
  the unrescaled injection campaign (`d_L < 0.10 Gpc` subset across all
  h_inj groups) and reuse the same scalar for every target h.
- The 4 `TestZeroFillBoundaryConvention` regression tests must continue to
  pass:
  - `test_zero_fill_below_first_bin_is_nonzero_for_valid_dL` — value > 0 ✓
  - `test_zero_fill_no_h_dependent_step_for_close_dL` — h-spread < 0.20 ✓
    (anchor is h-independent, so spread is bounded by the linear-interp
    slope difference across h's; should remain well within the 0.20 budget)
  - `test_zero_fill_above_dl_max_remains_zero` — unchanged ✓
  - `test_zero_fill_symmetry_invariant` — unchanged ✓
- The fix lives in `_build_grid_1d` (L518–522) only. All 6 production call
  sites in `bayesian_statistics.py` pick it up via the cached interpolator.
- Under `/physics-change` protocol: present old formula, new formula,
  reference to Step 1b empirical anchor as derivation, dimensional analysis
  (probability ∈ [0,1]), and a limiting case (d_L → 0 → p_max).
- Run the cluster re-eval template (`cluster/submit_phase44_eval.sh`) after
  the fix; acceptance: MAP ∈ [0.72, 0.74] AND 68% interval contains 0.73.

**Out of scope (defer to Phase 46+):**

- Docstring fix at L722–727 (nearest-neighbour vs linear): bundle with the
  patch but as a separate commit.
- The 4 MEDIUM physics caveats (wCDM, 10% σ_dL, WMAP cosmology, redshift
  uncertainty `(1+z)^3`).
- 2D BH-mass variant (interpolated, not zero_fill) — orthogonal.

---

## Reproducibility

All four scripts plus the lock-in note are reproducible from a fresh shell:

```bash
uv run python scripts/bias_investigation/test_08_bootstrap_map.py        # ~10 s
uv run python scripts/bias_investigation/test_09_first_bin_density.py    # ~5 s
uv run python scripts/bias_investigation/test_10_pdet_asymptote.py       # ~30 s
uv run python scripts/bias_investigation/test_11_window_proximity.py     # ~5 s
```

Outputs land in `scripts/bias_investigation/outputs/phase45/`. JSON files
contain raw numerics; PNG files give a one-glance summary.

## Caveat — local CRB is a proxy

The 412-event cluster posterior was produced from a CRB CSV
(`/pfs/.../run_20260401_seed200/cramer_rao_bounds.csv`, ~4500 events) that
is **not** rsync'd locally. Step 1c runs on
`simulations/prepared_cramer_rao_bounds.csv` (60 events at SNR≥20) as a
representative proxy. Posterior event IDs range 2..4489 → only 60
overlapped with the local CSV by index, but the σ/d_L *distribution*
generalizes (Fisher precision is a property of the waveform/PSD/SNR, not
of how many events were drawn). The "26/60 cross c_0" fraction (43%) is
the load-bearing finding; the absolute count is not.
