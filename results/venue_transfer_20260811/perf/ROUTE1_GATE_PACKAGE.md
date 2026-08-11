# [PHYSICS] Route 1 — adaptive Gauss-Hermite order in `completion_mass_factor_g`

Direction author-approved 2026-08-12; package RATIFIED by author 2026-08-12.

Implementation: `master_thesis_code/bayesian_inference/bayesian_statistics.py`
(`_G_I_HERMITE_NODES_FAST`, `_G_I_ADAPT_T`, `_G_I_ADAPT_MAX_RELWIDTH` module
constants; `completion_mass_factor_g(..., adaptive: bool = True)`). Tests:
`master_thesis_code_test/test_route1_adaptive_hermite.py`.

## 1. Old formula

Fixed 64-node Gauss-Hermite contraction, unconditionally, for every row —
`master_thesis_code/bayesian_inference/bayesian_statistics.py:2001-2061`
(pre-change, commit `39b7e75b`), the operative lines being:

```python
# :2053 (old)
x_nodes, x_weights = roots_hermite(n_hermite)   # n_hermite = _G_I_HERMITE_NODES = 64
# dM/dx_M at each z: the mass scale the dimensionless coordinate rides on.
scale = det_M_z / (1.0 + np.asarray(z_nodes, dtype=np.float64))  # (k,)
mu_cond = 1.0 + proj_d_L_to_M * (np.asarray(d_L_fraction, dtype=np.float64) - 1.0)  # (k,)
# Gauss-Hermite for E_{x~N(mu,sigma)}[phi_x]: nodes mu + sqrt(2) sigma t_j.
x_M = mu_cond[:, None] + math.sqrt(2.0) * sigma_cond_M * x_nodes[None, :]  # (k, n_h)
M_source = x_M * scale[:, None]
phi_x = dark_mass_density_per_mass(M_source) * scale[:, None]
return np.asarray((phi_x @ x_weights) / math.sqrt(math.pi), dtype=np.float64)
```

`n_hermite` defaults to `_G_I_HERMITE_NODES = 64` and every call — regardless
of how narrow the conditional Gaussian in `x_M` is relative to the nearest
`phi` breakpoint — pays the full 64-node contraction cost. Per the profiled
cost model (`results/venue_transfer_20260811/perf/PERF_ROADMAP.md`), this
`g_i` leg is the dominant contributor (91.1%) to seed wall time.

## 2. New formula

Per-row adaptive order: contract at `_G_I_HERMITE_NODES_FAST = 8` unless a
row triggers either fallback condition below, in which case it is contracted
at the pinned `_G_I_HERMITE_NODES = 64` instead. Both orders use the
identical contraction code path (`_contract_group` helper) — only the
`roots_hermite(order)` argument and the row subset differ.

**Fallback condition 1 — relative half-width.** With
`w = sqrt(2) * sigma_cond_M * _G_I_ADAPT_T` (`_G_I_ADAPT_T = 6.0`, a scalar,
constant across rows within one call since `sigma_cond_M` is event-level):

```
fallback_1 = (w > _G_I_ADAPT_MAX_RELWIDTH * mu_cond) OR (mu_cond <= 0)
```
with `_G_I_ADAPT_MAX_RELWIDTH = 0.02`. The `mu_cond <= 0` branch is a safety
fallback (never observed in the harvest; guards against pathological calls).

**Fallback condition 2 — breakpoint straddle.** With
`lo = (mu_cond - w) * scale`, `hi = (mu_cond + w) * scale` (`scale > 0`
always):

```
fallback_2 = any(lo < b < hi for b in {M_SOURCE_FRAME_MIN, 1.0e5, M_SOURCE_FRAME_MAX})
```

where `1.0e5` is the `emri_rate.kappa_cap` `M_turn` (Eq. 30 surrogate)
turn-over — the interior kink of the two-segment power law
`phi(M)` sits at.

`fallback = fallback_1 OR fallback_2`. Rows with `fallback == True` use
`n=64`; rows with `fallback == False` use `n=8`. `adaptive=False` (or a
non-default explicit `n_hermite`) bypasses all of the above and reproduces
the old single-group `n_hermite`-order contraction verbatim — byte-for-byte
identical to today's output. An all-`fallback=True` call under
`adaptive=True` takes the identical single-group `n=64` code path as
`adaptive=False`, so the two are bit-identical in that regime too (verified:
`test_forced_straddle_is_bit_identical_to_convention`,
`test_relwidth_criterion_triggers_fallback`).

## 3. Reference

- **Abramowitz & Stegun (1964), Eq. 25.4.46** — the Gauss-Hermite quadrature
  truncation error term: an `n`-point rule integrates any polynomial times
  `e^{-t^2}` exactly to degree `2n-1`; the remainder for a smooth non-
  polynomial integrand is controlled by the `(2n)`-th derivative and shrinks
  rapidly with both `n` and the relative narrowness of the sampled window
  around the expansion point. At `n=8` the rule is exact to degree 15.
- **Piecewise power-law integrand structure** — established by the phi
  affine-swap physics change (commit `87c6670b`,
  `_phi_ln_dark_mass_affine_coeffs`): `ln phi(M)` is exactly affine in
  `log10 M` on each side of the single kink at `M=1e5`, i.e. `phi(M)` is a
  pure power law `M^p` on each segment (exponents from Babak et al. 2017 Eqs.
  5, 23, 26-27, 30, 31x34: max `|p| = 1.43` combining the `dn/dlog10 M`,
  `R_eff`, and `kappa` factors). When the +-`_G_I_ADAPT_T` sigma window stays
  within one segment (fallback_2 == False) and is narrow relative to
  `mu_cond` (fallback_1 == False), the Gauss-Hermite integrand is a single
  smooth power-law branch times a Gaussian, and the truncation error at
  `n=8` for `relwidth <= 0.02` and `|p| <= 1.43` is many orders of magnitude
  below `1e-12` — confirmed empirically below, not just asserted from the
  error-term shape.
- **Empirical study**:
  `results/venue_transfer_20260811/perf/route1_study/ROUTE1_STUDY.md`.
  41,049,200 harvested production z-nodes (1394 calls, one realistic-venue
  seed run, `Tc` cell, `h_true=0.730`, `balls=real_k`, `sigma_mode=glade`,
  `n_events_cap=30`); `sigma_cond_M` median 6.276e-7 (dimensionless, `x_M`
  units), p90/p99/max all 6.276e-7 (event-level clustering). Zero straddling
  at `t_tol in {4, 5, 6}` on the entire harvest — `mu_cond` sits within
  `~5e-7` of 1 while the nearest breakpoint sits `>=0.07` away in `x_M`
  units, roughly 5-6 orders of magnitude beyond `sigma_cond_M * t_tol`. Max
  relative error at `n=8` vs `n=256` (reference): **1.305e-15**, at or near
  float64 machine epsilon. All 12 `(n_low, t_tol)` candidate pairs in
  `{8,12,16,24} x {4,5,6}` pass both acceptance criteria (non-straddling max
  rel. err `<1e-12` vs `n=256`; overall max rel. err `<1e-10` vs the `n=64`
  convention) by 3-5 orders of magnitude of margin. `n_low=8, t_tol=6` (the
  values shipped as `_G_I_HERMITE_NODES_FAST`, `_G_I_ADAPT_T`) is the most
  conservative (widest window) member of the winning set, chosen for margin
  rather than raw speed.

## 4. Dimensional analysis

Unchanged. The integrand `phi_x(x_M; z) = phi(x_M * scale) * scale`, the
Gaussian conditional `N(mu_cond, sigma_cond_M)`, and the quadrature measure
(`dx_M`, dimensionless) are exactly as before — only the number of
quadrature nodes used to approximate the (still exact, still
dimensionally-consistent) integral changes. `g_i` remains a density in
`x_M`, units `1/x_M`, matching `mz_integral`'s measure as documented in the
function's existing docstring. No new physical quantity, constant, or
formula term is introduced; `_G_I_ADAPT_T` and `_G_I_ADAPT_MAX_RELWIDTH` are
pure quadrature-control numbers (dimensionless order/tolerance knobs), not
physical parameters.

## 5. Limiting cases

- **`sigma_cond_M -> 0`** (point evaluation): `w -> 0`, so fallback_1 never
  fires and fallback_2 only fires if `mu_cond * scale` sits exactly on a
  breakpoint (measure zero). The Gauss-Hermite sum collapses toward the
  single-point evaluation `phi_x(mu_cond)`, matching the pinned convention's
  own `sigma -> 0` limit to the accuracy already exercised by the existing
  L5 property test; regression-covered here by
  `test_sigma_to_zero_point_evaluation` (rtol 1e-10 vs `adaptive=False`,
  finite and positive).
- **Flat-`phi` exactness at any order**: if `phi` were locally constant
  (zero-degree polynomial) the Gauss-Hermite rule is exact for any `n >= 1`
  by the classical GH exactness property — the adaptive scheme reduces to a
  strict accuracy improvement over the already-exact case, never a
  regression.
- **Forced breakpoint straddle**: constructed with `sigma_cond_M=0.12`,
  `det_M_z=2e5`, `z` near 1.0 so every row's +-6 sigma window crosses the
  `M=1e5` kink — `adaptive=True` output is bit-identical
  (`np.array_equal`) to `adaptive=False`, because an all-fallback call takes
  the same single-group `n=64` path. Covered by
  `test_forced_straddle_is_bit_identical_to_convention`.
- **Relative-half-width criterion in isolation**: `sigma_cond_M=0.01` with a
  breakpoint-free window (`det_M_z=8e5`) — `w > 0.02*mu_cond` for every row,
  so all rows fall back on width alone; output is bit-identical to
  `adaptive=False`. Covered by `test_relwidth_criterion_triggers_fallback`,
  which also independently re-derives the fallback mask from the same
  formula and asserts it against the observed all-fallback behaviour.
- **Explicit non-default `n_hermite`**: `n_hermite=32` disables the adaptive
  path unconditionally (per spec: `n_hermite != _G_I_HERMITE_NODES`
  triggers the old code path regardless of `adaptive`); `adaptive=True` and
  `adaptive=False` give bit-identical results at `n_hermite=32`. Covered by
  `test_explicit_n_hermite_override_bypasses_adaptive`.
- **Regression pins unchanged**: `test_phi_interpolation_regression.py` and
  `test_fixb_pathA_mixture.py` (both exercise `completion_mass_factor_g` at
  the default `n_hermite=64`, `adaptive=True`) pass unmodified against their
  existing pinned values, since realistic-narrow `sigma_cond_M` in those
  fixtures never trips a fallback condition and the fast-path result matches
  the pinned n=64 value to float64 noise.

## Registered tolerance class

Same tolerance class as the phi affine-swap physics change (commit
`87c6670b`): rel `1e-8` on the 2D channel is the registered ceiling this
Route lives well inside of (measured max rel. err `1.3e-15`, ~7 orders of
margin).

## Projected performance impact

Projected seed-wall speedup (from the study's cost model,
`g_i` leg = 91.1% of seed wall,
`speedup = 1/(0.089 + 0.911*(avg_nodes/64))`, `avg_nodes = 8` under
zero-fallback on the harvested venue): **4.93x**. This is a *projection*
from the harvested distribution's zero-fallback regime, not yet re-measured
end-to-end post-implementation — re-measurement (seed-wall timing on a full
realistic-venue run with the merged code) is the natural next step and is
explicitly **not** claimed as delivered by this package. A microbenchmark on
this branch (`completion_mass_factor_g` alone, `k=4000` realistic-narrow
rows, `sigma_cond_M=5e-7`, all-fast-path) measured **23.5x** — expected to
exceed the seed-wall projection since it isolates the `g_i` leg itself from
the other 8.9% of wall time the cost model already accounts for.

## Open note

The `n=64` convention's own quadrature defect at the kink (how well `n=64`
itself resolves the straddling case, vs. e.g. a split Gauss-Legendre
reference) is **unmeasurable on this harvested venue** — zero harvested
z-nodes ever straddle a breakpoint at any tested `t_tol` (Table 3 of
`ROUTE1_STUDY.md` is vacuous, `n_rows=0`). This stays an open question for
any future "fat-sigma" venue (larger `sigma_cond_M` relative to the distance
between `mu_cond*scale` and the nearest breakpoint) — the fallback path in
this implementation always defers to the unmodified `n=64` code, so it
inherits whatever accuracy that convention has always had; this package
makes no claim about improving or even characterizing that accuracy, only
about matching it byte-for-byte whenever it fires.

## Adversarial verification & measured certification (2026-08-12)

- Verdict CONFIRMED-WITH-CAVEATS (independent xhigh verifier). Analytic GH-8 truncation bound at the relwidth threshold: ~2.5e-37 relative (prefactor 8!·sqrt(pi)/(2^8·16!) ≈ 1.34e-11, derivative product ≈ 7.9e13 at p=-1.43, (0.02/6)^16 ≈ 2.4e-40) — the package's "<<1e-12" is conservative by ~25 orders. Numeric hunt (6160 adversarial placements): max in-support non-fallback deviation 9.6e-16.
- REGISTERED BEHAVIORAL DIVERGENCE (orchestrator decision, pending author ratification with the rest): rows whose entire ±6-sigma window lies off the phi band return exactly 0.0 under adaptive vs ~1e-19 quadrature-tail dust under the n=64 convention (GH-64 weight beyond |t|=6 is 2.7e-18 of total). Exact zero adopted as the more physical value (off-band density is zero by construction); unreachable in the production query distribution (nearest breakpoint >= 0.07 in x_M vs sigma ~ 6e-7); side effect: closed_loop_gfrac's b2>0 guard would drop an all-off-support event instead of keeping ln(dust) — accepted.
- Notes: sigma=0 rows are fast-path point evaluations, NOT bit-identical to n=64 (3.4e-16 — no claim broken); `scale > 0` is an unchecked assumption (det_M_z<0 is safe via off-band masking); mixed-call fallback rows within 1 ULP of full-call (registered BLAS-shape class); fast order now pinned by test.
- Measured certification (route1_counterfactual_smoke, Tc(0.730) registered smoke seed): max abs 1.14e-13 / max rel 1.26e-14 over 5 differing leaves, 1D channel byte-identical — tighter than the phi swap's own 5.15e-9. Seed wall: 124.92 s (baseline) -> 88.03 s (phi swap) -> 13.47 s (Route 1) = 9.28x cumulative; adaptive-vs-convention in-process 6.79x.
