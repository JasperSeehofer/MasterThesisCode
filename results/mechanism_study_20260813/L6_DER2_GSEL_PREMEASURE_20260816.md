# L6-DER2 §4 step 1 — mirror pre-measurement of the fused `g_sel` candidate

**Date:** 2026-08-16 · **Authorized:** rows #112/#114 · **Status: PRESENTED, NOT
ADJUDICATED.** This is the mirror pre-measurement registered as the next step
in `L6_DER2_CORRECT_FORM_2D_20260816.md` §4 item 1. Numbers are reported
as-measured; they are not tuned toward the registered prediction, and
adjudication (the xhigh verifier, §4 item 2) is a separate, later step.

Script: `results/mechanism_study_20260813/l6_der2_gsel_premeasure.py`.
Output: `results/mechanism_study_20260813/L6_DER2_GSEL_PREMEASURE_output.json`.

## What was run

15 MN0X seed replays (seeds 20310808–20310822, full dose, real-K balls), the
`l6_c2_switch_decomposition.py` c1/c2 chunk-loop method, at k=20 (h=0.725) and
k=22 (h=0.735). Three estimator configs computed in one shared pass per
(seed, k):

- **`base`** — the coded original estimator (`ESTIMATOR_VARIANT_BASE`).
- **`afull`** — the A-FULL estimator (`ESTIMATOR_VARIANT_A_FULL`) mirrored
  verbatim from `venue_transfer._channel_terms_at_h`.
- **`gsel`** — the L6-DER2 §3 candidate: `afull`'s 1D channel unchanged;
  its 2D channel replaces the (S̄_φ(z) node-weight × coded `g`) pair by the
  single fused object `g_sel(z,f;h) = ∫dx_M N(x_M;μ_cond,σ_cond)·φ_x(x_M;z)·
  S(x_M·M_z_obs;z,h)`, with `S` the unmarginalized with-BH detection
  survival queried the same way `precompute_phi_marginal_survival` queries
  it (isotropic, `_wbh_z_kwargs` pass-through). Implemented as a new
  `g_sel_mass_factor` function reusing `completion_mass_factor_g`'s
  conditional-Gaussian and `φ_x` conventions verbatim, non-adaptive full
  order (n_hermite=64) — a documented convention choice (see the script's
  module docstring), not something the derivation itself specifies.

Runtime: ~66 min for the 15-seed pool (8 fork workers), plus ~20 min for the
direct-call cross-validation (dominated by `_loo_impostor_weights` being
recomputed at all 41 h-grid points in the real `venue_transfer` code path,
not a mirror inefficiency) — about 1h35m end to end on the dev machine.

## Validation gates (all bit-exact)

| Gate | Result |
|---|---|
| `base` vs stored MN0X `ln_post_1d`/`ln_post_2d` | max abs diff = 0.0 |
| `afull` mirror vs direct `log_channel_posteriors_ball_sigma_vector(..., estimator_variant="a_full")` (2 seeds × both k) | max abs diff ln1 = 0.0, ln2 = 0.0 |
| S≡1 refactor check: `g_sel_mass_factor(force_S_one=True)` vs `completion_mass_factor_g(..., n_hermite=64, adaptive=False)` (200 kernel-branch rows, 2 events, seed 20310808, k=20) | max relative diff = 0.0 |
| c1 bit-identity: `gsel` vs `afull` (both k, all 15 seeds) | max abs diff = 0.0 |

All four gates passed exactly, not just "within tolerance."

## Numbers (nats/h, mean ± SE over 15 seeds)

| Config | T1 | T2 | excess (T2−T1) |
|---|---|---|---|
| `base` | +2644.0 ± 46.5 | +2775.5 ± 46.5 | +131.5 ± 0.1 |
| `afull` | +30.6 ± 42.7 | +166.4 ± 42.7 | +135.8 ± 0.1 |
| `gsel` | +30.6 ± 42.7 (identical to `afull`, by construction) | +18.9 ± 42.9 | **−11.7 ± 1.0** |

**dT2(gsel) = T2(gsel) − T2(afull) = −147.5 ± 1.0 nats/h**
**d_excess = excess(gsel) − excess(afull) = −147.5 ± 1.0 nats/h**

## Measured vs registered prediction

L6-DER2 §2/§4 (committed 09c02c06, before this run) predicted qualitatively:
the 2D−1D excess collapses under the fused form (the coded channel B's
~+139 nats/h scale, per `l6_c2_switch_decomposition.py`'s `dT2_sb` ≈ −139,
cancelled to the "~few-nat level"), with the 1D channel bit-untouched. No
numeric target was pre-registered for *this specific* fused-form measurement.

Measured: excess drops from +135.8 ± 0.1 (`afull`) to −11.7 ± 1.0 (`gsel`) —
a collapse of −147.5 nats/h, landing at the few-nat scale as predicted,
though on the negative side rather than at/near zero. T1 is exactly
unchanged (bit-identical c1, verified). Both qualitative claims of the
prediction — collapse-to-few-nats and 1D-untouched — are borne out by this
measurement; whether −11.7 ± 1.0 (vs. a hypothetical exact 0) constitutes
"cancellation" in the sense the derivation intends is an adjudication
question, explicitly left to §4 item 2 (the xhigh verifier), not decided
here.

## Convention choices made (documented, not physics decisions)

1. **Non-adaptive, full-order (n=64) Hermite quadrature for `g_sel`.** The
   coded object's Route-1 adaptive fast order (n=8) is a truncation-error
   optimization derived for the smooth `φ_x` integrand alone; folding in the
   detection survival `S(x_M)` (a much sharper, horizon-cutoff-shaped
   function of mass) makes that error bound inapplicable, so this script
   pins the exact order everywhere. This is the main cost driver (a survival
   query per Hermite node per quadrature node per candidate pair).
2. **S query convention**: detector-frame mass `M_z = x_M · M_z_obs_i` (the
   coded `x_M` is already defined as `M_z / M_z_obs_i`), absolute `d_L(z;h)`
   the same node value the outer kernel/GW-density integral already
   computed, isotropic sky (`phi=theta=0`), `_wbh_z_kwargs` pass-through
   included for correctness under a future FIX-3 flag flip (verified inert
   here — `wbh_z_resolved` is `False` in this venue context).
3. The `a_full` direct-call cross-validation was restructured mid-run to
   reuse the pool's own `afull` values rather than recomputing them via the
   3-config mirror a second time (the mirror always computes all three
   configs together, so a naive validation call would re-pay the expensive
   `gsel` Hermite-survival cost for the validation seeds) — a runtime
   optimization, not a change of what is validated: the direct-call
   comparison itself is still against the untouched `venue_transfer` code
   path.

## Deviations from the task brief

- The task brief anticipated "tens of minutes"; the actual run took
  ~1h35m end to end, dominated by the `gsel` Hermite-survival cost (a
  survival query per Hermite node × 50 quadrature nodes × ~1.19M candidate
  pairs × 2 k-values, non-adaptive) and, unexpectedly, by
  `_loo_impostor_weights` being recomputed at every one of the 41 h-grid
  points inside the real `venue_transfer.log_channel_posteriors_ball_sigma_vector`
  direct call used for the `a_full` cross-check (an existing cost in the
  production code path, not something this script introduced).
- No other deviations: all three configs, both validation targets, and the
  S≡1 refactor check were implemented and run as specified.
