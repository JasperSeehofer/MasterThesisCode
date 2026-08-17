# PHYSICS-CHANGE GATE PRESENTATION — item 1: fused survival in both completion legs

**Date:** 2026-08-17 · **Authorization:** row #117 item 1 ([DO], ratified) + item 3 ([P4]
settled here) · **Verifier:** GO-WITH-AMENDMENTS
(`PROPOSAL_2D_SELECTION_FUSION_VERIFIER_ADDENDUM_20260817.md`); MAJOR-1..4 incorporated below ·
**Status: PRESENTED, AWAITING AUTHOR RULING ON G1–G3 (end of document). No code is written
before those rulings — per the binding default, the verifier surfaced decisions whose inputs
did not exist at row #117, so they return fresh.**

## 1. [P1] — 2D completion leg: `g_i` → `g_sel,prod`

**Old** (`bayesian_statistics.py:4344`, `completion_mass_factor_g:2012`; production calls with
NO quadrature kwargs ⇒ Route-1 `adaptive=True`, fast n=8 path on most rows at measured σ_cond):

    B_num_wbh = ∫ (1−f_k) p_gw [dVc/dz/(1+z)] g_i(z;h) dz
    g_i(z;h)  = ∫ dx_M N(x_M; μ_cond(z), σ_cond) φ_x(x_M; z)

**New** (same outer z-quadrature; new function, strictly additive — `completion_mass_factor_g`
is untouched for its external callers `validation/calibration_gate.py:767`,
`closed_loop_gfrac.py:524`, `venue_transfer.py:1346`):

    g_sel,prod(z;h) = ∫ dx_M N(x_M; μ_cond(z), σ_cond) φ_x(x_M; z) · S_4D(d_L(z;h), x_M·M_z,det,i)

with `S_4D` queried exactly as `precompute_phi_marginal_survival:1925-1948` queries it
(detector-frame mass, node d_L, isotropic, `_wbh_z_kwargs` rider); quadrature per ruling G1.

**Reference:** Mandel, Farr & Gair (2019) arXiv:1809.02063 Eqs. (5)–(7) (selected-prior
per-event likelihood); L6-DER2 §2–§3 + L6-DER3 §3 (latent-thresholded classification of the
pipeline's own detection model); venue arm rows #115–#116.

**Dimensional analysis:** `S_4D` is dimensionless ⇒ `g_sel,prod` remains a density in
`x_M = M_z/M_z,det,i` — the same measure as the catalogue leg's `mz_integral`, so gate (i)
addability is preserved by construction (verifier surface D: SOUND for the S-fusion itself).

**Limiting cases (amended per MAJOR-2):**
- `S≡1` recovers `g_i(adaptive=False, n=64)` **bit-exactly** (unit test); pinned-vs-adaptive
  production default differs at the ~1e-15 level (L6-DER2 addendum measured 1.1e-15) — recorded
  as a separate bound, not folded into the bit-exact claim.
- `σ_cond→0` gives `g_i·S(μ_cond·M_z,det)` — and per MAJOR-1 this is effectively the
  **production operating point**, not a corner: measured d_L-conditional σ_cond on the
  production CRB reference (`run_20260804_postfix/iiib/diagnostics/`, n=1590, Bishop 2.81–2.82
  on the fractional (d_L,M_z) block): p5/p50/p95 = 2.47e-8 / 8.80e-8 / 2.99e-7, max 2.7e-6.

**Regime statement of record (MAJOR-1, replaces the proposal's):** production's completion leg
sits in the sharp-likelihood limit; the expected action of the pair is 1D-dominated ([P2]),
with [P1] correct-form and possibly near-inert (N-2 §3.1's unmeasured M2 band |Σ| ≤ 20 nats/h
is the only prior bracket). A3 stands — this sets expectations, not claims; the [P5-3]
counterfactual must decompose the 1D and 2D contributions separately.

**Interactions:** the `:4352` φ-support warning is taught to distinguish φ-support-exit zeros
from S-induced (beyond-horizon) zeros, with a regression test (MINOR-2); `g_frac` diagnostics
and `freeze_g_frac_ref_h` wrap the ratio unchanged, but S̄_φ is additionally tabulated at the
freeze reference h so the counterfactual cannot ValueError off-grid (MINOR-1); `np.log10(M_z)`
guarded against non-positive Hermite nodes under `wbh_z_resolved` (MINOR-6, one line + test).

## 2. [P2] — 1D completion leg: S̄_φ numerator factor default-on

**Old** (`:4230–4290` assembly): `B_num = ∫ (1−f_k) p_gw [dVc/dz/(1+z)] dz` — selection-free;
the weighted form exists as `completion_numerator_integrand_sel_1d:4295` (default `"off"`).

**New:** `B_num = ∫ (1−f_k) p_gw [dVc/dz/(1+z)] S̄_φ(z;h) dz` — the existing branch promoted to
the `absolute_marginal` default; table READ, never rebuilt, exactly as coded.

**Reference:** T3′ (`N2_SELECTION_NUMERATOR_DERIVATION_20260805`, status: derivation ratified
in the L6 chain; the associated PRODUCTION measurement `run_20260805_n2sel1d/` — +30.9 nats/h
central-difference at h=0.73 — remains **DRAFT/unadjudicated** and is context, not evidence,
per L6-DER3 §4); L6-DER3 §4.

**Carried caveat (MINOR-4, of record):** the N-2 draft's honest gap 5 — in the G4b harness the
selection-inside factor only *calibrated* when paired with the σ(d_obs)-vs-σ(d_true)
noise-model companion (#67) — is the single most likely way this correction disappoints. The
venue arm's DS-G3 restoration covers calibration in-venue only; [P5-3] measures magnitude, not
calibration. This caveat rides into item 4's mini-prereg.

**Limiting cases:** `S̄_φ≡1` recovers old `B_num` exactly (unit test); `S̄_φ(z→z_max)→0`
removes beyond-horizon population support (currently overweighted).

**Pairing (hard, verified — surface B):** [P1] and [P2] ship in ONE commit or not at all.

## 3. [P4] — measure ruling (V2 prefactor + D-ii ratio form) — see G2

Verifier MAJOR-4: V2's materiality lives in the **catalogue** leg's broad host-mass Gaussian
(`sigma_gal_frac:5238`, the 60–200% regime) — the leg deferred by ratified row 2. At
completion-leg σ_cond (~1e-7) the ratio-vs-density deviation is ≲1e-6 from a tilt-neutral
per-event constant (L6-DER2 addendum bound, same regime). A completion-only measure change
would break gate (i) addability with the untouched `mz_integral`. The ruling is therefore
presented explicitly as G2, not folded silently.

## 4. Verification plan of record (amended [P5]; ships with the [PHYSICS] commit)

1. Unit: S≡1 → `g_i(adaptive=False, n=64)` bit-exact; S̄_φ≡1 → old `B_num` exact; recorded
   pinned-vs-adaptive bound; φ-support-warning zero-provenance test; freeze-ref-h table test;
   non-positive-node guard test.
2. Byte-identity: every non-`absolute_marginal` estimator path AND the named external callers
   of `completion_mass_factor_g` (function untouched; fusion is a new callable).
3. Regression recording: old-vs-new `B_num`/`B_num_wbh`/`g_frac` on a pinned event set,
   committed BEFORE the flip lands; PHYSICS-GATE-LEDGER row per gate run.
4. [P5-3] paired counterfactual (item 4, own mini-prereg): old vs fused on the same
   seeds/realizations, **with separate 1D-only ([P2]) and 2D-only ([P1]) decomposition legs**
   (MAJOR-1) — three cells, not one. Bands seeded from nothing (A3). Ceiling set from the
   pessimistic premeasure rate (row #116 item 2 discipline). Campaign re-run scope returns
   with its result.

## 5. Decision table — fresh rulings the verifier created (binding-default returns)

| # | Question | Options | Recommendation | Tag |
|---|---|---|---|---|
| G1 | [P1] quadrature | (A) pinned non-adaptive n_hermite=64 (venue-registered convention; conservative out-of-regime; ~8× node cost + one S query/node on the hot path — undoes the ratified 2026-08-12 Route-1 default there) · (B) keep Route-1 adaptive, S evaluated per node (perf-preserving; in-regime exact since S is locally constant over the ~1e-7 Hermite window; the Route-1 polynomial bound formally does not cover S) | **B**, with the recorded pinned-vs-adaptive regression bound (~1e-15 class) and a guard assertion that escalates to n=64 if the S-variation across the node window exceeds tolerance — honors the realistic-venue efficiency deliverable without giving up the conservative path | [RULE] |
| G2 | [P4] measure/V2 | (A) retain the ratio convention in BOTH legs; record V2 as a tracked systematic (G7 budget row) with the σ_cond immateriality evidence; re-open with [P3]/row #110 where it is material · (B) widen the measure ruling to both legs now (touches the deferred catalogue leg) | **A** — B contradicts ratified row 2's deferral | [RULE] |
| G3 | Row 2 re-confirmation | The ratified [P3] deferral was adjudicated on an inverted skew direction (MAJOR-3: catalogue leg is OVER-weighted under [P2], not down-weighted). Structure of defer-unless-material survives (the counterfactual measures the skew regardless of sign). Confirm the deferral stands on the corrected basis, or reopen row 2 | **Confirm deferral** | [RULE] |

Implementation of items 1+3 begins when G1–G3 are ruled; the full gate then runs
(implementation → checks → `[PHYSICS]` commit → PHYSICS-GATE-LEDGER row).

*Append-only from its commit.*
