# FINDING — the host-MASS kernel truncation biases the 2D H0 channel HIGH

**Date:** 2026-07-13 · **Trigger:** user insight that the catalogue mass error is
~50% (not 10%), so the mass kernel hits the same untruncated-vs-truncated
inconsistency as the low-z photo-z redshift kernel — and since mass is in the 2D
(with-BH-mass) channel only, it's a candidate for the 2D +0.025 residual and the
info-monotonicity violation (2D bias > 1D bias).

**Verdict:** ✅ Confirmed as a real, correctly-signed (HIGH), 2D-only mechanism.
Magnitude is leverage-dependent (~+0.003 to +0.009 in the tested regime), so it is
a **meaningful contributor** to the +0.025 2D residual — plausibly a large fraction
of the *extra* 2D bias (+0.012 over 1D) — though the toy cannot pin the exact
real-regime number. Not yet a production change; a `/physics-change`-gated
lognormal×R_eff truncated mass kernel is the indicated fix.

## 1. The mass error is ~60%, and the kernel is a LINEAR Gaussian (`mass_trunc_probe.py`)

The BH mass comes from Reines & Volonteri (2015) stellar→BH mass with **0.24 dex
intrinsic scatter** (dominant) + calibration. In linear terms the per-galaxy 1σ is
`BH_mass_error = BH_mass · √(σ_int² + d_α² + …)` with a **floor √(0.553²+0.184²) ≈
0.58**; typically σ_M/M ≈ 0.6–0.8. The 2D likelihood marginalises the host mass with
a **linear** Gaussian `N(M; M_g, σ_M)` (the `mz_integral`), so at σ_M/M ≈ 0.6:

- **P(M<0) = 4.8%** of every host's kernel mass is unphysical; near the EMRI bounds
  [M_min,M_max]=[1e4,1e7], **29% below M_min** (low-mass hosts) / **24% above M_max**
  (high-mass hosts).

## 2. G2d (the Eddington-in-M shift) breaks at the bounds (`mass_trunc_probe.py`)

G2d approximates `N(M;M_g,σ_M)·R_eff(M)` by a shifted Gaussian, EXACT only under a
locally log-linear R_eff and an *untruncated* Gaussian. Comparing the G2d effective
mass to the exact truncated + R_eff-weighted posterior mean at σ_rel=0.6:

| host M_g | interior 1e5–3e6 | near M_min (1.5e4) | near M_max (7e6) |
|---|---|---|---|
| (B_G2d − A_exact)/M_g | **< 1%** (G2d accurate) | **−15%** (wrong sign) | **+22%** |

So G2d is validated in the interior even at σ_rel=0.6, but is 15–28% wrong for
boundary hosts. **And 65% of R_eff-weighted EMRI hosts sit in the M_max boundary
zone** (median host mass 4.55e6, near M_max) — so the majority of 2D-channel events
are affected, not a rare edge.

## 3. The H0 impact is HIGH and leverage-dependent (`mass_kernel_h0_toy.py`)

Controlled single-host 2D estimator (mass↔M_z↔(1+z)↔H0 coupling), two arms differing
ONLY in the mass kernel — production (linear Gaussian + G2d) vs correct
(lognormal×R_eff truncated on [M_min,M_max]) — at moderate host z (0.3–0.5) to
isolate from the redshift-kernel effect. `diff = production − correct` = the
mass-kernel-induced H0 shift; the shared z-marginalisation bias is common-mode and
cancels. Multi-seed (n_events=1500):

| photo-z σ_z/z | mass-kernel H0 shift (production − correct) | control (correct arm) |
|---|---|---|
| 0.05 (near spec-z) | +0.0004 (mass channel barely used) | +0.003 (~clean → framework validated) |
| 0.15 | **+0.0027 ± 0.0002** | +0.022 |
| 0.30 | **+0.0089 ± 0.0008** | +0.075 |

- **Sign: HIGH** (production > correct) in every non-railing case — resolves the
  sign puzzle (a naive point-estimate argument gives LOW; the full marginalisation,
  which sees the truncated kernel's *shape*, gives HIGH).
- **Magnitude grows with photo-z leverage**: looser photo-z → the mass channel
  carries more of the z-constraint → larger mass-kernel bias. The real GLADE regime
  is loose (σ_z/z ~ 0.65), so the real effect is ≥ the +0.009 measured at σ_z/z=0.30
  (the toy rails at very wide σ_z, so the exact real number isn't pinned here).
- **Control validated**: at near-spec-z the correct arm is ~unbiased (+0.003) and
  the mass diff ~0 — the estimator framework is sound; the growing control bias with
  σ_z/z is the *redshift*-kernel effect (H1), cleanly separated (common-mode).

## 4. Unified picture (why 2D bias > 1D bias)

Both channels share ONE mechanism — a **large fractional measurement error
marginalised with an untruncated kernel near a physical boundary / through a
nonlinear transform**, which biases HIGH and grows with the fractional error:

- **1D** carries only the **redshift** effect (σ_z/z ~ O(1) at low-z photo-z) → +0.013.
- **2D** adds the **mass** effect (σ_M/M ~ 0.6, [M_min,M_max] bounds) → an extra
  HIGH bias, so **2D bias (+0.025) > 1D bias (+0.013)** — the info-monotonicity
  violation, mechanistically explained. The mass toy's +0.003…+0.009 is in the
  ballpark of the +0.012 extra 2D bias.

## 5. Indicated fix (user-gated, `/physics-change`)

Replace the linear-Gaussian host-mass kernel with the **lognormal (its true error
model) × R_eff population weight, truncated + renormalised on [M_min,M_max]** — the
mass analog of the redshift Candidate-B kernel. This subsumes the G2d shift (which
stays valid in the interior). Verify against the same regression-gate discipline
(σ→0 → spec limit; interior unchanged; boundary bias removed; H0 shift toward truth
in a venue-matched run) BEFORE production — the `volume_trunc` lesson.

## Caveats

- The H0 toy is a controlled isolation, not the full pipeline (no selection D(h);
  flat-prior photo-z anchor; moderate-z hosts; single candidate host/event). It
  establishes the **sign and rough magnitude** of the mass-kernel bias, not a
  campaign-grade number. A production A/B (new mode) or a mass-extended pp_coverage
  harness is the next quantitative step.
- σ_Mz=0.01 (GW redshifted-mass precision) is conservative; real EMRI M_z is far
  better-measured, which if anything sharpens the coupling.
