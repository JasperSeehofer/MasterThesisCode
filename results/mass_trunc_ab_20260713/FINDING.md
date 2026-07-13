# FINDING — EXP-45 `mass_trunc` is numerically sound but NOT the 2D residual driver

**Date:** 2026-07-13 · **Branch:** `physics/zero-host-completion-fallback` ·
**Verdict:** ⚠️ The truncated lognormal × R_eff host-mass kernel (`mass_trunc`) is a
**more physically correct** kernel (true error model, proper `[M_MIN, M_MAX]`
truncation, peak-aware quadrature) and is **numerically clean**, but its net effect
on the seed600 shallow-venue 2D H₀ bias is **small and in the WRONG direction**
(2D mean +0.0029, *away* from truth). It does **not** explain the 2D +0.025 residual
and is **not** the production fix. The isolated single-host toy over-stated the
effect by omitting the selection denominator. `volume_deconv` stays the byte-identical
default. Retained as an experimental, reproducible record (like `volume_trunc`).

## What was tested

Per `results/mass_trunc_ab_20260713/RUNBOOK.md` (pre-registered):
`mass_trunc` = the `volume_deconv` host-z kernel with the 2D (with-BH-mass) channel's
host-mass prior replaced from the linear-Gaussian G2d moment match
(`eddington_shifted_host_mass`) to the **truncated lognormal × R_eff prior** on
`[M_MIN, M_MAX] = [1e4, 1e7]` — Gauss-Hermite on the narrow GW M_z peak in the
numerator, Gauss-Legendre-in-lnM over a peak-aware window in the selection
denominator. Motivation + toy (sign HIGH, +0.016…+0.02 at the shallow leverage):
`results/mass_kernel_truncation_20260713/FINDINGS.md`.

Decisive gate: seed600 494-event shallow-venue A/B, `mass_trunc` vs `volume_deconv`,
same 7-point grid `[0.60…0.86]` as the N-5 / Eddington / volume_trunc drivers.
Driver: `scripts/mass_trunc_ab.py`. Raw result: `gate_result.json`.

## Result (truth h = 0.73)

| channel | mode | MAP | mean | edge | posterior shape |
|---|---|---|---|---|---|
| 1D | `volume_deconv` (baseline) | 0.73 | **0.7450** | 0.000 | 0.53 @0.73, 0.44 @0.76 |
| 1D | `mass_trunc` | 0.73 | **0.7450** | 0.000 | **byte-identical to baseline** |
| 2D | `volume_deconv` (baseline) | 0.76 | **0.7681** | 0.003 | 0.735 @0.76, 0.225 @0.80 |
| 2D | `mass_trunc` | 0.76 | **0.7710** | 0.010 | 0.693 @0.76, 0.271 @0.80 |

Δ(mass_trunc − deconv): **1D mean +0.0000 (exact), 2D mean +0.0029**.

- **HARD CORRECTNESS GATE — PASSED.** The 1D combined posterior is **bit-for-bit
  identical** between the arms (`one_d_byte_identical: true`): `mass_trunc` shares the
  `volume_deconv` host-z kernel and modifies ONLY the 4D mass term, exactly as
  designed (also pinned by `test_kernel_parity`'s `*_mt_3d == *_vd_3d` byte-equality).
- **Baseline validation:** the `volume_deconv` arm reproduces the established seed600
  subsample reference exactly (1D mean 0.745, 2D mean 0.768) → driver + data sound;
  any divergence is the `mass_trunc` kernel alone.

## Verdict vs pre-registered predictions

- **CALIBRATED** required `Δ2D_mean ≤ −0.005` (move DOWN toward truth). **Not met.**
- **BIASED / NULL** = `Δ2D_mean > −0.003` or wrong sign. **Met:** +0.0029, wrong sign.

⇒ **The mass-kernel truncation is EXONERATED as the 2D +0.025 residual driver.** The
change is real, deterministic (no MC noise — glz64 denominator), numerically stable
(no rail, edge mass 0.003→0.010, MAP unchanged at 0.76), and physically more correct
— but its full-pipeline H₀ effect is small and the *opposite* sign to what the toy
predicted. This is a genuine result to report, not to force (the `volume_trunc` lesson).

## Mechanism — why the toy (+0.016…0.02) ≠ the pipeline (+0.0029, flipped)

The toy (`mass_kernel_h0_toy.py`) isolated the **numerator** mass marginal for a
single moderate-z host with **no selection term**. There, replacing the untruncated
linear Gaussian with the truncated lognormal×R_eff lowers the effective host mass and
pulls H₀ down (production > correct). The full pipeline applies the SAME truncated
prior to the **selection denominator** `D_g = ∫ p_det(d_L(z), M(1+z)) p_M(M) dM`, and
the in-catalogue likelihood is the ratio `L_cat = Σ_g w_g N_g / Σ_g w_g D_g`:

- The prior-shape change moves `N_g` and `D_g` in the **same direction** (both are
  integrals against the identical `p_M`), so the ratio `N_g/D_g` is far less sensitive
  to the prior than the numerator alone — the toy's numerator-only shift largely
  **cancels** against the denominator.
- What survives is a small residual set by the *p_det weighting* inside `D_g` (the
  truncation trims mass where p_det differs from the numerator's GW weighting),
  leaving a net **+0.0029**, opposite the numerator-only sign.

This is the same class of lesson as every prior selection-term surprise in this
project: **the selection denominator is not a spectator.** A numerator-only toy
cannot forecast the pipeline H₀ shift.

## Disposition

- **Do NOT deploy** `mass_trunc` as a bias fix — it does not reduce the 2D residual
  (it slightly increases it) and adds ~10% eval cost (GH+GL vs analytic; 358 s vs
  327 s here).
- **Correctness note:** it *is* the more faithful kernel (true lognormal error;
  proper truncation vs the linear Gaussian's 4.8% P(M<0) and boundary leakage).
  If a future decision wants the exact kernel for *fidelity* rather than bias
  reduction, `mass_trunc` is the vetted, tested implementation — but on this venue
  the linear-Gaussian G2d approximation and the exact kernel agree to ~0.003 in H₀,
  so the approximation is **empirically validated as adequate** for the 2D channel.
- `volume_deconv` remains the byte-identical production default. `mass_trunc` is
  retained as an isolated, tested `normalization_mode` (experimental) + reproducible
  record. Golden pins: `test_kernel_parity` (9 `*_mt_*` cases); limiting cases:
  `test_mass_trunc_kernel.py`.

## The 2D +0.025 residual is still open

With the mass kernel exonerated (this) and the host-z numerator window exonerated
(`volume_trunc` falsified), the 2D venue residual remains unexplained at the harness
level. Per the deep-bias ledger it stays **campaign-gated (D4)** — the definitive
test is the multi-seed campaign on the real (non-subsample) venue, not further
single-venue kernel surgery.

## Caveats

- One venue (seed600 shallow, 494-event subsample). The sign/magnitude of the
  mass-kernel effect is venue-dependent; a deeper venue with a different host-mass /
  photo-z leverage could differ. The *exoneration as the +0.025 driver* is specific
  to this venue but is the venue where the residual lives.
- `mass_trunc` also validated the toy's core claim in ISOLATION is real — it is the
  pipeline coupling (denominator) that neutralises it, not an error in the toy.
