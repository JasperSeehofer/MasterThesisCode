# φ-slope decider note (row #122 item 4) — 2026-08-18

**Question (audit §C / report §9 flag 3):** is the persistent +0.01 2D-channel bias (present in
`off` and `fused` alike) caused by the 2D catalogue-leg mass overlap carrying no φ-prior weight
while the completion leg integrates against φ (instrument defect), or is it Malmquist-type venue
noise physics?

**Design:** exploratory (non-verdict-bearing), `phi_slope_decider.py`, seed 20271111 — V-deep
`off` cell, n=250, R=20, h_true=0.72, mass_slope ∈ {+1, 0, −1, −2} at fixed σ_m,gal.

**Result** (`phi_slope_decider_output.json`):

| mass_slope | bias2d | bias1d |
|---|---|---|
| +1.0 | +0.0062 | −0.0008 |
| 0.0 (flat φ) | +0.0122 | +0.0024 |
| −1.0 | +0.0138 | +0.0026 |
| −2.0 | +0.0094 | −0.0082 |

**Verdict: NOT the missing-φ-prior mechanism.** At flat φ (slope 0) a missing φ-prior weight is
exactly inert, yet the bias2d baseline is fully present (+0.0122). Slope dependence is secondary,
non-monotone, and ~2σ at this R. The +0.01 2D bias is venue noise physics (photo-z × completion
share + galaxy mass-observation error, per the audit's σ→0 switch-offs), i.e. an estimator/venue
property to be understood in its own right if ever needed — not an instrument bug blocking the
harness. The catalogue-leg overlap's form (no φ-deconvolution) remains a documented modeling
choice mirroring production (`bayesian_statistics.py:5522` analog), carried under the §6
venue-transfer caveat.
