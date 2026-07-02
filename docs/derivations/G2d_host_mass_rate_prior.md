# G2d — Host-mass rate prior in the with-BH-mass channel (Eddington-in-M fix)

**Decision (user, 2026-07-02):** implement the mass-population prior rather than quote a caveat —
the bias scales as σ_M², so any caveat quantified at GLADE's σ_M would not transfer across the
(σ_z, σ_M) forecast sweep, which is precisely the F5 axis the paper explores. Symmetric to the
z-channel `volume_deconv` fix.

## 1. The omission

The 2-D channel marginalises each candidate host's BH mass with the bare catalogue likelihood
N(M; M_g, σ_M) (numerator: the analytic M_z-fraction marginal, `bayesian_statistics.py`
`numerator_integrant_with_bh_mass`; denominator: the (z, M) importance sampler). Gray et al.
(2020) Eq. A.10 places the host **prior** p(s|M) inside the marginal: the probability that *this*
galaxy hosts the EMRI carries the per-MBH rate R_eff(M) (Babak et al. 2017; the same weight the
host draw and the sum-level w_g use). Omitting it treats all true masses within the measurement
scatter as a priori equally likely to host, which they are not: R_eff falls as
d log₁₀R_eff / d log₁₀M ≈ −0.43 per dex over the catalogue range.

## 2. The correct per-galaxy mass prior and its exact tilted-Gaussian form

    p_g(M) = N(M; M_g, σ_M²) · R_eff(M) / Z_M,   Z_M = ∫ N(M; M_g, σ_M²) R_eff(M) dM.        (1)

Expand ln R_eff to first order around the catalogue value:

    ln R_eff(M) ≈ ln R_eff(M_g) + (α_g / M_g)(M − M_g),   α_g = d ln R_eff / d ln M |_{M_g}.  (2)

A Gaussian times an exponential tilt is exactly a shifted Gaussian,
N(M; μ, σ²)·e^{tM} ∝ N(M; μ + tσ², σ²), and the normalisation Z_M cancels the constants, so

    p_g(M) = N(M; M_g^eff, σ_M²),   M_g^eff = M_g (1 + α_g σ_rel²),  σ_rel = σ_M / M_g.       (3)

Equation (3) is EXACT under (2) — no further approximation. This is the classic Eddington (1913)
correction: the true mass is systematically on the downhill side of the noisy estimate.

## 3. Implementation (gated to `volume_deconv` / `volume_global` kernels)

- **Numerator:** the mass prior enters only via `mu_gal_frac = host_M (1+z) / M_z_det`
  (and the unchanged σ): replace `host_M → M_g^eff` once per call (z-independent shift).
  The analytic Gaussian-product marginal (Bishop 2006, Eq. 2.81-2.82; internal derivation
  Eqs. 14.21-14.31) is preserved as-is.
- **Denominator (MC):** sample from the SAME tilted prior (proposal = prior), so the importance
  weights remain `p_det` exactly as before; only the sampler location moves. Numerator and
  denominator therefore carry the identical mass prior — the "counted exactly once" rule in M.
- α_g by central finite difference of `R_eff_per_mbh` at M_g (ε = 1%); σ_rel clipped at 2.
- `global` / `local_ratio` modes keep the bare Gaussian (legacy/diagnostic reproduction).

## 4. Residual of the log-linear model (stated, tested)

The neglected curvature contributes at O(σ_rel⁴ · d²lnR_eff/d(lnM)²). R_eff is close to a power
law over most of the catalogue mass range; curvature concentrates in the low-mass `kappa_cap`
roll-off. The regression test pins the tilted-Gaussian M-marginal against an exact numerical
quadrature of (1) at σ_rel ∈ {0.2, 0.55, 0.76} for representative masses and asserts the stated
tolerance, so the approximation quality is continuously enforced, not assumed.

## 5. Limiting cases

- σ_M → 0: M_g^eff → M_g — the bare Gaussian is recovered (spec-quality mass limit; the F5
  regime where the channel becomes informative is automatically bias-free).
- Flat rate (α_g = 0): no shift.
- Leading-order equivalence with the external check: shifting the catalogue masses by (3) on the
  PRE-fix code (`scripts/eddington_m_impact.py`) must reproduce the post-fix posterior to
  second order — the before/after triangulation used to validate the change.

## 6. What deliberately stays at next order (documented, unchanged)

- The sum-level rate weight w_g = R_eff(M_g)/(1+z_g) stays point-evaluated (same status as the
  z-channel fix; G2c deviation note).
- The 4-D BallTree candidate search still queries at the catalogue M_g (the prior belongs in the
  likelihood, not the search; the ±1.5σ candidate window is generous against the ≲0.25 |shift|).
- The catalogue-mass-function prior dn_cat/dM is not separately modelled (no model input exists
  for it); the implemented weight captures the rate slope, the dominant model-available gradient.
