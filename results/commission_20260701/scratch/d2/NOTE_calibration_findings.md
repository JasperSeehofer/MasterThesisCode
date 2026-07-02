# Independent calibration (P-P / coverage) test of the dark-siren H0 estimator

Investigator: d2 (independent). Date: 2026-07-01.
Scratch: `results/commission_20260701/scratch/d2/`. Synthetic only; the real
GLADE catalogue was never loaded. The repo inference code was read (to learn the
estimator's exact form) but NOT imported — every estimator here is reimplemented
from scratch.

## Question

Is the per-event selection-corrected likelihood
`p_i(h) = (β_G·L_cat + B_num) / D(h)` — with an in-catalogue term `L_cat` whose
host-redshift marginalization uses a **bare** Gaussian `N(z; z_gal, σ_z)** and a
comoving-volume completion term `B_num` — statistically calibrated? If not, what
makes it so?

## Exact form of the production estimator (from the code)

`bayesian_statistics.py::single_host_likelihood` (numerator, lines ~1789-1807):
the in-catalogue numerator integrates
`p_GW(d_L(z,h)) · N(z; z_gal, σ_z) dz` — **no** `dV_c/dz` factor (a bare Gaussian
in z). By contrast `D(h)` (l.335), `β_Gbar` (l.462) and the completion numerator
`B_num` (l.1641) all carry `dV_c/dz · 1/(1+z)`. So the **numerator uses a flat-in-z
prior while every selection/denominator term uses the comoving-volume prior** —
an internal inconsistency. Since `E(z)` is h-independent, `d_L(z,h)=A(z)/h`
exactly and every overall `1/h^3` cancels; only the *shape*
`w_pop(z) ∝ I(z)²/[E(z)(1+z)]` matters.

## Synthetic universe

Flat ΛCDM Ω_m=0.3. Population `n(z) ∝ dV_c/dz·(1+z)^-1`; smooth detection
selection `p_det(d_L)` (50% at 1.85 Gpc → median detected z≈0.3, Malmquist-biased);
z-dependent completeness `f(z)` (≈0.9 near, →0 by z≈0.5); photo-z error
σ_z=0.035 (the commission value); GW distance error 5%. ~200-250 events /
realization; catalogue galaxies trace `f(z)·w_pop(z)`; a handful-to-tens of
candidate hosts per event. HPD credible regions on a flat h-grid [0.60,0.86].

## Estimators tested

- **A (production)**: bare-Gaussian in-cat numerator + GLOBAL selection
  denominator, `p_i=(β_G L_cat+B_num)/D`.
- **B_local (commission's literal B)**: dV_c/dz-weighted numerator, LOCAL
  self-normalized ratio-of-sums.
- **B_exact (correct)**: per-galaxy volume-prior deconvolution
  `p_g(z)=N(z_g;z,σ_z)·w_pop(z)/Z_g`, `p_i=((1/n0)Σ num_g + B_num)/D`.
- **B_naive (diagnostic)**: numerator multiplied by w_pop but NOT renormalized.

## RESULT 1 — clean controlled isolation (single host, f≈1, no completion)

Purely the in-catalogue numerator z-prior, everything else identical.
120 realizations × 250 events each, three injected H0.

| H0_true | prior  | cov50 | cov68 | cov90 | rail | MAP bias |
|---------|--------|-------|-------|-------|------|----------|
| 0.66 | FLAT (production) | 0.02 | 0.02 | 0.03 | 0.00 | **−0.025** |
| 0.66 | VOLUME (fix)      | 0.61 | 0.73 | 0.88 | 0.00 | −0.002 |
| 0.72 | FLAT (production) | 0.00 | 0.02 | 0.03 | 0.00 | **−0.024** |
| 0.72 | VOLUME (fix)      | 0.53 | 0.66 | 0.88 | 0.00 | −0.003 |
| 0.78 | FLAT (production) | 0.03 | 0.03 | 0.08 | 0.00 | **−0.022** |
| 0.78 | VOLUME (fix)      | 0.55 | 0.73 | 0.88 | 0.00 | −0.002 |

The FLAT (production) numerator's coverage **collapses to ≈0-3%** at all three
truths: a fixed −0.024 (≈3.3%) low bias in H0 dwarfs the ~0.008 statistical
spread of 250 events, so the truth almost never lands in even the 90% CI. The
VOLUME-weighted numerator is well-calibrated (coverage ≈ nominal, bias ≈0).

MAP-tracks-truth: BOTH track (MAP shifts ~1:1 with H0_true), but FLAT carries a
rigid −0.024 offset — it is not "MAP independent of truth", it is a biased-but-
responsive estimator. So the failure is a **bias**, not a dead estimator.

## RESULT 2 — σ_z² signature (mechanism confirmation)

FLAT MAP bias vs σ_z (h_true=0.72): −0.0016 / −0.0064 / −0.023 / −0.046 for
σ_z = 0.005 / 0.015 / 0.035 / 0.050 — i.e. it grows ≈ **σ_z²** and →0 as σ_z→0,
while the VOLUME bias stays ≈−0.002 at every σ_z. This is the textbook signature
of an omitted redshift prior: the bias is
`≈ σ_z² · d ln(dV_c/dz)/dz`, a finite-photo-z Eddington/Malmquist-in-z effect,
maximal exactly in the σ_z≈0.035 regime the catalogue lives in.

## RESULT 3 — full estimators (completion term + interlopers, H0_true=0.72)

120 realizations, ~160 events each, incomplete f(z), catalogue interlopers with a
sky-localization candidate weight, completion term B_num + selection D(h)/β_G.
All four are `p_i=(β_G L_cat + B_num)/D`; they differ only in the in-cat numerator.

| estimator | cov50 | cov68 | cov90 | rail | MAP (truth 0.72) | bias | slope |
|-----------|-------|-------|-------|------|------------------|------|-------|
| A_prod   (production bare-Gaussian num) | 0.00 | 0.00 | 0.02 | 0.00 | 0.682 | **−0.038** | 0.35 |
| B_corr   (volume-prior deconvolution)   | 0.40 | 0.54 | 0.82 | 0.00 | 0.707 | −0.013 | 0.35 |
| B_naive  (dVc mult, no renorm)          | 0.42 | 0.61 | 0.82 | 0.00 | 0.708 | −0.012 | 0.35 |
| A_global (literal global denom)         | 0.00 | 0.00 | 0.00 | 1.00 | 0.860 rail | — | 0.00 |

- **A_prod sits −0.024 below B_corr at every injected H0** (0.659/0.670/0.684/
  0.691/0.700 vs 0.683/0.694/0.708/0.715/0.725 for H0=0.66→0.78) — the identical
  numerator-prior bias from Result 1, reproduced inside the full completion
  machinery. Fixing the numerator prior (B_corr) moves the MAP 0.682→0.707 and
  recovers most of the coverage (0.00→0.40/0.54/0.82).
- A residual −0.013 low bias in B_corr and the compressed slope≈0.35 (shared by
  all non-rail estimators) come from the **completion term being only weakly
  H0-informative**: with an incomplete catalogue the out-of-catalogue term drags
  the MAP toward a fixed selection-geometry value. This is a *separate*, smaller
  effect, sensitive to the synthetic's completeness/interloper modeling — so
  RESULT 1 (complete, single host, no completion) is the decisive verdict.
- **Caveat on A_global**: production's literal GLOBAL selection denominator
  (`β_G·Σ_local num / Σ_global`) needs the discrete catalogue density n0 to match
  β_G exactly; in a from-scratch synthetic that normalization is delicate and it
  railed here. That is a limitation of the synthetic, NOT evidence about the real
  code (which sums the full GLADE catalogue). The z-prior verdict does not depend
  on it — Results 1/2 use no global denominator.

## Conclusion

The production-style estimator is **NOT statistically calibrated**. Its
in-catalogue numerator marginalizes the host photo-z against a *flat* redshift
prior while the selection denominator uses the *comoving-volume* prior; with
σ_z≈0.035 this omission Eddington-underestimates each host's true redshift and
therefore H0 by ≈2-3% (≈σ_z²·d ln(dV_c/dz)/dz), a fixed bias that destroys
coverage (≈0-3% vs nominal) even though the MAP still tracks the truth. What
makes it calibrated is **weighting the host-redshift integrand by the same
comoving-volume element dV_c/dz·(1+z)^-1 used in D(h)** — specifically the
per-galaxy volume-prior deconvolution `p_g(z) ∝ N(z_g;z,σ_z)·w_pop(z)`, which
restores coverage from ≈0 to nominal in the clean test and removes the −0.024
bias in the full test.

The dominant, unambiguous defect is the **missing comoving-volume prior in the
in-catalogue numerator**. Two secondary points, each less firmly established by a
from-scratch synthetic and worth a follow-up on the real pipeline: (i) the
completion (out-of-catalogue) term is only weakly H0-informative and can drag the
MAP toward a selection-geometry value when the catalogue is very incomplete;
(ii) production's GLOBAL selection-denominator restructure is only correct if the
discrete catalogue number density n0 exactly reproduces β_G — a normalization
that should be checked directly against the real GLADE sum. Neither of these
changes the primary verdict: **with σ_z≈0.035 the production bare-Gaussian
numerator is not calibrated, and volume-weighting the host-z integrand fixes it.**

## Reproduce
- `clean_singlehost_test.py`  — Results 1 (coverage) and the σ_z scan (Result 2).
- `pp_coverage_test.py`       — Result 3 full estimators; writes `coverage_results.json`.
- `make_pp_plot.py`           — `pp_coverage_plot.png` + `clean_pp_summary.json`.
All pure-numpy/scipy, no repo imports, no GLADE; run with `uv run python <file>`.
