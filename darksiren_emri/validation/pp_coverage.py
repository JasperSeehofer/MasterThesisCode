"""Independent P-P / coverage harness for the dark-siren H0 estimator family.

This module is the calibration instrument for the dark-siren estimator: it
builds a synthetic galaxy catalogue and EMRI detections in a flat LambdaCDM
universe, runs a from-scratch single-host dark-siren H0 estimator with a
switchable host-redshift kernel (``"bare"`` Gaussian ``N(z; z_gal, sigma_z)``
vs ``"volume"``-weighted ``N(z; z_gal, sigma_z) * dV_c/dz / (1+z)``), and
measures frequentist P-P coverage (50/68/90% HPD credible intervals) and MAP
bias across many realizations and injected truths.

Scientific independence
-----------------------
The harness is pure numpy/scipy and deliberately does NOT import the
production inference code (``darksiren_emri.bayesian_inference``). That
independence is its scientific value: it re-derives the estimator from the
written formulas, so a calibration failure here cannot be explained away as a
shared implementation bug. It was written from scratch by the 2026-07-01
verification commission (investigator d2); the original scratch version,
findings note and reference outputs live in
``results/commission_20260701/scratch/d2/`` (see
``NOTE_calibration_findings.md`` and ``coverage_results.json``) and the
commission verification report section 7.

Key commission finding reproduced by this harness: with photo-z scatter
``sigma_z ~= 0.035`` the bare-Gaussian host-z kernel carries a fixed
``~ -sigma_z^2 * d ln(dV_c/dz)/dz`` (Eddington/Malmquist-in-z) low bias in H0
that collapses coverage to ~0-3%, while the volume-weighted kernel is
calibrated (coverage ~= nominal, bias ~= 0).

Catalogue-support-truncated mode (``PPCoverageConfig.z_support``): splits the
detected population by true host redshift so hosts with ``z_host >=
z_support`` become zero-host events driven by the pure-completion likelihood
B_num(h)/D(h) — the ``L_cat -> 0`` limit of the Gray et al. (2020,
arXiv:1908.06050, Eqs. 29+32) mixture that production commit ``8db6c6e``
(issue #29) installed in ``bayesian_statistics.py``. ``z_support=None``
(default) reproduces the pre-2026-07-10 harness bit-identically. See
``.planning/HANDOFF-LOCAL-NO-CLUSTER-20260710.md`` (item L-A) and
``results/pp_coverage_deepvenue_20260710/RUNBOOK.md``.

Mixture modes (``PPCoverageConfig.mixture_mode``, EXP-41 / handoff item N-1,
``.planning/HANDOFF-DEEP-BIAS-MECHANISM-20260710.md``): ``"two_branch"``
(default) keeps the clean limit above bit-identically; ``"gray"`` gives
host-found events the full Gray et al. (2020, Eqs. 29+32) mixture
``(beta_G * L_cat_i + B_num) / D`` with the per-host selection denominator
``D_g_i`` of Eqs. A.9/A.10 (the production commit ``713fbd1`` analog) while
zero-host events keep the pure-completion ``B_num/D`` branch;
``"conditioned"`` is the membership-conditioned inverse (N-2b probe):
host events ``N_i / beta_G``, zero-host events ``B_num / beta_Gbar``;
``"exact"`` (quick task 260711-117) is the membership-truncated exact
kernel: under this harness's generative model detection is conditioned once
via ``1/D(h)`` with NO p_det inside the numerator (Mandel, Farr & Gair 2019,
arXiv:1809.02063) and catalogue membership ``G = 1[z_true < z_support]`` is
part of the observed data, so the exact host-event numerator is the
volume-kernel integral TRUNCATED at the support edge ``z_support`` (no
beta_G, no D_g_i) while zero-host events keep ``B_num/D`` — the two branches
tile ``[0, Z_MAX_POP]`` exactly (the support split of the Gray et al. 2020,
arXiv:1908.06050, Eqs. 29+32 completion mixture). The
``membership_on_observed`` flag (N-2d probe) decides catalogue membership on
the observed ``z_gal`` instead of the true ``z_host``.
``"absolute"`` (2026-07-26, harness analog of
``results/lcat_h_dependence_20260725/DERIVATION_ESTIMATOR_REDESIGN.md``
Variant 1, Eq. (2)) is the absolute-mass marginal: host events get
``p_i(h) = [N_i(h) + B_num_i(h)] / D(h)`` with NO self-normalization of the
catalogue term by any per-event or per-host selection denominator (neither
``D_g_i`` as in "gray" nor ``beta_G`` as a multiplicative weight). This is
the harness realization of production's ``A_i(h) = Sigma_ball w_g N_g(h) /
n_bar_w(h)`` with ``n_bar_w(h) = Sigma_glob(h) / beta_G(h)``: because the
harness's synthetic "catalogue" is a smooth one-candidate-per-event
continuum population (not a discrete multi-galaxy table sharing a Sigma_glob
distinct from beta_G), the continuum idealization ``Sigma_glob(h) ==
beta_G(h)`` collapses ``n_bar_w(h)`` to 1 identically, and
``A_i(h) = N_i(h)`` exactly (the ``beta_G`` factor that "gray" mode applies
to ``L_cat_i`` cancels against the same ``beta_G`` inside ``n_bar_w``). Both
terms are summed for EVERY host-found event (not just the zero-host
fallback) using the SAME ``B_num_i`` completion integral zero-host events
use, so the zero-host branch (``B_num/D``, shared with "two_branch"/"gray")
is the continuous ``N_i -> 0`` limit of this formula, not a separate branch.
HARNESS-FIDELITY CAVEAT: this harness structurally cannot construct genuine
impostor-only candidate balls (multiple false candidates carrying zero
true-host mass) because each event has exactly one candidate (its own noisy
``z_gal``) rather than a shared multi-galaxy catalogue queried by many
events; it therefore cannot exercise Variant 1's core impostor-suppression
claim (derivation Sec 3.4(b)). It CAN test Variant 1's other two claims: the
continuous zero-host/near-zero-host fallback limit and the unbiased
complete-catalogue limit.

Catalogue / impostor-ball universe (``PPCoverageConfig.catalogue_mode``,
2026-07-26) — see ``results/pp_impostor_harness_20260726/
DERIVATION_HARNESS_ANALOG.md`` for the full derivation and the
production<->harness correspondence table. Every mode described above builds a
universe in which each detected event has EXACTLY ONE candidate host (its own
noisy observed redshift), so no estimator that must CHOOSE among candidates can
be exercised. ``catalogue_mode=True`` replaces that generative model with a
discrete, frozen, shared galaxy catalogue plus hard sky localization balls:

* ``n_galaxies`` galaxies are drawn once per run with true redshifts from the
  comoving-volume galaxy number density ``n_gal(z) propto dV_c/dz =
  (1+z) w_pop(z)`` on ``[Z_MIN, Z_MAX_POP]`` and directions uniform on the
  sphere.
* A galaxy is CATALOGUED iff ``z_true < z_support`` (the harness's sky-averaged
  completeness ``fbar(z) = 1[z < z_support]``); catalogued galaxies carry a
  noisy observed redshift ``z_obs = z_true + N(0, sigma_z)``.
* An EMRI host is drawn from ALL galaxies with rate weight
  ``w_g = 1/(1+z_g)`` (so the host redshift density is
  ``n_gal(z) w(z) = w_pop(z)``, identical to the continuum harness) and
  detected with probability ``p_det(A(z)/h_true)``.
* The GW sky datum is a hard cap of solid-angle fraction ``sky_frac =
  dOmega/4pi``, positioned so the true host is UNIFORM inside it (making the
  flat in-cap sky likelihood exact). The candidate ball is every CATALOGUED
  galaxy inside the cap — the true host if it is catalogued, plus genuine
  foreground/background impostors, or impostors only when the host is not
  catalogued.

Catalogue-mode estimators (``mixture_mode``): ``"lcat"`` (the legacy
self-normalized Gray-A9 ratio-of-sums that production's ``volume_deconv``
implements), ``"absolute"`` (production ``absolute_marginal``, V1) and
``"generator_marginal"`` (the production stack of
``results/lcat_h_dependence_20260725/DERIVATION_GENERATOR_CONSISTENT_NORM.md``
and ``DERIVATION_ZRESOLVED_SURVIVAL.md``). All three share the same per-event
form ``p_i = [in-catalogue term + B_num,i] / denominator`` and differ ONLY in
how the discrete catalogue sum is put on the same absolute scale as the
continuum completion term:

===================  ==========================================  ==================
mixture_mode         in-catalogue term                            denominator
===================  ==========================================  ==================
``lcat``             ``beta_G(h) * (Sum_ball w N)/(Sum_ball w D_g)``  ``D = beta_G + beta_Gbar``
``absolute``         ``(Sum_ball w N) / (n_bar_w(h) * sky_frac)``     ``D = beta_G + beta_Gbar``
``generator_marginal`` ``(Sum_ball w N) / (n_hat_w * sky_frac)``      ``D_gen = Sigma_glob/n_hat_w + beta_Gbar``
===================  ==========================================  ==================

with ``n_bar_w(h) = Sigma_glob(h)/beta_G(h)`` (production's Option-A
calibration) and ``n_hat_w = W_cat / V_f`` (production's generator-consistent
draw-side rate-weight density; h-independent in this harness because the
common ``h^-3`` of the comoving volume cancels between every numerator and
denominator term — all four terms are homogeneous of degree one in
``w_pop``). The ``sky_frac`` factor is the harness's explicit analog of
production's pixel solid angles: it converts the all-sky rate-weight density
``n_hat_w`` into the expected in-cap density, so that
``E[in-catalogue term] = int fbar(z) w_pop(z) p_GW(z;h) dz`` tiles exactly
against ``B_num,i = int (1-fbar) w_pop p_GW dz``.

Remaining simplifications (the harness is NOT production; see the derivation
note section 6 for the full list). (i) The z-resolved-survival fix
(``DERIVATION_ZRESOLVED_SURVIVAL.md``) is VACUOUS here: this harness's
``p_det(A(z)/h)`` is an exact deterministic function of ``d_L``, so the
harness's survival is already the z-conditional ``S(d_L|z)`` — there is no
detector-frame-mass lift and hence no pooled-vs-conditional discrepancy to
repair. The harness therefore sits in FIX-2's FIXED state by construction.
(ii) The harness catalogue is drawn from exactly the population the estimator
models, so production's Option-A identity ``Sigma_glob = n_hat_w * beta_G``
holds up to Poisson noise and the sigma_z kernel asymmetry; ``absolute`` and
``generator_marginal`` therefore nearly coincide here, and the harness cannot
adjudicate FIX-3 on its own. What the harness CAN adjudicate is the
misassociation mechanism that motivated V1: ``lcat`` vs the two absolute-mass
forms on identical impostor-bearing universes. (iii) No mass dimension, no
pixelated completeness, no GW sky-likelihood shape (hard cap only), and the
per-galaxy redshift kernel is NOT truncated at the completeness edge — the
production-faithful "above-edge leak" that ``mixture_mode="exact"`` studies for
the single-candidate universe.

Prior-tilt probe (``PPCoverageConfig.inference_wpop_tilt``, handoff item N-3,
``.planning/HANDOFF-DEEP-BIAS-MECHANISM-20260710.md``): multiplies ONLY the
INFERENCE-side population weight w_pop(z) by ``exp(gamma * z)`` — the
generative truth draw (``_sample_detected_redshifts``) is never tilted — so
the harness measures inference-prior *misspecification* against a fixed
truth. The gate is strict (``gamma == 0.0`` returns the untilted weight
object unchanged), keeping the default path bit-identical.

Mass channel (``PPCoverageConfig.mass_channel``, 2026-08-17; ledger row #120
item 2 / D-2, built to the ``[A3]`` acceptance criteria of
``results/campaign51_20260728/realistic_20260729/
CLAIM_PRODUCTION_CALIBRATION_HARNESS_20260817.md`` §4). Every mode above is
mass-free: production's SECOND observable — the detector-frame BH mass — and
therefore the whole 2D leg that the selection fusion ([PHYSICS] commit
``2b10b8b8``, ledger rows #117-#118) touches could not be exercised.
``mass_channel=True`` (requires ``catalogue_mode``) adds:

* **A mass observable.** Every galaxy carries a source-frame BH mass drawn
  from ``phi(M)`` (:func:`dark_mass_density_per_mass`, a single power law on
  ``[M_SOURCE_MIN, M_SOURCE_MAX]`` — the harness's analog of production's
  piecewise Babak et al. 2017 ``phi``); catalogued galaxies carry a noisy
  fractional mass measurement (``sigma_m_gal_frac``). Each detected event
  carries a measured detector-frame mass ``M_z,det,i`` drawn jointly with
  ``d_L_obs`` from a 2x2 fractional covariance
  ``(sigma_dl_frac, sigma_mz_frac, rho_dl_mz)`` — the harness's ``cov_4d``
  ``(d_L, M_z)`` block, from which ``proj`` and ``sigma_cond`` follow exactly
  as production reads them off ``cov_4d`` (Bishop 2006 PRML Eqs. 2.81-2.82).
* **Mass-dependent selection.** ``S_4D(d_L, M_z)``
  (:func:`survival_with_mass`) rescales the detection horizon as
  ``d50 (M_z/1e6)^mass_horizon_index``; ``mass_horizon_index = 0`` recovers
  the mass-blind :func:`detection_probability` EXACTLY, so a mass-BEARING cell
  must set it ``> 0``. Its phi-marginal ``S_bar_phi(z;h)``
  (:func:`phi_marginal_survival_table`, the analog of production's
  ``precompute_phi_marginal_survival``) replaces ``p_det`` in EVERY selection
  integral (``D(h)``, ``beta_G``, ``Sigma_glob``) — the harness analog of
  production's phi-convention legs ``precompute_phi_selection_integrals``.
* **Two channels per realization**, sharing one generative universe and one
  denominator, exactly as production's per-event both-channel diagnostics: the
  **1D** channel discards the mass observable from the likelihood (it still
  drives selection) and is reported in the usual top-level result block; the
  **2D** channel adds a per-candidate Gaussian mass overlap to the catalogue
  leg (production's analytic ``mz_integral``) and the completion-leg mass
  factor to the completion leg, and is reported under the nested
  ``"mass_channel_2d"`` key.
* **The completion-leg mass factor** :func:`completion_mass_factor_g` and its
  fused form :func:`completion_mass_factor_g_sel` (``S_4D`` inside the SAME
  ``dx_M``), the harness analogs of production's ``completion_mass_factor_g``
  (``bayesian_statistics.py:2022``) and ``completion_mass_factor_g_sel``
  (``:2155``). The h grid is an ARRAY AXIS of the node block, so ``g`` is
  recomputed at every h-grid point by construction — never frozen, never
  elided.

Selection cell (``PPCoverageConfig.selection_cell``) mirrors production's
``selection_in_completion_numerator`` (``bayesian_statistics.py:3010``,
``:3037-3047``): ``"off"`` = the pre-#118 estimator, ``"1d"`` = [P2] only
(``S_bar_phi(z;h)`` inside the 1D completion numerator, ``:4495-4514``),
``"2d"`` = [P1] only (``g_sel`` in the 2D completion leg, ``:4592-4609``),
``"fused"`` = the landed production pairing. The two legs are channel-local by
construction, so ``off``/``2d`` share their 1D result bit-for-bit and
``off``/``1d`` share their 2D result (pinned by a test).

Noise-model cells (``--noise-model``; Q-0 audit, 2026-08-17). The committed
"const-sigma" convention conflates two error sub-terms: **(a)** sigma
evaluated at a SCATTERED ``d_L_obs`` instead of ``d_L_true``, and **(b)** the
width not varying across the z-integral (the dropped ``1/sigma(z)``).
Production carries **only (b)**: its per-event sigma is the Fisher CRB frozen
at the injected truth and there is NO measurement scatter — ``d_L_obs`` is
identically ``d_L_true`` on the production path
(``bayesian_statistics.py:3543``, ``:3613``, ``:4442-4443``;
``detection.py:133-136``). The three cells are therefore
``const`` = (a)+(b) (the committed convention, ``gw_measurement_scatter=True``
+ ``sigma_dl_model_in_likelihood=False``), ``model`` = neither
(``sigma_dl_model_in_likelihood=True``), and ``production`` = **(b) only**
(``gw_measurement_scatter=False`` + const sigma) — the production-faithful
cell. The discarded-not-skipped RNG draw keeps every scatter/no-scatter pair
on the same random stream, so the A/B is paired.

Runtime (measured 2026-08-17, single CPU core, dev machine; nh = 66 h-grid
nodes, ``n_z_quad=160``, ``n_hermite=24``, ``n_galaxies=200000``,
``sky_frac=1e-4``, ``mixture_mode="absolute"``): a production-N realization
(``n_events=1600``, both channels, ``selection_cell="fused"``,
``mass_horizon_index=0.25``) takes **~8.5 s**; ``selection_cell="off"``
(no ``S_4D`` per Hermite node) **~3.1 s**; the pre-existing mass-free
catalogue realization at the same N is **~0.4 s** and is untouched by this
extension (its code path is unchanged, and byte-identity is pinned by
``darksiren_emri_test/validation/test_pp_coverage_mass.py``). A full
mass-bearing [R-3]-scale cell (3 truths x 120 realizations at N=1600) is
therefore ~50 CPU-min per truth, embarrassingly parallel over truths. Tune
``event_chunk`` (events per vectorized block, default 16) for the memory /
speed trade-off; it never changes results.

Units: ``h`` in [100 km/s/Mpc]; distances in Gpc. Cosmology: flat LambdaCDM.
"""

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy.special import erfc

C_KM_S = 299_792.458
OMEGA_M = 0.30
OMEGA_L = 0.70

D50_GPC = 1.85  # 50% detection-probability luminosity distance [Gpc]
W_PDET_GPC = 0.30  # detection roll-off width [Gpc]

Z_MIN = 1e-4
Z_MAX_POP = 0.95  # population / catalogue redshift ceiling

# ----------------------------------------------------------------------------
# Cosmology tables (flat LambdaCDM): d_L(z, h) = A(z) / h with A in Gpc, and
# population weight w_pop(z) propto dV_c/dz / (1+z) (the 1/h^3 cancels).
# ----------------------------------------------------------------------------
_Z_GRID: npt.NDArray[np.float64] = np.linspace(0.0, 1.5, 15_001)
_E_OF_Z: npt.NDArray[np.float64] = np.sqrt(OMEGA_M * (1.0 + _Z_GRID) ** 3 + OMEGA_L)
_INV_E: npt.NDArray[np.float64] = 1.0 / _E_OF_Z
_I_OF_Z: npt.NDArray[np.float64] = np.concatenate(
    [
        np.array([0.0]),
        np.cumsum(0.5 * (_INV_E[1:] + _INV_E[:-1]) * np.diff(_Z_GRID)),
    ]
)
# A(z) = (1+z) * (c / 100 km/s/Mpc) * I(z) in Mpc, / 1000 -> Gpc.
_A_GPC: npt.NDArray[np.float64] = (1.0 + _Z_GRID) * (C_KM_S / 100.0) * _I_OF_Z / 1000.0
# w_pop(z) propto I(z)^2 / E(z) / (1+z)  (comoving volume element / (1+z))
_W_POP: npt.NDArray[np.float64] = np.where(
    _Z_GRID > 0.0, _I_OF_Z**2 / _E_OF_Z / (1.0 + _Z_GRID), 0.0
)


def comoving_amplitude_of_z(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Interpolate A(z) [Gpc] such that d_L(z, h) = A(z) / h.

    Args:
        z: Redshift values.

    Returns:
        A(z) in Gpc.
    """
    return np.interp(z, _Z_GRID, _A_GPC)


def z_of_comoving_amplitude(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Invert A(z): redshift at which d_L * h equals ``a`` [Gpc].

    Args:
        a: Amplitude values A = d_L * h in Gpc.

    Returns:
        Redshift values.
    """
    return np.interp(a, _A_GPC, _Z_GRID)


def population_weight_of_z(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Population redshift weight w_pop(z) propto dV_c/dz / (1+z) (unnormalized).

    Args:
        z: Redshift values.

    Returns:
        Unnormalized population weight at each redshift.
    """
    return np.interp(z, _Z_GRID, _W_POP)


def galaxy_number_weight_of_z(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Galaxy NUMBER weight n_gal(z) propto dV_c/dz (unnormalized, catalogue mode).

    The harness's galaxies have constant comoving number density, so their
    redshift density is the bare comoving volume element ``dV_c/dz``; the
    ``1/(1+z)`` of the RATE weight ``w_pop(z) = dV_c/dz / (1+z)`` is the
    per-galaxy EMRI rate suppression (observer-frame time dilation), carried
    separately by :func:`host_rate_weight_of_z`. Hence
    ``n_gal(z) * w(z) == w_pop(z)`` identically, which is what makes the
    catalogue-mode host draw share its redshift density with the continuum
    harness's :func:`_sample_detected_redshifts`.

    Args:
        z: Redshift values.

    Returns:
        Unnormalized galaxy number weight at each redshift.
    """
    return np.asarray(
        (1.0 + np.asarray(z, dtype=np.float64)) * population_weight_of_z(z), dtype=np.float64
    )


def host_rate_weight_of_z(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Per-galaxy EMRI rate weight w(z) = 1/(1+z) (observer-frame time dilation).

    Harness analog of production's ``w_g = R_eff_per_mbh(M_g)/(1+z_g)``
    (``bayesian_statistics._rate_weight``) with the mass factor dropped: this
    harness has no mass dimension, so every galaxy has the same intrinsic rate.

    Args:
        z: Redshift values.

    Returns:
        Rate weight at each redshift.
    """
    return np.asarray(1.0 / (1.0 + np.asarray(z, dtype=np.float64)), dtype=np.float64)


def _inference_population_weight(
    z: npt.NDArray[np.float64], tilt: float
) -> npt.NDArray[np.float64]:
    """Inference-side population weight w_pop(z) * exp(tilt * z) (N-3 prior-tilt probe).

    ``tilt == 0.0`` returns ``population_weight_of_z(z)`` UNCHANGED (strict gate ->
    bit-identical default path, all golden pins hold). ``tilt != 0.0`` multiplies by
    ``exp(tilt * z)`` — the prior-misspecification perturbation applied to INFERENCE-side
    w_pop only. The generative truth draw (``_sample_detected_redshifts``) never calls this
    and is therefore never tilted, so the probe measures inference-prior misspecification
    against a fixed truth.

    Args:
        z: Redshift values.
        tilt: Exponential tilt coefficient gamma [1/z].

    Returns:
        Tilted (or, at ``tilt == 0.0``, untilted) unnormalized population weight.
    """
    w = population_weight_of_z(z)
    if tilt == 0.0:
        return w
    return np.asarray(w * np.exp(tilt * np.asarray(z)), dtype=np.float64)


def _inference_galaxy_number_weight(
    z: npt.NDArray[np.float64], tilt: float
) -> npt.NDArray[np.float64]:
    """Inference-side galaxy number weight ``(1+z) * w_pop_tilted(z)`` (catalogue mode).

    Keeps the ``n_gal * w == w_pop`` identity exact under the N-3 prior tilt.

    Args:
        z: Redshift values.
        tilt: Exponential tilt coefficient gamma [1/z] (0.0 = untilted).

    Returns:
        Unnormalized galaxy number weight.
    """
    return np.asarray(
        (1.0 + np.asarray(z, dtype=np.float64)) * _inference_population_weight(z, tilt),
        dtype=np.float64,
    )


def detection_probability(
    d_L: npt.NDArray[np.float64],
    d50: float = D50_GPC,
    w_pdet: float = W_PDET_GPC,
) -> npt.NDArray[np.float64]:
    """Smooth Malmquist detection probability p_det(d_L).

    Args:
        d_L: Luminosity distance values in Gpc.
        d50: 50% detection-probability luminosity distance [Gpc]. Defaults to
            the module ``D50_GPC`` (1.85, the commission venue); lower values
            model a shallower detection horizon (N-4 venue-depth probe).
        w_pdet: Detection roll-off width [Gpc]. Defaults to ``W_PDET_GPC``.

    Returns:
        Detection probability in [0, 1] (50% at ``d50``).
    """
    return np.asarray(
        0.5 * erfc((np.asarray(d_L) - d50) / (np.sqrt(2.0) * w_pdet)),
        dtype=np.float64,
    )


def _norm_pdf(
    x: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64] | float,
    sig: npt.NDArray[np.float64] | float,
) -> npt.NDArray[np.float64]:
    """Gaussian probability density N(x; mu, sig)."""
    return np.asarray(
        np.exp(-0.5 * ((x - mu) / sig) ** 2) / (np.sqrt(2.0 * np.pi) * sig),
        dtype=np.float64,
    )


def _sample_detected_redshifts(
    h_true: float,
    n: int,
    rng: np.random.Generator,
    ngrid: int = 2000,
    d50: float = D50_GPC,
    w_pdet: float = W_PDET_GPC,
) -> npt.NDArray[np.float64]:
    """Draw host redshifts from the detected population w_pop(z) * p_det(d_L(z, h))."""
    zg = np.linspace(Z_MIN, Z_MAX_POP, ngrid)
    pdf = np.clip(
        population_weight_of_z(zg)
        * detection_probability(comoving_amplitude_of_z(zg) / h_true, d50, w_pdet),
        0.0,
        None,
    )
    cdf = np.concatenate([np.array([0.0]), np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(zg))])
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, zg)


def _hpd_contains(
    h_grid: npt.NDArray[np.float64],
    post: npt.NDArray[np.float64],
    h_true: float,
    level: float,
) -> bool:
    """Return True if ``h_true`` lies inside the HPD credible region of mass ``level``."""
    dh = np.gradient(h_grid)
    mass = post * dh
    order = np.argsort(post)[::-1]
    csum = np.cumsum(mass[order])
    k = int(np.searchsorted(csum, level))
    k = min(k, order.size - 1)
    thresh = float(post[order[k]])
    p_true = float(np.interp(h_true, h_grid, post))
    return p_true >= thresh


@dataclass
class PPCoverageConfig:
    """Configuration for the P-P / coverage harness.

    Args:
        n_realizations: Independent synthetic universes per injected truth.
        n_events: Detected EMRI events per realization.
        sigma_z: Host photo-z scatter (commission value 0.035).
        sigma_dl_frac: Fractional GW luminosity-distance error.
        injected_truths: Injected H0 values [100 km/s/Mpc]; defaults include
            near-grid-edge truths to exercise rail behaviour.
        seed: Master seed; all randomness flows from
            ``np.random.default_rng(seed)`` (fully deterministic).
        kernel: Host-z kernel — ``"bare"`` Gaussian (production-style) or
            ``"volume"``-weighted (calibrated).
        h_min: Lower edge of the H0 grid.
        h_max: Upper edge of the H0 grid.
        h_step: H0 grid spacing.
        n_z_quad: Per-event redshift quadrature points.
        inference_wpop_tilt: N-3 prior-tilt probe gamma [1/z]: multiplies the
            INFERENCE-side w_pop by exp(gamma * z) at every inference call
            site (host kernel, B_num, D(h), beta_G), gated strictly on
            != 0.0; the generative truth draw is untouched. Default 0.0 is
            bit-identical to the untilted harness.
        z_support: Catalogue support ceiling: true hosts with z_host <
            z_support are in the catalogue (existing single-host kernel
            branch); z_host >= z_support are zero-host events using the
            pure-completion likelihood B_num/D (issue #29 analog). None
            (default) => no truncation, bit-identical to the pre-2026-07-10
            harness.
        mixture_mode: Estimator composition under z_support truncation.
            "two_branch" (default, pre-2026-07-11 behaviour): in-catalogue
            events use the bare kernel numerator N_i/D, zero-host events
            B_num/D. "gray": in-catalogue events use the full Gray et al.
            (2020, arXiv:1908.06050, Eqs. 29+32) mixture
            (beta_G * L_cat_i + B_num)/D with L_cat_i = N_i/D_g_i and the
            per-host selection denominator D_g_i of Eqs. A.9/A.10 (production
            commit 713fbd1 analog); zero-host events keep B_num/D.
            "conditioned": membership-conditioned inverse (N-2b probe) —
            in-catalogue N_i/beta_G, zero-host B_num/beta_Gbar.
            "exact": membership-truncated exact kernel (260711-117).
            Derivation: under the harness generative model detection is
            conditioned once via 1/D(h) with NO p_det inside the numerator
            (Mandel, Farr & Gair 2019, arXiv:1809.02063), and catalogue
            membership G = 1[z_true < z_support] is part of the observed
            data; conditioning the host-z kernel on G truncates its support
            at z_support, so the exact host-event likelihood is the
            volume-kernel numerator integrated over
            [z_lo, min(z_hi, z_support)] divided by the shared D(h) — no
            beta_G weight, no per-host D_g_i. Zero-host events keep B_num/D,
            so the two branches tile [0, Z_MAX_POP] exactly (the support
            split of the Gray et al. 2020, arXiv:1908.06050, Eqs. 29+32
            completion mixture). This removes the above-edge kernel leak
            that the two_branch/gray host numerators carry.
            "absolute" (2026-07-26): the absolute-mass marginal (harness
            analog of DERIVATION_ESTIMATOR_REDESIGN.md Variant 1, Eq. 2):
            host events get [N_i(h) + B_num_i(h)]/D(h) — the catalogue
            numerator N_i enters WITHOUT any self-normalization (no beta_G
            weight, no D_g_i division), summed directly with the SAME
            B_num_i completion integral zero-host events use. Zero-host
            events keep B_num/D (the continuous N_i -> 0 limit). See the
            module-docstring "absolute" paragraph for the harness-fidelity
            caveat (no impostor-ball mechanism exists in this harness).
            "lcat" and "generator_marginal" (2026-07-26) are CATALOGUE-MODE
            ONLY (require ``catalogue_mode=True``): "lcat" is the legacy
            self-normalized Gray-A9 ratio-of-sums (production
            ``volume_deconv``) and "generator_marginal" the
            generator-consistent normalization
            (``DERIVATION_GENERATOR_CONSISTENT_NORM.md`` Eqs. 3-5).
            "absolute" works in BOTH universes (see the module docstring
            correspondence table for its catalogue-mode form).
            Modes other than "two_branch" require z_support (ValueError
            otherwise).
        catalogue_mode: Use the discrete-catalogue / impostor-ball generative
            model (see the module docstring). Requires ``z_support`` and
            ``mixture_mode`` in {"lcat", "absolute", "generator_marginal"}.
        n_galaxies: Galaxies in the synthetic catalogue (catalogue mode).
        sky_frac: GW sky-localization cap solid-angle fraction dOmega/(4 pi);
            the expected candidate-ball occupancy is
            ``n_galaxies * sky_frac * (catalogued fraction)``.
        resample_catalogue_per_realization: Redraw the catalogue per
            realization instead of freezing one shared table for the run.
        membership_on_observed: Decide catalogue membership on the OBSERVED
            z_gal (< z_support) instead of the true z_host (production's
            BallTree sees measured redshifts; N-2d probe). Default False keeps
            the true-z routing bit-identical.
        pdet_in_numerator: Latent-detection exact-inverse probe (quick task
            260711-27m, floor mechanism). The harness generative model decides
            detection on the TRUE z (``_sample_detected_redshifts`` draws z
            from w_pop * p_det BEFORE the dL_obs/z_gal noise draws), so
            detection is independent of the data given z and the exact
            conditional keeps p_det(A(z)/h) INSIDE the numerator integrals:

                p(data, G | detected, h)
                    = int 1_G(z) p_GW(dL_obs|z,h) [N(z; z_gal, sigma_z)]
                      p_det(A(z)/h) w_pop(z) dz / D(h).

            The Mandel, Farr & Gair (2019, arXiv:1809.02063) no-p_det-inside
            form applies when detection is a deterministic function of the
            OBSERVED data; for latent-thresholded detection the factor stays
            inside. When True, both branch numerators (host kernel integral in
            every mixture_mode, and the zero-host completion integral B_num)
            are multiplied by p_det(A(z)/h) on the quadrature grid. Default
            False keeps every existing mode bit-identical.
        sigma_dl_model_in_likelihood: σ(dL_obs)-vs-σ(dL_true) noise-model probe
            (quick task 260711-hx1, floor mechanism). The generative distance
            noise is drawn with σ = sigma_dl_frac * dL_true (line
            ``dL_obs = dL_host + N(0, sigma_dl_frac * dL_host)``), so the noise
            width scales with the TRUE distance and varies along the redshift
            integral. The default inference likelihood approximates this with a
            CONSTANT, observed-distance width ``sig_dl_i = sigma_dl_frac *
            dL_obs`` — an O(sigma_dl_frac**2) mismatch that leaves a small
            σ_z-independent residual. When True, the GW-likelihood factor is
            evaluated with the z-dependent model/true-distance width
            ``sigma_dl_frac * A(z)/h`` (shape ``(nz, nh)``), carrying its own
            1/σ(z) normalization via ``_norm_pdf`` — i.e.
            ``N(dL_obs; A(z)/h, sigma_dl_frac * A(z)/h)``. Applies to the host
            kernel numerator (every mixture_mode) and the completion B_num; the
            p_det SELECTION integrals (``D(h)``, gray ``D_g_i``) are unchanged
            (they integrate p_det, not the GW likelihood). Combined with
            ``pdet_in_numerator=True`` this is the fully-consistent exact
            conditional for the latent-thresholded generative model. Default
            False keeps every existing mode bit-identical.
    """

    n_realizations: int = 120
    n_events: int = 250
    sigma_z: float = 0.035
    # Flat peculiar-velocity redshift-error term, added in quadrature to
    # sigma_z for BOTH the generative truth scatter and the inference kernel
    # (the calibrated case). Default 0.0 keeps the committed anchor runs
    # bit-identical. Issue #16: the production kernel uses
    # (1+z) * SIGMA_V_PEC_KM_S / c; this harness knob is flat because its
    # sigma_z is flat too.
    sigma_z_pv: float = 0.0
    sigma_dl_frac: float = 0.05
    injected_truths: list[float] = field(default_factory=lambda: [0.62, 0.72, 0.84])
    seed: int = 20260701
    kernel: Literal["bare", "volume"] = "volume"
    h_min: float = 0.600
    h_max: float = 0.860
    h_step: float = 0.004
    n_z_quad: int = 160
    inference_wpop_tilt: float = 0.0
    z_support: float | None = None
    mixture_mode: Literal[
        "two_branch", "gray", "conditioned", "exact", "absolute", "lcat", "generator_marginal"
    ] = "two_branch"
    membership_on_observed: bool = False
    # --- Catalogue / impostor-ball universe (2026-07-26) ------------------
    # catalogue_mode=True replaces the one-candidate-per-event generative model
    # with a discrete frozen galaxy catalogue plus hard sky-localization balls,
    # so estimators that must CHOOSE among candidate hosts can be exercised.
    # Requires z_support (the completeness edge fbar(z) = 1[z < z_support]) and
    # mixture_mode in {"lcat", "absolute", "generator_marginal"}. Default False
    # keeps every pre-existing mode and golden pin bit-identical.
    catalogue_mode: bool = False
    n_galaxies: int = 200_000
    # Sky-localization cap solid-angle fraction dOmega/(4 pi).
    sky_frac: float = 1.0e-4
    # Redraw the galaxy catalogue for every realization (independent universes)
    # instead of freezing one shared catalogue for the whole run (production-
    # faithful: production has ONE GLADE+ table). Default False.
    resample_catalogue_per_realization: bool = False
    pdet_in_numerator: bool = False
    sigma_dl_model_in_likelihood: bool = False
    # Detection-horizon knobs (N-4 shallow-venue depth probe). Defaults =
    # module D50_GPC/W_PDET_GPC (the commission venue, z_median ~ 0.3); lower
    # d50_gpc models a shallower venue (e.g. d50_gpc=0.25 -> z_median ~ 0.046,
    # the seed600 shallow regime). Used by detection_probability everywhere the
    # generative population, D(h), beta_G and the p_det factors are built.
    d50_gpc: float = D50_GPC
    w_pdet_gpc: float = W_PDET_GPC
    # Clamp the observed photo-z z_gal at Z_MIN in the generative model (H1
    # clamp-isolation diagnostic, 2026-07-13). Default True keeps the committed
    # anchor runs bit-identical. Setting False lets z_gal go below Z_MIN (raw
    # unclamped photo-z), isolating whether the shallow-venue high bias is driven
    # by the boundary clamp on the measurement rather than the kernel itself.
    clamp_zgal: bool = True
    # --- GW measurement scatter (Q-0 audit, 2026-08-17) -------------------
    # The existing "const-sigma" convention conflates two error sub-terms:
    # (a) sigma evaluated at a SCATTERED d_L_obs instead of d_L_true, and
    # (b) the width not varying across the z-integral (the dropped 1/sigma(z)).
    # Production carries ONLY (b): its per-event sigma is the Fisher CRB frozen
    # at the injected truth and there is NO measurement scatter — d_L_obs is
    # identically d_L_true on the production path (bayesian_statistics.py:3543,
    # :3613, :4442-4443; detection.py:133-136). Setting
    # gw_measurement_scatter=False removes the (d_L, M_z) measurement draw
    # (the RNG draw is still made and DISCARDED so paired A/B cells stay on the
    # same random stream), leaving the production-faithful cell
    # (no scatter + const sigma). Default True is bit-identical to every
    # committed run.
    gw_measurement_scatter: bool = True
    # --- Mass channel ([A3] criterion (i), 2026-08-17) --------------------
    # mass_channel=True adds a genuine second (mass) observable on top of
    # catalogue_mode: galaxies carry BH masses, detection becomes
    # S_4D(d_L, M_z), the catalogue leg carries a per-candidate Gaussian mass
    # overlap and the completion leg the mass factor g / g_sel recomputed at
    # every h. Requires catalogue_mode. Default False keeps every pre-existing
    # mode and golden pin bit-identical.
    mass_channel: bool = False
    mass_slope: float = 0.0
    # S_4D horizon index alpha_M; 0.0 reduces the survival EXACTLY to the
    # mass-blind p_det (limiting case). A mass-BEARING cell must set > 0.
    mass_horizon_index: float = 0.0
    sigma_mz_frac: float = 0.10
    rho_dl_mz: float = 0.0
    sigma_m_gal_frac: float = 0.30
    mass_rate_index: float = 0.0
    n_hermite: int = 24
    n_mass_quad: int = 400
    n_z_survival: int = 1500
    # Selection cell, mirroring production's
    # selection_in_completion_numerator ('off' = pre-#118 estimator, '1d' =
    # [P2] only, '2d' = [P1] only, 'fused' = the landed production pairing).
    selection_cell: Literal["off", "1d", "2d", "fused"] = "off"
    # Events processed per vectorized block (memory/speed trade-off).
    event_chunk: int = 16

    def h_grid(self) -> npt.NDArray[np.float64]:
        """Return the H0 evaluation grid."""
        return np.arange(self.h_min, self.h_max + 0.5 * self.h_step, self.h_step)


def _completion_numerator(
    dL_obs_i: float,
    sig_dl_i: float,
    z_support: float,
    h_grid: npt.NDArray[np.float64],
    n_z_quad: int,
    tilt: float,
    pdet_in_numerator: bool = False,
    sigma_dl_frac: float = 0.05,
    sigma_dl_model_in_likelihood: bool = False,
    d50: float = D50_GPC,
    w_pdet: float = W_PDET_GPC,
) -> npt.NDArray[np.float64]:
    """Pure-completion numerator B_num(h) above the catalogue support edge.

    B_num(h) = int p_GW(A(z)/h) w_pop(z) dz over
    [max(Z_MIN, z_support, z_GW_lo), min(Z_MAX_POP, z_GW_hi)] — no kernel
    padding (there is no kernel here), capped at Z_MAX_POP (issue #30
    parallel; matches D(h)) and sharing D(h)'s exact unnormalized measure
    (no extra h-dependent normalization). The GW window is the +-5 sigma d_L
    support mapped through the h-grid edges. Returns the 1e-300 floor when
    the window is empty.

    Args:
        dL_obs_i: Observed GW luminosity distance [Gpc].
        sig_dl_i: Absolute d_L uncertainty [Gpc].
        z_support: Catalogue support ceiling (lower edge of the completion
            volume).
        h_grid: H0 evaluation grid.
        n_z_quad: Redshift quadrature points.
        tilt: Inference-side w_pop tilt gamma (N-3 probe); 0.0 is untilted.
        pdet_in_numerator: Multiply the integrand by p_det(A(z)/h) — the
            latent-detection exact-inverse factor (260711-27m probe; see
            ``PPCoverageConfig.pdet_in_numerator``). Default False is
            bit-identical to the pre-probe behaviour.
        sigma_dl_frac: Fractional d_L uncertainty σ_f; only used when
            ``sigma_dl_model_in_likelihood`` is True (to form σ_f·A(z)/h).
        sigma_dl_model_in_likelihood: Use the z-dependent model/true-distance
            GW-likelihood width σ_f·A(z)/h (with its 1/σ(z) normalization)
            instead of the constant ``sig_dl_i = σ_f·dL_obs`` (260711-hx1 probe;
            see ``PPCoverageConfig.sigma_dl_model_in_likelihood``). Default
            False is bit-identical to the pre-probe behaviour.

    Returns:
        B_num evaluated on ``h_grid`` (shape ``(nh,)``).
    """
    z_lo_b = max(
        Z_MIN,
        z_support,
        float(z_of_comoving_amplitude(np.asarray((dL_obs_i - 5 * sig_dl_i) * h_grid.min()))),
    )
    z_hi_b = min(
        Z_MAX_POP,
        float(z_of_comoving_amplitude(np.asarray((dL_obs_i + 5 * sig_dl_i) * h_grid.max()))),
    )
    if z_hi_b <= z_lo_b:
        return np.full(h_grid.size, 1e-300)
    zq_b = np.linspace(z_lo_b, z_hi_b, n_z_quad)
    wq_b = np.gradient(zq_b)
    dLg_b = comoving_amplitude_of_z(zq_b)[:, None] / h_grid[None, :]  # (nz, nh)
    # Model/true-distance width σ_f·A(z)/h (z-dependent, 1/σ(z) via _norm_pdf)
    # vs the constant observed-distance σ_f·dL_obs (260711-hx1 floor probe).
    sig_b: npt.NDArray[np.float64] | float = (
        sigma_dl_frac * dLg_b if sigma_dl_model_in_likelihood else sig_dl_i
    )
    pGW_b = _norm_pdf(dLg_b, dL_obs_i, sig_b)  # (nz, nh)
    if pdet_in_numerator:
        # Latent-detection exact inverse: detection is decided on the true z,
        # so p_det(A(z)/h) stays inside the numerator (260711-27m probe).
        pGW_b = pGW_b * detection_probability(dLg_b, d50, w_pdet)  # (nz, nh)
    wpop_b = _inference_population_weight(zq_b, tilt)  # unnormalized
    return np.asarray((wq_b * wpop_b) @ pGW_b, dtype=np.float64)  # (nh,)


def _run_realization(
    h_true: float,
    h_grid: npt.NDArray[np.float64],
    log_Dh: npt.NDArray[np.float64],
    config: PPCoverageConfig,
    rng: np.random.Generator,
    beta_G: npt.NDArray[np.float64] | None = None,
    beta_Gbar: npt.NDArray[np.float64] | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    int,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    int,
    int,
]:
    """Simulate one realization and return the accumulated log-likelihood on ``h_grid``.

    Clean single-host limit (fully complete catalogue, one candidate host per
    event): the per-event likelihood is p_i(h) = num(h) / D(h) with

        num_bare(h)   = int p_GW(A(z)/h) N(z; z_g, sigma_z) dz
        num_volume(h) = int p_GW(A(z)/h) N(z; z_g, sigma_z) w_pop(z) / Z_g dz
        D(h)          = int p_det(A(z)/h) w_pop(z) dz

    so only the host-z kernel differs between the two estimator variants.

    Catalogue-support-truncated mode (``config.z_support`` not None): true
    hosts with ``z_host >= z_support`` are treated as zero-host events and
    use the pure-completion likelihood B_num(h)/D(h) instead — the
    ``L_cat -> 0`` limit of the Gray et al. (2020, arXiv:1908.06050, Eqs.
    29+32) mixture that production commit ``8db6c6e`` (issue #29) installed
    in ``bayesian_statistics.py``; see also Gray, Messenger & Veitch (2022,
    arXiv:2111.04629, Eq. 5) and
    ``docs/derivations/G2a_completion_sky_marginal_4pi.md`` limiting case 2.
    With ``config.membership_on_observed`` the split uses the observed
    ``z_gal`` instead of the true ``z_host`` (N-2d probe).

    Mixture modes (``config.mixture_mode``, require ``z_support``):

    - ``"gray"``: in-catalogue events get the full Gray et al. (2020, Eqs.
      29+32) mixture ``(beta_G * L_cat_i + B_num) / D`` with
      ``L_cat_i = N_i / D_g_i``; ``N_i`` is the two_branch kernel numerator
      and ``D_g_i = int p_det(A(z)/h) K_i(z) dz`` the per-host selection
      denominator over the SAME normalized kernel (Eqs. A.9/A.10; production
      commit ``713fbd1`` analog). The kernel is NOT truncated at z_support
      (production-faithful leak). Zero-host events keep ``B_num/D``.
    - ``"conditioned"``: membership-conditioned inverse (N-2b probe) —
      in-catalogue ``N_i / beta_G``, zero-host ``B_num / beta_Gbar``.
    - ``"exact"``: membership-truncated exact kernel (260711-117) —
      in-catalogue events integrate the SAME volume-kernel numerator but
      truncated at the catalogue support edge (``z_hi -> min(z_hi,
      z_support)``), divided by the shared ``D(h)`` (no ``beta_G``, no
      ``D_g_i``). Under the harness generative model detection is
      conditioned once via ``1/D(h)`` with no p_det inside the numerator
      (Mandel, Farr & Gair 2019, arXiv:1809.02063) and membership
      ``G = 1[z_true < z_support]`` is observed data, which removes the
      above-edge kernel leak the two_branch/gray numerators carry.
      Zero-host events keep ``B_num/D``, so the two branches tile
      ``[0, Z_MAX_POP]`` exactly (Gray et al. 2020 support split).
    - ``"absolute"``: absolute-mass marginal (harness analog of
      ``results/lcat_h_dependence_20260725/DERIVATION_ESTIMATOR_REDESIGN.md``
      Variant 1, Eq. 2) — host events get ``[N_i(h) + B_num_i(h)] / D(h)``:
      the two_branch kernel numerator ``N_i`` summed DIRECTLY with the same
      ``B_num_i`` completion integral zero-host events use, with NO per-event
      self-normalization (no ``beta_G`` weight, no ``D_g_i`` division).
      Zero-host events keep ``B_num/D`` (the continuous ``N_i -> 0`` limit of
      the same formula). See the module docstring for the harness-fidelity
      caveat (no impostor-ball mechanism in this harness).

    Args:
        h_true: Injected truth.
        h_grid: H0 evaluation grid.
        log_Dh: Log of the shared selection denominator D(h) on ``h_grid``.
        config: Harness configuration.
        rng: Realization RNG.
        beta_G: In-catalogue selection integral on ``h_grid`` (required for
            mixture modes other than "two_branch").
        beta_Gbar: Out-of-catalogue selection integral ``D - beta_G``
            (required for "conditioned").

    Returns:
        Tuple of (accumulated log-likelihood on ``h_grid``, number of
        zero-host events, host-branch log-likelihood, completion-branch
        log-likelihood, number of host-branch events, number of
        completion-branch events). The host/completion accumulators are
        diagnostics; ``logL`` is their sum and keeps the exact pre-existing
        float ops in the exact event order (two_branch bit-identity).
    """
    # One effective sigma for the truth-scatter draw AND the kernel keeps the
    # generative model and the inference consistent (calibrated case).
    sigma_z = float(np.hypot(config.sigma_z, config.sigma_z_pv))
    z_host = _sample_detected_redshifts(
        h_true, config.n_events, rng, d50=config.d50_gpc, w_pdet=config.w_pdet_gpc
    )
    dL_host = comoving_amplitude_of_z(z_host) / h_true
    _dl_draw = rng.normal(0.0, config.sigma_dl_frac * dL_host)
    # Q-0 (2026-08-17): production has NO GW measurement scatter — d_L_obs is
    # identically d_L_true there. The draw is made and discarded so a paired
    # scatter/no-scatter A/B stays on the same random stream.
    dL_obs = np.clip(dL_host + (_dl_draw if config.gw_measurement_scatter else 0.0), 1e-3, None)
    sig_dl = config.sigma_dl_frac * dL_obs
    z_gal_raw = z_host + rng.normal(0.0, sigma_z, config.n_events)
    # H1 clamp-isolation diagnostic: default clamps z_gal at Z_MIN (committed
    # behaviour); when disabled, z_gal keeps its raw (possibly < Z_MIN) value so
    # the kernel/quadrature sees the unclamped measurement.
    z_gal = np.clip(z_gal_raw, Z_MIN, None) if config.clamp_zgal else z_gal_raw

    if config.mixture_mode in ("gray", "conditioned"):
        if beta_G is None or beta_Gbar is None:
            raise ValueError(
                f"mixture_mode={config.mixture_mode!r} requires precomputed beta_G/beta_Gbar"
            )
        beta_G_h: npt.NDArray[np.float64] = beta_G
        log_beta_G: npt.NDArray[np.float64] = np.log(np.clip(beta_G, 1e-300, None))
        log_beta_Gbar: npt.NDArray[np.float64] = np.log(np.clip(beta_Gbar, 1e-300, None))
    else:  # unused sentinels; two_branch and exact never touch them
        beta_G_h = np.zeros(0)
        log_beta_G = np.zeros(0)
        log_beta_Gbar = np.zeros(0)

    logL = np.zeros(h_grid.size)
    logL_host = np.zeros(h_grid.size)
    logL_completion = np.zeros(h_grid.size)
    n_zero_host = 0
    n_host = 0
    n_comp = 0
    for i in range(config.n_events):
        if config.z_support is not None:
            zs: float = config.z_support
            member_z = float(z_gal[i]) if config.membership_on_observed else float(z_host[i])
            if member_z >= zs:
                # Zero-host / out-of-catalogue event.
                n_zero_host += 1
                num_b = _completion_numerator(
                    float(dL_obs[i]),
                    float(sig_dl[i]),
                    zs,
                    h_grid,
                    config.n_z_quad,
                    config.inference_wpop_tilt,
                    config.pdet_in_numerator,
                    config.sigma_dl_frac,
                    config.sigma_dl_model_in_likelihood,
                    config.d50_gpc,
                    config.w_pdet_gpc,
                )
                if config.mixture_mode == "conditioned":
                    # Membership-conditioned inverse: B_num / beta_Gbar.
                    term_b = np.log(np.clip(num_b, 1e-300, None)) - log_beta_Gbar
                else:
                    # two_branch, gray AND absolute share the pure-completion
                    # B_num/D branch (Gray et al. 2020 Eqs. 29+32, L_cat -> 0
                    # / N_i -> 0 limit).
                    term_b = np.log(np.clip(num_b, 1e-300, None)) - log_Dh
                logL += term_b
                logL_completion += term_b
                n_comp += 1
                continue
        z_lo = max(
            Z_MIN,
            float(z_of_comoving_amplitude(np.asarray((dL_obs[i] - 5 * sig_dl[i]) * h_grid.min())))
            - 4 * sigma_z,
        )
        z_hi = min(
            float(_Z_GRID[-1]),
            float(z_of_comoving_amplitude(np.asarray((dL_obs[i] + 5 * sig_dl[i]) * h_grid.max())))
            + 4 * sigma_z,
        )
        if config.mixture_mode == "exact":
            # Membership-truncated exact kernel (Mandel-Farr-Gair 2019,
            # arXiv:1809.02063: detection conditioned once via 1/D(h), no
            # p_det in the numerator; catalogue membership
            # G = 1[z_true < z_support] is part of the observed data). The
            # exact host-event numerator integrates the volume kernel only
            # over the in-catalogue support [z_lo, min(z_hi, zs)], removing
            # the above-edge kernel leak that the two_branch / gray
            # numerators carry. Zero-host events keep B_num/D, so the two
            # branches tile [0, Z_MAX_POP] exactly (Gray et al. 2020,
            # arXiv:1908.06050, Eqs. 29+32 support split). z_support is
            # guaranteed not None here (run_coverage raises otherwise).
            assert config.z_support is not None
            z_hi = min(z_hi, float(config.z_support))
            if z_hi <= z_lo:
                # Empty truncated window -> the 1e-300 completion-style floor.
                num_floor = np.full(h_grid.size, 1e-300, dtype=np.float64)
                term = np.log(np.clip(num_floor, 1e-300, None)) - log_Dh
                logL += term
                logL_host += term
                n_host += 1
                continue
        zq = np.linspace(z_lo, z_hi, config.n_z_quad)
        wq = np.gradient(zq)
        dLg = comoving_amplitude_of_z(zq)[:, None] / h_grid[None, :]  # (nz, nh)
        # Model/true-distance width σ_f·A(z)/h (z-dependent, 1/σ(z) via _norm_pdf)
        # vs the constant observed-distance σ_f·dL_obs (260711-hx1 floor probe).
        sig_gw: npt.NDArray[np.float64] | float = (
            config.sigma_dl_frac * dLg if config.sigma_dl_model_in_likelihood else float(sig_dl[i])
        )
        pGW = _norm_pdf(dLg, float(dL_obs[i]), sig_gw)  # (nz, nh)
        if config.pdet_in_numerator:
            # Latent-detection exact inverse (260711-27m probe): detection is
            # decided on the true z, so p_det(A(z)/h) stays inside the host
            # numerator too (numerator-only — gray-mode D_g_i is unchanged).
            pGW = pGW * detection_probability(dLg, config.d50_gpc, config.w_pdet_gpc)  # (nz, nh)
        kernel_z = _norm_pdf(zq, float(z_gal[i]), sigma_z)  # (nz,)
        if config.kernel == "volume":
            kernel_z = kernel_z * _inference_population_weight(zq, config.inference_wpop_tilt)
            kernel_z = kernel_z / max(float(np.trapezoid(kernel_z, zq)), 1e-300)
        num = (wq * kernel_z) @ pGW  # (nh,)
        if config.mixture_mode == "gray" and config.z_support is not None:
            # Full Gray (2020) mixture: (beta_G * L_cat_i + B_num) / D with
            # L_cat_i = N_i / D_g_i over the SAME normalized kernel K_i
            # (Eqs. A.9/A.10; production commit 713fbd1 analog). The kernel
            # is NOT truncated at z_support (production-faithful leak).
            D_g_i = (wq * kernel_z) @ detection_probability(
                dLg, config.d50_gpc, config.w_pdet_gpc
            )  # (nh,)
            L_cat_i = num / np.clip(D_g_i, 1e-300, None)
            B_num_i = _completion_numerator(
                float(dL_obs[i]),
                float(sig_dl[i]),
                float(config.z_support),
                h_grid,
                config.n_z_quad,
                config.inference_wpop_tilt,
                config.pdet_in_numerator,
                config.sigma_dl_frac,
                config.sigma_dl_model_in_likelihood,
                config.d50_gpc,
                config.w_pdet_gpc,
            )
            mixture = beta_G_h * L_cat_i + B_num_i  # linear space, per event
            term = np.log(np.clip(mixture, 1e-300, None)) - log_Dh
        elif config.mixture_mode == "absolute" and config.z_support is not None:
            # Absolute-mass marginal (DERIVATION_ESTIMATOR_REDESIGN.md
            # Variant 1, Eq. 2): A_i(h) = N_i(h) with NO self-normalization
            # (no beta_G weight, no D_g_i division — n_bar_w collapses to 1
            # in this harness's continuum-population idealization, see
            # module docstring). Summed directly with the SAME B_num_i
            # completion integral zero-host events use.
            B_num_i_abs = _completion_numerator(
                float(dL_obs[i]),
                float(sig_dl[i]),
                float(config.z_support),
                h_grid,
                config.n_z_quad,
                config.inference_wpop_tilt,
                config.pdet_in_numerator,
                config.sigma_dl_frac,
                config.sigma_dl_model_in_likelihood,
                config.d50_gpc,
                config.w_pdet_gpc,
            )
            mixture_abs = num + B_num_i_abs  # linear space, absolute mass
            term = np.log(np.clip(mixture_abs, 1e-300, None)) - log_Dh
        elif config.mixture_mode == "conditioned" and config.z_support is not None:
            # Membership-conditioned inverse: N_i / beta_G (no B_num, no
            # D_g_i ratio).
            term = np.log(np.clip(num, 1e-300, None)) - log_beta_G
        else:
            term = np.log(np.clip(num, 1e-300, None)) - log_Dh
        logL += term
        logL_host += term
        n_host += 1
    return logL, n_zero_host, logL_host, logL_completion, n_host, n_comp


# ----------------------------------------------------------------------------
# Catalogue / impostor-ball universe (2026-07-26).
#
# Derivation: results/pp_impostor_harness_20260726/DERIVATION_HARNESS_ANALOG.md.
# Nothing below ever reads an injected truth: the estimator sees only the
# catalogue table (observed redshifts + directions), the completeness edge
# z_support, the event's (d_L_obs, cap centre) and the hypothesis h.
# ----------------------------------------------------------------------------

CATALOGUE_MIXTURE_MODES: tuple[str, ...] = ("lcat", "absolute", "generator_marginal")


@dataclass
class SyntheticCatalogue:
    """Frozen discrete galaxy catalogue plus its precomputed selection scalars.

    Attributes:
        z_true: True redshifts of ALL galaxies (catalogued or not), shape (N,).
        direction: Unit direction vectors of all galaxies, shape (N, 3).
        catalogued: Boolean mask ``z_true < z_support`` (the harness's
            sky-averaged completeness ``fbar(z) = 1[z < z_support]``).
        cat_index: Indices of catalogued galaxies into the full arrays.
        z_obs: Observed (photo-z scattered) redshifts of the CATALOGUED
            galaxies, shape (N_cat,) — the only redshift information the
            estimator ever sees.
        inv_norm: Per-catalogued-galaxy ``1/Z_g`` with
            ``Z_g = int n_gal(z) N(z_obs,g; z, sigma_z) dz`` over the
            population support, i.e. the normalizer of the galaxy's true-z
            posterior ``p(z|z_obs,g) propto n_gal(z) N(z_obs,g; z, sigma_z)``.
        tree: KD-tree over the catalogued galaxies' 3D unit vectors (the
            harness analog of production's BallTree sky index).
        chord_radius: Euclidean chord radius corresponding to the cap
            half-angle of ``sky_frac``.
        w_cat: ``W_cat = Sum_g int dmu_g`` — total draw-eligible catalogue rate
            weight (h-independent scalar).
        v_f: ``V_f = int fbar(z) w_pop(z) dz`` — the completeness-weighted
            population rate-weight volume (h-independent in this harness).
        n_hat_w: ``W_cat / V_f`` — the generator-consistent draw-side
            rate-weight density.
        sigma_glob: ``Sigma_glob(h) = Sum_g int dmu_g p_det(A(z)/h)`` on the
            h grid, shape (nh,).
        host_draw_p: Normalized host-draw probability per galaxy at the
            injected truth (rate weight times detection probability); set by
            :func:`_catalogue_host_draw_probabilities`.
        mass_true: Source-frame BH masses of ALL galaxies [M_sun], shape
            ``(N,)``, or None when the mass channel is off.
        mass_obs: Observed (noisy) source-frame BH masses of the CATALOGUED
            galaxies [M_sun], shape ``(N_cat,)``, or None when the mass
            channel is off — the only mass information the estimator sees.
    """

    z_true: npt.NDArray[np.float64]
    direction: npt.NDArray[np.float64]
    catalogued: npt.NDArray[np.bool_]
    cat_index: npt.NDArray[np.int64]
    z_obs: npt.NDArray[np.float64]
    inv_norm: npt.NDArray[np.float64]
    tree: Any
    chord_radius: float
    w_cat: float
    v_f: float
    n_hat_w: float
    sigma_glob: npt.NDArray[np.float64]
    mass_true: npt.NDArray[np.float64] | None = None
    mass_obs: npt.NDArray[np.float64] | None = None


def _sample_galaxy_redshifts(
    n: int, rng: np.random.Generator, tilt: float, ngrid: int = 4000
) -> npt.NDArray[np.float64]:
    """Draw galaxy true redshifts from the comoving number density n_gal(z).

    Args:
        n: Number of galaxies.
        rng: Random generator.
        tilt: N-3 prior tilt (applied to the generative galaxy density so that
            the tilted inference weight remains the correct model; 0.0 default).
        ngrid: Inverse-CDF grid resolution.

    Returns:
        Redshifts on ``[Z_MIN, Z_MAX_POP]``, shape ``(n,)``.
    """
    zg = np.linspace(Z_MIN, Z_MAX_POP, ngrid)
    pdf = np.clip(_inference_galaxy_number_weight(zg, tilt), 0.0, None)
    cdf = np.concatenate([np.array([0.0]), np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(zg))])
    cdf /= cdf[-1]
    return np.asarray(np.interp(rng.random(n), cdf, zg), dtype=np.float64)


def _random_unit_vectors(n: int, rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """Draw ``n`` directions uniform on the unit sphere, shape ``(n, 3)``."""
    v = rng.normal(size=(n, 3))
    return np.asarray(v / np.linalg.norm(v, axis=1, keepdims=True), dtype=np.float64)


def _perturb_within_cap(
    axis: npt.NDArray[np.float64], cos_theta_c: float, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Draw directions uniform inside the cap of half-angle ``theta_c`` about ``axis``.

    Used to place the GW localization cap CENTRE relative to the true host
    direction: if the centre is uniform in the cap about the host, then the
    host is uniform in the cap about the centre (the cap is a symmetric
    relation), which makes the flat in-cap sky likelihood
    ``p(sky data | direction) propto 1[direction in cap]`` exact rather than an
    approximation. No truth leaks into the estimator: only the cap centre is
    passed on.

    Args:
        axis: Unit vectors to perturb about, shape ``(n, 3)``.
        cos_theta_c: Cosine of the cap half-angle.
        rng: Random generator.

    Returns:
        Unit vectors, shape ``(n, 3)``.
    """
    n = axis.shape[0]
    cos_psi: npt.NDArray[np.float64] = np.asarray(
        cos_theta_c + (1.0 - cos_theta_c) * rng.random(n), dtype=np.float64
    )
    sin_psi: npt.NDArray[np.float64] = np.sqrt(np.clip(1.0 - cos_psi**2, 0.0, None))
    phi = 2.0 * np.pi * rng.random(n)
    # Build an orthonormal basis (axis, e1, e2) per row.
    helper = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    near_pole = np.abs(axis[:, 2]) > 0.9
    helper[near_pole] = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, helper)
    e1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(axis, e1)
    out = (
        cos_psi[:, None] * axis
        + (sin_psi * np.cos(phi))[:, None] * e1
        + (sin_psi * np.sin(phi))[:, None] * e2
    )
    return np.asarray(out / np.linalg.norm(out, axis=1, keepdims=True), dtype=np.float64)


def _posterior_normalizers(
    z_obs: npt.NDArray[np.float64], sigma_z: float, tilt: float, ngrid: int = 3000
) -> npt.NDArray[np.float64]:
    """Compute ``Z_g = int_{Z_MIN}^{Z_MAX_POP} n_gal(z) N(z_obs,g; z, sigma_z) dz``.

    Evaluated on a ``z_obs`` lookup grid and interpolated (``Z(z_obs)`` is a
    smooth function of the observed redshift).

    Args:
        z_obs: Observed catalogue redshifts, shape ``(N_cat,)``.
        sigma_z: Photo-z scatter.
        tilt: N-3 inference prior tilt.
        ngrid: Quadrature / lookup resolution.

    Returns:
        ``Z_g`` per galaxy, shape ``(N_cat,)``.
    """
    zq = np.linspace(Z_MIN, Z_MAX_POP, ngrid)
    wq = np.gradient(zq)
    ngal = _inference_galaxy_number_weight(zq, tilt)
    lo = float(min(Z_MIN, z_obs.min())) - 6.0 * sigma_z
    hi = float(max(Z_MAX_POP, z_obs.max())) + 6.0 * sigma_z
    obs_grid = np.linspace(lo, hi, ngrid)
    kern = _norm_pdf(zq[None, :], obs_grid[:, None], sigma_z)  # (ngrid_obs, ngrid_z)
    table = kern @ (wq * ngal)  # (ngrid_obs,)
    return np.asarray(np.interp(z_obs, obs_grid, table), dtype=np.float64)


def _smeared_catalogue_density(
    z_obs: npt.NDArray[np.float64],
    inv_norm: npt.NDArray[np.float64],
    sigma_z: float,
    z_eval: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Evaluate ``Khat(z) = Sum_g N(z; z_obs,g, sigma_z) / Z_g`` on ``z_eval``.

    Computed by linearly (cloud-in-cell) depositing ``z_obs`` with weights
    ``1/Z_g`` on a uniform grid of spacing ``sigma_z/16`` and convolving with
    the Gaussian kernel: O(dz^2) pointwise accuracy at O(N + M K) cost instead
    of the O(N * M) direct sum. CIC rather than nearest-node deposition is
    required — nearest-node is only first order and leaves a ~2e-3 pointwise
    error at this spacing (measured).

    Args:
        z_obs: Observed catalogue redshifts, shape ``(N_cat,)``.
        inv_norm: Per-galaxy ``1/Z_g``, shape ``(N_cat,)``.
        sigma_z: Photo-z scatter.
        z_eval: Redshifts at which to evaluate, shape ``(M,)``.

    Returns:
        ``Khat`` on ``z_eval``, shape ``(M,)``.
    """
    lo = float(min(z_obs.min(), z_eval.min())) - 6.0 * sigma_z
    hi = float(max(z_obs.max(), z_eval.max())) + 6.0 * sigma_z
    dz = sigma_z / 16.0
    nbin = int(np.ceil((hi - lo) / dz)) + 1
    grid = lo + dz * np.arange(nbin)
    pos = (z_obs - lo) / dz
    i0 = np.clip(np.floor(pos).astype(np.int64), 0, nbin - 2)
    frac = pos - i0
    binned = np.zeros(nbin, dtype=np.float64)
    np.add.at(binned, i0, inv_norm * (1.0 - frac))
    np.add.at(binned, i0 + 1, inv_norm * frac)
    half = int(np.ceil(6.0 * sigma_z / dz))
    offsets = np.asarray(dz * np.arange(-half, half + 1), dtype=np.float64)
    kern = _norm_pdf(offsets, 0.0, sigma_z)
    dens = np.convolve(binned, kern, mode="same")
    return np.asarray(np.interp(z_eval, grid, dens), dtype=np.float64)


def _build_catalogue(
    config: PPCoverageConfig,
    h_grid: npt.NDArray[np.float64],
    rng: np.random.Generator,
    survival_table: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
) -> SyntheticCatalogue:
    """Build one frozen synthetic galaxy catalogue and its selection precomputes.

    Args:
        config: Harness configuration (``catalogue_mode`` semantics).
        h_grid: H0 evaluation grid.
        rng: Random generator for the catalogue draw.
        survival_table: ``(z_grid, S_bar_phi)`` from
            :func:`phi_marginal_survival_table`, required when
            ``config.mass_channel`` is set: the global selection precompute
            ``Sigma_glob(h)`` then contracts ``S_bar_phi`` in place of the
            mass-blind ``p_det`` (harness analog of production's
            phi-convention legs).

    Returns:
        The catalogue with ``W_cat``, ``V_f``, ``n_hat_w`` and
        ``Sigma_glob(h)`` precomputed.

    Raises:
        ValueError: If ``config.z_support`` is None, or the catalogue is empty.
    """
    from scipy.spatial import cKDTree  # local import: only catalogue mode needs it

    if config.z_support is None:
        raise ValueError("catalogue_mode requires z_support (the completeness edge fbar).")
    sigma_z = float(np.hypot(config.sigma_z, config.sigma_z_pv))
    zs = float(config.z_support)

    z_true = _sample_galaxy_redshifts(config.n_galaxies, rng, config.inference_wpop_tilt)
    direction = _random_unit_vectors(config.n_galaxies, rng)
    catalogued = np.asarray(z_true < zs, dtype=np.bool_)
    cat_index = np.asarray(np.flatnonzero(catalogued), dtype=np.int64)
    if cat_index.size == 0:
        raise ValueError(
            f"catalogue_mode: no galaxy has z_true < z_support={zs}; raise n_galaxies or z_support."
        )
    z_obs_raw = z_true[cat_index] + rng.normal(0.0, sigma_z, cat_index.size)
    z_obs = np.clip(z_obs_raw, Z_MIN, None) if config.clamp_zgal else z_obs_raw

    inv_norm = 1.0 / np.clip(
        _posterior_normalizers(z_obs, sigma_z, config.inference_wpop_tilt), 1e-300, None
    )

    # Mass channel: source-frame BH masses for every galaxy, plus the noisy
    # mass observable the estimator sees for the CATALOGUED ones. Drawn after
    # the redshift/direction/photo-z draws so the mass-free stream is unchanged.
    mass_true: npt.NDArray[np.float64] | None = None
    mass_obs: npt.NDArray[np.float64] | None = None
    inv_norm_eff = inv_norm
    if config.mass_channel:
        mass_true = _sample_galaxy_masses(config.n_galaxies, rng, config.mass_slope)
        mass_obs = np.clip(
            mass_true[cat_index] * (1.0 + rng.normal(0.0, config.sigma_m_gal_frac, cat_index.size)),
            1e-3,
            None,
        )
        # Per-galaxy EMRI rate weight mass factor (production: R_eff(M_g)).
        inv_norm_eff = inv_norm * (mass_obs / M_REF_MSUN) ** config.mass_rate_index

    cos_theta_c = 1.0 - 2.0 * config.sky_frac
    chord_radius = float(2.0 * np.sin(0.5 * np.arccos(np.clip(cos_theta_c, -1.0, 1.0))))
    tree = cKDTree(direction[cat_index])

    # Global (all-sky) catalogue selection precomputes, on D(h)'s own node
    # convention so that at z_support >= Z_MAX_POP the limiting identities hold.
    zint = np.linspace(Z_MIN, Z_MAX_POP, 3000)
    wint = np.gradient(zint)
    khat = _smeared_catalogue_density(z_obs, inv_norm_eff, sigma_z, zint)
    rho_cat = _inference_population_weight(zint, config.inference_wpop_tilt) * khat
    w_cat = float(np.sum(wint * rho_cat))
    if config.mass_channel:
        if survival_table is None:
            raise ValueError("mass_channel catalogue build requires the S_bar_phi table")
        pdet_grid = _interp_survival_table(zint, survival_table[0], survival_table[1])  # (nz, nh)
    else:
        pdet_grid = detection_probability(
            comoving_amplitude_of_z(zint)[:, None] / h_grid[None, :],
            config.d50_gpc,
            config.w_pdet_gpc,
        )  # (nz, nh)
    sigma_glob = np.asarray((wint * rho_cat) @ pdet_grid, dtype=np.float64)  # (nh,)

    zvf = np.linspace(Z_MIN, min(zs, Z_MAX_POP), 3000)
    v_f = float(np.trapezoid(_inference_population_weight(zvf, config.inference_wpop_tilt), zvf))

    return SyntheticCatalogue(
        z_true=z_true,
        direction=direction,
        catalogued=catalogued,
        cat_index=cat_index,
        z_obs=np.asarray(z_obs, dtype=np.float64),
        inv_norm=np.asarray(inv_norm, dtype=np.float64),
        tree=tree,
        chord_radius=chord_radius,
        w_cat=w_cat,
        v_f=v_f,
        n_hat_w=w_cat / max(v_f, 1e-300),
        sigma_glob=sigma_glob,
        mass_true=mass_true,
        mass_obs=mass_obs,
    )


def _run_realization_catalogue(
    h_true: float,
    h_grid: npt.NDArray[np.float64],
    log_Dh: npt.NDArray[np.float64],
    config: PPCoverageConfig,
    rng: np.random.Generator,
    catalogue: SyntheticCatalogue,
    beta_G: npt.NDArray[np.float64],
    beta_Gbar: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], dict[str, float]]:
    """Simulate one catalogue-mode realization; return log-likelihood + diagnostics.

    Per-event likelihood (see the module docstring correspondence table and
    ``results/pp_impostor_harness_20260726/DERIVATION_HARNESS_ANALOG.md``):

        p_i(h) = [ T_i(h) + B_num,i(h) ] / Den(h)

    with the ball measure ``dmu_g(z) = [n_gal(z) N(z_obs,g; z, sigma_z)/Z_g]
    w(z) dz`` and

        Sum_ball w N (h) = Sum_{g in ball} int dmu_g(z) p_GW(dL_obs,i | A(z)/h)
        Sum_ball w D_g(h) = Sum_{g in ball} int dmu_g(z) p_det(A(z)/h)

        lcat:               T_i = beta_G(h) * (Sum_ball w N)/(Sum_ball w D_g)
                            Den  = D(h) = beta_G + beta_Gbar
        absolute:           T_i = (Sum_ball w N) / (n_bar_w(h) * sky_frac)
                            n_bar_w(h) = Sigma_glob(h)/beta_G(h),  Den = D(h)
        generator_marginal: T_i = (Sum_ball w N) / (n_hat_w * sky_frac)
                            n_hat_w = W_cat/V_f,
                            Den = D_gen(h) = Sigma_glob(h)/n_hat_w + beta_Gbar(h)

    An empty ball gives ``T_i = 0`` in every mode, so ``B_num/Den`` is the
    continuous limit rather than a separate branch.

    Args:
        h_true: Injected truth (used ONLY by the generative model).
        h_grid: H0 evaluation grid.
        log_Dh: ``log D(h)`` on the grid.
        config: Harness configuration.
        rng: Realization RNG.
        catalogue: The (frozen or per-realization) synthetic catalogue.
        beta_G: In-catalogue selection integral on ``h_grid``.
        beta_Gbar: Out-of-catalogue selection integral ``D - beta_G``.

    Returns:
        Tuple of the accumulated log-likelihood on ``h_grid`` and a dict of
        realization diagnostics (``completion_fraction`` = fraction of events
        whose true host is NOT in the catalogue, ``empty_ball_fraction``,
        ``mean_ball_size``, ``host_in_ball_fraction``, ``impostor_fraction``).
    """
    sigma_z = float(np.hypot(config.sigma_z, config.sigma_z_pv))
    zs = float(config.z_support) if config.z_support is not None else Z_MAX_POP
    mode = config.mixture_mode

    # --- generative model -------------------------------------------------
    w_host = host_rate_weight_of_z(catalogue.z_true)
    p_draw = w_host * detection_probability(
        comoving_amplitude_of_z(catalogue.z_true) / h_true, config.d50_gpc, config.w_pdet_gpc
    )
    p_draw = np.clip(p_draw, 0.0, None)
    p_draw = p_draw / p_draw.sum()
    host_idx = rng.choice(catalogue.z_true.size, size=config.n_events, p=p_draw)
    z_host = catalogue.z_true[host_idx]
    dL_host = comoving_amplitude_of_z(z_host) / h_true
    # Q-0 (2026-08-17): see _run_realization — the draw is discarded, not
    # skipped, when gw_measurement_scatter is off.
    _dl_draw = rng.normal(0.0, config.sigma_dl_frac * dL_host)
    dL_obs = np.clip(dL_host + (_dl_draw if config.gw_measurement_scatter else 0.0), 1e-3, None)
    sig_dl = config.sigma_dl_frac * dL_obs
    cos_theta_c = 1.0 - 2.0 * config.sky_frac
    cap_centre = _perturb_within_cap(catalogue.direction[host_idx], cos_theta_c, rng)
    balls: list[list[int]] = catalogue.tree.query_ball_point(cap_centre, catalogue.chord_radius)

    # Membership of the true host in the CATALOGUE (not in the ball).
    host_catalogued = catalogue.catalogued[host_idx]
    # Position of the host inside the catalogued arrays, -1 when not catalogued.
    cat_pos = np.full(catalogue.z_true.size, -1, dtype=np.int64)
    cat_pos[catalogue.cat_index] = np.arange(catalogue.cat_index.size, dtype=np.int64)
    host_cat_pos = cat_pos[host_idx]

    # Option-A calibration n_bar_w(h) = Sigma_glob(h)/beta_G(h) ("absolute");
    # generator-consistent n_hat_w = W_cat/V_f ("generator_marginal").
    n_bar_w: npt.NDArray[np.float64] = catalogue.sigma_glob / np.clip(beta_G, 1e-300, None)
    if mode == "generator_marginal":
        d_gen = catalogue.sigma_glob / catalogue.n_hat_w + beta_Gbar
        log_den = np.log(np.clip(d_gen, 1e-300, None))
        scale = catalogue.n_hat_w * config.sky_frac  # in-catalogue term divisor
    else:  # "absolute" and "lcat" share D(h) = beta_G + beta_Gbar
        log_den = log_Dh
        scale = float("nan")  # unused

    logL = np.zeros(h_grid.size)
    n_empty = 0
    ball_sizes: list[int] = []
    n_host_in_ball = 0
    n_impostor = 0
    n_ball_total = 0
    for i in range(config.n_events):
        ball = np.asarray(balls[i], dtype=np.int64)
        ball_sizes.append(int(ball.size))
        n_ball_total += int(ball.size)
        in_ball = bool(host_cat_pos[i] >= 0 and np.any(ball == host_cat_pos[i]))
        n_host_in_ball += int(in_ball)
        n_impostor += int(ball.size) - int(in_ball)

        z_lo = max(
            Z_MIN,
            float(z_of_comoving_amplitude(np.asarray((dL_obs[i] - 5 * sig_dl[i]) * h_grid.min())))
            - 4 * sigma_z,
        )
        z_hi = min(
            Z_MAX_POP,
            float(z_of_comoving_amplitude(np.asarray((dL_obs[i] + 5 * sig_dl[i]) * h_grid.max())))
            + 4 * sigma_z,
        )
        B_num_i = _completion_numerator(
            float(dL_obs[i]),
            float(sig_dl[i]),
            zs,
            h_grid,
            config.n_z_quad,
            config.inference_wpop_tilt,
            config.pdet_in_numerator,
            config.sigma_dl_frac,
            config.sigma_dl_model_in_likelihood,
            config.d50_gpc,
            config.w_pdet_gpc,
        )
        if ball.size == 0 or z_hi <= z_lo:
            n_empty += int(ball.size == 0)
            logL += np.log(np.clip(B_num_i, 1e-300, None)) - log_den
            continue

        zq = np.linspace(z_lo, z_hi, config.n_z_quad)
        wq = np.gradient(zq)
        dLg = comoving_amplitude_of_z(zq)[:, None] / h_grid[None, :]  # (nz, nh)
        sig_gw: npt.NDArray[np.float64] | float = (
            config.sigma_dl_frac * dLg if config.sigma_dl_model_in_likelihood else float(sig_dl[i])
        )
        pGW = _norm_pdf(dLg, float(dL_obs[i]), sig_gw)  # (nz, nh)
        if config.pdet_in_numerator:
            pGW = pGW * detection_probability(dLg, config.d50_gpc, config.w_pdet_gpc)
        # Ball rate-weight measure density: n_gal(z) * Sum_g N(z; z_obs,g,
        # sigma_z)/Z_g * w(z) == w_pop(z) * Khat_ball(z).
        khat_ball = np.sum(
            _norm_pdf(zq[None, :], catalogue.z_obs[ball][:, None], sigma_z)
            * catalogue.inv_norm[ball][:, None],
            axis=0,
        )  # (nz,)
        rho_ball = _inference_population_weight(zq, config.inference_wpop_tilt) * khat_ball
        sum_wN = np.asarray((wq * rho_ball) @ pGW, dtype=np.float64)  # (nh,)

        if mode == "lcat":
            pdet_q = detection_probability(dLg, config.d50_gpc, config.w_pdet_gpc)  # (nz, nh)
            sum_wD = np.asarray((wq * rho_ball) @ pdet_q, dtype=np.float64)  # (nh,)
            term_cat = beta_G * (sum_wN / np.clip(sum_wD, 1e-300, None))
        elif mode == "absolute":
            term_cat = sum_wN / np.clip(n_bar_w * config.sky_frac, 1e-300, None)
        else:  # generator_marginal
            term_cat = sum_wN / max(scale, 1e-300)
        logL += np.log(np.clip(term_cat + B_num_i, 1e-300, None)) - log_den

    n = float(config.n_events)
    diagnostics = {
        "completion_fraction": float(np.sum(~host_catalogued) / n),
        "empty_ball_fraction": n_empty / n,
        "mean_ball_size": float(np.mean(ball_sizes)),
        "host_in_ball_fraction": n_host_in_ball / n,
        "impostor_fraction": (n_impostor / n_ball_total) if n_ball_total > 0 else 0.0,
    }
    return logL, diagnostics


# ----------------------------------------------------------------------------
# Mass channel ([A3] criterion (i), 2026-08-17; ledger row #120 item 2 / D-2).
#
# A genuine SECOND (mass) observable on top of the catalogue/impostor-ball
# universe, so the production estimator's 2D leg — and in particular the
# completion-leg mass factor g / g_sel that the selection fusion (commit
# ``2b10b8b8``, ledger rows #117-#118) touches — can be calibration-tested.
# Nothing here imports production code: every object below is a self-contained
# re-derivation of the production FORM, cross-referenced function by function.
# ----------------------------------------------------------------------------

# Source-frame BH mass support and reference scale [M_sun]. Harness analogs of
# production's ``M_SOURCE_FRAME_MIN`` / ``M_SOURCE_FRAME_MAX``; the numbers are
# the harness's own (a single power law, not production's piecewise Babak+2017
# ``phi``), chosen only to span the same decades.
M_SOURCE_MIN: float = 1.0e4
M_SOURCE_MAX: float = 1.0e7
M_REF_MSUN: float = 1.0e6

SELECTION_CELLS: tuple[str, ...] = ("off", "1d", "2d", "fused")


def dark_mass_density_per_mass(
    M: npt.NDArray[np.float64], slope: float = 0.0
) -> npt.NDArray[np.float64]:
    r"""Normalized source-frame BH mass density ``phi(M)`` [1/M_sun].

    Harness analog of production's
    ``bayesian_statistics.dark_mass_density_per_mass`` (the Babak et al. 2017,
    arXiv:1703.09722 ``phi``): a single power law
    ``phi(M) propto M^{-(1 + slope)}`` on ``[M_SOURCE_MIN, M_SOURCE_MAX]``,
    zero outside, normalized to unit integral in ``dM``. ``slope = 0`` is flat
    in ``ln M``. The harness deliberately keeps ONE branch (no kink): the
    production kink only motivates the Route-1 breakpoint-straddle escalation,
    which is a quadrature detail, not a calibration mechanism.

    Args:
        M: Source-frame masses [M_sun].
        slope: Power-law slope offset (0.0 = flat in ``ln M``).

    Returns:
        ``phi(M)`` with the same shape as ``M``; zero off the support.

    References:
        Babak et al. (2017), arXiv:1703.09722 — the production ``phi``.
    """
    m = np.asarray(M, dtype=np.float64)
    if slope == 0.0:
        norm = math.log(M_SOURCE_MAX / M_SOURCE_MIN)
    else:
        norm = (M_SOURCE_MIN**-slope - M_SOURCE_MAX**-slope) / slope
    safe = np.where(m > 0.0, m, 1.0)
    dens = safe ** (-(1.0 + slope)) / norm
    inside = (m >= M_SOURCE_MIN) & (m <= M_SOURCE_MAX)
    return np.asarray(np.where(inside, dens, 0.0), dtype=np.float64)


def survival_with_mass(
    d_L: npt.NDArray[np.float64],
    M_z: npt.NDArray[np.float64],
    d50: float = D50_GPC,
    w_pdet: float = W_PDET_GPC,
    mass_horizon_index: float = 0.0,
) -> npt.NDArray[np.float64]:
    r"""Mass-dependent detection survival ``S_4D(d_L, M_z)`` in ``[0, 1]``.

    Harness analog of production's with-BH-mass survival object
    (``SimulationDetectionProbability.detection_probability_with_bh_mass_interpolated``,
    the ``S_4D`` that :func:`completion_mass_factor_g_sel` and
    ``precompute_phi_marginal_survival`` query). The harness makes the mass
    dependence a horizon rescaling — the physical content of an SNR-limited
    survey, where a heavier detector-frame BH is louder:

    .. math::

        S_\mathrm{4D}(d_L, M_z) = \tfrac12
            \mathrm{erfc}\!\left(\frac{d_L - d_{50}(M_z)}{\sqrt2\, w}\right),
        \qquad d_{50}(M_z) = d_{50}\,(M_z / M_\mathrm{ref})^{\alpha_M}

    with ``alpha_M = mass_horizon_index``. ``alpha_M = 0`` reduces this
    EXACTLY to the mass-blind :func:`detection_probability` (limiting case,
    pinned by a test), so a mass-bearing cell is obtained by setting
    ``alpha_M > 0``. Unlike production the harness survival is ANALYTIC, so
    ``g_sel``'s per-Hermite-node ``S`` queries need no interpolation and carry
    no grid error — a deliberate simplification (production's own quadrature
    guard ``_G_SEL_S_VAR_TOL`` guards interpolation error, not physics).

    Args:
        d_L: Luminosity distances [Gpc].
        M_z: DETECTOR-frame BH masses [M_sun] (``M (1+z)``), exactly the pair
            production queries.
        d50: Mass-reference 50% detection distance [Gpc].
        w_pdet: Roll-off width [Gpc].
        mass_horizon_index: Horizon power-law index ``alpha_M``.

    Returns:
        Survival probability, broadcast shape of ``d_L`` and ``M_z``.
    """
    if mass_horizon_index == 0.0:
        return detection_probability(np.broadcast_arrays(d_L, M_z)[0], d50, w_pdet)
    mz = np.clip(np.asarray(M_z, dtype=np.float64), 1e-30, None)
    d50_eff = d50 * (mz / M_REF_MSUN) ** mass_horizon_index
    return np.asarray(
        0.5 * erfc((np.asarray(d_L, dtype=np.float64) - d50_eff) / (np.sqrt(2.0) * w_pdet)),
        dtype=np.float64,
    )


def phi_marginal_survival_table(
    h_grid: npt.NDArray[np.float64],
    *,
    mass_slope: float = 0.0,
    mass_horizon_index: float = 0.0,
    d50: float = D50_GPC,
    w_pdet: float = W_PDET_GPC,
    n_z: int = 1500,
    n_mass_quad: int = 400,
    z_chunk: int = 150,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Tabulate the phi-marginal survival ``S_bar_phi(z; h)``.

    Harness analog of production's
    ``bayesian_statistics.precompute_phi_marginal_survival``
    (``bayesian_statistics.py:1869``):

    .. math::

        \bar S_\phi(z;h) = \int \phi(M)\,
            S_\mathrm{4D}\bigl(d_L(z;h),\, M(1+z)\bigr)\,\mathrm{d}M

    — ONE contraction of the SAME survival object
    (:func:`survival_with_mass`) that :func:`completion_mass_factor_g_sel`
    queries per Hermite node, so the tower identity holds by construction
    exactly as production's ``r_phi == 1``. The table is built once per run and
    READ by ``np.interp`` afterwards, mirroring production's "read, never
    rebuild" discipline (``bayesian_statistics.py:4503-4507``).

    Args:
        h_grid: H0 evaluation grid.
        mass_slope: ``phi`` slope (see :func:`dark_mass_density_per_mass`).
        mass_horizon_index: ``S_4D`` horizon index ``alpha_M``.
        d50: 50% detection distance [Gpc].
        w_pdet: Roll-off width [Gpc].
        n_z: Redshift table nodes on ``[Z_MIN, Z_MAX_POP]``.
        n_mass_quad: ``ln M`` quadrature nodes.
        z_chunk: Redshift nodes contracted per block (memory control).

    Returns:
        ``(z_grid, S_bar_phi)`` with ``S_bar_phi`` of shape ``(n_z, nh)``.
    """
    z_grid = np.linspace(Z_MIN, Z_MAX_POP, n_z)
    ln_m = np.linspace(math.log(M_SOURCE_MIN), math.log(M_SOURCE_MAX), n_mass_quad)
    m_grid = np.exp(ln_m)
    # phi(M) dM = phi(M) M dlnM. The weights are renormalized ON THE
    # QUADRATURE GRID so that S == 1 gives S_bar_phi == 1 exactly: the trapezoid
    # rule loses half a bin at each edge of the compact phi support, which would
    # otherwise put a ~1/n_mass_quad multiplicative offset between S_bar_phi and
    # the mass-blind p_det it must reduce to at mass_horizon_index = 0.
    phi_m = dark_mass_density_per_mass(m_grid, mass_slope) * m_grid  # (nM,)
    phi_m = phi_m / float(np.trapezoid(phi_m, ln_m))
    a_z = comoving_amplitude_of_z(z_grid)  # (nz,)
    out = np.empty((z_grid.size, h_grid.size), dtype=np.float64)
    for start in range(0, z_grid.size, z_chunk):
        sl = slice(start, min(start + z_chunk, z_grid.size))
        d_l = a_z[sl][:, None, None] / h_grid[None, :, None]  # (k, nh, 1)
        m_z = m_grid[None, None, :] * (1.0 + z_grid[sl][:, None, None])  # (k, 1, nM)
        s_4d = survival_with_mass(d_l, m_z, d50, w_pdet, mass_horizon_index)  # (k, nh, nM)
        out[sl] = np.trapezoid(s_4d * phi_m[None, None, :], ln_m, axis=2)
    return z_grid, out


def _interp_survival_table(
    z: npt.NDArray[np.float64],
    z_grid: npt.NDArray[np.float64],
    s_table: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Read ``S_bar_phi`` at arbitrary redshifts, one ``np.interp`` per h column.

    Mirrors production's accessor convention exactly: linear interpolation with
    endpoint clamping outside the table domain
    (``bayesian_statistics.py:4528``).

    Args:
        z: Query redshifts, shape ``(k,)`` (flattened by the caller).
        z_grid: Table redshift nodes, shape ``(n_z,)``.
        s_table: Table values, shape ``(n_z, nh)``.

    Returns:
        ``S_bar_phi`` at the query points, shape ``(k, nh)``.
    """
    zq = np.asarray(z, dtype=np.float64).ravel()
    out = np.empty((zq.size, s_table.shape[1]), dtype=np.float64)
    for j in range(s_table.shape[1]):
        out[:, j] = np.interp(zq, z_grid, s_table[:, j])
    return out


def completion_mass_factor_g(
    z_nodes: npt.NDArray[np.float64],
    d_L_fraction: npt.NDArray[np.float64],
    det_M_z: float,
    proj_d_L_to_M: float,
    sigma_cond_M: npt.NDArray[np.float64] | float,
    *,
    n_hermite: int = 24,
    mass_slope: float = 0.0,
) -> npt.NDArray[np.float64]:
    r"""Completion-leg mass density ``g_i(z;h)`` at quadrature nodes.

    Harness analog of production's
    ``bayesian_statistics.completion_mass_factor_g``
    (``bayesian_statistics.py:2022``), same mathematical form:

    .. math::

        g_i(z;h) = \int \mathrm{d}x_M\,
            \mathcal{N}\bigl(x_M; \mu_\mathrm{cond}(z;h), \sigma_\mathrm{cond}\bigr)\,
            \phi_x(x_M; z),\qquad
        \phi_x(x_M;z) = \phi\Bigl(x_M \frac{M_{z,\mathrm{det},i}}{1+z}\Bigr)
            \frac{M_{z,\mathrm{det},i}}{1+z}

    with ``x_M = M_z / M_z,det,i`` the dimensionless mass coordinate the 2D
    catalogue leg's mass overlap is a density in, and
    ``mu_cond(z;h) = 1 + proj (d_L(z;h)/d_L,det - 1)``. Because ``mu_cond``
    carries ``d_L(z;h)``, ``g_i`` is a function of ``h`` and is RECOMPUTED at
    every h-grid node by construction (the last axis of the node arrays is the
    h grid) — never frozen, never elided.

    Differences from production, deliberate: Gauss-Hermite at a fixed order
    (no Route-1 adaptive split — the harness ``phi`` has no breakpoints), and
    ``sigma_cond_M`` may be an array so the harness's noise-model toggle can
    make the conditional width track the MODELLED mass rather than the
    observed one.

    Args:
        z_nodes: Redshift nodes, any shape ``S`` (typically ``(nz, nh)``).
        d_L_fraction: ``d_L(z;h)/d_L,det,i`` at the same nodes, shape ``S``.
        det_M_z: The event's measured detector-frame BH mass ``M_z,det,i``.
        proj_d_L_to_M: ``rho sigma_M / sigma_D`` — the 2x2 block projection
            (production: ``cov_4d[3,2]/cov_4d[2,2]``).
        sigma_cond_M: ``sigma_M sqrt(1 - rho^2)`` — scalar, or an array of
            shape ``S`` under the model-sigma noise convention.
        n_hermite: Gauss-Hermite order.
        mass_slope: ``phi`` slope.

    Returns:
        ``g_i`` at the nodes, shape ``S``, in units of ``1/x_M``.

    References:
        Mandel, Farr & Gair (2019), arXiv:1809.02063, Eqs. (5)-(7).
        Gray et al. (2020), arXiv:1908.06050, Eq. (A.19).
    """
    return _mass_factor_core(
        z_nodes,
        d_L_fraction,
        det_M_z,
        proj_d_L_to_M,
        sigma_cond_M,
        n_hermite=n_hermite,
        mass_slope=mass_slope,
        survival=None,
    )


def completion_mass_factor_g_sel(
    z_nodes: npt.NDArray[np.float64],
    d_L_gpc: npt.NDArray[np.float64],
    d_L_fraction: npt.NDArray[np.float64],
    det_M_z: float,
    proj_d_L_to_M: float,
    sigma_cond_M: npt.NDArray[np.float64] | float,
    *,
    n_hermite: int = 24,
    mass_slope: float = 0.0,
    mass_horizon_index: float = 0.0,
    d50: float = D50_GPC,
    w_pdet: float = W_PDET_GPC,
) -> npt.NDArray[np.float64]:
    r"""FUSED completion-leg mass density ``g_sel(z;h)`` — survival INSIDE ``dx_M``.

    Harness analog of production's
    ``bayesian_statistics.completion_mass_factor_g_sel``
    (``bayesian_statistics.py:2155``), the [P1] leg of the selection fusion
    landed in ``2b10b8b8`` (ledger rows #117-#118):

    .. math::

        g_\mathrm{sel}(z;h) = \int \mathrm{d}x_M\,
            \mathcal{N}\bigl(x_M; \mu_\mathrm{cond}(z;h), \sigma_\mathrm{cond}\bigr)\,
            \phi_x(x_M;z)\,
            S_\mathrm{4D}\bigl(d_L(z;h),\, x_M M_{z,\mathrm{det},i}\bigr)

    — :func:`completion_mass_factor_g` with the detection survival integrated
    against the observed-mass likelihood in the SAME single ``dx_M`` (the
    selected-population prior of a latent-thresholded detection model). The
    survival is queried at the DETECTOR-frame Hermite-node mass
    ``x_M M_z,det,i`` and the node's absolute ``d_L(z;h)`` — exactly the pair
    :func:`phi_marginal_survival_table` contracts, so the two legs share one
    survival object as production requires.

    Limiting cases (both pinned by tests): ``mass_horizon_index = 0`` with the
    ``d_L``-only survival still multiplies ``g_i`` by a mass-independent
    factor, and ``sigma_cond -> 0`` gives ``g_i * S(mu_cond M_z,det)`` — per
    row #118 / MAJOR-1 effectively the production operating point (measured
    ``d_L``-conditional ``sigma_cond`` p50 = 8.8e-8).

    Args:
        z_nodes: Redshift nodes, any shape ``S``.
        d_L_gpc: Absolute ``d_L(z;h)`` [Gpc] at the same nodes, shape ``S``.
        d_L_fraction: ``d_L(z;h)/d_L,det,i`` at the same nodes, shape ``S``.
        det_M_z: The event's measured detector-frame BH mass.
        proj_d_L_to_M: 2x2 block projection ``rho sigma_M / sigma_D``.
        sigma_cond_M: Conditional mass width (scalar or shape ``S``).
        n_hermite: Gauss-Hermite order.
        mass_slope: ``phi`` slope.
        mass_horizon_index: ``S_4D`` horizon index.
        d50: 50% detection distance [Gpc].
        w_pdet: Roll-off width [Gpc].

    Returns:
        ``g_sel`` at the nodes, shape ``S``, in units of ``1/x_M``.

    References:
        Mandel, Farr & Gair (2019), arXiv:1809.02063, Eqs. (5)-(7).
    """

    def _survival(
        m_z: npt.NDArray[np.float64], _z: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return survival_with_mass(
            np.asarray(d_L_gpc, dtype=np.float64)[..., None],
            m_z,
            d50,
            w_pdet,
            mass_horizon_index,
        )

    return _mass_factor_core(
        z_nodes,
        d_L_fraction,
        det_M_z,
        proj_d_L_to_M,
        sigma_cond_M,
        n_hermite=n_hermite,
        mass_slope=mass_slope,
        survival=_survival,
    )


def _mass_factor_core(
    z_nodes: npt.NDArray[np.float64],
    d_L_fraction: npt.NDArray[np.float64],
    det_M_z: float,
    proj_d_L_to_M: float,
    sigma_cond_M: npt.NDArray[np.float64] | float,
    *,
    n_hermite: int,
    mass_slope: float,
    survival: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    | None,
) -> npt.NDArray[np.float64]:
    """Shared Gauss-Hermite contraction behind ``g`` and ``g_sel``.

    Args:
        z_nodes: Redshift nodes, shape ``S``.
        d_L_fraction: ``d_L(z;h)/d_L,det,i``, shape ``S``.
        det_M_z: Measured detector-frame BH mass.
        proj_d_L_to_M: 2x2 block projection.
        sigma_cond_M: Conditional width (scalar or shape ``S``).
        n_hermite: Gauss-Hermite order.
        mass_slope: ``phi`` slope.
        survival: ``(M_z, z) -> S_4D`` on shape ``S + (n_hermite,)`` arrays, or
            None for the unfused factor.

    Returns:
        The contracted mass factor, shape ``S``.
    """
    from scipy.special import roots_hermite  # local: only the mass channel needs it

    z_arr = np.asarray(z_nodes, dtype=np.float64)
    frac = np.asarray(d_L_fraction, dtype=np.float64)
    sig = np.asarray(sigma_cond_M, dtype=np.float64)
    x_nodes, x_weights = roots_hermite(n_hermite)
    # dM/dx_M at each node: the mass scale the dimensionless coordinate rides on.
    scale = det_M_z / (1.0 + z_arr)  # S
    mu_cond = 1.0 + proj_d_L_to_M * (frac - 1.0)  # S
    # Gauss-Hermite for E_{x~N(mu,sigma)}[.]: nodes mu + sqrt(2) sigma t_j.
    x_M = mu_cond[..., None] + math.sqrt(2.0) * sig[..., None] * x_nodes  # S + (n_h,)
    phi_x = dark_mass_density_per_mass(x_M * scale[..., None], mass_slope) * scale[..., None]
    if survival is not None:
        m_z_query = x_M * det_M_z  # detector-frame query mass (S_bar_phi's pair)
        s_4d = np.where(m_z_query > 0.0, survival(np.clip(m_z_query, 1e-30, None), z_arr), 0.0)
        phi_x = phi_x * s_4d
    return np.asarray((phi_x @ x_weights) / math.sqrt(math.pi), dtype=np.float64)


def _completion_numerator_batch(
    dL_obs: npt.NDArray[np.float64],
    sig_dl: npt.NDArray[np.float64],
    z_support: float,
    h_grid: npt.NDArray[np.float64],
    config: PPCoverageConfig,
    survival_table: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None,
    det_M_z: npt.NDArray[np.float64] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Vectorized completion numerators ``B_num`` (1D) and ``B_num_wbh`` (2D).

    The batched, mass-aware analog of :func:`_completion_numerator`: it
    evaluates the same integral for a whole block of events at once (the
    [A3] criterion (ii) production-N vectorization) and adds the two
    fusion legs of production commit ``2b10b8b8``:

    * 1D ([P2], ``bayesian_statistics.py:4495-4514``): the integrand is
      multiplied by ``S_bar_phi(z;h)`` when the selection cell is ``1d`` or
      ``fused``.
    * 2D ([P1], ``bayesian_statistics.py:4592-4609``): the integrand is
      multiplied by the completion mass factor — ``g_sel`` when the cell is
      ``2d`` or ``fused``, plain ``g`` otherwise. Both are recomputed at every
      h-grid node (the h axis is an array axis of the node grid).

    Args:
        dL_obs: Observed GW distances [Gpc], shape ``(B,)``.
        sig_dl: Absolute distance uncertainties [Gpc], shape ``(B,)``.
        z_support: Completeness edge (lower limit of the completion volume).
        h_grid: H0 evaluation grid, shape ``(nh,)``.
        config: Harness configuration.
        survival_table: ``(z_grid, S_bar_phi)`` or None (no mass channel).
        det_M_z: Per-event measured detector-frame BH masses, shape ``(B,)``,
            or None to skip the 2D leg.

    Returns:
        ``(B_num, B_num_wbh)``, each of shape ``(B, nh)``. ``B_num_wbh`` is
        ``B_num`` when ``det_M_z`` is None.
    """
    nh = h_grid.size
    nb = dL_obs.size
    nz = config.n_z_quad
    z_lo = np.maximum(
        max(Z_MIN, z_support),
        z_of_comoving_amplitude((dL_obs - 5.0 * sig_dl) * h_grid.min()),
    )
    z_hi = np.minimum(Z_MAX_POP, z_of_comoving_amplitude((dL_obs + 5.0 * sig_dl) * h_grid.max()))
    empty = z_hi <= z_lo
    z_hi_safe = np.where(empty, z_lo + 1e-9, z_hi)
    zq = np.linspace(z_lo, z_hi_safe, nz, axis=-1)  # (B, nz)
    wq = np.gradient(zq, axis=1)  # (B, nz)
    a_z = comoving_amplitude_of_z(zq)  # (B, nz)
    dLg = a_z[:, :, None] / h_grid[None, None, :]  # (B, nz, nh)
    sig_b: npt.NDArray[np.float64] = (
        config.sigma_dl_frac * dLg
        if config.sigma_dl_model_in_likelihood
        else np.broadcast_to(sig_dl[:, None, None], dLg.shape)
    )
    integrand = _norm_pdf(dLg, dL_obs[:, None, None], sig_b)  # (B, nz, nh)
    if config.pdet_in_numerator:
        integrand = integrand * detection_probability(dLg, config.d50_gpc, config.w_pdet_gpc)
    wpop = _inference_population_weight(zq, config.inference_wpop_tilt)  # (B, nz)
    base = integrand * (wq * wpop)[:, :, None]  # (B, nz, nh)

    sel_1d = config.selection_cell in ("1d", "fused")
    sel_2d = config.selection_cell in ("2d", "fused")
    if sel_1d:
        if survival_table is None:
            raise ValueError("selection_cell '1d'/'fused' needs the S_bar_phi table")
        s_bar = _interp_survival_table(zq, survival_table[0], survival_table[1])
        base_1d = base * s_bar.reshape(nb, nz, nh)
    else:
        base_1d = base
    b_num = np.asarray(base_1d.sum(axis=1), dtype=np.float64)  # (B, nh)

    if det_M_z is None:
        b_num_wbh = b_num
    else:
        proj, sigma_cond = _mass_conditional_parameters(config)
        frac = dLg / dL_obs[:, None, None]
        sig_cond_eff: npt.NDArray[np.float64] | float = sigma_cond
        if config.sigma_dl_model_in_likelihood:
            # Model-sigma convention: the conditional width tracks the MODELLED
            # mass fraction rather than the observed one (the mass-channel
            # counterpart of the #67 sigma(dL_true) noise model).
            sig_cond_eff = sigma_cond * np.clip(1.0 + proj * (frac - 1.0), 1e-6, None)
        g_all = np.empty_like(base)
        for i in range(nb):
            sig_i = sig_cond_eff[i] if isinstance(sig_cond_eff, np.ndarray) else sig_cond_eff
            if sel_2d:
                g_all[i] = completion_mass_factor_g_sel(
                    zq[i][:, None] * np.ones((1, nh)),
                    dLg[i],
                    frac[i],
                    float(det_M_z[i]),
                    proj,
                    sig_i,
                    n_hermite=config.n_hermite,
                    mass_slope=config.mass_slope,
                    mass_horizon_index=config.mass_horizon_index,
                    d50=config.d50_gpc,
                    w_pdet=config.w_pdet_gpc,
                )
            else:
                g_all[i] = completion_mass_factor_g(
                    zq[i][:, None] * np.ones((1, nh)),
                    frac[i],
                    float(det_M_z[i]),
                    proj,
                    sig_i,
                    n_hermite=config.n_hermite,
                    mass_slope=config.mass_slope,
                )
        b_num_wbh = np.asarray((base * g_all).sum(axis=1), dtype=np.float64)

    floor = np.full(nh, 1e-300)
    b_num = np.where(empty[:, None], floor, b_num)
    b_num_wbh = np.where(empty[:, None], floor, b_num_wbh)
    return b_num, b_num_wbh


def _mass_conditional_parameters(config: PPCoverageConfig) -> tuple[float, float]:
    """Return ``(proj, sigma_cond)`` of the ``(d_L_frac, M_z_frac)`` 2x2 block.

    Production reads these off ``cov_4d`` (``proj = cov_4d[3,2]/cov_4d[2,2]``,
    ``sigma_cond = sqrt(cov_4d[3,3] - cov_4d[3,2]^2/cov_4d[2,2])``); the
    harness's fractional covariance is built from
    ``(sigma_dl_frac, sigma_mz_frac, rho_dl_mz)``, giving the same Gaussian
    conditional (Bishop 2006, PRML Eqs. 2.81-2.82).

    Args:
        config: Harness configuration.

    Returns:
        ``(proj_d_L_to_M, sigma_cond_M)`` in fractional mass units.
    """
    rho = config.rho_dl_mz
    proj = rho * config.sigma_mz_frac / config.sigma_dl_frac
    sigma_cond = config.sigma_mz_frac * math.sqrt(max(1.0 - rho * rho, 0.0))
    return proj, max(sigma_cond, 1e-12)


def _sample_galaxy_masses(
    n: int, rng: np.random.Generator, slope: float, ngrid: int = 4000
) -> npt.NDArray[np.float64]:
    """Draw source-frame BH masses from ``phi(M)`` by inverse CDF.

    Args:
        n: Number of galaxies.
        rng: Random generator.
        slope: ``phi`` slope.
        ngrid: Inverse-CDF resolution in ``ln M``.

    Returns:
        Masses [M_sun], shape ``(n,)``.
    """
    ln_m = np.linspace(math.log(M_SOURCE_MIN), math.log(M_SOURCE_MAX), ngrid)
    m = np.exp(ln_m)
    pdf = dark_mass_density_per_mass(m, slope) * m
    cdf = np.concatenate([np.array([0.0]), np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(ln_m))])
    cdf /= cdf[-1]
    return np.asarray(np.exp(np.interp(rng.random(n), cdf, ln_m)), dtype=np.float64)


def _run_realization_catalogue_mass(
    h_true: float,
    h_grid: npt.NDArray[np.float64],
    log_Dh: npt.NDArray[np.float64],
    config: PPCoverageConfig,
    rng: np.random.Generator,
    catalogue: SyntheticCatalogue,
    beta_G: npt.NDArray[np.float64],
    beta_Gbar: npt.NDArray[np.float64],
    survival_table: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, float]]:
    """Two-channel (1D + 2D mass) catalogue-mode realization, vectorized in event blocks.

    The mass-bearing counterpart of :func:`_run_realization_catalogue`. Both
    channels share ONE generative universe and ONE denominator, exactly as
    production's per-event both-channel diagnostics do:

    * **1D channel** — the mass observable is DISCARDED from the likelihood
      (it still drives selection). Per-event
      ``p_i = [T_i + B_num,i] / Den`` with the same ``T_i`` forms as
      :func:`_run_realization_catalogue`, and ``B_num`` carrying
      ``S_bar_phi(z;h)`` under selection cells ``1d``/``fused`` ([P2]).
    * **2D channel** — every candidate's ball measure additionally carries the
      Gaussian mass overlap
      ``mz_g(z;h) = INTEGRAL N(x; mu_cond(z;h), sigma_cond) N(x; mu_gal,g(z), sigma_gal) dx``
      (production's analytic ``mz_integral``, ``bayesian_statistics.py:5522``),
      and the completion leg carries the mass factor ``g_i``/``g_sel``
      recomputed at every h.

    Selection integrals (``D``, ``beta_G``, ``Sigma_glob``) are built from
    ``S_bar_phi`` rather than the mass-blind ``p_det`` whenever the mass
    channel is on — the harness analog of production's phi-convention legs
    ``precompute_phi_selection_integrals`` (``bayesian_statistics.py:1964``).

    Args:
        h_true: Injected truth (generative model only).
        h_grid: H0 evaluation grid.
        log_Dh: ``log D(h)`` (phi convention) on the grid.
        config: Harness configuration.
        rng: Realization RNG.
        catalogue: The synthetic catalogue (mass columns required).
        beta_G: In-catalogue selection integral (phi convention).
        beta_Gbar: Out-of-catalogue selection integral (phi convention).
        survival_table: ``(z_grid, S_bar_phi)``.

    Returns:
        ``(logL_1d, logL_2d, diagnostics)``.
    """
    if catalogue.mass_true is None or catalogue.mass_obs is None:
        raise ValueError("mass_channel realization needs a mass-bearing catalogue")
    sigma_z = float(np.hypot(config.sigma_z, config.sigma_z_pv))
    zs = float(config.z_support) if config.z_support is not None else Z_MAX_POP
    mode = config.mixture_mode
    nh = h_grid.size
    proj, sigma_cond = _mass_conditional_parameters(config)

    # --- generative model -------------------------------------------------
    m_true = catalogue.mass_true
    rate_mass_all = (m_true / M_REF_MSUN) ** config.mass_rate_index
    z_all = catalogue.z_true
    p_draw = (
        rate_mass_all
        * host_rate_weight_of_z(z_all)
        * survival_with_mass(
            comoving_amplitude_of_z(z_all) / h_true,
            m_true * (1.0 + z_all),
            config.d50_gpc,
            config.w_pdet_gpc,
            config.mass_horizon_index,
        )
    )
    p_draw = np.clip(p_draw, 0.0, None)
    p_draw = p_draw / p_draw.sum()
    host_idx = rng.choice(z_all.size, size=config.n_events, p=p_draw)
    z_host = z_all[host_idx]
    dL_host = comoving_amplitude_of_z(z_host) / h_true
    m_z_host = m_true[host_idx] * (1.0 + z_host)
    # Correlated fractional (d_L, M_z) measurement errors — the harness's
    # cov_4d (2,3) block.
    cov = np.array(
        [
            [
                config.sigma_dl_frac**2,
                config.rho_dl_mz * config.sigma_dl_frac * config.sigma_mz_frac,
            ],
            [
                config.rho_dl_mz * config.sigma_dl_frac * config.sigma_mz_frac,
                config.sigma_mz_frac**2,
            ],
        ]
    )
    eps = rng.multivariate_normal(np.zeros(2), cov, size=config.n_events)
    if not config.gw_measurement_scatter:
        # Q-0 (2026-08-17) production-faithful cell: no measurement scatter in
        # EITHER observable; the draw is discarded, not skipped, so the paired
        # scatter/no-scatter cells share the random stream.
        eps = np.zeros_like(eps)
    dL_obs = np.clip(dL_host * (1.0 + eps[:, 0]), 1e-3, None)
    det_M_z = np.clip(m_z_host * (1.0 + eps[:, 1]), 1e-3, None)
    sig_dl = config.sigma_dl_frac * dL_obs
    cos_theta_c = 1.0 - 2.0 * config.sky_frac
    cap_centre = _perturb_within_cap(catalogue.direction[host_idx], cos_theta_c, rng)
    balls: list[list[int]] = catalogue.tree.query_ball_point(cap_centre, catalogue.chord_radius)

    host_catalogued = catalogue.catalogued[host_idx]
    cat_pos = np.full(z_all.size, -1, dtype=np.int64)
    cat_pos[catalogue.cat_index] = np.arange(catalogue.cat_index.size, dtype=np.int64)
    host_cat_pos = cat_pos[host_idx]

    n_bar_w: npt.NDArray[np.float64] = catalogue.sigma_glob / np.clip(beta_G, 1e-300, None)
    if mode == "generator_marginal":
        d_gen = catalogue.sigma_glob / catalogue.n_hat_w + beta_Gbar
        log_den = np.log(np.clip(d_gen, 1e-300, None))
        scale = catalogue.n_hat_w * config.sky_frac
    else:
        log_den = log_Dh
        scale = float("nan")

    # Per-catalogued-galaxy rate weight mass factor (production: R_eff(M_g)).
    cat_rate_mass = (catalogue.mass_obs / M_REF_MSUN) ** config.mass_rate_index

    logL_1d = np.zeros(nh)
    logL_2d = np.zeros(nh)
    n_empty = 0
    n_ball_total = 0
    n_host_in_ball = 0
    ball_sizes = np.asarray([len(b) for b in balls], dtype=np.int64)

    chunk = max(int(config.event_chunk), 1)
    for start in range(0, config.n_events, chunk):
        stop = min(start + chunk, config.n_events)
        idx = np.arange(start, stop)
        nb = idx.size
        b_num, b_num_wbh = _completion_numerator_batch(
            dL_obs[idx],
            sig_dl[idx],
            zs,
            h_grid,
            config,
            survival_table,
            det_M_z[idx],
        )  # (nb, nh) each

        z_lo = np.maximum(
            Z_MIN,
            z_of_comoving_amplitude((dL_obs[idx] - 5.0 * sig_dl[idx]) * h_grid.min())
            - 4.0 * sigma_z,
        )
        z_hi = np.minimum(
            Z_MAX_POP,
            z_of_comoving_amplitude((dL_obs[idx] + 5.0 * sig_dl[idx]) * h_grid.max())
            + 4.0 * sigma_z,
        )
        bad_window = z_hi <= z_lo
        z_hi = np.where(bad_window, z_lo + 1e-9, z_hi)
        zq = np.linspace(z_lo, z_hi, config.n_z_quad, axis=-1)  # (nb, nz)
        wq = np.gradient(zq, axis=1)
        dLg = comoving_amplitude_of_z(zq)[:, :, None] / h_grid[None, None, :]  # (nb, nz, nh)
        sig_gw: npt.NDArray[np.float64] = (
            config.sigma_dl_frac * dLg
            if config.sigma_dl_model_in_likelihood
            else np.broadcast_to(sig_dl[idx][:, None, None], dLg.shape)
        )
        pGW = _norm_pdf(dLg, dL_obs[idx][:, None, None], sig_gw)  # (nb, nz, nh)
        if config.pdet_in_numerator:
            pGW = pGW * detection_probability(dLg, config.d50_gpc, config.w_pdet_gpc)
        wpop_q = _inference_population_weight(zq, config.inference_wpop_tilt)  # (nb, nz)

        # Ragged ball members flattened with their in-chunk event index.
        mem_event = np.concatenate(
            [np.full(len(balls[i]), k, dtype=np.int64) for k, i in enumerate(idx)]
            + [np.zeros(0, dtype=np.int64)]
        )
        mem_gal = np.concatenate(
            [np.asarray(balls[i], dtype=np.int64) for i in idx] + [np.zeros(0, dtype=np.int64)]
        )
        n_ball_total += int(mem_gal.size)
        for i in idx:
            if host_cat_pos[i] >= 0 and host_cat_pos[i] in balls[i]:
                n_host_in_ball += 1
            if len(balls[i]) == 0:
                n_empty += 1

        rho_1d = np.zeros((nb, config.n_z_quad))
        rho_2d = np.zeros((nb, config.n_z_quad, nh))
        if mem_gal.size:
            kern = (
                _norm_pdf(zq[mem_event], catalogue.z_obs[mem_gal][:, None], sigma_z)
                * catalogue.inv_norm[mem_gal][:, None]
                * cat_rate_mass[mem_gal][:, None]
            )  # (nmem, nz)
            np.add.at(rho_1d, mem_event, kern)
            # 2D: Gaussian mass overlap per candidate, recomputed at every h.
            mu_cond = 1.0 + proj * (dLg / dL_obs[idx][:, None, None] - 1.0)  # (nb, nz, nh)
            sig_cond_eff: npt.NDArray[np.float64] = (
                sigma_cond * np.clip(mu_cond, 1e-6, None)
                if config.sigma_dl_model_in_likelihood
                else np.full_like(mu_cond, sigma_cond)
            )
            mu_gal = (
                catalogue.mass_obs[mem_gal][:, None]
                * (1.0 + zq[mem_event])
                / det_M_z[idx][mem_event][:, None]
            )  # (nmem, nz)
            sig_gal = config.sigma_m_gal_frac * np.clip(mu_gal, 1e-12, None)
            s2 = sig_cond_eff[mem_event] ** 2 + (sig_gal**2)[:, :, None]  # (nmem, nz, nh)
            mz = np.exp(-0.5 * (mu_cond[mem_event] - mu_gal[:, :, None]) ** 2 / s2) / np.sqrt(
                2.0 * np.pi * s2
            )
            np.add.at(rho_2d, mem_event, kern[:, :, None] * mz)

        common = (wq * wpop_q)[:, :, None] * pGW  # (nb, nz, nh)
        sum_wN_1d = np.asarray((common * rho_1d[:, :, None]).sum(axis=1), dtype=np.float64)
        sum_wN_2d = np.asarray((common * rho_2d).sum(axis=1), dtype=np.float64)
        if mode == "lcat":
            pdet_q = _interp_survival_table(zq, survival_table[0], survival_table[1]).reshape(
                nb, config.n_z_quad, nh
            )
            base_d = (wq * wpop_q)[:, :, None] * pdet_q
            sum_wD = np.asarray((base_d * rho_1d[:, :, None]).sum(axis=1), dtype=np.float64)
            term_1d = beta_G[None, :] * (sum_wN_1d / np.clip(sum_wD, 1e-300, None))
            term_2d = beta_G[None, :] * (sum_wN_2d / np.clip(sum_wD, 1e-300, None))
        elif mode == "absolute":
            denom_scale = np.clip(n_bar_w * config.sky_frac, 1e-300, None)[None, :]
            term_1d = sum_wN_1d / denom_scale
            term_2d = sum_wN_2d / denom_scale
        else:  # generator_marginal
            term_1d = sum_wN_1d / max(scale, 1e-300)
            term_2d = sum_wN_2d / max(scale, 1e-300)
        empty_mask = bad_window[:, None]
        term_1d = np.where(empty_mask, 0.0, term_1d)
        term_2d = np.where(empty_mask, 0.0, term_2d)
        logL_1d += np.log(np.clip(term_1d + b_num, 1e-300, None)).sum(axis=0) - nb * log_den
        logL_2d += np.log(np.clip(term_2d + b_num_wbh, 1e-300, None)).sum(axis=0) - nb * log_den

    n = float(config.n_events)
    diagnostics = {
        "completion_fraction": float(np.sum(~host_catalogued) / n),
        "empty_ball_fraction": n_empty / n,
        "mean_ball_size": float(np.mean(ball_sizes)),
        "host_in_ball_fraction": n_host_in_ball / n,
        "impostor_fraction": (
            (n_ball_total - n_host_in_ball) / n_ball_total if n_ball_total > 0 else 0.0
        ),
    }
    return logL_1d, logL_2d, diagnostics


def run_coverage(config: PPCoverageConfig) -> dict[str, Any]:
    """Run the P-P / coverage test for one kernel choice.

    Args:
        config: Harness configuration; all randomness is seeded from
            ``config.seed`` via ``np.random.default_rng``.

    Returns:
        JSON-serializable dict with keys ``"config"`` (the config as a dict)
        and ``"results"`` — one entry per injected truth (stringified H0)
        containing ``coverage`` (fractions at 50/68/90% HPD),
        ``rail_fraction``, ``map_mean``, ``map_std``, ``map_median``,
        ``map_bias`` (map_mean - truth), ``completion_fraction`` (mean
        fraction of events routed into the ``z_support`` zero-host
        pure-completion branch per realization; 0.0 when ``z_support`` is
        None or >= ``Z_MAX_POP``), and the per-branch tilt diagnostics
        ``dlogL_dh_host_mean`` / ``dlogL_dh_completion_mean`` (mean over
        realizations of d(logL_branch)/dh at the grid node nearest h_true;
        None — JSON null — when a branch had no events in any realization).

    Raises:
        ValueError: If ``config.mixture_mode`` is not "two_branch" and
            ``config.z_support`` is None (the Gray mixture and the
            membership-truncated exact kernel are only defined with a
            catalogue-support edge).
    """
    h_grid = config.h_grid()
    if config.selection_cell not in SELECTION_CELLS:
        raise ValueError(
            f"selection_cell must be one of {SELECTION_CELLS}, got {config.selection_cell!r}"
        )
    if config.mass_channel and not config.catalogue_mode:
        raise ValueError(
            "mass_channel=True requires catalogue_mode=True: the 2D catalogue leg is a "
            "per-CANDIDATE mass overlap, which needs a discrete multi-galaxy ball."
        )
    if config.selection_cell != "off" and not config.mass_channel:
        raise ValueError(
            f"selection_cell={config.selection_cell!r} requires mass_channel=True (the "
            "S_bar_phi / g_sel legs it switches only exist in the mass channel)."
        )
    # Mass channel: the phi-marginal survival table replaces the mass-blind
    # p_det in EVERY selection integral (harness analog of production's
    # phi-convention legs, bayesian_statistics.precompute_phi_selection_integrals).
    survival_table: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None
    if config.mass_channel:
        survival_table = phi_marginal_survival_table(
            h_grid,
            mass_slope=config.mass_slope,
            mass_horizon_index=config.mass_horizon_index,
            d50=config.d50_gpc,
            w_pdet=config.w_pdet_gpc,
            n_z=config.n_z_survival,
            n_mass_quad=config.n_mass_quad,
        )

    def _selection_kernel(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Detection weight in the selection integrals, shape ``(nz, nh)``."""
        if survival_table is not None:
            return _interp_survival_table(z, survival_table[0], survival_table[1])
        return detection_probability(
            comoving_amplitude_of_z(z)[:, None] / h_grid[None, :],
            config.d50_gpc,
            config.w_pdet_gpc,
        )

    # Selection denominator D(h) = int p_det(A(z)/h) w_pop(z) dz (shared);
    # with the mass channel, p_det -> S_bar_phi(z;h).
    zint = np.linspace(Z_MIN, Z_MAX_POP, 3000)
    wpop = _inference_population_weight(zint, config.inference_wpop_tilt)
    Dh = np.trapezoid(_selection_kernel(zint) * wpop[:, None], zint, axis=0)
    log_Dh = np.log(Dh)

    # In-catalogue selection integral beta_G(h) = int_{Z_MIN}^{zs} p_det w_pop
    # dz, precomputed once like log_Dh (only for mixture modes) on D(h)'s OWN
    # node convention (np.linspace(..., 3000)) so that at z_support >=
    # Z_MAX_POP beta_G == Dh exactly (limiting-case identity).
    beta_G: npt.NDArray[np.float64] | None = None
    beta_Gbar: npt.NDArray[np.float64] | None = None
    if config.mixture_mode != "two_branch" and config.z_support is None:
        raise ValueError(
            "mixture_mode='gray'/'conditioned'/'exact'/'absolute'/'lcat'/"
            "'generator_marginal' requires z_support: the Gray mixture, the "
            "membership-truncated exact kernel, the absolute-mass marginal and the "
            "catalogue-mode estimators are only defined with a catalogue-support edge."
        )
    if config.catalogue_mode and config.mixture_mode not in CATALOGUE_MIXTURE_MODES:
        raise ValueError(
            f"catalogue_mode=True supports mixture_mode in {CATALOGUE_MIXTURE_MODES}, "
            f"got {config.mixture_mode!r}."
        )
    if not config.catalogue_mode and config.mixture_mode in ("lcat", "generator_marginal"):
        raise ValueError(
            f"mixture_mode={config.mixture_mode!r} is catalogue-mode only "
            "(it needs a discrete multi-galaxy candidate ball): set catalogue_mode=True."
        )
    if config.catalogue_mode and config.kernel != "volume":
        raise ValueError(
            "catalogue_mode derives its per-galaxy redshift kernel from the catalogue's "
            "own generative model (n_gal(z) prior x photo-z Gaussian); kernel='bare' is "
            "not defined there. Use kernel='volume' (the default)."
        )
    if config.catalogue_mode or config.mixture_mode in ("gray", "conditioned"):
        # exact needs neither beta_G nor beta_Gbar (no mixture weight, no
        # conditioned denominators): only gray/conditioned and the
        # catalogue-mode estimators compute them.
        assert config.z_support is not None  # guarded above
        zbg = np.linspace(Z_MIN, min(config.z_support, Z_MAX_POP), 3000)
        beta_G = np.asarray(
            np.trapezoid(
                _selection_kernel(zbg)
                * _inference_population_weight(zbg, config.inference_wpop_tilt)[:, None],
                zbg,
                axis=0,
            ),
            dtype=np.float64,
        )
        # Out-of-catalogue selection integral int_{zs}^{Z_MAX_POP} p_det w_pop dz.
        beta_Gbar = np.asarray(Dh - beta_G, dtype=np.float64)

    master = np.random.default_rng(config.seed)

    # Catalogue mode: one frozen shared catalogue for the whole run (production
    # has ONE GLADE+ table), unless resample_catalogue_per_realization is set.
    catalogue: SyntheticCatalogue | None = None
    if config.catalogue_mode and not config.resample_catalogue_per_realization:
        catalogue = _build_catalogue(
            config, h_grid, np.random.default_rng(config.seed + 1), survival_table
        )

    results: dict[str, Any] = {}
    levels = {"50": 0.50, "68": 0.68, "90": 0.90}
    for h_true in config.injected_truths:
        cov = {name: 0 for name in levels}
        rail = 0
        maps: list[float] = []
        completion_fractions: list[float] = []
        host_tilts: list[float] = []
        comp_tilts: list[float] = []
        cat_diag: list[dict[str, float]] = []
        cov2: dict[str, int] = {name: 0 for name in levels}
        rail2 = 0
        maps2: list[float] = []
        it_true = int(np.argmin(np.abs(h_grid - h_true)))
        for _ in range(config.n_realizations):
            rng = np.random.default_rng(int(master.integers(1 << 62)))
            if config.catalogue_mode:
                assert beta_G is not None and beta_Gbar is not None  # guarded above
                cat = (
                    _build_catalogue(config, h_grid, rng, survival_table)
                    if config.resample_catalogue_per_realization
                    else catalogue
                )
                assert cat is not None
                if config.mass_channel:
                    assert survival_table is not None
                    logL, logL_2d, diag = _run_realization_catalogue_mass(
                        h_true,
                        h_grid,
                        log_Dh,
                        config,
                        rng,
                        cat,
                        beta_G,
                        beta_Gbar,
                        survival_table,
                    )
                    post2 = np.exp(logL_2d - logL_2d.max())
                    post2 /= np.trapezoid(post2, h_grid)
                    mi2 = int(np.argmax(post2))
                    maps2.append(float(h_grid[mi2]))
                    if mi2 == 0 or mi2 == h_grid.size - 1:
                        rail2 += 1
                    for name, lv in levels.items():
                        if _hpd_contains(h_grid, post2, h_true, lv):
                            cov2[name] += 1
                else:
                    logL, diag = _run_realization_catalogue(
                        h_true, h_grid, log_Dh, config, rng, cat, beta_G, beta_Gbar
                    )
                cat_diag.append(diag)
                completion_fractions.append(diag["completion_fraction"])
            else:
                logL, n_zero_host, logL_host, logL_completion, n_host, n_comp = _run_realization(
                    h_true, h_grid, log_Dh, config, rng, beta_G=beta_G, beta_Gbar=beta_Gbar
                )
                completion_fractions.append(n_zero_host / config.n_events)
                if n_host > 0:
                    host_tilts.append(float(np.gradient(logL_host, h_grid)[it_true]))
                if n_comp > 0:
                    comp_tilts.append(float(np.gradient(logL_completion, h_grid)[it_true]))
            post = np.exp(logL - logL.max())
            post /= np.trapezoid(post, h_grid)
            mi = int(np.argmax(post))
            maps.append(float(h_grid[mi]))
            if mi == 0 or mi == h_grid.size - 1:
                rail += 1
            for name, lv in levels.items():
                if _hpd_contains(h_grid, post, h_true, lv):
                    cov[name] += 1
        n = config.n_realizations
        results[f"{h_true:.4f}"] = {
            "h_true": h_true,
            "coverage": {name: cov[name] / n for name in levels},
            "rail_fraction": rail / n,
            "map_mean": float(np.mean(maps)),
            "map_std": float(np.std(maps)),
            "map_median": float(np.median(maps)),
            "map_bias": float(np.mean(maps)) - h_true,
            # Per-realization MAPs, in seed order: the paired-read rule [A2]
            # requires cross-cell comparisons as per-realization deltas on a
            # shared seed stream, which aggregates alone cannot provide.
            "maps": [float(m) for m in maps],
            "completion_fraction": float(np.mean(completion_fractions)),
            # None (JSON null) is the deliberate empty sentinel — NEVER NaN
            # (NaN != NaN would break full-dict equality comparisons).
            "dlogL_dh_host_mean": float(np.mean(host_tilts)) if host_tilts else None,
            "dlogL_dh_completion_mean": (float(np.mean(comp_tilts)) if comp_tilts else None),
        }
        if config.mass_channel:
            # The 2D (with-BH-mass) channel of the SAME realizations — the
            # harness counterpart of production's per-event both-channel
            # diagnostics. The top-level block above stays the 1D channel.
            results[f"{h_true:.4f}"]["mass_channel_2d"] = {
                "coverage": {name: cov2[name] / n for name in levels},
                "rail_fraction": rail2 / n,
                "map_mean": float(np.mean(maps2)),
                "map_std": float(np.std(maps2)),
                "map_median": float(np.median(maps2)),
                "map_bias": float(np.mean(maps2)) - h_true,
                "maps": [float(m) for m in maps2],
            }
        if cat_diag:
            for key in (
                "empty_ball_fraction",
                "mean_ball_size",
                "host_in_ball_fraction",
                "impostor_fraction",
            ):
                results[f"{h_true:.4f}"][key] = float(np.mean([d[key] for d in cat_diag]))
    return {"config": asdict(config), "results": results}


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: run the harness and write a JSON results file.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(
        description="Independent P-P/coverage calibration harness for the "
        "dark-siren H0 estimator (commission d2 provenance)."
    )
    parser.add_argument("--n-realizations", type=int, default=120)
    parser.add_argument("--n-events", type=int, default=250)
    parser.add_argument("--sigma-z", type=float, default=0.035)
    parser.add_argument("--sigma-z-pv", type=float, default=0.0)
    parser.add_argument("--sigma-dl-frac", type=float, default=0.05)
    parser.add_argument("--truths", type=float, nargs="+", default=[0.62, 0.72, 0.84])
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--kernel", choices=["bare", "volume"], default="volume")
    parser.add_argument("--output", type=Path, default=Path("pp_coverage_results.json"))
    parser.add_argument(
        "--z-support",
        type=float,
        default=None,
        help="Catalogue support ceiling: true hosts with z_host >= z_support become "
        "zero-host events using the pure-completion likelihood B_num/D (issue #29 "
        "analog). Default None disables truncation (bit-identical to the "
        "pre-2026-07-10 harness).",
    )
    parser.add_argument(
        "--mixture-mode",
        choices=[
            "two_branch",
            "gray",
            "conditioned",
            "exact",
            "absolute",
            "lcat",
            "generator_marginal",
        ],
        default="two_branch",
        help="Estimator composition under z_support truncation: 'two_branch' "
        "(default; in-catalogue events bare N_i/D, zero-host B_num/D — the "
        "pre-2026-07-11 behaviour), 'gray' (in-catalogue events get the full "
        "Gray et al. 2020 Eqs. 29+32 mixture (beta_G*L_cat_i + B_num)/D with "
        "the per-host D_g_i of Eqs. A.9/A.10; zero-host unchanged), "
        "'conditioned' (membership-conditioned inverse: N_i/beta_G and "
        "B_num/beta_Gbar), 'exact' (in-catalogue events use the "
        "volume-kernel numerator TRUNCATED at z_support — the "
        "membership-truncated exact kernel, no beta_G, no D_g_i; zero-host "
        "events keep B_num/D), or 'absolute' (the absolute-mass marginal: "
        "in-catalogue events get (N_i + B_num_i)/D with NO self-normalization "
        "of N_i, harness analog of "
        "results/lcat_h_dependence_20260725/DERIVATION_ESTIMATOR_REDESIGN.md "
        "Variant 1), 'lcat' / 'generator_marginal' (CATALOGUE MODE ONLY: the "
        "legacy self-normalized Gray-A9 ratio-of-sums, and the "
        "generator-consistent normalization of "
        "results/lcat_h_dependence_20260725/DERIVATION_GENERATOR_CONSISTENT_NORM.md). "
        "Modes other than 'two_branch' require --z-support.",
    )
    parser.add_argument(
        "--catalogue-mode",
        action="store_true",
        help="Use the discrete-catalogue / impostor-ball generative model: a frozen "
        "shared galaxy table plus hard sky-localization caps, so estimators that must "
        "CHOOSE among candidate hosts can be exercised. Requires --z-support and "
        "--mixture-mode in {lcat, absolute, generator_marginal}. See "
        "results/pp_impostor_harness_20260726/DERIVATION_HARNESS_ANALOG.md.",
    )
    parser.add_argument(
        "--n-galaxies",
        type=int,
        default=200_000,
        help="Galaxies in the synthetic catalogue (catalogue mode; default 200000).",
    )
    parser.add_argument(
        "--sky-frac",
        type=float,
        default=1.0e-4,
        help="GW sky-localization cap solid-angle fraction dOmega/(4 pi) (catalogue "
        "mode; default 1e-4). Expected ball occupancy = n_galaxies * sky_frac * "
        "catalogued fraction.",
    )
    parser.add_argument(
        "--resample-catalogue-per-realization",
        action="store_true",
        help="Redraw the galaxy catalogue every realization (independent universes) "
        "instead of freezing one shared table for the run (production-faithful default).",
    )
    parser.add_argument(
        "--n-z-quad",
        type=int,
        default=160,
        help="Per-event redshift quadrature points (config.n_z_quad). Raise for "
        "small-sigma_z runs so the host-z Gaussian kernel is sampled by >=4 "
        "points per sigma_z (e.g. --n-z-quad 480 at sigma_z=0.002).",
    )
    parser.add_argument(
        "--membership-on-observed",
        action="store_true",
        help="Decide catalogue membership on the OBSERVED z_gal (< z_support) "
        "instead of the true z_host (N-2d membership-determination probe).",
    )
    parser.add_argument(
        "--pdet-in-numerator",
        action="store_true",
        help="Latent-detection exact-inverse probe (260711-27m): multiply both "
        "branch numerators (host kernel integral and completion B_num) by "
        "p_det(A(z)/h) — the factor the exact conditional keeps inside when "
        "detection is decided on the latent true z rather than the observed "
        "data (Mandel-Farr-Gair 2019, arXiv:1809.02063, applies only to "
        "data-thresholded detection). Default off is bit-identical.",
    )
    parser.add_argument(
        "--d50-gpc",
        type=float,
        default=D50_GPC,
        help="50%% detection-horizon luminosity distance [Gpc] (N-4 shallow-venue "
        "depth probe). Default 1.85 = commission venue (z_median ~ 0.3). Lower values "
        "model a shallower venue: d50-gpc 0.25 -> z_median ~ 0.046 (seed600 regime).",
    )
    parser.add_argument(
        "--w-pdet-gpc",
        type=float,
        default=W_PDET_GPC,
        help="Detection roll-off width [Gpc] (default 0.30). Scale with --d50-gpc to "
        "keep a comparable fractional horizon sharpness in shallow-venue runs.",
    )
    parser.add_argument(
        "--sigma-model-in-likelihood",
        action="store_true",
        help="σ(dL_obs)-vs-σ(dL_true) noise-model floor probe (260711-hx1): "
        "evaluate the GW-likelihood factor with the z-dependent model/true-distance "
        "width σ_f·A(z)/h (carrying its own 1/σ(z) normalization) instead of the "
        "constant observed-distance σ_f·dL_obs. Applies to the host kernel numerator "
        "(every mixture_mode) and the completion B_num; the p_det selection integrals "
        "(D(h), gray D_g_i) are unchanged. Combined with --pdet-in-numerator it is the "
        "fully-consistent exact conditional for the latent-thresholded model. Default "
        "off is bit-identical.",
    )
    parser.add_argument(
        "--inference-wpop-tilt",
        type=float,
        default=0.0,
        help="N-3 prior-tilt probe gamma: multiplies the INFERENCE-side w_pop "
        "by exp(gamma*z) at every inference call site; the generative truth "
        "draw is untouched. Default 0.0 is bit-identical to the untilted "
        "harness.",
    )
    parser.add_argument(
        "--h-step",
        type=float,
        default=0.004,
        help="H0 grid spacing config.h_step; lower for finer floor-discriminator grids.",
    )
    parser.add_argument(
        "--noise-model",
        choices=["const", "model", "production"],
        default=None,
        help="Three-way GW noise-model cell (Q-0 audit, 2026-08-17). 'const' = "
        "scattered d_L_obs + constant sigma = sigma_f*d_L_obs (the committed "
        "harness convention; carries error sub-terms (a) sigma-at-scattered-obs "
        "AND (b) no width variation across the z-integral). 'model' = the "
        "z-dependent width sigma_f*A(z)/h (neither sub-term). 'production' = NO "
        "measurement scatter (d_L_obs == d_L_true) + constant sigma at the truth "
        "— carries sub-term (b) ONLY, which is what production does. Overrides "
        "--sigma-model-in-likelihood / --no-gw-scatter when given.",
    )
    parser.add_argument(
        "--no-gw-scatter",
        action="store_true",
        help="Remove the GW (d_L, M_z) measurement scatter: d_L_obs == d_L_true "
        "(production has no measurement scatter). The RNG draw is still made and "
        "discarded so paired scatter/no-scatter cells share the random stream.",
    )
    parser.add_argument(
        "--mass-channel",
        action="store_true",
        help="Enable the second (mass) observable: galaxies carry BH masses, "
        "detection becomes S_4D(d_L, M_z), the catalogue leg carries a "
        "per-candidate Gaussian mass overlap and the completion leg the mass "
        "factor g / g_sel recomputed at every h. Requires --catalogue-mode. The "
        "2D-channel coverage/bias block is reported under 'mass_channel_2d'.",
    )
    parser.add_argument("--mass-slope", type=float, default=0.0)
    parser.add_argument(
        "--mass-horizon-index",
        type=float,
        default=0.0,
        help="S_4D horizon index alpha_M: d50(M_z) = d50 (M_z/1e6)^alpha_M. 0.0 "
        "reduces the survival exactly to the mass-blind p_det; a mass-BEARING "
        "cell must set it > 0 (e.g. 0.25).",
    )
    parser.add_argument("--sigma-mz-frac", type=float, default=0.10)
    parser.add_argument("--rho-dl-mz", type=float, default=0.0)
    parser.add_argument("--sigma-m-gal-frac", type=float, default=0.30)
    parser.add_argument("--mass-rate-index", type=float, default=0.0)
    parser.add_argument("--n-hermite", type=int, default=24)
    parser.add_argument(
        "--selection-cell",
        choices=list(SELECTION_CELLS),
        default="off",
        help="Mirror of production's selection_in_completion_numerator: 'off' = "
        "pre-#118 estimator, '1d' = [P2] only (S_bar_phi in the 1D completion "
        "numerator), '2d' = [P1] only (fused g_sel in the 2D completion leg), "
        "'fused' = the landed production pairing. Requires --mass-channel.",
    )
    parser.add_argument("--event-chunk", type=int, default=16)
    args = parser.parse_args(argv)

    sigma_model = args.sigma_model_in_likelihood
    gw_scatter = not args.no_gw_scatter
    if args.noise_model is not None:
        sigma_model = args.noise_model == "model"
        gw_scatter = args.noise_model != "production"

    config = PPCoverageConfig(
        n_realizations=args.n_realizations,
        n_events=args.n_events,
        sigma_z=args.sigma_z,
        sigma_z_pv=args.sigma_z_pv,
        sigma_dl_frac=args.sigma_dl_frac,
        injected_truths=list(args.truths),
        seed=args.seed,
        kernel=args.kernel,
        h_step=args.h_step,
        n_z_quad=args.n_z_quad,
        inference_wpop_tilt=args.inference_wpop_tilt,
        z_support=args.z_support,
        mixture_mode=args.mixture_mode,
        membership_on_observed=args.membership_on_observed,
        pdet_in_numerator=args.pdet_in_numerator,
        sigma_dl_model_in_likelihood=sigma_model,
        d50_gpc=args.d50_gpc,
        w_pdet_gpc=args.w_pdet_gpc,
        catalogue_mode=args.catalogue_mode,
        n_galaxies=args.n_galaxies,
        sky_frac=args.sky_frac,
        resample_catalogue_per_realization=args.resample_catalogue_per_realization,
        gw_measurement_scatter=gw_scatter,
        mass_channel=args.mass_channel,
        mass_slope=args.mass_slope,
        mass_horizon_index=args.mass_horizon_index,
        sigma_mz_frac=args.sigma_mz_frac,
        rho_dl_mz=args.rho_dl_mz,
        sigma_m_gal_frac=args.sigma_m_gal_frac,
        mass_rate_index=args.mass_rate_index,
        n_hermite=args.n_hermite,
        selection_cell=args.selection_cell,
        event_chunk=args.event_chunk,
    )
    out = run_coverage(config)
    args.output.write_text(json.dumps(out, indent=2))
    for key, r in out["results"].items():
        print(
            f"h_true={key} [{config.kernel:6s}] "
            f"cov50={r['coverage']['50']:.2f} cov68={r['coverage']['68']:.2f} "
            f"cov90={r['coverage']['90']:.2f} rail={r['rail_fraction']:.2f} "
            f"MAP={r['map_mean']:.4f} bias={r['map_bias']:+.4f} "
            f"completion_fraction={r['completion_fraction']:.2f}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
