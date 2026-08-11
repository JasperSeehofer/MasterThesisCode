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

Units: ``h`` in [100 km/s/Mpc]; distances in Gpc. Cosmology: flat LambdaCDM.
"""

import argparse
import json
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
    dL_obs = np.clip(dL_host + rng.normal(0.0, config.sigma_dl_frac * dL_host), 1e-3, None)
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
) -> SyntheticCatalogue:
    """Build one frozen synthetic galaxy catalogue and its selection precomputes.

    Args:
        config: Harness configuration (``catalogue_mode`` semantics).
        h_grid: H0 evaluation grid.
        rng: Random generator for the catalogue draw.

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

    cos_theta_c = 1.0 - 2.0 * config.sky_frac
    chord_radius = float(2.0 * np.sin(0.5 * np.arccos(np.clip(cos_theta_c, -1.0, 1.0))))
    tree = cKDTree(direction[cat_index])

    # Global (all-sky) catalogue selection precomputes, on D(h)'s own node
    # convention so that at z_support >= Z_MAX_POP the limiting identities hold.
    zint = np.linspace(Z_MIN, Z_MAX_POP, 3000)
    wint = np.gradient(zint)
    khat = _smeared_catalogue_density(z_obs, inv_norm, sigma_z, zint)
    rho_cat = _inference_population_weight(zint, config.inference_wpop_tilt) * khat
    w_cat = float(np.sum(wint * rho_cat))
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
    dL_obs = np.clip(dL_host + rng.normal(0.0, config.sigma_dl_frac * dL_host), 1e-3, None)
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
    # Selection denominator D(h) = int p_det(A(z)/h) w_pop(z) dz (shared).
    zint = np.linspace(Z_MIN, Z_MAX_POP, 3000)
    wpop = _inference_population_weight(zint, config.inference_wpop_tilt)
    Dh = np.trapezoid(
        detection_probability(
            comoving_amplitude_of_z(zint)[:, None] / h_grid[None, :],
            config.d50_gpc,
            config.w_pdet_gpc,
        )
        * wpop[:, None],
        zint,
        axis=0,
    )
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
                detection_probability(
                    comoving_amplitude_of_z(zbg)[:, None] / h_grid[None, :],
                    config.d50_gpc,
                    config.w_pdet_gpc,
                )
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
        catalogue = _build_catalogue(config, h_grid, np.random.default_rng(config.seed + 1))

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
        it_true = int(np.argmin(np.abs(h_grid - h_true)))
        for _ in range(config.n_realizations):
            rng = np.random.default_rng(int(master.integers(1 << 62)))
            if config.catalogue_mode:
                assert beta_G is not None and beta_Gbar is not None  # guarded above
                cat = (
                    _build_catalogue(config, h_grid, rng)
                    if config.resample_catalogue_per_realization
                    else catalogue
                )
                assert cat is not None
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
            "completion_fraction": float(np.mean(completion_fractions)),
            # None (JSON null) is the deliberate empty sentinel — NEVER NaN
            # (NaN != NaN would break full-dict equality comparisons).
            "dlogL_dh_host_mean": float(np.mean(host_tilts)) if host_tilts else None,
            "dlogL_dh_completion_mean": (float(np.mean(comp_tilts)) if comp_tilts else None),
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
    args = parser.parse_args(argv)

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
        sigma_dl_model_in_likelihood=args.sigma_model_in_likelihood,
        d50_gpc=args.d50_gpc,
        w_pdet_gpc=args.w_pdet_gpc,
        catalogue_mode=args.catalogue_mode,
        n_galaxies=args.n_galaxies,
        sky_frac=args.sky_frac,
        resample_catalogue_per_realization=args.resample_catalogue_per_realization,
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
