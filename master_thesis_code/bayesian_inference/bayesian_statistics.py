"""Hubble constant posterior evaluation.

:class:`BayesianStatistics` loads saved Cramér-Rao bounds and orchestrates the
full Hubble-constant posterior evaluation using the real GLADE galaxy catalog,
simulation-based :class:`~master_thesis_code.bayesian_inference.simulation_detection_probability.SimulationDetectionProbability`,
full Fisher-matrix covariance, and multiprocessing.

Invoked via ``main.py:evaluate()`` / ``--evaluate`` CLI flag.
Output is written to ``simulations/posteriors/`` as JSON.
"""

import csv
import json
import logging
import math
import multiprocessing as mp
import os
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.integrate import dblquad, fixed_quad, quad
from scipy.stats import multivariate_normal, norm

from master_thesis_code.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)
from master_thesis_code.constants import (
    CRAMER_RAO_BOUNDS_OUTPUT_PATH,
    INJECTION_DATA_DIR,
    PREPARED_CRAMER_RAO_BOUNDS_PATH,
    SNR_THRESHOLD,
    H,
)
from master_thesis_code.cosmological_model import LamCDMScenario, Model1CrossCheck
from master_thesis_code.datamodels.detection import (
    Detection,
    _sky_localization_uncertainty,
)
from master_thesis_code.emri_rate import R_eff_per_mbh
from master_thesis_code.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    HostGalaxy,
    InternalCatalogColumns,
)
from master_thesis_code.galaxy_catalogue.pixel_completeness import (
    CompletenessModel,
    from_cache_or_build,
)
from master_thesis_code.physical_relations import (
    comoving_volume_element,
    dist,
    dist_to_redshift,
    dist_vectorized,
    get_redshift_outer_bounds,
)

_LOGGER = logging.getLogger()

DEFAULT_GALAXY_Z_ERROR = 0.0015
GALAXY_LIKELIHOODS = "galaxy_likelihoods"
ADDITIONAL_GALAXIES_WITHOUT_BH_MASS = "additional_galaxies_without_bh_mass"

FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD = 0.10

# Fixed-quad order for D(h) precomputation
_DH_QUAD_ORDER: int = 100


def weighted_ratio_of_sums(
    numerators: Sequence[float],
    denominators: Sequence[float],
    weights: Sequence[float],
) -> float:
    r"""Weighted in-catalog ratio-of-sums likelihood ``(Σ w·N) / (Σ w·D)``.

    Generalizes the equal-weight Gray et al. (2020) in-catalog term
    ``L_cat = (Σ_g N_g) / (Σ_g D_g)`` (Eq. A.9/A.10) by weighting each candidate
    host galaxy ``g`` by an astrophysical rate prior ``w(g)``:

    .. math::

        L_\mathrm{cat} = \frac{\sum_g w(g)\,N_g}{\sum_g w(g)\,D_g}.

    The weight enters numerator and denominator identically, so

    * any overall rescaling of ``w`` cancels (SCALING INVARIANCE), and
    * constant weights reproduce the plain ratio of sums exactly (the
      equal-weight Change-2 limit).

    This is the inference-side counterpart of the rate-weighted host draw
    :meth:`~master_thesis_code.galaxy_catalogue.handler.GalaxyCatalogueHandler.draw_rate_weighted_hosts`.

    Args:
        numerators: Per-host likelihood numerators ``N_g`` (host-aligned).
        denominators: Per-host selection denominators ``D_g`` (host-aligned,
            same order as ``numerators``).
        weights: Per-host rate weights ``w(g)`` (host-aligned, same order as
            ``numerators`` / ``denominators``).

    Returns:
        The weighted ratio of sums, or ``0.0`` when the weighted denominator
        ``Σ w·D`` is non-positive (matching the unweighted guard).

    References:
        Gray et al. (2020), arXiv:1908.06050, Eqs. (A.9)/(A.10) — in-catalog
            ratio-of-sums likelihood, here weighted by a galaxy rate prior.
    """
    # Σ w·N / Σ w·D — the weight cancels overall normalization (incl. C_NORM),
    # leaving only the relative galaxy weighting (Gray et al. 2020, arXiv:1908.06050).
    w = np.asarray(weights, dtype=np.float64)
    num = np.asarray(numerators, dtype=np.float64)
    den = np.asarray(denominators, dtype=np.float64)
    weighted_den_sum = float(np.sum(w * den))
    if weighted_den_sum <= 0.0:
        return 0.0
    weighted_num_sum = float(np.sum(w * num))
    return weighted_num_sum / weighted_den_sum


def weighted_sum(values: Sequence[float], weights: Sequence[float]) -> float:
    r"""Rate-weighted sum ``Σ_g w_g · v_g`` (the in-catalogue numerator building block).

    The partition-norm in-catalogue likelihood is
    ``L_cat = (Σ_local w_g N_g) / (Σ_global w_g D_g)`` (Gray et al. 2020,
    arXiv:1908.06050, Eqs. A.10 / 29) where the GW-likelihood numerator sum runs
    over the local candidate ball but the SELECTION denominator runs over the full
    catalogue (:func:`precompute_global_catalog_selection`). This helper returns
    the weighted sum of either; an empty input yields ``0.0``.

    Args:
        values: Per-host values ``v_g`` (host-aligned).
        weights: Per-host rate weights ``w_g`` (same order as *values*).

    Returns:
        ``Σ_g w_g · v_g`` (``0.0`` for empty inputs).
    """
    if len(values) == 0:
        return 0.0
    return float(
        np.sum(np.asarray(weights, dtype=np.float64) * np.asarray(values, dtype=np.float64))
    )


def _rate_weight(host: HostGalaxy) -> float:
    r"""Per-MBH EMRI-rate host weight ``w(g) = R_eff_per_mbh(M_g) / (1 + z_g)``.

    IDENTICAL to the weight used by the rate-weighted simulation host draw
    (:meth:`~master_thesis_code.galaxy_catalogue.handler.GalaxyCatalogueHandler.draw_rate_weighted_hosts`),
    closing the generative loop. ``host.M`` is the SOURCE-FRAME catalog BH mass
    (the detector-frame lift ``M_z = M·(1+z)`` is applied only inside
    :func:`single_host_likelihood`, never to ``host.M``), so this evaluates
    ``R_eff`` at the same mass the draw uses.

    Args:
        host: Candidate host galaxy (carries source-frame ``M`` and redshift ``z``).

    Returns:
        The scalar per-MBH rate weight ``R_eff_per_mbh(host.M) / (1 + host.z)``.

    References:
        Babak et al. (2017), arXiv:1703.09722 — effective per-MBH EMRI rate
            (:func:`master_thesis_code.emri_rate.R_eff_per_mbh`).
        Gray et al. (2020), arXiv:1908.06050 — galaxy weighting of the in-catalog
            dark-siren likelihood.
    """
    # host.M is SOURCE-FRAME (see handler NOTE on the redshifted-mass convention).
    return float(R_eff_per_mbh(host.M)) / (1.0 + host.z)


def precompute_completion_denominator(
    h_values: list[float],
    detection_probability_obj: SimulationDetectionProbability,
    Omega_m: float,
    Omega_DE: float,
    *,
    quad_n: int = _DH_QUAD_ORDER,
) -> dict[float, float]:
    """Precompute the completion-term denominator D(h) for each h value.

    Gray et al. (2020), arXiv:1908.06050, Eqs. 33 / A.19: the out-of-catalogue
    selection denominator integrates the detection probability against the EMRI
    population prior over the detectable volume.

    .. math::

        D(h) = \\int_{z_{\\min}}^{z_{\\max}(h)} P_{\\det}(d_L(z,h))
               \\,\\frac{1}{1+z}\\,\\frac{dV_c}{dz\\,d\\Omega}\\, dz

    where ``z_max(h)`` is the redshift corresponding to the P_det grid's
    maximum ``d_L`` at the given h, and ``1/(1+z)`` is the source-to-detector
    time dilation (matching ``comp_num`` and the event sampler
    :func:`master_thesis_code.emri_rate.p_pop_unnormalized`).

    Role in the partition-norm likelihood:
        ``D(h)`` is the FULL-volume selection normalisation
        ``D(h) = beta_G(h) + beta_Gbar(h)`` -- the denominator of the single
        per-event ratio ``p_i = (beta_G L_cat + B_num) / D(h)`` (:meth:`p_Di`).
        It carries **no** ``(1-f)`` factor: the incompleteness lives in its
        missing-volume partner
        :func:`precompute_missing_completion_denominator`
        (``beta_Gbar = INTEGRAL (1-f) P_det dVc/(1+z)``), and the in-catalogue
        share is recovered by ``beta_G = D(h) - beta_Gbar``. The selection-weighted
        catalog membership weight ``w_G = beta_G/D(h) = beta_G/(beta_G+beta_Gbar)``
        (Gray Eq. 29) is now computed EXACTLY -- it replaced the earlier scalar
        narrow-window approximation ``completeness(z_det)``.

    Modeling assumption (still in force): **constant comoving number density**
        for the missing galaxies -- the galaxy number density ``n_gal(z)`` and the
        mass-integrated rate ``INTEGRAL dM R_EMRI(z,M)`` are taken z-independent
        (the latter exact under the ``p0=1`` surrogate), so they are overall
        constants that **cancel** between the discrete catalogue sums and the
        continuous integrals (Option A; see
        :func:`precompute_global_catalog_selection`). Departures (clustering,
        rate/MF evolution) are second order.

    Args:
        h_values: List of Hubble parameter values to evaluate.
        detection_probability_obj: SimulationDetectionProbability instance
            (must have ``get_dl_max`` and
            ``detection_probability_without_bh_mass_interpolated_zero_fill``).
        Omega_m: Matter density parameter.
        Omega_DE: Dark energy density parameter.
        quad_n: Gauss-Legendre quadrature order (default 100).

    Returns:
        Dict mapping h -> D(h) in units of Mpc^3/sr.
    """
    D_h_table: dict[float, float] = {}

    for h in h_values:
        dl_max = detection_probability_obj.get_dl_max(h)
        z_max = dist_to_redshift(dl_max, h=h)
        z_min = 1e-6

        def _denom_integrand(
            z: npt.NDArray[np.float64],
            _h: float = h,
        ) -> npt.NDArray[np.float64]:
            d_L: npt.NDArray[np.float64] = np.asarray(
                dist_vectorized(z, h=_h), dtype=np.float64
            )  # Gpc
            phi = np.zeros_like(z)  # marginalized; value does not matter
            theta = np.zeros_like(z)
            p_det = detection_probability_obj.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_L, phi, theta, h=_h
            )
            dVc: npt.NDArray[np.float64] = np.atleast_1d(
                np.asarray(comoving_volume_element(z, h=_h), dtype=np.float64)
            )
            # Population prior R_EMRI(z,M)/(1+z) * dVc/dz (emri_rate.p_pop_unnormalized):
            # the 1/(1+z) is the source->detector time dilation. The mass-integrated
            # rate INTEGRAL dM R_EMRI(z,M) is z-independent under the p0=1 surrogate, so it
            # is an overall constant that cancels in L_comp = comp_num/D(h); only 1/(1+z)
            # survives here. Babak et al. (2017), arXiv:1703.09722 (rate); Mandel-Farr-Gair
            # (2019), arXiv:1809.02063 (detector-frame rate density).
            return np.asarray(p_det, dtype=np.float64) * dVc / (1.0 + z)

        D_h: float = fixed_quad(_denom_integrand, z_min, z_max, n=quad_n)[0]
        D_h_table[h] = D_h
        _LOGGER.info(
            "D(h=%.4f) = %.6e  [z_max=%.4f, dl_max=%.4f Gpc]",
            h,
            D_h,
            z_max,
            dl_max,
        )

    # --- Red flag checks ---
    D_values = list(D_h_table.values())
    if any(d <= 0 for d in D_values):
        _LOGGER.warning(
            "D(h) <= 0 for some h values: %s",
            {h: d for h, d in D_h_table.items() if d <= 0},
        )
    if len(D_values) > 1:
        ratio = max(D_values) / max(min(D_values), 1e-300)
        if ratio > 10:
            _LOGGER.warning("D(h) varies by %.1fx across h grid (max/min)", ratio)
        if max(D_values) - min(D_values) < 1e-10 * max(D_values):
            _LOGGER.warning("D(h) is nearly identical for all h — h-dependence may not be captured")

    return D_h_table


def precompute_missing_completion_denominator(
    h_values: list[float],
    detection_probability_obj: SimulationDetectionProbability,
    completeness: CompletenessModel,
    *,
    quad_n: int = _DH_QUAD_ORDER,
) -> dict[float, float]:
    r"""Precompute the missing-volume selection integral ``beta_Gbar(h)``.

    The ``(1-f(z))`` companion of :func:`precompute_completion_denominator`
    (which returns the **unchanged** full-volume ``D(h) = beta_G + beta_Gbar``).
    Gray et al. (2020), arXiv:1908.06050, Eq. (33): the out-of-catalogue
    selection integral weights the full detection denominator by the
    *incompleteness* ``1 - f(z)``, i.e. it integrates only over the galaxies the
    catalogue is missing:

    .. math::

        \beta_{\bar G}(h) = \int_{z_{\min}}^{z_{\max}(h)} \bigl(1 - f(z)\bigr)\,
            P_{\det}(d_L(z,h))\,\frac{1}{1+z}\,\frac{dV_c}{dz}\, dz .

    The in-catalogue selection normalisation is then
    ``beta_G(h) = D(h) - beta_Gbar(h) = INTEGRAL f(z) P_det (1/(1+z)) dVc``.
    ``f(z) = completeness.get_completeness_at_redshift(z, h)`` is the SAME
    completeness call the generator uses
    (:func:`master_thesis_code.dark_siren_injection.compute_global_catalog_fraction`
    and ``_draw_dark_redshifts``), so the inference completion population and the
    injected dark population are bit-for-bit identical.

    Args:
        h_values: Hubble parameter values to evaluate.
        detection_probability_obj: Same object passed to
            :func:`precompute_completion_denominator` (provides ``get_dl_max``
            and ``detection_probability_without_bh_mass_interpolated_zero_fill``).
        completeness: Catalogue completeness ``f(z)`` (Gray Eq. 9). Evaluated
            sky-marginalised, identically to the generator.
        quad_n: Gauss-Legendre quadrature order (default
            :data:`_DH_QUAD_ORDER`), matching ``D(h)``.

    Returns:
        Dict mapping ``h -> beta_Gbar(h)`` in units of Mpc^3/sr (same as
        ``D(h)``).

    References:
        Gray et al. (2020), arXiv:1908.06050, Eq. (33) — out-of-catalogue
            selection denominator (here the missing ``(1-f)`` fraction).
    """
    beta_Gbar_table: dict[float, float] = {}

    for h in h_values:
        dl_max = detection_probability_obj.get_dl_max(h)
        z_max = dist_to_redshift(dl_max, h=h)
        z_min = 1e-6

        def _missing_denom_integrand(
            z: npt.NDArray[np.float64],
            _h: float = h,
        ) -> npt.NDArray[np.float64]:
            d_L: npt.NDArray[np.float64] = np.asarray(
                dist_vectorized(z, h=_h), dtype=np.float64
            )  # Gpc
            phi = np.zeros_like(z)  # sky-marginalized; matches D(h)
            theta = np.zeros_like(z)
            p_det = detection_probability_obj.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_L, phi, theta, h=_h
            )
            dVc: npt.NDArray[np.float64] = np.atleast_1d(
                np.asarray(comoving_volume_element(z, h=_h), dtype=np.float64)
            )
            # Incompleteness weight (1 - f_bar(z,h)). Eq. (33) in Gray et al.
            # (2020), arXiv:1908.06050, with the SKY-AVERAGED completeness
            # f_bar = (1/Npix) sum_k f_k (Gray-Messenger-Veitch 2022 Eq. 3, Change
            # 5.2): beta_Gbar is the SKY-INTEGRATED missing-volume selection, so the
            # sky-uniform p_det pulls out and only the Omega-marginal f_bar enters.
            # f_bar is the SAME object the generator's F uses, closing the
            # sim/inference loop. The 1/(1+z) and dVc match D(h) so beta_G =
            # D - beta_Gbar. (Valid because p_det is sky-uniform; if real LISA sky
            # dependence is restored this must become sum_k f_k p_det(z,Omega_k).)
            f_z = np.clip(
                np.asarray(completeness.f_bar(z, _h), dtype=np.float64),
                0.0,
                1.0,
            )
            return (1.0 - f_z) * np.asarray(p_det, dtype=np.float64) * dVc / (1.0 + z)

        beta_Gbar: float = fixed_quad(_missing_denom_integrand, z_min, z_max, n=quad_n)[0]
        beta_Gbar_table[h] = beta_Gbar
        _LOGGER.info(
            "beta_Gbar(h=%.4f) = %.6e  [z_max=%.4f]",
            h,
            beta_Gbar,
            z_max,
        )

    return beta_Gbar_table


def precompute_global_catalog_selection(
    h_values: list[float],
    galaxy_catalog: GalaxyCatalogueHandler,
    detection_probability_obj: SimulationDetectionProbability,
    *,
    with_bh_mass: bool,
) -> dict[float, float]:
    r"""Precompute the GLOBAL in-catalogue selection denominator (Option A).

    The partition-norm restructure forms the in-catalogue likelihood as
    ``L_cat = (sum_local w_g N_g) / (sum_global w_g D_g)`` where the SELECTION
    denominator runs over the FULL catalogue out to the detection horizon
    ``z_max(h)``, NOT the per-event candidate ball. Globalising the denominator
    makes ``L_cat`` scale-free, so the per-galaxy <-> per-volume number-density
    factor ``n_gal`` cancels against the continuous
    ``beta_G(h) = D(h) - beta_Gbar(h)`` and no calibration constant is needed
    (Gray et al. 2020, arXiv:1908.06050, Eq. 29: the discrete catalogue sum is
    the Monte-Carlo realisation of ``beta_G = INTEGRAL f P_det dVc/(1+z)``).

    .. math::

        \Sigma_{\mathrm{global}}(h) = \sum_{g:\, z_g < z_{\max}(h)}
            w_g\, P_{\det}\bigl(d_L(z_g, h)\bigr),
        \qquad w_g = \frac{R_\mathrm{eff}(M_g)}{1 + z_g}.

    The weight ``w_g`` is IDENTICAL to the rate-weighted host draw
    (:meth:`~master_thesis_code.galaxy_catalogue.handler.GalaxyCatalogueHandler.draw_rate_weighted_hosts`)
    and the in-catalogue likelihood weight (:func:`_rate_weight`). ``P_det`` is
    evaluated SKY-MARGINALISED (``phi = theta = 0``), on the same footing as the
    completion ``D(h)`` / ``beta_Gbar`` (the per-galaxy sky dependence is
    deferred to the pixelated-completeness change). ``D_g ~= P_det(z_g)`` uses
    the narrow galaxy-redshift-PDF limit. The sum is event-INDEPENDENT, so it is
    precomputed once per ``h`` like ``D(h)``.

    Args:
        h_values: Hubble parameter values to evaluate.
        galaxy_catalog: Loaded catalogue handler (its ``reduced_galaxy_catalog``
            is summed over; same rows the rate-weighted draw uses).
        detection_probability_obj: Detection probability (provides ``get_dl_max``
            and the 3D / 4D ``P_det`` accessors).
        with_bh_mass: ``False`` uses the 3D (sky+distance) ``P_det`` (the
            without-BH-mass channel); ``True`` uses the 4D
            (sky+distance+observer-frame mass ``M_z = M_g(1+z_g)``) ``P_det``,
            the global companion of the with-BH-mass catalogue sum.

    Returns:
        Dict mapping ``h -> sum_global w_g D_g(h)`` (dimensionless rate-weighted
        detection count).

    References:
        Gray et al. (2020), arXiv:1908.06050, Eq. (29) — ``beta_G`` selection
            integral (here its discrete catalogue realisation).
        Babak et al. (2017), arXiv:1703.09722 — per-MBH rate ``R_eff``
            (:func:`master_thesis_code.emri_rate.R_eff_per_mbh`).
    """
    catalog = galaxy_catalog.reduced_galaxy_catalog
    z_all = np.asarray(
        catalog[InternalCatalogColumns.REDSHIFT].to_numpy(dtype=np.float64),
        dtype=np.float64,
    )
    M_all = np.asarray(
        catalog[InternalCatalogColumns.BH_MASS].to_numpy(dtype=np.float64),
        dtype=np.float64,
    )

    global_table: dict[float, float] = {}
    for h in h_values:
        z_max = dist_to_redshift(detection_probability_obj.get_dl_max(h), h=h)
        # Eligible galaxies: inside the detectable volume (z < z_max(h)) with a
        # finite source-frame mass. Galaxies beyond z_max(h) have P_det ~= 0 and
        # do not contribute to the selection normalisation.
        eligible = (z_all < z_max) & np.isfinite(M_all) & (M_all > 0.0)
        z_g = z_all[eligible]
        M_g = M_all[eligible]
        if z_g.size == 0:
            global_table[h] = 0.0
            _LOGGER.warning(
                "Global catalog selection (with_bh=%s): no eligible galaxy z<%.4f.",
                with_bh_mass,
                z_max,
            )
            continue

        # w_g = R_eff_per_mbh(M_g)/(1+z_g): the EXACT rate weight the draw and the
        # in-catalogue likelihood use (Babak et al. 2017; Gray et al. 2020).
        w_g = np.asarray(R_eff_per_mbh(M_g), dtype=np.float64) / (1.0 + z_g)
        d_L_g = np.asarray(dist_vectorized(z_g, h=h), dtype=np.float64)  # Gpc
        phi = np.zeros_like(z_g)  # sky-marginalized, matching D(h)
        theta = np.zeros_like(z_g)
        if with_bh_mass:
            M_z_g = M_g * (1.0 + z_g)  # observer-frame mass (P_det grid axis)
            p_det = np.asarray(
                detection_probability_obj.detection_probability_with_bh_mass_interpolated(
                    d_L_g, M_z_g, phi, theta, h=h
                ),
                dtype=np.float64,
            )
        else:
            p_det = np.asarray(
                detection_probability_obj.detection_probability_without_bh_mass_interpolated_zero_fill(
                    d_L_g, phi, theta, h=h
                ),
                dtype=np.float64,
            )
        global_table[h] = float(np.sum(w_g * p_det))
        _LOGGER.info(
            "Global catalog selection (with_bh=%s) sum_w_Dg(h=%.4f) = %.6e  "
            "[%d eligible galaxies, z_max=%.4f]",
            with_bh_mass,
            h,
            global_table[h],
            z_g.size,
            z_max,
        )

    return global_table


# Module-level globals used by child_process_init for multiprocessing worker state
redshift_upper_integration_limit: float = 0.0
redshift_lower_integration_limit: float = 0.0
bh_mass_upper_integration_limit: float = 0.0
bh_mass_lower_integration_limit: float = 0.0
detection_probability: Any = None
# Gray et al. (2020), arXiv:1908.06050, Eq. A.19:
# Precomputed completion-term denominator D(h) for each h in the evaluation grid
D_h_table: dict[float, float] = {}
# Legacy global kept for single_host_likelihood_integration_testing() and
# single_host_likelihood_grid() — not used by the optimized production path.
detection_likelihood_gaussians_by_detection_index: Any = None

# Pre-computed Gaussian arrays (replace frozen scipy multivariate_normal objects)
means_3d: npt.NDArray[np.float64] = np.empty(0)
cov_inv_3d: npt.NDArray[np.float64] = np.empty(0)
log_norm_3d: npt.NDArray[np.float64] = np.empty(0)
means_4d: npt.NDArray[np.float64] = np.empty(0)
cov_inv_4d: npt.NDArray[np.float64] = np.empty(0)
log_norm_4d: npt.NDArray[np.float64] = np.empty(0)
det_index_to_slot: dict[int, int] = {}

# Pre-computed conditional distribution parameters for BH mass branch
sigma2_cond_arr: npt.NDArray[np.float64] = np.empty(0)
proj_arr: npt.NDArray[np.float64] = np.empty(0)

# Pre-extracted detection parameters (avoid pickling Detection objects per starmap call)
det_d_L_arr: npt.NDArray[np.float64] = np.empty(0)
det_d_L_unc_arr: npt.NDArray[np.float64] = np.empty(0)
det_M_arr: npt.NDArray[np.float64] = np.empty(0)
det_phi_arr: npt.NDArray[np.float64] = np.empty(0)
det_theta_arr: npt.NDArray[np.float64] = np.empty(0)


def _check_covariance_quality(
    cov: npt.NDArray[np.float64],
    threshold: float,
) -> tuple[float, bool]:
    """Check whether a covariance matrix is numerically degenerate.

    Computes the condition number of *cov* and returns whether it exceeds
    *threshold*.  A high condition number indicates near-singularity that
    would make ``np.linalg.pinv`` and ``np.linalg.slogdet`` unreliable.

    Args:
        cov: Square covariance matrix to check.
        threshold: Condition-number threshold above which the matrix is
            considered degenerate.

    Returns:
        A tuple ``(condition_number, should_exclude)`` where
        *condition_number* is ``float(np.linalg.cond(cov))`` and
        *should_exclude* is ``True`` when ``condition_number > threshold``.
    """
    cond = float(np.linalg.cond(cov))
    return cond, cond > threshold


def _mvn_pdf(
    x: npt.NDArray[np.float64],
    mean: npt.NDArray[np.float64],
    cov_inv: npt.NDArray[np.float64],
    log_norm: float,
) -> npt.NDArray[np.float64]:
    """Evaluate multivariate normal PDF using pre-computed inverse and log-normalization.

    Equivalent to ``scipy.stats.multivariate_normal.pdf()`` but avoids repeated
    Cholesky decompositions by using pre-computed Sigma^{-1} and
    log((2*pi)^{-k/2} * |Sigma|^{-1/2}).

    Args:
        x: Evaluation points, shape ``(N, k)`` or ``(k,)``.
        mean: Mean vector, shape ``(k,)``.
        cov_inv: Inverse covariance matrix, shape ``(k, k)``.
        log_norm: Pre-computed log-normalization constant.

    Returns:
        PDF values, shape ``(N,)``.
    """
    diff = np.atleast_2d(x) - mean  # (N, k)
    maha = np.sum(diff @ cov_inv * diff, axis=-1)  # (N,)
    result: npt.NDArray[np.float64] = np.exp(log_norm - 0.5 * maha)
    return result


class BayesianStatistics:
    """Hubble constant posterior evaluation.

    Loads saved Cramér-Rao bounds from CSV, constructs a simulation-based
    :class:`SimulationDetectionProbability`, builds multivariate-normal GW
    likelihoods from the full Fisher-matrix covariance, and evaluates
    per-detection posteriors over an H₀ grid using a multiprocessing pool.

    Invoked via ``main.py:evaluate()`` (``--evaluate`` CLI flag).
    Output is written to ``simulations/posteriors/`` as JSON.
    """

    cramer_rao_bounds: pd.DataFrame
    detection: Detection
    cosmological_model: LamCDMScenario
    h: float
    Omega_m: float
    Omega_DE: float
    w_0: float
    w_a: float
    h_values: list[float]
    h_values_with_bh_mass: list[float]
    galaxy_weights: dict[str, dict[str, list[float]]]
    additional_galaxies_without_bh_mass: dict[str, dict[str, list[float]]]
    posterior_data: dict[int, list[float]]
    posterior_data_with_bh_mass: dict[int | str, Any]

    def __init__(self) -> None:
        self.h_values = []
        self.h_values_with_bh_mass = []
        self.galaxy_weights = {}
        self.additional_galaxies_without_bh_mass = {}
        self.posterior_data = {}
        self.posterior_data_with_bh_mass = {}
        self.cramer_rao_bounds = pd.read_csv(PREPARED_CRAMER_RAO_BOUNDS_PATH)
        self.true_cramer_rao_bounds = pd.read_csv(CRAMER_RAO_BOUNDS_OUTPUT_PATH)
        _LOGGER.info(f"Loaded {len(self.cramer_rao_bounds)} detections...")
        self.cosmological_model = LamCDMScenario()
        self.h = self.cosmological_model.h.fiducial_value
        self.Omega_m = self.cosmological_model.Omega_m.fiducial_value
        self.Omega_DE = 1 - self.Omega_m
        self.w_0 = self.cosmological_model.w_0
        self.w_a = self.cosmological_model.w_a
        self.catalog_only: bool = False
        self._diagnostic_rows: list[dict[str, object]] = []

    def evaluate(
        self,
        galaxy_catalog: GalaxyCatalogueHandler,
        cosmological_model: Model1CrossCheck,
        h_value: float,
        num_workers: int | None = None,
        catalog_only: bool = False,
        pdet_dl_bins: int = 60,
        pdet_mass_bins: int = 40,
        pdet_estimator: str = "local_linear",
        fisher_cond_threshold: float = 1e16,
    ) -> None:
        self.catalog_only = catalog_only
        self._diagnostic_rows = []
        if catalog_only:
            _LOGGER.info("catalog_only mode: f_i=1, L_comp=0 (skipping completion integral)")
        _LOGGER.info(f"Computing posteriors for h = {h_value}...")
        if (h_value < self.cosmological_model.h.lower_limit) or (
            h_value > self.cosmological_model.h.upper_limit
        ):
            raise ValueError("Hubble constant out of bounds.")

        _LOGGER.debug(f"Loaded {len(self.cramer_rao_bounds)} detections...")
        # Filter detections: SNR threshold + relative d_L error
        n_before = len(self.cramer_rao_bounds)
        self.cramer_rao_bounds = self.cramer_rao_bounds[
            self.cramer_rao_bounds["SNR"] >= SNR_THRESHOLD
        ]
        _LOGGER.info(
            f"SNR filter (>= {SNR_THRESHOLD}): {n_before} -> {len(self.cramer_rao_bounds)} detections"
        )
        for index, detection in self.cramer_rao_bounds.iterrows():
            detection = Detection(detection)
            if use_detection(detection) is False:
                self.cramer_rao_bounds.drop(index, inplace=True)
        _LOGGER.info(
            f"After quality filtering: {len(self.cramer_rao_bounds)} detections "
            f"(d_L relative error < {FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD})"
        )
        # parameter limitations
        REDSHIFT_LOWER_LIMIT = 0.0
        REDSHIFT_UPPER_LIMIT = cosmological_model.max_redshift
        BH_MASS_LOWER_LIMIT = cosmological_model.parameter_space.M.lower_limit
        BH_MASS_UPPER_LIMIT = cosmological_model.parameter_space.M.upper_limit

        _LOGGER.debug("Creating detection probability functions...")
        detection_probability = SimulationDetectionProbability(
            injection_data_dir=INJECTION_DATA_DIR,
            snr_threshold=SNR_THRESHOLD,
            dl_bins=pdet_dl_bins,
            mass_bins=pdet_mass_bins,
            estimator=pdet_estimator,  # type: ignore[arg-type]
        )
        _LOGGER.debug("Detection probability functions created.")

        # Pre-warm P_det grid cache for target h -- avoids N workers each building
        # the same grid independently after pool spawn
        detection_probability._get_or_build_grid(h_value)
        _LOGGER.debug("P_det grid pre-warmed for h=%.4f.", h_value)

        # Validate P_det grid coverage for observed events
        detection_probability.validate_coverage(h_value, self.cramer_rao_bounds)

        # Gray et al. (2020), arXiv:1908.06050, Eq. A.19:
        # Precompute completion-term denominator D(h) over full detectable volume.
        # D(h) is event-independent; compute once per h-value.
        _D_h_table = precompute_completion_denominator(
            h_values=[h_value],
            detection_probability_obj=detection_probability,
            Omega_m=self.Omega_m,
            Omega_DE=self.Omega_DE,
        )
        _LOGGER.info("D(h) precomputed for %d h-value(s).", len(_D_h_table))

        # Gray et al. (2020), arXiv:1908.06050, Eq. 9 + Gray-Messenger-Veitch 2022,
        # arXiv:2111.04629 (Change 5): per-HEALPix-pixel completeness f_k(z,Omega,h),
        # loaded from the SAME frozen cached m_th map the EMRI injection uses (C1
        # consistency; main.py:injection_campaign). f_bar weights beta_Gbar, f_k(event
        # pixel) weights the completion numerator B_num below.
        completeness = from_cache_or_build()

        # Partition-norm precomputes (Option A), consumed by p_Di's single ratio
        # p_i = (beta_G L_cat + B_num)/D(h). beta_Gbar(h) = INTEGRAL (1-f) P_det
        # dVc/(1+z) (Gray et al. 2020, arXiv:1908.06050, Eq. 33);
        # beta_G(h) = D(h) - beta_Gbar(h) (Eq. 29); and the global in-catalogue
        # selection sums sum_global w_g D_g for both channels (Eq. 29 discrete
        # realisation) that make L_cat scale-free so n_gal cancels.
        _beta_Gbar_table = precompute_missing_completion_denominator(
            h_values=[h_value],
            detection_probability_obj=detection_probability,
            completeness=completeness,
        )
        _beta_G_table = {h: _D_h_table[h] - _beta_Gbar_table[h] for h in _D_h_table}
        _global_cat_denom_no_bh = precompute_global_catalog_selection(
            h_values=[h_value],
            galaxy_catalog=galaxy_catalog,
            detection_probability_obj=detection_probability,
            with_bh_mass=False,
        )
        _global_cat_denom_with_bh = precompute_global_catalog_selection(
            h_values=[h_value],
            galaxy_catalog=galaxy_catalog,
            detection_probability_obj=detection_probability,
            with_bh_mass=True,
        )
        _w_G_preview = (
            _beta_G_table[h_value] / _D_h_table[h_value]
            if _D_h_table.get(h_value, 0.0) > 0.0
            else float("nan")
        )
        _LOGGER.info(
            "Partition-norm: w_G=beta_G/D(h)=%.4f, sum_w_Dg(no_bh)=%.4e, sum_w_Dg(with_bh)=%.4e",
            _w_G_preview,
            _global_cat_denom_no_bh.get(h_value, float("nan")),
            _global_cat_denom_with_bh.get(h_value, float("nan")),
        )

        _LOGGER.debug("Pre-computing Gaussian arrays for GW likelihoods...")
        _t0 = time.perf_counter()

        det_indices = list(self.cramer_rao_bounds.index)
        n_det = len(det_indices)
        _det_index_to_slot: dict[int, int] = {
            int(idx): slot for slot, idx in enumerate(det_indices)
        }

        # Pre-allocate arrays for 3D (without BH mass) and 4D (with BH mass) Gaussians
        _means_3d = np.zeros((n_det, 3))
        _cov_inv_3d = np.zeros((n_det, 3, 3))
        _log_norm_3d = np.zeros(n_det)
        _means_4d = np.zeros((n_det, 4))
        _cov_inv_4d = np.zeros((n_det, 4, 4))
        _log_norm_4d = np.zeros(n_det)

        # Conditional distribution pre-computations for BH mass branch
        _sigma2_cond_arr = np.zeros(n_det)
        _proj_arr = np.zeros((n_det, 3))

        # Fisher quality: condition numbers and exclusion mask
        _excluded_mask = np.zeros(n_det, dtype=bool)
        _cond_3d = np.zeros(n_det, dtype=np.float64)
        _cond_4d = np.zeros(n_det, dtype=np.float64)
        _eigen_3d: dict[int, npt.NDArray[np.float64]] = {}  # flagged slots only
        _eigen_4d: dict[int, npt.NDArray[np.float64]] = {}  # flagged slots only

        # Pre-extracted detection scalar parameters (avoid pickling Detection objects)
        _det_d_L = np.zeros(n_det)
        _det_d_L_unc = np.zeros(n_det)
        _det_M = np.zeros(n_det)
        _det_phi = np.zeros(n_det)
        _det_theta = np.zeros(n_det)

        for index, row in self.cramer_rao_bounds.iterrows():
            det = Detection(row)
            slot = _det_index_to_slot[int(index)]

            # Store detection scalars
            _det_d_L[slot] = det.d_L
            _det_d_L_unc[slot] = det.d_L_uncertainty
            _det_M[slot] = det.M
            _det_phi[slot] = det.phi
            _det_theta[slot] = det.theta

            # Build 3D covariance (without BH mass)
            cov_3d = np.array(
                [
                    [
                        det.phi_error**2,
                        det.theta_phi_covariance,
                        det.d_L_phi_covariance / det.d_L,
                    ],
                    [
                        det.theta_phi_covariance,
                        det.theta_error**2,
                        det.d_L_theta_covariance / det.d_L,
                    ],
                    [
                        det.d_L_phi_covariance / det.d_L,
                        det.d_L_theta_covariance / det.d_L,
                        det.d_L_uncertainty**2 / det.d_L**2,
                    ],
                ]
            )

            # Build 4D covariance (with BH mass)
            cov_4d = np.array(
                [
                    [
                        det.phi_error**2,
                        det.theta_phi_covariance,
                        det.d_L_phi_covariance / det.d_L,
                        det.M_phi_covariance / det.M,
                    ],
                    [
                        det.theta_phi_covariance,
                        det.theta_error**2,
                        det.d_L_theta_covariance / det.d_L,
                        det.M_theta_covariance / det.M,
                    ],
                    [
                        det.d_L_phi_covariance / det.d_L,
                        det.d_L_theta_covariance / det.d_L,
                        det.d_L_uncertainty**2 / det.d_L**2,
                        det.d_L_M_covariance / det.d_L / det.M,
                    ],
                    [
                        det.M_phi_covariance / det.M,
                        det.M_theta_covariance / det.M,
                        det.d_L_M_covariance / det.d_L / det.M,
                        det.M_uncertainty**2 / det.M**2,
                    ],
                ]
            )

            # Compute condition numbers for degeneracy detection (per D-01, D-02)
            cond_3d, exclude_3d = _check_covariance_quality(cov_3d, fisher_cond_threshold)
            cond_4d, exclude_4d = _check_covariance_quality(cov_4d, fisher_cond_threshold)
            _cond_3d[slot] = cond_3d
            _cond_4d[slot] = cond_4d

            if exclude_3d or exclude_4d:
                _excluded_mask[slot] = True
                _eigen_3d[slot] = np.linalg.eigh(cov_3d)[0]
                _eigen_4d[slot] = np.linalg.eigh(cov_4d)[0]
                _LOGGER.warning(
                    "Detection %d excluded: cond_3d=%.2e, cond_4d=%.2e (threshold=%.2e)",
                    int(index),
                    cond_3d,
                    cond_4d,
                    fisher_cond_threshold,
                )
                continue

            # 3D Gaussian: mean, inverse, log-normalization
            _means_3d[slot] = [det.phi, det.theta, 1]
            _cov_inv_3d[slot] = np.linalg.pinv(cov_3d)
            _sign_3d, logdet_3d = np.linalg.slogdet(cov_3d)
            if _sign_3d <= 0:
                _excluded_mask[slot] = True
                _eigen_3d[slot] = np.linalg.eigh(cov_3d)[0]
                _eigen_4d[slot] = np.linalg.eigh(cov_4d)[0]
                _LOGGER.warning(
                    "Detection %d excluded: slogdet sign_3d=%d (non-positive definite)",
                    int(index),
                    _sign_3d,
                )
                continue
            _log_norm_3d[slot] = -0.5 * (3 * np.log(2 * np.pi) + logdet_3d)

            # 4D Gaussian: mean, inverse, log-normalization
            _means_4d[slot] = [det.phi, det.theta, 1, 1]
            _cov_inv_4d[slot] = np.linalg.pinv(cov_4d)
            _sign_4d, logdet_4d = np.linalg.slogdet(cov_4d)
            if _sign_4d <= 0:
                _excluded_mask[slot] = True
                _eigen_3d[slot] = np.linalg.eigh(cov_3d)[0]
                _eigen_4d[slot] = np.linalg.eigh(cov_4d)[0]
                _LOGGER.warning(
                    "Detection %d excluded: slogdet sign_4d=%d (non-positive definite)",
                    int(index),
                    _sign_4d,
                )
                continue
            _log_norm_4d[slot] = -0.5 * (4 * np.log(2 * np.pi) + logdet_4d)

            # Conditional distribution for BH mass branch
            # Bishop (2006) PRML Eq. 2.81-2.82
            cov_obs = cov_4d[:3, :3]  # = cov_3d
            cov_cross = cov_4d[3, :3]
            cov_mz = cov_4d[3, 3]
            cov_obs_inv = _cov_inv_3d[slot]  # reuse already-computed inverse
            _sigma2_cond_arr[slot] = max(float(cov_mz - cov_cross @ cov_obs_inv @ cov_cross), 1e-30)
            _proj_arr[slot] = cov_cross @ cov_obs_inv

        # Log Fisher quality summary (D-11)
        n_flagged = int(_excluded_mask.sum())
        top5_worst = sorted(
            [
                (int(idx), float(_cond_3d[slot]), float(_cond_4d[slot]))
                for idx, slot in _det_index_to_slot.items()
            ],
            key=lambda t: max(t[1], t[2]),
            reverse=True,
        )[:5]
        _LOGGER.info(
            "Fisher quality: %d total, %d flagged/excluded (%.1f%%). Top-5 worst cond: %s",
            n_det,
            n_flagged,
            100 * n_flagged / max(n_det, 1),
            [(idx, f"{c3:.2e}", f"{c4:.2e}") for idx, c3, c4 in top5_worst],
        )

        # Store index mapping on the instance for use in p_Di completion term
        self._det_index_to_slot = _det_index_to_slot
        self._means_3d = _means_3d
        self._cov_inv_3d = _cov_inv_3d
        self._log_norm_3d = _log_norm_3d
        self._det_d_L = _det_d_L
        self._det_d_L_unc = _det_d_L_unc
        self._det_M = _det_M
        self._det_phi = _det_phi
        self._det_theta = _det_theta
        self._D_h_table = _D_h_table
        # Partition-norm precompute tables (Option A) -- stored for the
        # restructure commit; not yet read by p_Di.
        self._beta_Gbar_table = _beta_Gbar_table
        self._beta_G_table = _beta_G_table
        self._global_cat_denom_no_bh = _global_cat_denom_no_bh
        self._global_cat_denom_with_bh = _global_cat_denom_with_bh
        self._excluded_mask = _excluded_mask
        self._cond_3d = _cond_3d
        self._cond_4d = _cond_4d
        self._eigen_3d = _eigen_3d
        self._eigen_4d = _eigen_4d
        self._fisher_cond_threshold = fisher_cond_threshold

        _LOGGER.info(
            "Gaussian precomputation: %.2fs (%d detections)",
            time.perf_counter() - _t0,
            n_det,
        )

        self.h = h_value

        if num_workers is None:
            try:
                available_cpus = len(os.sched_getaffinity(0))
            except AttributeError:
                available_cpus = os.cpu_count() or 1
            num_workers = max(1, available_cpus - 2)
        _LOGGER.debug(f"Using {num_workers} worker(s) for multiprocessing pool.")

        _t0 = time.perf_counter()
        # forkserver with module preloading: the server imports heavy modules
        # once, then workers inherit them via copy-on-write — eliminates 126×
        # Python startup + module import on the shared cluster filesystem.
        # Fallback: if forkserver is unavailable, use spawn (always safe).
        _ctx: mp.context.BaseContext
        if "forkserver" in mp.get_all_start_methods():
            _fs_ctx = mp.get_context("forkserver")
            _fs_ctx.set_forkserver_preload(
                [
                    "numpy",
                    "scipy.interpolate",
                    "scipy.integrate",
                    "scipy.stats",
                    "pandas",
                    "master_thesis_code.bayesian_inference.simulation_detection_probability",
                    "master_thesis_code.physical_relations",
                ]
            )
            _ctx = _fs_ctx
        else:
            _ctx = mp.get_context("spawn")
        _LOGGER.info("Multiprocessing context: %s", _ctx.get_start_method())
        with _ctx.Pool(
            num_workers,
            initializer=child_process_init,
            initargs=(
                REDSHIFT_LOWER_LIMIT,
                REDSHIFT_UPPER_LIMIT,
                BH_MASS_LOWER_LIMIT,
                BH_MASS_UPPER_LIMIT,
                detection_probability,
                _means_3d,
                _cov_inv_3d,
                _log_norm_3d,
                _means_4d,
                _cov_inv_4d,
                _log_norm_4d,
                _det_index_to_slot,
                _sigma2_cond_arr,
                _proj_arr,
                _det_d_L,
                _det_d_L_unc,
                _det_M,
                _det_phi,
                _det_theta,
                _D_h_table,
            ),
        ) as pool:
            _LOGGER.info(
                "Pool spawn (%d workers): %.2fs",
                num_workers,
                time.perf_counter() - _t0,
            )
            self.p_D(
                galaxy_catalog=galaxy_catalog,
                redshift_upper_limit=REDSHIFT_UPPER_LIMIT,
                pool=pool,
                completeness=completeness,
                detection_probability_obj=detection_probability,
            )
        _LOGGER.info(f"posteriors comupted for h = {self.h}")

        if not os.path.isdir("simulations/posteriors"):
            os.makedirs("simulations/posteriors")
        if not os.path.isdir("simulations/posteriors_with_bh_mass"):
            os.makedirs("simulations/posteriors_with_bh_mass")

        # 4-decimal precision required to distinguish Phase-50 superdense
        # midpoints (Δh=0.0005, e.g. 0.7205 / 0.7215) from the dense Δh=0.001
        # grid (0.720 / 0.721 / 0.722). Rounding to 3 decimals collapses each
        # midpoint onto a neighbouring dense filename, so the second writer
        # silently overwrites the first. Posteriors share filenames only when
        # the underlying h-values agree to 4 decimals.
        h_label = str(np.round(self.h, 4)).replace(".", "_")
        with open(
            f"simulations/posteriors/h_{h_label}.json",
            "w",
        ) as file:
            data = {str(key): value for key, value in self.posterior_data.items()}
            json.dump(data | {"h": self.h}, file)

        with open(
            f"simulations/posteriors_with_bh_mass/h_{h_label}.json",
            "w",
        ) as file:
            # update existing data

            data = {str(key): value for key, value in self.posterior_data_with_bh_mass.items()}
            json.dump(data | {"h": self.h}, file)

        # Write per-event diagnostic CSV
        if self._diagnostic_rows:
            diagnostic_csv_path = "simulations/diagnostics/event_likelihoods.csv"
            self._write_diagnostic_csv(diagnostic_csv_path)

        # Write Fisher quality CSV (per D-12)
        self._write_fisher_quality_csv()

        # Generate Fisher quality diagnostic plot (per D-06, D-07)
        from master_thesis_code.plotting.fisher_plots import plot_fisher_diagnostics

        plot_fisher_diagnostics(
            cond_3d=self._cond_3d,
            cond_4d=self._cond_4d,
            excluded_mask=self._excluded_mask,
            eigen_3d=self._eigen_3d,
            eigen_4d=self._eigen_4d,
            det_d_L=self._det_d_L,
            det_M=self._det_M,
            det_index_to_slot=self._det_index_to_slot,
            threshold=self._fisher_cond_threshold,
            output_dir="simulations",
        )

    def _write_fisher_quality_csv(self) -> None:
        """Write per-event Fisher matrix condition numbers and exclusion flags to CSV.

        Columns: detection_index, cond_3d, cond_4d, excluded.
        Written once per evaluation run to ``simulations/fisher_quality.csv``.
        """
        rows = [
            {
                "detection_index": int(idx),
                "cond_3d": float(self._cond_3d[slot]),
                "cond_4d": float(self._cond_4d[slot]),
                "excluded": bool(self._excluded_mask[slot]),
            }
            for idx, slot in self._det_index_to_slot.items()
        ]
        df = pd.DataFrame(rows)
        csv_path = os.path.join("simulations", "fisher_quality.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        _LOGGER.info("Fisher quality CSV written to %s (%d rows)", csv_path, len(rows))

    def _write_diagnostic_csv(self, csv_path: str) -> None:
        """Write per-event diagnostic rows to CSV (append mode, header on first write).

        Args:
            csv_path: Path to the output CSV file.
        """
        if not self._diagnostic_rows:
            return

        fieldnames = [
            "event_idx",
            "h",
            "w_G",
            "L_cat_no_bh",
            "L_cat_with_bh",
            "B_num",
            "L_comp",
            "combined_no_bh",
            "combined_with_bh",
        ]

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        write_header = not os.path.isfile(csv_path)

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(self._diagnostic_rows)

        _LOGGER.info("Wrote %d diagnostic rows to %s", len(self._diagnostic_rows), csv_path)

    def p_D(
        self,
        galaxy_catalog: GalaxyCatalogueHandler,
        redshift_upper_limit: float,
        pool: mp.pool.Pool,
        completeness: CompletenessModel,
        detection_probability_obj: SimulationDetectionProbability,
    ) -> None:
        count = 0
        _det_times: list[float] = []
        self.posterior_data_with_bh_mass[GALAXY_LIKELIHOODS] = {}
        self.posterior_data_with_bh_mass[ADDITIONAL_GALAXIES_WITHOUT_BH_MASS] = {}
        for index, detection in self.cramer_rao_bounds.iterrows():
            _t_det = time.perf_counter()
            slot = self._det_index_to_slot[int(index)]
            if self._excluded_mask[slot]:
                _LOGGER.debug("Skipping excluded detection %d (Fisher quality)", int(index))
                continue
            _LOGGER.info(f"Progess: detections: {count}/{len(self.cramer_rao_bounds)}...")
            count += 1
            try:
                self.posterior_data[index]
            except KeyError:
                self.posterior_data[index] = []
                self.posterior_data_with_bh_mass[index] = []

            self.detection = Detection(detection)

            z_min, z_max = get_redshift_outer_bounds(
                distance=self.detection.d_L,
                distance_error=self.detection.d_L_uncertainty,
                h_min=self.cosmological_model.h.lower_limit,
                h_max=self.cosmological_model.h.upper_limit,
                Omega_m_min=self.cosmological_model.Omega_m.lower_limit,
                Omega_m_max=self.cosmological_model.Omega_m.upper_limit,
                sigma_multiplier=2.0,
            )

            z_max = min(z_max, redshift_upper_limit)

            possible_hosts = galaxy_catalog.get_possible_hosts_from_ball_tree(
                phi=self.detection.phi,
                theta=self.detection.theta,
                phi_sigma=self.detection.phi_error,
                theta_sigma=self.detection.theta_error,
                cov_theta_phi=self.detection.theta_phi_covariance,  # COORD-04: 2×2 sky Fisher off-diagonal
                z_min=z_min,
                z_max=z_max,
                M_z=self.detection.M,
                M_z_sigma=self.detection.M_uncertainty,
                sigma_multiplier=1.5,  # type: ignore[arg-type]
            )

            if possible_hosts is None:
                _LOGGER.debug("no possible hosts found...")
                continue
            possible_hosts, possible_hosts_with_bh_mass = possible_hosts  # type: ignore[assignment]
            _LOGGER.info(
                f"possible hosts found {len(possible_hosts)}/{len(possible_hosts_with_bh_mass)}..."
            )

            """
            if len(possible_hosts_with_bh_mass) == 0:
                detection_galaxy = _get_closest_possible_host(
                    self.detection, possible_hosts
                )
            else:
                detection_galaxy = _get_closest_possible_host(
                    self.detection, possible_hosts_with_bh_mass
                )

            self.detection.phi = detection_galaxy.phiS
            self.detection.theta = detection_galaxy.qS
            """

            event_likelihood, event_likelihood_with_bh_mass = self.p_Di(
                possible_host_galaxies=possible_hosts,  # type: ignore[arg-type]
                possible_host_galaxies_with_bh_mass=possible_hosts_with_bh_mass,
                detection_index=index,
                pool=pool,
                completeness=completeness,
                detection_probability_obj=detection_probability_obj,
            )

            self.posterior_data[index].append(event_likelihood)
            self.posterior_data_with_bh_mass[index].append(event_likelihood_with_bh_mass)

            _det_time = time.perf_counter() - _t_det
            _det_times.append(_det_time)
            if count % 100 == 0 or count == len(self.cramer_rao_bounds):
                _LOGGER.info(
                    "Detection %d/%d: last=%.2fs, avg=%.2fs, est_remaining=%.0fs",
                    count,
                    len(self.cramer_rao_bounds),
                    _det_time,
                    np.mean(_det_times),
                    np.mean(_det_times) * (len(self.cramer_rao_bounds) - count),
                )
            _LOGGER.debug(
                f"event likelihood: {event_likelihood}\nevent likelihood with bh mass: {event_likelihood_with_bh_mass}"
            )

    def p_Di(
        self,
        possible_host_galaxies: list[HostGalaxy],
        possible_host_galaxies_with_bh_mass: list[HostGalaxy],
        detection_index: int,
        pool: mp.pool.Pool,
        completeness: CompletenessModel,
        detection_probability_obj: SimulationDetectionProbability,
    ) -> tuple[float, float]:
        # start parallel computation
        _LOGGER.info(f"start parallel computation with: {pool}")
        start = time.time()
        # remove duplicates from possible_host_galaxies already covered in possible_host_galaxies_with_bh_mass

        hosts_with_bh_mass_set = set(possible_host_galaxies_with_bh_mass)

        possible_host_galaxies_reduced = [
            host for host in possible_host_galaxies if host not in hosts_with_bh_mass_set
        ]

        _LOGGER.debug(
            f"reduced possible hosts galaxies to unique, removed {len(possible_host_galaxies) - len(possible_host_galaxies_reduced)} galaxies."
        )

        chunksize = math.ceil(len(possible_host_galaxies_reduced) / pool._processes)  # type: ignore[attr-defined]
        chunksize_with_bh_mass = math.ceil(
            len(possible_host_galaxies_with_bh_mass) / pool._processes  # type: ignore[attr-defined]
        )
        results_with_bh_mass = pool.starmap(
            single_host_likelihood,
            [
                (
                    host.phiS,
                    host.qS,
                    host.z,
                    host.z_error,
                    host.M,
                    host.M_error,
                    detection_index,
                    self.h,
                    True,
                )
                for host in possible_host_galaxies_with_bh_mass
            ],
            chunksize=chunksize_with_bh_mass,
        )

        results_without_blackhole_mass = pool.starmap(
            single_host_likelihood,
            [
                (
                    host.phiS,
                    host.qS,
                    host.z,
                    host.z_error,
                    host.M,
                    host.M_error,
                    detection_index,
                    self.h,
                    False,
                )
                for host in possible_host_galaxies_reduced
            ],
            chunksize=chunksize,
        )
        end = time.time()
        _LOGGER.info(f"parallel computing took: {end - start}s")

        galaxy_likelihoods = list(
            zip(
                [galaxy.catalog_index for galaxy in possible_host_galaxies_with_bh_mass],
                results_with_bh_mass,
            )
        )

        self.posterior_data_with_bh_mass[GALAXY_LIKELIHOODS][detection_index] = galaxy_likelihoods

        additional_likelihoods = list(
            zip(
                [galaxy.catalog_index for galaxy in possible_host_galaxies_reduced],
                results_without_blackhole_mass,
            )
        )

        self.posterior_data_with_bh_mass[ADDITIONAL_GALAXIES_WITHOUT_BH_MASS][detection_index] = (
            additional_likelihoods
        )

        # --- In-catalogue weighted sums (Gray et al. 2020, Eqs. 24-25, A.9/A.10) ---
        # Per-MBH EMRI-rate weight w(g) = R_eff_per_mbh(M_g)/(1+z_g), IDENTICAL to
        # the simulation host draw (draw_rate_weighted_hosts): P(g) ∝ w(g). host.M is
        # the SOURCE-FRAME catalog BH mass (the detector-frame lift M_z = M·(1+z)
        # lives only inside single_host_likelihood, never on host.M). The overall
        # normalization (including emri_rate.C_NORM) cancels in every ratio below.
        # all_results_without_bh is ordered reduced + with_bh, so its weights MUST
        # follow the SAME host order. Babak et al. (2017), arXiv:1703.09722 (rate).
        if len(results_without_blackhole_mass) == 0 and len(results_with_bh_mass) == 0:
            _LOGGER.warning(f"Detection {detection_index}: no catalog results found")
            weights_with_bh: list[float] = []
            weights_without_bh: list[float] = []
            all_results_without_bh: list[Any] = []
        else:
            weights_with_bh = [_rate_weight(host) for host in possible_host_galaxies_with_bh_mass]
            weights_without_bh = [
                _rate_weight(host) for host in possible_host_galaxies_reduced
            ] + weights_with_bh
            all_results_without_bh = list(results_without_blackhole_mass) + list(
                results_with_bh_mass
            )

        # --- Per-event likelihood: Gray et al. (2020), arXiv:1908.06050, Eq. 9 + 29 ---
        # Single selection-normalized ratio
        #     p_i = (beta_G(h) * L_cat + B_num(h)) / D(h)
        # equivalently w_G*L_cat + (1-w_G)*L_comp with the EXACT event-INDEPENDENT
        # selection weight w_G = beta_G/D(h) = beta_G/(beta_G+beta_Gbar) (Eq. 29),
        # which REPLACES the old scalar mixing weight completeness(z_det). The
        # incompleteness (1-f(z)) lives INSIDE the completion numerator B_num and
        # denominator beta_Gbar; there is NO scalar (1-f_i) prefactor (keeping one on
        # top of the inside-(1-f) would compute (1-f)^2 and double-count).
        if self.catalog_only:
            # Pure-catalog cross-check (validation mode): the per-event in-catalogue
            # likelihood is the self-normalized LOCAL ratio of sums, no completion.
            # Unchanged from the convex-mix era (f_i=1, L_comp=0 => p_i = L_cat), so
            # this mode stays byte-identical.
            L_cat_without_bh_mass = weighted_ratio_of_sums(
                [r[0] for r in all_results_without_bh],
                [r[1] for r in all_results_without_bh],
                weights_without_bh,
            )
            if len(results_with_bh_mass) > 0:
                L_cat_with_bh_mass = weighted_ratio_of_sums(
                    [r[2] for r in results_with_bh_mass],
                    [r[3] for r in results_with_bh_mass],
                    weights_with_bh,
                )
            else:
                L_cat_with_bh_mass = 0.0
            combined_without_bh_mass = float(L_cat_without_bh_mass)
            combined_with_bh_mass = float(L_cat_with_bh_mass)
            w_G = 1.0
            B_num = 0.0
            L_comp = 0.0
        else:
            D_h: float = self._D_h_table.get(self.h, 0.0)
            beta_G: float = self._beta_G_table.get(self.h, 0.0)
            beta_Gbar: float = self._beta_Gbar_table.get(self.h, 0.0)
            global_denom_no_bh: float = self._global_cat_denom_no_bh.get(self.h, 0.0)
            global_denom_with_bh: float = self._global_cat_denom_with_bh.get(self.h, 0.0)

            # L_cat = (Σ_local w_g N_g) / (Σ_GLOBAL w_g D_g): GW-likelihood numerator
            # over the local candidate ball (p_GW self-truncates, so the local sum is
            # effectively the global in-catalogue numerator), SELECTION denominator
            # over the full catalogue out to z_max(h) (Eq. 29, precomputed in
            # precompute_global_catalog_selection). Globalising the denominator makes
            # L_cat scale-free, so beta_G*L_cat reconstructs the in-catalogue
            # numerator with the per-galaxy<->per-volume n_gal factor CANCELLED
            # (Option A; without it beta_G^global/beta_G^local would bias H0).
            cat_num_sum_no_bh = weighted_sum(
                [r[0] for r in all_results_without_bh], weights_without_bh
            )
            L_cat_without_bh_mass = (
                cat_num_sum_no_bh / global_denom_no_bh if global_denom_no_bh > 0 else 0.0
            )
            if len(results_with_bh_mass) > 0:
                cat_num_sum_with_bh = weighted_sum(
                    [r[2] for r in results_with_bh_mass], weights_with_bh
                )
                L_cat_with_bh_mass = (
                    cat_num_sum_with_bh / global_denom_with_bh if global_denom_with_bh > 0 else 0.0
                )
            else:
                L_cat_with_bh_mass = 0.0

            # B_num(h) = INTEGRAL (1-f(z)) p_GW(z) (1/(1+z)) dVc/dz dz : the completion
            # numerator with the incompleteness weight (1-f(z)). Gray et al. (2020),
            # arXiv:1908.06050, Eq. 32 -- GW likelihood × population prior ONLY; the
            # (1-f) is the smooth-completeness form of the catalog-edge lower limit
            # and is EXACTLY the dark population the generator draws
            # (dark_siren_injection._draw_dark_redshifts). f(z) is evaluated on the
            # quadrature grid (NOT at z_det); p_det stays solely in the denominator
            # D(h) (Mandel-Farr-Gair 2019, arXiv:1809.02063). 1/(1+z) matches D(h),
            # beta_Gbar, and the event sampler (emri_rate.p_pop_unnormalized).
            integration_limit_sigma_multiplier = 4.0
            z_upper = dist_to_redshift(
                self.detection.d_L
                + integration_limit_sigma_multiplier * self.detection.d_L_uncertainty,
                h=self.h,
            )
            z_lower = dist_to_redshift(
                self.detection.d_L
                - integration_limit_sigma_multiplier * self.detection.d_L_uncertainty,
                h=self.h,
            )
            z_lower = max(z_lower, 1e-6)  # avoid z=0 singularity in volume element

            FIXED_QUAD_N = 50
            _comp_slot = self._det_index_to_slot[detection_index]
            _comp_mean_3d = self._means_3d[_comp_slot]
            _comp_cov_inv_3d = self._cov_inv_3d[_comp_slot]
            _comp_log_norm_3d = float(self._log_norm_3d[_comp_slot])
            _comp_det_d_L = self._det_d_L[_comp_slot]
            # Change 5.3: the completion numerator weights the incompleteness at the
            # EVENT's sky pixel, (1 - f_{k(Omega_e)}(z)). p_GW delta-collapses the sky
            # integral, so f is evaluated at the single pixel containing the detection
            # direction (ecliptic phi/theta). Gray-Messenger-Veitch 2022,
            # arXiv:2111.04629, Eq. (5) (out-of-catalog branch). Computed once per
            # event; Omega-independent completeness gives the identical Task-A B_num.
            _event_pixel = completeness.ang2pix(self.detection.phi, self.detection.theta)

            def completion_numerator_integrand(
                z: npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
                d_L: npt.NDArray[np.float64] = np.asarray(
                    dist_vectorized(z, h=self.h), dtype=np.float64
                )  # Gpc
                d_L_fraction = d_L / _comp_det_d_L  # dimensionless
                phi = np.full_like(z, self.detection.phi)
                theta = np.full_like(z, self.detection.theta)

                p_gw: npt.NDArray[np.float64] = _mvn_pdf(
                    np.vstack([phi, theta, d_L_fraction]).T,
                    _comp_mean_3d,
                    _comp_cov_inv_3d,
                    _comp_log_norm_3d,
                )
                dVc: npt.NDArray[np.float64] = np.atleast_1d(
                    np.asarray(comoving_volume_element(z, h=self.h), dtype=np.float64)
                )
                # Eq. (32) in Gray et al. (2020), arXiv:1908.06050, with the per-pixel
                # incompleteness weight (1-f_{k(Omega_e)}(z)): GW likelihood × (1-f_k)
                # population prior, f_k evaluated at the EVENT pixel (Change 5.3,
                # Gray-Messenger-Veitch 2022 Eq. 5). f_k is the SAME completeness call
                # the generator uses (dark_siren_injection W_k sampler, restricted to
                # this pixel up to p_pop->p_GW), so B_num integrates exactly the
                # injected dark density at the event direction.
                f_z: npt.NDArray[np.float64] = np.clip(
                    np.asarray(
                        completeness.f_k(z, _event_pixel, self.h),
                        dtype=np.float64,
                    ),
                    0.0,
                    1.0,
                )
                return (1.0 - f_z) * p_gw * dVc / (1.0 + z)

            B_num = float(
                fixed_quad(completion_numerator_integrand, z_lower, z_upper, n=FIXED_QUAD_N)[0]
            )

            # Grid coverage flag: warn if numerator 4-sigma window exceeds P_det grid
            d_L_upper = self.detection.d_L + 4.0 * self.detection.d_L_uncertainty
            dl_max_grid = detection_probability_obj.get_dl_max(self.h)
            if d_L_upper > dl_max_grid:
                _LOGGER.warning(
                    "Detection %d: 4-sigma d_L upper (%.4f Gpc) exceeds P_det grid max (%.4f Gpc)",
                    detection_index,
                    d_L_upper,
                    dl_max_grid,
                )

            # Single ratio p_i = (beta_G*L_cat + B_num)/D(h). w_G = beta_G/D(h) is the
            # event-independent selection-weighted catalog membership probability
            # (Eq. 29). Tier 3 audit (2026-05-04): the outer -N log D subtraction in
            # combine_log_space stays disabled (D(h) normalizes here, per-event).
            if D_h > 0:
                w_G = beta_G / D_h
                combined_without_bh_mass = float((beta_G * L_cat_without_bh_mass + B_num) / D_h)
                combined_with_bh_mass = float((beta_G * L_cat_with_bh_mass + B_num) / D_h)
            else:
                _LOGGER.warning(f"Detection {detection_index}: D(h) is zero, using L_cat only")
                w_G = 1.0
                combined_without_bh_mass = float(L_cat_without_bh_mass)
                combined_with_bh_mass = float(L_cat_with_bh_mass)
            # Diagnostic-only completion likelihood L_comp = B_num/beta_Gbar (the
            # single ratio never divides by beta_Gbar, which -> 0 as f -> 1).
            L_comp = float(B_num / beta_Gbar) if beta_Gbar > 0 else 0.0

        _LOGGER.debug(
            f"Detection {detection_index}: w_G={w_G:.4f}, "
            f"L_cat_no_bh={L_cat_without_bh_mass:.6e}, "
            f"L_cat_with_bh={L_cat_with_bh_mass:.6e}, B_num={B_num:.6e}, L_comp={L_comp:.6e}"
        )

        # Record diagnostic row for every event
        self._diagnostic_rows.append(
            {
                "event_idx": detection_index,
                "h": self.h,
                "w_G": w_G,
                "L_cat_no_bh": L_cat_without_bh_mass,
                "L_cat_with_bh": L_cat_with_bh_mass,
                "B_num": B_num,
                "L_comp": L_comp,
                "combined_no_bh": combined_without_bh_mass,
                "combined_with_bh": combined_with_bh_mass,
            }
        )

        return (combined_without_bh_mass, combined_with_bh_mass)


def use_detection(detection: Detection) -> bool:
    sky_localization_uncertainty = _sky_localization_uncertainty(
        phi_error=detection.phi_error,
        theta=detection.theta,
        theta_error=detection.theta_error,
        cov_theta_phi=detection.theta_phi_covariance,
    )
    distance_relative_error = detection.d_L_uncertainty / detection.d_L

    if distance_relative_error < FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD:
        return True
    _LOGGER.debug(
        f"Detection skipped: distance_relative_error {distance_relative_error} > {FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD}, sky_localization_uncertainty {sky_localization_uncertainty}"
    )
    return False


def single_host_likelihood_grid(
    possible_host: HostGalaxy,
    detection: Detection,
    detection_index: int,
    h: float,
    evaluate_with_bh_mass: bool,
) -> list[float]:
    global redshift_upper_integration_limit
    global redshift_lower_integration_limit
    global bh_mass_upper_integration_limit
    global bh_mass_lower_integration_limit
    global detection_probability
    global detection_likelihood_gaussians_by_detection_index

    # find sharpest peak
    print(possible_host.z, possible_host.z_error)
    print(detection.d_L, detection.d_L_uncertainty)
    return []


def single_host_likelihood(
    host_phiS: float,
    host_qS: float,
    host_z: float,
    host_z_error: float,
    host_M: float,
    host_M_error: float,
    detection_index: int,
    h: float,
    evaluate_with_bh_mass: bool,
) -> list[float]:
    global redshift_upper_integration_limit
    global redshift_lower_integration_limit
    global bh_mass_upper_integration_limit
    global bh_mass_lower_integration_limit
    global detection_probability
    global means_3d, cov_inv_3d, log_norm_3d
    global means_4d, cov_inv_4d, log_norm_4d
    global det_index_to_slot
    global sigma2_cond_arr, proj_arr
    global det_d_L_arr, det_d_L_unc_arr, det_M_arr, det_phi_arr, det_theta_arr

    FIXED_QUAD_N = 50

    slot = det_index_to_slot[detection_index]
    _det_d_L = float(det_d_L_arr[slot])
    _det_d_L_unc = float(det_d_L_unc_arr[slot])
    _det_M = float(det_M_arr[slot])
    _mean_3d = means_3d[slot]
    _cov_inv_3d = cov_inv_3d[slot]
    _log_norm_3d = float(log_norm_3d[slot])

    integration_limit_sigma_multiplier = 4.0

    numerator_integration_upper_redshift_limit = dist_to_redshift(
        _det_d_L + integration_limit_sigma_multiplier * _det_d_L_unc, h=h
    )
    numerator_integration_lower_redshift_limit = dist_to_redshift(
        _det_d_L - integration_limit_sigma_multiplier * _det_d_L_unc, h=h
    )
    denominator_integration_upper_redshift_limit = (
        host_z + integration_limit_sigma_multiplier * host_z_error
    )
    denominator_integration_lower_redshift_limit = (
        host_z - integration_limit_sigma_multiplier * host_z_error
    )

    # construct normal distribution for redshift and mass for host galaxy
    galaxy_redshift_normal_distribution = norm(loc=host_z, scale=host_z_error)

    # Sky localization weight (phi, theta) is inside the GW likelihood Gaussian.
    # Verified correct by Phase 14 derivation (Sec. 2.7): the 3D/4D GW Gaussian
    # naturally encodes the sky position weight -- this is NOT a source of error.
    def numerator_integrant_without_bh_mass(z: npt.NDArray[np.float64]) -> Any:
        d_L = dist_vectorized(z, h=h)
        # fraction = d_L_model / d_L_measured; matches covariance σ²/d_L_measured²
        luminosity_distance_fraction = d_L / _det_d_L
        phi = np.full_like(z, host_phiS)
        theta = np.full_like(z, host_qS)

        # Eq. (A.10) in Gray et al. (2020), arXiv:1908.06050: the in-catalogue
        # numerator carries the GW likelihood p(x_GW|z,Omega,H0) and the galaxy
        # redshift uncertainty p(z) ONLY.  The detection probability
        # p_det = p(D_GW|z,Omega,H0) appears solely in the denominator D_g (below);
        # an extra p_det in the numerator is the Mandel-Farr-Gair (2019)
        # "most common mistake" (arXiv:1809.02063) and biases H0 high.
        return _mvn_pdf(
            np.vstack([phi, theta, luminosity_distance_fraction]).T,
            _mean_3d,
            _cov_inv_3d,
            _log_norm_3d,
        ) * galaxy_redshift_normal_distribution.pdf(z)

    def denominator_integrant_without_bh_mass(z: npt.NDArray[np.float64]) -> Any:
        d_L = dist_vectorized(z, h=h)
        phi = np.full_like(z, host_phiS)
        theta = np.full_like(z, host_qS)
        # Gray et al. (2020), arXiv:1908.06050, Eq. A.19: shared p_det function
        # with D(h) denominator (STAT-03 symmetry, commit a70d1a2).  Phase 44:
        # NN-fill below first bin (real injection statistic), zero above
        # injection horizon.
        p_det = detection_probability.detection_probability_without_bh_mass_interpolated_zero_fill(
            d_L, phi, theta, h=h
        )
        return p_det * galaxy_redshift_normal_distribution.pdf(z)

    (
        single_host_likelihood_numerator_without_bh_mass,
        single_host_likelihood_numerator_without_bh_mass_error,
    ) = fixed_quad(
        numerator_integrant_without_bh_mass,
        numerator_integration_lower_redshift_limit,
        numerator_integration_upper_redshift_limit,
        n=FIXED_QUAD_N,
    )
    (
        single_host_likelihood_denominator_without_bh_mass,
        single_host_likelihood_denominator_without_bh_mass_error,
    ) = fixed_quad(
        denominator_integrant_without_bh_mass,
        denominator_integration_lower_redshift_limit,
        denominator_integration_upper_redshift_limit,
        n=FIXED_QUAD_N,
    )

    # STAT-04: Per-event off-grid quadrature weight diagnostic.
    # Estimate the fraction of the integration window that lies outside the P_det grid.
    # Grid bounds are the first/last bin centres of the 1D interpolator grid.
    # Attribute access: detection_probability._get_or_build_grid(h)[1].grid[0] → d_L centres.
    _, _interp_1d = detection_probability._get_or_build_grid(h)
    _dl_centers = _interp_1d.grid[0]
    _dl_grid_min = float(_dl_centers[0])
    _dl_grid_max = float(_dl_centers[-1])

    # Numerator window: d_L(z_det ± 4σ)  [redshift limits → d_L limits]
    _dl_lower_num = float(
        dist_vectorized(np.array([numerator_integration_lower_redshift_limit]), h=h)[0]
    )
    _dl_upper_num = float(
        dist_vectorized(np.array([numerator_integration_upper_redshift_limit]), h=h)[0]
    )
    _window_num = _dl_upper_num - _dl_lower_num
    if _window_num > 0.0:
        _below_min_num = max(0.0, min(_dl_upper_num, _dl_grid_min) - _dl_lower_num) / _window_num
        _above_max_num = max(0.0, _dl_upper_num - max(_dl_lower_num, _dl_grid_max)) / _window_num
        quadrature_weight_outside_grid_numerator = float(
            np.clip(_below_min_num + _above_max_num, 0.0, 1.0)
        )
    else:
        quadrature_weight_outside_grid_numerator = 0.0

    # Denominator window: d_L(z_gal ± 4σ_z)  [redshift limits → d_L limits]
    _dl_lower_den = float(
        dist_vectorized(np.array([denominator_integration_lower_redshift_limit]), h=h)[0]
    )
    _dl_upper_den = float(
        dist_vectorized(np.array([denominator_integration_upper_redshift_limit]), h=h)[0]
    )
    _window_den = _dl_upper_den - _dl_lower_den
    if _window_den > 0.0:
        _below_min_den = max(0.0, min(_dl_upper_den, _dl_grid_min) - _dl_lower_den) / _window_den
        _above_max_den = max(0.0, _dl_upper_den - max(_dl_lower_den, _dl_grid_max)) / _window_den
        quadrature_weight_outside_grid_denominator = float(
            np.clip(_below_min_den + _above_max_den, 0.0, 1.0)
        )
    else:
        quadrature_weight_outside_grid_denominator = 0.0

    if (
        quadrature_weight_outside_grid_numerator > 0.05
        or quadrature_weight_outside_grid_denominator > 0.05
    ):
        _LOGGER.warning(
            "Event %d: >5%% quadrature weight outside P_det grid — "
            "numerator=%.3f, denominator=%.3f",
            detection_index,
            quadrature_weight_outside_grid_numerator,
            quadrature_weight_outside_grid_denominator,
        )

    if evaluate_with_bh_mass:
        galaxy_mass_normal_distribution = norm(loc=host_M, scale=host_M_error)

        # Pre-computed conditional distribution parameters for analytic M_z marginalization
        # Eqs. (14.23)-(14.28) in derivations/dark_siren_likelihood.md
        # Ref: Bishop (2006) PRML Eq. 2.81-2.82 (multivariate normal conditioning)
        _sigma2_cond = float(sigma2_cond_arr[slot])
        _proj = proj_arr[slot]
        _mu_obs_4d = means_4d[slot]

        def numerator_integrant_with_bh_mass(z: npt.NDArray[np.float64]) -> Any:
            d_L = dist_vectorized(z, h=h)
            luminosity_distance_fraction = d_L / _det_d_L
            phi = np.full_like(z, host_phiS)
            theta = np.full_like(z, host_qS)

            # Eq. (A.10) in Gray et al. (2020), arXiv:1908.06050: the in-catalogue
            # numerator carries the GW likelihood and mass/redshift priors ONLY.
            # p_det = p(D_GW|...) is applied solely in the denominator (below);
            # a numerator p_det is the Mandel-Farr-Gair (2019) "most common
            # mistake" (arXiv:1809.02063) and biases H0 high.

            # 3D marginal Gaussian: p(phi, theta, d_L_frac)
            # The 3D marginal is the upper-left 3x3 block of the 4D covariance
            gw_3d = _mvn_pdf(
                np.vstack([phi, theta, luminosity_distance_fraction]).T,
                _mean_3d,
                _cov_inv_3d,
                _log_norm_3d,
            )

            # Conditional mean of M_z_frac given (phi_gal, theta_gal, d_L_frac)
            x_obs = np.vstack([phi, theta, luminosity_distance_fraction]).T  # (N, 3)
            mu_cond = _mu_obs_4d[3] + (x_obs - _mu_obs_4d[:3]) @ _proj  # (N,)

            # Galaxy mass in M_z_frac coordinates: M_z_frac = M_gal * (1+z) / M_z_det
            # Eq. (14.22) in derivations/dark_siren_likelihood.md
            # NOTE: (1+z) here is CORRECT -- it is the coordinate transform, not a Jacobian
            mu_gal_frac = host_M * (1 + z) / _det_M
            sigma_gal_frac = host_M_error * (1 + z) / _det_M

            # Analytic Gaussian product integral:
            # ∫ N(x; μ_cond, σ²_cond) · N(x; μ_gal, σ²_gal) dx
            #   = N(μ_cond; μ_gal, σ²_cond + σ²_gal)
            # Eq. (14.31) in derivations/dark_siren_likelihood.md
            sigma2_sum = _sigma2_cond + sigma_gal_frac**2
            mz_integral = np.exp(-0.5 * (mu_cond - mu_gal_frac) ** 2 / sigma2_sum) / np.sqrt(
                2 * np.pi * sigma2_sum
            )

            # Eq. (A.10) in Gray et al. (2020): GW likelihood x mass-marginal x
            # galaxy z-prior; p_det removed from the numerator (denominator-only).
            # Eq. (14.32) in derivations/dark_siren_likelihood.md
            # No /(1+z) factor: Jacobian absorbed by Gaussian rescaling (Eq. 14.21)
            return gw_3d * mz_integral * galaxy_redshift_normal_distribution.pdf(z)

        single_host_likelihood_numerator_with_bh_mass = fixed_quad(
            numerator_integrant_with_bh_mass,
            numerator_integration_lower_redshift_limit,
            numerator_integration_upper_redshift_limit,
            n=FIXED_QUAD_N,
        )[0]

        # Eq. (14.33) in derivations/dark_siren_likelihood.md
        # Denominator: p_det(d_L, M_z, phi, theta) * p_gal(z) * p_gal(M)
        # No GW likelihood, no mz_integral, no /(1+z) -- confirmed correct by Phase 14
        def denominator_integrant_with_bh_mass_vectorized(
            M: npt.NDArray[np.float64], z: npt.NDArray[np.float64]
        ) -> Any:
            d_L = dist_vectorized(z, h=h)
            M_z = M * (1 + z)
            phi = np.full_like(M, host_phiS)
            theta = np.full_like(M, host_qS)
            # p_det in observer-frame M_z (consistent with grid axis and
            # the numerator's host_M·(1+z) hypothesis).
            p_det = detection_probability.detection_probability_with_bh_mass_interpolated(
                d_L, M_z, phi, theta, h=h
            )
            return (
                p_det
                * galaxy_redshift_normal_distribution.pdf(z)
                * galaxy_mass_normal_distribution.pdf(M)
            )

        # MC importance sampling for 2D denominator integral over (z, M).
        # Proposal distribution: q(z, M) = p_gal(z) * p_gal(M).
        # After cancellation, weights = p_det (each in [0, 1]).
        # Relative MC error ~ std(p_det) / (sqrt(N) * mean(p_det)) ~ 1% for N=10000.
        # Numerator uses fixed_quad (1D over z, mass analytically marginalized) --
        # the quadrature-vs-MC asymmetry is a numerical choice, not a physics error.
        N_SAMPLES = 10_000
        z_samples = galaxy_redshift_normal_distribution.rvs(size=N_SAMPLES)
        M_samples = galaxy_mass_normal_distribution.rvs(size=N_SAMPLES)

        numerator_integrant_from_samples = denominator_integrant_with_bh_mass_vectorized(
            M_samples, z_samples
        )

        sampling_pdf = galaxy_redshift_normal_distribution.pdf(
            z_samples
        ) * galaxy_mass_normal_distribution.pdf(M_samples)
        weights = numerator_integrant_from_samples / sampling_pdf

        single_host_likelihood_denominator_with_bh_mass = np.mean(weights)

        return [
            single_host_likelihood_numerator_without_bh_mass,
            single_host_likelihood_denominator_without_bh_mass,
            single_host_likelihood_numerator_with_bh_mass,
            single_host_likelihood_denominator_with_bh_mass,
            quadrature_weight_outside_grid_numerator,
            quadrature_weight_outside_grid_denominator,
        ]
    return [
        single_host_likelihood_numerator_without_bh_mass,
        single_host_likelihood_denominator_without_bh_mass,
        quadrature_weight_outside_grid_numerator,
        quadrature_weight_outside_grid_denominator,
    ]


def single_host_likelihood_integration_testing(
    possible_host: HostGalaxy,
    detection: Detection,
    detection_index: int,
    h: float,
    evaluate_with_bh_mass: bool,
) -> list[float]:
    global redshift_upper_integration_limit
    global redshift_lower_integration_limit
    global bh_mass_upper_integration_limit
    global bh_mass_lower_integration_limit
    global detection_probability
    global detection_likelihood_gaussians_by_detection_index

    ABS_ERROR = 1e-20

    # construct normal distribution for redshift and mass for host galaxy
    galaxy_redshift_normal_distribution = norm(loc=possible_host.z, scale=possible_host.z_error)

    # Sky localization weight (phi, theta) is inside the GW likelihood Gaussian.
    # Verified correct by Phase 14 derivation (Sec. 2.7) -- not a source of error.
    def numerator_integrant_without_bh_mass(z: float) -> float:
        d_L = dist(z, h=h)
        luminosity_distance_fraction = d_L / detection.d_L
        # Gray et al. (2020), arXiv:1908.06050, Eq. A.19: shared p_det function
        # with D(h) denominator (STAT-03 symmetry).  Phase 44 boundary convention:
        # NN-fill below first bin, zero above injection horizon.
        return float(
            detection_probability.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_L, possible_host.phiS, possible_host.qS, h=h
            )
            * detection_likelihood_gaussians_by_detection_index[detection_index][0].pdf(
                [possible_host.phiS, possible_host.qS, luminosity_distance_fraction]
            )
            * galaxy_redshift_normal_distribution.pdf(z)
        )

    def denominator_integrant_without_bh_mass(z: float) -> float:
        d_L = dist(z, h=h)
        # Gray et al. (2020), arXiv:1908.06050, Eq. A.19: shared p_det function
        # with D(h) denominator (STAT-03 symmetry).  Phase 44 boundary convention:
        # NN-fill below first bin, zero above injection horizon.
        return float(
            detection_probability.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_L, possible_host.phiS, possible_host.qS, h=h
            )
            * galaxy_redshift_normal_distribution.pdf(z)
        )

    (
        single_host_likelihood_numerator_without_bh_mass,
        single_host_likelihood_numerator_without_bh_mass_error,
    ) = quad(
        numerator_integrant_without_bh_mass,
        redshift_lower_integration_limit,
        redshift_upper_integration_limit,
        epsabs=ABS_ERROR,
    )
    (
        single_host_likelihood_denominator_without_bh_mass,
        single_host_likelihood_denominator_without_bh_mass_error,
    ) = quad(
        denominator_integrant_without_bh_mass,
        redshift_lower_integration_limit,
        redshift_upper_integration_limit,
        epsabs=ABS_ERROR,
    )

    print(
        f"Numerator without bh m:{single_host_likelihood_numerator_without_bh_mass}, error estimation: {single_host_likelihood_numerator_without_bh_mass_error}",
        flush=True,
    )
    print(
        f"Denominator without bh m:{single_host_likelihood_denominator_without_bh_mass}, error estimation {single_host_likelihood_denominator_without_bh_mass_error}",
        flush=True,
    )

    if evaluate_with_bh_mass:
        galaxy_mass_normal_distribution = norm(loc=possible_host.M, scale=possible_host.M_error)
        """
        # double integral version
        def numerator_integrant_with_bh_mass(M: float, z: float) -> float:
            d_L = dist(z, h=h)
            M_z = M * (1 + z)
            luminosity_distance_fraction = d_L / detection.d_L
            redshifted_mass_fraction = M_z / detection.M
            return (
                detection_probability.detection_probability_with_bh_mass_interpolated(
                    d_L, M_z, possible_host.phiS, possible_host.qS, h=h
                )
                * detection_likelihood_gaussians_by_detection_index[
                    detection_index
                ][1].pdf(
                    [possible_host.phiS, possible_host.qS, luminosity_distance_fraction, redshifted_mass_fraction]
                )
                * galaxy_redshift_normal_distribution.pdf(z)
                * galaxy_mass_normal_distribution.pdf(M)
            )

        def denominator_integrant_with_bh_mass(M: float, z: float) -> float:
            d_L = dist(z, h=h)
            M_z = M * (1 + z)
            return (
                detection_probability.detection_probability_with_bh_mass_interpolated(
                    d_L, M_z, possible_host.phiS, possible_host.qS, h=h
                )
                * galaxy_redshift_normal_distribution.pdf(z)
                * galaxy_mass_normal_distribution.pdf(M)
            )
        start = time.time()
        single_host_likelihood_numerator_with_bh_mass, single_host_likelihood_numerator_without_bh_mass_error = dblquad(
            numerator_integrant_with_bh_mass,
            redshift_lower_integration_limit,
            redshift_upper_integration_limit,
            lambda z: bh_mass_lower_integration_limit,
            lambda z: bh_mass_upper_integration_limit,
            epsabs=ABS_ERROR
        )
        single_host_likelihood_denominator_with_bh_mass, single_host_likelihood_denominator_with_bh_mass_error = dblquad(
            denominator_integrant_with_bh_mass,
            redshift_lower_integration_limit,
            redshift_upper_integration_limit,
            lambda m: bh_mass_lower_integration_limit,
            lambda m: bh_mass_upper_integration_limit,
            epsabs=ABS_ERROR
        )
        end = time.time()
        print(f"Time taken for double integral: {end - start}", flush=True)

        print(f"Numerator with bh m:{single_host_likelihood_numerator_with_bh_mass}, error estimation: {single_host_likelihood_numerator_without_bh_mass_error}", flush=True)
        print(f"Denominator with bh m:{single_host_likelihood_denominator_with_bh_mass}, error estimation {single_host_likelihood_denominator_with_bh_mass_error}", flush=True)
        """

        # Analytic marginalization over M_z_frac (same as production path)
        # Ref: Bishop (2006) PRML Eq. 2.81-2.82
        gaussian_4d_test = detection_likelihood_gaussians_by_detection_index[detection_index][1]
        cov_4d_test = np.asarray(gaussian_4d_test.cov)
        mu_obs_4d_test = np.asarray(gaussian_4d_test.mean)
        cov_obs_test = cov_4d_test[:3, :3]
        cov_cross_test = cov_4d_test[3, :3]
        cov_mz_test = cov_4d_test[3, 3]
        cov_obs_inv_test = np.linalg.pinv(cov_obs_test)
        sigma2_cond_test = float(cov_mz_test - cov_cross_test @ cov_obs_inv_test @ cov_cross_test)
        sigma2_cond_test = max(sigma2_cond_test, 1e-30)
        proj_test = cov_cross_test @ cov_obs_inv_test
        try:
            gaussian_3d_marginal_test = multivariate_normal(
                mean=mu_obs_4d_test[:3], cov=cov_obs_test
            )
        except np.linalg.LinAlgError:
            _LOGGER.warning(
                "Testing path: degenerate 3D covariance for detection %d — skipping",
                detection_index,
            )
            return [0.0]

        def numerator_integrant_with_bh_mass(z: float) -> float:
            d_L = dist(z, h=h)
            luminosity_distance_fraction = d_L / detection.d_L

            x_obs_test = np.array(
                [possible_host.phiS, possible_host.qS, luminosity_distance_fraction]
            )
            gw_3d = float(gaussian_3d_marginal_test.pdf(x_obs_test))

            mu_cond = float(mu_obs_4d_test[3] + proj_test @ (x_obs_test - mu_obs_4d_test[:3]))
            mu_gal_frac = possible_host.M * (1 + z) / detection.M
            sigma_gal_frac = possible_host.M_error * (1 + z) / detection.M
            sigma2_sum = sigma2_cond_test + sigma_gal_frac**2
            mz_integral = float(
                np.exp(-0.5 * (mu_cond - mu_gal_frac) ** 2 / sigma2_sum)
                / np.sqrt(2 * np.pi * sigma2_sum)
            )

            # Eq. (14.32) in derivations/dark_siren_likelihood.md
            # No /(1+z) factor: Jacobian absorbed by Gaussian rescaling (Eq. 14.21)
            return float(
                detection_probability.detection_probability_with_bh_mass_interpolated(
                    d_L, detection.M, possible_host.phiS, possible_host.qS, h=h
                )
                * gw_3d
                * mz_integral
                * galaxy_redshift_normal_distribution.pdf(z)
            )

        def denominator_integrant_with_bh_mass(M: float, z: float) -> float:
            d_L = dist(z, h=h)
            M_z = M * (1 + z)
            return float(
                detection_probability.detection_probability_with_bh_mass_interpolated(
                    d_L, M_z, possible_host.phiS, possible_host.qS, h=h
                )
                * galaxy_redshift_normal_distribution.pdf(z)
                * galaxy_mass_normal_distribution.pdf(M)
            )

        start = time.time()
        (
            single_host_likelihood_numerator_with_bh_mass,
            single_host_likelihood_numerator_with_bh_mass_error,
        ) = quad(
            numerator_integrant_with_bh_mass,
            redshift_lower_integration_limit,
            redshift_upper_integration_limit,
            epsabs=ABS_ERROR,
        )

        (
            single_host_likelihood_denominator_with_bh_mass,
            single_host_likelihood_denominator_with_bh_mass_error,
        ) = dblquad(
            denominator_integrant_with_bh_mass,
            galaxy_redshift_normal_distribution.mean()
            - 5 * galaxy_redshift_normal_distribution.std(),
            galaxy_redshift_normal_distribution.mean()
            + 5 * galaxy_redshift_normal_distribution.std(),
            lambda m: (
                galaxy_mass_normal_distribution.mean() - 5 * galaxy_mass_normal_distribution.std()
            ),
            lambda m: (
                galaxy_mass_normal_distribution.mean() + 5 * galaxy_mass_normal_distribution.std()
            ),
            epsabs=ABS_ERROR,
        )
        end = time.time()
        print(f"Time taken for delta function approximation: {end - start}s", flush=True)

        print(
            f"Numerator with bh m:{single_host_likelihood_numerator_with_bh_mass}, error estimation: {single_host_likelihood_numerator_with_bh_mass_error}",
            flush=True,
        )
        print(
            f"Denominator with bh m:{single_host_likelihood_denominator_with_bh_mass}, error estimation {single_host_likelihood_denominator_with_bh_mass_error}",
            flush=True,
        )

        # monte carlo integration denominator 2D
        start = time.time()

        def denominator_integrant_with_bh_mass_vectorized(
            M: npt.NDArray[np.float64], z: npt.NDArray[np.float64]
        ) -> Any:
            d_L = dist_vectorized(z, h=h)
            M_z = M * (1 + z)
            phi = np.ones_like(M) * possible_host.phiS
            theta = np.ones_like(M) * possible_host.qS
            return (
                detection_probability.detection_probability_with_bh_mass_interpolated(
                    d_L, M_z, phi, theta, h=h
                )
                * galaxy_redshift_normal_distribution.pdf(z)
                * galaxy_mass_normal_distribution.pdf(M)
            )

        N_SAMPLES = 100_00
        z_samples = galaxy_redshift_normal_distribution.rvs(size=N_SAMPLES)
        M_samples = galaxy_mass_normal_distribution.rvs(size=N_SAMPLES)

        numerator_integrant_from_samples = denominator_integrant_with_bh_mass_vectorized(
            M_samples, z_samples
        )

        sampling_pdf = galaxy_redshift_normal_distribution.pdf(
            z_samples
        ) * galaxy_mass_normal_distribution.pdf(M_samples)
        weights = numerator_integrant_from_samples / sampling_pdf

        integral = np.mean(weights)
        integral_error = np.std(weights) / np.sqrt(N_SAMPLES)
        end = time.time()
        print(f"Time taken for monte carlo integration: {end - start}s", flush=True)
        print(
            f"Monte Carlo denominator integral with bh mass: {integral}, error estimation: {integral_error}",
            flush=True,
        )
        print(
            f"Integration difference: {abs(single_host_likelihood_denominator_with_bh_mass - integral)}",
            flush=True,
        )

        return [
            single_host_likelihood_numerator_without_bh_mass,
            single_host_likelihood_denominator_without_bh_mass,
            single_host_likelihood_numerator_with_bh_mass,
            single_host_likelihood_denominator_with_bh_mass,
        ]
    return [
        single_host_likelihood_numerator_without_bh_mass,
        single_host_likelihood_denominator_without_bh_mass,
    ]


def child_process_init(
    redshift_lower_limit: float,
    redshift_upper_limit: float,
    bh_mass_lower_limit: float,
    bh_mass_upper_limit: float,
    current_detection_probability: SimulationDetectionProbability,
    current_means_3d: npt.NDArray[np.float64],
    current_cov_inv_3d: npt.NDArray[np.float64],
    current_log_norm_3d: npt.NDArray[np.float64],
    current_means_4d: npt.NDArray[np.float64],
    current_cov_inv_4d: npt.NDArray[np.float64],
    current_log_norm_4d: npt.NDArray[np.float64],
    current_det_index_to_slot: dict[int, int],
    current_sigma2_cond_arr: npt.NDArray[np.float64],
    current_proj_arr: npt.NDArray[np.float64],
    current_det_d_L_arr: npt.NDArray[np.float64],
    current_det_d_L_unc_arr: npt.NDArray[np.float64],
    current_det_M_arr: npt.NDArray[np.float64],
    current_det_phi_arr: npt.NDArray[np.float64],
    current_det_theta_arr: npt.NDArray[np.float64],
    current_D_h_table: dict[float, float] | None = None,
) -> None:
    global redshift_upper_integration_limit
    global redshift_lower_integration_limit
    global bh_mass_upper_integration_limit
    global bh_mass_lower_integration_limit
    global detection_probability
    global means_3d, cov_inv_3d, log_norm_3d
    global means_4d, cov_inv_4d, log_norm_4d
    global det_index_to_slot
    global sigma2_cond_arr, proj_arr
    global det_d_L_arr, det_d_L_unc_arr, det_M_arr, det_phi_arr, det_theta_arr
    global D_h_table

    redshift_upper_integration_limit = redshift_upper_limit
    redshift_lower_integration_limit = redshift_lower_limit
    bh_mass_upper_integration_limit = bh_mass_upper_limit
    bh_mass_lower_integration_limit = bh_mass_lower_limit
    detection_probability = current_detection_probability
    means_3d = current_means_3d
    cov_inv_3d = current_cov_inv_3d
    log_norm_3d = current_log_norm_3d
    means_4d = current_means_4d
    cov_inv_4d = current_cov_inv_4d
    log_norm_4d = current_log_norm_4d
    det_index_to_slot = current_det_index_to_slot
    sigma2_cond_arr = current_sigma2_cond_arr
    proj_arr = current_proj_arr
    det_d_L_arr = current_det_d_L_arr
    det_d_L_unc_arr = current_det_d_L_unc_arr
    det_M_arr = current_det_M_arr
    det_phi_arr = current_det_phi_arr
    det_theta_arr = current_det_theta_arr
    if current_D_h_table is not None:
        D_h_table = current_D_h_table


def _get_closest_possible_host(
    detection: Detection, possible_hosts: list[HostGalaxy]
) -> HostGalaxy:
    distances = [
        _distance_spherical_coordinates(
            phi1=detection.phi,
            theta1=detection.theta,
            phi2=host.phiS,
            theta2=host.qS,
        )
        for host in possible_hosts
    ]
    return possible_hosts[int(np.argmin(distances))]


def _distance_spherical_coordinates(
    phi1: float, theta1: float, phi2: float, theta2: float
) -> float:
    return float(
        np.arccos(
            np.sin(theta1) * np.sin(theta2) + np.cos(theta1) * np.cos(theta2) * np.cos(phi1 - phi2)
        )
    )


def compute_sigma_deviation(
    sigma: float, sigma_error: float, h_mean: float, h_mean_error: float
) -> tuple[float, float]:
    sigma_dev = (h_mean - H) / sigma
    sigma_dev_error = float(np.sqrt((sigma_error * sigma_dev) ** 2 + (h_mean_error) ** 2) / sigma)
    return sigma_dev, sigma_dev_error
