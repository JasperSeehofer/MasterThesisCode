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
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.integrate import dblquad, fixed_quad, quad
from scipy.special import ndtr, roots_hermite, roots_legendre
from scipy.stats import multivariate_normal, norm

from master_thesis_code.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)
from master_thesis_code.constants import (
    CRAMER_RAO_BOUNDS_OUTPUT_PATH,
    HOST_DRAW_Z_MAX,
    INJECTION_DATA_DIR,
    PREPARED_CRAMER_RAO_BOUNDS_PATH,
    SIGMA_V_PEC_KM_S,
    SNR_THRESHOLD,
    SPEED_OF_LIGHT_KM_S,
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

# Issue #40(a) decomposition flag (redteam F2/F3): the in-catalogue NUMERATOR
# host-z kernel, historically bundled into normalization_mode. "auto" preserves
# that bundling exactly (delta kernel iff generator_marginal); "point" /
# "volume_deconv" force the numerator kernel independently of the
# normalization leg (n_hat_w / D_gen machinery, which stays mode-selected).
HOST_Z_KERNEL_CHOICES = ("auto", "point", "volume_deconv")


def resolve_host_z_kernel(host_z_kernel: str, normalization_mode: str) -> str:
    """Resolve the numerator host-z kernel selection to 'point' or 'volume_deconv'.

    Decomposition flag for issue #40(a): makes the delta-kernel (point/point)
    in-catalogue numerator separately selectable from the normalization leg.
    ``"auto"`` reproduces the historical bundling — the delta kernel if and
    only if ``normalization_mode == "generator_marginal"`` — so the production
    default path is unchanged. Explicit ``"point"`` / ``"volume_deconv"``
    override the numerator kernel only; the selection-normalization machinery
    (``n_hat_w``/``D_gen`` vs ``n_bar_w``/``D``) remains governed by
    ``normalization_mode``.

    Args:
        host_z_kernel: One of ``HOST_Z_KERNEL_CHOICES``.
        normalization_mode: The in-catalogue normalization mode (see ``p_Di``).

    Returns:
        ``"point"`` (delta kernel at the catalogue z_g) or ``"volume_deconv"``
        (the mode's own quadrature kernel — volume-deconvolved in the
        *_marginal modes, bare Gaussian in "global"/"local_ratio").
    """
    if host_z_kernel not in HOST_Z_KERNEL_CHOICES:
        raise ValueError(
            f"unknown host_z_kernel: {host_z_kernel!r} (expected one of {HOST_Z_KERNEL_CHOICES})"
        )
    if host_z_kernel == "auto":
        return "point" if normalization_mode == "generator_marginal" else "volume_deconv"
    return host_z_kernel


GALAXY_LIKELIHOODS = "galaxy_likelihoods"
ADDITIONAL_GALAXIES_WITHOUT_BH_MASS = "additional_galaxies_without_bh_mass"

FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD = 0.10

# Fixed-quad order for D(h) precomputation
_DH_QUAD_ORDER: int = 100

# Gauss-Legendre order for the outer z-integral of the with-BH-mass selection
# denominator (the "glz64" semi-analytic estimator). The inner M-integral is
# exact (erf-sum); the only residual error is the outer z-quadrature over the
# p_det d_L-grid kinks, which n=64 pushes to <= 2.8e-4 worst-case (spec-z hosts
# 1e-8..1e-5) -- far below the ~1-5% MC noise it replaces.
_BH_DENOM_QUAD_ORDER: int = 64

# Gauss-Legendre nodes/weights shared by the batched host kernel. Identical to
# what scipy.integrate.fixed_quad uses internally (its _cached_roots_legendre is
# a cache around scipy.special.roots_legendre), so the batched quadrature
# reproduces fixed_quad bit-for-bit per host row.
_HOST_QUAD_N: int = 50
_GL_NODES_50, _GL_WEIGHTS_50 = roots_legendre(_HOST_QUAD_N)
_GL_NODES_64, _GL_WEIGHTS_64 = roots_legendre(_BH_DENOM_QUAD_ORDER)

# --- mass_trunc host-mass kernel (EXP-45, 2026-07-13) --------------------------
# The 2D (with-BH-mass) channel's `mass_trunc` mode replaces the linear-Gaussian
# G2d moment match (eddington_shifted_host_mass) with the TRUE per-galaxy host-mass
# prior: the Reines & Volonteri (2015) lognormal measurement error x the Babak
# et al. (2017) R_eff population weight, TRUNCATED + renormalised on the physical
# EMRI mass window [M_MIN, M_MAX] (the ParameterSpace.M bound; asserted against it
# in the kernel tests to guard drift). Two quadratures:
#   * Gauss-Hermite (weight e^{-t^2}) resolves the NARROW GW M_z peak in the
#     numerator mass-marginal -- placing nodes ON the peak, the exact fix for the
#     fixed_quad(50) aliasing that FALSIFIED volume_trunc (results/volume_trunc_ab_*).
#   * Gauss-Legendre in ln M integrates the SMOOTH normalisation Z_M and the
#     selection-denominator inner-M integral over the wide window.
_MASS_TRUNC_M_MIN: float = 1.0e4
_MASS_TRUNC_M_MAX: float = 1.0e7
_MASS_TRUNC_SIGMA_LNM_FLOOR: float = 1.0e-6
_MASS_TRUNC_GH_ORDER: int = 24
_MASS_TRUNC_GL_ORDER: int = 64
_MT_GH_NODES, _MT_GH_WEIGHTS = roots_hermite(_MASS_TRUNC_GH_ORDER)  # int e^{-t^2} g(t) dt
_MT_GL_NODES, _MT_GL_WEIGHTS = roots_legendre(_MASS_TRUNC_GL_ORDER)  # [-1, 1]

# Normalisation constant of the standard normal pdf; same value scipy.stats.norm
# divides by (scipy.stats._continuous_distns._norm_pdf_C).
_NORM_PDF_C: float = float(np.sqrt(2 * np.pi))

# Upper bound on hosts per batched-kernel chunk (see _starmap_host_batches).
_MAX_BATCH_CHUNK: int = 2048


def _gaussian_pdf(
    x: npt.NDArray[np.float64],
    loc: npt.NDArray[np.float64],
    scale: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Gaussian pdf replicating ``scipy.stats.norm(loc, scale).pdf(x)`` exactly.

    Reproduces scipy's operation order — ``y = (x - loc)/scale`` then
    ``exp(-y**2/2.0)/sqrt(2*pi)/scale`` — so results are bit-identical to the
    frozen-distribution path while skipping its per-construction ``rv_frozen``
    machinery (the profiled ~15-18% ``_construct_doc``/argument-parsing waste).
    All arguments broadcast.

    Args:
        x: Evaluation points.
        loc: Gaussian mean(s).
        scale: Gaussian standard deviation(s), > 0.

    Returns:
        Pdf values, broadcast shape of the inputs.
    """
    y = (x - loc) / scale
    result: npt.NDArray[np.float64] = np.exp(-(y**2) / 2.0) / _NORM_PDF_C / scale
    return result


def _batched_gl_nodes(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    nodes: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Map Gauss-Legendre reference nodes onto per-row integration windows.

    Replicates ``scipy.integrate.fixed_quad``'s affine node map
    ``y = (b - a)*(x + 1)/2.0 + a`` with a leading batch axis: for windows
    ``[a_i, b_i]`` returns the ``(n, len(nodes))`` node array whose row ``i``
    is bit-identical to the nodes fixed_quad would use for ``[a_i, b_i]``.

    Args:
        a: Lower window bounds, shape ``(n,)``.
        b: Upper window bounds, shape ``(n,)``.
        nodes: Gauss-Legendre reference nodes on ``[-1, 1]``.

    Returns:
        Node array of shape ``(n, len(nodes))``.
    """
    result: npt.NDArray[np.float64] = (b - a)[:, None] * (nodes + 1)[None, :] / 2.0 + a[:, None]
    return result


def _batched_gl_reduce(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Per-row Gauss-Legendre reduction replicating ``fixed_quad``'s sum.

    Computes ``(b - a)/2.0 * sum(w * values, axis=-1)`` per row — the same
    expression (and float reduction order, contiguous last axis) that
    ``fixed_quad`` evaluates for a single window.

    Args:
        a: Lower window bounds, shape ``(n,)``.
        b: Upper window bounds, shape ``(n,)``.
        weights: Gauss-Legendre weights, shape ``(k,)``.
        values: Integrand values at the mapped nodes, shape ``(n, k)``.

    Returns:
        Integral per row, shape ``(n,)``.
    """
    result: npt.NDArray[np.float64] = (b - a) / 2.0 * np.sum(weights * values, axis=-1)
    return result


def eddington_shifted_host_mass(host_M: float, host_M_error: float) -> float:
    """Effective host mass under the rate-weighted (Eddington-in-M) prior.

    The per-galaxy mass prior N(M; M_g, sigma_M^2) * R_eff(M) / Z_M is, under a
    locally log-linear R_eff (exponential-tilt identity), EXACTLY the shifted
    Gaussian N(M; M_g (1 + alpha sigma_rel^2), sigma_M^2) with
    ``alpha = dln R_eff / dln M |_{M_g}`` and sigma_rel = sigma_M / M_g.
    Classic Eddington (1913) correction; derivation and curvature-residual
    control in docs/derivations/G2d_host_mass_rate_prior.md (G7 row 9).

    Args:
        host_M: Catalogue (source-frame) host BH mass estimate [M_sun].
        host_M_error: 1-sigma mass uncertainty [M_sun].

    Returns:
        The shifted effective mass M_g^eff [M_sun]; equals host_M when the
        uncertainty is zero/invalid (bare-Gaussian limit).
    """
    if host_M <= 0.0 or host_M_error <= 0.0 or not math.isfinite(host_M_error):
        return host_M
    # EXACT posterior mean of N(M; M_g, sigma^2) * R_eff(M) / Z_M by quadrature
    # (moment matching). The local-slope (log-linear tilt) form gets the SIGN
    # wrong near the kappa_cap low-mass roll-off at GLADE's sigma_rel ~ 1, where
    # R_eff RISES with M — caught by the G2d regression tests.
    sigma = min(host_M_error, 2.0 * host_M)
    lo = max(host_M - 5.0 * sigma, 1e3)
    hi = host_M + 5.0 * sigma
    M_grid = np.linspace(lo, hi, 401)
    w = np.exp(-0.5 * ((M_grid - host_M) / sigma) ** 2) * np.asarray(
        R_eff_per_mbh(M_grid), dtype=np.float64
    )
    Z = float(np.trapezoid(w, M_grid))
    if not math.isfinite(Z) or Z <= 0.0:
        return host_M
    return float(np.trapezoid(M_grid * w, M_grid) / Z)


def _mass_trunc_lnM_weight(
    M: npt.NDArray[np.float64],
    host_M: float | npt.NDArray[np.float64],
    sigma_lnM: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Unnormalised truncated host-mass prior as a density w.r.t. ``d ln M``.

    Returns ``LN(M; M_g, sigma_lnM) * R_eff(M) * M`` (the trailing ``* M`` converts
    the density in ``M`` into a density in ``ln M``, so ``Z_M = int w d ln M``):

    .. math::

        w(\ln M) = \frac{R_\mathrm{eff}(M)}{\sigma_{\ln M}\sqrt{2\pi}}
                   \exp\!\Big[-\tfrac12\big(\tfrac{\ln M-\ln M_g}{\sigma_{\ln M}}\big)^2\Big].

    The caller applies the ``[M_MIN, M_MAX]`` truncation mask (this function does
    not). ``M``, ``host_M``, ``sigma_lnM`` broadcast against each other.

    References:
        Reines & Volonteri (2015), arXiv:1508.06274, Sec. 4.1 (0.24 dex lognormal
        scatter -> Gaussian in ln M_BH); Babak et al. (2017), arXiv:1703.09722
        (per-MBH R_eff population weight).
    """
    ln_ratio = (np.log(M) - np.log(host_M)) / sigma_lnM
    weight: npt.NDArray[np.float64] = (
        np.exp(-0.5 * ln_ratio * ln_ratio)
        * np.asarray(R_eff_per_mbh(M), dtype=np.float64)
        / (sigma_lnM * np.sqrt(2.0 * np.pi))
    )
    return weight


def _mass_trunc_sigma_lnM(
    host_M: float | npt.NDArray[np.float64], host_M_error: float | npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    r"""Recover the lognormal width ``sigma_lnM = host_M_error / host_M``.

    The catalogue stores the *linear* 1-sigma ``host_M_error = M_g * sigma_lnM``
    (``handler._empiric_stellar_mass_to_BH_mass_relation``), i.e. the first-order
    linearisation of the Reines & Volonteri lognormal error. Dividing recovers the
    underlying log-space width the ``mass_trunc`` kernel uses. Floored at
    ``_MASS_TRUNC_SIGMA_LNM_FLOOR`` so ``sigma -> 0`` yields the spec-mass limit.
    """
    return np.maximum(
        np.asarray(host_M_error, dtype=np.float64) / np.asarray(host_M, dtype=np.float64),
        _MASS_TRUNC_SIGMA_LNM_FLOOR,
    )


_MASS_TRUNC_LNM_HALF_WIDTH: float = 10.0  # +/- N sigma_lnM lnM integration window


def _mass_trunc_lnM_window(
    host_M: float | npt.NDArray[np.float64], sigma_lnM: float | npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Per-host ``[ln_lo, ln_hi]`` integration window: the prior peak +/- N sigma_lnM,
    clipped to ``[ln M_MIN, ln M_MAX]``.

    The truncated lognormal x R_eff prior is negligible (``exp(-N^2/2)``) outside
    ``ln M_g +/- N sigma_lnM``, so centring the ``ln M`` quadrature on the peak (i)
    respects the ``[M_MIN, M_MAX]`` truncation and (ii) RESOLVES the peak for ANY
    ``sigma_lnM`` -- a full-window Gauss-Legendre would miss a narrow spike (the
    same peak-aliasing that falsified volume_trunc). The centre is clipped so the
    window stays valid even for a host mass at/beyond a bound. Returns two arrays
    broadcasting to the shape of ``host_M`` / ``sigma_lnM``.
    """
    ln_min = math.log(_MASS_TRUNC_M_MIN)
    ln_max = math.log(_MASS_TRUNC_M_MAX)
    ln_mg = np.clip(np.log(np.asarray(host_M, dtype=np.float64)), ln_min, ln_max)
    half_w = _MASS_TRUNC_LNM_HALF_WIDTH * np.asarray(sigma_lnM, dtype=np.float64)
    return np.maximum(ln_min, ln_mg - half_w), np.minimum(ln_max, ln_mg + half_w)


def _mass_trunc_log_normalisation(
    host_M: float | npt.NDArray[np.float64], sigma_lnM: float | npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    r"""Per-host normalisation ``Z_M = int LN(M;M_g,sigma) R_eff(M) dM`` (truncated).

    Gauss-Legendre in ``u = ln M`` over the peak-aware window
    (:func:`_mass_trunc_lnM_window`). ``host_M`` / ``sigma_lnM`` are scalar or shape
    ``(n,)``; the result carries a trailing size matching their broadcast shape
    (a length-1 array for scalar input -- callers take ``.item()``).
    """
    ln_lo, ln_hi = _mass_trunc_lnM_window(host_M, sigma_lnM)  # (...,)
    half = 0.5 * (ln_hi - ln_lo)
    mid = 0.5 * (ln_hi + ln_lo)
    M_nodes = np.exp(mid[..., None] + half[..., None] * _MT_GL_NODES)  # (..., G)
    hM = np.asarray(host_M, dtype=np.float64)[..., None]  # (..., 1)
    sg = np.asarray(sigma_lnM, dtype=np.float64)[..., None]  # (..., 1)
    w = _mass_trunc_lnM_weight(M_nodes, hM, sg)  # (..., G)
    z_m: npt.NDArray[np.float64] = half * np.sum(w * _MT_GL_WEIGHTS, axis=-1)  # (...,)
    return z_m


def _mass_trunc_mz_integral(
    mu_cond: npt.NDArray[np.float64],
    sigma_cond: float,
    one_plus_z: npt.NDArray[np.float64],
    det_M: float,
    host_M: float | npt.NDArray[np.float64],
    sigma_lnM: float | npt.NDArray[np.float64],
    Z_M: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Mass-marginal factor of the with-BH-mass numerator, ``mass_trunc`` kernel.

    Replaces the analytic Gaussian-product ``mz_integral`` (linear-Gaussian mass
    prior) with

    .. math::

        \int \mathcal{N}\big(a;\mu_\mathrm{cond},\sigma_\mathrm{cond}\big)\,p_M(M)\,dM,
        \qquad a = M(1+z)/M_\mathrm{det},

    where ``p_M`` is the truncated lognormal x R_eff prior. The GW factor is a sharp
    Gaussian in ``a``; substituting ``a = mu_cond + sqrt(2) sigma_cond t`` gives the
    exact Gauss-Hermite form (A&S 25.4.46) -- nodes land ON the GW peak, so no
    aliasing over the wide mass window:

    .. math::

        \mathrm{mz} = \frac{1}{\sqrt\pi}\sum_k w_k^\mathrm{GH}\,p_M(M_k)\,
                      \frac{M_\mathrm{det}}{1+z},\quad
        M_k = \big(\mu_\mathrm{cond}+\sqrt2\,\sigma_\mathrm{cond}\,t_k\big)\frac{M_\mathrm{det}}{1+z}.

    ``mu_cond`` / ``one_plus_z`` are the per-z-node arrays ``(..., K)``; ``host_M`` /
    ``sigma_lnM`` / ``Z_M`` are scalar (scalar path) or ``(n,)`` (batch, leading
    axis = ``mu_cond.shape[:-1]``). Returns ``(..., K)``.
    """
    a = mu_cond[..., None] + np.sqrt(2.0) * sigma_cond * _MT_GH_NODES  # (..., K, G)
    opz = one_plus_z[..., None]  # (..., K, 1)
    M = a * det_M / opz  # (..., K, G) rest-frame mass at each GH node
    inside = (M >= _MASS_TRUNC_M_MIN) & (M <= _MASS_TRUNC_M_MAX)
    M_safe = np.where(inside, M, _MASS_TRUNC_M_MIN)  # keep logs finite; masked below
    # Host params -> (..., 1, 1) to broadcast against M of shape (..., K, G).
    hM = np.asarray(host_M, dtype=np.float64).reshape(np.shape(host_M) + (1, 1))
    sg = np.asarray(sigma_lnM, dtype=np.float64).reshape(np.shape(sigma_lnM) + (1, 1))
    ZM = np.asarray(Z_M, dtype=np.float64).reshape(np.shape(Z_M) + (1, 1))
    # p_M(M) as a density in M: LN*R_eff/Z_M = (lnM-weight)/(M Z_M); 0 outside window.
    p_M = np.where(inside, _mass_trunc_lnM_weight(M_safe, hM, sg) / (M_safe * ZM), 0.0)
    p_a = p_M * det_M / opz  # push forward to the a coordinate (|dM/da|)
    mz: npt.NDArray[np.float64] = (p_a @ _MT_GH_WEIGHTS) / np.sqrt(np.pi)  # (..., K)
    return mz


def _mass_trunc_denominator_inner_m_integral(
    z: npt.NDArray[np.float64],
    detection_probability: Any,
    host_phiS: float,
    host_qS: float,
    host_M: float,
    sigma_lnM: float,
    Z_M: float,
    h: float,
) -> npt.NDArray[np.float64]:
    r"""Inner mass integral of the with-BH-mass selection denominator, ``mass_trunc``.

    Returns, per redshift ``z_j``,
    ``g(z) = int p_det(d_L(z), M(1+z)) p_M(M) dM`` with the truncated lognormal x
    R_eff prior. Gauss-Legendre in ``ln M`` over the peak-aware window
    (:func:`_mass_trunc_lnM_window`, the SAME support as ``Z_M``); the erf-sum
    closed form (Gaussian-prior only) does not apply. p_det is evaluated at
    ``(d_L(z), M(1+z))`` via the same interpolator the erf-sum path uses.
    """
    z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))  # (n_z,)
    ln_lo, ln_hi = _mass_trunc_lnM_window(host_M, sigma_lnM)  # scalars
    half = 0.5 * (ln_hi - ln_lo)
    mid = 0.5 * (ln_hi + ln_lo)
    M_nodes = np.exp(mid + half * _MT_GL_NODES)  # (G,)
    n_z, n_g = z_arr.size, M_nodes.size
    d_L = dist_vectorized(z_arr, h=h)  # (n_z,)
    m_z = M_nodes[None, :] * (1.0 + z_arr)[:, None]  # (n_z, G) detector-frame mass
    p = np.asarray(
        detection_probability.detection_probability_with_bh_mass_interpolated(
            np.repeat(d_L, n_g),
            m_z.reshape(-1),
            np.full(n_z * n_g, host_phiS),
            np.full(n_z * n_g, host_qS),
            h=h,
        ),
        dtype=np.float64,
    ).reshape(n_z, n_g)
    w = _mass_trunc_lnM_weight(M_nodes, host_M, sigma_lnM) / Z_M  # (G,) normalised p_M dlnM
    inner_m: npt.NDArray[np.float64] = half * ((p * w[None, :]) @ _MT_GL_WEIGHTS)  # (n_z,)
    return inner_m


def _mass_trunc_denominator_inner_m_integral_batch(
    z: npt.NDArray[np.float64],
    detection_probability: Any,
    host_phiS: npt.NDArray[np.float64],
    host_qS: npt.NDArray[np.float64],
    host_M: npt.NDArray[np.float64],
    sigma_lnM: npt.NDArray[np.float64],
    Z_M: npt.NDArray[np.float64],
    h: float,
) -> npt.NDArray[np.float64]:
    """Host-batched twin of :func:`_mass_trunc_denominator_inner_m_integral`.

    ``z`` has shape ``(n, n_z)``; host parameters have shape ``(n,)``. Row ``i`` is
    bit-identical to the scalar function called with ``z[i]`` and host ``i``'s
    parameters -- one ``p_det`` interpolator call covers all ``n * n_z * G`` points.
    Per-host peak-aware ``ln M`` window (same as ``Z_M``).
    """
    n, n_z = z.shape
    ln_lo, ln_hi = _mass_trunc_lnM_window(host_M, sigma_lnM)  # (n,), (n,)
    half = 0.5 * (ln_hi - ln_lo)  # (n,)
    mid = 0.5 * (ln_hi + ln_lo)  # (n,)
    M_nodes = np.exp(mid[:, None] + half[:, None] * _MT_GL_NODES)  # (n, G)
    n_g = M_nodes.shape[1]
    d_L = dist_vectorized(z.reshape(-1), h=h)  # (n*n_z,)
    m_z = M_nodes[:, None, :] * (1.0 + z[:, :, None])  # (n, n_z, G)
    p = np.asarray(
        detection_probability.detection_probability_with_bh_mass_interpolated(
            np.repeat(d_L, n_g),
            m_z.reshape(-1),
            np.repeat(host_phiS, n_z * n_g),
            np.repeat(host_qS, n_z * n_g),
            h=h,
        ),
        dtype=np.float64,
    ).reshape(n, n_z, n_g)
    w = (
        _mass_trunc_lnM_weight(M_nodes, host_M[:, None], sigma_lnM[:, None]) / Z_M[:, None]
    )  # (n, G) normalised p_M dlnM
    inner_m: npt.NDArray[np.float64] = half[:, None] * ((p * w[:, None, :]) @ _MT_GL_WEIGHTS)
    return inner_m  # (n, n_z)


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


def _sky_aware_selection_available(
    completeness: CompletenessModel | None,
    detection_probability_obj: SimulationDetectionProbability,
) -> bool:
    r"""True iff both objects support the sky-resolved selection path (Change 1-4).

    Requires the detection-probability object to expose the ecliptic-latitude
    band survival (:meth:`survival_per_band`, :meth:`band_edges_sin_beta`) AND a
    per-pixel completeness (:meth:`pixel_centers`).  When either is absent
    (e.g. a mock ``p_det`` or the all-sky :class:`GladeCatalogCompleteness`), the
    selection integrals fall back to the EXACT sky-marginalised formulas -- which
    is also the ``n_sky_bands == 1`` isotropic limit (test T1).
    """
    return (
        completeness is not None
        and hasattr(completeness, "pixel_centers")
        and hasattr(completeness, "f_pixels")
        and hasattr(detection_probability_obj, "survival_per_band")
        and hasattr(detection_probability_obj, "band_edges_sin_beta")
    )


def _sky_band_pixel_map(
    completeness: CompletenessModel,
    detection_probability_obj: SimulationDetectionProbability,
) -> tuple[npt.NDArray[np.int_], int, int]:
    r"""Assign every HEALPix pixel centre to a ``p_det`` ecliptic-latitude band.

    Uses the SAME equal-|sin beta| band edges as the injection ``p_det`` build
    (:meth:`SimulationDetectionProbability.band_edges_sin_beta`) so the sky
    marginal is invariant (test T3).  ``beta = pi/2 - theta`` =>
    ``|sin beta| = |cos theta|``.  The sky prior is uniform ``1/Npix`` (equal-area
    pixels): pixels are counted, NOT galaxy-weighted (guardrail).

    Returns
    -------
    (band_of_pixel, n_bands, npix)
    """
    phi_k, theta_k = completeness.pixel_centers()  # type: ignore[attr-defined]
    sin_beta_abs = np.abs(np.cos(np.asarray(theta_k, dtype=np.float64)))  # |sin beta|
    edges = np.asarray(detection_probability_obj.band_edges_sin_beta(), dtype=np.float64)
    n_bands = int(edges.size - 1)
    band_of_pixel = np.clip(
        np.searchsorted(edges, sin_beta_abs, side="right") - 1, 0, n_bands - 1
    ).astype(np.int_)
    return band_of_pixel, n_bands, int(sin_beta_abs.size)


def _zres_z_kwargs(
    detection_probability_obj: Any,
    z: float | npt.NDArray[np.float64],
) -> dict[str, Any]:
    r"""FIX-2 pass-through: the conditioning redshift for z-resolved p_det queries.

    [PHYSICS] z-resolved detection survival (E1 FIX-2): when the detection-
    probability object is built with ``pdet_z_resolved=True``, every 3D
    (without-BH-mass) survival query must be conditioned on the redshift the
    caller is already holding — ``S(d_L(z;h) | z)`` replaces the pooled
    ``S(d_L(z;h))`` COHERENTLY across all selection integrals (D, beta_Gbar,
    Sigma_glob incl. the smeared branch, per-host D_g, sky-band variants).
    When the flag is off (or a mock p_det without the attribute is used), the
    call is byte-identical to the pre-FIX-2 form (no ``z`` keyword passed).

    References:
        Finn & Chernoff (1993), arXiv:gr-qc/9301003; Finn (1996),
            arXiv:gr-qc/9601048 — horizon-survival p_det.
        Mandel, Farr & Gair (2019), arXiv:1809.02063 — selection evaluated at
            the population AT HYPOTHESIS, which specifies z.
        results/lcat_h_dependence_20260725/DERIVATION_ZRESOLVED_SURVIVAL.md
            §5.1 (consumer coherence rule: pass the z you are already holding).
    """
    # `is True` (not truthiness): MagicMock test doubles auto-create truthy
    # attributes; only the real boolean property may activate the pass-through.
    if getattr(detection_probability_obj, "z_resolved", False) is True:
        return {"z": z}
    return {}


def precompute_completion_denominator(
    h_values: list[float],
    detection_probability_obj: SimulationDetectionProbability,
    Omega_m: float,
    Omega_DE: float,
    *,
    completeness: CompletenessModel | None = None,
    quad_n: int = _DH_QUAD_ORDER,
    z_max_cap: float | None = None,
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

    # Change 2: sky-resolved full-volume selection.  When the sky-aware path is
    # available, D(h) = INTEGRAL (1/Npix) sum_k p_det(d_L(z,h), Omega_k) dVc/(1+z) dz
    # is evaluated efficiently as sum_b (n_pix_b/Npix) S_b(d_L(z,h)) -- p_det
    # depends on Omega only through band(beta) (equal-solid-angle sky sum).
    # Gray, Gerosa et al. (2023), arXiv:2308.02281, Eq. (2.3) -- per-pixel GW
    # selection sum; Mandel-Farr-Gair (2019), arXiv:1809.02063, Eq. 6.
    _sky_aware = _sky_aware_selection_available(completeness, detection_probability_obj)
    if _sky_aware:
        assert completeness is not None
        _band_of_pixel, _n_bands, _npix = _sky_band_pixel_map(
            completeness, detection_probability_obj
        )
        # c_b = n_pix_b / Npix : uniform-sky (equal-area) fraction per band.
        _c_b = np.bincount(_band_of_pixel, minlength=_n_bands).astype(np.float64) / float(_npix)

    for h in h_values:
        dl_max = detection_probability_obj.get_dl_max(h)
        z_max = dist_to_redshift(dl_max, h=h)
        # [PHYSICS] Selection-domain cap (issue #30): keep the selection integrals
        # on the SAME z-domain as the numerator-side candidate window (p_D caps its
        # BallTree z-window at max_redshift), so an analysis truncation moves
        # numerator and denominator TOGETHER and beta_G = D - beta_Gbar remains an
        # identity on one domain. No-op at current constants: the p_det horizon
        # z_max(h) <= ~1.33 for h in [0.60, 0.86] < max_redshift = 1.5.
        # Mandel, Farr & Gair (2019), arXiv:1809.02063 (selection function must
        # match the event-inclusion criterion).
        if z_max_cap is not None:
            z_max = min(z_max, z_max_cap)
        z_min = 1e-6

        def _denom_integrand(
            z: npt.NDArray[np.float64],
            _h: float = h,
        ) -> npt.NDArray[np.float64]:
            d_L: npt.NDArray[np.float64] = np.asarray(
                dist_vectorized(z, h=_h), dtype=np.float64
            )  # Gpc
            if _sky_aware:
                # (1/Npix) sum_k p_det(Omega_k) = sum_b (n_pix_b/Npix) S_b(d_L).
                # Gray 2023 arXiv:2308.02281 Eq. 2.3 (per-pixel selection sum).
                # FIX-2: S_b(d_L | z) at the quadrature node's own z.
                s_band = np.asarray(
                    detection_probability_obj.survival_per_band(
                        d_L, **_zres_z_kwargs(detection_probability_obj, z)
                    ),
                    dtype=np.float64,
                )  # (n_bands, Z)
                p_det: npt.NDArray[np.float64] = _c_b @ s_band  # (Z,)
            else:
                phi = np.zeros_like(z)  # marginalized; value does not matter
                theta = np.zeros_like(z)
                p_det = np.asarray(
                    detection_probability_obj.detection_probability_without_bh_mass_interpolated_zero_fill(
                        d_L, phi, theta, h=_h, **_zres_z_kwargs(detection_probability_obj, z)
                    ),
                    dtype=np.float64,
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
    z_max_cap: float | None = None,
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

    # Change 3: sky-resolved missing-completion selection.  When the sky-aware
    # path is available this evaluates the caveat's own prescription
    # beta_Gbar(h) = INTEGRAL (1/Npix) sum_k (1 - f_k(z,h)) p_det(d_L(z,h), Omega_k)
    #                dVc/(1+z) dz
    # efficiently as sum_b S1mf_b(z) S_b(d_L), with the per-band incompleteness
    # sum S1mf_b(z) = (1/Npix) sum_{k in band b}(1 - f_k(z)).  ZoA/empty pixels
    # (f_k=0) contribute the FULL p_det(Omega_k) -- exactly where dark hosts
    # concentrate.  Gray et al. (2020), arXiv:1908.06050, Eq. (33);
    # Gray-Messenger-Veitch (2022), arXiv:2111.04629, Eq. (5).
    _sky_aware = _sky_aware_selection_available(completeness, detection_probability_obj)
    if _sky_aware:
        _band_of_pixel, _n_bands, _npix = _sky_band_pixel_map(
            completeness, detection_probability_obj
        )
        # Boolean (n_bands, npix) membership for the per-band pixel reduction.
        _band_membership = _band_of_pixel[None, :] == np.arange(_n_bands)[:, None]

    for h in h_values:
        dl_max = detection_probability_obj.get_dl_max(h)
        z_max = dist_to_redshift(dl_max, h=h)
        # [PHYSICS] Selection-domain cap (issue #30) — same domain as D(h); see
        # precompute_completion_denominator. No-op at current constants.
        if z_max_cap is not None:
            z_max = min(z_max, z_max_cap)
        z_min = 1e-6

        def _missing_denom_integrand(
            z: npt.NDArray[np.float64],
            _h: float = h,
        ) -> npt.NDArray[np.float64]:
            d_L: npt.NDArray[np.float64] = np.asarray(
                dist_vectorized(z, h=_h), dtype=np.float64
            )  # Gpc
            dVc: npt.NDArray[np.float64] = np.atleast_1d(
                np.asarray(comoving_volume_element(z, h=_h), dtype=np.float64)
            )
            if _sky_aware:
                # Per-pixel (1 - f_k(z)) summed per band, divided by Npix, then
                # weighted by that band's survival S_b(d_L).  Sky-uniform prior
                # 1/Npix (equal-area pixels). Gray 2023 arXiv:2308.02281 Eq. 2.3;
                # GMV 2022 arXiv:2111.04629 Eq. 5.
                f_pix = np.clip(
                    np.asarray(completeness.f_pixels(z, _h), dtype=np.float64),  # type: ignore[attr-defined]
                    0.0,
                    1.0,
                )  # (Z, npix)
                one_minus_f = 1.0 - f_pix  # (Z, npix)
                # S1mf_b(z) = (1/Npix) sum_{k in band b}(1 - f_k(z)) -> (n_bands, Z)
                s1mf_b = (_band_membership.astype(np.float64) @ one_minus_f.T) / float(_npix)
                # FIX-2: S_b(d_L | z) at the quadrature node's own z.
                s_band = np.asarray(
                    detection_probability_obj.survival_per_band(
                        d_L, **_zres_z_kwargs(detection_probability_obj, z)
                    ),
                    dtype=np.float64,
                )  # (n_bands, Z)
                integrand = np.einsum("bz,bz->z", s1mf_b, s_band)
                return np.asarray(integrand, dtype=np.float64) * dVc / (1.0 + z)
            # Isotropic fallback: (1 - f_bar(z)) <p_det>_iso (the exact
            # n_sky_bands==1 limit).  Valid because p_det is sky-uniform here.
            phi = np.zeros_like(z)  # sky-marginalized; matches D(h)
            theta = np.zeros_like(z)
            p_det = np.asarray(
                detection_probability_obj.detection_probability_without_bh_mass_interpolated_zero_fill(
                    d_L, phi, theta, h=_h, **_zres_z_kwargs(detection_probability_obj, z)
                ),
                dtype=np.float64,
            )
            f_z = np.clip(
                np.asarray(completeness.f_bar(z, _h), dtype=np.float64),
                0.0,
                1.0,
            )
            return (1.0 - f_z) * p_det * dVc / (1.0 + z)

        beta_Gbar: float = fixed_quad(_missing_denom_integrand, z_min, z_max, n=quad_n)[0]
        beta_Gbar_table[h] = beta_Gbar
        _LOGGER.info(
            "beta_Gbar(h=%.4f) = %.6e  [z_max=%.4f]",
            h,
            beta_Gbar,
            z_max,
        )

    return beta_Gbar_table


def compute_catalog_draw_weight_total(
    galaxy_catalog: GalaxyCatalogueHandler,
    z_max: float = HOST_DRAW_Z_MAX,
) -> float:
    r"""Total draw-eligible catalogue rate weight ``W_cat`` (h-independent scalar).

    .. math::

        W_\mathrm{cat} = \sum_{g:\, z_g < z_\mathrm{max}} w_g,
        \qquad w_g = \frac{R_\mathrm{eff}(M_g)}{1 + z_g},

    over the SAME pruned catalogue rows and the SAME eligibility mask
    (``z_g < HOST_DRAW_Z_MAX``, no other cut) that the generator's in-catalogue
    host draw normalizes over — this is exactly ``total_weight`` in
    :meth:`~master_thesis_code.galaxy_catalogue.handler.GalaxyCatalogueHandler.draw_rate_weighted_hosts`.
    It is the draw-side companion normalizer of the completeness-weighted
    population volume :func:`precompute_completeness_population_volume`; their
    ratio ``n_hat_w = W_cat / V_f(h)`` is the generator-consistent rate-weight
    density that replaces the Option-A calibration ``n_bar_w = Sigma_glob/beta_G``
    in the ``generator_marginal`` normalization mode.

    ``W_cat`` carries NO ``P_det`` and NO ``h`` dependence: it normalizes the
    draw, not the detection (domain note in the derivation packet §3.2). Any
    analysis-depth cap (issue-#30 ``z_max_cap``) must be applied to ``z_max``
    HERE and in ``V_f`` together with the candidate window (f29a5e7 principle:
    numerator and denominator move together).

    Args:
        galaxy_catalog: Loaded catalogue handler (its ``reduced_galaxy_catalog``
            is summed over; same rows the rate-weighted draw uses).
        z_max: Exclusive upper redshift bound of the draw eligibility. Defaults
            to :data:`~master_thesis_code.constants.HOST_DRAW_Z_MAX`.

    Returns:
        ``W_cat`` in ``yr^-1`` (the ``emri_rate.C_NORM`` scale cancels in every
        ratio it enters).

    References:
        - results/lcat_h_dependence_20260725/DERIVATION_GENERATOR_CONSISTENT_NORM.md
          §2.3 Eq. (4) (spec; W_cat anchor: 6.3477e8 over 9,060,017 pruned rows).
        - master_thesis_code/galaxy_catalogue/handler.py, draw_rate_weighted_hosts
          (the generator draw this normalizer replicates).
        - Babak et al. (2017), arXiv:1703.09722 — per-MBH rate ``R_eff``.
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
    # Draw eligibility: z_g < z_max ONLY — the exact mask of
    # draw_rate_weighted_hosts (no mass mask; the catalogue is already pruned).
    eligible = z_all < z_max
    z_g = z_all[eligible]
    M_g = M_all[eligible]
    if z_g.size == 0:
        raise ValueError(
            f"No galaxy in the reduced catalog has redshift < z_max = {z_max}; "
            "cannot form the draw-side rate-weight total W_cat."
        )
    # w_g = R_eff_per_mbh(M_g)/(1+z_g): IDENTICAL to draw_rate_weighted_hosts,
    # _rate_weight and precompute_global_catalog_selection.
    # Eq. (4) in DERIVATION_GENERATOR_CONSISTENT_NORM.md §2.3.
    w_g = np.asarray(R_eff_per_mbh(M_g), dtype=np.float64) / (1.0 + z_g)
    W_cat = float(np.sum(w_g))
    _LOGGER.info(
        "Catalog draw-weight total W_cat = %.6e yr^-1 over %d eligible galaxies (z < %.4f)",
        W_cat,
        z_g.size,
        z_max,
    )
    return W_cat


def precompute_completeness_population_volume(
    h_values: list[float],
    completeness: CompletenessModel,
    *,
    z_min: float = 1e-6,
    z_max: float = HOST_DRAW_Z_MAX,
    n_grid: int = 4096,
) -> dict[float, float]:
    r"""Completeness-weighted population volume ``V_f(h)`` (per steradian).

    .. math::

        V_f(h) = \int_{z_\mathrm{min}}^{z_\mathrm{max}} \bar f(z, h)\,
                 \frac{1}{1+z}\,\frac{dV_c}{dz\,d\Omega}\, dz ,

    the SAME integral the generator's in-catalogue mixture fraction ``F`` uses
    (``F = V_f / V_tot``, :func:`master_thesis_code.dark_siren_injection.compute_global_catalog_fraction`),
    with the SAME sky-averaged completeness ``f_bar`` and the SAME trapezoid
    quadrature convention (linspace grid, ``_DEFAULT_Z_GRID_POINTS = 4096``
    nodes there; the completeness curve is piecewise linear, so the trapezoid
    rule is exact-in-shape and more robust than Gauss-Legendre on the kinks).
    Under the frozen pixel ``m_th`` map ``f_bar`` is h-invariant, so
    ``V_f(h) = V_f(0.73) (0.73/h)^3`` exactly (``dV_c/dz`` carries the ``h^-3``);
    the table is still evaluated per-h so a future h-dependent completeness
    model flows through unchanged (derivation packet §2.2 note).

    Role: ``n_hat_w(h) = W_cat / V_f(h)`` is the generator draw-side rate-weight
    density of the ``generator_marginal`` mode — the calibration contains NO
    ``P_det``, so the Option-A identity ``Sigma_glob = n_hat_w * beta_G`` is
    never invoked (the whole point of the mode).

    Args:
        h_values: Hubble parameter values to evaluate.
        completeness: Catalogue completeness model (``f_bar`` accessor), the
            SAME frozen-cache object the generator uses (C1 consistency).
        z_min: Lower integration bound (matches the generator's
            ``_DEFAULT_Z_MIN = 1e-6``).
        z_max: Upper integration bound — the DRAW depth
            :data:`~master_thesis_code.constants.HOST_DRAW_Z_MAX`, NOT the
            detection horizon ``z_max(h)`` (domain note, derivation §3.2). An
            issue-#30 depth cap must move this together with ``W_cat``.
        n_grid: Trapezoid grid nodes (default 4096, the generator convention).

    Returns:
        Dict mapping ``h -> V_f(h)`` in ``Mpc^3 sr^-1`` (same measure as
        ``D(h)``/``beta_Gbar(h)``). Validation anchor:
        ``V_f(0.73) = 2.3237e8`` (generator_norm_Vf_tables.json).

    References:
        - results/lcat_h_dependence_20260725/DERIVATION_GENERATOR_CONSISTENT_NORM.md
          §2.2-§2.3, Eq. (4) (spec and numeric anchors).
        - master_thesis_code/dark_siren_injection.py,
          compute_global_catalog_fraction — the generator integral replicated.
        - Gray et al. (2020), arXiv:1908.06050, Eq. (9) — completeness ``f(z)``.
    """
    V_f_table: dict[float, float] = {}
    z_grid = np.linspace(z_min, z_max, n_grid, dtype=np.float64)
    for h in h_values:
        f_z = np.clip(np.asarray(completeness.f_bar(z_grid, h), dtype=np.float64), 0.0, 1.0)
        # (1/(1+z)) dVc/dz: the generator's _redshift_population_weight (the
        # mass-integrated rate is z-independent under p0=1 and cancels).
        # Eq. (4) in DERIVATION_GENERATOR_CONSISTENT_NORM.md §2.3.
        w_pop = np.asarray(comoving_volume_element(z_grid, h=h), dtype=np.float64) / (1.0 + z_grid)
        V_f = float(np.trapezoid(f_z * w_pop, z_grid))
        if not (V_f > 0.0):
            raise ValueError(
                f"Completeness population volume V_f(h={h}) = {V_f} is non-positive; "
                "cannot form the draw-side rate-weight density n_hat_w."
            )
        V_f_table[h] = V_f
        _LOGGER.info("V_f(h=%.4f) = %.6e Mpc^3/sr  [z in (%.1e, %.4f)]", h, V_f, z_min, z_max)
    return V_f_table


def _smeared_global_pdet_expectation(
    z_g: npt.NDArray[np.float64],
    M_g: npt.NDArray[np.float64],
    z_err_g: npt.NDArray[np.float64],
    theta_g: npt.NDArray[np.float64] | None,
    h: float,
    detection_probability_obj: SimulationDetectionProbability,
    *,
    with_bh_mass: bool,
    sky_aware: bool,
    n_quad: int = 50,
    chunk_size: int = 200_000,
) -> npt.NDArray[np.float64]:
    r"""Per-galaxy sigma_z-smeared selection weight ``E_{z~kernel_g}[P_det(d_L(z;h))]``.

    [PHYSICS] num/denom sigma_z symmetry (issue #30 estimator redesign, risk R4):
    ``Sigma_global``'s point evaluation ``P_det(d_L(z_g;h))`` is replaced by the
    expectation over the SAME volume-deconvolved host-z kernel the in-catalogue
    numerator ``N_g`` uses (``single_host_likelihood``):

    .. math::

        p_g(z) \propto \mathcal{N}(z; z_g, \sigma_{\mathrm{eff},g})\,
        \frac{dV_c/dz}{1+z},\qquad
        \sigma_{\mathrm{eff},g}^2 = \sigma_{z,g}^2
            + \bigl((1+z_g)\,\sigma_{v,\mathrm{pec}}/c\bigr)^2,

    integrated by Gauss-Legendre (n=50, the numerator's ``FIXED_QUAD_N``) over
    ``[max(z_g - 4 sigma_eff, 1e-6), z_g + 4 sigma_eff]`` — window, floor, and
    PV-inflation all mirrored from the numerator kernel. The NORMALIZED kernel is
    exactly h-invariant (``dV_c/dz = h^{-3} f(z)`` cancels in the per-galaxy
    normalization), so smearing changes only the ``P_det`` realization, never the
    kernel itself. Limiting case ``sigma_eff -> 0``: the kernel collapses to
    ``delta(z - z_g)`` and the point-evaluated form is recovered exactly.

    With-BH-mass channel: the observer-frame mass tracks the smeared redshift,
    ``M_z(z) = M_g (1+z)`` (consistent z-propagation). The galaxy MASS-ERROR
    kernel of the numerator is intentionally NOT mirrored here (pre-existing
    point-``M_g`` treatment retained; tracked separately under issue #24).

    References:
        results/lcat_h_dependence_20260725/DERIVATION_ESTIMATOR_REDESIGN.md
            §3.3 (measured n_bar_w residual) + §7 risk R4 (this remediation).
        Gray et al. (2020), arXiv:1908.06050, Eqs. A.10/33 (kernel form, as in
            the numerator).
    """
    x_nodes, x_weights = roots_legendre(n_quad)
    x_nodes = np.asarray(x_nodes, dtype=np.float64)
    x_weights = np.asarray(x_weights, dtype=np.float64)
    out = np.empty_like(z_g)
    sigma_z_pv = (1.0 + z_g) * SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S
    # Tiny floor keeps the affine window non-degenerate; at 1e-10 the kernel is
    # numerically a delta and the expectation equals the point evaluation.
    sigma_eff = np.maximum(np.sqrt(z_err_g**2 + sigma_z_pv**2), 1e-10)
    for start in range(0, z_g.size, chunk_size):
        sl = slice(start, min(start + chunk_size, z_g.size))
        zc = z_g[sl]
        se = sigma_eff[sl]
        lo = np.maximum(zc - 4.0 * se, 1e-6)  # numerator's z-floor (1e-6)
        hi = np.maximum(zc + 4.0 * se, lo + 1e-12)
        c = 0.5 * (hi + lo)
        s = 0.5 * (hi - lo)
        z_nodes = c[:, None] + s[:, None] * x_nodes[None, :]  # (n, K)
        gauss = np.exp(-0.5 * ((z_nodes - zc[:, None]) / se[:, None]) ** 2)
        w_pop = np.asarray(comoving_volume_element(z_nodes.ravel(), h=h), dtype=np.float64).reshape(
            z_nodes.shape
        ) / (1.0 + z_nodes)
        kern = gauss * w_pop * (s[:, None] * x_weights[None, :])
        norm_g = np.sum(kern, axis=1)
        norm_g = np.where(norm_g > 0.0, norm_g, 1.0)
        d_L_nodes = np.asarray(dist_vectorized(z_nodes.ravel(), h=h), dtype=np.float64)
        zeros = np.zeros_like(d_L_nodes)
        if with_bh_mass:
            M_z_nodes = (M_g[sl][:, None] * (1.0 + z_nodes)).ravel()
            p_nodes = np.asarray(
                detection_probability_obj.detection_probability_with_bh_mass_interpolated(
                    d_L_nodes, M_z_nodes, zeros, zeros, h=h
                ),
                dtype=np.float64,
            )
        elif sky_aware and theta_g is not None:
            sin_beta = np.abs(np.cos(theta_g[sl]))
            _edges = np.asarray(detection_probability_obj.band_edges_sin_beta(), dtype=np.float64)
            _n_bands = int(_edges.size - 1)
            band = np.clip(np.searchsorted(_edges, sin_beta, side="right") - 1, 0, _n_bands - 1)
            # FIX-2: the smear kernel and the conditioning coordinate are the
            # SAME z (packet §5.1): E_{z~kernel_g}[S(d_L(z;h) | z, band)] —
            # the expectation stays outside.
            s_band = np.asarray(
                detection_probability_obj.survival_per_band(
                    d_L_nodes,
                    **_zres_z_kwargs(detection_probability_obj, z_nodes.ravel()),
                ),
                dtype=np.float64,
            )  # (n_bands, n*K)
            band_rep = np.repeat(band, n_quad)
            p_nodes = s_band[band_rep, np.arange(band_rep.size)]
        else:
            p_nodes = np.asarray(
                detection_probability_obj.detection_probability_without_bh_mass_interpolated_zero_fill(
                    d_L_nodes,
                    zeros,
                    zeros,
                    h=h,
                    **_zres_z_kwargs(detection_probability_obj, z_nodes.ravel()),
                ),
                dtype=np.float64,
            )
        out[sl] = np.sum(kern * p_nodes.reshape(z_nodes.shape), axis=1) / norm_g
    return out


def precompute_global_catalog_selection(
    h_values: list[float],
    galaxy_catalog: GalaxyCatalogueHandler,
    detection_probability_obj: SimulationDetectionProbability,
    *,
    with_bh_mass: bool,
    z_max_cap: float | None = None,
    smear_sigma_z: bool = False,
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
    if smear_sigma_z:
        if InternalCatalogColumns.REDSHIFT_ERROR not in catalog.columns:
            raise ValueError(
                "smear_sigma_z=True requires the catalogue REDSHIFT_MEASUREMENT_ERROR column"
            )
        z_err_all = np.asarray(
            catalog[InternalCatalogColumns.REDSHIFT_ERROR].to_numpy(dtype=np.float64),
            dtype=np.float64,
        )
    else:
        z_err_all = np.zeros_like(z_all)  # unused on the point-evaluated path
    # Change 4: each galaxy's REAL ecliptic sky (PHI_S/THETA_S are ecliptic
    # longitude/colatitude after COORD-03). The catalog galaxies ARE the
    # Monte-Carlo sky sampling of the in-catalog channel (they trace LSS), so
    # feeding Omega_g into p_det is the correct MC estimator of
    # beta_G = INTEGRAL f p_det dVc/(1+z). Gray et al. (2020), arXiv:1908.06050,
    # Eq. 8 (antenna response varies over sky); Gray 2023 arXiv:2308.02281 Eq. 2.3.
    _has_sky_cols = (InternalCatalogColumns.PHI_S in catalog.columns) and (
        InternalCatalogColumns.THETA_S in catalog.columns
    )
    # Sky-aware only for the 3D (without-BH-mass) channel; the 4D with-BH-mass
    # sky x M_z survival is statistics-starved, so it stays ISOTROPIC (below).
    _sky_aware = (
        (not with_bh_mass)
        and _has_sky_cols
        and hasattr(detection_probability_obj, "detection_probability_without_bh_mass_sky")
    )
    if _sky_aware:
        # Only the ecliptic COLATITUDE is needed (azimuthal symmetry of the
        # orbit-averaged response, Cutler 1998); phi is not used.
        theta_all = np.asarray(
            catalog[InternalCatalogColumns.THETA_S].to_numpy(dtype=np.float64),
            dtype=np.float64,
        )

    global_table: dict[float, float] = {}
    for h in h_values:
        z_max = dist_to_redshift(detection_probability_obj.get_dl_max(h), h=h)
        # [PHYSICS] Selection-domain cap (issue #30) — same domain as D(h); see
        # precompute_completion_denominator. No-op at current constants.
        if z_max_cap is not None:
            z_max = min(z_max, z_max_cap)
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
        if smear_sigma_z:
            # [PHYSICS] num/denom sigma_z symmetry (issue #30 redesign, risk R4):
            # E_{z~kernel_g}[P_det] over the numerator's volume-deconvolved host-z
            # kernel replaces the point evaluation P_det(d_L(z_g;h)). Opt-in via
            # --smear_global_selection; sigma_eff -> 0 recovers the point form.
            # DERIVATION_ESTIMATOR_REDESIGN.md §3.3/§7-R4; Gray et al. (2020),
            # arXiv:1908.06050, Eqs. A.10/33 (kernel as in the numerator N_g).
            p_det = _smeared_global_pdet_expectation(
                z_g,
                M_g,
                z_err_all[eligible],
                theta_all[eligible] if _sky_aware else None,
                h,
                detection_probability_obj,
                with_bh_mass=with_bh_mass,
                sky_aware=_sky_aware,
            )
        elif with_bh_mass:
            # FLAG (user-approved, statistics-starved): the with-BH-mass 4D
            # sky x M_z survival is too noisy at NSIDE resolution, so this branch
            # stays ISOTROPIC (phi=theta=0, sky-marginalised 2D accessor). The
            # residual sky-selection systematic (<~1%) applies to the with-BH-mass
            # posterior ONLY, not the primary result. PHYSICS-CHANGE-PROTOCOL §9.3.
            M_z_g = M_g * (1.0 + z_g)  # observer-frame mass (P_det grid axis)
            phi_iso = np.zeros_like(z_g)
            theta_iso = np.zeros_like(z_g)
            p_det = np.asarray(
                detection_probability_obj.detection_probability_with_bh_mass_interpolated(
                    d_L_g, M_z_g, phi_iso, theta_iso, h=h
                ),
                dtype=np.float64,
            )
        elif _sky_aware:
            # Sky-resolved p_det at each galaxy's real ecliptic latitude, using the
            # IDENTICAL flat per-band survival that D(h) and beta_Gbar use (NOT the
            # interpolated accessor) so p_det(Omega) is ONE shared object across all
            # selection integrals. Otherwise the p_det convention would not cancel in
            # beta_G/Sigma_global and would rescale the in-catalogue channel weight,
            # reintroducing the sky bias. Same equal-|sin beta| edges + side="right"
            # band assignment as _sky_band_pixel_map (test T3 / T8).
            # Gray et al. (2020), arXiv:1908.06050, Eq. 8; Cutler 1998 arXiv:gr-qc/9703068.
            theta_g = theta_all[eligible]
            sin_beta_g = np.abs(np.cos(theta_g))  # |sin beta| = |cos theta|
            _edges = np.asarray(detection_probability_obj.band_edges_sin_beta(), dtype=np.float64)
            _n_bands = int(_edges.size - 1)
            band_g = np.clip(np.searchsorted(_edges, sin_beta_g, side="right") - 1, 0, _n_bands - 1)
            # FIX-2: the galaxy's own z_g is the conditioning point (packet
            # §5.1): S(d_L(z_g;h) | z_g, band).
            s_band = np.asarray(
                detection_probability_obj.survival_per_band(
                    d_L_g, **_zres_z_kwargs(detection_probability_obj, z_g)
                ),
                dtype=np.float64,
            )  # (n_bands, n_gal)
            p_det = s_band[band_g, np.arange(band_g.size)]
        else:
            phi_iso = np.zeros_like(z_g)  # isotropic fallback (matches D(h))
            theta_iso = np.zeros_like(z_g)
            p_det = np.asarray(
                detection_probability_obj.detection_probability_without_bh_mass_interpolated_zero_fill(
                    d_L_g,
                    phi_iso,
                    theta_iso,
                    h=h,
                    **_zres_z_kwargs(detection_probability_obj, z_g),
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
# Legacy global kept for single_host_likelihood_integration_testing() (the dev-only
# cross-check twin) — not used by the optimized production path.
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
    # In-catalogue normalization (set by evaluate()); "generator_marginal" is the
    # production default since 2026-07-26 (author-ratified adoption,
    # results/lcat_h_dependence_20260725/MULTISEED_READOUT_20260726.md;
    # derivation: DERIVATION_GENERATOR_CONSISTENT_NORM.md). See evaluate() for
    # the legacy modes ("volume_deconv"/"global"/"local_ratio").
    _normalization_mode: str = "generator_marginal"
    # Issue #40(a): numerator host-z kernel decomposition flag (set by
    # evaluate()); "auto" reproduces the historical mode bundling exactly.
    _host_z_kernel: str = "auto"
    # G4: base seed for the deterministic with-BH-mass MC denominator streams.
    _base_seed: int = 0
    # generator_marginal precomputes (set by evaluate() when the mode is active):
    # W_cat = draw-eligible catalogue rate-weight total (yr^-1, h-independent),
    # _V_f_table[h] = completeness-weighted population volume (Mpc^3/sr).
    # n_hat_w = W_cat/V_f(h) replaces the Option-A n_bar_w = Sigma_glob/beta_G.
    _W_cat: float = 0.0
    # Author decision 1 of the derivation packet §7: which catalogue-selection
    # sum enters D_gen. "4d_exact" (primary, generator-exact per (G-ii)) uses
    # Sigma_glob_wbh (each galaxy's actual M_z inside the 4D p_det); "3d_shared"
    # (documented diagnostic) uses the pooled-3D Sigma_glob shared with beta_Gbar.
    _dgen_catalog_selection: str = "4d_exact"

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
        self._V_f_table: dict[float, float] = {}

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
        # [PHYSICS] production defaults since 2026-07-26 (author-ratified,
        # results/lcat_h_dependence_20260725/MULTISEED_READOUT_20260726.md).
        normalization_mode: str = "generator_marginal",
        base_seed: int = 0,
        allow_low_pdet_coverage: bool = False,
        h_values: Sequence[float] | None = None,
        smear_global_selection: bool = False,
        dgen_catalog_selection: str = "4d_exact",
        pdet_z_resolved: bool = True,
        # Issue #40(a): numerator host-z kernel decomposition flag; "auto"
        # preserves the historical bundling (delta kernel iff
        # generator_marginal) — production default path unchanged.
        host_z_kernel: str = "auto",
    ) -> None:
        # h-grid fusion (opt-in): when h_values is given it supersedes h_value
        # and ALL h-invariant setup — catalogue/BallTree (passed in), injection
        # pooling + p_det grid, completeness, Fisher staging, worker pool — is
        # paid once for the whole grid. The D(h)/beta/global-selection
        # precomputes are h-list-native already. Per-h outputs (posterior
        # JSONs, event-likelihood diagnostics) are written as each h completes,
        # preserving per-h failure granularity. The single-h path (h_values
        # None) is byte-compatible with the pre-fusion behaviour.
        _h_list: list[float] = (
            [float(h_value)] if h_values is None else [float(x) for x in h_values]
        )
        if not _h_list:
            raise ValueError("h_values must contain at least one value")
        self.catalog_only = catalog_only
        # G4: deterministic seed for the with-BH-mass MC denominator (threaded to
        # single_host_likelihood workers; per-call streams derived per host).
        self._base_seed = int(base_seed) if base_seed is not None else 0
        # In-catalogue normalization for the non-catalog_only Gray single ratio
        # (commission de-rail study, 2026-07-01):
        #   "global"        -> legacy partition-norm:  L_cat = (Σ_local w_g N_g)/(Σ_GLOBAL w_g D_g)
        #   "local_ratio"   -> Gray A.9/A.10 literal:  L_cat = (Σ_local w_g N_g)/(Σ_local w_g D_g)   [fix #2]
        #   "volume_deconv" -> local ratio with the host-z Gaussian deconvolved through the
        #                      comoving-volume prior dVc/(1+z) (per-galaxy renormalised)          [fix #1]
        #   "volume_global" -> DIAGNOSTIC ONLY (G3 ablation cube): fix #1's volume kernel with
        #                      the legacy GLOBAL denominator — isolates the marginal effect of
        #                      each fix ingredient. Not for production results.
        # The kernel (bare vs volume-deconvolved) is threaded into single_host_likelihood.
        # Default "volume_deconv": Gray et al. (2020) arXiv:1908.06050 Eqs. A.9/A.10 + volume-
        # consistent host-z prior; P-P-calibrated (INDEPENDENT-VERIFICATION-REPORT-20260701 §7).
        #   "volume_trunc" -> EXPERIMENTAL / FALSIFIED (Part 1, 2026-07-12): the volume kernel with
        #                     the in-catalogue NUMERATOR integrated over the per-host galaxy window
        #                     [z_g-4sigma, z_g+4sigma] (shared with Z_g / D_g) and the lower z-limit
        #                     floored at 0 instead of 1e-6. No-op on the deep venue by construction.
        #                     DO NOT USE FOR PRODUCTION: the seed600 shallow A/B FALSIFIED it — it
        #                     worsens the shallow bias (1D mean 0.745 -> 0.800), because fixed_quad
        #                     n=50 aliases the narrow GW peak over the wide host window AND the exact
        #                     numerator tilts high. Kept as a diagnostic + reproducible record.
        #                     results/volume_trunc_ab_20260712/FINDING.md; scoping §7b (Gray A.10 + G2b §1.4).
        #   "absolute_marginal" -> the absolute-mass per-event host marginal (issue #30 estimator
        #                     redesign, Variant 1): p_i = (A_i + B_num)/D with the ABSOLUTE
        #                     catalogue mass A_i = (Sigma_ball w_g N_g)/n_bar_w and the
        #                     rate-weight density calibration n_bar_w = Sigma_glob(h)/beta_G(h).
        #                     Algebraically identical to "volume_global" (A_i/D = w_G * L_cat_global
        #                     exactly), but FIRST-CLASS: derived from the exhaustive per-event
        #                     host marginal rather than as an ablation diagnostic. Replaces the
        #                     self-normalized ratio-of-sums, whose event-local per-galaxy->per-volume
        #                     conversion Sigma_ball w_g D_g lets impostor-only candidate balls carry
        #                     O(1) weight against the completion term (the deep-venue rail;
        #                     results/lcat_h_dependence_20260725/DERIVATION_ESTIMATOR_REDESIGN.md).
        #                     Empty balls flow through A_i = 0 -> p_i = B_num/D continuously (the
        #                     issue-#29 fallback emerges as a limit, not a branch). Uses the
        #                     volume_deconv host-z kernel. NOT the default until the validation
        #                     gates (DERIVATION_ESTIMATOR_REDESIGN.md section 6) pass.
        #                     Eq. (15) in Chen, Fishbach & Holz (2018), arXiv:1712.06531;
        #                     Eq. (2.4) in Gray et al. (2023), arXiv:2308.02281.
        #   "mass_trunc"   -> EXPERIMENTAL (EXP-45, 2026-07-13): the volume_deconv host-z kernel PLUS
        #                     the 2D (with-BH-mass) host-mass prior replaced by the truncated
        #                     lognormal x R_eff prior on [M_MIN, M_MAX] (Gauss-Hermite numerator,
        #                     Gauss-Legendre-in-lnM denominator), superseding the linear-Gaussian G2d
        #                     moment match. Tests the host-mass-kernel truncation as the 2D +0.025
        #                     residual driver (results/mass_kernel_truncation_20260713/FINDINGS.md).
        #                     1D channel is byte-identical to volume_deconv (no mass term). Gated
        #                     behind the flag until the seed600 A/B; volume_deconv stays the default.
        #   "generator_marginal" -> [PHYSICS] the generator-consistent selection normalization
        #                     (E1 FIX-3, approved 2026-07-26): the exact per-event marginal under
        #                     the injection pipeline's own generative recipe. Two substitutions
        #                     relative to "absolute_marginal" (nothing else changes):
        #                       n_bar_w = Sigma_glob/beta_G  ->  n_hat_w = W_cat/V_f(h)
        #                       D = beta_G + beta_Gbar       ->  D_gen = Sigma_glob_sel/n_hat_w + beta_Gbar
        #                     with W_cat the draw-eligible catalogue rate-weight total (the
        #                     generator draw's own normalizer) and V_f(h) the completeness-weighted
        #                     population volume (the generator's F = V_f/V_tot integral). The
        #                     Option-A constant-comoving-density identity is never invoked: no
        #                     model integral is compared against a discrete catalogue sum. The
        #                     sigma_z pairing is point/point (generator-exact, premise verified:
        #                     draw_rate_weighted_hosts copies catalogue rows verbatim and
        #                     set_host_galaxy_parameters uses host_z unscattered;
        #                     handler.draw_z_and_mass_from_gaussian is dead code): the in-catalogue
        #                     numerator N_g is the GW likelihood POINT-evaluated at the catalogue
        #                     z_g (delta kernel) and Sigma_glob stays point-evaluated
        #                     (--smear_global_selection is rejected in this mode). D_gen's
        #                     catalogue term uses Sigma_glob_wbh ("4d_exact", generator-exact per
        #                     (G-ii)); the "3d_shared" pooled-survival variant is reachable via
        #                     dgen_catalog_selection as a documented diagnostic. Empty balls
        #                     reduce continuously to B_num/D_gen.
        #                     results/lcat_h_dependence_20260725/DERIVATION_GENERATOR_CONSISTENT_NORM.md
        #                     Eqs. (3)-(5); Mandel, Farr & Gair (2019), arXiv:1809.02063 (single
        #                     selection factor alpha(h)); Fishbach et al. (2019), arXiv:1807.05667,
        #                     Eqs. (3)-(5) (mixture structure).
        if normalization_mode not in (
            "global",
            "local_ratio",
            "volume_deconv",
            "volume_global",
            "volume_trunc",
            "mass_trunc",
            "absolute_marginal",
            "generator_marginal",
        ):
            raise ValueError(f"unknown normalization_mode: {normalization_mode!r}")
        if dgen_catalog_selection not in ("4d_exact", "3d_shared"):
            raise ValueError(
                f"unknown dgen_catalog_selection: {dgen_catalog_selection!r} "
                "(expected '4d_exact' or '3d_shared')"
            )
        # Issue #40(a): validate the kernel choice up front (raises on unknown);
        # the resolved value is recomputed identically inside the worker kernels.
        _resolved_kernel = resolve_host_z_kernel(host_z_kernel, normalization_mode)
        if host_z_kernel != "auto":
            _LOGGER.info(
                "host_z_kernel=%r overrides the mode-bundled numerator kernel "
                "(resolved: %s numerator with %s normalization) — diagnostic "
                "decomposition, issue #40(a)",
                host_z_kernel,
                _resolved_kernel,
                normalization_mode,
            )
        if normalization_mode == "generator_marginal" and smear_global_selection:
            # The mode is DEFINED with the point/point sigma_z pairing (generator-
            # exact, derivation §4.3): a smeared Sigma_glob would silently break
            # the approved pairing, so reject the combination loudly.
            raise ValueError(
                "normalization_mode='generator_marginal' uses the point/point "
                "sigma_z pairing (generator-exact); --smear_global_selection is "
                "incompatible with it. Drop the flag (or use 'absolute_marginal' "
                "for the kernel/smeared pairing)."
            )
        if normalization_mode == "global":
            warnings.warn(
                "normalization_mode='global' is mis-calibrated for photometric-redshift "
                "catalogues (~0% P-P coverage; posterior rails to the grid edge — see "
                ".planning/INDEPENDENT-VERIFICATION-REPORT-20260701.md §7). Use the default "
                "'volume_deconv' unless deliberately reproducing the railed baseline.",
                UserWarning,
                stacklevel=2,
            )
        self._normalization_mode = normalization_mode
        self._host_z_kernel = host_z_kernel
        self._dgen_catalog_selection = dgen_catalog_selection
        self._diagnostic_rows = []
        if catalog_only:
            _LOGGER.info("catalog_only mode: f_i=1, L_comp=0 (skipping completion integral)")
        _LOGGER.info(
            f"Computing posteriors for h = {_h_list[0] if len(_h_list) == 1 else _h_list}..."
        )
        for _h_check in _h_list:
            if (_h_check < self.cosmological_model.h.lower_limit) or (
                _h_check > self.cosmological_model.h.upper_limit
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
            # Stale-pool gate (issue #20): the pool must span the host-draw
            # volume; a z_cut = 0.5-era pool at depth 1.5 yields p_det = 0
            # for essentially all events — silent garbage posteriors.
            expected_z_max=HOST_DRAW_Z_MAX,
            allow_shallow_pool=allow_low_pdet_coverage,
            # FIX-2 (opt-in): z-resolved detection survival S(d_L | z); every
            # 3D consumer below passes its own z via _zres_z_kwargs.
            pdet_z_resolved=pdet_z_resolved,
        )
        _LOGGER.debug("Detection probability functions created.")

        # Pre-warm P_det grid cache for target h -- avoids N workers each building
        # the same grid independently after pool spawn
        for _h_warm in _h_list:
            detection_probability._get_or_build_grid(_h_warm)
            _LOGGER.debug("P_det grid pre-warmed for h=%.4f.", _h_warm)

        # Validate P_det grid coverage for observed events — HARD gate
        # (readiness sweep A2-STALE-POOL-GATE, 2026-07-03): a warning buried
        # in one of 38 per-task logs does not stop a campaign from burning
        # its cpu-h budget on p_det = 0 posteriors. Grid mode gates on every h.
        for _h_cov in _h_list:
            coverage_fraction = detection_probability.validate_coverage(
                _h_cov, self.cramer_rao_bounds
            )
            if coverage_fraction < 0.95 and not allow_low_pdet_coverage:
                msg = (
                    f"P_det grid covers only {coverage_fraction:.1%} of events' "
                    "4-sigma d_L windows (< 95%). The injection pool is likely stale "
                    "or too shallow for this event set. Regenerate the pool, or pass "
                    "--allow_low_pdet_coverage to proceed deliberately."
                )
                raise RuntimeError(msg)

        # Gray et al. (2020), arXiv:1908.06050, Eq. 9 + Gray-Messenger-Veitch 2022,
        # arXiv:2111.04629 (Change 5): per-HEALPix-pixel completeness f_k(z,Omega,h),
        # loaded from the SAME frozen cached m_th map the EMRI injection uses (C1
        # consistency; main.py:injection_campaign). f_bar weights beta_Gbar, f_k(event
        # pixel) weights the completion numerator B_num below.  Built BEFORE D(h) so
        # the sky-resolved selection (Change 2-4) can share its pixel grid.
        completeness = from_cache_or_build()

        # Gray et al. (2020), arXiv:1908.06050, Eq. A.19 + Gray 2023 arXiv:2308.02281
        # Eq. 2.3 (Change 2): sky-resolved completion-term denominator D(h) over the
        # full detectable volume, D(h) = INTEGRAL (1/Npix) sum_k p_det(d_L,Omega_k)
        # dVc/(1+z). D(h) is event-independent; compute once per h-value.
        _D_h_table = precompute_completion_denominator(
            h_values=_h_list,
            detection_probability_obj=detection_probability,
            Omega_m=self.Omega_m,
            Omega_DE=self.Omega_DE,
            completeness=completeness,
            z_max_cap=REDSHIFT_UPPER_LIMIT,
        )
        _LOGGER.info("D(h) precomputed for %d h-value(s).", len(_D_h_table))

        # Partition-norm precomputes (Option A), consumed by p_Di's single ratio
        # p_i = (beta_G L_cat + B_num)/D(h). beta_Gbar(h) = INTEGRAL (1-f) P_det
        # dVc/(1+z) (Gray et al. 2020, arXiv:1908.06050, Eq. 33);
        # beta_G(h) = D(h) - beta_Gbar(h) (Eq. 29); and the global in-catalogue
        # selection sums sum_global w_g D_g for both channels (Eq. 29 discrete
        # realisation) that make L_cat scale-free so n_gal cancels.
        _beta_Gbar_table = precompute_missing_completion_denominator(
            h_values=_h_list,
            detection_probability_obj=detection_probability,
            completeness=completeness,
            z_max_cap=REDSHIFT_UPPER_LIMIT,
        )
        _beta_G_table = {h: _D_h_table[h] - _beta_Gbar_table[h] for h in _D_h_table}
        _global_cat_denom_no_bh = precompute_global_catalog_selection(
            h_values=_h_list,
            galaxy_catalog=galaxy_catalog,
            detection_probability_obj=detection_probability,
            with_bh_mass=False,
            z_max_cap=REDSHIFT_UPPER_LIMIT,
            smear_sigma_z=smear_global_selection,
        )
        _global_cat_denom_with_bh = precompute_global_catalog_selection(
            h_values=_h_list,
            galaxy_catalog=galaxy_catalog,
            detection_probability_obj=detection_probability,
            with_bh_mass=True,
            z_max_cap=REDSHIFT_UPPER_LIMIT,
            smear_sigma_z=smear_global_selection,
        )
        # generator_marginal precomputes: the draw-side calibration pair
        # (W_cat, V_f(h)). Domain = min(draw depth, analysis cap) so an
        # issue-#30 depth truncation moves W_cat/V_f together with the
        # candidate window (derivation §3.2 domain note; f29a5e7 principle).
        # Eqs. (3)-(5) in DERIVATION_GENERATOR_CONSISTENT_NORM.md.
        if normalization_mode == "generator_marginal":
            _draw_domain_z_max = min(HOST_DRAW_Z_MAX, REDSHIFT_UPPER_LIMIT)
            self._W_cat = compute_catalog_draw_weight_total(
                galaxy_catalog, z_max=_draw_domain_z_max
            )
            self._V_f_table = precompute_completeness_population_volume(
                _h_list,
                completeness,
                z_max=_draw_domain_z_max,
            )
            for _h_gen in _h_list:
                _n_hat_w = self._W_cat / self._V_f_table[_h_gen]
                _sigma_sel = (
                    _global_cat_denom_with_bh[_h_gen]
                    if dgen_catalog_selection == "4d_exact"
                    else _global_cat_denom_no_bh[_h_gen]
                )
                _D_gen_prev = _sigma_sel / _n_hat_w + _beta_Gbar_table[_h_gen]
                _LOGGER.info(
                    "generator_marginal(h=%.4f): n_hat_w=%.4f, D_gen=%.6e (%s), P_cat_det=%.4f",
                    _h_gen,
                    _n_hat_w,
                    _D_gen_prev,
                    dgen_catalog_selection,
                    (_sigma_sel / _n_hat_w) / _D_gen_prev if _D_gen_prev > 0 else float("nan"),
                )
        for _h_prev in _h_list:
            _w_G_preview = (
                _beta_G_table[_h_prev] / _D_h_table[_h_prev]
                if _D_h_table.get(_h_prev, 0.0) > 0.0
                else float("nan")
            )
            _LOGGER.info(
                "Partition-norm: w_G=beta_G/D(h)=%.4f, sum_w_Dg(no_bh)=%.4e, sum_w_Dg(with_bh)=%.4e",
                _w_G_preview,
                _global_cat_denom_no_bh.get(_h_prev, float("nan")),
                _global_cat_denom_with_bh.get(_h_prev, float("nan")),
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
            # Per-h evaluation loop (one iteration in single-h mode). Setup
            # above — data, p_det grids, completeness, D(h)/beta/global tables,
            # Fisher staging, worker pool — is h-invariant and shared; each
            # iteration resets the per-h accumulators, runs the detection loop,
            # and writes that h's outputs immediately (per-h failure
            # granularity is preserved in grid mode).
            for _h_run in _h_list:
                self.h = _h_run
                if h_values is not None:
                    # Grid mode: per-h accumulators so each JSON carries exactly
                    # one likelihood per event (the canonical production shape).
                    # Single-h mode intentionally keeps the legacy semantics:
                    # repeated evaluate() calls on one instance accumulate one
                    # value per h into posterior_data (integration-test harness
                    # contract; production single-h runs are fresh processes).
                    self.posterior_data = {}
                    self.posterior_data_with_bh_mass = {}
                    self._diagnostic_rows = []

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

                    data = {
                        str(key): value for key, value in self.posterior_data_with_bh_mass.items()
                    }
                    json.dump(data | {"h": self.h}, file)

                # Write per-event diagnostic CSV (append-mode, rows are h-tagged)
                if self._diagnostic_rows:
                    diagnostic_csv_path = "simulations/diagnostics/event_likelihoods.csv"
                    self._write_diagnostic_csv(diagnostic_csv_path)

        # Write Fisher quality CSV (per D-12) — h-invariant, once per run
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
        _n_zero_host = 0
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
                if self.catalog_only:
                    # The catalog-only cross-check has no completion term, so a
                    # zero-host event carries no information in this mode — keep
                    # the legacy skip (mode stays byte-identical).
                    _LOGGER.debug("no possible hosts found (catalog_only): skipping event")
                    continue
                # [PHYSICS] Zero-host pure-completion fallback (issue #29): an event
                # whose localization volume contains no catalogue galaxy still
                # contributes the pure-completion likelihood p_i = B_num(h)/D(h) —
                # the exact L_cat -> 0 limit of the mixture
                # p_i = (beta_G L_cat + B_num)/D computed in p_Di. The pre-2026-07-10
                # `continue` silently conditioned the event sample on catalogue
                # support (58% of depth-1.5 campaign events dropped) and railed the
                # combined posterior; see FINDINGS_COMBINE_20260710.md.
                # Eqs. (29)+(32) in Gray et al. (2020), arXiv:1908.06050;
                # Eq. (5) in Gray, Messenger & Veitch (2022), arXiv:2111.04629;
                # docs/derivations/G2a_completion_sky_marginal_4pi.md, limiting case 2.
                _n_zero_host += 1
                _LOGGER.warning(
                    "Detection %d: no catalogue hosts in the localization volume — "
                    "pure-completion fallback p_i = B_num/D (issue #29)",
                    int(index),
                )
                candidate_hosts: list[HostGalaxy] = []
                candidate_hosts_with_bh_mass: list[HostGalaxy] = []
            else:
                candidate_hosts, candidate_hosts_with_bh_mass = possible_hosts
                _LOGGER.info(
                    f"possible hosts found {len(candidate_hosts)}/{len(candidate_hosts_with_bh_mass)}..."
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
                possible_host_galaxies=candidate_hosts,
                possible_host_galaxies_with_bh_mass=candidate_hosts_with_bh_mass,
                detection_index=index,
                pool=pool,
                completeness=completeness,
                detection_probability_obj=detection_probability_obj,
                redshift_upper_limit=redshift_upper_limit,
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

        # Host-lookup yield metric (issue #29 process fix): the zero-host rate is a
        # first-class health signal — 58-60% on the depth-1.5 campaign was visible
        # in per-event lines but tracked by nothing.
        _LOGGER.info(
            "Host-lookup yield at h=%.4f: %d/%d events with catalogue hosts, "
            "%d pure-completion (zero-host) fallbacks",
            self.h,
            count - _n_zero_host,
            count,
            _n_zero_host,
        )

    def p_Di(
        self,
        possible_host_galaxies: list[HostGalaxy],
        possible_host_galaxies_with_bh_mass: list[HostGalaxy],
        detection_index: int,
        pool: mp.pool.Pool,
        completeness: CompletenessModel,
        detection_probability_obj: SimulationDetectionProbability,
        redshift_upper_limit: float = HOST_DRAW_Z_MAX,
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

        # Host-batched dispatch: one vectorized single_host_likelihood_batch task
        # per worker chunk instead of one scalar single_host_likelihood task per
        # host. Same chunk count as the old chunksize=ceil(n/processes) policy,
        # same per-host values (see test_kernel_batch_equivalence.py).
        results_with_bh_mass = _starmap_host_batches(
            pool,
            possible_host_galaxies_with_bh_mass,
            detection_index,
            self.h,
            True,
            self._normalization_mode,
            self._host_z_kernel,
        )

        results_without_blackhole_mass = _starmap_host_batches(
            pool,
            possible_host_galaxies_reduced,
            detection_index,
            self.h,
            False,
            self._normalization_mode,
            self._host_z_kernel,
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
            # generator_marginal draw-side calibration (0.0 outside that mode).
            n_hat_w: float = 0.0

            # In-catalogue term L_cat. Normalization modes:
            #   "global"/"volume_global"/"absolute_marginal":
            #     L_cat = (Σ_local w_g N_g) / (Σ_GLOBAL w_g D_g) -- the partition-norm single
            #     ratio; the SELECTION denominator runs over the full catalogue (Eq. 29,
            #     precompute_global_catalog_selection), making L_cat scale-free so
            #     beta_G*L_cat reconstructs the in-catalogue numerator with the
            #     per-galaxy<->per-volume n_gal factor cancelled (Option A).
            #     "absolute_marginal" (issue #30 estimator redesign, Variant 1) is this same
            #     branch adopted as a FIRST-CLASS mode, derived from the exhaustive per-event
            #     host marginal: beta_G*L_cat_global = A_i = (Σ_ball w_g N_g)/n_bar_w with
            #     n_bar_w = Σ_glob(h)/beta_G(h) the catalogue-estimated rate-weight density,
            #     so the assembly below reads p_i = (A_i + B_num)/D exactly.
            #     Eq. (15) in Chen, Fishbach & Holz (2018), arXiv:1712.06531;
            #     Eq. (2.4) in Gray et al. (2023), arXiv:2308.02281.
            #     HISTORY NOTE: the 2026-07-01/02 commission finding that the global
            #     normalization "pins the mode to the grid edge" predates BOTH the G2a
            #     completion-sky-marginal fix and the issue-#29 zero-host fallback -- it
            #     evaluated the global catalogue term inside a broken mixture and is
            #     CONFOUNDED; no verdict on the current stack contradicts this branch
            #     (results/lcat_h_dependence_20260725/DERIVATION_ESTIMATOR_REDESIGN.md §0).
            #   "local_ratio"/"volume_deconv": L_cat = (Σ_local w_g N_g)/(Σ_local w_g D_g) --
            #     the Gray A.9/A.10 literal local self-normalized ratio-of-sums (numerator and
            #     per-host selection denominator over the SAME candidate ball). This was the
            #     2026-07-01 de-rail fix (#2); "volume_deconv" additionally uses the
            #     volume-deconvolved host-z prior inside N_g/D_g (#1, threaded via
            #     single_host_likelihood). Its event-local per-galaxy->per-volume conversion
            #     Σ_ball w_g D_g is scale-inconsistent with the marginal and lets
            #     impostor-only balls carry O(1) weight (the deep-venue rail, issue #30).
            #   Gray et al. (2020), arXiv:1908.06050, Eqs. A.9 / A.10 / 29.
            #   "generator_marginal": A_i = (Σ_ball w_g N_g) / n_hat_w with the DRAW-SIDE
            #     calibration n_hat_w = W_cat/V_f(h) — no P_det inside the conversion, so the
            #     Option-A identity Sigma_glob = n_hat_w*beta_G is never invoked. ONE n_hat_w
            #     for both channels (the conversion is population-side, channel-independent;
            #     derivation §4.2 — this also removes the per-channel Option-A substitution
            #     n_bar_w_wbh = Sigma_glob_wbh/beta_G).
            #     Eqs. (3)-(4) in DERIVATION_GENERATOR_CONSISTENT_NORM.md;
            #     Mandel, Farr & Gair (2019), arXiv:1809.02063 (selection convention).
            if self._normalization_mode == "generator_marginal":
                _V_f_h: float = self._V_f_table.get(self.h, 0.0)
                # n_hat_w = W_cat / V_f(h)  [yr^-1 sr Mpc^-3, same units as n_bar_w]
                # Eq. (4) in DERIVATION_GENERATOR_CONSISTENT_NORM.md §2.3.
                n_hat_w = self._W_cat / _V_f_h if _V_f_h > 0.0 else 0.0
                if n_hat_w <= 0.0:
                    _LOGGER.warning(
                        "Detection %s: n_hat_w <= 0 (W_cat=%.3e, V_f=%.3e) — "
                        "catalogue term dropped",
                        detection_index,
                        self._W_cat,
                        _V_f_h,
                    )
                cat_num_sum_no_bh = weighted_sum(
                    [r[0] for r in all_results_without_bh], weights_without_bh
                )
                # A_i = (Σ_ball w_g N_g) / n_hat_w; empty ball -> A_i = 0 exactly.
                L_cat_without_bh_mass = cat_num_sum_no_bh / n_hat_w if n_hat_w > 0.0 else 0.0
                if len(results_with_bh_mass) > 0:
                    cat_num_sum_with_bh = weighted_sum(
                        [r[2] for r in results_with_bh_mass], weights_with_bh
                    )
                    L_cat_with_bh_mass = cat_num_sum_with_bh / n_hat_w if n_hat_w > 0.0 else 0.0
                else:
                    L_cat_with_bh_mass = 0.0
            elif self._normalization_mode in ("global", "volume_global", "absolute_marginal"):
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
                        cat_num_sum_with_bh / global_denom_with_bh
                        if global_denom_with_bh > 0
                        else 0.0
                    )
                else:
                    L_cat_with_bh_mass = 0.0
            else:
                # local self-normalized ratio-of-sums (Gray A.9/A.10) -- de-rail fix #2/#1
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
            # Domain-matched to D(h): Eq. (32) in Gray et al. (2020), arXiv:1908.06050;
            # analysis-depth cap per f29a5e7. B_num shares the SAME functional form as
            # D(h)/beta_Gbar(h)/Sigma_global(h) (all `(1-f) p_det|p_GW dVc/(1+z)`), and
            # all three are already capped at `min(z_max(h), max_redshift)`
            # (z_max_cap, see precompute_completion_denominator and the
            # candidate-host window cap in p_D). Without this cap, B_num integrated
            # population density beyond the analysis depth while its own denominator
            # D(h) did not -- mismatched domains in the same ratio p_i = B_num/D(h).
            z_upper = min(z_upper, redshift_upper_limit)

            FIXED_QUAD_N = 50
            _comp_slot = self._det_index_to_slot[detection_index]
            _comp_mean_3d = self._means_3d[_comp_slot]
            _comp_cov_inv_3d = self._cov_inv_3d[_comp_slot]
            _comp_det_d_L = self._det_d_L[_comp_slot]
            # [PHYSICS] De-rail fix (2026-07-01): the completion numerator marginalises
            # the GW likelihood over the UNKNOWN dark-host sky direction with the
            # isotropic prior 1/(4π) — NOT the peak sky density. The isotropic
            # sky-marginal of the 3D GW Gaussian is a 1D Gaussian in d_L_fraction with
            # variance Σ[2,2] (Σ = cov = inv(cov_inv)) and mean mean_3d[2] (=1). This
            # makes B_num's sky treatment consistent with the completion denominator
            # D(h) = ∫ (1/Npix) Σ_k p_det(Ω_k) · dVc/(1+z) dz (sky-averaged p_det).
            # Eq. (32) in Gray et al. (2020), arXiv:1908.06050.
            _comp_cov_3d = np.linalg.inv(_comp_cov_inv_3d)
            _comp_sigma_dLfrac = float(np.sqrt(_comp_cov_3d[2, 2]))
            _comp_mean_dLfrac = float(_comp_mean_3d[2])
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
                # [PHYSICS] isotropic-sky-marginalised GW likelihood (see the precompute
                # above): (sin θ_det/4π) · N(d_L_fraction; 1, σ_marg). Replaces the peak
                # sky density _mvn_pdf([φ_det, θ_det, d_L_fraction], …), which over-counted
                # the completion term by ~4π·(peak sky density) (~5000× at σ_sky≈2°) and
                # pinned the H0 posterior to the grid edge.
                # The sin(θ_det) is the solid-angle Jacobian: the Fisher Gaussian is a
                # density in the bare coordinates (φ_S, q_S), so its isotropic marginal
                # over dΩ = sinθ dθ dφ picks up sinθ at the (narrow) beam position.
                # Eq. (32) in Gray et al. (2020), arXiv:1908.06050; derivation:
                # docs/derivations/G2a_completion_sky_marginal_4pi.md Eq. (10).
                p_gw: npt.NDArray[np.float64] = (
                    norm.pdf(d_L_fraction, loc=_comp_mean_dLfrac, scale=_comp_sigma_dLfrac)
                    * np.sin(self.detection.theta)
                    / (4.0 * np.pi)
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

            if z_lower >= z_upper:
                # The event's entire 4-sigma window lies beyond the analysis depth
                # (redshift_upper_limit): no population support survives the cap, so
                # the completion numerator vanishes rather than integrating an
                # inverted [z_lower, z_upper] interval (which would return a
                # negative fixed_quad result, not 0).
                B_num = 0.0
            else:
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
            # Under "absolute_marginal" this line IS the marginal p_i = (A_i + B_num)/D
            # (A_i = beta_G*L_cat_global; empty ball -> A_i = 0 -> p_i = B_num/D, the
            # issue-#29 fallback as a continuous limit of the same expression).
            # Eq. (15) in Chen, Fishbach & Holz (2018), arXiv:1712.06531;
            # Eq. (2.4) in Gray et al. (2023), arXiv:2308.02281.
            if self._normalization_mode == "generator_marginal":
                # [PHYSICS] Generator-consistent master denominator (E1 FIX-3):
                #     D_gen(h) = Sigma_glob_sel(h)/n_hat_w(h) + beta_Gbar(h)
                # replaces D = beta_G + beta_Gbar. Sigma_glob_sel is the with-BH
                # catalogue-selection sum Sigma_glob_wbh under the primary
                # "4d_exact" convention (generator-exact per (G-ii): each galaxy
                # detects at its actual M_z) or the pooled-3D Sigma_glob under
                # the "3d_shared" diagnostic; ONE D_gen serves both posterior
                # channels either way (derivation §4.2/§7 decision 1). The
                # marginal is p_i = (A_i + B_num)/D_gen; empty balls flow through
                # A_i = 0 -> p_i = B_num/D_gen continuously (issue-#29 fallback
                # as a limit, not a branch). In the p_det -> 1 limit
                # Sigma_glob -> W_cat, hence D_gen -> V_f + beta_Gbar = D and the
                # current estimator is recovered algebraically (derivation §5d).
                # Eqs. (3)+(5) in DERIVATION_GENERATOR_CONSISTENT_NORM.md;
                # Chen, Fishbach & Holz (2018), arXiv:1712.06531, Eq. (15);
                # Gray et al. (2023), arXiv:2308.02281, Eq. (2.4).
                _sigma_glob_sel: float = (
                    global_denom_with_bh
                    if self._dgen_catalog_selection == "4d_exact"
                    else global_denom_no_bh
                )
                _a_cat: float = _sigma_glob_sel / n_hat_w if n_hat_w > 0.0 else 0.0
                D_gen: float = _a_cat + beta_Gbar
                if D_gen > 0:
                    # Diagnostic: P_hat(cat|det,h) = (Sigma_glob_sel/n_hat_w)/D_gen —
                    # the generator-consistent detected-catalogue share (replaces
                    # the w_G = beta_G/D diagnostic; derivation §4.4).
                    w_G = _a_cat / D_gen
                    combined_without_bh_mass = float((L_cat_without_bh_mass + B_num) / D_gen)
                    combined_with_bh_mass = float((L_cat_with_bh_mass + B_num) / D_gen)
                else:
                    _LOGGER.warning(
                        f"Detection {detection_index}: D_gen(h) is zero, using A_i only"
                    )
                    w_G = 1.0
                    combined_without_bh_mass = float(L_cat_without_bh_mass)
                    combined_with_bh_mass = float(L_cat_with_bh_mass)
            elif D_h > 0:
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


def _bh_mass_denominator_inner_m_integral(
    z: npt.NDArray[np.float64],
    detection_probability: Any,
    host_phiS: float,
    host_qS: float,
    host_M_eff: float,
    host_M_error: float,
    h: float,
) -> npt.NDArray[np.float64]:
    r"""Exact inner mass integral of the with-BH-mass selection denominator.

    Returns, per redshift ``z_j``,

    .. math::

        g(z) = \int p_\mathrm{det}\big(d_L(z),\, M(1+z)\big)\,
               \mathcal{N}(M;\, M_g^\mathrm{eff},\, \sigma_M)\, dM .

    ``p_det`` is bilinearly interpolated (``method="linear"``) and constant-clamped
    in ``M_z`` outside the injection grid (``simulation_detection_probability``
    clips ``M_z`` to ``[M_centers[0], M_centers[-1]]``), so at fixed ``d_L(z)`` it
    is *exactly* piecewise-linear in ``M_z`` between the interpolator's ``M_z``
    knots.  The integral of a piecewise-linear function against a Gaussian is the
    closed-form erf-sum over the knots ``M_k = M_center_k / (1 + z)``:

    .. math::

        \int_{M_k}^{M_{k+1}} (c_0 + c_1 M)\,\mathcal{N}(M;\mu,\sigma)\,dM
        = c_0\,\Delta\Phi + c_1\,(\mu\,\Delta\Phi - \sigma\,\Delta\phi),

    with ``c_1`` the per-segment slope, plus constant-clamp tails
    ``p_0\,\Phi(a_0) + p_{-1}(1-\Phi(a_{-1}))``, ``a_k = (M_k-\mu)/\sigma``.  This
    is exact for the interpolant (zero ``M``-quadrature error) and replaces the
    10k-sample Monte-Carlo that carried ~1-5% noise.  The ``M_z`` knots are read
    from the live interpolator, so the integral automatically tracks any change
    to the injection-grid resolution.

    Reference:
        Owen (1980), *A table of normal integrals*, Commun. Statist. B9(4),
        389-419 (Gaussian zeroth/first-moment identities).
    """
    z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
    interp_2d, _ = detection_probability._get_or_build_grid(h)
    m_centers = np.asarray(interp_2d.grid[1], dtype=np.float64)  # M_z grid knots
    n_k = m_centers.size
    d_L = dist_vectorized(z_arr, h=h)

    # p_det at every (z_j, M_center_k) -> (n_z, K), one interpolator call.
    dl_zz = np.repeat(d_L, n_k)
    mm = np.tile(m_centers, z_arr.size)
    phi = np.full_like(dl_zz, host_phiS)
    theta = np.full_like(dl_zz, host_qS)
    p = np.asarray(
        detection_probability.detection_probability_with_bh_mass_interpolated(
            dl_zz, mm, phi, theta, h=h
        ),
        dtype=np.float64,
    ).reshape(z_arr.size, n_k)

    mu = host_M_eff
    sigma = host_M_error
    # Knot positions in rest-frame M (M_z = M(1+z)); increasing in k for z >= 0.
    m_knots = m_centers[None, :] / (1.0 + z_arr[:, None])  # (n_z, K)
    a = (m_knots - mu) / sigma
    big_phi = ndtr(a)  # standard-normal CDF (identical to norm.cdf)
    small_phi = np.exp(-0.5 * a * a) / np.sqrt(2.0 * np.pi)  # standard-normal pdf
    # Constant-clamp tails (p_det flat below the first / above the last knot).
    val = p[:, 0] * big_phi[:, 0] + p[:, -1] * (1.0 - big_phi[:, -1])
    # Interior linear segments: int (c0 + c1 M) N dM, c1 = per-segment slope.
    d_big = big_phi[:, 1:] - big_phi[:, :-1]  # (n_z, K-1)
    int_m_n = mu * d_big - sigma * (small_phi[:, 1:] - small_phi[:, :-1])  # ∫ M N dM
    dm = m_knots[:, 1:] - m_knots[:, :-1]
    slope = (p[:, 1:] - p[:, :-1]) / dm
    val = val + np.sum(p[:, :-1] * d_big + slope * (int_m_n - m_knots[:, :-1] * d_big), axis=1)
    return np.asarray(val, dtype=np.float64)


def _bh_mass_denominator_inner_m_integral_batch(
    z: npt.NDArray[np.float64],
    detection_probability: Any,
    host_phiS: npt.NDArray[np.float64],
    host_qS: npt.NDArray[np.float64],
    host_M_eff: npt.NDArray[np.float64],
    host_M_error: npt.NDArray[np.float64],
    h: float,
) -> npt.NDArray[np.float64]:
    """Host-batched twin of :func:`_bh_mass_denominator_inner_m_integral`.

    Evaluates the exact erf-sum inner mass integral for ``n`` hosts at once:
    ``z`` has shape ``(n, n_z)`` (per-host outer-quadrature nodes) and the host
    parameters have shape ``(n,)``. Row ``i`` of the result is bit-identical to
    the scalar function called with ``z[i]`` and host ``i``'s parameters — the
    arithmetic per (host, node, knot) element is unchanged; only a leading host
    axis is added, and the single ``p_det`` interpolator call covers all
    ``n * n_z * K`` points at once (amortising ``_find_indices``).

    Args:
        z: Redshift nodes, shape ``(n, n_z)``.
        detection_probability: ``SimulationDetectionProbability`` instance.
        host_phiS: Host ecliptic azimuths, shape ``(n,)``.
        host_qS: Host ecliptic polar angles, shape ``(n,)``.
        host_M_eff: Effective (Eddington-shifted) host masses, shape ``(n,)``.
        host_M_error: Host mass 1-sigma errors, shape ``(n,)``.
        h: Dimensionless Hubble parameter.

    Returns:
        Inner-integral values ``g(z)``, shape ``(n, n_z)``.
    """
    n, n_z = z.shape
    interp_2d, _ = detection_probability._get_or_build_grid(h)
    m_centers = np.asarray(interp_2d.grid[1], dtype=np.float64)  # M_z grid knots
    n_k = m_centers.size
    d_L = dist_vectorized(z.reshape(-1), h=h)  # (n*n_z,)

    # p_det at every (host_i, z_j, M_center_k) -> (n, n_z, K), one interpolator call.
    dl_zz = np.repeat(d_L, n_k)
    mm = np.tile(m_centers, n * n_z)
    phi = np.repeat(host_phiS, n_z * n_k)
    theta = np.repeat(host_qS, n_z * n_k)
    p = np.asarray(
        detection_probability.detection_probability_with_bh_mass_interpolated(
            dl_zz, mm, phi, theta, h=h
        ),
        dtype=np.float64,
    ).reshape(n, n_z, n_k)

    mu = host_M_eff[:, None, None]
    sigma = host_M_error[:, None, None]
    # Knot positions in rest-frame M (M_z = M(1+z)); increasing in k for z >= 0.
    m_knots = m_centers[None, None, :] / (1.0 + z[:, :, None])  # (n, n_z, K)
    a = (m_knots - mu) / sigma
    big_phi = ndtr(a)  # standard-normal CDF (identical to norm.cdf)
    small_phi = np.exp(-0.5 * a * a) / np.sqrt(2.0 * np.pi)  # standard-normal pdf
    # Constant-clamp tails (p_det flat below the first / above the last knot).
    val = p[:, :, 0] * big_phi[:, :, 0] + p[:, :, -1] * (1.0 - big_phi[:, :, -1])
    # Interior linear segments: int (c0 + c1 M) N dM, c1 = per-segment slope.
    d_big = big_phi[:, :, 1:] - big_phi[:, :, :-1]  # (n, n_z, K-1)
    int_m_n = mu * d_big - sigma * (small_phi[:, :, 1:] - small_phi[:, :, :-1])  # ∫ M N dM
    dm = m_knots[:, :, 1:] - m_knots[:, :, :-1]
    slope = (p[:, :, 1:] - p[:, :, :-1]) / dm
    val = val + np.sum(
        p[:, :, :-1] * d_big + slope * (int_m_n - m_knots[:, :, :-1] * d_big), axis=2
    )
    return np.asarray(val, dtype=np.float64)


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
    # [PHYSICS] production default since 2026-07-26 (MULTISEED_READOUT_20260726.md)
    normalization_mode: str = "generator_marginal",
    base_seed: int = 0,
    # Issue #40(a): numerator host-z kernel decomposition flag; "auto" == the
    # historical bundling (delta kernel iff generator_marginal). No value
    # change on the default path.
    host_z_kernel: str = "auto",
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

    # [PHYSICS] volume_trunc (Part 1, 2026-07-12): shallow-venue host-z kernel
    # correction. It reuses the volume-deconvolved kernel machinery (same w_pop)
    # but (i) floors the lower z-limit at 0 instead of 1e-6 and (ii) integrates
    # the in-catalogue NUMERATOR over the per-host galaxy window
    # [z_g-4sigma, z_g+4sigma] (shared with Z_g and D_g) instead of the
    # event-level GW window, so N_g, D_g and Z_g share ONE truncated support.
    # No-op on the deep venue by construction (z_g-4sigma > 0 there). Gray et al.
    # (2020) arXiv:1908.06050 Eq. A.10; docs/derivations/G2b_host_z_volume_prior.md
    # §1.4; .planning/PRODUCTION-KERNEL-FIX-SCOPING-20260712.md §7b.
    # EXPERIMENTAL / FALSIFIED — the seed600 A/B rejected this (worsens shallow bias:
    # fixed_quad n=50 aliases the narrow GW peak over the wide host window; exact
    # numerator also tilts high). Not for production. results/volume_trunc_ab_20260712/.
    _use_volume_trunc = normalization_mode == "volume_trunc"

    # [PHYSICS] mass_trunc (EXP-45, 2026-07-13): truncated lognormal x R_eff host-mass
    # prior in the 2D channel (see module-level _MASS_TRUNC_* + _mass_trunc_* helpers).
    # Shares the volume_deconv host-z kernel; differs ONLY in the with-BH-mass
    # mass-marginal (numerator + selection denominator). No effect without BH mass.
    _use_mass_trunc = normalization_mode == "mass_trunc"

    # [PHYSICS] generator_marginal (E1 FIX-3, approved 2026-07-26): point/point
    # sigma_z pairing. The generator draws hosts at their catalogue z verbatim and
    # detects at d_L(z_g; h_inj) — no sigma_z scatter anywhere on the production
    # path (draw_rate_weighted_hosts copies rows; set_host_galaxy_parameters uses
    # host_z unscattered; draw_z_and_mass_from_gaussian is dead code), so the
    # generator-exact in-catalogue numerator is the GW likelihood POINT-evaluated
    # at z_g: N_g = p(x | z_g, Omega_g[, M]) — the delta-kernel limit of the
    # volume-deconvolved host-z kernel (which is byte-identical in every other
    # mode). The per-host selection denominator D_g keeps the kernel machinery
    # (diagnostic only in this mode; the assembly never divides by it).
    # DERIVATION_GENERATOR_CONSISTENT_NORM.md §4.3 (G-iii); Mandel, Farr & Gair
    # (2019), arXiv:1809.02063 (P(det|x)=1 for detected data in numerators).
    # Issue #40(a): the delta-kernel numerator is now selectable independently
    # of the normalization leg via host_z_kernel ("auto" == this bundling).
    _use_generator_point = resolve_host_z_kernel(host_z_kernel, normalization_mode) == "point"

    # [PHYSICS] Issue #16 (user decision 2026-07-03): marginalize the residual
    # host peculiar-velocity dispersion into the host-z kernel.
    #   sigma_z_pv = (1 + z_g) * sigma_v / c
    # Davis et al. (2011), arXiv:1012.2912, Eqs. (1)/(A1) for the (1+z) factor
    # (z_obs = z_cos + (1 + z_cos) v_pec / c); added in quadrature to the
    # catalogue redshift error per standard practice (Mastrogiovanni et al.
    # 2023, arXiv:2305.10488, Sec. IV; EMRI precedent with the (1+z) factor:
    # Laghi et al. 2021, arXiv:2102.01708). The catalogue z_error already
    # carries GLADE+'s PV-CORRECTION error (or the 0.0015 parse-time floor);
    # SIGMA_V_PEC_KM_S is the residual (uncorrected/nonlinear) dispersion on
    # top of it. Applied ONCE here: every downstream consumer (window bounds,
    # Z_g renormalization, prior pdf, D_g, MC proposal + sampling_pdf) flows
    # through this single sigma and the one norm() object below, so the term
    # cannot double-count inside the likelihood. The ball-tree candidate
    # window and catalogue pruning (handler.py) intentionally keep the bare
    # catalogue z_error — a ±1σ, second-order candidate-list effect.
    sigma_z_pv = (1.0 + host_z) * SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S
    host_z_error_eff = float(np.sqrt(host_z_error**2 + sigma_z_pv**2))

    numerator_integration_upper_redshift_limit = dist_to_redshift(
        _det_d_L + integration_limit_sigma_multiplier * _det_d_L_unc, h=h
    )
    numerator_integration_lower_redshift_limit = dist_to_redshift(
        _det_d_L - integration_limit_sigma_multiplier * _det_d_L_unc, h=h
    )
    denominator_integration_upper_redshift_limit = (
        host_z + integration_limit_sigma_multiplier * host_z_error_eff
    )
    # [PHYSICS] clamp to z >= 0: for low-z photo-z hosts (z_g < 4 sigma_z) the window
    # would extend to unphysical z < 0 where comoving_volume_element still returns
    # positive values, silently adding prior mass to Z_g / D_g (G2b derivation note,
    # docs/derivations/G2b_host_z_volume_prior.md). Matches B_num's and D(h)'s z_min.
    # volume_trunc floors at exactly 0 (w_pop ∝ z² → 0 there, so this is a near-no-op
    # relative to 1e-6; the substantive volume_trunc change is the numerator window).
    _z_lower_floor = 0.0 if _use_volume_trunc else 1e-6
    denominator_integration_lower_redshift_limit = max(
        host_z - integration_limit_sigma_multiplier * host_z_error_eff, _z_lower_floor
    )

    # construct normal distribution for redshift and mass for host galaxy
    galaxy_redshift_normal_distribution = norm(loc=host_z, scale=host_z_error_eff)

    # [PHYSICS] De-rail fix #1 (commission, 2026-07-01): in-catalogue host-redshift prior.
    # "global"/"local_ratio" use the BARE photo-z Gaussian N(z; z_g, sigma_z) (unchanged
    # behaviour). "volume_deconv" DECONVOLVES the photo-z through the comoving-volume prior:
    #     p_g(z) = N(z; z_g, sigma_z) * w_pop(z) / Z_g ,  w_pop(z) = dVc/dz * (1+z)^-1 ,
    #     Z_g = INTEGRAL N(z; z_g, sigma_z) w_pop(z) dz  (per-galaxy renormalisation),
    # so the in-catalogue numerator AND denominator share the SAME z-prior that the
    # selection denominator D(h) = INTEGRAL (1/Npix) sum_k p_det * dVc/(1+z) dz already
    # carries. Removes the missing dd_L/dz-Jacobian Jensen bias (commission report bug #1).
    # Gray et al. (2020), arXiv:1908.06050, Eqs. A.10 / 33.
    # "volume_global" (diagnostic, G3 ablation cube) uses the SAME volume kernel
    # with the legacy global denominator selected in p_Di.
    # "volume_trunc" (shallow-venue Part 1) shares this volume-kernel weight and
    # differs only in the numerator integration support + z-floor (see above).
    # "mass_trunc" shares the SAME volume-deconvolved host-z kernel (only the
    # with-BH-mass mass-marginal differs), so it joins this set.
    # "absolute_marginal" (issue #30 Variant 1) keeps the volume_deconv kernel
    # unchanged (the kernel is exactly h-invariant, D1 §2 fact 2); only the p_Di
    # assembly differs. DERIVATION_ESTIMATOR_REDESIGN.md §3.1.
    # "generator_marginal" joins this set for the DENOMINATOR/Z_g machinery only
    # (byte-identical to absolute_marginal there); its NUMERATOR is the
    # point-evaluated delta kernel (see _use_generator_point above).
    _use_volume_deconv = normalization_mode in (
        "volume_deconv",
        "volume_global",
        "volume_trunc",
        "mass_trunc",
        "absolute_marginal",
        "generator_marginal",
    )
    _z_prior_norm = 1.0
    if _use_volume_deconv:

        def _z_prior_unnorm(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            w_pop = np.asarray(comoving_volume_element(z, h=h), dtype=np.float64) / (1.0 + z)
            base = np.asarray(galaxy_redshift_normal_distribution.pdf(z), dtype=np.float64)
            return base * w_pop

        _z_prior_norm = float(
            fixed_quad(
                _z_prior_unnorm,
                denominator_integration_lower_redshift_limit,
                denominator_integration_upper_redshift_limit,
                n=FIXED_QUAD_N,
            )[0]
        )
        if _z_prior_norm <= 0.0:
            _z_prior_norm = 1.0

    def galaxy_redshift_prior_pdf(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        base = np.asarray(galaxy_redshift_normal_distribution.pdf(z), dtype=np.float64)
        if _use_volume_deconv:
            w_pop = np.asarray(comoving_volume_element(z, h=h), dtype=np.float64) / (1.0 + z)
            return base * w_pop / _z_prior_norm
        return base

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
        ) * galaxy_redshift_prior_pdf(z)

    def denominator_integrant_without_bh_mass(z: npt.NDArray[np.float64]) -> Any:
        d_L = dist_vectorized(z, h=h)
        phi = np.full_like(z, host_phiS)
        theta = np.full_like(z, host_qS)
        # Gray et al. (2020), arXiv:1908.06050, Eq. A.19: shared p_det function
        # with D(h) denominator (STAT-03 symmetry, commit a70d1a2).  Phase 44:
        # NN-fill below first bin (real injection statistic), zero above
        # injection horizon.
        # FIX-2: per-host D_g conditions on the node z (packet §5.1).
        p_det = detection_probability.detection_probability_without_bh_mass_interpolated_zero_fill(
            d_L, phi, theta, h=h, **_zres_z_kwargs(detection_probability, z)
        )
        return p_det * galaxy_redshift_prior_pdf(z)

    # volume_trunc integrates the numerator over the per-host galaxy window (shared
    # with Z_g and D_g) so the truncated host-z prior spans ONE support; the default
    # modes keep the event-level GW window [d_L(z_det ± 4σ)].
    if _use_volume_trunc:
        numerator_quad_lower = denominator_integration_lower_redshift_limit
        numerator_quad_upper = denominator_integration_upper_redshift_limit
    else:
        numerator_quad_lower = numerator_integration_lower_redshift_limit
        numerator_quad_upper = numerator_integration_upper_redshift_limit

    if _use_generator_point:
        # [PHYSICS] delta-kernel numerator: N_g = p(x | z_g, Omega_g) — the GW
        # 3D Gaussian point-evaluated at the catalogue redshift (the volume
        # weight normalizes away in the delta limit: w_pop(z_g)/w_pop(z_g) = 1).
        # DERIVATION_GENERATOR_CONSISTENT_NORM.md §4.3 (fully generator-exact).
        _z_point = np.array([host_z], dtype=np.float64)
        _d_L_point = np.asarray(dist_vectorized(_z_point, h=h), dtype=np.float64)
        _ldf_point = _d_L_point / _det_d_L
        _phi_point = np.full_like(_z_point, host_phiS)
        _theta_point = np.full_like(_z_point, host_qS)
        _x_obs_point = np.vstack([_phi_point, _theta_point, _ldf_point]).T
        single_host_likelihood_numerator_without_bh_mass = float(
            _mvn_pdf(_x_obs_point, _mean_3d, _cov_inv_3d, _log_norm_3d)[0]
        )
    else:
        (
            single_host_likelihood_numerator_without_bh_mass,
            single_host_likelihood_numerator_without_bh_mass_error,
        ) = fixed_quad(
            numerator_integrant_without_bh_mass,
            numerator_quad_lower,
            numerator_quad_upper,
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
        # [PHYSICS] G2d Eddington-in-M: in the calibrated kernels the host-mass
        # prior is the rate-weighted N(M; M_g, sigma_M) R_eff(M) / Z_M, which under
        # a locally log-linear R_eff is EXACTLY the shifted Gaussian
        # N(M; M_g (1 + alpha sigma_rel^2), sigma_M). Applied identically in the
        # numerator (mu_gal_frac) and the denominator sampler (proposal = prior,
        # so the importance weights stay p_det) — "counted exactly once" in M.
        # Empirical impact at GLADE sigma_M: 2D-channel mean shifts -0.020 in h
        # (.planning/gate/G7row9_eddington_m_impact.json). Derivation + residual
        # control: docs/derivations/G2d_host_mass_rate_prior.md.
        # mass_trunc computes the FULL truncated lognormal x R_eff mass marginal, so
        # it needs neither the G2d point shift nor the linear sigma_M; every other
        # calibrated mode uses the moment-matched effective mass.
        _host_M_eff = (
            eddington_shifted_host_mass(host_M, host_M_error)
            if (_use_volume_deconv and not _use_mass_trunc)
            else host_M
        )
        if _use_mass_trunc:
            # sigma_lnM (recovered from the stored linear error) + per-host Z_M for
            # the truncated lognormal x R_eff prior (see _mass_trunc_* helpers).
            _sigma_lnM = float(_mass_trunc_sigma_lnM(host_M, host_M_error))
            _Z_M = _mass_trunc_log_normalisation(host_M, _sigma_lnM).item()

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

            if _use_mass_trunc:
                # Truncated lognormal x R_eff mass marginal via Gauss-Hermite on the
                # narrow GW M_z peak (EXP-45). Supersedes the analytic Gaussian product.
                mz_integral = _mass_trunc_mz_integral(
                    mu_cond, math.sqrt(_sigma2_cond), 1.0 + z, _det_M, host_M, _sigma_lnM, _Z_M
                )
            else:
                # Galaxy mass in M_z_frac coordinates: M_z_frac = M_gal * (1+z) / M_z_det
                # Eq. (14.22) in derivations/dark_siren_likelihood.md
                # NOTE: (1+z) here is CORRECT -- it is the coordinate transform, not a Jacobian
                # _host_M_eff carries the G2d Eddington-in-M rate-prior shift (see above).
                mu_gal_frac = _host_M_eff * (1 + z) / _det_M
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
            return gw_3d * mz_integral * galaxy_redshift_prior_pdf(z)

        if _use_generator_point:
            # [PHYSICS] delta-kernel with-BH numerator: gw_3d(z_g) * mz(z_g).
            # The galaxy MASS-error kernel is intentionally retained (pre-existing
            # point-M treatment tracked under issue #24; only the z-kernel
            # collapses per (G-iii)). DERIVATION_GENERATOR_CONSISTENT_NORM.md §4.3.
            _gw_3d_point = _mvn_pdf(_x_obs_point, _mean_3d, _cov_inv_3d, _log_norm_3d)
            _mu_cond_point = _mu_obs_4d[3] + (_x_obs_point - _mu_obs_4d[:3]) @ _proj
            # mass_trunc is a distinct mode; the generator point path always uses
            # the analytic Gaussian product (Eq. 14.31) at z = z_g.
            _mu_gal_frac_point = _host_M_eff * (1 + _z_point) / _det_M
            _sigma_gal_frac_point = host_M_error * (1 + _z_point) / _det_M
            _sigma2_sum_point = _sigma2_cond + _sigma_gal_frac_point**2
            _mz_point = np.exp(
                -0.5 * (_mu_cond_point - _mu_gal_frac_point) ** 2 / _sigma2_sum_point
            ) / np.sqrt(2 * np.pi * _sigma2_sum_point)
            single_host_likelihood_numerator_with_bh_mass = float((_gw_3d_point * _mz_point)[0])
        else:
            single_host_likelihood_numerator_with_bh_mass = fixed_quad(
                numerator_integrant_with_bh_mass,
                numerator_quad_lower,
                numerator_quad_upper,
                n=FIXED_QUAD_N,
            )[0]

        # Eq. (14.33) in derivations/dark_siren_likelihood.md
        # Denominator D_g = INTEGRAL p_gal(z) [ INTEGRAL p_det(d_L(z), M(1+z)) N(M) dM ] dz.
        # No GW likelihood, no mz_integral, no /(1+z) -- confirmed correct by Phase 14.
        #
        # [PHYSICS] 2026-07-08: EXACT semi-analytic estimator ("glz64"), replacing the
        # 10k-sample MC. p_det is piecewise-linear in M_z on the injection grid, so the
        # inner M-integral is closed form (erf-sum, zero M-quadrature error;
        # _bh_mass_denominator_inner_m_integral), and the outer z-integral is
        # Gauss-Legendre over the SAME host window [den_lo, den_hi] as the 3D
        # denominator and the Z_g normalisation. Deterministic, ~200x more accurate
        # than the MC (its ~1-5% noise removed) and ~4.5x faster. The MC sampled the
        # UNTRUNCATED z-Gaussian and over-counted the beyond-window / z<0 tail (~0.5%
        # for wide photo-z hosts); the host prior N(z; z_g, sigma_z) is normalised over
        # this window (Z_g), so D_g is a proper window-averaged p_det in [0, 1].
        # Owen (1980) first-moment identity; Gray et al. (2020), arXiv:1908.06050 Eq. A.19.
        def denominator_integrant_with_bh_mass(z: npt.NDArray[np.float64]) -> Any:
            if _use_mass_trunc:
                # Same truncated lognormal x R_eff prior as the numerator, so N_g and
                # D_g share ONE mass prior (Gauss-Legendre in ln M; the erf-sum closed
                # form is Gaussian-prior-only and does not apply).
                inner_m = _mass_trunc_denominator_inner_m_integral(
                    z, detection_probability, host_phiS, host_qS, host_M, _sigma_lnM, _Z_M, h
                )
            else:
                inner_m = _bh_mass_denominator_inner_m_integral(
                    z, detection_probability, host_phiS, host_qS, _host_M_eff, host_M_error, h
                )
            return inner_m * galaxy_redshift_prior_pdf(z)

        single_host_likelihood_denominator_with_bh_mass = fixed_quad(
            denominator_integrant_with_bh_mass,
            denominator_integration_lower_redshift_limit,
            denominator_integration_upper_redshift_limit,
            n=_BH_DENOM_QUAD_ORDER,
        )[0]

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


def single_host_likelihood_batch(
    host_phiS: npt.NDArray[np.float64],
    host_qS: npt.NDArray[np.float64],
    host_z: npt.NDArray[np.float64],
    host_z_error: npt.NDArray[np.float64],
    host_M: npt.NDArray[np.float64],
    host_M_error: npt.NDArray[np.float64],
    detection_index: int,
    h: float,
    evaluate_with_bh_mass: bool,
    # [PHYSICS] production default since 2026-07-26 (MULTISEED_READOUT_20260726.md)
    normalization_mode: str = "generator_marginal",
    # Issue #40(a): numerator host-z kernel decomposition flag ("auto" == the
    # historical bundling; see the scalar kernel and resolve_host_z_kernel).
    host_z_kernel: str = "auto",
) -> npt.NDArray[np.float64]:
    """Host-batched twin of :func:`single_host_likelihood`.

    Computes the per-host likelihood integrals for ``n`` candidate hosts of one
    detection in a single vectorized pass. Row ``i`` of the result equals
    ``single_host_likelihood(...)`` called with host ``i``'s scalars — the same
    physics, the same quadrature (fixed_quad's exact affine node map and
    reduction, see :func:`_batched_gl_nodes`/:func:`_batched_gl_reduce`), the
    same Gaussian pdf operation order (:func:`_gaussian_pdf`) — with the host
    loop moved from Python/starmap into the array axis. Eliminated per-host
    costs: ``scipy.stats.norm`` frozen-distribution construction, the
    event-level ``dist_to_redshift`` window calls (now once per batch), and
    per-host ``p_det`` interpolator calls (now one call over all hosts' nodes).

    Reads the ``child_process_init`` worker globals (the subset the scalar
    kernel actually uses). ``base_seed`` is intentionally absent: it was a
    dead parameter of the scalar signature (vestigial from the removed MC
    denominator).

    Args:
        host_phiS: Host ecliptic azimuths, shape ``(n,)``.
        host_qS: Host ecliptic polar angles, shape ``(n,)``.
        host_z: Host redshifts, shape ``(n,)``.
        host_z_error: Host redshift 1-sigma errors, shape ``(n,)``.
        host_M: Host BH masses [M_sun], shape ``(n,)``.
        host_M_error: Host BH mass 1-sigma errors, shape ``(n,)``.
        detection_index: CRB row index of the detection.
        h: Dimensionless Hubble parameter.
        evaluate_with_bh_mass: Include the with-BH-mass channel.
        normalization_mode: In-catalogue normalization mode (see ``p_Di``).
        host_z_kernel: Numerator host-z kernel selection (issue #40a);
            ``"auto"`` reproduces the historical mode bundling.

    Returns:
        Array of shape ``(n, 6)`` when ``evaluate_with_bh_mass`` else
        ``(n, 4)``; columns match the scalar kernel's return list.
    """
    global detection_probability
    global means_3d, cov_inv_3d, log_norm_3d
    global means_4d
    global det_index_to_slot
    global sigma2_cond_arr, proj_arr
    global det_d_L_arr, det_d_L_unc_arr, det_M_arr

    n = int(host_z.size)
    if n == 0:
        return np.empty((0, 6 if evaluate_with_bh_mass else 4), dtype=np.float64)

    slot = det_index_to_slot[detection_index]
    _det_d_L = float(det_d_L_arr[slot])
    _det_d_L_unc = float(det_d_L_unc_arr[slot])
    _det_M = float(det_M_arr[slot])
    _mean_3d = means_3d[slot]
    _cov_inv_3d = cov_inv_3d[slot]
    _log_norm_3d = float(log_norm_3d[slot])

    integration_limit_sigma_multiplier = 4.0

    # Residual peculiar-velocity dispersion folded into the host-z kernel —
    # identical formula and references as the scalar kernel (issue #16).
    sigma_z_pv = (1.0 + host_z) * SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S
    host_z_error_eff = np.sqrt(host_z_error**2 + sigma_z_pv**2)

    # Numerator window depends only on the event (and h): computed once per batch.
    numerator_integration_upper_redshift_limit = dist_to_redshift(
        _det_d_L + integration_limit_sigma_multiplier * _det_d_L_unc, h=h
    )
    numerator_integration_lower_redshift_limit = dist_to_redshift(
        _det_d_L - integration_limit_sigma_multiplier * _det_d_L_unc, h=h
    )
    den_hi = host_z + integration_limit_sigma_multiplier * host_z_error_eff
    # z >= 0 clamp: same G2b rationale as the scalar kernel. volume_trunc floors at
    # exactly 0 (w_pop ∝ z² → 0 there) instead of 1e-6.
    _use_volume_trunc = normalization_mode == "volume_trunc"
    # mass_trunc (EXP-45): truncated lognormal x R_eff host-mass prior in the 2D
    # channel; shares the volume_deconv host-z kernel (see scalar path).
    _use_mass_trunc = normalization_mode == "mass_trunc"
    # generator_marginal (E1 FIX-3): point/point sigma_z pairing — the numerator
    # is the GW likelihood POINT-evaluated at the catalogue z_g (delta kernel);
    # see the scalar kernel for the physics comment and references. Issue
    # #40(a): selectable independently via host_z_kernel ("auto" == bundling).
    _use_generator_point = resolve_host_z_kernel(host_z_kernel, normalization_mode) == "point"
    _z_lower_floor = 0.0 if _use_volume_trunc else 1e-6
    den_lo = np.maximum(
        host_z - integration_limit_sigma_multiplier * host_z_error_eff, _z_lower_floor
    )

    # generator_marginal joins the volume_deconv set for the DENOMINATOR/Z_g
    # machinery only (byte-identical to absolute_marginal there); its numerator
    # is point-evaluated (see _use_generator_point).
    _use_volume_deconv = normalization_mode in (
        "volume_deconv",
        "volume_global",
        "volume_trunc",
        "mass_trunc",
        "absolute_marginal",
        "generator_marginal",
    )

    # Per-host denominator quadrature nodes (fixed_quad affine map, n=50).
    y_den = _batched_gl_nodes(den_lo, den_hi, _GL_NODES_50)  # (n, 50)
    gauss_den = _gaussian_pdf(y_den, host_z[:, None], host_z_error_eff[:, None])

    z_prior_norm = np.ones(n, dtype=np.float64)
    w_pop_den: npt.NDArray[np.float64] | None = None
    if _use_volume_deconv:
        y_den_flat = y_den.reshape(-1)
        w_pop_den = (
            np.asarray(comoving_volume_element(y_den_flat, h=h), dtype=np.float64)
            / (1.0 + y_den_flat)
        ).reshape(n, _HOST_QUAD_N)
        z_prior_norm = _batched_gl_reduce(den_lo, den_hi, _GL_WEIGHTS_50, gauss_den * w_pop_den)
        z_prior_norm = np.where(z_prior_norm <= 0.0, 1.0, z_prior_norm)

    # Numerator quadrature nodes, all shaped (n, 50). Default modes share ONE
    # event-level window across every host (the shared-node optimization — the
    # per-host arrays are broadcast views of the shared (50,) nodes). volume_trunc
    # integrates the numerator over each host's galaxy window [den_lo, den_hi]
    # (== the denominator nodes y_den), so the numerator becomes genuinely
    # per-host; the shared-node optimization is dropped for the numerator only
    # (the denominator path is already per-host). y_num_nodes carries (1 + z) for
    # the with-BH-mass mass-fraction coordinate transform below.
    if _use_generator_point:
        # [PHYSICS] delta-kernel numerator (generator_marginal): a single "node"
        # column at the catalogue z_g keeps the downstream (n, k) machinery
        # shared; no quadrature reduce is applied (see below). Scalar-twin ops:
        # dist at host_z, fraction against the event d_L.
        # DERIVATION_GENERATOR_CONSISTENT_NORM.md §4.3 (G-iii).
        y_num_nodes = host_z[:, None]  # (n, 1)
        d_L_num_point = np.asarray(dist_vectorized(host_z, h=h), dtype=np.float64)  # (n,)
        luminosity_distance_fraction: npt.NDArray[np.floating[Any]] = (d_L_num_point / _det_d_L)[
            :, None
        ]  # (n, 1)
        num_reduce_lo = np.zeros(n)  # unused on the point path
        num_reduce_hi = np.zeros(n)  # unused on the point path
    elif _use_volume_trunc:
        y_num_nodes = y_den  # (n, 50)
        d_L_num = dist_vectorized(y_num_nodes.reshape(-1), h=h).reshape(n, _HOST_QUAD_N)
        luminosity_distance_fraction = d_L_num / _det_d_L  # (n, 50)
        num_reduce_lo = den_lo
        num_reduce_hi = den_hi
    else:
        y_num_1d = (
            numerator_integration_upper_redshift_limit - numerator_integration_lower_redshift_limit
        ) * (_GL_NODES_50 + 1) / 2.0 + numerator_integration_lower_redshift_limit  # (50,)
        y_num_nodes = np.broadcast_to(y_num_1d[None, :], (n, _HOST_QUAD_N))  # (n, 50)
        d_L_num = dist_vectorized(y_num_1d, h=h)  # (50,)
        luminosity_distance_fraction = np.broadcast_to(
            (d_L_num / _det_d_L)[None, :], (n, _HOST_QUAD_N)
        )  # (n, 50)
        num_reduce_lo = np.full(n, numerator_integration_lower_redshift_limit)
        num_reduce_hi = np.full(n, numerator_integration_upper_redshift_limit)

    w_pop_num: npt.NDArray[np.float64] | None = None
    if _use_volume_deconv and not _use_generator_point:
        if _use_volume_trunc:
            # Numerator nodes == denominator nodes -> reuse the denominator w_pop.
            w_pop_num = w_pop_den
        else:
            w_pop_num_1d = np.asarray(comoving_volume_element(y_num_1d, h=h), dtype=np.float64) / (
                1.0 + y_num_1d
            )
            w_pop_num = np.broadcast_to(w_pop_num_1d[None, :], (n, _HOST_QUAD_N))  # (n, 50)

    def _z_prior_pdf_at(
        z_nodes: npt.NDArray[np.float64], w_pop: npt.NDArray[np.float64] | None
    ) -> npt.NDArray[np.float64]:
        """Per-host z-prior pdf at ``(n, k)`` nodes; mirrors galaxy_redshift_prior_pdf."""
        base = _gaussian_pdf(z_nodes, host_z[:, None], host_z_error_eff[:, None])
        if _use_volume_deconv:
            assert w_pop is not None
            return base * w_pop / z_prior_norm[:, None]
        return base

    # Point mode has no numerator z-kernel: prior_num stays None (delta kernel).
    prior_num: npt.NDArray[np.float64] | None = (
        None if _use_generator_point else _z_prior_pdf_at(y_num_nodes, w_pop_num)
    )  # (n, 50) in the quadrature modes
    # (n, 50); same values the scalar integrand recomputes at y_den
    if _use_volume_deconv and w_pop_den is not None:
        prior_den = gauss_den * w_pop_den / z_prior_norm[:, None]
    else:
        prior_den = gauss_den

    # 3D GW likelihood at the numerator nodes, batched over hosts.
    # k_num = 1 in the generator point mode (single delta-kernel column);
    # _HOST_QUAD_N in every quadrature mode (value-identical to the pre-change
    # constant-shape code there).
    _k_num = int(y_num_nodes.shape[1])
    x_obs = np.empty((n, _k_num, 3), dtype=np.float64)
    x_obs[:, :, 0] = host_phiS[:, None]
    x_obs[:, :, 1] = host_qS[:, None]
    x_obs[:, :, 2] = luminosity_distance_fraction  # (n, k_num)
    gw_3d = _mvn_pdf(x_obs.reshape(n * _k_num, 3), _mean_3d, _cov_inv_3d, _log_norm_3d)
    gw_3d = gw_3d.reshape(n, _k_num)

    if _use_generator_point:
        # [PHYSICS] N_g = p(x | z_g, Omega_g): point value, no reduce.
        # DERIVATION_GENERATOR_CONSISTENT_NORM.md §4.3.
        numerator_without_bh_mass = gw_3d[:, 0]
    else:
        assert prior_num is not None
        numerator_without_bh_mass = _batched_gl_reduce(
            num_reduce_lo,
            num_reduce_hi,
            _GL_WEIGHTS_50,
            gw_3d * prior_num,
        )

    # 3D denominator: batched p_det lookup over all hosts' nodes at once.
    d_L_den = dist_vectorized(y_den.reshape(-1), h=h)
    p_det_den = np.asarray(
        detection_probability.detection_probability_without_bh_mass_interpolated_zero_fill(
            d_L_den,
            np.repeat(host_phiS, _HOST_QUAD_N),
            np.repeat(host_qS, _HOST_QUAD_N),
            h=h,
            # FIX-2: per-host D_g conditions on the node z (packet §5.1).
            **_zres_z_kwargs(detection_probability, y_den.reshape(-1)),
        ),
        dtype=np.float64,
    ).reshape(n, _HOST_QUAD_N)
    denominator_without_bh_mass = _batched_gl_reduce(
        den_lo, den_hi, _GL_WEIGHTS_50, p_det_den * prior_den
    )

    # STAT-04 off-grid quadrature-weight diagnostics (same expressions as scalar).
    _, _interp_1d = detection_probability._get_or_build_grid(h)
    _dl_centers = _interp_1d.grid[0]
    _dl_grid_min = float(_dl_centers[0])
    _dl_grid_max = float(_dl_centers[-1])

    # Numerator side is event-level: identical for every host of this batch.
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
        _w_num_scalar = float(np.clip(_below_min_num + _above_max_num, 0.0, 1.0))
    else:
        _w_num_scalar = 0.0
    quadrature_weight_outside_grid_numerator = np.full(n, _w_num_scalar, dtype=np.float64)

    # Denominator side is per-host.
    _dl_lower_den = dist_vectorized(den_lo, h=h)
    _dl_upper_den = dist_vectorized(den_hi, h=h)
    _window_den = _dl_upper_den - _dl_lower_den
    with np.errstate(divide="ignore", invalid="ignore"):
        _below_min_den = (
            np.maximum(0.0, np.minimum(_dl_upper_den, _dl_grid_min) - _dl_lower_den) / _window_den
        )
        _above_max_den = (
            np.maximum(0.0, _dl_upper_den - np.maximum(_dl_lower_den, _dl_grid_max)) / _window_den
        )
        quadrature_weight_outside_grid_denominator = np.where(
            _window_den > 0.0,
            np.clip(_below_min_den + _above_max_den, 0.0, 1.0),
            0.0,
        )

    for _flagged in np.flatnonzero(
        (quadrature_weight_outside_grid_numerator > 0.05)
        | (quadrature_weight_outside_grid_denominator > 0.05)
    ):
        _LOGGER.warning(
            "Event %d: >5%% quadrature weight outside P_det grid — "
            "numerator=%.3f, denominator=%.3f",
            detection_index,
            quadrature_weight_outside_grid_numerator[_flagged],
            quadrature_weight_outside_grid_denominator[_flagged],
        )

    if not evaluate_with_bh_mass:
        return np.column_stack(
            [
                numerator_without_bh_mass,
                denominator_without_bh_mass,
                quadrature_weight_outside_grid_numerator,
                quadrature_weight_outside_grid_denominator,
            ]
        )

    # --- with-BH-mass channel ---
    # G2d Eddington-in-M shift: scalar helper kept per host (data-dependent
    # early returns/clamps; negligible cost) — bit-identical to the scalar path.
    # mass_trunc uses neither the point shift nor the linear sigma_M (it integrates
    # the full truncated lognormal x R_eff prior), so skip the per-host quadrature.
    if _use_volume_deconv and not _use_mass_trunc:
        host_M_eff = np.array(
            [
                eddington_shifted_host_mass(float(m), float(dm_))
                for m, dm_ in zip(host_M, host_M_error)
            ],
            dtype=np.float64,
        )
    else:
        host_M_eff = np.asarray(host_M, dtype=np.float64)

    if _use_mass_trunc:
        # Per-host sigma_lnM (recovered from the stored linear error) and Z_M for the
        # truncated lognormal x R_eff prior; (n,)-vectorised, bit-identical to scalar.
        sigma_lnM = _mass_trunc_sigma_lnM(host_M, host_M_error)  # (n,)
        Z_M = _mass_trunc_log_normalisation(host_M, sigma_lnM)  # (n,)

    _sigma2_cond = float(sigma2_cond_arr[slot])
    _proj = proj_arr[slot]
    _mu_obs_4d = means_4d[slot]

    # Conditional mean of M_z_frac given (phi, theta, d_L_frac); Eq. (14.23)-(14.28).
    mu_cond = (_mu_obs_4d[3] + (x_obs.reshape(n * _k_num, 3) - _mu_obs_4d[:3]) @ _proj).reshape(
        n, _k_num
    )
    # (1 + z) mass-fraction coordinate transform at the numerator nodes y_num_nodes
    # (n, 50): broadcast of the shared window for the default modes, the per-host
    # galaxy window for volume_trunc.
    if _use_mass_trunc:
        # Truncated lognormal x R_eff mass marginal via Gauss-Hermite on the narrow
        # GW M_z peak (EXP-45); (n, 50) matches the analytic branch shape.
        mz_integral = _mass_trunc_mz_integral(
            mu_cond, math.sqrt(_sigma2_cond), 1.0 + y_num_nodes, _det_M, host_M, sigma_lnM, Z_M
        )
    else:
        mu_gal_frac = host_M_eff[:, None] * (1 + y_num_nodes) / _det_M
        sigma_gal_frac = host_M_error[:, None] * (1 + y_num_nodes) / _det_M

        # Analytic Gaussian product integral, Eq. (14.31).
        sigma2_sum = _sigma2_cond + sigma_gal_frac**2
        mz_integral = np.exp(-0.5 * (mu_cond - mu_gal_frac) ** 2 / sigma2_sum) / np.sqrt(
            2 * np.pi * sigma2_sum
        )

    if _use_generator_point:
        # [PHYSICS] with-BH point numerator: gw_3d(z_g) * mz(z_g); the galaxy
        # mass-error kernel is retained (issue #24), only the z-kernel collapses.
        # DERIVATION_GENERATOR_CONSISTENT_NORM.md §4.3.
        numerator_with_bh_mass = (gw_3d * mz_integral)[:, 0]
    else:
        assert prior_num is not None
        numerator_with_bh_mass = _batched_gl_reduce(
            num_reduce_lo,
            num_reduce_hi,
            _GL_WEIGHTS_50,
            gw_3d * mz_integral * prior_num,
        )

    # Semi-analytic denominator (glz64): batched erf-sum inner-M + GL outer-z.
    y_bh = _batched_gl_nodes(den_lo, den_hi, _GL_NODES_64)  # (n, 64)
    if _use_mass_trunc:
        # Same truncated lognormal x R_eff prior as the numerator (GL in ln M); shares
        # the mass prior between N_g and D_g. Row i bit-identical to the scalar path.
        inner_m = _mass_trunc_denominator_inner_m_integral_batch(
            y_bh, detection_probability, host_phiS, host_qS, host_M, sigma_lnM, Z_M, h
        )
    else:
        inner_m = _bh_mass_denominator_inner_m_integral_batch(
            y_bh, detection_probability, host_phiS, host_qS, host_M_eff, host_M_error, h
        )
    w_pop_bh: npt.NDArray[np.float64] | None = None
    if _use_volume_deconv:
        y_bh_flat = y_bh.reshape(-1)
        w_pop_bh = (
            np.asarray(comoving_volume_element(y_bh_flat, h=h), dtype=np.float64)
            / (1.0 + y_bh_flat)
        ).reshape(n, _BH_DENOM_QUAD_ORDER)
    prior_bh = _z_prior_pdf_at(y_bh, w_pop_bh)
    denominator_with_bh_mass = _batched_gl_reduce(
        den_lo, den_hi, _GL_WEIGHTS_64, inner_m * prior_bh
    )

    return np.column_stack(
        [
            numerator_without_bh_mass,
            denominator_without_bh_mass,
            numerator_with_bh_mass,
            denominator_with_bh_mass,
            quadrature_weight_outside_grid_numerator,
            quadrature_weight_outside_grid_denominator,
        ]
    )


def _hosts_to_arrays(
    hosts: list[HostGalaxy],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Column-major float64 arrays (phiS, qS, z, z_error, M, M_error) for a host list."""
    return (
        np.array([host.phiS for host in hosts], dtype=np.float64),
        np.array([host.qS for host in hosts], dtype=np.float64),
        np.array([host.z for host in hosts], dtype=np.float64),
        np.array([host.z_error for host in hosts], dtype=np.float64),
        np.array([host.M for host in hosts], dtype=np.float64),
        np.array([host.M_error for host in hosts], dtype=np.float64),
    )


def _starmap_host_batches(
    pool: mp.pool.Pool,
    hosts: list[HostGalaxy],
    detection_index: int,
    h: float,
    evaluate_with_bh_mass: bool,
    normalization_mode: str,
    host_z_kernel: str = "auto",
) -> list[list[float]]:
    """Dispatch the batched host kernel over worker processes.

    Splits ``hosts`` into at most ``pool._processes`` contiguous chunks (order
    preserved) and runs :func:`single_host_likelihood_batch` on each chunk in
    parallel. Returns one ``list[float]`` per host in the original order —
    exactly the structure the per-host ``single_host_likelihood`` starmap
    produced.

    Args:
        pool: Multiprocessing pool initialised via ``child_process_init``.
        hosts: Candidate hosts for the detection.
        detection_index: CRB row index of the detection.
        h: Dimensionless Hubble parameter.
        evaluate_with_bh_mass: Include the with-BH-mass channel.
        normalization_mode: In-catalogue normalization mode.
        host_z_kernel: Numerator host-z kernel selection (issue #40a).

    Returns:
        Per-host result rows in input order.
    """
    n = len(hosts)
    if n == 0:
        return []
    arrays = _hosts_to_arrays(hosts)
    # One chunk per worker, but never more than _MAX_BATCH_CHUNK hosts per
    # chunk: the with-BH erf-sum block allocates ~(chunk, 64, 40) float64
    # intermediates (~300-400 MB/worker at 1080 hosts), so few-worker runs on
    # events with tens of thousands of candidates must split further. Chunk
    # boundaries do not affect values (order-preserving; gated by
    # test_starmap_host_batches_ordering_and_chunking).
    n_chunks = min(n, max(pool._processes, math.ceil(n / _MAX_BATCH_CHUNK)))  # type: ignore[attr-defined]
    chunk_indices = np.array_split(np.arange(n), n_chunks)
    jobs = [
        tuple(a[idx] for a in arrays)
        + (detection_index, h, evaluate_with_bh_mass, normalization_mode, host_z_kernel)
        for idx in chunk_indices
    ]
    chunk_results = pool.starmap(single_host_likelihood_batch, jobs)
    rows: list[list[float]] = []
    for chunk in chunk_results:
        rows.extend(chunk.tolist())
    return rows


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
    # [PHYSICS] Issue #16: mirror the production kernel's residual-PV quadrature
    # (see single_host_likelihood) so the integration-testing twin stays a
    # faithful cross-check of the production path.
    _sigma_z_pv = (1.0 + possible_host.z) * SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S
    _z_error_eff = float(np.sqrt(possible_host.z_error**2 + _sigma_z_pv**2))
    galaxy_redshift_normal_distribution = norm(loc=possible_host.z, scale=_z_error_eff)

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
                d_L,
                possible_host.phiS,
                possible_host.qS,
                h=h,
                **_zres_z_kwargs(detection_probability, z),
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
                d_L,
                possible_host.phiS,
                possible_host.qS,
                h=h,
                **_zres_z_kwargs(detection_probability, z),
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
