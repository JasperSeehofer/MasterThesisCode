"""Hubble constant posterior evaluation.

:class:`BayesianStatistics` loads saved Cramér-Rao bounds and orchestrates the
full Hubble-constant posterior evaluation using the real GLADE galaxy catalog,
simulation-based :class:`~darksiren_emri.bayesian_inference.simulation_detection_probability.SimulationDetectionProbability`,
full Fisher-matrix covariance, and multiprocessing.

Invoked via ``main.py:evaluate()`` / ``--evaluate`` CLI flag.
Output is written to ``simulations/posteriors/`` as JSON.
"""

import csv
import functools
import json
import logging
import math
import multiprocessing as mp
import os
import time
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.integrate import dblquad, fixed_quad, quad
from scipy.special import ndtr, roots_hermite, roots_legendre
from scipy.stats import multivariate_normal, norm

from darksiren_emri.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)
from darksiren_emri.constants import (
    CRAMER_RAO_BOUNDS_OUTPUT_PATH,
    HOST_DRAW_Z_MAX,
    INJECTION_DATA_DIR,
    M_SOURCE_FRAME_MAX,
    M_SOURCE_FRAME_MIN,
    PREPARED_CRAMER_RAO_BOUNDS_PATH,
    SIGMA_V_PEC_KM_S,
    SNR_THRESHOLD,
    SPEED_OF_LIGHT_KM_S,
    H,
)
from darksiren_emri.cosmological_model import LamCDMScenario, Model1CrossCheck
from darksiren_emri.dark_siren_injection import (
    _redshift_population_weight,
    dark_mass_log10_density_unnormalised,
)
from darksiren_emri.datamodels.detection import (
    Detection,
    _sky_localization_uncertainty,
)
from darksiren_emri.emri_rate import R_eff_per_mbh
from darksiren_emri.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    HostGalaxy,
    InternalCatalogColumns,
)
from darksiren_emri.galaxy_catalogue.pixel_completeness import (
    CompletenessModel,
    from_cache_or_build,
)
from darksiren_emri.physical_relations import (
    comoving_volume_element,
    dist,
    dist_to_redshift,
    dist_vectorized,
    get_redshift_outer_bounds,
)

_LOGGER = logging.getLogger()

# Per-process dedup state for the out-of-grid quadrature warning: the check runs
# per (event, host) and produced O(10^5) identical lines per task on large
# campaigns, so we warn once per event and count suppressed repeats instead.
_quadrature_outside_grid_warned_events: set[int] = set()
_quadrature_outside_grid_suppressed_repeats: int = 0


def _warn_quadrature_weight_outside_grid(
    detection_index: int,
    weight_outside_numerator: float,
    weight_outside_denominator: float,
) -> None:
    """Emit the >5% out-of-grid quadrature warning at most once per event.

    Subsequent occurrences for the same event (per worker process) only
    increment ``_quadrature_outside_grid_suppressed_repeats``. Logging-only:
    the returned diagnostic weights are unaffected.
    """
    global _quadrature_outside_grid_suppressed_repeats
    if detection_index in _quadrature_outside_grid_warned_events:
        _quadrature_outside_grid_suppressed_repeats += 1
        return
    _quadrature_outside_grid_warned_events.add(detection_index)
    _LOGGER.warning(
        "Event %d: >5%% quadrature weight outside P_det grid — "
        "numerator=%.3f, denominator=%.3f (repeats for this event suppressed)",
        detection_index,
        weight_outside_numerator,
        weight_outside_denominator,
    )


# [P3-IMP] GATE E-P3 (A13) engagement evidence: per-process, per-dispatch-path
# one-shot log that the twin cell's factor was actually applied. Read from run
# logs by the registered scorer (PREREGISTRATION_P3_TWIN_20260822.md §4).
_p3_engagement_logged_paths: set[str] = set()


def _p3_engagement_log_once(path: str) -> None:
    if path in _p3_engagement_logged_paths:
        return
    _p3_engagement_logged_paths.add(path)
    _LOGGER.info(
        "[P3-IMP] catalogue_numerator_survival='phi' ENGAGED in the %s host "
        "path (GATE E-P3 dispatch evidence; once per worker process)",
        path,
    )


# Per-process dedup state for the C7 ZoA host-z-kernel fallback (see
# _warn_zoa_hostz_kernel_fallback): hosts whose HEALPix pixel is empty carry
# f_k == 0 at every redshift, so the catalogued-host intensity f*w_pop vanishes
# on the whole window and the kernel falls back to the pre-C7 (f == 1) form.
_zoa_hostz_kernel_fallback_warned: bool = False
_zoa_hostz_kernel_fallback_hosts: int = 0


def _warn_zoa_hostz_kernel_fallback(detection_index: int, n_hosts: int) -> None:
    """Warn once per worker that the ZoA host-z-kernel fallback engaged.

    ``n_hosts`` hosts of this event sit in a pixel whose completeness is
    identically zero across their whole ``+/-4 sigma`` window. There the
    catalogued-host intensity carries no information, so the kernel reverts to
    the pre-C7 ``w_pop``-only form for those hosts (GATE_PACKAGE_FINAL.md
    §1.1 B5: elementwise clamping is forbidden — it would install a kink where
    ``f_k`` crosses a floor partway across the window).
    """
    global _zoa_hostz_kernel_fallback_warned, _zoa_hostz_kernel_fallback_hosts
    _zoa_hostz_kernel_fallback_hosts += n_hosts
    if _zoa_hostz_kernel_fallback_warned:
        return
    _zoa_hostz_kernel_fallback_warned = True
    _LOGGER.warning(
        "Event %d: %d host(s) have f_k == 0 across the whole host-z window "
        "(empty/ZoA pixel) — host-z kernel falls back to the w_pop-only form "
        "for those hosts (further occurrences in this worker are suppressed).",
        detection_index,
        n_hosts,
    )


DEFAULT_GALAXY_Z_ERROR = 0.0015

# Issue #40(a) decomposition flag (redteam F2/F3): the in-catalogue NUMERATOR
# host-z kernel, historically bundled into normalization_mode. "auto" preserves
# that bundling exactly (delta kernel iff generator_marginal); "point" /
# "volume_deconv" force the numerator kernel independently of the
# normalization leg (n_hat_w / D_gen machinery, which stays mode-selected).
HOST_Z_KERNEL_CHOICES = ("auto", "point", "volume_deconv")


def resolve_host_z_kernel(
    host_z_kernel: str, normalization_mode: str, *, catalogue_scattered: bool = False
) -> str:
    """Resolve the numerator host-z kernel selection to 'point' or 'volume_deconv'.

    Decomposition flag for issue #40(a): makes the delta-kernel (point/point)
    in-catalogue numerator separately selectable from the normalization leg.
    ``"auto"`` reproduces the historical bundling — the delta kernel if and
    only if ``normalization_mode == "generator_marginal"`` — so the production
    default path is unchanged. Explicit ``"point"`` / ``"volume_deconv"``
    override the numerator kernel only; the selection-normalization machinery
    (``n_hat_w``/``D_gen`` vs ``n_bar_w``/``D``) remains governed by
    ``normalization_mode``.

    [PHYSICS] Scatter guard (realistic host-observation model, RATIFIED
    2026-07-29, docs/derivations/realistic_host_observation_model.md §3.1/§9
    guard 1): on a SCATTERED observed-catalogue realization
    (``catalogue_scattered=True``, i.e. sidecar sigma_scale > 0) the point
    (delta) kernel loses its licence — the realized data-generating process is
    exactly the width-sigma marginal, so point evaluation is a model error of
    order sigma_z * dln d_L/dz relative to sigma_dL/d_L. The guard is
    ONE-DIRECTIONAL: unscattered catalogues (default False) keep every
    baseline mode.

    Args:
        host_z_kernel: One of ``HOST_Z_KERNEL_CHOICES``.
        normalization_mode: The in-catalogue normalization mode (see ``p_Di``).
        catalogue_scattered: True iff the loaded catalogue is a scattered
            observed realization (``GalaxyCatalogueHandler.scattered``).

    Returns:
        ``"point"`` (delta kernel at the catalogue z_g) or ``"volume_deconv"``
        (the mode's own quadrature kernel — volume-deconvolved in the
        ``*_marginal`` modes, bare Gaussian in "global"/"local_ratio").

    Raises:
        ValueError: Unknown choice, or a point-resolving kernel on a
            scattered catalogue.
    """
    if host_z_kernel not in HOST_Z_KERNEL_CHOICES:
        raise ValueError(
            f"unknown host_z_kernel: {host_z_kernel!r} (expected one of {HOST_Z_KERNEL_CHOICES})"
        )
    resolved = (
        ("point" if normalization_mode == "generator_marginal" else "volume_deconv")
        if host_z_kernel == "auto"
        else host_z_kernel
    )
    if catalogue_scattered and resolved == "point":
        raise ValueError(
            "host-z kernel resolves to 'point' but the loaded catalogue is a "
            "SCATTERED observed realization (sidecar sigma_scale > 0): the "
            "delta kernel's licence is the unscattered-generator premise, which "
            "is false by construction under the realized noise — the truth is "
            "distributed with width sigma_z around each observed row, exactly "
            "the marginal the width kernel computes "
            "(docs/derivations/realistic_host_observation_model.md §3.1, guard "
            "§9.1). Use --host_z_kernel volume_deconv with "
            "--normalization_mode absolute_marginal (the ratified real-data "
            "pairing, [RATIFY-R3])."
        )
    return resolved


# [PHYSICS] Issue #40 remainder (2D mass-marginal, RATIFIED 2026-07-27,
# docs/derivations/mass_marginal_2d_kernel.md §4 item 1): the 2D host-MASS
# kernel, historically bundled into normalization_mode ("mass_trunc" mode).
# "auto" preserves that bundling exactly (trunc_lognormal iff mass_trunc);
# "gaussian" / "trunc_lognormal" force the mass kernel independently, so the
# ratified real-data combination (absolute_marginal normalization x
# volume_deconv host-z kernel x trunc_lognormal mass kernel) is expressible.
HOST_MASS_KERNEL_CHOICES = ("auto", "gaussian", "trunc_lognormal")


def resolve_host_mass_kernel(
    host_mass_kernel: str,
    normalization_mode: str,
    host_z_kernel: str,
    *,
    catalogue_scattered: bool = False,
) -> str:
    """Resolve the 2D host-mass kernel selection to 'gaussian' or 'trunc_lognormal'.

    Decomposition flag for the #40 remainder (RATIFY-M3/M4,
    docs/derivations/mass_marginal_2d_kernel.md): makes the truncated
    lognormal x R_eff mass kernel separately selectable from the
    normalization leg. ``"auto"`` reproduces the historical bundling — the
    truncated kernel if and only if ``normalization_mode == "mass_trunc"`` —
    so the production default path is unchanged.

    Guard (derivation §3.3): the delta-kernel (point) host-z numerator path
    always evaluates the analytic Gaussian mass product at the catalogue
    ``host_M`` (issue #24 point-M treatment), while the trunc_lognormal
    denominator carries the LN x R_eff prior — N_g and D_g would silently use
    DIFFERENT mass priors, violating the counted-once-in-M invariant. That
    combination raises instead of running silently.

    [PHYSICS] Scatter guard (realistic host-observation model, RATIFIED
    2026-07-29, docs/derivations/realistic_host_observation_model.md §9,
    with-BH channel): on a SCATTERED observed-catalogue realization the
    point-M treatment — the analytic mass product point-anchored by the
    delta host-z numerator (issue #24 pairing) — loses its licence exactly
    like the z delta kernel: the realized catalogue mass carries the
    ~0.24 dex forward scatter of §1.3, so BOTH resolved mass kernels are
    only licensed together with a width host-z numerator. The guard is
    enforced by resolving the host-z kernel under ``catalogue_scattered``
    (one-directional; unscattered catalogues keep every baseline pairing).

    Args:
        host_mass_kernel: One of ``HOST_MASS_KERNEL_CHOICES``.
        normalization_mode: The in-catalogue normalization mode (see ``p_Di``).
        host_z_kernel: The (unresolved) numerator host-z kernel selection;
            resolved internally via :func:`resolve_host_z_kernel` for the
            prior-consistency guard.
        catalogue_scattered: True iff the loaded catalogue is a scattered
            observed realization (``GalaxyCatalogueHandler.scattered``).

    Returns:
        ``"gaussian"`` (analytic Gaussian mass product + G2d moment-matched
        shift in the calibrated kernels) or ``"trunc_lognormal"`` (the
        ratified truncated lognormal x R_eff kernel, GH numerator with
        small-sigma crossover + GL-in-lnM denominator).

    Raises:
        ValueError: Unknown choice, or the prior-inconsistent combination of
            a point host-z numerator with the trunc_lognormal mass kernel.
    """
    if host_mass_kernel not in HOST_MASS_KERNEL_CHOICES:
        raise ValueError(
            f"unknown host_mass_kernel: {host_mass_kernel!r} "
            f"(expected one of {HOST_MASS_KERNEL_CHOICES})"
        )
    resolved = (
        ("trunc_lognormal" if normalization_mode == "mass_trunc" else "gaussian")
        if host_mass_kernel == "auto"
        else host_mass_kernel
    )
    if catalogue_scattered:
        # Scatter guard (§9, with-BH channel): a point-resolving host-z
        # numerator would point-anchor the mass product on a catalogue whose
        # masses now carry the realized ~0.24 dex scatter — refuse regardless
        # of which width mass kernel is selected. Raises inside
        # resolve_host_z_kernel with the derivation-cited message.
        resolve_host_z_kernel(host_z_kernel, normalization_mode, catalogue_scattered=True)
    if (
        resolved == "trunc_lognormal"
        and resolve_host_z_kernel(host_z_kernel, normalization_mode) == "point"
    ):
        raise ValueError(
            "host_mass_kernel='trunc_lognormal' is prior-inconsistent with a "
            "point (delta-kernel) host-z numerator: the point path evaluates "
            "the analytic Gaussian mass product while the denominator carries "
            "the truncated lognormal x R_eff prior — N_g and D_g would use "
            "different mass priors (counted-once-in-M violation, "
            "docs/derivations/mass_marginal_2d_kernel.md §3.3). Use "
            "host_z_kernel='volume_deconv' (or a non-point-resolving mode) "
            "with the truncated mass kernel."
        )
    return resolved


def validate_scatter_guards(
    normalization_mode: str,
    host_z_kernel: str,
    host_mass_kernel: str,
    catalogue_scattered: bool,
) -> None:
    """Enforce the scattered-catalogue prior-consistency guard set.

    [PHYSICS] Realistic host-observation model (RATIFIED 2026-07-29,
    docs/derivations/realistic_host_observation_model.md §3.4 / §9): when the
    loaded catalogue is a SCATTERED observed realization (sidecar
    sigma_scale > 0), refuse

    1. any point-resolving host-z kernel (§9 guard 1, via
       :func:`resolve_host_z_kernel`),
    2. ``normalization_mode == 'generator_marginal'`` altogether (§9 guard 2:
       its selection leg is derived FOR the point/point unscattered-generator
       premise, DERIVATION_GENERATOR_CONSISTENT_NORM.md §4.3), and
    3. the point-anchored with-BH mass pairing (§9, via
       :func:`resolve_host_mass_kernel`).

    The guard is ONE-DIRECTIONAL (§9 guard 3): on an unscattered catalogue
    (``catalogue_scattered=False``, including sigma_scale = 0 realizations
    and legacy no-sidecar catalogues) this is a no-op and every baseline
    mode stays permitted.

    Args:
        normalization_mode: The in-catalogue normalization mode.
        host_z_kernel: The (unresolved) numerator host-z kernel selection.
        host_mass_kernel: The (unresolved) 2D host-mass kernel selection.
        catalogue_scattered: ``GalaxyCatalogueHandler.scattered`` of the
            loaded catalogue.

    Raises:
        ValueError: Any refused combination under scatter.
    """
    if not catalogue_scattered:
        return
    if normalization_mode == "generator_marginal":
        raise ValueError(
            "normalization_mode='generator_marginal' is refused on a SCATTERED "
            "observed-catalogue realization (sidecar sigma_scale > 0): the "
            "mode's selection leg is derived FOR the point/point unscattered-"
            "generator premise (DERIVATION_GENERATOR_CONSISTENT_NORM.md §4.3), "
            "which the realized noise falsifies by construction "
            "(docs/derivations/realistic_host_observation_model.md §3.3/§3.4, "
            "guard §9.2). Use --normalization_mode absolute_marginal "
            "--host_z_kernel volume_deconv (the ratified real-data pairing, "
            "[RATIFY-R3])."
        )
    resolve_host_z_kernel(host_z_kernel, normalization_mode, catalogue_scattered=True)
    resolve_host_mass_kernel(
        host_mass_kernel, normalization_mode, host_z_kernel, catalogue_scattered=True
    )


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
#
# [DIAGNOSTIC] MTC_HOST_QUAD_N env override (mass_marginal_2d_kernel.md §3.8
# branch (e) discriminator): raises the numerator/host z-quadrature order
# stack-wide (numerator, Z_g norm, B_num completion — every FIXED_QUAD_N
# consumer) for the n=50-vs-200 aliasing check. Unset -> 50, byte-identical
# (kernel-parity goldens). Scalar and batch kernels share this constant, so
# bit-parity between them is preserved under the override.
_HOST_QUAD_N: int = int(os.environ.get("MTC_HOST_QUAD_N", "50"))
_GL_NODES_50, _GL_WEIGHTS_50 = roots_legendre(_HOST_QUAD_N)
_GL_NODES_64, _GL_WEIGHTS_64 = roots_legendre(_BH_DENOM_QUAD_ORDER)

# [DIAGNOSTIC] MTC_ABLATE_MZ_PROJ=1 env override (mass_marginal_2d_kernel.md
# §3.8 branch (b) discriminator): drops the d_L-M_z CRB cross-covariance in
# the 2D numerator's Gaussian conditioning — the conditional
# N(a; mu_cond(z), sigma2_cond) becomes the MARGINAL N(a; mu_4, Sigma_44)
# (consistent pair: zero proj AND marginal variance, not just a zeroed proj).
# Applied at the single precompute site, so scalar/batch consume identically.
# Unset -> production conditioning, byte-identical.
_ABLATE_MZ_PROJ: bool = os.environ.get("MTC_ABLATE_MZ_PROJ", "") == "1"
if _ABLATE_MZ_PROJ or _HOST_QUAD_N != 50:
    _LOGGER.warning(
        "[DIAGNOSTIC OVERRIDES ACTIVE] MTC_ABLATE_MZ_PROJ=%s MTC_HOST_QUAD_N=%d — "
        "NOT a production configuration (mass_marginal_2d_kernel.md §3.8 b/e discriminators)",
        _ABLATE_MZ_PROJ,
        _HOST_QUAD_N,
    )

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
# [PHYSICS] RATIFY-M3 small-sigma crossover (mass_marginal_2d_kernel.md §3.3):
# the GW-peak-centred Gauss-Hermite quadrature is exact only while the mass
# prior is at least as wide as the GH node coverage (sigma_gal >= K*sigma_cond
# in the fraction coordinate, sigma_gal = sigma_lnM * M_g(1+z)/M_det). Below
# that the PRIOR is the spike and falls between the GW-centred nodes (GH-24
# aliases it; returns exactly 0 at the sigma_lnM floor). The kernel therefore
# falls back to the analytic Gaussian product there — where the lognormal/
# Gaussian family difference is O(sigma_lnM) and truncation is negligible for
# interior hosts — restoring the sigma_lnM -> 0 spec-mass limit (C0-continuity
# bar; pinned by test_mass_trunc_kernel crossover tests).
# IMPLEMENTATION CORRECTION (found by the kernel-parity goldens, recorded in
# the derivation §3.3): the width condition ALONE misfires for mass-mismatched
# hosts (a_gal << 1 makes the LINEARIZED width sigma_gal tiny even when the
# prior is broad, sigma_lnM ~ 0.7, and its fat lognormal tail at the GW peak
# is exactly what GH integrates correctly — the Gaussian fallback would
# replace that tail with exp(-thousands), e.g. golden near_lowmass_bound_mt_4d
# 0.061 -> 7e-15). Genuine aliasing requires a Gaussian-like spike, i.e.
# sigma_lnM itself small: an in-span spike with moderate sigma_lnM is
# impossible (a_gal ~ mu_cond forces sigma_gal ~ sigma_lnM > K*sigma_cond).
# The crossover therefore ALSO requires sigma_lnM <= the family-validity cap.
_MASS_TRUNC_GH_CROSSOVER_K: float = 5.0
_MASS_TRUNC_GH_CROSSOVER_SIGMA_LNM_MAX: float = 0.1
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


def _host_pixels(
    completeness: CompletenessModel,
    host_phiS: npt.NDArray[np.float64],
    host_qS: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    """HEALPix pixel index ``k(g)`` of every host, from its own sky position.

    ``host_qS`` is the ecliptic *colatitude* — the same convention
    :meth:`CompletenessModel.ang2pix` takes and the same one the completion
    numerator uses for the event pixel.
    """
    return np.array(
        [completeness.ang2pix(float(phi), float(theta)) for phi, theta in zip(host_phiS, host_qS)],
        dtype=np.int64,
    )


def _completeness_at_host_nodes(
    completeness: CompletenessModel,
    z_nodes: npt.NDArray[np.float64],
    host_pixels: npt.NDArray[np.int64],
    h: float,
) -> npt.NDArray[np.float64]:
    """Per-host catalogue completeness ``f_{k(g)}(z)`` at ``(n, k)`` nodes.

    Vectorised over the redshift nodes; the only Python-level loop is over the
    *distinct* HEALPix pixels of the batch (``f_k`` is a per-pixel accessor),
    never over hosts or nodes.

    Args:
        completeness: Per-pixel completeness model (``f_k`` accessor).
        z_nodes: Redshift nodes, shape ``(n, k)``.
        host_pixels: HEALPix pixel index per host, shape ``(n,)``.
        h: Dimensionless Hubble parameter.

    Returns:
        ``f_{k(g)}(z)`` clipped to ``[0, 1]``, shape ``(n, k)``.
    """
    out = np.empty(z_nodes.shape, dtype=np.float64)
    for pixel in np.unique(host_pixels):
        rows = host_pixels == pixel
        values = np.asarray(
            completeness.f_k(z_nodes[rows].reshape(-1), int(pixel), h), dtype=np.float64
        )
        out[rows] = values.reshape(-1, z_nodes.shape[1])
    return np.clip(out, 0.0, 1.0)


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


def _eddington_shifted_host_mass_batch(
    host_M: npt.NDArray[np.float64],
    host_M_error: npt.NDArray[np.float64],
    n_grid: int = 401,
    chunk_size: int = 100_000,
) -> npt.NDArray[np.float64]:
    """Vectorised twin of :func:`eddington_shifted_host_mass` over an array.

    Row ``i`` uses the SAME moment-matching quadrature (same grid construction
    -- ``[M_g - 5 sigma, M_g + 5 sigma]`` clamped to ``sigma <= 2 M_g`` and
    ``M >= 1e3``, ``n_grid`` nodes -- and the same trapezoid reduction) as the
    scalar function, only batched over a leading host axis (never a per-host
    Python loop). Used where the scalar function's own per-host list
    comprehension (the small-``n`` per-event-candidate twin at
    ``single_host_likelihood_batch``) would be prohibitively slow -- e.g. over
    the full ``reduced_galaxy_catalog`` (tens of millions of rows, instrument
    J's ``--sigma4d_mass_kernel=kernel``,
    results/prod2d_closure_20260818/PREREGISTRATION_TILT_BATTERY.md §1).
    Chunked (``chunk_size`` rows) to bound the ``(chunk, n_grid)`` intermediates.

    Args:
        host_M: Catalogue (source-frame) host BH mass estimates [M_sun].
        host_M_error: 1-sigma mass uncertainties [M_sun].
        n_grid: Quadrature nodes per host (matches the scalar default, 401).
        chunk_size: Hosts per chunk.

    Returns:
        Shifted effective masses [M_sun], same shape as ``host_M``; a row
        equals ``host_M`` when its uncertainty is zero/invalid (bare-Gaussian
        limit, same guard as the scalar function).
    """
    host_M = np.asarray(host_M, dtype=np.float64)
    host_M_error = np.asarray(host_M_error, dtype=np.float64)
    out = np.array(host_M, dtype=np.float64, copy=True)
    valid = (host_M > 0.0) & (host_M_error > 0.0) & np.isfinite(host_M_error)
    valid_idx = np.flatnonzero(valid)
    for start in range(0, valid_idx.size, chunk_size):
        idx = valid_idx[start : start + chunk_size]
        M_v = host_M[idx]
        sigma = np.minimum(host_M_error[idx], 2.0 * M_v)
        lo = np.maximum(M_v - 5.0 * sigma, 1e3)
        hi = M_v + 5.0 * sigma
        t = np.linspace(0.0, 1.0, n_grid)
        M_grid = lo[:, None] + (hi - lo)[:, None] * t[None, :]  # (chunk, n_grid)
        w = np.exp(-0.5 * ((M_grid - M_v[:, None]) / sigma[:, None]) ** 2) * np.asarray(
            R_eff_per_mbh(M_grid.reshape(-1)), dtype=np.float64
        ).reshape(M_grid.shape)
        Z = np.trapezoid(w, M_grid, axis=1)
        num = np.trapezoid(M_grid * w, M_grid, axis=1)
        finite = np.isfinite(Z) & (Z > 0.0)
        shifted = np.where(finite, num / np.where(finite, Z, 1.0), M_v)
        out[idx] = shifted
    return out


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

    Small-sigma crossover (RATIFY-M3, mass_marginal_2d_kernel.md §3.3):
    elementwise, where ``sigma_gal = sigma_lnM * host_M (1+z) / det_M <=
    _MASS_TRUNC_GH_CROSSOVER_K * sigma_cond`` the GW-centred GH nodes cannot
    resolve the (now-narrow) prior, and the analytic Gaussian product
    ``N(mu_cond; mu_gal, sigma_cond^2 + sigma_gal^2)`` is used instead —
    recovering the spec-mass limit continuously (the family difference is
    O(sigma_lnM) there; truncation negligible for interior hosts).
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
    mz_gh: npt.NDArray[np.float64] = (p_a @ _MT_GH_WEIGHTS) / np.sqrt(np.pi)  # (..., K)
    # [PHYSICS] RATIFY-M3 crossover: analytic Gaussian product where the prior
    # is narrower than the GH node coverage (sigma_gal <= K * sigma_cond).
    # Eq. (14.31) in derivations/dark_siren_likelihood.md (Gaussian product);
    # docs/derivations/mass_marginal_2d_kernel.md §3.3 / §3.7 cases 1 & 8.
    mu_gal = (
        np.asarray(host_M, dtype=np.float64).reshape(np.shape(host_M) + (1,)) * one_plus_z / det_M
    )  # (..., K) = a_gal, the prior centre in the fraction coordinate
    sigma_gal = (
        np.asarray(sigma_lnM, dtype=np.float64).reshape(np.shape(sigma_lnM) + (1,)) * mu_gal
    )  # (..., K) linearized prior width in the fraction coordinate
    # Both conditions required: unresolvable by the GW-centred nodes AND
    # Gaussian-like (family cap) — see the IMPLEMENTATION CORRECTION note at
    # the constants. Broad mass-mismatched hosts (small a_gal, large
    # sigma_lnM) stay on GH, preserving the fat-tail-at-the-GW-peak physics.
    narrow = (sigma_gal <= _MASS_TRUNC_GH_CROSSOVER_K * sigma_cond) & (
        np.asarray(sigma_lnM, dtype=np.float64).reshape(np.shape(sigma_lnM) + (1,))
        <= _MASS_TRUNC_GH_CROSSOVER_SIGMA_LNM_MAX
    )  # (..., K)
    sigma2_sum = sigma_cond**2 + sigma_gal**2
    mz_gauss = np.exp(-0.5 * (mu_cond - mu_gal) ** 2 / sigma2_sum) / np.sqrt(
        2.0 * np.pi * sigma2_sum
    )
    mz: npt.NDArray[np.float64] = np.where(narrow, mz_gauss, mz_gh)
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
    # [PHYSICS] FIX-3 §7.1 pointwise switch (fix3_zmz_catalog_selection.md
    # §3.5 table row 3): each GL node holds its z, so the joint conditional
    # S(d_L(z) | z, M(1+z)) is queried per node when the flag is on; the
    # kwarg is absent (byte-identical) when it is off.
    p = np.asarray(
        detection_probability.detection_probability_with_bh_mass_interpolated(
            np.repeat(d_L, n_g),
            m_z.reshape(-1),
            np.full(n_z * n_g, host_phiS),
            np.full(n_z * n_g, host_qS),
            h=h,
            **_wbh_z_kwargs(detection_probability, np.repeat(z_arr, n_g)),
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
    # [PHYSICS] FIX-3 §7.1 pointwise switch — identical convention to the
    # scalar twin (bit-parity): z per query node when the flag is on.
    p = np.asarray(
        detection_probability.detection_probability_with_bh_mass_interpolated(
            np.repeat(d_L, n_g),
            m_z.reshape(-1),
            np.repeat(host_phiS, n_z * n_g),
            np.repeat(host_qS, n_z * n_g),
            h=h,
            **_wbh_z_kwargs(detection_probability, np.repeat(z.reshape(-1), n_g)),
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
    :meth:`~darksiren_emri.galaxy_catalogue.handler.GalaxyCatalogueHandler.draw_rate_weighted_hosts`.

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
    (:meth:`~darksiren_emri.galaxy_catalogue.handler.GalaxyCatalogueHandler.draw_rate_weighted_hosts`),
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
            (:func:`darksiren_emri.emri_rate.R_eff_per_mbh`).
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


def _wbh_z_kwargs(
    detection_probability_obj: Any,
    z: float | npt.NDArray[np.float64],
) -> dict[str, Any]:
    r"""FIX-3 §7.1 pass-through: conditioning redshift for with-BH p_det queries.

    [PHYSICS] joint z x M_z-resolved with-BH detection survival
    (docs/derivations/fix3_zmz_catalog_selection.md [RATIFY-Z5]): when the
    detection-probability object is built with ``pdet_wbh_z_resolved=True``,
    EVERY with-BH (2D) survival query must be conditioned on the redshift the
    caller is already holding — ``S(d_L(z;h) | z, M_z)`` replaces the
    pooled-in-z ``S(d_L(z;h) | M_z)`` ATOMICALLY across all with-BH selection
    legs (Sigma_glob_wbh incl. the smeared branch, per-host erf-sum and
    mass_trunc inner-M integrals).  When the flag is off (or a mock p_det
    without the attribute is used), the call is byte-identical to the
    pre-FIX-3 form (no ``z`` keyword passed).

    References:
        Mandel, Farr & Gair (2019), arXiv:1809.02063 — selection at the
            population AT HYPOTHESIS, which specifies (z, M_z) jointly.
        docs/derivations/fix3_zmz_catalog_selection.md §3.3 ("pass the z you
            are already holding") and §3.5 (atomic-switch rule).
    """
    # `is True` (not truthiness): MagicMock test doubles auto-create truthy
    # attributes; only the real boolean property may activate the pass-through.
    if getattr(detection_probability_obj, "wbh_z_resolved", False) is True:
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
    :func:`darksiren_emri.emri_rate.p_pop_unnormalized`).

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
    (:func:`darksiren_emri.dark_siren_injection.compute_global_catalog_fraction`
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
    :meth:`~darksiren_emri.galaxy_catalogue.handler.GalaxyCatalogueHandler.draw_rate_weighted_hosts`.
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
            to :data:`~darksiren_emri.constants.HOST_DRAW_Z_MAX`.

    Returns:
        ``W_cat`` in ``yr^-1`` (the ``emri_rate.C_NORM`` scale cancels in every
        ratio it enters).

    References:
        - results/lcat_h_dependence_20260725/DERIVATION_GENERATOR_CONSISTENT_NORM.md
          §2.3 Eq. (4) (spec). W_cat anchor of record (pin R5,
          FIXB_PATHA_PACKAGE.md §5, 2026-08-04): **W_cat = 1.2493e9 over
          20,834,171 pruned rows** of the campaign-#51 reduced catalogue. The
          earlier "6.3477e8 over 9,060,017 pruned rows" anchor belonged to the
          pre-#51 catalogue snapshot and is superseded (package erratum).
        - darksiren_emri/galaxy_catalogue/handler.py, draw_rate_weighted_hosts
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
    (``F = V_f / V_tot``, :func:`darksiren_emri.dark_siren_injection.compute_global_catalog_fraction`),
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
            :data:`~darksiren_emri.constants.HOST_DRAW_Z_MAX`, NOT the
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
        - darksiren_emri/dark_siren_injection.py,
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


# [HIER] θ-hook inventory counters (PA-HIER-16 corroborant): each in-scope
# site increments its counter when its θ branch actually runs, so a driver can
# assert the θ-aware path was entered. NOTE: sites 2.1/2.2 execute inside
# multiprocessing workers under production dispatch — increments land in the
# worker process, not the parent; the decisive engagement evidence is the
# per-term ln L diagnostics (PA-HIER-23), never this counter alone.
_THETA_HOOK_COUNTERS: dict[str, int] = {
    "site_2_1": 0,
    "site_2_2": 0,
    "site_2_3": 0,
    # [HIER] site 2.3phi (PHYSICS_CHANGE_THETA_DIVISOR_20260830.md section 2.2,
    # row #255 tree 2 node T1.1): the theta-consistent no-BH phi divisor
    # ratio rho(theta). Incremented once per (h) pass when engaged.
    "site_2_3_phi": 0,
}


def _theta_hook_count(site: str) -> None:
    _THETA_HOOK_COUNTERS[site] += 1


def _validate_theta(theta_b: float, theta_s: float) -> None:
    """Guard pattern, not a silent no-op: θ = (b, s) needs s > 0 and finite b."""
    if not (theta_s > 0.0) or not np.isfinite(theta_s) or not np.isfinite(theta_b):
        raise ValueError(f"theta requires finite b and s > 0, got (b, s) = ({theta_b}, {theta_s})")


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
    # [HIER] θ-hook site 2.3 (PHYSICS_CHANGE_THETA_HOOK_20260828.md, row #216).
    # (0.0, 1.0) is the literal-skip identity (GATE T-ID).
    theta_b: float = 0.0,
    theta_s: float = 1.0,
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
    if theta_b != 0.0 or theta_s != 1.0:
        # Sec. 2 in Ma, Hu & Huterer (2006), arXiv:astro-ph/0506614 — affine
        # photo-z systematic (b, s). HIER §1.2 s-placement (row #221 item 4;
        # 2026-08-29 note supersedes the 2026-08-28 "s on the folded width"
        # pin): s scales the RAW z_err_g BEFORE the PV fold; b is unchanged
        # (still AFTER the fold, using sigma_z_pv from the UNSHIFTED z_g);
        # the 1e-10 floor re-applies after the combine so the delta limit
        # stays exact.
        _validate_theta(theta_b, theta_s)
        _theta_hook_count("site_2_3")
        sigma_eff = np.maximum(np.sqrt((theta_s * z_err_g) ** 2 + sigma_z_pv**2), 1e-10)
        z_g = z_g + theta_b * (1.0 + z_g)
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
            # [PHYSICS] FIX-3 §7.1 one-z rule (fix3_zmz_catalog_selection.md
            # §3.3): under sigma_z smearing the smear z, the joint-conditioning
            # z, and the (1+z) mass lift ride the SAME z per query node —
            # counted once (project_pdet_hypothesis_convention).
            p_nodes = np.asarray(
                detection_probability_obj.detection_probability_with_bh_mass_interpolated(
                    d_L_nodes,
                    M_z_nodes,
                    zeros,
                    zeros,
                    h=h,
                    **_wbh_z_kwargs(detection_probability_obj, z_nodes.ravel()),
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


# ===========================================================================
# Path (A): ONE detection model — the phi-marginal survival S_bar_phi
# ===========================================================================
# [PHYSICS] FIXB_PATHA_PACKAGE.md §3 (2026-08-04), joint C9+C8 mass-consistent
# mixture. The shipped estimator carried the separately fitted mass-blind
# survival S_3D in the completion/partition legs while the catalogue leg
# carried the mass-aware S_4D at catalogue masses; nothing enforced the tower
# identity S_3D = INTEGRAL phi S_4D dM and it failed by 8.8-11.4% (gate (ii-b)
# r_phi(0.73) = 0.885984 on the production object, measured 2026-08-23;
# supersedes the stale "0.9119" quote of unresolved pool provenance --
# PROPOSAL_SIGMA_PHI_DIVISOR_20260822.md [A11], row #176), with 89-133% of the
# h-slope of r(h) being that mismatch rather than Malmquist physics. Path (A)
# replaces S_3D by the phi-marginal of the SAME S_4D in the completion and
# catalogue-numerator slots, so the tower identity holds by construction there
# (r_phi == 1). The no-BH catalogue GLOBAL DIVISOR is the fourth slot: with
# catalogue_global_selection="phi" now the production default under
# absolute_marginal (rows #171-#178), it too carries Sigma^phi, so r_phi == 1
# now holds for that slot as well. The surviving ratio
# r_Malm = Sigma^4D/Sigma^phi is a pure Malmquist ratio.
#
# References:
#     Mandel, Farr & Gair (2019), arXiv:1809.02063, Eqs. (5)-(7) — selection
#         alpha must use the SAME population and detection model as every
#         numerator (assumption A2), applied to the hybrid population density
#         of GATE_PACKAGE_FINAL.md Appendix A.
#     Turski et al. (2023), arXiv:2302.12037, Eq. (8) — completion numerator
#         and denominator carry the population mass density.
#     Gray et al. (2020), arXiv:1908.06050, Eq. (A.19) — catalogue/completion
#         partition structure.
#     Babak et al. (2017), arXiv:1703.09722, Eqs. (5), (23), (31)x(34) — phi
#         and R_eff.

# Quadrature conventions of the measured anchors (FIXB_PATHA_PACKAGE.md §5;
# instruments fixb_measurements/iid_pathA.py, fixb_x15_attribution/pathA_recomb.py):
# 600 log10-M nodes on [1e4, 1e7] (the generator's own `_DEFAULT_M_GRID_POINTS`
# convention) and 1500 z nodes on (1e-6, z_max(h)], trapezoid in both.
_PHI_LOG10_M_GRID_POINTS = 600
_S_PHI_Z_GRID_POINTS = 1500
_S_PHI_Z_CHUNK = 250
# Gauss-Hermite order of the g_i mass-kernel contraction (N8).
_G_I_HERMITE_NODES = 64

# [PHYSICS] Route 1 (2026-08-12), RATIFIED by author 2026-08-12:
# adaptive Gauss-Hermite order for the g_i contraction. Fast order n=8 is exact
# to degree 15; for the piecewise power-law integrand (max |exponent| 1.43) the
# GH truncation bound stays << 1e-12 whenever the relative half-width
# sqrt(2)*sigma_cond*t/mu <= _G_I_ADAPT_MAX_RELWIDTH and the +-t-sigma window
# crosses no breakpoint. Study: results/venue_transfer_20260811/perf/route1_study/
# (41.0M harvested production z-nodes: zero straddling, max rel err 1.3e-15 at n=8).
_G_I_HERMITE_NODES_FAST = 8
_G_I_ADAPT_T = 6.0
_G_I_ADAPT_MAX_RELWIDTH = 0.02

# [PHYSICS] G1 guard (ledger row #118, 2026-08-17): the fused g_sel keeps the
# Route-1 adaptive order, but the polynomial-exactness bound does not cover the
# S_4D factor. A fast-path row is escalated to the pinned order whenever the
# RELATIVE variation of S_4D across the +-_G_I_ADAPT_T-sigma Hermite window
# exceeds this tolerance: |S(hi) - S(lo)| > tol * max(S(hi), S(lo)). Within the
# tolerance the S-induced departure from the Route-1 error class is bounded by
# tol itself (S piecewise-linear in the interpolated grid; a within-cell window
# contributes a polynomial factor the fast order already integrates exactly).
_G_SEL_S_VAR_TOL = 1e-6

# [PHYSICS] D1 remedy (ii), monitoring half (author decision 2026-08-04):
# the campaign-#51 detections were selected by SNR >= 20 AND p0 in
# [10.002, 15.998] (the stale snapshot-era `ParameterSpace.p0` bound guard in
# the 5-point-stencil derivative), a MASS band-pass that no inference selection
# object models. Its class-conditional retention ratio under the joint
# selection S_and = P(SNR >= 20 AND p0 in W | d_L, M_z) was measured on the
# production pool at s_G/s_D = 0.7305 +- 0.4%
# (fixb_x15_attribution/CAND_B_CRB_FILTER.md, `cand_b_joint_selection.py`).
# Only this RATIO enters the class-share rescaling below, so the monitored
# gate-(ii) consistency number can be scored under S_and without rebuilding
# the selection objects (which would re-pin R10-R12; retiring the stale bounds
# is deferred to the next campaign — see TODO.md).
P0_WINDOW_CLASS_RETENTION_RATIO = 0.7305


@functools.lru_cache(maxsize=4)
def _phi_dark_mass_log10_grid(
    n_grid: int = _PHI_LOG10_M_GRID_POINTS,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    r"""The normalised dark-host mass density ``phi`` on its log10-M grid.

    Returns ``(log10_M, M, phi, Z_phi)`` where ``phi`` is a normalised density
    in ``log10 M`` on ``[M_SOURCE_FRAME_MIN, M_SOURCE_FRAME_MAX]``
    (``INTEGRAL phi dlog10 M = 1`` by the same trapezoid rule the generator's
    inverse-CDF sampler uses) and ``Z_phi`` is the normalisation that was
    divided out. The unnormalised density comes from
    :func:`~darksiren_emri.dark_siren_injection.dark_mass_log10_density_unnormalised`
    — the density the injected dark hosts are drawn from (never re-typed here).

    The arrays are returned read-only: they are cached module state.
    """
    log10_M = np.linspace(
        math.log10(M_SOURCE_FRAME_MIN), math.log10(M_SOURCE_FRAME_MAX), n_grid, dtype=np.float64
    )
    M_grid = 10.0**log10_M
    # Eqs. (5), (23), (31)x(34) in Babak et al. (2017), arXiv:1703.09722.
    phi = dark_mass_log10_density_unnormalised(M_grid)
    Z_phi = float(np.trapezoid(phi, log10_M))
    if not (Z_phi > 0.0):
        raise ValueError(f"dark-host mass density normalisation Z_phi = {Z_phi} is non-positive")
    phi = phi / Z_phi
    for arr in (log10_M, M_grid, phi):
        arr.setflags(write=False)
    return log10_M, M_grid, phi, Z_phi


@functools.lru_cache(maxsize=1)
def _phi_ln_dark_mass_affine_coeffs() -> tuple[float, float, float, float, float]:
    r"""Segment-affine coefficients of ``ln phi_unnorm`` vs ``log10 M``.

    [PHYSICS] perf/realistic-venue interpolation swap (2026-08-12), RATIFIED
    by author 2026-08-12. On the Babak band the unnormalised density
    ``phi(log10 M) = dn/dlog10 M x R_eff`` is an exact piecewise power law
    (Eqs. 5, 23, 26-27, 30, 31x34: exponents -0.3, -0.19, +0.06 and the
    kappa surrogate's +0.5 below ``M_turn``), because the Gamma min-cap never
    binds (max ratio 0.1253 < 1 on [1e4, 1e7]). Hence ``ln phi`` is exactly
    affine in ``log10 M`` on each side of the single kink at ``log10 M = 5``
    (``kappa_cap`` default ``M_turn = 1e5``): evaluating the two affine
    branches IS linear interpolation off the minimal kink-aligned 3-node
    grid, and reproduces the chain analytically — the only residual is the
    log/lerp/exp floating-point roundtrip, O(few ULP), adversarially measured
    max rel dev 1.8e-15 (14 ULP, 2M-sample sweep). Regression tripwire:
    ``test_phi_interpolation_regression.py::test_affinity_premise_still_holds``.

    The coefficients are DERIVED numerically from calls to the exact chain
    (:func:`~darksiren_emri.dark_siren_injection.dark_mass_log10_density_unnormalised`)
    at the three nodes — the density is never re-typed here
    (FIXB_PATHA_PACKAGE.md §3.2 contract).

    Returns:
        ``(kink, a_lo, b_lo, a_hi, b_hi)`` with
        ``ln phi = a + b * log10 M`` per segment.

    Tests that monkeypatch ``dark_mass_log10_density_unnormalised`` must
    ``cache_clear()`` this function as well as :func:`_phi_dark_mass_log10_grid`
    (the flat-phi Jacobian test in ``test_closed_loop_gfrac.py`` is the precedent).
    """
    lo = math.log10(M_SOURCE_FRAME_MIN)
    hi = math.log10(M_SOURCE_FRAME_MAX)
    # Eq. (30) surrogate turn-over in emri_rate.kappa_cap: M_turn = 1e5.
    kink = 5.0
    x = np.array([lo, kink, hi], dtype=np.float64)
    y = np.log(dark_mass_log10_density_unnormalised(10.0**x))
    b_lo = float((y[1] - y[0]) / (kink - lo))
    a_lo = float(y[0] - b_lo * lo)
    b_hi = float((y[2] - y[1]) / (hi - kink))
    a_hi = float(y[1] - b_hi * kink)
    return kink, a_lo, b_lo, a_hi, b_hi


def dark_mass_density_per_mass(
    M: npt.NDArray[np.float64],
    *,
    exact: bool = False,
) -> npt.NDArray[np.float64]:
    r"""``phi(M)``: the dark-host mass density per unit ``M`` (zero off-band).

    .. math::

        \phi(M) = \frac{1}{Z_\phi}\,
            \frac{\mathrm{d}n/\mathrm{d}\log_{10}M \; R_\mathrm{eff}(M)}
                 {M \ln 10},
        \qquad \int_{10^4}^{10^7}\phi(M)\,\mathrm{d}M = 1 ,

    the same ``phi`` as :func:`_phi_dark_mass_log10_grid` transformed from a
    density in ``log10 M`` to a density in ``M`` (``dlog10 M = dM/(M ln 10)``),
    with the IDENTICAL normalisation constant. Support is the Babak band; off
    the band the density is exactly zero (a dark host outside the band does not
    exist in the population, so ``g_i`` must not be extrapolated there).

    [PHYSICS] perf/realistic-venue (2026-08-12), RATIFIED by author 2026-08-12:
    the default path evaluates the chain as the two-segment affine form of
    ``ln phi`` in ``log10 M`` (:func:`_phi_ln_dark_mass_affine_coeffs` — the
    minimal kink-aligned lerp) instead of re-running the ``emri_rate.py``
    power-law chain per call (76.2% of measured seed wall time,
    ``results/venue_transfer_20260811/perf/PERF_ROADMAP.md`` §1-§2).
    The substitution is analytically exact (piecewise power law — see the
    table's docstring); residual is O(few ULP). ``exact=True`` restores the
    verbatim pre-swap evaluation for equivalence tests and counterfactuals.

    Args:
        M: Source-frame MBH masses in solar masses (any shape).
        exact: Evaluate the full ``emri_rate.py`` chain per call (pre-swap
            behaviour) instead of the interpolation table.

    Returns:
        ``phi(M)`` in ``M_sun^-1``, same shape as ``M``.
    """
    _, _, _, Z_phi = _phi_dark_mass_log10_grid()
    M_arr = np.asarray(M, dtype=np.float64)
    inside = (M_arr >= M_SOURCE_FRAME_MIN) & (M_arr <= M_SOURCE_FRAME_MAX)
    safe = np.where(inside, M_arr, M_SOURCE_FRAME_MIN)
    if exact:
        # Eqs. (5), (23), (31)x(34) in Babak et al. (2017), arXiv:1703.09722
        unnorm = dark_mass_log10_density_unnormalised(safe)
    else:
        # Same Eqs. via the analytically exact kink-aligned segment lerp.
        kink, a_lo, b_lo, a_hi, b_hi = _phi_ln_dark_mass_affine_coeffs()
        x = np.log10(safe)
        unnorm = np.exp(np.where(x < kink, a_lo + b_lo * x, a_hi + b_hi * x))
    density = unnorm / (safe * math.log(10.0)) / Z_phi
    return np.asarray(np.where(inside, density, 0.0), dtype=np.float64)


def precompute_phi_marginal_survival(
    h_values: list[float],
    detection_probability_obj: SimulationDetectionProbability,
    *,
    z_max_cap: float | None = None,
    n_z: int = _S_PHI_Z_GRID_POINTS,
    n_log10_M: int = _PHI_LOG10_M_GRID_POINTS,
) -> dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    r"""Tabulate the phi-marginal survival ``S_bar_phi(z;h)`` (path A, slot 0).

    [PHYSICS] FIXB_PATHA_PACKAGE.md §3.2 (2026-08-04):

    .. math::

        \bar S_\phi(z;h) \equiv \int \phi(\log_{10}M)\,
            S_\mathrm{4D}\bigl(d_L(z;h),\, M(1+z)\bigr)\,\mathrm{d}\log_{10}M

    ONE contraction over the **production** with-BH survival object — the
    pooled-2D 40-bin ``S(d_L | M_z)`` grid (``pdet_wbh_z_resolved = False`` in
    every run of record, cluster-verified 41/41), NOT the FIX-3 joint
    ``z x M_z`` grid. It is the SAME object
    :func:`precompute_global_catalog_selection` evaluates for ``Sigma^4D``, so
    the tower identity ``S_bar_phi = INTEGRAL phi S_4D dM`` holds by
    construction and ``r_phi == 1``. When the FIX-3 flag IS on, the
    conditioning redshift rides along per query node via
    :func:`_wbh_z_kwargs` (atomic-switch rule).

    Sky: the with-BH object is isotropic by standing decision (the 4D
    sky x M_z survival is statistics-starved; residual sky systematic bounded
    at ``Sigma^3D(sky)/Sigma^3D(iso) = 1.000202``, gate (ii-e)), so
    ``phi = theta = 0`` here exactly as in ``Sigma^4D``.

    Args:
        h_values: Hubble parameter values to tabulate.
        detection_probability_obj: The detection-probability object whose
            with-BH accessor defines ``S_4D``.
        z_max_cap: Analysis-depth cap (issue #30), applied as in ``D(h)``.
        n_z: Redshift nodes (trapezoid; anchor convention 1500).
        n_log10_M: ``log10 M`` nodes (trapezoid; anchor convention 600).

    Returns:
        ``h -> (z_grid, S_bar_phi(z_grid))``; ``S_bar_phi`` is dimensionless
        and in ``[0, 1]``.

    References:
        Mandel, Farr & Gair (2019), arXiv:1809.02063, Eqs. (5)-(7).
        Babak et al. (2017), arXiv:1703.09722 — ``phi``.
    """
    log10_M, M_grid, phi, _ = _phi_dark_mass_log10_grid(n_log10_M)
    table: dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
    for h in h_values:
        dl_max = detection_probability_obj.get_dl_max(h)
        z_max = dist_to_redshift(dl_max, h=h)
        if z_max_cap is not None:
            z_max = min(z_max, z_max_cap)
        z_grid = np.linspace(1e-6, z_max, n_z, dtype=np.float64)
        d_L_grid = np.asarray(dist_vectorized(z_grid, h=h), dtype=np.float64)  # Gpc
        s_phi = np.empty(n_z, dtype=np.float64)
        for start in range(0, n_z, _S_PHI_Z_CHUNK):
            sl = slice(start, min(start + _S_PHI_Z_CHUNK, n_z))
            z_chunk = z_grid[sl]
            # (n_chunk, n_M) query block: the SAME (d_L, M_z) pair Sigma^4D uses,
            # M_z = M (1 + z) with the one z per node (FIX-3 §7.1 one-z rule).
            d_L_block = np.repeat(d_L_grid[sl][:, None], M_grid.size, axis=1)
            M_z_block = M_grid[None, :] * (1.0 + z_chunk[:, None])
            z_block = np.repeat(z_chunk[:, None], M_grid.size, axis=1)
            zeros = np.zeros(d_L_block.size, dtype=np.float64)
            s_4d = np.asarray(
                detection_probability_obj.detection_probability_with_bh_mass_interpolated(
                    d_L_block.ravel(),
                    M_z_block.ravel(),
                    zeros,
                    zeros,
                    h=h,
                    **_wbh_z_kwargs(detection_probability_obj, z_block.ravel()),
                ),
                dtype=np.float64,
            ).reshape(d_L_block.shape)
            # S_bar_phi(z) = INTEGRAL phi(log10 M) S_4D dlog10 M (§3.2).
            s_phi[sl] = np.trapezoid(s_4d * phi[None, :], log10_M, axis=1)
        table[h] = (z_grid, s_phi)
        _LOGGER.info(
            "S_bar_phi(h=%.4f): z_max=%.4f, S_bar_phi(z_min)=%.7g, S_bar_phi(z_max)=%.7g, "
            "max=%.7g [%d z x %d log10M nodes]",
            h,
            z_max,
            float(s_phi[0]),
            float(s_phi[-1]),
            float(np.max(s_phi)),
            n_z,
            M_grid.size,
        )
    return table


def precompute_phi_selection_integrals(
    h_values: list[float],
    phi_survival_table: dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    completeness: CompletenessModel,
) -> tuple[dict[float, float], dict[float, float]]:
    r"""The phi-convention partition legs ``beta_G^phi`` and ``beta_Gbar^phi``.

    [PHYSICS] FIXB_PATHA_PACKAGE.md §3.2 (2026-08-04), slots 1 and 3:

    .. math::

        \beta_G^\phi(h)  &= \int \bar f(z;h)\,\bar S_\phi(z;h)\,p_\mathrm{pop}(z;h)\,\mathrm{d}z \\
        \beta_{\bar G}^\phi(h) &= \int \bigl(1-\bar f(z;h)\bigr)\,\bar S_\phi(z;h)\,
            p_\mathrm{pop}(z;h)\,\mathrm{d}z \\
        D^\phi(h) &= \beta_G^\phi + \beta_{\bar G}^\phi

    the exact analogues of the legacy ``beta_G``/``beta_Gbar``/``D`` with the
    fitted mass-blind ``S_3D`` replaced by ``S_bar_phi``. ``p_pop`` is the
    generator's own
    :func:`~darksiren_emri.dark_siren_injection._redshift_population_weight`
    (``dVc/dz/(1+z)``) and ``f_bar`` the same sky-marginalised completeness the
    legacy ``beta_Gbar`` uses; the legs are ISOTROPIC because the with-BH
    survival object they contract is (gate (ii-e) bounds the residual at
    2e-4). These are NEW tables: the legacy ones stay untouched so the
    ``generator_marginal`` assembly remains byte-identical (gate (iii-a)).

    Args:
        h_values: Hubble parameter values.
        phi_survival_table: Output of :func:`precompute_phi_marginal_survival`.
        completeness: Catalogue completeness ``f_bar(z, h)`` (Gray Eq. 9).

    Returns:
        ``(beta_G_phi_table, beta_Gbar_phi_table)``, both in the units of
        ``p_pop dz`` — identical to the legacy ``D``/``beta_Gbar``.

    References:
        Gray et al. (2020), arXiv:1908.06050, Eqs. (29), (33), (A.19).
        Mandel, Farr & Gair (2019), arXiv:1809.02063, Eqs. (5)-(7).
    """
    beta_G_phi: dict[float, float] = {}
    beta_Gbar_phi: dict[float, float] = {}
    for h in h_values:
        z_grid, s_phi = phi_survival_table[h]
        p_pop = np.asarray(_redshift_population_weight(z_grid, h), dtype=np.float64)
        f_bar = np.clip(np.asarray(completeness.f_bar(z_grid, h), dtype=np.float64), 0.0, 1.0)
        # Eq. (29) / Eq. (33) in Gray et al. (2020) with S_3D -> S_bar_phi.
        beta_G_phi[h] = float(np.trapezoid(f_bar * s_phi * p_pop, z_grid))
        beta_Gbar_phi[h] = float(np.trapezoid((1.0 - f_bar) * s_phi * p_pop, z_grid))
        _LOGGER.info(
            "phi-convention legs(h=%.4f): beta_G_phi=%.7g, beta_Gbar_phi=%.7g, D_phi=%.7g",
            h,
            beta_G_phi[h],
            beta_Gbar_phi[h],
            beta_G_phi[h] + beta_Gbar_phi[h],
        )
    return beta_G_phi, beta_Gbar_phi


def completion_mass_factor_g(
    z_nodes: npt.NDArray[np.float64],
    d_L_fraction: npt.NDArray[np.float64],
    det_M_z: float,
    proj_d_L_to_M: float,
    sigma_cond_M: float,
    *,
    n_hermite: int = _G_I_HERMITE_NODES,
    adaptive: bool = True,
) -> npt.NDArray[np.float64]:
    r"""The 2D completion leg's mass density ``g_i(z;h)`` at quadrature nodes.

    [PHYSICS] (N8), GATE_PACKAGE_FINAL.md §2.2 / FIXB_PATHA_PACKAGE.md §3.2:

    .. math::

        g_i(z;h) = \int \mathrm{d}x_M\,
            \mathcal{N}\bigl(x_M;\mu_\mathrm{cond}(z),\sigma_\mathrm{cond}\bigr)\,
            \phi_x(x_M;z), \qquad
        \phi_x(x_M;z) = \phi\Bigl(x_M \frac{M_{z,\mathrm{det},i}}{1+z}\Bigr)
            \frac{M_{z,\mathrm{det},i}}{1+z}

    with ``x_M = M_z/M_z,det,i`` the dimensionless mass coordinate the 2D
    catalogue leg's ``mz_integral`` is a density in, ``mu_cond(z) = 1 +
    proj (d_L_frac(z) - 1)`` and ``sigma_cond`` the Gaussian conditional of the
    ``(d_L_frac, M_z_frac)`` 2x2 block of ``cov_4d`` (Bishop 2006 PRML Eqs.
    2.81-2.82). ``g_i`` is a density in ``x_M`` — the SAME measure as
    ``mz_integral`` — so the 2D catalogue and completion legs become addable
    and the 2D measure invariance is preserved exactly (gate (i)).

    The factor sits **inside** the completion quadrature (the ``z``-dependence
    of ``mu_cond`` and of the ``1/(1+z)`` mass lift is not separable); the 1D
    completion numerator stays unmultiplied — the 1D observable set is
    ``cov_obs = cov_4d[:3, :3]``, its M-integral collapses, and inserting
    ``g`` there would be an MFG double count (gate (iv)).

    Args:
        z_nodes: Quadrature redshift nodes, shape ``(k,)``.
        d_L_fraction: ``d_L(z;h)/d_L,det,i`` at the same nodes, shape ``(k,)``.
        det_M_z: The event's own measured detector-frame BH mass ``M_z,det,i``.
        proj_d_L_to_M: ``cov_4d[3,2]/cov_4d[2,2]`` (the 2x2 block projection).
        sigma_cond_M: ``sqrt(cov_4d[3,3] - cov_4d[3,2]^2/cov_4d[2,2])``.
        n_hermite: Gauss-Hermite order (default 64, the measured convention).
        adaptive: Route 1 (2026-08-12, RATIFIED by author 2026-08-12). When
            ``True`` and ``n_hermite == _G_I_HERMITE_NODES`` (the default,
            unmodified order), each row is contracted at the fast order
            ``_G_I_HERMITE_NODES_FAST`` (n=8) unless it triggers one of two
            fallback conditions, in which case it is contracted at the
            pinned ``n_hermite`` (n=64) instead:

            1. **Relative half-width criterion** — the +-``_G_I_ADAPT_T``
               sigma Gauss-Hermite window is not narrow relative to
               ``mu_cond``: ``w = sqrt(2) sigma_cond_M _G_I_ADAPT_T >
               _G_I_ADAPT_MAX_RELWIDTH * mu_cond`` (or ``mu_cond <= 0``,
               treated as fallback for safety).
            2. **Breakpoint-straddle criterion** — the mass window
               ``[(mu_cond - w) * scale, (mu_cond + w) * scale]`` contains
               any of the ``phi`` breakpoints ``M_SOURCE_FRAME_MIN``,
               ``1.0e5`` (``emri_rate.kappa_cap`` ``M_turn``, Eq. 30
               surrogate), ``M_SOURCE_FRAME_MAX``.

            Off the fallback set, ``phi`` restricted to the +-6 sigma window
            is a single power-law branch (see
            :func:`_phi_ln_dark_mass_affine_coeffs`), so Gauss-Hermite of
            order n integrates a polynomial-times-Gaussian exactly to degree
            ``2n - 1``; the truncation error of the Taylor remainder beyond
            that degree (Abramowitz & Stegun 25.4.46) is bounded by the
            relative half-width to the ``2n`` power, which for n=8 and
            ``relwidth <= _G_I_ADAPT_MAX_RELWIDTH = 0.02`` stays << 1e-12 for
            the piecewise power-law exponents in play (max exponent
            magnitude 1.43;
            see ``results/venue_transfer_20260811/perf/ROUTE1_GATE_PACKAGE.md``
            and the harvested-node study cited there). Passing ``adaptive=False``
            restores the pinned n=64 single-group convention byte-for-byte,
            regardless of the mask; an all-fallback ``adaptive=True`` call is
            likewise byte-for-byte identical to ``adaptive=False`` because both
            take the identical single-group n=64 contraction path.

    Returns:
        ``g_i`` at the nodes, shape ``(k,)``, in units of ``1/x_M``.

    References:
        Mandel, Farr & Gair (2019), arXiv:1809.02063, Eqs. (5)-(7).
        Turski et al. (2023), arXiv:2302.12037, Eq. (8).
        Gray et al. (2020), arXiv:1908.06050, Eq. (A.19).
        Babak et al. (2017), arXiv:1703.09722 — ``phi``.
        Abramowitz & Stegun (1964), Eq. 25.4.46 — Gauss-Hermite truncation
        error term (Route 1 adaptive-order bound).
    """

    def _contract_group(
        order: int,
        mu_cond_group: npt.NDArray[np.float64],
        scale_group: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        x_nodes, x_weights = roots_hermite(order)
        # Gauss-Hermite for E_{x~N(mu,sigma)}[phi_x]: nodes mu + sqrt(2) sigma t_j.
        x_M = mu_cond_group[:, None] + math.sqrt(2.0) * sigma_cond_M * x_nodes[None, :]  # (k, n_h)
        M_source = x_M * scale_group[:, None]
        phi_x = dark_mass_density_per_mass(M_source) * scale_group[:, None]
        return np.asarray((phi_x @ x_weights) / math.sqrt(math.pi), dtype=np.float64)

    # dM/dx_M at each z: the mass scale the dimensionless coordinate rides on.
    scale = det_M_z / (1.0 + np.asarray(z_nodes, dtype=np.float64))  # (k,)
    mu_cond = 1.0 + proj_d_L_to_M * (np.asarray(d_L_fraction, dtype=np.float64) - 1.0)  # (k,)

    if not adaptive or n_hermite != _G_I_HERMITE_NODES:
        return _contract_group(n_hermite, mu_cond, scale)

    w = math.sqrt(2.0) * sigma_cond_M * _G_I_ADAPT_T  # scalar half-width
    lo_bound = (mu_cond - w) * scale  # (k,) — scale > 0 always
    hi_bound = (mu_cond + w) * scale  # (k,)
    breakpoints = (
        M_SOURCE_FRAME_MIN,
        1.0e5,  # emri_rate.kappa_cap M_turn (Eq. 30 surrogate)
        M_SOURCE_FRAME_MAX,
    )
    straddles = np.zeros_like(mu_cond, dtype=np.bool_)
    for b in breakpoints:
        straddles |= (lo_bound < b) & (b < hi_bound)
    fallback = (w > _G_I_ADAPT_MAX_RELWIDTH * mu_cond) | (mu_cond <= 0.0) | straddles

    if bool(np.all(fallback)):
        return _contract_group(_G_I_HERMITE_NODES, mu_cond, scale)
    if bool(np.all(~fallback)):
        return _contract_group(_G_I_HERMITE_NODES_FAST, mu_cond, scale)

    out = np.empty_like(mu_cond)
    out[fallback] = _contract_group(_G_I_HERMITE_NODES, mu_cond[fallback], scale[fallback])
    out[~fallback] = _contract_group(_G_I_HERMITE_NODES_FAST, mu_cond[~fallback], scale[~fallback])
    return out


def completion_mass_factor_g_sel(
    z_nodes: npt.NDArray[np.float64],
    d_L_gpc: npt.NDArray[np.float64],
    d_L_fraction: npt.NDArray[np.float64],
    det_M_z: float,
    proj_d_L_to_M: float,
    sigma_cond_M: float,
    *,
    s_query: Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
        npt.NDArray[np.float64],
    ],
    n_hermite: int = _G_I_HERMITE_NODES,
    adaptive: bool = True,
) -> npt.NDArray[np.float64]:
    r"""The FUSED 2D completion mass density ``g_sel,prod(z;h)`` at quadrature nodes.

    [PHYSICS] selection fusion [P1] (ledger rows #117-#118, 2026-08-17;
    ``docs/derivations/GATE_PRESENTATION_SELECTION_FUSION_20260817.md`` §1):

    .. math::

        g_{\mathrm{sel,prod}}(z;h) = \int \mathrm{d}x_M\,
            \mathcal{N}\bigl(x_M;\mu_\mathrm{cond}(z),\sigma_\mathrm{cond}\bigr)\,
            \phi_x(x_M;z)\,
            S_\mathrm{4D}\bigl(d_L(z;h),\, x_M\,M_{z,\mathrm{det},i}\bigr)

    — :func:`completion_mass_factor_g` with the detection survival integrated
    against the observed-mass likelihood in the SAME single ``dx_M``
    (Mandel, Farr & Gair 2019, arXiv:1809.02063, Eqs. (5)-(7): the selected
    population prior of a latent-thresholded detection model; L6-DER2 §2-§3,
    L6-DER3 §3). ``S_4D`` is dimensionless, so ``g_sel,prod`` remains a
    density in ``x_M`` — the same measure as ``mz_integral`` — and the 2D
    catalogue/completion addability (gate (i)) is preserved by construction.

    Quadrature (G1 ruling, row #118): Route-1 adaptive is KEPT, with ``S_4D``
    evaluated per Hermite node, plus a guard that escalates a fast-path row to
    the pinned order when the relative S-variation across the node window
    exceeds ``_G_SEL_S_VAR_TOL`` (the Route-1 polynomial bound does not cover
    ``S``; within the tolerance the S-induced error is bounded by the
    tolerance itself). ``s_query(d_L_gpc, M_z, z) -> S_4D`` must query the
    SAME with-BH survival object :func:`precompute_phi_marginal_survival`
    queries (detector-frame mass, absolute d_L in Gpc, isotropic sky,
    ``_wbh_z_kwargs`` rider) — the caller owns that closure so this function
    stays detection-model-agnostic and CPU-testable.

    Limiting cases: ``S ≡ 1`` recovers :func:`completion_mass_factor_g`
    bit-exactly on BOTH the ``adaptive=False`` and the adaptive path (the
    guard adds no escalations when the S-variation is zero, so the group
    partition is identical); ``sigma_cond -> 0`` gives
    ``g_i * S(mu_cond M_z,det)`` — per row #118/MAJOR-1 this is effectively
    the production operating point (measured d_L-conditional sigma_cond
    p50 = 8.8e-8).

    Args:
        z_nodes: Quadrature redshift nodes, shape ``(k,)``.
        d_L_gpc: Absolute ``d_L(z;h)`` at the nodes in Gpc, shape ``(k,)``.
        d_L_fraction: ``d_L(z;h)/d_L,det,i`` at the same nodes, shape ``(k,)``.
        det_M_z: The event's measured detector-frame BH mass ``M_z,det,i``.
        proj_d_L_to_M: ``cov_4d[3,2]/cov_4d[2,2]`` (2x2 block projection).
        sigma_cond_M: ``sqrt(cov_4d[3,3] - cov_4d[3,2]^2/cov_4d[2,2])``.
        s_query: Flat survival accessor ``(d_L_gpc, M_z, z) -> S_4D`` on
            equal-shape 1-D arrays; values in ``[0, 1]``.
        n_hermite: Pinned Gauss-Hermite order (default 64).
        adaptive: Route-1 adaptive order (G1: kept, with the S-variation
            guard). ``False`` restores the pinned single-group convention.

    Returns:
        ``g_sel,prod`` at the nodes, shape ``(k,)``, in units of ``1/x_M``.

    References:
        Mandel, Farr & Gair (2019), arXiv:1809.02063, Eqs. (5)-(7).
        Gray et al. (2020), arXiv:1908.06050, Eq. (A.19).
        Babak et al. (2017), arXiv:1703.09722 — ``phi``.
    """

    def _contract_group_sel(
        order: int,
        mu_cond_group: npt.NDArray[np.float64],
        scale_group: npt.NDArray[np.float64],
        d_L_group: npt.NDArray[np.float64],
        z_group: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        x_nodes, x_weights = roots_hermite(order)
        # Gauss-Hermite for E_{x~N(mu,sigma)}[phi_x S]: nodes mu + sqrt(2) sigma t_j.
        x_M = mu_cond_group[:, None] + math.sqrt(2.0) * sigma_cond_M * x_nodes[None, :]  # (k, n_h)
        M_source = x_M * scale_group[:, None]
        phi_x = dark_mass_density_per_mass(M_source) * scale_group[:, None]
        M_z = x_M * det_M_z  # detector-frame query mass, exactly S_bar_phi's pair
        # MINOR-6 guard (row #118): a non-positive Hermite node mass must not
        # reach the survival interpolator (log10 under wbh_z_resolved); phi is
        # already zero there, and S is forced to 0 so NaN cannot poison the row.
        pos = M_z > 0.0
        s = np.zeros_like(M_z)
        if bool(np.any(pos)):
            s[pos] = np.asarray(
                s_query(
                    np.repeat(d_L_group[:, None], order, axis=1)[pos],
                    M_z[pos],
                    np.repeat(z_group[:, None], order, axis=1)[pos],
                ),
                dtype=np.float64,
            )
        return np.asarray(((phi_x * s) @ x_weights) / math.sqrt(math.pi), dtype=np.float64)

    z_arr = np.asarray(z_nodes, dtype=np.float64)
    d_L_arr = np.asarray(d_L_gpc, dtype=np.float64)
    # dM/dx_M at each z: the mass scale the dimensionless coordinate rides on.
    scale = det_M_z / (1.0 + z_arr)  # (k,)
    mu_cond = 1.0 + proj_d_L_to_M * (np.asarray(d_L_fraction, dtype=np.float64) - 1.0)  # (k,)

    if not adaptive or n_hermite != _G_I_HERMITE_NODES:
        return _contract_group_sel(n_hermite, mu_cond, scale, d_L_arr, z_arr)

    # Identical fallback criteria to completion_mass_factor_g (Route 1) ...
    w = math.sqrt(2.0) * sigma_cond_M * _G_I_ADAPT_T  # scalar half-width
    lo_bound = (mu_cond - w) * scale  # (k,) — scale > 0 always
    hi_bound = (mu_cond + w) * scale  # (k,)
    breakpoints = (
        M_SOURCE_FRAME_MIN,
        1.0e5,  # emri_rate.kappa_cap M_turn (Eq. 30 surrogate)
        M_SOURCE_FRAME_MAX,
    )
    straddles = np.zeros_like(mu_cond, dtype=np.bool_)
    for b in breakpoints:
        straddles |= (lo_bound < b) & (b < hi_bound)
    fallback = (w > _G_I_ADAPT_MAX_RELWIDTH * mu_cond) | (mu_cond <= 0.0) | straddles

    # ... plus the G1 S-variation guard on the would-be-fast rows.
    fast = ~fallback
    if bool(np.any(fast)):
        m_lo = (mu_cond[fast] - w) * det_M_z
        m_hi = (mu_cond[fast] + w) * det_M_z
        s_lo = np.where(
            m_lo > 0.0,
            np.asarray(
                s_query(d_L_arr[fast], np.where(m_lo > 0.0, m_lo, 1.0), z_arr[fast]),
                dtype=np.float64,
            ),
            0.0,
        )
        s_hi = np.where(
            m_hi > 0.0,
            np.asarray(
                s_query(d_L_arr[fast], np.where(m_hi > 0.0, m_hi, 1.0), z_arr[fast]),
                dtype=np.float64,
            ),
            0.0,
        )
        s_var_exceeds = np.abs(s_hi - s_lo) > _G_SEL_S_VAR_TOL * np.maximum(s_hi, s_lo)
        if bool(np.any(s_var_exceeds)):
            fast_idx = np.flatnonzero(fast)
            fallback[fast_idx[s_var_exceeds]] = True

    if bool(np.all(fallback)):
        return _contract_group_sel(_G_I_HERMITE_NODES, mu_cond, scale, d_L_arr, z_arr)
    if bool(np.all(~fallback)):
        return _contract_group_sel(_G_I_HERMITE_NODES_FAST, mu_cond, scale, d_L_arr, z_arr)

    out = np.empty_like(mu_cond)
    out[fallback] = _contract_group_sel(
        _G_I_HERMITE_NODES, mu_cond[fallback], scale[fallback], d_L_arr[fallback], z_arr[fallback]
    )
    out[~fallback] = _contract_group_sel(
        _G_I_HERMITE_NODES_FAST,
        mu_cond[~fallback],
        scale[~fallback],
        d_L_arr[~fallback],
        z_arr[~fallback],
    )
    return out


def path_a_mixture_objects(
    beta_G_phi: float,
    beta_Gbar_phi: float,
    sigma_phi: float,
    sigma_4d: float,
) -> dict[str, float]:
    r"""Assemble the path-(A) mixture scalars from the four selection legs.

    [PHYSICS] FIXB_PATHA_PACKAGE.md §3.2 (2026-08-04):

    .. math::

        \hat n_w^\phi &= \Sigma^\phi/\beta_G^\phi \\
        r_\mathrm{Malm} &= \Sigma^\mathrm{4D}/\Sigma^\phi \\
        \alpha_G^\phi &= \Sigma^\mathrm{4D}/\hat n_w^\phi
            = \beta_G^\phi\, r_\mathrm{Malm} \\
        \tilde D^\phi &= \alpha_G^\phi + \beta_{\bar G}^\phi , \qquad
        \tilde w_G = \alpha_G^\phi/\tilde D^\phi

    with ``D^phi = beta_G^phi + beta_Gbar^phi`` reported alongside (the 1D
    channel's re-derived full-volume normalisation in the same convention).
    ``n_hat_w^phi`` is mass-blind by construction, so ``alpha_G^phi`` is the
    only place the Malmquist-selected catalogue masses enter — ``r_Malm`` is a
    pure Malmquist ratio and ``r_phi == 1`` identically for the WITH-BH leg
    this function assembles. The no-BH catalogue GLOBAL DIVISOR is a separate,
    fourth slot not consumed here (see ``catalogue_global_selection``); it
    carries its own r_phi, now also == 1 in production under the "phi"
    default (rows #171-#178).

    Args:
        beta_G_phi: ``beta_G^phi(h)``.
        beta_Gbar_phi: ``beta_Gbar^phi(h)``.
        sigma_phi: ``Sigma^phi(h)`` (mass-blind catalogue sum).
        sigma_4d: ``Sigma^4D(h)`` (mass-aware catalogue sum, same catalogue).

    Returns:
        ``{"n_hat_w_phi", "r_Malm", "alpha_G_phi", "D_phi", "D_tilde_phi",
        "w_tilde_G"}``. Degenerate legs (non-positive) yield zeros/NaN rather
        than raising, so a single bad ``h`` cannot abort a grid.

    References:
        Mandel, Farr & Gair (2019), arXiv:1809.02063, Eqs. (5)-(7);
        GATE_PACKAGE_FINAL.md Appendix A (hybrid population density).
    """
    D_phi = beta_G_phi + beta_Gbar_phi
    n_hat_w_phi = sigma_phi / beta_G_phi if beta_G_phi > 0.0 else 0.0
    r_Malm = sigma_4d / sigma_phi if sigma_phi > 0.0 else 0.0
    alpha_G_phi = sigma_4d / n_hat_w_phi if n_hat_w_phi > 0.0 else 0.0
    D_tilde_phi = alpha_G_phi + beta_Gbar_phi
    w_tilde_G = alpha_G_phi / D_tilde_phi if D_tilde_phi > 0.0 else float("nan")
    return {
        "n_hat_w_phi": float(n_hat_w_phi),
        "r_Malm": float(r_Malm),
        "alpha_G_phi": float(alpha_G_phi),
        "D_phi": float(D_phi),
        "D_tilde_phi": float(D_tilde_phi),
        "w_tilde_G": float(w_tilde_G),
    }


def path_a_completion_numerators(
    B_num: float,
    B_num_wbh: float,
    beta_Gbar_phi: float,
    beta_Gbar: float,
    mode: str = "derived",
) -> tuple[float, float, float]:
    r"""Path-(A) completion-leg numerators under the selected convention.

    [PHYSICS] docs/derivations/bscale_completion_normalization.md §6
    (ledger rows #130-#131); MFG (2019) arXiv:1809.02063 Eqs. (5)-(7).

    ``mode="derived"`` (production default): the completion leg is already a
    p_pop-measure integral as constructed (memo §2), so no transfer factor
    exists -- ``B_num_phi = B_num``, ``B_num_wbh_phi = B_num_wbh``.

    ``mode="legacy"``: preserves the un-derived
    ``B_scale = beta_Gbar_phi/beta_Gbar`` multiplier
    (``FIXB_PATHA_PACKAGE.md`` §3.2, 2026-08-04) for byte-identical
    reproduction of historical runs. The multiplication imports the
    difference of two detection models' volume-response slopes onto the
    completion leg -- an MFG-A2 violation (memo §3) -- and is retained ONLY
    as an instrument.

    Args:
        B_num: 1D completion numerator (unscaled).
        B_num_wbh: 2D (with-BH-mass) completion numerator (unscaled).
        beta_Gbar_phi: ``beta_Gbar^phi(h)``.
        beta_Gbar: legacy ``beta_Gbar(h)``.
        mode: ``"derived"`` (default) or ``"legacy"``.

    Returns:
        ``(B_num_phi, B_num_wbh_phi, B_scale_diagnostic)``. ``B_scale`` is
        always reported (1.0 under ``"derived"``) as a diagnostic; it is not
        written to any output column (verified: the diagnostics CSV records
        the unscaled ``B_num``/``B_num_wbh``, not the phi-scaled values).

    References:
        Mandel, Farr & Gair (2019), arXiv:1809.02063, Eqs. (5)-(7).
    """
    B_scale = beta_Gbar_phi / beta_Gbar if beta_Gbar > 0.0 else 0.0
    if mode == "legacy":
        return B_num * B_scale, B_num_wbh * B_scale, B_scale
    return B_num, B_num_wbh, 1.0


def _log_path_a_selection_objects(
    h: float,
    *,
    beta_G_phi: float,
    beta_Gbar_phi: float,
    sigma_phi: float,
    sigma_4d: float,
    w_G_legacy: float,
) -> None:
    """Log the path-(A) mixture scalars at 7 s.f. plus the monitored gate (ii).

    The legacy ``w_G = beta_G/D`` is logged under the RENAMED key
    ``w_G_legacy`` — it is no longer the operative mixture weight but it is not
    overwritten either (FIXB_PATHA_PACKAGE.md §5 instrumentation).
    """
    obj = path_a_mixture_objects(beta_G_phi, beta_Gbar_phi, sigma_phi, sigma_4d)
    _LOGGER.info(
        "path-A(h=%.4f): w_tilde_G=%.7g, alpha_G_phi=%.7g, r_Malm=%.7g, "
        "n_hat_w_phi=%.7g, D_phi=%.7g, D_tilde_phi=%.7g, Sigma_phi=%.7g, "
        "Sigma_4D=%.7g | w_G_legacy=%.7g",
        h,
        obj["w_tilde_G"],
        obj["alpha_G_phi"],
        obj["r_Malm"],
        obj["n_hat_w_phi"],
        obj["D_phi"],
        obj["D_tilde_phi"],
        sigma_phi,
        sigma_4d,
        w_G_legacy,
    )
    # Monitored consistency number (NOT evidence): gate (ii) is demoted
    # (FIXB_PATHA_PACKAGE.md §2). Scored BOTH raw (SNR-only selection objects,
    # i.e. ignoring the p0 window the pipeline actually applied) and under the
    # joint selection S_and, per author decision D1 remedy (ii).
    _LOGGER.info(
        "path-A(h=%.4f) monitored gate(ii): predicted in-cat share = %.7g "
        "(SNR-only) -> %.7g (S_and, rho=%.4f). Monitored consistency number, "
        "not evidence.",
        h,
        obj["w_tilde_G"],
        rescore_class_share_joint_selection(obj["w_tilde_G"]),
        P0_WINDOW_CLASS_RETENTION_RATIO,
    )


def write_selection_table_json(
    h: float,
    *,
    beta_G_phi: float,
    beta_Gbar_phi: float,
    sigma_phi: float,
    sigma_4d: float,
    directory: str = ".",
    # [HIER] site 2.3phi instrumentation (PHYSICS_CHANGE_THETA_DIVISOR_
    # 20260830.md section 2.2, row #255 tree 2 node T1.1). None (default) =
    # theta_phi_divisor="off" or the theta=(0,1) literal skip -- byte-
    # identical JSON (no new keys written), matching every pre-existing call.
    sigma_phi_theta: float | None = None,
    sigma_phi_smear_truth: float | None = None,
    rho_theta: float | None = None,
    kappa_smear_over_point: float | None = None,
    n_degenerate_rows: float | None = None,
    w_share_degenerate: float | None = None,
    theta_b: float | None = None,
    theta_s: float | None = None,
    theta_phi_divisor: str | None = None,
) -> str:
    r"""P4 banked source (PREREGISTRATION_TILT_BATTERY.md §2, N-2(J)).

    Dumps the per-h selection table ``{beta_G_phi, beta_Gbar_phi, sigma_phi,
    sigma_4d, r_Malm}`` to ``<directory>/selection_tables_h_<label>.json`` so
    instrument J's engagement gate (``max_h |r_Malm,kernel/r_Malm,point - 1| >
    1e-4``) is scored from files, not log-line scraping. ``r_Malm =
    sigma_4d/sigma_phi`` (:func:`path_a_mixture_objects`) is the quantity J's
    ``--sigma4d_mass_kernel=kernel`` moves (``sigma_4d``) while ``sigma_phi``
    (mass-blind) stays fixed. Always written whenever the phi-selection
    objects are computed, independent of ``--sigma4d_mass_kernel``/
    ``--eddington_m``.

    Args:
        h: Dimensionless Hubble parameter.
        beta_G_phi: Phi-marginal in-catalogue selection integral.
        beta_Gbar_phi: Phi-marginal completion selection integral.
        sigma_phi: Global mass-blind catalogue selection sum Sigma^phi(h).
        sigma_4d: Global with-BH-mass catalogue selection sum Sigma^4D(h).
        directory: Output directory (default: the current working directory,
            i.e. the run's working directory).
        sigma_phi_theta: [HIER] site 2.3phi (optional): the registered
            theta-consistent divisor ``Sigma_phi_reg(theta;h)`` actually
            consumed at :5215-5219 when engaged. ``None`` omits the key.
        sigma_phi_smear_truth: The site-2.3phi ``(0,1)`` normaliser pass
            ``Sigma_phi_smear((0,1);h)`` (section 2.1).
        rho_theta: The per-node ratio ``rho(theta;h)`` (section 2.3).
        kappa_smear_over_point: ``Sigma_phi_smear((0,1);h)/Sigma_phi_point(h)``
            (section 5.5) -- a REPORTED diagnostic, never a divisor.
        n_degenerate_rows: Count of degenerate transformed windows at this
            node (section 2.4).
        w_share_degenerate: Their total ``w_g``-share of the eligible pool.
        theta_b: The node's theta_b (recorded for T1.2 gate scoring from files).
        theta_s: The node's theta_s.
        theta_phi_divisor: The node's ``theta_phi_divisor`` flag value.

    Returns:
        The path written.
    """
    obj = path_a_mixture_objects(beta_G_phi, beta_Gbar_phi, sigma_phi, sigma_4d)
    h_label = str(np.round(h, 4)).replace(".", "_")
    path = os.path.join(directory, f"selection_tables_h_{h_label}.json")
    payload: dict[str, float | str] = {
        "h": h,
        "beta_G_phi": beta_G_phi,
        "beta_Gbar_phi": beta_Gbar_phi,
        "sigma_phi": sigma_phi,
        "sigma_4d": sigma_4d,
        "r_Malm": obj["r_Malm"],
    }
    # [HIER] site 2.3phi instrumentation: only written when engaged (None ==
    # off / literal-skip), so the payload is byte-identical (same key set)
    # whenever the divisor is off -- R1/R2/regression-item byte-identity.
    if sigma_phi_theta is not None:
        payload["sigma_phi_theta"] = sigma_phi_theta
    if sigma_phi_smear_truth is not None:
        payload["sigma_phi_smear_truth"] = sigma_phi_smear_truth
    if rho_theta is not None:
        payload["rho_theta"] = rho_theta
    if kappa_smear_over_point is not None:
        payload["kappa_smear_over_point"] = kappa_smear_over_point
    if n_degenerate_rows is not None:
        payload["n_degenerate_rows"] = n_degenerate_rows
    if w_share_degenerate is not None:
        payload["w_share_degenerate"] = w_share_degenerate
    if theta_b is not None:
        payload["theta_b"] = theta_b
    if theta_s is not None:
        payload["theta_s"] = theta_s
    if theta_phi_divisor is not None:
        payload["theta_phi_divisor"] = theta_phi_divisor
    with open(path, "w") as sel_file:
        json.dump(payload, sel_file, indent=2)
    return path


def rescore_class_share_joint_selection(
    predicted_in_catalogue_share: float,
    retention_ratio: float = P0_WINDOW_CLASS_RETENTION_RATIO,
) -> float:
    r"""Rescore a predicted in-catalogue share under the joint selection S_and.

    [PHYSICS] D1 remedy (ii), monitoring half (FIXB_PATHA_PACKAGE.md §8, author
    decision 2026-08-04). The pipeline's realized detections passed
    ``SNR >= 20 AND p0 in [10.002, 15.998]``; the inference selection objects
    model only the SNR leg. Because the p0 window is a MASS band-pass, it
    retains the two classes unequally, and the class-share discriminator under
    the joint selection depends only on the RATIO of the class retentions:

    .. math::

        \mathrm{share}_\mathrm{and} =
            \frac{\mathrm{share}\,\rho}{\mathrm{share}\,\rho + (1-\mathrm{share})},
        \qquad \rho \equiv s_G/s_{\bar G} = 0.7305 \pm 0.4\% .

    Verified against the measurement of record: ``share = 0.07280503``
    (generator-closure class share, SNR-only) rescores to ``0.054249`` versus
    the measured ``0.0542477`` (``z = -0.48``).

    This is a MONITORED CONSISTENCY NUMBER, not evidence: gate (ii) is demoted
    (FIXB_PATHA_PACKAGE.md §2), and the number is conditional both on the
    generator-closure convention and on modelling a filter whose existence is
    the package's own defect finding.

    Args:
        predicted_in_catalogue_share: Predicted in-catalogue share under the
            SNR-only selection.
        retention_ratio: ``s_G/s_Gbar`` under the joint selection.

    Returns:
        The predicted share under ``S_and``.

    References:
        fixb_x15_attribution/CAND_B_CRB_FILTER.md (measurement, instrument
            ``cand_b_joint_selection.py``); FIXB_PATHA_PACKAGE.md §2, §8 (D1).
    """
    share = float(predicted_in_catalogue_share)
    if not (0.0 < share < 1.0):
        return share
    scaled = share * float(retention_ratio)
    return scaled / (scaled + (1.0 - share))


def precompute_global_catalog_selection(
    h_values: list[float],
    galaxy_catalog: GalaxyCatalogueHandler,
    detection_probability_obj: SimulationDetectionProbability,
    *,
    with_bh_mass: bool,
    z_max_cap: float | None = None,
    smear_sigma_z: bool = False,
    phi_survival_table: dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]
    | None = None,
    # Instrument J (results/prod2d_closure_20260818/
    # PREREGISTRATION_TILT_BATTERY.md §1, P2 registered kernel): "point"
    # (default) is byte-identical to the pre-flag path; "kernel" applies only
    # inside the with_bh_mass=True branch (never with_bh_mass=False).
    sigma4d_mass_kernel: str = "point",
    # Instrument E (PREREGISTRATION_TILT_BATTERY.md §1): governs whether the
    # "kernel" mode's per-galaxy mean M_eff_g carries the Eddington-in-M shift
    # (the SAME shift production's per-event D_g uses). No effect under
    # sigma4d_mass_kernel="point" (unchanged point evaluation at raw M_g).
    eddington_m: str = "on",
    # [HIER] θ-hook site 2.3 (PHYSICS_CHANGE_THETA_HOOK_20260828.md, row #216):
    # forwarded to _smeared_global_pdet_expectation. The registered site is the
    # smeared kernel's width/window lines ONLY, so a non-identity θ REQUIRES
    # smear_sigma_z=True (guard below) — the point-evaluation branches carry no
    # kernel for θ to reparametrize.
    theta_b: float = 0.0,
    theta_s: float = 1.0,
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
    (:meth:`~darksiren_emri.galaxy_catalogue.handler.GalaxyCatalogueHandler.draw_rate_weighted_hosts`)
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
        phi_survival_table: [PHYSICS] path (A), FIXB_PATHA_PACKAGE.md §3.2
            slot 2. When given (and only with ``with_bh_mass=False``,
            ``smear_sigma_z=False``), ``P_det`` is replaced by the phi-marginal
            survival ``S_bar_phi(z_g;h)`` of
            :func:`precompute_phi_marginal_survival`, producing

            .. math::

                \Sigma^\phi(h) = \sum_g w_g\,\bar S_\phi(z_g;h) ,

            the mass-BLIND companion of ``Sigma^4D`` on the SAME catalogue
            rows, the SAME eligibility mask and the SAME weights ``w_g`` — so
            ``r_Malm(h) = Sigma^4D/Sigma^phi`` is a pure Malmquist ratio and
            ``n_hat_w^phi = Sigma^phi/beta_G^phi`` cannot inherit the Malmquist
            bias. Sharing this function (rather than a parallel one) is what
            guarantees the two sums are computed on the catalogue the run
            actually loads — the convention rule of decision D2, whose
            violation produced the retired mixed-catalogue ``r_Malm = 0.4304``.
        sigma4d_mass_kernel: Instrument J (results/prod2d_closure_20260818/
            PREREGISTRATION_TILT_BATTERY.md §1, P2 registered kernel).
            ``"point"`` (default) evaluates ``P_det`` at the point observer-
            frame mass ``M_z = M_g(1+z_g)``, byte-identical to the pre-flag
            path. ``"kernel"`` (only valid with ``with_bh_mass=True``)
            replaces the point evaluation by ``E_{M~N(M_eff_g,
            sigma_g^2)}[S_4D(d_L_g, M(1+z_g))]`` via the erf-sum inner-M
            machinery of :func:`_bh_mass_denominator_inner_m_integral_batch`,
            with ``sigma_g`` = the catalogue ``BH_MASS_ERROR`` column.
        eddington_m: Instrument E (PREREGISTRATION_TILT_BATTERY.md §1).
            Governs whether ``sigma4d_mass_kernel="kernel"``'s per-galaxy mean
            ``M_eff_g`` carries the Eddington-in-M shift (``"on"``, default)
            or the raw ``M_g`` (``"off"``). No effect under ``"point"``.

    Returns:
        Dict mapping ``h -> sum_global w_g D_g(h)`` (dimensionless rate-weighted
        detection count).

    References:
        Gray et al. (2020), arXiv:1908.06050, Eq. (29) — ``beta_G`` selection
            integral (here its discrete catalogue realisation).
        Babak et al. (2017), arXiv:1703.09722 — per-MBH rate ``R_eff``
            (:func:`darksiren_emri.emri_rate.R_eff_per_mbh`).
    """
    if (theta_b != 0.0 or theta_s != 1.0) and not smear_sigma_z:
        # [HIER] θ-hook site 2.3: guard pattern, not a silent no-op — the
        # registered site is the smeared kernel; the point branches would
        # silently ignore θ (PHYSICS_CHANGE_THETA_HOOK_20260828.md §2).
        raise ValueError(
            "theta (site 2.3) requires smear_sigma_z=True — the registered "
            "site is the smeared host-z kernel; got "
            f"(theta_b, theta_s) = ({theta_b}, {theta_s}) with smear_sigma_z=False"
        )
    if phi_survival_table is not None and (with_bh_mass or smear_sigma_z):
        raise ValueError(
            "phi_survival_table is the mass-blind phi-marginal leg Sigma^phi: it "
            "must be requested with with_bh_mass=False and smear_sigma_z=False "
            "(FIXB_PATHA_PACKAGE.md §3.2 slot 2)."
        )
    if sigma4d_mass_kernel not in ("point", "kernel"):
        raise ValueError(
            f"sigma4d_mass_kernel must be 'point' or 'kernel', got {sigma4d_mass_kernel!r}"
        )
    if sigma4d_mass_kernel == "kernel" and not with_bh_mass:
        raise ValueError(
            "sigma4d_mass_kernel='kernel' requires with_bh_mass=True (the "
            "without-BH-mass channel never consumes the mass kernel)."
        )
    if eddington_m not in ("on", "off"):
        raise ValueError(f"eddington_m must be 'on' or 'off', got {eddington_m!r}")
    catalog = galaxy_catalog.reduced_galaxy_catalog
    z_all = np.asarray(
        catalog[InternalCatalogColumns.REDSHIFT].to_numpy(dtype=np.float64),
        dtype=np.float64,
    )
    M_all = np.asarray(
        catalog[InternalCatalogColumns.BH_MASS].to_numpy(dtype=np.float64),
        dtype=np.float64,
    )
    if sigma4d_mass_kernel == "kernel":
        # Instrument J: sigma_g for the mass-smearing kernel (linear M_sun,
        # the same catalogue column D_g's host_M_error argument reads).
        M_error_all = np.asarray(
            catalog[InternalCatalogColumns.BH_MASS_ERROR].to_numpy(dtype=np.float64),
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
    # The phi-marginal leg contracts the ISOTROPIC with-BH object, so it must not
    # take the sky-aware 3D path (FIXB_PATHA_PACKAGE.md §3.2; gate (ii-e) bounds
    # the residual sky systematic at Sigma^3D(sky)/Sigma^3D(iso) = 1.000202).
    _sky_aware = (
        (not with_bh_mass)
        and phi_survival_table is None
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
        if phi_survival_table is not None:
            # [PHYSICS] path (A) slot 2 (FIXB_PATHA_PACKAGE.md §3.2):
            # Sigma^phi(h) = sum_g w_g S_bar_phi(z_g;h). The survival is a
            # smooth function of z alone once phi is contracted out, so it is
            # read off the S_bar_phi table by linear interpolation at the
            # galaxies' own redshifts (the table is built on 1500 nodes over
            # the same [1e-6, z_max(h)] domain; interpolation error 8e-7 at the
            # anchors). Mandel, Farr & Gair (2019), arXiv:1809.02063 Eq. (6).
            _z_phi_grid, _s_phi_grid = phi_survival_table[h]
            p_det = np.interp(z_g, _z_phi_grid, _s_phi_grid)
        elif smear_sigma_z:
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
                theta_b=theta_b,
                theta_s=theta_s,
            )
        elif with_bh_mass:
            # FLAG (user-approved, statistics-starved): the with-BH-mass 4D
            # sky x M_z survival is too noisy at NSIDE resolution, so this branch
            # stays ISOTROPIC (phi=theta=0, sky-marginalised 2D accessor). The
            # residual sky-selection systematic (<~1%) applies to the with-BH-mass
            # posterior ONLY, not the primary result. PHYSICS-CHANGE-PROTOCOL §9.3.
            phi_iso = np.zeros_like(z_g)
            theta_iso = np.zeros_like(z_g)
            if sigma4d_mass_kernel == "kernel":
                # [PHYSICS] Instrument J registered kernel (results/
                # prod2d_closure_20260818/PREREGISTRATION_TILT_BATTERY.md §1,
                # P2): replaces the point evaluation at M_z_g = M_g(1+z_g) by
                # the expectation over the per-galaxy Eddington-shifted mass
                # prior, via the SAME erf-sum inner-M machinery production's
                # own D_g uses. sigma_g = catalogue BH_MASS_ERROR. M_eff_g
                # carries the SAME Eddington-in-M shift D_g uses, gated by
                # --eddington_m (instrument E); NO R_eff/mass_trunc lognormal
                # inside the kernel (w_g stays the point rate weight computed
                # above; Sigma^phi is untouched -- it contains no per-galaxy
                # mass evaluation).
                sigma_g = M_error_all[eligible]
                M_eff_g = (
                    _eddington_shifted_host_mass_batch(M_g, sigma_g) if eddington_m == "on" else M_g
                )
                p_det = _sigma4d_mass_kernel_expectation(
                    z_g,
                    M_eff_g,
                    sigma_g,
                    phi_iso,
                    theta_iso,
                    h,
                    detection_probability_obj,
                )
            else:
                M_z_g = M_g * (1.0 + z_g)  # observer-frame mass (P_det grid axis)
                # [PHYSICS] FIX-3 §7.1 [RATIFY-Z1/Z5]: Sigma_glob_wbh's averaging
                # measure is the CATALOGUE's joint (z_g, M_z,g) — when the flag is
                # on the galaxy's own z_g conditions the query, S(d_L(z_g;h) |
                # z_g, M_g(1+z_g)); sky stays isotropic (unchanged decision).
                # docs/derivations/fix3_zmz_catalog_selection.md §3.1 (K1)/(K2).
                p_det = np.asarray(
                    detection_probability_obj.detection_probability_with_bh_mass_interpolated(
                        d_L_g,
                        M_z_g,
                        phi_iso,
                        theta_iso,
                        h=h,
                        **_wbh_z_kwargs(detection_probability_obj, z_g),
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
            "Global catalog selection (with_bh=%s%s) sum_w_Dg(h=%.4f) = %.6e  "
            "[%d eligible galaxies, z_max=%.4f]",
            with_bh_mass,
            ", phi_marginal" if phi_survival_table is not None else "",
            h,
            global_table[h],
            z_g.size,
            z_max,
        )
        if with_bh_mass and global_table[h] > 0.0:
            _log_sigma_4d_mass_band_shares(w_g * p_det, M_g, z_g, detection_probability_obj, h)

    return global_table


def _phi_divisor_kernel_pass(
    z_g: npt.NDArray[np.float64],
    z_err_g: npt.NDArray[np.float64],
    w_g: npt.NDArray[np.float64],
    host_pixels: npt.NDArray[np.int64],
    completeness: CompletenessModel,
    h: float,
    phi_z_grid: npt.NDArray[np.float64],
    phi_s_grid: npt.NDArray[np.float64],
    theta_b: float,
    theta_s: float,
    *,
    n_quad: int = 50,
    chunk_size: int = 200_000,
) -> tuple[float, int, float]:
    r"""One theta-node pass of the site-2.3phi kernel sum.

    [HIER] PHYSICS_CHANGE_THETA_DIVISOR_20260830.md section 2.1 (row #255,
    tree 2 node T1.1). Computes ``sum_{g eligible} w_g * S_tilde_g(theta;h)``
    with the SAME per-host kernel form as the site-2.2 numerator
    (``single_host_likelihood_batch``, :7117-7208; the C7-core kernel
    ``N(z; z_g^theta, sigma_g^theta) * f_k(g)(z;h) * dVc/dz(z;h) / (1+z)``, the
    +/-4 sigma window, the 1e-6 floor, GL-50 quadrature) — NOT the
    theta-INERT point evaluation and NOT the pre-C7 bare smear branch of
    :func:`_smeared_global_pdet_expectation` (section 1g, disclosed and left
    untouched). ``S_bar_phi`` is read off ``phi_z_grid``/``phi_s_grid`` by
    plain (endpoint-clamped) ``np.interp``, the same table-object convention
    :func:`precompute_global_catalog_selection`'s phi branch uses.

    A degenerate transformed window (``hi <= lo``, only possible at ``b < 0``,
    section 2.4) contributes exactly ``0`` (zero physical support => zero
    survival mass) and is counted, never integrated.

    Row-chunked over the ACTIVE (non-degenerate) rows only, for memory
    (mirrors :func:`_smeared_global_pdet_expectation` and the harness's
    ``_KERNEL_SMEAR_CHUNK`` precedent, correspondence_1d.py:1248-1345): each
    row's contribution depends only on its own window/nodes, so the final
    reduction is a single ``np.sum`` over the completed per-row array,
    independent of ``chunk_size`` (regression R8).

    Args:
        z_g: Eligible-row redshifts (UNSHIFTED), shape ``(n,)``.
        z_err_g: Eligible-row RAW catalogue redshift errors (before the s
            scale and the PV fold), shape ``(n,)``.
        w_g: Eligible-row rate weights ``R_eff_per_mbh(M_g)/(1+z_g)``.
        host_pixels: HEALPix pixel index per row (:func:`_host_pixels`).
        completeness: Per-pixel completeness model.
        h: Dimensionless Hubble parameter.
        phi_z_grid: ``S_bar_phi`` table's z-grid for this ``h``.
        phi_s_grid: ``S_bar_phi`` table's survival values for this ``h``.
        theta_b: Affine photo-z bias offset.
        theta_s: Affine photo-z scale.
        n_quad: Gauss-Legendre order (default 50, GL-50).
        chunk_size: Row-chunk size for the active-row loop.

    Returns:
        ``(sigma_phi_smear, n_degenerate_rows, w_degenerate)`` — the kernel
        sum, the degenerate-row count, and their total ``w_g`` weight.

    References:
        Ma, Hu & Huterer (2006), arXiv:astro-ph/0506614, sec. 2.
        Mandel, Farr & Gair (2019), arXiv:1809.02063, Eqs. (5)-(7).
        Gray et al. (2020), arXiv:1908.06050, Eq. (A.10).
    """
    x_nodes, x_weights = roots_legendre(n_quad)
    x_nodes = np.asarray(x_nodes, dtype=np.float64)
    x_weights = np.asarray(x_weights, dtype=np.float64)

    # Section 2.1: sigma_pv,g from the UNSHIFTED z_g, s on the RAW catalogue
    # error BEFORE the PV fold -- the same registered form and operation
    # order as site 2.2 (:7117-7129).
    sigma_z_pv = (1.0 + z_g) * SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S
    z_g_theta = z_g + theta_b * (1.0 + z_g)
    sigma_g_theta = np.sqrt((theta_s * z_err_g) ** 2 + sigma_z_pv**2)
    lo = np.maximum(z_g_theta - 4.0 * sigma_g_theta, 1e-6)
    hi = z_g_theta + 4.0 * sigma_g_theta
    degenerate = hi <= lo
    n_degenerate = int(np.count_nonzero(degenerate))
    w_degenerate = float(np.sum(w_g[degenerate])) if n_degenerate else 0.0

    idx_active = np.nonzero(~degenerate)[0]
    m = idx_active.size
    contrib = np.zeros(m, dtype=np.float64)
    for start in range(0, m, chunk_size):
        sl = slice(start, min(start + chunk_size, m))
        rows = idx_active[sl]
        lo_a = lo[rows]
        hi_a = hi[rows]
        zc_a = z_g_theta[rows]
        se_a = sigma_g_theta[rows]
        pix_a = host_pixels[rows]

        z_nodes = _batched_gl_nodes(lo_a, hi_a, x_nodes)
        gauss = _gaussian_pdf(z_nodes, zc_a[:, None], se_a[:, None])
        w_pop = (
            np.asarray(comoving_volume_element(z_nodes.reshape(-1), h=h), dtype=np.float64)
            / (1.0 + z_nodes.reshape(-1))
        ).reshape(z_nodes.shape)
        f_host = _completeness_at_host_nodes(completeness, z_nodes, pix_a, h)
        zoa = ~np.any(f_host > 0.0, axis=1)
        if bool(np.any(zoa)):
            f_host = f_host.copy()
            f_host[zoa, :] = 1.0
        kern = gauss * f_host * w_pop
        z_row = _batched_gl_reduce(lo_a, hi_a, x_weights, kern)
        s_phi_nodes = np.interp(z_nodes, phi_z_grid, phi_s_grid)
        numer = _batched_gl_reduce(lo_a, hi_a, x_weights, kern * s_phi_nodes)
        z_row_safe = np.where(z_row > 0.0, z_row, 1.0)
        s_tilde = np.where(z_row > 0.0, numer / z_row_safe, 0.0)
        contrib[sl] = w_g[rows] * s_tilde

    return float(np.sum(contrib)), n_degenerate, w_degenerate


def precompute_phi_divisor_theta_ratio(
    h_values: list[float],
    galaxy_catalog: GalaxyCatalogueHandler,
    completeness: CompletenessModel,
    phi_survival_table: dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    *,
    theta_b: float,
    theta_s: float,
    n_quad: int = 50,
    chunk_size: int = 200_000,
) -> dict[float, dict[str, float]]:
    r"""[HIER] site 2.3phi: the theta-consistent no-BH divisor ratio ``rho(theta)``.

    [PHYSICS] PHYSICS_CHANGE_THETA_DIVISOR_20260830.md sections 2.1-2.4 (row
    #255, tree 2 node T1.1). The forensic (B1_1_S0A_DEFECT_FORENSIC_20260829.md,
    mechanism (i)) localised the S0-A b-axis non-null to the no-BH catalogue
    divisor Sigma^phi carrying no theta-dependence in any BUILT form
    (:func:`precompute_global_catalog_selection`'s phi branch, :2906-2915,
    :3022), while the site-2.2 numerator DOES reparametrize under theta. This
    function restores the theta-dependence as a per-``(h, theta)`` SCALAR
    ratio to the stored point table, computed EXACTLY (not to first order):

    .. math::

        \rho(\theta; h) = \frac{\Sigma^\phi_\mathrm{smear}(\theta; h)}
            {\Sigma^\phi_\mathrm{smear}((0,1); h)}, \qquad
        \Sigma^\phi_\mathrm{smear}(\theta; h) = \sum_{g\ \mathrm{eligible}}
            w_g\, \tilde S_g(\theta; h),

    with :math:`\tilde S_g(\theta;h)` the theta-consistent per-galaxy
    survival — the SAME C7-core host-z kernel the site-2.2 numerator
    integrates against (:func:`_phi_divisor_kernel_pass`), evaluated on the
    galaxy's OWN transformed window. Two full pool passes are performed per
    ``h`` (the theta pass and the ``(0,1)`` normaliser pass); the caller forms
    the registered divisor ``Sigma_phi_reg(theta;h) = Sigma_phi_point(h) *
    rho(theta;h)``, so the object returned here is deliberately the RATIO,
    never a replacement table (section 2.3: dividing a banked node's
    ``L_cat,i(theta)`` by this scalar is algebraically identical to
    re-evaluating with the registered divisor — exact, not a linearisation).

    Eligibility (decision D2): identical mask to
    :func:`precompute_global_catalog_selection`'s phi branch and to
    ``Sigma^4D`` — ``z_g < z_max(h)`` (read off ``phi_survival_table``'s own
    capped z-grid domain, so no separate ``z_max_cap`` argument is needed)
    with finite, positive source-frame mass. The weights ``w_g`` are the
    IDENTICAL rate weights.

    Args:
        h_values: Hubble parameter values to evaluate.
        galaxy_catalog: Loaded catalogue handler (same rows the phi-point
            divisor sums over).
        completeness: Per-pixel completeness model (the C7-core ``f_k``
            accessor site 2.2 uses).
        phi_survival_table: Output of :func:`precompute_phi_marginal_survival`
            (``h -> (z_grid, S_bar_phi(z_grid))``); its domain fixes the
            eligibility ``z_max(h)``.
        theta_b: Affine photo-z bias offset (Ma, Hu & Huterer 2006, sec. 2).
        theta_s: Affine photo-z scale.
        n_quad: Gauss-Legendre order (default 50, GL-50, matching site 2.2).
        chunk_size: Row-chunk size (memory only; bit-identical sums for any
            chunk size, regression R8).

    Returns:
        ``h -> {"sigma_phi_smear_theta", "sigma_phi_smear_truth", "rho",
        "n_degenerate_rows", "w_share_degenerate"}``.

    References:
        Ma, Hu & Huterer (2006), arXiv:astro-ph/0506614, sec. 2.
        Mandel, Farr & Gair (2019), arXiv:1809.02063, Eqs. (5)-(7).
        Gray et al. (2020), arXiv:1908.06050, Eq. (29).
        B1_1_S0A_DEFECT_FORENSIC_20260829.md (mechanism (i), E10/E11/E20).
    """
    _validate_theta(theta_b, theta_s)
    catalog = galaxy_catalog.reduced_galaxy_catalog
    if InternalCatalogColumns.REDSHIFT_ERROR not in catalog.columns:
        raise ValueError(
            "precompute_phi_divisor_theta_ratio requires the catalogue "
            "REDSHIFT_MEASUREMENT_ERROR column (the site-2.3phi kernel width)."
        )
    z_all = np.asarray(
        catalog[InternalCatalogColumns.REDSHIFT].to_numpy(dtype=np.float64), dtype=np.float64
    )
    M_all = np.asarray(
        catalog[InternalCatalogColumns.BH_MASS].to_numpy(dtype=np.float64), dtype=np.float64
    )
    z_err_all = np.asarray(
        catalog[InternalCatalogColumns.REDSHIFT_ERROR].to_numpy(dtype=np.float64), dtype=np.float64
    )
    phi_s_all = np.asarray(
        catalog[InternalCatalogColumns.PHI_S].to_numpy(dtype=np.float64), dtype=np.float64
    )
    theta_s_col_all = np.asarray(
        catalog[InternalCatalogColumns.THETA_S].to_numpy(dtype=np.float64), dtype=np.float64
    )

    result: dict[float, dict[str, float]] = {}
    for h in h_values:
        z_grid, s_phi_grid = phi_survival_table[h]
        z_max = float(z_grid[-1])
        # Decision D2: the SAME eligibility mask precompute_global_catalog_
        # selection's phi branch applies (:2890) -- the table's own domain
        # already carries the run's z_max_cap.
        eligible = (z_all < z_max) & np.isfinite(M_all) & (M_all > 0.0)
        z_g = z_all[eligible]
        z_err_g = z_err_all[eligible]
        M_g = M_all[eligible]
        w_g = np.asarray(R_eff_per_mbh(M_g), dtype=np.float64) / (1.0 + z_g)
        host_pixels = _host_pixels(completeness, phi_s_all[eligible], theta_s_col_all[eligible])

        sigma_theta, n_deg_theta, w_deg_theta = _phi_divisor_kernel_pass(
            z_g,
            z_err_g,
            w_g,
            host_pixels,
            completeness,
            h,
            z_grid,
            s_phi_grid,
            theta_b,
            theta_s,
            n_quad=n_quad,
            chunk_size=chunk_size,
        )
        sigma_truth, _n_deg_truth, _w_deg_truth = _phi_divisor_kernel_pass(
            z_g,
            z_err_g,
            w_g,
            host_pixels,
            completeness,
            h,
            z_grid,
            s_phi_grid,
            0.0,
            1.0,
            n_quad=n_quad,
            chunk_size=chunk_size,
        )
        w_total = float(np.sum(w_g))
        rho = sigma_theta / sigma_truth if sigma_truth > 0.0 else 0.0
        w_share_degenerate = (w_deg_theta / w_total) if w_total > 0.0 else 0.0
        result[h] = {
            "sigma_phi_smear_theta": sigma_theta,
            "sigma_phi_smear_truth": sigma_truth,
            "rho": rho,
            "n_degenerate_rows": float(n_deg_theta),
            "w_share_degenerate": w_share_degenerate,
        }
        _LOGGER.info(
            "site 2.3phi(h=%.4f, theta=(%.6g,%.6g)): Sigma_phi_smear(theta)=%.7g, "
            "Sigma_phi_smear(0,1)=%.7g, rho=%.7g, n_degenerate=%d, "
            "w_share_degenerate=%.4e",
            h,
            theta_b,
            theta_s,
            sigma_theta,
            sigma_truth,
            rho,
            n_deg_theta,
            w_share_degenerate,
        )
    return result


def _log_sigma_4d_mass_band_shares(
    contributions: npt.NDArray[np.float64],
    M_g: npt.NDArray[np.float64],
    z_g: npt.NDArray[np.float64],
    detection_probability_obj: SimulationDetectionProbability,
    h: float,
) -> None:
    r"""T9 instrumentation: the mass-band decomposition of ``Sigma^4D(h)``.

    Logs the share of ``Sigma^4D`` carried by catalogue rows in the Babak band
    ``[1e4, 1e7]``, above it, below it, and by rows whose observer-frame mass
    ``M_z = M_g(1+z_g)`` falls outside the with-BH grid's ``M_z`` centres (where
    the accessor clamps to the nearest edge rather than extrapolating).

    This is the standing refutation of F8 ("the clamp dominates ``Sigma^4D``"):
    the measured shares of record at ``h = 0.73`` are
    **98.980 % in-band / 0.999 % above / 0.020 % hard-clamped-high**
    (FIXB_PATHA_PACKAGE.md §5 T9, gate (ii-c); instrument
    ``fixb_measurements/iic_clamp_decomposition.py``). Diagnostic only — no
    computed value depends on it.
    """
    if not hasattr(detection_probability_obj, "_grid_support"):
        return
    try:
        _, _, _, M_centers = detection_probability_obj._grid_support()
    except Exception:  # pragma: no cover - diagnostic must never break a run
        _LOGGER.debug("T9 Sigma^4D band shares unavailable (no grid support).")
        return
    total = float(np.sum(contributions))
    if not (total > 0.0):
        return
    M_z = M_g * (1.0 + z_g)
    M_lo = float(np.asarray(M_centers, dtype=np.float64)[0])
    M_hi = float(np.asarray(M_centers, dtype=np.float64)[-1])
    masks = {
        "in_band": (M_g >= M_SOURCE_FRAME_MIN) & (M_g <= M_SOURCE_FRAME_MAX),
        "above_band": M_g > M_SOURCE_FRAME_MAX,
        "below_band": M_g < M_SOURCE_FRAME_MIN,
        "clamped_hi": M_z > M_hi,
        "clamped_lo": M_z < M_lo,
    }
    shares = {k: 100.0 * float(np.sum(contributions[m])) / total for k, m in masks.items()}
    _LOGGER.info(
        "T9 Sigma^4D(h=%.4f) mass-band shares [%%]: in_band=%.4f above=%.4f below=%.4f "
        "clamped_hi=%.4f clamped_lo=%.4f  (of record at 0.73: 98.980/0.999/~0/0.020)",
        h,
        shares["in_band"],
        shares["above_band"],
        shares["below_band"],
        shares["clamped_hi"],
        shares["clamped_lo"],
    )


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

# (N8) d_L-only 2x2 block conditional scalars for g_i / completion_mass_factor_g
# (prod2d closure counterfactual instrument, "neutralized" mode; results/
# prod2d_closure_20260818/PREREGISTRATION_PROD_COUNTERFACTUAL.md §1 V1').
# Distinct from sigma2_cond_arr/proj_arr above, which condition on the FULL 3D
# observed vector (phi, theta, d_L_frac) for the candidate's own numerator;
# these condition on d_L_frac ONLY (cov_4d[3,2]/cov_4d[2,2] block), mirroring
# self._proj_d_L_to_M / self._sigma_cond_M (:3749-3750) used by the completion
# leg's own g_i.
proj_d_L_to_M_arr: npt.NDArray[np.float64] = np.empty(0)
sigma_cond_M_arr: npt.NDArray[np.float64] = np.empty(0)

# Pre-extracted detection parameters (avoid pickling Detection objects per starmap call)
det_d_L_arr: npt.NDArray[np.float64] = np.empty(0)
det_d_L_unc_arr: npt.NDArray[np.float64] = np.empty(0)
det_M_arr: npt.NDArray[np.float64] = np.empty(0)
det_phi_arr: npt.NDArray[np.float64] = np.empty(0)
det_theta_arr: npt.NDArray[np.float64] = np.empty(0)

# Per-HEALPix-pixel catalogue completeness f_k, threaded into the worker
# processes by child_process_init so the host-z kernel can evaluate f at the
# HOST's pixel (C7-core, GATE_PACKAGE_FINAL.md §1.2). ``None`` reproduces the
# pre-C7 kernel exactly (f == 1 everywhere): every unit test that installs the
# worker globals by hand and does not set this stays byte-identical.
completeness_model: CompletenessModel | None = None


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
    # #40 remainder (RATIFIED 2026-07-27): 2D host-mass kernel decomposition
    # flag (set by evaluate()); "auto" reproduces the mass_trunc bundling.
    _host_mass_kernel: str = "auto"
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
    # Prod2d closure counterfactual instrument (set by evaluate(); results/
    # prod2d_closure_20260818/PREREGISTRATION_PROD_COUNTERFACTUAL.md §1):
    # "production" reproduces the pre-flag path exactly. Class-level default
    # (not just __init__) so instances built via object.__new__ (existing
    # p_Di unit-test harnesses) still resolve a value at self._catalogue_mass_overlap.
    _catalogue_mass_overlap: str = "production"
    _catalogue_mass_error_scale: float = 1.0
    # Completion-leg normalization convention (docs/derivations/
    # bscale_completion_normalization.md §6/§7, ledger rows #130-#131).
    # "derived" (default, [PHYSICS]): B_num_phi = B_num, B_num_wbh_phi =
    # B_num_wbh (no multiplier -- the MFG-derivation-complete form).
    # "legacy": preserves the un-derived beta_Gbar_phi/beta_Gbar multiplier
    # for byte-identical reproduction of historical runs.
    _completion_b_scale: str = "derived"
    # Instrument E (results/prod2d_closure_20260818/
    # PREREGISTRATION_TILT_BATTERY.md §1). "on" (default) is byte-identical to
    # the pre-flag path.
    _eddington_m: str = "on"
    # Instrument J (results/prod2d_closure_20260818/
    # PREREGISTRATION_TILT_BATTERY.md §1). "point" (default) is byte-identical
    # to the pre-flag path.
    _sigma4d_mass_kernel: str = "point"
    # [P3-IMP] catalogue-leg twin cell (PREREGISTRATION_P3_TWIN_20260822.md §2).
    # "off" (default) is byte-identical to the pre-flag path.
    _catalogue_numerator_survival: str = "off"
    # [P3-2D] the with-BH catalogue-leg twin: production adoption (row #223
    # standing grant, charter node B7.3; PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md).
    # "mz_sel" (default) = the production with-BH catalogue numerator: S_4D
    # inside the candidate's own mass quadrature (row #<adoption>); explicit
    # "off" = the pre-adoption COUNTERFACTUAL (no per-candidate survival
    # factor).
    _catalogue_numerator_survival_2d: str = "mz_sel"
    # Centering sub-option ("unset"/"raw"/"eff"): "eff" (default) is the
    # adopted centering (A20_REVIEW_P3_2D_DESIGN_20260825.md F2); REFUSED
    # ("unset") until explicitly set when the twin is engaged -- no silent
    # default.
    _catalogue_numerator_survival_2d_center: str = "eff"
    # B-DEN falsifier instrument (docs/derivations/
    # completion_numerator_data_measure.md §6; AMENDMENT A-5,
    # results/prod2d_closure_20260818/PREREGISTRATION_1D_CORRESPONDENCE.md).
    # "ratio" (default) is byte-identical to the pre-flag path.
    _completion_event_measure: str = "ratio"
    # [P3-RPHI] the fourth Path-A slot, ADOPTED (docs/derivations/
    # PROPOSAL_SIGMA_PHI_DIVISOR_20260822.md §2/§6(ii); rows #172-#178).
    # evaluate()'s "auto" default resolves this to "phi" (Sigma^phi =
    # _global_cat_selection_phi, the SAME table already built by Path A for
    # the weight chain) under absolute_marginal, else "s3d" (the separately
    # fitted Sigma^3D = _global_cat_denom_no_bh). This bare class-level
    # fallback (pre-evaluate() / object.__new__ harnesses) stays the inert
    # "s3d" literal, matching the other un-resolved instrumentation flags.
    _catalogue_global_selection: str = "s3d"
    # Candidate-host mass pre-filter window (ledger rows #198-#202; adopted
    # per docs/derivations/PROPOSAL_MASS_FILTER_SYMMETRIC_20260825.md sec 7(a)).
    # "symmetric" (default, PRODUCTION) scales BH_MASS_ERROR by
    # sigma_multiplier on both sides of the window, matching the GW-side
    # convention. "asymmetric" is the explicit COUNTERFACTUAL pinning the
    # pre-flag path (galaxy error at its bare x1 value -- the retired
    # ±1.5σ-vs-±1σ window). Single read site: the mask branch in
    # get_possible_hosts_from_ball_tree; this attribute is inert plumbing.
    _mass_filter_sigma: str = "symmetric"
    # Mass-window GEOMETRY instrument flag (charter node B5.1,
    # PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md §2; ledger rows
    # #220-#223). "linear" (default, PRODUCTION, byte-identical) is the
    # pre-flag interval-overlap form; "log" re-expresses both sides in
    # ln-space. Not a production posterior at "log". Single read site: the
    # mask branch in get_possible_hosts_from_ball_tree; these two attributes
    # are inert plumbing.
    _mass_filter_geometry: str = "linear"
    _mass_filter_k: float = 1.5
    # [HIER] site 2.3phi theta-consistent no-BH divisor instrument
    # (PHYSICS_CHANGE_THETA_DIVISOR_20260830.md, row #255 tree 2 node T1.1).
    # "off" (default, PRODUCTION, byte-identical): the no-BH catalogue
    # divisor stays Sigma_phi_point (or Sigma^3D under catalogue_global_
    # selection="s3d") -- no code path change. "on" arms the theta-consistent
    # ratio; at theta=(0,1) the literal skip applies (GATE T-ID, no floating
    # operation performed). Single read site: the global_denom_no_bh branch
    # in p_Di.
    _theta_phi_divisor: str = "off"
    # Sky-cone-radius instrument flag (PHYSICS_CHANGE_THETA_DIVISOR_20260830.md
    # §2.5): decouples the sky search radius from the (now-mass-only)
    # mass_filter_k. Default 1.5 matches the pre-flag sigma_multiplier
    # literal -- byte-identical. Single read site: the sigma_multiplier
    # argument of get_possible_hosts_from_ball_tree.
    _sky_cone_k: float = 1.5
    # INSTRUMENTATION (T2.2, row #255 tree 2 node T2.2; A10 = instrumentation
    # guard, not a physics gate). None (default) => OFF, byte-identical: no
    # computed value is read or written differently. A directory path arms a
    # READ-ONLY per-(event, candidate) diagnostic serialiser -- see
    # B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md §6.
    _candidate_dump_dir: str | None = None
    # [HIER] mass-aware 1D catalogue leg instrument (row #255 tree 2 node
    # T2.3, PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md §2). "off" (default,
    # PRODUCTION, byte-identical): sites N1/D1/W1 stay exactly as before --
    # no code path change. "on" (guarded: normalization_mode=
    # "absolute_marginal", catalogue_numerator_survival resolving to "phi",
    # catalogue_global_selection resolving to "phi", theta_phi_divisor="off")
    # replaces the WITHOUT-BH catalogue numerator's per-candidate survival by
    # S_4D(d_L(z;h), M_g(1+z)) (the with-BH object Sigma_4D already
    # evaluates), Sigma_4D as the global divisor (ALREADY IN HAND) and
    # alpha_G_phi as the mixture weight -- the no-mass-likelihood image of
    # the 2D assembly. Not a production posterior; the production-default
    # flip is a fresh [RULE] (row #169's Appendix B pairing).
    _catalogue_leg_1d_mass_aware: str = "off"

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
        # INSTRUMENTATION (default OFF): reference h for the frozen-g_frac
        # counterfactual. None => the production path, byte-identical.
        self._freeze_g_frac_ref_h: float | None = None
        # INSTRUMENTATION (default OFF): N-2 selection-in-numerator
        # counterfactual cell. "off" => the production path, byte-identical.
        self._selection_in_completion_numerator: str = "off"
        # INSTRUMENTATION (default OFF): prod2d closure counterfactual
        # instrument (results/prod2d_closure_20260818/
        # PREREGISTRATION_PROD_COUNTERFACTUAL.md §1). "production" =>
        # the pre-flag path, byte-identical.
        self._catalogue_mass_overlap: str = "production"
        self._catalogue_mass_error_scale: float = 1.0
        # Completion-leg normalization convention (docs/derivations/
        # bscale_completion_normalization.md §6/§7; ledger rows #130-#131).
        # "derived" (default) => byte-different from pre-change runs (the
        # derivation-complete form); "legacy" reproduces the un-derived
        # beta_Gbar_phi/beta_Gbar multiplier for historical-run reproduction.
        self._completion_b_scale: str = "derived"
        # Tilt-ledger battery counterfactual instruments (results/
        # prod2d_closure_20260818/PREREGISTRATION_TILT_BATTERY.md §1).
        # "on"/"point" => the pre-flag production path, byte-identical.
        self._eddington_m: str = "on"
        self._sigma4d_mass_kernel: str = "point"
        # [P3-IMP] twin cell (PREREGISTRATION_P3_TWIN_20260822.md §2): "off"
        # => the pre-flag production path, byte-identical.
        self._catalogue_numerator_survival: str = "off"
        # [P3-2D] the with-BH catalogue-leg twin, ADOPTED (row #223 standing
        # grant, charter node B7.3): "mz_sel"/"eff" => the production
        # with-BH catalogue numerator; explicit "off" = the COUNTERFACTUAL.
        self._catalogue_numerator_survival_2d: str = "mz_sel"
        self._catalogue_numerator_survival_2d_center: str = "eff"
        # B-DEN falsifier instrument (docs/derivations/
        # completion_numerator_data_measure.md §6; AMENDMENT A-5). "ratio"
        # (default) => the pre-flag production path, byte-identical.
        self._completion_event_measure: str = "ratio"
        # [P3-RPHI] the fourth Path-A slot, ADOPTED (docs/derivations/
        # PROPOSAL_SIGMA_PHI_DIVISOR_20260822.md §2/§6(ii); rows #172-#178):
        # "s3d" is this bare pre-evaluate() fallback; evaluate()'s "auto"
        # default resolves to "phi" under absolute_marginal.
        self._catalogue_global_selection: str = "s3d"
        # Candidate-host mass pre-filter window (rows #198-#202 adoption):
        # "symmetric" (default) = production; "asymmetric" pins the pre-flag
        # counterfactual.
        self._mass_filter_sigma: str = "symmetric"
        # Mass-window GEOMETRY instrument flag (charter node B5.1, ledger
        # rows #220-#223): "linear" (default) = production, byte-identical;
        # "log" is the ln-space re-expression, not a production posterior.
        self._mass_filter_geometry: str = "linear"
        self._mass_filter_k: float = 1.5
        # [HIER] site 2.3phi theta-consistent no-BH divisor instrument (row
        # #255 tree 2 node T1.1): "off" (default) = production, byte-identical.
        self._theta_phi_divisor: str = "off"
        # Sky-cone-radius instrument flag (same reference): default 1.5
        # matches the pre-flag sigma_multiplier literal, byte-identical.
        self._sky_cone_k: float = 1.5
        # INSTRUMENTATION (T2.2, row #255 tree 2 node T2.2): None (default)
        # => production path, byte-identical (GATE BI). See evaluate()'s
        # candidate_dump_dir parameter.
        self._candidate_dump_dir = None
        self._candidate_dump_rows: list[dict[str, object]] = []
        self._candidate_dump_event_rows: list[dict[str, object]] = []
        self._candidate_dump_warned: bool = False
        # [HIER] mass-aware 1D catalogue leg instrument (row #255 tree 2 node
        # T2.3): "off" (default) = production, byte-identical.
        self._catalogue_leg_1d_mass_aware: str = "off"

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
        # FIX-3 §7.1 (default OFF, byte-identical): joint z x M_z-resolved
        # with-BH detection survival; requires pdet_z_resolved=True
        # (RATIFY-Z7 guard in SimulationDetectionProbability).
        # docs/derivations/fix3_zmz_catalog_selection.md.
        pdet_wbh_z_resolved: bool = False,
        # Issue #40(a): numerator host-z kernel decomposition flag; "auto"
        # preserves the historical bundling (delta kernel iff
        # generator_marginal) — production default path unchanged.
        host_z_kernel: str = "auto",
        # #40 remainder (RATIFIED 2026-07-27, mass_marginal_2d_kernel.md §4):
        # 2D host-mass kernel decomposition flag; "auto" preserves the
        # historical bundling (trunc_lognormal iff mass_trunc) — production
        # default path unchanged.
        host_mass_kernel: str = "auto",
        # INSTRUMENTATION (default None = OFF, byte-identical): gate (vii)
        # follow-up frozen-g_frac counterfactual. When set to h_ref, every
        # event's 2D completion numerator becomes B_num(h) * g_ref with
        # g_ref = B_num_wbh(h_ref)/B_num(h_ref) — see p_Di. Not a physics
        # change; the 1D channel and both catalogue legs are untouched.
        freeze_g_frac_ref_h: float | None = None,
        # [PHYSICS] selection fusion (ledger rows #117-#118, 2026-08-17;
        # GATE_PRESENTATION_SELECTION_FUSION_20260817.md). "auto" resolves to
        # "fused" under absolute_marginal (the production default: [P1]+[P2]
        # paired — S_bar_phi in the 1D completion numerator AND the fused
        # g_sel,prod in the 2D leg) and to "off" otherwise (generator_marginal
        # and legacy paths stay byte-identical, gate (iii-a)). Explicit
        # "off"/"1d"/"2d" are the item-4 counterfactual decomposition cells
        # (pre-#118 estimator / [P2]-only / [P1]-only).
        selection_in_completion_numerator: str = "auto",
        # Production counterfactual instrument (prod2d closure, results/
        # prod2d_closure_20260818/PREREGISTRATION_PROD_COUNTERFACTUAL.md §1):
        # "production" (default) is byte-identical to the pre-flag path.
        # "neutralized" (V1') replaces the 2D catalogue leg's per-candidate
        # mz_integral with the completion leg's population mass factor
        # (completion_mass_factor_g). "inflated" (V2) scales the numerator
        # width sigma_gal by catalogue_mass_error_scale while freezing the
        # Eddington-shifted mean. Never a production posterior.
        catalogue_mass_overlap: str = "production",
        catalogue_mass_error_scale: float = 1.0,
        # Completion-leg normalization convention (docs/derivations/
        # bscale_completion_normalization.md §6/§7, ledger rows #130-#131):
        # "derived" (default, [PHYSICS]) drops the un-derived
        # beta_Gbar_phi/beta_Gbar multiplier (B_num_phi = B_num). "legacy"
        # preserves it for byte-identical reproduction of historical runs.
        completion_b_scale: str = "derived",
        # Instrument E (results/prod2d_closure_20260818/
        # PREREGISTRATION_TILT_BATTERY.md §1): "on" (default) is byte-identical
        # to the pre-flag path; "off" assigns the raw (unshifted) host_M to
        # _host_M_eff, re-measuring s_Edd at the current operating point.
        eddington_m: str = "on",
        # Instrument J (results/prod2d_closure_20260818/
        # PREREGISTRATION_TILT_BATTERY.md §1, P2 registered kernel): "point"
        # (default) is byte-identical to the pre-flag path; "kernel" replaces
        # the with-BH-mass global catalogue selection's point p_det evaluation
        # by the expectation over the Eddington-shifted per-galaxy mass prior
        # (the erf-sum inner-M machinery, matched to production's own D_g
        # kernel).
        sigma4d_mass_kernel: str = "point",
        # B-DEN falsifier instrument (docs/derivations/
        # completion_numerator_data_measure.md §6; AMENDMENT A-5, results/
        # prod2d_closure_20260818/PREREGISTRATION_1D_CORRESPONDENCE.md):
        # "ratio" (default) is byte-identical to the pre-flag path — the
        # completion numerator's GW event term is a density in the
        # dimensionless distance ratio d_L(z;h)/d_L,det. "data" replaces it
        # with the same Gaussian measurement model expressed as a density in
        # the observable d_L,det, so the numerator normalizes to the same
        # measure as the completion denominator (MFG 2019 arXiv:1809.02063
        # Eqs. (5)-(7)).
        completion_event_measure: str = "ratio",
        # [P3-IMP] catalogue-leg twin, ADOPTED (docs/derivations/
        # PROPOSAL_CATALOGUE_TWIN_PRODUCTION_20260825.md §2/§6; row #195):
        # "auto" (default) resolves exactly like catalogue_global_selection
        # and selection_in_completion_numerator: "phi" under
        # normalization_mode="absolute_marginal" (the S_bar_phi table this
        # cell reads is only built there), else "off" (every other
        # normalization mode stays byte-identical to the pre-adoption path).
        # "off" is now the explicit COUNTERFACTUAL under absolute_marginal
        # (the pre-adoption production path: no per-candidate survival factor
        # in the WITHOUT-BH catalogue numerator). "phi" multiplies the
        # WITHOUT-BH catalogue numerator integrand per host by the
        # phi-marginal survival S_bar_phi(z;h) read from the SAME
        # precompute_phi_marginal_survival table the mixture normalizer
        # beta_G_phi integrates (:2065). "phi_flat" (the K-flat kill arm) is
        # NOT touched by "auto" -- it stays an explicit-only counterfactual
        # cell, same as before. The with-BH catalogue numerator is
        # deliberately untouched (registered invariant); beta_G_phi, D~_phi,
        # and the Sigma-chain are UNTOUCHED (proposal §2, Appendix B as
        # ratified).
        catalogue_numerator_survival: str = "auto",
        # [P3-2D] the with-BH catalogue-leg twin: 2D bounded identity test
        # (row #223 standing grant, charter node B7.3;
        # PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md). "mz_sel" (default) =
        # the production with-BH catalogue numerator: multiplies the
        # WITH-BH catalogue numerator's mass integrand by S_4D(d_L(z;h),
        # x*M_z,det) inside the candidate's own mass quadrature (the
        # product-Gaussian identity; see _mz_sel_2d_expectation). Explicit
        # "off" = the pre-adoption COUNTERFACTUAL. The without-BH twin above
        # is deliberately untouched.
        catalogue_numerator_survival_2d: str = "mz_sel",
        # Centering sub-option ("raw"=host_M, "eff"=host_M_eff) for the
        # product-Gaussian mean fed to the S_4D quadrature: "eff" (default)
        # is the adopted centering (A20_REVIEW_P3_2D_DESIGN_20260825.md F2);
        # REQUIRED explicitly (no silent default) when
        # catalogue_numerator_survival_2d="mz_sel" is combined with
        # "unset".
        catalogue_numerator_survival_2d_center: str = "eff",
        # [P3-RPHI] the fourth Path-A slot, ADOPTED (docs/derivations/
        # PROPOSAL_SIGMA_PHI_DIVISOR_20260822.md §2/§6(ii); rows #172-#178).
        # "auto" (default) resolves exactly like
        # selection_in_completion_numerator: "phi" under
        # normalization_mode="absolute_marginal" (the SAME catalogue-weighted
        # sum Path A already builds for the weight chain, on the same
        # rows/weights/eligibility as Sigma^4D -- makes the estimator's own
        # r_phi==1 docstring invariant true for this slot), else "s3d" (every
        # other normalization mode stays byte-identical -- the S_bar_phi
        # table this divisor reads is only built under absolute_marginal).
        # "s3d" is now the explicit COUNTERFACTUAL (the pre-adoption
        # production path: the no-BH catalogue divisor is the separately
        # fitted Sigma^3D). The with-BH leg is deliberately untouched.
        catalogue_global_selection: str = "auto",
        # Candidate-host mass pre-filter window (rows #198-#202; adopted per
        # PROPOSAL_MASS_FILTER_SYMMETRIC_20260825.md sec 7(a)). "symmetric"
        # (default, PRODUCTION): the GW mass window
        # ((M_z ± M_z_sigma*sigma_multiplier)/(1+z)) is compared against the
        # galaxy's BH_MASS ± BH_MASS_ERROR*sigma_multiplier -- the single-k
        # interval-overlap window. "asymmetric" is the explicit COUNTERFACTUAL
        # pinning the retired pre-flag path (galaxy error at x1). Purely
        # plumbing: validated and read at exactly one site, the
        # mass_filter_mask branch in
        # galaxy_catalogue/handler.py:get_possible_hosts_from_ball_tree.
        mass_filter_sigma: str = "symmetric",
        # Mass-window GEOMETRY instrument flag (charter node B5.1,
        # PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md §2; ledger rows
        # #220-#223). "linear" (default, PRODUCTION, byte-identical): the
        # mass_filter_sigma interval-overlap window above, unchanged in
        # form. "log": the SAME window re-expressed in ln-space on both
        # sides (small-error correspondence on the GW side, the existing
        # R&V15 ln-space budget BH_MASS_ERROR/BH_MASS on the candidate
        # side -- no re-derivation). Orthogonal to mass_filter_sigma (that
        # flag still selects the candidate-side multiplier convention under
        # EITHER geometry). Never a production posterior at "log". Purely
        # plumbing: validated and read at exactly one site, the
        # mass_filter_mask branch in
        # galaxy_catalogue/handler.py:get_possible_hosts_from_ball_tree.
        mass_filter_geometry: str = "linear",
        # Mass-window half-width in units of sigma, decoupled from
        # sigma_multiplier (which after this flag's introduction sets ONLY
        # the sky-cone search radius, never the mass window). Default 1.5
        # matches the current call-site sigma_multiplier=1.5, so the
        # default pairing of both new flags is byte-identical to the
        # pre-flag path (same reference as mass_filter_geometry).
        mass_filter_k: float = 1.5,
        # Sky-cone-radius instrument flag (PHYSICS_CHANGE_THETA_DIVISOR_
        # 20260830.md §2.5, row #255 tree 2 node T1.1): decouples the sky
        # search radius from mass_filter_k (which after B5.1 already governs
        # ONLY the mass window). Default 1.5 matches the pre-flag
        # sigma_multiplier literal at the get_possible_hosts_from_ball_tree
        # call site -- byte-identical mask, candidate list and every
        # downstream value. Purely plumbing: validated (finite, > 0) and read
        # at exactly one site.
        sky_cone_k: float = 1.5,
        # [HIER] θ-hook (C1, PHYSICS_CHANGE_THETA_HOOK_20260828.md, ledger row
        # #216): affine photo-z systematic θ = (b, s) — z̃ = z + b(1+z),
        # σ̃_eff = s·σ_eff at estimator sites 2.1/2.2/2.3 (Ma, Hu & Huterer
        # 2006, arXiv:astro-ph/0506614, Sec. 2). (0.0, 1.0) is the literal-skip
        # identity: the production path is byte-identical by construction (GATE
        # T-ID). NEVER applied generator-side (GATE GEN-FROZEN, PA-HIER-2).
        theta_b: float = 0.0,
        theta_s: float = 1.0,
        # [HIER] C2 (PA-HIER-23) OAT toggle: which in-scope site(s) receive θ;
        # the others are forced to their (0, 1) evaluation. Site 2.3 requires
        # smear_global_selection=True (the registered site is the smeared
        # kernel).
        theta_sites: str = "all",
        # [HIER] site 2.3phi (PHYSICS_CHANGE_THETA_DIVISOR_20260830.md §2.2,
        # row #255 tree 2 node T1.1): the theta-consistent no-BH divisor
        # ratio ρ(θ). "off" (default) is byte-identical -- no code path
        # change. "on" arms the divisor; INDEPENDENT of theta_sites (composes
        # with "2.2" for the CoR-P/CoR-M form of record) and valid with
        # smear_global_selection=False. At θ=(0,1) the literal skip applies
        # (GATE T-ID). Guards: "on" with catalogue_global_selection resolving
        # to "s3d", or with normalization_mode != "absolute_marginal", raises.
        theta_phi_divisor: str = "off",
        # [HIER] mass-aware 1D catalogue leg instrument (row #255 tree 2 node
        # T2.3, PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md §2). "off"
        # (default, PRODUCTION) is byte-identical -- sites N1/D1/W1 unchanged.
        # "on" replaces the WITHOUT-BH catalogue numerator's per-candidate
        # survival S_bar_phi(z;h) by S_4D(d_L(z;h), M_g(1+z)) (mirroring
        # self._sigma4d_mass_kernel), Sigma_4D as the global divisor and
        # alpha_G_phi as the mixture weight -- the no-mass-likelihood image
        # of the 2D assembly (:6524). Guarded (raises at setup, not a silent
        # no-op): requires normalization_mode="absolute_marginal",
        # catalogue_numerator_survival resolving to "phi",
        # catalogue_global_selection resolving to "phi" and
        # theta_phi_divisor="off" (no theta-consistent Sigma_4D exists).
        # COUNTERFACTUAL, never a production posterior; the production
        # default flip is a fresh [RULE] (section 11 of the gate doc).
        catalogue_leg_1d_mass_aware: str = "off",
        # INSTRUMENTATION (T2.2, row #255 tree 2 node T2.2; A10 = instrumentation
        # guard, not a physics gate; B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md
        # §6). None (default) is byte-identical -- no code path change, no
        # value read or written differently (GATE BI). When set to a directory
        # path, a READ-ONLY per-(event, candidate) diagnostic serialiser writes
        # per_candidate_h_<label>.csv and per_event_h_<label>.csv there, built
        # entirely from state p_Di already computed for a normal run (the
        # candidate_hosts lists, posterior_data_with_bh_mass, the phi-survival
        # table, the diagnostic row); it never writes into anything the
        # likelihood consumes.
        candidate_dump_dir: str | None = None,
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
        # INSTRUMENTATION (T2.2, row #255 A10): None => byte-identical, no
        # directory created, nothing collected or written below.
        self._candidate_dump_dir = candidate_dump_dir
        if self._candidate_dump_dir is not None:
            os.makedirs(self._candidate_dump_dir, exist_ok=True)
        # [HIER] θ-hook (C1/C2): validate, then store for p_Di's site-2.2
        # dispatch and the site-2.3 precompute below. The identity default
        # stores (0.0, 1.0, "all") and engages nothing.
        _theta_engaged = theta_b != 0.0 or theta_s != 1.0
        if _theta_engaged:
            _validate_theta(theta_b, theta_s)
        if theta_sites not in ("all", "2.1", "2.2", "2.3"):
            raise ValueError(
                f"theta_sites must be 'all', '2.1', '2.2' or '2.3', got {theta_sites!r}"
            )
        if _theta_engaged and theta_sites in ("all", "2.3") and not smear_global_selection:
            raise ValueError(
                "theta with site 2.3 enabled requires smear_global_selection=True "
                "(the registered site is the smeared global-selection kernel); "
                "pass theta_sites='2.1'/'2.2' or enable smearing"
            )
        self._theta_b = float(theta_b)
        self._theta_s = float(theta_s)
        self._theta_sites = str(theta_sites)
        if _theta_engaged:
            _LOGGER.warning(
                "INSTRUMENTATION ACTIVE: theta=(b=%.6g, s=%.6g), sites=%s — the "
                "host-z kernel is reparametrized ([HIER] θ-hook). This run is a "
                "COUNTERFACTUAL/profile point, not a production posterior.",
                theta_b,
                theta_s,
                theta_sites,
            )
        self._freeze_g_frac_ref_h = (
            float(freeze_g_frac_ref_h) if freeze_g_frac_ref_h is not None else None
        )
        if self._freeze_g_frac_ref_h is not None:
            _LOGGER.warning(
                "INSTRUMENTATION ACTIVE: --freeze_g_frac_ref_h=%.6g — the 2D "
                "completion leg uses B_num(h)*g_frac(h_ref) instead of "
                "B_num_wbh(h). This run is a COUNTERFACTUAL, not a production "
                "posterior.",
                self._freeze_g_frac_ref_h,
            )
        _sel_cell = str(selection_in_completion_numerator)
        if _sel_cell not in ("auto", "off", "1d", "2d", "fused"):
            raise ValueError(
                "selection_in_completion_numerator must be 'auto', 'off', '1d', '2d' "
                f"or 'fused', got {selection_in_completion_numerator!r}"
            )
        if _sel_cell == "auto":
            # [PHYSICS] rows #117-#118: fused survival is the absolute_marginal
            # default; every other normalization mode stays byte-identical.
            _sel_cell = "fused" if normalization_mode == "absolute_marginal" else "off"
        self._selection_in_completion_numerator = _sel_cell
        if _sel_cell != "off":
            if normalization_mode != "absolute_marginal":
                raise ValueError(
                    f"selection_in_completion_numerator={_sel_cell!r} requires "
                    "normalization_mode='absolute_marginal' (the S_bar_phi table it "
                    f"reuses is only built there); got {normalization_mode!r}"
                )
            if _sel_cell == "fused":
                _LOGGER.info(
                    "[PHYSICS] selection fusion ACTIVE (rows #117-#118): the 1D "
                    "completion numerator carries S_bar_phi(z;h) and the 2D leg "
                    "uses the fused g_sel,prod (S_4D inside the mass quadrature). "
                    "This is the production absolute_marginal estimator."
                )
            else:
                _LOGGER.warning(
                    "INSTRUMENTATION ACTIVE: --selection_in_completion_numerator=%s — "
                    "a single-leg selection cell ('1d' = [P2]-only, '2d' = "
                    "[P1]-only). The paired production form is 'fused'; this run "
                    "is a COUNTERFACTUAL decomposition cell, not a production "
                    "posterior.",
                    _sel_cell,
                )
        elif normalization_mode == "absolute_marginal":
            _LOGGER.warning(
                "COUNTERFACTUAL: selection_in_completion_numerator='off' under "
                "absolute_marginal — the legacy pre-#118 estimator (no survival "
                "factor in either completion leg). Not a production posterior."
            )
        # [P3-IMP] catalogue-leg twin, ADOPTED (docs/derivations/
        # PROPOSAL_CATALOGUE_TWIN_PRODUCTION_20260825.md §2/§6; row #195).
        _cat_num_surv = str(catalogue_numerator_survival)
        if _cat_num_surv not in ("auto", "off", "phi", "phi_flat"):
            raise ValueError(
                "catalogue_numerator_survival must be 'auto', 'off', 'phi' or "
                f"'phi_flat', got {catalogue_numerator_survival!r}"
            )
        if _cat_num_surv == "auto":
            # row #195: per-candidate S_bar_phi in the catalogue numerator is
            # production under absolute_marginal (the S_bar_phi table it
            # reads is only built there); every other normalization mode
            # stays byte-identical on "off". "phi_flat" (the K-flat kill arm)
            # is never reached by "auto" -- explicit-only, as before.
            _cat_num_surv = "phi" if normalization_mode == "absolute_marginal" else "off"
        if _cat_num_surv in ("phi", "phi_flat"):
            if normalization_mode != "absolute_marginal":
                raise ValueError(
                    "catalogue_numerator_survival='phi' requires "
                    "normalization_mode='absolute_marginal' (the S_bar_phi table it "
                    f"reads is only built there); got {normalization_mode!r}"
                )
            if _cat_num_surv == "phi":
                _LOGGER.info(
                    '[PHYSICS] catalogue_numerator_survival="phi" ACTIVE (row #195): '
                    "the WITHOUT-BH catalogue numerator carries per-candidate "
                    "S_bar_phi(z;h) inside the z-quadrature. This is the production "
                    "absolute_marginal estimator."
                )
            else:
                _LOGGER.warning(
                    "COUNTERFACTUAL: catalogue_numerator_survival=%r — per-host "
                    "S_bar_phi in the catalogue numerator ([P3-IMP] twin cell; "
                    "K-flat CONSTANT table). Not a production posterior.",
                    _cat_num_surv,
                )
        elif normalization_mode == "absolute_marginal":
            _LOGGER.warning(
                'COUNTERFACTUAL: catalogue_numerator_survival="off" under '
                "absolute_marginal — the pre-adoption WITHOUT-BH catalogue "
                "numerator (no per-candidate survival factor). Not a production "
                "posterior."
            )
        self._catalogue_numerator_survival = _cat_num_surv
        # [P3-2D] the with-BH catalogue-leg twin, ADOPTED (row #223 standing
        # grant, charter node B7.3; PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md).
        if catalogue_numerator_survival_2d not in ("off", "mz_sel"):
            raise ValueError(
                "catalogue_numerator_survival_2d must be 'off' or 'mz_sel', "
                f"got {catalogue_numerator_survival_2d!r}"
            )
        if catalogue_numerator_survival_2d == "mz_sel":
            if catalogue_numerator_survival_2d_center not in ("raw", "eff"):
                raise ValueError(
                    "catalogue_numerator_survival_2d='mz_sel' requires "
                    "catalogue_numerator_survival_2d_center to be explicitly "
                    "'raw' or 'eff' (no silent default -- the centering "
                    "choice is PENDING the pre-execution review, "
                    "PREREGISTRATION_P3_2D_20260825.md §2(i)); got "
                    f"{catalogue_numerator_survival_2d_center!r}"
                )
            if catalogue_numerator_survival_2d_center == "eff":
                _LOGGER.info(
                    '[PHYSICS] catalogue_numerator_survival_2d="mz_sel" '
                    '(center="eff") ACTIVE (adopted under row #223, charter B7.3): the WITH-BH catalogue '
                    "numerator carries S_4D(d_L(z;h), x*M_z,det) inside its "
                    "own mass quadrature (the product-Gaussian identity). "
                    "This is the production with-BH catalogue-leg twin "
                    "estimator."
                )
            else:
                _LOGGER.warning(
                    "COUNTERFACTUAL: catalogue_numerator_survival_2d=%r "
                    "(center=%r) — S_4D inside the with-BH catalogue "
                    "numerator's own mass quadrature with the RAW "
                    "(non-Eddington-shifted) centering ([P3-2D] twin "
                    "instrument only; the adopted centering is 'eff'). Not "
                    "a production posterior.",
                    catalogue_numerator_survival_2d,
                    catalogue_numerator_survival_2d_center,
                )
        else:
            _LOGGER.warning(
                'COUNTERFACTUAL: catalogue_numerator_survival_2d="off" — '
                "the pre-adoption WITH-BH catalogue numerator (no "
                "per-candidate survival factor inside the mass quadrature). "
                "Not a production posterior."
            )
        self._catalogue_numerator_survival_2d = catalogue_numerator_survival_2d
        self._catalogue_numerator_survival_2d_center = catalogue_numerator_survival_2d_center
        # [P3-RPHI] the fourth Path-A slot, ADOPTED (docs/derivations/
        # PROPOSAL_SIGMA_PHI_DIVISOR_20260822.md §2/§6(ii); rows #172-#178).
        _cat_glob_sel = str(catalogue_global_selection)
        if _cat_glob_sel not in ("auto", "s3d", "phi"):
            raise ValueError(
                "catalogue_global_selection must be 'auto', 's3d' or 'phi', "
                f"got {catalogue_global_selection!r}"
            )
        if _cat_glob_sel == "auto":
            # rows #172-#178: Sigma^phi is production under absolute_marginal
            # (the S_bar_phi table it reads is only built there); every other
            # normalization mode stays byte-identical on Sigma^3D.
            _cat_glob_sel = "phi" if normalization_mode == "absolute_marginal" else "s3d"
        if _cat_glob_sel == "phi":
            if normalization_mode != "absolute_marginal":
                raise ValueError(
                    "catalogue_global_selection='phi' requires "
                    "normalization_mode='absolute_marginal' (the Sigma^phi table it "
                    f"reads is only built there); got {normalization_mode!r}"
                )
            _LOGGER.info(
                '[PHYSICS] catalogue_global_selection="phi" ACTIVE (rows #172-#178): '
                "the no-BH catalogue divisor is Σ^φ. This is the production "
                "absolute_marginal estimator."
            )
        elif normalization_mode == "absolute_marginal":
            _LOGGER.warning(
                'COUNTERFACTUAL: catalogue_global_selection="s3d" under '
                "absolute_marginal — no-BH catalogue divisor Σ³ᴰ (the "
                "pre-adoption [P3-RPHI] slot). Not a production posterior."
            )
        self._catalogue_global_selection = _cat_glob_sel
        # [DEFECT] candidate-host mass pre-filter window (ledger row #198):
        # opaque plumbing only -- validated at the single read site
        # (get_possible_hosts_from_ball_tree's mass_filter_mask branch), not
        # here, so there is exactly one place that interprets the value.
        self._mass_filter_sigma = str(mass_filter_sigma)
        # Mass-window GEOMETRY instrument flag (charter node B5.1, ledger
        # rows #220-#223): same "opaque plumbing" convention as
        # mass_filter_sigma above -- validated at the single read site in
        # get_possible_hosts_from_ball_tree, not here.
        self._mass_filter_geometry = str(mass_filter_geometry)
        self._mass_filter_k = float(mass_filter_k)
        # Sky-cone-radius instrument flag (PHYSICS_CHANGE_THETA_DIVISOR_
        # 20260830.md §2.5): guard pattern, not a silent no-op.
        self._sky_cone_k = float(sky_cone_k)
        if not (self._sky_cone_k > 0.0) or not np.isfinite(self._sky_cone_k):
            raise ValueError(f"sky_cone_k must be finite and > 0, got {sky_cone_k}")
        # [HIER] site 2.3phi theta-consistent no-BH divisor instrument
        # (PHYSICS_CHANGE_THETA_DIVISOR_20260830.md §2.2, row #255 tree 2 node
        # T1.1). Guard pattern: "on" requires a phi table to transform
        # (catalogue_global_selection resolving to "phi", i.e.
        # normalization_mode="absolute_marginal", which precompute_
        # global_catalog_selection's phi branch also requires) -- checked
        # here so a misconfiguration raises at setup, not silently no-ops.
        _theta_phi_div = str(theta_phi_divisor)
        if _theta_phi_div not in ("off", "on"):
            raise ValueError(f"theta_phi_divisor must be 'off' or 'on', got {theta_phi_divisor!r}")
        if _theta_phi_div == "on":
            if self._catalogue_global_selection != "phi":
                raise ValueError(
                    "theta_phi_divisor='on' requires catalogue_global_selection "
                    "to resolve to 'phi' (site 2.3phi transforms the phi-"
                    f"marginal divisor); resolved to "
                    f"{self._catalogue_global_selection!r} (no phi table to "
                    "transform)"
                )
            if normalization_mode != "absolute_marginal":
                raise ValueError(
                    "theta_phi_divisor='on' requires normalization_mode="
                    "'absolute_marginal' (no phi objects are built otherwise); "
                    f"got {normalization_mode!r}"
                )
        self._theta_phi_divisor = _theta_phi_div
        # Prod2d closure counterfactual instrument (results/
        # prod2d_closure_20260818/PREREGISTRATION_PROD_COUNTERFACTUAL.md §1,
        # P8): validated here the same way selection_in_completion_numerator
        # is validated above.
        _cat_mass_overlap = str(catalogue_mass_overlap)
        if _cat_mass_overlap not in ("production", "neutralized", "inflated"):
            raise ValueError(
                "catalogue_mass_overlap must be 'production', 'neutralized' "
                f"or 'inflated', got {catalogue_mass_overlap!r}"
            )
        _cat_mass_error_scale = float(catalogue_mass_error_scale)
        if _cat_mass_error_scale != 1.0 and _cat_mass_overlap != "inflated":
            raise ValueError(
                "catalogue_mass_error_scale != 1.0 requires "
                "catalogue_mass_overlap='inflated' "
                f"(got catalogue_mass_overlap={_cat_mass_overlap!r}, "
                f"catalogue_mass_error_scale={_cat_mass_error_scale!r})."
            )
        self._catalogue_mass_overlap = _cat_mass_overlap
        self._catalogue_mass_error_scale = _cat_mass_error_scale
        # Completion-leg normalization convention (docs/derivations/
        # bscale_completion_normalization.md §6/§7; ledger rows #130-#131).
        _completion_b_scale = str(completion_b_scale)
        if _completion_b_scale not in ("derived", "legacy"):
            raise ValueError(
                f"completion_b_scale must be 'derived' or 'legacy', got {completion_b_scale!r}"
            )
        self._completion_b_scale = _completion_b_scale
        if _completion_b_scale == "legacy":
            _LOGGER.warning(
                "INSTRUMENTATION ACTIVE: --completion_b_scale=legacy — the "
                "completion-leg numerator carries the un-derived "
                "beta_Gbar_phi/beta_Gbar multiplier (docs/derivations/"
                "bscale_completion_normalization.md §6, DEFECT). Kept for "
                "historical-run reproduction only; the production default is "
                "'derived'."
            )
        if _cat_mass_overlap != "production":
            _LOGGER.warning(
                "INSTRUMENTATION ACTIVE: --catalogue_mass_overlap=%s "
                "(catalogue_mass_error_scale=%.6g) — a prod2d closure "
                "counterfactual (results/prod2d_closure_20260818/"
                "PREREGISTRATION_PROD_COUNTERFACTUAL.md §1). Not a production "
                "posterior.",
                _cat_mass_overlap,
                _cat_mass_error_scale,
            )
        # Tilt-ledger battery counterfactual instruments (results/
        # prod2d_closure_20260818/PREREGISTRATION_TILT_BATTERY.md §1/§2/§6).
        # Validated here the same way catalogue_mass_overlap is validated
        # above (defence in depth against argparse's choices= gate).
        _eddington_m = str(eddington_m)
        if _eddington_m not in ("on", "off"):
            raise ValueError(f"eddington_m must be 'on' or 'off', got {eddington_m!r}")
        self._eddington_m = _eddington_m
        if _eddington_m == "off":
            _LOGGER.warning(
                "INSTRUMENTATION ACTIVE: --eddington_m=off — the 2D catalogue "
                "leg's mu_gal and the per-host D_g erf-sum both use the raw "
                "(unshifted) host_M instead of eddington_shifted_host_mass "
                "(results/prod2d_closure_20260818/PREREGISTRATION_TILT_BATTERY.md "
                "§1 instrument E). Not a production posterior."
            )
        _sigma4d_mass_kernel = str(sigma4d_mass_kernel)
        if _sigma4d_mass_kernel not in ("point", "kernel"):
            raise ValueError(
                f"sigma4d_mass_kernel must be 'point' or 'kernel', got {sigma4d_mass_kernel!r}"
            )
        self._sigma4d_mass_kernel = _sigma4d_mass_kernel
        if _sigma4d_mass_kernel == "kernel":
            _LOGGER.warning(
                "INSTRUMENTATION ACTIVE: --sigma4d_mass_kernel=kernel — the "
                "global with-BH-mass catalogue selection Sigma^4D replaces its "
                "point p_det evaluation with the registered mass-smearing "
                "kernel (results/prod2d_closure_20260818/"
                "PREREGISTRATION_TILT_BATTERY.md §1 instrument J). Not a "
                "production posterior."
            )
        # [HIER] mass-aware 1D catalogue leg instrument (row #255 tree 2 node
        # T2.3, PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md §2.1). Guard
        # pattern (raises at setup, not a silent no-op): "on" requires a phi
        # numerator/divisor to REPLACE (catalogue_numerator_survival and
        # catalogue_global_selection both resolving to "phi", i.e.
        # normalization_mode="absolute_marginal") and no theta-consistent
        # Sigma_phi_reg divisor already armed (theta_phi_divisor="off" --
        # Sigma_4D is theta-inert under the registered form, T1.1 invariant
        # 9; no theta-consistent Sigma_4D exists).
        _cat_leg_1d_ma = str(catalogue_leg_1d_mass_aware)
        if _cat_leg_1d_ma not in ("off", "on"):
            raise ValueError(
                "catalogue_leg_1d_mass_aware must be 'off' or 'on', got "
                f"{catalogue_leg_1d_mass_aware!r}"
            )
        if _cat_leg_1d_ma == "on":
            if self._catalogue_numerator_survival != "phi":
                raise ValueError(
                    "catalogue_leg_1d_mass_aware='on' requires "
                    "catalogue_numerator_survival to resolve to 'phi' (site N1 "
                    "replaces the phi per-candidate survival); resolved to "
                    f"{self._catalogue_numerator_survival!r}"
                )
            if self._catalogue_global_selection != "phi":
                raise ValueError(
                    "catalogue_leg_1d_mass_aware='on' requires "
                    "catalogue_global_selection to resolve to 'phi' (site D1 "
                    "replaces the phi global divisor); resolved to "
                    f"{self._catalogue_global_selection!r}"
                )
            if self._theta_phi_divisor != "off":
                raise ValueError(
                    "catalogue_leg_1d_mass_aware='on' requires "
                    "theta_phi_divisor='off' (Sigma_4D is theta-inert under "
                    "the registered form; no theta-consistent Sigma_4D "
                    f"exists); got theta_phi_divisor={self._theta_phi_divisor!r}"
                )
            _LOGGER.warning(
                "COUNTERFACTUAL: catalogue_leg_1d_mass_aware='on' (row #255 "
                "tree 2 node T2.3) — the WITHOUT-BH catalogue numerator "
                "carries S_4D(d_L(z;h), M_g(1+z)) in place of S_bar_phi(z;h), "
                "Sigma_4D replaces Sigma^phi as the global divisor and "
                "alpha_G_phi replaces beta_G_phi as the mixture weight "
                "(PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md §2). Not a "
                "production posterior; the production-default flip returns "
                "to the author as a fresh [RULE]."
            )
        self._catalogue_leg_1d_mass_aware = _cat_leg_1d_ma
        # B-DEN falsifier instrument (docs/derivations/
        # completion_numerator_data_measure.md §6; AMENDMENT A-5). Validated
        # here the same way the other instrument flags above are.
        _completion_event_measure = str(completion_event_measure)
        if _completion_event_measure not in ("ratio", "data"):
            raise ValueError(
                "completion_event_measure must be 'ratio' or 'data', got "
                f"{completion_event_measure!r}"
            )
        self._completion_event_measure = _completion_event_measure
        if _completion_event_measure == "data":
            _LOGGER.warning(
                "INSTRUMENTATION ACTIVE: --completion_event_measure=data — the "
                "completion numerator's GW event term is evaluated as a "
                "density in the observable d_L,det instead of the "
                "dimensionless distance ratio (docs/derivations/"
                "completion_numerator_data_measure.md §6). This is the B-DEN "
                "falsifier instrument, not a production posterior."
            )
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
        # #40 remainder: validate the mass-kernel choice up front (raises on
        # unknown AND on the prior-inconsistent point-z x trunc-mass
        # combination); recomputed identically inside the worker kernels.
        _resolved_mass_kernel = resolve_host_mass_kernel(
            host_mass_kernel, normalization_mode, host_z_kernel
        )
        if host_mass_kernel != "auto":
            _LOGGER.info(
                "host_mass_kernel=%r overrides the mode-bundled 2D mass kernel "
                "(resolved: %s mass marginal with %s normalization) — "
                "docs/derivations/mass_marginal_2d_kernel.md",
                host_mass_kernel,
                _resolved_mass_kernel,
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
        # [PHYSICS] Realistic host-observation model (campaign #53, RATIFIED
        # 2026-07-29): one-directional scattered-catalogue guard set (§3.4/§9).
        # getattr fallback: tests construct lightweight stand-in catalogues
        # without the sidecar plumbing — those are unscattered by definition.
        self._catalogue_scattered = bool(getattr(galaxy_catalog, "scattered", False))
        validate_scatter_guards(
            normalization_mode=normalization_mode,
            host_z_kernel=host_z_kernel,
            host_mass_kernel=host_mass_kernel,
            catalogue_scattered=self._catalogue_scattered,
        )
        if self._catalogue_scattered:
            _LOGGER.info(
                "Scattered observed-catalogue realization loaded (sidecar: %s) — "
                "point host-z kernel and generator_marginal refused; width "
                "kernels are load-bearing "
                "(docs/derivations/realistic_host_observation_model.md).",
                getattr(galaxy_catalog, "realization_metadata", None),
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
        self._host_mass_kernel = host_mass_kernel
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
        # Host BH masses are SOURCE-frame: integrate over the Babak et al.
        # (2017) arXiv:1703.09722 band from constants (the single mass
        # boundary, issue #51). parameter_space.M.limits are now the
        # detector-frame M_z domain and must not be used here.
        BH_MASS_LOWER_LIMIT = M_SOURCE_FRAME_MIN
        BH_MASS_UPPER_LIMIT = M_SOURCE_FRAME_MAX

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
            # FIX-3 §7.1 (opt-in): joint z x M_z with-BH survival; every
            # with-BH consumer passes its own z via _wbh_z_kwargs
            # (atomic-switch rule, fix3_zmz_catalog_selection.md §3.5).
            pdet_wbh_z_resolved=pdet_wbh_z_resolved,
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
        # [HIER] θ-hook site 2.3 dispatch (OAT toggle, PA-HIER-23): θ reaches
        # the global-selection denominator only when site 2.3 is enabled.
        _theta_b_23, _theta_s_23 = (
            (self._theta_b, self._theta_s) if self._theta_sites in ("all", "2.3") else (0.0, 1.0)
        )
        _global_cat_denom_no_bh = precompute_global_catalog_selection(
            h_values=_h_list,
            galaxy_catalog=galaxy_catalog,
            detection_probability_obj=detection_probability,
            with_bh_mass=False,
            z_max_cap=REDSHIFT_UPPER_LIMIT,
            smear_sigma_z=smear_global_selection,
            theta_b=_theta_b_23,
            theta_s=_theta_s_23,
        )
        _global_cat_denom_with_bh = precompute_global_catalog_selection(
            h_values=_h_list,
            galaxy_catalog=galaxy_catalog,
            detection_probability_obj=detection_probability,
            with_bh_mass=True,
            z_max_cap=REDSHIFT_UPPER_LIMIT,
            smear_sigma_z=smear_global_selection,
            sigma4d_mass_kernel=self._sigma4d_mass_kernel,
            eddington_m=self._eddington_m,
            theta_b=_theta_b_23,
            theta_s=_theta_s_23,
        )
        # [PHYSICS] Path (A): ONE detection model (FIXB_PATHA_PACKAGE.md §3.2,
        # 2026-08-04). Only the production mixture mode consumes the
        # phi-convention tables; the legacy tables above stay untouched so the
        # generator_marginal assembly is byte-identical (gate (iii-a)).
        _phi_survival_table: dict[
            float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        ] = {}
        _beta_G_phi_table: dict[float, float] = {}
        _beta_Gbar_phi_table: dict[float, float] = {}
        _global_cat_selection_phi: dict[float, float] = {}
        # [HIER] site 2.3phi (row #255 tree 2 node T1.1): stays empty unless
        # theta_phi_divisor="on" AND theta != (0,1) -- the consumer (p_Di)
        # falls back to _global_cat_selection_phi (the point table) whenever
        # this dict has no entry for the current h (GATE T-ID literal skip,
        # and the "off" default).
        _global_cat_selection_phi_theta: dict[float, float] = {}
        _use_phi_selection = normalization_mode == "absolute_marginal"
        if _use_phi_selection:
            # MINOR-1 (row #118): the frozen-g_frac counterfactual re-enters the
            # 1D integrand at h_ref; with the S_bar_phi factor on (fused/'1d')
            # an off-grid h_ref would ValueError in the table lookup, so the
            # reference h is tabulated alongside the evaluation grid.
            _phi_h_list = list(_h_list)
            if (
                self._freeze_g_frac_ref_h is not None
                and self._freeze_g_frac_ref_h not in _phi_h_list
            ):
                _phi_h_list.append(self._freeze_g_frac_ref_h)
            _phi_survival_table = precompute_phi_marginal_survival(
                h_values=_phi_h_list,
                detection_probability_obj=detection_probability,
                z_max_cap=REDSHIFT_UPPER_LIMIT,
            )
            _beta_G_phi_table, _beta_Gbar_phi_table = precompute_phi_selection_integrals(
                h_values=_h_list,
                phi_survival_table=_phi_survival_table,
                completeness=completeness,
            )
            # Sigma^phi on the SAME catalogue rows/weights/eligibility as
            # Sigma^4D (decision D2's self-consistency rule).
            _global_cat_selection_phi = precompute_global_catalog_selection(
                h_values=_h_list,
                galaxy_catalog=galaxy_catalog,
                detection_probability_obj=detection_probability,
                with_bh_mass=False,
                z_max_cap=REDSHIFT_UPPER_LIMIT,
                smear_sigma_z=False,
                phi_survival_table=_phi_survival_table,
            )
            # [HIER] site 2.3phi (PHYSICS_CHANGE_THETA_DIVISOR_20260830.md
            # §2.2, row #255 tree 2 node T1.1): theta-consistent no-BH
            # divisor ratio. INDEPENDENT of theta_sites; engages only when
            # armed AND theta != (0,1). At theta=(0,1) the literal skip
            # applies (GATE T-ID) -- _global_cat_selection_phi_theta stays
            # empty and the consumer (p_Di) falls back to the stored point
            # table object itself, no floating operation performed.
            _theta_phi_engaged = self._theta_phi_divisor == "on" and (
                self._theta_b != 0.0 or self._theta_s != 1.0
            )
            _phi_divisor_ratio_table: dict[float, dict[str, float]] = {}
            if _theta_phi_engaged:
                _theta_hook_count("site_2_3_phi")
                _phi_divisor_ratio_table = precompute_phi_divisor_theta_ratio(
                    h_values=_h_list,
                    galaxy_catalog=galaxy_catalog,
                    completeness=completeness,
                    phi_survival_table=_phi_survival_table,
                    theta_b=self._theta_b,
                    theta_s=self._theta_s,
                )
                for _h_r in _h_list:
                    _global_cat_selection_phi_theta[_h_r] = (
                        _global_cat_selection_phi[_h_r] * _phi_divisor_ratio_table[_h_r]["rho"]
                    )
            elif self._theta_phi_divisor == "on":
                _LOGGER.info(
                    "site 2.3phi armed, identity theta, divisor = point table "
                    "(theta=(0,1), GATE T-ID literal skip)."
                )
            for _h_phi in _h_list:
                _log_path_a_selection_objects(
                    _h_phi,
                    beta_G_phi=_beta_G_phi_table[_h_phi],
                    beta_Gbar_phi=_beta_Gbar_phi_table[_h_phi],
                    sigma_phi=_global_cat_selection_phi[_h_phi],
                    sigma_4d=_global_cat_denom_with_bh[_h_phi],
                    w_G_legacy=(
                        _beta_G_table[_h_phi] / _D_h_table[_h_phi]
                        if _D_h_table.get(_h_phi, 0.0) > 0.0
                        else float("nan")
                    ),
                )
                # P4 (PREREGISTRATION_TILT_BATTERY.md §2 N-2(J) banked
                # source): dump the per-h selection table to a small JSON in
                # the working directory so instrument J's engagement gate is
                # scored from files, not log-line scraping. Always written
                # whenever the phi-selection objects above are computed
                # (independent of --sigma4d_mass_kernel/--eddington_m).
                # [HIER] site 2.3phi instrumentation (section 2.2): so the
                # T1.2 gates are scored from files, not log-line scraping.
                _ratio_h = _phi_divisor_ratio_table.get(_h_phi)
                _kappa_h: float | None = None
                if _ratio_h is not None and _global_cat_selection_phi[_h_phi] > 0.0:
                    _kappa_h = _ratio_h["sigma_phi_smear_truth"] / _global_cat_selection_phi[_h_phi]
                write_selection_table_json(
                    _h_phi,
                    beta_G_phi=_beta_G_phi_table[_h_phi],
                    beta_Gbar_phi=_beta_Gbar_phi_table[_h_phi],
                    sigma_phi=_global_cat_selection_phi[_h_phi],
                    sigma_4d=_global_cat_denom_with_bh[_h_phi],
                    sigma_phi_theta=_global_cat_selection_phi_theta.get(_h_phi),
                    sigma_phi_smear_truth=(
                        _ratio_h["sigma_phi_smear_truth"] if _ratio_h is not None else None
                    ),
                    rho_theta=(_ratio_h["rho"] if _ratio_h is not None else None),
                    kappa_smear_over_point=_kappa_h,
                    n_degenerate_rows=(
                        _ratio_h["n_degenerate_rows"] if _ratio_h is not None else None
                    ),
                    w_share_degenerate=(
                        _ratio_h["w_share_degenerate"] if _ratio_h is not None else None
                    ),
                    theta_b=(self._theta_b if self._theta_phi_divisor == "on" else None),
                    theta_s=(self._theta_s if self._theta_phi_divisor == "on" else None),
                    theta_phi_divisor=(
                        self._theta_phi_divisor if self._theta_phi_divisor == "on" else None
                    ),
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
                # w_G at 7 significant figures (INSTR-2): the h-dependence of the
                # catalogue partition weight is a ~1e-5 effect at the noise floor of
                # the 2D-bias investigation -- %.4f rounded it away.
                "Partition-norm: w_G=beta_G/D(h)=%.7g, sum_w_Dg(no_bh)=%.4e, sum_w_Dg(with_bh)=%.4e",
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
        # [PHYSICS] (N8) g_i's mass kernel: the (d_L_frac, M_z_frac) 2x2 block
        # conditional of cov_4d (GATE_PACKAGE_FINAL.md §2.2 item 2). The 2D
        # completion leg knows only d_L (p_gw has already collapsed the sky),
        # so g_i conditions on d_L alone — NOT on the 3-observable
        # (_proj_arr, _sigma2_cond_arr) pair the catalogue leg uses.
        _proj_d_L_to_M_arr = np.zeros(n_det)
        _sigma_cond_M_arr = np.zeros(n_det)

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
            if _ABLATE_MZ_PROJ:
                # [DIAGNOSTIC] branch (b): independent-M_z Gaussian — marginal
                # variance, zero projection (see _ABLATE_MZ_PROJ constant).
                _sigma2_cond_arr[slot] = max(float(cov_mz), 1e-30)
                _proj_arr[slot] = 0.0
            else:
                _sigma2_cond_arr[slot] = max(
                    float(cov_mz - cov_cross @ cov_obs_inv @ cov_cross), 1e-30
                )
                _proj_arr[slot] = cov_cross @ cov_obs_inv
            # (N8) d_L-only 2x2 block conditional for g_i: mu_cond(z) =
            # 1 + proj (d_L_frac - 1), sigma_cond^2 = S_MM - S_dM^2/S_dd.
            # Bishop (2006) PRML Eqs. 2.81-2.82 on the 2x2 block of cov_4d.
            _s_dd = float(cov_4d[2, 2])
            _s_dm = float(cov_4d[2, 3])
            _proj_d_L_to_M_arr[slot] = _s_dm / _s_dd if _s_dd > 0.0 else 0.0
            _sigma_cond_M_arr[slot] = math.sqrt(
                max(float(cov_mz) - (_s_dm * _s_dm / _s_dd if _s_dd > 0.0 else 0.0), 1e-30)
            )

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
        # Path-(A) phi-convention tables (FIXB_PATHA_PACKAGE.md §3.2) and the
        # (N8) g_i per-event mass-kernel scalars.
        self._use_phi_selection = _use_phi_selection
        self._beta_G_phi_table = _beta_G_phi_table
        self._beta_Gbar_phi_table = _beta_Gbar_phi_table
        self._global_cat_selection_phi = _global_cat_selection_phi
        # [HIER] site 2.3phi (row #255 tree 2 node T1.1): the registered
        # theta-consistent divisor Sigma_phi_reg(theta;h) = Sigma_phi_point(h)
        # * rho(theta;h); empty unless armed AND theta != (0,1) (see above).
        self._global_cat_selection_phi_theta = _global_cat_selection_phi_theta
        # The S_bar_phi(z;h) table itself (h -> (z_grid, S_bar_phi)). Read by
        # the N-2 counterfactual in p_Di through the SAME np.interp accessor
        # precompute_global_catalog_selection uses for Sigma^phi — one object,
        # one interpolation, never a re-typed copy.
        self._phi_survival_table = _phi_survival_table
        self._proj_d_L_to_M = _proj_d_L_to_M_arr
        self._sigma_cond_M = _sigma_cond_M_arr
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
                    "darksiren_emri.bayesian_inference.simulation_detection_probability",
                    "darksiren_emri.physical_relations",
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
                completeness,
                _proj_d_L_to_M_arr,
                _sigma_cond_M_arr,
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
                    if self._candidate_dump_dir is not None:
                        self._candidate_dump_rows = []
                        self._candidate_dump_event_rows = []

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

                # Observed-catalogue provenance (§6.1 item 3): record the
                # realization sidecar next to the posteriors so every leg of a
                # run is verifiably tied to ONE realization. Written as a
                # SEPARATE file (not inside the per-event h_*.json) so the
                # per-event posterior schema — and the sigma_scale = 0
                # bit-identity gate [RATIFY-R6] on those files — is untouched.
                _realization_metadata = getattr(galaxy_catalog, "realization_metadata", None)
                if _realization_metadata is not None:
                    for _prov_dir in (
                        "simulations/posteriors",
                        "simulations/posteriors_with_bh_mass",
                    ):
                        with open(
                            os.path.join(_prov_dir, "realization_provenance.json"), "w"
                        ) as _prov_file:
                            json.dump(_realization_metadata, _prov_file, indent=2)

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

                # T2.2 (row #255 A10) candidate-dump instrumentation: OFF by
                # default (self._candidate_dump_dir is None), byte-identical.
                if self._candidate_dump_dir is not None and (
                    self._candidate_dump_rows or self._candidate_dump_event_rows
                ):
                    self._write_candidate_dump_csvs(self._candidate_dump_dir)

        # Write Fisher quality CSV (per D-12) — h-invariant, once per run
        self._write_fisher_quality_csv()

        # Generate Fisher quality diagnostic plot (per D-06, D-07)
        from darksiren_emri.plotting.fisher_plots import plot_fisher_diagnostics

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
            # Path-(A) instrumentation (FIXB_PATHA_PACKAGE.md §5): the legacy
            # beta_G/D weight is RENAMED to w_G_legacy (kept, not overwritten)
            # and the operative w~_G plus its ingredients are emitted at 7 s.f.
            "w_G_legacy",
            "w_tilde_G",
            "alpha_G_phi",
            "r_Malm",
            "D_tilde_phi",
            "L_cat_no_bh",
            "L_cat_with_bh",
            "B_num",
            "B_num_wbh",
            "g_frac",
            "L_comp",
            "combined_no_bh",
            "combined_with_bh",
            # [HIER] C2 (PA-HIER-23): separable ln L terms — num_log − den_log
            # is the event's ln L; the OAT toggle matrix reads these, never the
            # aggregate.
            "den_log_term",
            "num_log_term_no_bh",
            "num_log_term_with_bh",
        ]
        # 7 significant figures on the path-(A) diagnostics.
        _seven_sf = ("w_G_legacy", "w_tilde_G", "alpha_G_phi", "r_Malm", "D_tilde_phi", "g_frac")

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        write_header = not os.path.isfile(csv_path)

        rows_out: list[dict[str, Any]] = []
        for row in self._diagnostic_rows:
            out = {key: row.get(key, "") for key in fieldnames}
            for key in _seven_sf:
                value = out.get(key, "")
                if isinstance(value, float):
                    out[key] = f"{value:.7g}"
            rows_out.append(out)

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows_out)

        _LOGGER.info("Wrote %d diagnostic rows to %s", len(self._diagnostic_rows), csv_path)

    def _collect_candidate_dump_rows(
        self,
        *,
        detection_index: int,
        candidate_hosts: list[HostGalaxy],
        candidate_hosts_with_bh_mass: list[HostGalaxy],
        detection_row: pd.Series,
        galaxy_catalog: GalaxyCatalogueHandler,
        completeness: CompletenessModel,
        detection_probability_obj: SimulationDetectionProbability,
    ) -> None:
        """T2.2 (row #255 A10) read-only per-candidate diagnostic serialiser.

        Builds the per-(event, candidate) and per-event dump rows of
        B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md section 6.2, called AFTER
        ``p_Di`` returns for this event. Reads only already-computed state
        (``self.posterior_data_with_bh_mass``, ``self._phi_survival_table``,
        ``self._diagnostic_rows``) — it never writes into anything the
        likelihood consumes (GATE BI). Never raises: any failure is caught,
        logged once, and the run proceeds with whatever rows were collected
        before the failure (diagnostic-only, never affects the posterior).

        Args:
            detection_index: CRB row index of the detection (event_idx).
            candidate_hosts: The 1D (no-BH) candidate ball for this event.
            candidate_hosts_with_bh_mass: The 2D (with-BH mass filter) ball.
            detection_row: The raw CRB row (for the ``z_true`` truth column
                when present; NaN otherwise).
            galaxy_catalog: Handler used to translate the true host position
                for the ``is_true_host`` flag.
            completeness: Completeness model for ``f_bar``/``f_k``.
            detection_probability_obj: Survival-estimator object for the
                per-candidate ``S_4D(z_g, M_g; h)`` diagnostic read.
        """
        try:
            h = float(self.h)
            det = self.detection
            d_hat = float(det.d_L)
            sigma_dL = float(det.d_L_uncertainty)

            cand_by_idx: dict[int, HostGalaxy] = {
                host.catalog_index: host for host in candidate_hosts
            }
            cand_by_idx.update({host.catalog_index: host for host in candidate_hosts_with_bh_mass})

            _phi_table = getattr(self, "_phi_survival_table", {})
            if h in _phi_table:
                _z_phi_grid, _s_phi_grid = _phi_table[h]
            else:
                _z_phi_grid, _s_phi_grid = None, None

            _translated: int | None = None
            if det.host_galaxy_index >= 0:
                _translated = galaxy_catalog.resolve_host_recovery_position(det.host_galaxy_index)

            galaxy_likelihoods = self.posterior_data_with_bh_mass.get(GALAXY_LIKELIHOODS, {}).get(
                detection_index, []
            )
            additional_likelihoods = self.posterior_data_with_bh_mass.get(
                ADDITIONAL_GALAXIES_WITHOUT_BH_MASS, {}
            ).get(detection_index, [])

            def _emit(catalog_index: int, result_row: Sequence[float], batch: str) -> None:
                host = cand_by_idx.get(catalog_index)
                if host is None:
                    return
                z_g = float(host.z)
                M_g = float(host.M)
                N_g_used = float(result_row[0])
                D_g = float(result_row[1])
                w_g = float(_rate_weight(host))
                if _z_phi_grid is not None and _s_phi_grid is not None:
                    s_bar_phi_zg = float(np.interp(z_g, _z_phi_grid, _s_phi_grid))
                else:
                    s_bar_phi_zg = float("nan")
                try:
                    d_L_g = float(dist(z_g, h=h))
                    s_4d_zg_mg = float(
                        detection_probability_obj.detection_probability_with_bh_mass_interpolated(
                            d_L_g, M_g * (1.0 + z_g), 0.0, 0.0, h=h
                        )
                    )
                    u_g = (d_L_g - d_hat) / sigma_dL if sigma_dL > 0.0 else float("nan")
                except Exception:
                    s_4d_zg_mg = float("nan")
                    u_g = float("nan")
                self._candidate_dump_rows.append(
                    {
                        "event_idx": detection_index,
                        "h": h,
                        "catalog_index": int(catalog_index),
                        "batch": batch,
                        "z_g": z_g,
                        "z_err_g": float(host.z_error),
                        "M_g": M_g,
                        "M_err_g": float(host.M_error),
                        "phiS_g": float(host.phiS),
                        "qS_g": float(host.qS),
                        "w_g": w_g,
                        "N_g_used": N_g_used,
                        "D_g": D_g,
                        "s_bar_phi_zg": s_bar_phi_zg,
                        "s_4d_zg_mg": s_4d_zg_mg,
                        "u_g": u_g,
                        # Optional per section 6.2 (sky Fisher block Mahalanobis
                        # distance) -- not computed by this hook; column kept
                        # for schema stability (T2_2 schema test).
                        "sky_mahalanobis": float("nan"),
                        "is_true_host": bool(
                            _translated is not None and catalog_index == _translated
                        ),
                    }
                )

            for catalog_index, result_row in galaxy_likelihoods:
                _emit(catalog_index, result_row, "with_bh")
            for catalog_index, result_row in additional_likelihoods:
                _emit(catalog_index, result_row, "no_bh_only")

            _diag = self._diagnostic_rows[-1] if self._diagnostic_rows else {}
            z_true = float("nan")
            try:
                if "z_true" in detection_row.index:
                    _zt = detection_row["z_true"]
                    if _zt is not None and not (isinstance(_zt, float) and math.isnan(_zt)):
                        z_true = float(_zt)
            except Exception:
                z_true = float("nan")

            f_bar_z_true = float("nan")
            f_k_z_true = float("nan")
            if np.isfinite(z_true):
                try:
                    f_bar_z_true = float(completeness.f_bar(z_true, h=h))
                    _pix = completeness.ang2pix(det.phi, det.theta)
                    f_k_z_true = float(completeness.f_k(z_true, _pix, h=h))
                except Exception:
                    pass

            self._candidate_dump_event_rows.append(
                {
                    "event_idx": detection_index,
                    "h": h,
                    "d_hat": d_hat,
                    "sigma_dL": sigma_dL,
                    "z_true": z_true,
                    "host_galaxy_index": int(det.host_galaxy_index),
                    "n_cand_no_bh": len(candidate_hosts),
                    "n_cand_with_bh": len(candidate_hosts_with_bh_mass),
                    "f_bar_z_true": f_bar_z_true,
                    "f_k_z_true": f_k_z_true,
                    "L_cat_no_bh": _diag.get("L_cat_no_bh", float("nan")),
                    "B_num": _diag.get("B_num", float("nan")),
                    "D_tilde_phi": _diag.get("D_tilde_phi", float("nan")),
                }
            )
        except Exception:
            if not self._candidate_dump_warned:
                _LOGGER.warning(
                    "candidate_dump_dir instrumentation raised for detection "
                    "%d; further dump rows may be incomplete (diagnostic-only, "
                    "never affects the posterior)",
                    detection_index,
                    exc_info=True,
                )
                self._candidate_dump_warned = True

    def _write_candidate_dump_csvs(self, directory: str) -> None:
        """Write the T2.2 per-candidate / per-event dump CSVs for the current h.

        One overwrite write per h, matching the ``write_selection_table_json``
        naming convention: ``per_candidate_h_<label>.csv`` /
        ``per_event_h_<label>.csv``.

        Args:
            directory: Output directory (``candidate_dump_dir``).
        """
        h_label = str(np.round(self.h, 4)).replace(".", "_")
        os.makedirs(directory, exist_ok=True)
        if self._candidate_dump_rows:
            cand_path = os.path.join(directory, f"per_candidate_h_{h_label}.csv")
            pd.DataFrame(self._candidate_dump_rows).to_csv(cand_path, index=False)
            _LOGGER.info(
                "Wrote %d candidate-dump rows to %s",
                len(self._candidate_dump_rows),
                cand_path,
            )
        if self._candidate_dump_event_rows:
            event_path = os.path.join(directory, f"per_event_h_{h_label}.csv")
            pd.DataFrame(self._candidate_dump_event_rows).to_csv(event_path, index=False)
            _LOGGER.info(
                "Wrote %d event-dump rows to %s",
                len(self._candidate_dump_event_rows),
                event_path,
            )

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
        # Per-class Sigma ln p_i accumulators (INSTR-1): split the per-h posterior
        # contribution by host provenance (host_galaxy_index >= 0 = in-catalogue,
        # -1 = dark). Both channels are tracked so a per-class 1D-vs-2D divergence
        # is visible directly in the log instead of having to be reconstructed from
        # the per-event diagnostics CSV afterwards.
        _n_in_cat_class = 0
        _n_dark_class = 0
        # P6 host-recovery counters (INSTR-3): of the in-catalogue events, how many
        # had their TRUE host actually returned by the candidate search, per channel.
        _n_recovered_no_bh = 0
        _n_recovered_with_bh = 0
        _sum_ln_p_in_cat_no_bh = 0.0
        _sum_ln_p_in_cat_with_bh = 0.0
        _sum_ln_p_dark_no_bh = 0.0
        _sum_ln_p_dark_with_bh = 0.0
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
                sigma_multiplier=self._sky_cone_k,  # type: ignore[arg-type]
                mass_filter_sigma=self._mass_filter_sigma,
                mass_filter_geometry=self._mass_filter_geometry,
                mass_filter_k=self._mass_filter_k,
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

            # Per-class Sigma ln p_i bookkeeping (INSTR-1): read-only accumulation of
            # values already computed above. A zero-valued p_i contributes -inf, which
            # IS the true class sum -- do not clip or filter it.
            _ln_p_no_bh = math.log(event_likelihood) if event_likelihood > 0.0 else float("-inf")
            _ln_p_with_bh = (
                math.log(event_likelihood_with_bh_mass)
                if event_likelihood_with_bh_mass > 0.0
                else float("-inf")
            )
            if self.detection.host_galaxy_index >= 0:
                _n_in_cat_class += 1
                _sum_ln_p_in_cat_no_bh += _ln_p_no_bh
                _sum_ln_p_in_cat_with_bh += _ln_p_with_bh
                # P6: was the TRUE host among the candidates the production search
                # actually returned? Checked against those exact lists (never a
                # second, parallel search), so the counter can only ever agree
                # with what the estimator consumed. The translation handles the
                # observed-catalogue case where injection-time and evaluation
                # frames have pruned different rows.
                _translated = galaxy_catalog.resolve_host_recovery_position(
                    self.detection.host_galaxy_index
                )
                if _translated is not None:
                    if _translated in {host.catalog_index for host in candidate_hosts}:
                        _n_recovered_no_bh += 1
                    if _translated in {host.catalog_index for host in candidate_hosts_with_bh_mass}:
                        _n_recovered_with_bh += 1
            else:
                _n_dark_class += 1
                _sum_ln_p_dark_no_bh += _ln_p_no_bh
                _sum_ln_p_dark_with_bh += _ln_p_with_bh

            # T2.2 (row #255 A10) candidate-dump instrumentation: OFF by
            # default (self._candidate_dump_dir is None) -- this branch is
            # never entered on the production path (GATE BI).
            if self._candidate_dump_dir is not None:
                self._collect_candidate_dump_rows(
                    detection_index=int(index),
                    candidate_hosts=candidate_hosts,
                    candidate_hosts_with_bh_mass=candidate_hosts_with_bh_mass,
                    detection_row=detection,
                    galaxy_catalog=galaxy_catalog,
                    completeness=completeness,
                    detection_probability_obj=detection_probability_obj,
                )

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

        # Per-class Sigma ln p_i (INSTR-1): the two lines the 2026-07 2D-bias
        # investigation had to reconstruct by hand from ad-hoc scripts. Both
        # channels on one line per class, 7 significant figures.
        _LOGGER.info(
            "Per-class Sigma ln p_i (h=%.4f): IN-CAT (N=%d) 1D=%.7g 2D=%.7g",
            self.h,
            _n_in_cat_class,
            _sum_ln_p_in_cat_no_bh,
            _sum_ln_p_in_cat_with_bh,
        )
        _LOGGER.info(
            "Per-class Sigma ln p_i (h=%.4f): DARK (N=%d) 1D=%.7g 2D=%.7g",
            self.h,
            _n_dark_class,
            _sum_ln_p_dark_no_bh,
            _sum_ln_p_dark_with_bh,
        )

        # P6 host-recovery (INSTR-3): numerator and denominator stated explicitly so
        # the rate can never be misread. _n_in_cat_class IS the denominator -- there
        # is no second, separately-maintained "in-cat events seen" count.
        _LOGGER.info(
            "P6 host-recovery (h=%.4f): 1D %d/%d hosts recovered/in-cat events seen "
            "(%.7g%%), 2D %d/%d hosts recovered/in-cat events seen (%.7g%%)",
            self.h,
            _n_recovered_no_bh,
            _n_in_cat_class,
            (100.0 * _n_recovered_no_bh / _n_in_cat_class if _n_in_cat_class else float("nan")),
            _n_recovered_with_bh,
            _n_in_cat_class,
            (100.0 * _n_recovered_with_bh / _n_in_cat_class if _n_in_cat_class else float("nan")),
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
        # [HIER] C2 (PA-HIER-23): the denominator actually dividing this
        # event's combined likelihood — NaN until an assembly branch sets it
        # (the catalog_only bypass never does), so ln L decomposes into a
        # separable numerator-log-term and denominator-log-term at the
        # diagnostic append.
        _den_used = float("nan")
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
        # [P3-IMP] twin cell: the flag+table go to BOTH host batches — the
        # with-BH batch's r[0] no-BH numerator also feeds L_cat_no_bh (A13).
        _cat_surv = self._catalogue_numerator_survival
        if _cat_surv == "phi":
            _cat_surv_table = self._phi_survival_table[float(self.h)]
        elif _cat_surv == "phi_flat":
            # [P3-IMP] K-flat kill arm (PREREGISTRATION_P3_TWIN_20260822.md §3):
            # the catalogue consumer receives a CONSTANT table (the real table's
            # grid-mean) while the normalizer legs keep the real table object —
            # only this per-call slice is flattened, so the §6 invariants hold.
            _z_kf, _s_kf = self._phi_survival_table[float(self.h)]
            _cat_surv_table = (_z_kf, np.full_like(_s_kf, float(np.mean(_s_kf))))
            _cat_surv = "phi"  # workers see the same engaged cell semantics
        else:
            _cat_surv_table = None
        # [HIER] θ-hook site 2.2 dispatch: the batch kernel receives θ only
        # when site 2.2 is enabled by the OAT toggle (PA-HIER-23); otherwise it
        # is forced to the (0, 1) identity evaluation.
        _theta_b_22, _theta_s_22 = (
            (getattr(self, "_theta_b", 0.0), getattr(self, "_theta_s", 1.0))
            if getattr(self, "_theta_sites", "all") in ("all", "2.2")
            else (0.0, 1.0)
        )
        results_with_bh_mass = _starmap_host_batches(
            pool,
            possible_host_galaxies_with_bh_mass,
            detection_index,
            self.h,
            True,
            self._normalization_mode,
            self._host_z_kernel,
            self._host_mass_kernel,
            self._catalogue_mass_overlap,
            self._catalogue_mass_error_scale,
            self._eddington_m,
            _cat_surv,
            _cat_surv_table,
            self._catalogue_numerator_survival_2d,
            self._catalogue_numerator_survival_2d_center,
            theta_b=_theta_b_22,
            theta_s=_theta_s_22,
            catalogue_leg_1d_mass_aware=self._catalogue_leg_1d_mass_aware,
            sigma4d_mass_kernel=self._sigma4d_mass_kernel,
        )

        results_without_blackhole_mass = _starmap_host_batches(
            pool,
            possible_host_galaxies_reduced,
            detection_index,
            self.h,
            False,
            self._normalization_mode,
            self._host_z_kernel,
            self._host_mass_kernel,
            self._catalogue_mass_overlap,
            self._catalogue_mass_error_scale,
            self._eddington_m,
            _cat_surv,
            _cat_surv_table,
            self._catalogue_numerator_survival_2d,
            self._catalogue_numerator_survival_2d_center,
            theta_b=_theta_b_22,
            theta_s=_theta_s_22,
            catalogue_leg_1d_mass_aware=self._catalogue_leg_1d_mass_aware,
            sigma4d_mass_kernel=self._sigma4d_mass_kernel,
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
            B_num_wbh = 0.0
            g_frac_used = float("nan")
            L_comp = 0.0
            w_G_legacy = 1.0
            alpha_G_phi = float("nan")
            r_Malm = float("nan")
            D_tilde_phi = float("nan")
        else:
            D_h: float = self._D_h_table.get(self.h, 0.0)
            beta_G: float = self._beta_G_table.get(self.h, 0.0)
            beta_Gbar: float = self._beta_Gbar_table.get(self.h, 0.0)
            # [P3-RPHI] the fourth Path-A slot (docs/derivations/
            # PROPOSAL_SIGMA_PHI_DIVISOR_20260822.md §2/§6(ii)): "phi" swaps
            # the no-BH catalogue divisor Sigma^3D for Sigma^phi (the SAME
            # table Path A already builds for the weight chain, :3878). The
            # with-BH leg (global_denom_with_bh) is deliberately untouched.
            # [HIER] site 2.3phi (PHYSICS_CHANGE_THETA_DIVISOR_20260830.md
            # §2.1-2.2, row #255 tree 2 node T1.1): when the theta-consistent
            # divisor is armed AND engaged (theta != (0,1)),
            # _global_cat_selection_phi_theta[h] holds the registered
            # Sigma_phi_reg(theta;h) = Sigma_phi_point(h) * rho(theta;h); its
            # ``.get`` falls through to the stored point-table VALUE itself
            # (identity of object, no floating operation) whenever the dict
            # has no entry for this h -- the "off" default and the GATE T-ID
            # literal skip at theta=(0,1) are both this same fallback.
            global_denom_with_bh: float = self._global_cat_denom_with_bh.get(self.h, 0.0)
            # [HIER T2.3] mass-aware 1D catalogue leg instrument
            # (PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md §2.2 site D1,
            # row #255 tree 2 node T2.3): "on" (guarded at setup) replaces
            # the no-BH catalogue divisor by Sigma_4D (global_denom_with_bh,
            # ALREADY IN HAND, no new computation) -- the SAME divisor
            # Sigma_4D's own with-BH branch already computes. "off"
            # (default): byte-identical to the pre-flag ternary below.
            global_denom_no_bh: float = (
                global_denom_with_bh
                if getattr(self, "_catalogue_leg_1d_mass_aware", "off") == "on"
                else (
                    getattr(self, "_global_cat_selection_phi_theta", {}).get(
                        self.h, self._global_cat_selection_phi.get(self.h, 0.0)
                    )
                    if getattr(self, "_catalogue_global_selection", "s3d") == "phi"
                    else self._global_cat_denom_no_bh.get(self.h, 0.0)
                )
            )
            # generator_marginal draw-side calibration (0.0 outside that mode).
            n_hat_w: float = 0.0
            # Path-(A) diagnostics (NaN outside the phi-convention branch); the
            # legacy w_G = beta_G/D is RENAMED, never overwritten.
            w_G_legacy = beta_G / D_h if D_h > 0.0 else float("nan")
            alpha_G_phi = float("nan")
            r_Malm = float("nan")
            D_tilde_phi = float("nan")

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
            FIXED_QUAD_N = _HOST_QUAD_N
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
            # B-DEN falsifier instrument precondition (docs/derivations/
            # completion_numerator_data_measure.md §6): the 'data' event-measure
            # mapping loc=d_L, scale=sigma_frac*d_L is the delta-method image of
            # d_L/d_hat ~ N(mu_frac, sigma_frac) ONLY at mu_frac == 1 exactly.
            # mu_frac is hardcoded to 1 in _means_3d's construction (~:3992) --
            # verified once per event here, not assumed, so a future change to
            # that construction cannot silently corrupt the 'data' path.
            if getattr(self, "_completion_event_measure", "ratio") == "data":
                assert _comp_mean_dLfrac == 1.0, (
                    "completion_event_measure='data' assumes _comp_mean_dLfrac == 1.0 "
                    f"(the ratio's mean is hardcoded to 1 in _means_3d); got "
                    f"{_comp_mean_dLfrac!r} -- the data-measure mapping needs re-deriving."
                )
            # Change 5.3: the completion numerator weights the incompleteness at the
            # EVENT's sky pixel, (1 - f_{k(Omega_e)}(z)). p_GW delta-collapses the sky
            # integral, so f is evaluated at the single pixel containing the detection
            # direction (ecliptic phi/theta). Gray-Messenger-Veitch 2022,
            # arXiv:2111.04629, Eq. (5) (out-of-catalog branch). Computed once per
            # event; Omega-independent completeness gives the identical Task-A B_num.
            _event_pixel = completeness.ang2pix(self.detection.phi, self.detection.theta)

            def completion_numerator_integrand(
                z: npt.NDArray[np.float64],
                h_eval: float,
            ) -> npt.NDArray[np.float64]:
                d_L: npt.NDArray[np.float64] = np.asarray(
                    dist_vectorized(z, h=h_eval), dtype=np.float64
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
                # [PHYSICS] B-DEN falsifier instrument (default OFF, byte-identical
                # 'ratio' path): docs/derivations/completion_numerator_data_measure.md
                # §2, §6; MFG (2019) arXiv:1809.02063 Eqs. (5)-(7). The 'ratio' form
                # below is a density in the dimensionless distance ratio d_L/d_L,det,
                # integrated against dz — NOT a density in the observable, so
                # integral dd_L,det p_gw != 1 (it is proportional to d_L(z;h), see the
                # memo §2). '_comp_mean_dLfrac' is always exactly 1.0 (hardcoded in
                # _means_3d construction, bayesian_statistics.py ~:3992: the ratio's
                # mean is centred at unity, no bias/centering offset to carry over,
                # verified once per event above), so the 'data' form's mean maps to
                # d_L(z;h) exactly, not d_L(z;h)/_comp_mean_dLfrac -- verified, not
                # assumed (memo §6 New).
                _event_measure = getattr(self, "_completion_event_measure", "ratio")
                if _event_measure == "data":
                    p_gw: npt.NDArray[np.float64] = (
                        norm.pdf(
                            _comp_det_d_L,
                            loc=d_L,
                            scale=_comp_sigma_dLfrac * d_L,
                        )
                        * np.sin(self.detection.theta)
                        / (4.0 * np.pi)
                    )
                else:
                    p_gw = (
                        norm.pdf(d_L_fraction, loc=_comp_mean_dLfrac, scale=_comp_sigma_dLfrac)
                        * np.sin(self.detection.theta)
                        / (4.0 * np.pi)
                    )
                dVc: npt.NDArray[np.float64] = np.atleast_1d(
                    np.asarray(comoving_volume_element(z, h=h_eval), dtype=np.float64)
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
                        completeness.f_k(z, _event_pixel, h_eval),
                        dtype=np.float64,
                    ),
                    0.0,
                    1.0,
                )
                return (1.0 - f_z) * p_gw * dVc / (1.0 + z)

            # [PHYSICS] the [P2] leg of the selection fusion (rows #117-#118;
            # production default under absolute_marginal via the 'fused' cell,
            # promoted from the N-2 '1d' instrumentation branch, T3'). The
            # 1D channel discards M_z^obs, so under a latent-thresholded
            # detection model the MFG numerator's P(det|theta) survives the
            # z-quadrature as the phi-marginal survival S_bar_phi(z;h):
            #     B_num^{1d} = INTEGRAL (1-f_k) p_gw dVc/(1+z) S_bar_phi(z;h) dz
            # (derivation draft N2_SELECTION_NUMERATOR_DERIVATION_20260805 (T3')).
            # S_bar_phi is READ, never rebuilt: the same table
            # precompute_phi_marginal_survival produced for beta^phi/D~^phi, via
            # the same np.interp accessor precompute_global_catalog_selection
            # uses for Sigma^phi. Outside the table's [1e-6, z_max(h)] domain
            # np.interp clamps to the endpoints, matching that accessor exactly.
            # [PHYSICS] rows #117-#118: 'fused' (the absolute_marginal
            # production cell) = [P1]+[P2] paired; '1d'/'2d' are the item-4
            # counterfactual decomposition cells; 'off' (and legacy instances
            # without the attribute) is the pre-#118 estimator.
            _sel_cell = getattr(self, "_selection_in_completion_numerator", "off")
            _sel_1d = _sel_cell in ("1d", "fused")
            _sel_2d = _sel_cell in ("2d", "fused")

            def completion_numerator_integrand_sel_1d(
                z: npt.NDArray[np.float64],
                h_eval: float,
            ) -> npt.NDArray[np.float64]:
                base = completion_numerator_integrand(z, h_eval)
                _table = getattr(self, "_phi_survival_table", {})
                if h_eval not in _table:
                    raise ValueError(
                        "selection_in_completion_numerator='1d': no S_bar_phi table "
                        f"entry for h={h_eval!r} (tabulated: {sorted(_table)!r})"
                    )
                _z_phi_grid, _s_phi_grid = _table[h_eval]
                s_bar_phi: npt.NDArray[np.float64] = np.interp(
                    np.asarray(z, dtype=np.float64), _z_phi_grid, _s_phi_grid
                )
                return base * s_bar_phi

            # [PHYSICS] (N8) the 2D completion leg's own numerator
            # (GATE_PACKAGE_FINAL.md §2.2, FIXB_PATHA_PACKAGE.md §3.2):
            #     B_num_wbh = INTEGRAL (1-f_k) p_gw dVc/(1+z) g_i(z;h) dz
            # with g_i INSIDE the quadrature — mu_cond(z) and the (1+z) mass
            # lift both depend on z, so the factor is not separable. The 1D
            # B_num stays unmultiplied (gate (iv)).
            _use_g_inside = bool(getattr(self, "_use_phi_selection", False))
            _g_slot = self._det_index_to_slot[detection_index]
            _g_proj = (
                float(getattr(self, "_proj_d_L_to_M", np.zeros(1))[_g_slot])
                if _use_g_inside
                else 0.0
            )
            _g_sigma = (
                float(getattr(self, "_sigma_cond_M", np.zeros(1))[_g_slot])
                if _use_g_inside
                else 0.0
            )
            _g_det_M_z = float(self.detection.M)

            def completion_numerator_integrand_with_bh_mass(
                z: npt.NDArray[np.float64],
                h_eval: float,
            ) -> npt.NDArray[np.float64]:
                base = completion_numerator_integrand(z, h_eval)
                d_L_mass: npt.NDArray[np.float64] = np.asarray(
                    dist_vectorized(z, h=h_eval), dtype=np.float64
                )
                z_arr = np.asarray(z, dtype=np.float64)

                def _s_query(
                    dl_q: npt.NDArray[np.float64],
                    m_z_q: npt.NDArray[np.float64],
                    z_q: npt.NDArray[np.float64],
                ) -> npt.NDArray[np.float64]:
                    # The SAME with-BH survival query S_bar_phi is built from
                    # (precompute_phi_marginal_survival): detector-frame mass,
                    # absolute node d_L, isotropic sky, _wbh_z_kwargs rider.
                    _zeros = np.zeros(dl_q.size, dtype=np.float64)
                    return np.asarray(
                        detection_probability_obj.detection_probability_with_bh_mass_interpolated(
                            dl_q,
                            m_z_q,
                            _zeros,
                            _zeros,
                            h=h_eval,
                            **_wbh_z_kwargs(detection_probability_obj, z_q),
                        ),
                        dtype=np.float64,
                    )

                # g_i / g_sel,prod is a density in x_M = M_z/M_z,det,i — the
                # SAME measure as the 2D catalogue leg's mz_integral, so the
                # two legs are addable and the 2D MAP stays invariant under a
                # rescaling of that measure (gate (i)). Under the fused cell
                # ([PHYSICS] rows #117-#118, MFG 2019 arXiv:1809.02063 Eqs.
                # (5)-(7)) the detection survival S_4D sits INSIDE the same
                # dx_M — the selected-prior form of the latent-thresholded
                # detection model (L6-DER3 §3).
                if _sel_2d:
                    g_i = completion_mass_factor_g_sel(
                        z_arr,
                        d_L_mass,
                        d_L_mass / _comp_det_d_L,
                        _g_det_M_z,
                        _g_proj,
                        _g_sigma,
                        s_query=_s_query,
                    )
                else:
                    g_i = completion_mass_factor_g(
                        z_arr,
                        d_L_mass / _comp_det_d_L,
                        _g_det_M_z,
                        _g_proj,
                        _g_sigma,
                    )
                _support_exit = (g_i <= 0.0) & (base > 0.0)
                _n_support_exit = int(np.count_nonzero(_support_exit))
                if _n_support_exit:
                    if _sel_2d:
                        # MINOR-2 (row #118): distinguish beyond-horizon zeros
                        # (S_4D = 0) from genuine phi-support exits — the two
                        # have opposite physical readings.
                        _mu_bad = 1.0 + _g_proj * (d_L_mass[_support_exit] / _comp_det_d_L - 1.0)
                        _m_bad = np.where(_mu_bad > 0.0, _mu_bad, 1.0) * _g_det_M_z
                        _s_bad = _s_query(d_L_mass[_support_exit], _m_bad, z_arr[_support_exit]) * (
                            _mu_bad > 0.0
                        )
                        _n_horizon = int(np.count_nonzero(_s_bad <= 0.0))
                        _LOGGER.warning(
                            "Detection %d: fused completion mass factor zero at "
                            "%d/%d quadrature nodes (%d beyond the detection "
                            "horizon, S_4D=0; %d off the phi support).",
                            detection_index,
                            _n_support_exit,
                            g_i.size,
                            _n_horizon,
                            _n_support_exit - _n_horizon,
                        )
                    else:
                        _LOGGER.warning(
                            "Detection %d: g_i left the phi support (%d/%d quadrature "
                            "nodes with zero mass density) — the completion leg's "
                            "population mass prior has no weight there.",
                            detection_index,
                            _n_support_exit,
                            g_i.size,
                        )
                return base * g_i

            def _completion_numerators(h_eval: float) -> tuple[float, float]:
                """Completion-leg numerators ``(B_num, B_num_wbh)`` at ``h_eval``.

                Factored out of the inline body (2026-08-04) so the SAME
                quadrature can be re-evaluated at a reference Hubble value for
                the frozen-g_frac counterfactual (``--freeze_g_frac_ref_h``).
                Called with ``self.h`` it reproduces the pre-factorisation
                expressions verbatim — the default path is byte-identical.
                """
                z_upper = dist_to_redshift(
                    self.detection.d_L
                    + integration_limit_sigma_multiplier * self.detection.d_L_uncertainty,
                    h=h_eval,
                )
                z_lower = dist_to_redshift(
                    self.detection.d_L
                    - integration_limit_sigma_multiplier * self.detection.d_L_uncertainty,
                    h=h_eval,
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

                if z_lower >= z_upper:
                    # The event's entire 4-sigma window lies beyond the analysis depth
                    # (redshift_upper_limit): no population support survives the cap, so
                    # the completion numerator vanishes rather than integrating an
                    # inverted [z_lower, z_upper] interval (which would return a
                    # negative fixed_quad result, not 0).
                    return 0.0, 0.0
                if _sel_1d:
                    # [PHYSICS] S_bar_phi-weighted 1D numerator — the PRODUCTION
                    # absolute_marginal path since rows #117-#118 ('fused'; also
                    # the '1d' decomposition cell). The `else` below is the
                    # pre-#118 expression, kept verbatim for the 'off'/'2d'
                    # counterfactual cells and every other normalization mode.
                    b_num = float(
                        fixed_quad(
                            lambda z: completion_numerator_integrand_sel_1d(z, h_eval),
                            z_lower,
                            z_upper,
                            n=FIXED_QUAD_N,
                        )[0]
                    )
                else:
                    b_num = float(
                        fixed_quad(
                            lambda z: completion_numerator_integrand(z, h_eval),
                            z_lower,
                            z_upper,
                            n=FIXED_QUAD_N,
                        )[0]
                    )
                b_num_wbh = (
                    float(
                        fixed_quad(
                            lambda z: completion_numerator_integrand_with_bh_mass(z, h_eval),
                            z_lower,
                            z_upper,
                            n=FIXED_QUAD_N,
                        )[0]
                    )
                    if _use_g_inside
                    else b_num
                )
                return b_num, b_num_wbh

            B_num, B_num_wbh = _completion_numerators(self.h)
            # g_frac = B_num_wbh/B_num is the completion leg's mass-consistency
            # factor; unfrozen it carries its own h-dependence through the whole
            # quadrature. The diagnostics column below records the value ACTUALLY
            # used, so a frozen run is self-describing.
            g_frac_used = (B_num_wbh / B_num) if B_num > 0.0 else float("nan")
            # getattr: p_Di is exercised by tests that build the instance with
            # ``object.__new__`` (no __init__), matching the surrounding style.
            _freeze_ref_h: float | None = getattr(self, "_freeze_g_frac_ref_h", None)
            if _freeze_ref_h is not None:
                # INSTRUMENTATION ONLY (gate (vii) follow-up, default OFF — no
                # computed value changes when the flag is None; nothing inside
                # this branch runs on the production path). Pins each event's
                # g_frac to its OWN value at the reference h, so the 2D
                # completion leg becomes B_num(h) * g_ref(h_ref): the h-slope of
                # the mass factor is removed while every other h-dependence
                # (catalogue leg, w~_G, B_num itself, the 1D channel) still
                # moves. Counterfactual diagnostic, not a physics change.
                _b_ref, _b_wbh_ref = _completion_numerators(float(_freeze_ref_h))
                if _b_ref > 0.0:
                    g_frac_used = _b_wbh_ref / _b_ref
                    B_num_wbh = B_num * g_frac_used
                else:
                    _LOGGER.warning(
                        "Detection %s: frozen-g reference B_num(h_ref=%.6g) is zero — "
                        "B_num_wbh left unfrozen for this event",
                        detection_index,
                        _freeze_ref_h,
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
                _den_used = D_gen if D_gen > 0 else 1.0
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
            elif _use_g_inside and self.h in getattr(self, "_beta_G_phi_table", {}):
                # [PHYSICS] Path (A): ONE detection model, mass-consistent
                # mixture (FIXB_PATHA_PACKAGE.md §3.2, 2026-08-04):
                #   1D: p_i = (beta_G^phi L_cat^1D + B_num^phi) / D~^phi
                #   2D: p_i = (alpha_G^phi L_cat^2D + B_num^phi g_i) / D~^phi
                # with B_num^phi = B_num (derived form, docs/derivations/
                # bscale_completion_normalization.md §6, ledger rows
                # #130-#131: the completion leg is already a p_pop-measure
                # integral as constructed, so no transfer factor exists; the
                # legacy beta_Gbar^phi/beta_Gbar multiplier was a defect,
                # kept only under --completion_b_scale legacy) and
                # alpha_G^phi = beta_G^phi r_Malm the mass-aware
                # in-catalogue selection.
                # The tower identity S_3D = INTEGRAL phi S_4D dM now holds by
                # construction (r_phi == 1) and r_Malm is a pure Malmquist
                # ratio. Mandel, Farr & Gair (2019), arXiv:1809.02063,
                # Eqs. (5)-(7) (assumption A2, hybrid population density of
                # GATE_PACKAGE_FINAL.md Appendix A); Turski et al. (2023),
                # arXiv:2302.12037, Eq. (8); Gray et al. (2020),
                # arXiv:1908.06050, Eq. (A.19).
                beta_G_phi = self._beta_G_phi_table[self.h]
                beta_Gbar_phi = self._beta_Gbar_phi_table[self.h]
                sigma_phi = self._global_cat_selection_phi.get(self.h, 0.0)
                path_a = path_a_mixture_objects(
                    beta_G_phi, beta_Gbar_phi, sigma_phi, global_denom_with_bh
                )
                alpha_G_phi = path_a["alpha_G_phi"]
                D_tilde_phi = path_a["D_tilde_phi"]
                r_Malm = path_a["r_Malm"]
                # Derived form B_num^phi = B_num — docs/derivations/
                # bscale_completion_normalization.md §6; MFG (2019)
                # arXiv:1809.02063 Eqs. (5)-(7). Legacy
                # beta_Gbar^phi/beta_Gbar multiplication was un-derived
                # (defect, ledger row #131); kept only under
                # --completion_b_scale legacy for historical-run reproduction.
                B_num_phi, B_num_wbh_phi, _B_scale = path_a_completion_numerators(
                    B_num,
                    B_num_wbh,
                    beta_Gbar_phi,
                    beta_Gbar,
                    mode=getattr(self, "_completion_b_scale", "derived"),
                )
                _den_used = D_tilde_phi if D_tilde_phi > 0.0 else 1.0
                if D_tilde_phi > 0.0:
                    w_G = path_a["w_tilde_G"]
                    # [HIER T2.3] mass-aware 1D catalogue leg instrument
                    # (PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md §2.2/§2.3
                    # site W1, row #255 tree 2 node T2.3): "on" (guarded at
                    # setup) replaces beta_G_phi by alpha_G_phi -- the
                    # IDENTICAL float the 2D assembly below already consumes
                    # (:6501) -- the no-mass-likelihood image of the 2D
                    # mixture. "off": byte-identical (beta_G_phi, unchanged).
                    _cat_num_weight_no_bh = (
                        alpha_G_phi
                        if getattr(self, "_catalogue_leg_1d_mass_aware", "off") == "on"
                        else beta_G_phi
                    )
                    combined_without_bh_mass = float(
                        (_cat_num_weight_no_bh * L_cat_without_bh_mass + B_num_phi) / D_tilde_phi
                    )
                    combined_with_bh_mass = float(
                        (alpha_G_phi * L_cat_with_bh_mass + B_num_wbh_phi) / D_tilde_phi
                    )
                else:
                    _LOGGER.warning(
                        "Detection %s: D~^phi(h) is zero, using L_cat only", detection_index
                    )
                    w_G = 1.0
                    combined_without_bh_mass = float(L_cat_without_bh_mass)
                    combined_with_bh_mass = float(L_cat_with_bh_mass)
            elif D_h > 0:
                w_G = beta_G / D_h
                _den_used = D_h
                combined_without_bh_mass = float((beta_G * L_cat_without_bh_mass + B_num) / D_h)
                combined_with_bh_mass = float((beta_G * L_cat_with_bh_mass + B_num) / D_h)
            else:
                _LOGGER.warning(f"Detection {detection_index}: D(h) is zero, using L_cat only")
                w_G = 1.0
                _den_used = 1.0
                combined_without_bh_mass = float(L_cat_without_bh_mass)
                combined_with_bh_mass = float(L_cat_with_bh_mass)
            # Diagnostic-only completion likelihood L_comp = B_num/beta_Gbar (the
            # single ratio never divides by beta_Gbar, which -> 0 as f -> 1).
            L_comp = float(B_num / beta_Gbar) if beta_Gbar > 0 else 0.0

        _LOGGER.debug(
            f"Detection {detection_index}: w_G={w_G:.7g} (w_G_legacy={w_G_legacy:.7g}), "
            f"L_cat_no_bh={L_cat_without_bh_mass:.6e}, "
            f"L_cat_with_bh={L_cat_with_bh_mass:.6e}, B_num={B_num:.6e}, "
            f"B_num_wbh={B_num_wbh:.6e}, L_comp={L_comp:.6e}"
        )

        # Record diagnostic row for every event. The path-(A) columns
        # (w_tilde_G, alpha_G_phi, r_Malm, D_tilde_phi, B_num_wbh, g_frac) ship
        # WITH the change and are written at 7 significant figures — the
        # h-dependence of the partition weight is a ~1e-5 effect
        # (FIXB_PATHA_PACKAGE.md §5 instrumentation).
        self._diagnostic_rows.append(
            {
                "event_idx": detection_index,
                "h": self.h,
                "w_G": w_G,
                "w_G_legacy": w_G_legacy,
                "w_tilde_G": w_G if getattr(self, "_use_phi_selection", False) else float("nan"),
                "alpha_G_phi": alpha_G_phi,
                "r_Malm": r_Malm,
                "D_tilde_phi": D_tilde_phi,
                "L_cat_no_bh": L_cat_without_bh_mass,
                "L_cat_with_bh": L_cat_with_bh_mass,
                "B_num": B_num,
                "B_num_wbh": B_num_wbh,
                "g_frac": g_frac_used,
                "L_comp": L_comp,
                "combined_no_bh": combined_without_bh_mass,
                "combined_with_bh": combined_with_bh_mass,
                # [HIER] C2 (PA-HIER-23): separable per-event ln L decomposition
                # ln L = num_log_term − den_log_term. The numerator is recovered
                # as combined × den_used (exactly what the assembly divided);
                # NaN where the term is non-positive or no denominator applied.
                "den_log_term": math.log(_den_used)
                if np.isfinite(_den_used) and _den_used > 0.0
                else float("nan"),
                "num_log_term_no_bh": math.log(combined_without_bh_mass * _den_used)
                if np.isfinite(_den_used) and combined_without_bh_mass * _den_used > 0.0
                else float("nan"),
                "num_log_term_with_bh": math.log(combined_with_bh_mass * _den_used)
                if np.isfinite(_den_used) and combined_with_bh_mass * _den_used > 0.0
                else float("nan"),
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
    d_L = dist_vectorized(z_arr, h=h)
    if getattr(detection_probability, "wbh_z_resolved", False) is True:
        # [PHYSICS] FIX-3 §7.1 erf-sum path (fix3_zmz_catalog_selection.md
        # §3.3-C convention 2, choice (a)): the joint-grid knot values at
        # (d_L(z_j), z_j, m_k), lifted to M_z knots 10^{m_k}, with the
        # interpolant treated as PIECEWISE-LINEAR IN M_z between them — the
        # closed-form erf-sum below stays exact for that interpolant.
        m_centers, p = detection_probability.wbh_joint_knot_values(
            np.asarray(d_L, dtype=np.float64), z_arr
        )
        m_centers = np.asarray(m_centers, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)  # (n_z, K)
        n_k = m_centers.size
    else:
        interp_2d, _ = detection_probability._get_or_build_grid(h)
        m_centers = np.asarray(interp_2d.grid[1], dtype=np.float64)  # M_z grid knots
        n_k = m_centers.size

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
    d_L = dist_vectorized(z.reshape(-1), h=h)  # (n*n_z,)
    if getattr(detection_probability, "wbh_z_resolved", False) is True:
        # [PHYSICS] FIX-3 §7.1 erf-sum path — identical convention to the
        # scalar twin (bit-parity): joint-grid knot values per (host, z-node),
        # piecewise-linear in M_z between the lifted knots (§3.3-C choice (a)).
        m_centers, p_flat = detection_probability.wbh_joint_knot_values(
            np.asarray(d_L, dtype=np.float64), z.reshape(-1)
        )
        m_centers = np.asarray(m_centers, dtype=np.float64)
        n_k = m_centers.size
        p = np.asarray(p_flat, dtype=np.float64).reshape(n, n_z, n_k)
    else:
        interp_2d, _ = detection_probability._get_or_build_grid(h)
        m_centers = np.asarray(interp_2d.grid[1], dtype=np.float64)  # M_z grid knots
        n_k = m_centers.size

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


def _sigma4d_mass_kernel_expectation(
    z_g: npt.NDArray[np.float64],
    M_eff_g: npt.NDArray[np.float64],
    sigma_g: npt.NDArray[np.float64],
    phi_g: npt.NDArray[np.float64],
    theta_g: npt.NDArray[np.float64],
    h: float,
    detection_probability_obj: Any,
    chunk_size: int = 200_000,
) -> npt.NDArray[np.float64]:
    r"""Instrument J registered kernel (results/prod2d_closure_20260818/
    PREREGISTRATION_TILT_BATTERY.md §1, P2): per-galaxy

    .. math::

        p_{\det,g} = \mathbb{E}_{M \sim \mathcal{N}(M_{\mathrm{eff},g},\,
        \sigma_g^2)}\bigl[S_{4D}(d_L(z_g),\, M(1+z_g))\bigr],

    reusing the erf-sum inner-M closed form of
    :func:`_bh_mass_denominator_inner_m_integral_batch` (the SAME exact
    quadrature against the piecewise-linear ``p_det`` interpolant that
    production's own ``D_g`` uses -- a single ``z``-node per galaxy). ``sigma_g
    -> 0`` collapses the Gaussian kernel to a delta function and recovers the
    point evaluation ``S_4D(d_L(z_g), M_eff_g(1+z_g))`` exactly (pinned
    limiting-case test). NO R_eff/truncated-lognormal mass prior enters this
    kernel (unlike ``_mass_trunc``): ``w_g`` stays the point rate weight
    outside this function, so ``R_eff`` is counted exactly once and cancels in
    ``r_Malm`` (registered, P2).

    Chunked at ``chunk_size`` rows (catalogue-scale call site: tens of
    millions of eligible galaxies), mirroring
    :func:`_smeared_global_pdet_expectation`.

    Args:
        z_g: Eligible galaxy redshifts, shape ``(n,)``.
        M_eff_g: Eddington-shifted (or raw, per ``--eddington_m``) mean host
            masses [M_sun], shape ``(n,)``.
        sigma_g: Catalogue ``BH_MASS_ERROR`` 1-sigma mass uncertainties
            [M_sun], shape ``(n,)``.
        phi_g: Ecliptic azimuths (isotropic zeros on the production path),
            shape ``(n,)``.
        theta_g: Ecliptic polar angles (isotropic zeros), shape ``(n,)``.
        h: Dimensionless Hubble parameter.
        detection_probability_obj: ``SimulationDetectionProbability`` instance.
        chunk_size: Galaxies per chunk.

    Returns:
        Per-galaxy ``p_det,g``, shape ``(n,)``.
    """
    n = int(z_g.size)
    out = np.empty(n, dtype=np.float64)
    # Numerical floor only (BH_MASS_ERROR is always > 0 in the production
    # catalogue); avoids a division by zero for a synthetic sigma_g == 0 test
    # input while still recovering the point evaluation in that limit.
    sigma_floor = np.maximum(sigma_g, 1e-8)
    for start in range(0, n, chunk_size):
        sl = slice(start, min(start + chunk_size, n))
        z_col = z_g[sl][:, None]  # (chunk, 1): a single quadrature node per galaxy
        val = _bh_mass_denominator_inner_m_integral_batch(
            z_col,
            detection_probability_obj,
            phi_g[sl],
            theta_g[sl],
            M_eff_g[sl],
            sigma_floor[sl],
            h,
        )
        out[sl] = val[:, 0]
    return out


def catalogue_leg_1d_mass_aware_factor(
    z: float | npt.NDArray[np.float64],
    M_g: float | npt.NDArray[np.float64],
    M_g_error: float | npt.NDArray[np.float64],
    h: float,
    sigma4d_mass_kernel: str,
    eddington_m: str,
    detection_probability_obj: Any,
) -> npt.NDArray[np.float64]:
    r"""catalogue_leg_1d_mass_aware="on" site N1 factor (row #255 tree 2 node
    T2.3, PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md §2.2/§2.3): replaces the
    population-average catalogue-numerator survival ``S_bar_phi(z;h)`` by the
    SAME per-galaxy with-BH survival Sigma_4D already evaluates for that
    galaxy --

    .. math::

        S_{4D}\bigl(d_L(z;h),\, M_g(1+z)\bigr)

    (``sigma4d_mass_kernel="point"``) or its Gaussian-mass-kernel expectation

    .. math::

        \mathbb{E}_{M \sim \mathcal{N}(M_{\mathrm{eff},g},\,\sigma_g^2)}
        \bigl[S_{4D}(d_L(z;h),\, M(1+z))\bigr]

    (``sigma4d_mass_kernel="kernel"``, REPORTED-ONLY -- no registered F3
    prediction uses this branch). This mirrors ``self._sigma4d_mass_kernel``
    exactly, reusing the SAME accessor and SAME isotropic-sky convention as
    Sigma_4D's own with-BH branch (``precompute_global_catalog_selection``,
    point query :3022-3038 / kernel query :2996-3020) and the T2.2 hook's
    ``s_4d_zg_mg`` column -- the registered coupling rule (§2.2): the
    numerator's mass measure can never differ from the divisor's (Sigma_4D),
    so an unpaired point/kernel combination is exactly the
    [NUMERATOR-ONLY-CLEAN] defect class.

    ``z``, ``M_g``, ``M_g_error`` must already share one broadcast shape (the
    caller's z-node axis) -- callers pass e.g. ``host_M[:, None]`` against
    ``y_num_nodes`` of shape ``(n, k)``, or a bare scalar/1-D array in the
    scalar kernel.

    Args:
        z: Redshift nodes, any shape.
        M_g: Raw catalogue BH masses [M_sun] (HostGalaxy.M), broadcastable to
            ``z``'s shape.
        M_g_error: Catalogue ``BH_MASS_ERROR`` [M_sun], broadcastable to
            ``z``'s shape (read only when ``sigma4d_mass_kernel="kernel"``).
        h: Dimensionless Hubble parameter.
        sigma4d_mass_kernel: "point" or "kernel" -- mirrors
            ``self._sigma4d_mass_kernel`` (Sigma_4D's own convention); no
            independent knob (§2.2's registered coupling rule).
        eddington_m: "on"/"off" -- mirrors ``self._eddington_m`` (read only
            under "kernel").
        detection_probability_obj: ``SimulationDetectionProbability`` instance
            (the SAME accessor Sigma_4D and the T2.2 hook query).

    Returns:
        S_4D per node, same shape as ``z``.

    References:
        Mandel, Farr & Gair (2019), arXiv:1809.02063, Eqs. (5)-(7)
            (assumption A2: the selection integral must use the SAME
            detection model as the numerator, for a catalogue galaxy of
            KNOWN mass).
    """
    z_arr = np.asarray(z, dtype=np.float64)
    shape = z_arr.shape
    z_flat = z_arr.reshape(-1)
    M_flat = np.broadcast_to(np.asarray(M_g, dtype=np.float64), shape).reshape(-1).copy()
    zeros = np.zeros_like(z_flat)
    if sigma4d_mass_kernel == "point":
        d_L_flat = np.asarray(dist_vectorized(z_flat, h=h), dtype=np.float64)
        M_z_flat = M_flat * (1.0 + z_flat)
        out = np.asarray(
            detection_probability_obj.detection_probability_with_bh_mass_interpolated(
                d_L_flat,
                M_z_flat,
                zeros,
                zeros,
                h=h,
                **_wbh_z_kwargs(detection_probability_obj, z_flat),
            ),
            dtype=np.float64,
        )
    elif sigma4d_mass_kernel == "kernel":
        Merr_flat = np.broadcast_to(np.asarray(M_g_error, dtype=np.float64), shape).reshape(-1)
        M_eff_flat = (
            _eddington_shifted_host_mass_batch(M_flat, Merr_flat) if eddington_m == "on" else M_flat
        )
        out = _sigma4d_mass_kernel_expectation(
            z_flat, M_eff_flat, Merr_flat, zeros, zeros, h, detection_probability_obj
        )
    else:
        raise ValueError(
            f"sigma4d_mass_kernel must be 'point' or 'kernel', got {sigma4d_mass_kernel!r}"
        )
    return out.reshape(shape)


def _mz_sel_2d_expectation(
    mu_star: npt.NDArray[np.float64],
    sigma_star: npt.NDArray[np.float64],
    z_nodes: npt.NDArray[np.float64],
    d_L_nodes: npt.NDArray[np.float64],
    det_M: float,
    detection_probability: Any,
    host_phiS: float,
    host_qS: float,
    h: float,
) -> npt.NDArray[np.float64]:
    r"""[P3-2D] scalar host: ``E[S_4D(d_L(z;h), x*M_z,det)]`` under the
    product-Gaussian ``x ~ N(mu_star, sigma_star^2)``.

    PREREGISTRATION_P3_2D_20260825.md §1/§2: the with-BH catalogue numerator's
    mass integrand gains the survival INSIDE the candidate's own mass
    quadrature, ``mz_sel = INTEGRAL N(x;mu_cond,sigma_cond) p_gal(x;z)
    S_4D(d_L(z;h), x*M_z,det) dx``. The product of the two Gaussians factors
    (completing the square) as

    .. math::

        \mathcal N(x;\mu_\mathrm{cond},\sigma_\mathrm{cond}^2)\,
        \mathcal N(x;\mu_\mathrm{gal},\sigma_\mathrm{gal}^2)
        = \mathcal N(\mu_\mathrm{cond};\mu_\mathrm{gal},
                     \sigma_\mathrm{cond}^2+\sigma_\mathrm{gal}^2)\,
          \mathcal N(x;\mu_\star,\sigma_\star^2),

    .. math::

        \mu_\star = \frac{\mu_\mathrm{cond}\sigma_\mathrm{gal}^2 +
                     \mu_\mathrm{gal}\sigma_\mathrm{cond}^2}
                    {\sigma_\mathrm{cond}^2+\sigma_\mathrm{gal}^2},
        \qquad
        \sigma_\star^2 = \frac{\sigma_\mathrm{cond}^2\sigma_\mathrm{gal}^2}
                         {\sigma_\mathrm{cond}^2+\sigma_\mathrm{gal}^2},

    so ``mz_sel = mz_integral * E[S_4D]`` where ``mz_integral`` is exactly the
    analytic Gaussian-product prefactor already computed at the call site
    (Eq. 14.31 in derivations/dark_siren_likelihood.md) and ``E[S_4D]`` is
    this function's return value. Gauss-Hermite quadrature (the SAME
    ``_MT_GH_NODES``/``_MT_GH_WEIGHTS`` the ``mass_trunc`` kernel already uses,
    :func:`_mass_trunc_mz_integral`) centred on ``mu_star`` resolves ``S_4D``
    over the product-Gaussian's own narrow support -- no aliasing, no new
    table (consumes the EXISTING
    ``detection_probability_with_bh_mass_interpolated`` accessor). Sharp-GW-
    mass limit (``sigma_star -> 0``): ``E[S_4D] -> S_4D(d_L, mu_cond*M_z,det)``
    -- the stage-0 §1 registered limit.

    Args:
        mu_star: Product-Gaussian mean (mass-fraction coordinate), ``(K,)``.
        sigma_star: Product-Gaussian std (mass-fraction coordinate), ``(K,)``.
        z_nodes: The z quadrature nodes ``mu_star`` was built at, ``(K,)``
            (threaded to the accessor only when it is z-resolved,
            :func:`_wbh_z_kwargs`).
        d_L_nodes: ``d_L(z_nodes; h)``, ``(K,)``.
        det_M: The detection's own detector-frame mass ``M_z,det`` [M_sun].
        detection_probability: ``SimulationDetectionProbability`` instance.
        host_phiS: Host ecliptic azimuth.
        host_qS: Host ecliptic polar angle.
        h: Dimensionless Hubble parameter.

    Returns:
        ``E[S_4D]``, shape ``(K,)``.
    """
    n_g = _MT_GH_NODES.size
    a_nodes = mu_star[:, None] + np.sqrt(2.0) * sigma_star[:, None] * _MT_GH_NODES  # (K, G)
    m_z_flat = (a_nodes * det_M).reshape(-1)
    d_L_flat = np.repeat(np.asarray(d_L_nodes, dtype=np.float64), n_g)
    z_flat = np.repeat(np.asarray(z_nodes, dtype=np.float64), n_g)
    S_4D = np.asarray(
        detection_probability.detection_probability_with_bh_mass_interpolated(
            d_L_flat,
            m_z_flat,
            np.full(d_L_flat.size, host_phiS),
            np.full(d_L_flat.size, host_qS),
            h=h,
            **_wbh_z_kwargs(detection_probability, z_flat),
        ),
        dtype=np.float64,
    ).reshape(mu_star.shape[0], n_g)
    expectation: npt.NDArray[np.float64] = (S_4D @ _MT_GH_WEIGHTS) / np.sqrt(np.pi)
    return expectation


def _mz_sel_2d_expectation_batch(
    mu_star: npt.NDArray[np.float64],
    sigma_star: npt.NDArray[np.float64],
    z_nodes: npt.NDArray[np.float64],
    d_L_nodes: npt.NDArray[np.float64],
    det_M: float,
    detection_probability: Any,
    host_phiS: npt.NDArray[np.float64],
    host_qS: npt.NDArray[np.float64],
    h: float,
) -> npt.NDArray[np.float64]:
    """Host-batched twin of :func:`_mz_sel_2d_expectation`.

    ``mu_star``/``sigma_star``/``z_nodes``/``d_L_nodes`` have shape ``(n, K)``;
    ``host_phiS``/``host_qS`` have shape ``(n,)``. Row ``i`` is bit-identical
    to the scalar function called with host ``i``'s arrays -- the arithmetic
    per ``(host, z-node, GH-node)`` element is unchanged; only the host axis
    is added, in ONE ``detection_probability_with_bh_mass_interpolated`` call.
    """
    n, k = mu_star.shape
    n_g = _MT_GH_NODES.size
    a_nodes = mu_star[..., None] + np.sqrt(2.0) * sigma_star[..., None] * _MT_GH_NODES  # (n,k,G)
    m_z_flat = (a_nodes * det_M).reshape(-1)
    d_L_flat = np.repeat(d_L_nodes.reshape(-1), n_g)
    z_flat = np.repeat(z_nodes.reshape(-1), n_g)
    phi_flat = np.repeat(host_phiS, k * n_g)
    theta_flat = np.repeat(host_qS, k * n_g)
    S_4D = np.asarray(
        detection_probability.detection_probability_with_bh_mass_interpolated(
            d_L_flat,
            m_z_flat,
            phi_flat,
            theta_flat,
            h=h,
            **_wbh_z_kwargs(detection_probability, z_flat),
        ),
        dtype=np.float64,
    ).reshape(n, k, n_g)
    expectation: npt.NDArray[np.float64] = (S_4D @ _MT_GH_WEIGHTS) / np.sqrt(np.pi)
    return expectation


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
    # #40 remainder (RATIFIED 2026-07-27): 2D host-mass kernel decomposition
    # flag; "auto" == the historical bundling (trunc_lognormal iff
    # mass_trunc). No value change on the default path.
    host_mass_kernel: str = "auto",
    # Prod2d closure counterfactual instrument (results/prod2d_closure_20260818/
    # PREREGISTRATION_PROD_COUNTERFACTUAL.md §1). "production" (default) is
    # byte-identical to the pre-flag path.
    catalogue_mass_overlap: str = "production",
    catalogue_mass_error_scale: float = 1.0,
    # Instrument E (results/prod2d_closure_20260818/
    # PREREGISTRATION_TILT_BATTERY.md §1). "on" (default) is byte-identical to
    # the pre-flag path; "off" assigns the raw (unshifted) host_M to
    # _host_M_eff, switching the numerator mass prior AND the per-host D_g
    # erf-sum together (the single assignment below).
    eddington_m: str = "on",
    # [P3-IMP] twin cell (PREREGISTRATION_P3_TWIN_20260822.md §2); scalar twin
    # of the batch flag — same semantics, same table-slice input.
    catalogue_numerator_survival: str = "off",
    catalogue_survival_table: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
    # [P3-2D] the with-BH catalogue-leg twin: 2D bounded identity test (stage
    # 2) (results/campaign51_20260728/realistic_20260729/
    # PREREGISTRATION_P3_2D_20260825.md §2(i)); scalar twin of the batch flag.
    # "off" (default) is byte-identical to the pre-flag path; "mz_sel"
    # multiplies the WITH-BH catalogue numerator's mass integrand by
    # S_4D(d_L(z;h), x*M_z,det) inside the candidate's own mass quadrature
    # (see _mz_sel_2d_expectation). The WITHOUT-BH numerator is untouched.
    catalogue_numerator_survival_2d: str = "off",
    # Centering sub-option ("raw"=host_M, "eff"=host_M_eff) for the
    # product-Gaussian mean fed to the S_4D quadrature. REFUSED ("unset",
    # the default) until explicitly set when the twin is engaged -- the
    # choice is PENDING the pre-execution review (prereg §2(i)); no silent
    # default.
    catalogue_numerator_survival_2d_center: str = "unset",
    # [HIER] θ-hook site 2.1 (PHYSICS_CHANGE_THETA_HOOK_20260828.md, ledger row
    # #216): affine photo-z systematic reparametrization z̃ = z + b(1+z),
    # σ̃_eff = s·σ_eff. (0.0, 1.0) is the literal-skip identity (GATE T-ID) —
    # the default path never executes the θ branch.
    theta_b: float = 0.0,
    theta_s: float = 1.0,
    # [HIER T2.3] mass-aware 1D catalogue leg instrument (row #255 tree 2
    # node T2.3, PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md §2); scalar
    # twin of the batch flag — same semantics.
    catalogue_leg_1d_mass_aware: str = "off",
    sigma4d_mass_kernel: str = "point",
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
    global proj_d_L_to_M_arr, sigma_cond_M_arr
    global det_d_L_arr, det_d_L_unc_arr, det_M_arr, det_phi_arr, det_theta_arr
    global completeness_model

    if eddington_m not in ("on", "off"):
        raise ValueError(f"eddington_m must be 'on' or 'off', got {eddington_m!r}")

    # [P3-IMP] twin cell (PREREGISTRATION_P3_TWIN_20260822.md §2/§4).
    if catalogue_numerator_survival not in ("off", "phi"):
        raise ValueError(
            "catalogue_numerator_survival must be 'off' or 'phi', got "
            f"{catalogue_numerator_survival!r}"
        )
    _cat_surv_on = catalogue_numerator_survival == "phi"
    if _cat_surv_on:
        if catalogue_survival_table is None:
            raise ValueError("catalogue_numerator_survival='phi' requires catalogue_survival_table")
        _p3_engagement_log_once("scalar")

    # [HIER T2.3] mass-aware 1D catalogue leg instrument (row #255 tree 2
    # node T2.3, PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md §2); scalar
    # twin of the batch validation.
    if catalogue_leg_1d_mass_aware not in ("off", "on"):
        raise ValueError(
            "catalogue_leg_1d_mass_aware must be 'off' or 'on', got "
            f"{catalogue_leg_1d_mass_aware!r}"
        )
    _cat_leg_1d_ma_on = catalogue_leg_1d_mass_aware == "on"
    if _cat_leg_1d_ma_on and not _cat_surv_on:
        raise ValueError(
            "catalogue_leg_1d_mass_aware='on' requires "
            "catalogue_numerator_survival='phi' (site N1 replaces the phi "
            "per-candidate survival)"
        )

    # [P3-2D] the with-BH catalogue-leg twin (PREREGISTRATION_P3_2D_20260825.md §2(i)).
    if catalogue_numerator_survival_2d not in ("off", "mz_sel"):
        raise ValueError(
            "catalogue_numerator_survival_2d must be 'off' or 'mz_sel', "
            f"got {catalogue_numerator_survival_2d!r}"
        )
    _cat_surv_2d_on = catalogue_numerator_survival_2d == "mz_sel"
    if _cat_surv_2d_on and catalogue_numerator_survival_2d_center not in ("raw", "eff"):
        raise ValueError(
            "catalogue_numerator_survival_2d='mz_sel' requires "
            "catalogue_numerator_survival_2d_center to be explicitly 'raw' or "
            "'eff' (no silent default -- the centering choice is PENDING the "
            "pre-execution review, PREREGISTRATION_P3_2D_20260825.md §2(i)); "
            f"got {catalogue_numerator_survival_2d_center!r}"
        )

    FIXED_QUAD_N = _HOST_QUAD_N

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

    # [PHYSICS] truncated lognormal x R_eff host-mass prior in the 2D channel
    # (EXP-45; RATIFIED 2026-07-27, docs/derivations/mass_marginal_2d_kernel.md
    # gates M1-M5: lognormal family from the Reines & Volonteri log-space fit,
    # truncated + renormalized on ParameterSpace.M, GH numerator with the
    # small-sigma crossover, GL-in-lnM denominator, counted-once-in-M). See
    # module-level _MASS_TRUNC_* + _mass_trunc_* helpers. Differs ONLY in the
    # with-BH-mass mass-marginal (numerator + selection denominator); no
    # effect without BH mass. Selectable independently of the normalization
    # leg via host_mass_kernel ("auto" == the historical mass_trunc bundling);
    # the point-z x trunc-mass combination raises (prior-consistency guard).
    _use_mass_trunc = (
        resolve_host_mass_kernel(host_mass_kernel, normalization_mode, host_z_kernel)
        == "trunc_lognormal"
    )
    # [P3-2D] the twin only composes with the production Gaussian-product
    # with-BH mass-marginal branch (mz_integral's "else" below) -- guard
    # pattern, not a silent no-op, when combined with an incompatible
    # instrument (PREREGISTRATION_P3_2D_20260825.md §2(i)).
    if _cat_surv_2d_on and (_use_mass_trunc or catalogue_mass_overlap != "production"):
        raise ValueError(
            "catalogue_numerator_survival_2d='mz_sel' composes only with the "
            "production Gaussian-product with-BH mass-marginal branch; got "
            f"host_mass_kernel resolving to mass_trunc={_use_mass_trunc!r}, "
            f"catalogue_mass_overlap={catalogue_mass_overlap!r}"
        )

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
    if theta_b != 0.0 or theta_s != 1.0:
        # [HIER] θ-hook site 2.1 — Sec. 2 in Ma, Hu & Huterer (2006),
        # arXiv:astro-ph/0506614. HIER §1.2 s-placement (row #221 item 4;
        # 2026-08-29 note in PHYSICS_CHANGE_THETA_HOOK_20260828.md supersedes
        # the 2026-08-28 "s scales the folded width" pin): s scales the RAW
        # host_z_error BEFORE the PV quadrature fold; b is unchanged — it
        # still shifts the kernel centre AFTER the fold, using sigma_z_pv
        # computed above from the UNSHIFTED host_z.
        _validate_theta(theta_b, theta_s)
        _theta_hook_count("site_2_1")
        host_z_error_eff = float(np.sqrt((theta_s * host_z_error) ** 2 + sigma_z_pv**2))
        host_z = host_z + theta_b * (1.0 + host_z)

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
    # [PHYSICS] C7-core (scalar twin of the batched kernel): the in-catalogue
    # host-z prior's population weight is the CATALOGUED-host intensity
    # f_{k(g)}(z) * w_pop(z). f is evaluated at the HOST's HEALPix pixel with the
    # SAME completeness callable B_num and beta_Gbar use, and enters the
    # numerator AND Z_g (hence D_g), so rho_g stays a unit-mass density in z and
    # p_det stays out of the numerator. ZoA: if f_k == 0 across the whole host
    # window the kernel falls back to the pre-C7 w_pop-only form (no clamp).
    # Theorem (T) + partition argument, GATE_PACKAGE_FINAL.md §1.2 (2026-08-04);
    # structure: Gray et al. (2020) arXiv:1908.06050 Eq. (A.10); Turski et al.
    # (2023) arXiv:2302.12037 Eq. (4)
    _completeness = completeness_model
    _host_pixel: int | None = None
    if _use_volume_deconv and _completeness is not None:
        _host_pixel = _completeness.ang2pix(host_phiS, host_qS)
        _f_probe = _completeness_at_host_nodes(
            _completeness,
            _batched_gl_nodes(
                np.array([denominator_integration_lower_redshift_limit]),
                np.array([denominator_integration_upper_redshift_limit]),
                _GL_NODES_50,
            ),
            np.array([_host_pixel], dtype=np.int64),
            h,
        )
        if not bool(np.any(_f_probe > 0.0)):
            _host_pixel = None
            _warn_zoa_hostz_kernel_fallback(detection_index, 1)

    def _w_pop_eff(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """``w_pop(z) = dV_c/dz / (1 + z)``, times ``f_{k(g)}(z)`` (C7-core)."""
        w_pop = np.asarray(comoving_volume_element(z, h=h), dtype=np.float64) / (1.0 + z)
        if _host_pixel is not None and _completeness is not None:
            f_k = np.clip(
                np.asarray(_completeness.f_k(z, _host_pixel, h), dtype=np.float64), 0.0, 1.0
            )
            return w_pop * f_k
        return w_pop

    _z_prior_norm = 1.0
    if _use_volume_deconv:

        def _z_prior_unnorm(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            w_pop = _w_pop_eff(z)
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
            return base * _w_pop_eff(z) / _z_prior_norm
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
        _num = _mvn_pdf(
            np.vstack([phi, theta, luminosity_distance_fraction]).T,
            _mean_3d,
            _cov_inv_3d,
            _log_norm_3d,
        ) * galaxy_redshift_prior_pdf(z)
        if _cat_surv_on:
            if _cat_leg_1d_ma_on:
                # [HIER T2.3] site N1, quadrature path (scalar twin):
                # S_4D(d_L(z;h), M_g(1+z)) at each quadrature node.
                _num = _num * catalogue_leg_1d_mass_aware_factor(
                    np.asarray(z, dtype=np.float64),
                    host_M,
                    host_M_error,
                    h,
                    sigma4d_mass_kernel,
                    eddington_m,
                    detection_probability,
                )
            else:
                # [P3-IMP] twin cell: per-host S_bar_phi factor (endpoint-clamped
                # np.interp — the completion_numerator_integrand_sel_1d convention),
                # from the SAME table the mixture normalizer beta_G_phi integrates.
                assert catalogue_survival_table is not None
                _z_s, _s_phi = catalogue_survival_table
                _num = _num * np.interp(np.asarray(z, dtype=np.float64), _z_s, _s_phi)
        return _num

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
        if _cat_surv_on:
            if _cat_leg_1d_ma_on:
                # [HIER T2.3] site N1, point path (scalar twin):
                # S_4D(d_L(z;h), M_g(1+z)) — the T2.2 hook's s_4d_zg_mg
                # column exactly.
                single_host_likelihood_numerator_without_bh_mass *= float(
                    catalogue_leg_1d_mass_aware_factor(
                        np.array([host_z], dtype=np.float64),
                        host_M,
                        host_M_error,
                        h,
                        sigma4d_mass_kernel,
                        eddington_m,
                        detection_probability,
                    )[0]
                )
            else:
                # [P3-IMP] twin cell, delta-kernel branch: point-evaluated factor.
                assert catalogue_survival_table is not None
                _z_s, _s_phi = catalogue_survival_table
                single_host_likelihood_numerator_without_bh_mass *= float(
                    np.interp(host_z, _z_s, _s_phi)
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
        _warn_quadrature_weight_outside_grid(
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
        # Impact re-measured by the E instrument (--eddington_m {on,off}; results/
        # prod2d_closure_20260818/PREREGISTRATION_TILT_BATTERY.md §1 R-E), which
        # replaces the stale -0.020 empirical-impact figure (audit finding 5) with
        # a direct s_Edd,new = mean_h(baseline) - mean_h(E-off) measurement at the
        # current operating point; the pre-D_g-fix expectation anchor is a
        # -0.0022-class shift (docs/gates/G7row9_N5). "off" is a guard pattern
        # (the shift is simply never computed), so "on" stays bit-identical.
        _host_M_eff = (
            host_M
            if eddington_m == "off"
            else (
                eddington_shifted_host_mass(host_M, host_M_error)
                if (_use_volume_deconv and not _use_mass_trunc)
                else host_M
            )
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
            elif catalogue_mass_overlap != "production":
                # [PHYSICS] prod2d closure counterfactual instrument (results/
                # prod2d_closure_20260818/PREREGISTRATION_PROD_COUNTERFACTUAL.md
                # §1). Guard pattern: the production float stream (the final
                # ``else`` branch below) is reached ONLY when
                # catalogue_mass_overlap == "production" and is never touched
                # here.
                mu_gal_frac = _host_M_eff * (1 + z) / _det_M
                if catalogue_mass_overlap == "neutralized":
                    # V1': replace the candidate's own mz_integral with the
                    # SAME population mass factor the completion leg uses
                    # (completion_mass_factor_g, :2022) at the candidate's own
                    # z-nodes, with the event's det_M_z / projection /
                    # sigma_cond -- the candidate becomes mass-UNINFORMATIVE.
                    # Both legs remain densities in the same x_M measure
                    # (registered normalization coherence statement, §1 V1').
                    mz_integral = completion_mass_factor_g(
                        np.asarray(z, dtype=np.float64),
                        np.asarray(luminosity_distance_fraction, dtype=np.float64),
                        _det_M,
                        float(proj_d_L_to_M_arr[slot]),
                        float(sigma_cond_M_arr[slot]),
                    )
                elif catalogue_mass_overlap == "inflated":
                    # V2: host_M_error -> k*host_M_error ONLY in the numerator
                    # width sigma_gal; the Eddington-shifted mean _host_M_eff
                    # (above) stays computed from the UNSCALED host_M_error
                    # (frozen mu_eff, §1 V2).
                    sigma_gal_frac = (host_M_error * catalogue_mass_error_scale) * (1 + z) / _det_M
                    sigma2_sum = _sigma2_cond + sigma_gal_frac**2
                    mz_integral = np.exp(
                        -0.5 * (mu_cond - mu_gal_frac) ** 2 / sigma2_sum
                    ) / np.sqrt(2 * np.pi * sigma2_sum)
                else:
                    raise ValueError(
                        "catalogue_mass_overlap must be 'production', "
                        f"'neutralized' or 'inflated', got {catalogue_mass_overlap!r}"
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

                if _cat_surv_2d_on:
                    # [P3-2D] the with-BH catalogue-leg twin: S_4D inside the
                    # candidate's own mass quadrature (product-Gaussian
                    # identity, see _mz_sel_2d_expectation). "raw"/"eff"
                    # centering only changes the product-Gaussian MEAN fed to
                    # the S_4D quadrature; sigma_gal_frac (the measurement
                    # width) is unchanged either way.
                    _mu_gal_surv = (
                        (host_M if catalogue_numerator_survival_2d_center == "raw" else _host_M_eff)
                        * (1 + z)
                        / _det_M
                    )
                    _mu_star = (
                        mu_cond * sigma_gal_frac**2 + _mu_gal_surv * _sigma2_cond
                    ) / sigma2_sum
                    _sigma_star = np.sqrt(_sigma2_cond * sigma_gal_frac**2 / sigma2_sum)
                    mz_integral = mz_integral * _mz_sel_2d_expectation(
                        _mu_star,
                        _sigma_star,
                        np.asarray(z, dtype=np.float64),
                        np.asarray(d_L, dtype=np.float64),
                        _det_M,
                        detection_probability,
                        host_phiS,
                        host_qS,
                        h,
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
            if _cat_surv_2d_on:
                # [P3-2D] delta-kernel branch: same product-Gaussian factor,
                # evaluated at z = z_g (see the quadrature branch above).
                _mu_gal_surv_point = (
                    (host_M if catalogue_numerator_survival_2d_center == "raw" else _host_M_eff)
                    * (1 + _z_point)
                    / _det_M
                )
                _mu_star_point = (
                    _mu_cond_point * _sigma_gal_frac_point**2 + _mu_gal_surv_point * _sigma2_cond
                ) / _sigma2_sum_point
                _sigma_star_point = np.sqrt(
                    _sigma2_cond * _sigma_gal_frac_point**2 / _sigma2_sum_point
                )
                _mz_point = _mz_point * _mz_sel_2d_expectation(
                    _mu_star_point,
                    _sigma_star_point,
                    _z_point,
                    _d_L_point,
                    _det_M,
                    detection_probability,
                    host_phiS,
                    host_qS,
                    h,
                )
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
    # #40 remainder: 2D host-mass kernel decomposition flag ("auto" == the
    # historical mass_trunc bundling; see resolve_host_mass_kernel).
    host_mass_kernel: str = "auto",
    # Prod2d closure counterfactual instrument (results/prod2d_closure_20260818/
    # PREREGISTRATION_PROD_COUNTERFACTUAL.md §1). "production" (default) is
    # byte-identical to the pre-flag path.
    catalogue_mass_overlap: str = "production",
    catalogue_mass_error_scale: float = 1.0,
    # Instrument E (results/prod2d_closure_20260818/
    # PREREGISTRATION_TILT_BATTERY.md §1). "on" (default) is byte-identical to
    # the pre-flag path; "off" assigns the raw (unshifted) host_M to
    # host_M_eff.
    eddington_m: str = "on",
    # [P3-IMP] twin cell (PREREGISTRATION_P3_TWIN_20260822.md §2). "off"
    # (default) is byte-identical to the pre-flag path; "phi" multiplies the
    # WITHOUT-BH numerator integrand by S_bar_phi(z;h) from
    # catalogue_survival_table (endpoint-clamped np.interp — the same
    # convention as completion_numerator_integrand_sel_1d).
    catalogue_numerator_survival: str = "off",
    catalogue_survival_table: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
    # [P3-2D] the with-BH catalogue-leg twin: 2D bounded identity test (stage
    # 2) (results/campaign51_20260728/realistic_20260729/
    # PREREGISTRATION_P3_2D_20260825.md §2(i)); production dispatch path. "off"
    # (default) is byte-identical to the pre-flag path; "mz_sel" multiplies
    # the WITH-BH catalogue numerator's mass integrand by S_4D(d_L(z;h),
    # x*M_z,det) inside the candidate's own mass quadrature (see
    # _mz_sel_2d_expectation_batch). The WITHOUT-BH numerator is untouched.
    catalogue_numerator_survival_2d: str = "off",
    # Centering sub-option ("raw"=host_M, "eff"=host_M_eff): REFUSED
    # ("unset", the default) until explicitly set when the twin is engaged --
    # the choice is PENDING the pre-execution review (prereg §2(i)); no
    # silent default.
    catalogue_numerator_survival_2d_center: str = "unset",
    # [HIER] θ-hook site 2.2 — production's actual dispatch path
    # (PHYSICS_CHANGE_THETA_HOOK_20260828.md, ledger row #216). (0.0, 1.0) is
    # the literal-skip identity (GATE T-ID); semantics identical to the scalar
    # twin's site 2.1.
    theta_b: float = 0.0,
    theta_s: float = 1.0,
    # [HIER T2.3] mass-aware 1D catalogue leg instrument (row #255 tree 2
    # node T2.3, PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md §2). "off"
    # (default) is byte-identical to the pre-flag path; "on" replaces site
    # N1's per-candidate S_bar_phi factor by S_4D(d_L(z;h), M_g(1+z))
    # (mirroring sigma4d_mass_kernel). Requires catalogue_numerator_survival
    # to resolve to "phi" (validated by the caller, evaluate()'s setup guard).
    catalogue_leg_1d_mass_aware: str = "off",
    sigma4d_mass_kernel: str = "point",
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
        host_mass_kernel: 2D host-mass kernel selection (#40 remainder);
            ``"auto"`` reproduces the historical mass_trunc bundling.
        catalogue_mass_overlap: Prod2d closure counterfactual instrument
            ("production"/"neutralized"/"inflated"); see
            results/prod2d_closure_20260818/
            PREREGISTRATION_PROD_COUNTERFACTUAL.md §1.
        catalogue_mass_error_scale: Width multiplier ``k`` for "inflated".
        catalogue_numerator_survival_2d: [P3-2D] twin cell ("off"/"mz_sel");
            PREREGISTRATION_P3_2D_20260825.md §2(i).
        catalogue_numerator_survival_2d_center: Centering sub-option
            ("unset"/"raw"/"eff"); REQUIRED to be "raw" or "eff" when the
            twin cell is "mz_sel".

    Returns:
        Array of shape ``(n, 6)`` when ``evaluate_with_bh_mass`` else
        ``(n, 4)``; columns match the scalar kernel's return list.
    """
    global detection_probability
    global means_3d, cov_inv_3d, log_norm_3d
    global means_4d
    global det_index_to_slot
    global sigma2_cond_arr, proj_arr
    global proj_d_L_to_M_arr, sigma_cond_M_arr
    global det_d_L_arr, det_d_L_unc_arr, det_M_arr
    global completeness_model

    if eddington_m not in ("on", "off"):
        raise ValueError(f"eddington_m must be 'on' or 'off', got {eddington_m!r}")

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
    if theta_b != 0.0 or theta_s != 1.0:
        # [HIER] θ-hook site 2.2 — Sec. 2 in Ma, Hu & Huterer (2006),
        # arXiv:astro-ph/0506614. HIER §1.2 s-placement (row #221 item 4;
        # 2026-08-29 note supersedes the 2026-08-28 "s on the folded width"
        # pin): s scales the RAW host_z_error BEFORE the PV fold; b is
        # unchanged (still AFTER the fold, using sigma_z_pv from the
        # UNSHIFTED host_z) — same registered form as the scalar twin (2.1).
        _validate_theta(theta_b, theta_s)
        _theta_hook_count("site_2_2")
        host_z_error_eff = np.sqrt((theta_s * host_z_error) ** 2 + sigma_z_pv**2)
        host_z = host_z + theta_b * (1.0 + host_z)

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
    # Truncated lognormal x R_eff host-mass prior in the 2D channel (EXP-45;
    # RATIFIED 2026-07-27, docs/derivations/mass_marginal_2d_kernel.md).
    # Selectable independently via host_mass_kernel (see scalar path).
    _use_mass_trunc = (
        resolve_host_mass_kernel(host_mass_kernel, normalization_mode, host_z_kernel)
        == "trunc_lognormal"
    )
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

    # [PHYSICS] C7-core: the in-catalogue host-z kernel's population prior is the
    # CATALOGUED-host intensity f_{k(g)}(z) * w_pop(z), not w_pop(z) alone — the
    # discrete catalogue sum then reconstructs the population intensity
    # (E[sum_g w_g rho_g] = lambda) and the two legs of the mixture partition the
    # z-density (catalogue f*w_pop, completion (1-f)*w_pop). f enters the
    # numerator AND Z_g (and hence D_g), so rho_g stays a unit-mass density in z;
    # p_det stays OUT of the numerator (normalisation slot only).
    # Theorem (T) + partition argument, GATE_PACKAGE_FINAL.md §1.2 (2026-08-04);
    # structure: Gray et al. (2020) arXiv:1908.06050 Eq. (A.10); Turski et al.
    # (2023) arXiv:2302.12037 Eq. (4)
    f_host_den: npt.NDArray[np.float64] | None = None
    host_pixels: npt.NDArray[np.int64] | None = None
    zoa_rows: npt.NDArray[np.bool_] | None = None
    if _use_volume_deconv and completeness_model is not None:
        host_pixels = _host_pixels(completeness_model, host_phiS, host_qS)
        f_host_den = _completeness_at_host_nodes(completeness_model, y_den, host_pixels, h)
        # ZoA branch (B5): per-pixel all-zero window -> pre-C7 kernel for those
        # hosts. Never an elementwise clamp (that installs a kink mid-window).
        zoa_rows = ~np.any(f_host_den > 0.0, axis=1)
        if bool(np.any(zoa_rows)):
            f_host_den[zoa_rows, :] = 1.0
            _warn_zoa_hostz_kernel_fallback(detection_index, int(np.count_nonzero(zoa_rows)))

    z_prior_norm = np.ones(n, dtype=np.float64)
    w_pop_den: npt.NDArray[np.float64] | None = None
    if _use_volume_deconv:
        y_den_flat = y_den.reshape(-1)
        w_pop_den = (
            np.asarray(comoving_volume_element(y_den_flat, h=h), dtype=np.float64)
            / (1.0 + y_den_flat)
        ).reshape(n, _HOST_QUAD_N)
        if f_host_den is not None:
            w_pop_den = w_pop_den * f_host_den
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
            # Numerator nodes == denominator nodes -> reuse the denominator w_pop
            # (which already carries f_{k(g)} on those very nodes).
            w_pop_num = w_pop_den
        else:
            w_pop_num_1d = np.asarray(comoving_volume_element(y_num_1d, h=h), dtype=np.float64) / (
                1.0 + y_num_1d
            )
            w_pop_num = np.broadcast_to(w_pop_num_1d[None, :], (n, _HOST_QUAD_N))  # (n, 50)
            # [PHYSICS] C7-core: same f_{k(g)}(z) factor as Z_g, at the numerator
            # nodes (the event-level window, shared across hosts of this batch).
            # Theorem (T) + partition argument, GATE_PACKAGE_FINAL.md §1.2
            # (2026-08-04); structure: Gray et al. (2020) arXiv:1908.06050
            # Eq. (A.10); Turski et al. (2023) arXiv:2302.12037 Eq. (4)
            if host_pixels is not None:
                f_host_num = _completeness_at_host_nodes(
                    completeness_model,  # type: ignore[arg-type]
                    y_num_nodes,
                    host_pixels,
                    h,
                )
                if zoa_rows is not None:
                    f_host_num[zoa_rows, :] = 1.0
                w_pop_num = w_pop_num * f_host_num

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

    # [P3-IMP] twin cell (PREREGISTRATION_P3_TWIN_20260822.md §2): per-host
    # S_bar_phi factor in the WITHOUT-BH numerator only. GATE E-P3 engagement
    # log fires once per worker process per path (A13 dispatch evidence).
    if catalogue_numerator_survival not in ("off", "phi"):
        raise ValueError(
            "catalogue_numerator_survival must be 'off' or 'phi', got "
            f"{catalogue_numerator_survival!r}"
        )
    # NOTE (A13): applied for BOTH evaluate_with_bh_mass values — the with-BH
    # host batch's r[0] is ALSO a no-BH numerator that feeds L_cat_no_bh (the
    # all_results_without_bh concatenation in the caller), so gating on the
    # channel flag would silently engage the cell on a host subset only.
    _cat_surv_on = catalogue_numerator_survival == "phi"
    if _cat_surv_on:
        if catalogue_survival_table is None:
            raise ValueError("catalogue_numerator_survival='phi' requires catalogue_survival_table")
        _p3_engagement_log_once("batch")

    # [HIER T2.3] mass-aware 1D catalogue leg instrument (row #255 tree 2
    # node T2.3, PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md §2). Defence in
    # depth: evaluate() already validates this at setup; re-validated here
    # the same way catalogue_numerator_survival_2d is below.
    if catalogue_leg_1d_mass_aware not in ("off", "on"):
        raise ValueError(
            "catalogue_leg_1d_mass_aware must be 'off' or 'on', got "
            f"{catalogue_leg_1d_mass_aware!r}"
        )
    _cat_leg_1d_ma_on = catalogue_leg_1d_mass_aware == "on"
    if _cat_leg_1d_ma_on and not _cat_surv_on:
        raise ValueError(
            "catalogue_leg_1d_mass_aware='on' requires "
            "catalogue_numerator_survival='phi' (site N1 replaces the phi "
            "per-candidate survival)"
        )

    # [P3-2D] the with-BH catalogue-leg twin (PREREGISTRATION_P3_2D_20260825.md §2(i)).
    if catalogue_numerator_survival_2d not in ("off", "mz_sel"):
        raise ValueError(
            "catalogue_numerator_survival_2d must be 'off' or 'mz_sel', "
            f"got {catalogue_numerator_survival_2d!r}"
        )
    _cat_surv_2d_on = catalogue_numerator_survival_2d == "mz_sel"
    if _cat_surv_2d_on and catalogue_numerator_survival_2d_center not in ("raw", "eff"):
        raise ValueError(
            "catalogue_numerator_survival_2d='mz_sel' requires "
            "catalogue_numerator_survival_2d_center to be explicitly 'raw' or "
            "'eff' (no silent default -- the centering choice is PENDING the "
            "pre-execution review, PREREGISTRATION_P3_2D_20260825.md §2(i)); "
            f"got {catalogue_numerator_survival_2d_center!r}"
        )
    # The twin only composes with the production Gaussian-product with-BH
    # mass-marginal branch (mz_integral's "else" below) -- guard pattern.
    if _cat_surv_2d_on and (_use_mass_trunc or catalogue_mass_overlap != "production"):
        raise ValueError(
            "catalogue_numerator_survival_2d='mz_sel' composes only with the "
            "production Gaussian-product with-BH mass-marginal branch; got "
            f"host_mass_kernel resolving to mass_trunc={_use_mass_trunc!r}, "
            f"catalogue_mass_overlap={catalogue_mass_overlap!r}"
        )

    if _use_generator_point:
        # [PHYSICS] N_g = p(x | z_g, Omega_g): point value, no reduce.
        # DERIVATION_GENERATOR_CONSISTENT_NORM.md §4.3.
        numerator_without_bh_mass = gw_3d[:, 0]
        if _cat_surv_on:
            if _cat_leg_1d_ma_on:
                # [HIER T2.3] site N1, point path: S_4D(d_L(z;h), M_g(1+z))
                # in place of S_bar_phi(z) -- the T2.2 hook's s_4d_zg_mg
                # column exactly.
                _surv_factor = catalogue_leg_1d_mass_aware_factor(
                    host_z,
                    host_M,
                    host_M_error,
                    h,
                    sigma4d_mass_kernel,
                    eddington_m,
                    detection_probability,
                )
            else:
                assert catalogue_survival_table is not None
                _z_s, _s_phi = catalogue_survival_table
                _surv_factor = np.interp(host_z, _z_s, _s_phi)
            numerator_without_bh_mass = numerator_without_bh_mass * _surv_factor
    else:
        assert prior_num is not None
        _num_integrand = gw_3d * prior_num
        if _cat_surv_on:
            if _cat_leg_1d_ma_on:
                # [HIER T2.3] site N1, quadrature path: S_4D evaluated at
                # each node's own (z, M_g(1+z)) -- the 2D twin's own
                # per-node evaluation (:8280-8290 below) mirrored exactly.
                _surv_factor = catalogue_leg_1d_mass_aware_factor(
                    y_num_nodes,
                    host_M[:, None],
                    host_M_error[:, None],
                    h,
                    sigma4d_mass_kernel,
                    eddington_m,
                    detection_probability,
                )
            else:
                assert catalogue_survival_table is not None
                _z_s, _s_phi = catalogue_survival_table
                _surv_factor = np.interp(y_num_nodes, _z_s, _s_phi)
            _num_integrand = _num_integrand * _surv_factor
        numerator_without_bh_mass = _batched_gl_reduce(
            num_reduce_lo,
            num_reduce_hi,
            _GL_WEIGHTS_50,
            _num_integrand,
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
        _warn_quadrature_weight_outside_grid(
            detection_index,
            float(quadrature_weight_outside_grid_numerator[_flagged]),
            float(quadrature_weight_outside_grid_denominator[_flagged]),
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
    # Instrument E (--eddington_m {on,off}; PREREGISTRATION_TILT_BATTERY.md §1):
    # "off" is a guard pattern -- host_M_eff is assigned raw host_M directly and
    # eddington_shifted_host_mass is never called, so "on" (default) reproduces
    # the pre-flag path bit-identically.
    if eddington_m == "off":
        host_M_eff = np.asarray(host_M, dtype=np.float64)
    elif _use_volume_deconv and not _use_mass_trunc:
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
    elif catalogue_mass_overlap != "production":
        # [PHYSICS] prod2d closure counterfactual instrument (results/
        # prod2d_closure_20260818/PREREGISTRATION_PROD_COUNTERFACTUAL.md §1).
        # Guard pattern: the production float stream (the `else` branch below)
        # is reached ONLY when catalogue_mass_overlap == "production" and is
        # never touched here.
        mu_gal_frac = host_M_eff[:, None] * (1 + y_num_nodes) / _det_M
        if catalogue_mass_overlap == "neutralized":
            # V1': replace the candidate's own mz_integral with the SAME
            # population mass factor the completion leg uses
            # (completion_mass_factor_g, :2022) at the candidate's own
            # z-nodes, with the event's det_M_z / projection / sigma_cond --
            # the candidate becomes mass-UNINFORMATIVE. Both legs remain
            # densities in the same x_M measure (registered normalization
            # coherence statement, §1 V1').
            _z_flat = np.broadcast_to(y_num_nodes, mu_gal_frac.shape).reshape(-1)
            _dlf_flat = np.broadcast_to(luminosity_distance_fraction, mu_gal_frac.shape).reshape(-1)
            mz_integral = completion_mass_factor_g(
                _z_flat,
                _dlf_flat,
                _det_M,
                float(proj_d_L_to_M_arr[slot]),
                float(sigma_cond_M_arr[slot]),
            ).reshape(mu_gal_frac.shape)
        elif catalogue_mass_overlap == "inflated":
            # V2: host_M_error -> k*host_M_error ONLY in the numerator width
            # sigma_gal; the Eddington-shifted mean host_M_eff (above) stays
            # computed from the UNSCALED host_M_error (frozen mu_eff, §1 V2).
            sigma_gal_frac = (
                (host_M_error * catalogue_mass_error_scale)[:, None] * (1 + y_num_nodes) / _det_M
            )
            sigma2_sum = _sigma2_cond + sigma_gal_frac**2
            mz_integral = np.exp(-0.5 * (mu_cond - mu_gal_frac) ** 2 / sigma2_sum) / np.sqrt(
                2 * np.pi * sigma2_sum
            )
        else:
            raise ValueError(
                "catalogue_mass_overlap must be 'production', 'neutralized' "
                f"or 'inflated', got {catalogue_mass_overlap!r}"
            )
    else:
        mu_gal_frac = host_M_eff[:, None] * (1 + y_num_nodes) / _det_M
        sigma_gal_frac = host_M_error[:, None] * (1 + y_num_nodes) / _det_M

        # Analytic Gaussian product integral, Eq. (14.31).
        sigma2_sum = _sigma2_cond + sigma_gal_frac**2
        mz_integral = np.exp(-0.5 * (mu_cond - mu_gal_frac) ** 2 / sigma2_sum) / np.sqrt(
            2 * np.pi * sigma2_sum
        )

        if _cat_surv_2d_on:
            # [P3-2D] the with-BH catalogue-leg twin: S_4D inside the
            # candidate's own mass quadrature (product-Gaussian identity, see
            # _mz_sel_2d_expectation_batch). Flows through BOTH the
            # generator-point AND quadrature reduce below via mz_integral.
            _host_M_for_surv = (
                host_M if catalogue_numerator_survival_2d_center == "raw" else host_M_eff
            )
            mu_gal_surv = _host_M_for_surv[:, None] * (1 + y_num_nodes) / _det_M
            mu_star = (mu_cond * sigma_gal_frac**2 + mu_gal_surv * _sigma2_cond) / sigma2_sum
            sigma_star = np.sqrt(_sigma2_cond * sigma_gal_frac**2 / sigma2_sum)
            d_L_at_num = np.asarray(luminosity_distance_fraction * _det_d_L, dtype=np.float64)
            mz_integral = mz_integral * _mz_sel_2d_expectation_batch(
                mu_star,
                sigma_star,
                y_num_nodes,
                d_L_at_num,
                _det_M,
                detection_probability,
                host_phiS,
                host_qS,
                h,
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
        # [PHYSICS] C7-core: D_g integrates the SAME rho_g as the numerator, so
        # the 64-node with-BH-mass denominator carries f_{k(g)}(z) too.
        # Theorem (T) + partition argument, GATE_PACKAGE_FINAL.md §1.2
        # (2026-08-04); structure: Gray et al. (2020) arXiv:1908.06050
        # Eq. (A.10); Turski et al. (2023) arXiv:2302.12037 Eq. (4)
        if host_pixels is not None:
            f_host_bh = _completeness_at_host_nodes(
                completeness_model,  # type: ignore[arg-type]
                y_bh,
                host_pixels,
                h,
            )
            if zoa_rows is not None:
                f_host_bh[zoa_rows, :] = 1.0
            w_pop_bh = w_pop_bh * f_host_bh
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
    host_mass_kernel: str = "auto",
    catalogue_mass_overlap: str = "production",
    catalogue_mass_error_scale: float = 1.0,
    eddington_m: str = "on",
    catalogue_numerator_survival: str = "off",
    catalogue_survival_table: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
    catalogue_numerator_survival_2d: str = "off",
    catalogue_numerator_survival_2d_center: str = "unset",
    theta_b: float = 0.0,
    theta_s: float = 1.0,
    catalogue_leg_1d_mass_aware: str = "off",
    sigma4d_mass_kernel: str = "point",
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
        host_mass_kernel: 2D host-mass kernel selection (#40 remainder).
        catalogue_mass_overlap: Prod2d closure counterfactual instrument
            ("production"/"neutralized"/"inflated"); see
            results/prod2d_closure_20260818/
            PREREGISTRATION_PROD_COUNTERFACTUAL.md §1.
        catalogue_mass_error_scale: Width multiplier ``k`` for "inflated".
        eddington_m: Instrument E ("on"/"off"); see
            results/prod2d_closure_20260818/PREREGISTRATION_TILT_BATTERY.md §1.
        catalogue_numerator_survival: [P3-IMP] twin cell ("off"/"phi");
            PREREGISTRATION_P3_TWIN_20260822.md §2.
        catalogue_survival_table: The per-h ``(z_grid, s_phi_grid)`` slice of
            the phi-marginal survival table; required when the twin cell is
            "phi", ignored otherwise.
        catalogue_numerator_survival_2d: [P3-2D] the with-BH catalogue-leg
            twin ("off"/"mz_sel"); PREREGISTRATION_P3_2D_20260825.md §2(i).
        catalogue_numerator_survival_2d_center: Centering sub-option
            ("unset"/"raw"/"eff"); REQUIRED to be "raw" or "eff" when the
            twin cell is "mz_sel".
        catalogue_leg_1d_mass_aware: [HIER T2.3] mass-aware 1D catalogue leg
            instrument ("off"/"on");
            PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md §2.
        sigma4d_mass_kernel: Mirrors ``self._sigma4d_mass_kernel`` for the
            mass-aware 1D leg's own point/kernel form (§2.2's registered
            coupling rule); ignored unless
            ``catalogue_leg_1d_mass_aware="on"``.

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
        + (
            detection_index,
            h,
            evaluate_with_bh_mass,
            normalization_mode,
            host_z_kernel,
            host_mass_kernel,
            catalogue_mass_overlap,
            catalogue_mass_error_scale,
            eddington_m,
            catalogue_numerator_survival,
            catalogue_survival_table,
            catalogue_numerator_survival_2d,
            catalogue_numerator_survival_2d_center,
            theta_b,
            theta_s,
            catalogue_leg_1d_mass_aware,
            sigma4d_mass_kernel,
        )
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
    current_completeness: CompletenessModel | None = None,
    # (N8) d_L-only 2x2 block conditional scalars, threaded for the
    # "neutralized" prod2d closure counterfactual (results/
    # prod2d_closure_20260818/PREREGISTRATION_PROD_COUNTERFACTUAL.md §1 V1').
    # None (default, e.g. existing hand-built worker-global tests) leaves the
    # module globals at their prior value -- byte-identical, since only the
    # "neutralized" mode reads them.
    current_proj_d_L_to_M_arr: npt.NDArray[np.float64] | None = None,
    current_sigma_cond_M_arr: npt.NDArray[np.float64] | None = None,
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
    global proj_d_L_to_M_arr, sigma_cond_M_arr
    global det_d_L_arr, det_d_L_unc_arr, det_M_arr, det_phi_arr, det_theta_arr
    global D_h_table
    global completeness_model

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
    if current_proj_d_L_to_M_arr is not None:
        proj_d_L_to_M_arr = current_proj_d_L_to_M_arr
    if current_sigma_cond_M_arr is not None:
        sigma_cond_M_arr = current_sigma_cond_M_arr
    det_d_L_arr = current_det_d_L_arr
    det_d_L_unc_arr = current_det_d_L_unc_arr
    det_M_arr = current_det_M_arr
    det_phi_arr = current_det_phi_arr
    det_theta_arr = current_det_theta_arr
    if current_D_h_table is not None:
        D_h_table = current_D_h_table
    # C7-core: the per-pixel completeness the host-z kernel evaluates at the
    # HOST's pixel. Threaded here (not passed per task) so the batched and the
    # scalar kernel read the SAME object in every worker.
    completeness_model = current_completeness


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
