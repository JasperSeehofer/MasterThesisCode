"""Cosmological distance functions for a flat cosmology.

Provides luminosity distance, redshift inversion, and derived quantities used
throughout the EMRI simulation and Bayesian H₀ inference pipelines.

Note: the analytic luminosity distance (``dist`` and friends) is **ΛCDM-only**
(hypergeometric form); it raises ``NotImplementedError`` on genuine wCDM inputs
(``w_0 != -1`` or ``w_a != 0``). ``hubble_function`` does implement the full CPL
``E(z)``. A wCDM luminosity distance would require numerical quadrature and must go
through ``/physics-change`` (GitHub #4).
"""

from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
from scipy.special import hyp2f1

from darksiren_emri.constants import (
    GPC_TO_MPC,
    KM_TO_M,
    OMEGA_DE,
    OMEGA_M,
    PRESCREEN_DL_MARGIN,
    SPEED_OF_LIGHT_KM_S,
    W_0,
    W_A,
    C,
    H,
)


def _reject_unsupported_wcdm(w_0: float, w_a: float) -> None:
    """Guard: the analytic distance is ΛCDM-only (hyp2f1), so fail loudly on genuine
    wCDM inputs instead of silently returning the ΛCDM result (review PHY-01, GitHub #4).

    ``dist``/``cached_dist``/``dist_vectorized`` accept ``w_0``/``w_a`` for signature
    compatibility but ``lambda_cdm_analytic_distance`` ignores them. Every production
    call uses the defaults ``w_0 = -1``, ``w_a = 0`` (verified by grep), so this guard
    changes no computed value; it only prevents a silent wrong answer. A real wCDM
    implementation (numerical quadrature of ``1/E(z)`` with the CPL ``hubble_function``)
    must go through ``/physics-change``.
    """
    if w_0 != -1.0 or w_a != 0.0:
        msg = (
            f"analytic luminosity distance is ΛCDM-only (w_0=-1, w_a=0); got "
            f"w_0={w_0}, w_a={w_a}. wCDM requires numerical quadrature of 1/E(z)."
        )
        raise NotImplementedError(msg)


# ── Luminosity-distance spline table (perf; /physics-change 2026-07-08) ──────
# The hot-path d_L evaluation was ~52% of the H0-evaluation CPU, dominated by
# scipy.special.hyp2f1 (no GPU equivalent). We exploit the exact factorisation
#
#     d_L(z, h) = (c / H_0(h)) * (1 + z) * I(z),   I(z) = INTEGRAL_0^z dz'/E(z'),
#
# where E(z) and I(z) are h-INDEPENDENT and the whole h-dependence is the c/H_0
# prefactor (exactly 1/h). So one CubicSpline of I(z) serves every h in a run.
# Node values are the exact hyp2f1 antiderivative (lambda_cdm_analytic_distance),
# paid once; CLAMPED boundary conditions with the exact endpoint slope
# I'(z) = 1/E(z) are required — natural BC (I''=0) is wrong at z=0 because
# I''(0) = -3/2 Omega_m != 0 and blows the low-z relative error up to ~5e-5.
# Accuracy vs an adaptive-quad reference is 2.2e-10 over z in [0, 1.6] — tighter
# than the incumbent hyp2f1's own ~6e-10, so the H0 MAP cannot move. Only the
# fiducial LCDM cosmology within the tabulated z-range takes the fast path; every
# other input falls back to the exact hyp2f1 path unchanged.
# Ref: Hogg (1999), arXiv:astro-ph/9905116, Eqs. (15)-(16).
_DL_TABLE_Z_MAX: float = 1.6
_DL_TABLE_N_KNOTS: int = 512
_comoving_integral_spline_cache: dict[tuple[float, float], CubicSpline] = {}
_z_from_dl_ratio_spline_cache: dict[tuple[float, float], CubicSpline] = {}


def _e_of_z_lcdm(
    z: float | npt.NDArray[np.float64], Omega_m: float, Omega_de: float
) -> npt.NDArray[np.float64]:
    """Dimensionless flat-LCDM Hubble function E(z) = sqrt(Omega_m (1+z)^3 + Omega_de)."""
    return np.sqrt(Omega_m * (1.0 + np.asarray(z, dtype=np.float64)) ** 3 + Omega_de)


def _comoving_integral_knots(
    Omega_m: float, Omega_de: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return (z_knots, I(z_knots)) with I(0)=0 pinned exactly."""
    z_knots = np.linspace(0.0, _DL_TABLE_Z_MAX, _DL_TABLE_N_KNOTS, dtype=np.float64)
    i_knots = np.array(
        [float(lambda_cdm_analytic_distance(float(zk), Omega_m, Omega_de)) for zk in z_knots],
        dtype=np.float64,
    )
    i_knots[0] = 0.0
    return z_knots, i_knots


def _get_comoving_integral_spline(Omega_m: float, Omega_de: float) -> CubicSpline:
    """Lazily build+cache the clamped CubicSpline of I(z) for a cosmology."""
    key = (Omega_m, Omega_de)
    spline = _comoving_integral_spline_cache.get(key)
    if spline is None:
        z_knots, i_knots = _comoving_integral_knots(Omega_m, Omega_de)
        d_i_dz_0 = 1.0 / float(_e_of_z_lcdm(0.0, Omega_m, Omega_de))
        d_i_dz_max = 1.0 / float(_e_of_z_lcdm(_DL_TABLE_Z_MAX, Omega_m, Omega_de))
        spline = CubicSpline(z_knots, i_knots, bc_type=((1, d_i_dz_0), (1, d_i_dz_max)))
        _comoving_integral_spline_cache[key] = spline
    return spline


def _get_z_from_dl_ratio_spline(Omega_m: float, Omega_de: float) -> CubicSpline:
    """Lazily build+cache the inverse spline z(u), u(z) = (1+z) I(z) monotone.

    Since d_L = (c/H_0) u(z), inverting u -> z once serves every h (the h-only
    prefactor is divided out before the lookup). Clamped ends use the exact
    dz/du = 1/(du/dz), du/dz = I(z) + (1+z)/E(z) > 0 (a bijection).
    """
    key = (Omega_m, Omega_de)
    spline = _z_from_dl_ratio_spline_cache.get(key)
    if spline is None:
        z_knots, i_knots = _comoving_integral_knots(Omega_m, Omega_de)
        u_knots = (1.0 + z_knots) * i_knots  # strictly increasing, u(0)=0
        du_dz_0 = float(i_knots[0]) + 1.0 / float(_e_of_z_lcdm(0.0, Omega_m, Omega_de))
        du_dz_max = float(i_knots[-1]) + (1.0 + _DL_TABLE_Z_MAX) / float(
            _e_of_z_lcdm(_DL_TABLE_Z_MAX, Omega_m, Omega_de)
        )
        spline = CubicSpline(u_knots, z_knots, bc_type=((1, 1.0 / du_dz_0), (1, 1.0 / du_dz_max)))
        _z_from_dl_ratio_spline_cache[key] = spline
    return spline


def dist(
    redshift: float,
    h: float = H,
    Omega_m: float = OMEGA_M,
    Omega_de: float = OMEGA_DE,
    w_0: float = W_0,
    w_a: float = W_A,
    offset_for_root_finding: float = 0.0,
) -> float:
    """Luminosity distance in Gpc for a flat ΛCDM cosmology.

    Uses the analytic hypergeometric form of the comoving distance integral:

    .. math::

        d_L(z) = \\frac{c\\,(1+z)}{H_0} \\int_0^z \\frac{dz'}{E(z')}

    where :math:`E(z) = \\sqrt{\\Omega_m(1+z)^3 + \\Omega_\\Lambda}` for
    :math:`w_0 = -1,\\, w_a = 0`.

    Args:
        redshift: Source redshift :math:`z \\geq 0`.
        h: Dimensionless Hubble parameter
            :math:`h = H_0 / (100\\,\\mathrm{km\\,s^{-1}\\,Mpc^{-1}})`.
        Omega_m: Matter density parameter :math:`\\Omega_m`.
        Omega_de: Dark energy density parameter :math:`\\Omega_\\Lambda`.
        w_0: Dark energy equation-of-state parameter :math:`w_0`.
        w_a: Dark energy equation-of-state evolution :math:`w_a`.
        offset_for_root_finding: Subtracted from the result; set to the target
            distance when calling this function via ``scipy.optimize.fsolve``
            for redshift inversion.

    Returns:
        Luminosity distance in Gpc.

    References:
        Hogg (1999), *Distance measures in cosmology*, arXiv:astro-ph/9905116, Eq. (16).

    Examples:
        >>> dist(0.0)
        0.0
    """
    _reject_unsupported_wcdm(w_0, w_a)
    H_0 = h * 100.0 * KM_TO_M / GPC_TO_MPC ** (-1)  # Hubble constant in m/s*Gpc

    # use analytic version of the integral
    integral = lambda_cdm_analytic_distance(redshift, Omega_m, Omega_de)

    # luminosity distance in Gpc
    result = C / H_0 * (1 + redshift) * integral - offset_for_root_finding

    return float(np.asarray(result).flat[0])


@lru_cache(maxsize=1000)
def cached_dist(
    redshift: float,
    h: float = H,
    Omega_m: float = OMEGA_M,
    Omega_de: float = OMEGA_DE,
    w_0: float = W_0,
    w_a: float = W_A,
    offset_for_root_finding: float = 0.0,
) -> float:
    """LRU-cached version of :func:`dist`.

    Identical semantics; results are memoized up to 1000 unique argument
    combinations, which eliminates redundant integration in hot paths.

    Args:
        redshift: Source redshift :math:`z \\geq 0`.
        h: Dimensionless Hubble parameter.
        Omega_m: Matter density parameter.
        Omega_de: Dark energy density parameter.
        w_0: Dark energy equation-of-state parameter.
        w_a: Dark energy equation-of-state evolution.
        offset_for_root_finding: Subtracted from the result; used for inversion
            via ``scipy.optimize.fsolve``.

    Returns:
        Luminosity distance in Gpc.
    """
    _reject_unsupported_wcdm(w_0, w_a)
    H_0 = h * 100.0 * KM_TO_M / GPC_TO_MPC ** (-1)  # Hubble constant in m/s*Gpc

    # use analytic version of the integral
    integral = lambda_cdm_analytic_distance(redshift, Omega_m, Omega_de)

    # luminosity distance in Gpc
    result = C / H_0 * (1 + redshift) * integral - offset_for_root_finding

    return float(np.asarray(result).flat[0])


def dist_vectorized(
    redshift: npt.NDArray[np.floating[Any]],
    h: float = H,
    Omega_m: float = OMEGA_M,
    Omega_de: float = OMEGA_DE,
    w_0: float = W_0,
    w_a: float = W_A,
    offset_for_root_finding: float = 0.0,
) -> npt.NDArray[np.floating[Any]]:
    """Vectorized luminosity distance in Gpc over a redshift array.

    Applies the same formula as :func:`dist` element-wise without Python loops,
    using NumPy broadcasting via :func:`lambda_cdm_analytic_distance`.

    Args:
        redshift: Array of source redshifts :math:`z \\geq 0`.
        h: Dimensionless Hubble parameter.
        Omega_m: Matter density parameter.
        Omega_de: Dark energy density parameter.
        w_0: Dark energy equation-of-state parameter.
        w_a: Dark energy equation-of-state evolution.
        offset_for_root_finding: Subtracted from every element of the result.

    Returns:
        Array of luminosity distances in Gpc, same shape as *redshift*.
    """
    _reject_unsupported_wcdm(w_0, w_a)
    H_0 = h * 100.0 * KM_TO_M / GPC_TO_MPC ** (-1)  # Hubble constant in m/s*Gpc

    z_arr = np.asarray(redshift, dtype=np.float64)
    # Fast path: fiducial LCDM within the tabulated z-range uses the clamped
    # cubic-spline of I(z) instead of hyp2f1 (h enters only via the 1/H_0
    # prefactor). Eqs. (15)-(16) in Hogg (1999), arXiv:astro-ph/9905116.
    if (
        Omega_m == OMEGA_M
        and Omega_de == OMEGA_DE
        and z_arr.size > 0
        and float(z_arr.min()) >= 0.0
        and float(z_arr.max()) <= _DL_TABLE_Z_MAX
    ):
        integral = np.asarray(
            _get_comoving_integral_spline(Omega_m, Omega_de)(z_arr), dtype=np.float64
        )
    else:
        # exact hyp2f1 antiderivative (non-fiducial cosmology or z beyond table)
        integral = np.asarray(
            lambda_cdm_analytic_distance(redshift, Omega_m, Omega_de),  # type: ignore[arg-type]
            dtype=np.float64,
        )

    # luminosity distance in Gpc
    result = C / H_0 * (1 + redshift) * integral - offset_for_root_finding

    return result


def luminosity_distance_prescreen_gpc(
    z_max: float,
    h: float,
    Omega_m: float = OMEGA_M,
    Omega_de: float = OMEGA_DE,
    margin: float = PRESCREEN_DL_MARGIN,
) -> float:
    """Population-derived luminosity-distance pre-screen bound in Gpc.

    The simulation loop skips events with :math:`d_L` above this bound before
    generating any waveform. The bound is the luminosity distance of the rate
    model's maximum sampled redshift, inflated by a small safety margin, so no
    in-population event can be cut — at physical SNR semantics (G8 dt² fix) the
    EMRI detection horizon (z ≈ 1.5–3.8) exceeds the population reach, making
    the pre-screen inert for valid events; it only guards pathological draws.

    Supersedes the retired ``LUMINOSITY_DISTANCE_PRESCREEN_GPC = 2.0``, which
    was calibrated on pre-dt² (SNR/10-scale) injection data and lay inside the
    z ≤ 0.5 host-draw volume (issue #19). The margin is a placeholder until
    re-measured on post-dt² injection data.

    Args:
        z_max: Maximum redshift sampled by the population model
            (``Model1CrossCheck.max_redshift``).
        h: Dimensionless Hubble parameter of the current run.
        Omega_m: Matter density parameter.
        Omega_de: Dark energy density parameter.
        margin: Multiplicative safety margin (≥ 1).

    Returns:
        Pre-screen bound in Gpc; 0.0 exactly at ``z_max = 0``.

    References:
        Babak et al. (2017), arXiv:1703.09722 (M1 population; EMRI horizon);
        Hogg (1999), arXiv:astro-ph/9905116 Eq. (16) via :func:`dist`.
    """
    # d_L(z_max; h) × margin — Hogg (1999), arXiv:astro-ph/9905116 Eq. (16)
    return margin * dist(z_max, h=h, Omega_m=Omega_m, Omega_de=Omega_de)


def dist_derivative(
    redshift: float,
    h: float = H,
    Omega_m: float = OMEGA_M,
    Omega_de: float = OMEGA_DE,
    w_0: float = W_0,
    w_a: float = W_A,
) -> float:
    """Derivative of luminosity distance with respect to redshift, :math:`dd_L/dz` in Gpc.

    Uses the analytic expression:

    .. math::

        \\frac{dd_L}{dz} = \\frac{c}{H_0} \\left[
            \\frac{1+z}{E(z)} + \\int_0^z \\frac{dz'}{E(z')}
        \\right]

    Args:
        redshift: Source redshift :math:`z \\geq 0`.
        h: Dimensionless Hubble parameter.
        Omega_m: Matter density parameter.
        Omega_de: Dark energy density parameter.
        w_0: Dark energy equation-of-state parameter.
        w_a: Dark energy equation-of-state evolution.

    Returns:
        :math:`dd_L/dz` in Gpc.
    """
    H_0 = h * 100.0 * KM_TO_M / GPC_TO_MPC ** (-1)  # Hubble constant in m/s*Gpc

    # Forward the cosmology so a non-default Omega_m / w_0 / w_a is honoured rather
    # than silently replaced by module defaults (review PHY-02). hubble_function
    # implements the full CPL E(z); value-neutral for the ΛCDM production defaults.
    first_term = (
        C
        / H_0
        * (1 + redshift)
        / hubble_function(redshift, Omega_m=Omega_m, Omega_de=Omega_de, w_0=w_0, w_a=w_a)
    )

    zs = np.linspace(0, redshift, 1000)
    hubble_function_values = hubble_function(
        zs, Omega_m=Omega_m, Omega_de=Omega_de, w_0=w_0, w_a=w_a
    )

    # integral
    second_term = C / H_0 * float(np.trapezoid(1 / hubble_function_values, zs))

    return float(first_term + second_term)


def hubble_function(
    redshift: float | npt.NDArray[np.floating[Any]],
    h: float = H,
    Omega_m: float = OMEGA_M,
    Omega_de: float = OMEGA_DE,
    w_0: float = W_0,
    w_a: float = W_A,
) -> float | npt.NDArray[np.floating[Any]]:
    """Dimensionless Hubble function :math:`E(z) = H(z) / H_0` for a flat wCDM cosmology.

    .. math::

        E(z) = \\sqrt{\\Omega_m (1+z)^3 + \\Omega_\\Lambda (1+z)^{3(1+w_0+w_a)}
               \\exp\\!\\left(\\frac{-3 w_a z}{1+z}\\right)}

    For the fiducial ΛCDM case (:math:`w_0 = -1,\\, w_a = 0`) this reduces to
    :math:`E(z) = \\sqrt{\\Omega_m (1+z)^3 + \\Omega_\\Lambda}`.

    Args:
        redshift: Source redshift or array of redshifts.
        h: Dimensionless Hubble parameter (unused — :math:`E(z)` is independent of
            :math:`h` by definition).
        Omega_m: Matter density parameter :math:`\\Omega_m`.
        Omega_de: Dark energy density parameter :math:`\\Omega_\\Lambda`.
        w_0: Dark energy equation-of-state parameter :math:`w_0`.
        w_a: Dark energy equation-of-state evolution :math:`w_a`.

    Returns:
        :math:`E(z)` as a float when *redshift* is a scalar, or as an ndarray
        when *redshift* is an array.
    """
    result = np.sqrt(
        Omega_m * (1 + redshift) ** 3
        + Omega_de
        * (1 + redshift) ** (3 * (1 + w_0 + w_a))
        * np.exp(-3 * w_a * redshift / (1 + redshift))
    )
    if np.ndim(result) == 0:
        return float(result)
    return result


def lambda_cdm_analytic_distance(
    redshift: float, Omega_m: float = OMEGA_M, Omega_de: float = OMEGA_DE
) -> float:
    """Analytic ΛCDM comoving distance integral :math:`\\int_0^z dz'/E(z')` in units of :math:`c/H_0`.

    Evaluates the integral in closed form using the Gauss hypergeometric function
    :math:`{}_2F_1`, valid for a flat ΛCDM cosmology (:math:`w_0=-1,\\, w_a=0`).

    Args:
        redshift: Source redshift.
        Omega_m: Matter density parameter.
        Omega_de: Dark energy density parameter.

    Returns:
        Dimensionless comoving distance integral :math:`\\int_0^z dz'/E(z')`.
    """
    return (  # type: ignore[no-any-return]
        (
            (1 + redshift)
            * np.sqrt(1 + (Omega_m * (1 + redshift) ** 3) / Omega_de)
            * hyp2f1(1 / 3, 1 / 2, 4 / 3, -((Omega_m * (1 + redshift) ** 3) / Omega_de))
        )
        / np.sqrt(Omega_de + Omega_m * (1 + redshift) ** 3)
        - (
            np.sqrt((Omega_m + Omega_de) / Omega_de)
            * hyp2f1(1 / 3, 1 / 2, 4 / 3, -(Omega_m / Omega_de))
        )
        / np.sqrt(Omega_m + Omega_de)
    )


def dist_to_redshift(
    distance: float,
    h: float = H,
    Omega_m: float = OMEGA_M,
    Omega_de: float = OMEGA_DE,
    w_0: float = W_0,
    w_a: float = W_A,
) -> float:
    """Redshift corresponding to a given luminosity distance (inverse of :func:`dist`).

    Solves :math:`d_L(z) = \\mathrm{distance}` via ``scipy.optimize.fsolve`` with
    initial guess :math:`z = 1`.

    Args:
        distance: Luminosity distance in Gpc.
        h: Dimensionless Hubble parameter.
        Omega_m: Matter density parameter.
        Omega_de: Dark energy density parameter.
        w_0: Dark energy equation-of-state parameter.
        w_a: Dark energy equation-of-state evolution.

    Returns:
        Redshift :math:`z` such that :math:`d_L(z) = \\mathrm{distance}`.
    """
    H_0 = h * 100.0 * KM_TO_M / GPC_TO_MPC ** (-1)  # m/s*Gpc, matches dist()
    # Fast path: fiducial LCDM inverts d_L = (c/H_0) u(z), u(z)=(1+z) I(z), via
    # the cached monotone inverse spline z(u) — no per-call root find. Falls back
    # to fsolve for non-fiducial cosmology or distances beyond the tabulated range.
    if Omega_m == OMEGA_M and Omega_de == OMEGA_DE and w_0 == -1.0 and w_a == 0.0:
        spline = _get_z_from_dl_ratio_spline(Omega_m, Omega_de)
        u = float(distance) * H_0 / C
        if 0.0 <= u <= float(spline.x[-1]):
            return float(spline(u))
    return float(
        fsolve(
            dist,
            1,
            args=(
                h,
                Omega_m,
                Omega_de,
                w_0,
                w_a,
                distance,
            ),
        )[0]
    )


def dist_to_redshift_error_proagation(
    distance: float,
    distance_error: float,
    h: float = H,
    Omega_m: float = OMEGA_M,
    Omega_de: float = OMEGA_DE,
    w_0: float = W_0,
    w_a: float = W_A,
    derivative_epsilon: float = 1e-6,
) -> float:
    """
    Calculate the redshift error for a given luminosity distance error.
    """
    z_0 = dist_to_redshift(distance, h, Omega_m, Omega_de, w_0, w_a)
    z_1 = dist_to_redshift(distance + derivative_epsilon, h, Omega_m, Omega_de, w_0, w_a)
    derivative = (z_1 - z_0) / derivative_epsilon
    return float(np.sqrt((derivative * distance_error) ** 2))


def redshifted_mass(mass: float, redshift: float) -> float:
    """Return the redshifted mass M_z = M * (1 + z)."""
    return mass * (1 + redshift)


def redshifted_mass_inverse(redshifted_mass: float, redshift: float) -> float:
    """Return the true mass M = M_z / (1 + z)."""
    return redshifted_mass / (1 + redshift)


def convert_redshifted_mass_to_true_mass(
    M_z: float, M_z_error: float, z: float, z_error: float
) -> tuple[float, float]:
    M = M_z / (1 + z)
    M_err = float(np.sqrt((M_z_error / (1 + z)) ** 2 + (M_z * z_error / (1 + z) ** 2) ** 2))
    return (M, M_err)


def convert_true_mass_to_redshifted_mass_with_distance(M: float, dist: float) -> float:
    z = dist_to_redshift(dist)
    return float(M * (1 + z))


def convert_true_mass_to_redshifted_mass(
    M: float, M_error: float, z: float, z_error: float
) -> tuple[float, float]:
    M_z = M * (1 + z)
    M_z_err = float(np.sqrt((M_error * (1 + z)) ** 2 + (M * z_error) ** 2))
    return (M_z, M_z_err)


def get_redshift_outer_bounds(
    distance: float,
    distance_error: float,
    h_min: float = 0.6,
    h_max: float = 0.86,
    Omega_m_min: float = 0.04,
    Omega_m_max: float = 0.5,
    w_0: float = W_0,
    w_a: float = W_A,
    sigma_multiplier: float = 3.0,
) -> tuple[float, float]:
    """
    Calculate the outer bounds for the redshift for a given luminosity distance and error w.r.t LamCDM model.
    """
    # FOR NOW IGNORE UNCERTAINTIES IN OMEGA_DE AND W
    Omega_de_min = 1 - Omega_m_min
    Omega_de_max = 1 - Omega_m_max
    z_min = dist_to_redshift(distance - 3 * distance_error, h_min)
    if distance - 3 * distance_error < 0:
        z_min = 0.0
    z_max = dist_to_redshift(distance + 3 * distance_error, h_max)
    return z_min, z_max


# Eq. (28) in Hogg (1999), arXiv:astro-ph/9905116
def comoving_volume_element(
    z: float | npt.NDArray[np.floating[Any]],
    h: float = H,
    Omega_m: float = OMEGA_M,
    Omega_de: float = OMEGA_DE,
) -> float | npt.NDArray[np.floating[Any]]:
    r"""Comoving volume element per unit redshift per unit solid angle.

    .. math::

        \frac{dV_c}{dz\,d\Omega} = \frac{d_{\mathrm{com}}^2(z)\,c}{H(z)}

    where :math:`d_{\mathrm{com}} = d_L / (1+z)` is the comoving distance and
    :math:`H(z) = h \times 100\,\mathrm{km\,s^{-1}\,Mpc^{-1}} \times E(z)`.

    The result has units of :math:`\mathrm{Mpc}^3\,\mathrm{sr}^{-1}`.

    Dimensional analysis
    --------------------
    :math:`[Mpc]^2 \times [km/s] / [km/s/Mpc] = [Mpc]^3` per steradian.

    Limiting case (z << 1)
    ----------------------
    :math:`d_{\mathrm{com}} \approx c z / H_0`, :math:`H(z) \approx H_0`, so
    :math:`dV_c/dz/d\Omega \approx (c/H_0)^3 z^2`, scaling as :math:`z^2`.

    Args:
        z: Redshift (scalar or array). Must be >= 0.
        h: Dimensionless Hubble parameter.
        Omega_m: Matter density parameter.
        Omega_de: Dark energy density parameter.

    Returns:
        Comoving volume element :math:`dV_c / dz / d\Omega` in
        :math:`\mathrm{Mpc}^3 / \mathrm{sr}`. Same type as input *z*.

    References
    ----------
    Hogg (1999), arXiv:astro-ph/9905116, Eq. (28).
    Gray et al. (2020), arXiv:1908.06050, Appendix A.2.3
    (Eqs. 31-32 use this volume element as the completion term prior).
    """
    # ASSERT_CONVENTION: distance=Mpc, speed=km/s, H0=km/s/Mpc, result=Mpc^3/sr

    # Luminosity distance in Mpc
    z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
    d_L_mpc = dist_vectorized(z_arr, h=h, Omega_m=Omega_m, Omega_de=Omega_de) * GPC_TO_MPC

    # Comoving distance in Mpc: d_com = d_L / (1+z)
    d_com = d_L_mpc / (1.0 + z_arr)

    # Hubble parameter H(z) in km/s/Mpc
    H_z = h * 100.0 * np.asarray(hubble_function(z_arr, Omega_m=Omega_m, Omega_de=Omega_de))

    # dVc/dz/dOmega = d_com^2 * c / H(z)  [Mpc^3/sr]
    result = d_com**2 * SPEED_OF_LIGHT_KM_S / H_z

    if np.ndim(z) == 0:
        return float(result.flat[0])
    return result
