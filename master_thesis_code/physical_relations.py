"""Cosmological distance functions for a flat wCDM universe.

Provides luminosity distance, redshift inversion, and derived quantities used
throughout the EMRI simulation and Bayesian H₀ inference pipelines.
"""

from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import fsolve
from scipy.special import hyp2f1

from master_thesis_code.constants import (
    GPC_TO_MPC,
    KM_TO_M,
    OMEGA_DE,
    OMEGA_M,
    SPEED_OF_LIGHT_KM_S,
    W_0,
    W_A,
    C,
    H,
)


def dist(
    redshift: float,
    h: float = H,
    Omega_m: float = OMEGA_M,
    Omega_de: float = OMEGA_DE,
    w_0: float = W_0,
    w_a: float = W_A,
    offset_for_root_finding: float = 0.0,
) -> float:
    """Luminosity distance in Gpc for a flat wCDM cosmology.

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
    H_0 = h * 100.0 * KM_TO_M / GPC_TO_MPC ** (-1)  # Hubble constant in m/s*Gpc

    # use analytic version of the integral
    integral = lambda_cdm_analytic_distance(redshift, Omega_m, Omega_de)  # type: ignore[arg-type]

    # luminosity distance in Gpc
    result = C / H_0 * (1 + redshift) * integral - offset_for_root_finding

    return result


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

    first_term = C / H_0 * (1 + redshift) / hubble_function(redshift)

    zs = np.linspace(0, redshift, 1000)
    hubble_function_values = hubble_function(zs)

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
