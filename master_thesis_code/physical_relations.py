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
    """
    Calculate the luminosity distance in Gpc.

    :param redshift: redshift
    :param Omega_m: matter density parameter
    :param Omega_de: dark energy density parameter
    :param w_0: dark energy equation of state parameter
    :param w_a: dark energy equation of state parameter
    :return: luminosity distance in Gpc
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
    """
    Calculate the luminosity distance in Gpc.

    :param redshift: redshift
    :param Omega_m: matter density parameter
    :param Omega_de: dark energy density parameter
    :param w_0: dark energy equation of state parameter
    :param w_a: dark energy equation of state parameter
    :return: luminosity distance in Gpc
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
    """
    Calculate the luminosity distance in Gpc (vectorized over redshift array).

    :param redshift: redshift array
    :param Omega_m: matter density parameter
    :param Omega_de: dark energy density parameter
    :param w_0: dark energy equation of state parameter
    :param w_a: dark energy equation of state parameter
    :return: luminosity distance array in Gpc
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
    """
    Calculate the redshift for a given luminosity distance.

    :param distance: luminosity distance in Gpc
    :param Omega_m: matter density parameter
    :param Omega_de: dark energy density parameter
    :param w_0: dark energy equation of state parameter
    :param w_a: dark energy equation of state parameter
    :return: redshift
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


def visualize() -> None:
    import matplotlib.pyplot as plt

    zs = np.linspace(0, 2, 1000)
    distances = [dist(z) for z in zs]
    plt.plot(zs, distances)
    plt.xlabel("Redshift")
    plt.ylabel("Distance [Gpc]")
    plt.show()
    redshifts = [dist_to_redshift(d) for d in distances]
    plt.plot(distances, redshifts)
    plt.xlabel("Distance [Gpc]")
    plt.ylabel("Redshift")
    plt.show()

    # check lower and upper bound
    distance = 0.5
    distance_error = 0.05
    for h in np.linspace(0.6, 0.83, 3):
        for Omega_m in np.linspace(0.04, 0.5, 3):
            z = dist_to_redshift(distance, h, h, Omega_m, Omega_m)
            plt.scatter([z], [h], label=f"Omega_m={Omega_m}, h={h}")
    plt.xlabel("Redshift")
    plt.ylabel("Hubble constant")
    plt.legend()
    plt.show()
