import numpy as np
from scipy.optimize import fsolve
from master_thesis_code.constants import (
    C,
    H,
    OMEGA_M,
    OMEGA_DE,
    W_0,
    W_A,
    GPC_TO_MPC,
    KM_TO_M,
)


def dist(
    redshift,
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
    zs = np.linspace(0, redshift, 1000)

    H_0 = h * 100.0 * KM_TO_M * GPC_TO_MPC ** (-1)

    # Hubble parameter
    hubble = np.array(
        [
            np.sqrt(
                Omega_m * (1 + z) ** 3
                + Omega_de
                * (1 + z) ** (3 * (1 + w_0 + w_a))
                * np.exp(-3 * w_a * z / (1 + z))
            )
            for z in zs
        ]
    )

    # integral
    integral = np.trapz(1 / hubble, zs)
    print(integral)

    # luminosity distance
    result = C / H_0 * (1 + redshift) * integral - offset_for_root_finding
    print(result)
    return result


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
    :param redshift_min: minimum redshift
    :param redshift_max: maximum redshift
    :param redshift_steps: number of steps
    :return: redshift
    """
    return fsolve(
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
    z_1 = dist_to_redshift(
        distance + derivative_epsilon, h, Omega_m, Omega_de, w_0, w_a
    )
    derivative = (z_1 - z_0) / derivative_epsilon
    return np.sqrt((derivative * distance_error) ** 2)
