import pandas as pd
import time

# test execution time of single_host_likelihood function


def test_single_host_likelihood():
    from master_thesis_code.cosmological_model import Detection, single_host_likelihood
    from master_thesis_code.galaxy_catalogue.handler import (
        HostGalaxy,
        InternalCatalogColumns,
    )
    from master_thesis_code.constants import H, OMEGA_DE, OMEGA_M, W_0, W_A

    host_galaxy_parameters = pd.Series(
        {
            InternalCatalogColumns.PHI_S: 0.5,
            InternalCatalogColumns.THETA_S: 0.3,
            InternalCatalogColumns.REDSHIFT: 0.1,
            InternalCatalogColumns.REDSHIFT_ERROR: 0.01,
            InternalCatalogColumns.BH_MASS: 1000000,
            InternalCatalogColumns.BH_MASS_ERROR: 0.1,
        }
    )

    detection_parameters = pd.Series(
        {
            "dist": 1.0,
            "delta_dist_delta_dist": 0.1,
            "phiS": 0.5,
            "delta_phiS_delta_phiS": 0.05,
            "qS": 0.3,
            "delta_qS_delta_qS": 0.03,
            "M": 1000000,
            "delta_M_delta_M": 0.07,
            "delta_phiS_delta_qS": 0.00005,
            "delta_phiS_delta_M": 0.007,
            "delta_qS_delta_M": 0.003,
        }
    )

    host_galaxy = HostGalaxy(host_galaxy_parameters)

    detection = Detection(detection_parameters)

    start = time.time()
    results = single_host_likelihood(
        possible_host=host_galaxy,
        detection=detection,
        h=H,
        Omega_de=OMEGA_DE,
        Omega_m=OMEGA_M,
        w_0=W_0,
        w_a=W_A,
        evaluate_with_bh_mass=True,
    )
    end = time.time()
    duration = end - start
    assert duration > 1
