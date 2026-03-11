"""Thin re-export shim for backward compatibility.

All logic has been extracted to:
  - master_thesis_code/datamodels/galaxy.py         (Galaxy, GalaxyCatalog)
  - master_thesis_code/datamodels/emri_detection.py (EMRIDetection)
  - master_thesis_code/bayesian_inference/bayesian_inference.py (BayesianInference, dist_array)

This module re-exports every symbol that external code imports from this path so that
existing imports continue to work without changes.
"""

import multiprocessing as mp
from statistics import NormalDist as NormalDist
from time import time

import numpy as np

from master_thesis_code.bayesian_inference.bayesian_inference import (
    BayesianInference as BayesianInference,
)
from master_thesis_code.bayesian_inference.bayesian_inference import (
    dist_array as dist_array,
)
from master_thesis_code.constants import (
    OMEGA_DE as OMEGA_LAMBDA,  # noqa: F401  re-exported
)
from master_thesis_code.constants import (
    OMEGA_M as OMEGA_M,
)
from master_thesis_code.constants import (
    SPEED_OF_LIGHT_KM_S as SPEED_OF_LIGHT,  # noqa: F401  re-exported
)
from master_thesis_code.constants import (
    TRUE_HUBBLE_CONSTANT as TRUE_HUBBLE_CONSTANT,
)
from master_thesis_code.datamodels.emri_detection import EMRIDetection as EMRIDetection
from master_thesis_code.datamodels.galaxy import Galaxy as Galaxy
from master_thesis_code.datamodels.galaxy import GalaxyCatalog as GalaxyCatalog
from master_thesis_code.physical_relations import (
    dist as dist,
)
from master_thesis_code.physical_relations import (
    dist_to_redshift as dist_to_redshift,
)
from master_thesis_code.physical_relations import (
    lambda_cdm_analytic_distance as lambda_cdm_analytic_distance,
)
from master_thesis_code.physical_relations import (
    redshifted_mass as redshifted_mass,
)
from master_thesis_code.physical_relations import (
    redshifted_mass_inverse as redshifted_mass_inverse,
)

if __name__ == "__main__":
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    from master_thesis_code.plotting import apply_style

    apply_style()

    galaxy_catalog = GalaxyCatalog(use_truncnorm=False, use_comoving_volume=True)
    NUMBER_OF_GALAXIES = 1000
    STEPS = 20
    NUMBER_OF_NEW_DETECTIONS_PER_STEP = 3
    compare_with_truncnorm = False

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.suptitle(
        f"Galaxy Catalog: {NUMBER_OF_GALAXIES}, Detections: {NUMBER_OF_NEW_DETECTIONS_PER_STEP * STEPS}"
    )
    norm = plt.Normalize(0, STEPS * NUMBER_OF_NEW_DETECTIONS_PER_STEP - 1)
    color_map = cm.ScalarMappable(norm=norm, cmap="viridis")

    for i in range(STEPS):
        start_time = time()

        while True:
            galaxy_catalog.remove_all_galaxies()
            galaxy_catalog.create_random_catalog(NUMBER_OF_GALAXIES)
            if (
                len(galaxy_catalog.get_possible_host_galaxies())
                > STEPS * NUMBER_OF_NEW_DETECTIONS_PER_STEP
            ):
                print(
                    f"Galaxy catalog set up with {len(galaxy_catalog.catalog)} galaxies with {len(galaxy_catalog.get_possible_host_galaxies())} possible host galaxies."
                )
                break

        host_galaxies = galaxy_catalog.get_unique_host_galaxies_from_catalog(
            number_of_host_galaxies=NUMBER_OF_NEW_DETECTIONS_PER_STEP * STEPS,
        )

        emri_detections = [
            EMRIDetection.from_host_galaxy(host_galaxy) for host_galaxy in host_galaxies
        ]
        bayesian_inference = BayesianInference(
            galaxy_catalog=galaxy_catalog, emri_detections=emri_detections
        )

        # Inference
        hubble_values = np.linspace(0.6, 0.8, 60)
        with mp.Pool() as pool:
            posterior_distribution = pool.map(bayesian_inference.posterior, hubble_values)

        likelihoods = np.array(posterior_distribution).T

        # plot combined likelihood
        combined_posterior = np.prod(likelihoods, axis=0)
        ax.plot(
            hubble_values,
            combined_posterior / max(combined_posterior),
            color=color_map.to_rgba(i * NUMBER_OF_NEW_DETECTIONS_PER_STEP),  # type: ignore[arg-type]
            linestyle="solid",
            linewidth=1,
        )

        # evaluate with bh mass information
        bayesian_inference.use_bh_mass = True
        with mp.Pool() as pool:
            posterior_distribution = pool.map(bayesian_inference.posterior, hubble_values)

        likelihoods = np.array(posterior_distribution).T
        likelihoods = likelihoods / np.max(likelihoods)

        # plot combined posterior
        combined_posterior = np.prod(likelihoods, axis=0)

        ax.plot(
            hubble_values,
            combined_posterior / np.max(combined_posterior),
            color=color_map.to_rgba(i * NUMBER_OF_NEW_DETECTIONS_PER_STEP),  # type: ignore[arg-type]
            linestyle="dotted",
            label=rf"iteration ${i}$",
            linewidth=1.5,
        )

        if compare_with_truncnorm:
            galaxy_catalog_with_truncnorm = galaxy_catalog
            galaxy_catalog_with_truncnorm._use_truncnorm = True
            bayesian_inference_with_truncnorm = BayesianInference(
                galaxy_catalog=galaxy_catalog_with_truncnorm,
                emri_detections=emri_detections,
            )
            with mp.Pool() as pool:
                posterior_distribution_with_truncnorm = pool.map(
                    bayesian_inference_with_truncnorm.posterior, hubble_values
                )
            likelihoods_with_truncnorm = np.array(posterior_distribution_with_truncnorm).T
            combined_posterior_with_truncnorm = np.prod(likelihoods_with_truncnorm, axis=0)
            ax.plot(
                hubble_values,
                combined_posterior_with_truncnorm / max(combined_posterior_with_truncnorm),
                color=color_map.to_rgba(i * NUMBER_OF_NEW_DETECTIONS_PER_STEP),  # type: ignore[arg-type]
                linestyle="dashdot",
            )
        print(f"Finished iteration {i + 1} of {STEPS} in {time() - start_time:.2f}s.")

    ax.vlines(
        TRUE_HUBBLE_CONSTANT,
        0,
        1,
        color="black",
        linestyles="dashed",
        label="True Hubble Constant",
    )
    color_map.set_array([])
    fig.colorbar(color_map, ax=ax, label="Number of detections")
    ax.legend()
    plt.show()
    plt.close(fig)
