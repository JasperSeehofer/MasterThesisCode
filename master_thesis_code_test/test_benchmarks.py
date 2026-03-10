"""Benchmark tests for performance-critical code paths.

Run with: pytest -m slow --benchmark-only
These are excluded from the fast CI path (marked as slow).
"""

from typing import Any

import numpy as np
import pytest

from master_thesis_code.datamodels.emri_detection import EMRIDetection
from master_thesis_code.datamodels.galaxy import GalaxyCatalog


def _make_emri_detection(seed: int = 0) -> EMRIDetection:
    rng = np.random.default_rng(seed)
    return EMRIDetection(
        measured_luminosity_distance=float(rng.uniform(0.5, 5.0)),
        measured_redshifted_mass=float(rng.uniform(1e5, 1e6)),
        measured_right_ascension=float(rng.uniform(0, 2 * np.pi)),
        measured_declination=float(rng.uniform(0, np.pi)),
    )


@pytest.mark.slow
def test_benchmark_likelihood(benchmark: Any) -> None:
    """Benchmark BayesianInference.likelihood for a single detection."""
    from master_thesis_code.bayesian_inference.bayesian_inference import BayesianInference

    catalog = GalaxyCatalog(use_truncnorm=True, use_comoving_volume=False)
    for _ in range(200):
        catalog.add_random_galaxy()

    detection = _make_emri_detection(0)
    inference = BayesianInference(
        galaxy_catalog=catalog,
        emri_detections=[detection],
    )

    def run() -> float:
        return inference.likelihood(
            hubble_constant=0.73,
            measured_luminosity_distance=detection.measured_luminosity_distance,
            measured_redshifted_mass=detection.measured_redshifted_mass,
            detection_index=0,
        )

    benchmark(run)


@pytest.mark.slow
def test_benchmark_galaxy_catalog_distribution(benchmark: Any) -> None:
    """Benchmark GalaxyCatalog.evaluate_galaxy_distribution for a catalog of 500 galaxies."""
    catalog = GalaxyCatalog(use_truncnorm=True, use_comoving_volume=False)
    for _ in range(500):
        catalog.add_random_galaxy()

    def run() -> None:
        catalog.evaluate_galaxy_distribution(0.3)

    benchmark(run)
