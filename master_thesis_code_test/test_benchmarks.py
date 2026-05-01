"""Benchmark tests for performance-critical code paths.

Run with: pytest -m slow --benchmark-only
These are excluded from the fast CI path (marked as slow).
"""

from typing import Any

import pytest

from master_thesis_code.datamodels.galaxy import GalaxyCatalog


@pytest.mark.slow
def test_benchmark_galaxy_catalog_distribution(benchmark: Any) -> None:
    """Benchmark GalaxyCatalog.evaluate_galaxy_distribution for a catalog of 500 galaxies."""
    catalog = GalaxyCatalog(use_truncnorm=True, use_comoving_volume=False)
    for _ in range(500):
        catalog.add_random_galaxy()

    def run() -> None:
        catalog.evaluate_galaxy_distribution(0.3)

    benchmark(run)
