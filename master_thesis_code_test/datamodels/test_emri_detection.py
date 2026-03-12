"""Tests for datamodels/emri_detection.py."""

from master_thesis_code.datamodels.emri_detection import EMRIDetection
from master_thesis_code.datamodels.galaxy import Galaxy


def _make_galaxy(redshift: float = 0.1, mass: float = 1e5) -> Galaxy:
    return Galaxy(
        redshift=redshift,
        central_black_hole_mass=mass,
        right_ascension=1.0,
        declination=0.5,
    )


def test_from_host_galaxy_no_noise_float_fields() -> None:
    """Regression: measured_luminosity_distance and measured_redshifted_mass must be float."""
    galaxy = _make_galaxy()
    detection = EMRIDetection.from_host_galaxy(galaxy, use_measurement_noise=False)
    assert isinstance(detection.measured_luminosity_distance, float)
    assert isinstance(detection.measured_redshifted_mass, float)


def test_from_host_galaxy_no_noise_positive() -> None:
    galaxy = _make_galaxy(redshift=0.2)
    detection = EMRIDetection.from_host_galaxy(galaxy, use_measurement_noise=False)
    assert detection.measured_luminosity_distance > 0
    assert detection.measured_redshifted_mass > 0


def test_from_host_galaxy_sky_angles_preserved() -> None:
    galaxy = _make_galaxy()
    detection = EMRIDetection.from_host_galaxy(galaxy, use_measurement_noise=False)
    assert detection.measured_right_ascension == galaxy.right_ascension
    assert detection.measured_declination == galaxy.declination


def test_from_host_galaxy_with_noise_positive() -> None:
    """With noise, distance and mass should still be positive (high SNR scenario)."""
    import numpy as np

    rng = np.random.default_rng(42)
    galaxy = _make_galaxy(redshift=0.1, mass=1e5)
    detection = EMRIDetection.from_host_galaxy(galaxy, use_measurement_noise=True, rng=rng)
    assert detection.measured_luminosity_distance > 0
    assert detection.measured_redshifted_mass > 0
