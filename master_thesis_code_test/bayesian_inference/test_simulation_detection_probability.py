"""Tests for SimulationDetectionProbability class.

Tests verify that the class loads injection CSVs, applies SNR threshold at
evaluation time, builds P_det(z, M | h) grids, interpolates between h values,
and is pickle-safe for multiprocessing.
"""

import pickle

import numpy as np
import pandas as pd
import pytest


def _create_synthetic_injection_csv(
    directory: str,
    h_value: float,
    n_rows: int = 200,
    seed: int = 42,
) -> None:
    """Create a synthetic injection CSV with known properties."""
    rng = np.random.default_rng(seed)
    h_label = f"{h_value:.2f}".replace(".", "p")

    z = rng.uniform(0.01, 1.0, size=n_rows)
    M = rng.uniform(1e5, 5e5, size=n_rows)  # noqa: N806
    phiS = rng.uniform(0, 2 * np.pi, size=n_rows)  # noqa: N806
    qS = rng.uniform(0, np.pi, size=n_rows)  # noqa: N806
    # Mix of SNR values: half above threshold, half below
    snr = np.where(rng.random(n_rows) > 0.5, rng.uniform(25, 50, n_rows), rng.uniform(5, 15, n_rows))
    # Compute approximate luminosity distance (just for CSV columns; not used by grid builder)
    luminosity_distance = z * 4.0  # rough approximation in Gpc

    df = pd.DataFrame(
        {
            "z": z,
            "M": M,
            "phiS": phiS,
            "qS": qS,
            "SNR": snr,
            "h_inj": h_value,
            "luminosity_distance": luminosity_distance,
        }
    )
    df.to_csv(f"{directory}/injection_h_{h_label}_task_001.csv", index=False)


@pytest.fixture()
def injection_dir(tmp_path: object) -> str:
    """Create a temporary directory with synthetic injection CSVs for h=0.70 and h=0.80."""
    d = str(tmp_path)
    _create_synthetic_injection_csv(d, h_value=0.70, seed=42)
    _create_synthetic_injection_csv(d, h_value=0.80, seed=123)
    return d


@pytest.fixture()
def empty_dir(tmp_path: object) -> str:
    """Return an empty directory path."""
    return str(tmp_path)


class TestSimulationDetectionProbabilityConstruction:
    """Test 1: Constructor loads synthetic injection CSVs and builds internal grids."""

    def test_constructor_loads_csvs_and_builds_grids(self, injection_dir: str) -> None:
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        # Should have detected two h values
        assert len(pdet._h_grid) == 2
        assert 0.70 in pdet._h_grid
        assert 0.80 in pdet._h_grid
        # Should have interpolators for each h value
        assert len(pdet._interpolators) == 2
        assert len(pdet._interpolators_1d) == 2


class TestSNRThresholdAppliedAtEvaluation:
    """Test 2: SNR threshold applied at evaluation time."""

    def test_threshold_filters_events_correctly(self, tmp_path: object) -> None:
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        d = str(tmp_path)
        rng = np.random.default_rng(99)
        n = 500
        # All events at z ~ 0.1-0.3, M ~ 2e5-3e5
        z_detected = rng.uniform(0.1, 0.3, n)
        M_detected = rng.uniform(2e5, 3e5, n)  # noqa: N806
        # Half with high SNR, half with low SNR
        snr_high = np.full(n, 25.0)
        snr_low = np.full(n, 10.0)

        # Region A: high SNR events (z=0.1-0.3)
        df_high = pd.DataFrame(
            {
                "z": z_detected,
                "M": M_detected,
                "phiS": rng.uniform(0, 2 * np.pi, n),
                "qS": rng.uniform(0, np.pi, n),
                "SNR": snr_high,
                "h_inj": 0.70,
                "luminosity_distance": z_detected * 4.0,
            }
        )

        # Region B: low SNR events only (z=0.7-0.9)
        z_undetected = rng.uniform(0.7, 0.9, n)
        M_undetected = rng.uniform(2e5, 3e5, n)  # noqa: N806
        df_low = pd.DataFrame(
            {
                "z": z_undetected,
                "M": M_undetected,
                "phiS": rng.uniform(0, 2 * np.pi, n),
                "qS": rng.uniform(0, np.pi, n),
                "SNR": snr_low,
                "h_inj": 0.70,
                "luminosity_distance": z_undetected * 4.0,
            }
        )

        df = pd.concat([df_high, df_low], ignore_index=True)
        df.to_csv(f"{d}/injection_h_0p70_task_001.csv", index=False)

        pdet = SimulationDetectionProbability(
            injection_data_dir=d,
            snr_threshold=20.0,
            h_grid=[0.70],
        )

        # Query at region A center (z=0.2, M=2.5e5) -- should have P_det > 0
        p_a = pdet._interpolators[0.70](np.array([[0.2, 2.5e5]]))[0]
        assert p_a > 0.0, f"Expected P_det > 0 in detected region, got {p_a}"

        # Query at region B center (z=0.8, M=2.5e5) -- should have P_det == 0
        p_b = pdet._interpolators[0.70](np.array([[0.8, 2.5e5]]))[0]
        assert p_b == 0.0, f"Expected P_det == 0 in undetected region, got {p_b}"


class TestDetectionProbabilityWithoutBHMass:
    """Test 3: detection_probability_without_bh_mass_interpolated returns float in [0, 1]."""

    def test_returns_valid_probability(self, injection_dir: str) -> None:
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        # Use a luminosity distance that maps to a redshift within the grid
        # dist(z=0.3, h=0.70) ~ 1.5 Gpc approximately
        result = pdet.detection_probability_without_bh_mass_interpolated(
            1.5, 1.0, 0.5, h=0.70
        )
        assert isinstance(result, (float, np.floating))
        assert 0.0 <= float(result) <= 1.0


class TestDetectionProbabilityWithBHMass:
    """Test 4: detection_probability_with_bh_mass_interpolated with h interpolation."""

    def test_returns_valid_probability_interpolated_h(self, injection_dir: str) -> None:
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        # h=0.75 is between 0.70 and 0.80, should interpolate
        result = pdet.detection_probability_with_bh_mass_interpolated(
            1.5, 3e5, 1.0, 0.5, h=0.75
        )
        assert isinstance(result, (float, np.floating))
        assert 0.0 <= float(result) <= 1.0


class TestPickleSafety:
    """Test 5: Class instance is pickle-safe."""

    def test_pickle_roundtrip(self, injection_dir: str) -> None:
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        data = pickle.dumps(pdet)
        pdet_restored = pickle.loads(data)  # noqa: S301

        # Verify restored instance works
        result = pdet_restored.detection_probability_without_bh_mass_interpolated(
            1.5, 1.0, 0.5, h=0.70
        )
        assert 0.0 <= float(result) <= 1.0


class TestEmptyDirectory:
    """Test 6: Constructor with empty directory raises informative error."""

    def test_empty_dir_raises(self, empty_dir: str) -> None:
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        with pytest.raises(FileNotFoundError, match="No injection CSV files found"):
            SimulationDetectionProbability(
                injection_data_dir=empty_dir,
                snr_threshold=20.0,
            )
