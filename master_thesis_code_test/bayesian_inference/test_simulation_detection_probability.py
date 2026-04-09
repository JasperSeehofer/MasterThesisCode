"""Tests for SimulationDetectionProbability class.

Tests verify that the class loads injection CSVs, pools all events, applies
SNR threshold at evaluation time, builds P_det grids via SNR rescaling, and
is pickle-safe for multiprocessing.
"""

import pickle

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from master_thesis_code.physical_relations import dist_vectorized


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
    # Compute luminosity distance using the actual cosmological model
    luminosity_distance = dist_vectorized(z, h=h_value)
    # Compute SNR: use a simple model where SNR ~ 1/d_L * intrinsic_loudness
    # Intrinsic loudness varies by source (mass-dependent etc.)
    intrinsic_loudness = rng.uniform(10.0, 80.0, n_rows)
    snr = intrinsic_loudness / np.maximum(luminosity_distance, 1e-10)

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


def _create_controlled_injection_csv(
    directory: str,
    h_value: float,
    z_vals: npt.NDArray[np.float64],
    M_vals: npt.NDArray[np.float64],
    snr_vals: npt.NDArray[np.float64],
    suffix: str = "task_001",
) -> None:
    """Create an injection CSV with fully controlled values."""
    h_label = f"{h_value:.2f}".replace(".", "p")
    n = len(z_vals)
    luminosity_distance = dist_vectorized(z_vals, h=h_value)

    df = pd.DataFrame(
        {
            "z": z_vals,
            "M": M_vals,
            "phiS": np.zeros(n),
            "qS": np.zeros(n),
            "SNR": snr_vals,
            "h_inj": h_value,
            "luminosity_distance": luminosity_distance,
        }
    )
    df.to_csv(f"{directory}/injection_h_{h_label}_{suffix}.csv", index=False)


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
    """Test 1: Constructor loads synthetic injection CSVs and pools events."""

    def test_constructor_loads_csvs_and_pools_events(self, injection_dir: str) -> None:
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        # Should have detected two h values
        assert len(pdet._h_values_found) == 2
        assert 0.70 in pdet._h_values_found
        assert 0.80 in pdet._h_values_found
        # Should have pooled all events (200 + 200 = 400)
        assert len(pdet._pooled_df) == 400

    def test_h_grid_deprecation_warning(self, injection_dir: str) -> None:
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        with pytest.warns(DeprecationWarning, match="h_grid.*deprecated"):
            SimulationDetectionProbability(
                injection_data_dir=injection_dir,
                snr_threshold=20.0,
                h_grid=[0.70, 0.80],
            )


class TestSNRThresholdAppliedAtEvaluation:
    """Test 2: SNR threshold applied at evaluation time via rescaling."""

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

        h_val = 0.70

        # Region A: high SNR events (z=0.1-0.3)
        dl_detected = dist_vectorized(z_detected, h=h_val)
        df_high = pd.DataFrame(
            {
                "z": z_detected,
                "M": M_detected,
                "phiS": rng.uniform(0, 2 * np.pi, n),
                "qS": rng.uniform(0, np.pi, n),
                "SNR": snr_high,
                "h_inj": h_val,
                "luminosity_distance": dl_detected,
            }
        )

        # Region B: low SNR events only (z=0.7-0.9)
        z_undetected = rng.uniform(0.7, 0.9, n)
        M_undetected = rng.uniform(2e5, 3e5, n)  # noqa: N806
        dl_undetected = dist_vectorized(z_undetected, h=h_val)
        df_low = pd.DataFrame(
            {
                "z": z_undetected,
                "M": M_undetected,
                "phiS": rng.uniform(0, 2 * np.pi, n),
                "qS": rng.uniform(0, np.pi, n),
                "SNR": snr_low,
                "h_inj": h_val,
                "luminosity_distance": dl_undetected,
            }
        )

        df = pd.concat([df_high, df_low], ignore_index=True)
        df.to_csv(f"{d}/injection_h_0p70_task_001.csv", index=False)

        pdet = SimulationDetectionProbability(
            injection_data_dir=d,
            snr_threshold=20.0,
        )

        # Query at region A center -- should have P_det > 0
        # Region A: z=0.1-0.3, d_L at h=0.70, center ~ dist(0.2, 0.70)
        dl_center_a = float(dist_vectorized(np.array([0.2]), h=h_val)[0])
        interp_2d, _ = pdet._get_or_build_grid(h_val)
        p_a = interp_2d(np.array([[dl_center_a, 2.5e5]]))[0]
        assert p_a > 0.0, f"Expected P_det > 0 in detected region, got {p_a}"

        # Query at region B center -- should have P_det == 0
        dl_center_b = float(dist_vectorized(np.array([0.8]), h=h_val)[0])
        p_b = interp_2d(np.array([[dl_center_b, 2.5e5]]))[0]
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
        result = pdet.detection_probability_without_bh_mass_interpolated(1.5, 1.0, 0.5, h=0.70)
        assert isinstance(result, float | np.floating)
        assert 0.0 <= float(result) <= 1.0


class TestDetectionProbabilityWithBHMass:
    """Test 4: detection_probability_with_bh_mass_interpolated works at any h."""

    def test_returns_valid_probability_at_any_h(self, injection_dir: str) -> None:
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        # h=0.75 is between the injection h values -- should work via rescaling
        result = pdet.detection_probability_with_bh_mass_interpolated(1.5, 3e5, 1.0, 0.5, h=0.75)
        assert isinstance(result, float | np.floating)
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
        # Pre-build a grid so the cache has content
        pdet.detection_probability_without_bh_mass_interpolated(1.5, 1.0, 0.5, h=0.70)

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


class TestPoolsAllHValues:
    """Test 7: Constructor pools events from ALL h values."""

    def test_pools_all_h_values(self, tmp_path: object) -> None:
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        d = str(tmp_path)
        n1, n2 = 150, 250
        _create_synthetic_injection_csv(d, h_value=0.70, n_rows=n1, seed=42)
        _create_synthetic_injection_csv(d, h_value=0.80, n_rows=n2, seed=123)

        pdet = SimulationDetectionProbability(
            injection_data_dir=d,
            snr_threshold=20.0,
        )
        # Total pooled count should equal sum of both files
        assert len(pdet._pooled_df) == n1 + n2
        # Both h values should be recorded
        assert set(pdet._h_values_found) == {0.70, 0.80}


class TestSNRRescalingConsistency:
    """Test 8: SNR rescaling produces consistent results.

    Verifies:
    1. Identity: P_det at h_inj matches per-h grid (no rescaling needed)
    2. Monotonicity: higher h -> lower d_L -> higher SNR -> higher P_det
    3. Numerical consistency: rescaled SNR values match expected d_L ratio
    """

    def test_snr_rescaling_identity(self, tmp_path: object) -> None:
        """When queried at h = h_inj, rescaling is identity (d_L ratio = 1)."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        d = str(tmp_path)
        h_inj = 0.70
        n = 500
        rng = np.random.default_rng(42)

        z = rng.uniform(0.05, 0.8, n)
        M = rng.uniform(1e5, 5e5, n)  # noqa: N806
        dl = dist_vectorized(z, h=h_inj)
        # SNR that gives a mix of detected/undetected
        snr = rng.uniform(5, 40, n)

        _create_controlled_injection_csv(d, h_inj, z, M, snr)

        pdet = SimulationDetectionProbability(
            injection_data_dir=d,
            snr_threshold=20.0,
        )

        # Query at the injection h -- rescaling ratio should be 1.0
        # Check that the internal rescaling produces exactly the raw SNR
        d_L_target, snr_rescaled = pdet._rescale_snr(h_inj)

        # d_L should match the injection d_L (recomputed from z, h_inj)
        np.testing.assert_allclose(d_L_target, dl, rtol=1e-10)
        # SNR should be unchanged
        np.testing.assert_allclose(snr_rescaled, snr, rtol=1e-10)

    def test_snr_rescaling_direction(self, tmp_path: object) -> None:
        """Higher h -> lower d_L -> higher SNR -> more detections.

        For a fixed source at redshift z:
          d_L(z, h) ~ 1/h  (from d_L = c(1+z)/(H0) * integral)
          SNR ~ 1/d_L ~ h

        So increasing h should increase SNR and thus P_det.
        """
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        d = str(tmp_path)
        h_inj = 0.70
        n = 1000
        rng = np.random.default_rng(42)

        z = rng.uniform(0.1, 0.5, n)
        M = rng.uniform(2e5, 4e5, n)  # noqa: N806
        # SNR near threshold to see a change
        snr = rng.uniform(15, 25, n)

        _create_controlled_injection_csv(d, h_inj, z, M, snr)

        pdet = SimulationDetectionProbability(
            injection_data_dir=d,
            snr_threshold=20.0,
        )

        # Rescale to higher h -> SNR should increase
        _, snr_at_high_h = pdet._rescale_snr(0.80)
        _, snr_at_low_h = pdet._rescale_snr(0.60)

        # On average, SNR at high h should be higher than at low h
        assert np.mean(snr_at_high_h) > np.mean(snr_at_low_h), (
            f"Expected SNR(h=0.80)={np.mean(snr_at_high_h):.2f} > "
            f"SNR(h=0.60)={np.mean(snr_at_low_h):.2f}"
        )

        # More events should be detected at high h
        n_det_high = np.sum(snr_at_high_h >= 20.0)
        n_det_low = np.sum(snr_at_low_h >= 20.0)
        assert n_det_high >= n_det_low, (
            f"Expected more detections at h=0.80 ({n_det_high}) than at h=0.60 ({n_det_low})"
        )

    def test_snr_rescaling_numerical_ratio(self, tmp_path: object) -> None:
        """Verify SNR rescaling matches the expected d_L ratio."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        d = str(tmp_path)
        h_inj = 0.70
        h_target = 0.80
        n = 100
        rng = np.random.default_rng(42)

        z = rng.uniform(0.1, 0.5, n)
        M = rng.uniform(2e5, 4e5, n)  # noqa: N806
        snr = rng.uniform(10, 50, n)

        _create_controlled_injection_csv(d, h_inj, z, M, snr)

        pdet = SimulationDetectionProbability(
            injection_data_dir=d,
            snr_threshold=20.0,
        )

        d_L_target, snr_rescaled = pdet._rescale_snr(h_target)

        # Expected: SNR_target = SNR_raw * d_L(z, h_inj) / d_L(z, h_target)
        d_L_inj = dist_vectorized(z, h=h_inj)
        d_L_tgt = dist_vectorized(z, h=h_target)
        expected_snr = snr * d_L_inj / d_L_tgt

        np.testing.assert_allclose(snr_rescaled, expected_snr, rtol=1e-10)
        np.testing.assert_allclose(d_L_target, d_L_tgt, rtol=1e-10)


class TestQualityFlags:
    """Test 9: Quality flags work with lazy grid construction."""

    def test_quality_flags_available_after_query(self, injection_dir: str) -> None:
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        # Quality flags should be populated after accessing them
        flags = pdet.quality_flags(h=0.70)
        assert "n_total" in flags
        assert "n_detected" in flags
        assert "reliable" in flags
        assert "dl_edges" in flags
        assert "M_edges" in flags
        assert "n_eff" in flags


class TestGridCaching:
    """Test 10: Grid caching and LRU eviction."""

    def test_cache_hit(self, injection_dir: str) -> None:
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        # First call builds the grid
        pdet._get_or_build_grid(0.70)
        assert 0.70 in pdet._grid_cache

        # Second call should be a cache hit (same object)
        interp1 = pdet._grid_cache[0.70]
        pdet._get_or_build_grid(0.70)
        interp2 = pdet._grid_cache[0.70]
        assert interp1 is interp2

    def test_lru_eviction(self, injection_dir: str) -> None:
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            _MAX_CACHE_SIZE,
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        # Fill cache beyond max size
        for i in range(_MAX_CACHE_SIZE + 5):
            h_val = 0.60 + i * 0.01
            pdet._get_or_build_grid(h_val)

        # Cache should not exceed max size
        assert len(pdet._grid_cache) <= _MAX_CACHE_SIZE


class TestConfigurableBins:
    """Tests for configurable dl_bins and mass_bins parameters."""

    def test_custom_bins_grid_shape(self, injection_dir: str) -> None:
        """Custom dl_bins=10, mass_bins=5 produces grid with correct shape."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
            dl_bins=10,
            mass_bins=5,
        )
        assert pdet._dl_bins == 10
        assert pdet._mass_bins == 5

        # Build a grid and verify quality flags shape
        flags = pdet.quality_flags(h=0.70)
        assert flags["n_total"].shape == (10, 5)
        assert flags["n_detected"].shape == (10, 5)

    def test_default_bins(self, injection_dir: str) -> None:
        """Default construction uses dl_bins=60, mass_bins=40."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        assert pdet._dl_bins == 60
        assert pdet._mass_bins == 40

    def test_pickle_preserves_bins(self, injection_dir: str) -> None:
        """Pickle roundtrip preserves custom bin counts."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
            dl_bins=15,
            mass_bins=8,
        )
        # Pre-warm a grid so cache has content
        pdet._get_or_build_grid(0.70)

        data = pickle.dumps(pdet)
        pdet_restored = pickle.loads(data)  # noqa: S301

        assert pdet_restored._dl_bins == 15
        assert pdet_restored._mass_bins == 8


class TestCoverageValidation:
    """Tests for validate_coverage() method."""

    def test_full_coverage(self, injection_dir: str) -> None:
        """All events well within grid -> coverage == 1.0."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        # Build grid to learn the d_L range
        pdet._get_or_build_grid(0.70)
        _, interp_1d = pdet._grid_cache[0.70]
        dl_centers = interp_1d.grid[0]
        dl_mid = float(dl_centers[len(dl_centers) // 2])

        # Create CRB DataFrame with events well inside grid
        crb_df = pd.DataFrame(
            {
                "luminosity_distance": [dl_mid] * 10,
                "delta_luminosity_distance_delta_luminosity_distance": [0.001] * 10,
            }
        )

        coverage = pdet.validate_coverage(0.70, crb_df)
        assert coverage == 1.0

    def test_partial_coverage(self, injection_dir: str) -> None:
        """Some events have 4-sigma d_L bounds outside grid -> coverage < 1.0."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        pdet._get_or_build_grid(0.70)
        _, interp_1d = pdet._grid_cache[0.70]
        dl_centers = interp_1d.grid[0]
        dl_max = float(dl_centers[-1])
        dl_mid = float(dl_centers[len(dl_centers) // 2])

        # Mix: 5 events inside, 5 events with huge sigma that extends beyond grid
        crb_df = pd.DataFrame(
            {
                "luminosity_distance": [dl_mid] * 5 + [dl_max] * 5,
                "delta_luminosity_distance_delta_luminosity_distance": [0.001] * 5
                + [dl_max**2] * 5,  # sigma = dl_max, so 4*sigma >> grid
            }
        )

        coverage = pdet.validate_coverage(0.70, crb_df)
        assert 0.0 < coverage < 1.0

    def test_coverage_warning_logged(
        self, injection_dir: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """WARNING logged when coverage < 95%."""
        import logging

        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        pdet._get_or_build_grid(0.70)
        _, interp_1d = pdet._grid_cache[0.70]
        dl_centers = interp_1d.grid[0]
        dl_max = float(dl_centers[-1])

        # All events have huge sigma -> all outside grid -> 0% coverage
        crb_df = pd.DataFrame(
            {
                "luminosity_distance": [dl_max] * 10,
                "delta_luminosity_distance_delta_luminosity_distance": [dl_max**2] * 10,
            }
        )

        with caplog.at_level(
            logging.WARNING,
            logger="master_thesis_code.bayesian_inference.simulation_detection_probability",
        ):
            pdet.validate_coverage(0.70, crb_df)

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("below 95%" in msg for msg in warning_msgs), (
            f"Expected warning about coverage, got: {warning_msgs}"
        )

    def test_coverage_info_logged(
        self, injection_dir: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """INFO log is always emitted with coverage percentage."""
        import logging

        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        pdet._get_or_build_grid(0.70)
        _, interp_1d = pdet._grid_cache[0.70]
        dl_centers = interp_1d.grid[0]
        dl_mid = float(dl_centers[len(dl_centers) // 2])

        crb_df = pd.DataFrame(
            {
                "luminosity_distance": [dl_mid] * 5,
                "delta_luminosity_distance_delta_luminosity_distance": [0.001] * 5,
            }
        )

        with caplog.at_level(
            logging.INFO,
            logger="master_thesis_code.bayesian_inference.simulation_detection_probability",
        ):
            pdet.validate_coverage(0.70, crb_df)

        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("P_det grid coverage" in msg for msg in info_msgs), (
            f"Expected info about coverage, got: {info_msgs}"
        )
