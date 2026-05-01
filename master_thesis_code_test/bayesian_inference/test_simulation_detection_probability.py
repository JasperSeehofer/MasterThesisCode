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


class TestZeroFillBoundaryConvention:
    """Phase 44 regressions: detection_probability_without_bh_mass_interpolated_zero_fill
    must remain nonzero below the first bin centre (Phase 44 invariant) and
    zero only above the injection horizon.

    Pre-fix the function zeroed any d_L < dl_centers[0] = dl_max/120.  Because
    dl_max(h) ∝ 1/h, this created a moving threshold c_0(h) ∝ 1/h that produced
    a +145.7 log-unit MAP bias toward h_max for events with d_L ≈ c_0.

    Phase 45 amends ``test_zero_fill_below_first_bin_is_nonzero_for_valid_dL``:
    Plan 45-02 made the value at ``d_L = c_0/2`` the linear midpoint between
    ``_P_MAX_EMPIRICAL_ANCHOR`` and ``p̂(c_0)``.  Plan 45-04 (hybrid) extends the
    layout with an intermediate anchor at ``(0.05, 1.0)``; the value at
    ``d_L = c_0/2`` now follows the hybrid two-segment linear interp formula
    (linear from ``(0, 0.7931)`` to ``(0.05, 1.0)`` on ``[0, 0.05]``; linear
    from ``(0.05, 1.0)`` to ``(c_0, p̂(c_0))`` on ``[0.05, c_0]``).  The
    assertion is still fixture-independent.
    """

    def test_zero_fill_below_first_bin_is_nonzero_for_valid_dL(self, injection_dir: str) -> None:
        """d_L below dl_centers[0] follows the Phase 45 hybrid linear-interp formula.

        Fixture-independent: Plan 45-04 inserts an intermediate anchor at
        ``(_D_INTERMEDIATE_ANCHOR_GPC, _P_INTERMEDIATE_EMPIRICAL) = (0.05, 1.0)``
        when ``c_0(h) > 0.05``.  The value at ``d_L = c_0/2`` is therefore:
        - on segment ``[0, 0.05]`` (when ``c_0/2 < 0.05``):
          ``P_MAX + (P_INTERMEDIATE - P_MAX) * (c_0/2) / 0.05``
        - on segment ``[0.05, c_0]`` (when ``c_0/2 >= 0.05``):
          ``P_INTERMEDIATE + (p̂(c_0) - P_INTERMEDIATE) * (c_0/2 - 0.05) / (c_0 - 0.05)``
        - in the c_0 ≤ 0.05 fallback (no intermediate):
          ``0.5 * (P_MAX + p̂(c_0))`` (Plan 45-02 formula).
        """
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            _D_INTERMEDIATE_ANCHOR_GPC,
            _P_INTERMEDIATE_EMPIRICAL,
            _P_MAX_EMPIRICAL_ANCHOR,
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        h = 0.70
        pdet._get_or_build_grid(h)
        _, interp_1d = pdet._grid_cache[h]
        # Phase 45 layout: grid[0] = 0.0 (Plan 45-02 anchor); grid[1] is either
        # _D_INTERMEDIATE_ANCHOR_GPC (Plan 45-04 hybrid) OR c_0 (fallback when
        # c_0 ≤ _D_INTERMEDIATE_ANCHOR_GPC).
        grid_axis = interp_1d.grid[0]
        anchor_dl = float(grid_axis[0])
        assert anchor_dl == 0.0  # Plan 45-02 anchor at d_L=0
        if abs(float(grid_axis[1]) - _D_INTERMEDIATE_ANCHOR_GPC) < 1e-12:
            c0 = float(grid_axis[2])  # hybrid layout
        else:
            c0 = float(grid_axis[1])  # fallback layout
        assert c0 > 0.0

        d_query = 0.5 * c0
        p_below = pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
            d_L=d_query, phi=0.0, theta=0.0, h=h
        )
        p_at = float(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_L=c0, phi=0.0, theta=0.0, h=h
            )
        )

        # Plan 45-04 hybrid linear-interp formula (fixture-independent):
        if c0 <= _D_INTERMEDIATE_ANCHOR_GPC:
            # Fallback: Plan 45-02 single-anchor layout.
            expected_p_below = 0.5 * (_P_MAX_EMPIRICAL_ANCHOR + p_at)
        elif d_query > _D_INTERMEDIATE_ANCHOR_GPC:
            # Hybrid right segment: linear from (0.05, 1.0) to (c_0, p_at).
            expected_p_below = _P_INTERMEDIATE_EMPIRICAL + (p_at - _P_INTERMEDIATE_EMPIRICAL) * (
                d_query - _D_INTERMEDIATE_ANCHOR_GPC
            ) / (c0 - _D_INTERMEDIATE_ANCHOR_GPC)
        else:
            # Hybrid left segment: linear from (0, 0.7931) to (0.05, 1.0).
            expected_p_below = (
                _P_MAX_EMPIRICAL_ANCHOR
                + (_P_INTERMEDIATE_EMPIRICAL - _P_MAX_EMPIRICAL_ANCHOR)
                * d_query
                / _D_INTERMEDIATE_ANCHOR_GPC
            )

        assert float(p_below) == pytest.approx(expected_p_below, rel=1e-6), (
            f"Phase 45 hybrid: at d_L={d_query:.6f} (c_0={c0:.6f}), expected "
            f"{expected_p_below:.6f}; got p_below={p_below}"
        )
        # The first-bin estimate must be a real probability, not the old zero.
        assert 0.0 < float(p_below) <= 1.0, (
            f"p_det must be in (0, 1] for d_L < c_0 inside the first bin, got {p_below}"
        )

    def test_zero_fill_no_h_dependent_step_for_close_dL(self, injection_dir: str) -> None:
        """At fixed d_L just below the c_0(h=0.70) threshold, p_det varies smoothly with h."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        # Pick a d_L that straddles c_0(h) across the h grid pre-fix.
        # c_0(0.70) ≈ dl_max(0.70)/120; pick d_L slightly below that.
        pdet._get_or_build_grid(0.70)
        _, interp_70 = pdet._grid_cache[0.70]
        c0_70 = float(interp_70.grid[0][0])
        d_L_test = 0.5 * c0_70  # well inside the first bin, below c_0(0.70)

        p_vals: dict[float, float] = {}
        for h in (0.65, 0.70, 0.75, 0.80, 0.85):
            p_vals[h] = float(
                pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
                    d_L=d_L_test, phi=0.0, theta=0.0, h=h
                )
            )

        # All non-zero (pre-fix at least one would be 0 because of the moving threshold).
        assert all(p > 0.0 for p in p_vals.values()), (
            f"Phase 44: p_det must be nonzero for d_L={d_L_test:.4f} Gpc at all h, got {p_vals}"
        )

        # Largest consecutive Δ across h-grid below 0.20 (synthetic data is noisier than real).
        h_sorted = sorted(p_vals)
        diffs = [abs(p_vals[h2] - p_vals[h1]) for h1, h2 in zip(h_sorted[:-1], h_sorted[1:])]
        assert max(diffs) < 0.20, (
            f"Phase 44: p_det jumps too sharply across h grid (suggests resurfaced "
            f"h-dependent threshold artifact): {p_vals}"
        )

    def test_zero_fill_above_dl_max_remains_zero(self, injection_dir: str) -> None:
        """Above dl_centers[-1] p_det is zero (source beyond injection horizon)."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        h = 0.70
        pdet._get_or_build_grid(h)
        _, interp_1d = pdet._grid_cache[h]
        dl_max = float(interp_1d.grid[0][-1])

        result = pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
            d_L=dl_max * 1.5, phi=0.0, theta=0.0, h=h
        )
        assert float(result) == 0.0, (
            f"Phase 44: above dl_max the source is beyond the injection horizon; "
            f"p_det must be 0.  Got {result} for d_L={dl_max * 1.5:.4f}, "
            f"dl_max={dl_max:.4f}."
        )

    def test_zero_fill_symmetry_invariant(self) -> None:
        """STAT-03 contract: numerator and denominator paths in L_comp/L_cat must
        share the same p_det function (commit a70d1a2).  Phase 44 preserves this
        by editing the function body, not the call sites.

        This test catches accidental divergence (e.g. someone replacing one site
        with the non-zero-fill variant for "performance").
        """
        import inspect

        from master_thesis_code.bayesian_inference import bayesian_statistics as bs

        src = inspect.getsource(bs)
        n_calls = src.count("detection_probability_without_bh_mass_interpolated_zero_fill")
        # 6 expected: precompute_completion_denominator (1) +
        # p_Di.completion_numerator_integrand (1) +
        # single_host_likelihood (numerator + denominator = 2) +
        # single_host_likelihood_integration_testing (numerator + denominator = 2)
        # = 6 production sites.  Plus 1 docstring/comment reference allowed.
        assert n_calls >= 6, (
            f"Expected >= 6 zero_fill call sites in bayesian_statistics.py "
            f"(Phase 38 STAT-03 invariant, commit a70d1a2), got {n_calls}.  "
            f"Numerator/denominator symmetry may be broken."
        )


class TestPhase45EmpiricalAnchor:
    """Phase 45 regressions: detection_probability_without_bh_mass_interpolated_zero_fill
    must use the empirical anchor (0, _P_MAX_EMPIRICAL_ANCHOR) prepended to
    the histogram grid, so that on [0, c_0) the result is a linear
    interpolation between the empirical asymptote and the first bin centre.

    Pre-Phase-45 the interpolator linearly extrapolated through bins 0 and 1
    to ≈0.748 at d_L=0 (vs the empirical ≈1.0), biasing L_comp downward at
    low h and producing a residual MAP=0.7650 vs truth 0.73 on the
    412-event production posterior.  See
    ``.gpd/phases/45-p-det-first-bin-asymptote-fix/45-01-SUMMARY.md`` and
    ``scripts/bias_investigation/outputs/phase45/p_max_h_independence.json``
    for the empirical derivation (LR homogeneity p=0.199, pooled Wilson 95%
    lower bound 0.7931 across all h_inj groups).
    """

    def test_anchor_value_in_unit_interval(self) -> None:
        """`_P_MAX_EMPIRICAL_ANCHOR` is a float in [0, 1] (probability)."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            _P_MAX_EMPIRICAL_ANCHOR,
        )

        assert isinstance(_P_MAX_EMPIRICAL_ANCHOR, float), (
            f"_P_MAX_EMPIRICAL_ANCHOR must be float, got {type(_P_MAX_EMPIRICAL_ANCHOR)}"
        )
        assert 0.0 <= _P_MAX_EMPIRICAL_ANCHOR <= 1.0, (
            f"_P_MAX_EMPIRICAL_ANCHOR must be in [0, 1], got {_P_MAX_EMPIRICAL_ANCHOR}"
        )

    def test_anchor_value_within_wilson_ci(self) -> None:
        """`_P_MAX_EMPIRICAL_ANCHOR` lies in the pooled Wilson 95% CI from
        Plan 45-01 ([0.7931, 0.9418]).

        This test catches accidental edits that drift the constant outside
        the empirically-defensible range derived in Plan 45-01.  The
        recommended default (conservative) is the lower bound 0.7931 so the
        anchor cannot overshoot truth on production posteriors.
        """
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            _P_MAX_EMPIRICAL_ANCHOR,
        )

        # Wilson 95% CI from p_max_h_independence.json (pooled n=63/71).
        ci_lower = 0.7931
        ci_upper = 0.9418
        assert ci_lower <= _P_MAX_EMPIRICAL_ANCHOR <= ci_upper, (
            f"_P_MAX_EMPIRICAL_ANCHOR={_P_MAX_EMPIRICAL_ANCHOR} outside Plan 45-01 "
            f"Wilson 95% CI [{ci_lower}, {ci_upper}]; see "
            f"scripts/bias_investigation/outputs/phase45/p_max_h_independence.json."
        )

    def test_anchor_at_dL_zero_equals_empirical_constant(self, injection_dir: str) -> None:
        """interp(d_L=0; h=0.73) == _P_MAX_EMPIRICAL_ANCHOR exactly."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            _P_MAX_EMPIRICAL_ANCHOR,
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        # The synthetic fixture covers h=0.70 and h=0.80; query at h=0.73 is
        # SNR-rescaled internally by SimulationDetectionProbability.
        result = pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
            d_L=0.0, phi=0.0, theta=0.0, h=0.73
        )
        assert float(result) == pytest.approx(_P_MAX_EMPIRICAL_ANCHOR, rel=1e-9), (
            f"Phase 45: interp(d_L=0; h=0.73) must equal "
            f"_P_MAX_EMPIRICAL_ANCHOR={_P_MAX_EMPIRICAL_ANCHOR} exactly; "
            f"got {result}"
        )

    def test_anchor_h_independent(self, injection_dir: str) -> None:
        """interp(d_L=0; h) is the same constant for every h
        (the anchor is module-level, not per-h)."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            _P_MAX_EMPIRICAL_ANCHOR,
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )

        values: dict[float, float] = {}
        for h in (0.65, 0.70, 0.73, 0.80, 0.85):
            v = pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_L=0.0, phi=0.0, theta=0.0, h=h
            )
            values[h] = float(v)

        h_spread = max(values.values()) - min(values.values())
        assert h_spread < 1e-9, (
            f"Phase 45: interp(d_L=0) must be h-INDEPENDENT (the anchor is a "
            f"module-level scalar).  h-spread = {h_spread}, values = {values}"
        )
        for h, v in values.items():
            assert v == pytest.approx(_P_MAX_EMPIRICAL_ANCHOR, rel=1e-9), (
                f"interp(d_L=0; h={h}) = {v} != _P_MAX_EMPIRICAL_ANCHOR={_P_MAX_EMPIRICAL_ANCHOR}"
            )

    def test_interp_at_c0_unchanged_by_anchor(self, injection_dir: str) -> None:
        """interp(d_L=c_0) equals the unchanged histogram first-bin estimate p̂(c_0).

        The Phase 45 anchors at d_L=0 (Plan 45-02) and d_L=0.05 (Plan 45-04)
        must NOT perturb the value at the first histogram bin centre c_0:
        linear interpolation between two distinct endpoint values passes
        through both.  We verify this by reproducing the histogram logic from
        `_build_grid_1d` directly and comparing.
        """
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            _D_INTERMEDIATE_ANCHOR_GPC,
            SimulationDetectionProbability,
        )
        from master_thesis_code.physical_relations import dist_vectorized

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        h = 0.70
        pdet._get_or_build_grid(h)
        _, interp_1d = pdet._grid_cache[h]
        # Phase 45 layout: grid[0] = 0.0 anchor; grid[1] is either the
        # intermediate (0.05) or the first bin centre (fallback).
        grid_axis = interp_1d.grid[0]
        if abs(float(grid_axis[1]) - _D_INTERMEDIATE_ANCHOR_GPC) < 1e-12:
            c0 = float(grid_axis[2])  # hybrid layout
        else:
            c0 = float(grid_axis[1])  # fallback layout

        # Reproduce histogram first-bin estimate p̂(c_0) at h=0.70 directly.
        pooled = pdet._pooled_df
        dl_inj = pooled["luminosity_distance"].values
        snr_raw = pooled["SNR"].values
        z = pooled["z"].values
        dl_target = dist_vectorized(z, h=h)
        snr_rescaled = snr_raw * dl_inj / dl_target
        dl_max = float(np.max(dl_target)) * 1.1
        dl_edges = np.linspace(0, dl_max, 60 + 1)  # _DEFAULT_DL_BINS = 60
        total_counts, _ = np.histogram(dl_target, bins=dl_edges)
        detected_counts, _ = np.histogram(dl_target[snr_rescaled >= 20.0], bins=dl_edges)
        p_hat_c0 = (
            float(detected_counts[0]) / float(total_counts[0]) if total_counts[0] > 0 else 0.0
        )

        result = pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
            d_L=c0, phi=0.0, theta=0.0, h=h
        )
        assert float(result) == pytest.approx(p_hat_c0, rel=1e-9), (
            f"Phase 45: interp(d_L=c_0; h={h}) must equal p̂(c_0)={p_hat_c0:.10f} "
            f"(unchanged from histogram bin 0); got {result}"
        )

    def test_interp_below_c0_strictly_lifted(self) -> None:
        """On the PRODUCTION injection fixture (where p̂(c_0) ≈ 0.544 at h=0.73),
        interp(c_0/2; h=0.73) is strictly above p̂(c_0) and bounded above by
        the intermediate anchor value 1.0 (Plan 45-04 hybrid).

        This test uses the real `simulations/injections/` campaign rather
        than the synthetic 200-event fixture, because Plan 45-01 established
        empirically that p̂(c_0) ≈ 0.544 < 0.7931 ≈ _P_MAX_EMPIRICAL_ANCHOR
        at h=0.73 there.  On the synthetic fixture the magnitudes are
        dataset-specific and a strict-lift assertion would be fragile.

        Skipped if the production injection directory is not available
        (e.g. on CI machines without the dataset checked out).
        """
        import os
        import pathlib

        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            _D_INTERMEDIATE_ANCHOR_GPC,
            _P_INTERMEDIATE_EMPIRICAL,
            _P_MAX_EMPIRICAL_ANCHOR,
            SimulationDetectionProbability,
        )

        prod_dir = pathlib.Path("simulations/injections")
        if not prod_dir.is_dir() or not any(prod_dir.glob("injection_h_*.csv")):
            pytest.skip(
                "Production injection campaign at simulations/injections/ "
                "not available; this test only runs in the dev/research env."
            )
        # Sanity: ensure the test runs from a directory where simulations/
        # is reachable (i.e. project root).
        if not os.path.isabs(str(prod_dir)):
            prod_dir = pathlib.Path.cwd() / "simulations/injections"

        pdet = SimulationDetectionProbability(
            injection_data_dir=str(prod_dir),
            snr_threshold=20.0,
        )
        h = 0.73
        pdet._get_or_build_grid(h)
        _, interp_1d = pdet._grid_cache[h]
        # Phase 45 layout: grid[0]=0 anchor; grid[1] is either 0.05 (hybrid)
        # or c_0 (fallback). Read c_0 from the appropriate position.
        grid_axis = interp_1d.grid[0]
        if abs(float(grid_axis[1]) - _D_INTERMEDIATE_ANCHOR_GPC) < 1e-12:
            c0 = float(grid_axis[2])  # hybrid layout
        else:
            c0 = float(grid_axis[1])  # fallback layout
        d_L_test = 0.5 * c0

        p_at = float(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_L=c0, phi=0.0, theta=0.0, h=h
            )
        )
        p_below = float(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_L=d_L_test, phi=0.0, theta=0.0, h=h
            )
        )

        # On the production fixture, p̂(c_0) ≈ 0.544 < 0.7931 ≈ anchor — the
        # pre-Phase-45 bias-diagnosis precondition is preserved by the hybrid.
        assert p_at < _P_MAX_EMPIRICAL_ANCHOR, (
            f"Production fixture sanity: expected p̂(c_0)={p_at} < "
            f"_P_MAX_EMPIRICAL_ANCHOR={_P_MAX_EMPIRICAL_ANCHOR}; if this fails "
            "the pre-Phase-45 bias diagnosis no longer applies to this dataset."
        )
        # Plan 45-04 hybrid: at d_L = c_0/2 the value lies on one of the two
        # hybrid segments, both bounded above by _P_INTERMEDIATE_EMPIRICAL=1.0
        # and strictly above p̂(c_0).
        assert p_below > p_at, (
            f"Phase 45 hybrid: interp(c_0/2) must strictly exceed p̂(c_0)={p_at}; got {p_below}"
        )
        assert p_below <= _P_INTERMEDIATE_EMPIRICAL, (
            f"Phase 45 hybrid: interp(c_0/2) must be bounded above by the "
            f"intermediate anchor _P_INTERMEDIATE_EMPIRICAL={_P_INTERMEDIATE_EMPIRICAL}; "
            f"got {p_below}"
        )

    def test_docstring_states_linear_and_anchor(self) -> None:
        """The docstring of detection_probability_without_bh_mass_interpolated_zero_fill
        must state the boundary convention: linear interpolation lifted by an
        empirical anchor (Plan 45-02) plus a hybrid intermediate anchor at
        d_L=0.05 (Plan 45-04).
        """
        import inspect

        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        doc = inspect.getdoc(
            SimulationDetectionProbability.detection_probability_without_bh_mass_interpolated_zero_fill
        )
        assert doc is not None, "missing docstring"
        doc_lower = doc.lower()
        assert "linear" in doc_lower, (
            f"Phase 45: docstring must state 'linear' (interp/extrap), got:\n{doc}"
        )
        assert "empirical anchor" in doc_lower or "empirical anchors" in doc_lower, (
            f"Phase 45: docstring must reference 'empirical anchor', got:\n{doc}"
        )
        assert "phase 45" in doc_lower, f"Phase 45: docstring must cite 'Phase 45', got:\n{doc}"
        # Plan 45-04: docstring must additionally mention the hybrid layout.
        assert "hybrid" in doc_lower or "intermediate" in doc_lower, (
            f"Phase 45 Plan 45-04: docstring must mention 'hybrid' or "
            f"'intermediate' anchor, got:\n{doc}"
        )

    # ------------------------------------------------------------------
    # Plan 45-04 hybrid (intermediate anchor at d_L = 0.05 Gpc, value = 1.0)
    # ------------------------------------------------------------------

    def test_intermediate_value_at_005_equals_constant(self, injection_dir: str) -> None:
        """interp(d_L=0.05; h) == _P_INTERMEDIATE_EMPIRICAL=1.0 for all h with c_0(h) > 0.05."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            _D_INTERMEDIATE_ANCHOR_GPC,
            _P_INTERMEDIATE_EMPIRICAL,
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        for h in (0.65, 0.70, 0.73, 0.80, 0.85):
            pdet._get_or_build_grid(h)
            grid_axis = pdet._grid_cache[h][1].grid[0]
            # Hybrid layout precondition: c_0(h) > _D_INTERMEDIATE_ANCHOR_GPC.
            if abs(float(grid_axis[1]) - _D_INTERMEDIATE_ANCHOR_GPC) >= 1e-12:
                pytest.skip(
                    f"h={h}: c_0={float(grid_axis[1])} <= 0.05; intermediate anchor "
                    "absent (fallback layout); not the regime this test covers."
                )
            v = pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_L=_D_INTERMEDIATE_ANCHOR_GPC, phi=0.0, theta=0.0, h=h
            )
            assert float(v) == pytest.approx(_P_INTERMEDIATE_EMPIRICAL, rel=1e-12), (
                f"Plan 45-04: interp(d_L=0.05; h={h}) must equal "
                f"_P_INTERMEDIATE_EMPIRICAL={_P_INTERMEDIATE_EMPIRICAL}; got {v}"
            )

    def test_intermediate_h_independent(self, injection_dir: str) -> None:
        """h-spread at d_L=0.05 is exactly 0 (intermediate is a fixed scalar)."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            _D_INTERMEDIATE_ANCHOR_GPC,
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        values: dict[float, float] = {}
        for h in (0.65, 0.70, 0.73, 0.80, 0.85):
            v = pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_L=_D_INTERMEDIATE_ANCHOR_GPC, phi=0.0, theta=0.0, h=h
            )
            values[h] = float(v)
        spread = max(values.values()) - min(values.values())
        assert spread < 1e-12, (
            f"Plan 45-04: interp(d_L=0.05) must be h-INDEPENDENT (intermediate "
            f"is a module-level scalar at a fixed physical position).  "
            f"h-spread = {spread}, values = {values}"
        )

    def test_anchor_at_zero_unchanged(self, injection_dir: str) -> None:
        """interp(d_L=0; h) still equals _P_MAX_EMPIRICAL_ANCHOR=0.7931 (Plan 45-02 invariant)."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            _P_MAX_EMPIRICAL_ANCHOR,
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        for h in (0.65, 0.70, 0.73, 0.80, 0.85):
            v = pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_L=0.0, phi=0.0, theta=0.0, h=h
            )
            assert float(v) == pytest.approx(_P_MAX_EMPIRICAL_ANCHOR, rel=1e-12), (
                f"Plan 45-04: d_L=0 anchor invariant violated at h={h}; got {v}"
            )

    def test_interp_at_c0_unchanged_by_intermediate(self, injection_dir: str) -> None:
        """Plan 45-04 invariant: interp(d_L=c_0; h) still equals p̂(c_0).

        The intermediate anchor at d_L=0.05 is BETWEEN d_L=0 and d_L=c_0; the
        linear interpolant on [0.05, c_0] passes through both (0.05, 1.0) and
        (c_0, p̂(c_0)), so the value at c_0 is unchanged by Plan 45-04.
        """
        import numpy as np

        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            _D_INTERMEDIATE_ANCHOR_GPC,
            SimulationDetectionProbability,
        )
        from master_thesis_code.physical_relations import dist_vectorized

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        h = 0.70
        pdet._get_or_build_grid(h)
        _, interp_1d = pdet._grid_cache[h]
        grid_axis = interp_1d.grid[0]
        if abs(float(grid_axis[1]) - _D_INTERMEDIATE_ANCHOR_GPC) < 1e-12:
            c0 = float(grid_axis[2])  # hybrid
        else:
            c0 = float(grid_axis[1])  # fallback

        # Reproduce histogram p̂(c_0) directly.
        pooled = pdet._pooled_df
        z = pooled["z"].values
        snr_raw = pooled["SNR"].values
        dl_inj = pooled["luminosity_distance"].values
        dl_target = dist_vectorized(z, h=h)
        snr_rescaled = snr_raw * dl_inj / dl_target
        dl_max = float(np.max(dl_target)) * 1.1
        dl_edges = np.linspace(0, dl_max, 60 + 1)
        total_counts, _ = np.histogram(dl_target, bins=dl_edges)
        detected_counts, _ = np.histogram(dl_target[snr_rescaled >= 20.0], bins=dl_edges)
        p_hat_c0 = (
            float(detected_counts[0]) / float(total_counts[0]) if total_counts[0] > 0 else 0.0
        )

        v = pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
            d_L=c0, phi=0.0, theta=0.0, h=h
        )
        assert float(v) == pytest.approx(p_hat_c0, rel=1e-9), (
            f"Plan 45-04: interp(d_L=c_0; h={h}) must equal p̂(c_0)={p_hat_c0:.10f} "
            f"(Plan 45-02 invariant under hybrid; intermediate anchor at 0.05 "
            f"does not perturb the c_0 value); got {v}"
        )

    def test_below_intermediate_strict_lift(self) -> None:
        """interp(d_L=0.025; h=0.73) on production fixture equals the linear-interp
        midpoint of (0.7931, 1.0) = 0.89655 AND strictly exceeds the pre-hybrid value.
        """
        import json
        import pathlib

        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            _P_INTERMEDIATE_EMPIRICAL,
            _P_MAX_EMPIRICAL_ANCHOR,
            SimulationDetectionProbability,
        )

        prod_dir = pathlib.Path("simulations/injections")
        if not prod_dir.is_dir() or not any(prod_dir.glob("injection_h_*.csv")):
            pytest.skip("Production injection campaign not available.")

        # Read pre-hybrid baseline; required for the strict-lift assertion.
        pre_path = pathlib.Path(
            "scripts/bias_investigation/outputs/phase45/pre_hybrid_interp_probe.json"
        )
        if not pre_path.exists():
            pytest.fail(
                "pre_hybrid_interp_probe.json missing; run Plan 45-04 Task 1 first: "
                "uv run python -m scripts.bias_investigation.probe_interp_values "
                "--output-name pre_hybrid_interp_probe.json --label pre-hybrid"
            )
        pre = json.loads(pre_path.read_text())
        pre_value = next(
            r["value"] for r in pre["rows"] if r["h"] == 0.73 and r["d_L_gpc"] == 0.025
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=str(prod_dir),
            snr_threshold=20.0,
        )
        v = float(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_L=0.025, phi=0.0, theta=0.0, h=0.73
            )
        )
        expected = 0.5 * (_P_MAX_EMPIRICAL_ANCHOR + _P_INTERMEDIATE_EMPIRICAL)
        assert v == pytest.approx(expected, rel=1e-9), (
            f"Plan 45-04 linear-interp midpoint identity on [0, 0.05]: "
            f"interp(0.025; h=0.73) must equal 0.5*(0.7931+1.0)={expected}; got {v}"
        )
        assert v > pre_value, (
            f"Plan 45-04 strict lift: post-hybrid interp(0.025; h=0.73)={v} must "
            f"exceed pre-hybrid value {pre_value} from pre_hybrid_interp_probe.json"
        )

    def test_above_intermediate_strict_lift(self) -> None:
        """interp(d_L=0.075; h=0.73) on production fixture matches the linear-interp
        prediction on segment [0.05, c_0] AND strictly exceeds pre-hybrid value.
        """
        import json
        import pathlib

        import numpy as np

        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            _D_INTERMEDIATE_ANCHOR_GPC,
            _P_INTERMEDIATE_EMPIRICAL,
            SimulationDetectionProbability,
        )
        from master_thesis_code.physical_relations import dist_vectorized

        prod_dir = pathlib.Path("simulations/injections")
        if not prod_dir.is_dir() or not any(prod_dir.glob("injection_h_*.csv")):
            pytest.skip("Production injection campaign not available.")

        pre_path = pathlib.Path(
            "scripts/bias_investigation/outputs/phase45/pre_hybrid_interp_probe.json"
        )
        if not pre_path.exists():
            pytest.fail("pre_hybrid_interp_probe.json missing; run Plan 45-04 Task 1.")
        pre = json.loads(pre_path.read_text())
        pre_value = next(
            r["value"] for r in pre["rows"] if r["h"] == 0.73 and r["d_L_gpc"] == 0.075
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=str(prod_dir),
            snr_threshold=20.0,
        )
        h = 0.73
        pdet._get_or_build_grid(h)
        _, interp_1d = pdet._grid_cache[h]
        grid_axis = interp_1d.grid[0]
        c0 = float(grid_axis[2])  # hybrid layout (production fixture has c_0 > 0.05)

        # Reproduce p̂(c_0) for the prediction.
        pooled = pdet._pooled_df
        z = pooled["z"].values
        snr_raw = pooled["SNR"].values
        dl_inj = pooled["luminosity_distance"].values
        dl_target = dist_vectorized(z, h=h)
        snr_rescaled = snr_raw * dl_inj / dl_target
        dl_max = float(np.max(dl_target)) * 1.1
        dl_edges = np.linspace(0, dl_max, 60 + 1)
        total_counts, _ = np.histogram(dl_target, bins=dl_edges)
        detected_counts, _ = np.histogram(dl_target[snr_rescaled >= 20.0], bins=dl_edges)
        p_at = float(detected_counts[0]) / float(total_counts[0]) if total_counts[0] > 0 else 0.0

        v = float(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_L=0.075, phi=0.0, theta=0.0, h=h
            )
        )
        expected = _P_INTERMEDIATE_EMPIRICAL + (p_at - _P_INTERMEDIATE_EMPIRICAL) * (
            0.075 - _D_INTERMEDIATE_ANCHOR_GPC
        ) / (c0 - _D_INTERMEDIATE_ANCHOR_GPC)
        assert v == pytest.approx(expected, rel=1e-3), (
            f"Plan 45-04 linear-interp on [0.05, c_0]: interp(0.075; h=0.73)={v} "
            f"must match prediction {expected:.6f} (c_0={c0:.6f}, p̂(c_0)={p_at:.6f})"
        )
        assert v > pre_value, (
            f"Plan 45-04 strict lift: post-hybrid interp(0.075; h=0.73)={v} must "
            f"exceed pre-hybrid value {pre_value}"
        )

    def test_edge_case_small_c0_skips_intermediate(self, tmp_path: object) -> None:
        """When dl_max(h) is small enough that c_0 ≤ 0.05, the intermediate
        anchor must be SKIPPED (fallback to Plan 45-02 single-anchor layout)
        to preserve grid monotonicity.
        """
        import numpy as np
        import pandas as pd

        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            _D_INTERMEDIATE_ANCHOR_GPC,
            SimulationDetectionProbability,
        )

        # Synthesize an injection campaign with d_L bounded so c_0 < 0.05.
        # c_0 = dl_max / (2*N) with N=60 default; so we need dl_max < 6.0 Gpc.
        # 1.0 * 1.1 = 1.1 Gpc → c_0 ≈ 0.00917 << 0.05.
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "z": rng.uniform(0.01, 0.10, size=n),
                "M": rng.uniform(1e6, 1e7, size=n),
                "phiS": rng.uniform(0.0, 2.0 * np.pi, size=n),
                "qS": rng.uniform(0.0, np.pi, size=n),
                "SNR": rng.uniform(15.0, 40.0, size=n),
                "h_inj": np.full(n, 0.73),
                "luminosity_distance": rng.uniform(0.01, 1.0, size=n),
            }
        )
        tmp_dir = str(tmp_path)
        df.to_csv(f"{tmp_dir}/injection_h_0p73_task_0.csv", index=False)

        pdet = SimulationDetectionProbability(
            injection_data_dir=tmp_dir,
            snr_threshold=20.0,
        )
        # Use a very high h so dl_max(h) shrinks and c_0 < 0.05.
        h = 0.85
        pdet._get_or_build_grid(h)
        _, interp_1d = pdet._grid_cache[h]
        grid_axis = interp_1d.grid[0]
        # Intermediate must NOT be present in the grid when c_0 ≤ 0.05.
        c0 = float(grid_axis[1])  # under fallback, grid[0]=0.0, grid[1]=c_0
        assert c0 <= _D_INTERMEDIATE_ANCHOR_GPC, (
            f"Test premise violated: synthetic fixture has c_0={c0} > 0.05; "
            "the fallback branch wasn't exercised."
        )
        # Structural test: 0.05 must NOT appear in the grid axis (this is the
        # fixture-independent invariant of the fallback branch).
        assert _D_INTERMEDIATE_ANCHOR_GPC not in grid_axis.tolist(), (
            f"Plan 45-04 fallback: when c_0 ≤ 0.05, the intermediate anchor "
            f"must be skipped; grid={grid_axis.tolist()[:5]}..."
        )
        # No exception must be raised when querying past c_0 (regression
        # against accidental strict-monotonicity violations on the grid).
        v = float(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_L=_D_INTERMEDIATE_ANCHOR_GPC, phi=0.0, theta=0.0, h=h
            )
        )
        # Value must remain a valid probability (Phase 44 + Phase 45 invariant).
        assert 0.0 <= v <= 1.0, (
            f"Plan 45-04 fallback: interp(0.05) under c_0 ≤ 0.05 must remain in [0, 1]; got {v}"
        )
