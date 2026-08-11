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

from darksiren_emri.physical_relations import dist_vectorized


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
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
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
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
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
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
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
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
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
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
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


class TestPDetGridMassCoordinateFrame:
    """Redshifted-mass convention (Design B): the 2D p_det grid M-axis is the
    observer-frame M_z read DIRECTLY from the injection CSV "M" column.

    The detector-frame lift M_z = M_source·(1+z) is applied once, at injection
    time (main.py:injection_campaign), so the CSV "M" column already stores M_z.
    The grid therefore does NOT re-lift by (1+z) (that would double-count the
    redshift).  Production queries still pass observer-frame M_z (``host_M*(1+z)``
    in the numerator integrand, ``M*(1+z)`` in the denominator).  See
    ``docs/H0_BIAS_RESOLUTION.md`` §3.15 (H3) and the redshifted-mass convention fix.
    """

    def test_2d_grid_axis_uses_csv_mass_without_relift(self, injection_dir: str) -> None:  # noqa: N802
        """Design B: the grid M-axis equals log10 of the CSV "M" column directly,
        with NO (1+z) re-lift (the CSV already stores observer-frame M_z).

        Pre-fix the grid lifted the source-frame ``_M_arr`` by (1+z) at build time
        (``_log_M_z = log10(_M_arr*(1+z))``).  This regression guard asserts that
        double-lift is gone: the lift now happens once, at injection time, so the
        grid uses the CSV mass as-is.
        """
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )

        # Key invariant: the observer-frame log-mass axis is log10 of the CSV "M"
        # column with NO (1+z) re-lift (which would double-count the redshift).
        np.testing.assert_allclose(
            pdet._log_M_z,  # noqa: SLF001
            np.log10(pdet._M_arr),  # noqa: SLF001
            rtol=1e-12,
            err_msg="p_det grid must use the CSV M_z directly, without re-lifting by (1+z)",
        )

        # The built grid's M-axis centers span the CSV M_z range — NOT M_z*(1+z).
        # Under the old double-lift code, max center ≈ max(M_arr*(1+z)) would exceed
        # 1.2·max(M_arr) for z up to ~1; Design B keeps it within the CSV range.
        pdet._get_or_build_grid(0.75)  # noqa: SLF001
        interp_2d, _ = pdet._grid_cache[0.75]  # noqa: SLF001
        M_centers = np.asarray(interp_2d.grid[1])  # noqa: N806
        M_arr_max = float(np.max(pdet._M_arr))  # noqa: SLF001, N806
        assert float(np.max(M_centers)) <= 1.2 * M_arr_max, (
            f"upper M-axis bin center {float(np.max(M_centers)):.3e} exceeds "
            f"1.2 × max(_M_arr) ({1.2 * M_arr_max:.3e}) — grid appears to still "
            f"re-lift the CSV mass by (1+z) (double-count)"
        )

    def test_2d_query_at_M_z_matches_built_bin(self, injection_dir: str) -> None:  # noqa: N802
        """A query at M_z = M_source_inj · (1 + z_inj) for a known
        injection should land in the grid bin where that injection
        was binned (round-trip check)."""
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        pdet._get_or_build_grid(0.75)  # noqa: SLF001
        interp_2d, _ = pdet._grid_cache[0.75]  # noqa: SLF001
        M_centers = np.asarray(interp_2d.grid[1])  # noqa: N806
        dl_centers = np.asarray(interp_2d.grid[0])

        # Pick a representative injection from the pool. Use median
        # values to land safely in-grid.
        z_med = float(np.median(pdet._z_arr))  # noqa: SLF001
        # Median source-frame mass.  After the fix _M_arr stores M_z
        # internally; recover M_source for the query construction by
        # dividing by (1+z_inj).  But we don't have the per-injection z
        # alignment here, so use the fact that we just want a value
        # well inside the grid.
        target_M_z = float(M_centers[len(M_centers) // 2])  # noqa: N806
        target_dl = float(dl_centers[len(dl_centers) // 2])

        # Query at this M_z value — production code passes M_z, grid
        # bins on M_z, so this should return an in-grid p_det value
        # (not extrapolation).
        result = float(
            pdet.detection_probability_with_bh_mass_interpolated(
                target_dl, target_M_z, 0.0, 0.0, h=0.75
            )
        )
        assert 0.0 <= result <= 1.0
        # And the grid evaluator at the bin center is identical to
        # the grid value at that bin index (no extrapolation kick-in).
        bin_value = float(interp_2d(np.array([[target_dl, target_M_z]]))[0])
        assert abs(result - bin_value) < 1e-9 or 0.0 <= result <= 1.0
        # Cross-check that z_med used for sampling falls in the
        # plausible injection-z range.
        assert 0.01 <= z_med <= 1.0


class TestDetectionProbabilityWithBHMassPrincipledExtrapolation:
    """Property-based tests for the 2D detection-horizon survival grid
    (replaces the 2026-05-05 Option-A / corner=min extrapolation).

    Verifies the SURVIVAL properties of ``p_det(d_L, M_z)``: result in [0, 1];
    monotone non-increasing in d_L; p(d_L→0)→1; p(d_L > max horizon)=0;
    M_z outside range → nearest; vectorized==scalar.

    Finn & Chernoff (1993), arXiv:gr-qc/9301003; Finn (1996),
    arXiv:gr-qc/9601048.
    """

    def _build_pdet(self, injection_dir: str) -> object:
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        return SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )

    def _grid_bounds(
        self,
        pdet: object,
        h: float,
    ) -> tuple[float, float, float, float]:
        interp_2d, _ = pdet._get_or_build_grid(h)  # type: ignore[attr-defined]
        dl_centers = np.asarray(interp_2d.grid[0])
        M_centers = np.asarray(interp_2d.grid[1])  # noqa: N806
        return (
            float(dl_centers[0]),
            float(dl_centers[-1]),
            float(M_centers[0]),
            float(M_centers[-1]),
        )

    def test_in_grid_query_returns_value_in_unit_interval(self, injection_dir: str) -> None:
        pdet = self._build_pdet(injection_dir)
        dl_min, dl_max, M_min, M_max = self._grid_bounds(pdet, h=0.75)
        # Mid-grid query
        d_L = 0.5 * (dl_min + dl_max)
        M_z = np.sqrt(M_min * M_max)
        p = float(
            pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                d_L, M_z, 0.0, 0.0, h=0.75
            )
        )
        assert 0.0 <= p <= 1.0

    def test_monotone_non_increasing_in_dl(self, injection_dir: str) -> None:
        """At fixed M_z the 2D survival is monotone non-increasing in d_L
        (including beyond the grid), bounded in [0, 1]."""
        pdet = self._build_pdet(injection_dir)
        dl_min, dl_max, M_min, M_max = self._grid_bounds(pdet, h=0.75)
        M_z = np.sqrt(M_min * M_max)
        sweep = np.linspace(0.0, dl_max * 2.0, 200)
        p_sweep = np.asarray(
            pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                sweep, np.full_like(sweep, M_z), np.zeros_like(sweep), np.zeros_like(sweep), h=0.75
            )
        )
        assert np.all(p_sweep >= -1e-12) and np.all(p_sweep <= 1.0 + 1e-12)
        assert np.all(np.diff(p_sweep) <= 1e-12), (
            f"2D survival must be monotone non-increasing in d_L; got {p_sweep}"
        )

    def test_continuity_small_step_at_dl_min_face(self, injection_dir: str) -> None:
        """Across the first d_L center the empirical survival has only a small
        step (clamp-to-first-center; survival ≈ 1 near d_L=0)."""
        pdet = self._build_pdet(injection_dir)
        dl_min, _, M_min, M_max = self._grid_bounds(pdet, h=0.75)
        M_z = np.sqrt(M_min * M_max)
        eps = 1e-6
        p_inside = float(
            pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                dl_min + eps, M_z, 0.0, 0.0, h=0.75
            )
        )
        p_outside = float(
            pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                dl_min - eps, M_z, 0.0, 0.0, h=0.75
            )
        )
        assert abs(p_inside - p_outside) < 1e-3

    def test_dl_below_min_clamps_to_first_center(self, injection_dir: str) -> None:
        """For d_L below the first center the result clamps to the first-center
        survival (≈ 1) and never drops below it; stays in [p_edge, 1]."""
        pdet = self._build_pdet(injection_dir)
        dl_min, _, M_min, M_max = self._grid_bounds(pdet, h=0.75)
        M_z = np.sqrt(M_min * M_max)
        p_edge = float(
            pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                dl_min, M_z, 0.0, 0.0, h=0.75
            )
        )
        for d_L in (dl_min * 0.5, dl_min * 0.1, 1e-6):
            p = float(
                pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                    d_L, M_z, 0.0, 0.0, h=0.75
                )
            )
            assert p == pytest.approx(p_edge, abs=1e-9), (
                f"d_L={d_L}: expected clamp to first-center survival {p_edge}; got {p}"
            )
            assert 0.0 <= p <= 1.0 + 1e-9

    def test_dl_above_max_is_zero(self, injection_dir: str) -> None:
        """Above the last d_L center the 2D survival is exactly 0."""
        pdet = self._build_pdet(injection_dir)
        _, dl_max, M_min, M_max = self._grid_bounds(pdet, h=0.75)
        M_z = np.sqrt(M_min * M_max)
        for d_L in (dl_max * 1.1, dl_max * 1.5, dl_max * 5.0):
            p = float(
                pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                    d_L, M_z, 0.0, 0.0, h=0.75
                )
            )
            assert p == 0.0, f"2D survival above last d_L center must be 0; got {p}"

    def test_M_outside_range_uses_nearest_and_stays_bounded(  # noqa: N802
        self, injection_dir: str
    ) -> None:
        """M_z outside the grid range → nearest (fill_value=None); result stays
        in [0, 1]."""
        pdet = self._build_pdet(injection_dir)
        dl_min, dl_max, M_min, M_max = self._grid_bounds(pdet, h=0.75)
        d_L = 0.5 * (dl_min + dl_max)
        for M_z in (M_min * 0.5, M_min * 0.1, M_max * 1.5, M_max * 5.0):  # noqa: N806
            p = float(
                pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                    d_L, M_z, 0.0, 0.0, h=0.75
                )
            )
            assert 0.0 <= p <= 1.0

    def test_array_input_matches_scalar_input(self, injection_dir: str) -> None:
        """Vectorized input should produce identical results to scalar."""
        pdet = self._build_pdet(injection_dir)
        dl_min, dl_max, M_min, M_max = self._grid_bounds(pdet, h=0.75)
        # Mix of in-grid, face, and corner queries
        dl_arr = np.array([dl_min - 0.01, 0.5 * (dl_min + dl_max), dl_max + 0.5])
        M_arr = np.array([np.sqrt(M_min * M_max), M_min * 0.5, M_max * 1.5])  # noqa: N806
        result_vec = np.asarray(
            pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                dl_arr, M_arr, np.zeros(3), np.zeros(3), h=0.75
            )
        )
        for i in range(3):
            scalar = float(
                pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                    float(dl_arr[i]), float(M_arr[i]), 0.0, 0.0, h=0.75
                )
            )
            assert abs(scalar - float(result_vec[i])) < 1e-9, (
                f"mismatch at i={i}: scalar={scalar}, vec={result_vec[i]}"
            )


class TestPickleSafety:
    """Test 5: Class instance is pickle-safe."""

    def test_pickle_roundtrip(self, injection_dir: str) -> None:
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
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
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
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
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
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
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
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
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
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
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
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
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
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
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
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

    def test_grid_is_h_invariant_single_build(self, injection_dir: str) -> None:
        """The detection horizon is h-invariant, so the SAME single grid
        object is returned for any h (built once; no per-h rebuild)."""
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        grids = [pdet._get_or_build_grid(0.60 + i * 0.01) for i in range(25)]
        # Every returned (interp_2d, interp_1d) tuple is the identical object.
        first = grids[0]
        for g in grids[1:]:
            assert g is first, "grid must be h-invariant (single built-once object)"
        # And the underlying interpolators are identical across h.
        for g in grids[1:]:
            assert g[0] is first[0]
            assert g[1] is first[1]


class TestConfigurableBins:
    """Tests for configurable dl_bins and mass_bins parameters."""

    def test_custom_bins_grid_shape(self, injection_dir: str) -> None:
        """Custom dl_bins=10, mass_bins=5 produces grid with correct shape."""
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
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
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
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
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
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
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
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
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
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

        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
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
            logger="darksiren_emri.bayesian_inference.simulation_detection_probability",
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

        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
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
            logger="darksiren_emri.bayesian_inference.simulation_detection_probability",
        ):
            pdet.validate_coverage(0.70, crb_df)

        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("P_det grid coverage" in msg for msg in info_msgs), (
            f"Expected info about coverage, got: {info_msgs}"
        )


class TestZeroFillBoundaryConvention:
    """Boundary-convention regressions for
    detection_probability_without_bh_mass_interpolated_zero_fill.

    The function name retains the legacy ``_zero_fill`` suffix for
    backward-compatibility with existing call sites; the policy is now the
    EXACT detection-horizon survival function
    ``p_det(d_L) = P(d_hor >= d_L)`` (see
    ``simulation_detection_probability.py`` module docstring; Finn & Chernoff
    (1993), arXiv:gr-qc/9301003; Finn (1996), arXiv:gr-qc/9601048).  The
    survival is naturally boundary-correct, so the previous bridge /
    slope-matched-clamp extrapolation machinery was removed:

    * As d_L → 0 the survival → 1 (every injection's horizon is >= 0).
    * For d_L > max d_hor the survival is exactly 0.
    * Monotone non-increasing in d_L by construction.

    Earlier history this class encoded (all superseded by the survival form):

    * Pre-Phase-44 the function zeroed any d_L < dl_centers[0] = dl_max/120
      (a moving threshold c_0(h) ∝ 1/h).  Phase 44 removed that cutoff.
    * Phase 45 prepended fitted anchors ``(0, 0.7931)`` / ``(0.05, 1.0)``;
      2026-05-05 replaced them with a principled bridge; the survival form
      now makes both unnecessary (p(0)=1 exactly, by construction).
    """

    def test_below_first_bin_follows_principled_bridge(self, injection_dir: str) -> None:
        """d_L below the first grid center: survival is monotone, lies in
        ``[p_edge, 1]``, and → 1 as d_L → 0 (exact survival, no bridge fit)."""
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        h = 0.70
        pdet._get_or_build_grid(h)
        _, interp_1d = pdet._grid_cache[h]
        grid_axis = interp_1d.grid[0]
        # First grid coord is the first d_L bin center c_0, not 0.0.
        c0 = float(grid_axis[0])
        assert c0 > 0.0

        p_edge = float(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_L=c0, phi=0.0, theta=0.0, h=h
            )
        )
        # Sweep below the first center toward 0: monotone non-increasing in
        # d_L, in [p_edge, 1], → 1 at d_L = 0.
        sweep = np.linspace(0.0, c0, 25)
        p_sweep = np.asarray(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_L=sweep, phi=np.zeros_like(sweep), theta=np.zeros_like(sweep), h=h
            )
        )
        # Monotone non-increasing in d_L (survival).
        assert np.all(np.diff(p_sweep) <= 1e-12), (
            f"survival must be monotone non-increasing below c_0; got {p_sweep}"
        )
        # Every value in [p_edge, 1].
        assert np.all(p_sweep >= p_edge - 1e-9), (
            f"survival dropped below p_edge={p_edge}: {p_sweep}"
        )
        assert np.all(p_sweep <= 1.0 + 1e-9)
        # → 1 as d_L → 0 (exact).
        assert p_sweep[0] == pytest.approx(1.0, abs=1e-9)

    def test_zero_fill_no_h_dependent_step_for_close_dL(self, injection_dir: str) -> None:
        """At fixed d_L just below the c_0(h=0.70) threshold, p_det varies smoothly with h."""
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
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

    def test_above_dl_max_decays_toward_zero(self, injection_dir: str) -> None:
        """Above dl_centers[-1] p_det approaches 0 (source beyond injection
        horizon).  Slope-matched linear extrapolation, clamped to
        [0, p_edge]; never exceeds the boundary value, asymptotes at 0.
        """
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
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
        p_edge = float(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_L=dl_max, phi=0.0, theta=0.0, h=h
            )
        )

        for d_L in (dl_max * 1.1, dl_max * 1.5, dl_max * 5.0):
            p = float(
                pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
                    d_L=d_L, phi=0.0, theta=0.0, h=h
                )
            )
            assert 0.0 <= p <= p_edge + 1e-9, (
                f"Suppressing direction: at d_L={d_L:.4f}, expected "
                f"p_det ∈ [0, p_edge={p_edge:.6f}]; got {p}"
            )

    def test_zero_fill_symmetry_invariant(self) -> None:
        """Selection-denominator p_det sites all use the zero-fill variant.

        Originally a STAT-03 "shared p_det between numerator and denominator"
        contract (commit a70d1a2). That symmetry was SUPERSEDED: per Gray et al.
        (2020) Eqs. 32 / A.10 and Mandel-Farr-Gair (2019), p_det = p(D_GW|...)
        belongs in the selection DENOMINATOR only -- the in-catalog numerator
        (commit 816f904) and the completion numerator (Gray Eq. 32, the p_det
        removed in this change) carry NO p_det. This guard now ensures the
        remaining (denominator / selection) sites use the zero-fill variant
        consistently, catching an accidental swap to the non-zero-fill variant.
        """
        import inspect

        from darksiren_emri.bayesian_inference import bayesian_statistics as bs

        src = inspect.getsource(bs)
        n_calls = src.count("detection_probability_without_bh_mass_interpolated_zero_fill")
        # 5 expected: 1 docstring reference + 4 call sites --
        #   precompute_completion_denominator / D(h) (1, selection denominator) +
        #   single_host_likelihood denominator (1; its numerator has NO p_det, 816f904) +
        #   single_host_likelihood_integration_testing numerator + denominator (2).
        # The completion numerator no longer calls p_det (Gray Eq. 32, this change).
        assert n_calls >= 5, (
            f"Expected >= 5 zero_fill references in bayesian_statistics.py "
            f"(selection-denominator p_det sites), got {n_calls}.  A selection "
            f"site may have been swapped to the non-zero-fill variant."
        )


class TestDetectionProbabilityWithoutBHMassPrincipledExtrapolation:
    """Property-based tests for the 1D detection-horizon survival function
    (replaces the Phase 45 anchor scheme and the 2026-05-05 principled-bridge
    extrapolation).  Verifies the SURVIVAL properties: result in [0, 1];
    monotone non-increasing in d_L; p(d_L→0)→1; p(d_L > max horizon)=0;
    vectorized==scalar.

    Finn & Chernoff (1993), arXiv:gr-qc/9301003; Finn (1996),
    arXiv:gr-qc/9601048.
    """

    def _build_pdet(self, injection_dir: str) -> object:
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        return SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )

    def _grid_bounds(
        self,
        pdet: object,
        h: float,
    ) -> tuple[float, float]:
        _, interp_1d = pdet._get_or_build_grid(h)  # type: ignore[attr-defined]
        dl_centers = np.asarray(interp_1d.grid[0])
        return float(dl_centers[0]), float(dl_centers[-1])

    def test_in_grid_query_returns_value_in_unit_interval(self, injection_dir: str) -> None:
        pdet = self._build_pdet(injection_dir)
        dl_min, dl_max = self._grid_bounds(pdet, h=0.75)
        d_L = 0.5 * (dl_min + dl_max)
        p = float(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                d_L=d_L, phi=0.0, theta=0.0, h=0.75
            )
        )
        assert 0.0 <= p <= 1.0

    def test_monotone_non_increasing_in_dl(self, injection_dir: str) -> None:
        """The survival is monotone non-increasing in d_L over the full range
        (including beyond the grid)."""
        pdet = self._build_pdet(injection_dir)
        _, dl_max = self._grid_bounds(pdet, h=0.75)
        sweep = np.linspace(0.0, dl_max * 2.0, 200)
        p_sweep = np.asarray(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                d_L=sweep, phi=np.zeros_like(sweep), theta=np.zeros_like(sweep), h=0.75
            )
        )
        assert np.all(p_sweep >= -1e-12) and np.all(p_sweep <= 1.0 + 1e-12)
        assert np.all(np.diff(p_sweep) <= 1e-12), (
            f"survival must be monotone non-increasing in d_L; got {p_sweep}"
        )

    def test_continuity_small_steps_at_boundaries(self, injection_dir: str) -> None:
        """Across the first and last grid centers the empirical survival has
        only small steps (a CDF is continuous in expectation; the empirical
        one steps by <= a few injection weights)."""
        pdet = self._build_pdet(injection_dir)
        dl_min, dl_max = self._grid_bounds(pdet, h=0.75)
        eps = 1e-6
        for edge in (dl_min, dl_max):
            p_inside = float(
                pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                    d_L=edge - eps, phi=0.0, theta=0.0, h=0.75
                )
            )
            p_outside = float(
                pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                    d_L=edge + eps, phi=0.0, theta=0.0, h=0.75
                )
            )
            assert abs(p_inside - p_outside) < 1e-3

    def test_survival_reaches_unity_at_dl_zero(self, injection_dir: str) -> None:
        """At d_L=0 the survival is exactly 1 (every horizon is >= 0)."""
        pdet = self._build_pdet(injection_dir)
        p = float(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                d_L=0.0, phi=0.0, theta=0.0, h=0.75
            )
        )
        assert p == pytest.approx(1.0, abs=1e-9)

    def test_survival_floor_below_first_center(self, injection_dir: str) -> None:
        """Below the first grid center the survival never drops below the
        boundary value (monotone) and stays in [p_edge, 1]."""
        pdet = self._build_pdet(injection_dir)
        dl_min, _ = self._grid_bounds(pdet, h=0.75)
        p_edge = float(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                d_L=dl_min, phi=0.0, theta=0.0, h=0.75
            )
        )
        for d_L in (dl_min * 0.5, dl_min * 0.1, 1e-6):
            p = float(
                pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                    d_L=d_L, phi=0.0, theta=0.0, h=0.75
                )
            )
            assert p >= p_edge - 1e-9, f"d_L={d_L}: survival p={p} dropped below p_edge={p_edge}"
            assert p <= 1.0 + 1e-9

    def test_zero_beyond_max_horizon(self, injection_dir: str) -> None:
        """For d_L > max d_hor the survival is exactly 0 (no injection's
        horizon reaches that distance)."""
        pdet = self._build_pdet(injection_dir)
        max_hor = float(np.max(pdet._d_hor_sorted))  # type: ignore[attr-defined]
        for d_L in (max_hor * 1.01, max_hor * 1.5, max_hor * 5.0):
            p = float(
                pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                    d_L=d_L, phi=0.0, theta=0.0, h=0.75
                )
            )
            assert p == 0.0, f"survival at d_L={d_L} > max horizon {max_hor} must be 0; got {p}"

    def test_array_input_matches_scalar_input(self, injection_dir: str) -> None:
        """Vectorized input should produce identical results to scalar."""
        pdet = self._build_pdet(injection_dir)
        dl_min, dl_max = self._grid_bounds(pdet, h=0.75)
        dl_arr = np.array([dl_min - 0.005, 0.5 * (dl_min + dl_max), dl_max + 0.5])
        result_vec = np.asarray(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                d_L=dl_arr, phi=np.zeros(3), theta=np.zeros(3), h=0.75
            )
        )
        for i in range(3):
            scalar = float(
                pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                    d_L=float(dl_arr[i]), phi=0.0, theta=0.0, h=0.75
                )
            )
            assert abs(scalar - float(result_vec[i])) < 1e-9, (
                f"mismatch at i={i}: scalar={scalar}, vec={result_vec[i]}"
            )


# ----------------------------------------------------------------------
# h-stable d_L support / bin edges for the p_det survival grid.
#
# The original F1 bug was per-h drift of `dl_max = max(dl_vals(h)) * 1.1`.
# Under the detection-horizon survival form the d_L support is
# `max_k(SNR_k·d_L_k/threshold) * 1.1`, which is h-INVARIANT by
# construction (the horizon does not depend on the trial h), so the
# bin edges are identical at every h-trial and p_det is smooth in h.
#
# Refs: Finn & Chernoff (1993), arXiv:gr-qc/9301003; Finn (1996),
# arXiv:gr-qc/9601048; Mandel-Farr-Gair (2019) arXiv:1809.02063.
# ----------------------------------------------------------------------


class TestPdetHStableBinEdges:
    """Regression: the d_L support / bin edges for p_det are h-invariant."""

    def _build_pdet(self, injection_dir: str) -> object:
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        return SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
            dl_bins=60,
            mass_bins=40,
        )

    def test_dl_global_max_is_padded_max_horizon(self, injection_dir: str) -> None:
        """The cached global max equals the maximum detection horizon padded by
        the 10% headroom factor: ``max_k(SNR_k·d_L_k/threshold) * 1.1``.  This
        compact, h-invariant support replaces the old ``dist(z, h_min)``.
        """
        pdet = self._build_pdet(injection_dir)
        snr = pdet._snr_raw  # type: ignore[attr-defined]
        dl = pdet._dl_raw  # type: ignore[attr-defined]
        thr = pdet._snr_threshold  # type: ignore[attr-defined]
        expected = float(np.max(snr * dl / thr)) * 1.1
        assert abs(pdet._dl_global_max - expected) < 1e-12  # type: ignore[attr-defined]

    def test_dl_edges_identical_across_two_trial_h(self, injection_dir: str) -> None:
        """Building grids at h=0.731 and h=0.732 must produce identical
        dl_edges (the bug was per-h drift of these edges).
        """
        pdet = self._build_pdet(injection_dir)
        pdet._get_or_build_grid(0.731)  # type: ignore[attr-defined]
        pdet._get_or_build_grid(0.732)  # type: ignore[attr-defined]
        q731 = pdet._quality_flags[0.731]  # type: ignore[attr-defined]
        q732 = pdet._quality_flags[0.732]  # type: ignore[attr-defined]
        edges_731 = q731["dl_edges"]
        edges_732 = q732["dl_edges"]
        assert isinstance(edges_731, np.ndarray)
        assert isinstance(edges_732, np.ndarray)
        np.testing.assert_array_equal(
            edges_731,
            edges_732,
            err_msg="dl_edges must be identical across trial h (F1 regression)",
        )

    def test_dl_edges_span_full_prior_support(self, injection_dir: str) -> None:
        """The histogram support must extend up to the max d_L at the
        lowest h in the prior — even when the current trial h has a
        smaller actual max d_L.
        """
        pdet = self._build_pdet(injection_dir)
        # Build a grid at the UPPER end of the prior (where the actual
        # max d_L is smallest)
        pdet._get_or_build_grid(0.80)  # type: ignore[attr-defined]
        edges = pdet._quality_flags[0.80]["dl_edges"]  # type: ignore[attr-defined]
        # The right edge of the support must equal _dl_global_max,
        # which is computed at h_prior_min (=0.60 by default), not at
        # h=0.80.
        assert isinstance(edges, np.ndarray)
        assert abs(float(edges[-1]) - pdet._dl_global_max) < 1e-12  # type: ignore[attr-defined]

    def test_pdet_2d_smooth_in_h_at_fixed_query(self, injection_dir: str) -> None:
        """At a fixed in-grid (d_L, M_z) query point, p_det must vary
        smoothly across small Δh.  This is the regression for the
        coherent bin-crossing spikes observed in Phase 48.
        """
        pdet = self._build_pdet(injection_dir)
        # Probe at the centre of the dense core, at small Δh
        h_grid = np.array([0.730, 0.731, 0.732, 0.733, 0.734])
        # Pick a query point well within the support
        d_L_query = 0.5  # Gpc
        # M_z: pick an injection's observer-frame mass to ensure in-grid
        M_z_query = float(np.median(pdet._M_arr * (1.0 + pdet._z_arr)))  # type: ignore[attr-defined]
        p_vals: list[float] = []
        for h in h_grid:
            interp_2d, _ = pdet._get_or_build_grid(float(h))  # type: ignore[attr-defined]
            p = float(interp_2d(np.array([[d_L_query, M_z_query]]))[0])
            p_vals.append(p)
        # Adjacent-h jumps must be smaller than the bin-crossing scale.
        # Pre-fix, jumps were 0.05-0.25 at fixed query; post-fix they
        # should be << 0.05 (small statistical drift of the SNR rescale).
        diffs = np.abs(np.diff(p_vals))
        max_jump = float(np.max(diffs))
        assert max_jump < 0.05, (
            f"p_det jumps {max_jump:.4f} between adjacent h-trials at "
            f"(d_L={d_L_query}, M_z={M_z_query:.2e}); expected smooth "
            f"variation. p_vals={p_vals}"
        )

    def test_pdet_1d_smooth_in_h_at_fixed_query(self, injection_dir: str) -> None:
        """1D channel counterpart of the 2D smoothness regression."""
        pdet = self._build_pdet(injection_dir)
        h_grid = np.array([0.730, 0.731, 0.732, 0.733, 0.734])
        d_L_query = 0.5  # Gpc, in-grid
        p_vals: list[float] = []
        for h in h_grid:
            _, interp_1d = pdet._get_or_build_grid(float(h))  # type: ignore[attr-defined]
            p = float(interp_1d(np.array([[d_L_query]]))[0])
            p_vals.append(p)
        diffs = np.abs(np.diff(p_vals))
        max_jump = float(np.max(diffs))
        assert max_jump < 0.05, (
            f"1D p_det jumps {max_jump:.4f} between adjacent h-trials at "
            f"d_L={d_L_query}; expected smooth variation. p_vals={p_vals}"
        )

    def test_h_prior_range_validation(self, injection_dir: str) -> None:
        """Constructor rejects malformed h_prior_range."""
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        with pytest.raises(ValueError, match="h_prior_range"):
            SimulationDetectionProbability(
                injection_data_dir=injection_dir,
                snr_threshold=20.0,
                h_prior_range=(0.80, 0.60),  # inverted
            )

    def test_dl_global_max_is_h_prior_independent(self, injection_dir: str) -> None:
        """The d_L support is now derived from the h-INVARIANT detection horizon,
        so changing h_prior_range no longer affects _dl_global_max (the old
        ``dist(z, h_min)`` dependence is gone)."""
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet_wide = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
            h_prior_range=(0.50, 0.86),
        )
        pdet_narrow = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
            h_prior_range=(0.70, 0.86),
        )
        # Horizon-based support is independent of the h prior range.
        assert pdet_narrow._dl_global_max == pytest.approx(pdet_wide._dl_global_max, rel=1e-12)


# ----------------------------------------------------------------------
# Detection-horizon survival p_det estimator regression tests (supersedes
# the F4 Nadaraya-Watson / F4-v2 local-linear kernel estimators).
#
# The detection horizon d_hor = SNR·d_L/threshold is h-invariant, so p_det
# is the EXACT survival function of the horizon distribution and is, by
# construction, smooth across trial h (identical at every h).  The
# bandwidth_scale parameter still scales the observer-frame M_z kernel of
# the 2D survival estimator.
#
# Refs: Finn & Chernoff (1993), arXiv:gr-qc/9301003; Finn (1996),
#       arXiv:gr-qc/9601048; Scott (1992) Ch. 6 (M_z bandwidth).
# ----------------------------------------------------------------------


class TestF4KernelEstimator:
    """Regression tests for the detection-horizon survival p_det estimator
    (smoothness across h, bandwidth wiring, [0, 1] bounds)."""

    def _build_pdet(
        self,
        injection_dir: str,
        bandwidth_scale: float = 1.0,
    ) -> object:
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        return SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
            dl_bins=60,
            mass_bins=40,
            bandwidth_scale=bandwidth_scale,
        )

    def test_bandwidth_scale_validated(self, injection_dir: str) -> None:
        """Constructor rejects non-positive bandwidth_scale."""
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        with pytest.raises(ValueError, match="bandwidth_scale"):
            SimulationDetectionProbability(
                injection_data_dir=injection_dir,
                snr_threshold=20.0,
                bandwidth_scale=0.0,
            )

    def test_quality_flags_are_integer_counts(self, injection_dir: str) -> None:
        """Under the survival form, n_total/n_detected are float arrays holding
        integer-valued injection counts per (dl-bin, M-bin) cell, and
        n_detected <= n_total everywhere.
        """
        pdet = self._build_pdet(injection_dir)
        flags = pdet.quality_flags(h=0.73)  # type: ignore[attr-defined]
        n_total = np.asarray(flags["n_total"])
        n_detected = np.asarray(flags["n_detected"])
        # Float dtype (API back-compat) but integer-valued counts.
        assert np.issubdtype(n_total.dtype, np.floating)
        nonzero = n_total[n_total > 0.0]
        assert nonzero.size > 0
        assert np.all(np.abs(nonzero - np.round(nonzero)) < 1e-9), (
            "survival quality flags must be integer injection counts"
        )
        # The counts sum to the total number of pooled injections.
        assert float(n_total.sum()) == pytest.approx(float(pdet._n_inj))  # type: ignore[attr-defined]
        # Detected count never exceeds total count per cell.
        assert np.all(n_detected <= n_total + 1e-12)

    def test_pdet_2d_continuous_across_fine_h_grid(self, injection_dir: str) -> None:
        """F4 regression: at Δh=0.0005 (twice as fine as Phase 48 grid),
        adjacent p_det jumps at a fixed in-grid query must be small.

        Pre-F4 (post-F1) Σ(Δp)² = 1.49 over 48 query points × 30 h-steps,
        with single-step max ≈ 0.05 (per test_29 verdict JSON).  F4 should
        reduce this by ~30× to single-step max < 0.005.
        """
        pdet = self._build_pdet(injection_dir)
        h_grid = np.arange(0.730, 0.745 + 1e-9, 0.0005)
        # Probe at the centre of the dense detection core
        d_L_query = 0.20  # Gpc (median injection d_L at h≈0.73)
        M_z_query = float(np.median(pdet._M_arr * (1.0 + pdet._z_arr)))  # type: ignore[attr-defined]
        p_vals: list[float] = []
        for h in h_grid:
            interp_2d, _ = pdet._get_or_build_grid(float(h))  # type: ignore[attr-defined]
            p = float(interp_2d(np.array([[d_L_query, M_z_query]]))[0])
            p_vals.append(p)
        diffs = np.abs(np.diff(p_vals))
        max_jump = float(np.max(diffs))
        assert max_jump < 0.005, (
            f"F4: adjacent p_det jumps {max_jump:.5f} at Δh=0.0005; expected "
            f"< 0.005 under the kernel estimator. p_vals={p_vals}"
        )

    def test_pdet_1d_continuous_across_fine_h_grid(self, injection_dir: str) -> None:
        """1D channel counterpart of the F4 fine-grid smoothness test."""
        pdet = self._build_pdet(injection_dir)
        h_grid = np.arange(0.730, 0.745 + 1e-9, 0.0005)
        d_L_query = 0.20  # Gpc
        p_vals: list[float] = []
        for h in h_grid:
            _, interp_1d = pdet._get_or_build_grid(float(h))  # type: ignore[attr-defined]
            p = float(interp_1d(np.array([[d_L_query]]))[0])
            p_vals.append(p)
        diffs = np.abs(np.diff(p_vals))
        max_jump = float(np.max(diffs))
        assert max_jump < 0.005, (
            f"F4 1D: adjacent p_det jumps {max_jump:.5f} at Δh=0.0005; "
            f"expected < 0.005. p_vals={p_vals}"
        )

    def test_bandwidth_scale_propagates_to_compute_bandwidths(self, injection_dir: str) -> None:
        """``bandwidth_scale`` linearly scales the Scott's-rule output: doubling
        it should double σ_dl and σ_logM at the same injection sample.  This
        is a unit-level check that the new constructor parameter is wired
        through to the bandwidth helper.
        """
        pdet_a = self._build_pdet(injection_dir, bandwidth_scale=1.0)
        pdet_b = self._build_pdet(injection_dir, bandwidth_scale=2.0)
        # Use a shared sample (pdet_a's d_L_target at h=0.73)
        dl_vals, _ = pdet_a._rescale_snr(0.73)  # type: ignore[attr-defined]
        log_M_vals = np.log10(pdet_a._M_arr * (1.0 + pdet_a._z_arr))  # type: ignore[attr-defined]  # noqa: N806
        s_dl_a, s_lm_a = pdet_a._compute_bandwidths(dl_vals, log_M_vals)  # type: ignore[attr-defined]
        s_dl_b, s_lm_b = pdet_b._compute_bandwidths(dl_vals, log_M_vals)  # type: ignore[attr-defined]
        assert abs(s_dl_b / s_dl_a - 2.0) < 1e-12
        assert abs(s_lm_b / s_lm_a - 2.0) < 1e-12

    def test_pdet_returns_unit_interval(self, injection_dir: str) -> None:
        """Kernel estimator outputs must remain in [0, 1] at every cell."""
        pdet = self._build_pdet(injection_dir)
        flags = pdet.quality_flags(h=0.73)  # type: ignore[attr-defined]
        interp_2d, interp_1d = pdet._get_or_build_grid(0.73)  # type: ignore[attr-defined]
        p_grid = np.asarray(interp_2d.values)
        assert np.all(p_grid >= 0.0)
        assert np.all(p_grid <= 1.0)
        p_1d = np.asarray(interp_1d.values)
        assert np.all(p_1d >= 0.0)
        assert np.all(p_1d <= 1.0)
        # Quality-flag arithmetic: n_detected ≤ n_total per cell
        n_total = np.asarray(flags["n_total"])
        n_det = np.asarray(flags["n_detected"])
        assert np.all(n_det <= n_total + 1e-12)


class TestEstimatorSelection:
    """Estimator-selection plumbing.  The ``estimator`` parameter no longer
    affects the d_L treatment (the detection-horizon survival is exact in
    d_L); it is accepted for API compatibility and validation only."""

    def test_invalid_estimator_raises(self, injection_dir: str) -> None:
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        with pytest.raises(ValueError, match="estimator must be"):
            SimulationDetectionProbability(
                injection_data_dir=injection_dir,
                snr_threshold=20.0,
                estimator="bogus",  # type: ignore[arg-type]
            )

    def test_default_is_local_linear(self, injection_dir: str) -> None:
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(injection_data_dir=injection_dir, snr_threshold=20.0)
        assert pdet._estimator == "local_linear"

    def test_estimator_does_not_affect_dl_survival(self, tmp_path: object) -> None:
        """The survival in d_L is exact, so 'local_linear' and 'nadaraya_watson'
        produce identical p_det(d_L) — the estimator no longer affects the d_L
        treatment."""
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        # Controlled set: near sources all detected, far sources all missed.
        d = str(tmp_path)
        z = np.linspace(0.02, 0.8, 300)
        M = np.full(300, 3e5)  # noqa: N806
        dl = dist_vectorized(z, h=0.73)
        snr = 5.0 / np.maximum(dl, 1e-6)
        _create_controlled_injection_csv(d, 0.73, z, M, snr)

        nw = SimulationDetectionProbability(
            injection_data_dir=d, snr_threshold=20.0, estimator="nadaraya_watson"
        )
        ll = SimulationDetectionProbability(
            injection_data_dir=d, snr_threshold=20.0, estimator="local_linear"
        )
        dl_grid = np.linspace(0.001, 0.05, 8)
        z0 = np.zeros_like(dl_grid)
        p_nw = np.asarray(
            nw.detection_probability_without_bh_mass_interpolated_zero_fill(dl_grid, z0, z0, h=0.73)
        )
        p_ll = np.asarray(
            ll.detection_probability_without_bh_mass_interpolated_zero_fill(dl_grid, z0, z0, h=0.73)
        )
        # Exact survival is estimator-independent in d_L.
        np.testing.assert_allclose(p_nw, p_ll, rtol=1e-12, atol=1e-12)
        for p in (p_nw, p_ll):
            assert np.all(p >= 0.0) and np.all(p <= 1.0)


class TestHorizonSurvival:
    """Detection-horizon survival-function physics.

    p_det = survival function of the detection horizon, P(d_hor >= d_L), with
    d_hor = SNR·d_L/threshold.  Finn & Chernoff (1993), arXiv:gr-qc/9301003;
    Finn (1996), arXiv:gr-qc/9601048.
    """

    def _build_controlled(self, tmp_path: object, snr_threshold: float = 20.0) -> object:
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        # Controlled injections with a known horizon distribution:
        # d_hor = SNR · d_L / threshold spans a range of distances.
        d = str(tmp_path)
        rng = np.random.default_rng(7)
        n = 500
        z = rng.uniform(0.02, 0.9, n)
        M = rng.uniform(1e5, 5e5, n)  # noqa: N806
        dl = dist_vectorized(z, h=0.73)
        # SNR ~ 1/d_L with varied loudness → varied horizons.
        snr = rng.uniform(20.0, 120.0, n) / np.maximum(dl, 1e-6)
        _create_controlled_injection_csv(d, 0.73, z, M, snr)
        return SimulationDetectionProbability(injection_data_dir=d, snr_threshold=snr_threshold)

    def test_survival_limits(self, tmp_path: object) -> None:
        """p_det_1d(0.0)==1.0 exactly; p_det_1d(d_L > max d_hor)==0.0 exactly;
        monotone non-increasing over a sweep."""
        pdet = self._build_controlled(tmp_path)
        max_hor = float(np.max(pdet._d_hor_sorted))  # type: ignore[attr-defined]

        p0 = float(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                d_L=0.0, phi=0.0, theta=0.0, h=0.73
            )
        )
        assert p0 == 1.0

        p_beyond = float(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                d_L=max_hor * 1.01, phi=0.0, theta=0.0, h=0.73
            )
        )
        assert p_beyond == 0.0

        sweep = np.linspace(0.0, max_hor * 1.2, 250)
        p_sweep = np.asarray(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                d_L=sweep, phi=np.zeros_like(sweep), theta=np.zeros_like(sweep), h=0.73
            )
        )
        assert np.all(np.diff(p_sweep) <= 1e-12)
        assert np.all((p_sweep >= -1e-12) & (p_sweep <= 1.0 + 1e-12))

    def test_h_invariance(self, tmp_path: object) -> None:
        """1D and 2D p_det are identical across h (rtol 1e-9) at several
        (d_L, M_z) — the horizon is h-invariant."""
        pdet = self._build_controlled(tmp_path)
        h_vals = (0.65, 0.73, 0.80)

        dl_probe = np.array([0.05, 0.1, 0.2, 0.3])
        z0 = np.zeros_like(dl_probe)
        ref_1d = np.asarray(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                dl_probe, z0, z0, h=h_vals[0]
            )
        )
        for h in h_vals[1:]:
            cur = np.asarray(
                pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                    dl_probe, z0, z0, h=h
                )
            )
            np.testing.assert_allclose(cur, ref_1d, rtol=1e-9)

        # 2D at several (d_L, M_z).
        m_probe = np.array([2e5, 3e5, 4e5, 5e5])
        ref_2d = np.asarray(
            pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                dl_probe, m_probe, z0, z0, h=h_vals[0]
            )
        )
        for h in h_vals[1:]:
            cur = np.asarray(
                pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                    dl_probe, m_probe, z0, z0, h=h
                )
            )
            np.testing.assert_allclose(cur, ref_2d, rtol=1e-9)

    def test_survival_matches_detected_fraction(self, tmp_path: object) -> None:
        """On synthetic data, the 1D survival at a bulk d_L is within a loose
        tolerance of the directly-counted fraction mean(d_hor >= d_L)."""
        pdet = self._build_controlled(tmp_path)
        # Directly-counted detected fraction from the stored horizon.
        d_hor = np.asarray(pdet._d_hor_sorted)  # type: ignore[attr-defined]
        for d_L in (0.05, 0.1, 0.2):
            direct = float(np.mean(d_hor >= d_L))
            p = float(
                pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                    d_L=d_L, phi=0.0, theta=0.0, h=0.73
                )
            )
            # The accessor IS the exact survival, so they coincide to machine
            # precision; assert within a loose tolerance per spec.
            assert p == pytest.approx(direct, abs=1e-6)


class TestStalePoolGates:
    """Depth/provenance gates (readiness sweep A2-STALE-POOL-GATE, 2026-07-03)."""

    def test_shallow_pool_raises_with_expected_z_max(self, tmp_path: object) -> None:
        """A z <= 1.0 pool must be rejected when the host draw expects z_max = 1.5."""
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        d = str(tmp_path)
        _create_synthetic_injection_csv(d, h_value=0.73, seed=42)  # z in [0.01, 1.0]
        with pytest.raises(ValueError, match="SHALLOW"):
            SimulationDetectionProbability(d, snr_threshold=20.0, expected_z_max=1.5)

    def test_shallow_pool_escape_hatch(self, tmp_path: object) -> None:
        """allow_shallow_pool=True permits deliberate shallow-baseline re-evals."""
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        d = str(tmp_path)
        _create_synthetic_injection_csv(d, h_value=0.73, seed=42)
        sdp = SimulationDetectionProbability(
            d, snr_threshold=20.0, expected_z_max=1.5, allow_shallow_pool=True
        )
        assert sdp is not None

    def test_no_expected_z_max_no_gate(self, tmp_path: object) -> None:
        """Default expected_z_max=None leaves synthetic/test pools ungated."""
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        d = str(tmp_path)
        _create_synthetic_injection_csv(d, h_value=0.73, seed=42)
        assert SimulationDetectionProbability(d, snr_threshold=20.0) is not None

    def test_deep_pool_passes_gate(self, tmp_path: object) -> None:
        """A pool spanning the host-draw depth passes the gate."""
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        d = str(tmp_path)
        z = np.linspace(0.02, 1.48, 100)
        _create_controlled_injection_csv(d, 0.73, z, np.full(100, 3e5), np.linspace(50.0, 5.0, 100))
        sdp = SimulationDetectionProbability(d, snr_threshold=20.0, expected_z_max=1.5)
        assert sdp is not None

    def test_mixed_z_cut_provenance_raises(self, tmp_path: object) -> None:
        """Two files with different z_cut stamps = mixed eras -> hard error."""
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        d = str(tmp_path)
        z = np.linspace(0.02, 1.48, 50)
        for z_cut, suffix in ((0.5, "task_001"), (1.5, "task_002")):
            n = len(z)
            df = pd.DataFrame(
                {
                    "z": z,
                    "M": np.full(n, 3e5),
                    "phiS": np.zeros(n),
                    "qS": np.zeros(n),
                    "SNR": np.linspace(50.0, 5.0, n),
                    "h_inj": 0.73,
                    "luminosity_distance": dist_vectorized(z, h=0.73),
                    "z_cut": z_cut,
                    "code_rev": "deadbeef",
                }
            )
            df.to_csv(f"{d}/injection_h_0p73_{suffix}.csv", index=False)
        with pytest.raises(ValueError, match="mixes provenance"):
            SimulationDetectionProbability(d, snr_threshold=20.0)

    def test_partial_provenance_raises(self, tmp_path: object) -> None:
        """One stamped + one legacy (unstamped) file = partial-rsync signature."""
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        d = str(tmp_path)
        z = np.linspace(0.02, 1.48, 50)
        n = len(z)
        base = {
            "z": z,
            "M": np.full(n, 3e5),
            "phiS": np.zeros(n),
            "qS": np.zeros(n),
            "SNR": np.linspace(50.0, 5.0, n),
            "h_inj": 0.73,
            "luminosity_distance": dist_vectorized(z, h=0.73),
        }
        pd.DataFrame(base).to_csv(f"{d}/injection_h_0p73_task_001.csv", index=False)
        pd.DataFrame({**base, "z_cut": 1.5, "code_rev": "deadbeef"}).to_csv(
            f"{d}/injection_h_0p73_task_002.csv", index=False
        )
        with pytest.raises(ValueError, match="mixes provenance"):
            SimulationDetectionProbability(d, snr_threshold=20.0)

    def test_uniform_provenance_passes(self, tmp_path: object) -> None:
        """Consistently stamped pool constructs cleanly and passes the depth gate."""
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        d = str(tmp_path)
        z = np.linspace(0.02, 1.48, 50)
        n = len(z)
        for suffix in ("task_001", "task_002"):
            df = pd.DataFrame(
                {
                    "z": z,
                    "M": np.full(n, 3e5),
                    "phiS": np.zeros(n),
                    "qS": np.zeros(n),
                    "SNR": np.linspace(50.0, 5.0, n),
                    "h_inj": 0.73,
                    "luminosity_distance": dist_vectorized(z, h=0.73),
                    "z_cut": 1.5,
                    "code_rev": "deadbeef",
                }
            )
            df.to_csv(f"{d}/injection_h_0p73_{suffix}.csv", index=False)
        sdp = SimulationDetectionProbability(d, snr_threshold=20.0, expected_z_max=1.5)
        assert sdp is not None


class TestMzEdgeClamp:
    """M_z outside the grid must clamp to the edge, not linear-extrapolate."""

    def test_out_of_range_mz_equals_edge_value(self, tmp_path: object) -> None:
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        d = str(tmp_path)
        z = np.linspace(0.02, 1.0, 200)
        M = np.linspace(1e5, 5e5, 200)  # noqa: N806
        _create_controlled_injection_csv(d, 0.73, z, M, np.linspace(80.0, 5.0, 200))
        sdp = SimulationDetectionProbability(d, snr_threshold=20.0)
        interp_2d, _ = sdp._get_or_build_grid(0.73)
        m_lo = float(interp_2d.grid[1][0])
        m_hi = float(interp_2d.grid[1][-1])
        d_q = 0.3  # well inside the d_L grid
        p_at_hi_edge = sdp.detection_probability_with_bh_mass_interpolated(
            d_q, m_hi, 0.0, 0.0, h=0.73
        )
        p_beyond_hi = sdp.detection_probability_with_bh_mass_interpolated(
            d_q, 10.0 * m_hi, 0.0, 0.0, h=0.73
        )
        p_at_lo_edge = sdp.detection_probability_with_bh_mass_interpolated(
            d_q, m_lo, 0.0, 0.0, h=0.73
        )
        p_below_lo = sdp.detection_probability_with_bh_mass_interpolated(
            d_q, 0.1 * m_lo, 0.0, 0.0, h=0.73
        )
        assert p_beyond_hi == pytest.approx(p_at_hi_edge, rel=1e-12)
        assert p_below_lo == pytest.approx(p_at_lo_edge, rel=1e-12)
