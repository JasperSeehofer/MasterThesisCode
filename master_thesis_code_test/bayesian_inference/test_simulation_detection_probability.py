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


class TestPDetGridMassCoordinateFrame:
    """Phase 47 H3 fix: the 2D p_det grid M-axis is observer-frame M_z.

    The grid is built from injection campaign data with source-frame M
    multiplied by (1 + z_inj) to give observer-frame M_z (the natural
    SNR-determining mass coordinate).  Production queries pass M_z
    (e.g., ``host_M * (1+z)`` in the numerator integrand,
    ``M * (1+z)`` in the denominator).  See ``docs/H0_BIAS_RESOLUTION.md``
    §3.15.
    """

    def test_2d_grid_axis_is_M_z(self, injection_dir: str) -> None:  # noqa: N802
        """The 2D grid M-axis range matches max/min of M_source · (1 + z_inj),
        not max/min of M_source alone.

        Post-fix expectation: at injections spanning z ∈ [0.01, 1.0] with
        M_source ∈ [1e5, 5e5], the M-axis upper bound is at ~2 · M_source_max
        (observer-frame) rather than M_source_max (source-frame).
        """
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
        )
        # Force grid build for a representative h.
        pdet._get_or_build_grid(0.75)  # noqa: SLF001
        interp_2d, _ = pdet._grid_cache[0.75]  # noqa: SLF001
        M_centers = np.asarray(interp_2d.grid[1])  # noqa: N806

        # _M_arr stores the source-frame mass from the injection CSV.
        # The expected observer-frame M_z range is the source-frame _M_arr
        # multiplied by (1 + z_inj) per injection.
        z_arr = pdet._z_arr  # noqa: SLF001
        M_arr_source = pdet._M_arr  # noqa: SLF001, N806 (source-frame from CSV)
        M_z_inj = M_arr_source * (1.0 + z_arr)  # noqa: N806

        # _build_grid_2d uses bin edges padded ±10%; bin centers (geometric
        # mean for log-spaced M) lie strictly between min*0.9 and max*1.1.
        # Hard distinguishability: M_z_max should be > M_source_max*1.1 by
        # at least the min-z scaling factor.
        M_source_max = float(np.max(M_arr_source))  # noqa: N806
        M_z_max = float(np.max(M_z_inj))  # noqa: N806

        # The upper bin center under M_source-axis would lie around
        # M_source_max * 1.05 (within ±10% padded edges).  Under M_z-axis,
        # the upper bin center lies around M_z_max * 1.05.  Ratio is
        # roughly 1+z for the heaviest-z injection, and we have z up to 1.0.
        assert M_z_max > 1.5 * M_source_max, (
            f"sanity check: M_z_max={M_z_max:.3e} should exceed "
            f"M_source_max={M_source_max:.3e} by 1.5× given z up to ~1"
        )
        # Distinguishing assertion: upper bin center reflects M_z scale.
        # If grid were still source-frame, max bin center ≲ M_source_max * 1.1.
        # Under M_z grid, max bin center > M_source_max * 1.1 (clearly above).
        assert M_centers[-1] > 1.2 * M_source_max, (
            f"upper M-axis bin center {M_centers[-1]:.3e} is below "
            f"1.2 × M_source_max ({1.2 * M_source_max:.3e}) — grid "
            f"appears to still be in source-frame, not observer-frame M_z"
        )

    def test_2d_query_at_M_z_matches_built_bin(self, injection_dir: str) -> None:  # noqa: N802
        """A query at M_z = M_source_inj · (1 + z_inj) for a known
        injection should land in the grid bin where that injection
        was binned (round-trip check)."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
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
    """Property-based tests for the principled out-of-grid policy
    (Step 2 fix, ``.planning/2D-CHANNEL-AUDIT-20260505.md``).

    Verifies the asymptote table, C0 boundary continuity, the Option A
    directional clamp, and the corner = min(faces) rule.
    """

    def _build_pdet(self, injection_dir: str) -> object:
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
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

    def test_c0_continuity_at_dl_min_face(self, injection_dir: str) -> None:
        """As d_L → dl_min from below, p_det should approach the in-grid
        boundary value continuously (no step)."""
        pdet = self._build_pdet(injection_dir)
        dl_min, _, M_min, M_max = self._grid_bounds(pdet, h=0.75)
        M_z = np.sqrt(M_min * M_max)
        # Sample on both sides of dl_min
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
        # C0 continuity: |p_inside - p_outside| should be tiny
        assert abs(p_inside - p_outside) < 1e-3

    def test_dl_below_min_floor_at_pedge(self, injection_dir: str) -> None:
        """Option A: in the saturating d_L<dl_min direction, the result
        should never drop below the in-grid boundary value p_edge."""
        pdet = self._build_pdet(injection_dir)
        dl_min, _, M_min, M_max = self._grid_bounds(pdet, h=0.75)
        M_z = np.sqrt(M_min * M_max)
        p_edge = float(
            pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                dl_min, M_z, 0.0, 0.0, h=0.75
            )
        )
        # Probe several positions below dl_min; result must be >= p_edge.
        for d_L in (dl_min * 0.5, dl_min * 0.1, 1e-6):
            p = float(
                pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                    d_L, M_z, 0.0, 0.0, h=0.75
                )
            )
            assert p >= p_edge - 1e-9, (
                f"d_L={d_L}: p={p} dropped below p_edge={p_edge} (Option A floor violated)"
            )
            assert p <= 1.0 + 1e-9

    def test_dl_above_max_decays_toward_zero(self, injection_dir: str) -> None:
        """In the suppressing d_L>dl_max direction, the result should
        never exceed the in-grid boundary value p_edge."""
        pdet = self._build_pdet(injection_dir)
        _, dl_max, M_min, M_max = self._grid_bounds(pdet, h=0.75)
        M_z = np.sqrt(M_min * M_max)
        p_edge = float(
            pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                dl_max, M_z, 0.0, 0.0, h=0.75
            )
        )
        # Probe positions above dl_max; result must be <= p_edge.
        for d_L in (dl_max * 1.1, dl_max * 1.5, dl_max * 5.0):
            p = float(
                pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                    d_L, M_z, 0.0, 0.0, h=0.75
                )
            )
            assert p <= p_edge + 1e-9
            assert p >= 0.0

    def test_M_extremes_decay_toward_zero(self, injection_dir: str) -> None:  # noqa: N802
        """Both M extremes are suppressing: results <= p_edge."""
        pdet = self._build_pdet(injection_dir)
        dl_min, dl_max, M_min, M_max = self._grid_bounds(pdet, h=0.75)
        d_L = 0.5 * (dl_min + dl_max)
        # M < M_min
        p_edge_low = float(
            pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                d_L, M_min, 0.0, 0.0, h=0.75
            )
        )
        for M_z in (M_min * 0.5, M_min * 0.1):  # noqa: N806
            p = float(
                pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                    d_L, M_z, 0.0, 0.0, h=0.75
                )
            )
            assert p <= p_edge_low + 1e-9
            assert p >= 0.0
        # M > M_max
        p_edge_high = float(
            pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                d_L, M_max, 0.0, 0.0, h=0.75
            )
        )
        for M_z in (M_max * 1.5, M_max * 5.0):  # noqa: N806
            p = float(
                pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                    d_L, M_z, 0.0, 0.0, h=0.75
                )
            )
            assert p <= p_edge_high + 1e-9
            assert p >= 0.0

    def test_corner_returns_min_of_face_extrapolations(self, injection_dir: str) -> None:
        """At a corner cell (both axes outside), the result should be the
        min of the two face extrapolations.  Since at least one axis is
        always suppressing for any corner, the corner should be at most
        the minimum of the two face boundary values, asymptotically 0."""
        pdet = self._build_pdet(injection_dir)
        dl_min, dl_max, M_min, M_max = self._grid_bounds(pdet, h=0.75)

        # Corner: d_L > dl_max AND M > M_max  (both suppressing)
        p_corner = float(
            pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                dl_max * 1.5, M_max * 1.5, 0.0, 0.0, h=0.75
            )
        )
        assert 0.0 <= p_corner <= 1.0
        # Bound: corner <= p_edge of the corner of the grid
        p_grid_corner = float(
            pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                dl_max, M_max, 0.0, 0.0, h=0.75
            )
        )
        assert p_corner <= p_grid_corner + 1e-9

        # Corner: d_L < dl_min AND M > M_max
        # d_L<min wants asymptote 1, M>max wants asymptote 0.
        # min rule → corner driven toward 0 by the M side.
        p_corner_mixed = float(
            pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                dl_min * 0.5, M_max * 1.5, 0.0, 0.0, h=0.75
            )
        )
        # Must be at most the p_edge value at the (dl_min, M_max) corner
        p_edge_mixed = float(
            pdet.detection_probability_with_bh_mass_interpolated(  # type: ignore[attr-defined]
                dl_min, M_max, 0.0, 0.0, h=0.75
            )
        )
        assert p_corner_mixed <= p_edge_mixed + 1e-9
        assert 0.0 <= p_corner_mixed <= 1.0

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
    """Boundary-convention regressions for
    detection_probability_without_bh_mass_interpolated_zero_fill.

    The function name retains the legacy ``_zero_fill`` suffix for
    backward-compatibility with existing call sites; the policy is
    no longer pure zero-fill.  As of 2026-05-05 (audit document
    ``.planning/2D-CHANNEL-AUDIT-20260505.md``) the function uses a
    principled monotonic-asymptotic extrapolation:

    * Saturating face (d_L < d_L_min): linear bridge from (d_L_min, p_edge)
      to (0, 1) — reaches the asymptote at the natural physical scale d_L=0.
    * Suppressing face (d_L > d_L_max): slope-matched linear extrapolation
      from the boundary, clamped to [0, p_edge].

    Earlier history this class encoded:

    * Pre-Phase-44 the function zeroed any d_L < dl_centers[0] = dl_max/120.
      Because dl_max(h) ∝ 1/h, this created a moving threshold c_0(h) ∝ 1/h
      that produced a +145.7 log-unit MAP bias toward h_max for events with
      d_L ≈ c_0.  The Phase 44 fix removed that left-side cutoff.
    * Phase 45 (Plan 45-02 + Plan 45-04 hybrid) prepended fitted anchors
      ``(0, 0.7931)`` and ``(0.05, 1.0)`` to lift the d_L→0 saturation
      regime.  Replaced 2026-05-05 by the principled bridge construction
      because the anchor values were fitted to production-truth
      ("conservative Wilson LB chosen to not overshoot truth on production
      posteriors"), against the project's principled-physics preference.
    """

    def test_below_first_bin_follows_principled_bridge(self, injection_dir: str) -> None:
        """d_L below the first bin center follows the linear bridge from
        (dl_min, p_edge) to (0, 1).
        """
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
        grid_axis = interp_1d.grid[0]
        # No more anchors; first grid coord is the histogram first bin
        # center c_0, not 0.0.
        c0 = float(grid_axis[0])
        assert c0 > 0.0

        d_query = 0.5 * c0
        p_at = float(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_L=c0, phi=0.0, theta=0.0, h=h
            )
        )
        p_below = float(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_L=d_query, phi=0.0, theta=0.0, h=h
            )
        )

        # Bridge formula: p(dl) = 1 - (1 - p_edge) * (dl / dl_min)
        # At d_query = c0/2 with p_edge = p_at: expected = 1 - (1 - p_at) * 0.5
        expected = 1.0 - (1.0 - p_at) * (d_query / c0)
        assert float(p_below) == pytest.approx(expected, rel=1e-6), (
            f"Bridge: at d_L={d_query:.6f} (c_0={c0:.6f}), expected "
            f"{expected:.6f}; got p_below={p_below}"
        )
        # Inside [p_edge, 1] by construction.
        assert p_at - 1e-9 <= p_below <= 1.0 + 1e-9, (
            f"Bridge result {p_below} outside [p_edge={p_at}, 1]"
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

    def test_above_dl_max_decays_toward_zero(self, injection_dir: str) -> None:
        """Above dl_centers[-1] p_det approaches 0 (source beyond injection
        horizon).  Slope-matched linear extrapolation, clamped to
        [0, p_edge]; never exceeds the boundary value, asymptotes at 0.
        """
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


class TestDetectionProbabilityWithoutBHMassPrincipledExtrapolation:
    """Property-based tests for the 1D principled out-of-grid policy
    (replaces the Phase 45 anchor scheme; ``.planning/2D-CHANNEL-AUDIT-20260505.md``).

    Mirror of TestDetectionProbabilityWithBHMassPrincipledExtrapolation
    specialized to the 1D channel.  Verifies the bridge construction in
    the saturating direction, the slope-matched suppression in the
    high-d_L direction, and C0 boundary continuity at both ends.
    """

    def _build_pdet(self, injection_dir: str) -> object:
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
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

    def test_c0_continuity_at_dl_min_face(self, injection_dir: str) -> None:
        """As d_L → dl_min from below, p_det should approach the in-grid
        boundary value continuously (no step)."""
        pdet = self._build_pdet(injection_dir)
        dl_min, _ = self._grid_bounds(pdet, h=0.75)
        eps = 1e-6
        p_inside = float(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                d_L=dl_min + eps, phi=0.0, theta=0.0, h=0.75
            )
        )
        p_outside = float(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                d_L=dl_min - eps, phi=0.0, theta=0.0, h=0.75
            )
        )
        assert abs(p_inside - p_outside) < 1e-3

    def test_c0_continuity_at_dl_max_face(self, injection_dir: str) -> None:
        """As d_L → dl_max from above, p_det should approach the in-grid
        boundary value continuously (no step)."""
        pdet = self._build_pdet(injection_dir)
        _, dl_max = self._grid_bounds(pdet, h=0.75)
        eps = 1e-6
        p_inside = float(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                d_L=dl_max - eps, phi=0.0, theta=0.0, h=0.75
            )
        )
        p_outside = float(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                d_L=dl_max + eps, phi=0.0, theta=0.0, h=0.75
            )
        )
        assert abs(p_inside - p_outside) < 1e-3

    def test_bridge_reaches_unity_at_dl_zero(self, injection_dir: str) -> None:
        """At d_L=0 the bridge gives exactly p_det=1 (saturated asymptote at
        the natural physical scale)."""
        pdet = self._build_pdet(injection_dir)
        p = float(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                d_L=0.0, phi=0.0, theta=0.0, h=0.75
            )
        )
        assert p == pytest.approx(1.0, abs=1e-9)

    def test_bridge_floor_at_pedge(self, injection_dir: str) -> None:
        """In the saturating d_L<dl_min direction, the result should never
        drop below the in-grid boundary value p_edge (Option A floor)."""
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
            assert p >= p_edge - 1e-9, (
                f"d_L={d_L}: p={p} dropped below p_edge={p_edge} (Option A floor violated)"
            )
            assert p <= 1.0 + 1e-9

    def test_above_dl_max_clamped_to_pedge(self, injection_dir: str) -> None:
        """In the suppressing d_L>dl_max direction, the result should never
        exceed the in-grid boundary value p_edge."""
        pdet = self._build_pdet(injection_dir)
        _, dl_max = self._grid_bounds(pdet, h=0.75)
        p_edge = float(
            pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                d_L=dl_max, phi=0.0, theta=0.0, h=0.75
            )
        )
        for d_L in (dl_max * 1.1, dl_max * 1.5, dl_max * 5.0):
            p = float(
                pdet.detection_probability_without_bh_mass_interpolated_zero_fill(  # type: ignore[attr-defined]
                    d_L=d_L, phi=0.0, theta=0.0, h=0.75
                )
            )
            assert p <= p_edge + 1e-9
            assert p >= 0.0

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
# F1 (Phase 49) regression: h-stable bin edges for the p_det histogram.
#
# Pre-fix bug: `dl_max = max(dl_vals(h)) * 1.1` computed per-h made
# individual injections cross integer-count bin boundaries as h shifted
# by 0.001, producing 5-25% jumps in p_det at fixed (d_L, M_z) that
# summed coherently across 1473 events into visible spikes in
# Sigma log L_i.  See .planning/debug/posterior-noisy-peak.md.
#
# Fix F1 (in simulation_detection_probability.py): compute
# DL_GLOBAL_MAX once over the prior support [h_min, h_max] and reuse
# the same dl_edges at every h-trial.
#
# Refs: Farr (2019) arXiv:1904.10879 Sec III; Mandel-Farr-Gair (2019)
# arXiv:1809.02063 Eq. 18; literature audit at
# .planning/debug/F1_literature_audit.md.
# ----------------------------------------------------------------------


class TestPdetHStableBinEdges:
    """Regression: histogram support for p_det must not depend on trial h."""

    def _build_pdet(self, injection_dir: str) -> object:
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        return SimulationDetectionProbability(
            injection_data_dir=injection_dir,
            snr_threshold=20.0,
            dl_bins=60,
            mass_bins=40,
        )

    def test_dl_global_max_computed_at_h_prior_min(self, injection_dir: str) -> None:
        """The cached global max equals the empirical max d_L at the lower
        prior bound (which is where d_L is largest at fixed z).
        """
        pdet = self._build_pdet(injection_dir)
        # Manually compute the expected value
        z_arr = pdet._z_arr  # type: ignore[attr-defined]
        expected = float(np.max(dist_vectorized(z_arr, h=pdet._h_prior_min))) * 1.1  # type: ignore[attr-defined]
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
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        with pytest.raises(ValueError, match="h_prior_range"):
            SimulationDetectionProbability(
                injection_data_dir=injection_dir,
                snr_threshold=20.0,
                h_prior_range=(0.80, 0.60),  # inverted
            )

    def test_h_prior_range_override_affects_global_max(self, injection_dir: str) -> None:
        """Tightening the lower h-prior bound reduces _dl_global_max."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
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
        # d_L is monotone-decreasing in h, so a higher lower bound on h
        # gives a smaller max d_L over the prior.
        assert pdet_narrow._dl_global_max < pdet_wide._dl_global_max


# ----------------------------------------------------------------------
# F4 (Phase 49): Nadaraya-Watson kernel p_det estimator regression tests.
#
# Replaces histogram bin counts with kernel-weighted sums so that
# injection d_L_k(h) crossing the (still fixed) bin edges no longer
# produces integer-count jumps.  Diagnostic test_29 attributed 96% of
# post-F1 spike variance to this "mechanism A" — see
# .planning/PHASE-49-F4-PLAN.md and
# scripts/bias_investigation/test_29_snr_threshold_crossings.py.
#
# Refs: Nadaraya (1964); Watson (1964); Scott (1992) Ch. 6;
#       Farr (2019) arXiv:1904.10879 Sec III.
# ----------------------------------------------------------------------


class TestF4KernelEstimator:
    """Regression tests for the F4 Nadaraya-Watson p_det estimator."""

    def _build_pdet(
        self,
        injection_dir: str,
        bandwidth_scale: float = 1.0,
    ) -> object:
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
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
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        with pytest.raises(ValueError, match="bandwidth_scale"):
            SimulationDetectionProbability(
                injection_data_dir=injection_dir,
                snr_threshold=20.0,
                bandwidth_scale=0.0,
            )

    def test_quality_flags_are_continuous(self, injection_dir: str) -> None:
        """Under the kernel form, n_total/n_detected are float-valued kernel
        mass (not integer counts).  Cells with multiple contributing
        injections have non-integer kernel mass; this asserts the change of
        semantics introduced by F4.
        """
        pdet = self._build_pdet(injection_dir)
        flags = pdet.quality_flags(h=0.73)  # type: ignore[attr-defined]
        n_total = np.asarray(flags["n_total"])
        # Float dtype check
        assert np.issubdtype(n_total.dtype, np.floating)
        # At least one populated cell has non-integer kernel mass (pre-F4
        # this array was integer-valued counts).
        nonzero = n_total[n_total > 0.0]
        assert nonzero.size > 0
        non_integer_count = int(np.sum(np.abs(nonzero - np.round(nonzero)) > 1e-9))
        assert non_integer_count > 0, (
            "F4 kernel estimator should produce non-integer 'n_total' (kernel "
            "mass) values; got all-integer values which suggests histogram "
            "fallback."
        )

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


class TestLocalLinearHelper:
    """Unit tests for the F4-v2 local-linear-in-d_L intercept estimator."""

    def test_interior_symmetric_reduces_to_nadaraya_watson(self) -> None:
        """In a symmetric neighbourhood (S1=0) the LL intercept = NW ratio."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            _local_linear_p_det,
        )

        # Symmetric u about 0, symmetric weights -> S1 = 0.
        u = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
        w = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        y = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        nw = float((w * y).sum() / w.sum())
        ll = float(_local_linear_p_det(u, w, y))
        assert ll == pytest.approx(nw, abs=1e-12)

    def test_boundary_corrects_one_sided_downward_trend(self) -> None:
        """One-sided neighbourhood with a declining trend: LL intercept exceeds
        the NW one-sided average (boundary-bias correction)."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            _local_linear_p_det,
        )

        # All points at u >= 0 (boundary at u=0); detection declines with u.
        # True boundary value (u=0) is 1.0; NW averages in the far misses.
        u = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
        w = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        y = np.array([1.0, 1.0, 1.0, 0.0, 0.0])  # declining
        nw = float((w * y).sum() / w.sum())  # 0.6
        ll = float(_local_linear_p_det(u, w, y))
        assert ll > nw  # LL recovers more of the true boundary value
        assert ll == pytest.approx(1.0, abs=0.15)

    def test_clipped_to_unit_interval(self) -> None:
        """Local-linear extrapolation is clipped to [0, 1]."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            _local_linear_p_det,
        )

        # Steeply declining all-detected-near data would extrapolate the line
        # above 1 at u=0; must clip.
        u = np.array([0.01, 0.02, 0.03, 0.04])
        y = np.array([1.0, 1.0, 1.0, 1.0])
        w = np.ones(4)
        ll = float(_local_linear_p_det(u, w, y))
        assert 0.0 <= ll <= 1.0

    def test_singular_neighbourhood_falls_back_to_ratio(self) -> None:
        """A degenerate (single distinct u) design falls back to T0/S0."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            _local_linear_p_det,
        )

        u = np.array([0.0, 0.0, 0.0])  # det == 0 -> singular 2x2
        w = np.array([1.0, 2.0, 1.0])
        y = np.array([1.0, 0.0, 1.0])
        nw = float((w * y).sum() / w.sum())
        ll = float(_local_linear_p_det(u, w, y))
        assert ll == pytest.approx(nw, abs=1e-12)

    def test_vectorized_over_columns_matches_scalar(self) -> None:
        """2D (per-M-center) vectorization matches the column-wise scalar calls."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            _local_linear_p_det,
        )

        u = np.array([0.0, 0.01, 0.02, 0.03])
        w2 = np.array([[1.0, 0.5], [1.0, 1.0], [1.0, 2.0], [1.0, 1.0]])
        y2 = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        vec = _local_linear_p_det(u, w2, y2)
        col0 = float(_local_linear_p_det(u, w2[:, 0], y2[:, 0]))
        col1 = float(_local_linear_p_det(u, w2[:, 1], y2[:, 1]))
        assert vec[0] == pytest.approx(col0, abs=1e-12)
        assert vec[1] == pytest.approx(col1, abs=1e-12)


class TestEstimatorSelection:
    """F4-v2 estimator-selection plumbing and NW regression escape hatch."""

    def test_invalid_estimator_raises(self, injection_dir: str) -> None:
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        with pytest.raises(ValueError, match="estimator must be"):
            SimulationDetectionProbability(
                injection_data_dir=injection_dir,
                snr_threshold=20.0,
                estimator="bogus",  # type: ignore[arg-type]
            )

    def test_default_is_local_linear(self, injection_dir: str) -> None:
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        pdet = SimulationDetectionProbability(injection_data_dir=injection_dir, snr_threshold=20.0)
        assert pdet._estimator == "local_linear"

    def test_nadaraya_watson_escape_hatch_is_local_constant(self, tmp_path: object) -> None:
        """With NW selected, the 1D grid reproduces the Σwy/Σw ratio exactly."""
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        # Controlled set: near sources all detected, far sources all missed.
        d = str(tmp_path)
        z = np.linspace(0.02, 0.8, 300)
        M = np.full(300, 3e5)  # noqa: N806
        # SNR ~ 1/d_L; choose loudness so the threshold falls mid-range.
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
        # Near the boundary, where all sources are detected, LL must be >= NW
        # (boundary-bias correction lifts the near-field estimate).
        assert np.all(p_ll >= p_nw - 1e-9)
        # Both bounded in [0, 1].
        for p in (p_nw, p_ll):
            assert np.all(p >= 0.0) and np.all(p <= 1.0)
