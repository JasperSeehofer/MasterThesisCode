"""Tests for posterior combination module."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from master_thesis_code.bayesian_inference.posterior_combination import (
    CombinationStrategy,
    apply_strategy,
    build_likelihood_array,
    combine_log_space,
    combine_posteriors,
    generate_comparison_table,
    generate_diagnostic_report,
    load_posterior_jsons,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def posteriors_dir(tmp_path: Path) -> Path:
    """Create a temp dir with 3 synthetic posterior JSON files."""
    data_06 = {"0": [1.0], "1": [0.0], "2": [], "h": 0.6}
    data_07 = {"0": [2.0], "1": [0.0], "2": [], "h": 0.7}
    data_08 = {"0": [3.0], "1": [0.5], "2": [], "h": 0.8}

    for name, data in [
        ("h_0_6.json", data_06),
        ("h_0_7.json", data_07),
        ("h_0_8.json", data_08),
    ]:
        (tmp_path / name).write_text(json.dumps(data))

    return tmp_path


@pytest.fixture()
def larger_posteriors_dir(tmp_path: Path) -> Path:
    """Create a temp dir with more realistic data for comparison table tests."""
    rng = np.random.default_rng(42)
    h_values = [0.6, 0.7, 0.8]
    for h in h_values:
        data: dict[str, list[float] | float] = {}
        for i in range(20):
            if i == 5:
                data[str(i)] = [0.0]  # one zero event
            elif i == 10:
                data[str(i)] = []  # one empty event
            else:
                data[str(i)] = [float(rng.uniform(0.1, 100.0))]
            data["h"] = h
        fname = f"h_0_{str(h).replace('0.', '')}.json"
        (tmp_path / fname).write_text(json.dumps(data))
    return tmp_path


# ---------------------------------------------------------------------------
# test_load_posterior_jsons
# ---------------------------------------------------------------------------


class TestLoadPosteriorJsons:
    def test_returns_sorted_h_values(self, posteriors_dir: Path) -> None:
        h_values, _ = load_posterior_jsons(posteriors_dir)
        assert h_values == [0.6, 0.7, 0.8]

    def test_skips_empty_events(self, posteriors_dir: Path) -> None:
        _, event_likelihoods = load_posterior_jsons(posteriors_dir)
        # Event 2 had [] in all files, so should not be present
        assert 2 not in event_likelihoods

    def test_includes_zero_events(self, posteriors_dir: Path) -> None:
        _, event_likelihoods = load_posterior_jsons(posteriors_dir)
        # Event 1 had [0.0] in files 0.6 and 0.7, and [0.5] in 0.8
        assert 1 in event_likelihoods
        assert event_likelihoods[1][0.6] == 0.0
        assert event_likelihoods[1][0.8] == 0.5

    def test_event_0_has_all_h_values(self, posteriors_dir: Path) -> None:
        h_values, event_likelihoods = load_posterior_jsons(posteriors_dir)
        assert set(event_likelihoods[0].keys()) == set(h_values)
        assert event_likelihoods[0][0.6] == 1.0
        assert event_likelihoods[0][0.7] == 2.0
        assert event_likelihoods[0][0.8] == 3.0


# ---------------------------------------------------------------------------
# test_build_likelihood_array
# ---------------------------------------------------------------------------


class TestBuildLikelihoodArray:
    def test_shape_and_values(self) -> None:
        event_likelihoods = {
            0: {0.6: 1.0, 0.7: 2.0},
            1: {0.6: 0.0, 0.7: 3.0},
        }
        h_values = [0.6, 0.7]
        arr, indices = build_likelihood_array(h_values, event_likelihoods)
        assert arr.shape == (2, 2)
        np.testing.assert_array_equal(arr[0], [1.0, 2.0])
        np.testing.assert_array_equal(arr[1], [0.0, 3.0])
        assert indices == [0, 1]

    def test_missing_h_value_becomes_nan(self) -> None:
        event_likelihoods = {
            0: {0.6: 1.0},  # missing 0.7
        }
        h_values = [0.6, 0.7]
        arr, _ = build_likelihood_array(h_values, event_likelihoods)
        assert arr.shape == (1, 2)
        assert arr[0, 0] == 1.0
        assert np.isnan(arr[0, 1])


# ---------------------------------------------------------------------------
# test_apply_strategy
# ---------------------------------------------------------------------------


class TestApplyStrategy:
    def test_strategy_naive(self) -> None:
        likelihoods = np.array([[1.0, 2.0], [0.0, 3.0]])
        result, excluded = apply_strategy(likelihoods, CombinationStrategy.NAIVE)
        assert excluded == 0
        # Zero replaced with tiny, not exact zero
        assert result[1, 0] > 0.0
        assert result[1, 0] == pytest.approx(np.finfo(float).tiny)
        # Non-zero values unchanged
        assert result[0, 0] == 1.0
        assert result[0, 1] == 2.0

    def test_strategy_exclude(self) -> None:
        likelihoods = np.array([[1.0, 2.0], [0.0, 3.0], [4.0, 5.0]])
        result, excluded = apply_strategy(likelihoods, CombinationStrategy.EXCLUDE)
        assert excluded == 1
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result[0], [1.0, 2.0])
        np.testing.assert_array_equal(result[1], [4.0, 5.0])

    def test_strategy_per_event_floor(self) -> None:
        likelihoods = np.array([[5.0, 0.0, 10.0]])
        result, excluded = apply_strategy(
            likelihoods, CombinationStrategy.PER_EVENT_FLOOR
        )
        assert excluded == 0
        # min(5.0, 10.0) / 100 = 0.05
        assert result[0, 1] == pytest.approx(0.05)
        # Non-zero values unchanged
        assert result[0, 0] == 5.0
        assert result[0, 2] == 10.0

    def test_strategy_per_event_floor_all_zero(self) -> None:
        likelihoods = np.array([[0.0, 0.0]])
        result, excluded = apply_strategy(
            likelihoods, CombinationStrategy.PER_EVENT_FLOOR
        )
        assert excluded == 0
        assert result[0, 0] == pytest.approx(np.finfo(float).tiny)
        assert result[0, 1] == pytest.approx(np.finfo(float).tiny)

    def test_strategy_physics_floor_basic(self) -> None:
        """Physics floor: zeros replaced with min(nonzero) per event."""
        likelihoods = np.array([[5.0, 0.0, 10.0]])
        result, excluded = apply_strategy(
            likelihoods, CombinationStrategy.PHYSICS_FLOOR
        )
        assert excluded == 0
        # Floor = min nonzero = 5.0 (NOT divided by 100)
        np.testing.assert_array_equal(result[0], [5.0, 5.0, 10.0])

    def test_strategy_physics_floor_per_event(self) -> None:
        """Physics floor is independent per event (D-03)."""
        likelihoods = np.array([[5.0, 0.0, 10.0], [1.0, 2.0, 0.0]])
        result, excluded = apply_strategy(
            likelihoods, CombinationStrategy.PHYSICS_FLOOR
        )
        assert excluded == 0
        # Row 0: floor = 5.0
        np.testing.assert_array_equal(result[0], [5.0, 5.0, 10.0])
        # Row 1: floor = 1.0
        np.testing.assert_array_equal(result[1], [1.0, 2.0, 1.0])

    def test_strategy_physics_floor_all_zero_excluded(self) -> None:
        """All-zero event is excluded (no nonzero value for floor)."""
        likelihoods = np.array([[0.0, 0.0], [1.0, 2.0]])
        result, excluded = apply_strategy(
            likelihoods, CombinationStrategy.PHYSICS_FLOOR
        )
        assert excluded == 1
        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result[0], [1.0, 2.0])

    def test_strategy_physics_floor_no_zeros(self) -> None:
        """No zeros: array unchanged, excluded_count=0."""
        likelihoods = np.array([[1.0, 2.0]])
        result, excluded = apply_strategy(
            likelihoods, CombinationStrategy.PHYSICS_FLOOR
        )
        assert excluded == 0
        np.testing.assert_array_equal(result[0], [1.0, 2.0])

    def test_strategy_physics_floor_logs_floor_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Floor application is logged with event index and floor value."""
        likelihoods = np.array([[5.0, 0.0, 10.0]])
        with caplog.at_level(logging.INFO):
            apply_strategy(likelihoods, CombinationStrategy.PHYSICS_FLOOR)
        assert "physics floor" in caplog.text.lower()
        assert "5.000000e+00" in caplog.text or "5.0" in caplog.text

    def test_strategy_physics_floor_all_zero_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """All-zero event exclusion is logged as warning."""
        likelihoods = np.array([[0.0, 0.0], [1.0, 2.0]])
        with caplog.at_level(logging.WARNING):
            apply_strategy(likelihoods, CombinationStrategy.PHYSICS_FLOOR)
        assert "all-zero" in caplog.text.lower() or "excluding" in caplog.text.lower()


# ---------------------------------------------------------------------------
# test_combine_log_space
# ---------------------------------------------------------------------------


class TestCombineLogSpace:
    def test_simple_combination(self) -> None:
        # Two events, two h-bins: products are 2*4=8, 3*5=15
        likelihoods = np.array([[2.0, 3.0], [4.0, 5.0]])
        posterior = combine_log_space(likelihoods)
        expected = np.array([8.0 / 23.0, 15.0 / 23.0])
        np.testing.assert_allclose(posterior, expected, rtol=1e-10)

    def test_sums_to_one(self) -> None:
        likelihoods = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        posterior = combine_log_space(likelihoods)
        assert posterior.sum() == pytest.approx(1.0)

    def test_no_underflow_500_events(self) -> None:
        """500 events with small likelihoods must produce valid posterior."""
        rng = np.random.default_rng(123)
        likelihoods = rng.uniform(0.005, 0.02, size=(500, 15))
        posterior = combine_log_space(likelihoods)
        assert not np.any(np.isnan(posterior))
        assert not np.any(posterior == 0.0)
        assert posterior.sum() == pytest.approx(1.0)
        assert posterior.shape == (15,)


# ---------------------------------------------------------------------------
# test_generate_diagnostic_report
# ---------------------------------------------------------------------------


class TestGenerateDiagnosticReport:
    def test_contains_required_sections(self) -> None:
        h_values = [0.6, 0.7, 0.8]
        likelihoods = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 1.0],
            [5.0, 6.0, 7.0],
        ])
        detection_indices = [10, 20, 30]
        report = generate_diagnostic_report(h_values, likelihoods, detection_indices)
        assert "# Zero-Likelihood Diagnostic Report" in report or "## Zero-Likelihood" in report
        assert "20" in report  # Detection index 20 has zeros
        assert "0.6" in report  # h-bin with zero

    def test_no_zeros_report(self) -> None:
        h_values = [0.6, 0.7]
        likelihoods = np.array([[1.0, 2.0], [3.0, 4.0]])
        detection_indices = [0, 1]
        report = generate_diagnostic_report(h_values, likelihoods, detection_indices)
        assert "0" in report  # Should still have a summary


# ---------------------------------------------------------------------------
# test_generate_comparison_table
# ---------------------------------------------------------------------------


class TestGenerateComparisonTable:
    def test_contains_all_strategies(self) -> None:
        h_values = np.array([0.6, 0.7, 0.8])
        likelihoods = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
            [5.0, 6.0, 7.0],
        ])
        detection_indices = [0, 1, 2]
        table = generate_comparison_table(h_values, likelihoods, detection_indices, "test")
        assert "naive" in table.lower()
        assert "exclude" in table.lower()
        assert "per-event-floor" in table.lower()
        assert "physics-floor" in table.lower()
        assert "MAP" in table or "map" in table.lower()

    def test_markdown_table_format(self) -> None:
        h_values = np.array([0.6, 0.7, 0.8])
        likelihoods = np.array([[1.0, 2.0, 3.0]])
        detection_indices = [0]
        table = generate_comparison_table(h_values, likelihoods, detection_indices, "test")
        # Should contain pipe characters for markdown tables
        assert "|" in table


# ---------------------------------------------------------------------------
# test_combine_posteriors (end-to-end)
# ---------------------------------------------------------------------------


class TestCombinePosteriors:
    def test_end_to_end(self, posteriors_dir: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = combine_posteriors(
            posteriors_dir=str(posteriors_dir),
            strategy="exclude",
            output_dir=str(output_dir),
        )
        # Result dict has required keys
        assert "h_values" in result
        assert "posterior" in result
        assert "strategy" in result
        assert "n_events_total" in result
        assert "n_events_used" in result
        assert "n_events_excluded" in result
        assert result["strategy"] == "exclude"

        # Output files created
        assert (output_dir / "combined_posterior.json").exists()
        assert (output_dir / "diagnostic_report.md").exists()
        assert (output_dir / "comparison_table.md").exists()

        # Posterior sums to ~1.0
        posterior = np.array(result["posterior"])
        assert posterior.sum() == pytest.approx(1.0, abs=1e-6)

    def test_physics_floor_works(
        self, posteriors_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = combine_posteriors(
            posteriors_dir=str(posteriors_dir),
            strategy="physics-floor",
            output_dir=str(output_dir),
        )
        # Should use physics-floor directly (no fallback)
        assert result["strategy"] == "physics-floor"
        # Output files created
        assert (output_dir / "combined_posterior.json").exists()
        assert (output_dir / "diagnostic_report.md").exists()
        # Posterior sums to ~1.0
        posterior = np.array(result["posterior"])
        assert posterior.sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Integration tests against real campaign data
# ---------------------------------------------------------------------------

# Campaign data may live in the main worktree (not copied to git worktrees).
# Try both relative and absolute paths.
_CAMPAIGN_CANDIDATES = [
    Path("results/h_sweep_20260401/posteriors"),
    Path("/home/jasper/Repositories/MasterThesisCode/results/h_sweep_20260401/posteriors"),
]
CAMPAIGN_DIR: Path | None = next(
    (p for p in _CAMPAIGN_CANDIDATES if p.exists()), None
)


@pytest.mark.skipif(
    CAMPAIGN_DIR is None, reason="Campaign data not available"
)
class TestCampaignIntegration:
    """Integration tests against the real h_sweep_20260401 campaign data."""

    def _campaign_dir(self) -> Path:
        assert CAMPAIGN_DIR is not None
        return CAMPAIGN_DIR

    def test_load_real_posteriors(self) -> None:
        """Verify loading all 15 h-value JSON files."""
        h_values, event_likelihoods = load_posterior_jsons(self._campaign_dir())
        assert len(h_values) == 15
        assert h_values[0] == pytest.approx(0.6)
        assert h_values[-1] == pytest.approx(0.86)
        assert len(event_likelihoods) >= 530  # ~538 events minus some empties

    def test_naive_strategy_all_zeros(self) -> None:
        """Naive strategy should produce valid normalized posterior."""
        h_values, event_likelihoods = load_posterior_jsons(self._campaign_dir())
        likelihoods, _det_indices = build_likelihood_array(h_values, event_likelihoods)
        processed, excluded = apply_strategy(likelihoods, CombinationStrategy.NAIVE)
        assert excluded == 0
        posterior = combine_log_space(processed)
        assert np.isfinite(posterior).all()
        assert pytest.approx(np.sum(posterior), abs=0.01) == 1.0

    def test_exclude_strategy_map(self) -> None:
        """Exclude strategy should give MAP within expected range."""
        h_values, event_likelihoods = load_posterior_jsons(self._campaign_dir())
        likelihoods, _det_indices = build_likelihood_array(h_values, event_likelihoods)
        processed, excluded = apply_strategy(likelihoods, CombinationStrategy.EXCLUDE)
        assert excluded >= 17  # At least 17 zero-events excluded
        posterior = combine_log_space(processed)
        h_arr = np.array(h_values)
        map_h = h_arr[np.argmax(posterior)]
        # Known baseline: MAP in [0.60, 0.86] range for "without BH mass" with exclude
        assert 0.60 <= map_h <= 0.86, f"MAP {map_h} outside expected range"

    def test_diagnostic_report_on_real_data(self) -> None:
        """Diagnostic report should identify the known all-zero events."""
        h_values, event_likelihoods = load_posterior_jsons(self._campaign_dir())
        likelihoods, det_indices = build_likelihood_array(h_values, event_likelihoods)
        report = generate_diagnostic_report(h_values, likelihoods, det_indices)
        assert "163" in report  # Known all-zeros event
        assert "223" in report  # Known all-zeros event
        assert "507" in report  # Known all-zeros event
        assert "## Zero-Event Detail" in report

    def test_comparison_table_on_real_data(self) -> None:
        """Comparison table should have rows for all 4 strategies."""
        h_values, event_likelihoods = load_posterior_jsons(self._campaign_dir())
        h_arr = np.array(h_values)
        likelihoods, det_indices = build_likelihood_array(h_values, event_likelihoods)
        table = generate_comparison_table(h_arr, likelihoods, det_indices, "without_bh_mass")
        assert "naive" in table.lower()
        assert "exclude" in table.lower()
        assert "per-event-floor" in table.lower()
        assert "physics-floor" in table.lower()

    def test_full_combine_posteriors_output(self, tmp_path: Path) -> None:
        """End-to-end: combine_posteriors writes all output files."""
        result = combine_posteriors(
            posteriors_dir=str(self._campaign_dir()),
            strategy="exclude",
            output_dir=str(tmp_path),
        )
        # Check output files
        assert (tmp_path / "combined_posterior.json").exists()
        assert (tmp_path / "diagnostic_report.md").exists()
        assert (tmp_path / "comparison_table.md").exists()
        # Check JSON schema
        import json

        with open(tmp_path / "combined_posterior.json") as f:
            data = json.load(f)
        assert "h_values" in data
        assert "posterior" in data
        assert "strategy" in data
        assert data["strategy"] == "exclude"
        assert len(data["h_values"]) == 15
        assert len(data["posterior"]) == 15
        assert data["n_events_excluded"] >= 17
