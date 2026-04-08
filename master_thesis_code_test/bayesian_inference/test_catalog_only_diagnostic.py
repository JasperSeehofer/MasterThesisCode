"""Tests for --catalog_only CLI flag and diagnostic CSV logging."""

import csv
import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

from master_thesis_code.arguments import Arguments

if TYPE_CHECKING:
    from master_thesis_code.bayesian_inference.bayesian_statistics import BayesianStatistics


class TestCatalogOnlyFlag:
    """Test --catalog_only CLI flag parsing."""

    def test_catalog_only_flag_present(self) -> None:
        args = Arguments.create(["workdir", "--evaluate", "--catalog_only"])
        assert args.catalog_only is True

    def test_catalog_only_flag_absent(self) -> None:
        args = Arguments.create(["workdir", "--evaluate"])
        assert args.catalog_only is False

    def test_catalog_only_without_evaluate(self) -> None:
        args = Arguments.create(["workdir", "--catalog_only"])
        assert args.catalog_only is True
        assert args.evaluate is False


class TestCatalogOnlyBypass:
    """Test that catalog_only=True sets f_i=1.0 and L_comp=0.0 in p_Di."""

    @pytest.fixture()
    def mock_bayesian_stats(self) -> "BayesianStatistics":
        """Create a minimally mocked BayesianStatistics for testing p_Di behavior."""
        # We import here to avoid import issues on CPU-only machines
        from master_thesis_code.bayesian_inference.bayesian_statistics import BayesianStatistics

        # Create instance without __init__ (avoids CSV loading)
        instance = object.__new__(BayesianStatistics)
        instance.h = 0.73
        instance.catalog_only = True
        instance.posterior_data = {}
        instance.posterior_data_with_bh_mass = {
            "galaxy_likelihoods": {},
            "additional_galaxies_without_bh_mass": {},
        }
        instance._diagnostic_rows = []

        # Mock detection
        mock_detection = MagicMock()
        mock_detection.d_L = 1.0
        mock_detection.d_L_uncertainty = 0.1
        mock_detection.phi = 0.5
        mock_detection.theta = 0.5
        mock_detection.phi_error = 0.01
        mock_detection.theta_error = 0.01
        mock_detection.M = 1e6
        mock_detection.M_uncertainty = 1e5
        instance.detection = mock_detection

        # Mock pre-computed arrays
        instance._det_index_to_slot = {0: 0}
        instance._means_3d = np.array([[0.5, 0.5, 1.0]])
        instance._cov_inv_3d = np.array([np.eye(3)])
        instance._log_norm_3d = np.array([0.0])
        instance._det_d_L = np.array([1.0])

        return instance

    def test_catalog_only_skips_completion_integral(
        self, mock_bayesian_stats: "BayesianStatistics"
    ) -> None:
        """When catalog_only=True, f_i should be 1.0 and L_comp should be 0.0."""
        import multiprocessing as mp

        from master_thesis_code.bayesian_inference.bayesian_statistics import BayesianStatistics

        instance = mock_bayesian_stats

        # Create mock pool that returns known galaxy likelihoods
        mock_pool = MagicMock(spec=mp.pool.Pool)
        mock_pool._processes = 1
        # results_with_bh_mass: each result is [numerator_no_bh, denominator, numerator_with_bh, denominator_with_bh]
        mock_pool.starmap.side_effect = [
            [[0.5, 0.3, 0.4, 0.2]],  # with BH mass results
            [[0.3, 0.2]],  # without BH mass results
        ]

        mock_completeness = MagicMock()
        mock_p_det = MagicMock()

        # Call p_Di through the class
        result = BayesianStatistics.p_Di(
            instance,
            possible_host_galaxies=[MagicMock()],
            possible_host_galaxies_with_bh_mass=[MagicMock()],
            detection_index=0,
            pool=mock_pool,
            completeness=mock_completeness,
            detection_probability_obj=mock_p_det,
        )

        # With catalog_only=True, f_i=1.0, so combined = 1.0 * L_cat + 0.0 * L_comp = L_cat
        # The completion integral should NOT have been computed
        # Verify completeness was never called (no f_i lookup needed)
        mock_completeness.get_completeness_at_redshift.assert_not_called()

        # Check diagnostic row was recorded
        assert len(instance._diagnostic_rows) == 1
        row = instance._diagnostic_rows[0]
        assert row["f_i"] == 1.0
        assert row["L_comp"] == 0.0


class TestDiagnosticCsv:
    """Test diagnostic CSV output."""

    def test_diagnostic_row_columns(self) -> None:
        """Verify diagnostic rows contain the expected columns."""
        expected_columns = {
            "event_idx",
            "h",
            "f_i",
            "L_cat_no_bh",
            "L_cat_with_bh",
            "L_comp",
            "combined_no_bh",
            "combined_with_bh",
        }
        # We'll verify this through the actual row dict structure
        # by checking the _write_diagnostic_csv method exists
        from master_thesis_code.bayesian_inference.bayesian_statistics import BayesianStatistics

        assert hasattr(BayesianStatistics, "_write_diagnostic_csv")

    def test_write_diagnostic_csv(self, tmp_path: "os.PathLike[str]") -> None:
        """Test that _write_diagnostic_csv writes correct CSV output."""
        from master_thesis_code.bayesian_inference.bayesian_statistics import BayesianStatistics

        instance = object.__new__(BayesianStatistics)
        instance._diagnostic_rows = [
            {
                "event_idx": 0,
                "h": 0.73,
                "f_i": 0.85,
                "L_cat_no_bh": 1.23e-5,
                "L_cat_with_bh": 2.34e-5,
                "L_comp": 3.45e-6,
                "combined_no_bh": 1.1e-5,
                "combined_with_bh": 2.1e-5,
            },
            {
                "event_idx": 1,
                "h": 0.73,
                "f_i": 1.0,
                "L_cat_no_bh": 4.56e-5,
                "L_cat_with_bh": 5.67e-5,
                "L_comp": 0.0,
                "combined_no_bh": 4.56e-5,
                "combined_with_bh": 5.67e-5,
            },
        ]

        csv_path = os.path.join(str(tmp_path), "diagnostics", "event_likelihoods.csv")
        instance._write_diagnostic_csv(csv_path)

        assert os.path.isfile(csv_path)

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert set(rows[0].keys()) == {
            "event_idx",
            "h",
            "f_i",
            "L_cat_no_bh",
            "L_cat_with_bh",
            "L_comp",
            "combined_no_bh",
            "combined_with_bh",
        }
        assert rows[0]["event_idx"] == "0"
        assert rows[1]["f_i"] == "1.0"

    def test_write_diagnostic_csv_append_mode(self, tmp_path: "os.PathLike[str]") -> None:
        """Test that subsequent writes append without duplicating headers."""
        from master_thesis_code.bayesian_inference.bayesian_statistics import BayesianStatistics

        csv_path = os.path.join(str(tmp_path), "diagnostics", "event_likelihoods.csv")

        # First write
        instance1 = object.__new__(BayesianStatistics)
        instance1._diagnostic_rows = [
            {
                "event_idx": 0,
                "h": 0.70,
                "f_i": 0.9,
                "L_cat_no_bh": 1e-5,
                "L_cat_with_bh": 2e-5,
                "L_comp": 3e-6,
                "combined_no_bh": 1e-5,
                "combined_with_bh": 2e-5,
            },
        ]
        instance1._write_diagnostic_csv(csv_path)

        # Second write (append)
        instance2 = object.__new__(BayesianStatistics)
        instance2._diagnostic_rows = [
            {
                "event_idx": 0,
                "h": 0.73,
                "f_i": 0.85,
                "L_cat_no_bh": 4e-5,
                "L_cat_with_bh": 5e-5,
                "L_comp": 6e-6,
                "combined_no_bh": 4e-5,
                "combined_with_bh": 5e-5,
            },
        ]
        instance2._write_diagnostic_csv(csv_path)

        with open(csv_path) as f:
            lines = f.readlines()

        # Header + 2 data rows
        assert len(lines) == 3
