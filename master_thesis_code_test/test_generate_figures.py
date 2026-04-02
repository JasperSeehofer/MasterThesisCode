"""Integration tests for the batch figure generation pipeline.

Tests ``generate_figures()`` and the ``_check_file_size`` helper
from ``master_thesis_code.main``.
"""

import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from master_thesis_code.plotting._data import PARAMETER_NAMES


def _make_crb_csv(tmp_path: Path, n_rows: int = 5) -> Path:
    """Create a minimal CRB CSV in *tmp_path* with required columns."""
    rng = np.random.default_rng(42)
    n_params = len(PARAMETER_NAMES)

    # Build parameter value columns
    data: dict[str, list[float]] = {}
    for p in PARAMETER_NAMES:
        data[p] = rng.uniform(0.1, 10.0, size=n_rows).tolist()

    # Add SNR and redshift columns
    data["SNR"] = rng.uniform(20.0, 100.0, size=n_rows).tolist()
    data["redshift"] = rng.uniform(0.01, 1.0, size=n_rows).tolist()

    # Build the 105 delta_*_delta_* covariance columns (lower triangle)
    for i in range(n_params):
        for j in range(i + 1):
            col = f"delta_{PARAMETER_NAMES[i]}_delta_{PARAMETER_NAMES[j]}"
            if i == j:
                data[col] = rng.uniform(0.001, 0.1, size=n_rows).tolist()
            else:
                data[col] = rng.uniform(-0.01, 0.01, size=n_rows).tolist()

    df = pd.DataFrame(data)
    csv_path = tmp_path / "cramer_rao_bounds_0.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


class TestGenerateFigures:
    """Tests for the generate_figures() function."""

    def test_produces_pdf_with_crb_data(self, tmp_path: Path) -> None:
        """generate_figures with CRB CSV produces at least one PDF."""
        _make_crb_csv(tmp_path)

        from master_thesis_code.main import generate_figures

        generate_figures(str(tmp_path))

        figures_dir = tmp_path / "figures"
        pdfs = list(figures_dir.glob("*.pdf"))
        assert len(pdfs) >= 1, f"Expected at least 1 PDF, found {len(pdfs)}"

    def test_graceful_degradation_empty_dir(self, tmp_path: Path) -> None:
        """generate_figures with no data completes without error."""
        from master_thesis_code.main import generate_figures

        # Should not raise
        generate_figures(str(tmp_path))

    def test_check_file_size_warns_over_2mb(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """_check_file_size logs warning when file exceeds 2 MB."""
        from master_thesis_code.main import _check_file_size

        fake_path = str(tmp_path / "big.pdf")
        with patch("os.path.getsize", return_value=3_000_000):
            with caplog.at_level(logging.WARNING):
                _check_file_size(fake_path, "big_figure")

        assert any("exceeds 2 MB" in rec.message for rec in caplog.records)

    def test_check_file_size_no_warn_under_2mb(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """_check_file_size does not warn for files under 2 MB."""
        from master_thesis_code.main import _check_file_size

        fake_path = str(tmp_path / "small.pdf")
        with patch("os.path.getsize", return_value=100):
            with caplog.at_level(logging.WARNING):
                _check_file_size(fake_path, "small_figure")

        assert not any("exceeds 2 MB" in rec.message for rec in caplog.records)
