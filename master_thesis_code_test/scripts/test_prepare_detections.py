"""Tests for batch-compatible prepare_detections script."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.prepare_detections import main


def _write_valid_cramer_rao_csv(path: Path) -> None:
    """Write a minimal valid Cramer-Rao bounds CSV that Detection can parse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Detection.__init__ reads these columns from a pd.Series
    data = {
        "M": [5e5],
        "luminosity_distance": [1.0],
        "phiS": [1.0],
        "qS": [0.5],
        "delta_phiS_delta_phiS": [0.01],  # must be positive to avoid skip
        "delta_qS_delta_qS": [0.01],
        "delta_M_delta_M": [1e8],
        "delta_luminosity_distance_delta_luminosity_distance": [0.01],
        "delta_phiS_delta_qS": [0.001],
        "delta_phiS_delta_M": [0.001],
        "delta_qS_delta_M": [0.001],
        "delta_luminosity_distance_delta_M": [0.001],
        "delta_qS_delta_luminosity_distance": [0.001],
        "delta_phiS_delta_luminosity_distance": [0.001],
        "SNR": [25.0],
        "host_galaxy_index": [0],
    }
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


class TestMainCallable:
    """main() function exists and is callable with argv parameter."""

    def test_main_exists(self) -> None:
        assert callable(main)

    def test_main_accepts_argv(self) -> None:
        """main() should accept a list of strings as argv."""
        import inspect

        sig = inspect.signature(main)
        assert "argv" in sig.parameters


class TestMissingInput:
    """main() with missing input CSV exits with non-zero code."""

    def test_missing_csv_exits_with_error(self, tmp_path: Path) -> None:
        sim_dir = tmp_path / "simulations"
        sim_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(SystemExit) as exc_info:
            main(["--workdir", str(tmp_path)])
        assert exc_info.value.code != 0


class TestSuccessfulPrepare:
    """main() with valid CSV produces output file."""

    def test_output_csv_created(self, tmp_path: Path) -> None:
        input_path = tmp_path / "simulations" / "cramer_rao_bounds.csv"
        _write_valid_cramer_rao_csv(input_path)
        main(["--workdir", str(tmp_path)])
        output_path = tmp_path / "simulations" / "prepared_cramer_rao_bounds.csv"
        assert output_path.exists()

    def test_output_has_rows(self, tmp_path: Path) -> None:
        input_path = tmp_path / "simulations" / "cramer_rao_bounds.csv"
        _write_valid_cramer_rao_csv(input_path)
        main(["--workdir", str(tmp_path)])
        output_path = tmp_path / "simulations" / "prepared_cramer_rao_bounds.csv"
        df = pd.read_csv(output_path)
        assert len(df) >= 1
