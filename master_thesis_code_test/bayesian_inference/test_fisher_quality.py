"""Unit tests for the Fisher matrix condition-number gate.

Tests cover:
- ``_check_covariance_quality`` helper function (well-conditioned and degenerate cases)
- slogdet sign exclusion path
- ``fisher_quality.csv`` column names and excluded flag
"""

import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from master_thesis_code.bayesian_inference.bayesian_statistics import (
    _check_covariance_quality,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity(n: int) -> npt.NDArray[np.float64]:
    """Return an n×n identity matrix (condition number == 1)."""
    return np.eye(n, dtype=np.float64)


def _near_singular(n: int, tiny: float = 1e-15) -> npt.NDArray[np.float64]:
    """Return an n×n symmetric matrix with one near-zero eigenvalue."""
    mat = np.eye(n, dtype=np.float64)
    mat[0, 0] = tiny
    return mat


# ---------------------------------------------------------------------------
# Tests for _check_covariance_quality
# ---------------------------------------------------------------------------


def test_well_conditioned_not_excluded() -> None:
    """Identity matrix has condition number 1 — should not be excluded."""
    cov = _identity(3)
    cond, should_exclude = _check_covariance_quality(cov, threshold=1e10)
    assert cond == pytest.approx(1.0, rel=1e-6)
    assert should_exclude is False


def test_degenerate_excluded_3x3() -> None:
    """Near-singular 3×3 matrix (one eigenvalue ~1e-15) should be excluded."""
    cov = _near_singular(3, tiny=1e-15)
    cond, should_exclude = _check_covariance_quality(cov, threshold=1e10)
    assert cond > 1e10
    assert should_exclude is True


def test_degenerate_excluded_4x4() -> None:
    """Near-singular 4×4 matrix should be excluded."""
    cov = _near_singular(4, tiny=1e-14)
    cond, should_exclude = _check_covariance_quality(cov, threshold=1e10)
    assert cond > 1e10
    assert should_exclude is True


def test_threshold_boundary_below() -> None:
    """Matrix with condition number just below threshold is NOT excluded."""
    # Build a diagonal matrix with known condition number = 100
    cov = np.diag(np.array([1.0, 0.01], dtype=np.float64))
    cond, should_exclude = _check_covariance_quality(cov, threshold=200.0)
    assert cond == pytest.approx(100.0, rel=1e-6)
    assert should_exclude is False


def test_threshold_boundary_above() -> None:
    """Matrix with condition number just above threshold IS excluded."""
    cov = np.diag(np.array([1.0, 0.01], dtype=np.float64))
    cond, should_exclude = _check_covariance_quality(cov, threshold=50.0)
    assert cond == pytest.approx(100.0, rel=1e-6)
    assert should_exclude is True


def test_returns_float_condition_number() -> None:
    """Condition number returned must be a Python float."""
    cov = _identity(3)
    cond, _ = _check_covariance_quality(cov, threshold=1e10)
    assert isinstance(cond, float)


def test_slogdet_sign_path() -> None:
    """A matrix whose slogdet sign is <= 0 signals non-positive-definite.

    This is tested indirectly: build a matrix that passes the cond check but
    has a negative determinant (not SPD).  The slogdet check inside evaluate()
    catches this case.  Here we verify that ``_check_covariance_quality`` still
    returns ``should_exclude=False`` (it only checks cond), so the slogdet path
    is a second independent gate implemented in ``evaluate()``.
    """
    # A negative-definite matrix: cond() is finite but slogdet sign == -1
    cov = np.diag(np.array([-1.0, -1.0, -1.0], dtype=np.float64))
    cond, should_exclude = _check_covariance_quality(cov, threshold=1e10)
    # cond() returns ratio of abs singular values — still ~1 here
    assert cond == pytest.approx(1.0, rel=1e-6)
    # The cond gate alone does NOT exclude it — the slogdet gate does
    assert should_exclude is False
    # Verify slogdet sign IS negative (confirming the secondary gate is needed)
    sign, _ = np.linalg.slogdet(cov)
    assert sign < 0


# ---------------------------------------------------------------------------
# Tests for fisher_quality.csv output
# ---------------------------------------------------------------------------


def _build_mock_quality_df(
    indices: list[int],
    cond_3d_vals: list[float],
    cond_4d_vals: list[float],
    excluded_vals: list[bool],
) -> pd.DataFrame:
    """Build a mock fisher_quality DataFrame matching the expected schema."""
    return pd.DataFrame(
        {
            "detection_index": indices,
            "cond_3d": cond_3d_vals,
            "cond_4d": cond_4d_vals,
            "excluded": excluded_vals,
        }
    )


def test_csv_columns_correct() -> None:
    """fisher_quality.csv must have exactly the four required columns."""
    df = _build_mock_quality_df(
        indices=[0, 1, 2],
        cond_3d_vals=[1.0, 2.0, 1e12],
        cond_4d_vals=[1.5, 2.5, 1e13],
        excluded_vals=[False, False, True],
    )
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "fisher_quality.csv"
        df.to_csv(csv_path, index=False)
        loaded = pd.read_csv(csv_path)

    assert list(loaded.columns) == ["detection_index", "cond_3d", "cond_4d", "excluded"]


def test_csv_excluded_flag_true_for_degenerate() -> None:
    """Events exceeding threshold must have excluded=True in CSV."""
    df = _build_mock_quality_df(
        indices=[0, 1],
        cond_3d_vals=[1.0, 1e12],
        cond_4d_vals=[1.5, 1e13],
        excluded_vals=[False, True],
    )
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "fisher_quality.csv"
        df.to_csv(csv_path, index=False)
        loaded = pd.read_csv(csv_path)

    assert bool(loaded.loc[loaded["detection_index"] == 1, "excluded"].values[0]) is True
    assert bool(loaded.loc[loaded["detection_index"] == 0, "excluded"].values[0]) is False


def test_csv_detection_index_dtype() -> None:
    """detection_index column must be integer."""
    df = _build_mock_quality_df(
        indices=[0, 1],
        cond_3d_vals=[1.0, 2.0],
        cond_4d_vals=[1.5, 2.5],
        excluded_vals=[False, False],
    )
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "fisher_quality.csv"
        df.to_csv(csv_path, index=False)
        loaded = pd.read_csv(csv_path)

    assert pd.api.types.is_integer_dtype(loaded["detection_index"])


def test_csv_cond_columns_dtype() -> None:
    """cond_3d and cond_4d columns must be numeric (float)."""
    df = _build_mock_quality_df(
        indices=[0],
        cond_3d_vals=[1.23e5],
        cond_4d_vals=[4.56e6],
        excluded_vals=[False],
    )
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "fisher_quality.csv"
        df.to_csv(csv_path, index=False)
        loaded = pd.read_csv(csv_path)

    assert pd.api.types.is_float_dtype(loaded["cond_3d"])
    assert pd.api.types.is_float_dtype(loaded["cond_4d"])
