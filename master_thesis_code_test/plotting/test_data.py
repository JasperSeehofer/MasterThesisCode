"""Unit tests for the CRB data layer module (_data.py).

Tests cover covariance reconstruction, parameter name constants,
intrinsic/extrinsic partitioning, and label key mapping.
"""

import numpy as np
import pandas as pd

from master_thesis_code.plotting._data import (
    EXTRINSIC,
    INTRINSIC,
    PARAMETER_NAMES,
    label_key,
    reconstruct_covariance,
)


class TestReconstructCovariance:
    """Tests for reconstruct_covariance()."""

    def test_reconstruct_shape(self, sample_crb_row: pd.Series) -> None:
        """Reconstructed covariance has shape (14, 14)."""
        cov = reconstruct_covariance(sample_crb_row)
        assert cov.shape == (14, 14)

    def test_reconstruct_symmetric(self, sample_crb_row: pd.Series) -> None:
        """Reconstructed covariance is symmetric: cov == cov.T."""
        cov = reconstruct_covariance(sample_crb_row)
        np.testing.assert_array_equal(cov, cov.T)

    def test_reconstruct_diagonal_positive(self, sample_crb_row: pd.Series) -> None:
        """All diagonal elements are positive for a PD input."""
        cov = reconstruct_covariance(sample_crb_row)
        assert np.all(np.diag(cov) > 0)

    def test_reconstruct_roundtrip(self) -> None:
        """Build a CSV row from a known symmetric matrix, reconstruct, get same matrix."""
        rng = np.random.default_rng(123)
        n = len(PARAMETER_NAMES)
        A = rng.random((n, n))
        original = A @ A.T  # symmetric positive-definite

        # Build a pd.Series mimicking a CRB CSV row
        data: dict[str, float] = {}
        for name in PARAMETER_NAMES:
            data[name] = 1.0  # dummy param values
        for i in range(n):
            for j in range(i + 1):
                data[f"delta_{PARAMETER_NAMES[i]}_delta_{PARAMETER_NAMES[j]}"] = original[i, j]

        row = pd.Series(data)
        reconstructed = reconstruct_covariance(row)
        np.testing.assert_allclose(reconstructed, original)

    def test_reconstruct_dtype(self, sample_crb_row: pd.Series) -> None:
        """Reconstructed covariance has float64 dtype."""
        cov = reconstruct_covariance(sample_crb_row)
        assert cov.dtype == np.float64


class TestParameterNames:
    """Tests for PARAMETER_NAMES constant."""

    def test_parameter_names_length(self) -> None:
        """PARAMETER_NAMES has exactly 14 entries."""
        assert len(PARAMETER_NAMES) == 14

    def test_parameter_names_order(self) -> None:
        """First is 'M', 7th (index 6) is 'luminosity_distance'."""
        assert PARAMETER_NAMES[0] == "M"
        assert PARAMETER_NAMES[6] == "luminosity_distance"

    def test_parameter_names_last(self) -> None:
        """Last parameter is 'Phi_r0'."""
        assert PARAMETER_NAMES[-1] == "Phi_r0"


class TestIntrinsicExtrinsic:
    """Tests for INTRINSIC and EXTRINSIC partition."""

    def test_intrinsic_extrinsic_partition(self) -> None:
        """INTRINSIC + EXTRINSIC == PARAMETER_NAMES as sets, no overlap."""
        combined = set(INTRINSIC) | set(EXTRINSIC)
        assert combined == set(PARAMETER_NAMES)
        # No overlap
        assert len(set(INTRINSIC) & set(EXTRINSIC)) == 0

    def test_intrinsic_count(self) -> None:
        """6 intrinsic parameters."""
        assert len(INTRINSIC) == 6

    def test_extrinsic_count(self) -> None:
        """8 extrinsic parameters."""
        assert len(EXTRINSIC) == 8


class TestLabelKey:
    """Tests for label_key() mapping."""

    def test_label_key_mapped_luminosity_distance(self) -> None:
        """label_key('luminosity_distance') == 'd_L'."""
        assert label_key("luminosity_distance") == "d_L"

    def test_label_key_mapped_x0(self) -> None:
        """label_key('x0') == 'Y0'."""
        assert label_key("x0") == "Y0"

    def test_label_key_identity(self) -> None:
        """label_key('M') == 'M' (no mapping needed)."""
        assert label_key("M") == "M"

    def test_label_key_identity_angles(self) -> None:
        """Angle parameters pass through unchanged."""
        assert label_key("qS") == "qS"
        assert label_key("phiK") == "phiK"
        assert label_key("Phi_phi0") == "Phi_phi0"
