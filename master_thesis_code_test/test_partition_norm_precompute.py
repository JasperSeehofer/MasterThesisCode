"""Tests for the partition-norm precompute functions (Commit 1, additive).

Covers the two new additive precomputes that the partition-norm restructure
(Option A) will consume in a later [PHYSICS] commit, but which do NOT change any
posterior yet:

* ``precompute_missing_completion_denominator`` -> ``beta_Gbar(h)`` =
  INTEGRAL (1-f) P_det dVc/(1+z) (Gray et al. 2020, arXiv:1908.06050, Eq. 33),
  with ``beta_G = D(h) - beta_Gbar`` (Eq. 29).
* ``precompute_global_catalog_selection`` -> ``sum_global w_g D_g`` over the full
  catalog out to the detection horizon (Eq. 29, discrete realisation).

All tests run on CPU (no GPU marker); dependencies are mocked / synthetic.
"""

from typing import cast
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from master_thesis_code.bayesian_inference.bayesian_statistics import (
    precompute_completion_denominator,
    precompute_global_catalog_selection,
    precompute_missing_completion_denominator,
)
from master_thesis_code.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)
from master_thesis_code.emri_rate import R_eff_per_mbh
from master_thesis_code.galaxy_catalogue.glade_completeness import GladeCatalogCompleteness
from master_thesis_code.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    InternalCatalogColumns,
)
from master_thesis_code.physical_relations import dist_to_redshift

# Reuse the smooth 1D P_det mock the completion-term tests already validate.
from master_thesis_code_test.test_completion_term_fix import _make_mock_pdet

_H = 0.73
_OMEGA_M = 0.25
_OMEGA_DE = 0.75


def _constant_completeness(fraction_percent: float) -> GladeCatalogCompleteness:
    """A completeness model with a constant ``f(z) = fraction_percent/100``."""
    return GladeCatalogCompleteness(
        distance=[0.0, 1.0e7],
        completeness=[fraction_percent, fraction_percent],
    )


# ======================================================================
# precompute_missing_completion_denominator (beta_Gbar)
# ======================================================================


def test_beta_gbar_constant_completeness_is_one_minus_f_times_D() -> None:
    """For constant f=c, beta_Gbar = (1-c) D(h) and beta_G = c D(h) exactly."""
    mock_pdet = _make_mock_pdet(dl_max=5.0)
    c = 0.40
    D = precompute_completion_denominator([_H], mock_pdet, Omega_m=_OMEGA_M, Omega_DE=_OMEGA_DE)
    beta_Gbar = precompute_missing_completion_denominator(
        [_H], mock_pdet, completeness=_constant_completeness(100.0 * c)
    )
    # (1-f) pulls out of the linear quadrature when f is constant.
    assert beta_Gbar[_H] == pytest.approx((1.0 - c) * D[_H], rel=1e-9)
    beta_G = D[_H] - beta_Gbar[_H]
    assert beta_G == pytest.approx(c * D[_H], rel=1e-9)


def test_beta_gbar_full_completeness_is_zero() -> None:
    """f -> 1 everywhere => beta_Gbar -> 0 (no missing volume) => beta_G = D(h)."""
    mock_pdet = _make_mock_pdet(dl_max=5.0)
    D = precompute_completion_denominator([_H], mock_pdet, Omega_m=_OMEGA_M, Omega_DE=_OMEGA_DE)
    beta_Gbar = precompute_missing_completion_denominator(
        [_H], mock_pdet, completeness=_constant_completeness(100.0)
    )
    assert beta_Gbar[_H] == pytest.approx(0.0, abs=1e-6 * D[_H])


def test_beta_gbar_zero_completeness_is_full_D() -> None:
    """f -> 0 everywhere => beta_Gbar -> D(h) (all volume missing) => beta_G = 0."""
    mock_pdet = _make_mock_pdet(dl_max=5.0)
    D = precompute_completion_denominator([_H], mock_pdet, Omega_m=_OMEGA_M, Omega_DE=_OMEGA_DE)
    beta_Gbar = precompute_missing_completion_denominator(
        [_H], mock_pdet, completeness=_constant_completeness(0.0)
    )
    assert beta_Gbar[_H] == pytest.approx(D[_H], rel=1e-9)


def test_beta_gbar_realistic_completeness_strictly_between_zero_and_D() -> None:
    """Realistic GLADE+ completeness gives 0 < beta_Gbar < D(h) (interior split)."""
    mock_pdet = _make_mock_pdet(dl_max=5.0)
    D = precompute_completion_denominator([_H], mock_pdet, Omega_m=_OMEGA_M, Omega_DE=_OMEGA_DE)
    beta_Gbar = precompute_missing_completion_denominator(
        [_H], mock_pdet, completeness=GladeCatalogCompleteness()
    )
    assert 0.0 < beta_Gbar[_H] < D[_H]
    beta_G = D[_H] - beta_Gbar[_H]
    assert 0.0 < beta_G < D[_H]


# ======================================================================
# precompute_global_catalog_selection (sum_global w_g D_g)
# ======================================================================


class _FakeCatalogDF:
    """Minimal catalog handler exposing ``reduced_galaxy_catalog`` columns."""

    def __init__(self, z: list[float], M: list[float]) -> None:
        self.reduced_galaxy_catalog = pd.DataFrame(
            {
                InternalCatalogColumns.REDSHIFT: np.asarray(z, dtype=np.float64),
                InternalCatalogColumns.BH_MASS: np.asarray(M, dtype=np.float64),
            }
        )


def _make_constant_pdet(dl_max: float = 5.0, value: float = 1.0) -> MagicMock:
    """Mock P_det = ``value`` inside [0, dl_max] (both 3D and 4D accessors), 0 outside."""
    mock = MagicMock(spec=SimulationDetectionProbability)
    mock.get_dl_max = lambda h: dl_max

    def _zero_fill(
        d_L: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
        *,
        h: float,
    ) -> npt.NDArray[np.float64]:
        dl = np.atleast_1d(np.asarray(d_L, dtype=np.float64))
        out = np.full_like(dl, value)
        out[(dl < 0.0) | (dl > dl_max)] = 0.0
        return out

    def _with_bh(
        d_L: npt.NDArray[np.float64],
        M_z: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
        *,
        h: float,
    ) -> npt.NDArray[np.float64]:
        dl = np.atleast_1d(np.asarray(d_L, dtype=np.float64))
        out = np.full_like(dl, value)
        out[(dl < 0.0) | (dl > dl_max)] = 0.0
        return out

    mock.detection_probability_without_bh_mass_interpolated_zero_fill = _zero_fill
    mock.detection_probability_with_bh_mass_interpolated = _with_bh
    return mock


def test_global_selection_equals_sum_of_rate_weights_when_pdet_unity() -> None:
    """With P_det == 1 inside the grid, sum_global w_g D_g = sum_g R_eff(M_g)/(1+z_g)."""
    z = [0.05, 0.10, 0.15]
    M = [1.0e5, 5.0e5, 1.0e6]
    catalog = cast(GalaxyCatalogueHandler, _FakeCatalogDF(z, M))
    mock_pdet = _make_constant_pdet(dl_max=5.0, value=1.0)

    expected = float(
        np.sum(np.asarray(R_eff_per_mbh(np.asarray(M)), dtype=np.float64) / (1.0 + np.asarray(z)))
    )
    result = precompute_global_catalog_selection([_H], catalog, mock_pdet, with_bh_mass=False)
    assert result[_H] == pytest.approx(expected, rel=1e-9)


def test_global_selection_excludes_galaxies_beyond_horizon() -> None:
    """Galaxies with z_g >= z_max(h) (beyond P_det grid) are dropped from the sum."""
    dl_max = 5.0
    z_max = dist_to_redshift(dl_max, h=_H)
    z = [0.05, 0.10, 10.0 * z_max]  # last is far beyond the horizon
    M = [1.0e5, 5.0e5, 1.0e6]
    catalog = cast(GalaxyCatalogueHandler, _FakeCatalogDF(z, M))
    mock_pdet = _make_constant_pdet(dl_max=dl_max, value=1.0)

    expected = float(
        np.sum(
            np.asarray(R_eff_per_mbh(np.asarray(M[:2])), dtype=np.float64)
            / (1.0 + np.asarray(z[:2]))
        )
    )
    result = precompute_global_catalog_selection([_H], catalog, mock_pdet, with_bh_mass=False)
    assert result[_H] == pytest.approx(expected, rel=1e-9)


def test_global_selection_drops_nonfinite_masses() -> None:
    """Galaxies with NaN / non-positive mass do not contribute to the sum."""
    z = [0.05, 0.10, 0.15]
    M = [1.0e5, float("nan"), -1.0]
    catalog = cast(GalaxyCatalogueHandler, _FakeCatalogDF(z, M))
    mock_pdet = _make_constant_pdet(dl_max=5.0, value=1.0)

    expected = float(R_eff_per_mbh(np.asarray([1.0e5]))[0] / (1.0 + 0.05))
    result = precompute_global_catalog_selection([_H], catalog, mock_pdet, with_bh_mass=False)
    assert result[_H] == pytest.approx(expected, rel=1e-9)


def test_global_selection_with_bh_mass_channel_runs_and_is_positive() -> None:
    """The 4D (with-BH-mass) channel evaluates and returns a finite positive sum."""
    z = [0.05, 0.10, 0.15]
    M = [1.0e5, 5.0e5, 1.0e6]
    catalog = cast(GalaxyCatalogueHandler, _FakeCatalogDF(z, M))
    mock_pdet = _make_constant_pdet(dl_max=5.0, value=1.0)

    expected = float(
        np.sum(np.asarray(R_eff_per_mbh(np.asarray(M)), dtype=np.float64) / (1.0 + np.asarray(z)))
    )
    result = precompute_global_catalog_selection([_H], catalog, mock_pdet, with_bh_mass=True)
    # P_det == 1 for both channels in the mock, so the 4D sum matches the 3D one.
    assert result[_H] == pytest.approx(expected, rel=1e-9)
    assert result[_H] > 0.0


def test_global_selection_empty_catalog_returns_zero() -> None:
    """No eligible galaxy => sum is 0.0 (defensive guard)."""
    catalog = cast(
        GalaxyCatalogueHandler, _FakeCatalogDF([10.0], [1.0e5])
    )  # single galaxy far beyond horizon
    mock_pdet = _make_constant_pdet(dl_max=5.0, value=1.0)
    result = precompute_global_catalog_selection([_H], catalog, mock_pdet, with_bh_mass=False)
    assert result[_H] == 0.0
