"""Tests for the sigma_z-smeared global in-catalogue selection sum (issue #30, R4).

``precompute_global_catalog_selection(..., smear_sigma_z=True)`` replaces the
per-galaxy point evaluation ``P_det(d_L(z_g;h))`` by the expectation over the
numerator's volume-deconvolved host-z kernel (num/denom sigma_z symmetry;
``results/lcat_h_dependence_20260725/DERIVATION_ESTIMATOR_REDESIGN.md`` §3.3 +
§7 risk R4). All tests run on CPU with synthetic catalogs and mocked P_det.
"""

from typing import cast
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from scipy.special import roots_legendre

import master_thesis_code.bayesian_inference.bayesian_statistics as bs
from master_thesis_code.bayesian_inference.bayesian_statistics import (
    precompute_global_catalog_selection,
)
from master_thesis_code.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)
from master_thesis_code.emri_rate import R_eff_per_mbh
from master_thesis_code.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    InternalCatalogColumns,
)
from master_thesis_code.physical_relations import (
    comoving_volume_element,
    dist_vectorized,
)

_H = 0.73


class _FakeCatalogDF:
    """Minimal catalog handler with a REDSHIFT_MEASUREMENT_ERROR column."""

    def __init__(self, z: list[float], M: list[float], z_err: list[float]) -> None:
        self.reduced_galaxy_catalog = pd.DataFrame(
            {
                InternalCatalogColumns.REDSHIFT: np.asarray(z, dtype=np.float64),
                InternalCatalogColumns.BH_MASS: np.asarray(M, dtype=np.float64),
                InternalCatalogColumns.REDSHIFT_ERROR: np.asarray(z_err, dtype=np.float64),
            }
        )


def _make_linear_pdet(dl_max: float = 5.0) -> MagicMock:
    """Mock P_det(d_L) = max(0, 1 - d_L/dl_max): smooth, strictly d_L-dependent."""
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
        return np.clip(1.0 - dl / dl_max, 0.0, 1.0)

    def _with_bh(
        d_L: npt.NDArray[np.float64],
        M_z: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
        *,
        h: float,
    ) -> npt.NDArray[np.float64]:
        dl = np.atleast_1d(np.asarray(d_L, dtype=np.float64))
        return np.clip(1.0 - dl / dl_max, 0.0, 1.0)

    mock.detection_probability_without_bh_mass_interpolated_zero_fill = _zero_fill
    mock.detection_probability_with_bh_mass_interpolated = _with_bh
    return mock


def _make_constant_pdet(dl_max: float = 5.0) -> MagicMock:
    """Mock P_det = 1 everywhere inside the grid (E[1] = 1 normalization check)."""
    mock = MagicMock(spec=SimulationDetectionProbability)
    mock.get_dl_max = lambda h: dl_max

    def _one(
        d_L: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
        *,
        h: float,
    ) -> npt.NDArray[np.float64]:
        dl = np.atleast_1d(np.asarray(d_L, dtype=np.float64))
        return np.ones_like(dl)

    mock.detection_probability_without_bh_mass_interpolated_zero_fill = _one
    mock.detection_probability_with_bh_mass_interpolated = _one
    return mock


def _manual_smeared_sum(
    z: npt.NDArray[np.float64],
    M: npt.NDArray[np.float64],
    z_err: npt.NDArray[np.float64],
    h: float,
    dl_max: float,
    sigma_v_pec_km_s: float,
) -> float:
    """Independent re-implementation of the smeared sum for the isotropic 3D branch."""
    x_nodes, x_weights = roots_legendre(50)
    total = 0.0
    for z_g, M_g, se_g in zip(z, M, z_err, strict=True):
        sigma_pv = (1.0 + z_g) * sigma_v_pec_km_s / bs.SPEED_OF_LIGHT_KM_S
        sigma_eff = max(np.sqrt(se_g**2 + sigma_pv**2), 1e-10)
        lo = max(z_g - 4.0 * sigma_eff, 1e-6)
        hi = max(z_g + 4.0 * sigma_eff, lo + 1e-12)
        c, s = 0.5 * (hi + lo), 0.5 * (hi - lo)
        z_nodes = c + s * x_nodes
        gauss = np.exp(-0.5 * ((z_nodes - z_g) / sigma_eff) ** 2)
        w_pop = np.asarray(comoving_volume_element(z_nodes, h=h), dtype=np.float64) / (
            1.0 + z_nodes
        )
        kern = gauss * w_pop * (s * x_weights)
        d_L_nodes = np.asarray(dist_vectorized(z_nodes, h=h), dtype=np.float64)
        p_nodes = np.clip(1.0 - d_L_nodes / dl_max, 0.0, 1.0)
        expectation = float(np.sum(kern * p_nodes) / np.sum(kern))
        w_g = float(np.asarray(R_eff_per_mbh(np.asarray([M_g])), dtype=np.float64)[0]) / (1.0 + z_g)
        total += w_g * expectation
    return total


def test_smear_flag_default_off_is_point_evaluated() -> None:
    """Default call (no flag) must match the point-evaluated legacy sum exactly."""
    z = [0.05, 0.10, 0.15]
    M = [1.0e5, 5.0e5, 9.0e5]
    z_err = [0.02, 0.03, 0.01]
    catalog = cast(GalaxyCatalogueHandler, _FakeCatalogDF(z, M, z_err))
    mock_pdet = _make_linear_pdet()

    z_arr = np.asarray(z)
    d_L = np.asarray(dist_vectorized(z_arr, h=_H), dtype=np.float64)
    p_point = np.clip(1.0 - d_L / 5.0, 0.0, 1.0)
    w = np.asarray(R_eff_per_mbh(np.asarray(M)), dtype=np.float64) / (1.0 + z_arr)
    expected_point = float(np.sum(w * p_point))

    result_default = precompute_global_catalog_selection(
        [_H], catalog, mock_pdet, with_bh_mass=False
    )
    result_explicit_off = precompute_global_catalog_selection(
        [_H], catalog, mock_pdet, with_bh_mass=False, smear_sigma_z=False
    )
    assert result_default[_H] == pytest.approx(expected_point, rel=1e-12)
    assert result_default[_H] == result_explicit_off[_H]


def test_smeared_matches_independent_manual_computation() -> None:
    """Smeared sum equals a from-scratch reimplementation (isotropic 3D branch)."""
    z = [0.08, 0.20]
    M = [2.0e5, 7.0e5]
    z_err = [0.035, 0.0017]
    catalog = cast(GalaxyCatalogueHandler, _FakeCatalogDF(z, M, z_err))
    mock_pdet = _make_linear_pdet()

    expected = _manual_smeared_sum(
        np.asarray(z), np.asarray(M), np.asarray(z_err), _H, 5.0, bs.SIGMA_V_PEC_KM_S
    )
    result = precompute_global_catalog_selection(
        [_H], catalog, mock_pdet, with_bh_mass=False, smear_sigma_z=True
    )
    assert result[_H] == pytest.approx(expected, rel=1e-10)


def test_sigma_z_to_zero_limit_recovers_point_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """sigma_eff -> 0 (z_err ~ 0 AND sigma_v_pec = 0): smeared == point form."""
    monkeypatch.setattr(bs, "SIGMA_V_PEC_KM_S", 0.0)
    z = [0.05, 0.12]
    M = [1.0e5, 8.0e5]
    z_err = [1.0e-14, 1.0e-14]
    catalog = cast(GalaxyCatalogueHandler, _FakeCatalogDF(z, M, z_err))
    mock_pdet = _make_linear_pdet()

    smeared = precompute_global_catalog_selection(
        [_H], catalog, mock_pdet, with_bh_mass=False, smear_sigma_z=True
    )
    point = precompute_global_catalog_selection(
        [_H], catalog, mock_pdet, with_bh_mass=False, smear_sigma_z=False
    )
    assert smeared[_H] == pytest.approx(point[_H], rel=1e-8)


def test_smeared_expectation_is_normalized() -> None:
    """With P_det == 1 inside the grid, E[P_det] == 1: smeared sum == sum of weights."""
    z = [0.05, 0.10]
    M = [1.0e5, 5.0e5]
    z_err = [0.035, 0.035]
    catalog = cast(GalaxyCatalogueHandler, _FakeCatalogDF(z, M, z_err))
    mock_pdet = _make_constant_pdet()

    z_arr = np.asarray(z)
    expected = float(
        np.sum(np.asarray(R_eff_per_mbh(np.asarray(M)), dtype=np.float64) / (1.0 + z_arr))
    )
    result = precompute_global_catalog_selection(
        [_H], catalog, mock_pdet, with_bh_mass=False, smear_sigma_z=True
    )
    assert result[_H] == pytest.approx(expected, rel=1e-12)


def test_smearing_softens_a_hard_selection_edge() -> None:
    """A galaxy just inside a P_det cliff loses weight under smearing (convexity).

    Point evaluation sees P_det = 1; the smeared kernel straddles the cliff and
    must return E[P_det] strictly between 0 and 1. Direction only — no tuning.
    """
    dl_max = 5.0
    mock = MagicMock(spec=SimulationDetectionProbability)
    mock.get_dl_max = lambda h: dl_max
    dl_cliff = float(np.asarray(dist_vectorized(np.asarray([0.10]), h=_H))[0])

    def _step(
        d_L: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
        *,
        h: float,
    ) -> npt.NDArray[np.float64]:
        dl = np.atleast_1d(np.asarray(d_L, dtype=np.float64))
        return (dl <= dl_cliff).astype(np.float64)

    mock.detection_probability_without_bh_mass_interpolated_zero_fill = _step
    mock.detection_probability_with_bh_mass_interpolated = _step

    z = [0.099]  # just inside the cliff at z = 0.10
    M = [5.0e5]
    z_err = [0.035]  # photo-z width straddles the cliff
    catalog = cast(GalaxyCatalogueHandler, _FakeCatalogDF(z, M, z_err))

    point = precompute_global_catalog_selection(
        [_H], catalog, mock, with_bh_mass=False, smear_sigma_z=False
    )
    smeared = precompute_global_catalog_selection(
        [_H], catalog, mock, with_bh_mass=False, smear_sigma_z=True
    )
    w_g = float(np.asarray(R_eff_per_mbh(np.asarray(M)), dtype=np.float64)[0]) / (1.0 + z[0])
    assert point[_H] == pytest.approx(w_g, rel=1e-12)  # point sees P_det = 1
    assert 0.0 < smeared[_H] < point[_H]  # smearing feels the cliff


def test_smear_requires_redshift_error_column() -> None:
    """smear_sigma_z=True without the z-error column must raise, not degrade silently."""

    class _NoErrCatalog:
        def __init__(self) -> None:
            self.reduced_galaxy_catalog = pd.DataFrame(
                {
                    InternalCatalogColumns.REDSHIFT: np.asarray([0.05], dtype=np.float64),
                    InternalCatalogColumns.BH_MASS: np.asarray([1.0e5], dtype=np.float64),
                }
            )

    catalog = cast(GalaxyCatalogueHandler, _NoErrCatalog())
    with pytest.raises(ValueError, match="REDSHIFT_MEASUREMENT_ERROR"):
        precompute_global_catalog_selection(
            [_H], catalog, _make_linear_pdet(), with_bh_mass=False, smear_sigma_z=True
        )
