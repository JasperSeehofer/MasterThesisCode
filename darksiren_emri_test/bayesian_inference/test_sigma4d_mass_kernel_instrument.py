"""Regression tests for the tilt-ledger battery instrument J.

``--sigma4d_mass_kernel {point,kernel}`` is a pre-registered, author-approved
counterfactual instrument (results/prod2d_closure_20260818/
PREREGISTRATION_TILT_BATTERY.md v2, sections 1/2/6, P2 registered kernel).
Under "kernel", the with-BH-mass branch of
``precompute_global_catalog_selection`` replaces the point evaluation of
``S_4D`` at the observer-frame mass ``M_z_g = M_g(1+z_g)`` by the expectation
over a Gaussian mass prior centred on the (optionally Eddington-shifted)
effective mass, reusing the erf-sum inner-M machinery of
``_bh_mass_denominator_inner_m_integral_batch`` via the new
``_sigma4d_mass_kernel_expectation`` helper.

Gates pinned here (limiting cases from the gate presentation, §6):
  (i)   sigma_g -> 0 collapses the kernel expectation onto the point
        evaluation (pinned, synthetic small catalogue).
  (ii)  the kernel result stays in [0, 1] (it is a probability).
  (iii) "point" (default) is byte-identical to omitting the flag.
  (iv)  the P4 selection-table JSON dump exists and round-trips.
"""

import json
import os
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from darksiren_emri.bayesian_inference.bayesian_statistics import (
    _eddington_shifted_host_mass_batch,
    _sigma4d_mass_kernel_expectation,
    eddington_shifted_host_mass,
    precompute_global_catalog_selection,
    write_selection_table_json,
)
from darksiren_emri.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)
from darksiren_emri.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    InternalCatalogColumns,
)
from darksiren_emri.physical_relations import dist_vectorized
from darksiren_emri_test.bayesian_inference.test_bh_denominator_semianalytic import (
    make_grid2d_pdet,
)

_H = 0.73


class _StubDetectionProbabilityWithDlMax:
    """``make_grid2d_pdet`` stub + ``get_dl_max``, for the global catalogue path."""

    def __init__(self, kind: str = "peaked") -> None:
        self._grid2d = make_grid2d_pdet(kind)

    def get_dl_max(self, h: float) -> float:
        return float(self._grid2d._dl[-1])

    def _get_or_build_grid(self, h: float) -> tuple[Any, Any]:
        return self._grid2d._get_or_build_grid(h)

    def detection_probability_with_bh_mass_interpolated(
        self,
        d_L: npt.NDArray[np.float64],
        M_z: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
        h: float,
    ) -> npt.NDArray[np.float64]:
        return self._grid2d.detection_probability_with_bh_mass_interpolated(d_L, M_z, phi, theta, h)


class _FakeCatalogWithMassError:
    """Minimal catalog handler exposing REDSHIFT/BH_MASS/BH_MASS_ERROR."""

    def __init__(self, z: list[float], M: list[float], M_err: list[float]) -> None:
        self.reduced_galaxy_catalog = pd.DataFrame(
            {
                InternalCatalogColumns.REDSHIFT: np.asarray(z, dtype=np.float64),
                InternalCatalogColumns.BH_MASS: np.asarray(M, dtype=np.float64),
                InternalCatalogColumns.BH_MASS_ERROR: np.asarray(M_err, dtype=np.float64),
            }
        )


def _precompute(
    catalog: _FakeCatalogWithMassError,
    dp: _StubDetectionProbabilityWithDlMax,
    **kwargs: Any,
) -> dict[float, float]:
    """Typed wrapper: the stub/fake objects mirror the production interface
    structurally (duck typing) but are not subclasses, so mypy needs an
    explicit cast at the boundary."""
    return precompute_global_catalog_selection(
        h_values=[_H],
        galaxy_catalog=cast(GalaxyCatalogueHandler, catalog),
        detection_probability_obj=cast(SimulationDetectionProbability, dp),
        **kwargs,
    )


# ── (i)/(ii) unit-level: _sigma4d_mass_kernel_expectation ───────────────────


def test_kernel_expectation_collapses_to_point_as_sigma_to_zero() -> None:
    dp = _StubDetectionProbabilityWithDlMax()
    z_g = np.array([0.05, 0.15, 0.30])
    M_g = np.array([3.0e5, 6.0e5, 1.5e6])
    phi = np.zeros_like(z_g)
    theta = np.zeros_like(z_g)

    tiny_sigma = np.full_like(z_g, 1e-6)
    kernel_val = _sigma4d_mass_kernel_expectation(z_g, M_g, tiny_sigma, phi, theta, _H, dp)
    d_L_g = np.asarray(dist_vectorized(z_g, h=_H), dtype=np.float64)
    M_z_g = M_g * (1.0 + z_g)
    point_val = np.asarray(
        dp.detection_probability_with_bh_mass_interpolated(d_L_g, M_z_g, phi, theta, h=_H),
        dtype=np.float64,
    )
    np.testing.assert_allclose(kernel_val, point_val, rtol=2e-3, atol=2e-3)


def test_kernel_expectation_in_unit_interval() -> None:
    dp = _StubDetectionProbabilityWithDlMax()
    rng = np.random.default_rng(20260819)
    n = 50
    z_g = rng.uniform(0.01, 0.9, n)
    M_g = 10.0 ** rng.uniform(4.5, 6.5, n)
    sigma_g = M_g * rng.uniform(0.05, 0.5, n)
    phi = np.zeros(n)
    theta = np.zeros(n)
    val = _sigma4d_mass_kernel_expectation(z_g, M_g, sigma_g, phi, theta, _H, dp)
    assert np.all(val >= -1e-12)
    assert np.all(val <= 1.0 + 1e-12)


def test_kernel_expectation_chunking_matches_single_pass() -> None:
    """Row order/values must not depend on the chunk boundary (chunked loop)."""
    dp = _StubDetectionProbabilityWithDlMax()
    rng = np.random.default_rng(1)
    n = 37
    z_g = rng.uniform(0.01, 0.5, n)
    M_g = 10.0 ** rng.uniform(4.5, 6.0, n)
    sigma_g = M_g * 0.2
    phi = np.zeros(n)
    theta = np.zeros(n)
    full = _sigma4d_mass_kernel_expectation(z_g, M_g, sigma_g, phi, theta, _H, dp, chunk_size=200)
    chunked = _sigma4d_mass_kernel_expectation(z_g, M_g, sigma_g, phi, theta, _H, dp, chunk_size=5)
    np.testing.assert_allclose(full, chunked, rtol=1e-12)


# ── Eddington-shift vectorised twin ──────────────────────────────────────────


def test_eddington_shifted_host_mass_batch_matches_scalar() -> None:
    M = np.array([1.0e5, 3.0e5, 1.0e6, 5.0e6])
    M_err = np.array([1.0e4, 6.0e4, 2.0e5, 1.0e6])
    batch = _eddington_shifted_host_mass_batch(M, M_err)
    scalar = np.array([eddington_shifted_host_mass(float(m), float(e)) for m, e in zip(M, M_err)])
    np.testing.assert_allclose(batch, scalar, rtol=1e-6)


def test_eddington_shifted_host_mass_batch_guards_invalid_sigma() -> None:
    M = np.array([2.0e5, 4.0e5])
    M_err = np.array([0.0, -1.0])
    out = _eddington_shifted_host_mass_batch(M, M_err)
    np.testing.assert_array_equal(out, M)


# ── (iii) "point" == default, and precompute_global_catalog_selection wiring ─


def _make_catalog() -> _FakeCatalogWithMassError:
    z = [0.05, 0.10, 0.15, 0.25]
    M = [3.0e5, 5.0e5, 8.0e5, 1.2e6]
    M_err = [3.0e4, 5.0e4, 8.0e4, 1.2e5]
    return _FakeCatalogWithMassError(z, M, M_err)


def test_precompute_point_mode_default_matches_explicit_point() -> None:
    dp = _StubDetectionProbabilityWithDlMax()
    catalog = _make_catalog()
    default = _precompute(catalog, dp, with_bh_mass=True)
    explicit = _precompute(catalog, dp, with_bh_mass=True, sigma4d_mass_kernel="point")
    assert default == explicit


def test_precompute_kernel_mode_sigma_to_zero_matches_point() -> None:
    dp = _StubDetectionProbabilityWithDlMax()
    z = [0.05, 0.10, 0.15, 0.25]
    M = [3.0e5, 5.0e5, 8.0e5, 1.2e6]
    tiny_err = [1e-3] * 4
    catalog = _FakeCatalogWithMassError(z, M, tiny_err)
    point = _precompute(
        catalog, dp, with_bh_mass=True, sigma4d_mass_kernel="point", eddington_m="off"
    )
    kernel = _precompute(
        catalog, dp, with_bh_mass=True, sigma4d_mass_kernel="kernel", eddington_m="off"
    )
    np.testing.assert_allclose(kernel[_H], point[_H], rtol=5e-3)


def test_kernel_mode_requires_with_bh_mass() -> None:
    dp = _StubDetectionProbabilityWithDlMax()
    catalog = _make_catalog()
    with pytest.raises(ValueError):
        _precompute(catalog, dp, with_bh_mass=False, sigma4d_mass_kernel="kernel")


def test_unknown_sigma4d_mass_kernel_raises() -> None:
    dp = _StubDetectionProbabilityWithDlMax()
    catalog = _make_catalog()
    with pytest.raises(ValueError):
        _precompute(catalog, dp, with_bh_mass=True, sigma4d_mass_kernel="bogus")


def test_unknown_eddington_m_raises() -> None:
    dp = _StubDetectionProbabilityWithDlMax()
    catalog = _make_catalog()
    with pytest.raises(ValueError):
        _precompute(
            catalog,
            dp,
            with_bh_mass=True,
            sigma4d_mass_kernel="kernel",
            eddington_m="bogus",
        )


# ── (iv) P4 JSON dump exists and round-trips ─────────────────────────────────


def test_selection_table_json_round_trips(tmp_path: Any) -> None:
    path = write_selection_table_json(
        0.73,
        beta_G_phi=1.5,
        beta_Gbar_phi=0.5,
        sigma_phi=2.0,
        sigma_4d=1.8,
        directory=str(tmp_path),
    )
    assert os.path.isfile(path)
    with open(path) as f:
        data = json.load(f)
    assert data["h"] == pytest.approx(0.73)
    assert data["beta_G_phi"] == pytest.approx(1.5)
    assert data["beta_Gbar_phi"] == pytest.approx(0.5)
    assert data["sigma_phi"] == pytest.approx(2.0)
    assert data["sigma_4d"] == pytest.approx(1.8)
    assert data["r_Malm"] == pytest.approx(1.8 / 2.0)


def test_selection_table_json_filename_uses_rounded_h_label(tmp_path: Any) -> None:
    path = write_selection_table_json(
        0.72055,
        beta_G_phi=1.0,
        beta_Gbar_phi=1.0,
        sigma_phi=1.0,
        sigma_4d=1.0,
        directory=str(tmp_path),
    )
    assert (
        os.path.basename(path) == "selection_tables_h_0_7206.json"
        or os.path.basename(path) == "selection_tables_h_0_7205.json"
    )
