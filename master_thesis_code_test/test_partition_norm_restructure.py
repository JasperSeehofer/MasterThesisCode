"""Tests for the partition-norm restructure of p_Di (the atomic single-ratio commit).

Verifies the corrected per-event likelihood
    p_i = (beta_G(h) * L_cat + B_num(h)) / D(h)
        = w_G * L_cat + (1 - w_G) * L_comp,   w_G = beta_G/D(h) = beta_G/(beta_G+beta_Gbar)
(Gray et al. 2020, arXiv:1908.06050, Eq. 9 + 29), with the incompleteness (1-f(z))
INSIDE the completion numerator/denominator and the scalar completeness(z_det)
mixing weight DROPPED.

These exercise p_Di's partition (non-catalog_only) branch with mocked precompute
tables and pool results, so the algebraic identity and the f->1 / f->0 limits are
checked directly. CPU-only (no GPU marker).
"""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from master_thesis_code.bayesian_inference.bayesian_statistics import BayesianStatistics
from master_thesis_code.emri_rate import R_eff_per_mbh

_H = 0.73


def _run_p_Di(
    *,
    f_const: float,
    D_h: float,
    beta_Gbar: float,
    global_no_bh: float,
    global_with_bh: float,
) -> dict[str, Any]:
    """Run p_Di's partition branch with controlled tables; return its diagnostic row.

    beta_G is set to D_h - beta_Gbar (as evaluate() does). Two hosts (one reduced,
    one with-BH-mass), both M=1e6, z=0.1, so each carries the same rate weight
    w0 = R_eff_per_mbh(1e6)/1.1.
    """
    instance = object.__new__(BayesianStatistics)
    instance.h = _H
    instance.catalog_only = False
    instance.posterior_data = {}
    instance.posterior_data_with_bh_mass = {
        "galaxy_likelihoods": {},
        "additional_galaxies_without_bh_mass": {},
    }
    instance._diagnostic_rows = []

    mock_detection = MagicMock()
    mock_detection.d_L = 1.0
    mock_detection.d_L_uncertainty = 0.1
    mock_detection.phi = 0.5
    mock_detection.theta = 0.5
    mock_detection.M = 1e6
    mock_detection.M_uncertainty = 1e5
    instance.detection = mock_detection

    instance._det_index_to_slot = {0: 0}
    instance._means_3d = np.array([[0.5, 0.5, 1.0]])
    instance._cov_inv_3d = np.array([np.eye(3)])
    instance._log_norm_3d = np.array([0.0])
    instance._det_d_L = np.array([1.0])

    # Partition-norm precompute tables (consistent: beta_G = D_h - beta_Gbar).
    instance._D_h_table = {_H: D_h}
    instance._beta_Gbar_table = {_H: beta_Gbar}
    instance._beta_G_table = {_H: D_h - beta_Gbar}
    instance._global_cat_denom_no_bh = {_H: global_no_bh}
    instance._global_cat_denom_with_bh = {_H: global_with_bh}

    mock_pool = MagicMock()
    mock_pool._processes = 1
    # First starmap -> with-BH results [num_no_bh, den, num_with_bh, den_with_bh];
    # second -> without-BH results [num_no_bh, den].
    mock_pool.starmap.side_effect = [
        [[0.5, 0.3, 0.4, 0.2]],
        [[0.3, 0.2]],
    ]

    mock_completeness = MagicMock()
    # Change 5.3: B_num evaluates the per-pixel completeness f_k at the event pixel
    # (constant f here -> Omega-independent, the Task-A limiting case).
    mock_completeness.ang2pix.return_value = 0
    mock_completeness.f_k.side_effect = lambda z, k, h: np.full_like(
        np.asarray(z, dtype=np.float64), f_const
    )
    mock_completeness.get_completeness_at_redshift.side_effect = lambda z, h: np.full_like(
        np.asarray(z, dtype=np.float64), f_const
    )

    mock_p_det = MagicMock()
    mock_p_det.get_dl_max.return_value = 10.0

    mock_host = MagicMock()
    mock_host.M = 1e6
    mock_host.z = 0.1
    mock_host.catalog_index = 0
    mock_host_with_bh = MagicMock()
    mock_host_with_bh.M = 1e6
    mock_host_with_bh.z = 0.1
    mock_host_with_bh.catalog_index = 1

    combined_no_bh, combined_with_bh = BayesianStatistics.p_Di(
        instance,
        possible_host_galaxies=[mock_host],
        possible_host_galaxies_with_bh_mass=[mock_host_with_bh],
        detection_index=0,
        pool=mock_pool,
        completeness=mock_completeness,
        detection_probability_obj=mock_p_det,
    )
    row = instance._diagnostic_rows[0]
    assert row["combined_no_bh"] == combined_no_bh
    assert row["combined_with_bh"] == combined_with_bh
    return row


def _w0() -> float:
    return float(R_eff_per_mbh(np.asarray([1e6]))[0]) / 1.1


def test_single_ratio_equals_convex_combination_with_w_G() -> None:
    """combined = w_G*L_cat + (1-w_G)*L_comp (the single-ratio <-> convex identity)."""
    row = _run_p_Di(f_const=0.5, D_h=1.0e9, beta_Gbar=0.5e9, global_no_bh=2.0, global_with_bh=1.5)
    w_G = row["w_G"]
    assert w_G == pytest.approx(0.5)  # beta_G/D_h = 0.5e9/1.0e9
    # No-BH and with-BH channels both follow the convex identity with the SAME w_G.
    assert row["combined_no_bh"] == pytest.approx(
        w_G * row["L_cat_no_bh"] + (1.0 - w_G) * row["L_comp"], rel=1e-12
    )
    assert row["combined_with_bh"] == pytest.approx(
        w_G * row["L_cat_with_bh"] + (1.0 - w_G) * row["L_comp"], rel=1e-12
    )


def test_l_cat_uses_global_denominator() -> None:
    """L_cat = (Sigma_local w_g N_g) / (Sigma_global w_g D_g) -- global denominator."""
    global_no_bh = 2.0
    global_with_bh = 1.5
    row = _run_p_Di(
        f_const=0.5,
        D_h=1.0e9,
        beta_Gbar=0.5e9,
        global_no_bh=global_no_bh,
        global_with_bh=global_with_bh,
    )
    w0 = _w0()
    # cat_num_sum_no_bh = w0*(0.3 + 0.5); cat_num_sum_with_bh = w0*0.4.
    assert row["L_cat_no_bh"] == pytest.approx(w0 * (0.3 + 0.5) / global_no_bh, rel=1e-12)
    assert row["L_cat_with_bh"] == pytest.approx(w0 * 0.4 / global_with_bh, rel=1e-12)


def test_f_to_one_limit_recovers_pure_catalog() -> None:
    """f->1 everywhere: B_num=0, w_G=1 => p_i = L_cat (pure catalog)."""
    row = _run_p_Di(f_const=1.0, D_h=1.0e9, beta_Gbar=0.0, global_no_bh=2.0, global_with_bh=1.5)
    assert row["w_G"] == pytest.approx(1.0)
    assert row["B_num"] == pytest.approx(0.0, abs=1e-30)
    assert row["combined_no_bh"] == pytest.approx(row["L_cat_no_bh"], rel=1e-12)
    assert row["combined_with_bh"] == pytest.approx(row["L_cat_with_bh"], rel=1e-12)


def test_f_to_zero_limit_is_pure_completion() -> None:
    """f->0 everywhere: beta_G=0, w_G=0 => p_i = L_comp (pure completion)."""
    row = _run_p_Di(f_const=0.0, D_h=1.0e9, beta_Gbar=1.0e9, global_no_bh=2.0, global_with_bh=1.5)
    assert row["w_G"] == pytest.approx(0.0)
    assert row["B_num"] > 0.0  # full completion numerator (1-f = 1)
    # Both channels collapse onto the shared completion likelihood L_comp.
    assert row["combined_no_bh"] == pytest.approx(row["L_comp"], rel=1e-12)
    assert row["combined_with_bh"] == pytest.approx(row["L_comp"], rel=1e-12)


def test_w_G_equals_beta_G_over_D() -> None:
    """The recorded selection weight is w_G = beta_G/D(h) = beta_G/(beta_G+beta_Gbar)."""
    D_h = 8.0e8
    beta_Gbar = 3.0e8
    row = _run_p_Di(f_const=0.5, D_h=D_h, beta_Gbar=beta_Gbar, global_no_bh=2.0, global_with_bh=1.5)
    beta_G = D_h - beta_Gbar
    assert row["w_G"] == pytest.approx(beta_G / D_h)
    assert row["w_G"] == pytest.approx(beta_G / (beta_G + beta_Gbar))
