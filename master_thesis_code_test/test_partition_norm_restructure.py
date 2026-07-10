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

import contextlib
import inspect
import warnings
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
    norm_mode: str = "global",
) -> dict[str, Any]:
    """Run p_Di's partition branch with controlled tables; return its diagnostic row.

    beta_G is set to D_h - beta_Gbar (as evaluate() does). Two hosts (one reduced,
    one with-BH-mass), both M=1e6, z=0.1, so each carries the same rate weight
    w0 = R_eff_per_mbh(1e6)/1.1.

    ``norm_mode`` selects the in-catalogue L_cat normalization (commission de-rail
    study): "global" (partition-norm single ratio) vs "local_ratio"/"volume_deconv"
    (Gray A.9/A.10 local ratio-of-sums). The kernel difference between local_ratio and
    volume_deconv lives inside single_host_likelihood (mocked here), so at the p_Di
    level the two share the same ratio-of-sums normalization.
    """
    instance = object.__new__(BayesianStatistics)
    instance.h = _H
    instance._normalization_mode = norm_mode
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
    # second -> without-BH results [num_no_bh, den]. The batched dispatch
    # (_starmap_host_batches) returns one (n_hosts, n_cols) array per chunk.
    mock_pool.starmap.side_effect = [
        [np.array([[0.5, 0.3, 0.4, 0.2]])],
        [np.array([[0.3, 0.2]])],
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


# --- Commission de-rail study: in-catalogue normalization modes (fixes #2 / #1) ---
# Mocked starmap results (see _run_p_Di): without-BH host -> [N=0.3, D=0.2];
# with-BH host -> [N_nobh=0.5, D_nobh=0.3, N_bh=0.4, D_bh=0.2]. Equal rate weights.


def test_local_ratio_mode_is_gray_ratio_of_sums() -> None:
    """ "local_ratio" (de-rail fix #2): L_cat = (Sum w N_g)/(Sum w D_g), Gray A.9/A.10.

    Equal weights cancel, so L_cat_no_bh = (0.3+0.5)/(0.2+0.3) = 1.6 and
    L_cat_with_bh = 0.4/0.2 = 2.0 -- independent of the GLOBAL denominators.
    """
    row = _run_p_Di(
        f_const=0.5,
        D_h=1.0e9,
        beta_Gbar=0.5e9,
        global_no_bh=2.0,
        global_with_bh=1.5,
        norm_mode="local_ratio",
    )
    assert row["L_cat_no_bh"] == pytest.approx((0.3 + 0.5) / (0.2 + 0.3))
    assert row["L_cat_with_bh"] == pytest.approx(0.4 / 0.2)


def test_volume_deconv_shares_ratio_of_sums_normalization() -> None:
    """ "volume_deconv" (fix #1) uses the SAME local ratio-of-sums at the p_Di level.

    The volume-prior deconvolution changes the per-host N_g/D_g INSIDE
    single_host_likelihood (mocked here), not the p_Di normalization, so with identical
    mocked N_g/D_g the L_cat matches "local_ratio" and differs from "global".
    """
    kw = dict(f_const=0.5, D_h=1.0e9, beta_Gbar=0.5e9, global_no_bh=2.0, global_with_bh=1.5)
    row_vol = _run_p_Di(norm_mode="volume_deconv", **kw)
    row_loc = _run_p_Di(norm_mode="local_ratio", **kw)
    row_glob = _run_p_Di(norm_mode="global", **kw)
    assert row_vol["L_cat_no_bh"] == pytest.approx(row_loc["L_cat_no_bh"])
    assert row_vol["L_cat_with_bh"] == pytest.approx(row_loc["L_cat_with_bh"])
    # global uses the GLOBAL denominator (Sum w N / global_no_bh), a different value.
    assert row_glob["L_cat_no_bh"] != pytest.approx(row_loc["L_cat_no_bh"])


def test_unknown_normalization_mode_rejected() -> None:
    """evaluate() rejects an unknown normalization_mode early (guards typos)."""
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(ValueError, match="unknown normalization_mode"):
        # The guard fires right after catalog_only is set, before any catalog/model use,
        # so passing None for those is safe -- the bogus mode raises first.
        BayesianStatistics.evaluate(instance, None, None, 0.73, normalization_mode="bogus")  # type: ignore[arg-type]


def test_volume_global_diagnostic_mode_uses_global_denominator() -> None:
    """'volume_global' (G3 ablation-cube diagnostic) takes the GLOBAL L_cat branch.

    At the p_Di level it must be identical to 'global' (the volume-kernel
    difference lives inside single_host_likelihood, mocked here); it exists to
    isolate fix #1's kernel from fix #2's denominator in the ablation cube.
    """
    kw = dict(f_const=0.5, D_h=1.0e9, beta_Gbar=0.5e9, global_no_bh=2.0, global_with_bh=1.5)
    row_vg = _run_p_Di(norm_mode="volume_global", **kw)
    row_glob = _run_p_Di(norm_mode="global", **kw)
    row_loc = _run_p_Di(norm_mode="local_ratio", **kw)
    assert row_vg["L_cat_no_bh"] == pytest.approx(row_glob["L_cat_no_bh"])
    assert row_vg["L_cat_with_bh"] == pytest.approx(row_glob["L_cat_with_bh"])
    assert row_vg["L_cat_no_bh"] != pytest.approx(row_loc["L_cat_no_bh"])


def test_volume_global_mode_accepted_by_guard() -> None:
    """evaluate() accepts the diagnostic 'volume_global' mode (no ValueError)."""
    instance = object.__new__(BayesianStatistics)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # must not trigger the 'global' calibration warning
        with contextlib.suppress(AttributeError):
            BayesianStatistics.evaluate(
                instance,
                None,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                0.73,
                normalization_mode="volume_global",
            )


def test_default_normalization_mode_is_volume_deconv() -> None:
    """The library default is the P-P-calibrated estimator, matching the CLI default.

    'global' (~0% coverage on photo-z catalogues) must be an explicit opt-in only.
    """
    sig = inspect.signature(BayesianStatistics.evaluate)
    assert sig.parameters["normalization_mode"].default == "volume_deconv"
    assert BayesianStatistics._normalization_mode == "volume_deconv"


def test_base_seed_threaded_to_workers() -> None:
    """G4 reproducibility: evaluate() exposes base_seed (default 0 = deterministic)
    and single_host_likelihood accepts it (the MC denominator stream derives from
    (base_seed, detection_index, host_z, host_M))."""
    from master_thesis_code.bayesian_inference.bayesian_statistics import single_host_likelihood

    assert inspect.signature(BayesianStatistics.evaluate).parameters["base_seed"].default == 0
    assert BayesianStatistics._base_seed == 0
    shl_params = inspect.signature(single_host_likelihood).parameters
    assert "base_seed" in shl_params and shl_params["base_seed"].default == 0


def test_global_mode_emits_calibration_warning() -> None:
    """Explicitly requesting the legacy 'global' mode warns about mis-calibration."""
    instance = object.__new__(BayesianStatistics)
    with pytest.warns(UserWarning, match="mis-calibrated"):
        # The warning fires right after the mode guard; the bare instance then hits
        # AttributeError at the h-bounds check, which is irrelevant to this test.
        with contextlib.suppress(AttributeError):
            BayesianStatistics.evaluate(instance, None, None, 0.73, normalization_mode="global")  # type: ignore[arg-type]


def test_calibrated_modes_do_not_warn() -> None:
    """'volume_deconv' and 'local_ratio' run without a calibration warning."""
    for mode in ("volume_deconv", "local_ratio"):
        instance = object.__new__(BayesianStatistics)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with contextlib.suppress(AttributeError):
                BayesianStatistics.evaluate(instance, None, None, 0.73, normalization_mode=mode)  # type: ignore[arg-type]
