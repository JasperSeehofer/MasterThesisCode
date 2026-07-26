"""Tests for the ``absolute_marginal`` normalization mode (issue #30, Variant 1).

The absolute-mass per-event host marginal replaces the self-normalized
ratio-of-sums assembly by

    p_i(h) = (A_i(h) + B_num,i(h)) / D(h),
    A_i(h) = (Sum_ball w_g N_g) / n_bar_w(h),   n_bar_w(h) = Sigma_glob(h) / beta_G(h),

which is algebraically identical to the pre-existing (dormant) "volume_global"
branch of ``p_Di``: ``A_i/D = w_G * L_cat_global`` exactly. These tests pin

* the equivalence claim (new-mode p_i == w_G*L_cat_global-assembled marginal),
* byte-identity of the existing production modes (volume_deconv/local_ratio/global),
* the derivation's limiting cases: empty ball -> B_num/D exactly and continuously,
  f -> 1 recovering the full-catalogue Gray/Gair ratio-of-sums, and the h^-3
  cancellation in the calibration constant n_bar_w,
* both channels (without / with BH mass) throughout.

References:
    Eq. (15) in Chen, Fishbach & Holz (2018), arXiv:1712.06531.
    Eq. (2.4) in Gray et al. (2023), arXiv:2308.02281.
    results/lcat_h_dependence_20260725/DERIVATION_ESTIMATOR_REDESIGN.md (spec).

CPU-only (no GPU marker); ``single_host_likelihood`` is mocked at the pool level.
"""

import contextlib
import inspect
import warnings
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from master_thesis_code.arguments import Arguments
from master_thesis_code.bayesian_inference.bayesian_statistics import BayesianStatistics
from master_thesis_code_test.test_partition_norm_restructure import _run_p_Di, _w0

_H = 0.73


def _run_p_Di_custom(
    *,
    norm_mode: str,
    f_const: float,
    D_h: float,
    beta_Gbar: float,
    global_no_bh: float,
    global_with_bh: float,
    with_hosts: bool,
    numerator_scale: float = 1.0,
    h: float = _H,
) -> dict[str, Any]:
    """Run p_Di's partition branch with controlled tables and OPTIONAL hosts.

    Mirrors :func:`master_thesis_code_test.test_partition_norm_restructure._run_p_Di`
    but supports the empty candidate ball (``with_hosts=False`` — the issue-#29
    zero-host path flows through ``p_Di`` with empty host lists) and a numerator
    scale factor (continuity probe: ``N_g -> numerator_scale * N_g``).

    Args:
        norm_mode: In-catalogue normalization mode under test.
        f_const: Constant completeness ``f(z) = f_const``.
        D_h: Mocked completion denominator ``D(h)``.
        beta_Gbar: Mocked out-of-catalogue selection ``beta_Gbar(h)``;
            ``beta_G = D_h - beta_Gbar``.
        global_no_bh: Mocked ``Sigma_glob(h)`` for the without-BH-mass channel.
        global_with_bh: Mocked ``Sigma_glob(h)`` for the with-BH-mass channel.
        with_hosts: ``False`` runs the empty-ball limit (no starmap results).
        numerator_scale: Scale applied to the mocked per-host GW numerators.
        h: Hubble hypothesis value stored on the instance.

    Returns:
        The single diagnostic row recorded by ``p_Di``.
    """
    instance = object.__new__(BayesianStatistics)
    instance.h = h
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

    instance._D_h_table = {h: D_h}
    instance._beta_Gbar_table = {h: beta_Gbar}
    instance._beta_G_table = {h: D_h - beta_Gbar}
    instance._global_cat_denom_no_bh = {h: global_no_bh}
    instance._global_cat_denom_with_bh = {h: global_with_bh}

    mock_pool = MagicMock()
    mock_pool._processes = 1
    s = numerator_scale
    hosts: list[Any]
    hosts_with_bh: list[Any]
    if with_hosts:
        mock_pool.starmap.side_effect = [
            [np.array([[0.5 * s, 0.3, 0.4 * s, 0.2]])],
            [np.array([[0.3 * s, 0.2]])],
        ]
        mock_host = MagicMock()
        mock_host.M = 1e6
        mock_host.z = 0.1
        mock_host.catalog_index = 0
        mock_host_with_bh = MagicMock()
        mock_host_with_bh.M = 1e6
        mock_host_with_bh.z = 0.1
        mock_host_with_bh.catalog_index = 1
        hosts = [mock_host]
        hosts_with_bh = [mock_host_with_bh]
    else:
        # Empty candidate ball: _starmap_host_batches returns [] without
        # touching the pool (issue-#29 events reach p_Di with empty lists).
        hosts = []
        hosts_with_bh = []

    mock_completeness = MagicMock()
    mock_completeness.ang2pix.return_value = 0
    mock_completeness.f_k.side_effect = lambda z, k, h: np.full_like(
        np.asarray(z, dtype=np.float64), f_const
    )
    mock_completeness.get_completeness_at_redshift.side_effect = lambda z, h: np.full_like(
        np.asarray(z, dtype=np.float64), f_const
    )

    mock_p_det = MagicMock()
    mock_p_det.get_dl_max.return_value = 10.0

    combined_no_bh, combined_with_bh = BayesianStatistics.p_Di(
        instance,
        possible_host_galaxies=hosts,
        possible_host_galaxies_with_bh_mass=hosts_with_bh,
        detection_index=0,
        pool=mock_pool,
        completeness=mock_completeness,
        detection_probability_obj=mock_p_det,
    )
    row = instance._diagnostic_rows[0]
    assert row["combined_no_bh"] == combined_no_bh
    assert row["combined_with_bh"] == combined_with_bh
    return row


_KW: dict[str, float] = {
    "f_const": 0.5,
    "D_h": 1.0e9,
    "beta_Gbar": 0.4e9,
    "global_no_bh": 2.0,
    "global_with_bh": 1.5,
}


# --- Equivalence: p_i(absolute_marginal) == (A_i + B_num)/D, both channels ---


def test_absolute_marginal_matches_assembled_marginal() -> None:
    """new-mode p_i equals the w_G*L_cat_global-assembled marginal to <=1e-12.

    A_i = (Sum_ball w_g N_g)/n_bar_w with n_bar_w = Sigma_glob/beta_G, so
    p_i = (A_i + B_num)/D must match the code's (beta_G*L_cat_global + B_num)/D
    assembled independently from the mocked pieces — both channels.
    """
    row = _run_p_Di(norm_mode="absolute_marginal", **_KW)
    D_h = _KW["D_h"]
    beta_G = _KW["D_h"] - _KW["beta_Gbar"]
    w0 = _w0()
    B_num = float(row["B_num"])
    # Mocked starmap: without-BH numerators 0.3 (reduced) + 0.5 (with-bh host);
    # with-BH numerator 0.4. Equal rate weights w0.
    for sum_wN, glob, key in [
        (w0 * (0.3 + 0.5), _KW["global_no_bh"], "combined_no_bh"),
        (w0 * 0.4, _KW["global_with_bh"], "combined_with_bh"),
    ]:
        n_bar_w = glob / beta_G
        A_i = sum_wN / n_bar_w
        assert float(row[key]) == pytest.approx((A_i + B_num) / D_h, rel=1e-12)


def test_absolute_marginal_identical_to_volume_global_branch() -> None:
    """The new mode takes exactly the dormant global branch (both channels)."""
    row_am = _run_p_Di(norm_mode="absolute_marginal", **_KW)
    row_vg = _run_p_Di(norm_mode="volume_global", **_KW)
    assert row_am["L_cat_no_bh"] == pytest.approx(row_vg["L_cat_no_bh"], rel=1e-15)
    assert row_am["L_cat_with_bh"] == pytest.approx(row_vg["L_cat_with_bh"], rel=1e-15)
    assert row_am["combined_no_bh"] == pytest.approx(row_vg["combined_no_bh"], rel=1e-15)
    assert row_am["combined_with_bh"] == pytest.approx(row_vg["combined_with_bh"], rel=1e-15)


# --- Default-mode regression: existing modes unchanged ---


def test_existing_modes_unchanged_regression() -> None:
    """volume_deconv/local_ratio/global p_Di values match their analytic forms.

    Pins the pre-change behaviour of every pre-existing mode at the p_Di level
    (the new mode must not perturb them): local modes use the ball-local
    ratio-of-sums, global uses Sigma_glob — with the convex-identity assembly
    p_i = w_G*L_cat + (1-w_G)*L_comp in all cases.
    """
    w0 = _w0()
    for mode in ("volume_deconv", "local_ratio"):
        row = _run_p_Di(norm_mode=mode, **_KW)
        assert row["L_cat_no_bh"] == pytest.approx((0.3 + 0.5) / (0.2 + 0.3), rel=1e-12)
        assert row["L_cat_with_bh"] == pytest.approx(0.4 / 0.2, rel=1e-12)
        w_G = float(row["w_G"])
        assert row["combined_no_bh"] == pytest.approx(
            w_G * row["L_cat_no_bh"] + (1.0 - w_G) * row["L_comp"], rel=1e-12
        )
    row_g = _run_p_Di(norm_mode="global", **_KW)
    assert row_g["L_cat_no_bh"] == pytest.approx(w0 * (0.3 + 0.5) / _KW["global_no_bh"], rel=1e-12)
    assert row_g["L_cat_with_bh"] == pytest.approx(w0 * 0.4 / _KW["global_with_bh"], rel=1e-12)


# --- Limiting case (b): empty ball -> B_num/D exactly, and continuously ---


def test_empty_ball_reduces_to_pure_completion_exactly() -> None:
    """Empty candidate ball: A_i = 0 (empty sum) => p_i = B_num/D identically.

    The issue-#29 fallback emerges as the continuous limit of the SAME
    expression (beta_G*L_cat_global + B_num)/D with L_cat_global = 0 — bitwise
    equal to B_num/D since beta_G*0.0 + B_num == B_num exactly.
    """
    row = _run_p_Di_custom(norm_mode="absolute_marginal", with_hosts=False, **_KW)
    B_over_D = float(row["B_num"]) / _KW["D_h"]
    assert float(row["L_cat_no_bh"]) == 0.0
    assert float(row["L_cat_with_bh"]) == 0.0
    assert float(row["combined_no_bh"]) == B_over_D
    assert float(row["combined_with_bh"]) == B_over_D


def test_empty_ball_limit_is_continuous_in_catalogue_mass() -> None:
    """p_i(eps) - p_i(0) is LINEAR in the ball's absolute mass (no branch jump).

    Scaling every per-host numerator by eps scales A_i by eps, so the departure
    from the pure-completion value must be exactly eps * beta_G * L_cat(1)/D —
    the strong (quantitative) continuity of the derivation's limiting case (b).
    """
    row_full = _run_p_Di_custom(
        norm_mode="absolute_marginal", with_hosts=True, numerator_scale=1.0, **_KW
    )
    row_eps = _run_p_Di_custom(
        norm_mode="absolute_marginal", with_hosts=True, numerator_scale=1e-8, **_KW
    )
    row_zero = _run_p_Di_custom(norm_mode="absolute_marginal", with_hosts=False, **_KW)
    beta_G = _KW["D_h"] - _KW["beta_Gbar"]
    for key, cat_key in [
        ("combined_no_bh", "L_cat_no_bh"),
        ("combined_with_bh", "L_cat_with_bh"),
    ]:
        delta = float(row_eps[key]) - float(row_zero[key])
        expected = 1e-8 * beta_G * float(row_full[cat_key]) / _KW["D_h"]
        assert delta == pytest.approx(expected, rel=1e-9)


# --- Limiting case (a): f -> 1 recovers the full-catalogue Gray/Gair form ---


def test_f_to_one_recovers_full_catalog_ratio_of_sums() -> None:
    """f == 1 everywhere: B_num = 0, beta_G = D => p_i = (Sum_ball w N)/Sigma_glob.

    The catalogue-complete limit is the FULL-catalogue ratio-of-sums of
    Gray et al. (2020) Eq. A.9 / Gair et al. (2023) Eq. 15 (numerator
    self-truncated to the ball, selection denominator over all galaxies) —
    NOT the ball-local self-normalized form. Both channels.
    """
    row = _run_p_Di_custom(
        norm_mode="absolute_marginal",
        with_hosts=True,
        f_const=1.0,
        D_h=1.0e9,
        beta_Gbar=0.0,
        global_no_bh=2.0,
        global_with_bh=1.5,
    )
    w0 = _w0()
    assert float(row["B_num"]) == pytest.approx(0.0, abs=1e-30)
    assert float(row["combined_no_bh"]) == pytest.approx(w0 * (0.3 + 0.5) / 2.0, rel=1e-12)
    assert float(row["combined_with_bh"]) == pytest.approx(w0 * 0.4 / 1.5, rel=1e-12)


# --- h^-3 cancellation in n_bar_w ---


def test_h_cubed_cancellation_in_calibration() -> None:
    """The catalogue term A_i/D is invariant under the pure-volume h-rescaling.

    In flat LCDM, dV_c/dz carries h^-3 exactly, so D, beta_G, beta_Gbar scale by
    s = (h'/h)^-3 while the discrete sums Sum_ball w N and Sigma_glob carry NO
    h^-3 (fixed galaxy count). n_bar_w = Sigma_glob/beta_G then supplies the
    required h^+3 and A_i/D = beta_G * (Sum w N)/(Sigma_glob * D) is invariant:
    the spurious volume scale cancels (derivation section 3.3). Tested with
    f = 1 (B_num = 0) so the catalogue term is isolated; both channels.
    """
    s = (0.60 / 0.73) ** -3  # pure-volume rescaling of D, beta_G, beta_Gbar
    row_a = _run_p_Di_custom(
        norm_mode="absolute_marginal",
        with_hosts=True,
        f_const=1.0,
        D_h=1.0e9,
        beta_Gbar=0.0,
        global_no_bh=2.0,
        global_with_bh=1.5,
    )
    row_b = _run_p_Di_custom(
        norm_mode="absolute_marginal",
        with_hosts=True,
        f_const=1.0,
        D_h=1.0e9 * s,
        beta_Gbar=0.0,
        global_no_bh=2.0,
        global_with_bh=1.5,
    )
    assert float(row_a["combined_no_bh"]) == pytest.approx(
        float(row_b["combined_no_bh"]), rel=1e-12
    )
    assert float(row_a["combined_with_bh"]) == pytest.approx(
        float(row_b["combined_with_bh"]), rel=1e-12
    )


# --- Wiring: guard, warnings, CLI, defaults ---


def test_absolute_marginal_accepted_without_warning() -> None:
    """evaluate() accepts 'absolute_marginal' with no calibration warning."""
    instance = object.__new__(BayesianStatistics)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with contextlib.suppress(AttributeError):
            BayesianStatistics.evaluate(
                instance,
                None,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                0.73,
                normalization_mode="absolute_marginal",
            )


def test_cli_exposes_absolute_marginal_and_default_is_production() -> None:
    """--normalization_mode accepts 'absolute_marginal'; the default is the
    production 'generator_marginal' (flipped 2026-07-26)."""
    args = Arguments.create(["wd", "--evaluate", "--normalization_mode", "absolute_marginal"])
    assert args.normalization_mode == "absolute_marginal"
    args_default = Arguments.create(["wd", "--evaluate"])
    assert args_default.normalization_mode == "generator_marginal"


def test_library_default_is_production() -> None:
    """The evaluate() default and class default are 'generator_marginal'
    (production since 2026-07-26)."""
    sig = inspect.signature(BayesianStatistics.evaluate)
    assert sig.parameters["normalization_mode"].default == "generator_marginal"
    assert BayesianStatistics._normalization_mode == "generator_marginal"
