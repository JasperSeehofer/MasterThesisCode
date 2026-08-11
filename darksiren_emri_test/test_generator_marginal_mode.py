"""Tests for the ``generator_marginal`` normalization mode (E1 FIX-3).

The generator-consistent selection normalization replaces the Option-A
calibration of ``absolute_marginal`` by the generator's own draw-side pair:

    p_i(h)   = (Sum_ball w_g N_g / n_hat_w(h) + B_num,i(h)) / D_gen(h),
    n_hat_w  = W_cat / V_f(h)                (draw-side; NO P_det inside),
    D_gen(h) = Sigma_glob_sel(h)/n_hat_w(h) + beta_Gbar(h),

with the point/point sigma_z pairing (N_g point-evaluated at the catalogue
z_g; Sigma_glob point-evaluated). These tests pin

* the assembly formula (both channels, 4d_exact and 3d_shared conventions),
* the Option-A limiting case: when the tables satisfy the constant-comoving-
  density identity Sigma_glob = n_hat_w * beta_G, generator_marginal reduces
  ROW-WISE to absolute_marginal (derivation limiting case a),
* empty ball -> B_num/D_gen exactly and continuously,
* the h^3 identity of the calibration: V_f proportional to h^-3 exactly under an
  h-invariant completeness, hence d ln n_hat_w/dh == 3/h,
* W_cat as the exact draw normalizer (synthetic catalogue),
* the point/point kernel: numerator == GW likelihood at z_g, batch == scalar
  bit-for-bit, denominator columns byte-identical to the volume_deconv
  machinery, and the sigma_z -> 0 limit of the volume_deconv numerator
  recovering the point value,
* byte-identity of all existing modes (p_Di level; the committed kernel golden
  pins in test_kernel_parity are the kernel-level gate),
* wiring: evaluate() accepts the mode without warning, rejects
  --smear_global_selection with it, validates dgen_catalog_selection; CLI
  choice exposed; defaults unchanged.

References:
    results/lcat_h_dependence_20260725/DERIVATION_GENERATOR_CONSISTENT_NORM.md
        (the approved spec; Eqs. (3)-(5), limiting cases section 5).
    Mandel, Farr & Gair (2019), arXiv:1809.02063 — selection convention.
    Chen, Fishbach & Holz (2018), arXiv:1712.06531, Eq. (15).

CPU-only; ``single_host_likelihood`` results are mocked at the pool level for
the p_Di tests, and the kernel tests use the test_kernel_parity fixtures.
"""

import contextlib
import inspect
import warnings
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import darksiren_emri.bayesian_inference.bayesian_statistics as bs
from darksiren_emri.arguments import Arguments
from darksiren_emri.bayesian_inference.bayesian_statistics import (
    BayesianStatistics,
    compute_catalog_draw_weight_total,
    precompute_completeness_population_volume,
)
from darksiren_emri.emri_rate import R_eff_per_mbh
from darksiren_emri.galaxy_catalogue.handler import InternalCatalogColumns
from darksiren_emri_test.bayesian_inference.test_kernel_parity import (
    _case_grid,
    _install_worker_globals,
)
from darksiren_emri_test.test_partition_norm_restructure import _w0

_H = 0.73


# ── p_Di-level harness ───────────────────────────────────────────────────────


def _run_p_Di_gen(
    *,
    norm_mode: str,
    f_const: float,
    D_h: float,
    beta_Gbar: float,
    global_no_bh: float,
    global_with_bh: float,
    with_hosts: bool,
    W_cat: float = 3.0,
    V_f: float = 1.0e9,
    dgen_catalog_selection: str = "4d_exact",
    numerator_scale: float = 1.0,
    h: float = _H,
) -> dict[str, Any]:
    """Run p_Di's partition branch with controlled tables including the
    generator_marginal precomputes (W_cat, V_f table, D_gen convention).

    Mirrors ``test_absolute_marginal_mode._run_p_Di_custom`` and adds the
    generator-mode instance state.
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
    instance._W_cat = W_cat
    instance._V_f_table = {h: V_f}
    instance._dgen_catalog_selection = dgen_catalog_selection

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


_KW: dict[str, Any] = {
    "f_const": 0.5,
    "D_h": 1.0e9,
    "beta_Gbar": 0.4e9,
    "global_no_bh": 2.0,
    "global_with_bh": 1.5,
}


# ── Assembly formula (Eqs. (3)-(5)) ─────────────────────────────────────────


@pytest.mark.parametrize("dgen_sel", ["4d_exact", "3d_shared"])
def test_generator_marginal_assembly_formula(dgen_sel: str) -> None:
    """p_i == (Sum_ball w N / n_hat_w + B_num)/D_gen, both channels, both
    D_gen catalogue-selection conventions; ONE n_hat_w and ONE D_gen shared."""
    W_cat, V_f = 3.0, 1.0e9
    row = _run_p_Di_gen(
        norm_mode="generator_marginal",
        with_hosts=True,
        W_cat=W_cat,
        V_f=V_f,
        dgen_catalog_selection=dgen_sel,
        **_KW,
    )
    n_hat_w = W_cat / V_f
    sigma_sel = _KW["global_with_bh"] if dgen_sel == "4d_exact" else _KW["global_no_bh"]
    D_gen = sigma_sel / n_hat_w + _KW["beta_Gbar"]
    B_num = float(row["B_num"])
    w0 = _w0()
    # Mocked starmap: no-BH numerators 0.3 (reduced) + 0.5 (with-bh host);
    # with-BH numerator 0.4; equal rate weights w0. SAME n_hat_w both channels.
    for sum_wN, key in [
        (w0 * (0.3 + 0.5), "combined_no_bh"),
        (w0 * 0.4, "combined_with_bh"),
    ]:
        A_i = sum_wN / n_hat_w
        assert float(row[key]) == pytest.approx((A_i + B_num) / D_gen, rel=1e-12)
    # Diagnostic w_G column carries P_hat(cat|det) = (Sigma_sel/n_hat_w)/D_gen.
    assert float(row["w_G"]) == pytest.approx((sigma_sel / n_hat_w) / D_gen, rel=1e-12)


# ── Limiting case (a): Option-A tables -> reduces to absolute_marginal ──────


def test_option_a_tables_reduce_to_absolute_marginal_row_wise() -> None:
    """When the catalogue realizes Option A (Sigma_glob = n_hat_w * beta_G and
    one shared Sigma_glob), generator_marginal == absolute_marginal row-wise.

    Constructed identity: choose W_cat/V_f == Sigma_glob/beta_G. Then
    n_hat_w = n_bar_w and D_gen = Sigma_glob/n_hat_w + beta_Gbar
    = beta_G + beta_Gbar = D, so every per-event value coincides
    (derivation section 5, case a). Tolerance 1e-10 per the task spec;
    the identity is exact up to floating-point.
    """
    D_h, beta_Gbar = 1.0e9, 0.4e9
    beta_G = D_h - beta_Gbar
    sigma_glob = 2.0  # ONE shared catalogue-selection sum (both channels)
    V_f = 1.0e9
    W_cat = (sigma_glob / beta_G) * V_f  # constructs n_hat_w == n_bar_w exactly
    common = {
        "f_const": 0.5,
        "D_h": D_h,
        "beta_Gbar": beta_Gbar,
        "global_no_bh": sigma_glob,
        "global_with_bh": sigma_glob,
        "with_hosts": True,
    }
    row_gen = _run_p_Di_gen(
        norm_mode="generator_marginal",
        W_cat=W_cat,
        V_f=V_f,
        dgen_catalog_selection="3d_shared",
        **common,  # type: ignore[arg-type]
    )
    row_abs = _run_p_Di_gen(norm_mode="absolute_marginal", **common)  # type: ignore[arg-type]
    for key in ("combined_no_bh", "combined_with_bh", "B_num"):
        assert float(row_gen[key]) == pytest.approx(float(row_abs[key]), rel=1e-10)


# ── Limiting case: empty ball -> B_num/D_gen exactly and continuously ───────


def test_empty_ball_reduces_to_pure_completion_over_D_gen() -> None:
    """Empty candidate ball: A_i = 0 (empty sum) => p_i = B_num/D_gen exactly."""
    W_cat, V_f = 3.0, 1.0e9
    row = _run_p_Di_gen(
        norm_mode="generator_marginal", with_hosts=False, W_cat=W_cat, V_f=V_f, **_KW
    )
    n_hat_w = W_cat / V_f
    D_gen = _KW["global_with_bh"] / n_hat_w + _KW["beta_Gbar"]
    assert float(row["L_cat_no_bh"]) == 0.0
    assert float(row["L_cat_with_bh"]) == 0.0
    assert float(row["combined_no_bh"]) == float(row["B_num"]) / D_gen
    assert float(row["combined_with_bh"]) == float(row["B_num"]) / D_gen


def test_empty_ball_limit_is_continuous_in_catalogue_mass() -> None:
    """p_i(eps) - p_i(0) is LINEAR in the ball's absolute mass (no branch)."""
    kw = dict(_KW)
    row_full = _run_p_Di_gen(
        norm_mode="generator_marginal", with_hosts=True, numerator_scale=1.0, **kw
    )
    row_eps = _run_p_Di_gen(
        norm_mode="generator_marginal", with_hosts=True, numerator_scale=1e-8, **kw
    )
    row_zero = _run_p_Di_gen(norm_mode="generator_marginal", with_hosts=False, **kw)
    n_hat_w = 3.0 / 1.0e9
    D_gen = _KW["global_with_bh"] / n_hat_w + _KW["beta_Gbar"]
    for key, cat_key in [
        ("combined_no_bh", "L_cat_no_bh"),
        ("combined_with_bh", "L_cat_with_bh"),
    ]:
        delta = float(row_eps[key]) - float(row_zero[key])
        expected = 1e-8 * float(row_full[cat_key]) / D_gen
        assert delta == pytest.approx(expected, rel=1e-9)


# ── Existing modes byte-identical at the p_Di level ──────────────────────────


def test_existing_modes_unchanged_regression() -> None:
    """volume_deconv/local_ratio/global/absolute_marginal keep their analytic
    p_Di forms (the new mode must not perturb any of them). The kernel-level
    gate is the committed golden file in test_kernel_parity (unchanged)."""
    w0 = _w0()
    for mode in ("volume_deconv", "local_ratio"):
        row = _run_p_Di_gen(norm_mode=mode, with_hosts=True, **_KW)
        assert row["L_cat_no_bh"] == pytest.approx((0.3 + 0.5) / (0.2 + 0.3), rel=1e-12)
        assert row["L_cat_with_bh"] == pytest.approx(0.4 / 0.2, rel=1e-12)
        w_G = float(row["w_G"])
        assert row["combined_no_bh"] == pytest.approx(
            w_G * row["L_cat_no_bh"] + (1.0 - w_G) * row["L_comp"], rel=1e-12
        )
    row_am = _run_p_Di_gen(norm_mode="absolute_marginal", with_hosts=True, **_KW)
    beta_G = _KW["D_h"] - _KW["beta_Gbar"]
    B_num = float(row_am["B_num"])
    for sum_wN, glob, key in [
        (w0 * (0.3 + 0.5), _KW["global_no_bh"], "combined_no_bh"),
        (w0 * 0.4, _KW["global_with_bh"], "combined_with_bh"),
    ]:
        A_i = sum_wN / (glob / beta_G)
        assert float(row_am[key]) == pytest.approx((A_i + B_num) / _KW["D_h"], rel=1e-12)


# ── Precomputes: W_cat and V_f(h) ────────────────────────────────────────────


class _StubHandler:
    """Minimal stand-in for GalaxyCatalogueHandler.reduced_galaxy_catalog."""

    def __init__(self, z: np.ndarray, M: np.ndarray) -> None:
        self.reduced_galaxy_catalog = pd.DataFrame(
            {
                InternalCatalogColumns.REDSHIFT: z,
                InternalCatalogColumns.BH_MASS: M,
            }
        )


def test_W_cat_matches_draw_normalizer_on_synthetic_catalogue() -> None:
    """W_cat == Sum_{z_g < z_max} R_eff(M_g)/(1+z_g) — the exact total_weight of
    draw_rate_weighted_hosts, with the z<z_max eligibility respected."""
    z = np.array([0.1, 0.5, 1.2, 1.6, 2.0])
    M = np.array([1e5, 3e5, 1e6, 5e5, 2e5])
    handler = _StubHandler(z, M)
    got = compute_catalog_draw_weight_total(handler, z_max=1.5)  # type: ignore[arg-type]
    mask = z < 1.5
    expected = float(np.sum(np.asarray(R_eff_per_mbh(M[mask]), dtype=np.float64) / (1.0 + z[mask])))
    assert got == pytest.approx(expected, rel=1e-14)
    # h-independence is structural: no h anywhere in the sum.
    with pytest.raises(ValueError):
        compute_catalog_draw_weight_total(_StubHandler(z, M), z_max=0.05)  # type: ignore[arg-type]


class _ConstShapeCompleteness:
    """h-invariant completeness f_bar(z) = exp(-z) (mirrors the frozen m_th map,
    whose f_bar is exactly h-independent — derivation section 2.2)."""

    def f_bar(self, z: np.ndarray, h: float) -> np.ndarray:
        return np.exp(-np.asarray(z, dtype=np.float64))


def test_V_f_h_cubed_identity_and_n_hat_w_log_slope() -> None:
    """V_f(h) = V_f(0.73) (0.73/h)^3 exactly under h-invariant f_bar, hence
    d ln n_hat_w/dh == 3/h (n_hat_w = W_cat/V_f is proportional to h^3)."""
    comp = _ConstShapeCompleteness()
    h_list = [0.60, 0.73, 0.86]
    table = precompute_completeness_population_volume(h_list, comp)  # type: ignore[arg-type]
    v073 = table[0.73]
    assert v073 > 0.0
    for h in h_list:
        assert table[h] == pytest.approx(v073 * (0.73 / h) ** 3, rel=1e-12)
    # Finite-difference log-slope of n_hat_w = W_cat/V_f at h = 0.73: exactly
    # +3/h up to O(dh^2) since n_hat_w carries a pure h^3.
    dh = 1e-3
    t = precompute_completeness_population_volume([0.73 - dh, 0.73 + dh], comp)  # type: ignore[arg-type]
    slope = (np.log(1.0 / t[0.73 + dh]) - np.log(1.0 / t[0.73 - dh])) / (2.0 * dh)
    assert slope == pytest.approx(3.0 / 0.73, rel=1e-5)


def test_V_f_rejects_nonpositive_volume() -> None:
    """Zero completeness => V_f = 0 => loud ValueError (no silent n_hat_w=inf)."""

    class _ZeroCompleteness:
        def f_bar(self, z: np.ndarray, h: float) -> np.ndarray:
            return np.zeros_like(np.asarray(z, dtype=np.float64))

    with pytest.raises(ValueError):
        precompute_completeness_population_volume([0.73], _ZeroCompleteness())  # type: ignore[arg-type]


# ── Kernel: point/point sigma_z pairing ──────────────────────────────────────


def _gen_case() -> dict[str, Any]:
    kw = dict(_case_grid()["near_photoz_match_vd_4d"])
    kw["normalization_mode"] = "generator_marginal"
    return kw


def test_point_numerator_equals_gw_likelihood_at_catalogue_z() -> None:
    """N_g (no-BH) == the 3D GW Gaussian point-evaluated at z_g exactly."""
    from darksiren_emri.physical_relations import dist_vectorized

    _install_worker_globals()
    kw = _gen_case()
    r = bs.single_host_likelihood(**kw)
    slot = bs.det_index_to_slot[kw["detection_index"]]
    dl = float(dist_vectorized(np.array([kw["host_z"]]), h=kw["h"])[0])
    x = np.array([[kw["host_phiS"], kw["host_qS"], dl / float(bs.det_d_L_arr[slot])]])
    expected = float(
        bs._mvn_pdf(x, bs.means_3d[slot], bs.cov_inv_3d[slot], float(bs.log_norm_3d[slot]))[0]
    )
    assert r[0] == expected


def test_generator_denominators_byte_identical_to_volume_deconv() -> None:
    """The D_g columns (diagnostic in this mode) keep the volume_deconv kernel
    machinery bit-for-bit — the mode changes numerators only."""
    _install_worker_globals()
    kw_vd = dict(_case_grid()["near_photoz_match_vd_4d"])
    kw_gm = _gen_case()
    r_vd = bs.single_host_likelihood(**kw_vd)
    r_gm = bs.single_host_likelihood(**kw_gm)
    assert r_gm[1] == r_vd[1]  # 3D denominator
    assert r_gm[3] == r_vd[3]  # 4D denominator
    # ... and the numerators genuinely differ (point vs kernel-smoothed).
    assert r_gm[0] != r_vd[0]


def test_generator_batch_equals_scalar_bitwise() -> None:
    """Batched kernel row-reproduces the scalar kernel bit-for-bit in the new
    mode (multi-host, heterogeneous z/M), both channels."""
    _install_worker_globals()
    kw = _gen_case()
    perturb = [1.0, 1.05, 0.95]
    scalar_rows = []
    for fac in perturb:
        skw = dict(kw)
        skw["host_z"] = kw["host_z"] * fac
        scalar_rows.append(bs.single_host_likelihood(**skw))
    scalar = np.array(scalar_rows, dtype=np.float64)
    _install_worker_globals()
    batch = bs.single_host_likelihood_batch(
        np.full(3, kw["host_phiS"]),
        np.full(3, kw["host_qS"]),
        np.array([kw["host_z"] * f for f in perturb]),
        np.full(3, kw["host_z_error"]),
        np.full(3, kw["host_M"]),
        np.full(3, kw["host_M_error"]),
        detection_index=kw["detection_index"],
        h=kw["h"],
        evaluate_with_bh_mass=True,
        normalization_mode="generator_marginal",
    )
    assert batch.shape == scalar.shape
    assert (batch == scalar).all()


def test_volume_deconv_numerator_collapses_to_point_as_sigma_to_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Limiting case: sigma_eff -> 0 collapses the volume-deconvolved z-kernel
    to a delta, recovering the generator point numerator (both channels).

    The quadrature must RESOLVE the collapsing kernel, so the host-window
    variant of the same kernel (``volume_trunc``: numerator integrated over
    [z_g - 4 sigma, z_g + 4 sigma]) is used — the event-window modes place no
    node on a 1e-9-wide spike by construction. The residual-PV floor is patched
    to 0 so sigma_eff is driven by host_z_error alone.
    """
    monkeypatch.setattr(bs, "SIGMA_V_PEC_KM_S", 0.0)
    _install_worker_globals()
    kw = dict(_case_grid()["near_photoz_match_vd_4d"])
    kw["host_z_error"] = 1e-9
    kw["normalization_mode"] = "volume_trunc"  # host-window kernel quadrature
    kw_gm = dict(kw)
    kw_gm["normalization_mode"] = "generator_marginal"
    r_kernel = bs.single_host_likelihood(**kw)
    r_gm = bs.single_host_likelihood(**kw_gm)
    assert r_kernel[0] == pytest.approx(r_gm[0], rel=1e-8)  # no-BH numerator
    assert r_kernel[2] == pytest.approx(r_gm[2], rel=1e-8)  # with-BH numerator


# ── Wiring: evaluate() guards, CLI, defaults ─────────────────────────────────


def test_generator_marginal_accepted_without_warning() -> None:
    """evaluate() accepts 'generator_marginal' with no calibration warning."""
    instance = object.__new__(BayesianStatistics)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with contextlib.suppress(AttributeError):
            BayesianStatistics.evaluate(
                instance,
                None,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                0.73,
                normalization_mode="generator_marginal",
            )


def test_generator_marginal_rejects_smear_global_selection() -> None:
    """The mode is DEFINED point/point: --smear_global_selection must raise."""
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(ValueError, match="point/point"):
        BayesianStatistics.evaluate(
            instance,
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            0.73,
            normalization_mode="generator_marginal",
            smear_global_selection=True,
        )


def test_invalid_dgen_catalog_selection_rejected() -> None:
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(ValueError, match="dgen_catalog_selection"):
        BayesianStatistics.evaluate(
            instance,
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            0.73,
            normalization_mode="generator_marginal",
            dgen_catalog_selection="5d_wrong",
        )


def test_cli_exposes_generator_marginal_as_default() -> None:
    """generator_marginal is the production default since 2026-07-26;
    volume_deconv stays selectable as the legacy mode."""
    args = Arguments.create(["wd", "--evaluate", "--normalization_mode", "volume_deconv"])
    assert args.normalization_mode == "volume_deconv"
    args_default = Arguments.create(["wd", "--evaluate"])
    assert args_default.normalization_mode == "generator_marginal"
    assert args_default.pdet_z_resolved is True
    args_legacy = Arguments.create(["wd", "--evaluate", "--no-pdet_z_resolved"])
    assert args_legacy.pdet_z_resolved is False


def test_library_defaults_production_stack() -> None:
    sig = inspect.signature(BayesianStatistics.evaluate)
    assert sig.parameters["normalization_mode"].default == "generator_marginal"
    assert sig.parameters["pdet_z_resolved"].default is True
    assert sig.parameters["dgen_catalog_selection"].default == "4d_exact"
    assert BayesianStatistics._normalization_mode == "generator_marginal"
    assert BayesianStatistics._dgen_catalog_selection == "4d_exact"


# ── Issue #40(a): host_z_kernel numerator decomposition flag ─────────────────


def test_host_z_kernel_auto_matches_historical_bundling_bitwise() -> None:
    """'auto' (default) reproduces the mode-bundled kernel bit-for-bit:
    generator_marginal default == explicit 'point'; volume_deconv default ==
    explicit 'volume_deconv'."""
    _install_worker_globals()
    kw_gm = _gen_case()
    r_auto = bs.single_host_likelihood(**kw_gm)
    r_point = bs.single_host_likelihood(**kw_gm, host_z_kernel="point")
    assert r_auto == r_point
    kw_vd = dict(_case_grid()["near_photoz_match_vd_4d"])
    r_vd_auto = bs.single_host_likelihood(**kw_vd)
    r_vd_expl = bs.single_host_likelihood(**kw_vd, host_z_kernel="volume_deconv")
    assert r_vd_auto == r_vd_expl


def test_host_z_kernel_decouples_numerator_from_normalization() -> None:
    """The decomposition works both ways at the kernel level (where the two
    modes differ ONLY in the numerator kernel substitution):
    generator_marginal + 'volume_deconv' == volume_deconv, and
    volume_deconv + 'point' == generator_marginal — all columns, bitwise."""
    _install_worker_globals()
    kw_vd = dict(_case_grid()["near_photoz_match_vd_4d"])
    kw_gm = _gen_case()
    r_vd = bs.single_host_likelihood(**kw_vd)
    r_gm = bs.single_host_likelihood(**kw_gm)
    r_gm_kernel = bs.single_host_likelihood(**kw_gm, host_z_kernel="volume_deconv")
    r_vd_point = bs.single_host_likelihood(**kw_vd, host_z_kernel="point")
    assert r_gm_kernel == r_vd
    assert r_vd_point == r_gm
    # sanity: the decoupled results genuinely differ from their mode defaults
    assert r_gm_kernel[0] != r_gm[0]
    assert r_vd_point[0] != r_vd[0]


def test_host_z_kernel_batch_matches_scalar_override() -> None:
    """Batched kernel honors the flag identically to the scalar kernel."""
    _install_worker_globals()
    kw = _gen_case()
    scalar = np.array(
        [bs.single_host_likelihood(**kw, host_z_kernel="volume_deconv")], dtype=np.float64
    )
    _install_worker_globals()
    batch = bs.single_host_likelihood_batch(
        np.full(1, kw["host_phiS"]),
        np.full(1, kw["host_qS"]),
        np.full(1, kw["host_z"]),
        np.full(1, kw["host_z_error"]),
        np.full(1, kw["host_M"]),
        np.full(1, kw["host_M_error"]),
        detection_index=kw["detection_index"],
        h=kw["h"],
        evaluate_with_bh_mass=True,
        normalization_mode="generator_marginal",
        host_z_kernel="volume_deconv",
    )
    assert (batch == scalar).all()


def test_host_z_kernel_rejects_unknown_value() -> None:
    """Unknown kernel names fail loudly at resolution time (e.g. the pending
    real-data 'pv_photoz' kernel, issue #40b — not yet derived)."""
    with pytest.raises(ValueError, match="host_z_kernel"):
        bs.resolve_host_z_kernel("pv_photoz", "generator_marginal")
    _install_worker_globals()
    kw = _gen_case()
    with pytest.raises(ValueError, match="host_z_kernel"):
        bs.single_host_likelihood(**kw, host_z_kernel="pv_photoz")


def test_cli_exposes_host_z_kernel_with_auto_default() -> None:
    """CLI: --host_z_kernel defaults to 'auto' (production path unchanged);
    'point'/'volume_deconv' selectable; library defaults match."""
    args_default = Arguments.create(["wd", "--evaluate"])
    assert args_default.host_z_kernel == "auto"
    args_point = Arguments.create(["wd", "--evaluate", "--host_z_kernel", "point"])
    assert args_point.host_z_kernel == "point"
    sig = inspect.signature(BayesianStatistics.evaluate)
    assert sig.parameters["host_z_kernel"].default == "auto"
    assert BayesianStatistics._host_z_kernel == "auto"
