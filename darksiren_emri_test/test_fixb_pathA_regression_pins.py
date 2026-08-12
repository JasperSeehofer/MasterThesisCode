"""Regression pins guarding the Fix-B path-(A) joint C9+C8 mixture change.

Written and committed **before** the physics change (``FIXB_PATHA_PACKAGE.md``
§5), so the numerical diff of the change is visible in the test diff rather
than hidden inside it. Two classes of pin live here:

**Must-not-move** (the change is wrong if any of these moves):

* **R1** — the 1D (without-BH-mass) numerator/denominator content is
  reproduced to ``rtol=1e-12`` (issue #51 gate P5, "path (A) touches
  selection objects and the 2D completion leg only, never the host kernel"):
  pinned as a values golden of the deterministic 1D block below (columns
  0-1 are the physics content ``N_g``/``D_g``; columns 2-3 are STAT-04
  quadrature-outside-grid diagnostics carried along in the same array, not
  numerator content themselves). A tolerance compare rather than a byte
  digest is deliberate — see the test docstring for why.
* **R1/(iii-a)** — the ``generator_marginal`` assembly consumes the *legacy*
  shared ``beta_G``/``beta_Gbar``/``D`` tables and must stay byte-identical;
  path (A) installs NEW phi-convention tables consumed by
  ``absolute_marginal`` only.
* **R2** — the ``catalog_only`` branch fact (``p_i == L_cat``, no completion).
* **R3** — the 1D channel is bitwise invariant under a rescaling of the 2D
  catalogue leg (the 1D numerator carries no mass density; gate (iv),
  ``cov_obs = cov_4d[:3, :3]``).
* **R4** — the legacy ``D``/``beta_Gbar``/``beta_G`` quadrature formulas
  reproduce their values on an analytic p_det, and the recorded production
  tables of record satisfy ``beta_G = D - beta_Gbar`` exactly.
* **R5** — ``V_f(0.73) = 2.3237e8`` (draw-side completeness volume).
* **R6** — the scatter guards still pass in the nominal configuration.

**Will move** (asserted at their OLD values so the change shows up as a diff):

* the operative in-catalogue mixture weight at ``h = 0.73``,
  ``w_G = beta_G/D = 0.1215039`` — path (A) replaces it by
  ``w~_G = alpha_G^phi/D~^phi`` (0.070802 on the delivered convention).
* the legacy single-ratio assembly ``p_i = (beta_G L_cat + B_num)/D`` with the
  same ``B_num`` in both channels (path (A) gives the 2D channel its own
  ``B_num_wbh = B_num * g_i``-inside and its own ``alpha_G^phi`` prefactor).

Selection tables of record are the production run's own log lines (campaign
#51 / gate-B, ``fixb_measurements/prod_selection_log_extract.txt``, replicated
in ``fixb_measurements/iid_pathA_results.json``); they are quoted here as
plain constants because reproducing them needs the 200k-injection pool and the
20.8M-row reduced catalogue (not unit-test inputs).

CPU-only; no GPU marker.
"""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest

from darksiren_emri.bayesian_inference.bayesian_statistics import (
    precompute_completeness_population_volume,
    precompute_completion_denominator,
    precompute_missing_completion_denominator,
    validate_scatter_guards,
)
from darksiren_emri_test.test_partition_norm_restructure import _run_p_Di, _w0

_H = 0.73

# --------------------------------------------------------------------------
# Selection tables of record (production logs, h = 0.73) — provenance:
# fixb_measurements/prod_selection_log_extract.txt / iid_pathA_results.json.
# --------------------------------------------------------------------------
_D_OF_RECORD = 1.520637e9
_BETA_GBAR_OF_RECORD = 1.335874e9
_BETA_G_OF_RECORD = 1.847630e8
_SIGMA_3D_OF_RECORD = 1.075654e9
_SIGMA_4D_OF_RECORD = 4.221903e8
# The pin that WILL move: the legacy operative mixture weight.
_W_G_LEGACY_OF_RECORD = 0.1215039


# ==========================================================================
# R4 — legacy selection tables and quadratures
# ==========================================================================
def test_R4_recorded_tables_satisfy_partition_identity() -> None:
    """beta_G = D - beta_Gbar holds on the tables of record (must not move)."""
    assert _D_OF_RECORD - _BETA_GBAR_OF_RECORD == pytest.approx(_BETA_G_OF_RECORD, rel=2e-6)
    assert _BETA_G_OF_RECORD / _D_OF_RECORD == pytest.approx(_W_G_LEGACY_OF_RECORD, rel=1e-5)


def _analytic_p_det() -> MagicMock:
    """p_det(d_L) = exp(-d_L) mock — smooth, positive, h-independent."""
    mock = MagicMock()
    mock.get_dl_max.return_value = 4.0
    del mock.z_resolved  # keep _zres_z_kwargs on the flag-off (pre-FIX-2) path
    del mock.wbh_z_resolved
    del mock.survival_per_band  # force the isotropic fallback branch
    del mock.detection_probability_without_bh_mass_sky

    def _p(
        d_L: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
        *,
        h: float,
    ) -> npt.NDArray[np.float64]:
        return np.exp(-np.asarray(d_L, dtype=np.float64))

    mock.detection_probability_without_bh_mass_interpolated_zero_fill.side_effect = _p
    return mock


def _constant_completeness(f_const: float) -> MagicMock:
    mock = MagicMock()
    del mock.f_pixels  # no sky-aware branch
    del mock.pixel_centers
    mock.f_bar.side_effect = lambda z, h: np.full_like(np.asarray(z, dtype=np.float64), f_const)
    return mock


def test_R4_legacy_D_and_beta_Gbar_quadrature_pins() -> None:
    """D(h) / beta_Gbar(h) on an analytic p_det — the legacy formulas, pinned."""
    p_det = _analytic_p_det()
    comp = _constant_completeness(0.25)
    D = precompute_completion_denominator(
        h_values=[_H],
        detection_probability_obj=p_det,
        Omega_m=0.2726,
        Omega_DE=0.7274,
        completeness=comp,
        z_max_cap=1.5,
    )
    beta_Gbar = precompute_missing_completion_denominator(
        h_values=[_H],
        detection_probability_obj=p_det,
        completeness=comp,
        z_max_cap=1.5,
    )
    # Pinned at the pre-change values (measured on this analytic mock).
    assert D[_H] == pytest.approx(3.4039191e8, rel=1e-6)
    assert beta_Gbar[_H] == pytest.approx(2.5529393e8, rel=1e-6)
    # (1 - f) with constant f: beta_Gbar = (1-f) D exactly.
    assert beta_Gbar[_H] == pytest.approx(0.75 * D[_H], rel=1e-12)


# ==========================================================================
# R5 — draw-side completeness volume anchor
# ==========================================================================
def test_R5_V_f_anchor_at_0p73() -> None:
    """V_f(0.73) = 2.3237e8 Mpc^3/sr on the frozen m_th completeness cache."""
    from darksiren_emri.galaxy_catalogue.pixel_completeness import from_cache_or_build

    completeness = from_cache_or_build()
    V_f = precompute_completeness_population_volume([_H], completeness)
    assert V_f[_H] == pytest.approx(2.3237e8, rel=1e-4)


# ==========================================================================
# R6 — scatter guards
# ==========================================================================
def test_R6_scatter_guards_pass_in_nominal_configuration() -> None:
    """The nominal (unscattered-inference) guard configuration still passes."""
    validate_scatter_guards(
        normalization_mode="absolute_marginal",
        host_z_kernel="volume_deconv",
        host_mass_kernel="gaussian",
        catalogue_scattered=False,
    )
    # Scattered catalogue: generator_marginal is refused (guard 2).
    with pytest.raises(ValueError):
        validate_scatter_guards(
            normalization_mode="generator_marginal",
            host_z_kernel="volume_deconv",
            host_mass_kernel="trunc_lognormal",
            catalogue_scattered=True,
        )


# ==========================================================================
# R1 — the 1D numerator content is byte-identical
# ==========================================================================
def _deterministic_1d_numerator() -> npt.NDArray[np.float64]:
    """A deterministic 1D (without-BH-mass) host-kernel numerator/denominator block.

    Exercises ``single_host_likelihood_batch``'s 1D leg on a fixed synthetic
    host set with the kernel-parity stub p_det, so the digest below is a
    content hash of the 1D ``(N_g, D_g)`` pair the mixture consumes.
    """
    import darksiren_emri.bayesian_inference.bayesian_statistics as bs
    from darksiren_emri_test.bayesian_inference.test_kernel_parity import (  # noqa: PLC0415
        _install_worker_globals,
    )

    _install_worker_globals()
    bs.redshift_lower_integration_limit = 1e-6
    bs.redshift_upper_integration_limit = 1.5
    bs.bh_mass_lower_integration_limit = 1e4
    bs.bh_mass_upper_integration_limit = 1e7

    n = 6
    host_phi = np.full(n, 1.2)
    host_theta = np.full(n, 1.0)
    host_z = np.linspace(0.05, 0.12, n)
    host_z_err = np.full(n, 0.02)
    host_M = np.geomspace(1e5, 1e7, n)
    host_M_err = 0.1 * host_M

    out = bs.single_host_likelihood_batch(
        host_phi,
        host_theta,
        host_z,
        host_z_err,
        host_M,
        host_M_err,
        0,
        _H,
        False,
        "absolute_marginal",
        "volume_deconv",
        "gaussian",
    )
    return np.ascontiguousarray(np.asarray(out, dtype=np.float64))


# Golden values for `_deterministic_1d_numerator()`, generated on this exact
# code path (git blame this constant, do not hand-edit). Shape (6, 4); rows
# are the 6 synthetic hosts, columns are:
#   0  N_g   — numerator_without_bh_mass (physics content, gate-P5-guarded)
#   1  D_g   — denominator_without_bh_mass (physics content, gate-P5-guarded)
#   2  quadrature_weight_outside_grid_numerator   (STAT-04 diagnostic, all 0)
#   3  quadrature_weight_outside_grid_denominator (STAT-04 diagnostic)
# Captured on the default (AVX-512 / X86_V4) numpy SIMD dispatch path.
_R1_GOLDEN: npt.NDArray[np.float64] = np.array(
    [
        [65.786031214837507, 0.94691259482656309, 0.0, 0.017035447849954366],
        [219.71706143750842, 0.93703502605520039, 0.0, 0.015238410507965405],
        [497.73979390007048, 0.9265400539580918, 0.0, 0.01376338380938529],
        [750.08852477352355, 0.91560873357452732, 0.0, 0.0],
        [742.04801834291038, 0.90436628298765964, 0.0, 0.0],
        [477.48316195576768, 0.89289753854892617, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def test_R1_one_d_numerator_content_digest() -> None:
    """The 1D numerator/denominator content must not move (issue #51 gate P5).

    Pins ``_deterministic_1d_numerator()`` against a values golden
    (``_R1_GOLDEN`` above) rather than a byte digest, compared at
    ``rtol=1e-12`` / ``atol=0.0``.

    Why not bitwise: this array is saturated with ``np.exp`` calls (the
    Gaussian/MVN kernels and the ``p_det = exp(-d_L/5)`` stub), and NumPy
    2.4's x86-64 wheels dispatch `exp`/`log`/`log1p`/`expm1`/`cbrt`/`x**1.5`
    to different (each <1-ULP-accurate, but mutually non-bit-identical) SIMD
    loops depending on the host CPU's available instruction set (baseline
    X86_V2 vs AVX2 X86_V3 vs AVX-512 X86_V4). GitHub's runner fleet is
    heterogeneous in this respect, so a bitwise md5 pin here tested which VM
    the job landed on, not the physics: measured divergence between the
    X86_V4 and X86_V3 paths on this exact array is 7/24 elements changed by
    1-3 ULP, max relative difference 3.73e-16 — the numerator/denominator
    content itself does not move.

    Why rtol=1e-12: about 4 orders of magnitude above the measured 3.73e-16
    hardware noise floor (so it does not re-introduce the flake), and about
    6 orders of magnitude below anything that could shift an H0 MAP (so it
    still catches a real regression in the host kernel).

    To re-verify hardware stability locally:
        NPY_DISABLE_CPU_FEATURES=X86_V4 uv run pytest \
            darksiren_emri_test/test_fixb_pathA_regression_pins.py::test_R1_one_d_numerator_content_digest
    should pass just the same as the unset-env default run.
    """
    vec = _deterministic_1d_numerator()
    np.testing.assert_allclose(
        vec,
        _R1_GOLDEN,
        rtol=1e-12,
        atol=0.0,
        err_msg=(
            "1D numerator/denominator content changed beyond hardware SIMD "
            "noise (rtol=1e-12); path (A) must not touch the host kernel "
            "(regression R1, #51 gate P5)."
        ),
    )


# ==========================================================================
# R1/(iii-a) — generator_marginal keeps the LEGACY shared tables
# ==========================================================================
def test_R1_generator_marginal_assembly_uses_legacy_tables() -> None:
    """D_gen = Sigma_glob_sel/n_hat_w + beta_Gbar (legacy tables), unchanged."""
    from darksiren_emri_test.test_generator_marginal_mode import (  # noqa: PLC0415
        _run_p_Di_gen,
    )

    row = _run_p_Di_gen(
        norm_mode="generator_marginal",
        with_hosts=True,
        f_const=0.5,
        D_h=1.0e9,
        beta_Gbar=0.5e9,
        global_no_bh=2.0,
        global_with_bh=1.5,
        W_cat=1.0e9,
        V_f=2.0e8,
    )
    n_hat_w = 1.0e9 / 2.0e8
    a_cat = 1.5 / n_hat_w  # 4d_exact convention
    D_gen = a_cat + 0.5e9
    assert row["w_G"] == pytest.approx(a_cat / D_gen, rel=1e-12)
    assert row["combined_with_bh"] == pytest.approx(
        (row["L_cat_with_bh"] + row["B_num"]) / D_gen, rel=1e-12
    )
    assert row["combined_no_bh"] == pytest.approx(
        (row["L_cat_no_bh"] + row["B_num"]) / D_gen, rel=1e-12
    )


# ==========================================================================
# R2 — catalog_only branch fact
# ==========================================================================
def _run_catalog_only() -> dict[str, Any]:
    from darksiren_emri.bayesian_inference.bayesian_statistics import BayesianStatistics

    instance = object.__new__(BayesianStatistics)
    instance.h = _H
    instance._normalization_mode = "absolute_marginal"
    instance.catalog_only = True
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
    instance.detection = mock_detection
    mock_pool = MagicMock()
    mock_pool._processes = 1
    mock_pool.starmap.side_effect = [
        [np.array([[0.5, 0.3, 0.4, 0.2]])],
        [np.array([[0.3, 0.2]])],
    ]
    host_a = MagicMock()
    host_a.M, host_a.z, host_a.catalog_index = 1e6, 0.1, 0
    host_b = MagicMock()
    host_b.M, host_b.z, host_b.catalog_index = 1e6, 0.1, 1
    BayesianStatistics.p_Di(
        instance,
        possible_host_galaxies=[host_a],
        possible_host_galaxies_with_bh_mass=[host_b],
        detection_index=0,
        pool=mock_pool,
        completeness=MagicMock(),
        detection_probability_obj=MagicMock(),
    )
    return instance._diagnostic_rows[0]


def test_R2_catalog_only_branch_fact() -> None:
    """catalog_only: w_G = 1, B_num = L_comp = 0, p_i = L_cat (both channels)."""
    row = _run_catalog_only()
    assert row["w_G"] == 1.0
    assert row["B_num"] == 0.0
    assert row["L_comp"] == 0.0
    assert row["combined_no_bh"] == row["L_cat_no_bh"]
    assert row["combined_with_bh"] == row["L_cat_with_bh"]


# ==========================================================================
# R3 — 1D measure invariance (bitwise)
# ==========================================================================
def test_R3_one_d_channel_bitwise_invariant_to_2d_leg_rescaling() -> None:
    """Rescaling the 2D catalogue leg leaves the 1D channel bitwise unchanged."""
    base = _run_p_Di(
        f_const=0.4,
        D_h=1.0e9,
        beta_Gbar=0.6e9,
        global_no_bh=2.0,
        global_with_bh=1.5,
        norm_mode="absolute_marginal",
    )
    scaled = _run_p_Di(
        f_const=0.4,
        D_h=1.0e9,
        beta_Gbar=0.6e9,
        global_no_bh=2.0,
        global_with_bh=1.5 * np.e,
        norm_mode="absolute_marginal",
    )
    assert scaled["combined_no_bh"] == base["combined_no_bh"]
    assert scaled["L_cat_no_bh"] == base["L_cat_no_bh"]
    assert scaled["B_num"] == base["B_num"]


# ==========================================================================
# WILL MOVE — the legacy mixture weight and assembly at h = 0.73
# ==========================================================================
def test_legacy_operative_weight_of_record_will_move() -> None:
    """OLD: the operative in-catalogue weight is w_G = beta_G/D = 0.1215039.

    Path (A) replaces the operative weight by w~_G = alpha_G^phi/D~^phi
    (0.070802 on the delivered convention, FIXB_PATHA_PACKAGE.md §5 R7/R8);
    the legacy value survives only as a RENAMED diagnostic (``w_G_legacy``).
    """
    row = _run_p_Di(
        f_const=0.5,
        D_h=_D_OF_RECORD,
        beta_Gbar=_BETA_GBAR_OF_RECORD,
        global_no_bh=_SIGMA_3D_OF_RECORD,
        global_with_bh=_SIGMA_4D_OF_RECORD,
        norm_mode="absolute_marginal",
    )
    assert row["w_G"] == pytest.approx(_W_G_LEGACY_OF_RECORD, rel=1e-5)


def test_legacy_single_ratio_assembly_will_move() -> None:
    """OLD: both channels share beta_G, D and the SAME B_num (no g_i, no alpha)."""
    row = _run_p_Di(
        f_const=0.5,
        D_h=_D_OF_RECORD,
        beta_Gbar=_BETA_GBAR_OF_RECORD,
        global_no_bh=_SIGMA_3D_OF_RECORD,
        global_with_bh=_SIGMA_4D_OF_RECORD,
        norm_mode="absolute_marginal",
    )
    beta_G = _D_OF_RECORD - _BETA_GBAR_OF_RECORD
    w0 = _w0()
    assert row["L_cat_no_bh"] == pytest.approx(w0 * (0.3 + 0.5) / _SIGMA_3D_OF_RECORD, rel=1e-12)
    assert row["L_cat_with_bh"] == pytest.approx(w0 * 0.4 / _SIGMA_4D_OF_RECORD, rel=1e-12)
    assert row["combined_no_bh"] == pytest.approx(
        (beta_G * row["L_cat_no_bh"] + row["B_num"]) / _D_OF_RECORD, rel=1e-12
    )
    assert row["combined_with_bh"] == pytest.approx(
        (beta_G * row["L_cat_with_bh"] + row["B_num"]) / _D_OF_RECORD, rel=1e-12
    )
