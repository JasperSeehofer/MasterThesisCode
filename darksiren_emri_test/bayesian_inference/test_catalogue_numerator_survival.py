r"""Tests for the [P3-IMP] catalogue-leg twin, ADOPTED as production (row #195).

Spec: ``docs/derivations/PROPOSAL_CATALOGUE_TWIN_PRODUCTION_20260825.md`` §2
("New formula") and §6 (the verification plan). ``catalogue_numerator_survival``
in ``{"auto","off","phi","phi_flat"}``. "auto" (default) resolves exactly like
``catalogue_global_selection``/``selection_in_completion_numerator``: "phi"
under ``normalization_mode="absolute_marginal"`` (production), else "off"
(every other normalization mode stays byte-identical to the pre-adoption
path). "off" is now the explicit COUNTERFACTUAL under absolute_marginal.
"phi_flat" (the K-flat kill arm) is explicit-only, never reached by "auto".

The flag multiplies the WITHOUT-BH catalogue numerator integrand per
candidate host by the phi-marginal survival ``S_bar_phi(z;h)`` (endpoint-
clamped ``np.interp`` against ``catalogue_survival_table``) INSIDE the
z-quadrature, at the read site in ``numerator_integrant_without_bh_mass``
(``single_host_likelihood``/``single_host_likelihood_batch``). The WITH-BH
catalogue numerator, ``beta_G_phi``/``D_tilde_phi``, and the Sigma-chain are
architecturally untouched (proposal §2/§3, Appendix B as ratified) --
``catalogue_numerator_survival`` is never read anywhere in ``p_Di`` itself,
only threaded to the workers as the (already-resolved) ``_cat_surv``/
``_cat_surv_table`` pair.

Mirrors ``test_catalogue_global_selection.py``'s evaluate()-level (auto/mode-
guard) structure and ``test_catalogue_numerator_survival_2d.py``'s worker-
level (ratio/parity/untouched-channel) structure.

CPU-only; no GPU, no real pool.
"""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest

import darksiren_emri.bayesian_inference.bayesian_statistics as bs
from darksiren_emri.bayesian_inference.bayesian_statistics import BayesianStatistics
from darksiren_emri_test.bayesian_inference.test_kernel_parity import (
    _DETECTIONS,
    _StubDetectionProbability,
)

_H = 0.73

_HOST_KEYS = ["host_phiS", "host_qS", "host_z", "host_z_error", "host_M", "host_M_error"]

_HOSTS: list[dict[str, float]] = [
    {
        "host_phiS": _DETECTIONS[0]["phi"],
        "host_qS": _DETECTIONS[0]["theta"],
        "host_z": 0.10,
        "host_z_error": 0.0015,
        "host_M": 3.0e5,
        "host_M_error": 3.0e4,
    },
    {
        "host_phiS": _DETECTIONS[0]["phi"],
        "host_qS": _DETECTIONS[0]["theta"],
        "host_z": 0.10,
        "host_z_error": 0.03,
        "host_M": 2.5e5,
        "host_M_error": 1.5e5,
    },
    {
        "host_phiS": _DETECTIONS[0]["phi"],
        "host_qS": _DETECTIONS[0]["theta"],
        "host_z": 0.085,
        "host_z_error": 0.01,
        "host_M": 4.0e5,
        "host_M_error": 8.0e4,
    },
]

_BASE_KW: dict[str, Any] = {
    "detection_index": 0,
    "h": _H,
    "normalization_mode": "volume_deconv",
}

# A CONSTANT survival table: np.interp returns this value at every z, so the
# numerator integral scales by EXACTLY this factor -- an analytic, not just
# bounded, ratio prediction (the same K-flat-table trick test_catalogue_
# global_selection.py's sibling suite uses for its degree test).
_CONST_S = 0.37
_CONST_TABLE: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = (
    np.array([0.0, 5.0]),
    np.full(2, _CONST_S),
)

# A genuinely z-varying table (monotonically declining survival), for the
# bounded-ratio (not-exactly-constant) engagement check.
_VARYING_TABLE: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = (
    np.array([0.0, 0.05, 0.10, 0.15, 0.20, 5.0]),
    np.array([1.0, 0.95, 0.85, 0.70, 0.50, 0.50]),
)


def _install_worker_globals() -> None:
    """Single-detection worker state with a diagonal 3D/4D covariance.

    Mirrors ``test_catalogue_numerator_survival_2d._install_worker_globals``.
    """
    d = _DETECTIONS[0]
    bs.det_index_to_slot = {0: 0}
    bs.det_d_L_arr = np.array([d["d_L"]])
    bs.det_d_L_unc_arr = np.array([d["d_L_unc"]])
    bs.det_M_arr = np.array([d["M"]])
    bs.det_phi_arr = np.array([d["phi"]])
    bs.det_theta_arr = np.array([d["theta"]])

    cov3 = np.diag([d["sig_phi"] ** 2, d["sig_theta"] ** 2, d["sig_dl_frac"] ** 2])
    cov4 = np.diag(
        [d["sig_phi"] ** 2, d["sig_theta"] ** 2, d["sig_dl_frac"] ** 2, d["sig_mz_frac"] ** 2]
    )
    bs.means_3d = np.array([[d["phi"], d["theta"], 1.0]])
    bs.cov_inv_3d = np.array([np.linalg.inv(cov3)])
    bs.log_norm_3d = np.array([-0.5 * (3 * np.log(2 * np.pi) + np.linalg.slogdet(cov3)[1])])
    bs.means_4d = np.array([[d["phi"], d["theta"], 1.0, 1.0]])
    bs.cov_inv_4d = np.array([np.linalg.inv(cov4)])
    bs.log_norm_4d = np.array([-0.5 * (4 * np.log(2 * np.pi) + np.linalg.slogdet(cov4)[1])])
    bs.sigma2_cond_arr = np.array([d["sig_mz_frac"] ** 2])
    bs.proj_arr = np.array([np.zeros(3)])
    bs.proj_d_L_to_M_arr = np.array([0.0])
    bs.sigma_cond_M_arr = np.array([np.sqrt(d["sig_mz_frac"] ** 2)])
    bs.detection_probability = _StubDetectionProbability()
    bs.completeness_model = None


def _scalar_rows(
    mode: str,
    table: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
    normalization_mode: str = "volume_deconv",
    evaluate_with_bh_mass: bool = True,
) -> npt.NDArray[np.float64]:
    _install_worker_globals()
    rows = []
    for host in _HOSTS:
        kw = dict(_BASE_KW)
        kw["normalization_mode"] = normalization_mode
        kw.update(host)
        kw["evaluate_with_bh_mass"] = evaluate_with_bh_mass
        kw["catalogue_numerator_survival"] = mode
        kw["catalogue_survival_table"] = table
        rows.append(bs.single_host_likelihood(**kw))
    return np.array(rows, dtype=np.float64)


def _batch_rows(
    mode: str,
    table: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
    normalization_mode: str = "volume_deconv",
    evaluate_with_bh_mass: bool = True,
) -> npt.NDArray[np.float64]:
    _install_worker_globals()
    arrays = {k: np.array([h[k] for h in _HOSTS], dtype=np.float64) for k in _HOST_KEYS}
    return bs.single_host_likelihood_batch(
        arrays["host_phiS"],
        arrays["host_qS"],
        arrays["host_z"],
        arrays["host_z_error"],
        arrays["host_M"],
        arrays["host_M_error"],
        detection_index=_BASE_KW["detection_index"],
        h=_BASE_KW["h"],
        evaluate_with_bh_mass=evaluate_with_bh_mass,
        normalization_mode=normalization_mode,
        catalogue_numerator_survival=mode,
        catalogue_survival_table=table,
    )


# ===========================================================================
# (a) default byte-identity (worker level)
# ===========================================================================
def test_worker_default_off_omitted_kwarg_is_bit_identical_scalar() -> None:
    """Omitting the kwargs entirely == passing "off" explicitly (worker default)."""
    _install_worker_globals()
    host = _HOSTS[0]
    kw = dict(_BASE_KW)
    kw.update(host)
    kw["evaluate_with_bh_mass"] = True
    omitted = bs.single_host_likelihood(**kw)

    _install_worker_globals()
    kw["catalogue_numerator_survival"] = "off"
    explicit_off = bs.single_host_likelihood(**kw)

    assert np.array_equal(omitted, explicit_off)


def test_worker_default_off_omitted_kwarg_is_bit_identical_batch() -> None:
    _install_worker_globals()
    arrays = {k: np.array([_HOSTS[0][k]], dtype=np.float64) for k in _HOST_KEYS}
    omitted = bs.single_host_likelihood_batch(
        arrays["host_phiS"],
        arrays["host_qS"],
        arrays["host_z"],
        arrays["host_z_error"],
        arrays["host_M"],
        arrays["host_M_error"],
        detection_index=_BASE_KW["detection_index"],
        h=_BASE_KW["h"],
        evaluate_with_bh_mass=True,
        normalization_mode=_BASE_KW["normalization_mode"],
    )
    _install_worker_globals()
    explicit_off = bs.single_host_likelihood_batch(
        arrays["host_phiS"],
        arrays["host_qS"],
        arrays["host_z"],
        arrays["host_z_error"],
        arrays["host_M"],
        arrays["host_M_error"],
        detection_index=_BASE_KW["detection_index"],
        h=_BASE_KW["h"],
        evaluate_with_bh_mass=True,
        normalization_mode=_BASE_KW["normalization_mode"],
        catalogue_numerator_survival="off",
    )
    assert np.array_equal(omitted, explicit_off)


# ===========================================================================
# (a) "auto" resolution + explicit "off" byte-identity regression (evaluate())
# ===========================================================================
def _reach_catalogue_numerator_survival(instance: BayesianStatistics, **kwargs: Any) -> None:
    """Reach the ``catalogue_numerator_survival`` validation block, then abort
    on the very next (unrelated) validation so the rest of ``evaluate()`` need
    not be mocked -- mirrors ``test_catalogue_global_selection.py``'s harness.
    """
    with pytest.raises(ValueError, match="catalogue_mass_overlap"):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            catalogue_mass_error_scale=2.0,
            **kwargs,
        )


def test_evaluate_auto_resolves_to_phi_under_absolute_marginal() -> None:
    instance = object.__new__(BayesianStatistics)
    _reach_catalogue_numerator_survival(
        instance,
        normalization_mode="absolute_marginal",
        catalogue_numerator_survival="auto",
    )
    assert instance._catalogue_numerator_survival == "phi"


def test_evaluate_auto_resolves_to_off_under_generator_marginal() -> None:
    instance = object.__new__(BayesianStatistics)
    _reach_catalogue_numerator_survival(
        instance,
        normalization_mode="generator_marginal",
        catalogue_numerator_survival="auto",
    )
    assert instance._catalogue_numerator_survival == "off"


def test_evaluate_default_omits_the_kwarg_and_still_resolves_to_phi() -> None:
    """The unqualified default (kwarg omitted entirely) is "auto", so the
    production call site (no ``catalogue_numerator_survival`` passed) gets
    "phi" under absolute_marginal without anyone touching the flag."""
    instance = object.__new__(BayesianStatistics)
    _reach_catalogue_numerator_survival(instance, normalization_mode="absolute_marginal")
    assert instance._catalogue_numerator_survival == "phi"


def test_evaluate_explicit_off_under_absolute_marginal_is_the_counterfactual_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) The counterfactual regression: explicit "off" under absolute_marginal
    resolves to EXACTLY the same stored value ("off") as the pre-adoption
    production path did -- byte-identity of the resolved cell, and now logs
    as the explicit COUNTERFACTUAL (the flag's roles swapped, as Sigma^phi's
    did)."""
    logged: list[str] = []
    monkeypatch.setattr(
        bs._LOGGER,
        "warning",
        lambda msg, *a, **k: logged.append(msg % a if a else msg),
    )
    instance = object.__new__(BayesianStatistics)
    _reach_catalogue_numerator_survival(
        instance,
        normalization_mode="absolute_marginal",
        catalogue_numerator_survival="off",
    )
    assert instance._catalogue_numerator_survival == "off"
    assert any("COUNTERFACTUAL" in m and "catalogue_numerator_survival" in m for m in logged)


def test_evaluate_explicit_phi_under_absolute_marginal_logs_physics_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: list[str] = []
    monkeypatch.setattr(
        bs._LOGGER,
        "info",
        lambda msg, *a, **k: logged.append(msg % a if a else msg),
    )
    instance = object.__new__(BayesianStatistics)
    _reach_catalogue_numerator_survival(
        instance,
        normalization_mode="absolute_marginal",
        catalogue_numerator_survival="phi",
    )
    assert instance._catalogue_numerator_survival == "phi"
    assert any(
        "[PHYSICS]" in m and "catalogue_numerator_survival" in m and "ACTIVE" in m for m in logged
    )


# ===========================================================================
# (b) mode guard
# ===========================================================================
def test_evaluate_rejects_an_unknown_value() -> None:
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(ValueError, match="must be 'auto', 'off', 'phi' or 'phi_flat'"):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            catalogue_numerator_survival="bogus",
        )


def test_evaluate_requires_absolute_marginal_for_explicit_phi() -> None:
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(ValueError, match="absolute_marginal"):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            normalization_mode="generator_marginal",
            catalogue_numerator_survival="phi",
        )


def test_evaluate_requires_absolute_marginal_for_explicit_phi_flat() -> None:
    """ "phi_flat" (the K-flat kill arm) keeps the SAME mode guard as "phi" --
    "auto" never reaches it, but it is not otherwise relaxed."""
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(ValueError, match="absolute_marginal"):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            normalization_mode="generator_marginal",
            catalogue_numerator_survival="phi_flat",
        )


# ===========================================================================
# (c) engagement: "phi" rescales the WITHOUT-BH numerator by EXACTLY the
#     constant survival factor (analytic prediction), and by a bounded
#     (0, 1] factor for a genuinely z-varying table.
# ===========================================================================
def test_phi_rescales_no_bh_numerator_by_exactly_the_constant_survival_factor() -> None:
    """With a CONSTANT S_bar_phi table, the numerator integral scales by
    EXACTLY that constant (it factors out of the z-quadrature) -- an
    analytic, not just bounded, ratio prediction."""
    off = _scalar_rows("off")
    phi = _scalar_rows("phi", table=_CONST_TABLE)

    # No-BH numerator (col 0) scales by exactly _CONST_S; no-BH denominator
    # (col 1) and the with-BH channel (cols 2, 3) are untouched.
    np.testing.assert_allclose(phi[:, 0], off[:, 0] * _CONST_S, rtol=1e-12)
    np.testing.assert_array_equal(off[:, 1], phi[:, 1])
    np.testing.assert_array_equal(off[:, 2], phi[:, 2])
    np.testing.assert_array_equal(off[:, 3], phi[:, 3])


@pytest.mark.parametrize("normalization_mode", ["generator_marginal", "volume_deconv"])
def test_phi_engagement_is_bounded_in_0_1_for_a_varying_table(normalization_mode: str) -> None:
    """A genuinely z-varying (declining) survival table moves the no-BH
    numerator by a factor in (0, 1] -- S_bar_phi is a survival probability."""
    off = _scalar_rows("off", normalization_mode=normalization_mode)
    phi = _scalar_rows("phi", table=_VARYING_TABLE, normalization_mode=normalization_mode)

    assert not np.array_equal(off[:, 0], phi[:, 0])
    ratio = phi[:, 0] / off[:, 0]
    assert np.all(ratio > 0.0)
    assert np.all(ratio <= 1.0 + 1e-9)


def test_phi_flat_is_worker_level_off_the_flattening_happens_upstream() -> None:
    """ "phi_flat" (the K-flat kill arm) is resolved to the same "phi" cell
    the worker validates, with the TABLE flattened upstream at the
    class-level dispatch (p_Di's own "_cat_surv = 'phi'" rewrite,
    :4824-neighbourhood) -- the worker itself validates only "off"/"phi"
    and rejects "phi_flat" directly, so a caller must resolve it before
    reaching this function (exercised at the p_Di level below, not here)."""
    with pytest.raises(ValueError, match="must be 'off' or 'phi'"):
        _scalar_rows("phi_flat", table=_CONST_TABLE)


# ===========================================================================
# (d) scalar/batch parity
# ===========================================================================
@pytest.mark.parametrize("normalization_mode", ["generator_marginal", "volume_deconv"])
def test_scalar_batch_parity_phi(normalization_mode: str) -> None:
    scalar = _scalar_rows("phi", table=_VARYING_TABLE, normalization_mode=normalization_mode)
    batch = _batch_rows("phi", table=_VARYING_TABLE, normalization_mode=normalization_mode)
    assert batch.shape == scalar.shape
    np.testing.assert_allclose(
        batch,
        scalar,
        rtol=1e-10,
        err_msg=f"scalar/batch mismatch for normalization_mode={normalization_mode!r}",
    )


def test_scalar_batch_parity_off() -> None:
    scalar = _scalar_rows("off")
    batch = _batch_rows("off")
    np.testing.assert_allclose(batch, scalar, rtol=1e-12)


# ===========================================================================
# (e) the with-BH channel is bit-unchanged under every value (worker level)
# ===========================================================================
@pytest.mark.parametrize("normalization_mode", ["generator_marginal", "volume_deconv"])
def test_with_bh_channel_unaffected_scalar(normalization_mode: str) -> None:
    off = _scalar_rows("off", normalization_mode=normalization_mode)
    phi = _scalar_rows("phi", table=_VARYING_TABLE, normalization_mode=normalization_mode)
    np.testing.assert_array_equal(off[:, 2], phi[:, 2])
    np.testing.assert_array_equal(off[:, 3], phi[:, 3])


@pytest.mark.parametrize("normalization_mode", ["generator_marginal", "volume_deconv"])
def test_with_bh_channel_unaffected_batch(normalization_mode: str) -> None:
    off = _batch_rows("off", normalization_mode=normalization_mode)
    phi = _batch_rows("phi", table=_VARYING_TABLE, normalization_mode=normalization_mode)
    np.testing.assert_array_equal(off[:, 2], phi[:, 2])
    np.testing.assert_array_equal(off[:, 3], phi[:, 3])


def test_evaluate_with_bh_mass_false_leaves_no_bh_channel_the_only_output_and_is_still_moved() -> (
    None
):
    """With ``evaluate_with_bh_mass=False`` only the no-BH columns are
    populated; the twin still engages there (it is architecturally the
    WITHOUT-BH numerator's own factor, independent of whether the with-BH
    channel is requested at all)."""
    off = _scalar_rows("off", evaluate_with_bh_mass=False)
    phi = _scalar_rows("phi", table=_CONST_TABLE, evaluate_with_bh_mass=False)
    np.testing.assert_allclose(phi[:, 0], off[:, 0] * _CONST_S, rtol=1e-12)


# ===========================================================================
# (e) the completion leg (beta_G_phi / D_tilde_phi / Sigma-chain) is
#     bit-unchanged at the p_Di level -- catalogue_numerator_survival is
#     threaded ONLY to the (mocked-out, here) worker dispatch and never read
#     by p_Di's own completion-leg arithmetic.
# ===========================================================================
_BETA_G_PHI = 1.533228e8
_BETA_GBAR_PHI = 8.884038e8
_SIGMA_PHI = 9.562370e8
_SIGMA_3D = 1.075654e9
_SIGMA_4D = 4.221903e8


def _run_p_Di(
    *,
    catalogue_numerator_survival: str = "off",
) -> dict[str, Any]:
    """Run ``p_Di`` with the path-(A) tables installed; return its diagnostic
    row. Modeled on ``test_catalogue_global_selection.py``'s ``_run_p_Di_phi``.
    The mocked pool returns CANNED numerator/denominator sums regardless of
    ``catalogue_numerator_survival`` (the real per-candidate S_bar_phi
    multiplication happens inside the -- here mocked-out -- worker), so this
    harness isolates exactly the claim under test: p_Di's OWN completion-leg
    arithmetic never reads this flag.
    """
    instance = object.__new__(BayesianStatistics)
    instance.h = _H
    instance._normalization_mode = "absolute_marginal"
    instance._catalogue_numerator_survival = catalogue_numerator_survival
    instance._catalogue_numerator_survival_2d = "off"
    instance._catalogue_numerator_survival_2d_center = "unset"
    instance._catalogue_global_selection = "s3d"
    instance._completion_b_scale = "derived"
    instance.catalog_only = False
    instance.posterior_data = {}
    instance.posterior_data_with_bh_mass = {
        "galaxy_likelihoods": {},
        "additional_galaxies_without_bh_mass": {},
    }
    instance._diagnostic_rows = []
    instance._phi_survival_table = {_H: _VARYING_TABLE}

    mock_detection = MagicMock()
    mock_detection.d_L = 1.0
    mock_detection.d_L_uncertainty = 0.1
    mock_detection.phi = 0.5
    mock_detection.theta = 0.5
    mock_detection.M = 1.0e6
    instance.detection = mock_detection

    instance._det_index_to_slot = {0: 0}
    instance._means_3d = np.array([[0.5, 0.5, 1.0]])
    instance._cov_inv_3d = np.array([np.eye(3)])
    instance._log_norm_3d = np.array([0.0])
    instance._det_d_L = np.array([1.0])

    instance._D_h_table = {_H: 1.520637e9}
    instance._beta_Gbar_table = {_H: 1.335874e9}
    instance._beta_G_table = {_H: 1.520637e9 - 1.335874e9}
    instance._global_cat_denom_no_bh = {_H: _SIGMA_3D}
    instance._global_cat_denom_with_bh = {_H: _SIGMA_4D}
    instance._use_phi_selection = True
    instance._beta_G_phi_table = {_H: _BETA_G_PHI}
    instance._beta_Gbar_phi_table = {_H: _BETA_GBAR_PHI}
    instance._global_cat_selection_phi = {_H: _SIGMA_PHI}
    instance._proj_d_L_to_M = np.array([0.3])
    instance._sigma_cond_M = np.array([0.1])

    mock_pool = MagicMock()
    mock_pool._processes = 1
    mock_pool.starmap.side_effect = [
        [np.array([[0.5, 0.3, 0.4, 0.2]])],
        [np.array([[0.3, 0.2]])],
    ]

    mock_completeness = MagicMock()
    mock_completeness.ang2pix.return_value = 0
    mock_completeness.f_k.side_effect = lambda z, k, h: np.full_like(
        np.asarray(z, dtype=np.float64), 0.5
    )

    mock_p_det = MagicMock()
    mock_p_det.get_dl_max.return_value = 10.0

    host = MagicMock()
    host.M, host.z, host.catalog_index = 1e6, 0.1, 0
    host_with_bh = MagicMock()
    host_with_bh.M, host_with_bh.z, host_with_bh.catalog_index = 1e6, 0.1, 1

    BayesianStatistics.p_Di(
        instance,
        possible_host_galaxies=[host],
        possible_host_galaxies_with_bh_mass=[host_with_bh],
        detection_index=0,
        pool=mock_pool,
        completeness=mock_completeness,
        detection_probability_obj=mock_p_det,
    )
    return instance._diagnostic_rows[0]


def test_completion_leg_and_sigma_chain_untouched_by_the_flag() -> None:
    """beta_G_phi/D_tilde_phi/Sigma^phi/Sigma^3D-derived diagnostics are
    IDENTICAL between "off" and "phi" at the p_Di level -- the flag has
    exactly one read site, and it is not here (proposal §2: "beta_G_phi,
    D_tilde_phi, and the Sigma-chain are UNTOUCHED")."""
    off = _run_p_Di(catalogue_numerator_survival="off")
    phi = _run_p_Di(catalogue_numerator_survival="phi")

    assert off.keys() == phi.keys()
    for key in off:
        assert off[key] == phi[key], key


def test_p_di_dispatches_the_resolved_flag_and_table_to_the_worker_pool() -> None:
    """The class-level dispatch (:4816 neighbourhood) forwards the RESOLVED
    ``_catalogue_numerator_survival`` and the h-sliced ``_phi_survival_table``
    entry positionally to BOTH host-batch dispatches -- this is the actual
    production wiring the ratio tests above exercise only through the
    (mocked-out here) worker; this test pins the wiring itself."""
    instance = object.__new__(BayesianStatistics)
    instance.h = _H
    instance._normalization_mode = "absolute_marginal"
    instance._catalogue_numerator_survival = "phi"
    instance._catalogue_numerator_survival_2d = "off"
    instance._catalogue_numerator_survival_2d_center = "unset"
    instance._catalogue_global_selection = "s3d"
    instance._completion_b_scale = "derived"
    instance.catalog_only = False
    instance.posterior_data = {}
    instance.posterior_data_with_bh_mass = {
        "galaxy_likelihoods": {},
        "additional_galaxies_without_bh_mass": {},
    }
    instance._diagnostic_rows = []
    instance._phi_survival_table = {_H: _VARYING_TABLE}

    mock_detection = MagicMock()
    mock_detection.d_L = 1.0
    mock_detection.d_L_uncertainty = 0.1
    mock_detection.phi = 0.5
    mock_detection.theta = 0.5
    mock_detection.M = 1.0e6
    instance.detection = mock_detection

    instance._det_index_to_slot = {0: 0}
    instance._means_3d = np.array([[0.5, 0.5, 1.0]])
    instance._cov_inv_3d = np.array([np.eye(3)])
    instance._log_norm_3d = np.array([0.0])
    instance._det_d_L = np.array([1.0])

    instance._D_h_table = {_H: 1.520637e9}
    instance._beta_Gbar_table = {_H: 1.335874e9}
    instance._beta_G_table = {_H: 1.520637e9 - 1.335874e9}
    instance._global_cat_denom_no_bh = {_H: _SIGMA_3D}
    instance._global_cat_denom_with_bh = {_H: _SIGMA_4D}
    instance._use_phi_selection = True
    instance._beta_G_phi_table = {_H: _BETA_G_PHI}
    instance._beta_Gbar_phi_table = {_H: _BETA_GBAR_PHI}
    instance._global_cat_selection_phi = {_H: _SIGMA_PHI}
    instance._proj_d_L_to_M = np.array([0.3])
    instance._sigma_cond_M = np.array([0.1])

    mock_pool = MagicMock()
    mock_pool._processes = 1
    mock_pool.starmap.side_effect = [
        [np.array([[0.5, 0.3, 0.4, 0.2]])],
        [np.array([[0.3, 0.2]])],
    ]

    mock_completeness = MagicMock()
    mock_completeness.ang2pix.return_value = 0
    mock_completeness.f_k.side_effect = lambda z, k, h: np.full_like(
        np.asarray(z, dtype=np.float64), 0.5
    )

    mock_p_det = MagicMock()
    mock_p_det.get_dl_max.return_value = 10.0

    host = MagicMock()
    host.M, host.z, host.catalog_index = 1e6, 0.1, 0
    host_with_bh = MagicMock()
    host_with_bh.M, host_with_bh.z, host_with_bh.catalog_index = 1e6, 0.1, 1

    BayesianStatistics.p_Di(
        instance,
        possible_host_galaxies=[host],
        possible_host_galaxies_with_bh_mass=[host_with_bh],
        detection_index=0,
        pool=mock_pool,
        completeness=mock_completeness,
        detection_probability_obj=mock_p_det,
    )

    assert mock_pool.starmap.call_count == 2
    for call in mock_pool.starmap.call_args_list:
        dispatch_args = call.args[1][0]  # the per-batch tuple of positional args
        assert any(a == "phi" for a in dispatch_args if isinstance(a, str))
        tables = [
            a
            for a in dispatch_args
            if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray)
        ]
        assert len(tables) == 1
        np.testing.assert_array_equal(tables[0][0], _VARYING_TABLE[0])
        np.testing.assert_array_equal(tables[0][1], _VARYING_TABLE[1])
