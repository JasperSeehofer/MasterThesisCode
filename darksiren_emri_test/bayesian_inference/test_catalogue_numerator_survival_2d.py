r"""Tests for the [P3-2D] with-BH catalogue-leg twin estimator counterfactual
flag (``--catalogue_numerator_survival_2d {off,mz_sel}`` +
``--catalogue_numerator_survival_2d_center {unset,raw,eff}``).

Spec: results/campaign51_20260728/realistic_20260729/
PREREGISTRATION_P3_2D_20260825.md §2(i); PRODUCTION-DEFAULT FLIP (row #223
standing grant, charter node B7.3;
PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md): "mz_sel"/"eff" is now the
CLASS/__init__/evaluate()/CLI/main.evaluate/run_mirror_seed_inprocess
default; explicit "off" (any center) is the pre-adoption COUNTERFACTUAL and
stays byte-identical to the pre-flag path in BOTH kernels
(``single_host_likelihood`` scalar, ``single_host_likelihood_batch`` --
production dispatches exclusively through the batch kernel via
``_starmap_host_batches``; the KERNEL-LEVEL defaults of these two functions
are deliberately left at "off"/"unset" -- only the evaluate()-and-above
declaration sites flipped). "mz_sel" multiplies the
WITH-BH catalogue numerator's mass integrand by
``S_4D(d_L(z;h), x*M_z,det)`` inside the candidate's own mass quadrature (the
product-Gaussian identity, ``_mz_sel_2d_expectation``/``_batch``), consuming
the EXISTING ``detection_probability_with_bh_mass_interpolated`` accessor.
The WITHOUT-BH numerator is architecturally untouched (a distinct closure/
branch the flag is never threaded into).

The centering sub-option ("raw"=host_M, "eff"=host_M_eff) is REFUSED
("unset", the default) until explicitly set whenever the twin is engaged --
no silent default; the choice is PENDING the pre-execution review.

Mirrors ``test_catalogue_mass_overlap.py``'s structure (gates (i)-(v)) and
``test_catalogue_global_selection.py``'s CLI-plumbing section.

CPU-only; no GPU, no real pool.
"""

import inspect
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import darksiren_emri.bayesian_inference.bayesian_statistics as bs
import darksiren_emri.main as main_module
import darksiren_emri.validation.correspondence_1d as c1d
from darksiren_emri.arguments import Arguments
from darksiren_emri.exceptions import ArgumentsError
from darksiren_emri_test.bayesian_inference.test_kernel_parity import (
    _DETECTIONS,
    _StubDetectionProbability,
)

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
    "h": 0.73,
    "normalization_mode": "volume_deconv",
}

_CENTER_CASES: list[str] = ["raw", "eff"]
_MODE_CASES: list[str] = ["generator_marginal", "volume_deconv", "absolute_marginal"]


def _install_worker_globals() -> None:
    """Single-detection worker state with a diagonal 3D/4D covariance.

    Mirrors ``test_catalogue_mass_overlap._install_worker_globals``.
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
    normalization_mode: str = "volume_deconv",
    center: str = "unset",
    evaluate_with_bh_mass: bool = True,
) -> npt.NDArray[np.float64]:
    _install_worker_globals()
    rows = []
    for host in _HOSTS:
        kw = dict(_BASE_KW)
        kw["normalization_mode"] = normalization_mode
        kw.update(host)
        kw["evaluate_with_bh_mass"] = evaluate_with_bh_mass
        kw["catalogue_numerator_survival_2d"] = mode
        kw["catalogue_numerator_survival_2d_center"] = center
        rows.append(bs.single_host_likelihood(**kw))
    return np.array(rows, dtype=np.float64)


def _batch_rows(
    mode: str,
    normalization_mode: str = "volume_deconv",
    center: str = "unset",
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
        catalogue_numerator_survival_2d=mode,
        catalogue_numerator_survival_2d_center=center,
    )


# ===========================================================================
# (a) default byte-identity
# ===========================================================================
def test_default_off_omitted_kwarg_is_bit_identical_scalar() -> None:
    """Omitting the kwargs entirely == passing "off" explicitly (default)."""
    _install_worker_globals()
    host = _HOSTS[0]
    kw = dict(_BASE_KW)
    kw.update(host)
    kw["evaluate_with_bh_mass"] = True
    omitted = bs.single_host_likelihood(**kw)

    _install_worker_globals()
    kw["catalogue_numerator_survival_2d"] = "off"
    kw["catalogue_numerator_survival_2d_center"] = "unset"
    explicit_off = bs.single_host_likelihood(**kw)

    assert np.array_equal(omitted, explicit_off)


def _single_host_arrays(host: dict[str, float]) -> tuple[npt.NDArray[np.float64], ...]:
    """One host as six ``(1,)`` positional arrays, matching the batch kernel's
    positional signature (avoids a ``*list[...]`` splat, which mypy cannot
    check against a fixed-arity call)."""
    return tuple(np.array([host[k]], dtype=np.float64) for k in _HOST_KEYS)


def test_default_off_omitted_kwarg_is_bit_identical_batch() -> None:
    _install_worker_globals()
    phiS, qS, z, z_error, M, M_error = _single_host_arrays(_HOSTS[0])
    omitted = bs.single_host_likelihood_batch(
        phiS,
        qS,
        z,
        z_error,
        M,
        M_error,
        detection_index=_BASE_KW["detection_index"],
        h=_BASE_KW["h"],
        evaluate_with_bh_mass=True,
        normalization_mode=_BASE_KW["normalization_mode"],
    )
    _install_worker_globals()
    explicit_off = bs.single_host_likelihood_batch(
        phiS,
        qS,
        z,
        z_error,
        M,
        M_error,
        detection_index=_BASE_KW["detection_index"],
        h=_BASE_KW["h"],
        evaluate_with_bh_mass=True,
        normalization_mode=_BASE_KW["normalization_mode"],
        catalogue_numerator_survival_2d="off",
        catalogue_numerator_survival_2d_center="unset",
    )
    assert np.array_equal(omitted, explicit_off)


@pytest.mark.parametrize("normalization_mode", _MODE_CASES)
def test_off_matches_the_pre_flag_golden_across_modes(normalization_mode: str) -> None:
    """ "off" is byte-identical to a call that never mentions the flag at all,
    across every normalization mode the twin composes with."""
    _install_worker_globals()
    host = _HOSTS[1]
    kw = dict(_BASE_KW)
    kw["normalization_mode"] = normalization_mode
    kw.update(host)
    kw["evaluate_with_bh_mass"] = True
    pre_flag = bs.single_host_likelihood(**kw)

    _install_worker_globals()
    off = bs.single_host_likelihood(
        **kw, catalogue_numerator_survival_2d="off", catalogue_numerator_survival_2d_center="unset"
    )
    assert np.array_equal(pre_flag, off)


# ===========================================================================
# (b) mode guard
# ===========================================================================
def test_scalar_unknown_top_level_value_raises() -> None:
    _install_worker_globals()
    kw = dict(_BASE_KW)
    kw.update(_HOSTS[0])
    kw["evaluate_with_bh_mass"] = True
    kw["catalogue_numerator_survival_2d"] = "bogus"
    with pytest.raises(ValueError, match="must be 'off' or 'mz_sel'"):
        bs.single_host_likelihood(**kw)


def test_batch_unknown_top_level_value_raises() -> None:
    _install_worker_globals()
    phiS, qS, z, z_error, M, M_error = _single_host_arrays(_HOSTS[0])
    with pytest.raises(ValueError, match="must be 'off' or 'mz_sel'"):
        bs.single_host_likelihood_batch(
            phiS,
            qS,
            z,
            z_error,
            M,
            M_error,
            detection_index=_BASE_KW["detection_index"],
            h=_BASE_KW["h"],
            evaluate_with_bh_mass=True,
            normalization_mode=_BASE_KW["normalization_mode"],
            catalogue_numerator_survival_2d="bogus",
        )


def test_scalar_mz_sel_with_unset_center_raises() -> None:
    _install_worker_globals()
    kw = dict(_BASE_KW)
    kw.update(_HOSTS[0])
    kw["evaluate_with_bh_mass"] = True
    kw["catalogue_numerator_survival_2d"] = "mz_sel"
    with pytest.raises(ValueError, match="no silent default"):
        bs.single_host_likelihood(**kw)


def test_batch_mz_sel_with_unset_center_raises() -> None:
    _install_worker_globals()
    phiS, qS, z, z_error, M, M_error = _single_host_arrays(_HOSTS[0])
    with pytest.raises(ValueError, match="no silent default"):
        bs.single_host_likelihood_batch(
            phiS,
            qS,
            z,
            z_error,
            M,
            M_error,
            detection_index=_BASE_KW["detection_index"],
            h=_BASE_KW["h"],
            evaluate_with_bh_mass=True,
            normalization_mode=_BASE_KW["normalization_mode"],
            catalogue_numerator_survival_2d="mz_sel",
        )


def test_scalar_mz_sel_with_bogus_center_raises() -> None:
    _install_worker_globals()
    kw = dict(_BASE_KW)
    kw.update(_HOSTS[0])
    kw["evaluate_with_bh_mass"] = True
    kw["catalogue_numerator_survival_2d"] = "mz_sel"
    kw["catalogue_numerator_survival_2d_center"] = "bogus"
    with pytest.raises(ValueError, match="'raw' or 'eff'"):
        bs.single_host_likelihood(**kw)


@pytest.mark.parametrize("center", _CENTER_CASES)
def test_scalar_mz_sel_composed_with_mass_trunc_raises(center: str) -> None:
    """The twin is a guard pattern outside its implemented branch: it raises
    rather than silently no-opping when combined with mass_trunc."""
    _install_worker_globals()
    kw = dict(_BASE_KW)
    kw["normalization_mode"] = "mass_trunc"
    kw.update(_HOSTS[0])
    kw["evaluate_with_bh_mass"] = True
    kw["catalogue_numerator_survival_2d"] = "mz_sel"
    kw["catalogue_numerator_survival_2d_center"] = center
    with pytest.raises(ValueError, match="production Gaussian-product"):
        bs.single_host_likelihood(**kw)


@pytest.mark.parametrize("center", _CENTER_CASES)
def test_scalar_mz_sel_composed_with_catalogue_mass_overlap_raises(center: str) -> None:
    _install_worker_globals()
    kw = dict(_BASE_KW)
    kw.update(_HOSTS[0])
    kw["evaluate_with_bh_mass"] = True
    kw["catalogue_numerator_survival_2d"] = "mz_sel"
    kw["catalogue_numerator_survival_2d_center"] = center
    kw["catalogue_mass_overlap"] = "neutralized"
    with pytest.raises(ValueError, match="production Gaussian-product"):
        bs.single_host_likelihood(**kw)


def test_evaluate_rejects_an_unknown_top_level_value() -> None:
    from unittest.mock import MagicMock

    instance = object.__new__(bs.BayesianStatistics)
    with pytest.raises(ValueError, match="must be 'off' or 'mz_sel'"):
        bs.BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            catalogue_numerator_survival_2d="bogus",
        )


def test_evaluate_mz_sel_with_unset_center_raises() -> None:
    """Re-pinned for the row #223 production-default flip
    (PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md §6.1 (a-vi)): evaluate()'s
    center default is now "eff", so the refusal path requires "unset" to be
    passed explicitly rather than relying on the (former) default."""
    from unittest.mock import MagicMock

    instance = object.__new__(bs.BayesianStatistics)
    with pytest.raises(ValueError, match="no silent default"):
        bs.BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            catalogue_numerator_survival_2d="mz_sel",
            catalogue_numerator_survival_2d_center="unset",
        )


# ===========================================================================
# (c) engagement: "mz_sel" moves L_cat_with_bh, ratio sane in (0, 1]
# ===========================================================================
@pytest.mark.parametrize("center", _CENTER_CASES)
@pytest.mark.parametrize("normalization_mode", _MODE_CASES)
def test_mz_sel_moves_with_bh_numerator_by_a_survival_factor_in_0_1(
    normalization_mode: str, center: str
) -> None:
    off = _scalar_rows("off", normalization_mode=normalization_mode)
    mz = _scalar_rows("mz_sel", normalization_mode=normalization_mode, center=center)

    # With-BH numerator (col 2) moves; everything else (no-BH cols 0/1,
    # with-BH denominator col 3) is untouched.
    assert not np.array_equal(off[:, 2], mz[:, 2])
    np.testing.assert_array_equal(off[:, 0], mz[:, 0])
    np.testing.assert_array_equal(off[:, 1], mz[:, 1])
    np.testing.assert_array_equal(off[:, 3], mz[:, 3])

    ratio = mz[:, 2] / off[:, 2]
    assert np.all(ratio > 0.0)
    assert np.all(ratio <= 1.0 + 1e-9)  # E[S_4D] is a survival expectation, <= 1


def test_mz_sel_sharp_gw_mass_limit_matches_point_s4d() -> None:
    """Sharp-GW-mass limit (stage-0 §1): as sigma_cond -> 0, E[S_4D] ->
    S_4D(d_L, mu_cond*M_z,det) -- so the "mz_sel"/"off" ratio approaches the
    POINT S_4D value at the conditional mean mass, not some other number."""
    d = dict(_DETECTIONS[0])
    d["sig_mz_frac"] = 1.0e-6  # collapse the GW mass-conditional width

    def install_sharp() -> None:
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

    install_sharp()
    host = _HOSTS[0]
    kw = dict(_BASE_KW)
    kw.update(host)
    kw["evaluate_with_bh_mass"] = True
    off = bs.single_host_likelihood(**kw)
    mz = bs.single_host_likelihood(
        **kw, catalogue_numerator_survival_2d="mz_sel", catalogue_numerator_survival_2d_center="eff"
    )
    ratio = mz[2] / off[2]

    # The independently-computed point S_4D at mu_cond*M_z,det (the GW-measured
    # detector-frame mass, sharp limit): mu_cond ~= 1.0 (event term's own mean),
    # so the point mass is close to det_M = _DETECTIONS[0]["M"].
    stub = _StubDetectionProbability()
    point_s4d = float(
        stub.detection_probability_with_bh_mass_interpolated(
            np.array([bs.dist_vectorized(np.array([host["host_z"]]), h=kw["h"])[0]]),
            np.array([d["M"]]),
            np.array([host["host_phiS"]]),
            np.array([host["host_qS"]]),
            kw["h"],
        )[0]
    )
    assert ratio == pytest.approx(point_s4d, rel=5e-3)


def test_r5_sigma_gal_zero_limit_matches_point_s4d_at_host_mass() -> None:
    """R5 (§6.2 item 6 of the row #223 flip's regression plan;
    PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md §4 row 5): as sigma_gal -> 0
    (mass-certain host), the Eddington shift vanishes (proportional to
    sigma_rel^2, eddington_shifted_host_mass) so mu*->mu_gal->host_M(1+z)/
    M_z,det, and E[S] -> S_4D(d_L(z;h), host_M*(1+z)) -- Sigma^4D's own
    per-row point query (:2692). Isolated via normalization_mode=
    "generator_marginal" -- the code's genuine delta-kernel branch
    (:6872-6896; "historical bundling: delta kernel iff generator_marginal",
    :3411/:6273), which evaluates at z=host_z directly with NO z-quadrature,
    so the ratio reduces to a clean single-z point comparison with no
    quadrature-resolution artefact (collapsing host_z_error instead, under
    volume_deconv, was tried and underflows the 50-node host-z quadrature to
    ~0 well before this limit is reached -- disclosed, not used)."""
    d = dict(_DETECTIONS[0])

    def install_delta() -> None:
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

    install_delta()
    host = dict(_HOSTS[0])
    host["host_M_error"] = 1.0  # collapse sigma_gal (mass-certain host)
    kw = dict(_BASE_KW)
    kw["normalization_mode"] = "generator_marginal"  # the delta-kernel branch
    kw.update(host)
    kw["evaluate_with_bh_mass"] = True
    off = bs.single_host_likelihood(**kw)
    mz = bs.single_host_likelihood(
        **kw, catalogue_numerator_survival_2d="mz_sel", catalogue_numerator_survival_2d_center="eff"
    )
    ratio = mz[2] / off[2]

    # The independently-computed point S_4D at host_M*(1+host_z) (the
    # Eddington-shift-vanished, mass-certain-host detector-frame mass).
    stub = _StubDetectionProbability()
    point_mass = host["host_M"] * (1.0 + host["host_z"])
    point_s4d = float(
        stub.detection_probability_with_bh_mass_interpolated(
            np.array([bs.dist_vectorized(np.array([host["host_z"]]), h=kw["h"])[0]]),
            np.array([point_mass]),
            np.array([host["host_phiS"]]),
            np.array([host["host_qS"]]),
            kw["h"],
        )[0]
    )
    assert ratio == pytest.approx(point_s4d, rel=5e-3)


# ===========================================================================
# (d) scalar/batch parity
# ===========================================================================
@pytest.mark.parametrize("center", _CENTER_CASES)
@pytest.mark.parametrize("normalization_mode", _MODE_CASES)
def test_scalar_batch_parity(normalization_mode: str, center: str) -> None:
    scalar = _scalar_rows("mz_sel", normalization_mode=normalization_mode, center=center)
    batch = _batch_rows("mz_sel", normalization_mode=normalization_mode, center=center)
    assert batch.shape == scalar.shape
    np.testing.assert_allclose(
        batch,
        scalar,
        rtol=1e-10,
        err_msg=f"scalar/batch mismatch for normalization_mode={normalization_mode!r} center={center!r}",
    )


def test_scalar_batch_parity_off() -> None:
    scalar = _scalar_rows("off")
    batch = _batch_rows("off")
    np.testing.assert_allclose(batch, scalar, rtol=1e-12)


# ===========================================================================
# (e) the 1D (without-BH) channel is bit-unchanged in every case
# ===========================================================================
@pytest.mark.parametrize("center", _CENTER_CASES)
@pytest.mark.parametrize("normalization_mode", _MODE_CASES)
def test_1d_channel_unaffected_scalar(normalization_mode: str, center: str) -> None:
    off = _scalar_rows("off", normalization_mode=normalization_mode, evaluate_with_bh_mass=False)
    mz = _scalar_rows(
        "mz_sel", normalization_mode=normalization_mode, center=center, evaluate_with_bh_mass=False
    )
    assert np.array_equal(off, mz)


@pytest.mark.parametrize("center", _CENTER_CASES)
@pytest.mark.parametrize("normalization_mode", _MODE_CASES)
def test_1d_channel_unaffected_batch(normalization_mode: str, center: str) -> None:
    off = _batch_rows("off", normalization_mode=normalization_mode, evaluate_with_bh_mass=False)
    mz = _batch_rows(
        "mz_sel", normalization_mode=normalization_mode, center=center, evaluate_with_bh_mass=False
    )
    assert np.array_equal(off, mz)


# ===========================================================================
# (f) centering sub-option is load-bearing (raw != eff when eddington_m shifts
#     the mean)
# ===========================================================================
def test_raw_and_eff_centering_differ_when_eddington_shift_is_material() -> None:
    raw = _scalar_rows("mz_sel", center="raw")
    eff = _scalar_rows("mz_sel", center="eff")
    # eddington_m="on" (default) shifts the mean, so "raw" (host_M) and "eff"
    # (host_M_eff) feed materially different product-Gaussian means into the
    # S_4D quadrature -- the with-BH numerator (col 2) must differ.
    assert not np.allclose(raw[:, 2], eff[:, 2])


# ===========================================================================
# CLI plumbing
# ===========================================================================
def test_cli_flag_defaults_to_mz_sel_and_eff() -> None:
    """The row #223 production-default flip
    (PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md; charter node B7.3): the
    with-BH catalogue-leg twin is now the CLI default. Supersedes
    test_cli_flag_defaults_to_off_and_unset (renamed; see
    test_cli_flag_explicit_off_and_unset_parses_and_validates for the
    counterfactual pin)."""
    args = Arguments.create([".", "--evaluate"])
    assert args.catalogue_numerator_survival_2d == "mz_sel"
    assert args.catalogue_numerator_survival_2d_center == "eff"
    assert args.to_dict()["catalogue_numerator_survival_2d"] == "mz_sel"
    assert args.to_dict()["catalogue_numerator_survival_2d_center"] == "eff"
    args.validate()  # mz_sel + eff is the adopted production pair


def test_cli_flag_explicit_off_and_unset_parses_and_validates() -> None:
    """The pre-adoption COUNTERFACTUAL stays reachable and byte-identical:
    explicit "off"/"unset" parses, validates, and stamps to_dict() (§6.1
    (a-vi))."""
    args = Arguments.create(
        [
            ".",
            "--evaluate",
            "--catalogue_numerator_survival_2d",
            "off",
            "--catalogue_numerator_survival_2d_center",
            "unset",
        ]
    )
    assert args.catalogue_numerator_survival_2d == "off"
    assert args.catalogue_numerator_survival_2d_center == "unset"
    assert args.to_dict()["catalogue_numerator_survival_2d"] == "off"
    assert args.to_dict()["catalogue_numerator_survival_2d_center"] == "unset"
    args.validate()  # off + unset is fine (center only enforced when engaged)


def test_cli_flag_parses_mz_sel_with_center() -> None:
    args = Arguments.create(
        [
            ".",
            "--evaluate",
            "--catalogue_numerator_survival_2d",
            "mz_sel",
            "--catalogue_numerator_survival_2d_center",
            "eff",
        ]
    )
    assert args.catalogue_numerator_survival_2d == "mz_sel"
    assert args.catalogue_numerator_survival_2d_center == "eff"
    args.validate()


def test_cli_flag_rejects_unknown_top_level_value() -> None:
    with pytest.raises(SystemExit):
        Arguments.create([".", "--evaluate", "--catalogue_numerator_survival_2d", "bogus"])


def test_cli_flag_rejects_unknown_center_value() -> None:
    with pytest.raises(SystemExit):
        Arguments.create([".", "--evaluate", "--catalogue_numerator_survival_2d_center", "bogus"])


def test_cli_validate_refuses_mz_sel_with_unset_center() -> None:
    """Re-pinned for the row #223 production-default flip
    (PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md §6.1 (a-vi)): the CLI's
    center default is now "eff", so the refusal path requires "unset" to be
    passed explicitly rather than relying on the (former) default."""
    args = Arguments.create(
        [
            ".",
            "--evaluate",
            "--catalogue_numerator_survival_2d",
            "mz_sel",
            "--catalogue_numerator_survival_2d_center",
            "unset",
        ]
    )
    with pytest.raises(ArgumentsError, match="no silent default"):
        args.validate()


# ===========================================================================
# Production-default flip pins (row #223 standing grant, charter node B7.3;
# PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md §6.2)
# ===========================================================================
def test_six_site_default_trace_is_mz_sel_and_eff(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The end-to-end default trace (cf4f8a2a precedent): class attribute,
    __init__ instance default, evaluate() signature default, argparse
    default, main.evaluate() signature default, and
    run_mirror_seed_inprocess signature default ALL resolve to
    ("mz_sel", "eff") after the row #223 flip. The KERNEL-level defaults
    (single_host_likelihood, single_host_likelihood_batch) are deliberately
    EXCLUDED from this trace -- they stay "off"/"unset" by design (§1.4)."""
    # 1. class attribute
    assert bs.BayesianStatistics._catalogue_numerator_survival_2d == "mz_sel"
    assert bs.BayesianStatistics._catalogue_numerator_survival_2d_center == "eff"

    # 2. __init__ instance default (its own CSV reads are irrelevant to this
    # flag; point them at trivial fixtures, matching
    # test_evaluation_pipeline.py's monkeypatch pattern).
    csv_path = tmp_path / "crb.csv"
    csv_path.write_text("x\n1\n")
    monkeypatch.setattr(bs, "PREPARED_CRAMER_RAO_BOUNDS_PATH", str(csv_path))
    monkeypatch.setattr(bs, "CRAMER_RAO_BOUNDS_OUTPUT_PATH", str(csv_path))
    instance = bs.BayesianStatistics()
    assert instance._catalogue_numerator_survival_2d == "mz_sel"
    assert instance._catalogue_numerator_survival_2d_center == "eff"

    # 3. evaluate() signature default
    eval_sig = inspect.signature(bs.BayesianStatistics.evaluate)
    assert eval_sig.parameters["catalogue_numerator_survival_2d"].default == "mz_sel"
    assert eval_sig.parameters["catalogue_numerator_survival_2d_center"].default == "eff"

    # 4. argparse default (CLI)
    args = Arguments.create([".", "--evaluate"])
    assert args.catalogue_numerator_survival_2d == "mz_sel"
    assert args.catalogue_numerator_survival_2d_center == "eff"

    # 5. main.evaluate() (module-level) signature default
    main_eval_sig = inspect.signature(main_module.evaluate)
    assert main_eval_sig.parameters["catalogue_numerator_survival_2d"].default == "mz_sel"
    assert main_eval_sig.parameters["catalogue_numerator_survival_2d_center"].default == "eff"

    # 6. run_mirror_seed_inprocess signature default
    mirror_sig = inspect.signature(c1d.run_mirror_seed_inprocess)
    assert mirror_sig.parameters["catalogue_numerator_survival_2d"].default == "mz_sel"
    assert mirror_sig.parameters["catalogue_numerator_survival_2d_center"].default == "eff"

    # Kernel-level defaults are UNCHANGED by design (worker convenience only;
    # production always receives the resolved value from the dispatch).
    scalar_sig = inspect.signature(bs.single_host_likelihood)
    assert scalar_sig.parameters["catalogue_numerator_survival_2d"].default == "off"
    assert scalar_sig.parameters["catalogue_numerator_survival_2d_center"].default == "unset"
    batch_sig = inspect.signature(bs.single_host_likelihood_batch)
    assert batch_sig.parameters["catalogue_numerator_survival_2d"].default == "off"
    assert batch_sig.parameters["catalogue_numerator_survival_2d_center"].default == "unset"


def test_kernel_default_pair_bit_identical_to_explicit_mz_sel_eff() -> None:
    """The resolved production pair ("mz_sel", "eff") that evaluate() now
    dispatches by default produces rows bit-identical to explicitly passing
    that same pair to the kernel directly (there is no separate "default"
    code path inside the kernel -- it always receives an explicit value from
    the dispatch, :5070-5071/:5090-5091)."""
    explicit = _scalar_rows("mz_sel", center="eff")
    also_explicit = _scalar_rows(
        bs.BayesianStatistics._catalogue_numerator_survival_2d,
        center=bs.BayesianStatistics._catalogue_numerator_survival_2d_center,
    )
    assert np.array_equal(explicit, also_explicit)

    explicit_batch = _batch_rows("mz_sel", center="eff")
    also_explicit_batch = _batch_rows(
        bs.BayesianStatistics._catalogue_numerator_survival_2d,
        center=bs.BayesianStatistics._catalogue_numerator_survival_2d_center,
    )
    assert np.array_equal(explicit_batch, also_explicit_batch)


def test_evaluate_default_logs_physics_info_line(caplog: pytest.LogCaptureFixture) -> None:
    """G-2 log block (§6.1 (a-iii)), exercised through evaluate() (the block
    lives inside evaluate(), not __init__ -- __init__ only sets the bare
    instance defaults, unvalidated). The resolved production default
    ("mz_sel", "eff") emits one INFO line naming the flag ACTIVE. Deliberately
    paired with an invalid, UNRELATED flag (catalogue_global_selection) to
    force a deterministic, immediate ValueError right after our block runs
    (the very next validation in evaluate(), :3754-3757) -- this avoids
    driving evaluate() into real host/pool computation against MagicMock()
    stand-ins for galaxy_catalog/cosmological_model."""
    from unittest.mock import MagicMock

    instance = object.__new__(bs.BayesianStatistics)
    with caplog.at_level("INFO"), pytest.raises(ValueError, match="catalogue_global_selection"):
        bs.BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            catalogue_global_selection="bogus",
        )
    physics_msgs = [
        r
        for r in caplog.records
        if r.levelname == "INFO"
        and 'catalogue_numerator_survival_2d="mz_sel"' in r.message
        and "ACTIVE" in r.message
    ]
    assert len(physics_msgs) == 1


def test_evaluate_explicit_off_logs_counterfactual_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """G-2 log block (§6.1 (a-iii)): explicit "off" emits one WARNING naming
    it a COUNTERFACTUAL instead of the default's INFO/ACTIVE line. Same
    deterministic short-circuit as the test above."""
    from unittest.mock import MagicMock

    instance = object.__new__(bs.BayesianStatistics)
    with caplog.at_level("WARNING"), pytest.raises(ValueError, match="catalogue_global_selection"):
        bs.BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            catalogue_numerator_survival_2d="off",
            catalogue_global_selection="bogus",
        )
    counterfactual_msgs = [
        r
        for r in caplog.records
        if r.levelname == "WARNING"
        and 'catalogue_numerator_survival_2d="off"' in r.message
        and "COUNTERFACTUAL" in r.message
    ]
    assert len(counterfactual_msgs) == 1
    active_msgs = [r for r in caplog.records if "ACTIVE" in r.message]
    assert len(active_msgs) == 0
