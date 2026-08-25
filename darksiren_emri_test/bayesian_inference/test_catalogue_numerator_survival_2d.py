r"""Tests for the [P3-2D] with-BH catalogue-leg twin estimator counterfactual
flag (``--catalogue_numerator_survival_2d {off,mz_sel}`` +
``--catalogue_numerator_survival_2d_center {unset,raw,eff}``).

Spec: results/campaign51_20260728/realistic_20260729/
PREREGISTRATION_P3_2D_20260825.md §2(i). "off" (default) is byte-identical to
the pre-flag path in BOTH kernels (``single_host_likelihood`` scalar,
``single_host_likelihood_batch`` -- production dispatches exclusively through
the batch kernel via ``_starmap_host_batches``). "mz_sel" multiplies the
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

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import darksiren_emri.bayesian_inference.bayesian_statistics as bs
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
    from unittest.mock import MagicMock

    instance = object.__new__(bs.BayesianStatistics)
    with pytest.raises(ValueError, match="no silent default"):
        bs.BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            catalogue_numerator_survival_2d="mz_sel",
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
def test_cli_flag_defaults_to_off_and_unset() -> None:
    args = Arguments.create([".", "--evaluate"])
    assert args.catalogue_numerator_survival_2d == "off"
    assert args.catalogue_numerator_survival_2d_center == "unset"
    assert args.to_dict()["catalogue_numerator_survival_2d"] == "off"
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
    args = Arguments.create([".", "--evaluate", "--catalogue_numerator_survival_2d", "mz_sel"])
    with pytest.raises(ArgumentsError, match="no silent default"):
        args.validate()
