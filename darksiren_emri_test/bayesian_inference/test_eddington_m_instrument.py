"""Regression tests for the tilt-ledger battery instrument E.

``--eddington_m {on,off}`` is a pre-registered, author-approved counterfactual
instrument (results/prod2d_closure_20260818/PREREGISTRATION_TILT_BATTERY.md v2,
sections 1/2/6). Under "off" the 2D catalogue leg's ``_host_M_eff``/
``host_M_eff`` is assigned the raw (unshifted) ``host_M`` instead of
``eddington_shifted_host_mass(host_M, host_M_error)`` -- the SAME single
assignment feeds the numerator ``mu_gal`` AND the per-host D_g erf-sum in both
kernels (scalar ``single_host_likelihood``, batch
``single_host_likelihood_batch``).

Gates pinned here (limiting cases from the gate presentation, §6):
  (i)   "on" (default) is byte-identical to the pre-flag path (no value
        change vs. calling without the kwarg at all).
  (ii)  scalar<->batch parity under both "on" and "off".
  (iii) sigma_M -> 0 collapses "on" onto "off" (the Eddington shift vanishes
        in the bare-Gaussian limit -- eddington_shifted_host_mass's own
        sigma<=0 guard already returns the raw mass).
  (iv)  the 1D (without-BH-mass) channel is bit-identical under "off" (the
        shift lives in the with-BH path only).
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import darksiren_emri.bayesian_inference.bayesian_statistics as bs
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
    "base_seed": 20260819,
}

_MODES = ["on", "off"]


def _install_worker_globals() -> None:
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
    eddington_m: str | None, evaluate_with_bh_mass: bool = True
) -> npt.NDArray[np.float64]:
    _install_worker_globals()
    rows = []
    for host in _HOSTS:
        kw = dict(_BASE_KW)
        kw.update(host)
        kw["evaluate_with_bh_mass"] = evaluate_with_bh_mass
        if eddington_m is not None:
            kw["eddington_m"] = eddington_m
        rows.append(bs.single_host_likelihood(**kw))
    return np.array(rows, dtype=np.float64)


def _batch_rows(
    eddington_m: str | None, evaluate_with_bh_mass: bool = True
) -> npt.NDArray[np.float64]:
    _install_worker_globals()
    arrays = {k: np.array([h[k] for h in _HOSTS], dtype=np.float64) for k in _HOST_KEYS}
    kw: dict[str, Any] = dict(
        detection_index=_BASE_KW["detection_index"],
        h=_BASE_KW["h"],
        evaluate_with_bh_mass=evaluate_with_bh_mass,
        normalization_mode=_BASE_KW["normalization_mode"],
    )
    if eddington_m is not None:
        kw["eddington_m"] = eddington_m
    return bs.single_host_likelihood_batch(
        arrays["host_phiS"],
        arrays["host_qS"],
        arrays["host_z"],
        arrays["host_z_error"],
        arrays["host_M"],
        arrays["host_M_error"],
        **kw,
    )


# ── (i) "on" == default (no kwarg passed at all) ─────────────────────────────


def test_on_is_byte_identical_to_default_scalar() -> None:
    default = _scalar_rows(None)
    on = _scalar_rows("on")
    assert np.array_equal(default, on)


def test_on_is_byte_identical_to_default_batch() -> None:
    default = _batch_rows(None)
    on = _batch_rows("on")
    assert np.array_equal(default, on)


# ── (ii) scalar <-> batch parity under both modes ────────────────────────────


@pytest.mark.parametrize("mode", _MODES)
def test_scalar_batch_parity(mode: str) -> None:
    scalar = _scalar_rows(mode)
    batch = _batch_rows(mode)
    assert batch.shape == scalar.shape
    np.testing.assert_allclose(
        batch, scalar, rtol=1e-12, err_msg=f"scalar/batch mismatch for eddington_m={mode!r}"
    )


# ── (iii) sigma_M -> 0 collapses "on" onto "off" ─────────────────────────────


def test_sigma_m_zero_collapses_on_to_off_scalar() -> None:
    _install_worker_globals()
    host = dict(_HOSTS[0])
    host["host_M_error"] = 0.0
    kw_on = dict(_BASE_KW)
    kw_on.update(host)
    kw_on["evaluate_with_bh_mass"] = True
    kw_on["eddington_m"] = "on"
    on = bs.single_host_likelihood(**kw_on)

    kw_off = dict(kw_on)
    kw_off["eddington_m"] = "off"
    off = bs.single_host_likelihood(**kw_off)

    assert np.array_equal(on, off)


def test_sigma_m_zero_collapses_on_to_off_batch() -> None:
    _install_worker_globals()
    hosts = [dict(_HOSTS[0])]
    hosts[0]["host_M_error"] = 0.0
    arrays = {k: np.array([h[k] for h in hosts], dtype=np.float64) for k in _HOST_KEYS}
    common = dict(
        detection_index=_BASE_KW["detection_index"],
        h=_BASE_KW["h"],
        evaluate_with_bh_mass=True,
        normalization_mode=_BASE_KW["normalization_mode"],
    )
    on = bs.single_host_likelihood_batch(
        arrays["host_phiS"],
        arrays["host_qS"],
        arrays["host_z"],
        arrays["host_z_error"],
        arrays["host_M"],
        arrays["host_M_error"],
        eddington_m="on",
        **common,
    )
    off = bs.single_host_likelihood_batch(
        arrays["host_phiS"],
        arrays["host_qS"],
        arrays["host_z"],
        arrays["host_z_error"],
        arrays["host_M"],
        arrays["host_M_error"],
        eddington_m="off",
        **common,
    )
    assert np.array_equal(on, off)


# ── (iv) 1D channel bit-identical under "off" (the shift is with-BH only) ───


def test_1d_channel_bit_identical_under_off_scalar() -> None:
    on = _scalar_rows("on", evaluate_with_bh_mass=False)
    off = _scalar_rows("off", evaluate_with_bh_mass=False)
    assert np.array_equal(on, off)


def test_1d_channel_bit_identical_under_off_batch() -> None:
    on = _batch_rows("on", evaluate_with_bh_mass=False)
    off = _batch_rows("off", evaluate_with_bh_mass=False)
    assert np.array_equal(on, off)


# ── production DOES respond to sigma_M under "on" (sanity: the limiting-case
#    test above is not vacuous) ───────────────────────────────────────────────


def test_on_responds_to_nonzero_sigma_m_sanity() -> None:
    on_zero_sigma = _scalar_rows(None)  # host_M_error > 0 by construction in _HOSTS
    off = _scalar_rows("off")
    # At least one candidate's with-BH numerator (col 2) must differ between
    # the shifted ("on") and raw ("off") mass prior when sigma_M > 0.
    assert not np.array_equal(on_zero_sigma[:, 2], off[:, 2])


# ── rejection of unknown modes (defence in depth) ────────────────────────────


def test_unknown_mode_raises_scalar() -> None:
    _install_worker_globals()
    kw = dict(_BASE_KW)
    kw.update(_HOSTS[0])
    kw["evaluate_with_bh_mass"] = True
    kw["eddington_m"] = "bogus"
    with pytest.raises(ValueError):
        bs.single_host_likelihood(**kw)


def test_unknown_mode_raises_batch() -> None:
    _install_worker_globals()
    arrays = {k: np.array([_HOSTS[0][k]], dtype=np.float64) for k in _HOST_KEYS}
    with pytest.raises(ValueError):
        bs.single_host_likelihood_batch(
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
            eddington_m="bogus",
        )
