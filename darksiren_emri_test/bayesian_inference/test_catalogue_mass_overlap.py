"""Regression tests for the prod2d closure counterfactual instrument.

``--catalogue_mass_overlap {production,neutralized,inflated}`` (+
``--catalogue_mass_error_scale``) is a pre-registered, author-approved
counterfactual instrument (results/prod2d_closure_20260818/
PREREGISTRATION_PROD_COUNTERFACTUAL.md v2, sections 1/2/6). It switches the
2D catalogue leg's per-candidate ``mz_integral`` in BOTH kernels
(``single_host_likelihood`` scalar, ``single_host_likelihood_batch``), since
production dispatches exclusively through the batch kernel via
``_starmap_host_batches``.

Gates pinned here (limiting cases from the gate presentation, §6):
  (i)   default ("production") is byte-identical to the pre-flag path
        (covered by the existing kernel-parity/batch-equivalence goldens,
        which never pass ``catalogue_mass_overlap`` and therefore exercise
        the default -- this file adds the NEW-mode coverage).
  (ii)  scalar<->batch parity under every mode.
  (iii) "inflated" k=1.0 is identical to "production".
  (iv)  "neutralized" is independent of the candidate's own mass/M_error.
  (v)   the 1D (without-BH-mass) channel is unaffected by the mode in every
        case (the mode only touches the with-BH-mass mz_integral).
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

# Base host candidates (spec-z match, photo-z match, offset) against detection 0.
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

_MODE_CASES: list[tuple[str, float]] = [
    ("production", 1.0),
    ("neutralized", 1.0),
    ("inflated", 0.5),
    ("inflated", 1.0),
    ("inflated", 2.0),
]


def _install_worker_globals() -> None:
    """Install a single-detection worker state with the d_L-only 2x2 block.

    Mirrors ``test_kernel_parity._install_worker_globals`` (diagonal 3D/4D
    covariance) but also installs ``proj_d_L_to_M_arr``/``sigma_cond_M_arr``
    -- the (N8) d_L-only conditional the "neutralized" mode reads
    (:2022 ``completion_mass_factor_g``), distinct from the full-3D-observed
    ``proj_arr``/``sigma2_cond_arr`` the candidate's OWN numerator uses. With
    a diagonal covariance the two conditionals coincide (cross-terms are
    zero), which is what is asserted below.
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
    # (N8) d_L-only block: diagonal cov4 => zero cross term => the d_L-only
    # conditional coincides with the marginal, exactly as sigma2_cond_arr
    # above (proj vanishes for both).
    bs.proj_d_L_to_M_arr = np.array([0.0])
    bs.sigma_cond_M_arr = np.array([np.sqrt(d["sig_mz_frac"] ** 2)])
    bs.detection_probability = _StubDetectionProbability()
    bs.completeness_model = None


def _scalar_rows(
    mode: str, scale: float, evaluate_with_bh_mass: bool = True
) -> npt.NDArray[np.float64]:
    _install_worker_globals()
    rows = []
    for host in _HOSTS:
        kw = dict(_BASE_KW)
        kw.update(host)
        kw["evaluate_with_bh_mass"] = evaluate_with_bh_mass
        kw["catalogue_mass_overlap"] = mode
        kw["catalogue_mass_error_scale"] = scale
        rows.append(bs.single_host_likelihood(**kw))
    return np.array(rows, dtype=np.float64)


def _batch_rows(
    mode: str, scale: float, evaluate_with_bh_mass: bool = True
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
        normalization_mode=_BASE_KW["normalization_mode"],
        catalogue_mass_overlap=mode,
        catalogue_mass_error_scale=scale,
    )


# ── (ii) scalar <-> batch parity under every mode ────────────────────────────


@pytest.mark.parametrize("mode,scale", _MODE_CASES)
def test_scalar_batch_parity_every_mode(mode: str, scale: float) -> None:
    scalar = _scalar_rows(mode, scale)
    batch = _batch_rows(mode, scale)
    assert batch.shape == scalar.shape
    np.testing.assert_allclose(
        batch,
        scalar,
        rtol=1e-12,
        err_msg=f"scalar/batch mismatch for catalogue_mass_overlap={mode!r} scale={scale!r}",
    )


# ── (iii) inflated k=1.0 == production (bit-identical, same inputs) ─────────


def test_inflated_k1_identical_to_production_scalar() -> None:
    production = _scalar_rows("production", 1.0)
    inflated = _scalar_rows("inflated", 1.0)
    assert np.array_equal(production, inflated)


def test_inflated_k1_identical_to_production_batch() -> None:
    production = _batch_rows("production", 1.0)
    inflated = _batch_rows("inflated", 1.0)
    assert np.array_equal(production, inflated)


# ── (iv) neutralized is independent of the candidate's own mass/M_error ─────


def test_neutralized_independent_of_host_mass_scalar() -> None:
    """Perturbing host_M and host_M_error leaves 'neutralized' unchanged."""
    _install_worker_globals()
    base_kw = dict(_BASE_KW)
    base_kw.update(_HOSTS[0])
    base_kw["evaluate_with_bh_mass"] = True
    base_kw["catalogue_mass_overlap"] = "neutralized"
    base_kw["catalogue_mass_error_scale"] = 1.0

    baseline = bs.single_host_likelihood(**base_kw)

    perturbed_kw = dict(base_kw)
    perturbed_kw["host_M"] = base_kw["host_M"] * 3.7
    perturbed_kw["host_M_error"] = base_kw["host_M_error"] * 0.2
    perturbed = bs.single_host_likelihood(**perturbed_kw)

    # combined_with_bh numerator (index 2) is the mode-sensitive quantity.
    assert baseline[2] == perturbed[2]
    # 1D numerator/denominator (indices 0, 1) are trivially untouched too.
    assert baseline[0] == perturbed[0]
    assert baseline[1] == perturbed[1]


def test_neutralized_independent_of_host_mass_batch() -> None:
    _install_worker_globals()
    hosts = [dict(_HOSTS[0]), dict(_HOSTS[0])]
    hosts[1]["host_M"] = hosts[0]["host_M"] * 3.7
    hosts[1]["host_M_error"] = hosts[0]["host_M_error"] * 0.2
    arrays = {k: np.array([h[k] for h in hosts], dtype=np.float64) for k in _HOST_KEYS}
    out = bs.single_host_likelihood_batch(
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
        catalogue_mass_overlap="neutralized",
        catalogue_mass_error_scale=1.0,
    )
    # Row 0 (unperturbed) vs row 1 (host_M/host_M_error perturbed): the
    # with-BH-mass numerator (col 2) must be identical under neutralization.
    assert out[0, 2] == out[1, 2]


def test_production_responds_to_host_mass_perturbation_sanity() -> None:
    """Sanity check: 'production' (unlike 'neutralized') DOES respond to a
    host mass/M_error perturbation -- confirms the perturbation is
    physically load-bearing and the neutralized-invariance test above is not
    vacuous."""
    _install_worker_globals()
    base_kw = dict(_BASE_KW)
    base_kw.update(_HOSTS[0])
    base_kw["evaluate_with_bh_mass"] = True
    base_kw["catalogue_mass_overlap"] = "production"
    base_kw["catalogue_mass_error_scale"] = 1.0

    baseline = bs.single_host_likelihood(**base_kw)

    perturbed_kw = dict(base_kw)
    perturbed_kw["host_M"] = base_kw["host_M"] * 3.7
    perturbed_kw["host_M_error"] = base_kw["host_M_error"] * 0.2
    perturbed = bs.single_host_likelihood(**perturbed_kw)

    assert baseline[2] != perturbed[2]


# ── (v) the 1D channel is unaffected by the mode in every case ──────────────


@pytest.mark.parametrize("mode,scale", _MODE_CASES)
def test_1d_channel_unaffected_by_mode_scalar(mode: str, scale: float) -> None:
    production = _scalar_rows("production", 1.0, evaluate_with_bh_mass=False)
    other = _scalar_rows(mode, scale, evaluate_with_bh_mass=False)
    assert np.array_equal(production, other)


@pytest.mark.parametrize("mode,scale", _MODE_CASES)
def test_1d_channel_unaffected_by_mode_batch(mode: str, scale: float) -> None:
    production = _batch_rows("production", 1.0, evaluate_with_bh_mass=False)
    other = _batch_rows(mode, scale, evaluate_with_bh_mass=False)
    assert np.array_equal(production, other)


# ── rejection of unknown modes (defence in depth; argparse already rejects
#    at the CLI layer, this pins the kernel-level guard too) ────────────────


def test_unknown_mode_raises_scalar() -> None:
    _install_worker_globals()
    kw = dict(_BASE_KW)
    kw.update(_HOSTS[0])
    kw["evaluate_with_bh_mass"] = True
    kw["catalogue_mass_overlap"] = "bogus"
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
            catalogue_mass_overlap="bogus",
            catalogue_mass_error_scale=1.0,
        )
