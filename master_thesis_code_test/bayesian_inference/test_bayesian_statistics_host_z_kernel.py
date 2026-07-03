"""Numeric regression pins for the host-z kernel in ``single_host_likelihood``.

These tests pin the exact numerical output of the production Pipeline-B
per-host likelihood at fixed synthetic inputs, so any change to the host
redshift kernel (window bounds, ``norm(loc=host_z, scale=...)`` width,
volume-deconvolution weight) shows up as a deliberate pin update in the same
diff that changes the physics — the /physics-change regression requirement.

The worker globals normally installed by ``init_worker`` are set directly on
the module with a stub detection-probability object whose grid is wide enough
that the outside-grid quadrature weights are exactly zero.
"""

import numpy as np
import pytest

import master_thesis_code.bayesian_inference.bayesian_statistics as bs

# Synthetic detection: z ~ 0.1 event at h = 0.73 with 5% distance error.
_DET_D_L = 0.47  # Gpc
_DET_D_L_UNC = 0.0235  # Gpc (5%)
_DET_M = 3.3e5  # M_sun, observer-frame (redshifted) mass
_HOST_PHI = 1.2
_HOST_THETA = 1.0


class _StubDetectionProbability:
    """Deterministic, smooth p_det stub with a grid spanning any test window."""

    def __init__(self) -> None:
        self._dl_centers = np.linspace(0.01, 60.0, 300)

    def detection_probability_without_bh_mass_interpolated_zero_fill(
        self,
        d_L: np.ndarray,
        phi: np.ndarray,
        theta: np.ndarray,
        h: float,
    ) -> np.ndarray:
        return np.exp(-np.asarray(d_L, dtype=np.float64) / 5.0)

    def detection_probability_with_bh_mass_interpolated(
        self,
        d_L: np.ndarray,
        M_z: np.ndarray,
        phi: np.ndarray,
        theta: np.ndarray,
        h: float,
    ) -> np.ndarray:
        d = np.asarray(d_L, dtype=np.float64)
        m = np.asarray(M_z, dtype=np.float64)
        return np.exp(-d / 5.0) * np.exp(-((np.log10(m) - 5.5) ** 2))

    def _get_or_build_grid(self, h: float) -> tuple:
        class _Interp:
            grid = (self._dl_centers,)

        return None, _Interp()


def _install_worker_globals() -> None:
    """Install the module-level worker state single_host_likelihood reads."""
    cov_3d = np.diag([0.02**2, 0.02**2, 0.05**2])
    cov_4d = np.diag([0.02**2, 0.02**2, 0.05**2, 0.1**2])

    bs.det_index_to_slot = {0: 0}
    bs.det_d_L_arr = np.array([_DET_D_L])
    bs.det_d_L_unc_arr = np.array([_DET_D_L_UNC])
    bs.det_M_arr = np.array([_DET_M])
    bs.det_phi_arr = np.array([_HOST_PHI])
    bs.det_theta_arr = np.array([_HOST_THETA])

    bs.means_3d = np.array([[_HOST_PHI, _HOST_THETA, 1.0]])
    bs.cov_inv_3d = np.array([np.linalg.inv(cov_3d)])
    bs.log_norm_3d = np.array([-0.5 * (3 * np.log(2 * np.pi) + np.linalg.slogdet(cov_3d)[1])])

    bs.means_4d = np.array([[_HOST_PHI, _HOST_THETA, 1.0, 1.0]])
    # Diagonal 4D covariance: conditional variance of the M_z fraction is its
    # marginal variance and the projection vector vanishes.
    bs.sigma2_cond_arr = np.array([0.1**2])
    bs.proj_arr = np.zeros((1, 3))
    bs.cov_inv_4d = np.array([np.linalg.inv(cov_4d)])
    bs.log_norm_4d = np.array([-0.5 * (4 * np.log(2 * np.pi) + np.linalg.slogdet(cov_4d)[1])])

    bs.detection_probability = _StubDetectionProbability()


def _run_case(
    host_z: float,
    host_z_error: float,
    normalization_mode: str,
    evaluate_with_bh_mass: bool,
) -> list[float]:
    _install_worker_globals()
    return bs.single_host_likelihood(
        host_phiS=_HOST_PHI,
        host_qS=_HOST_THETA,
        host_z=host_z,
        host_z_error=host_z_error,
        host_M=3.0e5,
        host_M_error=3.0e4,
        detection_index=0,
        h=0.73,
        evaluate_with_bh_mass=evaluate_with_bh_mass,
        normalization_mode=normalization_mode,
        base_seed=42,
    )


def test_kernel_pin_volume_deconv_without_bh_mass() -> None:
    """Spec-z-like host (sigma_z = 0.0015) — volume-deconvolved kernel."""
    num, den, w_num, w_den = _run_case(0.10, 0.0015, "volume_deconv", False)
    assert num == pytest.approx(PIN_VD_NUM, rel=1e-9)
    assert den == pytest.approx(PIN_VD_DEN, rel=1e-9)
    assert w_num == 0.0
    assert w_den == 0.0


def test_kernel_pin_local_ratio_without_bh_mass() -> None:
    """Same host — bare photo-z Gaussian kernel (local_ratio mode)."""
    num, den, w_num, w_den = _run_case(0.10, 0.0015, "local_ratio", False)
    assert num == pytest.approx(PIN_LR_NUM, rel=1e-9)
    assert den == pytest.approx(PIN_LR_DEN, rel=1e-9)
    assert w_num == 0.0
    assert w_den == 0.0


def test_kernel_pin_volume_deconv_with_bh_mass() -> None:
    """With-BH-mass path incl. the seeded MC denominator (base_seed = 42)."""
    vals = _run_case(0.10, 0.0015, "volume_deconv", True)
    assert vals[0] == pytest.approx(PIN_VD_NUM, rel=1e-9)
    assert vals[1] == pytest.approx(PIN_VD_DEN, rel=1e-9)
    assert vals[2] == pytest.approx(PIN_VD_BH_NUM, rel=1e-9)
    assert vals[3] == pytest.approx(PIN_VD_BH_DEN, rel=1e-9)
    assert vals[4] == 0.0
    assert vals[5] == 0.0


def test_kernel_pin_low_z_window_clamp() -> None:
    """Low-z host (z_g < 4 sigma_z): lower window bound clamps to z = 1e-6.

    The numerator is exactly zero (the event window near z ~ 0.1 has no
    overlap with the host kernel at z = 0.004); the denominator and its
    outside-grid quadrature weight both depend directly on the kernel width,
    making the weight a sensitive canary for any sigma_z change.
    """
    num, den, w_num, w_den = _run_case(0.004, 0.0015, "volume_deconv", False)
    assert num == 0.0
    assert den == pytest.approx(PIN_CLAMP_DEN, rel=1e-9)
    assert w_num == 0.0
    assert w_den == pytest.approx(PIN_CLAMP_W_DEN, rel=1e-9)


# ── Pinned values (captured from the code as of the commit adding this file) ──
PIN_VD_NUM = 1622.0066615957417
PIN_VD_DEN = 0.9153058114326034
PIN_LR_NUM = 1607.5871614112384
PIN_LR_DEN = 0.9152832361769563
PIN_VD_BH_NUM = 4574.227429970933
PIN_VD_BH_DEN = 0.913118914446828
PIN_CLAMP_DEN = 0.9958992939448512
PIN_CLAMP_W_DEN = 0.24151081256750923
