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

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import darksiren_emri.bayesian_inference.bayesian_statistics as bs
from darksiren_emri.constants import H
from darksiren_emri_test.bayesian_inference.test_bh_denominator_semianalytic import (
    make_grid2d_pdet,
)

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
        # Real 2-D interp for the with-BH-mass path (the semi-analytic denominator
        # reads its M_z knots and needs a genuinely piecewise-linear p_det).
        self._grid2d = make_grid2d_pdet("peaked")

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
        return self._grid2d.detection_probability_with_bh_mass_interpolated(d_L, M_z, phi, theta, h)

    def _get_or_build_grid(self, h: float) -> tuple:
        class _Interp:
            grid = (self._dl_centers,)

        return self._grid2d._interp, _Interp()


_SENTINEL_DEFAULT_COMPLETENESS = object()


@pytest.fixture(autouse=True)
def _reset_completeness_global() -> Any:
    """Never leak this module's completeness stub into another test module."""
    yield
    bs.completeness_model = None


class _StubCompleteness:
    """Analytic, non-trivial per-pixel completeness with a switchable ZoA pixel.

    ``f_k(z) = 1 / (1 + (z / Z0)^2)`` on every ordinary pixel; on the ZoA pixel
    (an empty HEALPix cell, ``m_th = -inf`` in the real
    :class:`~darksiren_emri.galaxy_catalogue.pixel_completeness.PixelCompleteness`)
    it is exactly ``0`` at every redshift.

    The log-slope is analytic,
    ``gamma_f(z) = dln f / dln z = -2 (z/Z0)^2 / (1 + (z/Z0)^2)``, so
    ``gamma_f(0.1) = -0.4`` at ``Z0 = 0.2``: the reference value the C7 shift
    law ``Delta / e^2 = gamma_f`` is checked against.
    """

    Z0 = 0.2
    ZOA_PIXEL = 0
    ORDINARY_PIXEL = 7

    def __init__(self, zoa: bool = False, flat: bool = False) -> None:
        self._zoa = zoa
        self._flat = flat

    def ang2pix(self, phi: float, theta: float) -> int:
        return self.ZOA_PIXEL if self._zoa else self.ORDINARY_PIXEL

    def f_k(
        self,
        z: float | npt.NDArray[np.floating[Any]],
        k: int,
        h: float = H,
    ) -> float | npt.NDArray[np.float64]:
        z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
        if k == self.ZOA_PIXEL:
            out = np.zeros_like(z_arr)
        elif self._flat:
            out = np.ones_like(z_arr)
        else:
            out = 1.0 / (1.0 + (z_arr / self.Z0) ** 2)
        if np.ndim(z) == 0:
            return float(out[0])
        return out

    def f_bar(
        self,
        z: float | npt.NDArray[np.floating[Any]],
        h: float = H,
    ) -> float | npt.NDArray[np.float64]:
        return self.f_k(z, self.ORDINARY_PIXEL, h)

    def get_completeness_at_redshift(
        self,
        z: float | npt.NDArray[np.floating[Any]],
        h: float = H,
    ) -> float | npt.NDArray[np.float64]:
        return self.f_bar(z, h)


def gamma_f_reference(z: float, z0: float = _StubCompleteness.Z0) -> float:
    """Analytic ``dln f / dln z`` of :class:`_StubCompleteness` at ``z``."""
    u = (z / z0) ** 2
    return -2.0 * u / (1.0 + u)


def _install_worker_globals(completeness: Any = None) -> None:
    """Install the module-level worker state single_host_likelihood reads.

    ``completeness`` is the C7 host-z-kernel completeness object threaded into
    the workers by ``child_process_init``; ``None`` is the pre-C7 kernel.
    """
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
    bs.completeness_model = completeness


def _run_case(
    host_z: float,
    host_z_error: float,
    normalization_mode: str,
    evaluate_with_bh_mass: bool,
    completeness: Any = _SENTINEL_DEFAULT_COMPLETENESS,
) -> list[float]:
    """Run the scalar kernel; by default with the non-trivial C7 completeness."""
    if completeness is _SENTINEL_DEFAULT_COMPLETENESS:
        completeness = _StubCompleteness()
    _install_worker_globals(completeness)
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


def test_kernel_pin_volume_trunc_without_bh_mass() -> None:
    """volume_trunc (Part 1): numerator over the per-host galaxy window, z-floor 0.

    Spec-z-like host: the host window is narrow and contains the matched event,
    so the shared-support numerator lands within rel~1e-4 of volume_deconv here
    (the shallow-venue divergence is exercised on the seed600 A/B, not this pin).
    """
    num, den, w_num, w_den = _run_case(0.10, 0.0015, "volume_trunc", False)
    assert num == pytest.approx(PIN_VT_NUM, rel=1e-9)
    assert den == pytest.approx(PIN_VT_DEN, rel=1e-9)
    assert w_num == 0.0
    assert w_den == 0.0


def test_kernel_pin_volume_trunc_with_bh_mass() -> None:
    """volume_trunc with-BH-mass path (deterministic semi-analytic denominator)."""
    vals = _run_case(0.10, 0.0015, "volume_trunc", True)
    assert vals[0] == pytest.approx(PIN_VT_NUM, rel=1e-9)
    assert vals[1] == pytest.approx(PIN_VT_DEN, rel=1e-9)
    assert vals[2] == pytest.approx(PIN_VT_BH_NUM, rel=1e-9)
    assert vals[3] == pytest.approx(PIN_VT_BH_DEN, rel=1e-9)
    assert vals[4] == 0.0
    assert vals[5] == 0.0


def test_volume_trunc_sigma_z_to_zero_spec_limit() -> None:
    """Limiting case (scoping §6 gate 1): as sigma_z -> 0, p_g -> delta(z - z_g),
    so the volume_trunc likelihood ratio N_g/D_g converges to the bare
    spectroscopic (local_ratio) ratio for a matched host. Assert the relative gap
    shrinks monotonically over a decreasing sigma_z sequence and is < 5e-3 at the
    tightest rung.
    """
    host_z = 0.10  # matched to the stub event (d_L = 0.47 Gpc, z ~ 0.1 at h = 0.73)
    # Rungs stop at 1e-3: since the counted-once PV change (issue #40b,
    # SIGMA_V_PEC_KM_S default 0.0) sigma_eff equals the bare sigma_z, and below
    # ~1e-3 the n=50 host-window quadrature aliases the narrowing kernel (the
    # known volume_trunc aliasing mode) — a resolution artifact, not a physics
    # limit failure. Previously the runtime PV term floored sigma_eff at ~7.3e-4.
    sigmas = [0.005, 0.002, 0.001]
    gaps = []
    for sz in sigmas:
        num_vt, den_vt, _, _ = _run_case(host_z, sz, "volume_trunc", False)
        num_lr, den_lr, _, _ = _run_case(host_z, sz, "local_ratio", False)
        l_vt = num_vt / den_vt
        l_lr = num_lr / den_lr
        gaps.append(abs(l_vt - l_lr) / l_lr)
    # Strictly convergent toward the spec-z limit.
    assert all(gaps[i + 1] < gaps[i] for i in range(len(gaps) - 1)), gaps
    assert gaps[-1] < 5e-3, gaps


def test_volume_trunc_prior_shape_h_independent() -> None:
    """h-independence of the volume prior shape (scoping §6 gate 6, G2b §1.5).

    volume_trunc reuses volume_deconv's weight w_pop(z, h) = dV_c/dz / (1 + z).
    Because dV_c/dz factorizes as h^-3 * g(z), the h-dependence is z-separable and
    cancels against the per-galaxy normalization Z_g(h), leaving p_g(z) identical
    across trial h. Verify the separability directly on comoving_volume_element:
    its ratio at two h values is constant in z (to machine precision).
    """
    from darksiren_emri.physical_relations import comoving_volume_element

    z = np.linspace(0.01, 0.6, 32)
    ratio = np.asarray(comoving_volume_element(z, h=0.60), dtype=np.float64) / np.asarray(
        comoving_volume_element(z, h=0.85), dtype=np.float64
    )
    spread = float((ratio.max() - ratio.min()) / ratio.mean())
    assert spread < 1e-12, spread


# ── Pinned values ─────────────────────────────────────────────────────────────
# Re-pinned in the [PHYSICS] C7-core commit (GATE_PACKAGE_FINAL.md §1.2/§1.5,
# 2026-08-04): the host-z volume_deconv kernel now carries the catalogued-host
# intensity f_{k(g)}(z) * w_pop(z) in both the numerator and Z_g / D_g. The
# fixture installs a non-trivial per-pixel completeness (_StubCompleteness,
# f = 1/(1 + (z/0.2)^2), gamma_f(0.1) = -0.4), so PIN_VD_*, PIN_VT_* and
# PIN_CLAMP_* move; the pre-C7 values (asserted in the parent commit
# "test: pin pre-C7-fix volume_deconv kernel values") were
#   PIN_VD_NUM     1622.0066615458952  -> 1618.855141034943
#   PIN_VD_DEN     0.9153058114327306  -> 0.9153135911632229
#   PIN_VD_BH_NUM  4574.22742983034    -> 4565.355545079394
#   PIN_VD_BH_DEN  0.9427033282673192  -> 0.9427085122404929
#   PIN_CLAMP_DEN  0.9958992939449592  -> 0.9958996737905789
#   PIN_VT_NUM     1621.8905890993271  -> 1618.7417804538027
#   PIN_VT_DEN     0.9153058114327306  -> 0.9153135911632229
#   PIN_VT_BH_NUM  4573.901034625918   -> 4565.036772477752
#   PIN_VT_BH_DEN  0.9427033282673192  -> 0.9427085122404929
# The numerators fall (gamma_f < 0 => the kernel's z-weight tilts DOWN) and the
# denominators rise by ~1e-5 relative; PIN_CLAMP_W_DEN (a window-geometry
# diagnostic, not a kernel value) is unchanged. PIN_LR_* must NOT move and does
# not: verified bit-identical with and without bs.completeness_model (the
# historical PIN_LR_NUM literal differs from the delivered value by 1 ulp, which
# predates this change and is inside the rel=1e-9 pin tolerance).
#
# Re-pinned in the [PHYSICS] counted-once PV commit (issue #40b, RATIFIED
# 2026-07-26; docs/derivations/hostz_pv_photoz_kernel.md): SIGMA_V_PEC_KM_S
# defaults to 0.0 — the PV dispersion now lives in the parse-time per-class
# catalogue z_error, so sigma_z_eff here equals the bare test sigma_z_cat.
# The kernel narrows by the former ~11% issue-#16 broadening (sigma_z_pv was
# 7.34e-4 at z_g = 0.1) and the integrals move 0.2-0.5%; the clamp canary
# w_den moves 0.228 -> 0.242. Pre-change values are in the parent commit of
# this diff (physics-change protocol). [History: issue-#16 commit added the
# runtime broadening; [PHYSICS] 2026-07-08 replaced the MC BH denominator by
# the exact erf-sum — see test_bh_denominator_semianalytic.]
PIN_VD_NUM = 1618.855141034943
PIN_VD_DEN = 0.9153135911632229
PIN_LR_NUM = 1607.5871613613651
PIN_LR_DEN = 0.9152832361770838
PIN_VD_BH_NUM = 4565.355545079394
PIN_VD_BH_DEN = 0.9427085122404929
PIN_CLAMP_DEN = 0.9958996737905789
PIN_CLAMP_W_DEN = 0.24151081257029428

# volume_trunc (Part 1, 2026-07-12): the in-catalogue numerator is integrated over
# the per-host galaxy window [z_g - 4σ, z_g + 4σ] (shared with Z_g / D_g) and the
# lower z-limit floors at 0. For this spec-z-like host the denominator/normalization
# are unchanged (z_g - 4σ > 0, so no z-floor difference; D_g/Z_g byte-identical to
# volume_deconv → PIN_VT_DEN == PIN_VD_DEN), and the numerator lands within ~7e-6 of
# PIN_VD_NUM because the narrow host kernel is fully inside the GW window.
PIN_VT_NUM = 1618.7417804538027
PIN_VT_DEN = 0.9153135911632229
PIN_VT_BH_NUM = 4565.036772477752
PIN_VT_BH_DEN = 0.9427085122404929
