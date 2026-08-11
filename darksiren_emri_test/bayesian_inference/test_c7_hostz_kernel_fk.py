"""Acceptance tests for the C7-core host-z kernel factor ``f_{k(g)}(z)``.

The in-catalogue host-redshift kernel of the ``volume_deconv`` family now
carries the *catalogued-host* intensity

    rho_g(z) = N(z; z_g, sigma_eff) * w_pop(z; h) * f_{k(g)}(z) / Z_g ,
    Z_g      = INTEGRAL N(z; z_g, sigma_eff) * w_pop(z; h) * f_{k(g)}(z) dz ,

with ``f`` the SAME per-pixel completeness callable ``B_num`` and ``beta_Gbar``
use, evaluated at the HOST's HEALPix pixel (GATE_PACKAGE_FINAL.md §1.2,
2026-08-04).  This module implements the §1.5 acceptance tests:

* ``f == 1`` byte-identity with the pre-C7 kernel (strict generalisation);
* ``sigma_z -> 0`` exactness with a non-trivial ``f`` (``f(z_g)`` cancels
  between numerator and ``Z_g``);
* h-invariance of ``w_pop * f_k`` — including the exact ``m_star`` cancellation
  mechanism inside the real ``PixelCompleteness``;
* the renormalised shift law ``Delta / e^2 = gamma_f`` over three sigma rungs;
* the ZoA all-zero-window fallback branch (no elementwise clamp, warn once);
* batched/scalar parity with ``completeness`` threaded into the
  ``child_process_init`` worker globals.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import darksiren_emri.bayesian_inference.bayesian_statistics as bs
from darksiren_emri.galaxy_catalogue.pixel_completeness import PixelCompleteness
from darksiren_emri.physical_relations import comoving_volume_element
from darksiren_emri_test.bayesian_inference.test_bayesian_statistics_host_z_kernel import (
    _DET_D_L,
    _DET_D_L_UNC,
    _DET_M,
    _HOST_PHI,
    _HOST_THETA,
    _install_worker_globals,
    _run_case,
    _StubCompleteness,
    _StubDetectionProbability,
    gamma_f_reference,
)

_H = 0.73


@pytest.fixture(autouse=True)
def _reset_completeness_global() -> Any:
    yield
    bs.completeness_model = None


# ── (1) f == 1 byte-identity ────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["volume_deconv", "volume_trunc", "absolute_marginal"])
@pytest.mark.parametrize("with_bh", [False, True])
def test_f_identically_one_is_byte_identical_to_pre_c7(mode: str, with_bh: bool) -> None:
    """``f == 1`` reproduces the pre-C7 kernel bit-for-bit (gate §1.2 limit 2)."""
    head = _run_case(0.10, 0.0015, mode, with_bh, completeness=None)
    flat = _run_case(0.10, 0.0015, mode, with_bh, completeness=_StubCompleteness(flat=True))
    assert flat == head


# ── (2) sigma_z -> 0 exactness with a non-trivial f ─────────────────────────


def test_sigma_z_to_zero_is_f_independent() -> None:
    """As ``sigma_z -> 0`` the kernel returns the point kernel, f-independently.

    ``f(z_g)`` cancels between the numerator and ``Z_g``, so the relative gap
    between the C7 and the pre-C7 likelihood ratio shrinks monotonically toward
    zero.  Rungs stop at 1.5e-3 and the tolerance matches the existing
    ``volume_trunc`` spec-z gate (5e-3): below ~1e-3 the n=50 numerator
    quadrature over the *event* window aliases the narrowing host kernel and
    both legs hit a ~5e-4 resolution floor — a quadrature artifact, not a
    physics limit failure.  The clean quadratic statement is made on the kernel
    moment itself in ``test_shift_law_delta_over_e_squared_equals_gamma_f``.
    """
    gaps = []
    for sigma_z in (0.006, 0.003, 0.0015):
        num_f, den_f, _, _ = _run_case(0.10, sigma_z, "volume_deconv", False)
        num_0, den_0, _, _ = _run_case(0.10, sigma_z, "volume_deconv", False, completeness=None)
        gaps.append(abs((num_f / den_f) / (num_0 / den_0) - 1.0))
    assert all(gaps[i + 1] < gaps[i] for i in range(len(gaps) - 1)), gaps
    assert gaps[-1] < 5e-3, gaps


# ── (3) h-invariance of w_pop * f_k ─────────────────────────────────────────


def test_f_k_is_exactly_h_independent_via_m_star_cancellation() -> None:
    """``f_k(z, h)`` carries no h: the ``+5 log10 h`` in ``M_*`` cancels the
    ``-5 log10 h`` of the distance modulus (``pixel_completeness.py``; G2b gate
    6 extended to the C7 kernel factor, GATE_PACKAGE_FINAL.md §1.3 gate (v)).
    """
    m_th = np.full(48, 18.0)
    completeness = PixelCompleteness(m_th, nside=2)
    z = np.linspace(0.01, 0.6, 32)
    f_lo = np.asarray(completeness.f_k(z, 12, h=0.60), dtype=np.float64)
    f_hi = np.asarray(completeness.f_k(z, 12, h=0.86), dtype=np.float64)
    assert np.all(f_lo > 0.0)
    assert float(np.max(np.abs(f_lo / f_hi - 1.0))) < 1e-10


def test_w_pop_times_f_k_h_dependence_is_z_separable() -> None:
    """The full C7 kernel weight ``w_pop(z,h) * f_k(z,h)`` is h-separable.

    ``dV_c/dz`` factorises as ``h^-3 g(z)`` and ``f_k`` carries no h at all, so
    the product's h-dependence is a z-independent constant that cancels against
    ``Z_g(h)`` — the kernel shape ``rho_g(z)`` is exactly h-invariant.
    """
    m_th = np.full(48, 18.0)
    completeness = PixelCompleteness(m_th, nside=2)
    z = np.linspace(0.01, 0.6, 32)

    def weight(h: float) -> npt.NDArray[np.float64]:
        w_pop = np.asarray(comoving_volume_element(z, h=h), dtype=np.float64) / (1.0 + z)
        return w_pop * np.asarray(completeness.f_k(z, 12, h=h), dtype=np.float64)

    ratio = weight(0.60) / weight(0.85)
    spread = float((ratio.max() - ratio.min()) / ratio.mean())
    assert spread < 1e-10, spread


# ── (4) the renormalised shift law Delta / e^2 = gamma_f ────────────────────


class _ZReturningDetectionProbability(_StubDetectionProbability):
    """z-resolved p_det stub that returns the conditioning redshift itself.

    With this weight the per-host denominator ``D_g = INTEGRAL p_det rho_g dz``
    is exactly the first moment ``E_rho[z]`` of the host-z kernel (``rho_g`` is
    unit-normalised on the very same window and quadrature rule), which is the
    object the ``(3 + gamma_f)(sigma_z/z)^2`` law is a statement about.
    """

    z_resolved = True

    def detection_probability_without_bh_mass_interpolated_zero_fill(
        self,
        d_L: np.ndarray,
        phi: np.ndarray,
        theta: np.ndarray,
        h: float,
        z: np.ndarray | float | None = None,
    ) -> np.ndarray:
        assert z is not None, "the z-resolved pass-through must supply z"
        return np.asarray(z, dtype=np.float64)


def _kernel_mean_z(host_z: float, sigma_z: float, completeness: Any) -> float:
    """First moment of the host-z kernel, read off the production kernel."""
    _install_worker_globals(completeness)
    bs.detection_probability = _ZReturningDetectionProbability()
    _, den, _, _ = bs.single_host_likelihood(
        host_phiS=_HOST_PHI,
        host_qS=_HOST_THETA,
        host_z=host_z,
        host_z_error=sigma_z,
        host_M=3.0e5,
        host_M_error=3.0e4,
        detection_index=0,
        h=_H,
        evaluate_with_bh_mass=False,
        normalization_mode="volume_deconv",
        base_seed=42,
    )
    return float(den)


def test_shift_law_delta_over_e_squared_equals_gamma_f() -> None:
    """``Delta / e^2 -> gamma_f`` over three sigma rungs (gate §1.3 (ii)/(iii)).

    ``Delta`` is the fractional change of the kernel's mean redshift caused by
    the C7 factor and ``e = sigma_z / z_g``.  The adjudicated pre-C7 law
    ``E[z]/z_g = 1 + 3 e^2`` is *renormalised* to ``1 + (3 + gamma_f) e^2`` with
    ``gamma_f = dln f / dln z`` at ``z_g`` — it does not collapse.  Sign: the
    stub's ``gamma_f = -0.4 < 0``, so the kernel's mean redshift can only fall.
    """
    host_z = 0.10
    gamma_f = gamma_f_reference(host_z)
    assert gamma_f == pytest.approx(-0.4)

    ratios = []
    for e in (0.04, 0.02, 0.01):
        sigma_z = e * host_z
        mean_f = _kernel_mean_z(host_z, sigma_z, _StubCompleteness())
        mean_0 = _kernel_mean_z(host_z, sigma_z, None)
        # Delta = (E_f[z] - E_head[z]) / z_g; the shift is one-signed (down).
        assert mean_f < mean_0
        ratios.append(((mean_f - mean_0) / host_z) / e**2)
    for ratio in ratios:
        assert ratio == pytest.approx(gamma_f, rel=0.05), ratios
    # Converging on the analytic coefficient as e -> 0.
    assert abs(ratios[-1] - gamma_f) < abs(ratios[0] - gamma_f)


# ── (5) the ZoA all-zero-window branch ─────────────────────────────────────


def test_zoa_all_zero_window_falls_back_to_pre_c7_kernel(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """f_k == 0 on the whole window -> pre-C7 kernel, one warning, never NaN.

    Elementwise clamping is forbidden (GATE_PACKAGE_FINAL.md §1.1 B5): it would
    install a kink where f_k crosses a floor partway across the window.
    """
    bs._zoa_hostz_kernel_fallback_warned = False
    bs._zoa_hostz_kernel_fallback_hosts = 0

    head = _run_case(0.10, 0.0015, "volume_deconv", True, completeness=None)
    with caplog.at_level("WARNING"):
        zoa = _run_case(
            0.10, 0.0015, "volume_deconv", True, completeness=_StubCompleteness(zoa=True)
        )

    assert zoa == head
    assert all(np.isfinite(zoa))
    messages = [r.message for r in caplog.records if "ZoA pixel" in str(r.getMessage())]
    assert len(messages) == 1, [r.getMessage() for r in caplog.records]

    # Warn-once, count-always.
    _run_case(0.10, 0.0015, "volume_deconv", False, completeness=_StubCompleteness(zoa=True))
    assert bs._zoa_hostz_kernel_fallback_hosts == 2
    bs._zoa_hostz_kernel_fallback_warned = False
    bs._zoa_hostz_kernel_fallback_hosts = 0


# ── (6) batched / scalar parity with the threaded worker global ─────────────


def test_child_process_init_threads_completeness_into_worker_globals() -> None:
    """``child_process_init`` installs the completeness the kernel reads."""
    bs.completeness_model = None
    stub = _StubCompleteness()
    zeros = np.zeros((1, 3))
    bs.child_process_init(
        0.0,
        1.0,
        1e4,
        1e7,
        None,  # type: ignore[arg-type]
        np.zeros((1, 3)),
        np.zeros((1, 3, 3)),
        np.zeros(1),
        np.zeros((1, 4)),
        np.zeros((1, 4, 4)),
        np.zeros(1),
        {0: 0},
        np.zeros(1),
        zeros,
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        None,
        stub,
    )
    assert bs.completeness_model is stub


@pytest.mark.parametrize("with_bh", [False, True])
def test_batched_scalar_parity_with_completeness(with_bh: bool) -> None:
    """The batched kernel reproduces the scalar twin host-by-host under C7.

    Two hosts in the same (ordinary) pixel plus one in the ZoA pixel exercise
    both the f-carrying and the fallback row of the batched path.
    """
    hosts = [
        (0.10, 0.0015),
        (0.11, 0.0060),
        (0.09, 0.0030),
    ]
    completeness = _StubCompleteness()

    scalar_rows = []
    for host_z, host_z_error in hosts:
        scalar_rows.append(_run_case(host_z, host_z_error, "volume_deconv", with_bh, completeness))

    _install_worker_globals(completeness)
    batch = bs.single_host_likelihood_batch(
        np.full(len(hosts), _HOST_PHI),
        np.full(len(hosts), _HOST_THETA),
        np.array([h for h, _ in hosts]),
        np.array([e for _, e in hosts]),
        np.full(len(hosts), 3.0e5),
        np.full(len(hosts), 3.0e4),
        0,
        _H,
        with_bh,
        "volume_deconv",
    )
    assert batch.shape == (len(hosts), 6 if with_bh else 4)
    np.testing.assert_allclose(batch, np.array(scalar_rows), rtol=1e-9, atol=0.0)


class _MixedPixelCompleteness(_StubCompleteness):
    """Two-pixel sky: hosts at ``phi > 1.205`` sit in the empty (ZoA) pixel."""

    def ang2pix(self, phi: float, theta: float) -> int:
        return self.ZOA_PIXEL if phi > 1.205 else self.ORDINARY_PIXEL


def _batch_two_hosts(completeness: Any, phis: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    _install_worker_globals(completeness)
    return bs.single_host_likelihood_batch(
        phis,
        np.full(2, _HOST_THETA),
        np.array([0.10, 0.10]),
        np.array([0.0060, 0.0060]),
        np.full(2, 3.0e5),
        np.full(2, 3.0e4),
        0,
        _H,
        False,
        "volume_deconv",
    )


def test_batched_zoa_row_falls_back_per_row() -> None:
    """The ZoA fallback is per-row: one host reverts, its neighbour keeps f.

    Both hosts are identical except for their sky position, which places host 1
    in the empty pixel.  Row 0 must move relative to the pre-C7 kernel, row 1
    must be bit-identical to it.
    """
    phis = np.array([1.20, 1.21])
    head = _batch_two_hosts(None, phis)
    mixed = _batch_two_hosts(_MixedPixelCompleteness(), phis)
    assert np.all(np.isfinite(mixed))
    # Row 1 (ZoA pixel): pre-C7 kernel, bit-identical.
    np.testing.assert_array_equal(mixed[1], head[1])
    # Row 0 (ordinary pixel): carries f.
    assert mixed[0, 0] != head[0, 0]
    assert mixed[0, 1] != head[0, 1]


def test_batched_zoa_batch_falls_back_to_pre_c7() -> None:
    """An all-ZoA batch is bit-identical to the pre-C7 kernel; f-carrying isn't."""
    completeness_zoa = _StubCompleteness(zoa=True)
    _install_worker_globals(completeness_zoa)
    batch_zoa = bs.single_host_likelihood_batch(
        np.full(2, _HOST_PHI),
        np.full(2, _HOST_THETA),
        np.array([0.10, 0.11]),
        np.array([0.0015, 0.0060]),
        np.full(2, 3.0e5),
        np.full(2, 3.0e4),
        0,
        _H,
        False,
        "volume_deconv",
    )
    _install_worker_globals(None)
    batch_head = bs.single_host_likelihood_batch(
        np.full(2, _HOST_PHI),
        np.full(2, _HOST_THETA),
        np.array([0.10, 0.11]),
        np.array([0.0015, 0.0060]),
        np.full(2, 3.0e5),
        np.full(2, 3.0e4),
        0,
        _H,
        False,
        "volume_deconv",
    )
    np.testing.assert_array_equal(batch_zoa, batch_head)

    _install_worker_globals(_StubCompleteness())
    batch_f = bs.single_host_likelihood_batch(
        np.full(2, _HOST_PHI),
        np.full(2, _HOST_THETA),
        np.array([0.10, 0.11]),
        np.array([0.0015, 0.0060]),
        np.full(2, 3.0e5),
        np.full(2, 3.0e4),
        0,
        _H,
        False,
        "volume_deconv",
    )
    assert not np.allclose(batch_f[:, 0], batch_head[:, 0], rtol=1e-12)


def test_event_stubs_are_self_consistent() -> None:
    """Guard: the shared fixture still describes a z ~ 0.1 matched event."""
    assert _DET_D_L == pytest.approx(0.47)
    assert _DET_D_L_UNC == pytest.approx(0.0235)
    assert _DET_M == pytest.approx(3.3e5)
