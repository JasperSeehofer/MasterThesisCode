"""G8 gate: magnitude validation of the LISA noise-weighted inner product.

The standard GW inner product (e.g. Babak et al. 2021, arXiv:2108.01167, Eq. 20;
Finn 1992) for a monochromatic signal ``h(t) = A sin(2 pi f0 t)`` of duration
``T`` observed against a one-sided PSD ``S_n(f)`` evaluates in closed form to

    <h|h> = A^2 T / S_n(f0)

(from Parseval: the one-sided integral of ``|h~(f)|^2`` is ``A^2 T / 4``, and the
inner product carries the prefactor 4; ``S_n`` varies slowly across the bin).

The continuous Fourier transform relates to the raw DFT output as
``h~(f_k) = dt * X_k``; an implementation using un-normalized ``rfft`` output
against a physical PSD in 1/Hz therefore needs an explicit ``dt**2`` factor.

HISTORY (G8 soundness gate, 2026-07-02): the pre-fix implementation was missing
that ``dt**2`` and returned exactly ``<h|h>_physical / dt**2`` (code-SNR =
physical-SNR/10 at dt=10 s; the "SNR >= 20" catalogue was a physical-SNR >= 200
population). Fixed under the physics-change protocol; the magnitude test below
is the post-fix regression anchor. Derivation + five evidence lines:
docs/derivations/G8_dt2_inner_product_derivation.md
"""

import numpy as np
import numpy.fft as nfft
import pytest

from master_thesis_code.LISA_configuration import LisaTdiConfiguration
from master_thesis_code.parameter_estimation.parameter_estimation import ParameterEstimation


def _bare_pe() -> ParameterEstimation:
    """ParameterEstimation with only the attrs scalar_product_of_functions needs.

    Avoids constructing the FEW/fastlisaresponse generators (GPU/slow); same
    pattern as the gpu-marked scalar_product tests.
    """
    pe = object.__new__(ParameterEstimation)
    pe._xp = np
    pe._fft = nfft
    pe._psd_cache = {}
    pe.lisa_configuration = LisaTdiConfiguration()
    return pe


def _monochromatic_case(
    pe: ParameterEstimation, k: int, n: int = 2**17, amp: float = 1e-21
) -> tuple[float, float]:
    """Return (code_value, physical_value) for an exact-bin sinusoid at f0 = k/(n dt)."""
    dt = pe.dt
    T = n * dt
    f0 = k / T
    t = np.arange(n) * dt
    h = amp * np.sin(2.0 * np.pi * f0 * t)
    tdi = np.stack([h, np.zeros_like(h)])
    code = pe.scalar_product_of_functions(tdi, tdi)
    S_f0 = float(pe.lisa_configuration.power_spectral_density(np.array([f0]), channel="A")[0])
    physical = amp**2 * T / S_f0
    return code, physical


@pytest.mark.parametrize("k", [400, 2000])
def test_inner_product_matches_analytic_monochromatic(k: int) -> None:
    """REGRESSION ANCHOR (post-dt^2-fix): code == <h|h>_physical = A^2 T / S_n(f0)."""
    pe = _bare_pe()
    code, physical = _monochromatic_case(pe, k=k)
    assert code == pytest.approx(physical, rel=1e-6)


def test_inner_product_parseval_white_psd() -> None:
    """FFT-free cross-check: for constant S_n = S0, <h|h> = (2/S0) * sum h(t)^2 dt.

    Parseval collapses the inner product to pure time domain — no Fourier
    convention exists on the reference side, so this test discriminates the
    DFT normalization independently of L1's analytic benchmark.
    """
    from unittest.mock import patch

    pe = _bare_pe()
    rng = np.random.default_rng(42)
    n = 2**14
    t = np.arange(n) * pe.dt
    h = np.zeros(n)
    for f in np.linspace(5e-4, 2e-2, 30):
        h += rng.normal(0, 1e-21) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    tdi = np.stack([h, np.zeros_like(h)])
    S0 = 1e-36
    with patch.object(
        pe.lisa_configuration,
        "power_spectral_density",
        side_effect=lambda fs, channel="A": np.full_like(np.asarray(fs, dtype=float), S0),
    ):
        code = pe.scalar_product_of_functions(tdi, tdi)
    parseval = 2.0 / S0 * float(np.sum(h**2) * pe.dt)
    assert code == pytest.approx(parseval, rel=1e-4)


def test_inner_product_runs_on_cpu_numpy2() -> None:
    """The CPU path must not crash on numpy>=2 (trapz removed -> trapezoid)."""
    pe = _bare_pe()
    code, _ = _monochromatic_case(pe, k=400, n=2**14)
    assert code > 0.0
