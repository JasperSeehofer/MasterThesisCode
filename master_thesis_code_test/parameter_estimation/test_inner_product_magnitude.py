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

FINDING (2026-07-02, G8 soundness gate): ``scalar_product_of_functions`` is
missing that ``dt**2`` — it returns exactly ``<h|h>_physical / dt**2``
(ratio 0.01 at dt=10 s, machine-precision at multiple test frequencies).
Physically: code-SNR = physical-SNR / dt, so the "SNR >= 20" catalogue is a
physical-SNR >= 200 population, and Cramer-Rao uncertainties are dt x too
pessimistic for those events.

``test_inner_product_current_convention_pins_missing_dt2`` is the REGRESSION
ANCHOR required by the physics-change protocol: it pins the pre-fix behavior.
When the dt**2 fix lands (its own [PHYSICS] commit), flip the expected value
in that test to ``physical`` and delete this paragraph.
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
def test_inner_product_current_convention_pins_missing_dt2(k: int) -> None:
    """REGRESSION ANCHOR (pre-fix): code returns <h|h>_physical / dt**2 exactly."""
    pe = _bare_pe()
    code, physical = _monochromatic_case(pe, k=k)
    assert code == pytest.approx(physical / pe.dt**2, rel=1e-6)


def test_inner_product_runs_on_cpu_numpy2() -> None:
    """The CPU path must not crash on numpy>=2 (trapz removed -> trapezoid)."""
    pe = _bare_pe()
    code, _ = _monochromatic_case(pe, k=400, n=2**14)
    assert code > 0.0
