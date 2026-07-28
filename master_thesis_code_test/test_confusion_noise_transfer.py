"""Regression pins for the confusion-noise TDI transfer fix (#51 high-M audit).

Physics-change protocol two-step: THIS commit pins the OLD (buggy) behaviour —
the Cornish & Robson (2017) strain-referred confusion PSD S_c added to the
TDI-1 A-channel relative-frequency PSD WITHOUT the stochastic transfer factor
1.5*(2x sin x)^2, x = 2*pi*f*L/c (lisatools ``A1TDISens.stochastic_transform``
convention), overweighting confusion by ~1e6 at 0.2 mHz. The follow-up
``[PHYSICS]`` commit flips these pins to the corrected values.

Provenance: results/campaign51_20260728/highm_audit/HIGHM_AUDIT.md item 4;
bug introduced in commit 3bed9fc (Phase 9).
"""

import numpy as np
import pytest

from master_thesis_code.LISA_configuration import LisaTdiConfiguration


def _confusion_to_instrumental_ratio(f_hz: float) -> float:
    """PSD(with confusion)/PSD(without) at a single frequency."""
    with_c = LisaTdiConfiguration(include_confusion_noise=True)
    without_c = LisaTdiConfiguration(include_confusion_noise=False)
    fa = np.array([f_hz])
    return float(
        (
            with_c.power_spectral_density_a_channel(fa)
            / without_c.power_spectral_density_a_channel(fa)
        )[0]
    )


def test_confusion_ratio_at_0p2mhz_old() -> None:
    """OLD pin: raw strain-referred S_c dwarfs the TDI instrumental PSD ~1.4e6x."""
    assert _confusion_to_instrumental_ratio(2e-4) == pytest.approx(1.372656e6, rel=1e-3)


def test_confusion_ratio_at_1mhz_old() -> None:
    """OLD pin: ~7.5e4x at 1 mHz (physically absurd — LISA would be deaf)."""
    assert _confusion_to_instrumental_ratio(1e-3) == pytest.approx(7.523679e4, rel=1e-3)


def test_confusion_negligible_at_5mhz() -> None:
    """Above the confusion knee the ratio is ~1 (holds before AND after the fix)."""
    assert _confusion_to_instrumental_ratio(5e-3) == pytest.approx(1.0, abs=0.05)
