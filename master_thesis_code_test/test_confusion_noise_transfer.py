"""Regression tests for the confusion-noise TDI transfer fix (#51 high-M audit).

Physics-change protocol two-step: the previous commit pinned the OLD (buggy)
behaviour — the Cornish & Robson (2017) strain-referred confusion PSD S_c
added to the TDI-1 A-channel relative-frequency PSD WITHOUT the stochastic
transfer factor 1.5*(2x sin x)^2, x = 2*pi*f*L/c — giving absurd
confusion/instrumental ratios of 1.37e6 at 0.2 mHz and 7.5e4 at 1 mHz.

THIS commit applies the transfer (lisatools ``A1TDISens.stochastic_transform``
/ LDC convention) and flips the pins to the corrected values: confusion peaks
at ~4x instrumental in the 0.8-2.5 mHz band and is negligible elsewhere — the
expected physical LISA behaviour.

Provenance: results/campaign51_20260728/highm_audit/HIGHM_AUDIT.md item 4;
bug introduced in commit 3bed9fc (Phase 9).
"""

import numpy as np
import pytest

from master_thesis_code.LISA_configuration import C, L, LisaTdiConfiguration


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


def test_confusion_ratio_at_0p2mhz_corrected() -> None:
    """NEW pin: ~1.10 at 0.2 mHz (OLD buggy value: 1.372656e6)."""
    assert _confusion_to_instrumental_ratio(2e-4) == pytest.approx(1.099315, rel=1e-3)


def test_confusion_ratio_at_1mhz_corrected() -> None:
    """NEW pin: ~4.30 at 1 mHz — confusion peaks at a few x instrumental in the
    0.8-2.5 mHz band (OLD buggy value: 7.523679e4; 4.399186 at the retired
    t_obs_years = 4.0 — the [PHYSICS] mission-duration alignment to 4.5 yr,
    Colpi et al. 2024 arXiv:2402.07571, moves the subtraction knees ~3%;
    docs/derivations/plunge_window_initial_conditions.md SS7)."""
    assert _confusion_to_instrumental_ratio(1e-3) == pytest.approx(4.300683, rel=1e-3)


def test_confusion_negligible_at_5mhz() -> None:
    """Above the confusion knee the ratio is ~1 (holds before AND after the fix)."""
    assert _confusion_to_instrumental_ratio(5e-3) == pytest.approx(1.0, abs=0.05)


def test_transfer_matches_lisatools_convention() -> None:
    """Cross-check: our added confusion term equals 1.5*(2x sin x)^2 * S_c —
    the lisatools A1TDISens.stochastic_transform factor — across the band."""
    with_c = LisaTdiConfiguration(include_confusion_noise=True)
    without_c = LisaTdiConfiguration(include_confusion_noise=False)
    # Band restricted to where the confusion term is non-negligible relative to
    # the instrumental PSD: outside it the difference-of-PSDs suffers floating
    # cancellation far above any physical tolerance. (NB: importing lisatools
    # itself segfaults in the CPU-only env, so the package cross-check lives in
    # the audit record — HIGHM_AUDIT.md item 4 verified the factor against the
    # in-venv lisatools A1TDISens.stochastic_transform source.)
    f = np.logspace(np.log10(2e-4), np.log10(3e-3), 100)
    added = with_c.power_spectral_density_a_channel(
        f
    ) - without_c.power_spectral_density_a_channel(f)
    x = 2 * np.pi * f * L / C
    expected = 1.5 * (2 * x * np.sin(x)) ** 2 * with_c._confusion_noise(f)
    np.testing.assert_allclose(added, expected, rtol=1e-9)


def test_low_frequency_limit_transfer_vanishes() -> None:
    """Limiting case f -> 0: the transfer ~ 6x^4... -> 0, so confusion cannot
    dominate the DC limit (the buggy form diverged relative to instrumental)."""
    assert _confusion_to_instrumental_ratio(1e-5) == pytest.approx(1.0, abs=1e-2)
