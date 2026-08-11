"""Regression pins for the plunge-window initial-condition convention.

[PHYSICS] change 2026-07-28 (author-ratified): the snapshot draw p0 ~ U[10, 16]
(few's Pn5AAK input domain adopted as a prior in 2023) is replaced by the
plunge-window convention of the Babak et al. (2017) arXiv:1703.09722 rate
model: t_plunge ~ U[0, T_mission], p0 = root of t_insp(p0) = t_plunge.
Simultaneously the observation span moves from the unofficial T = 5 yr to the
official 4.5 yr (Colpi et al. 2024, arXiv:2402.07571).

The BEFORE-change pins (old uniform law, T = 5, t_obs_years = 4.0) live in
results/campaign51_20260728/plunge_window/old_pins_test_version.py and are
meant to be committed FIRST, then flipped to this file.

Derivation: docs/derivations/plunge_window_initial_conditions.md.
"""

import math

import numpy as np
import pytest

from darksiren_emri.constants import LISA_MISSION_DURATION_YEARS
from darksiren_emri.datamodels.parameter_space import ParameterSpace


def test_mission_duration_constant_is_official_4p5_yr() -> None:
    """Colpi et al. (2024) arXiv:2402.07571: 4.5 yr nominal science operations."""
    assert LISA_MISSION_DURATION_YEARS == 4.5


def test_parameter_estimation_T_is_mission_duration() -> None:
    """ParameterEstimation.T (class attribute; no init needed) tracks the constant.

    Old pin (pre-change): T = 5 (unofficial, hardcoded 2023). SNR/PSD
    consequence of 5 -> 4.5 is documented in the derivation doc SS7.
    """
    from darksiren_emri.parameter_estimation.parameter_estimation import (
        ParameterEstimation,
    )

    assert ParameterEstimation.T == LISA_MISSION_DURATION_YEARS == 4.5


def test_lisa_tdi_t_obs_years_is_mission_duration() -> None:
    """Confusion-noise foreground-subtraction span tracks the constant.

    Old pin (pre-change): t_obs_years = 4.0 (inconsistent with T = 5).
    """
    from darksiren_emri.LISA_configuration import LisaTdiConfiguration

    assert LisaTdiConfiguration().t_obs_years == LISA_MISSION_DURATION_YEARS == 4.5


def test_snapshot_draw_law_unchanged_for_archaeology() -> None:
    """--snapshot_ics path = bare randomize_parameters: the OLD p0 law survives.

    Byte-identical pin to the pre-change regression value (same seed, same
    stream): p0 ~ U[10, 16] via randomize_parameters, no extra rng draws.
    """
    ps = ParameterSpace()
    ps.randomize_parameters(np.random.default_rng(42))
    assert ps.p0.value == pytest.approx(14.184208174356183, abs=0.0)
    assert 10.0 <= ps.p0.value <= 16.0
    # Snapshot mode carries no plunge-time provenance.
    assert math.isnan(ps.t_plunge_yr)


def test_snapshot_draw_is_uniform_in_bounds() -> None:
    """Old-law shape check: p0 uniform on [10, 16] (KS against the uniform CDF)."""
    ps = ParameterSpace()
    rng = np.random.default_rng(123)
    values = []
    for _ in range(400):
        ps.randomize_parameter(ps.p0, rng)
        values.append(ps.p0.value)
    arr = np.sort(np.asarray(values))
    assert arr[0] >= 10.0 and arr[-1] <= 16.0
    u = (arr - 10.0) / 6.0
    ks = float(np.max(np.abs(u - (np.arange(1, len(u) + 1) - 0.5) / len(u))))
    assert ks < 0.07  # 400 samples: KS_0.99 ~ 0.0815


def test_plunge_window_draw_roundtrip() -> None:
    """NEW pin: the drawn p0 satisfies t_insp(p0) = t_plunge on the PN5 trajectory.

    Deterministic under seed: rng(7).uniform(0, 4.5) = 2.8129295997210013 yr.
    Tolerance 1e-2 relative, far above the measured realized accuracy
    (max 2.8e-4, results/campaign51_20260728/plunge_window/).
    """
    few = pytest.importorskip("few")  # noqa: F841
    from few.utils.constants import YRSID_SI

    from darksiren_emri.plunge_window import (
        _get_trajectory,
        draw_plunge_window_initial_conditions,
    )

    ps = ParameterSpace()
    ps.M.value = 1e6  # detector-frame M_z
    ps.mu.value = 10.0
    ps.a.value = 0.98
    ps.e0.value = 0.1
    ps.x0.value = 0.9

    t_plunge = draw_plunge_window_initial_conditions(
        ps, np.random.default_rng(7), LISA_MISSION_DURATION_YEARS
    )

    assert t_plunge == pytest.approx(2.8129295997210013, abs=0.0)
    assert ps.t_plunge_yr == t_plunge
    assert 0.0 <= t_plunge <= LISA_MISSION_DURATION_YEARS

    # Domain rule: p0 >= p_sep(a, e0, x0) + 0.05 (few's separatrix buffer),
    # NOT the retired snapshot [10, 16] clamp.
    traj = _get_trajectory()
    traj.func.add_fixed_parameters(ps.M.value, ps.mu.value, ps.a.value)
    p_lo = float(traj.func.min_p(ps.e0.value, ps.x0.value))
    assert ps.p0.value >= p_lo - 1e-9

    # Round trip: integrate the PN5 trajectory from the drawn p0; it must
    # plunge at t_plunge (within tolerance) using few's DEFAULT integrator
    # tolerance (the waveform-generation setting).
    out = traj(
        ps.M.value,
        ps.mu.value,
        ps.a.value,
        ps.p0.value,
        ps.e0.value,
        ps.x0.value,
        T=2.0 * t_plunge,
    )
    t_end_yr = float(out[0][-1]) / YRSID_SI
    assert t_end_yr == pytest.approx(t_plunge, rel=1e-2)


def test_plunge_window_high_mass_p0_below_10() -> None:
    """The physics the snapshot convention forbade: at M_z = 1e7 a plunge-window
    p0 lies well below 10 (measured 4.71 at t_plunge = 4.5 yr), inside few 2.0's
    ACTUAL Pn5AAK domain (its sanity check imposes no p0 >= 10 bound; MEASURED
    waveform generation succeeds at p0 = 7/8/9, M_z = 1e7)."""
    pytest.importorskip("few")
    from darksiren_emri.plunge_window import draw_plunge_window_initial_conditions

    ps = ParameterSpace()
    ps.M.value = 1e7
    ps.mu.value = 10.0
    ps.a.value = 0.98
    ps.e0.value = 0.1
    ps.x0.value = 0.9

    # rng(3).uniform(0, 4.5) is deterministic; any in-window t_plunge at this
    # mass must land below the retired snapshot floor of 10.
    draw_plunge_window_initial_conditions(ps, np.random.default_rng(3), LISA_MISSION_DURATION_YEARS)
    assert ps.p0.value < 10.0
    assert ps.p0.value > 1.8  # above the Kerr a=0.98 separatrix region


def test_randomize_parameters_resets_t_plunge() -> None:
    ps = ParameterSpace()
    ps.t_plunge_yr = 1.23
    ps.randomize_parameters(np.random.default_rng(0))
    assert math.isnan(ps.t_plunge_yr)


def test_t_plunge_yr_not_in_waveform_parameter_dict() -> None:
    """t_plunge is provenance, not a 15th waveform parameter."""
    ps = ParameterSpace()
    assert "t_plunge_yr" not in ps._parameters_to_dict()
    assert len(ps._parameters_to_dict()) == 14
