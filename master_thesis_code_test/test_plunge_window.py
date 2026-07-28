"""BEFORE-change regression pins — commit this file FIRST as
master_thesis_code_test/test_plunge_window.py, then replace it with the
post-change version (left in the working tree) in the [PHYSICS] commit.

Pins the pre-2026-07-28 state:
  * snapshot initial conditions p0 ~ U[10, 16] (uniform law + seeded value)
  * ParameterEstimation.T = 5 yr (unofficial, hardcoded 2023)
  * LisaTdiConfiguration.t_obs_years = 4.0 (inconsistent with T = 5)

Verified green against commit d31822c (pre-change tree).
Audit motivating the change: results/campaign51_20260728/highm_audit/
HIGHM_AUDIT.md item 1.
"""

import numpy as np
import pytest

from master_thesis_code.datamodels.parameter_space import ParameterSpace


def test_parameter_estimation_T_is_5_yr_unofficial() -> None:
    """OLD pin: T = 5 yr, hardcoded since 2023 (no literature reference).

    Will flip to LISA_MISSION_DURATION_YEARS = 4.5 (Colpi et al. 2024,
    arXiv:2402.07571) in the plunge-window [PHYSICS] commit.
    """
    from master_thesis_code.parameter_estimation.parameter_estimation import (
        ParameterEstimation,
    )

    assert ParameterEstimation.T == 5


def test_lisa_tdi_t_obs_years_is_4_yr() -> None:
    """OLD pin: confusion-noise t_obs_years default 4.0 (inconsistent with T=5)."""
    from master_thesis_code.LISA_configuration import LisaTdiConfiguration

    assert LisaTdiConfiguration().t_obs_years == 4.0


def test_snapshot_p0_draw_seeded_value() -> None:
    """OLD pin: byte-exact seeded snapshot draw, p0 ~ U[10, 16]."""
    ps = ParameterSpace()
    ps.randomize_parameters(np.random.default_rng(42))
    assert ps.p0.value == pytest.approx(14.184208174356183, abs=0.0)
    assert 10.0 <= ps.p0.value <= 16.0


def test_snapshot_p0_bounds_are_few_input_domain() -> None:
    """OLD pin: the [10, 16] bounds are few's Pn5AAK documented input domain
    adopted as a prior (HIGHM_AUDIT.md item 1 provenance)."""
    ps = ParameterSpace()
    assert ps.p0.lower_limit == 10.0
    assert ps.p0.upper_limit == 16.0


def test_snapshot_draw_is_uniform_in_bounds() -> None:
    """OLD pin: p0 uniform on [10, 16] (KS against the uniform CDF)."""
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
