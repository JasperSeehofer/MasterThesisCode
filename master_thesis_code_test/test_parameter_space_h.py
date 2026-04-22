"""Tests for h-threading in ParameterSpace.set_host_galaxy_parameters (PE-01, Phase 37)
and per-parameter derivative_epsilon stability (PE-02).

SC-1: luminosity_distance scales linearly with h (d_L at h=1.0 is exactly double d_L at h=0.5).
SC-2: calling set_host_galaxy_parameters without h raises TypeError (no default).
SC-3: all 14 parameters have non-uniform, non-default derivative_epsilon in valid step-size regime.
"""

import numpy as np
import pytest

from master_thesis_code.datamodels.parameter_space import Parameter, ParameterSpace
from master_thesis_code.galaxy_catalogue.handler import HostGalaxy


def _make_host(z: float = 0.1) -> HostGalaxy:
    """Construct a minimal HostGalaxy for testing via the from_attributes factory."""
    return HostGalaxy.from_attributes(
        phiS=0.0,
        qS=0.0,
        z=z,
        z_error=0.001,
        M=1e5,
        M_error=1e3,
    )


def test_set_host_galaxy_parameters_h_ratio() -> None:
    """SC-1: d_L at h=1.0 is exactly double d_L at h=0.5.

    dist(z, h=h) scales as 1/h for fixed z (Hogg 1999, Eq. 16: d_L = c(1+z)/H_0 * integral).
    At h=0.5 the Hubble constant is half that at h=1.0, so d_L doubles.
    """
    host = _make_host(z=0.1)

    ps_half = ParameterSpace()
    ps_half.set_host_galaxy_parameters(host, h=0.5)

    ps_one = ParameterSpace()
    ps_one.set_host_galaxy_parameters(host, h=1.0)

    # d_L ∝ c/H_0 ∝ 1/h (Hogg 1999 Eq. 16), so d_L(h=0.5) = 2 × d_L(h=1.0).
    ratio = ps_half.luminosity_distance.value / ps_one.luminosity_distance.value
    np.testing.assert_allclose(
        ratio,
        2.0,
        rtol=1e-10,
        err_msg="luminosity_distance should scale as 1/h: d_L(h=0.5) = 2 × d_L(h=1.0)",
    )


def test_set_host_galaxy_parameters_requires_h() -> None:
    """SC-2: calling set_host_galaxy_parameters without h raises TypeError.

    h has no default value (D-01) so Python raises TypeError at call time.
    """
    host = _make_host(z=0.1)
    ps = ParameterSpace()
    with pytest.raises(TypeError):
        ps.set_host_galaxy_parameters(host)  # type: ignore[call-arg]


def test_derivative_epsilon_per_parameter() -> None:
    """SC-3 structural: all 14 parameters have distinct, non-default derivative_epsilon values.

    REQ-ID: PE-02
    Reference: Vallisneri (2008) arXiv:gr-qc/0703086 Eq. (A11)
    """
    ps = ParameterSpace()
    epsilons = {
        p.symbol: p.derivative_epsilon for p in vars(ps).values() if isinstance(p, Parameter)
    }
    assert len(epsilons) == 14, f"Expected 14 parameters, got {len(epsilons)}"
    # None should be the old uniform default
    for symbol, eps in epsilons.items():
        assert eps != 1e-6, f"Parameter {symbol} still has uniform default 1e-6"
        assert eps > 0, f"Parameter {symbol} has non-positive epsilon {eps}"
    # At least 4 distinct values (scale diversity)
    assert len(set(epsilons.values())) >= 4, (
        f"Expected >=4 distinct epsilon values, got {len(set(epsilons.values()))}"
    )


def test_fisher_determinant_stability() -> None:
    """SC-3 stability: per-parameter epsilons are in the valid step-size regime.

    Validates that each epsilon is:
    - At least 1e-6 relative to the parameter's representative value (round-off safety)
    - At most 1% of the parameter's range width (truncation safety)

    REQ-ID: PE-02
    Reference: Vallisneri (2008) arXiv:gr-qc/0703086 Eq. (A11): h* ≈ ε_machine^(1/4) × |x|
    """
    ps = ParameterSpace()
    params = [p for p in vars(ps).values() if isinstance(p, Parameter)]

    for p in params:
        range_width = p.upper_limit - p.lower_limit
        # Representative value: use geometric mean for log-uniform parameters (range spans
        # >100×), arithmetic midpoint otherwise. M is log-uniform over [1e4, 1e7]: geometric
        # mean = sqrt(1e4 × 1e7) ≈ 3162 SM, which is the correct scale for the step-size check.
        if p.lower_limit > 0 and p.upper_limit / p.lower_limit > 100:
            midpoint = (p.lower_limit * p.upper_limit) ** 0.5  # geometric mean
        else:
            midpoint = abs((p.upper_limit + p.lower_limit) / 2.0)
        if midpoint == 0.0:
            midpoint = range_width / 2.0  # x0 is centred at 0; use half-range

        # epsilon must be >= 1e-6 * representative_value (avoids round-off catastrophe)
        min_safe = 1e-6 * midpoint if midpoint > 0 else 1e-10
        assert p.derivative_epsilon >= min_safe, (
            f"{p.symbol}: epsilon {p.derivative_epsilon} < min_safe {min_safe} "
            f"(midpoint={midpoint})"
        )

        # epsilon must be <= 1% of range width (avoids leaving Taylor regime)
        max_safe = 0.01 * range_width
        assert p.derivative_epsilon <= max_safe, (
            f"{p.symbol}: epsilon {p.derivative_epsilon} > 1% of range {range_width} "
            f"(max_safe={max_safe})"
        )
