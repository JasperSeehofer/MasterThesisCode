"""Tests for h-threading in ParameterSpace.set_host_galaxy_parameters (PE-01, Phase 37).

SC-1: luminosity_distance scales linearly with h (d_L at h=1.0 is exactly double d_L at h=0.5).
SC-2: calling set_host_galaxy_parameters without h raises TypeError (no default).
"""

import numpy as np
import pytest

from master_thesis_code.datamodels.parameter_space import ParameterSpace
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
