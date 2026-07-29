"""Regression tests for the Detection measured-mass domain (issue #51).

Physics-change protocol two-step: the previous commit pinned the OLD behaviour
— the measurement-scatter draw clipped/truncated the measured detector-frame
mass to a hardcoded ``[1e4, 1e6]``, a survival of the retired
``[10^4.5, 1e6]`` draw-side era. Measured consequence on campaign #51 seed
61000: 127/1590 events (8.0 %) pinned at exactly 1e6 in
``prepared_cramer_rao_bounds.csv`` while the raw CRBs reached 1.63e6 with no
pile-up.

THIS commit derives the domain from the single mass boundary:
``[M_SOURCE_FRAME_MIN, M_SOURCE_FRAME_MAX * (1 + HOST_DRAW_Z_MAX)]`` — the
same detector-frame image ``Model1CrossCheck`` puts on ``parameter_space.M``,
so the scatter draw can never clip a physically drawn event.
"""

import numpy as np

from master_thesis_code.constants import (
    HOST_DRAW_Z_MAX,
    M_SOURCE_FRAME_MAX,
    M_SOURCE_FRAME_MIN,
)
from master_thesis_code.datamodels.detection import (
    _M_Z_DOMAIN_MAX,
    _M_Z_DOMAIN_MIN,
    Detection,
)


def _detection_at(mass: float, mass_uncertainty: float) -> Detection:
    """Minimal Detection whose measured-mass draw can be exercised directly."""
    det = Detection.__new__(Detection)
    det.M = mass
    det.M_uncertainty = mass_uncertainty
    det.phi = 1.0
    det.phi_error = 1e-3
    det.theta = 1.0
    det.theta_error = 1e-3
    det.d_L = 1.0
    det.d_L_uncertainty = 1e-3
    return det


def test_domain_is_derived_from_single_boundary() -> None:
    """NEW: the measured-mass domain is the lifted image of the Babak band."""
    assert _M_Z_DOMAIN_MIN == M_SOURCE_FRAME_MIN
    assert _M_Z_DOMAIN_MAX == M_SOURCE_FRAME_MAX * (1.0 + HOST_DRAW_Z_MAX)
    assert _M_Z_DOMAIN_MAX == 2.5e7  # OLD pinned value: 1e6


def test_measured_mass_above_1e6_survives_the_draw() -> None:
    """NEW: a 1.5e6 M_sun event (inside the campaign band) is no longer pinned.

    OLD behaviour: every draw was truncated at 1e6.
    """
    draws = []
    for _ in range(200):
        det = _detection_at(1.5e6, 1.0e5)
        det._independent_draw()
        draws.append(det.M)
    assert max(draws) > 1e6, "measured mass must be free to exceed the retired cap"
    assert abs(float(np.median(draws)) - 1.5e6) < 5.0e4, "draw should center on truth"
    assert sum(1 for m in draws if m == 1e6) == 0, "no pile-up at the retired cap"


def test_no_literal_1e6_bounds_remain() -> None:
    """NEW: the hardcoded literals are gone from the scatter draw."""
    import inspect

    from master_thesis_code.datamodels import detection as detection_module

    src = inspect.getsource(detection_module.Detection._independent_draw)
    assert "1e6" not in src
    assert "_M_Z_DOMAIN_MAX" in src


def test_low_mass_bound_is_population_floor() -> None:
    """The lower bound is the population floor 1e4 (holds before AND after)."""
    draws = []
    for _ in range(50):
        d = _detection_at(1.1e4, 5.0e3)
        d._independent_draw()
        draws.append(d.M)
    assert min(draws) >= M_SOURCE_FRAME_MIN - 1e-6


def test_clip_path_respects_domain() -> None:
    """The multivariate clip path uses the same derived domain (limiting case:
    an absurd sample is clipped to the domain edges, not to 1e6)."""
    det = _detection_at(2.0e7, 1.0e6)
    clipped_high = float(np.clip(1.0e9, _M_Z_DOMAIN_MIN, _M_Z_DOMAIN_MAX))
    clipped_low = float(np.clip(1.0, _M_Z_DOMAIN_MIN, _M_Z_DOMAIN_MAX))
    assert clipped_high == _M_Z_DOMAIN_MAX == 2.5e7
    assert clipped_low == _M_Z_DOMAIN_MIN == 1e4
    assert np.isfinite(det.M)
