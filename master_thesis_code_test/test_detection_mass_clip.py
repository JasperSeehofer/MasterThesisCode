"""Regression pins for the Detection measured-mass clip (issue #51 leftover cap).

Physics-change protocol two-step: THIS commit pins the OLD behaviour — the
measurement-scatter draw clips/truncates the measured detector-frame mass to a
hardcoded ``[1e4, 1e6]`` (``datamodels/detection.py``), a survival of the
retired ``[10^4.5, 1e6]`` draw-side era. Measured consequence on campaign #51
seed 61000: 127/1590 events (8.0 %) pinned at exactly 1e6 in
``prepared_cramer_rao_bounds.csv`` while the raw CRBs reach 1.63e6 with no
pile-up.

The follow-up ``[PHYSICS]`` commit replaces the literals with the derived
detector-frame domain (constants.M_SOURCE_FRAME_* lifted by 1+HOST_DRAW_Z_MAX)
and flips these pins.
"""

import numpy as np

from master_thesis_code.datamodels.detection import Detection


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


def test_independent_draw_truncates_measured_mass_at_1e6_old() -> None:
    """OLD pin: a 1.5e6 M_sun event's measured mass can never exceed 1e6."""
    rng_draws = []
    for _ in range(50):
        det = _detection_at(1.5e6, 1.0e5)
        det._independent_draw()
        rng_draws.append(det.M)
    assert max(rng_draws) <= 1e6 + 1e-6, "old truncation bound was 1e6"


def test_clip_bounds_are_hardcoded_literals_old() -> None:
    """OLD pin: the clip bounds are the literals 1e4 / 1e6 (not derived)."""
    import inspect

    from master_thesis_code.datamodels import detection as detection_module

    src = inspect.getsource(detection_module)
    assert "np.clip(sample[3], 1e4, 1e6)" in src
    assert "(1e6 - self.M) / self.M_uncertainty" in src


def test_low_mass_bound_is_1e4() -> None:
    """The lower bound is the population floor 1e4 (holds before AND after)."""
    det = _detection_at(1.1e4, 5.0e3)
    draws = []
    for _ in range(50):
        d = _detection_at(1.1e4, 5.0e3)
        d._independent_draw()
        draws.append(d.M)
    assert min(draws) >= 1e4 - 1e-6
    assert np.isfinite(det.M)
