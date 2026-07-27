"""Regression pins for the EMRI mass-boundary single source of truth (issue #51).

This file is committed in two steps per the physics-change protocol:

1. THIS commit pins the OLD behaviour — the unjustified draw-side override
   ``M in [10^4.5, 10^6]`` introduced by ``cbe1a6f3`` (2024-06-20, "changed
   boundaries of mass", no recorded justification) at
   ``cosmological_model.py:179-180``.
2. The follow-up ``[PHYSICS]`` commit replaces these pins with the NEW
   behaviour (Babak et al. 2017 arXiv:1703.09722 source-frame valid band
   ``[1e4, 1e7]`` as the single scientific boundary in ``constants.py``,
   detector-frame domain = its ``(1+z_max)``-lifted image), so the diff of
   this file documents the numerical change.
"""

import numpy as np
import pytest

from master_thesis_code.cosmological_model import Model1CrossCheck


@pytest.fixture()
def model(monkeypatch: pytest.MonkeyPatch) -> Model1CrossCheck:
    """Model1CrossCheck without the expensive emcee burn-in (sampler unused here)."""
    monkeypatch.setattr(
        Model1CrossCheck, "setup_emri_events_sampler", lambda self: None
    )
    return Model1CrossCheck(rng=np.random.default_rng(0))


def test_draw_side_mass_bounds_old_override(model: Model1CrossCheck) -> None:
    """OLD pin: the cbe1a6f3 override narrows the Babak band to [10^4.5, 1e6]."""
    assert model.parameter_space.M.lower_limit == pytest.approx(10**4.5)
    assert model.parameter_space.M.upper_limit == pytest.approx(10**6.0)


def test_population_draw_rejects_above_old_cap(model: Model1CrossCheck) -> None:
    """OLD pin: a 5e6 M_sun source-frame draw is rejected (inside Babak band)."""
    assert model._log_probability(5.0e6, 0.5) == -np.inf


def test_population_draw_accepts_mid_band(model: Model1CrossCheck) -> None:
    """A mid-band draw (3e5 M_sun, z=0.5) is accepted with finite log-probability."""
    assert np.isfinite(model._log_probability(3.0e5, 0.5))
