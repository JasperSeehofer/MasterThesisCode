"""Regression tests for the EMRI mass-boundary single source of truth (issue #51).

History (physics-change protocol): the previous commit of this file pinned the
OLD behaviour — the unjustified ``cbe1a6f3`` draw-side override
``M in [10^4.5, 10^6]`` (``cosmological_model.py:179-180``, 2024-06-20, no
recorded justification) whose reuse as a detector-frame truncation produced
the hard ``log10 M_z = 6.000`` wall in every injection pool.

NEW behaviour under test (FIX-3 Amendment 2, 2026-07-27, author-ratified):
- ``constants.M_SOURCE_FRAME_MIN/MAX = [1e4, 1e7]`` — Babak et al. (2017)
  arXiv:1703.09722 source-frame mass-function valid band, THE single boundary.
- Population draw rejection happens on the source-frame constants.
- ``parameter_space.M`` limits are the DETECTOR-frame domain: the
  ``(1+max_redshift)``-lifted image of the source band (parameter holds
  ``M_z = M(1+z)``; Maggiore 2008 GW Vol. 1 Eq. 4.7), so no drawn event can
  hit the limits — no additional clamp anywhere in the pipeline.
"""

import numpy as np
import pytest

from darksiren_emri.constants import M_SOURCE_FRAME_MAX, M_SOURCE_FRAME_MIN
from darksiren_emri.cosmological_model import Model1CrossCheck
from darksiren_emri.datamodels.parameter_space import ParameterSpace


@pytest.fixture()
def model(monkeypatch: pytest.MonkeyPatch) -> Model1CrossCheck:
    """Model1CrossCheck without the expensive emcee burn-in (sampler unused here)."""
    monkeypatch.setattr(Model1CrossCheck, "setup_emri_events_sampler", lambda self: None)
    return Model1CrossCheck(rng=np.random.default_rng(0))


def test_single_source_constants_are_babak_band() -> None:
    """The scientific boundary is the Babak et al. (2017) valid band [1e4, 1e7]."""
    assert M_SOURCE_FRAME_MIN == 1e4
    assert M_SOURCE_FRAME_MAX == 1e7


def test_parameter_space_default_uses_constants() -> None:
    """ParameterSpace.M defaults read the constants (no duplicated literal)."""
    space = ParameterSpace()
    assert space.M.lower_limit == M_SOURCE_FRAME_MIN
    assert space.M.upper_limit == M_SOURCE_FRAME_MAX


def test_detector_frame_domain_is_lifted_image(model: Model1CrossCheck) -> None:
    """M limits = (1+z_max)-lifted image of the source band, NOT a clamp.

    OLD (pinned in the previous commit): [10^4.5, 1e6].
    NEW: [1e4, 1e7 * (1 + 1.5)] = [1e4, 2.5e7].
    """
    assert model.parameter_space.M.lower_limit == M_SOURCE_FRAME_MIN
    assert model.parameter_space.M.upper_limit == pytest.approx(
        M_SOURCE_FRAME_MAX * (1.0 + model.max_redshift)
    )
    assert model.parameter_space.M.upper_limit == pytest.approx(2.5e7)


def test_detector_frame_domain_tracks_max_redshift_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shallower --max_redshift override shrinks the lifted image consistently."""
    monkeypatch.setattr(Model1CrossCheck, "setup_emri_events_sampler", lambda self: None)
    shallow = Model1CrossCheck(rng=np.random.default_rng(0), max_redshift_override=0.5)
    assert shallow.parameter_space.M.upper_limit == pytest.approx(M_SOURCE_FRAME_MAX * 1.5)


def test_population_draw_accepts_full_babak_band(model: Model1CrossCheck) -> None:
    """NEW: 5e6 M_sun (rejected under the old 1e6 cap) is accepted; the whole
    band interior has finite log-probability with positive density."""
    for M in (1.5e4, 3.0e5, 5.0e6, 9.0e6):
        logp = model._log_probability(M, 0.5)
        assert np.isfinite(logp), f"M={M} rejected inside the Babak band"


def test_population_draw_rejects_outside_babak_band(model: Model1CrossCheck) -> None:
    """Draws outside the source-frame valid band are rejected."""
    assert model._log_probability(9.0e3, 0.5) == -np.inf
    assert model._log_probability(1.5e7, 0.5) == -np.inf


def test_emri_distribution_positive_on_widened_band(model: Model1CrossCheck) -> None:
    """The M1 rate density is positive across the full band and z range — the
    >= 10^6.25 coefficient branch now receives real traffic."""
    for log10M in np.linspace(4.05, 6.95, 12):
        for z in (0.05, 0.5, 1.0, 1.45):
            density = model.emri_distribution(10**log10M, z)
            assert np.isfinite(density)
            assert density > 0.0, f"density <= 0 at log10M={log10M:.2f}, z={z}"


def test_no_detector_frame_event_escapes_domain(model: Model1CrossCheck) -> None:
    """Structural no-clamp guarantee: max drawable M_z = M_max*(1+z_max) equals
    the domain upper limit exactly — the Fisher bounds check can never fire on
    an in-band draw (up to the 2-epsilon stencil margin, 2 M_sun at 2.5e7)."""
    max_drawable_mz = M_SOURCE_FRAME_MAX * (1.0 + model.max_redshift)
    assert max_drawable_mz <= model.parameter_space.M.upper_limit
