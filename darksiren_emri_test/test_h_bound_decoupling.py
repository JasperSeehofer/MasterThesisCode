"""Regression + new-behavior tests for the h-prior/admissibility decoupling.

Ref: results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/
b-hprior-fix/DECOUPLING_DESIGN.md (Research Graph 1, rows #293/#301/#304/#308).

`h.upper_limit` (0.86) is the HOST-WINDOW / prior-support bound consumed by
`get_redshift_outer_bounds(h_max=...)` and must stay frozen. A separate
`h_grid_admissibility_max` (1.00) widens only the evaluate() entry guard to
admit the ratified G-EXT wing. This file pins both bounds and the guard's
admission logic (mirrored directly, not via a full BayesianStatistics
instance, per the design's "small direct test of the bound logic is fine").
"""

from typing import Any

from darksiren_emri.cosmological_model import LamCDMScenario
from darksiren_emri.physical_relations import get_redshift_outer_bounds


def _guard_admissible_max(model: Any) -> float:
    """Mirror of the evaluate() entry-guard ceiling (bayesian_statistics.py:~4656)."""
    upper_limit: float = model.h.upper_limit
    return max(
        upper_limit,
        getattr(model, "h_grid_admissibility_max", upper_limit),
    )


def _guard_rejects(model: Any, h_value: float) -> bool:
    """Mirror of the evaluate() entry-guard bounds check."""
    admissible_max = _guard_admissible_max(model)
    return bool((h_value < model.h.lower_limit) or (h_value > admissible_max))


# ── Frozen host-window bound ────────────────────────────────────────────────


def test_lamcdm_h_upper_limit_is_0_86() -> None:
    """h.upper_limit (the host-window / prior-support bound) is FROZEN at 0.86."""
    model = LamCDMScenario()
    assert model.h.upper_limit == 0.86


def test_lamcdm_h_lower_limit_is_0_6() -> None:
    model = LamCDMScenario()
    assert model.h.lower_limit == 0.6


def test_get_redshift_outer_bounds_default_h_max_is_0_86() -> None:
    """get_redshift_outer_bounds's own h_max default is untouched by this design."""
    import inspect

    sig = inspect.signature(get_redshift_outer_bounds)
    assert sig.parameters["h_max"].default == 0.86


# ── New admissibility ceiling ───────────────────────────────────────────────


def test_lamcdm_h_grid_admissibility_max_is_1_00() -> None:
    model = LamCDMScenario()
    assert model.h_grid_admissibility_max == 1.00


# ── Guard admission logic ───────────────────────────────────────────────────


def test_guard_admits_wing_up_to_1_00() -> None:
    model = LamCDMScenario()
    assert not _guard_rejects(model, 0.87)
    assert not _guard_rejects(model, 1.00)


def test_guard_still_rejects_above_1_01() -> None:
    model = LamCDMScenario()
    assert _guard_rejects(model, 1.01)


def test_guard_still_rejects_below_lower_limit() -> None:
    model = LamCDMScenario()
    assert _guard_rejects(model, 0.59)


# ── Limiting case 2: degenerate ceiling reproduces old guard exactly ───────


def test_guard_degenerate_ceiling_equals_old_behavior() -> None:
    """h_grid_admissibility_max == h.upper_limit reproduces the pre-decoupling guard:
    the wing (e.g. h=0.87) is rejected again, exactly as it was before this design.
    """
    model = LamCDMScenario()
    model.h_grid_admissibility_max = model.h.upper_limit
    assert _guard_rejects(model, 0.87)
    assert not _guard_rejects(model, 0.86)


# ── Limiting case 3: absent attribute falls back to h.upper_limit ─────────


def test_guard_absent_attribute_falls_back_to_upper_limit() -> None:
    """A duck-typed scenario without h_grid_admissibility_max behaves like the old guard."""

    class _BareScenario:
        def __init__(self) -> None:
            self.h = LamCDMScenario().h

    bare = _BareScenario()
    assert _guard_rejects(bare, 0.87)
    assert not _guard_rejects(bare, 0.86)


# ── Mirror path: runtime widening of h.upper_limit keeps working ──────────


def test_guard_mirror_widening_of_upper_limit_still_admits() -> None:
    """correspondence_1d.py-style widening (h.upper_limit = max(h.upper_limit, eff_hi))
    keeps admitting the widened value through the guard via the max() clause.
    """
    model = LamCDMScenario()
    model.h.upper_limit = max(model.h.upper_limit, 0.95)
    assert not _guard_rejects(model, 0.95)
