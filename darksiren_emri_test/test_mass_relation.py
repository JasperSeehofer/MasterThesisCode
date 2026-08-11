"""Regression tests for the stellar-mass -> BH-mass relation (handler.py).

Pins the Reines & Volonteri (2015) AGN M_BH-M_*,total relation and the
[PHYSICS] error-budget fix: (A) the intrinsic scatter (epsilon_0 = 0.24 dex) is now
included in BH_mass_error (previously omitted -> ~3x under-estimate at the pivot),
and (B) the spurious "/10" in the stellar-mass-error term is removed.

Ref: Reines & Volonteri (2015), ApJ 813, 82, arXiv:1508.06274, Eq. (5) + Sec. 4.1.
"""

from __future__ import annotations

import numpy as np

from darksiren_emri.galaxy_catalogue.handler import (
    _empiric_MBH_to_M_stellar_relation,
    _empiric_stellar_mass_to_BH_mass_relation,
    beta,
)

# stellar_mass is in units of 1e10 Msun, so stellar_mass=10 <=> M_* = 1e11 Msun (the R&V pivot).
_PIVOT = 10.0


def test_forward_central_value_matches_reines_volonteri() -> None:
    """At M_* = 1e11 Msun, log10(M_BH) = 7.45 (R&V15 Eq. 5 intercept)."""
    BH_mass, _ = _empiric_stellar_mass_to_BH_mass_relation(_PIVOT, 0.0)
    assert np.isclose(np.log10(BH_mass), 7.45, atol=1e-6)
    assert np.isclose(BH_mass, 10**7.45, rtol=1e-6)


def test_forward_error_includes_intrinsic_scatter() -> None:
    """BH_mass_error now carries the 0.24 dex intrinsic scatter (the dominant term).

    NEW fractional error (CV) at the pivot ~= 0.592 (= sqrt(sigma_int^2 + d_alpha^2 + sm_term)).
    OLD (scatter omitted) was ~= 0.1845 -- a ~3.2x under-estimate. We assert the NEW value and
    that it is >3x the old fit-only floor.
    """
    BH_mass, BH_mass_error = _empiric_stellar_mass_to_BH_mass_relation(_PIVOT, 1.0)
    cv = BH_mass_error / BH_mass
    assert np.isclose(cv, 0.5919, atol=2e-3)  # NEW (with intrinsic scatter)
    assert cv > 3.0 * 0.1845  # was ~0.1845 before the fix (fit-error only)


def test_stellar_mass_error_term_has_no_spurious_factor_10() -> None:
    """The /10 operator-precedence bug is fixed: the stellar-mass-error term contributes
    (beta/M_* * sigma_*)^2 to the variance, NOT (beta/M_*/10 * sigma_*)^2.

    Isolate it as the variance increment from sigma_* = 0 -> 1 at the pivot.
    """
    _, err0 = _empiric_stellar_mass_to_BH_mass_relation(_PIVOT, 0.0)
    BH_mass, err1 = _empiric_stellar_mass_to_BH_mass_relation(_PIVOT, 1.0)
    cv0 = err0 / BH_mass
    cv1 = err1 / BH_mass
    dvar = cv1**2 - cv0**2
    assert np.isclose(dvar, (beta / _PIVOT * 1.0) ** 2, rtol=1e-6)  # = 0.105^2 (FIXED)
    assert not np.isclose(dvar, (beta / _PIVOT / 10 * 1.0) ** 2, rtol=1e-2)  # NOT the old /10 bug


def test_inverse_round_trips() -> None:
    """forward then inverse recovers the input stellar mass."""
    sm_in = 4.2  # 4.2e10 Msun
    BH_mass, _ = _empiric_stellar_mass_to_BH_mass_relation(sm_in, 0.0)
    sm_out, _ = _empiric_MBH_to_M_stellar_relation(BH_mass, 0.0)
    assert np.isclose(sm_out, sm_in, rtol=1e-9)


def test_inverse_error_uses_one_over_beta() -> None:
    """The inverse M_BH-error term scales as (1/beta)^2, not beta^2 (the previous bug)."""
    MBH = 1.0e7
    sm0, e0 = _empiric_MBH_to_M_stellar_relation(MBH, 0.0)
    sm1, e1 = _empiric_MBH_to_M_stellar_relation(MBH, 0.1 * MBH)
    # increment in (stellar_mass_error/stellar_mass)^2 from the M_BH-error term
    dvar = (e1 / sm1) ** 2 - (e0 / sm0) ** 2
    assert np.isclose(dvar, (0.1 / beta) ** 2, rtol=1e-6)  # FIXED (1/beta)
    assert not np.isclose(dvar, (0.1 * beta) ** 2, rtol=1e-3)  # NOT the old *beta bug
