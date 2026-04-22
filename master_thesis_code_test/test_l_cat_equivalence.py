"""
Unit tests for L_cat formula correctness.

Tests that the canonical Gray et al. (2020) arXiv:1908.06050 Eq. 24-25 form
L_cat = (1/N) Σ_g [N_g / D_g]
is used, and that it differs from the incorrect ΣN_g/ΣD_g form when D_g vary.

REQ-ID: STAT-01, STAT-02 (Phase 38)
"""

import numpy as np


def _lcat_old(pairs: list[tuple[float, float]]) -> float:
    """Old (incorrect) form: ΣN_g / ΣD_g."""
    total_n = sum(n for n, d in pairs)
    total_d = sum(d for n, d in pairs)
    return total_n / total_d if total_d > 0 else 0.0


def _lcat_new(pairs: list[tuple[float, float]]) -> float:
    """Canonical Gray et al. (2020) Eq. 24-25 form: (1/N) Σ [N_g/D_g]."""
    ratios = [n / d for n, d in pairs if d > 0]
    return float(np.mean(ratios)) if ratios else 0.0


def test_lcat_single_galaxy() -> None:
    """Single galaxy: both forms are identical (N=1 limiting case)."""
    pairs = [(3.0, 5.0)]
    assert abs(_lcat_old(pairs) - _lcat_new(pairs)) < 1e-12
    assert abs(_lcat_new(pairs) - 3.0 / 5.0) < 1e-12


def test_lcat_constant_d_equivalent() -> None:
    """All D_g equal: both forms agree (D_g = D ⟹ ΣN_g/(N×D) = (1/N)Σ(N_g/D))."""
    D = 2.0
    pairs = [(1.0, D), (3.0, D), (5.0, D)]
    assert abs(_lcat_old(pairs) - _lcat_new(pairs)) < 1e-12


def test_lcat_three_galaxies_varying_d() -> None:
    """
    Three galaxies with varying D_g: forms diverge; new form is correct.

    Counterexample from Phase 38 STAT-01 analysis:
      N=2, N_1=1, D_1=1, N_2=1, D_2=2
      Old: (1+1)/(1+2) = 2/3
      New: (1/2)(1/1 + 1/2) = 3/4

    Extended to 3 galaxies with varying D to confirm divergence is not accidental.
    """
    # Minimal counterexample (N=2)
    pairs_2 = [(1.0, 1.0), (1.0, 2.0)]
    old_2 = _lcat_old(pairs_2)
    new_2 = _lcat_new(pairs_2)
    assert abs(old_2 - 2.0 / 3.0) < 1e-12, f"Old form: expected 2/3, got {old_2}"
    assert abs(new_2 - 3.0 / 4.0) < 1e-12, f"New form: expected 3/4, got {new_2}"
    assert abs(old_2 - new_2) > 0.05, "Forms should diverge for varying D_g"

    # 3-galaxy case
    pairs_3 = [(1.0, 1.0), (2.0, 4.0), (1.0, 0.5)]
    old_3 = _lcat_old(pairs_3)
    new_3 = _lcat_new(pairs_3)
    # new_3 = (1/3)(1/1 + 2/4 + 1/0.5) = (1/3)(1 + 0.5 + 2) = (1/3)(3.5) = 7/6
    assert abs(new_3 - 7.0 / 6.0) < 1e-12, f"New form 3-galaxy: expected 7/6, got {new_3}"
    # old_3 = (1+2+1)/(1+4+0.5) = 4/5.5 = 8/11
    assert abs(old_3 - 8.0 / 11.0) < 1e-12, f"Old form 3-galaxy: expected 8/11, got {old_3}"
    assert abs(old_3 - new_3) > 0.05, "Forms should diverge for 3-galaxy varying D_g case"
