"""
Unit tests for the L_cat in-catalogue aggregation form.

The production catalog term (``BayesianStatistics.p_Di``) computes the
in-catalogue likelihood as the **ratio of sums** over candidate host galaxies

    L_cat = (Σ_g N_g) / (Σ_g D_g)

i.e. a single shared selection denominator, per Gray et al. (2020),
arXiv:1908.06050 **Eq. (A.9)/(A.10)** (verified against the paper appendix).

History: Phase 38 (REQ STAT-01/STAT-02) had switched this to the *mean of
per-galaxy self-normalized ratios* ``(1/N) Σ_g (N_g/D_g)``, citing a
mis-attributed "Eq. 24-25".  Reading Gray's actual appendix (Eq. A.9/A.10)
shows the in-catalogue likelihood is a ratio of sums with p_det = p(D_GW|...)
appearing ONLY in the (summed) denominator.  The mean-of-ratios form gives each
galaxy its own selection normalization instead of one population-level β(H0),
and (together with a spurious p_det in the numerator) biased the recovered H0
high.  Reverting to the ratio-of-sums form is empirically confirmed to move the
seed400 MAP toward truth (1D catalog-only 0.740 -> 0.735; full 1D 0.750 ->
0.740).  See docs/H0_BIAS_RESOLUTION.md §3.17.

These tests pin the algebraic identity and assert that the canonical
(ratio-of-sums) form is the one production uses.
"""

import numpy as np


def _lcat_ratio_of_sums(pairs: list[tuple[float, float]]) -> float:
    """Canonical Gray et al. (2020) Eq. (A.9/A.10) form: (Σ N_g) / (Σ D_g).

    This is the form used in production (``bayesian_statistics.py`` L_cat
    aggregation).
    """
    total_n = sum(n for n, d in pairs)
    total_d = sum(d for n, d in pairs)
    return total_n / total_d if total_d > 0 else 0.0


def _lcat_mean_of_ratios(pairs: list[tuple[float, float]]) -> float:
    """Superseded Phase-38 form: (1/N) Σ (N_g/D_g).

    Per-galaxy self-normalized ratios; NOT Gray Eq. A.9/A.10.  Kept here only to
    document the divergence from the canonical form.
    """
    ratios = [n / d for n, d in pairs if d > 0]
    return float(np.mean(ratios)) if ratios else 0.0


def test_lcat_single_galaxy() -> None:
    """Single galaxy: both forms are identical (N=1 limiting case)."""
    pairs = [(3.0, 5.0)]
    assert abs(_lcat_ratio_of_sums(pairs) - _lcat_mean_of_ratios(pairs)) < 1e-12
    assert abs(_lcat_ratio_of_sums(pairs) - 3.0 / 5.0) < 1e-12


def test_lcat_constant_d_equivalent() -> None:
    """All D_g equal: both forms agree (D_g = D ⟹ ΣN_g/(N×D) = (1/N)Σ(N_g/D))."""
    D = 2.0
    pairs = [(1.0, D), (3.0, D), (5.0, D)]
    assert abs(_lcat_ratio_of_sums(pairs) - _lcat_mean_of_ratios(pairs)) < 1e-12


def test_lcat_ratio_of_sums_is_canonical() -> None:
    """
    With varying D_g the two forms diverge; the canonical (ratio-of-sums) form
    is Gray Eq. (A.9/A.10).

      N=2, N_1=1, D_1=1, N_2=1, D_2=2
        ratio-of-sums (canonical): (1+1)/(1+2) = 2/3
        mean-of-ratios (Phase 38): (1/2)(1/1 + 1/2) = 3/4
    """
    # Minimal counterexample (N=2)
    pairs_2 = [(1.0, 1.0), (1.0, 2.0)]
    canonical_2 = _lcat_ratio_of_sums(pairs_2)
    superseded_2 = _lcat_mean_of_ratios(pairs_2)
    assert abs(canonical_2 - 2.0 / 3.0) < 1e-12, f"ratio-of-sums: expected 2/3, got {canonical_2}"
    assert abs(superseded_2 - 3.0 / 4.0) < 1e-12, (
        f"mean-of-ratios: expected 3/4, got {superseded_2}"
    )
    assert abs(canonical_2 - superseded_2) > 0.05, "Forms should diverge for varying D_g"

    # 3-galaxy case
    pairs_3 = [(1.0, 1.0), (2.0, 4.0), (1.0, 0.5)]
    canonical_3 = _lcat_ratio_of_sums(pairs_3)
    superseded_3 = _lcat_mean_of_ratios(pairs_3)
    # canonical_3 = (1+2+1)/(1+4+0.5) = 4/5.5 = 8/11
    assert abs(canonical_3 - 8.0 / 11.0) < 1e-12, (
        f"ratio-of-sums 3-gal: expected 8/11, got {canonical_3}"
    )
    # superseded_3 = (1/3)(1/1 + 2/4 + 1/0.5) = (1/3)(3.5) = 7/6
    assert abs(superseded_3 - 7.0 / 6.0) < 1e-12, (
        f"mean-of-ratios 3-gal: expected 7/6, got {superseded_3}"
    )
    assert abs(canonical_3 - superseded_3) > 0.05, (
        "Forms should diverge for 3-galaxy varying D_g case"
    )


def test_production_uses_ratio_of_sums() -> None:
    """Regression guard: the production L_cat aggregation must be ratio-of-sums.

    Verifies the source computes a (rate-WEIGHTED, CHANGE 3) ratio of sums
    ``(Σ w·N) / (Σ w·D)`` via the ``weighted_ratio_of_sums`` helper (Gray
    A.9/A.10) and NOT ``np.mean([r[0]/r[1] ...])`` (the superseded Phase-38
    mean-of-ratios). The constant-weight limit of ``weighted_ratio_of_sums``
    reproduces the plain ``sum(r[0]) / sum(r[1])`` form (pinned independently in
    test_rate_weighted_catalog.py::test_wros_constant_weight_limit_equals_plain_ratio).
    """
    import inspect

    from darksiren_emri.bayesian_inference import bayesian_statistics

    src = inspect.getsource(bayesian_statistics.BayesianStatistics.p_Di)
    assert "weighted_ratio_of_sums" in src, (
        "L_cat aggregation should use the (weighted) ratio-of-sums helper, Gray Eq. A.9/A.10"
    )
    assert "np.mean(ratios_without_bh)" not in src, (
        "L_cat must not use the superseded mean-of-ratios form"
    )
