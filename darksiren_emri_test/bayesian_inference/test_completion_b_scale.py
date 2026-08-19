"""Regression tests for the completion-leg normalization convention.

``--completion_b_scale {derived,legacy}`` (docs/derivations/
bscale_completion_normalization.md §6/§7, ledger rows #130-#131). The memo's
verdict: the ``B_scale = beta_Gbar^phi/beta_Gbar`` multiplier introduced by
``FIXB_PATHA_PACKAGE.md`` §3.2 (2026-08-04) on the path-(A) completion-leg
numerator was an un-derived defect (an MFG-A2 violation, two detection
models inside one likelihood term). The DERIVED form drops the multiplier
entirely (``B_num_phi = B_num``, ``B_num_wbh_phi = B_num_wbh``) and is now
the production default; the un-derived ``"legacy"`` form is kept only as an
instrument for byte-identical reproduction of historical runs.

``path_a_completion_numerators`` (bayesian_statistics.py) is the exact
helper the ``p_Di`` combine site calls -- these tests pin its outputs, and
combine them into the full single-ratio ``p_i`` formula (memo §7's old vs
new formula), so the "actual code path" is exercised, not a hand-rolled
reimplementation.
"""

import inspect

import pytest

from darksiren_emri.arguments import Arguments
from darksiren_emri.bayesian_inference.bayesian_statistics import (
    BayesianStatistics,
    path_a_completion_numerators,
)

# Synthetic path-(A) mixture scalars (arbitrary but physically-signed: all
# positive, beta_Gbar_phi != beta_Gbar so the legacy multiplier is non-trivial).
_BETA_GBAR_PHI = 3.7e6
_BETA_GBAR = 5.5e6
_L_CAT = 2.1e-3
_B_NUM = 4.4e-4
_B_NUM_WBH = 6.6e-4
_ALPHA_G_PHI = 1.2e6
_D_TILDE_PHI = 1.9e7


# ── formula-level pin: path_a_completion_numerators ─────────────────────────


def test_derived_mode_drops_the_multiplier() -> None:
    """Derived form: B_num_phi = B_num, B_num_wbh_phi = B_num_wbh (memo §6)."""
    b_num_phi, b_num_wbh_phi, b_scale = path_a_completion_numerators(
        _B_NUM, _B_NUM_WBH, _BETA_GBAR_PHI, _BETA_GBAR, mode="derived"
    )
    assert b_num_phi == _B_NUM
    assert b_num_wbh_phi == _B_NUM_WBH
    assert b_scale == 1.0


def test_legacy_mode_reproduces_the_old_multiplier() -> None:
    """Legacy form: B_num_phi = B_num * beta_Gbar_phi/beta_Gbar (the retracted
    FIXB_PATHA_PACKAGE.md §3.2 transfer line), preserved for historical-run
    reproduction."""
    expected_scale = _BETA_GBAR_PHI / _BETA_GBAR
    b_num_phi, b_num_wbh_phi, b_scale = path_a_completion_numerators(
        _B_NUM, _B_NUM_WBH, _BETA_GBAR_PHI, _BETA_GBAR, mode="legacy"
    )
    assert b_scale == pytest.approx(expected_scale)
    assert b_num_phi == pytest.approx(_B_NUM * expected_scale)
    assert b_num_wbh_phi == pytest.approx(_B_NUM_WBH * expected_scale)


def test_legacy_mode_degenerate_beta_gbar_yields_zero_scale() -> None:
    """beta_Gbar <= 0 guards against division by zero (matches the original
    inline ``if beta_Gbar > 0.0 else 0.0`` guard)."""
    b_num_phi, b_num_wbh_phi, b_scale = path_a_completion_numerators(
        _B_NUM, _B_NUM_WBH, _BETA_GBAR_PHI, 0.0, mode="legacy"
    )
    assert b_scale == 0.0
    assert b_num_phi == 0.0
    assert b_num_wbh_phi == 0.0


# ── formula-level pin: the full combine (memo §7 old vs new formula) ────────


def _combine_with_bh(mode: str) -> float:
    """The exact ``p_i`` assembly at the p_Di combine site (with-BH channel):
    ``(alpha_G_phi * L_cat + B_num_wbh_phi) / D_tilde_phi``."""
    _, b_num_wbh_phi, _ = path_a_completion_numerators(
        _B_NUM, _B_NUM_WBH, _BETA_GBAR_PHI, _BETA_GBAR, mode=mode
    )
    return float((_ALPHA_G_PHI * _L_CAT + b_num_wbh_phi) / _D_TILDE_PHI)


def test_legacy_combine_matches_old_arithmetic() -> None:
    """memo §7 'Old formula': (alpha*L_cat + B_num*beta_Gbar_phi/beta_Gbar)/D_tilde."""
    expected = (_ALPHA_G_PHI * _L_CAT + _B_NUM_WBH * (_BETA_GBAR_PHI / _BETA_GBAR)) / _D_TILDE_PHI
    assert _combine_with_bh("legacy") == pytest.approx(expected)


def test_derived_combine_matches_new_arithmetic() -> None:
    """memo §7 'New formula': (alpha*L_cat + B_num)/D_tilde (no transfer factor)."""
    expected = (_ALPHA_G_PHI * _L_CAT + _B_NUM_WBH) / _D_TILDE_PHI
    assert _combine_with_bh("derived") == pytest.approx(expected)


def test_derived_and_legacy_combine_differ() -> None:
    """Sanity: the two conventions are NOT numerically degenerate for these
    synthetic inputs (beta_Gbar_phi != beta_Gbar), so the pins above are
    load-bearing. Exact (non-approx) comparison: the alpha*L_cat term
    dominates these particular synthetic magnitudes, so the two combined
    values are numerically close but must not be bit-identical."""
    assert _combine_with_bh("derived") != _combine_with_bh("legacy")


def test_limiting_case_equal_detection_models_no_op() -> None:
    """memo §5 limiting case: if beta_Gbar_phi == beta_Gbar (S_bar_phi == S_3D),
    B_scale = 1 and legacy degenerates to derived -- the factor measures only
    the model mismatch, not physics."""
    _, b_num_wbh_phi_legacy, b_scale = path_a_completion_numerators(
        _B_NUM, _B_NUM_WBH, _BETA_GBAR, _BETA_GBAR, mode="legacy"
    )
    _, b_num_wbh_phi_derived, _ = path_a_completion_numerators(
        _B_NUM, _B_NUM_WBH, _BETA_GBAR, _BETA_GBAR, mode="derived"
    )
    assert b_scale == pytest.approx(1.0)
    assert b_num_wbh_phi_legacy == pytest.approx(b_num_wbh_phi_derived)


# ── mode default / selection / validation ────────────────────────────────────


def test_completion_b_scale_default_is_derived_on_evaluate_signature() -> None:
    sig = inspect.signature(BayesianStatistics.evaluate)
    assert sig.parameters["completion_b_scale"].default == "derived"


def test_completion_b_scale_class_default_is_derived() -> None:
    assert BayesianStatistics._completion_b_scale == "derived"


def test_completion_b_scale_legacy_selectable() -> None:
    """evaluate() accepts 'legacy' and records it on the instance (the guard
    fires early, before catalog/model use, so a bare instance + None args is safe)."""
    instance = object.__new__(BayesianStatistics)
    try:
        BayesianStatistics.evaluate(
            instance,
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            0.73,
            completion_b_scale="legacy",
        )
    except AttributeError:
        # Bare instance hits an unrelated attribute error further down the
        # method; irrelevant here -- what matters is the mode was accepted
        # and recorded before that point.
        pass
    assert instance._completion_b_scale == "legacy"


def test_completion_b_scale_invalid_value_raises() -> None:
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(ValueError, match="completion_b_scale must be"):
        BayesianStatistics.evaluate(
            instance,
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            0.73,
            completion_b_scale="bogus",
        )


# ── CLI wiring ────────────────────────────────────────────────────────────


def test_cli_completion_b_scale_default() -> None:
    args = Arguments.create(["."])
    assert args.completion_b_scale == "derived"


def test_cli_completion_b_scale_legacy_selectable() -> None:
    args = Arguments.create([".", "--completion_b_scale", "legacy"])
    assert args.completion_b_scale == "legacy"


def test_cli_completion_b_scale_invalid_raises() -> None:
    with pytest.raises(SystemExit):
        Arguments.create([".", "--completion_b_scale", "bogus"])
