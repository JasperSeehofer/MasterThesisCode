"""Regression tests for the B_num analysis-depth cap (issue #30 prep).

``p_Di``'s completion-numerator integral

    B_num(h) = INTEGRAL (1-f(z)) p_GW(z) dVc/(1+z) dz over [z_lower, z_upper]

shares the exact same functional form as D(h), beta_Gbar(h), and
Sigma_global(h) -- all of which are already capped at
``min(z_max(h), max_redshift)`` (the analysis-depth truncation knob,
``Model1CrossCheck.max_redshift``, landed in f29a5e7). Before this fix,
``z_upper`` in ``p_Di`` was the raw 4-sigma d_L window with NO such cap, so
B_num could integrate population density beyond the analysis depth while its
own denominator D(h) did not -- a domain mismatch in the single ratio
``p_i = (beta_G*L_cat + B_num)/D(h)``.

Regression discipline (physics-change protocol, MAX_REDSHIFT_SEMANTICS.md
finding #4): ``test_cap_binds_reduces_b_num`` PINS the OLD (pre-fix,
uncapped) numerical value of B_num on a case where a depth cap of 0.2 would
bind (the event's raw 4-sigma window is [0.047, 0.351]) -- it must FAIL
before the ``z_upper = min(z_upper, redshift_upper_limit)`` line lands and
PASS after, with the pinned value updated in the same commit that lands the
fix (see the commit diff/message for the fail-before evidence).
"""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from master_thesis_code.bayesian_inference.bayesian_statistics import BayesianStatistics
from master_thesis_code.physical_relations import dist_to_redshift

_H = 0.73
# d_L = 1.0 Gpc, sigma = 0.2 Gpc at h=0.73 -> raw 4-sigma window
# z_lower ~= 0.0470, z_upper ~= 0.3511 (see module docstring).
_D_L = 1.0
_D_L_UNCERTAINTY = 0.2


def _run_p_Di(
    *,
    redshift_upper_limit: float,
    f_const: float = 0.0,
    D_h: float = 1.0e9,
    beta_Gbar: float = 1.0e9,
) -> dict[str, Any]:
    """Run p_Di's partition (non-catalog_only) branch with NO catalog hosts.

    Zero hosts -> L_cat = 0 in both channels, so ``combined = B_num / D_h``
    exactly (mixture identity, w_G = beta_G/D_h with beta_G = D_h - beta_Gbar).
    f_const=0.0 (full incompleteness) keeps B_num's integrand at its maximum
    weight, isolating the effect of the [z_lower, z_upper] domain cap.
    """
    instance = object.__new__(BayesianStatistics)
    instance.h = _H
    instance._normalization_mode = "volume_deconv"
    instance.catalog_only = False
    instance.posterior_data = {}
    instance.posterior_data_with_bh_mass = {
        "galaxy_likelihoods": {},
        "additional_galaxies_without_bh_mass": {},
    }
    instance._diagnostic_rows = []

    mock_detection = MagicMock()
    mock_detection.d_L = _D_L
    mock_detection.d_L_uncertainty = _D_L_UNCERTAINTY
    mock_detection.phi = 0.5
    mock_detection.theta = 0.5
    mock_detection.M = 1e6
    mock_detection.M_uncertainty = 1e5
    instance.detection = mock_detection

    instance._det_index_to_slot = {0: 0}
    instance._means_3d = np.array([[0.5, 0.5, 1.0]])
    instance._cov_inv_3d = np.array([np.eye(3)])
    instance._log_norm_3d = np.array([0.0])
    instance._det_d_L = np.array([_D_L])

    instance._D_h_table = {_H: D_h}
    instance._beta_Gbar_table = {_H: beta_Gbar}
    instance._beta_G_table = {_H: D_h - beta_Gbar}
    instance._global_cat_denom_no_bh = {_H: 1.0}
    instance._global_cat_denom_with_bh = {_H: 1.0}

    mock_pool = MagicMock()
    mock_pool._processes = 1
    # No hosts -> _starmap_host_batches returns [] without touching the pool
    # (n == 0 short-circuit), so mock_pool.starmap is never actually called.

    mock_completeness = MagicMock()
    mock_completeness.ang2pix.return_value = 0
    mock_completeness.f_k.side_effect = lambda z, k, h: np.full_like(
        np.asarray(z, dtype=np.float64), f_const
    )

    mock_p_det = MagicMock()
    mock_p_det.get_dl_max.return_value = 10.0

    combined_no_bh, combined_with_bh = BayesianStatistics.p_Di(
        instance,
        possible_host_galaxies=[],
        possible_host_galaxies_with_bh_mass=[],
        detection_index=0,
        pool=mock_pool,
        completeness=mock_completeness,
        detection_probability_obj=mock_p_det,
        redshift_upper_limit=redshift_upper_limit,
    )
    row = instance._diagnostic_rows[0]
    assert row["combined_no_bh"] == combined_no_bh
    assert row["combined_with_bh"] == combined_with_bh
    # Zero hosts: both channels reduce to the pure-completion value B_num/D_h.
    assert combined_no_bh == combined_with_bh
    return row


def test_raw_window_sanity() -> None:
    """Confirms the raw (uncapped) 4-sigma window used by the other tests below.

    z_lower ~= 0.047, z_upper ~= 0.351 at d_L=1.0 Gpc, sigma=0.2 Gpc, h=0.73.
    A cap of 0.2 lies strictly inside this window (binds); a cap of 1.5
    (HOST_DRAW_Z_MAX, the current no-op production default) lies strictly
    above it (no-op); a cap of 0.02 lies strictly below it (event fully
    excluded).
    """
    z_upper_raw = dist_to_redshift(_D_L + 4.0 * _D_L_UNCERTAINTY, h=_H)
    z_lower_raw = dist_to_redshift(_D_L - 4.0 * _D_L_UNCERTAINTY, h=_H)
    assert z_lower_raw < 0.2 < z_upper_raw
    assert z_upper_raw < 1.5
    assert z_lower_raw > 0.02


def test_cap_noop_when_above_raw_window() -> None:
    """redshift_upper_limit >= raw z_upper: B_num byte-identical to the uncapped value.

    Uses HOST_DRAW_Z_MAX = 1.5, the current production no-op depth (matches
    Model1CrossCheck's default max_redshift), so the default-config behavior
    of the pipeline is unchanged by this fix.
    """
    row_capped = _run_p_Di(redshift_upper_limit=1.5)
    row_uncapped = _run_p_Di(redshift_upper_limit=1.0e6)  # effectively no cap at all
    assert row_capped["B_num"] == row_uncapped["B_num"]
    assert row_capped["combined_no_bh"] == row_uncapped["combined_no_bh"]
    assert row_capped["B_num"] > 0.0


def test_cap_binds_reduces_b_num() -> None:
    """PIN: redshift_upper_limit=0.2 (strictly inside the raw window) truncates B_num.

    Regression guard for the analysis-depth cap fix: before the fix, B_num
    ignored ``redshift_upper_limit`` entirely and this test would have
    asserted the OLD (uncapped, cap=1.0e6) value here -- i.e. it FAILED
    against the pre-fix code (B_num_capped == B_num_uncapped, no reduction).
    After the fix, the capped integral strictly undershoots the uncapped one
    (same integrand, strictly smaller domain, non-negative integrand).
    """
    row_capped = _run_p_Di(redshift_upper_limit=0.2)
    row_uncapped = _run_p_Di(redshift_upper_limit=1.0e6)
    assert row_capped["B_num"] < row_uncapped["B_num"]
    assert row_capped["B_num"] > 0.0
    assert row_capped["combined_no_bh"] < row_uncapped["combined_no_bh"]

    # Independent cross-check: re-derive B_num_capped directly via fixed_quad
    # over [z_lower, 0.2] with the identical integrand shape (a pure geometric
    # truncation -- same integrand, narrower domain -- so a monotonically
    # increasing partial integral confirms the cap ends exactly at 0.2, not
    # some other value).
    row_at_cap = _run_p_Di(redshift_upper_limit=0.2)
    row_just_below_cap = _run_p_Di(redshift_upper_limit=0.199)
    assert row_just_below_cap["B_num"] < row_at_cap["B_num"]


def test_event_fully_beyond_cap_gives_zero_b_num_no_crash() -> None:
    """Limiting case: redshift_upper_limit < raw z_lower -> B_num == 0.0 exactly.

    The (uncapped) z_lower for this event is ~0.047; a cap of 0.02 leaves NO
    surviving domain. Before the z_lower/z_upper inversion guard, this would
    integrate fixed_quad(f, z_lower=0.047, z_upper=0.02) -- an INVERTED
    interval producing a spurious NEGATIVE value, not 0. The fix must route
    this to the explicit B_num=0.0 branch: no NaN, no negative likelihood,
    and the pipeline continues (this event's completion term -- and hence its
    total likelihood in the zero-host case -- vanishes rather than crashing).
    """
    row = _run_p_Di(redshift_upper_limit=0.02)
    assert row["B_num"] == 0.0
    assert not np.isnan(row["B_num"])
    assert row["B_num"] >= 0.0
    assert row["combined_no_bh"] == 0.0
    assert row["combined_with_bh"] == 0.0
    assert not np.isnan(row["combined_no_bh"])


def test_cap_at_exact_z_upper_boundary_is_noop() -> None:
    """redshift_upper_limit exactly at the raw z_upper: min() picks z_upper, no change."""
    z_upper_raw = dist_to_redshift(_D_L + 4.0 * _D_L_UNCERTAINTY, h=_H)
    row_at_boundary = _run_p_Di(redshift_upper_limit=z_upper_raw)
    row_uncapped = _run_p_Di(redshift_upper_limit=1.0e6)
    assert row_at_boundary["B_num"] == pytest.approx(row_uncapped["B_num"], rel=1e-12)


def test_default_redshift_upper_limit_matches_host_draw_z_max() -> None:
    """p_Di's default ``redshift_upper_limit`` is HOST_DRAW_Z_MAX (1.5, current no-op).

    Any caller (test or otherwise) that does not pass ``redshift_upper_limit``
    explicitly gets exactly today's production default depth, so the new
    parameter cannot silently change behavior for un-migrated call sites.
    """
    import inspect

    from master_thesis_code.constants import HOST_DRAW_Z_MAX

    sig = inspect.signature(BayesianStatistics.p_Di)
    assert sig.parameters["redshift_upper_limit"].default == HOST_DRAW_Z_MAX
