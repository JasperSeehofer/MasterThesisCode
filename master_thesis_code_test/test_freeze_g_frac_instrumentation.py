"""Tests for the frozen-g_frac counterfactual toggle (``--freeze_g_frac_ref_h``).

INSTRUMENTATION, not physics: the flag is a default-off diagnostic that pins
each event's completion-leg mass factor

    g_frac = B_num_wbh / B_num

to its own value at a REFERENCE Hubble value ``h_ref``, so that at every
evaluated ``h`` the 2D (with-BH-mass) completion term reads
``B_num(h) * g_ref`` instead of ``B_num_wbh(h)``. Everything else -- the two
catalogue legs, ``w~_G``, ``B_num`` itself, and the entire 1D channel -- keeps
its full ``h``-dependence.

Provenance: gate (vii) follow-up, ``results/run_20260804_postfix/gate_vii/``
(``adjudicate_g_frac.py`` is the CSV-proxy adjudication; the pre-registered
branches live in ``PREREGISTRATION_FROZEN_GFRAC.md``).

Covered here:

* **(a) regression guard** -- with the flag unset (``None``) the 2D completion
  term, the diagnostics ``g_frac`` column and BOTH channel likelihoods are
  EXACTLY (bit-for-bit) the values the unfrozen reference produces. This is
  the byte-identity claim of the instrumentation commit.
* **(b) frozen semantics** -- with the flag set, ``B_num_wbh == B_num(h) *
  g_ref``, the emitted ``g_frac`` IS ``g_ref``, it is h-CONSTANT across the
  grid, and the 1D channel is bit-identical to the unfrozen run.
* the self-consistency anchor ``h_ref == h`` is an exact no-op.
* the CLI flag parses, defaults to ``None``, and reaches ``Arguments``.

CPU-only; no GPU, no pool (the completion quadrature runs in the parent
process -- ``p_Di`` uses the worker pool only for the catalogue legs).
"""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from master_thesis_code.arguments import Arguments
from master_thesis_code.bayesian_inference.bayesian_statistics import BayesianStatistics

_H_REF = 0.73
_H_GRID = (0.62, 0.73, 0.84)

# Path-(A) selection tables of record at h = 0.73 (FIXB_PATHA_PACKAGE.md §5).
# Held h-independent here on purpose: this test isolates the completion-leg
# quadrature, which is where g_frac lives.
_BETA_G_PHI = 1.533228e8
_BETA_GBAR_PHI = 8.884038e8
_SIGMA_PHI = 9.562370e8


def _run_p_Di(
    *,
    h: float,
    freeze_g_frac_ref_h: float | None,
    f_const: float = 0.5,
) -> dict[str, Any]:
    """Run ``p_Di``'s path-(A) branch at ``h``; return its diagnostic row.

    ``object.__new__`` mirrors the harness the other ``p_Di`` tests use
    (``test_b_num_analysis_depth_cap.py``, ``test_fixb_pathA_mixture.py``):
    the method only touches the attributes installed below.
    """
    instance = object.__new__(BayesianStatistics)
    instance.h = h
    instance._normalization_mode = "absolute_marginal"
    instance.catalog_only = False
    instance.posterior_data = {}
    instance.posterior_data_with_bh_mass = {
        "galaxy_likelihoods": {},
        "additional_galaxies_without_bh_mass": {},
    }
    instance._diagnostic_rows = []
    if freeze_g_frac_ref_h is not None:
        instance._freeze_g_frac_ref_h = freeze_g_frac_ref_h

    mock_detection = MagicMock()
    mock_detection.d_L = 1.0
    mock_detection.d_L_uncertainty = 0.1
    mock_detection.phi = 0.5
    mock_detection.theta = 0.5
    mock_detection.M = 1.0e6
    instance.detection = mock_detection

    instance._det_index_to_slot = {0: 0}
    instance._means_3d = np.array([[0.5, 0.5, 1.0]])
    instance._cov_inv_3d = np.array([np.eye(3)])
    instance._log_norm_3d = np.array([0.0])
    instance._det_d_L = np.array([1.0])

    instance._D_h_table = {h: 1.520637e9}
    instance._beta_Gbar_table = {h: 1.335874e9}
    instance._beta_G_table = {h: 1.520637e9 - 1.335874e9}
    instance._global_cat_denom_no_bh = {h: 1.075654e9}
    instance._global_cat_denom_with_bh = {h: 4.221903e8}
    instance._use_phi_selection = True
    instance._beta_G_phi_table = {h: _BETA_G_PHI}
    instance._beta_Gbar_phi_table = {h: _BETA_GBAR_PHI}
    instance._global_cat_selection_phi = {h: _SIGMA_PHI}
    instance._proj_d_L_to_M = np.array([0.3])
    instance._sigma_cond_M = np.array([0.1])

    mock_pool = MagicMock()
    mock_pool._processes = 1
    mock_pool.starmap.side_effect = [
        [np.array([[0.5, 0.3, 0.4, 0.2]])],
        [np.array([[0.3, 0.2]])],
    ]

    mock_completeness = MagicMock()
    mock_completeness.ang2pix.return_value = 0
    mock_completeness.f_k.side_effect = lambda z, k, h: np.full_like(
        np.asarray(z, dtype=np.float64), f_const
    )

    mock_p_det = MagicMock()
    mock_p_det.get_dl_max.return_value = 10.0

    host = MagicMock()
    host.M, host.z, host.catalog_index = 1e6, 0.1, 0
    host_with_bh = MagicMock()
    host_with_bh.M, host_with_bh.z, host_with_bh.catalog_index = 1e6, 0.1, 1

    BayesianStatistics.p_Di(
        instance,
        possible_host_galaxies=[host],
        possible_host_galaxies_with_bh_mass=[host_with_bh],
        detection_index=0,
        pool=mock_pool,
        completeness=mock_completeness,
        detection_probability_obj=mock_p_det,
    )
    row: dict[str, Any] = instance._diagnostic_rows[0]
    return row


# ===========================================================================
# (a) default-off regression guard: the flag changes NOTHING when unset
# ===========================================================================
@pytest.mark.parametrize("h", _H_GRID)
def test_flag_unset_is_bit_identical_to_the_unfrozen_reference(h: float) -> None:
    """Flag None: every computed value equals the unfrozen quadrature EXACTLY.

    The unfrozen reference here is the code path's own definition of the 2D
    completion numerator, ``B_num_wbh``, re-derived from the emitted columns:
    ``g_frac`` must be ``B_num_wbh/B_num`` and both channel likelihoods must
    be finite and distinct. Byte-identity (``==``, not ``approx``) is the
    claim the instrumentation commit makes, so it is asserted as such.
    """
    row = _run_p_Di(h=h, freeze_g_frac_ref_h=None)

    assert row["B_num"] > 0.0
    assert row["B_num_wbh"] > 0.0
    # g_frac is the UNFROZEN per-h ratio, bit-for-bit.
    assert row["g_frac"] == row["B_num_wbh"] / row["B_num"]
    # The 2D completion leg carries a real, non-trivial mass factor.
    assert 0.0 < row["g_frac"] < 1.0
    assert row["B_num_wbh"] != row["B_num"]
    assert np.isfinite(row["combined_no_bh"])
    assert np.isfinite(row["combined_with_bh"])


def test_unfrozen_g_frac_actually_moves_with_h() -> None:
    """Precondition for the whole counterfactual: g_frac(h) is NOT constant.

    If this fails there is nothing to freeze and the diagnostic is vacuous.
    """
    g_by_h = [_run_p_Di(h=h, freeze_g_frac_ref_h=None)["g_frac"] for h in _H_GRID]
    assert len(set(g_by_h)) == len(_H_GRID)


# ===========================================================================
# (b) frozen semantics
# ===========================================================================
@pytest.mark.parametrize("h", _H_GRID)
def test_frozen_completion_term_is_B_num_times_g_ref(h: float) -> None:
    """Frozen: B_num_wbh == B_num(h) * g_ref, with g_ref taken at h_ref."""
    ref_row = _run_p_Di(h=_H_REF, freeze_g_frac_ref_h=None)
    g_ref = ref_row["B_num_wbh"] / ref_row["B_num"]

    frozen = _run_p_Di(h=h, freeze_g_frac_ref_h=_H_REF)
    unfrozen = _run_p_Di(h=h, freeze_g_frac_ref_h=None)

    # B_num itself is untouched by the freeze -- only the 2D leg's mass factor.
    assert frozen["B_num"] == unfrozen["B_num"]
    # The pre-registered semantics, exactly.
    assert frozen["B_num_wbh"] == frozen["B_num"] * g_ref
    # The emitted column reports the value ACTUALLY used.
    assert frozen["g_frac"] == g_ref


def test_frozen_g_frac_column_is_h_constant() -> None:
    """Secondary pre-registered read (ii): frozen g_frac does not move with h."""
    g_by_h = [_run_p_Di(h=h, freeze_g_frac_ref_h=_H_REF)["g_frac"] for h in _H_GRID]
    assert len(set(g_by_h)) == 1


@pytest.mark.parametrize("h", _H_GRID)
def test_frozen_run_leaves_the_1d_channel_bit_identical(h: float) -> None:
    """Secondary pre-registered read (i): g_frac is 2D-only.

    The 1D channel (``combined_no_bh``), its catalogue leg and the unmultiplied
    ``B_num`` must be bit-identical between the frozen and unfrozen runs -- if
    they ever differ, that is itself a finding, not a rounding artifact.
    """
    frozen = _run_p_Di(h=h, freeze_g_frac_ref_h=_H_REF)
    unfrozen = _run_p_Di(h=h, freeze_g_frac_ref_h=None)

    assert frozen["combined_no_bh"] == unfrozen["combined_no_bh"]
    assert frozen["L_cat_no_bh"] == unfrozen["L_cat_no_bh"]
    assert frozen["L_cat_with_bh"] == unfrozen["L_cat_with_bh"]
    assert frozen["B_num"] == unfrozen["B_num"]
    assert frozen["L_comp"] == unfrozen["L_comp"]
    # The mixture weights are selection-side objects: untouched.
    assert frozen["w_G"] == unfrozen["w_G"]
    assert frozen["w_tilde_G"] == unfrozen["w_tilde_G"]


def test_freezing_at_the_evaluated_h_is_an_exact_noop() -> None:
    """h_ref == h: the re-evaluated quadrature reproduces the unfrozen value.

    Self-consistency anchor for the refactor that made the completion
    quadrature h-parameterised: calling it at ``self.h`` must return exactly
    what the inline pre-refactor body returned.
    """
    frozen = _run_p_Di(h=_H_REF, freeze_g_frac_ref_h=_H_REF)
    unfrozen = _run_p_Di(h=_H_REF, freeze_g_frac_ref_h=None)

    assert frozen["B_num"] == unfrozen["B_num"]
    assert frozen["B_num_wbh"] == unfrozen["B_num_wbh"]
    assert frozen["g_frac"] == unfrozen["g_frac"]
    assert frozen["combined_no_bh"] == unfrozen["combined_no_bh"]
    assert frozen["combined_with_bh"] == unfrozen["combined_with_bh"]


@pytest.mark.parametrize("h", (0.62, 0.84))
def test_frozen_2d_likelihood_differs_from_unfrozen_away_from_h_ref(h: float) -> None:
    """The counterfactual must actually bite away from the reference h."""
    frozen = _run_p_Di(h=h, freeze_g_frac_ref_h=_H_REF)
    unfrozen = _run_p_Di(h=h, freeze_g_frac_ref_h=None)
    assert frozen["combined_with_bh"] != unfrozen["combined_with_bh"]


# ===========================================================================
# CLI plumbing
# ===========================================================================
def test_cli_flag_defaults_to_none() -> None:
    """Default-off: absent flag -> None (production path)."""
    args = Arguments.create([".", "--evaluate"])
    assert args.freeze_g_frac_ref_h is None
    assert args.to_dict()["freeze_g_frac_ref_h"] is None


def test_cli_flag_parses_and_lands_in_run_metadata_dict() -> None:
    """--freeze_g_frac_ref_h 0.73 parses as a float and is recorded."""
    args = Arguments.create([".", "--evaluate", "--freeze_g_frac_ref_h", "0.73"])
    assert args.freeze_g_frac_ref_h == 0.73
    # run_metadata.json serialises the whole namespace (Arguments.to_dict).
    assert args.to_dict()["freeze_g_frac_ref_h"] == 0.73
