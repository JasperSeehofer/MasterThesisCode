"""Tests for the N-2 counterfactual toggle (``--selection_in_completion_numerator``).

INSTRUMENTATION, not physics: the flag is a default-off diagnostic. Under the
``"1d"`` cell the **1D** completion-leg numerator becomes

    B_num^{1d}(h) = INTEGRAL (1-f_k) p_gw dVc/(1+z) S_bar_phi(z;h) dz

with ``S_bar_phi`` READ from the table
:func:`~darksiren_emri.bayesian_inference.bayesian_statistics.precompute_phi_marginal_survival`
already builds for ``beta^phi``/``D~^phi`` — the same object, the same
``np.interp`` accessor ``precompute_global_catalog_selection`` uses for
``Sigma^phi``. Everything else (the 2D with-BH-mass leg, both catalogue legs,
``w~_G``, ``r_Malm``, ``D~^phi``) is the production object.

Provenance: ``.planning/derivation-gfrac-20260805/``
(``N2_SELECTION_NUMERATOR_DERIVATION_20260805.md.DRAFT`` (T3'), measurements M1
and M2); pre-registered branches in
``results/run_20260804_postfix/gate_vii/PREREGISTRATION_N2_SEL1D.md``.

Covered here:

* **(a) default-off byte-identity** -- with the flag ``"off"`` every emitted
  column is EXACTLY (``==``) what the pre-flag code path produces.
* **(b) the '1d' cell multiplies the completion integrand by S_bar_phi** --
  verified against an independent hand Gauss-Legendre quadrature on the
  captured integrand, and against exact scaling for a constant table.
* **(c) the 2D channel is bit-identical under '1d'** (measurement M2 deleted
  the 'both' arm; the 2D leg must not move at all).
* CLI parsing, choices, default, and run_metadata capture.

CPU-only; no GPU, no pool (the completion quadrature runs in the parent
process -- ``p_Di`` uses the worker pool only for the catalogue legs).
"""

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest

import darksiren_emri.bayesian_inference.bayesian_statistics as bs
from darksiren_emri.arguments import Arguments
from darksiren_emri.bayesian_inference.bayesian_statistics import BayesianStatistics

_H_GRID = (0.62, 0.73, 0.84)

# Path-(A) selection tables of record at h = 0.73 (FIXB_PATHA_PACKAGE.md §5).
# Held h-independent here on purpose: this test isolates the completion-leg
# quadrature, which is where the N-2 factor enters.
_BETA_G_PHI = 1.533228e8
_BETA_GBAR_PHI = 8.884038e8
_SIGMA_PHI = 9.562370e8

# A z-grid that comfortably covers the mock event's 4-sigma window
# (d_L = 1.0 +/- 0.1 Gpc => z <~ 0.35 at these h).
_Z_TABLE = np.linspace(1e-6, 1.0, 2001, dtype=np.float64)


def _s_phi_affine(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """A non-trivial, strictly positive, z-varying stand-in for S_bar_phi."""
    return np.asarray(0.20 + 0.55 * z, dtype=np.float64)


def _s_phi_const(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.full_like(np.asarray(z, dtype=np.float64), 0.375)


def _run_p_Di(
    *,
    h: float,
    selection_cell: str | None,
    s_phi_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] = _s_phi_affine,
    f_const: float = 0.5,
    table_h: float | None = None,
    p_det_obj: Any | None = None,
    freeze_ref_h: float | None = None,
    extra_table_h: list[float] | None = None,
) -> dict[str, Any]:
    """Run ``p_Di``'s path-(A) branch at ``h``; return its diagnostic row.

    ``object.__new__`` mirrors the harness the other ``p_Di`` tests use
    (``test_freeze_g_frac_instrumentation.py``, ``test_fixb_pathA_mixture.py``):
    the method only touches the attributes installed below. ``selection_cell``
    of ``None`` leaves the attribute unset entirely, which is how a legacy
    (pre-flag) instance looks to the ``getattr`` guard. ``table_h`` mis-keys the
    ``S_bar_phi`` table on purpose (missing-entry guard).
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
    if selection_cell is not None:
        instance._selection_in_completion_numerator = selection_cell
    if freeze_ref_h is not None:
        instance._freeze_g_frac_ref_h = freeze_ref_h
    _key = h if table_h is None else table_h
    _keys = [_key, *(extra_table_h or [])]
    instance._phi_survival_table = {k: (_Z_TABLE, s_phi_fn(_Z_TABLE)) for k in _keys}

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

    if p_det_obj is not None:
        mock_p_det = p_det_obj
    else:
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
# (a) default-off byte-identity
# ===========================================================================
@pytest.mark.parametrize("h", _H_GRID)
def test_flag_off_is_bit_identical_to_the_unflagged_instance(h: float) -> None:
    """``"off"`` and "attribute absent" produce EXACTLY the same row.

    "Attribute absent" is the pre-flag code path as seen through the
    ``getattr(..., "off")`` guard, so this is the byte-identity claim of the
    instrumentation commit, asserted with ``==`` rather than ``approx``.
    """
    legacy = _run_p_Di(h=h, selection_cell=None)
    off = _run_p_Di(h=h, selection_cell="off")

    assert off.keys() == legacy.keys()
    for key in legacy:
        assert off[key] == legacy[key], key


@pytest.mark.parametrize("h", _H_GRID)
def test_off_path_ignores_the_survival_table_entirely(h: float) -> None:
    """Default-off must not read S_bar_phi at all: changing it changes nothing."""
    a = _run_p_Di(h=h, selection_cell="off", s_phi_fn=_s_phi_affine)
    b = _run_p_Di(h=h, selection_cell="off", s_phi_fn=_s_phi_const)
    for key in a:
        assert a[key] == b[key], key


# ===========================================================================
# (b) the '1d' cell multiplies the completion integrand by S_bar_phi
# ===========================================================================
@pytest.mark.parametrize("h", _H_GRID)
def test_sel_1d_matches_a_hand_gauss_legendre_quadrature(
    h: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B_num under '1d' == hand quadrature of (production integrand x S_bar_phi).

    The production 1D integrand is captured by wrapping ``fixed_quad`` in the
    module namespace (the wrapper still delegates to the real routine, so the
    run itself is unchanged). The hand check then re-does the SAME fixed-order
    Gauss-Legendre rule with ``np.polynomial.legendre.leggauss`` and multiplies
    by the table's ``S_bar_phi`` at the nodes -- an independent evaluation of
    (T3'), not a re-call of the code under test.
    """
    captured: list[tuple[Callable[..., Any], float, float, int]] = []
    real_fixed_quad = bs.fixed_quad

    def _spy(func: Callable[..., Any], a: float, b: float, n: int) -> Any:
        captured.append((func, float(a), float(b), int(n)))
        return real_fixed_quad(func, a, b, n=n)

    monkeypatch.setattr(bs, "fixed_quad", _spy)

    off = _run_p_Di(h=h, selection_cell="off")
    # The FIRST fixed_quad call of the off-run is the 1D completion numerator.
    integrand_1d, z_lower, z_upper, n_quad = captured[0]
    captured.clear()

    sel = _run_p_Di(h=h, selection_cell="1d")

    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    half = 0.5 * (z_upper - z_lower)
    mid = 0.5 * (z_upper + z_lower)
    z_nodes = half * nodes + mid
    base = np.asarray(integrand_1d(z_nodes), dtype=np.float64)
    s_phi = np.interp(z_nodes, _Z_TABLE, _s_phi_affine(_Z_TABLE))
    hand = float(half * np.dot(weights, base * s_phi))

    assert sel["B_num"] == pytest.approx(hand, rel=1e-12, abs=0.0)
    # ... and it is genuinely different from the production value.
    assert sel["B_num"] != off["B_num"]
    # S_bar_phi < 1 everywhere on this window => the numerator shrinks.
    assert 0.0 < sel["B_num"] < off["B_num"]


@pytest.mark.parametrize("h", _H_GRID)
def test_sel_1d_with_a_constant_table_is_exact_scaling(h: float) -> None:
    """S_bar_phi = c (constant) => B_num^{1d} = c * B_num, a closed-form check."""
    off = _run_p_Di(h=h, selection_cell="off", s_phi_fn=_s_phi_const)
    sel = _run_p_Di(h=h, selection_cell="1d", s_phi_fn=_s_phi_const)
    assert sel["B_num"] == pytest.approx(0.375 * off["B_num"], rel=1e-13, abs=0.0)


def test_sel_1d_tilts_the_1d_channel_in_h() -> None:
    """The correction must carry an h-slope -- otherwise the run is vacuous.

    (T3') attributes the whole slope to the ``(1+z)`` mass lift; here the
    stand-in table rises with z, so the ratio to the production value must
    move with h rather than sit at a constant.
    """
    ratios = [
        _run_p_Di(h=h, selection_cell="1d")["B_num"] / _run_p_Di(h=h, selection_cell="off")["B_num"]
        for h in _H_GRID
    ]
    assert len(set(ratios)) == len(_H_GRID)


def test_sel_1d_raises_when_the_table_lacks_the_evaluated_h() -> None:
    """A missing S_bar_phi entry is a hard error, never a silent fallback."""
    with pytest.raises(ValueError, match="no S_bar_phi table entry"):
        _run_p_Di(h=0.73, selection_cell="1d", table_h=0.99)


def test_evaluate_rejects_an_unknown_cell() -> None:
    """``BayesianStatistics.evaluate`` validates the cell name itself."""
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(ValueError, match="must be 'auto', 'off', '1d', '2d' or 'fused'"):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            selection_in_completion_numerator="both",
        )


def test_evaluate_requires_absolute_marginal_for_the_1d_cell() -> None:
    """The S_bar_phi table only exists in ``absolute_marginal``; say so loudly."""
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(ValueError, match="absolute_marginal"):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            normalization_mode="generator_marginal",
            selection_in_completion_numerator="1d",
        )


# ===========================================================================
# (c) the 2D channel is bit-identical under '1d' (measurement M2)
# ===========================================================================
@pytest.mark.parametrize("h", _H_GRID)
def test_sel_1d_leaves_the_2d_completion_leg_bit_identical(h: float) -> None:
    """M2 deleted the 'both' arm: the 2D leg must not move by a single ulp.

    ``B_num_wbh`` is its OWN quadrature over the unmultiplied base integrand,
    and the 2D mixture reads ``B_num_wbh``, never ``B_num`` -- so the whole 2D
    channel, including its catalogue leg and the mixture weights, is untouched.
    """
    off = _run_p_Di(h=h, selection_cell="off")
    sel = _run_p_Di(h=h, selection_cell="1d")

    assert sel["B_num_wbh"] == off["B_num_wbh"]
    assert sel["combined_with_bh"] == off["combined_with_bh"]
    assert sel["L_cat_with_bh"] == off["L_cat_with_bh"]
    assert sel["L_cat_no_bh"] == off["L_cat_no_bh"]
    # Selection-side objects (P-9 expected NULLs): the toggle must not leak
    # into the normalisation.
    assert sel["w_G"] == off["w_G"]
    assert sel["w_tilde_G"] == off["w_tilde_G"]
    assert sel["alpha_G_phi"] == off["alpha_G_phi"]
    assert sel["r_Malm"] == off["r_Malm"]
    assert sel["D_tilde_phi"] == off["D_tilde_phi"]
    # ... while the 1D channel DOES move (that is the whole point).
    assert sel["combined_no_bh"] != off["combined_no_bh"]


@pytest.mark.parametrize("h", _H_GRID)
def test_diagnostics_schema_is_unchanged(h: float) -> None:
    """The CSV schema is a contract: '1d' adds and removes no column."""
    off = _run_p_Di(h=h, selection_cell="off")
    sel = _run_p_Di(h=h, selection_cell="1d")
    assert list(sel.keys()) == list(off.keys())


# ===========================================================================
# CLI plumbing
# ===========================================================================
def test_cli_flag_defaults_to_auto() -> None:
    """Default 'auto' ([PHYSICS] rows #117-#118): resolves per normalization
    mode inside ``evaluate`` ('fused' under absolute_marginal, 'off' otherwise).
    """
    args = Arguments.create([".", "--evaluate"])
    assert args.selection_in_completion_numerator == "auto"
    assert args.to_dict()["selection_in_completion_numerator"] == "auto"


def test_cli_flag_parses_and_lands_in_run_metadata_dict() -> None:
    """--selection_in_completion_numerator 1d parses and is recorded."""
    args = Arguments.create([".", "--evaluate", "--selection_in_completion_numerator", "1d"])
    assert args.selection_in_completion_numerator == "1d"
    # run_metadata.json serialises the whole namespace (Arguments.to_dict).
    assert args.to_dict()["selection_in_completion_numerator"] == "1d"


def test_cli_flag_rejects_unknown_cells() -> None:
    """The deleted 'both' arm (measurement M2) is not a selectable cell."""
    with pytest.raises(SystemExit):
        Arguments.create([".", "--evaluate", "--selection_in_completion_numerator", "both"])


def test_cli_flag_accepts_the_fusion_cells() -> None:
    """'2d' and 'fused' are selectable since rows #117-#118."""
    for cell in ("2d", "fused"):
        args = Arguments.create([".", "--evaluate", "--selection_in_completion_numerator", cell])
        assert args.selection_in_completion_numerator == cell
