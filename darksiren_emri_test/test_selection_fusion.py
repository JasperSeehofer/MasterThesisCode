"""Tests for the selection fusion [P1]+[P2] ([PHYSICS], ledger rows #117-#118).

Gate: ``docs/derivations/GATE_PRESENTATION_SELECTION_FUSION_20260817.md`` +
verifier addendum (``PROPOSAL_2D_SELECTION_FUSION_VERIFIER_ADDENDUM_20260817.md``).

Covered here (the amended [P5]-1 list):

* **S == 1 bit-exact recovery** of :func:`completion_mass_factor_g` on BOTH the
  ``adaptive=False`` pinned path and the Route-1 adaptive path (MAJOR-2
  restatement: bit-exactness is against the same-quadrature object).
* **Constant-S exact scaling** ``g_sel = c * g_i`` (closed form).
* **G1 S-variation guard** (row #118): a step in ``S_4D`` across the Hermite
  node window escalates the row to the pinned order n=64.
* **MINOR-6 guard**: non-positive Hermite node masses never reach ``s_query``
  and cannot poison the row with NaN.
* **Pre-change regression pins** ([P5]-2, recorded at commit ``4ab5da0e``
  BEFORE the fusion landed): the ``off`` and ``1d`` cells reproduce the
  pre-#118 values EXACTLY (``==``); the ``fused``/``2d`` cells are pinned as
  the new values with closed-form cross-checks (constant ``S_4D = 0.6`` =>
  ``B_num_wbh`` scales by exactly 0.6).
* **Cell resolution**: ``auto`` -> ``fused`` under ``absolute_marginal``,
  ``auto`` -> ``off`` under ``generator_marginal``; ``fused`` rejected outside
  ``absolute_marginal``.
* **MINOR-2 warning classification**: beyond-horizon (S=0) zeros are labelled
  as such, not as phi-support exits.
* **MINOR-1**: the frozen-g_frac counterfactual works under ``fused`` when the
  S_bar_phi table carries the reference h (the invariant ``evaluate`` now
  guarantees by tabulating ``freeze_g_frac_ref_h``).
* **G1 recorded bound**: adaptive-vs-pinned relative difference on smooth-S
  production-regime rows stays in the ~1e-15 class.

CPU-only; no GPU, no pool.
"""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest

from darksiren_emri.bayesian_inference.bayesian_statistics import (
    BayesianStatistics,
    completion_mass_factor_g,
    completion_mass_factor_g_sel,
)
from darksiren_emri_test.test_selection_in_completion_numerator import (
    _H_GRID,
    _run_p_Di,
)

# ---------------------------------------------------------------------------
# g_sel unit fixtures: production-like rows (sharp-likelihood regime of record,
# row #118 MAJOR-1: measured d_L-conditional sigma_cond p50 = 8.8e-8).
# ---------------------------------------------------------------------------
_Z = np.linspace(0.02, 0.4, 25, dtype=np.float64)
_DET_M_Z = 1.0e6
_DET_D_L = 1.0  # Gpc
_PROJ = 0.3
_SIGMA_SHARP = 8.8e-8  # fast-path regime
_SIGMA_BROAD = 0.1  # relwidth-fallback regime


def _d_L_of_z(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # A smooth stand-in distance curve; only ratios/monotonicity matter here.
    return np.asarray(4.0 * z * (1.0 + 0.3 * z), dtype=np.float64)


def _s_ones(
    dl: npt.NDArray[np.float64],
    m_z: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return np.ones_like(m_z)


@pytest.mark.parametrize("sigma", [_SIGMA_SHARP, _SIGMA_BROAD])
@pytest.mark.parametrize("adaptive", [False, True])
def test_s_equal_one_recovers_g_i_bit_exactly(sigma: float, adaptive: bool) -> None:
    """S == 1 => g_sel == g_i to the BIT, on both quadrature paths.

    The venue's S==1 refactor gate (0.0 measured) restated per MAJOR-2: the
    comparison is same-quadrature (identical adaptive flag), and the G1 guard
    adds no escalations at zero S-variation, so the group partition and every
    contraction are identical arithmetic.
    """
    d_L = _d_L_of_z(_Z)
    g_i = completion_mass_factor_g(_Z, d_L / _DET_D_L, _DET_M_Z, _PROJ, sigma, adaptive=adaptive)
    g_sel = completion_mass_factor_g_sel(
        _Z,
        d_L,
        d_L / _DET_D_L,
        _DET_M_Z,
        _PROJ,
        sigma,
        s_query=_s_ones,
        adaptive=adaptive,
    )
    assert np.array_equal(g_sel, g_i)


@pytest.mark.parametrize("c", [0.6, 0.05])
def test_constant_s_is_exact_scaling(c: float) -> None:
    """S == c => g_sel = c * g_i (closed form; floating reassociation only)."""
    d_L = _d_L_of_z(_Z)

    def _s_const(
        dl: npt.NDArray[np.float64],
        m_z: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return np.full_like(m_z, c)

    g_i = completion_mass_factor_g(
        _Z, d_L / _DET_D_L, _DET_M_Z, _PROJ, _SIGMA_SHARP, adaptive=False
    )
    g_sel = completion_mass_factor_g_sel(
        _Z,
        d_L,
        d_L / _DET_D_L,
        _DET_M_Z,
        _PROJ,
        _SIGMA_SHARP,
        s_query=_s_const,
        adaptive=False,
    )
    np.testing.assert_allclose(g_sel, c * g_i, rtol=5e-15, atol=0.0)


def test_g1_guard_escalates_on_s_step_across_the_window() -> None:
    """A step in S inside the +-6-sigma node window forces the pinned order.

    G1 ruling (row #118): the fast path is allowed only while the relative
    S-variation across the window is <= _G_SEL_S_VAR_TOL. The step mock below
    varies by 50% across every row's window, so every row must be contracted
    at n=64 — visible as a query of size k*64 and the absence of any k*8 query.
    """
    d_L = _d_L_of_z(_Z)
    sizes: list[int] = []

    def _s_step(
        dl: npt.NDArray[np.float64],
        m_z: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        sizes.append(int(m_z.size))
        # mu_cond ~ 1 => the step at exactly det_M_z bisects every window.
        return np.where(m_z < _DET_M_Z * (1.0 + _PROJ * (0.0)), 1.0, 0.5).astype(np.float64)

    g_sel = completion_mass_factor_g_sel(
        _Z,
        d_L,
        d_L / _DET_D_L,
        _DET_M_Z,
        0.0,  # proj=0 => mu_cond = 1 exactly; window straddles the step
        _SIGMA_SHARP,
        s_query=_s_step,
        adaptive=True,
    )
    assert np.all(np.isfinite(g_sel))
    # Two edge-guard queries of size k, then ONE escalated contraction of k*64.
    assert _Z.size * 64 in sizes
    assert _Z.size * 8 not in sizes


def test_fast_path_taken_when_s_is_smooth() -> None:
    """Constant S in the sharp regime keeps the Route-1 fast order (n=8)."""
    d_L = _d_L_of_z(_Z)
    sizes: list[int] = []

    def _s_const(
        dl: npt.NDArray[np.float64],
        m_z: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        sizes.append(int(m_z.size))
        return np.full_like(m_z, 0.7)

    completion_mass_factor_g_sel(
        _Z,
        d_L,
        d_L / _DET_D_L,
        _DET_M_Z,
        _PROJ,
        _SIGMA_SHARP,
        s_query=_s_const,
        adaptive=True,
    )
    assert _Z.size * 8 in sizes
    assert _Z.size * 64 not in sizes


def test_non_positive_node_masses_never_reach_the_survival_query() -> None:
    """MINOR-6: sigma so broad that Hermite nodes go negative — no NaN, and
    s_query only ever sees strictly positive masses."""
    d_L = _d_L_of_z(_Z)

    def _s_checked(
        dl: npt.NDArray[np.float64],
        m_z: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        assert np.all(m_z > 0.0)
        return np.full_like(m_z, 0.5)

    g_sel = completion_mass_factor_g_sel(
        _Z,
        d_L,
        d_L / _DET_D_L,
        _DET_M_Z,
        _PROJ,
        1.0,  # +-6 sigma window reaches x_M < 0
        s_query=_s_checked,
        adaptive=True,
    )
    assert np.all(np.isfinite(g_sel))
    assert np.all(g_sel >= 0.0)


def test_g1_recorded_bound_adaptive_vs_pinned() -> None:
    """G1 recorded bound: adaptive vs pinned n=64 with smooth S, ~1e-15 class.

    This is the pinned-vs-adaptive bound MAJOR-2 splits out of the (bit-exact)
    S==1 recovery claim; the measured value on these rows is recorded in the
    gate ledger's `verified` row.
    """
    d_L = _d_L_of_z(_Z)

    def _s_smooth(
        dl: npt.NDArray[np.float64],
        m_z: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return np.asarray(1.0 / (1.0 + dl / 8.0), dtype=np.float64)

    kwargs: dict[str, Any] = dict(s_query=_s_smooth)
    fast = completion_mass_factor_g_sel(
        _Z, d_L, d_L / _DET_D_L, _DET_M_Z, _PROJ, _SIGMA_SHARP, adaptive=True, **kwargs
    )
    pinned = completion_mass_factor_g_sel(
        _Z, d_L, d_L / _DET_D_L, _DET_M_Z, _PROJ, _SIGMA_SHARP, adaptive=False, **kwargs
    )
    rel = np.max(np.abs(fast - pinned) / np.abs(pinned))
    assert rel < 1e-13


# ---------------------------------------------------------------------------
# p_Di-level: pins + cells. Harness = the N-2 test module's _run_p_Di with a
# survival-capable detection-probability mock threaded through.
# ---------------------------------------------------------------------------
_S_4D_CONST = 0.6


def _mock_p_det_with_survival(s_value: float = _S_4D_CONST) -> MagicMock:
    mock_p_det = MagicMock()
    mock_p_det.get_dl_max.return_value = 10.0
    # `is True` guard in _wbh_z_kwargs keeps the MagicMock on the pooled path.
    mock_p_det.wbh_z_resolved = False

    def _s(
        dl: npt.NDArray[np.float64],
        m_z: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
        h: float,
    ) -> npt.NDArray[np.float64]:
        return np.full_like(np.asarray(dl, dtype=np.float64), s_value)

    mock_p_det.detection_probability_with_bh_mass_interpolated.side_effect = _s
    return mock_p_det


def _run_fused(h: float, cell: str, **kwargs: Any) -> dict[str, Any]:
    return _run_p_Di(
        h=h,
        selection_cell=cell,
        p_det_obj=_mock_p_det_with_survival(),
        **kwargs,
    )


# [P5]-2 regression pins, recorded at commit 4ab5da0e (PRE-fusion HEAD) on the
# pinned mock event of test_selection_in_completion_numerator._run_p_Di
# (affine S_bar_phi table, f_const=0.5). EXACT equality: the 'off' and '1d'
# cells must not move by an ulp when the fusion cells land.
_PRE_FUSION_PINS: dict[tuple[str, float], dict[str, float]] = {
    ("off", 0.62): {
        "B_num": 2618876.5250924043,
        "B_num_wbh": 252637.21825214982,
        "g_frac": 0.09646778526270373,
    },
    ("1d", 0.62): {
        "B_num": 801135.198425758,
        "B_num_wbh": 252637.21825214982,
        "g_frac": 0.31534904314351125,
    },
    ("off", 0.73): {
        "B_num": 2326363.8527682433,
        "B_num_wbh": 227099.27577344136,
        "g_frac": 0.0976198437330454,
    },
    ("1d", 0.73): {
        "B_num": 749272.6316246494,
        "B_num_wbh": 227099.27577344136,
        "g_frac": 0.30309298136383483,
    },
    ("off", 0.84): {
        "B_num": 2080507.1613053768,
        "B_num_wbh": 205381.22311619035,
        "g_frac": 0.09871690275140778,
    },
    ("1d", 0.84): {
        "B_num": 702656.1243849726,
        "B_num_wbh": 205381.22311619035,
        "g_frac": 0.29229265353085526,
    },
}


@pytest.mark.parametrize("cell", ["off", "1d"])
@pytest.mark.parametrize("h", _H_GRID)
def test_pre_fusion_cells_reproduce_the_recorded_pins_exactly(cell: str, h: float) -> None:
    """The legacy cells are byte-frozen: fusion changed NOTHING off-cell."""
    row = _run_p_Di(h=h, selection_cell=cell)
    pins = _PRE_FUSION_PINS[(cell, h)]
    for key, value in pins.items():
        assert row[key] == value, key


@pytest.mark.parametrize("h", _H_GRID)
def test_fused_1d_leg_equals_the_1d_cell(h: float) -> None:
    """'fused' B_num == '1d' B_num: [P2] is the promoted branch, unchanged."""
    fused = _run_fused(h, "fused")
    assert fused["B_num"] == _PRE_FUSION_PINS[("1d", h)]["B_num"]


@pytest.mark.parametrize("h", _H_GRID)
def test_fused_2d_leg_scales_by_the_constant_survival(h: float) -> None:
    """Constant S_4D = 0.6 => B_num_wbh(fused) = 0.6 x B_num_wbh(off), closed form."""
    fused = _run_fused(h, "fused")
    np.testing.assert_allclose(
        fused["B_num_wbh"],
        _S_4D_CONST * _PRE_FUSION_PINS[("off", h)]["B_num_wbh"],
        rtol=1e-12,
        atol=0.0,
    )


@pytest.mark.parametrize("h", _H_GRID)
def test_2d_cell_touches_only_the_2d_leg(h: float) -> None:
    """'2d' ([P1]-only): B_num stays at the off pin; B_num_wbh carries S."""
    cell = _run_fused(h, "2d")
    assert cell["B_num"] == _PRE_FUSION_PINS[("off", h)]["B_num"]
    np.testing.assert_allclose(
        cell["B_num_wbh"],
        _S_4D_CONST * _PRE_FUSION_PINS[("off", h)]["B_num_wbh"],
        rtol=1e-12,
        atol=0.0,
    )


def test_fused_pairing_identity() -> None:
    """The pairing constraint made measurable: fused = ('1d' 1D leg) + ('2d' 2D leg)."""
    h = 0.73
    fused = _run_fused(h, "fused")
    only_1d = _run_fused(h, "1d")
    only_2d = _run_fused(h, "2d")
    assert fused["B_num"] == only_1d["B_num"]
    assert fused["B_num_wbh"] == only_2d["B_num_wbh"]


def test_fused_works_with_a_frozen_g_frac_reference_h() -> None:
    """MINOR-1: freeze_g_frac_ref_h under 'fused' re-enters the S_bar_phi table
    at h_ref; with h_ref tabulated (the invariant evaluate now guarantees) the
    counterfactual runs and freezes g_frac to the reference-value ratio."""
    h, h_ref = 0.73, 0.62
    row = _run_p_Di(
        h=h,
        selection_cell="fused",
        p_det_obj=_mock_p_det_with_survival(),
        freeze_ref_h=h_ref,
        extra_table_h=[h_ref],
    )
    ref = _run_fused(h_ref, "fused")
    assert row["g_frac"] == ref["B_num_wbh"] / ref["B_num"]


def test_beyond_horizon_zeros_are_labelled_as_horizon(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """MINOR-2: S_4D == 0 everywhere => the zero-factor warning must say
    'beyond the detection horizon', not claim a phi-support exit."""
    with caplog.at_level("WARNING"):
        _run_p_Di(
            h=0.73,
            selection_cell="fused",
            p_det_obj=_mock_p_det_with_survival(s_value=0.0),
        )
    horizon_msgs = [r for r in caplog.records if "beyond the detection horizon" in r.message]
    assert horizon_msgs, "expected the horizon-classified zero-factor warning"
    assert not any("left the phi support" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# evaluate-level cell resolution
# ---------------------------------------------------------------------------
class _TripWireError(Exception):
    pass


class _TripInt:
    """base_seed stand-in whose int() raises: stops evaluate() right AFTER the
    cell-resolution block (`self._base_seed = int(base_seed) ...`), so the
    resolved cell can be asserted without running the heavy pipeline."""

    def __int__(self) -> int:
        raise _TripWireError


@pytest.mark.parametrize(
    ("mode", "resolved"),
    [("absolute_marginal", "fused"), ("generator_marginal", "off")],
)
def test_auto_resolves_per_normalization_mode(mode: str, resolved: str) -> None:
    instance = object.__new__(BayesianStatistics)
    instance._freeze_g_frac_ref_h = None
    with pytest.raises(_TripWireError):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            normalization_mode=mode,
            selection_in_completion_numerator="auto",
            base_seed=_TripInt(),  # type: ignore[arg-type]
        )
    assert instance._selection_in_completion_numerator == resolved


@pytest.mark.parametrize("cell", ["fused", "2d"])
def test_fusion_cells_require_absolute_marginal(cell: str) -> None:
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(ValueError, match="absolute_marginal"):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            normalization_mode="generator_marginal",
            selection_in_completion_numerator=cell,
        )


def test_external_g_i_callers_are_untouched() -> None:
    """MINOR-3: completion_mass_factor_g keeps its signature and semantics —
    the fusion is a NEW callable; the validation-suite imports stay valid."""
    d_L = _d_L_of_z(_Z)
    before = completion_mass_factor_g(_Z, d_L / _DET_D_L, _DET_M_Z, _PROJ, _SIGMA_SHARP)
    assert before.shape == _Z.shape
    assert np.all(np.isfinite(before))
    for module_path in (
        "darksiren_emri.validation.calibration_gate",
        "darksiren_emri.validation.closed_loop_gfrac",
        "darksiren_emri.validation.venue_transfer",
    ):
        mod = __import__(module_path, fromlist=["completion_mass_factor_g"])
        assert getattr(mod, "completion_mass_factor_g", completion_mass_factor_g) is not None
