"""Tests for the G-1 [C-SYM]/[P3] ``selection_cell`` extension: ``cat1d``/``symmetric``.

Covers the required tests t1-t5 registered in
``results/pp_coverage_csym_20260818/PREREGISTRATION_G1_CATLEG_SYMMETRY.md`` Sec 0:

* t1 -- byte-identity of the four PRE-EXISTING ``selection_cell`` modes
  (``off``/``1d``/``2d``/``fused``) against golden values recorded at this
  commit (the extension must not perturb any pre-existing code path);
* t2 -- ``S_bar_phi`` collapses to 1 (``d50 -> very large``): ``symmetric``
  reduces bit-exactly to ``fused`` and ``cat1d`` to ``off``;
* t3 -- the empty-catalogue-ball limit: same reduction, bit-exact;
* t4 -- generative-stream alignment: the shared-seed generative diagnostics
  (completion/ball statistics) are identical across ALL SIX
  ``selection_cell`` values -- ``selection_cell`` is estimator-side only;
* t5 -- the ``alpha_M = 0`` mass-blind reduction: at zero horizon index the
  1D-channel result of ``cat1d``/``symmetric`` is invariant to
  ``mass_slope`` (``phi(M)`` integrates out of a mass-blind ``S_bar_phi``
  exactly), mirroring the existing ``mass_horizon_index = 0`` reduction
  pattern in ``test_pp_coverage_mass.py``.

All tests are CPU-only and fast (no ``gpu``/``slow`` marker).
"""

import dataclasses
from typing import Any, Literal, get_args

from darksiren_emri.validation.pp_coverage import PPCoverageConfig, run_coverage
from darksiren_emri_test.validation.test_pp_coverage_mass import TINY_MASS

SelectionCell = Literal["off", "1d", "2d", "fused", "cat1d", "symmetric"]
ALL_SIX_CELLS: tuple[SelectionCell, ...] = get_args(SelectionCell)


def _one_d(block: dict[str, Any]) -> dict[str, Any]:
    """Strip the nested 2D-channel block for 1D-channel-only comparisons."""
    return {k: v for k, v in block.items() if k != "mass_channel_2d"}


def _run(config: PPCoverageConfig) -> dict[str, Any]:
    result: dict[str, Any] = run_coverage(config)["results"]["0.7200"]
    return result


# ---------------------------------------------------------------------------
# t1 -- byte-identity of the four pre-existing modes (prodcal-freeze guard)
# ---------------------------------------------------------------------------

# Golden values recorded from the extended harness (this commit) on
# TINY_MASS for the four PRE-EXISTING selection_cell modes. The extension
# adds ``cat1d``/``symmetric`` as new ``in (...)`` branch members only; it
# does not alter the code path taken by ``off``/``1d``/``2d``/``fused``. Any
# drift here means the extension broke a pre-existing mode: STOP (N-A).
GOLDEN_T1: dict[SelectionCell, dict[str, float]] = {
    "off": {
        "map_mean": 0.63,
        "map_std": 0.030000000000000027,
        "completion_fraction": 0.65,
        "empty_ball_fraction": 0.25,
        "mean_ball_size": 1.475,
        "host_in_ball_fraction": 0.35,
        "impostor_fraction": 0.7586405529953917,
    },
    "1d": {
        "map_mean": 0.62,
        "map_std": 0.020000000000000018,
        "completion_fraction": 0.65,
        "empty_ball_fraction": 0.25,
        "mean_ball_size": 1.475,
        "host_in_ball_fraction": 0.35,
        "impostor_fraction": 0.7586405529953917,
    },
    "2d": {
        "map_mean": 0.63,
        "map_std": 0.030000000000000027,
        "completion_fraction": 0.65,
        "empty_ball_fraction": 0.25,
        "mean_ball_size": 1.475,
        "host_in_ball_fraction": 0.35,
        "impostor_fraction": 0.7586405529953917,
    },
    "fused": {
        "map_mean": 0.62,
        "map_std": 0.020000000000000018,
        "completion_fraction": 0.65,
        "empty_ball_fraction": 0.25,
        "mean_ball_size": 1.475,
        "host_in_ball_fraction": 0.35,
        "impostor_fraction": 0.7586405529953917,
    },
}


def test_t1_pre_existing_modes_are_byte_identical() -> None:
    """off/1d/2d/fused are unperturbed by the cat1d/symmetric extension."""
    for cell, golden in GOLDEN_T1.items():
        block = _one_d(_run(dataclasses.replace(TINY_MASS, selection_cell=cell)))
        for key, value in golden.items():
            assert block[key] == value, f"{cell}.{key}: {block[key]!r} != {value!r}"


# ---------------------------------------------------------------------------
# t2 -- S_bar_phi collapses to 1 (d50 -> very large): the no-selection limit
# ---------------------------------------------------------------------------


def test_t2_symmetric_reduces_to_fused_at_large_d50() -> None:
    """At d50 >> any d_L in range, S_bar_phi -> 1 EXACTLY (erfc saturation).

    Multiplying by an array of exact 1.0 is bit-exact in IEEE-754, so
    'symmetric' (fused + the cat1d catalogue-leg factor) must reduce
    bit-exactly to 'fused' -- the catalogue-leg factor becomes the identity.
    """
    huge = dataclasses.replace(TINY_MASS, d50_gpc=1.0e6)
    symmetric = _run(dataclasses.replace(huge, selection_cell="symmetric"))
    fused = _run(dataclasses.replace(huge, selection_cell="fused"))
    assert symmetric == fused


def test_t2_cat1d_reduces_to_off_at_large_d50() -> None:
    """Same limit: 'cat1d' (off + the catalogue-leg factor) reduces to 'off'."""
    huge = dataclasses.replace(TINY_MASS, d50_gpc=1.0e6)
    cat1d = _run(dataclasses.replace(huge, selection_cell="cat1d"))
    off = _run(dataclasses.replace(huge, selection_cell="off"))
    assert cat1d == off


# ---------------------------------------------------------------------------
# t3 -- empty-catalogue-ball limit
# ---------------------------------------------------------------------------

# z_support well below the catalogue's populated redshift range at this
# n_galaxies: the catalogue itself is non-empty (>= 1 galaxy at z < z_support
# somewhere on the sky, so _build_catalogue does not raise), but no event's
# own host (and no chance impostor) ever lands inside its localization ball
# -- verified below via empty_ball_fraction == 1.0 exactly, the same
# diagnostic the harness already reports. This is a cleaner "genuinely empty
# ball" limit than sky_frac -> 0: at sky_frac == 0 the reciprocal cap
# construction (docstring of _perturb_within_cap) still ALWAYS places a
# catalogued host in its own zero-radius ball (bit-exact self-match), which
# additionally collides with mixture_mode="absolute"'s sky_frac-scaled
# denominator (near-singular, not a clean limit); this config avoids both.
EMPTY_BALL = dataclasses.replace(TINY_MASS, z_support=0.02, n_galaxies=200_000)


def test_t3_empty_ball_precondition_holds() -> None:
    """Precondition: every event's catalogue ball is genuinely empty."""
    off = _run(dataclasses.replace(EMPTY_BALL, selection_cell="off"))
    assert off["empty_ball_fraction"] == 1.0
    assert off["host_in_ball_fraction"] == 0.0


def test_t3_symmetric_reduces_to_fused_with_no_catalogue_candidates() -> None:
    symmetric = _run(dataclasses.replace(EMPTY_BALL, selection_cell="symmetric"))
    fused = _run(dataclasses.replace(EMPTY_BALL, selection_cell="fused"))
    assert symmetric == fused


def test_t3_cat1d_reduces_to_off_with_no_catalogue_candidates() -> None:
    cat1d = _run(dataclasses.replace(EMPTY_BALL, selection_cell="cat1d"))
    off = _run(dataclasses.replace(EMPTY_BALL, selection_cell="off"))
    assert cat1d == off


# ---------------------------------------------------------------------------
# t4 -- generative-stream alignment across all six selection_cell values
# ---------------------------------------------------------------------------


def test_t4_generative_stream_is_identical_across_all_six_cells() -> None:
    """selection_cell is estimator-side only: the shared seed's generative
    draws (host selection, sky placement, catalogue occupancy) must not
    move when selection_cell changes -- pinned via the diagnostics that are
    purely functions of the generative stream (never of the numerator math).
    """
    diagnostics_keys = (
        "completion_fraction",
        "empty_ball_fraction",
        "mean_ball_size",
        "host_in_ball_fraction",
        "impostor_fraction",
    )
    reference = None
    for cell in ALL_SIX_CELLS:
        block = _run(dataclasses.replace(TINY_MASS, selection_cell=cell))
        diag = {k: block[k] for k in diagnostics_keys}
        if reference is None:
            reference = diag
        else:
            assert diag == reference, f"selection_cell={cell!r} moved the generative stream"


# ---------------------------------------------------------------------------
# t5 -- alpha_M = 0 mass-blind reduction (slope invariance)
# ---------------------------------------------------------------------------


def test_t5_cat1d_1d_channel_is_mass_slope_invariant_at_zero_horizon_index() -> None:
    """At alpha_M = 0, S_4D is mass-independent, so phi(M) integrates out of
    S_bar_phi exactly (S_bar_phi(z;h) = S(d_L(z;h)), normalization pinned by
    test_phi_marginal_survival_matches_pdet_at_zero_horizon_index in
    test_pp_coverage_mass.py) -- and TINY_MASS's mass_rate_index = 0.0 makes
    the catalogue leg's rate weight mass-blind too. The 1D-channel result of
    'cat1d' must therefore be EXACTLY invariant to mass_slope, mirroring the
    existing alpha_M = 0 mass-blind-limit pattern.
    """
    base = dataclasses.replace(TINY_MASS, mass_horizon_index=0.0, selection_cell="cat1d")
    a = _one_d(_run(dataclasses.replace(base, mass_slope=0.0)))
    b = _one_d(_run(dataclasses.replace(base, mass_slope=0.7)))
    assert a == b


def test_t5_symmetric_1d_channel_is_mass_slope_invariant_at_zero_horizon_index() -> None:
    base = dataclasses.replace(TINY_MASS, mass_horizon_index=0.0, selection_cell="symmetric")
    a = _one_d(_run(dataclasses.replace(base, mass_slope=0.0)))
    b = _one_d(_run(dataclasses.replace(base, mass_slope=-0.4)))
    assert a == b


def test_t5_sanity_slope_matters_at_nonzero_horizon_index() -> None:
    """Negative control: the invariance above is a real alpha_M = 0 effect,
    not a vacuous / broken comparison -- at alpha_M > 0, mass_slope DOES
    change the cat1d 1D-channel result.
    """
    base = dataclasses.replace(TINY_MASS, mass_horizon_index=0.25, selection_cell="cat1d")
    a = _one_d(_run(dataclasses.replace(base, mass_slope=0.0)))
    b = _one_d(_run(dataclasses.replace(base, mass_slope=0.7)))
    assert a != b


# ---------------------------------------------------------------------------
# Basic API surface: SELECTION_CELLS / CLI plumbing
# ---------------------------------------------------------------------------


def test_new_cells_are_registered_and_finite() -> None:
    from darksiren_emri.validation.pp_coverage import SELECTION_CELLS

    assert "cat1d" in SELECTION_CELLS
    assert "symmetric" in SELECTION_CELLS
    for cell in ("cat1d", "symmetric"):  # type: SelectionCell
        res = _run(dataclasses.replace(TINY_MASS, selection_cell=cell))
        assert res["map_mean"] == res["map_mean"]  # not NaN
        assert res["mass_channel_2d"]["map_mean"] == res["mass_channel_2d"]["map_mean"]


def test_cat1d_and_symmetric_differ_from_off_and_fused_in_the_engaged_regime() -> None:
    """Sanity: on TINY_MASS's normal (non-limiting) venue, the new legs
    actually engage -- 'cat1d' != 'off' and 'symmetric' != 'fused'.
    """
    off = _one_d(_run(dataclasses.replace(TINY_MASS, selection_cell="off")))
    cat1d = _one_d(_run(dataclasses.replace(TINY_MASS, selection_cell="cat1d")))
    fused = _one_d(_run(dataclasses.replace(TINY_MASS, selection_cell="fused")))
    symmetric = _one_d(_run(dataclasses.replace(TINY_MASS, selection_cell="symmetric")))
    assert off != cat1d
    assert fused != symmetric
