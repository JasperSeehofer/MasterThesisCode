"""Regression + behaviour tests for the 2026-08-20 combine/weights corrections.

Ledger row #145 · AMENDMENT A-7 in
``results/prod2d_closure_20260818/PREREGISTRATION_1D_CORRESPONDENCE.md`` ·
``docs/derivations/GATE_PRESENTATION_SENTINEL_COMBINE_20260820.md``.

Two defects were corrected in ``darksiren_emri/validation/correspondence_1d.py``:

1. a zero per-event likelihood was floored **in log space** at ``-1.0e300``,
   which float64 absorbs (``-1e300 + (-358.62) == -1e300``);
2. posterior moments used ``np.gradient(grid)``, the central-difference
   derivative stencil, whose endpoint weights are double the trapezoid rule's.

The superseded behaviours remain reachable so the banked fleet stays
reproducible (GATE R-0a). These tests pin BOTH: the old numbers under the legacy
switches, and the corrected behaviour by default.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from darksiren_emri.validation import correspondence_1d as c1d

# The exact degenerate values every fully-masked seed reported before the fix.
# Verified bit-for-bit against 21 banked seeds (e.g. b0_seed900101.json).
LEGACY_DEGENERATE_MEAN_H = 0.7299999999999999
LEGACY_DEGENERATE_SIGMA_H = 0.07784719124788758
LEGACY_DEGENERATE_MAP_H = 0.6


def _grid() -> npt.NDArray[np.float64]:
    return np.array(sorted(c1d.H_GRID_41), dtype=np.float64)


def _write_csv(path: Path, vals: npt.NDArray[np.float64], grid: npt.NDArray[np.float64]) -> str:
    """Write an ``event_likelihoods.csv`` carrying ``vals`` as ``combined_no_bh``."""
    rows = []
    for ev in range(vals.shape[0]):
        for j, h in enumerate(grid):
            rows.append({"event_idx": ev, "h": float(h), "combined_no_bh": float(vals[ev, j])})
    out = str(path / "event_likelihoods.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


# ── the sentinel: absorption, and the artefact it manufactures ───────────────


def test_float64_absorbs_real_signal_into_the_sentinel() -> None:
    """The mechanism: a real log-likelihood is annihilated by the -1e300 floor."""
    assert -1.0e300 + (-358.6188494587322) == -1.0e300
    assert (-1.0e300 + (-358.62)) - (-1.0e300 + (-344.77)) == 0.0


def test_legacy_sentinel_reproduces_the_banked_degenerate_values(tmp_path: Path) -> None:
    """REGRESSION (old value): a seed with one all-zero event reports the grid midpoint.

    This is the artefact of record -- `mean_h` equal to the arithmetic midpoint of
    H_GRID_41, which coincides with H_TRUE, together with `map_h` at the grid's
    first node (<= R_LOW_THRESHOLD, so `r_low` is True) and full coverage.
    """
    grid = _grid()
    rng = np.random.default_rng(0)
    vals = rng.uniform(0.1, 1.0, size=(5, grid.size))
    vals[2, :] = 0.0  # one event zero at EVERY node -> every node sentinelled
    csv = _write_csv(tmp_path, vals, grid)

    stats = c1d.compute_seed_statistics(
        csv, seed=1, zero_handling="legacy_sentinel", weights_convention="legacy_gradient"
    )
    assert stats.mean_h == LEGACY_DEGENERATE_MEAN_H
    assert stats.sigma_h == LEGACY_DEGENERATE_SIGMA_H
    assert stats.map_h == LEGACY_DEGENERATE_MAP_H
    assert stats.r_low is True
    assert (stats.c50, stats.c68, stats.c90) == (True, True, True)


def test_legacy_degenerate_mean_is_exactly_the_grid_midpoint() -> None:
    """The 'truth to four decimals' was grid geometry: (0.600 + 0.860)/2 = 0.730."""
    grid = _grid()
    assert (grid[0] + grid[-1]) / 2.0 == pytest.approx(c1d.H_TRUE, abs=1e-15)
    assert LEGACY_DEGENERATE_MEAN_H == pytest.approx(c1d.H_TRUE, abs=1e-15)


def test_fixed_path_refuses_a_fully_uninformative_seed(tmp_path: Path) -> None:
    """Corrected behaviour: the same seed is refused, not silently scored."""
    grid = _grid()
    vals = np.zeros((3, grid.size))
    csv = _write_csv(tmp_path, vals, grid)
    with pytest.raises(ValueError, match="uninformative"):
        c1d.compute_seed_statistics(csv, seed=1)


def test_fixed_path_drops_the_all_zero_event_and_keeps_the_rest(tmp_path: Path) -> None:
    """An event that is zero everywhere carries no h-information and must drop out."""
    grid = _grid()
    rng = np.random.default_rng(1)
    good = rng.uniform(0.1, 1.0, size=(4, grid.size))
    with_dead = np.vstack([good, np.zeros((1, grid.size))])

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    csv_good = _write_csv(tmp_path / "a", good, grid)
    csv_dead = _write_csv(tmp_path / "b", with_dead, grid)

    s_good = c1d.compute_seed_statistics(csv_good, seed=1)
    s_dead = c1d.compute_seed_statistics(csv_dead, seed=1)
    assert s_dead.mean_h == pytest.approx(s_good.mean_h, abs=1e-12)
    assert s_dead.sigma_h == pytest.approx(s_good.sigma_h, abs=1e-12)


def test_sentinel_and_physics_floor_agree_when_no_zeros(tmp_path: Path) -> None:
    """N-0 bit-identity: with no zeros the correction is a strict no-op.

    This is the property that held on all 79 sentinel-free banked seeds.
    """
    grid = _grid()
    rng = np.random.default_rng(2)
    vals = rng.uniform(0.05, 2.0, size=(30, grid.size))
    legacy = c1d.combine_log_likelihood(vals, "legacy_sentinel")
    fixed = c1d.combine_log_likelihood(vals, "physics_floor")
    np.testing.assert_array_equal(legacy, fixed)


def test_sentinel_matches_minus_inf_whenever_a_node_survives() -> None:
    """The narrowing that makes the fix safe: identical unless EVERY node is masked."""
    grid = _grid()
    rng = np.random.default_rng(3)
    vals = rng.uniform(0.1, 1.0, size=(6, grid.size))
    vals[1, :10] = 0.0  # a partial mask -- nodes 10.. still survive
    sentinel = c1d.combine_log_likelihood(vals, "legacy_sentinel")

    with np.errstate(divide="ignore"):
        log_l = np.where(vals > 0.0, np.log(vals, where=vals > 0.0), -np.inf)
    exact = log_l.sum(axis=0)

    w = c1d.moment_weights(grid, "legacy_gradient")

    def mean_of(s: npt.NDArray[np.float64]) -> float:
        post = np.exp(s - np.nanmax(s))
        return float((post * grid * w).sum() / (post * w).sum())

    assert mean_of(sentinel) == pytest.approx(mean_of(exact), abs=0.0, rel=0.0)


def test_combine_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="unknown zero_handling"):
        c1d.combine_log_likelihood(np.ones((2, 3)), "nope")  # type: ignore[arg-type]


# ── moment weights ──────────────────────────────────────────────────────────


def test_gradient_weights_double_the_endpoints() -> None:
    """REGRESSION (old value): np.gradient over-counts the interval by one step."""
    grid = _grid()
    legacy = c1d.moment_weights(grid, "legacy_gradient")
    trap = c1d.moment_weights(grid, "trapezoid")
    assert legacy[0] == pytest.approx(2.0 * trap[0])
    assert legacy[-1] == pytest.approx(2.0 * trap[-1])
    assert legacy.sum() == pytest.approx(0.27)
    assert trap.sum() == pytest.approx(grid[-1] - grid[0])
    assert trap.sum() == pytest.approx(0.26)


def test_trapezoid_weights_match_gradient_in_the_interior() -> None:
    grid = _grid()
    legacy = c1d.moment_weights(grid, "legacy_gradient")
    trap = c1d.moment_weights(grid, "trapezoid")
    np.testing.assert_allclose(legacy[1:-1], trap[1:-1], rtol=0, atol=0)


def test_trapezoid_is_exact_for_a_flat_density() -> None:
    """Limiting case: a flat posterior must have its mean at the interval midpoint."""
    grid = _grid()
    for convention in ("trapezoid", "legacy_gradient"):
        w = c1d.moment_weights(grid, convention)
        post = np.ones_like(grid)
        mean = float((post * grid * w).sum() / (post * w).sum())
        assert mean == pytest.approx((grid[0] + grid[-1]) / 2.0, abs=1e-12)


def test_uniform_grid_trapezoid_weights_are_textbook() -> None:
    g = np.linspace(0.0, 1.0, 11)
    w = c1d.moment_weights(g, "trapezoid")
    d = g[1] - g[0]
    assert w[0] == pytest.approx(d / 2.0)
    assert w[-1] == pytest.approx(d / 2.0)
    np.testing.assert_allclose(w[1:-1], d, rtol=1e-12)
    assert w.sum() == pytest.approx(1.0)


def test_moment_weights_rejects_unknown_convention() -> None:
    with pytest.raises(ValueError, match="unknown moment-weight convention"):
        c1d.moment_weights(_grid(), "nope")  # type: ignore[arg-type]


# ── _hpd_contains is CORRECT and must stay that way ─────────────────────────


@pytest.mark.parametrize(("level", "z"), [(0.50, 0.6744897), (0.68, 0.9944579), (0.90, 1.6448536)])
def test_hpd_contains_matches_the_analytic_gaussian(level: float, z: float) -> None:
    """A verifier proposed 'fixing' this; it is right. Pin it so it is not 'fixed'."""
    g = np.linspace(0.5, 1.0, 201)
    w = np.gradient(g)
    post = np.exp(-0.5 * ((g - 0.75) / 0.05) ** 2)
    post /= (post * w).sum()
    for k, expected in ((z - 0.05, True), (z + 0.05, False)):
        target = int(np.argmin(np.abs(g - (0.75 + k * 0.05))))
        assert c1d._hpd_contains(post, w, target, level) is expected
