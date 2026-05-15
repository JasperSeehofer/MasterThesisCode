"""Canonical-MAP consistency tests across H0-posterior figures (Phase H).

These tests are the guard that prevents regression of the bug found on
2026-05-15: the same combined H0 posterior was being computed by three
different paths producing different MAP estimates (one at h=0.738 from the
production physics-floor combine, one at h≈0.73 from in-memory naive sums,
one bootstrap-subset draw from the M_z improvement bank).

The Phase A unification routed every H0-exposing figure through
:func:`master_thesis_code.plotting._helpers.load_canonical_combined_posterior`.
This module verifies that contract — synthetic per-event posteriors are
written into a temp directory, the loader and the public figure paths are
exercised on the same data, and the discrete/continuous MAPs are checked
to agree to better than one grid step.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from master_thesis_code.bayesian_inference.posterior_combination import (
    compute_canonical_combined_posterior,
    load_per_h_likelihoods,
    parabolic_refine_map,
)
from master_thesis_code.plotting._helpers import load_canonical_combined_posterior


def _write_synthetic_posteriors(
    base: Path,
    *,
    n_events: int,
    h_grid: np.ndarray,
    map_h: float,
    width: float,
    variant: str,
    rng: np.random.Generator,
) -> None:
    """Write per-h JSON files for a synthetic Gaussian-shaped per-event L(h).

    Each event peaks at ``map_h + jitter`` with width ``width``; the sum
    across events therefore peaks at ``map_h`` to high accuracy when
    ``n_events`` is large.
    """
    out_dir = base / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    # Per-event peak jitter so the joint MAP is well-defined.
    centers = map_h + rng.normal(0.0, width * 0.5, size=n_events)
    for h in h_grid:
        per_event: dict[str, list[float]] = {}
        for ev in range(n_events):
            L = float(np.exp(-0.5 * ((h - centers[ev]) / width) ** 2))
            per_event[str(ev)] = [L]
        payload: dict[str, object] = {"h": float(h)}
        payload.update(per_event)
        h_str_int = int(h)
        h_str_frac = int(round((h - int(h)) * 100))
        fname = f"h_{h_str_int}_{h_str_frac}.json"
        (out_dir / fname).write_text(json.dumps(payload))


class TestCanonicalCombinedPosteriorRoundtrip:
    def test_compute_canonical_recovers_synthetic_peak(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(2026_05_16)
        h_grid = np.linspace(0.60, 0.86, 27, dtype=np.float64)
        _write_synthetic_posteriors(
            tmp_path,
            n_events=120,
            h_grid=h_grid,
            map_h=0.74,
            width=0.04,
            variant="posteriors",
            rng=rng,
        )
        result = compute_canonical_combined_posterior(tmp_path / "posteriors")
        assert result["n_events_used"] == 120
        # Discrete MAP should sit at the grid point nearest 0.74.
        discrete_map = result["discrete_map"]
        continuous_map = result["continuous_map"]
        assert isinstance(discrete_map, float)
        assert isinstance(continuous_map, float)
        assert abs(discrete_map - 0.74) < 0.012  # 1 grid step at Δh=0.01
        # Continuous MAP should be sub-grid accurate.
        assert abs(continuous_map - 0.74) < 0.005

    def test_loader_writes_and_rereads_cache(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(2026_05_17)
        h_grid = np.linspace(0.60, 0.86, 27, dtype=np.float64)
        _write_synthetic_posteriors(
            tmp_path,
            n_events=60,
            h_grid=h_grid,
            map_h=0.73,
            width=0.05,
            variant="posteriors",
            rng=rng,
        )
        # First call computes + caches.
        h1, p1, meta1 = load_canonical_combined_posterior(tmp_path, "posteriors")
        cache_path = tmp_path / "posteriors" / "canonical_combined.json"
        assert cache_path.is_file()
        # Second call hits the cache and returns equivalent data.
        h2, p2, meta2 = load_canonical_combined_posterior(tmp_path, "posteriors")
        np.testing.assert_array_equal(h1, h2)
        np.testing.assert_array_equal(p1, p2)
        assert meta1["discrete_map"] == meta2["discrete_map"]
        assert meta1["continuous_map"] == meta2["continuous_map"]

    def test_refresh_recomputes_and_overwrites_cache(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(2026_05_18)
        h_grid = np.linspace(0.60, 0.86, 27, dtype=np.float64)
        _write_synthetic_posteriors(
            tmp_path,
            n_events=40,
            h_grid=h_grid,
            map_h=0.73,
            width=0.05,
            variant="posteriors",
            rng=rng,
        )
        load_canonical_combined_posterior(tmp_path, "posteriors")  # primes cache
        cache_path = tmp_path / "posteriors" / "canonical_combined.json"
        # Mutate the cache to a sentinel value.
        cache_path.write_text(
            json.dumps(
                {
                    "h_values": [0.6],
                    "posterior": [1.0],
                    "log_posterior": [0.0],
                    "n_events_used": 1,
                    "discrete_map": 0.6,
                    "continuous_map": 0.6,
                    "strategy": "sentinel",
                }
            )
        )
        # refresh=True should ignore the sentinel and recompute.
        h_arr, p_arr, meta = load_canonical_combined_posterior(tmp_path, "posteriors", refresh=True)
        assert meta["strategy"] == "raw-sum-log"
        assert meta["n_events_used"] == 40

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_canonical_combined_posterior(tmp_path, "posteriors")

    def test_bad_variant_raises_via_paper_figures(self, tmp_path: Path) -> None:
        from master_thesis_code.plotting.paper_figures import _load_combined_posterior

        with pytest.raises(ValueError, match="Unknown variant"):
            _load_combined_posterior("bogus", tmp_path)


class TestCanonicalMapAgreesAcrossFigurePaths:
    """The Phase A contract: every figure must consume the SAME canonical MAP.

    With Phase A in place, the discrete/continuous MAPs returned by:
    - ``compute_canonical_combined_posterior`` (the raw helper)
    - ``_load_combined_posterior`` (used by paper_h0_posterior + KDE)
    - ``load_canonical_combined_posterior`` (used by main.py manifest + interactive)
    must all agree exactly because they share the same implementation.
    """

    def test_three_loader_paths_return_identical_map(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(2026_05_19)
        h_grid = np.linspace(0.60, 0.86, 27, dtype=np.float64)
        _write_synthetic_posteriors(
            tmp_path,
            n_events=80,
            h_grid=h_grid,
            map_h=0.74,
            width=0.04,
            variant="posteriors",
            rng=rng,
        )
        from master_thesis_code.plotting.paper_figures import _load_combined_posterior

        a = compute_canonical_combined_posterior(tmp_path / "posteriors")
        b_h, b_p, b_meta = load_canonical_combined_posterior(tmp_path, "posteriors")
        c = _load_combined_posterior("posteriors", tmp_path)

        a_discrete = a["discrete_map"]
        a_continuous = a["continuous_map"]
        a_posterior = a["posterior"]
        assert isinstance(a_discrete, float)
        assert isinstance(a_continuous, float)
        assert isinstance(a_posterior, list)

        # Discrete MAPs match exactly.
        assert a_discrete == b_meta["discrete_map"]
        assert a_discrete == c["discrete_map"]
        # Continuous MAPs match exactly.
        assert a_continuous == b_meta["continuous_map"]
        assert a_continuous == c["map_h"]
        # Posterior arrays match exactly.
        np.testing.assert_array_equal(np.asarray(a_posterior), b_p)
        np.testing.assert_array_equal(np.asarray(c["posterior"]), b_p)


class TestParabolicRefine:
    def test_parabolic_refine_matches_quadratic_peak(self) -> None:
        h = np.array([0.7, 0.72, 0.74, 0.76, 0.78], dtype=np.float64)
        peak = 0.733
        log_post = -((h - peak) ** 2) / 0.01
        refined = parabolic_refine_map(h, log_post)
        assert abs(refined - peak) < 1e-4

    def test_parabolic_refine_falls_back_at_boundary(self) -> None:
        h = np.linspace(0.6, 0.86, 27, dtype=np.float64)
        # Monotonic increasing log-posterior — argmax is at the right edge.
        log_post = np.arange(len(h), dtype=np.float64)
        refined = parabolic_refine_map(h, log_post)
        assert refined == float(h[-1])


class TestLoadPerHLikelihoods:
    def test_round_trip(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(2026_05_20)
        h_grid = np.linspace(0.60, 0.86, 14, dtype=np.float64)
        _write_synthetic_posteriors(
            tmp_path,
            n_events=10,
            h_grid=h_grid,
            map_h=0.73,
            width=0.06,
            variant="posteriors",
            rng=rng,
        )
        h_values, log_L = load_per_h_likelihoods(tmp_path / "posteriors")
        assert len(h_values) == 14
        assert log_L.shape == (10, 14)
        # All entries should be finite (no NaN) and bounded above by 0.
        assert np.isfinite(log_L).all()
        assert (log_L <= 0.0 + 1e-12).all()
