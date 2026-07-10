"""End-to-end numeric parity pin for ``BayesianStatistics.evaluate`` (Pipeline B).

Companion to ``bayesian_inference/test_kernel_parity.py``. The kernel test pins
``single_host_likelihood`` against a *stub* p_det; this test pins the **full**
evaluate() path — real ``SimulationDetectionProbability`` (RegularGridInterpolator
p_det), BallTree host resolution, the completion denominator, and the
multiprocessing reduction — so the performance refactor's batched-interpolator
and host-dimension changes are covered too.

The synthetic pipeline is fully deterministic: seeded galaxy catalogue (rng 99),
seeded ``Model1CrossCheck`` (rng 42), ``base_seed=0`` default, and the 4D MC
denominator's per-(event,host) seeded stream. Per-event likelihoods across a
small H0 grid are therefore reproducible and pinned to a committed golden.

Regenerate (only on unmodified production, or as the reviewed value-update step
of an approved physics change) with::

    REGEN_PIPELINE_GOLDEN=1 uv run pytest -m slow \
        master_thesis_code_test/integration/test_pipeline_parity.py

Marked ``slow``: it spins the forkserver pool and evaluates a 3-point H0 grid.
"""

import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from master_thesis_code_test.integration.conftest import build_galaxy_catalog_for_n_detections
from master_thesis_code_test.integration.test_evaluation_pipeline import (
    FIXTURES_DIR,
    _create_synthetic_injection_csvs,
)

if TYPE_CHECKING:
    from master_thesis_code.cosmological_model import Model1CrossCheck

_GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "pipeline_parity_pins.json"

# H0 grid straddling the true value; small so the pool spins quickly.
_H_GRID: list[float] = [0.68, 0.73, 0.78]

# Per-event likelihoods pass through the real RegularGridInterpolator + BallTree,
# so reordered float ops in a refactor may perturb the last ~1-2 ULP. This gate
# is a "did a value move meaningfully" tripwire, not a bit-identity check.
_REL_TOL = 1e-9
_ABS_TOL = 1e-30


def _setup_sim_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage fixture CSVs + injections in tmp_path and point the module at them."""
    sim_dir = tmp_path / "simulations"
    sim_dir.mkdir()
    shutil.copy(
        FIXTURES_DIR / "synthetic_cramer_rao_bounds.csv",
        sim_dir / "cramer_rao_bounds.csv",
    )
    shutil.copy(
        FIXTURES_DIR / "synthetic_prepared_cramer_rao_bounds.csv",
        sim_dir / "prepared_cramer_rao_bounds.csv",
    )
    injection_dir = sim_dir / "injections"
    injection_dir.mkdir()
    _create_synthetic_injection_csvs(str(injection_dir))

    import master_thesis_code.bayesian_inference.bayesian_statistics as bs

    monkeypatch.setattr(
        bs, "PREPARED_CRAMER_RAO_BOUNDS_PATH", str(sim_dir / "prepared_cramer_rao_bounds.csv")
    )
    monkeypatch.setattr(bs, "CRAMER_RAO_BOUNDS_OUTPUT_PATH", str(sim_dir / "cramer_rao_bounds.csv"))
    monkeypatch.setattr(bs, "INJECTION_DATA_DIR", str(injection_dir))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(4)))


def _run_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cosmological_model: "Model1CrossCheck",
) -> dict[str, dict[str, list[float]]]:
    """Run evaluate() over the H0 grid; return per-event 1D and 2D likelihoods."""
    _setup_sim_env(tmp_path, monkeypatch)

    from master_thesis_code.bayesian_inference.bayesian_statistics import BayesianStatistics

    galaxy_catalog = build_galaxy_catalog_for_n_detections(5)
    bayesian_stats = BayesianStatistics()

    for h in _H_GRID:
        bayesian_stats.evaluate(
            galaxy_catalog=galaxy_catalog,
            cosmological_model=cosmological_model,
            h_value=float(h),
        )

    def _int_keyed(data: dict[Any, Any]) -> dict[str, list[float]]:
        # posterior_data_with_bh_mass also carries non-int bookkeeping keys
        # (GALAXY_LIKELIHOODS, ADDITIONAL_GALAXIES_WITHOUT_BH_MASS); keep only
        # the per-event integer-indexed likelihood lists.
        out: dict[str, list[float]] = {}
        for k, v in data.items():
            if isinstance(k, int) and isinstance(v, list):
                out[str(k)] = [float(x) for x in v]
        return out

    return {
        "1d": _int_keyed(bayesian_stats.posterior_data),
        "2d": _int_keyed(bayesian_stats.posterior_data_with_bh_mass),
    }


@pytest.mark.slow
def test_pipeline_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cosmological_model: "Model1CrossCheck",
) -> None:
    """Full evaluate() reproduces committed per-event likelihoods across the H0 grid."""
    np.random.seed(42)
    result = _run_pipeline(tmp_path, monkeypatch, cosmological_model)

    if os.environ.get("REGEN_PIPELINE_GOLDEN"):
        _GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_GOLDEN_PATH, "w") as f:
            json.dump({"h_grid": _H_GRID, **result}, f, indent=2, sort_keys=True)
        pytest.skip("pipeline golden regenerated")

    if not _GOLDEN_PATH.exists():
        pytest.skip(
            "pipeline parity golden not generated; run with REGEN_PIPELINE_GOLDEN=1 "
            "on unmodified production first"
        )

    with open(_GOLDEN_PATH) as f:
        golden = json.load(f)

    assert golden["h_grid"] == _H_GRID, "golden H0 grid differs; regenerate"
    for channel in ("1d", "2d"):
        got_ch = result[channel]
        exp_ch = golden[channel]
        assert set(got_ch) == set(exp_ch), (
            f"{channel}: event-index set changed {set(got_ch) ^ set(exp_ch)}"
        )
        for idx, exp_vals in exp_ch.items():
            got_vals = got_ch[idx]
            assert len(got_vals) == len(exp_vals), f"{channel}[{idx}]: length changed"
            for i, (g, e) in enumerate(zip(got_vals, exp_vals, strict=True)):
                assert g == pytest.approx(e, rel=_REL_TOL, abs=_ABS_TOL), (
                    f"{channel}[{idx}] h={_H_GRID[i]}: {g} != {e}"
                )


@pytest.mark.slow
def test_fused_h_grid_matches_sequential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cosmological_model: "Model1CrossCheck",
) -> None:
    """One fused evaluate(h_values=grid) pass == N sequential single-h passes.

    The fused mode shares all h-invariant setup (p_det grid, completeness,
    D(h)/beta tables, Fisher staging, worker pool) across the grid; per-h
    outputs must be numerically IDENTICAL (exact ==) to independent single-h
    evaluations, and each per-h JSON must carry exactly one likelihood per
    event (the canonical production shape).
    """
    np.random.seed(42)
    _setup_sim_env(tmp_path, monkeypatch)

    from master_thesis_code.bayesian_inference.bayesian_statistics import BayesianStatistics

    galaxy_catalog = build_galaxy_catalog_for_n_detections(5)

    # Sequential reference: legacy accumulation, one value per h per event.
    sequential = BayesianStatistics()
    for h in _H_GRID:
        sequential.evaluate(
            galaxy_catalog=galaxy_catalog,
            cosmological_model=cosmological_model,
            h_value=float(h),
        )
    seq_1d = {k: list(v) for k, v in sequential.posterior_data.items() if isinstance(k, int)}
    seq_2d = {
        k: list(v)
        for k, v in sequential.posterior_data_with_bh_mass.items()
        if isinstance(k, int) and isinstance(v, list)
    }

    # Fused run: fresh instance, one pass over the whole grid.
    fused = BayesianStatistics()
    fused.evaluate(
        galaxy_catalog=galaxy_catalog,
        cosmological_model=cosmological_model,
        h_value=float(_H_GRID[0]),  # superseded by h_values
        h_values=[float(h) for h in _H_GRID],
    )

    for i, h in enumerate(_H_GRID):
        h_label = str(np.round(h, 4)).replace(".", "_")
        for channel, seq_ch in (("posteriors", seq_1d), ("posteriors_with_bh_mass", seq_2d)):
            json_path = tmp_path / "simulations" / channel / f"h_{h_label}.json"
            assert json_path.exists(), f"fused mode did not write {json_path}"
            with open(json_path) as f:
                data = json.load(f)
            assert data["h"] == float(h)
            for idx, seq_vals in seq_ch.items():
                fused_vals = data[str(idx)]
                assert len(fused_vals) == 1, (
                    f"{channel} h={h}: expected exactly one likelihood per event, "
                    f"got {len(fused_vals)}"
                )
                assert fused_vals[0] == seq_vals[i], (
                    f"{channel}[{idx}] h={h}: fused {fused_vals[0]!r} != sequential {seq_vals[i]!r}"
                )
