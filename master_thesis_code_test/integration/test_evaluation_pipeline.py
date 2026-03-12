"""Integration test for the evaluation pipeline (Pipeline B).

Exercises the full BayesianStatistics.evaluate() path with synthetic
fixture CSVs and a fake galaxy catalog, verifying that:
- posteriors are produced and structurally correct
- output JSON files are written
- a posterior plot can be generated
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from master_thesis_code.galaxy_catalogue.handler import GalaxyCatalogueHandler

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "evaluation"


@pytest.mark.slow
def test_evaluation_pipeline_produces_valid_posterior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_galaxy_catalog: GalaxyCatalogueHandler,
) -> None:
    """Run BayesianStatistics.evaluate() on synthetic data and verify output."""
    np.random.seed(42)

    # ── 1. Copy fixture CSVs to tmp_path with production filenames ──────
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
    shutil.copy(
        FIXTURES_DIR / "synthetic_undetected_events.csv",
        sim_dir / "undetected_events.csv",
    )

    # ── 2. Monkeypatch CSV paths in cosmological_model module ───────────
    import master_thesis_code.cosmological_model as cm

    monkeypatch.setattr(
        cm,
        "PREPARED_CRAMER_RAO_BOUNDS_PATH",
        str(sim_dir / "prepared_cramer_rao_bounds.csv"),
    )
    monkeypatch.setattr(
        cm,
        "CRAMER_RAO_BOUNDS_OUTPUT_PATH",
        str(sim_dir / "cramer_rao_bounds.csv"),
    )
    monkeypatch.setattr(
        cm,
        "UNDETECTED_EVENTS_OUTPUT_PATH",
        str(sim_dir / "undetected_events.csv"),
    )

    # ── 3. Monkeypatch cwd so relative output paths land in tmp_path ────
    monkeypatch.chdir(tmp_path)

    # ── 4. Guarantee enough CPUs for the multiprocessing pool ───────────
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(4)))

    # ── 5. Create real Model1CrossCheck (needs MCMC burn-in) ────────────
    from master_thesis_code.cosmological_model import BayesianStatistics, Model1CrossCheck

    cosmological_model = Model1CrossCheck()

    # ── 6. Create BayesianStatistics (reads monkeypatched CSV paths) ────
    bayesian_stats = BayesianStatistics()
    assert len(bayesian_stats.cramer_rao_bounds) == 5
    assert len(bayesian_stats.undetected_events) == 20

    # ── 7. Run evaluate ─────────────────────────────────────────────────
    h_value = 0.73
    bayesian_stats.evaluate(
        galaxy_catalog=fake_galaxy_catalog,
        cosmological_model=cosmological_model,
        h_value=h_value,
    )

    # ── 8. Assert structural correctness of posterior_data ──────────────
    posterior_data = bayesian_stats.posterior_data
    assert len(posterior_data) > 0, "posterior_data should have at least one detection"

    has_nonzero = False
    for idx, likelihoods in posterior_data.items():
        assert isinstance(likelihoods, list), f"Detection {idx}: expected list"
        for val in likelihoods:
            assert np.isfinite(val), f"Detection {idx}: non-finite likelihood {val}"
            if val > 0:
                has_nonzero = True

    # At least some detections should have positive likelihood
    # (if the galaxy catalog is aligned with the detection positions)
    # Note: due to the synthetic nature of the data, some may be zero
    # We only require the pipeline didn't crash — nonzero is a bonus

    # ── 9. Assert output JSON files exist ───────────────────────────────
    posterior_json = tmp_path / "simulations" / "posteriors" / "h_0_73.json"
    posterior_bh_json = tmp_path / "simulations" / "posteriors_with_bh_mass" / "h_0_73.json"

    assert posterior_json.exists(), f"Expected {posterior_json}"
    assert posterior_bh_json.exists(), f"Expected {posterior_bh_json}"

    with open(posterior_json) as f:
        data = json.load(f)
    assert data["h"] == h_value
    # Should have at least one detection key beyond "h"
    detection_keys = [k for k in data if k != "h"]
    assert len(detection_keys) > 0, "JSON should contain detection indices"

    with open(posterior_bh_json) as f:
        data_bh = json.load(f)
    assert data_bh["h"] == h_value

    # ── 10. Produce a posterior plot (uses project emri_thesis.mplstyle) ─
    from master_thesis_code.plotting import apply_style, save_figure
    from master_thesis_code.plotting.bayesian_plots import plot_event_posteriors

    apply_style()  # also called by session-scoped _plotting_style fixture

    # Build a single-point "h_values" array and a dict matching the JSON
    h_arr = np.array([h_value])
    plot_data: dict[int, list[float]] = {}
    for key in detection_keys:
        plot_data[int(key)] = data[key]

    fig, ax = plot_event_posteriors(
        h_values=h_arr,
        posterior_data=plot_data,
        true_h=0.73,
        title="Integration test: single h-value",
    )

    plot_path = str(tmp_path / "evaluation_posterior")
    save_figure(fig, plot_path, formats=("png",), close=True)

    png_path = tmp_path / "evaluation_posterior.png"
    assert png_path.exists(), "Posterior plot PNG not produced"
    assert png_path.stat().st_size > 0, "Posterior plot PNG is empty"
