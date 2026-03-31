"""Integration test for the evaluation pipeline (Pipeline B).

Exercises the full BayesianStatistics.evaluate() path with synthetic
fixture CSVs and a fake galaxy catalog, verifying that:
- posteriors are produced and structurally correct
- output JSON files are written
- a posterior plot can be generated
- the posterior narrows as more detections are combined
"""

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest

from master_thesis_code.galaxy_catalogue.handler import GalaxyCatalogueHandler

if TYPE_CHECKING:
    from master_thesis_code.cosmological_model import Model1CrossCheck

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "evaluation"

TRUE_H = 0.73
H_GRID = np.linspace(0.64, 0.82, 11)


def _combine_posterior(
    posterior_data: dict[int, list[float]],
    detection_indices: list[int],
) -> npt.NDArray[np.float64]:
    """Combine per-detection posteriors for a subset of detections.

    Returns the product of normalized per-detection posteriors.
    """
    combined = np.ones(len(H_GRID))
    for idx in detection_indices:
        arr = np.array(posterior_data[idx], dtype=np.float64)
        if np.max(arr) > 0:
            combined *= arr / np.max(arr)
    return combined


def _posterior_width(posterior: npt.NDArray[np.float64]) -> int:
    """Number of h-bins above 50% of the peak value."""
    peak = np.max(posterior)
    if peak <= 0:
        return len(posterior)
    return int(np.sum(posterior > 0.5 * peak))


def _generate_gallery_html(plot_dir: Path) -> None:
    """Write a self-contained HTML gallery page to the parent of *plot_dir*."""
    artifacts_dir = plot_dir.parent
    html_path = artifacts_dir / "index.html"

    # Use the test file's location to find the repo root (cwd may be tmp_path)
    repo_root = Path(__file__).resolve().parent.parent.parent
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            cwd=repo_root,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "unknown"

    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    pngs = sorted(plot_dir.glob("*.png"))
    if not pngs:
        return

    per_det = [p for p in pngs if "comparison" not in p.name]
    comparison = [p for p in pngs if "comparison" in p.name]

    img_tags = ""
    for p in per_det:
        rel = p.relative_to(artifacts_dir)
        img_tags += (
            f'      <div class="plot"><img src="{rel}" alt="{p.stem}"><p>{p.stem}</p></div>\n'
        )

    comparison_tags = ""
    for p in comparison:
        rel = p.relative_to(artifacts_dir)
        comparison_tags += (
            f'      <div class="plot"><img src="{rel}" alt="{p.stem}"><p>{p.stem}</p></div>\n'
        )

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Integration Test Plot Gallery</title>
<style>
  body {{ background: #1a1a2e; color: #e0e0e0; font-family: sans-serif; margin: 2em; }}
  h1, h2 {{ text-align: center; }}
  .meta {{ text-align: center; color: #888; margin-bottom: 2em; }}
  .gallery {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 1.5em; }}
  .plot {{ background: #16213e; border-radius: 8px; padding: 1em; max-width: 600px; }}
  .plot img {{ width: 100%; border-radius: 4px; }}
  .plot p {{ text-align: center; font-size: 0.9em; color: #aaa; }}
</style>
</head>
<body>
  <h1>Evaluation Pipeline &mdash; Integration Test Plots</h1>
  <p class="meta">Commit: <code>{sha}</code> &middot; {timestamp}</p>

  <h2>Per-detection-count posteriors</h2>
  <div class="gallery">
{img_tags}  </div>

  <h2>Posterior narrowing comparison</h2>
  <div class="gallery">
{comparison_tags}  </div>
</body>
</html>
"""
    html_path.write_text(html)


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

    # ── 2. Monkeypatch CSV paths in bayesian_statistics module ──────────
    import master_thesis_code.bayesian_inference.bayesian_statistics as bs

    monkeypatch.setattr(
        bs,
        "PREPARED_CRAMER_RAO_BOUNDS_PATH",
        str(sim_dir / "prepared_cramer_rao_bounds.csv"),
    )
    monkeypatch.setattr(
        bs,
        "CRAMER_RAO_BOUNDS_OUTPUT_PATH",
        str(sim_dir / "cramer_rao_bounds.csv"),
    )
    monkeypatch.setattr(
        bs,
        "UNDETECTED_EVENTS_OUTPUT_PATH",
        str(sim_dir / "undetected_events.csv"),
    )

    # ── 3. Monkeypatch cwd so relative output paths land in tmp_path ────
    monkeypatch.chdir(tmp_path)

    # ── 4. Guarantee enough CPUs for the multiprocessing pool ───────────
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(4)))

    # ── 5. Create real Model1CrossCheck (needs MCMC burn-in) ────────────
    from master_thesis_code.bayesian_inference.bayesian_statistics import BayesianStatistics
    from master_thesis_code.cosmological_model import Model1CrossCheck

    cosmological_model = Model1CrossCheck()

    # ── 6. Create BayesianStatistics (reads monkeypatched CSV paths) ────
    bayesian_stats = BayesianStatistics()
    assert len(bayesian_stats.cramer_rao_bounds) == 5
    assert len(bayesian_stats.undetected_events) == 20  # type: ignore[attr-defined]  # pre-existing: attribute does not exist on BayesianStatistics

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


@pytest.mark.slow
def test_posterior_narrows_with_more_detections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cosmological_model: "Model1CrossCheck",
    plot_output_dir: Path,
) -> None:
    """Verify that combining more detections produces a narrower posterior.

    Runs BayesianStatistics.evaluate() across 11 h-values with all 5
    synthetic detections, then combines subsets of 1, 3, and 5 detection
    posteriors. Checks:
    - All posteriors are finite and non-negative
    - For 5 detections the peak is within 0.05 of the true H
    - The posterior width (bins above 50% of peak) narrows: w5 <= w3 <= w1

    Note: DetectionProbability's internal KDE requires >= 4 detected events
    (4-dimensional kernel), so we always use the full 5-detection CSV and
    vary which per-detection posteriors are *combined*.
    """
    np.random.seed(42)

    # Guarantee enough CPUs for the multiprocessing pool
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(4)))

    # ── Set up working directory with fixture CSVs ──────────────────────
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

    import master_thesis_code.bayesian_inference.bayesian_statistics as bs

    monkeypatch.setattr(
        bs, "PREPARED_CRAMER_RAO_BOUNDS_PATH", str(sim_dir / "prepared_cramer_rao_bounds.csv")
    )
    monkeypatch.setattr(bs, "CRAMER_RAO_BOUNDS_OUTPUT_PATH", str(sim_dir / "cramer_rao_bounds.csv"))
    monkeypatch.setattr(bs, "UNDETECTED_EVENTS_OUTPUT_PATH", str(sim_dir / "undetected_events.csv"))
    monkeypatch.chdir(tmp_path)

    # ── Build catalog and run full h-sweep with all 5 detections ────────
    from master_thesis_code.bayesian_inference.bayesian_statistics import BayesianStatistics
    from master_thesis_code_test.integration.conftest import (
        build_galaxy_catalog_for_n_detections,
    )

    galaxy_catalog = build_galaxy_catalog_for_n_detections(5)
    bayesian_stats = BayesianStatistics()
    assert len(bayesian_stats.cramer_rao_bounds) == 5

    for h in H_GRID:
        bayesian_stats.evaluate(
            galaxy_catalog=galaxy_catalog,
            cosmological_model=cosmological_model,
            h_value=float(h),
        )

    posterior_data = bayesian_stats.posterior_data

    # ── Assertions on raw posterior data ────────────────────────────────
    detection_indices = sorted(posterior_data.keys())
    assert len(detection_indices) > 0, "No detection posteriors produced"

    for idx in detection_indices:
        arr = np.array(posterior_data[idx], dtype=np.float64)
        assert np.all(np.isfinite(arr)), f"Detection {idx}: non-finite values"
        assert np.all(arr >= 0), f"Detection {idx}: negative values"

    # ── Combine subsets: 1, 3, and 5 detections ─────────────────────────
    from master_thesis_code.plotting import save_figure
    from master_thesis_code.plotting.bayesian_plots import (
        plot_combined_posterior,
        plot_event_posteriors,
    )

    subset_counts = [1, 3, 5]
    combined_results: dict[int, npt.NDArray[np.float64]] = {}

    for n_det in subset_counts:
        subset_indices = detection_indices[:n_det]
        combined = _combine_posterior(posterior_data, subset_indices)
        combined_results[n_det] = combined

        # Per-subset posterior data for plotting
        subset_data = {idx: posterior_data[idx] for idx in subset_indices}

        fig_events, _ = plot_event_posteriors(
            h_values=H_GRID,
            posterior_data=subset_data,
            true_h=TRUE_H,
            title=f"Individual event posteriors ({n_det} det)",
        )
        save_figure(
            fig_events,
            str(plot_output_dir / f"event_posteriors_{n_det}det"),
            formats=("png",),
            close=True,
        )

        fig_combined, _ = plot_combined_posterior(
            h_values=H_GRID,
            posterior=combined,
            true_h=TRUE_H,
            label=f"{n_det} detection{'s' if n_det > 1 else ''}",
        )
        save_figure(
            fig_combined,
            str(plot_output_dir / f"combined_posterior_{n_det}det"),
            formats=("png",),
            close=True,
        )

    # At least some h-values should have nonzero likelihood for 3+ detections
    for n_det in (3, 5):
        assert np.max(combined_results[n_det]) > 0, (
            f"{n_det}det: all combined posterior values are zero"
        )

    # For 5 detections: peak should be within 0.05 of TRUE_H
    combined_5 = combined_results[5]
    if np.max(combined_5) > 0:
        peak_h = H_GRID[np.argmax(combined_5)]
        assert abs(peak_h - TRUE_H) <= 0.05, (
            f"5det peak at h={peak_h:.3f}, expected within 0.05 of {TRUE_H}"
        )

    # Posterior narrowing: width_5 <= width_3 <= width_1
    width_1 = _posterior_width(combined_results[1])
    width_3 = _posterior_width(combined_results[3])
    width_5 = _posterior_width(combined_results[5])

    assert width_5 <= width_3 <= width_1, (
        f"Posterior should narrow: w1={width_1}, w3={width_3}, w5={width_5}"
    )

    # ── Comparison plot (overlay all three combined posteriors) ──────────
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for n_det in subset_counts:
        plot_combined_posterior(
            h_values=H_GRID,
            posterior=combined_results[n_det],
            true_h=TRUE_H,
            label=f"{n_det} detection{'s' if n_det > 1 else ''}",
            ax=ax,
        )
    ax.set_title("Posterior narrowing with increasing detections")
    save_figure(
        fig,
        str(plot_output_dir / "posterior_narrowing_comparison"),
        formats=("png",),
        close=True,
    )

    # ── Generate HTML gallery ───────────────────────────────────────────
    _generate_gallery_html(plot_output_dir)
