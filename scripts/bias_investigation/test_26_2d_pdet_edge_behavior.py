"""Test 26: characterize the 2D p_det out-of-grid extrapolation behavior.

Hypothesis (H2 from .planning/HANDOFF-2D-BIAS-INVESTIGATION-20260505.md):
the residual 2D-channel bias on the phase46-merged CRB (z=+37 at h=0.73,
+55 at h=0.60) is driven by raw scipy linear extrapolation of
``detection_probability_with_bh_mass_interpolated`` at the (d_L, M) grid
boundaries.

This script does NOT modify physics.  It only diagnoses the off-grid
behavior of the existing 2D p_det grid against the partial-panel CRB.

What it reports:

1. Grid bounds for h ∈ {0.60, 0.65, 0.70, 0.73} (d_L range, M range)
2. For each h_truth, classifies every event's (d_L, M_meas) as
   in_grid / d_L>max / d_L<min / M>max / M<min / corner, by direction
3. Per-h_trial cell classification (d_L_trial = d_L_meas; M_meas fixed),
   sweeping h_trial across the LamCDM-clamped 21-pt grid
4. For out-of-grid cells: raw scipy interp value, principled-limit value
   (per the asymptote table in the plan), and the difference
5. Comparison against the 1D p_det grid: the 1D path uses an anchor
   (Wilson 95% LB at d_L=0 = 0.7931, intermediate at d_L=0.05 = 1.0)
   and zero-fills above d_L_max.  We report whether 1D queries reach
   the anchored region or the d_L>max region for the same events.

Output: outputs/phase46_merged/2d_pdet_edge_behavior.json + summary
printed to stdout.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from darksiren_emri.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)
from darksiren_emri.constants import INJECTION_DATA_DIR
from darksiren_emri.physical_relations import dist_vectorized  # noqa: F401

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "outputs" / "phase46_merged"
PHASE46_CRB = (
    PROJECT_ROOT / "simulations" / "cluster_run_phase46_merged_20260504" / "cramer_rao_bounds.csv"
)
SNR_THRESHOLD = 20.0
H_TRUTHS = (0.60, 0.65, 0.70, 0.73)
H_GRID_HALF_WIDTH = 0.05
H_GRID_STEP = 0.005


def asymptote_for_direction(direction: str) -> float:
    """Principled out-of-grid asymptote per the plan's table.

    - d_L > d_L_max: 0 (SNR-suppressed)
    - d_L < d_L_min: 1 (SNR-saturated)
    - M > M_max: 0 (event-rate / model cutoff at high M_z)
    - M < M_min: 0 (SNR-suppressed at low M)
    - corner: min of the two face asymptotes (i.e. 0 unless both face
      asymptotes are 1, which only happens at d_L<d_L_min + M<M_min, but
      that combination has principled value 0 from the M side).
    """
    if direction == "in_grid":
        return float("nan")
    if direction == "d_L>max":
        return 0.0
    if direction == "d_L<min":
        return 1.0
    if direction == "M>max":
        return 0.0
    if direction == "M<min":
        return 0.0
    if direction.startswith("corner"):
        # Any corner involves at least one suppressing axis (M>max, M<min,
        # or d_L>max).  The principled value is 0 for all corners.
        return 0.0
    msg = f"unknown direction tag: {direction}"
    raise ValueError(msg)


def classify_cell(
    d_L: float,
    M: float,
    dl_min: float,
    dl_max: float,
    M_min: float,
    M_max: float,
) -> str:
    dl_below = d_L < dl_min
    dl_above = d_L > dl_max
    M_below = M < M_min  # noqa: N806
    M_above = M > M_max  # noqa: N806

    if not (dl_below or dl_above or M_below or M_above):
        return "in_grid"
    if (dl_below or dl_above) and (M_below or M_above):
        # corner case
        dl_tag = "d_L<min" if dl_below else "d_L>max"
        M_tag = "M<min" if M_below else "M>max"  # noqa: N806
        return f"corner:{dl_tag}+{M_tag}"
    if dl_below:
        return "d_L<min"
    if dl_above:
        return "d_L>max"
    if M_below:
        return "M<min"
    return "M>max"


def grid_bounds_2d(sdp: SimulationDetectionProbability, h: float) -> dict[str, float]:
    """Recover (d_L, M) bounds of the 2D p_det grid at h."""
    interp_2d, _ = sdp._get_or_build_grid(h)  # noqa: SLF001 (diagnostic-only access)
    dl_centers = np.asarray(interp_2d.grid[0])
    M_centers = np.asarray(interp_2d.grid[1])  # noqa: N806

    # The 2D grid uses bin-center coordinates.  Use centers as the
    # in-grid range (extrapolation kicks in at the centers, not the bin
    # edges, per scipy semantics).
    return {
        "dl_min": float(dl_centers[0]),
        "dl_max": float(dl_centers[-1]),
        "M_min": float(M_centers[0]),
        "M_max": float(M_centers[-1]),
        "n_dl": int(len(dl_centers)),
        "n_M": int(len(M_centers)),
    }


def grid_bounds_1d(sdp: SimulationDetectionProbability, h: float) -> dict[str, float]:
    """Recover d_L bounds of the 1D p_det grid at h (for comparison)."""
    _, interp_1d = sdp._get_or_build_grid(h)  # noqa: SLF001
    dl_centers = np.asarray(interp_1d.grid[0])
    spacing = float(dl_centers[1] - dl_centers[0])
    return {
        "dl_min": float(dl_centers[0]),
        "dl_max": float(dl_centers[-1]),
        "dl_max_with_half_spacing": float(dl_centers[-1] + spacing / 2),
        "n_dl": int(len(dl_centers)),
    }


def evaluate_pdet_2d_raw(
    sdp: SimulationDetectionProbability,
    h: float,
    d_L: npt.NDArray[np.float64],
    M: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Raw scipy interp value at (d_L, M), no clip.  Diagnostic-only."""
    interp_2d, _ = sdp._get_or_build_grid(h)  # noqa: SLF001
    points = np.column_stack([d_L, M])
    return np.asarray(interp_2d(points), dtype=np.float64)


def evaluate_pdet_2d_clipped(
    sdp: SimulationDetectionProbability,
    h: float,
    d_L: npt.NDArray[np.float64],
    M: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Production p_det_2d call (clipped to [0,1])."""
    return np.asarray(
        sdp.detection_probability_with_bh_mass_interpolated(
            d_L,
            M,
            np.zeros_like(d_L),
            np.zeros_like(d_L),
            h=h,
        ),
        dtype=np.float64,
    )


def analyze_one_truth(
    sdp: SimulationDetectionProbability,
    crb: pd.DataFrame,
    h_truth: float,
) -> dict[str, Any]:
    """Edge-behavior diagnostic at one h_truth."""
    n_events = len(crb)
    print(f"\n=== h_truth = {h_truth:.3f}  ({n_events} events) ===")

    # Build h-grid identical to test_24's LamCDM-clamped 21-pt grid
    half = H_GRID_HALF_WIDTH
    step = H_GRID_STEP
    h_grid = np.round(np.arange(h_truth - half, h_truth + half + step / 2, step), 4)
    h_grid = h_grid[(h_grid >= 0.60) & (h_grid <= 0.86)]
    print(f"  h_grid: [{h_grid[0]:.3f}..{h_grid[-1]:.3f}] ({len(h_grid)} points)")

    M_meas = crb["M"].values.astype(np.float64)  # noqa: N806
    d_L_meas = crb["luminosity_distance"].values.astype(np.float64)  # noqa: N806

    # Per-h_trial classification.  We use the event's measured d_L_meas
    # directly (this is approximately what the integrand probes — the
    # integrand evaluates p_det at d_L = dist(z_gal, h_trial) for hosts
    # near the event, which is centered on d_L_meas regardless of
    # h_trial).  For M, we use M_meas (= _det_M in the integrand —
    # constant in h).
    rows: list[dict[str, Any]] = []
    direction_counts_per_h: dict[float, dict[str, int]] = {}
    raw_disagreement_summary: list[dict[str, Any]] = []

    for h_trial in h_grid:
        bounds = grid_bounds_2d(sdp, float(h_trial))
        directions = [
            classify_cell(
                d_L_meas[i],
                M_meas[i],
                bounds["dl_min"],
                bounds["dl_max"],
                bounds["M_min"],
                bounds["M_max"],
            )
            for i in range(n_events)
        ]
        # Aggregate by direction
        dir_counts: dict[str, int] = {}
        for d in directions:
            dir_counts[d] = dir_counts.get(d, 0) + 1
        direction_counts_per_h[float(h_trial)] = dir_counts

        # For the OUT-of-grid cells, evaluate raw scipy and principled
        # asymptote, log the disagreement.
        out_mask = np.array([d != "in_grid" for d in directions])
        if out_mask.sum() == 0:
            continue
        raw_vals = evaluate_pdet_2d_raw(sdp, float(h_trial), d_L_meas[out_mask], M_meas[out_mask])
        clipped_vals = evaluate_pdet_2d_clipped(
            sdp, float(h_trial), d_L_meas[out_mask], M_meas[out_mask]
        )
        principled_vals = np.array(
            [asymptote_for_direction(d) for d in directions if d != "in_grid"],
            dtype=np.float64,
        )
        # Summary of disagreements |raw − principled|
        raw_disagreement_summary.append(
            {
                "h_trial": float(h_trial),
                "n_out_of_grid": int(out_mask.sum()),
                "raw_min": float(raw_vals.min()) if len(raw_vals) else None,
                "raw_max": float(raw_vals.max()) if len(raw_vals) else None,
                "raw_mean": float(raw_vals.mean()) if len(raw_vals) else None,
                "raw_below_zero": int((raw_vals < 0).sum()),
                "raw_above_one": int((raw_vals > 1).sum()),
                "clipped_mean": float(clipped_vals.mean()) if len(clipped_vals) else None,
                "principled_mean": float(principled_vals.mean()) if len(principled_vals) else None,
                "abs_clipped_minus_principled_mean": (
                    float(np.mean(np.abs(clipped_vals - principled_vals)))
                    if len(clipped_vals)
                    else None
                ),
                "abs_clipped_minus_principled_max": (
                    float(np.max(np.abs(clipped_vals - principled_vals)))
                    if len(clipped_vals)
                    else None
                ),
            }
        )

    # Top-level summary at h_truth
    grid_bounds_truth = grid_bounds_2d(sdp, float(h_truth))
    grid_bounds_truth_1d = grid_bounds_1d(sdp, float(h_truth))

    # Aggregate fractions across h_grid: mean over h_trial
    all_dir_keys: set[str] = set()
    for cd in direction_counts_per_h.values():
        all_dir_keys.update(cd.keys())
    fraction_summary: dict[str, dict[str, float]] = {}
    for d in all_dir_keys:
        counts = [direction_counts_per_h[h].get(d, 0) for h in direction_counts_per_h]
        fracs = [c / n_events for c in counts]
        fraction_summary[d] = {
            "min_frac": float(min(fracs)),
            "max_frac": float(max(fracs)),
            "mean_frac": float(np.mean(fracs)),
        }

    # Print summary
    print(
        f"  2D grid bounds: d_L ∈ [{grid_bounds_truth['dl_min']:.3f}, "
        f"{grid_bounds_truth['dl_max']:.3f}], "
        f"M ∈ [{grid_bounds_truth['M_min']:.3e}, {grid_bounds_truth['M_max']:.3e}]"
    )
    print(f"  Event d_L range: [{d_L_meas.min():.3f}, {d_L_meas.max():.3f}]")
    print(f"  Event M range: [{M_meas.min():.3e}, {M_meas.max():.3e}]")
    for d, stats in sorted(fraction_summary.items()):
        print(
            f"  {d:30s}  min={stats['min_frac']:.3f}  "
            f"mean={stats['mean_frac']:.3f}  max={stats['max_frac']:.3f}"
        )

    if raw_disagreement_summary:
        max_disagreement = max(
            (r["abs_clipped_minus_principled_max"] or 0.0 for r in raw_disagreement_summary),
            default=0.0,
        )
        print(
            f"  Max |clipped − principled| over h_grid × out-of-grid events: {max_disagreement:.4f}"
        )

    return {
        "h_truth": h_truth,
        "n_events": n_events,
        "grid_bounds_2d_at_h_truth": grid_bounds_truth,
        "grid_bounds_1d_at_h_truth": grid_bounds_truth_1d,
        "h_grid": h_grid.tolist(),
        "fraction_summary_per_direction": fraction_summary,
        "per_h_trial_direction_counts": direction_counts_per_h,
        "raw_vs_principled_per_h": raw_disagreement_summary,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("TEST 26 — 2D p_det out-of-grid extrapolation diagnostic (Step 1b)")
    print("=" * 72)

    print(f"\nLoading phase46-merged CRB: {PHASE46_CRB}")
    crb = pd.read_csv(PHASE46_CRB)
    print(f"  raw rows: {len(crb)}")
    crb = crb[crb["SNR"] >= SNR_THRESHOLD].reset_index(drop=True)
    print(f"  SNR>={SNR_THRESHOLD}: {len(crb)}")

    sdp = SimulationDetectionProbability(
        injection_data_dir=str(PROJECT_ROOT / INJECTION_DATA_DIR),
        snr_threshold=SNR_THRESHOLD,
    )

    results: dict[str, Any] = {
        "snr_threshold": SNR_THRESHOLD,
        "h_truths_analyzed": list(H_TRUTHS),
        "crb_path": str(PHASE46_CRB.relative_to(PROJECT_ROOT)),
        "n_events_post_snr_cut": len(crb),
        "per_truth": [],
    }

    for h_truth in H_TRUTHS:
        results["per_truth"].append(analyze_one_truth(sdp, crb, h_truth))

    out_path = OUTPUT_DIR / "2d_pdet_edge_behavior.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
