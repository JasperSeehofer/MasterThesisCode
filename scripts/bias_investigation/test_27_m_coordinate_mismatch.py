"""Test 27: pre-implementation diagnostic for the H3 fix.

Quantifies the bug surfaced during the 2026-05-05 plan-mode review of the
H3 plan (`docs/H0_BIAS_RESOLUTION.md` §4.7):

  (1) Numerator queries observation `_det_M` instead of hypothesis
      `host_M·(1+z)` at integration z.
  (2) Grid axis is binned in source-frame M while queries pass
      observer-frame M_z.

Both issues are fixed together by Option A: build grid in M_z everywhere
and pass `host_M·(1+z)` (varying with integration z) in the numerator.

This script does NOT modify production code.  It builds two
``SimulationDetectionProbability`` instances differing only in the
M-axis grid coordinate, queries both at sample integration points per
detection in the phase46-merged CRB, and reports the predicted joint
MAP shift if Option A is implemented.

Decision gate G_H3a:
- if predicted Δh has sign + magnitude consistent with closing the
  +0.0141 residual (Δh ≈ -0.013), proceed to Step 2 of the plan.
- if predicted shift has wrong sign, the framing has an analytical
  error — pause and re-derive.
- if predicted magnitude is much smaller than 0.01, H3 is not dominant
  — pivot to H3b (entropy mismatch) or H1 (realization-bootstrap).

Output: outputs/phase46_merged/2d_m_coordinate_mismatch.json + summary
to stdout.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import OrderedDict
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
from darksiren_emri.physical_relations import dist_vectorized

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "outputs" / "phase46_merged"
PHASE46_CRB = (
    PROJECT_ROOT / "simulations" / "cluster_run_phase46_merged_20260504" / "cramer_rao_bounds.csv"
)
SNR_THRESHOLD = 20.0
H_TRUTH = 0.73


def build_proposed_sdp(
    injection_data_dir: str, snr_threshold: float
) -> SimulationDetectionProbability:
    """Build an SDP with the M-axis in observer-frame M_z (Option A).

    Constructs a normal SDP, then monkeypatches ``_M_arr`` to be the
    observer-frame redshifted mass ``M_source · (1 + z_inj)`` for each
    injection.  Clears the grid cache so subsequent ``_get_or_build_grid``
    calls see the new M coordinate.
    """
    sdp = SimulationDetectionProbability(
        injection_data_dir=injection_data_dir,
        snr_threshold=snr_threshold,
    )
    # Replace source-frame M with observer-frame M_z.
    sdp._M_arr = sdp._M_arr * (1.0 + sdp._z_arr)  # noqa: SLF001
    sdp._grid_cache = OrderedDict()  # noqa: SLF001
    sdp._quality_flags = {}  # noqa: SLF001
    return sdp


def query_pdet(
    sdp: SimulationDetectionProbability,
    d_L: npt.NDArray[np.float64],
    M_arg: npt.NDArray[np.float64],
    h: float,
) -> npt.NDArray[np.float64]:
    """Production call to detection_probability_with_bh_mass_interpolated."""
    return np.asarray(
        sdp.detection_probability_with_bh_mass_interpolated(
            d_L,
            M_arg,
            np.zeros_like(d_L),
            np.zeros_like(d_L),
            h=h,
        ),
        dtype=np.float64,
    )


def grid_M_range(sdp: SimulationDetectionProbability, h: float) -> tuple[float, float]:
    interp_2d, _ = sdp._get_or_build_grid(h)  # noqa: SLF001
    M_centers = np.asarray(interp_2d.grid[1])  # noqa: N806
    return float(M_centers[0]), float(M_centers[-1])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("TEST 27 — H3 pre-implementation diagnostic")
    print("=" * 72)
    print(f"\nLoading phase46-merged CRB: {PHASE46_CRB}")
    crb = pd.read_csv(PHASE46_CRB)
    crb = crb[crb["SNR"] >= SNR_THRESHOLD].reset_index(drop=True)
    n_events = len(crb)
    print(f"  SNR>={SNR_THRESHOLD}: {n_events} events")

    print("\nBuilding production SDP (current: M-axis = M_source) ...")
    sdp_current = SimulationDetectionProbability(
        injection_data_dir=str(PROJECT_ROOT / INJECTION_DATA_DIR),
        snr_threshold=SNR_THRESHOLD,
    )

    print("Building proposed SDP (Option A: M-axis = M_z = M_source·(1+z_inj)) ...")
    sdp_proposed = build_proposed_sdp(str(PROJECT_ROOT / INJECTION_DATA_DIR), SNR_THRESHOLD)

    M_min_curr, M_max_curr = grid_M_range(sdp_current, H_TRUTH)  # noqa: N806
    M_min_prop, M_max_prop = grid_M_range(sdp_proposed, H_TRUTH)  # noqa: N806
    print(f"\nGrid M ranges at h={H_TRUTH}:")
    print(f"  current  (M_source axis): [{M_min_curr:.3e}, {M_max_curr:.3e}]")
    print(f"  proposed (M_z axis)     : [{M_min_prop:.3e}, {M_max_prop:.3e}]")
    print(f"  ratio (max/max)         : {M_max_prop / M_max_curr:.3f}")

    # Per-detection: estimate the host's central z via z = dist^{-1}(d_L, h).
    # Inverted via 1D interpolation on a pre-computed (z, d_L) table.
    dL_meas = crb["luminosity_distance"].values.astype(np.float64)  # noqa: N806
    M_meas = crb["M"].values.astype(np.float64)  # noqa: N806 (observer-frame ML M_z)
    z_table = np.linspace(1e-4, 5.0, 2000)
    dL_table = dist_vectorized(z_table, h=H_TRUTH)  # noqa: N806
    z_central = np.interp(dL_meas, dL_table, z_table).astype(np.float64)

    # Proxy for host_M (catalog source-frame mass): M_meas / (1+z_central)
    # In the production integrand host_M is sampled from the GLADE catalog;
    # here we use the truth-consistent estimate which is what the catalog
    # would converge to for a perfectly matched host.
    host_M_proxy = M_meas / (1.0 + z_central)  # noqa: N806

    # Sample integration z values: ±2σ_z window around z_central.
    # σ_z ≈ 0.013·(1+z)^3 (historical Pipeline-A galaxy model, removed with
    # datamodels/galaxy.py in the 2026-07-04 dead-code cleanup; production z-errors
    # come from the GLADE+ catalogue + the σ_v PV term in bayesian_statistics).
    sigma_z = 0.013 * (1.0 + z_central) ** 3  # noqa
    z_offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    # Per-detection integration grid: shape (n_events, n_offsets)
    z_int = z_central[:, None] + sigma_z[:, None] * z_offsets[None, :]
    z_int = np.clip(z_int, 1e-4, None)  # avoid z<=0

    # For each integration z, compute d_L(z, h) and the queries:
    # Current behaviour: M-arg = _det_M (constant in z)
    # Proposed behaviour: M-arg = host_M_proxy · (1+z) (varies with z)
    n_off = z_offsets.size
    delta_pdet_per_event = np.zeros(n_events, dtype=np.float64)
    delta_pdet_central = np.zeros(n_events, dtype=np.float64)
    pdet_current_central = np.zeros(n_events, dtype=np.float64)
    pdet_proposed_central = np.zeros(n_events, dtype=np.float64)

    for j in range(n_off):
        z_j = z_int[:, j]
        dL_j = dist_vectorized(z_j, h=H_TRUTH)  # noqa: N806

        M_arg_current = M_meas  # constant in z (current production behaviour)  # noqa: N806
        M_arg_proposed = host_M_proxy * (1.0 + z_j)  # noqa: N806

        p_curr = query_pdet(sdp_current, dL_j, M_arg_current, H_TRUTH)
        p_prop = query_pdet(sdp_proposed, dL_j, M_arg_proposed, H_TRUTH)

        delta_pdet_per_event += (p_prop - p_curr) / n_off  # mean over offsets
        if z_offsets[j] == 0.0:  # central point
            delta_pdet_central[:] = p_prop - p_curr
            pdet_current_central[:] = p_curr
            pdet_proposed_central[:] = p_prop

    # Per-event log-likelihood proxy:
    # the numerator integrand is roughly ∝ p_det · (Gaussian terms);
    # the Gaussian is sharply peaked at z_central, so the integrand's
    # mean p_det ≈ p_det at z_central.  Then num_proposed/num_current ≈
    # p_det_proposed_central / p_det_current_central, and Δlog num_i ≈
    # log(p_det_proposed/p_det_current) per event.
    # (The denominator's p_det also changes; assuming partial cancellation
    # in the ratio, we estimate the leading effect from the numerator.)
    eps = 1e-8
    log_ratio_central = np.log(np.maximum(pdet_proposed_central, eps)) - np.log(
        np.maximum(pdet_current_central, eps)
    )

    # Aggregate predicted MAP shift via Laplace approximation:
    # log P(h|D) ≈ Σ log L_i(h) + ... with Σ peaked near h_truth.
    # If a uniform-in-h shift Δh moves all per-event log L_i by their
    # mean value, then Σ Δlog L_i = N · mean(Δlog L_i).  At the MAP,
    # the second derivative gives σ_h^2 ≈ -1 / (d²log P / dh²).
    # The σ_boot estimate from the post-bridge-fix R1 was ≈0.0039
    # for 2D at 1473 events.  Predicted MAP shift Δh ≈
    #   (Σ Δlog L_i) / (-d²log P / dh²) · sign per event
    # but the simpler order-of-magnitude is Δh ≈ σ_boot · z_shift where
    # z_shift = -mean(Δlog L_i / σ_per_event_approx) — too fiddly to
    # compute precisely without the full sweep.  Report instead the
    # mean log-ratio and a heuristic MAP shift estimate.

    mean_log_ratio = float(np.mean(log_ratio_central))
    median_log_ratio = float(np.median(log_ratio_central))

    # Heuristic: if the proposed code shifts each per-event log L by
    # Δlog L (h_truth) compared to current, then the MAP shifts by
    # approximately Δh ≈ Δlog L · σ_boot² · (∂log L / ∂h)^{-1} per
    # event, summed.  A simpler bound: |Δh| ≲ σ_boot if the bias
    # collapses to truth.  Report mean log-ratio · sign.

    print("\n=== Per-event Δp_det at integration centre (current → proposed) ===")
    print(f"  mean Δp_det               : {float(np.mean(delta_pdet_central)):+.4f}")
    print(f"  median Δp_det             : {float(np.median(delta_pdet_central)):+.4f}")
    print(
        f"  Δp_det 25–75 percentile   : "
        f"[{float(np.percentile(delta_pdet_central, 25)):+.4f}, "
        f"{float(np.percentile(delta_pdet_central, 75)):+.4f}]"
    )
    print(f"  fraction with Δp_det > 0  : {float(np.mean(delta_pdet_central > 0)):.3f}")
    print(f"  fraction with |Δp_det|>0.05: {float(np.mean(np.abs(delta_pdet_central) > 0.05)):.3f}")

    print("\n=== Per-event log(p_det_proposed / p_det_current) at central z ===")
    print(f"  mean log-ratio            : {mean_log_ratio:+.4f}")
    print(f"  median log-ratio          : {median_log_ratio:+.4f}")
    print(f"  total Σ log-ratio over N  : {float(np.sum(log_ratio_central)):+.2f}")

    # Predicted MAP shift heuristic (rough):
    # σ_boot post-bridge-fix at h=0.73 / 2D / 1473 events = 0.0039
    # Joint log L curvature ~ N / σ_boot² ≈ 1473 / 0.0039² ≈ 9.7e7 per unit h².
    # Mean Δlog L per event ≈ mean_log_ratio.
    # Total Δlog L ≈ N · mean_log_ratio.
    # Predicted MAP shift Δh ≈ -Total Δlog L / (Σ ∂²log L / ∂h²).
    # Without knowing the sign of ∂log L / ∂h, we can only bound
    # |Δh| ≲ σ_boot · |z_equiv| where z_equiv = mean_log_ratio · √N / σ_boot.
    #
    # Order-of-magnitude bound: if the MAP currently sits at +3.6σ_boot
    # (=+0.0141), and the proposed change moves the integrand systematically,
    # then the MAP shift is bounded by the magnitude of the systematic
    # divided by the curvature.

    sigma_boot_postbridge_2d = 0.0039
    n = n_events

    # Heuristic: per-event log-ratio of magnitude γ shifts MAP by
    # approximately Δh ≈ γ / |∂log L / ∂h|_{per event}.
    # |∂log L / ∂h|_{per event} at the MAP ~ 1 / (σ_per_event · √N)
    # where σ_per_event ≈ σ_boot · √N.
    # Combining: Δh ≈ γ · σ_boot · √N.
    # This gives a rough scale; sign requires actual posterior re-evaluation.

    delta_h_scale_estimate = abs(mean_log_ratio) * sigma_boot_postbridge_2d * np.sqrt(n)

    print("\n=== Order-of-magnitude predicted MAP shift ===")
    print(f"  σ_boot post-bridge-fix (2D, h=0.73)  : {sigma_boot_postbridge_2d:.4f}")
    print(f"  |Δh| scale estimate                  : {delta_h_scale_estimate:.4f}")
    print("  (target: closing residual ≈ 0.0141)")

    out: dict[str, Any] = {
        "snr_threshold": SNR_THRESHOLD,
        "h_truth": H_TRUTH,
        "n_events": n_events,
        "crb_path": str(PHASE46_CRB.relative_to(PROJECT_ROOT)),
        "grid_M_range": {
            "current": [M_min_curr, M_max_curr],
            "proposed": [M_min_prop, M_max_prop],
            "ratio_max": M_max_prop / M_max_curr,
        },
        "delta_pdet_central": {
            "mean": float(np.mean(delta_pdet_central)),
            "median": float(np.median(delta_pdet_central)),
            "p25": float(np.percentile(delta_pdet_central, 25)),
            "p75": float(np.percentile(delta_pdet_central, 75)),
            "frac_positive": float(np.mean(delta_pdet_central > 0)),
            "frac_abs_gt_005": float(np.mean(np.abs(delta_pdet_central) > 0.05)),
        },
        "delta_pdet_window_mean": {
            "mean": float(np.mean(delta_pdet_per_event)),
            "median": float(np.median(delta_pdet_per_event)),
        },
        "log_ratio_central": {
            "mean": mean_log_ratio,
            "median": median_log_ratio,
            "sum_over_N": float(np.sum(log_ratio_central)),
        },
        "predicted_map_shift": {
            "sigma_boot_post_bridge_2d": sigma_boot_postbridge_2d,
            "abs_delta_h_scale_estimate": float(delta_h_scale_estimate),
            "target_residual_closure": 0.0141,
        },
    }

    out_path = OUTPUT_DIR / "2d_m_coordinate_mismatch.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
