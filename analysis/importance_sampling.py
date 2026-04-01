"""Importance sampling utilities for P_det grid construction.

Provides the self-normalized IS estimator, Kish effective sample size,
Wilson confidence intervals adapted for weighted samples, and diagnostic
functions for weight quality assessment.

Core estimator: Tiwari (2018), arXiv:1712.00482, Eq. 5-8.
Effective sample size: Kish (1965), Survey Sampling.
Wilson CI: Brown, Cai, DasGupta (2001) Stat. Sci. 16:101-133,
           with N_eff substitution per Owen (2013) Monte Carlo theory.

Conventions:
  - SI units: distances in Gpc, masses in solar masses, h dimensionless
  - Weights w_i = p(theta_i) / q(theta_i), dimensionless, positive
  - P_det dimensionless, in [0, 1]
  - N_eff dimensionless, 0 < N_eff <= N for non-empty bins
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SNR_THRESHOLD: float = 15.0


# ---------------------------------------------------------------------------
# Core IS functions
# ---------------------------------------------------------------------------
def weighted_histogram_estimator(
    detected_weights: npt.NDArray[np.float64],
    total_weights: npt.NDArray[np.float64],
) -> float:
    """Self-normalized importance sampling estimator for detection probability.

    Computes P_hat(B) = sum(detected_weights) / sum(total_weights).

    When all weights are 1.0, this reduces to N_det / N_total exactly.

    Args:
        detected_weights: Weights of detected events in the bin.
        total_weights: Weights of all events in the bin.

    Returns:
        Estimated detection probability in [0, 1].

    References:
        Tiwari (2018), arXiv:1712.00482, Eq. 5-8.
    """
    # Eq. 5-8 of Tiwari (2018): P_hat = sum(w_i * I_i) / sum(w_i)
    sum_total = np.sum(total_weights)
    if sum_total == 0.0:
        return 0.0
    return float(np.sum(detected_weights) / sum_total)


def kish_n_eff(weights: npt.NDArray[np.float64]) -> float:
    """Kish effective sample size from importance weights.

    Computes N_eff = (sum(w))^2 / sum(w^2).

    When all weights are equal, N_eff = len(weights) exactly.
    Proof: if w_i = c for all i, then (n*c)^2 / (n*c^2) = n.

    Args:
        weights: Array of importance weights (positive).

    Returns:
        Effective sample size. Zero if weights is empty or all-zero.

    References:
        Kish (1965), Survey Sampling, Chapter 11.
    """
    if len(weights) == 0:
        return 0.0
    sum_w = np.sum(weights)
    if sum_w == 0.0:
        return 0.0
    sum_w2 = np.sum(weights**2)
    # N_eff = (sum w)^2 / sum(w^2)
    return float(sum_w**2 / sum_w2)


def is_weighted_wilson_ci(
    detected_weights: npt.NDArray[np.float64],
    total_weights: npt.NDArray[np.float64],
    confidence_level: float = 0.9545,
) -> tuple[float, float]:
    """Wilson confidence interval adapted for effective sample size.

    Uses N_eff in place of n in the Wilson formula, giving wider CIs when
    weights are unequal (correctly reflecting reduced information).

    When all weights are 1.0, this gives the same CI as the standard
    Wilson score interval.

    Args:
        detected_weights: Weights of detected events in the bin.
        total_weights: Weights of all events in the bin.
        confidence_level: Confidence level (default 0.9545 = 95.45%).

    Returns:
        Tuple (ci_lower, ci_upper) clipped to [0, 1].

    References:
        Brown, Cai, DasGupta (2001) Stat. Sci. 16:101-133 (Wilson score).
        Owen (2013) Monte Carlo theory, N_eff substitution.
    """
    from scipy.stats import norm

    p_hat = weighted_histogram_estimator(detected_weights, total_weights)
    n_eff = kish_n_eff(total_weights)

    if n_eff < 1.0:
        return (0.0, 1.0)

    z = norm.ppf(0.5 + confidence_level / 2.0)
    z2 = z * z

    # Wilson score interval with n -> N_eff
    # Eq: (p_hat + z^2/(2n) +/- z*sqrt(p_hat*(1-p_hat)/n + z^2/(4n^2))) / (1 + z^2/n)
    denom = 1.0 + z2 / n_eff
    center = (p_hat + z2 / (2.0 * n_eff)) / denom
    half_width = z * np.sqrt(p_hat * (1.0 - p_hat) / n_eff + z2 / (4.0 * n_eff**2)) / denom

    ci_lower = float(np.clip(center - half_width, 0.0, 1.0))
    ci_upper = float(np.clip(center + half_width, 0.0, 1.0))
    return (ci_lower, ci_upper)


# ---------------------------------------------------------------------------
# Diagnostic functions
# ---------------------------------------------------------------------------
def weight_diagnostic(
    weights: npt.NDArray[np.float64],
    bin_indices: npt.NDArray[np.intp],
    n_detected_per_bin: npt.NDArray[np.float64],
) -> dict[str, Any]:
    """Compute per-bin and global weight quality diagnostics.

    Args:
        weights: Per-injection importance weights, shape (N,).
        bin_indices: Flat bin index for each injection, shape (N,).
            -1 for out-of-range events.
        n_detected_per_bin: Number of detected events per bin (flat).

    Returns:
        Dict with keys:
          - ``per_bin``: list of dicts with mean_weight, max_weight,
            max_mean_ratio, n_eff, n_total, n_eff_ratio.
          - ``global_n_eff``: total effective sample size.
          - ``global_farr_pass``: bool, whether sum(n_eff) > 4 * sum(n_det).
          - ``red_flags``: list of bin indices with max/mean ratio > 100.
    """
    n_bins = int(np.max(bin_indices) + 1) if len(bin_indices) > 0 else 0
    per_bin: list[dict[str, float]] = []
    total_n_eff = 0.0

    for b in range(n_bins):
        mask = bin_indices == b
        w_bin = weights[mask]
        if len(w_bin) == 0:
            per_bin.append(
                {
                    "mean_weight": 0.0,
                    "max_weight": 0.0,
                    "max_mean_ratio": 0.0,
                    "n_eff": 0.0,
                    "n_total": 0,
                    "n_eff_ratio": 0.0,
                }
            )
            continue
        mean_w = float(np.mean(w_bin))
        max_w = float(np.max(w_bin))
        n_eff_bin = kish_n_eff(w_bin)
        n_total = len(w_bin)
        total_n_eff += n_eff_bin
        per_bin.append(
            {
                "mean_weight": mean_w,
                "max_weight": max_w,
                "max_mean_ratio": max_w / mean_w if mean_w > 0 else 0.0,
                "n_eff": n_eff_bin,
                "n_total": n_total,
                "n_eff_ratio": n_eff_bin / n_total if n_total > 0 else 0.0,
            }
        )

    total_n_det = float(np.sum(n_detected_per_bin))
    global_farr_pass = total_n_eff > 4.0 * total_n_det

    red_flags = [i for i, d in enumerate(per_bin) if d["max_mean_ratio"] > 100.0]

    return {
        "per_bin": per_bin,
        "global_n_eff": total_n_eff,
        "global_farr_pass": global_farr_pass,
        "red_flags": red_flags,
    }


def farr_criterion_check(
    n_eff_grid: npt.NDArray[np.float64],
    n_detected_grid: npt.NDArray[np.float64],
) -> dict[str, Any]:
    """Check Farr (2019) criterion: N_eff > 4 * N_det.

    Args:
        n_eff_grid: Per-bin effective sample size, shape (dl_bins, M_bins).
        n_detected_grid: Per-bin detected count, shape (dl_bins, M_bins).

    Returns:
        Dict with keys:
          - ``global_pass``: bool, sum(N_eff) > 4 * sum(N_det).
          - ``per_bin_pass_fraction``: fraction of bins with n_det > 0
            where N_eff > 4 * N_det.
          - ``worst_bin``: (i, j) index of the bin with smallest
            N_eff / N_det ratio (among bins with N_det > 0).
          - ``worst_ratio``: the smallest N_eff / N_det ratio.

    References:
        Farr (2019), arXiv:1904.10879.
    """
    total_n_eff = float(np.sum(n_eff_grid))
    total_n_det = float(np.sum(n_detected_grid))

    global_pass = total_n_eff > 4.0 * total_n_det

    # Per-bin check (only where n_det > 0)
    active = n_detected_grid > 0
    if not np.any(active):
        return {
            "global_pass": global_pass,
            "per_bin_pass_fraction": 1.0,
            "worst_bin": (0, 0),
            "worst_ratio": float("inf"),
        }

    ratios = np.full_like(n_eff_grid, np.inf)
    ratios[active] = n_eff_grid[active] / n_detected_grid[active]

    per_bin_pass = np.sum(ratios[active] > 4.0)
    per_bin_pass_fraction = float(per_bin_pass / np.sum(active))

    worst_flat = int(np.argmin(ratios))
    worst_bin = np.unravel_index(worst_flat, ratios.shape)
    worst_ratio = float(ratios[worst_bin])

    return {
        "global_pass": global_pass,
        "per_bin_pass_fraction": per_bin_pass_fraction,
        "worst_bin": worst_bin,
        "worst_ratio": worst_ratio,
    }


# ---------------------------------------------------------------------------
# Verification / __main__ block
# ---------------------------------------------------------------------------
def _load_injection_data(
    csv_dir: str,
) -> dict[float, pd.DataFrame]:
    """Load injection CSVs grouped by h value.

    Args:
        csv_dir: Directory containing injection CSV files.

    Returns:
        Dict mapping h_value -> concatenated DataFrame.
    """
    patterns = [
        f"{csv_dir}/injection_h_*_task_*.csv",
        f"{csv_dir}/injection_h_*.csv",
    ]
    csv_files: list[str] = []
    for pat in patterns:
        csv_files.extend(glob.glob(pat))
    csv_files = sorted(set(csv_files))

    h_file_map: dict[float, list[str]] = {}
    h_pattern = re.compile(r"injection_h_(\d+p\d+)")
    for f in csv_files:
        match = h_pattern.search(f)
        if match:
            h_val = float(match.group(1).replace("p", "."))
            h_file_map.setdefault(h_val, []).append(f)

    result: dict[float, pd.DataFrame] = {}
    for h_val in sorted(h_file_map.keys()):
        dfs = [pd.read_csv(f) for f in h_file_map[h_val]]
        result[h_val] = pd.concat(dfs, ignore_index=True)
    return result


def run_uniform_weight_verification(csv_dir: str) -> bool:
    """Verify IS estimator with uniform weights matches standard N_det/N_total.

    This is the DECISIVE backward-compatibility test.

    Args:
        csv_dir: Directory containing injection CSV files.

    Returns:
        True if max absolute difference < 1e-14 for all h-values.
    """
    from master_thesis_code.bayesian_inference.simulation_detection_probability import (
        SimulationDetectionProbability,
    )

    all_pass = True

    # Build grids with default (no-weight) path
    sdp_default = SimulationDetectionProbability(csv_dir, SNR_THRESHOLD)

    # Build grids with explicit weights=1
    sdp_weighted = SimulationDetectionProbability(
        csv_dir,
        SNR_THRESHOLD,
        _force_unit_weights=True,
    )

    print("\n=== Uniform-Weight Recovery Test (DECISIVE) ===")
    print(
        f"{'h':>6s}  {'N_total':>8s}  {'N_det':>6s}  {'max|diff|':>12s}  "
        f"{'N_eff==n_total':>14s}  {'Farr':>5s}"
    )
    print("-" * 65)

    for h_val in sdp_default._h_grid:
        flags_default = sdp_default.quality_flags(h_val)
        flags_weighted = sdp_weighted.quality_flags(h_val)

        p_det_default = flags_default["n_detected"] / np.where(
            flags_default["n_total"] > 0, flags_default["n_total"], 1.0
        )
        p_det_default = np.where(flags_default["n_total"] > 0, p_det_default, 0.0)

        p_det_weighted = flags_weighted["n_detected"] / np.where(
            flags_weighted["n_total"] > 0, flags_weighted["n_total"], 1.0
        )
        p_det_weighted = np.where(flags_weighted["n_total"] > 0, p_det_weighted, 0.0)

        max_diff = float(np.max(np.abs(p_det_default - p_det_weighted)))

        # Check N_eff = n_total for uniform weights
        n_eff = flags_weighted["n_eff"]
        n_total_int = flags_default["n_total"]
        n_eff_match = float(np.max(np.abs(n_eff - n_total_int.astype(np.float64))))
        n_eff_ok = n_eff_match < 1e-14

        # Farr criterion
        farr = farr_criterion_check(n_eff, flags_weighted["n_detected"])

        n_total = int(np.sum(flags_default["n_total"]))
        n_det = int(np.sum(flags_default["n_detected"]))

        pass_str = "PASS" if max_diff < 1e-14 and n_eff_ok else "FAIL"
        if max_diff >= 1e-14 or not n_eff_ok:
            all_pass = False

        print(
            f"{h_val:6.2f}  {n_total:8d}  {n_det:6d}  {max_diff:12.2e}  "
            f"{'YES' if n_eff_ok else 'NO':>14s}  "
            f"{'YES' if farr['global_pass'] else 'NO':>5s}  {pass_str}"
        )

    print()
    if all_pass:
        print("RESULT: ALL PASSED -- IS estimator is backward-compatible")
    else:
        print("RESULT: FAILED -- IS estimator does NOT match standard estimator")

    return all_pass


def run_synthetic_weight_test() -> bool:
    """Test IS estimator with known non-uniform weights.

    Example: bin has events with weights [2, 1, 1, 1] and
    detected=[True, False, True, False].
    P_det = (2+1)/(2+1+1+1) = 3/5 = 0.6, not 2/4 = 0.5.
    N_eff = (2+1+1+1)^2 / (4+1+1+1) = 25/7 ~ 3.571.

    Returns:
        True if all synthetic tests pass.
    """
    print("\n=== Synthetic Non-Uniform Weight Test ===")

    all_pass = True

    # Test 1: Basic non-uniform weights
    total_w = np.array([2.0, 1.0, 1.0, 1.0])
    detected_w = np.array([2.0, 0.0, 1.0, 0.0])  # events 0 and 2 detected
    # Only detected events contribute to detected_weights
    det_w = total_w[np.array([True, False, True, False])]

    p_hat = weighted_histogram_estimator(det_w, total_w)
    expected_p = 3.0 / 5.0
    diff = abs(p_hat - expected_p)
    ok = diff < 1e-15
    print(
        f"  Test 1: P_det = {p_hat:.6f} (expected {expected_p:.6f}), "
        f"diff = {diff:.2e} {'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        all_pass = False

    # Test 2: N_eff for non-uniform weights
    n_eff = kish_n_eff(total_w)
    expected_neff = 25.0 / 7.0
    diff_neff = abs(n_eff - expected_neff)
    ok_neff = diff_neff < 1e-14
    print(
        f"  Test 2: N_eff = {n_eff:.6f} (expected {expected_neff:.6f}), "
        f"diff = {diff_neff:.2e} {'PASS' if ok_neff else 'FAIL'}"
    )
    if not ok_neff:
        all_pass = False

    # Test 3: Uniform weights reduce to standard estimator
    total_uniform = np.ones(100)
    det_uniform = np.ones(30)  # 30 detected out of 100
    p_hat_uniform = weighted_histogram_estimator(det_uniform, total_uniform)
    expected_uniform = 0.3
    diff_uniform = abs(p_hat_uniform - expected_uniform)
    ok_uniform = diff_uniform < 1e-15
    print(
        f"  Test 3: Uniform P_det = {p_hat_uniform:.6f} (expected {expected_uniform:.6f}), "
        f"diff = {diff_uniform:.2e} {'PASS' if ok_uniform else 'FAIL'}"
    )
    if not ok_uniform:
        all_pass = False

    # Test 4: N_eff = N for uniform weights
    n_eff_uniform = kish_n_eff(total_uniform)
    expected_neff_uniform = 100.0
    diff_neff_u = abs(n_eff_uniform - expected_neff_uniform)
    ok_neff_u = diff_neff_u < 1e-14
    print(
        f"  Test 4: N_eff = {n_eff_uniform:.1f} (expected {expected_neff_uniform:.1f}), "
        f"diff = {diff_neff_u:.2e} {'PASS' if ok_neff_u else 'FAIL'}"
    )
    if not ok_neff_u:
        all_pass = False

    # Test 5: Empty weights
    p_empty = weighted_histogram_estimator(np.array([]), np.array([]))
    n_eff_empty = kish_n_eff(np.array([]))
    ok_empty = p_empty == 0.0 and n_eff_empty == 0.0
    print(
        f"  Test 5: Empty -> P_det={p_empty}, N_eff={n_eff_empty} {'PASS' if ok_empty else 'FAIL'}"
    )
    if not ok_empty:
        all_pass = False

    # Test 6: N_eff bounds check -- N_eff <= N for any weight distribution
    rng = np.random.default_rng(42)
    for _ in range(100):
        w = rng.exponential(1.0, size=50)
        ne = kish_n_eff(w)
        if ne > len(w) + 1e-10:
            print(f"  Test 6: FAIL -- N_eff={ne} > N={len(w)}")
            all_pass = False
            break
    else:
        print("  Test 6: N_eff <= N for 100 random weight sets PASS")

    print()
    if all_pass:
        print("RESULT: ALL SYNTHETIC TESTS PASSED")
    else:
        print("RESULT: SYNTHETIC TESTS FAILED")
    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IS estimator verification on injection data")
    parser.add_argument(
        "--csv-dir",
        default="simulations/injections",
        help="Directory with injection CSVs",
    )
    args = parser.parse_args()

    # Run synthetic tests first (no data needed)
    synthetic_ok = run_synthetic_weight_test()

    # Run uniform-weight recovery test on real data
    csv_path = Path(args.csv_dir)
    if csv_path.exists():
        recovery_ok = run_uniform_weight_verification(str(csv_path))
    else:
        print(f"\nWARNING: {csv_path} not found, skipping recovery test.")
        recovery_ok = True  # not a failure, just no data

    if synthetic_ok and recovery_ok:
        print("\n=== ALL VERIFICATION PASSED ===")
    else:
        print("\n=== VERIFICATION FAILED ===")
        raise SystemExit(1)
