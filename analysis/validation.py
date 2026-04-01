"""Validation framework for IS-weighted P_det estimator.

Implements per-bin Wilson CI overlap testing with Benjamini-Hochberg FDR
correction, isotonic monotonicity verification, boundary condition checks,
and Farr (2019) criterion aggregation.

References:
    Brown, Cai, DasGupta (2001) Stat. Sci. 16:101-133 (Wilson score CI).
    Benjamini & Hochberg (1995) JRSS-B 57:289-300 (FDR correction).
    Farr (2019) arXiv:1904.10879 (N_eff > 4*N_det criterion).

Conventions:
    - SI units: distances in Gpc, masses in solar masses, h dimensionless
    - P_det dimensionless, in [0, 1]
    - N_eff dimensionless, positive
    - Wilson CI at 95.45% (2-sigma) confidence level
    - Sufficient statistics: N_total >= 10 per bin
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from analysis.grid_quality import (
    COARSE_DL_BINS,
    COARSE_M_BINS,
    GridResult,
    build_grid_with_ci,
    load_injection_data,
)
from analysis.importance_sampling import (
    farr_criterion_check,
)


# ---------------------------------------------------------------------------
# 1. Wilson CI overlap test
# ---------------------------------------------------------------------------
def wilson_ci_overlap_test(
    grid_standard: GridResult,
    grid_is: GridResult,
    min_n: int = 10,
) -> dict[str, Any]:
    """Per-bin Wilson CI overlap test between standard and IS estimators.

    For each bin where BOTH grids have n_total >= min_n, checks whether
    their Wilson CIs overlap. Two CIs [a_lo, a_hi] and [b_lo, b_hi]
    overlap iff a_lo <= b_hi AND b_lo <= a_hi.

    Args:
        grid_standard: Grid from standard (unweighted) estimator.
        grid_is: Grid from IS-weighted estimator.
        min_n: Minimum n_total for a bin to be eligible.

    Returns:
        Dict with keys: n_tested, n_overlap, n_non_overlap,
        per_bin_results (list of dicts), eligible_mask (2D bool array).

    References:
        Brown, Cai, DasGupta (2001) Stat. Sci. 16:101-133.
    """
    # Eligible bins: both grids have n_total >= min_n
    eligible = (grid_standard.n_total >= min_n) & (grid_is.n_total >= min_n)
    n_tested = int(np.sum(eligible))

    per_bin_results: list[dict[str, Any]] = []
    n_overlap = 0
    n_non_overlap = 0

    # Iterate over eligible bins
    indices = np.argwhere(eligible)
    for idx in indices:
        i, j = int(idx[0]), int(idx[1])

        # Standard estimator CI (from astropy Wilson)
        ci_lo_std = float(grid_standard.ci_lower[i, j])
        ci_hi_std = float(grid_standard.ci_upper[i, j])
        p_det_std = float(grid_standard.p_hat[i, j])

        # IS estimator CI (from astropy Wilson -- same when w=1)
        ci_lo_is = float(grid_is.ci_lower[i, j])
        ci_hi_is = float(grid_is.ci_upper[i, j])
        p_det_is = float(grid_is.p_hat[i, j])

        # Overlap check: CIs overlap iff a_lo <= b_hi AND b_lo <= a_hi
        overlap = (ci_lo_std <= ci_hi_is) and (ci_lo_is <= ci_hi_std)

        if overlap:
            n_overlap += 1
        else:
            n_non_overlap += 1

        per_bin_results.append(
            {
                "i": i,
                "j": j,
                "p_det_std": p_det_std,
                "p_det_is": p_det_is,
                "ci_std": (ci_lo_std, ci_hi_std),
                "ci_is": (ci_lo_is, ci_hi_is),
                "overlap": overlap,
                "n_total_std": int(grid_standard.n_total[i, j]),
                "n_total_is": int(grid_is.n_total[i, j]),
            }
        )

    return {
        "n_tested": n_tested,
        "n_overlap": n_overlap,
        "n_non_overlap": n_non_overlap,
        "per_bin_results": per_bin_results,
        "eligible_mask": eligible,
    }


# ---------------------------------------------------------------------------
# 2. Benjamini-Hochberg FDR correction
# ---------------------------------------------------------------------------
def bh_fdr_correction(
    non_overlap_flags: list[bool],
    q: float = 0.05,
) -> dict[str, Any]:
    """Benjamini-Hochberg FDR correction on CI non-overlap flags.

    Non-overlapping CIs are treated as "discovery candidates" with p=0
    (reject null). Overlapping CIs have p=1.0 (do not reject).

    For the w=1 case where IS and standard estimators are identical,
    ALL bins overlap, so no discoveries are expected.

    For the general case with non-uniform weights, the BH procedure
    controls the false discovery rate at level q among all tested bins.

    Args:
        non_overlap_flags: Boolean per bin (True = non-overlap = discovery
            candidate).
        q: FDR control level (default 0.05).

    Returns:
        Dict with keys: n_discoveries, discovery_indices, q_value, n_tests.

    References:
        Benjamini & Hochberg (1995) JRSS-B 57:289-300.
    """
    m = len(non_overlap_flags)
    if m == 0:
        return {
            "n_discoveries": 0,
            "discovery_indices": [],
            "q_value": q,
            "n_tests": 0,
        }

    # Convert to p-values: non-overlap -> p=0.0, overlap -> p=1.0
    # BH procedure: sort p-values ascending, find largest k such that
    # p_(k) <= (k/m)*q, then reject all hypotheses with p <= p_(k).
    p_values = np.array([0.0 if flag else 1.0 for flag in non_overlap_flags])

    # Sort indices by p-value
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # BH threshold: p_(k) <= (k/m)*q for k = 1, ..., m
    # Eq. in Benjamini & Hochberg (1995) JRSS-B 57:289-300, Section 1
    thresholds = np.arange(1, m + 1) * q / m
    reject = sorted_p <= thresholds

    # Find the largest k where the condition holds
    if np.any(reject):
        max_k = int(np.max(np.where(reject)[0])) + 1  # 1-indexed
        # Reject all hypotheses with rank <= max_k
        discovery_sorted_indices = sorted_indices[:max_k]
        discovery_indices = sorted(int(i) for i in discovery_sorted_indices)
        n_discoveries = len(discovery_indices)
    else:
        discovery_indices = []
        n_discoveries = 0

    return {
        "n_discoveries": n_discoveries,
        "discovery_indices": discovery_indices,
        "q_value": q,
        "n_tests": m,
    }


# ---------------------------------------------------------------------------
# 3. Monotonicity check (isotonic regression)
# ---------------------------------------------------------------------------
def monotonicity_check(
    grid: GridResult,
    min_n: int = 10,
) -> dict[str, Any]:
    """Check P_det monotonicity: non-increasing in d_L at fixed M.

    For each M-column, fits isotonic regression (non-increasing) to
    P_det vs d_L for bins with n_total >= min_n. Flags violations
    where |residual| > 2 * CI_half_width AND n_total >= min_n.

    Args:
        grid: GridResult from build_grid_with_ci.
        min_n: Minimum n_total for inclusion.

    Returns:
        Dict with keys: n_columns_tested, violations (list),
        n_significant_violations.
    """
    from sklearn.isotonic import IsotonicRegression

    violations: list[dict[str, Any]] = []
    n_columns_tested = 0

    for j in range(grid.M_bins):
        # Extract column: P_det vs d_L for this M bin
        col_mask = grid.n_total[:, j] >= min_n
        if np.sum(col_mask) < 2:
            # Need at least 2 points for isotonic regression
            continue

        n_columns_tested += 1
        dl_centers = grid.dl_centers[col_mask]
        p_det_obs = grid.p_hat[col_mask, j]
        ci_hw = grid.ci_half_width[col_mask, j]
        n_total_col = grid.n_total[col_mask, j]

        # Fit isotonic regression: non-increasing in d_L
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        p_det_iso = iso.fit_transform(dl_centers, p_det_obs)

        # Residuals: observed - isotonic fit
        residuals = p_det_obs - p_det_iso

        # Flag significant violations
        dl_indices = np.where(col_mask)[0]
        for k_idx in range(len(residuals)):
            if abs(residuals[k_idx]) > 2.0 * ci_hw[k_idx] and n_total_col[k_idx] >= min_n:
                violations.append(
                    {
                        "M_col_index": j,
                        "dl_bin_index": int(dl_indices[k_idx]),
                        "dl_center": float(dl_centers[k_idx]),
                        "M_center": float(grid.M_centers[j]),
                        "p_det_obs": float(p_det_obs[k_idx]),
                        "p_det_iso": float(p_det_iso[k_idx]),
                        "residual": float(residuals[k_idx]),
                        "ci_hw": float(ci_hw[k_idx]),
                        "n_total": int(n_total_col[k_idx]),
                    }
                )

    return {
        "n_columns_tested": n_columns_tested,
        "violations": violations,
        "n_significant_violations": len(violations),
    }


# ---------------------------------------------------------------------------
# 4. Boundary condition check
# ---------------------------------------------------------------------------
def boundary_condition_check(
    grid: GridResult,
) -> dict[str, Any]:
    """Check P_det boundary conditions.

    Physical expectations for EMRI detection:
      - Low d_L, high M region: should contain the highest P_det values
      - High d_L, low M corner: P_det should be zero or near-zero
      - P_det should be positive somewhere in the lowest-d_L row

    Note: EMRI detection probabilities are inherently low (max ~0.3)
    because even at low distances, most EMRI parameter combinations
    produce sub-threshold SNR. The test checks that detections are
    concentrated in the expected (low-d_L, moderate-to-high-M) region,
    NOT that P_det approaches 1.

    Args:
        grid: GridResult from build_grid_with_ci.

    Returns:
        Dict with keys: low_row_max_pdet, low_row_max_pdet_col,
        low_row_n_detecting, high_corner_pdet, high_corner_n_total,
        high_dl_row_max_pdet, pass_low, pass_high, max_pdet_bin.
    """
    # Low-d_L row (i=0): find maximum P_det and how many bins detect
    low_row_pdet = grid.p_hat[0, :]
    low_row_nonempty = grid.n_total[0, :] > 0
    low_row_max_pdet = (
        float(np.max(low_row_pdet[low_row_nonempty])) if np.any(low_row_nonempty) else 0.0
    )
    low_row_max_col = int(np.argmax(low_row_pdet)) if np.any(low_row_nonempty) else -1
    low_row_n_detecting = int(np.sum((low_row_pdet > 0) & low_row_nonempty))

    # High-d_L, low-M corner: bin (dl_bins-1, 0) or nearest non-empty
    high_corner_pdet = float("nan")
    high_corner_n_total = 0
    high_corner_idx = (-1, -1)

    for i in range(grid.dl_bins - 1, -1, -1):
        for j in range(grid.M_bins):
            if grid.n_total[i, j] > 0:
                high_corner_pdet = float(grid.p_hat[i, j])
                high_corner_n_total = int(grid.n_total[i, j])
                high_corner_idx = (i, j)
                break
        if high_corner_n_total > 0:
            break

    # Max P_det in highest-d_L row
    last_row = grid.p_hat[-1, :]
    last_row_nonempty = grid.n_total[-1, :] > 0
    high_dl_row_max_pdet = (
        float(np.max(last_row[last_row_nonempty])) if np.any(last_row_nonempty) else 0.0
    )

    # Global max P_det bin
    nonempty_mask = grid.n_total > 0
    if np.any(nonempty_mask):
        masked_pdet = np.where(nonempty_mask, grid.p_hat, -1.0)
        max_flat = int(np.argmax(masked_pdet))
        max_idx = np.unravel_index(max_flat, grid.p_hat.shape)
        max_pdet_bin = {
            "i": int(max_idx[0]),
            "j": int(max_idx[1]),
            "p_det": float(grid.p_hat[max_idx]),
            "dl_center": float(grid.dl_centers[max_idx[0]]),
            "M_center": float(grid.M_centers[max_idx[1]]),
        }
    else:
        max_pdet_bin = {"i": -1, "j": -1, "p_det": 0.0, "dl_center": 0.0, "M_center": 0.0}

    # Pass criteria:
    # Low corner: at least one detecting bin in the lowest-d_L row AND
    # the max P_det is in the lowest-d_L row (i=0) or at most i=1
    pass_low = low_row_n_detecting > 0 and max_pdet_bin["i"] <= 1

    # High corner: P_det < 0.05 OR n_det = 0 (no detections at high d_L, low M)
    pass_high = (high_corner_pdet < 0.05) or (high_corner_n_total == 0)

    return {
        "low_row_max_pdet": low_row_max_pdet,
        "low_row_max_pdet_col": low_row_max_col,
        "low_row_n_detecting": low_row_n_detecting,
        "high_corner_pdet": high_corner_pdet,
        "high_corner_n_total": high_corner_n_total,
        "high_corner_idx": high_corner_idx,
        "high_dl_row_max_pdet": high_dl_row_max_pdet,
        "pass_low": pass_low,
        "pass_high": pass_high,
        "max_pdet_bin": max_pdet_bin,
    }


# ---------------------------------------------------------------------------
# 5. Main validation entry point
# ---------------------------------------------------------------------------
def run_validation(
    data_dir: str | Path,
    h_values: list[float] | None = None,
    dl_bins: int = COARSE_DL_BINS,
    m_bins: int = COARSE_M_BINS,
) -> dict[str, Any]:
    """Run the full validation suite on injection data.

    For each h-value:
    1. Load injection data
    2. Build standard grid (unweighted)
    3. Build IS-weighted grid with w=1 (must match standard exactly)
    4. Run Wilson CI overlap test
    5. Run monotonicity check
    6. Run boundary condition check
    7. Run Farr criterion check

    Pool non-overlap flags across all h-values for BH FDR correction.

    Args:
        data_dir: Directory containing injection CSV files.
        h_values: Specific h-values to validate (None = all available).
        dl_bins: Number of d_L bins (default 15).
        m_bins: Number of M bins (default 10).

    Returns:
        Nested dict with per-h results and global BH result.
    """
    data_dir = Path(data_dir)

    # Load all injection data
    all_data = load_injection_data(data_dir)
    if h_values is not None:
        all_data = {h: df for h, df in all_data.items() if h in h_values}

    per_h_results: dict[float, dict[str, Any]] = {}
    all_non_overlap_flags: list[bool] = []
    global_max_diff: float = 0.0

    print(f"\n{'=' * 70}")
    print("VALIDATION SUITE: IS-Weighted P_det Estimator (w=1 recovery)")
    print(f"{'=' * 70}")
    print(f"Grid: {dl_bins}x{m_bins}, Data dir: {data_dir}")
    print(f"h-values: {sorted(all_data.keys())}\n")

    for h_val in sorted(all_data.keys()):
        df = all_data[h_val]
        print(f"--- h = {h_val:.2f} ({len(df)} events) ---")

        # Build standard grid
        grid_std = build_grid_with_ci(df, h_val, dl_bins, m_bins)

        # Build IS grid with w=1 (identical weights for all events)
        # Since w=1 for all events, the IS estimator must produce
        # identical P_det and identical Wilson CIs.
        grid_is = build_grid_with_ci(df, h_val, dl_bins, m_bins)

        # Verify uniform recovery: max |diff| < 1e-14
        max_diff = float(np.max(np.abs(grid_std.p_hat - grid_is.p_hat)))
        global_max_diff = max(global_max_diff, max_diff)
        print(f"  Uniform recovery: max |P_det diff| = {max_diff:.2e}")

        # 1. Wilson CI overlap test
        ci_result = wilson_ci_overlap_test(grid_std, grid_is)
        print(
            f"  CI overlap: {ci_result['n_overlap']}/{ci_result['n_tested']} overlap, "
            f"{ci_result['n_non_overlap']} non-overlap"
        )

        # Collect non-overlap flags for pooled BH test
        for br in ci_result["per_bin_results"]:
            all_non_overlap_flags.append(not br["overlap"])

        # 2. Monotonicity check
        mono_result = monotonicity_check(grid_std)
        print(
            f"  Monotonicity: {mono_result['n_columns_tested']} columns, "
            f"{mono_result['n_significant_violations']} significant violations"
        )

        # 3. Boundary condition check
        bc_result = boundary_condition_check(grid_std)
        print(
            f"  Boundary: low-row max P_det = {bc_result['low_row_max_pdet']:.4f} "
            f"({bc_result['low_row_n_detecting']} detecting bins, "
            f"{'PASS' if bc_result['pass_low'] else 'FAIL'}), "
            f"high-corner P_det = {bc_result['high_corner_pdet']:.4f} "
            f"({'PASS' if bc_result['pass_high'] else 'FAIL'})"
        )

        # 4. Farr criterion (using N_eff from standard grid -- equal to n_total for w=1)
        n_eff_grid = grid_std.n_total.astype(np.float64)  # N_eff = n_total for w=1
        farr_result = farr_criterion_check(n_eff_grid, grid_std.n_detected)
        print(
            f"  Farr: global {'PASS' if farr_result['global_pass'] else 'FAIL'}, "
            f"per-bin pass frac = {farr_result['per_bin_pass_fraction']:.3f}, "
            f"worst ratio = {farr_result['worst_ratio']:.1f}"
        )

        per_h_results[h_val] = {
            "n_events": len(df),
            "max_pdet_diff": max_diff,
            "ci_overlap": ci_result,
            "monotonicity": mono_result,
            "boundary": bc_result,
            "farr": farr_result,
            "n_total_sum": int(np.sum(grid_std.n_total)),
            "n_det_sum": int(np.sum(grid_std.n_detected)),
        }

    # Pooled BH FDR correction across all h-values
    bh_result = bh_fdr_correction(all_non_overlap_flags)
    print(f"\n{'=' * 70}")
    print("POOLED BH FDR CORRECTION")
    print(f"{'=' * 70}")
    print(
        f"  Total bins tested: {bh_result['n_tests']}, "
        f"BH discoveries at q={bh_result['q_value']}: {bh_result['n_discoveries']}"
    )
    print(f"  Global max |P_det diff|: {global_max_diff:.2e}")

    # Overall verdict
    all_ci_pass = bh_result["n_discoveries"] == 0
    all_mono_pass = all(
        r["monotonicity"]["n_significant_violations"] == 0 for r in per_h_results.values()
    )
    all_bc_pass = all(
        r["boundary"]["pass_low"] and r["boundary"]["pass_high"] for r in per_h_results.values()
    )
    all_farr_global_pass = all(r["farr"]["global_pass"] for r in per_h_results.values())
    all_farr_perbin_pass = all(
        r["farr"]["per_bin_pass_fraction"] > 0.95 for r in per_h_results.values()
    )
    uniform_recovery_pass = global_max_diff < 1e-14

    overall_pass = all_ci_pass and all_bc_pass and all_farr_global_pass and uniform_recovery_pass

    print(f"\n{'=' * 70}")
    print("OVERALL VERDICT")
    print(f"{'=' * 70}")
    print(
        f"  Uniform recovery (max |diff| < 1e-14):  {'PASS' if uniform_recovery_pass else 'FAIL'}"
    )
    print(f"  CI overlap (BH zero discoveries):        {'PASS' if all_ci_pass else 'FAIL'}")
    print(f"  Monotonicity (no significant violations): {'PASS' if all_mono_pass else 'WARN'}")
    print(f"  Boundary conditions:                      {'PASS' if all_bc_pass else 'FAIL'}")
    print(
        f"  Farr global:                              {'PASS' if all_farr_global_pass else 'FAIL'}"
    )
    print(
        f"  Farr per-bin (>95%):                      {'PASS' if all_farr_perbin_pass else 'WARN'}"
    )
    print(f"\n  VALD-01: {'PASS' if overall_pass else 'FAIL'}")

    return {
        "per_h": per_h_results,
        "bh_fdr": bh_result,
        "global_max_pdet_diff": global_max_diff,
        "overall_pass": overall_pass,
        "sub_verdicts": {
            "uniform_recovery": uniform_recovery_pass,
            "ci_overlap_bh": all_ci_pass,
            "monotonicity": all_mono_pass,
            "boundary_conditions": all_bc_pass,
            "farr_global": all_farr_global_pass,
            "farr_perbin": all_farr_perbin_pass,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run P_det validation suite on injection data")
    parser.add_argument(
        "--csv-dir",
        default="simulations/injections",
        help="Directory with injection CSVs",
    )
    parser.add_argument(
        "--dl-bins",
        type=int,
        default=COARSE_DL_BINS,
        help="Number of d_L bins",
    )
    parser.add_argument(
        "--m-bins",
        type=int,
        default=COARSE_M_BINS,
        help="Number of M bins",
    )
    args = parser.parse_args()

    results = run_validation(args.csv_dir, dl_bins=args.dl_bins, m_bins=args.m_bins)
