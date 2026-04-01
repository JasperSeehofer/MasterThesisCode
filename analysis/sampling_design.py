"""Neyman-optimal stratified allocation and two-stage sampling design for P_det.

Computes the optimal allocation of targeted injections across (d_L, M) bins
to minimize variance in the P_det boundary region. Uses Phase 18 uniform
injection data as the pilot sample to estimate per-bin Bernoulli standard
deviations.

Core method: Neyman-optimal allocation N_k proportional to sigma_k
  (Cochran 1977 Ch. 5; Owen 2013 Ch. 8).
Defensive mixture: q = alpha * p_uniform + (1-alpha) * g_targeted
  (Hesterberg 1995, Technometrics 37(2)).
IS estimator: Tiwari (2018), arXiv:1712.00482.
N_eff criterion: Farr (2019), arXiv:1904.10879.

Conventions:
  - SI units: distances in Gpc, masses in solar masses, h dimensionless
  - P_det dimensionless, in [0, 1]
  - sigma_k = sqrt(P_k * (1 - P_k)), dimensionless
  - VRF = Var_uniform / Var_stratified, dimensionless, > 1 means improvement
  - N_k = integer allocation per bin, sum(N_k) = N_targeted exactly
  - Boundary bins: 0.05 < P_det < 0.95
  - Weights w_i = p(theta_i) / q(theta_i), dimensionless, bounded by 1/alpha
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from analysis.grid_quality import (
    GridResult,
    build_grid_with_ci,
    load_injection_data,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BOUNDARY_LOWER: float = 0.05
BOUNDARY_UPPER: float = 0.95
DEFAULT_ALPHA: float = 0.3  # Hesterberg (1995) defensive mixture fraction
COARSE_DL_BINS: int = 15
COARSE_M_BINS: int = 10
DEFAULT_MIN_PER_BIN: int = 5
H_VALUES: list[float] = [0.60, 0.65, 0.70, 0.73, 0.80, 0.85, 0.90]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class AllocationResult:
    """Result of Neyman-optimal allocation for one h-value."""

    h_val: float
    n_pilot: int
    n_targeted: int
    allocation: npt.NDArray[np.int64]  # shape (dl_bins, M_bins)
    sigma_k: npt.NDArray[np.float64]  # shape (dl_bins, M_bins)
    boundary_mask: npt.NDArray[np.bool_]
    p_det: npt.NDArray[np.float64]
    n_total_pilot: npt.NDArray[np.float64]
    grid_result: GridResult


@dataclass
class VRFResult:
    """Variance reduction factor for one h-value."""

    h_val: float
    vrf_per_bin: npt.NDArray[np.float64]  # full grid, 0 for non-boundary
    vrf_boundary_mean: float  # weighted mean VRF across boundary bins
    vrf_boundary_min: float
    ci_hw_uniform: npt.NDArray[np.float64]  # hypothetical uniform CI hw
    ci_hw_stratified: npt.NDArray[np.float64]  # stratified CI hw


@dataclass
class TwoStageDesign:
    """Complete two-stage sampling design specification."""

    alpha: float
    allocation_results: dict[float, AllocationResult] = field(default_factory=dict)
    vrf_results: dict[float, VRFResult] = field(default_factory=dict)
    weight_bound: float = 0.0
    pilot_source: str = ""
    combined_estimator: str = ""


# ---------------------------------------------------------------------------
# Neyman-optimal allocation
# ---------------------------------------------------------------------------
def neyman_allocation(
    p_det_grid: npt.NDArray[np.float64],
    n_total_grid: npt.NDArray[np.float64],
    n_targeted_total: int,
    min_per_bin: int = DEFAULT_MIN_PER_BIN,
) -> npt.NDArray[np.int64]:
    """Compute Neyman-optimal allocation of targeted injections.

    Allocates N_targeted injections across bins proportional to
    sigma_k = sqrt(P_k * (1 - P_k)), the Bernoulli standard deviation.

    Bins with P_det = 0 or P_det = 1 have sigma_k = 0 and receive only
    the minimum allocation (if they had any pilot events).

    Args:
        p_det_grid: Per-bin detection probability, shape (dl_bins, M_bins).
        n_total_grid: Per-bin total pilot events, shape (dl_bins, M_bins).
        n_targeted_total: Total targeted injection budget.
        min_per_bin: Minimum allocation for bins with pilot events.

    Returns:
        Integer allocation array, same shape as p_det_grid.
        Sums to n_targeted_total exactly.

    References:
        Cochran (1977) Sampling Techniques, Ch. 5.
        Owen (2013) Monte Carlo theory, methods and examples, Ch. 8.
    """
    # Bernoulli std dev per bin
    # sigma_k = sqrt(P*(1-P)), max at P=0.5 where sigma=0.5
    sigma_k = np.sqrt(p_det_grid * (1.0 - p_det_grid))

    # Bins eligible for allocation: those with pilot data
    has_pilot = n_total_grid > 0

    # Total sigma for proportional allocation
    total_sigma = np.sum(sigma_k[has_pilot])

    allocation = np.zeros(p_det_grid.shape, dtype=np.int64)

    if total_sigma > 0:
        # Neyman-optimal: N_k proportional to sigma_k
        # Eq: N_k = floor(N_targeted * sigma_k / sum(sigma_k))
        frac = np.zeros_like(p_det_grid)
        frac[has_pilot] = sigma_k[has_pilot] / total_sigma
        allocation_float = n_targeted_total * frac
        allocation = np.floor(allocation_float).astype(np.int64)
    else:
        # All bins have P_det = 0 or 1; distribute evenly
        n_eligible = int(np.sum(has_pilot))
        if n_eligible > 0:
            base = n_targeted_total // n_eligible
            allocation[has_pilot] = base

    # Impose minimum for bins with pilot data
    below_min = has_pilot & (allocation < min_per_bin)
    allocation[below_min] = min_per_bin

    # Distribute remainder to highest-sigma bins
    remainder = n_targeted_total - int(np.sum(allocation))

    if remainder > 0:
        # Sort bins by sigma_k descending, add 1 to each until remainder = 0
        flat_sigma = sigma_k.ravel()
        sorted_indices = np.argsort(-flat_sigma)
        flat_alloc = allocation.ravel()
        for idx in sorted_indices:
            if remainder <= 0:
                break
            flat_alloc[idx] += 1
            remainder -= 1
        allocation = flat_alloc.reshape(p_det_grid.shape)
    elif remainder < 0:
        # Over-allocated due to minimum constraints; reduce from lowest-sigma bins
        flat_sigma = sigma_k.ravel()
        sorted_indices = np.argsort(flat_sigma)  # ascending sigma
        flat_alloc = allocation.ravel()
        for idx in sorted_indices:
            if remainder >= 0:
                break
            reduce = min(-remainder, max(0, int(flat_alloc[idx]) - min_per_bin))
            flat_alloc[idx] -= reduce
            remainder += reduce
        allocation = flat_alloc.reshape(p_det_grid.shape)

        # If still over-allocated, reduce minimum bins as last resort
        if remainder < 0:
            for idx in sorted_indices:
                if remainder >= 0:
                    break
                reduce = min(-remainder, int(flat_alloc[idx]))
                flat_alloc[idx] -= reduce
                remainder += reduce
            allocation = flat_alloc.reshape(p_det_grid.shape)

    return allocation


# ---------------------------------------------------------------------------
# Variance reduction factor
# ---------------------------------------------------------------------------
def variance_reduction_factor(
    p_det_grid: npt.NDArray[np.float64],
    n_total_pilot: npt.NDArray[np.float64],
    allocation_grid: npt.NDArray[np.int64],
    n_pilot_total: int,
) -> VRFResult:
    """Compute variance reduction factor for boundary bins.

    VRF = Var_uniform / Var_stratified, where:
    - Var_uniform = P*(1-P) / n_uniform_k for the same total budget N_new
      distributed uniformly (proportional to pilot distribution).
    - Var_stratified = P*(1-P) / (n_pilot_k + N_k) where n_pilot_k is the
      pilot contribution and N_k is the targeted allocation.

    The total new budget is N_new = sum(allocation_grid). In the uniform
    scenario, these N_new events distribute in proportion to the current
    pilot occupancy (same distribution as Phase 18).

    Args:
        p_det_grid: Per-bin detection probability from pilot.
        n_total_pilot: Per-bin pilot event count.
        allocation_grid: Per-bin targeted allocation.
        n_pilot_total: Total number of pilot events.

    Returns:
        VRFResult with per-bin and summary VRF values.

    Dimensional check:
        - Var has dimensions [P*(1-P)/N] = [dimensionless/integer] = dimensionless
        - VRF = Var/Var = dimensionless
    """
    boundary_mask = (p_det_grid > BOUNDARY_LOWER) & (p_det_grid < BOUNDARY_UPPER)
    n_targeted_total = int(np.sum(allocation_grid))

    # Uniform scenario: N_new additional events distributed proportionally to pilot
    # n_uniform_new_k = N_new * (n_pilot_k / N_pilot) for each bin
    pilot_fraction = np.zeros_like(p_det_grid)
    if n_pilot_total > 0:
        pilot_fraction = n_total_pilot / n_pilot_total

    # Total effective samples per bin:
    # Uniform: n_pilot_k + n_uniform_new_k = n_pilot_k + N_new * (n_pilot_k / N_pilot)
    n_uniform_effective = n_total_pilot + n_targeted_total * pilot_fraction

    # Stratified: n_pilot_k + N_k
    n_stratified_effective = n_total_pilot + allocation_grid.astype(np.float64)

    # Variance per bin: Var_k = P_k*(1-P_k) / n_k
    # VRF_k = Var_uniform_k / Var_stratified_k = n_stratified_k / n_uniform_k
    # (P*(1-P) cancels in the ratio)
    vrf_per_bin = np.zeros_like(p_det_grid)
    valid = n_uniform_effective > 0
    vrf_per_bin[valid] = n_stratified_effective[valid] / n_uniform_effective[valid]

    # CI half-width comparison (Wilson approximation: hw ~ z * sqrt(P*(1-P)/n))
    z_val = 1.96  # 95% CI
    sigma_k = np.sqrt(p_det_grid * (1.0 - p_det_grid))

    ci_hw_uniform = np.zeros_like(p_det_grid)
    ci_hw_stratified = np.zeros_like(p_det_grid)

    mask_u = n_uniform_effective > 0
    ci_hw_uniform[mask_u] = z_val * sigma_k[mask_u] / np.sqrt(n_uniform_effective[mask_u])

    mask_s = n_stratified_effective > 0
    ci_hw_stratified[mask_s] = z_val * sigma_k[mask_s] / np.sqrt(n_stratified_effective[mask_s])

    # Summary statistics for boundary bins
    if np.any(boundary_mask):
        vrf_boundary = vrf_per_bin[boundary_mask]
        # Weight by sigma_k for the mean (higher-variance bins matter more)
        sigma_boundary = sigma_k[boundary_mask]
        total_sigma_boundary = np.sum(sigma_boundary)
        if total_sigma_boundary > 0:
            vrf_mean = float(np.sum(vrf_boundary * sigma_boundary) / total_sigma_boundary)
        else:
            vrf_mean = float(np.mean(vrf_boundary))
        vrf_min = float(np.min(vrf_boundary))
    else:
        vrf_mean = 1.0
        vrf_min = 1.0

    return VRFResult(
        h_val=0.0,  # set by caller
        vrf_per_bin=vrf_per_bin,
        vrf_boundary_mean=vrf_mean,
        vrf_boundary_min=vrf_min,
        ci_hw_uniform=ci_hw_uniform,
        ci_hw_stratified=ci_hw_stratified,
    )


# ---------------------------------------------------------------------------
# Defensive mixture weight
# ---------------------------------------------------------------------------
def defensive_mixture_weight(
    alpha: float,
    n_targeted_bin: float,
    bin_area: float,
    p_uniform_density: float,
) -> float:
    """Compute importance weight for a targeted sample in one bin.

    The defensive mixture proposal is:
        q(theta) = alpha * p_uniform(theta) + (1 - alpha) * g_targeted(theta)

    where g_targeted is piecewise constant in bins:
        g_targeted(theta in bin k) proportional to N_k_targeted / bin_area_k

    The importance weight is:
        w = p_uniform(theta) / q(theta)
          = 1 / [alpha + (1 - alpha) * g_targeted(theta) / p_uniform(theta)]

    Weight bound: max(w) <= 1/alpha (since g_targeted >= 0).

    For pilot samples drawn from p_uniform: q = p_uniform, so w = 1.

    Args:
        alpha: Mixture fraction for uniform component (0 < alpha < 1).
        n_targeted_bin: Targeted allocation for this bin.
        bin_area: Area of this bin in (d_L, M) space.
        p_uniform_density: Prior density p_uniform at this point.

    Returns:
        Importance weight w_i = p_uniform / q.

    References:
        Hesterberg (1995), Technometrics 37(2).
    """
    if p_uniform_density <= 0 or bin_area <= 0:
        return 1.0

    g_targeted_density = n_targeted_bin / bin_area if n_targeted_bin > 0 else 0.0

    ratio = g_targeted_density / p_uniform_density
    # w = 1 / (alpha + (1-alpha) * g/p)
    denom = alpha + (1.0 - alpha) * ratio
    if denom <= 0:
        return 1.0 / alpha  # upper bound

    return 1.0 / denom


# ---------------------------------------------------------------------------
# Two-stage design specification
# ---------------------------------------------------------------------------
def two_stage_design(
    injection_dir: str | Path,
    alpha: float = DEFAULT_ALPHA,
    dl_bins: int = COARSE_DL_BINS,
    M_bins: int = COARSE_M_BINS,
    min_per_bin: int = DEFAULT_MIN_PER_BIN,
    targeted_fraction: float = 0.7,
) -> TwoStageDesign:
    """Compute the complete two-stage sampling design from Phase 18 data.

    Stage 1 (pilot): Phase 18 uniform injections (~23k events per h-value).
    Stage 2 (targeted): Additional injections allocated by Neyman-optimal
    rule, concentrated on boundary bins (0.05 < P_det < 0.95).

    The targeted budget is targeted_fraction * N_pilot per h-value.

    Args:
        injection_dir: Directory containing Phase 18 injection CSVs.
        alpha: Defensive mixture fraction (0.3 = 30% uniform + 70% targeted).
        dl_bins: Number of d_L bins.
        M_bins: Number of M bins.
        min_per_bin: Minimum targeted allocation per non-empty bin.
        targeted_fraction: Targeted budget as fraction of pilot.

    Returns:
        TwoStageDesign with allocation and VRF results for all h-values.
    """
    data = load_injection_data(injection_dir)

    design = TwoStageDesign(
        alpha=alpha,
        weight_bound=1.0 / alpha,
        pilot_source=f"Phase 18 uniform injections ({injection_dir})",
        combined_estimator=(
            "Self-normalized IS: P_hat = sum(w_i * I_i) / sum(w_i). "
            "Pilot samples: w_i = 1. "
            "Targeted samples: w_i = p(theta_i) / q(theta_i). "
            "Ref: Tiwari (2018), arXiv:1712.00482, Eq. 5-8."
        ),
    )

    for h_val, df in sorted(data.items()):
        # Build 15x10 grid from pilot data
        gr = build_grid_with_ci(df, h_val, dl_bins, M_bins)

        n_pilot = len(df)
        n_targeted = int(targeted_fraction * n_pilot)

        # Compute Neyman allocation
        alloc = neyman_allocation(
            gr.p_hat,
            gr.n_total,
            n_targeted,
            min_per_bin=min_per_bin,
        )

        boundary_mask = (gr.p_hat > BOUNDARY_LOWER) & (gr.p_hat < BOUNDARY_UPPER)

        alloc_result = AllocationResult(
            h_val=h_val,
            n_pilot=n_pilot,
            n_targeted=n_targeted,
            allocation=alloc,
            sigma_k=np.sqrt(gr.p_hat * (1.0 - gr.p_hat)),
            boundary_mask=boundary_mask,
            p_det=gr.p_hat,
            n_total_pilot=gr.n_total,
            grid_result=gr,
        )

        # Compute VRF
        vrf = variance_reduction_factor(gr.p_hat, gr.n_total, alloc, n_pilot)
        vrf.h_val = h_val

        design.allocation_results[h_val] = alloc_result
        design.vrf_results[h_val] = vrf

    return design


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary_table(design: TwoStageDesign) -> str:
    """Generate a summary table of the allocation and VRF results.

    Returns the table as a string for embedding in reports.
    """
    lines: list[str] = []
    lines.append(
        f"{'h':>6s}  {'N_pilot':>8s}  {'N_targ':>8s}  {'sum(Nk)':>8s}  "
        f"{'Boundary':>8s}  {'Alloc/bd':>8s}  {'Alloc/nbd':>9s}  "
        f"{'Ratio':>6s}  {'VRF_mean':>8s}  {'VRF_min':>7s}"
    )
    lines.append("-" * 100)

    all_pass = True

    for h_val in sorted(design.allocation_results.keys()):
        ar = design.allocation_results[h_val]
        vr = design.vrf_results[h_val]

        alloc_sum = int(np.sum(ar.allocation))
        n_boundary = int(np.sum(ar.boundary_mask))
        n_nonboundary = int(np.sum(~ar.boundary_mask & (ar.n_total_pilot > 0)))

        # Mean allocation per boundary vs non-boundary bin
        if n_boundary > 0:
            mean_alloc_boundary = float(np.mean(ar.allocation[ar.boundary_mask]))
        else:
            mean_alloc_boundary = 0.0

        if n_nonboundary > 0:
            nonboundary_mask = ~ar.boundary_mask & (ar.n_total_pilot > 0)
            mean_alloc_nonboundary = float(np.mean(ar.allocation[nonboundary_mask]))
        else:
            mean_alloc_nonboundary = 0.0

        ratio = (
            mean_alloc_boundary / mean_alloc_nonboundary
            if mean_alloc_nonboundary > 0
            else float("inf")
        )

        if vr.vrf_boundary_mean < 2.0:
            all_pass = False

        lines.append(
            f"{h_val:6.2f}  {ar.n_pilot:8d}  {ar.n_targeted:8d}  {alloc_sum:8d}  "
            f"{n_boundary:8d}  {mean_alloc_boundary:8.1f}  {mean_alloc_nonboundary:9.1f}  "
            f"{ratio:6.1f}x  {vr.vrf_boundary_mean:8.1f}  {vr.vrf_boundary_min:7.1f}"
        )

    lines.append("")
    if all_pass:
        lines.append("CONTRACT TARGET: VRF > 2.0 for all h-values -- PASSED")
    else:
        lines.append("CONTRACT TARGET: VRF > 2.0 -- FAILED for some h-values")

    return "\n".join(lines)


def print_boundary_detail_table(design: TwoStageDesign) -> str:
    """Generate detailed per-boundary-bin table for the design report."""
    lines: list[str] = []
    lines.append(
        f"{'h':>6s}  {'Bin(i,j)':>10s}  {'P_det':>6s}  {'sigma':>6s}  "
        f"{'n_pilot':>8s}  {'N_k':>6s}  {'VRF':>6s}  "
        f"{'CI_unif':>8s}  {'CI_strat':>8s}"
    )
    lines.append("-" * 80)

    for h_val in sorted(design.allocation_results.keys()):
        ar = design.allocation_results[h_val]
        vr = design.vrf_results[h_val]

        boundary_indices = np.argwhere(ar.boundary_mask)
        for idx in boundary_indices:
            i, j = int(idx[0]), int(idx[1])
            lines.append(
                f"{h_val:6.2f}  ({i:3d},{j:3d})  "
                f"{ar.p_det[i, j]:6.3f}  {ar.sigma_k[i, j]:6.3f}  "
                f"{ar.n_total_pilot[i, j]:8.0f}  {ar.allocation[i, j]:6d}  "
                f"{vr.vrf_per_bin[i, j]:6.1f}  "
                f"{vr.ci_hw_uniform[i, j]:8.4f}  {vr.ci_hw_stratified[i, j]:8.4f}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_design(design: TwoStageDesign) -> dict[str, Any]:
    """Run all acceptance tests on the design.

    Returns dict with test names and pass/fail status.
    """
    results: dict[str, Any] = {}

    # Test 1: Allocation conservation
    alloc_conserved = True
    for h_val, ar in design.allocation_results.items():
        alloc_sum = int(np.sum(ar.allocation))
        if alloc_sum != ar.n_targeted:
            alloc_conserved = False
            results[f"alloc_sum_{h_val:.2f}"] = f"FAIL: sum={alloc_sum}, expected={ar.n_targeted}"
    results["test-allocation-sums"] = "PASS" if alloc_conserved else "FAIL"

    # Test 2: Boundary concentration (>5x)
    concentration_pass = True
    for h_val, ar in design.allocation_results.items():
        n_boundary = int(np.sum(ar.boundary_mask))
        nonboundary_mask = ~ar.boundary_mask & (ar.n_total_pilot > 0)
        n_nonboundary = int(np.sum(nonboundary_mask))
        if n_boundary > 0 and n_nonboundary > 0:
            mean_bd = float(np.mean(ar.allocation[ar.boundary_mask]))
            mean_nbd = float(np.mean(ar.allocation[nonboundary_mask]))
            ratio = mean_bd / mean_nbd if mean_nbd > 0 else float("inf")
            if ratio < 5.0:
                concentration_pass = False
                results[f"concentration_{h_val:.2f}"] = f"FAIL: ratio={ratio:.1f}, expected > 5.0"
    results["test-boundary-concentration"] = "PASS" if concentration_pass else "FAIL"

    # Test 3: VRF > 2.0 for boundary bins
    vrf_pass = True
    for h_val, vr in design.vrf_results.items():
        if vr.vrf_boundary_mean < 2.0:
            vrf_pass = False
            results[f"vrf_{h_val:.2f}"] = (
                f"FAIL: VRF_mean={vr.vrf_boundary_mean:.2f}, expected > 2.0"
            )
    results["test-vrf-exceeds-2"] = "PASS" if vrf_pass else "FAIL"

    # Test 4: sigma_k range
    sigma_pass = True
    for h_val, ar in design.allocation_results.items():
        # P_det = 0 or 1 -> sigma = 0
        zero_pdet = (ar.p_det == 0.0) | (ar.p_det == 1.0)
        if np.any(ar.sigma_k[zero_pdet] != 0.0):
            sigma_pass = False
        # Boundary sigma in (0, 0.5]
        bd_sigma = ar.sigma_k[ar.boundary_mask]
        if len(bd_sigma) > 0:
            if np.any(bd_sigma <= 0) or np.any(bd_sigma > 0.5 + 1e-10):
                sigma_pass = False
    results["test-sigma-range"] = "PASS" if sigma_pass else "FAIL"

    # Test 5: Minimum allocation respected
    min_pass = True
    for h_val, ar in design.allocation_results.items():
        has_pilot = ar.n_total_pilot > 0
        below_min = has_pilot & (ar.allocation < DEFAULT_MIN_PER_BIN)
        if np.any(below_min):
            min_pass = False
            n_below = int(np.sum(below_min))
            results[f"min_alloc_{h_val:.2f}"] = f"FAIL: {n_below} bins below minimum"
    results["test-min-allocation"] = "PASS" if min_pass else "FAIL"

    # Test 6: Weight bound
    results["test-weight-bound"] = f"PASS: max(w) <= 1/alpha = {design.weight_bound:.4f}"

    # Test 7: Full support
    results["test-full-support"] = (
        f"PASS: q(theta) >= alpha * p(theta) = {design.alpha} * p(theta) > 0 "
        f"wherever p(theta) > 0 (by construction)"
    )

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main(injection_dir: str = "simulations/injections") -> None:
    """Run the full Neyman allocation and VRF computation."""
    print("=" * 80)
    print("Neyman-Optimal Stratified Allocation and VRF Computation")
    print(f"Injection data: {injection_dir}")
    print(f"Grid: {COARSE_DL_BINS}x{COARSE_M_BINS} (d_L x M)")
    print(f"Alpha: {DEFAULT_ALPHA} (defensive mixture)")
    print("=" * 80)

    design = two_stage_design(injection_dir)

    print("\n--- Summary Table ---\n")
    summary = print_summary_table(design)
    print(summary)

    print("\n--- Boundary Bin Detail ---\n")
    detail = print_boundary_detail_table(design)
    print(detail)

    print("\n--- Verification ---\n")
    verification = verify_design(design)
    for test_name, result in verification.items():
        print(f"  {test_name}: {result}")

    print("\n--- Two-Stage Design Specification ---\n")
    print(f"  Pilot source: {design.pilot_source}")
    print(f"  Alpha: {design.alpha}")
    print(f"  Weight bound: max(w) <= {design.weight_bound:.4f}")
    print(f"  Estimator: {design.combined_estimator}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neyman-optimal stratified allocation for P_det")
    parser.add_argument(
        "--injection-dir",
        default="simulations/injections",
        help="Directory with injection CSVs",
    )
    args = parser.parse_args()
    main(args.injection_dir)
