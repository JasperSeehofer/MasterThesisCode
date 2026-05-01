"""Test 13 (Audit A1): Δh=0.001 reinterpretation of cached posteriors.

Resolves the discrete-grid ambiguity flagged in the v2.3 audit programme.
The cluster MAP shifted 0.7650 → 0.7550 (one Δh=0.005 grid step) between
Plan 45-03 and Plan 45-05. Bootstrap σ_MAP ≈ 0.01 is comparable to the grid
spacing, so the discrete shift cannot be distinguished from a continuous
shift of arbitrary size on the existing 38-point grid.

This script:
  1. Loads per-event log-likelihoods from results/phase45_v2_posteriors/
     (1D `without_bh_mass` channel).
  2. Linearly interpolates log L_i(h) for each event to a Δh=0.001 grid in
     [0.70, 0.80].
  3. Linearly interpolates log D(h) to the same fine grid.
  4. Recomputes the joint posterior log p(h) = Σ log L_i(h) − N log D(h)
     and reports the continuous MAP, posterior mean, 68%/95% HPD.
  5. Cross-checks via cubic-spline interpolation; agreement within
     Δh=0.002 required (else escalate to Audit A6 native fine-grid sweep).
  6. Bootstrap (B=1000) on the fine grid for both linear and cubic.
  7. For the 2D `with_bh_mass` channel: interpolates the cached *joint*
     posterior (38 points → fine grid) since per-event JSONs are not cached
     locally. This is a coarser check but flags whether 2D MAP=0.745 is a
     grid artifact.

Pre-registered gates (set BEFORE running):
  G1a: continuous 1D MAP ∈ [0.720, 0.740] (±σ_boot of 0.730)
       → bias is a discrete-grid artifact; current state acceptable; A7 backstop only.
  G1b: continuous 1D MAP ∈ [0.745, 0.755], σ_boot ≈ 0.01
       → real shift; bias genuine ~+0.020 to +0.025; proceed to A2.
  G1c: continuous 1D MAP at 0.755 ± tight CI excluding 0.730 by > 3 σ_boot
       → bias robust regardless of grid; proceed to A2.

Cross-check gate:
  |cubic_MAP - linear_MAP| < 0.002 → A1 reliable; else escalate to A6.

Run from project root:
    uv run python scripts/bias_investigation/test_13_fine_grid_map.py
"""

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from master_thesis_code.bayesian_inference.posterior_combination import (
    CombinationStrategy,
    apply_strategy,
    build_likelihood_array,
    load_posterior_jsons,
)

POSTERIORS_DIR_1D = PROJECT_ROOT / "results" / "phase45_v2_posteriors"
POSTERIORS_DIR_2D = PROJECT_ROOT / "results" / "phase45_v2_posteriors_with_bh_mass"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "outputs" / "phase45"

H_TRUTH = 0.73
N_BOOTSTRAP = 1000
RANDOM_SEED = 20260501  # A1 seed (distinct from T08 seed 20260429)
STRATEGY = CombinationStrategy.PHYSICS_FLOOR

# Fine grid covers the whole 38-point support to avoid edge effects, then we
# focus the MAP search on the peak region for reporting.
FINE_GRID_LO, FINE_GRID_HI, FINE_GRID_STEP = 0.60, 0.86, 0.001
PEAK_REGION_LO, PEAK_REGION_HI = 0.70, 0.80

# Pre-registered acceptance windows (truth h=0.73)
GATE_G1A_LO, GATE_G1A_HI = 0.720, 0.740  # discrete-grid artifact band
GATE_G1B_LO, GATE_G1B_HI = 0.745, 0.755  # real shift band
CROSS_CHECK_TOL = 0.002  # linear-vs-cubic |Δ MAP|


def _hpd(samples: npt.NDArray[np.float64], frac: float) -> tuple[float, float]:
    """Highest posterior density interval (frac)."""
    sorted_s = np.sort(samples)
    n = len(sorted_s)
    width = int(np.ceil(frac * n))
    if width >= n:
        return float(sorted_s[0]), float(sorted_s[-1])
    diffs = sorted_s[width:] - sorted_s[: n - width]
    i = int(np.argmin(diffs))
    return float(sorted_s[i]), float(sorted_s[i + width])


def _interp_log_likelihoods(
    log_likes_coarse: npt.NDArray[np.float64],
    h_coarse: npt.NDArray[np.float64],
    h_fine: npt.NDArray[np.float64],
    method: str,
) -> npt.NDArray[np.float64]:
    """Interpolate per-event log-likelihoods from coarse to fine h-grid.

    Parameters
    ----------
    log_likes_coarse : (n_events, n_h_coarse) array of log-likelihoods.
    h_coarse : (n_h_coarse,) array of coarse-grid h-values.
    h_fine   : (n_h_fine,) array of fine-grid h-values.
    method   : 'linear' or 'cubic'.

    Returns (n_events, n_h_fine) array.
    """
    n_events = log_likes_coarse.shape[0]
    out = np.empty((n_events, len(h_fine)), dtype=np.float64)
    if method == "linear":
        for i in range(n_events):
            out[i, :] = np.interp(h_fine, h_coarse, log_likes_coarse[i, :])
    elif method == "cubic":
        for i in range(n_events):
            cs = CubicSpline(h_coarse, log_likes_coarse[i, :], extrapolate=False)
            out[i, :] = cs(h_fine)
    else:
        raise ValueError(f"Unknown interpolation method: {method!r}")
    return out


def _joint_log_posterior(
    log_likes_fine: npt.NDArray[np.float64],
    log_D_h_fine: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    n_events = log_likes_fine.shape[0]
    result: npt.NDArray[np.float64] = np.sum(log_likes_fine, axis=0) - n_events * log_D_h_fine
    return result


def _continuous_map_summary(
    log_post: npt.NDArray[np.float64],
    h_fine: npt.NDArray[np.float64],
    peak_lo: float,
    peak_hi: float,
) -> dict[str, float]:
    """Continuous MAP, posterior mean, 68%/95% HPD on the fine h-grid."""
    log_post = log_post - np.max(log_post)
    post = np.exp(log_post)
    post_norm = post / np.trapezoid(post, h_fine)
    map_idx = int(np.argmax(post_norm))
    map_h = float(h_fine[map_idx])
    mean = float(np.trapezoid(h_fine * post_norm, h_fine))
    cdf = np.cumsum(post_norm) * (h_fine[1] - h_fine[0])
    cdf /= cdf[-1]
    p16, p50, p84 = (float(np.interp(q, cdf, h_fine)) for q in (0.16, 0.5, 0.84))
    p2_5, p97_5 = (float(np.interp(q, cdf, h_fine)) for q in (0.025, 0.975))
    # Restrict reporting peak metric to the peak region in case of multimodality.
    in_peak = (h_fine >= peak_lo) & (h_fine <= peak_hi)
    if in_peak.any():
        peak_local_idx = int(np.argmax(post_norm[in_peak]))
        peak_global_idx = int(np.where(in_peak)[0][peak_local_idx])
        peak_local_map = float(h_fine[peak_global_idx])
    else:
        peak_local_map = map_h
    return {
        "map_h": map_h,
        "peak_local_map_h": peak_local_map,
        "mean": mean,
        "median": p50,
        "p16": p16,
        "p84": p84,
        "p2.5": p2_5,
        "p97.5": p97_5,
    }


def _bootstrap_continuous_map(
    log_likes_fine: npt.NDArray[np.float64],
    log_D_h_fine: npt.NDArray[np.float64],
    h_fine: npt.NDArray[np.float64],
    n_iter: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    n_events = log_likes_fine.shape[0]
    map_samples = np.empty(n_iter, dtype=np.float64)
    for b in range(n_iter):
        idx = rng.integers(0, n_events, size=n_events)
        log_post = np.sum(log_likes_fine[idx, :], axis=0) - n_events * log_D_h_fine
        map_samples[b] = float(h_fine[int(np.argmax(log_post))])
    return map_samples


def _classify_gate(map_h: float, sigma_boot: float, ci_lo: float, ci_hi: float) -> str:
    # Tolerance equal to half a fine-grid step absorbs floating-point endpoint
    # offsets from np.arange (e.g. 0.7550000000000001 ↔ 0.755).
    eps = 0.5 * FINE_GRID_STEP
    if (GATE_G1A_LO - eps) <= map_h <= (GATE_G1A_HI + eps):
        return "G1a (discrete-grid artifact; truth recovered)"
    if (GATE_G1B_LO - eps) <= map_h <= (GATE_G1B_HI + eps):
        return "G1b (real shift; bias genuine ~+0.020 to +0.025)"
    if not (ci_lo <= H_TRUTH <= ci_hi) and abs(map_h - H_TRUTH) > 3.0 * sigma_boot:
        return "G1c (robust bias; CI excludes truth by >3 σ_boot)"
    return "OTHER (between bands; document explicitly)"


def analyze_1d_channel() -> dict[str, Any]:
    print("\n=== A1 / 1D CHANNEL (without_bh_mass) ===\n")

    # 1. Load per-event likelihoods + cached D(h).
    h_values, event_likelihoods = load_posterior_jsons(POSTERIORS_DIR_1D)
    likelihoods, _ = build_likelihood_array(h_values, event_likelihoods)
    h_coarse = np.asarray(h_values, dtype=np.float64)
    print(f"Loaded {likelihoods.shape[0]} events × {likelihoods.shape[1]} h-bins")

    with open(POSTERIORS_DIR_1D / "combined_posterior.json") as f:
        cached = json.load(f)
    D_h_coarse = np.asarray(cached["D_h_per_h"], dtype=np.float64)
    log_D_h_coarse = np.log(D_h_coarse)
    cached_map_h = float(cached["map_h"])

    # 2. Apply combination strategy (physics-floor) → log-likelihoods.
    processed, n_excluded = apply_strategy(likelihoods, STRATEGY)
    log_likes_coarse = np.log(processed)
    print(
        f"After strategy '{STRATEGY.value}': {processed.shape[0]} events used, "
        f"{n_excluded} excluded"
    )

    # 3. Build fine grid and interpolate (linear baseline).
    h_fine = np.arange(FINE_GRID_LO, FINE_GRID_HI + 0.5 * FINE_GRID_STEP, FINE_GRID_STEP)
    log_likes_fine_lin = _interp_log_likelihoods(log_likes_coarse, h_coarse, h_fine, "linear")
    log_D_h_fine_lin = np.interp(h_fine, h_coarse, log_D_h_coarse)
    log_post_lin = _joint_log_posterior(log_likes_fine_lin, log_D_h_fine_lin)
    summary_lin = _continuous_map_summary(log_post_lin, h_fine, PEAK_REGION_LO, PEAK_REGION_HI)
    print(
        f"Linear:  continuous MAP = {summary_lin['peak_local_map_h']:.4f} "
        f"(mean = {summary_lin['mean']:.4f})"
    )

    # 4. Cross-check with cubic spline.
    log_likes_fine_cub = _interp_log_likelihoods(log_likes_coarse, h_coarse, h_fine, "cubic")
    cs_D = CubicSpline(h_coarse, log_D_h_coarse, extrapolate=False)
    log_D_h_fine_cub = cs_D(h_fine)
    valid = ~np.isnan(log_D_h_fine_cub) & ~np.isnan(log_likes_fine_cub).any(axis=0)
    h_fine_cub = h_fine[valid]
    log_post_cub = _joint_log_posterior(log_likes_fine_cub[:, valid], log_D_h_fine_cub[valid])
    summary_cub = _continuous_map_summary(log_post_cub, h_fine_cub, PEAK_REGION_LO, PEAK_REGION_HI)
    print(
        f"Cubic:   continuous MAP = {summary_cub['peak_local_map_h']:.4f} "
        f"(mean = {summary_cub['mean']:.4f})"
    )

    # 5. Cross-check tolerance.
    delta_lin_cub = abs(summary_lin["peak_local_map_h"] - summary_cub["peak_local_map_h"])
    cross_check_pass = delta_lin_cub < CROSS_CHECK_TOL
    print(
        f"|cubic - linear| = {delta_lin_cub:.4f}  "
        f"(tol {CROSS_CHECK_TOL}; "
        f"{'PASS' if cross_check_pass else 'FAIL → escalate to A6'})"
    )

    # 6. Bootstrap on linear fine grid.
    rng = np.random.default_rng(RANDOM_SEED)
    print(f"Bootstrapping linear fine-grid: B={N_BOOTSTRAP}, N_events={processed.shape[0]}...")
    map_samples_lin = _bootstrap_continuous_map(
        log_likes_fine_lin, log_D_h_fine_lin, h_fine, N_BOOTSTRAP, rng
    )
    sigma_boot_lin = float(np.std(map_samples_lin, ddof=1))
    p16_lin, p84_lin = _hpd(map_samples_lin, 0.68)
    p2_5_lin, p97_5_lin = _hpd(map_samples_lin, 0.95)
    median_boot = float(np.median(map_samples_lin))
    print(
        f"  σ_boot = {sigma_boot_lin:.4f}, "
        f"68% HPD = [{p16_lin:.4f}, {p84_lin:.4f}], "
        f"95% HPD = [{p2_5_lin:.4f}, {p97_5_lin:.4f}]"
    )

    # 7. Pre-registered gate classification.
    gate = _classify_gate(summary_lin["peak_local_map_h"], sigma_boot_lin, p16_lin, p84_lin)
    print(f"\n>>> Pre-registered gate verdict: {gate}\n")

    # 8. Plot.
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    in_peak = (h_fine >= 0.68) & (h_fine <= 0.82)
    log_post_lin_plot = log_post_lin[in_peak] - np.max(log_post_lin[in_peak])
    post_lin_plot = np.exp(log_post_lin_plot)
    post_lin_plot = post_lin_plot / np.trapezoid(post_lin_plot, h_fine[in_peak])
    in_peak_cub = (h_fine_cub >= 0.68) & (h_fine_cub <= 0.82)
    log_post_cub_plot = log_post_cub[in_peak_cub] - np.max(log_post_cub[in_peak_cub])
    post_cub_plot = np.exp(log_post_cub_plot)
    post_cub_plot = post_cub_plot / np.trapezoid(post_cub_plot, h_fine_cub[in_peak_cub])
    ax.plot(h_fine[in_peak], post_lin_plot, label="linear interp (Δh=0.001)", color="tab:blue")
    ax.plot(
        h_fine_cub[in_peak_cub],
        post_cub_plot,
        label="cubic-spline cross-check",
        color="tab:orange",
        ls="--",
    )
    ax.axvline(H_TRUTH, color="green", lw=2.0, label=f"truth h={H_TRUTH}")
    ax.axvline(cached_map_h, color="red", lw=2.0, label=f"discrete cluster MAP={cached_map_h:.4f}")
    ax.axvline(
        summary_lin["peak_local_map_h"],
        color="purple",
        lw=1.5,
        label=f"continuous MAP={summary_lin['peak_local_map_h']:.4f}",
    )
    ax.axvspan(p16_lin, p84_lin, color="purple", alpha=0.15, label="68% bootstrap HPD")
    ax.set_xlabel("h")
    ax.set_ylabel("p(h | data) (peak-normalized)")
    ax.set_title(
        f"Audit A1 (1D channel) — continuous MAP={summary_lin['peak_local_map_h']:.4f}, "
        f"σ_boot={sigma_boot_lin:.4f}, gate: {gate.split(' ')[0]}"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_png = OUTPUT_DIR / "fine_grid_map_1d.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")

    return {
        "channel": "without_bh_mass_1d",
        "n_events": int(processed.shape[0]),
        "n_excluded": int(n_excluded),
        "discrete_cached_map_h": cached_map_h,
        "continuous_map_h_linear": summary_lin["peak_local_map_h"],
        "continuous_map_h_cubic": summary_cub["peak_local_map_h"],
        "linear_summary": summary_lin,
        "cubic_summary": summary_cub,
        "delta_linear_cubic": float(delta_lin_cub),
        "cross_check_tol": CROSS_CHECK_TOL,
        "cross_check_pass": bool(cross_check_pass),
        "sigma_boot": sigma_boot_lin,
        "bootstrap_median": median_boot,
        "bootstrap_68_hpd": [p16_lin, p84_lin],
        "bootstrap_95_hpd": [p2_5_lin, p97_5_lin],
        "n_bootstrap": int(N_BOOTSTRAP),
        "random_seed": RANDOM_SEED,
        "gate_verdict": gate,
        "h_truth": H_TRUTH,
    }


def analyze_2d_channel_joint_only() -> dict[str, Any]:
    """Joint-posterior interpolation for the 2D channel (no per-event JSONs).

    This is a coarser proxy: we interpolate the *cached posterior* on the
    38-point grid to a fine grid using both linear and cubic, then compare
    continuous MAPs. Without per-event data we cannot bootstrap, so we report
    only continuous-MAP agreement vs the discrete cached MAP.
    """
    print("\n=== A1 / 2D CHANNEL (with_bh_mass, joint-posterior interp only) ===\n")

    with open(POSTERIORS_DIR_2D / "combined_posterior.json") as f:
        cached = json.load(f)
    h_coarse = np.asarray(cached["h_values"], dtype=np.float64)
    posterior_coarse = np.asarray(cached["posterior"], dtype=np.float64)
    cached_map_h = float(cached["map_h"])
    print(f"Cached 2D MAP = {cached_map_h:.4f}, posterior length = {len(posterior_coarse)}")

    # Renormalize and take log to interpolate in log-space (preserves peakedness).
    posterior_coarse = posterior_coarse / np.trapezoid(posterior_coarse, h_coarse)
    log_post_coarse = np.log(np.clip(posterior_coarse, 1e-300, None))

    h_fine = np.arange(FINE_GRID_LO, FINE_GRID_HI + 0.5 * FINE_GRID_STEP, FINE_GRID_STEP)

    # Linear in log-space.
    log_post_fine_lin = np.interp(h_fine, h_coarse, log_post_coarse)
    post_fine_lin = np.exp(log_post_fine_lin - np.max(log_post_fine_lin))
    post_fine_lin /= np.trapezoid(post_fine_lin, h_fine)
    in_peak = (h_fine >= PEAK_REGION_LO) & (h_fine <= PEAK_REGION_HI)
    map_lin = float(h_fine[in_peak][int(np.argmax(post_fine_lin[in_peak]))])

    # Cubic in log-space.
    cs = CubicSpline(h_coarse, log_post_coarse, extrapolate=False)
    log_post_fine_cub = cs(h_fine)
    valid = ~np.isnan(log_post_fine_cub)
    h_fine_v = h_fine[valid]
    log_post_fine_cub_v = log_post_fine_cub[valid]
    post_fine_cub = np.exp(log_post_fine_cub_v - np.max(log_post_fine_cub_v))
    post_fine_cub /= np.trapezoid(post_fine_cub, h_fine_v)
    in_peak_v = (h_fine_v >= PEAK_REGION_LO) & (h_fine_v <= PEAK_REGION_HI)
    map_cub = float(h_fine_v[in_peak_v][int(np.argmax(post_fine_cub[in_peak_v]))])

    delta = abs(map_lin - map_cub)
    print(f"Linear continuous MAP (log-space): {map_lin:.4f}")
    print(f"Cubic continuous MAP  (log-space): {map_cub:.4f}")
    print(
        f"|cubic - linear| = {delta:.4f} (tol {CROSS_CHECK_TOL}; "
        f"{'PASS' if delta < CROSS_CHECK_TOL else 'FAIL'})"
    )

    return {
        "channel": "with_bh_mass_2d_joint_only",
        "discrete_cached_map_h": cached_map_h,
        "continuous_map_h_linear": map_lin,
        "continuous_map_h_cubic": map_cub,
        "delta_linear_cubic": float(delta),
        "cross_check_tol": CROSS_CHECK_TOL,
        "cross_check_pass": bool(delta < CROSS_CHECK_TOL),
        "note": (
            "2D channel has no cached per-event JSONs locally; this is a "
            "joint-posterior interpolation only. No bootstrap. If the linear "
            "and cubic continuous MAPs differ from the discrete cached MAP "
            "by less than Δh=0.005, the discrete shift is at the grid floor; "
            "if they differ by more, the discrete grid was hiding a continuous "
            "shift."
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_1d = analyze_1d_channel()
    summary_2d = analyze_2d_channel_joint_only()

    full = {
        "audit": "A1 — Δh=0.001 reinterpretation",
        "h_truth": H_TRUTH,
        "fine_grid": {
            "lo": FINE_GRID_LO,
            "hi": FINE_GRID_HI,
            "step": FINE_GRID_STEP,
            "peak_region": [PEAK_REGION_LO, PEAK_REGION_HI],
        },
        "preregistered_gates": {
            "G1a_band": [GATE_G1A_LO, GATE_G1A_HI],
            "G1b_band": [GATE_G1B_LO, GATE_G1B_HI],
            "cross_check_tol": CROSS_CHECK_TOL,
        },
        "channel_1d_without_bh_mass": summary_1d,
        "channel_2d_with_bh_mass": summary_2d,
    }
    out_json = OUTPUT_DIR / "fine_grid_map.json"
    with open(out_json, "w") as f:
        json.dump(full, f, indent=2)
    print(f"\nWrote {out_json}")

    print("\n" + "=" * 60)
    print("AUDIT A1 SUMMARY")
    print("=" * 60)
    print("1D channel (per-event):")
    print(f"  Discrete cached MAP:  {summary_1d['discrete_cached_map_h']:.4f}")
    print(f"  Continuous MAP (lin): {summary_1d['continuous_map_h_linear']:.4f}")
    print(f"  Continuous MAP (cub): {summary_1d['continuous_map_h_cubic']:.4f}")
    print(f"  σ_boot:               {summary_1d['sigma_boot']:.4f}")
    print(
        f"  68% HPD:              [{summary_1d['bootstrap_68_hpd'][0]:.4f}, "
        f"{summary_1d['bootstrap_68_hpd'][1]:.4f}]"
    )
    print(f"  Cross-check:          {'PASS' if summary_1d['cross_check_pass'] else 'FAIL'}")
    print(f"  Pre-reg gate:         {summary_1d['gate_verdict']}")
    print("\n2D channel (joint-posterior interp only):")
    print(f"  Discrete cached MAP:  {summary_2d['discrete_cached_map_h']:.4f}")
    print(f"  Continuous MAP (lin): {summary_2d['continuous_map_h_linear']:.4f}")
    print(f"  Continuous MAP (cub): {summary_2d['continuous_map_h_cubic']:.4f}")


if __name__ == "__main__":
    main()
