"""Test 9 (Phase 45 Step 1a): First-bin d_L density shape.

Histograms d_L of all injection events at h=0.73 within
[0, 2 * dl_centers[0](h=0.73)] = [0, 0.20 Gpc] (the first p_det bin).
Classifies the shape as uniform / upper-edge-skew / lower-edge-skew so we
can interpret what ``p̂(c_0) ≈ 0.55`` is averaging over.

If skew is "upper-edge", events near the bin upper bound (low p_det)
dominate the bin mean → the mean *underestimates* p_det at d_L → 0
→ supports H1 (first-bin underestimate).

Run: ``uv run python scripts/bias_investigation/test_09_first_bin_density.py``
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.grid_quality import load_injection_data

OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "outputs" / "phase45"

H_INJ = 0.73
DL_BIN_UPPER = 0.20  # ~ 2 * dl_centers[0](h=0.73)
N_SUB_BINS = 10
SKEW_RATIO_UNIFORM = 1.5  # ratio threshold separating uniform from skewed


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inj_dir = PROJECT_ROOT / "simulations" / "injections"
    print(f"Loading injection campaign from {inj_dir}")
    injection_groups = load_injection_data(inj_dir)
    if H_INJ not in injection_groups:
        raise RuntimeError(f"h={H_INJ} not in injection groups: {sorted(injection_groups.keys())}")
    df = injection_groups[H_INJ]
    print(f"Pooled {len(df)} injections at h_inj={H_INJ}")

    dl = df["luminosity_distance"].values
    in_first_bin = dl < DL_BIN_UPPER
    n_first_bin = int(in_first_bin.sum())
    print(f"  injections within [0, {DL_BIN_UPPER}] Gpc: {n_first_bin}")
    if n_first_bin < 10:
        print("WARNING: <10 injections in first bin — skew classification noisy.")

    edges = np.linspace(0.0, DL_BIN_UPPER, N_SUB_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(dl[in_first_bin], bins=edges)

    n_lower3 = int(counts[:3].sum())
    n_upper3 = int(counts[-3:].sum())
    if n_lower3 == 0 and n_upper3 == 0:
        verdict = "empty"
        skew_ratio = float("nan")
    elif n_lower3 == 0:
        verdict = "upper-skew"
        skew_ratio = float("inf")
    else:
        skew_ratio = n_upper3 / n_lower3
        if skew_ratio > SKEW_RATIO_UNIFORM:
            verdict = "upper-skew"
        elif skew_ratio < 1.0 / SKEW_RATIO_UNIFORM:
            verdict = "lower-skew"
        else:
            verdict = "uniform"

    weighted_mean_dl = (
        float(np.average(centers, weights=counts)) if counts.sum() > 0 else float("nan")
    )

    summary = {
        "h_inj": H_INJ,
        "dl_bin_range_gpc": [0.0, DL_BIN_UPPER],
        "n_sub_bins": N_SUB_BINS,
        "edges_gpc": [float(e) for e in edges],
        "centers_gpc": [float(c) for c in centers],
        "counts": [int(c) for c in counts],
        "n_first_bin_total": n_first_bin,
        "n_lower3": n_lower3,
        "n_upper3": n_upper3,
        "skew_ratio_upper_over_lower": skew_ratio,
        "skew_ratio_threshold": SKEW_RATIO_UNIFORM,
        "verdict": verdict,
        "weighted_mean_dl_in_bin_gpc": weighted_mean_dl,
        "bin_midpoint_gpc": DL_BIN_UPPER / 2.0,
    }

    out_json = OUTPUT_DIR / "first_bin_density.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_json}")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(centers, counts, width=DL_BIN_UPPER / N_SUB_BINS * 0.9, edgecolor="black", alpha=0.7)
    ax.axvline(DL_BIN_UPPER / 2.0, color="green", ls="--", lw=1.5, label="bin midpoint c_0")
    if not np.isnan(weighted_mean_dl):
        ax.axvline(
            weighted_mean_dl,
            color="red",
            ls="--",
            lw=1.5,
            label=f"weighted mean d_L = {weighted_mean_dl:.4f}",
        )
    ax.set_xlabel("d_L within first p_det bin [Gpc]")
    ax.set_ylabel("injection count")
    ax.set_title(f"Phase 45 Step 1a — first-bin density at h={H_INJ}; verdict: {verdict}")
    ax.legend()
    fig.tight_layout()
    out_png = OUTPUT_DIR / "first_bin_density.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")

    print("")
    print(f"=== Phase 45 Step 1a verdict: {verdict.upper()} ===")
    print(f"  n_first_bin = {n_first_bin}")
    print(f"  lower 3 sub-bins: {n_lower3}  |  upper 3 sub-bins: {n_upper3}")
    print(f"  skew ratio (upper/lower) = {skew_ratio:.2f}")
    print(
        f"  weighted mean d_L in bin = {weighted_mean_dl:.4f}  (midpoint = {DL_BIN_UPPER / 2:.4f})"
    )


if __name__ == "__main__":
    main()
