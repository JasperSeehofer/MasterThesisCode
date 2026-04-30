"""Test 10 (Phase 45 Step 1b): empirical p_det asymptote at d_L → 0.

Two complementary measurements:

(a) Histogram-based detection rate on the close-by injection subset
    (d_L < threshold). Reports n_detected / n_total with Wilson 95% CI.
(b) Direct query of the production interpolator
    ``SimulationDetectionProbability`` at small d_L's to see what the
    likelihood currently uses.

The combination tells us:
- (a) gives the empirical anchor ``p_max_empirical`` if H1 is right.
- (b) confirms the interpolator returns the histogram first-bin mean
  ``p̂(c_0) ≈ 0.55`` for d_L < c_0 (nearest-neighbour fill).

Run: ``uv run python scripts/bias_investigation/test_10_pdet_asymptote.py``
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.stats import binom_conf_interval

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.grid_quality import load_injection_data
from master_thesis_code.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)
from master_thesis_code.constants import SNR_THRESHOLD

OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "outputs" / "phase45"
H_INJ = 0.73
DL_THRESHOLDS_GPC = [0.10, 0.15, 0.20]  # injection min is 0.051; <0.10 has only 16 events
QUERY_DLS_GPC = [0.001, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30]


def empirical_asymptote(df: pd.DataFrame, threshold: float, snr_thr: float) -> dict:
    """Empirical detection fraction in [0, threshold] Gpc with Wilson CI."""
    mask = df["luminosity_distance"] < threshold
    n_total = int(mask.sum())
    n_det = int(((df["SNR"] >= snr_thr) & mask).sum())
    if n_total == 0:
        return {
            "threshold_gpc": threshold,
            "n_total": 0,
            "n_detected": 0,
            "p_hat": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }
    p_hat = n_det / n_total
    ci = binom_conf_interval(n_det, n_total, confidence_level=0.95, interval="wilson")
    return {
        "threshold_gpc": threshold,
        "n_total": n_total,
        "n_detected": n_det,
        "p_hat": float(p_hat),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
    }


def query_interpolator(injection_dir: Path, h: float) -> list[dict]:
    """Build the production p_det interpolator and query at small d_L."""
    snr_dist = SimulationDetectionProbability(
        injection_data_dir=str(injection_dir),
        snr_threshold=float(SNR_THRESHOLD),
    )
    out: list[dict] = []
    for dl in QUERY_DLS_GPC:
        try:
            p = float(
                snr_dist.detection_probability_without_bh_mass_interpolated_zero_fill(
                    np.float64(dl), 0.0, 0.0, h=h
                )
            )
        except Exception as exc:
            p = float("nan")
            print(f"  query d_L={dl}: ERROR {exc}")
        out.append({"d_L_gpc": dl, "p_det": p})
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inj_dir = PROJECT_ROOT / "simulations" / "injections"
    print(f"Loading injection campaign from {inj_dir}")
    injection_groups = load_injection_data(inj_dir)
    df = injection_groups[H_INJ]
    print(f"Pooled {len(df)} injections at h_inj={H_INJ}")
    print(f"  d_L min = {df['luminosity_distance'].min():.4f} Gpc")
    print(f"  SNR threshold = {SNR_THRESHOLD}")

    # (a) Empirical asymptote at varying thresholds
    print("\n=== (a) Empirical detection fraction (Wilson 95% CI) ===")
    empirical = []
    for thr in DL_THRESHOLDS_GPC:
        result = empirical_asymptote(df, thr, float(SNR_THRESHOLD))
        empirical.append(result)
        print(
            f"  d_L < {result['threshold_gpc']} Gpc: "
            f"n={result['n_total']:3d}, n_det={result['n_detected']:3d}, "
            f"p_hat={result['p_hat']:.3f} [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]"
        )

    # (b) Interpolator query
    print("\n=== (b) Interpolator query (production zero_fill variant) ===")
    interp_values = query_interpolator(inj_dir, H_INJ)
    for v in interp_values:
        print(f"  p_det(d_L={v['d_L_gpc']:.3f} Gpc) = {v['p_det']:.4f}")

    # Suggested anchor: tightest empirical estimate (smallest threshold with n>=15)
    suggested = next((r for r in empirical if r["n_total"] >= 15), empirical[0])
    p_max_empirical = suggested["p_hat"]
    p_max_lower = suggested["ci_lower"]
    p_max_upper = suggested["ci_upper"]
    print(
        f"\nSuggested empirical anchor: p_max_empirical = {p_max_empirical:.3f} "
        f"[{p_max_lower:.3f}, {p_max_upper:.3f}] from d_L < {suggested['threshold_gpc']} Gpc "
        f"(n={suggested['n_total']})"
    )

    summary = {
        "h_inj": H_INJ,
        "snr_threshold": float(SNR_THRESHOLD),
        "empirical_by_threshold": empirical,
        "interpolator_query": interp_values,
        "p_max_empirical": float(p_max_empirical),
        "p_max_ci_95": [float(p_max_lower), float(p_max_upper)],
        "anchor_threshold_gpc": float(suggested["threshold_gpc"]),
        "anchor_n_total": int(suggested["n_total"]),
        "naive_asymptote_assumption": 1.0,
        "current_nn_fill_value_at_dl_zero": next(
            (v["p_det"] for v in interp_values if v["d_L_gpc"] <= 0.05), float("nan")
        ),
    }

    out_json = OUTPUT_DIR / "pdet_asymptote.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")

    # Plot
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    # Empirical: error-bar at threshold midpoint (we use threshold itself as x for clarity)
    thrs = [r["threshold_gpc"] for r in empirical]
    p_hats = [r["p_hat"] for r in empirical]
    ci_lows = [max(0.0, r["p_hat"] - r["ci_lower"]) for r in empirical]
    ci_highs = [max(0.0, r["ci_upper"] - r["p_hat"]) for r in empirical]
    ax.errorbar(
        thrs,
        p_hats,
        yerr=[ci_lows, ci_highs],
        fmt="o",
        color="C0",
        capsize=4,
        label="empirical p̂ (Wilson 95%)",
    )
    # Interpolator query
    interp_dls = [v["d_L_gpc"] for v in interp_values]
    interp_ps = [v["p_det"] for v in interp_values]
    ax.plot(interp_dls, interp_ps, "s--", color="C1", label="interpolator (zero-fill)")
    ax.axhline(1.0, color="green", ls=":", alpha=0.6, label="naive asymptote (1.0)")
    ax.axhline(
        p_max_empirical,
        color="C0",
        ls=":",
        alpha=0.5,
        label=f"empirical anchor = {p_max_empirical:.3f}",
    )
    ax.set_xlabel("d_L threshold (or query point) [Gpc]")
    ax.set_ylabel("p_det")
    ax.set_xscale("log")
    ax.set_xlim(5e-4, 0.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Phase 45 Step 1b — p_det asymptote at d_L → 0; h={H_INJ}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_png = OUTPUT_DIR / "pdet_asymptote.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
