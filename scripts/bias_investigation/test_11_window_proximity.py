"""Test 11 (Phase 45 Step 1c): 4σ window proximity to d_L = 0.

For each of the 412 production-seed200 events that fed the cached posterior,
computes ``d_L_det − 4σ_dL`` and counts how many would have their
integration window touch ``d_L = 0``.

The threshold is ``σ/d_L > 0.25`` (equivalently ``d_L/σ < 4``) — at that
ratio, the lower edge of the symmetric 4σ window crosses zero.

If ``n_events_touching_zero == 0``, the Alternative-C anchor at d_L=0 has
zero impact on the production posterior and Phase 45 needs a different
angle even if H1 is right (because the close events don't get integrated
that far down). If > 0, the anchor matters and its empirical value
(Test 10) governs the residual bias.

Run: ``uv run python scripts/bias_investigation/test_11_window_proximity.py``
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CRB_PATH = PROJECT_ROOT / "simulations" / "prepared_cramer_rao_bounds.csv"
POSTERIORS_DIR = PROJECT_ROOT / "results" / "phase44_posteriors"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "outputs" / "phase45"

SIGMA_MULT = 4.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load LOCAL production-like CRBs as proxy.
    # NOTE: the actual 412-event posterior was produced on the cluster from
    # /pfs/.../run_20260401_seed200/cramer_rao_bounds.csv (~4500 events) which
    # is not rsync'd back. The local prepared_cramer_rao_bounds.csv (542 rows,
    # SNR>=15) is from the same pipeline with a smaller campaign and serves as
    # a representative d_L/σ distribution proxy. Conclusions about the
    # *fraction* of events with windows touching zero generalize.
    crb = pd.read_csv(CRB_PATH)
    print(f"Loaded {len(crb)} rows from {CRB_PATH.name} (LOCAL proxy)")

    # 2. Filter to SNR>=20 (production threshold, see master_thesis_code.constants.SNR_THRESHOLD).
    sub = (
        crb[crb["SNR"] >= 20.0].copy().reset_index(drop=False).rename(columns={"index": "event_id"})
    )
    print(f"After SNR>=20 filter: {len(sub)} events")
    print("(Cluster posterior used 412 events from a larger campaign; local proxy ratio is")
    print(" what generalizes, not the absolute count.)")

    # Also report cross-check vs posterior IDs (informational only)
    h73_file = POSTERIORS_DIR / "h_0_73.json"
    with open(h73_file) as f:
        data = json.load(f)
    posterior_event_ids = sorted(
        int(k) for k in data.keys() if k != "h" and isinstance(data[k], list) and len(data[k]) > 0
    )
    print(
        f"Posterior IDs range: {min(posterior_event_ids)}..{max(posterior_event_ids)} ({len(posterior_event_ids)} ids; cluster CRB unavailable locally)"
    )

    # 4. Compute σ_dL = sqrt(variance) and the lower edge.
    var_col = "delta_luminosity_distance_delta_luminosity_distance"
    sub["sigma_dL"] = np.sqrt(sub[var_col].clip(lower=0.0))
    sub["dl_minus_4sigma"] = sub["luminosity_distance"] - SIGMA_MULT * sub["sigma_dL"]
    sub["sigma_over_dl"] = sub["sigma_dL"] / sub["luminosity_distance"]
    sub["dl_over_sigma"] = sub["luminosity_distance"] / sub["sigma_dL"]

    # 5. Count events whose 4σ window touches zero.
    touches_zero = sub["dl_minus_4sigma"] <= 0.0
    n_touch = int(touches_zero.sum())
    n_total = int(len(sub))
    print(f"\nEvents with 4σ window touching d_L=0:  {n_touch} / {n_total}")

    sigma_over_dl_max = float(sub["sigma_over_dl"].max())
    sigma_over_dl_p99 = float(sub["sigma_over_dl"].quantile(0.99))
    sigma_over_dl_p95 = float(sub["sigma_over_dl"].quantile(0.95))
    sigma_over_dl_p50 = float(sub["sigma_over_dl"].median())
    dl_over_sigma_min = float(sub["dl_over_sigma"].min())

    # Top close-event candidates
    closest = sub.nsmallest(10, "dl_minus_4sigma")[
        ["event_id", "luminosity_distance", "sigma_dL", "dl_minus_4sigma", "dl_over_sigma", "SNR"]
    ]
    print("\nTop-10 closest events to d_L=0 lower edge:")
    print(closest.to_string(index=False))

    # 6. Also: events where dl_minus_4sigma < dl_centers[0] ≈ 0.10 (where the
    # interpolator first matters for the L_comp integrand)
    n_below_c0 = int((sub["dl_minus_4sigma"] < 0.10).sum())
    print(f"\nEvents with 4σ window crossing dl_centers[0] ≈ 0.10 Gpc: {n_below_c0} / {n_total}")

    summary = {
        "n_events_total": n_total,
        "n_events_touching_zero": n_touch,
        "n_events_window_below_c0_010_gpc": n_below_c0,
        "sigma_mult": SIGMA_MULT,
        "sigma_over_dl_max": sigma_over_dl_max,
        "sigma_over_dl_p99": sigma_over_dl_p99,
        "sigma_over_dl_p95": sigma_over_dl_p95,
        "sigma_over_dl_median": sigma_over_dl_p50,
        "dl_over_sigma_min": dl_over_sigma_min,
        "threshold_sigma_over_dl_for_zero_touch": 1.0 / SIGMA_MULT,
        "closest_events": closest.to_dict(orient="records"),
        "verdict": "alternative_c_anchor_active"
        if n_below_c0 > 0
        else "alternative_c_anchor_inert",
    }

    out_json = OUTPUT_DIR / "window_proximity.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nWrote {out_json}")

    # 7. Plot.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax0, ax1 = axes
    ax0.scatter(
        sub["luminosity_distance"],
        sub["sigma_dL"],
        s=14,
        alpha=0.5,
        c=sub["SNR"],
        cmap="viridis",
    )
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel("d_L [Gpc]")
    ax0.set_ylabel("σ_dL [Gpc]")
    # Boundary line σ = d_L/4
    dl_grid = np.geomspace(sub["luminosity_distance"].min(), sub["luminosity_distance"].max(), 100)
    ax0.plot(
        dl_grid,
        dl_grid / SIGMA_MULT,
        "r-",
        lw=1.0,
        label=f"σ = d_L/{int(SIGMA_MULT)} (4σ window touches 0)",
    )
    ax0.axvline(0.10, color="orange", ls=":", lw=1.0, label="dl_centers[0] ≈ 0.10")
    ax0.set_title(f"Per-event d_L vs σ — {n_touch}/{n_total} touch d_L=0; {n_below_c0} cross c_0")
    ax0.legend(fontsize=8)

    ax1.hist(sub["dl_minus_4sigma"], bins=40, edgecolor="black", alpha=0.7)
    ax1.axvline(0.0, color="red", lw=1.5, label="d_L=0")
    ax1.axvline(0.10, color="orange", ls=":", lw=1.5, label="dl_centers[0] ≈ 0.10")
    ax1.set_xlabel("d_L − 4σ_dL [Gpc] (integration window lower edge)")
    ax1.set_ylabel("count")
    ax1.set_title("Distribution of 4σ-window lower edges")
    ax1.legend(fontsize=8)

    fig.tight_layout()
    out_png = OUTPUT_DIR / "window_proximity.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")

    print("")
    print(f"=== Phase 45 Step 1c verdict: {summary['verdict']} ===")
    print(f"  events touching d_L=0:                {n_touch}/{n_total}")
    print(f"  events crossing dl_centers[0]≈0.10:   {n_below_c0}/{n_total}")
    print(f"  min d_L/σ (most leveraged event):     {dl_over_sigma_min:.2f}")
    print(f"  median σ/d_L:                         {sigma_over_dl_p50:.4f}")


if __name__ == "__main__":
    main()
