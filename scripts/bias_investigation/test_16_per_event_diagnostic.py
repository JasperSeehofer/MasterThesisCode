"""Test 16 (Audit A4): per-event L_cat / L_comp / f_i diagnostic at 412 cluster events.

Resolves concerns #4, #5, #7 simultaneously by identifying which events drive
the +0.025 bias residual, the channel divergence, and whether the [0, c_0]
mechanism (Audit A3 hypothesis) is the right family of fixes.

Reads the per-event diagnostic CSV produced by `bayesian_statistics.py`'s
`_write_diagnostic_csv()` (line 696) under the cluster RUN_DIR and the
matching prepared Cramer-Rao bounds CSV. Both rsynced to
`simulations/cluster_run_phase45_20260501/` (Audit A4 setup).

Pre-registered gates (set BEFORE running):
  G4a: Bias concentrated in <20 events with d_L ≈ c_0(h=0.73)=0.10 Gpc.
       → [0, c_0] mechanism confirmed; A3 verdict applies.
  G4b: Bias broadly distributed (top-10 contribute <30% of total
       log-likelihood pull at the bias-driving direction).
       → mechanism is NOT first-bin asymptote; halt all anchor work,
       escalate to Audit A8 or A7.
  G4c: Channel divergence (1D-2D) concentrated in same events as overall
       bias → both channels operate on same lever; A3 result applies.

Run from project root:
    uv run python scripts/bias_investigation/test_16_per_event_diagnostic.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CLUSTER_DIR = PROJECT_ROOT / "simulations" / "cluster_run_phase45_20260501"
DIAGNOSTIC_CSV = CLUSTER_DIR / "event_likelihoods.csv"
PREPARED_CSV = CLUSTER_DIR / "prepared_cramer_rao_bounds.csv"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "outputs" / "phase45"

H_TRUTH = 0.73
H_BIAS_DIRECTION = 0.755  # 1D channel cluster MAP (away-from-truth direction)
C_0_CLUSTER = 0.10  # cluster first-bin midpoint at h=0.73
C_0_BAND_HI = 0.20  # boundary band (full first bin width)

# Pre-registered acceptance windows
TOP_K = 10
TOP_K_PULL_THRESHOLD = 0.30  # if top-10 contribute < 30% of total pull → broad


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AUDIT A4 — Per-event L_cat / L_comp / f_i diagnostic (412 cluster events)")
    print("=" * 70)

    # 1. Load CSVs.
    diag_raw = pd.read_csv(DIAGNOSTIC_CSV)
    crb = pd.read_csv(PREPARED_CSV)
    print(
        f"Loaded diagnostic CSV: {len(diag_raw)} rows, "
        f"{diag_raw['event_idx'].nunique()} events × {diag_raw['h'].nunique()} h-values"
    )
    # Deduplicate: append-mode CSV from multiple cluster runs (Plan 45-03 single
    # anchor + Plan 45-04 hybrid + possibly a third). The LAST entry per
    # (event_idx, h) is the most recent — the Plan 45-04 hybrid state.
    diag_raw["_row_order"] = np.arange(len(diag_raw))
    diag = (
        diag_raw.sort_values("_row_order")
        .groupby(["event_idx", "h"], as_index=False)
        .last()
        .drop(columns=["_row_order"])
    )
    print(f"After dedup (latest per event×h): {len(diag)} rows")
    print(f"Loaded prepared CRB CSV: {len(crb)} events")

    # The CRB CSV uses the row index as the event index in production code
    # (per `bayesian_statistics.py:312` `for index, detection in
    # self.cramer_rao_bounds.iterrows()`). Build a lookup from event_idx →
    # CRB row.
    crb["_index"] = crb.index
    snr_filtered = crb[crb["SNR"] >= 20.0].copy()
    print(f"After SNR≥20 filter: {len(snr_filtered)} events")

    # Verify event_idx in diag matches indices in snr_filtered.
    diag_event_set = set(diag["event_idx"].unique())
    crb_event_set = set(snr_filtered["_index"].values)
    overlap = diag_event_set & crb_event_set
    print(
        f"Event overlap (diagnostic ∩ CRB): {len(overlap)} / "
        f"diag={len(diag_event_set)}, crb={len(crb_event_set)}"
    )

    # 2. Pull log-likelihoods at h_truth=0.73 and h_bias=0.755.
    diag_truth = diag[np.isclose(diag["h"], H_TRUTH)].set_index("event_idx")
    diag_bias = diag[np.isclose(diag["h"], H_BIAS_DIRECTION)].set_index("event_idx")
    print(
        f"\nAt h={H_TRUTH}: {len(diag_truth)} events; at h={H_BIAS_DIRECTION}: {len(diag_bias)} events"
    )

    # 3. Per-event "pull" toward h_bias vs h_truth: log L_combined(h_bias) - log L_combined(h_truth).
    #    Positive pull = event prefers h_bias over h_truth (drives MAP up).
    common_idx = sorted(set(diag_truth.index) & set(diag_bias.index))
    print(f"Common event_idx with both h-values: {len(common_idx)}")

    pull_no_bh = pd.Series(
        np.log(diag_bias.loc[common_idx, "combined_no_bh"].values + 1e-300)
        - np.log(diag_truth.loc[common_idx, "combined_no_bh"].values + 1e-300),
        index=common_idx,
        name="pull_no_bh",
    )
    pull_with_bh = pd.Series(
        np.log(diag_bias.loc[common_idx, "combined_with_bh"].values + 1e-300)
        - np.log(diag_truth.loc[common_idx, "combined_with_bh"].values + 1e-300),
        index=common_idx,
        name="pull_with_bh",
    )

    # 4. Total pull and concentration.
    total_pull_no_bh = float(pull_no_bh.sum())
    total_pull_with_bh = float(pull_with_bh.sum())
    print("\nTotal log-likelihood pull h_truth → h_bias=0.755:")
    print(f"  1D channel (no_bh): {total_pull_no_bh:+.3f}")
    print(f"  2D channel (with_bh): {total_pull_with_bh:+.3f}")
    print(f"  (positive ⇒ events collectively favour h={H_BIAS_DIRECTION} over h={H_TRUTH})")

    # 5. Top-K events ranked by |pull|.
    top_k_no_bh = pull_no_bh.abs().sort_values(ascending=False).head(TOP_K)
    top_k_with_bh = pull_with_bh.abs().sort_values(ascending=False).head(TOP_K)
    top_k_pull_sum_no_bh = float(pull_no_bh.loc[top_k_no_bh.index].abs().sum())
    top_k_pull_sum_with_bh = float(pull_with_bh.loc[top_k_with_bh.index].abs().sum())
    abs_total_no_bh = float(pull_no_bh.abs().sum())
    abs_total_with_bh = float(pull_with_bh.abs().sum())
    frac_top_k_no_bh = top_k_pull_sum_no_bh / abs_total_no_bh if abs_total_no_bh > 0 else 0.0
    frac_top_k_with_bh = (
        top_k_pull_sum_with_bh / abs_total_with_bh if abs_total_with_bh > 0 else 0.0
    )
    print(f"\nTop-{TOP_K} events by |pull| (1D channel):")
    print(f"  contribute {frac_top_k_no_bh * 100:.1f}% of total |pull|")
    print(f"Top-{TOP_K} events by |pull| (2D channel):")
    print(f"  contribute {frac_top_k_with_bh * 100:.1f}% of total |pull|")

    # 6. Cross-reference top-K events with their d_L / sky angles / SNR.
    crb_by_idx = snr_filtered.set_index("_index")
    print(f"\nTop-{TOP_K} bias-driving events (1D channel):")
    print(
        f"  {'event_idx':>9} {'pull':>9} {'d_L_Gpc':>8} {'σ_dL_Gpc':>8} "
        f"{'SNR':>7} {'in_first_bin':>13} {'in_4σ_of_c_0':>13}"
    )
    top_event_details = []
    for ev in top_k_no_bh.index:
        row = crb_by_idx.loc[ev] if ev in crb_by_idx.index else None
        if row is None:
            continue
        d_L = float(row["luminosity_distance"])
        # σ_dL: pull from Fisher matrix diagonal entry for d_L
        sigma_dl = float(row.get("sigma_dL", row.get("luminosity_distance_uncertainty", np.nan)))
        snr = float(row["SNR"])
        in_first_bin = d_L < C_0_BAND_HI
        in_4sigma_window = (d_L - 4 * sigma_dl) < C_0_CLUSTER if not np.isnan(sigma_dl) else False
        pull_val = float(pull_no_bh[ev])
        print(
            f"  {int(ev):>9} {pull_val:>+9.3f} {d_L:>8.4f} {sigma_dl:>8.4f} "
            f"{snr:>7.1f} {str(in_first_bin):>13} {str(in_4sigma_window):>13}"
        )
        top_event_details.append(
            {
                "event_idx": int(ev),
                "pull_no_bh": pull_val,
                "d_L_Gpc": d_L,
                "sigma_dL_Gpc": sigma_dl if not np.isnan(sigma_dl) else None,
                "SNR": snr,
                "d_L_in_first_bin_below_2c0": bool(in_first_bin),
                "window_crosses_c0": bool(in_4sigma_window),
            }
        )

    # 7. Concentration verdict.
    n_first_bin = sum(1 for d in top_event_details if d["d_L_in_first_bin_below_2c0"])
    n_window_cross = sum(1 for d in top_event_details if d["window_crosses_c0"])
    print(f"\nOf top-{TOP_K} bias-driving events (1D):")
    print(f"  {n_first_bin}/{TOP_K} have d_L < 2·c_0 = 0.20 Gpc")
    print(f"  {n_window_cross}/{TOP_K} have 4σ window crossing c_0 = 0.10 Gpc")

    # 8. Channel divergence: per-event sign of (combined_no_bh - combined_with_bh).
    diag_at_truth = diag_truth.copy()
    diag_at_truth["channel_diff_log"] = np.log(diag_at_truth["combined_no_bh"] + 1e-300) - np.log(
        diag_at_truth["combined_with_bh"] + 1e-300
    )
    median_diff = float(diag_at_truth["channel_diff_log"].median())
    abs_diff = diag_at_truth["channel_diff_log"].abs()
    top_div = abs_diff.sort_values(ascending=False).head(TOP_K)
    print(f"\nChannel divergence at h={H_TRUTH}:")
    print(f"  median(log L_no_bh - log L_with_bh) = {median_diff:+.4f}")
    print(f"  Top-{TOP_K} divergence drivers:")
    print(f"  {'event_idx':>9} {'log_diff':>9} {'d_L_Gpc':>8}")
    div_details = []
    for ev in top_div.index:
        row = crb_by_idx.loc[ev] if ev in crb_by_idx.index else None
        if row is None:
            continue
        d_L = float(row["luminosity_distance"])
        diff = float(diag_at_truth.loc[ev, "channel_diff_log"])
        print(f"  {int(ev):>9} {diff:>+9.4f} {d_L:>8.4f}")
        div_details.append({"event_idx": int(ev), "channel_diff_log": diff, "d_L_Gpc": d_L})

    # Overlap of top-K bias-pull events and top-K channel-divergence events.
    bias_set = {d["event_idx"] for d in top_event_details}
    div_set = {d["event_idx"] for d in div_details}
    overlap_top_k = bias_set & div_set
    print(
        f"\nOverlap between top-{TOP_K} bias-pull events and top-{TOP_K} channel-divergence events: "
        f"{len(overlap_top_k)} events"
    )

    # 9. f_i distribution.
    f_i_at_truth = diag_at_truth["f_i"].values
    print(f"\nf_i distribution at h={H_TRUTH} (catalog completeness fraction):")
    print(f"  median: {float(np.median(f_i_at_truth)):.4f}")
    print(f"  mean:   {float(np.mean(f_i_at_truth)):.4f}")
    print(f"  fraction f_i > 0.5 (catalog-dominated): {float(np.mean(f_i_at_truth > 0.5)):.3f}")
    print(f"  fraction f_i < 0.1 (completion-dominated): {float(np.mean(f_i_at_truth < 0.1)):.3f}")

    # 10. Pre-registered gate classification.
    # G4a: bias concentrated in <20 boundary events, where boundary = d_L < 2*c_0
    # G4b: bias broadly distributed (top-10 contribute <30% of |pull|)
    # G4c: channel divergence drivers overlap with bias-pull drivers
    if frac_top_k_no_bh < TOP_K_PULL_THRESHOLD:
        gate = (
            f"G4b (bias broadly distributed; top-{TOP_K} contribute "
            f"{frac_top_k_no_bh * 100:.1f}% < {TOP_K_PULL_THRESHOLD * 100}% of |pull|)"
        )
    elif n_window_cross >= TOP_K // 2:
        gate = (
            f"G4a ([0, c_0] mechanism confirmed; {n_window_cross}/{TOP_K} top "
            f"events have window crossing c_0)"
        )
    else:
        gate = (
            "G4-MIXED (bias concentrated but not in [0, c_0] events; investigate individual events)"
        )
    print(f"\n>>> Pre-registered gate verdict: {gate}\n")

    # Channel divergence vs pull overlap → G4c
    g4c_satisfied = len(overlap_top_k) >= TOP_K // 2
    print(
        f"G4c (channel-divergence ∩ bias-pull overlap ≥ {TOP_K // 2}/{TOP_K}): "
        f"{'TRUE' if g4c_satisfied else 'FALSE'}"
    )

    # 11. Decompose MAP into Σ log L_i and -N log D(h) contributions.
    cluster_combined_path = (
        PROJECT_ROOT / "results" / "phase45_v2_posteriors" / "combined_posterior.json"
    )
    with open(cluster_combined_path) as f:
        cp = json.load(f)
    h_grid = np.array(cp["h_values"])
    D_h_grid = np.array(cp["D_h_per_h"])
    log_D_h = np.log(D_h_grid)

    # Build per-event log-L matrix at the cluster h_grid.
    log_L = np.full((len(common_idx), len(h_grid)), np.nan)
    for i, ev in enumerate(common_idx):
        sub = diag[diag["event_idx"] == ev].sort_values("h")
        for j, hv in enumerate(h_grid):
            idx = np.argmin(np.abs(sub["h"].values - hv))
            if abs(sub["h"].values[idx] - hv) < 1e-4:
                log_L[i, j] = np.log(max(sub["combined_no_bh"].values[idx], 1e-300))

    n_eff = len(common_idx)
    L_term = np.nansum(log_L, axis=0)
    D_term = -n_eff * log_D_h
    joint_log = L_term + D_term

    map_idx = int(np.argmax(joint_log))
    L_only_idx = int(np.argmax(L_term))
    print("\n" + "=" * 70)
    print("MAP DECOMPOSITION: Σ log L_i(h) vs −N log D(h)")
    print("=" * 70)
    print(f"  Joint MAP (Σ log L − N log D):  h={h_grid[map_idx]:.4f}")
    print(f"  Σ log L_i(h) alone peaks at:    h={h_grid[L_only_idx]:.4f}")
    print(f"  −N log D(h) peaks at h_max:     h={h_grid[int(np.argmax(D_term))]:.4f}")
    print()
    i_truth = np.argmin(np.abs(h_grid - H_TRUTH))
    i_map = map_idx
    print(
        f"  At h={H_TRUTH:.4f}:  Σ log L = {L_term[i_truth]:.2f}, -N log D = {D_term[i_truth]:.2f}"
    )
    print(
        f"  At h={h_grid[i_map]:.4f}: Σ log L = {L_term[i_map]:.2f}, -N log D = {D_term[i_map]:.2f}"
    )
    delta_L = float(L_term[i_map] - L_term[i_truth])
    delta_D = float(D_term[i_map] - D_term[i_truth])
    print(f"  Δ Σ log L: {delta_L:+.3f} (per-event preference)")
    print(f"  Δ -N log D: {delta_D:+.3f} (selection-function correction)")
    print(f"  Total Δ log p: {delta_L + delta_D:+.3f}")
    print()
    print(
        f"  *** D(h) effect is {abs(delta_D) / max(abs(delta_L), 1e-9):.1f}× larger than per-event L pull ***"
    )
    if (delta_L < 0) and (delta_D > 0):
        print(
            f"  *** Signs are OPPOSITE: events alone would prefer h={H_TRUTH}; D(h) pulls to h={h_grid[i_map]:.4f} ***"
        )

    summary_decomp = {
        "joint_map_h": float(h_grid[map_idx]),
        "L_only_peak_h": float(h_grid[L_only_idx]),
        "delta_L_truth_to_map": delta_L,
        "delta_minusN_logD_truth_to_map": delta_D,
        "delta_total_log_post_truth_to_map": delta_L + delta_D,
        "D_h_dominance_ratio": abs(delta_D) / max(abs(delta_L), 1e-9),
        "interpretation": (
            f"Per-event likelihoods alone peak at h={h_grid[L_only_idx]:.4f} "
            f"(within σ_boot of truth h={H_TRUTH}); D(h) selection-function "
            f"correction shifts MAP to h={h_grid[map_idx]:.4f} (+{(h_grid[map_idx] - h_grid[L_only_idx]):.4f})."
        ),
    }

    # 12. Save JSON summary.
    summary = {
        "audit": "A4 — Per-event L_cat/L_comp/f_i diagnostic (412 cluster events)",
        "n_events_in_diagnostic": int(diag["event_idx"].nunique()),
        "n_events_overlap_with_crb": len(overlap),
        "h_truth": H_TRUTH,
        "h_bias_direction": H_BIAS_DIRECTION,
        "c_0_cluster_Gpc": C_0_CLUSTER,
        "total_pull_h_truth_to_h_bias": {
            "1D_no_bh": total_pull_no_bh,
            "2D_with_bh": total_pull_with_bh,
        },
        "top_k": TOP_K,
        "frac_top_k_pull_no_bh": frac_top_k_no_bh,
        "frac_top_k_pull_with_bh": frac_top_k_with_bh,
        "top_event_details_1D": top_event_details,
        "top_event_divergence_details": div_details,
        "n_top_k_in_first_bin": n_first_bin,
        "n_top_k_with_window_crossing_c0": n_window_cross,
        "n_top_k_overlap_bias_and_divergence": len(overlap_top_k),
        "f_i_at_truth_stats": {
            "median": float(np.median(f_i_at_truth)),
            "mean": float(np.mean(f_i_at_truth)),
            "frac_above_0.5_catalog_dominated": float(np.mean(f_i_at_truth > 0.5)),
            "frac_below_0.1_completion_dominated": float(np.mean(f_i_at_truth < 0.1)),
        },
        "gate_verdict_concentration": gate,
        "gate_g4c_overlap": bool(g4c_satisfied),
        "map_decomposition": summary_decomp,
    }

    out_json = OUTPUT_DIR / "per_event_diagnostic.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
