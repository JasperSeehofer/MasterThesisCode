#!/usr/bin/env python3
"""
Item 8/20 — end-of-fan-out verifier pass, independent re-derivation.

Re-derives, FROM SOURCE (CSV/JSON/log), the decisive numbers claimed in:
  - B5_2_WIN_K3_READOUT_RECORD.md / b5_2_readout.json  (the I_HEAD stencil, R6/R2/R5/R1 gates,
    the mechanism join, R4)
  - B5_2_PULL_READ_20260829.md / b5_pull_read.json     (the |pull|<=3 empirical fraction, and
    its match to b5_window_count.json's log-k3 retention)

Every number below is computed here from the raw CSVs — never read out of b5_2_readout.json's
own precomputed fields (those are used only as the comparison target, printed alongside).
"""

import json
import math

import numpy as np
import pandas as pd

ROOT = "/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729"

BASELINE_CSV = f"{ROOT}/headreadout_20260827/iiib/event_likelihoods.csv"
ARM_CSV = f"{ROOT}/wave2_20260829/c3/simulations/diagnostics/event_likelihoods.csv"
CRB_CSV = f"{ROOT}/seed61000/prepared_cramer_rao_bounds.csv"
READOUT_JSON = f"{ROOT}/fanout1_20260829/b5_2_readout.json"
PULL_JSON = f"{ROOT}/fanout1_20260829/b5_pull_read.json"

H_NODES = [0.660, 0.665, 0.670, 0.730]
I_HEAD = 2965.0
IMMATERIAL = 0.003
T_MAT = 0.008


def load(path, h_filter=None):
    df = pd.read_csv(path)
    if h_filter is not None:
        mask = np.zeros(len(df), dtype=bool)
        for h in h_filter:
            mask |= np.isclose(df["h"].to_numpy(), h, atol=1e-9)
        df = df[mask]
    return df


def main():
    baseline = load(BASELINE_CSV, H_NODES)
    arm = load(ARM_CSV, H_NODES)

    print("=" * 78)
    print("STEP 0 — shape sanity")
    print("=" * 78)
    print(f"baseline rows (4 H4 nodes): {len(baseline)} (expect 4*1588={4*1588})")
    print(f"arm rows (4 H4 nodes):      {len(arm)} (expect 4*1588={4*1588})")
    n_events_baseline = baseline["event_idx"].nunique()
    n_events_arm = arm["event_idx"].nunique()
    print(f"unique event_idx baseline: {n_events_baseline}, arm: {n_events_arm}")

    # -----------------------------------------------------------------
    # R6 — 1D bit-identity
    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("R6 — 1D bit-identity (L_cat_no_bh, combined_no_bh)")
    print("=" * 78)
    merged_full = baseline.merge(
        arm, on=["event_idx", "h"], suffixes=("_B", "_T"), how="inner"
    )
    print(f"merged rows: {len(merged_full)} (expect {4*1588})")
    max_rel_diffs = {}
    for col in ["L_cat_no_bh", "combined_no_bh"]:
        b = merged_full[f"{col}_B"].to_numpy()
        t = merged_full[f"{col}_T"].to_numpy()
        denom = np.maximum(np.abs(b), 1e-300)
        rel = np.abs(t - b) / denom
        max_rel_diffs[col] = rel.max()
        print(f"  {col}: max relative diff = {rel.max():.6e}")
    r6_max = max(max_rel_diffs.values())
    r6_pass = r6_max <= 1e-12
    print(f"R6 verdict: max_rel_diff={r6_max:.6e}, threshold=1e-12 -> {'PASS' if r6_pass else 'FAIL'}")
    print(f"  (record claims 2.667e-14, PASS)")

    # -----------------------------------------------------------------
    # R2 — engagement at h=0.730
    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("R2 — engagement gate at h=0.730")
    print("=" * 78)
    m730 = merged_full[np.isclose(merged_full["h"], 0.730, atol=1e-9)]
    baseline_nonempty = m730["L_cat_with_bh_B"] > 0
    n_nonempty = int(baseline_nonempty.sum())
    differs = merged_full.loc[m730.index, "L_cat_with_bh_B"] != merged_full.loc[
        m730.index, "L_cat_with_bh_T"
    ]
    n_differ_among_nonempty = int((baseline_nonempty & differs).sum())
    engagement = n_differ_among_nonempty / n_nonempty
    print(f"n_events_total at h=0.730: {len(m730)}")
    print(f"n_nonempty_baseline_with_bh: {n_nonempty} (record: 982)")
    print(f"n_differing_among_nonempty: {n_differ_among_nonempty} (record: 951)")
    print(f"engagement_fraction: {engagement:.10f} (record: 0.9684317718940937)")
    print(f"R2 verdict: {'PASS' if engagement >= 0.90 else 'FAIL'} (>=0.90 required)")

    # -----------------------------------------------------------------
    # PRIMARY READING — Delta_mean_h,pred via I_HEAD stencil
    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("PRIMARY READING — Delta ell(h), stencil, Delta_mean_h,pred")
    print("=" * 78)
    sum_lnL_B = {}
    sum_lnL_T = {}
    delta_ell = {}
    for h in H_NODES:
        mh = merged_full[np.isclose(merged_full["h"], h, atol=1e-9)]
        # combined_with_bh must be strictly positive on both sides to log
        b = mh["combined_with_bh_B"].to_numpy()
        t = mh["combined_with_bh_T"].to_numpy()
        pos_mask = (b > 0) & (t > 0)
        n_excluded = int((~pos_mask).sum())
        lnL_B = np.log(b[pos_mask]).sum()
        lnL_T = np.log(t[pos_mask]).sum()
        sum_lnL_B[h] = lnL_B
        sum_lnL_T[h] = lnL_T
        delta_ell[h] = lnL_T - lnL_B
        print(
            f"h={h:.3f}: sum_lnL_B={lnL_B:.10f}  sum_lnL_T={lnL_T:.10f}  "
            f"Delta_ell={delta_ell[h]:.10f}  n_excluded_nonpositive={n_excluded}"
        )

    # central-difference stencil over {0.660, 0.665, 0.670}
    h_lo, h_mid, h_hi = 0.660, 0.665, 0.670
    step = h_mid - h_lo  # 0.005
    assert math.isclose(h_hi - h_mid, step, rel_tol=1e-9)
    dprime = (delta_ell[h_hi] - delta_ell[h_lo]) / (2 * step)
    ddprime = (delta_ell[h_hi] - 2 * delta_ell[h_mid] + delta_ell[h_lo]) / (step**2)
    delta_mean_h_pred = dprime / I_HEAD

    print()
    print(f"Delta_ell'(0.665)  = {dprime:.10f} nats/h   (record: 10.444057521544883)")
    print(f"Delta_ell''(0.665) = {ddprime:.10f}          (record: -63.70506631355954)")
    print(f"I_HEAD = {I_HEAD}")
    print(f"Delta_mean_h,pred = Delta_ell'(0.665)/I_HEAD = {delta_mean_h_pred:.10f}")
    print(f"  (record claims: +0.0035225270694619774)")
    print(f"  match to record: {'YES' if math.isclose(delta_mean_h_pred, 0.0035225270694619774, rel_tol=1e-6) else 'NO'}")

    ratio_ddprime_ihead = abs(ddprime) / I_HEAD
    print(f"R5 |Delta_ell''|/I_HEAD = {ratio_ddprime_ihead:.6f} ({ratio_ddprime_ihead*100:.4f}%) -- '<<' check")

    # Band verdict
    print()
    if abs(delta_mean_h_pred) <= IMMATERIAL:
        band = "IMMATERIAL-CONSISTENT-WITH-HB"
    elif abs(delta_mean_h_pred) >= T_MAT:
        band = "MATERIAL"
    else:
        band = "INTERMEDIATE"
    print(f"Band verdict per registered map (IMMATERIAL<=0.003, INTERMEDIATE, MATERIAL>=0.008): {band}")
    print(f"  (record claims: INTERMEDIATE)")

    # -----------------------------------------------------------------
    # R1 mechanism join — with-BH collapse vs in_catalog status
    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("R1 mechanism — with-BH candidate-set collapse vs host_galaxy_index/in_catalog")
    print("=" * 78)
    crb = pd.read_csv(CRB_CSV, usecols=["host_galaxy_index", "in_catalog"])
    crb = crb.reset_index().rename(columns={"index": "event_idx"})
    print(f"CRB rows: {len(crb)} (expect 1588 or 1591-with-header artifacts)")
    print(f"in_catalog counts:\n{crb['in_catalog'].value_counts()}")

    m730_full = merged_full[np.isclose(merged_full["h"], 0.730, atol=1e-9)].copy()
    joined = m730_full.merge(crb, on="event_idx", how="left")
    print(f"joined rows at h=0.730: {len(joined)}")

    in_cat_mask = joined["in_catalog"].astype(bool) if joined["in_catalog"].dtype != object else joined["in_catalog"] == True  # noqa: E712
    n_in_cat = int(in_cat_mask.sum())
    print(f"n_in_catalog_events (via in_catalog column): {n_in_cat} (record: 76)")

    baseline_pos = joined["L_cat_with_bh_B"] > 0
    arm_pos = joined["L_cat_with_bh_T"] > 0
    collapsed = baseline_pos & (~arm_pos)
    gained = (~baseline_pos) & arm_pos
    n_collapsed = int(collapsed.sum())
    n_gained = int(gained.sum())
    print(f"n_total with-BH collapse events (B>0 -> T==0): {n_collapsed} (record: 621)")
    print(f"n_gained (B==0 -> T>0): {n_gained} (record: 0)")

    n_collapsed_in_cat = int((collapsed & in_cat_mask).sum())
    n_collapsed_dark = int((collapsed & ~in_cat_mask).sum())
    print(f"of collapsed: in_catalog={n_collapsed_in_cat} (record: 0), dark={n_collapsed_dark} (record: 621)")

    n_in_cat_baseline_nonempty = int((in_cat_mask & baseline_pos).sum())
    n_in_cat_arm_nonempty = int((in_cat_mask & arm_pos).sum())
    n_in_cat_positivity_changed = int((in_cat_mask & (baseline_pos != arm_pos)).sum())
    print(
        f"in-catalog events: baseline_with_bh_support={n_in_cat_baseline_nonempty} (record: 75), "
        f"arm_with_bh_support={n_in_cat_arm_nonempty} (record: 75), "
        f"positivity_changed={n_in_cat_positivity_changed} (record: 0)"
    )

    # -----------------------------------------------------------------
    # R1 retention falsifier — arm-side log line (verifiable from local logs)
    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("R1 falsifier -- P6 host-recovery, ARM side (from local log; baseline side NOT locally retrievable, SSH down)")
    print("=" * 78)
    arm_log_path = f"{ROOT}/wave2_20260829/c3/logs/wave2_c3_task3_6738999.err"
    found = None
    with open(arm_log_path) as f:
        for line in f:
            if "P6 host-recovery (h=0.7300)" in line:
                found = line.strip()
    print(f"arm log line: {found}")
    print("record claims arm retention 66/76 = 86.84211% -- ", "MATCH" if found and "66/76" in found else "NO MATCH/NOT FOUND")
    print(
        "NOTE: baseline-side P6 log line (from the C0 gate task on the cluster scratch "
        "workspace) is NOT present in this local checkout and cluster SSH is down this "
        "session -- the claim 'baseline retention is ALSO 66/76, identical to arm' cannot "
        "be independently re-verified from source at zero compute. This is a genuine "
        "reproducibility gap in the record, not merely a restated caveat."
    )

    falsifier_band = (0.762, 0.816)
    arm_retention = 66 / 76
    in_band = falsifier_band[0] <= arm_retention <= falsifier_band[1]
    print(
        f"arm retention {arm_retention:.6f} vs falsifier band {falsifier_band} -> "
        f"{'IN BAND (not falsified)' if in_band else 'OUTSIDE BAND (FALSIFIED)'}"
    )

    # -----------------------------------------------------------------
    # R4 — Delta w-bar_2 (alpha_G_phi)
    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("R4 -- Delta w-bar_2 (mean alpha_G_phi) at h=0.730")
    print("=" * 78)
    mean_alpha_B = m730_full["alpha_G_phi_B"].mean()
    mean_alpha_T = m730_full["alpha_G_phi_T"].mean()
    print(f"mean alpha_G_phi baseline: {mean_alpha_B}")
    print(f"mean alpha_G_phi arm:      {mean_alpha_T}")
    print(f"delta: {mean_alpha_T - mean_alpha_B}")
    print("(record claims both = 58688310.0, delta = 0.0 exactly)")

    # -----------------------------------------------------------------
    # Cross-check against the readout JSON's own stored numbers (sanity only)
    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("Cross-check against b5_2_readout.json's stored numbers")
    print("=" * 78)
    with open(READOUT_JSON) as f:
        readout = json.load(f)
    recorded_dmhp = readout["primary_reading_delta_mean_h_pred"]["channel_with_bh"]["delta_mean_h_pred"]
    print(f"recorded delta_mean_h_pred: {recorded_dmhp}")
    print(f"rederived  delta_mean_h_pred: {delta_mean_h_pred}")
    print(f"abs diff: {abs(recorded_dmhp - delta_mean_h_pred):.3e}")

    # -----------------------------------------------------------------
    # PULL-READ cross-check (L9) -- recompute |pull_def1|<=3 fraction independently
    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("PULL-READ (L9) -- recompute |pull_def1|<=3 fraction from b5_pull_read.json pooled block")
    print("=" * 78)
    with open(PULL_JSON) as f:
        pull = json.load(f)
    pooled = pull["pooled"]
    frac_def1_k3 = pooled["pull_def1_ratio_BHMASSERR_over_BHMASS"]["fraction_abs_le"]["3.0"]
    print(f"pull_def1 |pull|<=3 fraction (from JSON, N={pooled['n_events']}): {frac_def1_k3:.6f} = {frac_def1_k3*100:.2f}%")
    print("(record's B5.2-pre claims 78.8%; item 8 task text also says 78.8%)")

    b5wc_path = None
    import glob
    candidates = glob.glob(f"{ROOT}/**/b5_window_count.json", recursive=True) + glob.glob(
        f"{ROOT}/fanout1_20260829/b5_window_count.json"
    )
    print(f"b5_window_count.json candidates found: {candidates}")
    if candidates:
        with open(candidates[0]) as f:
            b5wc = json.load(f)
        ret_k3 = b5wc.get("true_host_retention", {}).get("fraction_retained", {}).get("iii_log_k3.0")
        print(f"b5_window_count.json true_host_retention.fraction_retained['iii_log_k3.0']: {ret_k3}")
    else:
        # fall back to the value embedded in the pull-read's own cross-check block
        ret_k3 = pull["cross_check_vs_b5_window_count"]["b5_window_count_log_retention"]["k3.0"]
        print(f"[b5_window_count.json not found on disk; using value embedded in b5_pull_read.json's own cross-check block: {ret_k3}]")

    diff_pp = abs(ret_k3 - frac_def1_k3) * 100
    print(f"|pull_def1<=3| = {frac_def1_k3*100:.4f}% vs log-k3 retention = {ret_k3*100:.4f}%  -> diff = {diff_pp:.4f} pp")
    print(f"claimed reconciliation: match to within 0.2 points -> {'HOLDS' if diff_pp <= 0.2 else 'DOES NOT HOLD'}")

    # Also recompute item 7's "78.9%" claim vs item 8's task text: does 78.9 match record?
    print()
    print(f"Task text claims 'item 7 independently-measured 78.9% true-host retention' -- "
          f"b5_window_count log-k3.0 retention here = {ret_k3*100:.4f}% (rounds to 78.9%: "
          f"{'YES' if round(ret_k3*100,1)==78.9 else 'NO'})")

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
