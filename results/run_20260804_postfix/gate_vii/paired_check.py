#!/usr/bin/env python3
"""
Paired check of dark-survivor likelihood delta across venues.

Computes Δ_e = ln(L_cat_with_bh/L_cat_no_bh)@h=0.81 − same@h=0.73 for dark
survivors in each venue (iiib, joint_r1), forms the intersection, and reports
paired statistics.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Paths
iiib_likelihoods = "/home/jasper/Repositories/MasterThesisCode/results/run_20260804_postfix/iiib/diagnostics/event_likelihoods.csv"
iiib_bounds = "/home/jasper/Repositories/MasterThesisCode/results/run_20260804_postfix/iiib/diagnostics/prepared_cramer_rao_bounds.csv"
joint_likelihoods = "/home/jasper/Repositories/MasterThesisCode/results/run_20260804_postfix/joint_r1/diagnostics/event_likelihoods.csv"

# Read data
print("Reading likelihoods and bounds...")
iiib_lh = pd.read_csv(iiib_likelihoods)
joint_lh = pd.read_csv(joint_likelihoods)
bounds = pd.read_csv(iiib_bounds)

# Identify dark events (host_galaxy_index < 0)
# Note: bounds has one row per event, indexed by row number starting at 0
dark_event_indices = set(bounds[bounds['host_galaxy_index'] < 0].index.tolist())
print(f"Total dark events: {len(dark_event_indices)}")

# For each venue, identify survivors
def find_survivors(lh_df, venue_name):
    """
    A survivor is an event where L_cat_no_bh > 0 AND L_cat_with_bh > 0
    at ALL 41 h values.
    """
    # Group by event_idx and check conditions for all h values
    survivors = set()
    for event_idx in lh_df['event_idx'].unique():
        event_data = lh_df[lh_df['event_idx'] == event_idx]
        # Check if L_cat_no_bh > 0 AND L_cat_with_bh > 0 for all rows
        if (event_data['L_cat_no_bh'] > 0).all() and (event_data['L_cat_with_bh'] > 0).all():
            survivors.add(event_idx)
    print(f"{venue_name}: {len(survivors)} survivors out of {lh_df['event_idx'].nunique()}")
    return survivors

iiib_survivors = find_survivors(iiib_lh, "iiib")
joint_survivors = find_survivors(joint_lh, "joint_r1")

# Form dark survivors
iiib_dark_survivors = dark_event_indices & iiib_survivors
joint_dark_survivors = dark_event_indices & joint_survivors

print(f"iiib dark survivors: {len(iiib_dark_survivors)}")
print(f"joint_r1 dark survivors: {len(joint_dark_survivors)}")

# Form intersection
intersection = iiib_dark_survivors & joint_dark_survivors
print(f"Intersection (dark survivors in both venues): {len(intersection)}")

if len(intersection) == 0:
    print("ERROR: Empty intersection!")
    exit(1)

# Compute Δ_e for each venue on the intersection
def compute_delta(lh_df, event_indices, venue_name):
    """
    For each event in event_indices, compute:
    Δ_e = ln(L_cat_with_bh/L_cat_no_bh)@h=0.81 − same@h=0.73

    Returns dict: event_idx -> Δ_e
    """
    deltas = {}
    for event_idx in event_indices:
        event_data = lh_df[lh_df['event_idx'] == event_idx].sort_values('h')

        # Find rows for h=0.81 and h=0.73
        row_81 = event_data[event_data['h'] == 0.81]
        row_73 = event_data[event_data['h'] == 0.73]

        if len(row_81) == 0 or len(row_73) == 0:
            print(f"  WARNING: {venue_name} event {event_idx} missing h value")
            continue

        L_no_bh_81 = row_81['L_cat_no_bh'].values[0]
        L_with_bh_81 = row_81['L_cat_with_bh'].values[0]
        L_no_bh_73 = row_73['L_cat_no_bh'].values[0]
        L_with_bh_73 = row_73['L_cat_with_bh'].values[0]

        # Compute log ratio at each h
        log_ratio_81 = np.log(L_with_bh_81 / L_no_bh_81)
        log_ratio_73 = np.log(L_with_bh_73 / L_no_bh_73)

        # Δ_e = difference
        delta = log_ratio_81 - log_ratio_73
        deltas[event_idx] = delta

    print(f"{venue_name}: computed Δ_e for {len(deltas)} events")
    return deltas

iiib_deltas = compute_delta(iiib_lh, intersection, "iiib")
joint_deltas = compute_delta(joint_lh, intersection, "joint_r1")

# Ensure both have the same events (intersection alignment)
aligned_events = set(iiib_deltas.keys()) & set(joint_deltas.keys())
print(f"Aligned events in both deltas: {len(aligned_events)}")

# Extract paired data
delta_iiib = np.array([iiib_deltas[e] for e in sorted(aligned_events)])
delta_joint = np.array([joint_deltas[e] for e in sorted(aligned_events)])

# Compute paired statistics
d_e = delta_iiib - delta_joint  # per-event difference

# Ratio: guard division
ratio_e = []
n_near_zero = 0
for i in range(len(delta_joint)):
    if abs(delta_joint[i]) < 1e-6:
        n_near_zero += 1
    else:
        ratio_e.append(delta_iiib[i] / delta_joint[i])

ratio_e = np.array(ratio_e)

# Compute statistics
N = len(aligned_events)
median_d = np.median(d_e)
mean_d = np.mean(d_e)
std_d = np.std(d_e)

median_r = np.median(ratio_e) if len(ratio_e) > 0 else np.nan
p16_r = np.percentile(ratio_e, 16) if len(ratio_e) > 0 else np.nan
p84_r = np.percentile(ratio_e, 84) if len(ratio_e) > 0 else np.nan

# Spearman correlation
corr, pval = spearmanr(delta_iiib, delta_joint)

# Fraction with |r_e - 1| < threshold
frac_05 = np.sum(np.abs(ratio_e - 1) < 0.05) / len(ratio_e) if len(ratio_e) > 0 else np.nan
frac_20 = np.sum(np.abs(ratio_e - 1) < 0.20) / len(ratio_e) if len(ratio_e) > 0 else np.nan

# Aggregate per-event means on intersection
mean_delta_iiib = np.mean(delta_iiib)
mean_delta_joint = np.mean(delta_joint)

# Build output dictionary
output = {
    "intersection_size": N,
    "n_events_with_delta_joint_near_zero": n_near_zero,
    "n_events_in_ratio_statistics": len(ratio_e),
    "median_d_e": float(median_d),
    "mean_d_e": float(mean_d),
    "std_d_e": float(std_d),
    "median_r_e": float(median_r),
    "p16_r_e": float(p16_r),
    "p84_r_e": float(p84_r),
    "spearman_correlation": float(corr),
    "spearman_p_value": float(pval),
    "fraction_r_e_within_0_05": float(frac_05),
    "fraction_r_e_within_0_20": float(frac_20),
    "aggregate_mean_delta_iiib": float(mean_delta_iiib),
    "aggregate_mean_delta_joint": float(mean_delta_joint),
}

# Write JSON output
output_path = "/home/jasper/Repositories/MasterThesisCode/results/run_20260804_postfix/gate_vii/paired_check.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nOutput written to {output_path}")

# Print summary (7 sf where meaningful)
print("\n=== RESULTS ===")
print(f"Intersection size: {N}")
print(f"n_events with |Δ_e^joint| < 1e-6: {n_near_zero}")
print(f"n_events in ratio statistics: {len(ratio_e)}")
print(f"median(d_e): {median_d:.7g}")
print(f"mean(d_e): {mean_d:.7g}")
print(f"std(d_e): {std_d:.7g}")
print(f"median(r_e): {median_r:.7g}")
print(f"p16(r_e): {p16_r:.7g}")
print(f"p84(r_e): {p84_r:.7g}")
print(f"Spearman correlation: {corr:.7g}")
print(f"Spearman p-value: {pval:.7g}")
print(f"Fraction |r_e - 1| < 0.05: {frac_05:.7g}")
print(f"Fraction |r_e - 1| < 0.20: {frac_20:.7g}")
print(f"Aggregate mean Δ_e^iiib: {mean_delta_iiib:.7g}")
print(f"Aggregate mean Δ_e^joint: {mean_delta_joint:.7g}")
