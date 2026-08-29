#!/usr/bin/env python3
"""
Bonus verifier check for item 15 (C0 baseline gate): run the still-outstanding
REGISTRATION_C0_BASELINE_GATE_20260829.md §11.2 identity check on the retrieved c0 CSV
(SYNTHESIS_DOCKET_2_20260829.md sec7 item 7 lists this as not yet run). Zero additional
cluster compute -- the CSV is already local.

Checks, per bayesian_statistics.py:5827-5837 (cited in the registration):
  num_log_term_no_bh   - den_log_term == log(combined_no_bh)      (rows where finite)
  num_log_term_with_bh - den_log_term == log(combined_with_bh)    (rows where finite)
tolerance: <= 1e-12 absolute, on rows where both terms are finite.
NaN rows: check NaN-pattern agreement between the two channels' log-term columns.
"""
import csv
import math
from pathlib import Path

CSV = Path("/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729/wave2_20260829/c0/diagnostics/event_likelihoods.csv")

def to_float(s):
    try:
        v = float(s)
    except ValueError:
        return float("nan")
    return v

max_abs_no_bh = 0.0
max_abs_with_bh = 0.0
n_checked_no_bh = 0
n_checked_with_bh = 0
n_nan_rows = 0
nan_pattern_mismatches = 0
worst_no_bh = None
worst_with_bh = None

with open(CSV, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        den = to_float(row["den_log_term"])
        num_no = to_float(row["num_log_term_no_bh"])
        num_wbh = to_float(row["num_log_term_with_bh"])
        comb_no = to_float(row["combined_no_bh"])
        comb_wbh = to_float(row["combined_with_bh"])

        # no_bh channel
        finite_no = not (math.isnan(den) or math.isnan(num_no))
        if finite_no:
            expected = math.log(comb_no) if comb_no > 0 else float("nan")
            lhs = num_no - den
            if not math.isnan(expected):
                d = abs(lhs - expected)
                n_checked_no_bh += 1
                if d > max_abs_no_bh:
                    max_abs_no_bh = d
                    worst_no_bh = (row["event_idx"], lhs, expected, d)
            else:
                # combined <= 0 but num/den finite -- pattern mismatch
                nan_pattern_mismatches += 1
        else:
            n_nan_rows += 1
            if comb_no > 0:
                nan_pattern_mismatches += 1

        # with_bh channel
        finite_wbh = not (math.isnan(den) or math.isnan(num_wbh))
        if finite_wbh:
            expected = math.log(comb_wbh) if comb_wbh > 0 else float("nan")
            lhs = num_wbh - den
            if not math.isnan(expected):
                d = abs(lhs - expected)
                n_checked_with_bh += 1
                if d > max_abs_with_bh:
                    max_abs_with_bh = d
                    worst_with_bh = (row["event_idx"], lhs, expected, d)
            else:
                nan_pattern_mismatches += 1
        else:
            if comb_wbh > 0:
                nan_pattern_mismatches += 1

print(f"no_bh channel:   n_checked={n_checked_no_bh}, max_abs_diff={max_abs_no_bh!r}, worst={worst_no_bh}")
print(f"with_bh channel: n_checked={n_checked_with_bh}, max_abs_diff={max_abs_with_bh!r}, worst={worst_with_bh}")
print(f"nan-row count (den/num nonfinite in either channel, counted once per row): {n_nan_rows}")
print(f"nan-pattern mismatches (finite log-term but non-positive combined, or vice versa): {nan_pattern_mismatches}")

TOL = 1e-12
pass_ = (max_abs_no_bh <= TOL) and (max_abs_with_bh <= TOL) and (nan_pattern_mismatches == 0)
print(f"\nSec11.2 IDENTITY CHECK VERDICT (tol {TOL}): {'PASS' if pass_ else 'FAIL'}")
