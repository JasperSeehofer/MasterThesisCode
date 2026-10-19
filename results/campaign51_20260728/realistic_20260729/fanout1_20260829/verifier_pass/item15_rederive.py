#!/usr/bin/env python3
"""
Item 15 verifier re-derivation (fresh, from-source, no trust of the record's own numbers).

Re-derives, from the raw CSV/JSON artifacts named by REGISTRATION_C0_BASELINE_GATE_20260829.md
and the ledger row #246 / COMPUTE_LEDGER.md entries:

  (A) the C0 bit-identity gate: max |x - y| over the 14 non-trivial shared numeric columns,
      wave2_20260829/c0/diagnostics/event_likelihoods.csv (h=0.73, 1588 rows)
      vs headreadout_20260827/iiib/event_likelihoods.csv (filtered to h=0.73, should be 1588 rows)

  (B) posterior-file identity via independently computed md5 (not trusting the record's md5s)

  (C) the costing-anchor correction: 00:06:28 elapsed x 16 cpus/task = CPU-h; compare against the
      15-23 CPU-h estimate; and check that the 56-76 min/h-value anchor text
      (cluster/LAUNCHING_JOBS.md:47) names the 3355-event population, not this run's population.

No numbers are read from any record's own restatement -- only from the CSV/JSON/text sources.
"""
import csv
import hashlib
import re
import sys
from pathlib import Path

REPO = Path("/home/jasper/Repositories/darksiren-emri")
C0_CSV = REPO / "results/campaign51_20260728/realistic_20260729/wave2_20260829/c0/diagnostics/event_likelihoods.csv"
BANKED_CSV = REPO / "results/campaign51_20260728/realistic_20260729/headreadout_20260827/iiib/event_likelihoods.csv"
C0_RUN_META = REPO / "results/campaign51_20260728/realistic_20260729/wave2_20260829/c0/run_metadata_21.json"
C0_POST1 = REPO / "results/campaign51_20260728/realistic_20260729/wave2_20260829/c0/posteriors/h_0_73.json"
C0_POST2 = REPO / "results/campaign51_20260728/realistic_20260729/wave2_20260829/c0/posteriors_with_bh_mass/h_0_73.json"
BANKED_POST1 = REPO / "results/campaign51_20260728/realistic_20260729/headreadout_20260827/iiib/posteriors/h_0_73.json"
BANKED_POST2 = REPO / "results/campaign51_20260728/realistic_20260729/headreadout_20260827/iiib/posteriors_with_bh_mass/h_0_73.json"
LAUNCHING_JOBS = REPO / "cluster/LAUNCHING_JOBS.md"
COMPUTE_LEDGER = REPO / "results/campaign51_20260728/realistic_20260729/fanout1_20260829/COMPUTE_LEDGER.md"

SHARED_NUMERIC_COLS = [
    "w_G", "w_G_legacy", "w_tilde_G", "alpha_G_phi", "r_Malm", "D_tilde_phi",
    "L_cat_no_bh", "L_cat_with_bh", "B_num", "B_num_wbh", "g_frac", "L_comp",
    "combined_no_bh", "combined_with_bh",
]


def load_rows(path, h_filter=None):
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        n_read = 0
        for row in reader:
            n_read += 1
            if h_filter is not None:
                h = float(row["h"])
                if abs(h - h_filter) > 1e-9:
                    continue
            rows[row["event_idx"]] = row
    return header, rows, n_read


def main():
    print("=== Item 15 (C0 baseline gate) independent re-derivation ===\n")

    # ---- (A) load both CSVs ----
    print(f"[A] Loading candidate: {C0_CSV}")
    c0_header, c0_rows, c0_n_read = load_rows(C0_CSV)
    print(f"    header ({len(c0_header)} fields): {c0_header}")
    print(f"    rows read: {c0_n_read}, unique event_idx: {len(c0_rows)}")

    print(f"\n[A] Loading banked baseline: {BANKED_CSV} (filtering h=0.73)")
    banked_header, banked_rows, banked_n_read = load_rows(BANKED_CSV, h_filter=0.73)
    print(f"    header ({len(banked_header)} fields): {banked_header}")
    print(f"    total data rows in file (all h): {banked_n_read}")
    print(f"    rows at h=0.73: {len(banked_rows)}")

    # sanity: 65108 = 1588 * 41
    assert banked_n_read == 65108, f"expected 65108 total data rows, got {banked_n_read}"
    assert banked_n_read == 1588 * 41, "1588*41 identity check failed"

    # column-set check
    c0_col_set = set(c0_header)
    banked_col_set = set(banked_header)
    extra_in_c0 = c0_col_set - banked_col_set
    missing_from_c0 = banked_col_set - c0_col_set
    print(f"\n[A] columns only in c0 (candidate superset): {sorted(extra_in_c0)}")
    print(f"[A] columns in banked but missing from c0: {sorted(missing_from_c0)}")
    assert missing_from_c0 == set(), "banked columns missing from candidate -- gate cannot cover them"
    assert extra_in_c0 == {"den_log_term", "num_log_term_no_bh", "num_log_term_with_bh"}, (
        f"unexpected extra-column set: {extra_in_c0}"
    )

    # event_idx set check
    c0_idx = set(c0_rows.keys())
    banked_idx = set(banked_rows.keys())
    print(f"\n[A] event_idx set equal: {c0_idx == banked_idx}")
    print(f"    c0 count={len(c0_idx)}, banked(h=0.73) count={len(banked_idx)}")
    if c0_idx != banked_idx:
        print(f"    only in c0: {sorted(c0_idx - banked_idx)[:10]}")
        print(f"    only in banked: {sorted(banked_idx - c0_idx)[:10]}")

    common_idx = c0_idx & banked_idx

    # ---- decisive number: max abs diff over 14 non-trivial shared numeric columns ----
    max_abs_overall = 0.0
    max_abs_per_col = {c: 0.0 for c in SHARED_NUMERIC_COLS}
    max_abs_loc = None
    n_compared = 0
    n_nan_mismatch = 0

    for idx in common_idx:
        r_c0 = c0_rows[idx]
        r_bk = banked_rows[idx]
        for col in SHARED_NUMERIC_COLS:
            v_c0_raw = r_c0[col]
            v_bk_raw = r_bk[col]
            try:
                v_c0 = float(v_c0_raw)
                v_bk = float(v_bk_raw)
            except ValueError:
                n_nan_mismatch += 1
                continue
            # handle nan
            import math
            if math.isnan(v_c0) and math.isnan(v_bk):
                n_compared += 1
                continue
            if math.isnan(v_c0) or math.isnan(v_bk):
                n_nan_mismatch += 1
                continue
            d = abs(v_c0 - v_bk)
            n_compared += 1
            if d > max_abs_per_col[col]:
                max_abs_per_col[col] = d
            if d > max_abs_overall:
                max_abs_overall = d
                max_abs_loc = (idx, col, v_c0, v_bk)

    print(f"\n[A] DECISIVE NUMBER -- max_abs over {len(SHARED_NUMERIC_COLS)} columns, "
          f"{len(common_idx)} common events, {n_compared} value-pairs compared, "
          f"{n_nan_mismatch} nan-mismatches:")
    print(f"    max_abs = {max_abs_overall!r}")
    if max_abs_loc:
        print(f"    (at event_idx={max_abs_loc[0]}, col={max_abs_loc[1]}, "
              f"c0={max_abs_loc[2]!r}, banked={max_abs_loc[3]!r})")
    print("    per-column max_abs:")
    for col, v in max_abs_per_col.items():
        print(f"      {col:20s} {v!r}")

    gate_pass = (max_abs_overall == 0.0) and (n_nan_mismatch == 0) and (c0_idx == banked_idx)
    print(f"\n[A] GATE VERDICT (max_abs==0, no nan-mismatches, index sets equal): "
          f"{'PASS' if gate_pass else 'FAIL'}")

    # ---- (B) posterior identity via independent md5 ----
    def md5_of(path):
        return hashlib.md5(path.read_bytes()).hexdigest()

    print("\n[B] Posterior md5 identity (independently computed):")
    for label, a, b in [
        ("posteriors/h_0_73.json", C0_POST1, BANKED_POST1),
        ("posteriors_with_bh_mass/h_0_73.json", C0_POST2, BANKED_POST2),
    ]:
        ma, mb = md5_of(a), md5_of(b)
        print(f"    {label}: c0={ma} banked={mb} match={ma == mb}")

    # ---- (C) costing anchor correction ----
    print("\n[C] Costing-anchor correction re-derivation:")
    meta_text = C0_RUN_META.read_text()
    import json as _json
    meta = _json.loads(meta_text)
    slurm_job = meta["slurm"]["SLURM_JOB_ID"]
    cpus_per_task = int(meta["slurm"]["SLURM_CPUS_PER_TASK"])
    print(f"    run_metadata_21.json: SLURM_JOB_ID={slurm_job}, cpus_per_task={cpus_per_task}, "
          f"git_commit={meta['git_commit']}")
    assert slurm_job == "6738998", f"job id mismatch: {slurm_job}"
    assert cpus_per_task == 16

    # Elapsed 00:06:28 is claimed in the record from sacct; sacct itself is not retrievable
    # (no local logs/ dir under wave2_20260829/c0, and no ssh available per task constraints).
    # We can only check: (i) whether any local artifact independently states this Elapsed,
    # and (ii) the anchor-source claim in cluster/LAUNCHING_JOBS.md, and (iii) the internal
    # arithmetic of the CPU-h conversion itself (which does not depend on sacct access).
    elapsed_str = "00:06:28"
    h_, m_, s_ = (int(x) for x in elapsed_str.split(":"))
    elapsed_hours = h_ + m_ / 60 + s_ / 3600
    cpu_h = elapsed_hours * cpus_per_task
    print(f"    Elapsed (as claimed in record, sacct not independently retrievable without ssh): "
          f"{elapsed_str} = {elapsed_hours:.6f} h")
    print(f"    CPU-h = {elapsed_hours:.6f} h * {cpus_per_task} cpus = {cpu_h:.4f} CPU-h")
    est_lo, est_hi = 15, 23
    ratio_lo = est_lo / cpu_h
    ratio_hi = est_hi / cpu_h
    print(f"    vs estimate [{est_lo}, {est_hi}] CPU-h -> overestimate factor "
          f"[{ratio_lo:.2f}x, {ratio_hi:.2f}x]")

    print(f"\n[C] cluster/LAUNCHING_JOBS.md:47 anchor-source text:")
    lines = LAUNCHING_JOBS.read_text().splitlines()
    line47 = lines[46]  # 0-indexed
    print(f"    line 47: {line47!r}")
    m = re.search(r"56.{0,3}76 min per h-value @ (\d+) events", line47)
    if m:
        print(f"    events-in-anchor-population parsed: {m.group(1)}")
        anchor_pop = int(m.group(1))
        assert anchor_pop == 3355, f"anchor population mismatch: {anchor_pop}"
        assert anchor_pop != 1588, "anchor population equals this run's population -- claim refuted"
        print(f"    CONFIRMED: anchor population ({anchor_pop}) != this run's population (1588)")
    else:
        print("    WARNING: could not parse anchor line by regex -- inspect manually")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
