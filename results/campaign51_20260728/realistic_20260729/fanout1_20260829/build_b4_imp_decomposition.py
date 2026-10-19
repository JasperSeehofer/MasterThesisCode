#!/usr/bin/env python3
"""B4.1 [IMP] part 1 -- per-event covariate table from banked B-SEL/b0i arms.

launched under rows #222/#223 -- charter node B4.1 [IMP] part 1

Zero-compute reconstruction from banked artifacts under
results/campaign51_20260728/realistic_20260729/p3_b0_work/{bc,bt}_9001NN_work/.

What this script recovers (all provenance-tagged in the output CSV):
  - z_true, host_galaxy_index, in_catalog, host_draw_mode, s_tilde_phi_host,
    SNR: direct columns of prepared_cramer_rao_bounds.csv (source: the mirror
    evaluate() input, one row per original detection index 0..N-1).
  - candidate_count_no_bh, candidate_count_with_bh: reconstructed from the
    "possible hosts found A/B..." INFO log line in <arm>.log. These are
    printed once per (event, h) in the SAME iteration order as the
    diagnostics CSV rows for that h-block, for every event EXCEPT the
    zero-host fallback event (which prints a WARNING instead and is
    identified in the CSV by L_cat_no_bh == L_cat_with_bh == 0.0 exactly).
    Verified h-invariant for bc_900101 (bit-identical (n,m) tuples between
    the h=0.50 and h=0.52 blocks) -- structural, because the candidate
    ball's z-window is built from the WIDENED h-bounds spanning the whole
    H-grid, not the current per-call h (bayesian_statistics.py p_D loop /
    correspondence_1d.py run_mirror_seed_inprocess docstring "Note"). Only
    the first h-block's log lines are parsed per arm (zero extra compute).
  - L_cat_no_bh, L_cat_with_bh, B_num, B_num_wbh, g_frac, combined_no_bh,
    combined_with_bh: event_likelihoods.csv diagnostics, evaluated at the
    h-block nearest H=0.73 (constants.py fiducial) present in that arm's
    grid. These are CANDIDATE-BALL AGGREGATES (Gray-2020-style sums over
    every candidate host, true host included when recovered) -- NOT a
    per-candidate breakdown. No column here isolates the impostor-only
    contribution from the true-host contribution.

What is explicitly NOT recoverable (documented, not fabricated):
  - Per-candidate z, mass, sigma_z, or catalogue share c_i for the
    candidates inside each event's ball. The python HostGalaxy lists
    (`candidate_hosts`, `candidate_hosts_with_bh_mass`) built at
    bayesian_statistics.py ~4780-4830 are consumed by `p_Di` and then
    discarded -- never serialized. Confirmed by source grep (no
    per-candidate writer exists) and independently corroborated by
    CLAIM_P3_IMPOSTOR_CONVENTION_20260822.md's 2026-08-22 stage-1
    inventory finding ("the per-host terms FULL-F needs ... are never
    stored").
  - A per-event score at the true host (ln L_cat - ln L_true-host):
    requires exactly the per-candidate breakdown above to subtract the
    true host's own term out of the ball-summed L_cat. Not computable from
    any banked column.
  - A per-event in-ball flag (was the true host among the candidates
    returned): only a SEED-LEVEL aggregate exists (the "P6 host-recovery"
    INFO log line, e.g. "1D 91/106 ... (85.85%), 2D 32/106 ... (30.19%)"),
    logged once per h and h-invariant for the same structural reason as
    the candidate counts. Recorded as arm-level columns
    (recovered_no_bh_frac, recovered_with_bh_frac) -- NOT joinable to
    individual events without a fresh instrumented run.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
P3_B0_WORK = REPO_ROOT / "results/campaign51_20260728/realistic_20260729/p3_b0_work"
OUT_DIR = REPO_ROOT / "results/campaign51_20260728/realistic_20260729/fanout1_20260829"

SEEDS = [900101 + i for i in range(12)]
ARMS = ["bc", "bt"]
FIDUCIAL_H = 0.73

POSSIBLE_HOSTS_RE = re.compile(r"possible hosts found (\d+)/(\d+)\.\.\.")
RECOVERY_RE = re.compile(
    r"P6 host-recovery \(h=([\d.]+)\): "
    r"1D (\d+)/(\d+) hosts recovered/in-cat events seen \(([\d.]+)%\), "
    r"2D (\d+)/(\d+) hosts recovered/in-cat events seen \(([\d.]+)%\)"
)


def parse_arm_log(log_path: Path) -> tuple[list[tuple[int, int]], dict]:
    """Return (possible_hosts_counts_in_log_order, first_recovery_record)."""
    counts: list[tuple[int, int]] = []
    recovery: dict = {}
    with open(log_path) as f:
        for line in f:
            if "possible hosts found" in line:
                m = POSSIBLE_HOSTS_RE.search(line)
                if m:
                    counts.append((int(m.group(1)), int(m.group(2))))
            elif "P6 host-recovery" in line and not recovery:
                m = RECOVERY_RE.search(line)
                if m:
                    recovery = {
                        "h": float(m.group(1)),
                        "n_recovered_no_bh": int(m.group(2)),
                        "n_in_cat_no_bh": int(m.group(3)),
                        "pct_no_bh": float(m.group(4)),
                        "n_recovered_with_bh": int(m.group(5)),
                        "n_in_cat_with_bh": int(m.group(6)),
                        "pct_with_bh": float(m.group(7)),
                    }
    return counts, recovery


def build_arm_table(arm: str, seed: int) -> tuple[pd.DataFrame, dict, str]:
    """Build the per-event covariate rows for one (arm, seed). Returns
    (dataframe, recovery_dict, warning_string_or_empty)."""
    subdir = f"{arm}_{seed}"
    work = P3_B0_WORK / f"{subdir}_work" / f"seed{seed}"
    log_path = P3_B0_WORK / f"{subdir}.log"
    crb_path = work / "simulations" / "prepared_cramer_rao_bounds.csv"
    diag_path = work / "simulations" / "diagnostics" / "event_likelihoods.csv"

    warns: list[str] = []
    if not (log_path.is_file() and diag_path.is_file()):
        return pd.DataFrame(), {}, f"MISSING inputs for {subdir} (no log and/or diagnostics CSV)"

    crb = pd.read_csv(crb_path) if crb_path.is_file() else None
    if crb is None:
        warns.append(
            f"{subdir}: prepared_cramer_rao_bounds.csv not on this disk (large-CSV sync "
            "gap per BIAS_HISTORY_LEDGER push-reject note) -- z_true/host_galaxy_index/"
            "in_catalog/host_draw_mode/SNR are NaN for this arm; candidate counts and "
            "diagnostics aggregates are still recovered."
        )
    diag = pd.read_csv(diag_path)

    h_values = sorted(diag["h"].unique())
    h_first = h_values[0]
    h_fiducial = min(h_values, key=lambda h: abs(h - FIDUCIAL_H))

    block_first = diag[diag["h"] == h_first].reset_index(drop=True)
    block_fid = diag[diag["h"] == h_fiducial].set_index("event_idx")

    counts, recovery = parse_arm_log(log_path)

    zero_host_mask = (block_first["L_cat_no_bh"] == 0.0) & (block_first["L_cat_with_bh"] == 0.0)
    n_zero_host = int(zero_host_mask.sum())
    n_h_blocks = len(h_values)
    expected_block_size = len(block_first) - n_zero_host  # "possible hosts found" prints per h-block
    block1_counts = counts[:expected_block_size]

    if len(counts) != expected_block_size * n_h_blocks:
        warns.append(
            f"{subdir}: total log 'possible hosts found' count ({len(counts)}) does not "
            f"equal expected_block_size ({expected_block_size}) x n_h_blocks ({n_h_blocks}) "
            f"= {expected_block_size * n_h_blocks} -- candidate-count alignment SKIPPED for this arm."
        )
        cand_counts = {int(idx): (np.nan, np.nan) for idx in block_first["event_idx"]}
    elif len(block1_counts) != expected_block_size:
        warns.append(
            f"{subdir}: fewer log lines ({len(block1_counts)}) than expected non-zero-host "
            f"row count ({expected_block_size}) -- candidate-count alignment SKIPPED for this arm."
        )
        cand_counts = {int(idx): (np.nan, np.nan) for idx in block_first["event_idx"]}
    else:
        cand_counts = {}
        ci = 0
        for _, row in block_first.iterrows():
            eidx = int(row["event_idx"])
            if row["L_cat_no_bh"] == 0.0 and row["L_cat_with_bh"] == 0.0:
                cand_counts[eidx] = (0, 0)  # zero-host event: ball search returned nothing usable
            else:
                cand_counts[eidx] = block1_counts[ci]
                ci += 1

    rows = []
    for eidx in block_first["event_idx"]:
        eidx = int(eidx)
        crb_row = crb.iloc[eidx] if crb is not None else {}
        fid_row = block_fid.loc[eidx] if eidx in block_fid.index else None
        n_no_bh, n_with_bh = cand_counts.get(eidx, (np.nan, np.nan))
        in_cat_val = crb_row.get("in_catalog", np.nan) if crb is not None else np.nan
        rows.append(
            {
                "arm": arm,
                "seed": seed,
                "event_idx": eidx,
                "z_true": crb_row.get("z_true", np.nan),
                "host_galaxy_index": crb_row.get("host_galaxy_index", np.nan),
                "in_catalog": crb_row.get("in_catalog", np.nan),
                "host_draw_mode": crb_row.get("host_draw_mode", ""),
                "s_tilde_phi_host": crb_row.get("s_tilde_phi_host", np.nan),
                "SNR": crb_row.get("SNR", np.nan),
                "candidate_count_no_bh": n_no_bh,
                "candidate_count_with_bh": n_with_bh,
                "h_fiducial_used": h_fiducial,
                "L_cat_no_bh": fid_row["L_cat_no_bh"] if fid_row is not None else np.nan,
                "L_cat_with_bh": fid_row["L_cat_with_bh"] if fid_row is not None else np.nan,
                "B_num": fid_row["B_num"] if fid_row is not None else np.nan,
                "B_num_wbh": fid_row["B_num_wbh"] if fid_row is not None else np.nan,
                "g_frac": fid_row["g_frac"] if fid_row is not None else np.nan,
                "combined_no_bh": fid_row["combined_no_bh"] if fid_row is not None else np.nan,
                "combined_with_bh": fid_row["combined_with_bh"] if fid_row is not None else np.nan,
                # Best-available zero-compute impostor-share PROXY (see module
                # docstring / report for what this is and is not):
                # 1.0 exactly when the event's catalogue-leg sum is
                # STRUCTURALLY all-impostor (dark class: not a catalogue
                # member at all; or zero-host: the ball found nothing, so
                # L_cat==0 and no host of any kind contributes); NaN
                # (unknown, cannot bound further from banked columns) when
                # the event IS an in-catalog event with a non-empty ball,
                # because we cannot tell from aggregate columns alone
                # whether the true host is among the (n_no_bh) candidates
                # summed into L_cat_no_bh.
                "impostor_share_proxy_no_bh": (
                    1.0
                    if (in_cat_val is False) or (in_cat_val == 0) or (n_no_bh == 0)
                    else np.nan
                ),
            }
        )
    df = pd.DataFrame(rows)
    return df, recovery, "; ".join(warns)


def main() -> None:
    all_rows = []
    recovery_rows = []
    warnings = []
    for arm in ARMS:
        for seed in SEEDS:
            df, recovery, warn = build_arm_table(arm, seed)
            if warn:
                warnings.append(warn)
            if not df.empty:
                all_rows.append(df)
            if recovery:
                recovery_rows.append({"arm": arm, "seed": seed, **recovery})

    full = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    full.to_csv(OUT_DIR / "b4_imp_decomposition.csv", index=False)

    rec_df = pd.DataFrame(recovery_rows)
    rec_df.to_csv(OUT_DIR / "b4_imp_recovery_by_arm.csv", index=False)

    print(f"wrote {len(full)} event rows across {full['arm'].nunique() if not full.empty else 0} arms")
    print(f"wrote {len(rec_df)} arm-level recovery rows")
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(" -", w)
    else:
        print("No alignment warnings -- candidate-count reconstruction verified consistent in every arm.")

    # Quick sanity echo
    if not full.empty:
        print("\nper-arm event counts:")
        print(full.groupby(["arm", "seed"]).size())
        print("\ncandidate_count_no_bh NaN count (unaligned arms):", full["candidate_count_no_bh"].isna().sum())


if __name__ == "__main__":
    main()
