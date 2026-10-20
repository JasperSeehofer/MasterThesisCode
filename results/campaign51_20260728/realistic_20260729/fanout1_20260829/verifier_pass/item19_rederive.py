#!/usr/bin/env python3
"""
Item 19 (end-of-fan-out verifier pass, row #222 registration) — Compute ledger totals
and F4 compliance re-derivation.

Re-derives, from the elapsed-time / cpu-count primitives quoted at the point of
measurement in the record set (COMPUTE_LEDGER.md prose + the arm readout JSONs
b5_2_readout.json / b4_2_readout.json), the decisive numbers:

  1. C0 + C3 + C4 measured wave-2 cluster total, and its ratio against the
     registered 179-357 CPU-h estimate band (WAVE2_REGISTRATION_CHECK_20260829.md
     Sec 0 item 7 / COMPUTE_LEDGER.md "Wave 2 cost refinements" table).
  2. row #249's KW-Q1 (P2) figure: 6.152 CPU-h measured vs 8.4 CPU-h estimate,
     cross-checked against b4_2_readout.json's own cost_measured block (JSON
     source, not a record restating it).
  3. The row #252 chair fan-out-to-date total (cluster + local wave-2 + wave-1
     local).
  4. The wave-3 estimate arithmetic (41 x 16 x [274,398] s, x[2.2,3.0] joint_r1
     multiplier) quoted in SYNTHESIS_DOCKET_2_20260829.md line 47.
  5. F4 deadline-gate arithmetic: days remaining to workspace expiry 2026-09-23
     from the task's stated "as of" date 2026-08-30.

IMPORTANT CAVEAT (disclosed, not concealed): the underlying SLURM `Elapsed`
strings for C0/C3/C4 (e.g. "00:06:28") are NOT available in this repo as a raw
`sacct` dump -- no such file was retrieved before the cluster SSH outage (see
B7_2_TWIN_CF_READOUT_RECORD.md / PROPOSAL_2D_TWIN_ADOPTION_20260829.md Sec 15,
"Cluster SSH went down ... mid-retrieval"). This script re-derives the ARITHMETIC
from the elapsed-time strings as quoted, cross-checked for internal consistency
across every independent citation (COMPUTE_LEDGER.md, BIAS_HISTORY_LEDGER.md rows
#246-249/252, the JSON cost fields where they exist, PROPOSAL_2D_TWIN_ADOPTION
Sec 15). It CANNOT independently re-run `sacct` against job IDs 6738998-6739001
because this pass forbids ssh. That specific input (raw sacct authenticity) is
UNDETERMINED-AT-ZERO-COMPUTE, not confirmed or refuted, and is reported as such.
"""

import json
from pathlib import Path

REPO = Path("/home/jasper/Repositories/darksiren-emri")
FAN = REPO / "results/campaign51_20260728/realistic_20260729/fanout1_20260829"


def hms_to_hours(hms: str) -> float:
    h, m, s = (int(x) for x in hms.split(":"))
    return h + m / 60 + s / 3600


def main() -> None:
    print("=" * 78)
    print("1. C0 measured (job 6738998, Elapsed 00:06:28, 16 cpus/task)")
    c0 = hms_to_hours("00:06:28") * 16
    print(f"   -> {c0:.4f} CPU-h  (ledger states 1.7; row #252 states 1.72)")

    print("\n2. C3 measured (job 6738999 array 0-3, 16 cpus/task each)")
    c3_elapsed = ["00:04:50", "00:04:39", "00:04:36", "00:04:34"]
    c3_per_task = [hms_to_hours(e) * 16 for e in c3_elapsed]
    c3 = sum(c3_per_task)
    for e, v in zip(c3_elapsed, c3_per_task, strict=True):
        print(f"     {e} -> {v:.4f} CPU-h")
    print(f"   -> sum = {c3:.4f} CPU-h  (ledger states 4.97)")

    print("\n3. C4 measured (job 6739000 task0 + 6739001 tasks1-3, 16 cpus/task)")
    c4_elapsed = ["00:06:25", "00:06:38", "00:06:17", "00:06:10"]
    c4_per_task = [hms_to_hours(e) * 16 for e in c4_elapsed]
    c4 = sum(c4_per_task)
    for e, v in zip(c4_elapsed, c4_per_task, strict=True):
        print(f"     {e} -> {v:.4f} CPU-h")
    print(f"   -> sum = {c4:.4f} CPU-h  (ledger states 6.8)")

    cluster_total = c0 + c3 + c4
    print(f"\n4. C0+C3+C4 measured cluster total = {cluster_total:.4f} CPU-h "
          f"(ledger/row#252 state ~13.5 / 13.50)")

    band_lo, band_hi = 179, 357
    ratio_lo = band_lo / cluster_total
    ratio_hi = band_hi / cluster_total
    print(f"\n5. Ratio vs registered full wave-2-cluster band [{band_lo}, {band_hi}] CPU-h "
          f"(C0+C1+C3+C4, C1 NOT launched):")
    print(f"   {band_lo}/{cluster_total:.4f} = {ratio_lo:.2f}x .. "
          f"{band_hi}/{cluster_total:.4f} = {ratio_hi:.2f}x")
    print("   claimed range in the verifier item text: ~13-26x -> "
          f"{'CONSISTENT' if 12 <= ratio_lo <= 14 and 25 <= ratio_hi <= 28 else 'MISMATCH'}")

    # Apples-to-oranges check: comparable band excluding C1 (not launched/not incurred)
    c0_band = (15, 23)
    c3_band = (44, 137)
    c4_band = (60, 105)
    comp_lo = c0_band[0] + c3_band[0] + c4_band[0]
    comp_hi = c0_band[1] + c3_band[1] + c4_band[1]
    print(f"\n5b. Methodological check: estimate band for the 3 arms ACTUALLY launched "
          f"(C0+C3+C4 only, excluding C1's un-incurred 60-92 CPU-h estimate) = "
          f"[{comp_lo}, {comp_hi}] CPU-h")
    print(f"    -> ratio vs that narrower, launched-arms-only band: "
          f"{comp_lo / cluster_total:.2f}x .. {comp_hi / cluster_total:.2f}x")
    print("    NOTE: the 179-357 band includes C1's 60-92 CPU-h estimate for an arm that did "
          "NOT launch; comparing 3-arm measured cost against the 4-arm estimate band is a "
          "mild apples-to-oranges framing (inflates the favorable-miss ratio at the margins), "
          "though every individual arm's own ratio (8.8x-27.6x) already brackets both framings.")

    print("\n" + "=" * 78)
    print("6. Per-arm own ratio ranges (measured vs that arm's own estimate band)")
    for name, meas, (lo, hi) in [
        ("C0", c0, c0_band),
        ("C3", c3, c3_band),
        ("C4", c4, c4_band),
    ]:
        print(f"   {name}: measured {meas:.3f} CPU-h vs [{lo},{hi}] "
              f"-> {lo / meas:.2f}x .. {hi / meas:.2f}x")

    print("\n" + "=" * 78)
    print("7. P2 (KW-Q1) row #249 figure, cross-checked against b4_2_readout.json "
          "cost_measured block (source JSON, not a record restating it)")
    b4 = json.loads((FAN / "b4_2_readout.json").read_text())
    cm = b4["cost_measured"]
    main_cpu_h = cm["main_run_wall_s"] / 3600 * cm["main_run_cpu_per_job"]
    parity_cpu_h = cm["parity_run_wall_s"] / 3600 * cm["parity_run_cpu_per_job"]
    total = main_cpu_h + parity_cpu_h
    print(f"   main_run:   wall {cm['main_run_wall_s']:.3f}s x {cm['main_run_cpu_per_job']} cpu "
          f"= {main_cpu_h:.6f} CPU-h  (JSON main_run_cpu_h={cm['main_run_cpu_h']:.6f})")
    print(f"   parity_run: wall {cm['parity_run_wall_s']:.3f}s x {cm['parity_run_cpu_per_job']} cpu "
          f"= {parity_cpu_h:.6f} CPU-h  (JSON parity_run_cpu_h={cm['parity_run_cpu_h']:.6f})")
    print(f"   total re-derived = {total:.6f} CPU-h  (JSON total_cpu_h={cm['total_cpu_h']:.6f}; "
          f"ledger states 6.152)")
    est = cm["registered_estimate_cpu_h"]
    ratio = total / est
    print(f"   ratio measured/estimate = {ratio:.4f} "
          f"(JSON ratio_measured_over_estimate={cm['ratio_measured_over_estimate']:.4f}; "
          f"i.e. {(1 - ratio) * 100:.1f}% below the {est} CPU-h estimate; "
          "ledger states '~27% below')")

    print("\n" + "=" * 78)
    print("8. Row #252 chair fan-out-to-date total")
    local_wave2 = 6.152 + 11.51 + 10.42  # P2 + P0(S0-A) + S0-C
    wave1_local = 11.4
    fanout_total = cluster_total + local_wave2 + wave1_local
    print(f"   cluster (C0+C3+C4) = {cluster_total:.2f}")
    print(f"   local wave-2 (P2 6.152 + P0 11.51 + S0-C 10.42) = {local_wave2:.2f}")
    print(f"   wave-1 local (~11.4)")
    print(f"   -> fan-out total to date = {fanout_total:.2f} CPU-h "
          f"(row #252 states ~53.0)")

    print("\n" + "=" * 78)
    print("9. Wave-3 estimate arithmetic (SYNTHESIS_DOCKET_2 line 47 / chair (i))")
    n_tasks_per_venue = 41
    cpus = 16
    lo_s, hi_s = 274, 398
    iiib_lo = n_tasks_per_venue * cpus * lo_s / 3600
    iiib_hi = n_tasks_per_venue * cpus * hi_s / 3600
    joint_lo = iiib_lo * 2.2
    joint_hi = iiib_hi * 3.0
    total_lo = iiib_lo + joint_lo
    total_hi = iiib_hi + joint_hi
    print(f"   iiib:      {iiib_lo:.2f} - {iiib_hi:.2f} CPU-h "
          f"(docket states 49.93-72.52)")
    print(f"   joint_r1:  {joint_lo:.2f} - {joint_hi:.2f} CPU-h "
          f"(docket states 109.84-217.57)")
    print(f"   total:     {total_lo:.2f} - {total_hi:.2f} CPU-h "
          f"(docket states 159.77-290.10 / rounded 159.8-290.1)")
    print(f"   tasks: {n_tasks_per_venue * 2} (2 venues) "
          f"(docket states 82 tasks)")

    print("\n" + "=" * 78)
    print("10. F4 deadline gate")
    from datetime import date
    today = date(2026, 8, 30)
    expiry = date(2026, 9, 23)
    days_remaining = (expiry - today).days
    print(f"    today (task 'as of' date) = {today}, workspace expiry = {expiry}")
    print(f"    days remaining = {days_remaining} -> "
          f"{'COMFORTABLY CLEAR (>7 days, no arm mid-flight uncovered)' if days_remaining > 7 else 'TIGHT'}")


if __name__ == "__main__":
    main()
