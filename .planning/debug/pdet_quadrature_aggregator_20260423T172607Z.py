"""VERIFY-05 aggregator: per-h stats, all-h stats, histogram, Phase 41 verdict.

Reads the raw CSV produced by pdet_quadrature_parser_20260423T172607Z.py and produces:
  - Per-h summary table (mean, max, count per h-value)
  - All-h aggregate (across all WARNING rows)
  - Histogram of per-(h,event) numerator weights (10 bins on [0, 1])
  - Phase 41 trigger decision per D-18:
      mean_{h=0.73}(quadrature_weight_outside_grid_numerator) > 0.05 => Phase 41 triggers
    Where the mean uses the conservative lower bound:
      mean_lb = sum(numerator for h=0.73 rows) / N_events_total
    Unlogged events (weight <= 5%) are treated as 0.0.

Note on multiple rows per event:
  The WARNING fires once per host-galaxy call where the d_L window is outside the grid.
  Each event can have many potential host galaxies, so one event generates many rows.
  The D-18 trigger uses mean(per-event max_numerator) / N_events_total as the
  lower-bound mean. Per-event max is the correct unit (one value per unique event).
  Unlogged events contribute 0.0 (lower bound).

Usage: uv run python pdet_quadrature_aggregator_20260423T172607Z.py <N_events_total>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TS = (ROOT / ".planning" / "debug" / "verify_gate_timestamp.txt").read_text().strip()
TRUE_H = 0.73
PHASE_41_THRESHOLD = 0.05
BORDERLINE_LOW = 0.03
BORDERLINE_HIGH = PHASE_41_THRESHOLD  # 0.05

# N_events_total passed as argv[1]
if len(sys.argv) < 2:
    print("Usage: pdet_quadrature_aggregator_<ts>.py <N_events_total>", file=sys.stderr)
    sys.exit(1)
N_EVENTS_TOTAL = int(sys.argv[1])

raw_csv = ROOT / ".planning" / "debug" / f"pdet_quadrature_raw_{TS}.csv"
if not raw_csv.exists() or raw_csv.stat().st_size == 0:
    df = pd.DataFrame(
        columns=[
            "h",
            "event_idx",
            "quadrature_weight_outside_grid_numerator",
            "quadrature_weight_outside_grid_denominator",
            "source_file",
        ]
    )
else:
    df = pd.read_csv(raw_csv)

print(f"Loaded {len(df)} rows from {raw_csv}", file=sys.stderr)

# Per D-18: the Phase 41 trigger is on the NUMERATOR mean at h=0.73.
# Unlogged events have weight <= 0.05; we treat unlogged as 0.0 (conservative lower bound).
#
# IMPORTANT: The WARNING fires once per HOST-GALAXY CALL where the d_L window is outside
# the P_det grid, not once per event. A single event may have hundreds of potential host
# galaxies, each generating a separate WARNING row. Therefore the raw CSV has many rows
# per event.
#
# The correct per-D-18 aggregation is per-EVENT, not per-row:
#   - For each unique event at h=0.73, take max(numerator) across all host-galaxy calls.
#     This represents the worst-case off-grid fraction for that event.
#   - mean_lb = sum(max_numerator per unique event) / N_events_total
#   - Unlogged events (all host calls <= 5%) contribute 0.0 to the sum (lower bound).
#
# If we summed raw rows / N_events_total the result would be >> 1 (inflated by multi-host
# counting) and meaningless. The per-event max is the appropriate unit.


def _per_event_stats(sub: pd.DataFrame, n_total: int) -> dict:
    """Compute per-event statistics from a per-host-galaxy-call DataFrame."""
    if sub.empty:
        return {
            "n_rows_logged": 0,
            "n_unique_events_logged": 0,
            "n_unique_events_above_5pct_numerator": 0,
            "n_unique_events_above_5pct_denominator": 0,
            "logged_mean_numerator": float("nan"),
            "logged_max_numerator": float("nan"),
            "logged_mean_denominator": float("nan"),
            "logged_max_denominator": float("nan"),
            "per_event_max_mean_numerator_lb": 0.0,
            "per_event_max_mean_denominator_lb": 0.0,
        }
    ev_num = sub.groupby("event_idx")["quadrature_weight_outside_grid_numerator"].max()
    ev_den = sub.groupby("event_idx")["quadrature_weight_outside_grid_denominator"].max()
    n_above_num = int((ev_num > 0.05).sum())
    n_above_den = int((ev_den > 0.05).sum())
    return {
        "n_rows_logged": len(sub),
        "n_unique_events_logged": int(sub["event_idx"].nunique()),
        "n_unique_events_above_5pct_numerator": n_above_num,
        "n_unique_events_above_5pct_denominator": n_above_den,
        "logged_mean_numerator": float(sub["quadrature_weight_outside_grid_numerator"].mean()),
        "logged_max_numerator": float(sub["quadrature_weight_outside_grid_numerator"].max()),
        "logged_mean_denominator": float(sub["quadrature_weight_outside_grid_denominator"].mean()),
        "logged_max_denominator": float(sub["quadrature_weight_outside_grid_denominator"].max()),
        # D-18 decision variable: mean of per-event max / N_events_total (lower bound, unlogged=0)
        "per_event_max_mean_numerator_lb": float(ev_num.sum() / n_total),
        "per_event_max_mean_denominator_lb": float(ev_den.sum() / n_total),
    }


# --- Per-h stats ---
per_h_rows = []
unique_h = sorted(df["h"].unique()) if not df.empty else []
for h in unique_h:
    sub = df[df["h"] == h]
    stats = _per_event_stats(sub, N_EVENTS_TOTAL)
    per_h_rows.append({"h": float(h), "n_events_total": N_EVENTS_TOTAL, **stats})

# --- All-h aggregate (logged rows only) ---
if not df.empty:
    all_mean_num = float(df["quadrature_weight_outside_grid_numerator"].mean())
    all_max_num = float(df["quadrature_weight_outside_grid_numerator"].max())
    all_mean_den = float(df["quadrature_weight_outside_grid_denominator"].mean())
    all_max_den = float(df["quadrature_weight_outside_grid_denominator"].max())
else:
    all_mean_num = all_max_num = all_mean_den = all_max_den = 0.0

# --- Per-event histogram (aggregated across all h, numerator, logged rows only) ---
hist_bins = np.linspace(0.0, 1.0, 11)  # 10 bins on [0, 1] per D-17
if not df.empty:
    hist_counts_num, _ = np.histogram(
        df["quadrature_weight_outside_grid_numerator"], bins=hist_bins
    )
    hist_counts_den, _ = np.histogram(
        df["quadrature_weight_outside_grid_denominator"], bins=hist_bins
    )
else:
    hist_counts_num = np.zeros(10, dtype=int)
    hist_counts_den = np.zeros(10, dtype=int)

# --- Phase 41 trigger decision (D-18, W1 3-way verdict) ---
h_073_row = next((r for r in per_h_rows if abs(r["h"] - TRUE_H) < 1e-6), None)
if h_073_row is None:
    mean_073_num = 0.0
    trigger_verdict = "NOT-TRIGGERED"
    trigger_rationale = "No h=0.73 data in CSV (no event >5% at h=0.73) — Phase 41 NOT triggered"
else:
    mean_073_num = h_073_row["per_event_max_mean_numerator_lb"]
    if mean_073_num > BORDERLINE_HIGH:
        trigger_verdict = "PHASE-41-TRIGGERED"
        trigger_rationale = (
            f"mean_{{h=0.73}}(numerator) lower bound = {mean_073_num:.4f} > {BORDERLINE_HIGH:.2f}"
        )
    elif mean_073_num >= BORDERLINE_LOW:
        trigger_verdict = "PHASE-41-TRIGGER-BORDERLINE"
        trigger_rationale = (
            f"mean_{{h=0.73}}(numerator) lower bound = {mean_073_num:.4f} is inside "
            f"borderline band [{BORDERLINE_LOW:.2f}, {BORDERLINE_HIGH:.2f}] — user decides"
        )
    else:
        trigger_verdict = "NOT-TRIGGERED"
        trigger_rationale = (
            f"mean_{{h=0.73}}(numerator) lower bound = {mean_073_num:.4f} < {BORDERLINE_LOW:.2f}"
        )

# Back-compat boolean: True iff strictly triggered.
trigger_bool = trigger_verdict == "PHASE-41-TRIGGERED"

# --- Dominant events at h=0.73 (top 10 by max numerator across all host calls) ---
dom_events = []
if not df.empty:
    h73 = df[abs(df["h"] - TRUE_H) < 1e-6]
    if not h73.empty:
        # Aggregate per event: take max numerator across all host-galaxy calls
        h73_agg = (
            h73.groupby("event_idx")
            .agg(
                max_numerator=("quadrature_weight_outside_grid_numerator", "max"),
                max_denominator=("quadrature_weight_outside_grid_denominator", "max"),
                n_host_calls=("quadrature_weight_outside_grid_numerator", "count"),
            )
            .reset_index()
            .sort_values("max_numerator", ascending=False)
        )
        for _, row in h73_agg.head(10).iterrows():
            dom_events.append(
                {
                    "event_idx": int(row["event_idx"]),
                    "max_numerator": float(row["max_numerator"]),
                    "max_denominator": float(row["max_denominator"]),
                    "n_host_calls": int(row["n_host_calls"]),
                }
            )

# --- Write Markdown ---
DEBUG = ROOT / ".planning" / "debug"
md = DEBUG / f"pdet_quadrature_summary_{TS}.md"

lines = [
    "# VERIFY-05: P_det Quadrature-Weight-Outside-Grid Summary",
    "",
    f"**Timestamp:** {TS}",
    "**Phase:** 40 Wave 3",
    "**Requirement:** VERIFY-05",
    "",
    "**Phase 41 trigger rule (D-18, W1 3-way verdict):**",
    "  `mean_{h=0.73}(numerator)` lower bound (unlogged=0) vs borderline band:",
    f"  - `< {BORDERLINE_LOW:.2f}` → NOT-TRIGGERED",
    f"  - `[{BORDERLINE_LOW:.2f}, {BORDERLINE_HIGH:.2f}]` → PHASE-41-TRIGGER-BORDERLINE (user decides)",
    f"  - `> {BORDERLINE_HIGH:.2f}` → PHASE-41-TRIGGERED",
    "",
    "## Phase 41 Trigger Verdict",
    "",
    f"- `mean_{{h=0.73}}` all-events lower bound (unlogged=0) = **{mean_073_num:.4f}**",
    f"- Borderline band: [{BORDERLINE_LOW:.2f}, {BORDERLINE_HIGH:.2f}]",
    f"- **Verdict: {trigger_verdict}**",
    f"- Rationale: {trigger_rationale}",
    "",
    "## Interpretation Note",
    "",
    "STAT-04 logs a WARNING once per *host-galaxy call* where the d_L integration window",
    "extends outside the P_det injection grid, only when numerator > 0.05 OR denominator > 0.05.",
    "Events below that threshold are NOT captured. This means:",
    "",
    "- **`n_rows_logged`**: total WARNING rows (one per host-galaxy call above threshold),",
    "  NOT one per event. Events with many potential hosts generate many rows.",
    "- **`logged_mean_numerator`**: mean over WARNING rows — biased HIGH (all rows are >5%).",
    "- **`per_event_max_mean_numerator_lb`**: mean of per-event max(numerator) / N_events_total.",
    "  Per-event max collapses all host-galaxy calls for one event into a single value.",
    "  Unlogged events (all host calls <= 5%) contribute 0.0 — a conservative LOWER BOUND.",
    "  **This is the D-18 decision variable.**",
    "",
    "## Per-h Summary",
    "",
    "| h | n_unique_events_num | n_rows_logged | logged_mean_num | logged_max_num | per_ev_max_mean_num_lb | per_ev_max_mean_den_lb |",
    "|---|---------------------|---------------|-----------------|----------------|------------------------|------------------------|",
]
for r in per_h_rows:
    lines.append(
        f"| {r['h']:.4f} "
        f"| {r['n_unique_events_above_5pct_numerator']} "
        f"| {r['n_rows_logged']} "
        f"| {r['logged_mean_numerator']:.4f} "
        f"| {r['logged_max_numerator']:.4f} "
        f"| {r['per_event_max_mean_numerator_lb']:.4f} "
        f"| {r['per_event_max_mean_denominator_lb']:.4f} |"
    )

lines += [
    "",
    "## All-h Aggregate (Across All WARNING Rows, All h-Values)",
    "",
    f"- N_rows_logged = {len(df)}",
    f"- Mean numerator across all rows = {all_mean_num:.4f}",
    f"- Max numerator across all rows  = {all_max_num:.4f}",
    f"- Mean denominator across all rows = {all_mean_den:.4f}",
    f"- Max denominator across all rows  = {all_max_den:.4f}",
    f"- N_events_total (from CRB CSV) = {N_EVENTS_TOTAL}",
    "",
    "## Histogram of Numerator Weights (10 bins on [0, 1], All WARNING Rows, All h)",
    "",
    "| Bin range          | Count |",
    "|--------------------|-------|",
]
for i, c in enumerate(hist_counts_num):
    lo = hist_bins[i]
    hi = hist_bins[i + 1]
    lines.append(f"| [{lo:.2f}, {hi:.2f}) | {int(c)} |")

lines += [
    "",
    "## Top-10 Dominant Events at h=0.73 (by Max Numerator Across Host Calls)",
    "",
    "| event_idx | max_numerator | max_denominator | n_host_calls |",
    "|-----------|---------------|-----------------|--------------|",
]
if dom_events:
    for r in dom_events:
        lines.append(
            f"| {r['event_idx']} "
            f"| {r['max_numerator']:.4f} "
            f"| {r['max_denominator']:.4f} "
            f"| {r['n_host_calls']} |"
        )
    # Explicit event 2 callout (40-CONTEXT.md §specifics)
    ev2 = next((r for r in dom_events if r["event_idx"] == 2), None)
    if ev2:
        lines += [
            "",
            f"**Event 2 spotlight:** max_numerator={ev2['max_numerator']:.4f}, "
            f"max_denominator={ev2['max_denominator']:.4f}, "
            f"n_host_calls={ev2['n_host_calls']}.",
            "Event 2 was previously flagged in STATE.md Phase 38 as having 100% off-grid",
            "numerator weight (injection grid coverage gap). Every potential host galaxy for",
            "this event has its d_L integration window entirely outside the P_det grid.",
            "If Phase 41 triggers, the densified injection grid should prioritize the",
            "M x z x d_L region around this event's parameters.",
        ]
else:
    lines.append("| (no h=0.73 events above 5% threshold) | — | — | — |")

lines += [
    "",
    "## Artifacts",
    "",
    f"- Raw CSV: `.planning/debug/pdet_quadrature_raw_{TS}.csv`",
    f"- Parser driver: `.planning/debug/pdet_quadrature_parser_{TS}.py`",
    f"- Aggregator driver: `.planning/debug/pdet_quadrature_aggregator_{TS}.py`",
    f"- Machine-readable: `.planning/debug/pdet_quadrature_summary_{TS}.json`",
    "",
]

md.write_text("\n".join(lines))

# --- Write JSON ---
json_path = DEBUG / f"pdet_quadrature_summary_{TS}.json"
json_path.write_text(
    json.dumps(
        {
            "ts": TS,
            "phase_41_threshold": PHASE_41_THRESHOLD,
            "phase_41_borderline_low": BORDERLINE_LOW,
            "phase_41_borderline_high": BORDERLINE_HIGH,
            "phase_41_trigger": trigger_verdict,
            "phase_41_trigger_bool": bool(trigger_bool),
            "phase_41_trigger_rationale": trigger_rationale,
            "mean_h_073_numerator_lb": mean_073_num,
            "per_h": per_h_rows,
            "all_h_aggregate_logged_only": {
                "n_rows": len(df),
                "mean_numerator": all_mean_num,
                "max_numerator": all_max_num,
                "mean_denominator": all_mean_den,
                "max_denominator": all_max_den,
            },
            "histogram_numerator": {
                "bins": [float(x) for x in hist_bins],
                "counts": [int(c) for c in hist_counts_num],
            },
            "histogram_denominator": {
                "bins": [float(x) for x in hist_bins],
                "counts": [int(c) for c in hist_counts_den],
            },
            "dominant_events_h_073": dom_events,
            "n_events_total": N_EVENTS_TOTAL,
        },
        indent=2,
    )
)

print(f"VERIFY-05 verdict: {trigger_verdict}")
print(f"  mean_h_073_numerator_lb = {mean_073_num:.6f}")
print(f"  trigger_rationale: {trigger_rationale}")
print(f"  report: {md}")
print(f"  json: {json_path}")
