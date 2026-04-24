"""VERIFY-04 anisotropy audit: per-quartile MAP comparison.

Reads the prepared CRB CSV (for qS values via row-index = simulation step index),
the per-h posterior JSON files (for combined likelihoods, matching extract_baseline),
and the overall h-sweep posterior (for MAP_total and sigma).

Computes per-quartile MAP_q and flags any quartile with
|MAP_q - MAP_total| > sigma as a Stage-2 trigger (D-12 -- NOT an abort).

Likelihood source: uses simulations/posteriors/h_*.json files directly
(same source as extract_baseline) rather than the diagnostic CSV, because:
  - The diagnostic CSV has duplicate rows (multiple evaluation runs appended)
  - 23 events have combined_no_bh = 0 in the latest (Phase 40 sweep) CSV rows,
    making the group log-posterior -inf for every h-value
  - extract_baseline already skips zero-likelihood events (lk > 0 guard at line 162)
  - Using the same source ensures MAP_q is computed consistently with MAP_total

Note on 40-03 verdict: VERIFY-03 returned FAIL (SC-3 MAP=0.860) due to
extract_baseline lacking the D(h) denominator correction. VERIFY-04 is
independent of that finding -- we compute anisotropy relative to the same
biased MAP_total (0.86) so the comparison is internally self-consistent.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from master_thesis_code.bayesian_inference.evaluation_report import extract_baseline  # noqa: E402

TS = (ROOT / ".planning" / "debug" / "verify_gate_timestamp.txt").read_text().strip()
TRUE_H = 0.73

# --- Inputs (CRB path from sys.argv[1]) ---
if len(sys.argv) < 2:
    sys.exit("usage: anisotropy_driver.py <CRB_CSV>")
CRB_CSV = Path(sys.argv[1])
POSTERIORS_DIR = ROOT / "simulations" / "posteriors"
DIAG_CSV = ROOT / "simulations" / "diagnostics" / "event_likelihoods.csv"

# --- Step A: Overall posterior gives MAP_total and sigma (D-13.3) ---
print(f"Loading overall posterior from {POSTERIORS_DIR} ...")
snap = extract_baseline(posteriors_dir=POSTERIORS_DIR, crb_csv_path=None, true_h=TRUE_H)
MAP_total = snap.map_h
sigma = (snap.ci_upper - snap.ci_lower) / 2.0  # D-13.3: half of 68% CI width
print(f"MAP_total = {MAP_total:.4f}")
print(f"CI = [{snap.ci_lower:.4f}, {snap.ci_upper:.4f}]")
print(f"sigma = {sigma:.6f}")
print(f"n_events (in extract_baseline) = {snap.n_events}")

# --- Step B: Load all per-h posterior JSON files (same source as extract_baseline) ---
# Each file: {"h": float, "<event_idx_str>": [likelihood_value], ...}
# extract_baseline skips events where lk <= 0; we do the same.
print(f"\nLoading per-h posterior files from {POSTERIORS_DIR} ...")
h_files = sorted(POSTERIORS_DIR.glob("h_*.json"))
if not h_files:
    sys.exit(f"ERROR: No h_*.json files found in {POSTERIORS_DIR}")

# Build: per_event_per_h[event_key][h] = log(lk), or -inf if lk <= 0
h_grid_set: set[float] = set()
per_event_per_h: dict[str, dict[float, float]] = {}

for f in h_files:
    d: dict[str, object] = json.loads(f.read_text())
    h = float(d["h"])  # type: ignore[arg-type]
    h_grid_set.add(h)
    for k, v in d.items():
        if k == "h":
            continue
        lk = float(v[0]) if isinstance(v, list) and len(v) > 0 else 0.0  # type: ignore[index]
        log_lk = math.log(lk) if lk > 0 else -math.inf
        if k not in per_event_per_h:
            per_event_per_h[k] = {}
        per_event_per_h[k][h] = log_lk

h_grid = sorted(h_grid_set)
all_event_keys = sorted(per_event_per_h.keys(), key=int)  # event keys as strings
print(f"h_grid: {len(h_grid)} values from {min(h_grid):.3f} to {max(h_grid):.3f}")
print(f"Events found across all posteriors: {len(all_event_keys)}")

# Convert event keys (strings) to integers for join with CRB row indices
event_int_ids = sorted([int(k) for k in all_event_keys])

# Validate diagnostic CSV row count (warn-only -- we use JSON files, not CSV for likelihoods)
if DIAG_CSV.exists():
    n_diag_rows = sum(1 for _ in DIAG_CSV.open())
    print(f"Diagnostic CSV rows (info only): {n_diag_rows - 1} data rows")
    if n_diag_rows < 101:
        print(
            "WARNING: diagnostic CSV has fewer than 100 data rows -- "
            "Phase 40-03 sweep may not have completed."
        )

# --- Step C: Per-event qS from prepared CRB CSV (row-index = simulation step index) ---
print(f"\nLoading CRB CSV from {CRB_CSV} ...")
crb = pd.read_csv(CRB_CSV)
print(f"CRB rows: {len(crb)}")

col_qs = next((c for c in crb.columns if c.lower() == "qs"), None)
if col_qs is None:
    sys.exit(f"ERROR: CRB CSV missing qS column (found: {list(crb.columns)})")

# Validate event_int_ids against CRB size
max_event_id = max(event_int_ids)
if max_event_id >= len(crb):
    sys.exit(
        f"ERROR: max event_idx={max_event_id} >= CRB rows={len(crb)}. "
        "Use prepared_cramer_rao_bounds.csv (542 rows), not cramer_rao_bounds.csv (42 rows)."
    )

# Extract qS values for each detected event using row-index lookup
qS_values = crb[col_qs].iloc[event_int_ids].values.astype(float)
print(f"qS range: [{qS_values.min():.3f}, {qS_values.max():.3f}]")

# --- Step D: Equal-count quartile edges on |qS - pi/2| (D-14) ---
dist_from_equator = np.abs(qS_values - math.pi / 2.0)
q_edges = np.quantile(dist_from_equator, [0.00, 0.25, 0.50, 0.75, 1.00])
print(f"\nQuartile edges (on |qS - pi/2|): {[f'{e:.4f}' for e in q_edges]}")

quartile_labels = [
    "Q1 (nearest ecliptic equator)",
    "Q2",
    "Q3",
    "Q4 (furthest from equator)",
]

# Assign each event to a quartile via searchsorted on the inner 3 edges
q_assignments = np.searchsorted(q_edges[1:-1], dist_from_equator, side="right")
q_assignments = np.minimum(q_assignments, 3)

quartile_event_str_keys: list[list[str]] = [[] for _ in range(4)]
for ev_int_id, q in zip(event_int_ids, q_assignments):
    quartile_event_str_keys[q].append(str(ev_int_id))

for i, members in enumerate(quartile_event_str_keys):
    print(f"  Q{i + 1}: {len(members)} events")

# --- Step E: Per-quartile MAP via log-posterior re-combination ---
# For each quartile, sum log_lk over member events per h (skipping -inf, same as extract_baseline).
# Then argmax over h => MAP_q.
print()
per_quartile: list[dict[str, object]] = []

for q_idx, (label, members) in enumerate(zip(quartile_labels, quartile_event_str_keys)):
    if len(members) == 0:
        per_quartile.append(
            {
                "quartile": q_idx + 1,
                "label": label,
                "edge_lower": float(q_edges[q_idx]),
                "edge_upper": float(q_edges[q_idx + 1]),
                "n_events": 0,
                "event_idx_members": [],
                "n_events_with_finite_lk": 0,
                "MAP_q": None,
                "delta_MAP_abs": None,
                "sigma": float(sigma),
                "stage_2_trigger": False,
                "note": "empty quartile",
            }
        )
        continue

    # Build log-posterior array over h_grid for this quartile
    lp_array = np.zeros(len(h_grid), dtype=np.float64)
    n_finite_per_h = np.zeros(len(h_grid), dtype=int)

    for h_idx, h in enumerate(h_grid):
        for ev_k in members:
            if ev_k in per_event_per_h:
                lk = per_event_per_h[ev_k].get(h, -math.inf)
                if not math.isinf(lk):
                    lp_array[h_idx] += lk
                    n_finite_per_h[h_idx] += 1
            # Events missing from posteriors entirely: skip (no contribution)

    # The argmax is well-defined as long as at least one h has finite log-posterior
    finite_mask = np.isfinite(lp_array)
    n_finite_events = int(np.max(n_finite_per_h))

    if not finite_mask.any():
        MAP_q = float("nan")
        delta = float("nan")
        triggered = False
        note = "all log-posteriors are -inf (no events with non-zero likelihood)"
    else:
        best_idx = int(np.argmax(lp_array))
        MAP_q = float(h_grid[best_idx])
        delta = abs(MAP_q - MAP_total)
        triggered = bool(delta > sigma)
        note = ""

    print(
        f"  Q{q_idx + 1} ({len(members)} events, {n_finite_events} with finite lk): "
        f"MAP_q={MAP_q:.4f}  |DELTA|={delta:.4f}  sigma={sigma:.4f}  "
        f"trigger={'YES' if triggered else 'no'}"
    )

    per_quartile.append(
        {
            "quartile": q_idx + 1,
            "label": label,
            "edge_lower": float(q_edges[q_idx]),
            "edge_upper": float(q_edges[q_idx + 1]),
            "n_events": len(members),
            "event_idx_members": [int(k) for k in members],
            "n_events_with_finite_lk": n_finite_events,
            "MAP_q": MAP_q if not math.isnan(MAP_q) else None,
            "delta_MAP_abs": delta if not math.isnan(delta) else None,
            "sigma": float(sigma),
            "stage_2_trigger": triggered,
            "note": note,
        }
    )

# --- Step F: Write artifacts ---
DEBUG_DIR = ROOT / ".planning" / "debug"
report_md = DEBUG_DIR / f"anisotropy_audit_{TS}.md"
report_json = DEBUG_DIR / f"anisotropy_audit_{TS}.json"

any_trigger = any(bool(q["stage_2_trigger"]) for q in per_quartile)
verdict = "STAGE-2-TRIGGER" if any_trigger else "PASS"

lines = [
    "# VERIFY-04: Anisotropy Audit",
    "",
    f"**Timestamp:** {TS}",
    "**Phase:** 40 Wave 3",
    "**Requirement:** VERIFY-04",
    "**Rule (D-12):** `>1σ shift is a Stage-2 trigger for Phase 42 (not a blocker)`",
    "",
    "## Overall h=0.73 posterior",
    "",
    f"- `MAP_total` = {MAP_total:.4f}",
    f"- 68% CI: [{snap.ci_lower:.4f}, {snap.ci_upper:.4f}]",
    f"- σ = (CI_upper − CI_lower) / 2 = {sigma:.6f}",
    f"- N events in posteriors = {snap.n_events} (out of {len(all_event_keys)} with non-zero lk)",
    "",
    "**Note on MAP_total = 0.86:** `extract_baseline` sums log-likelihoods without the D(h)",
    "denominator correction (the SC-3 FAIL finding from VERIFY-03). The quartile comparison",
    "is internally self-consistent: all quartile MAPs and the threshold σ are derived from the",
    "same biased posterior. The anisotropy audit result is therefore valid regardless of the",
    "absolute MAP calibration issue.",
    "",
    "**Likelihood source:** per-h JSON posterior files (not the diagnostic CSV), matching",
    "`extract_baseline` behavior. Events with zero likelihood at a given h are skipped (same",
    "as `extract_baseline` line 162), ensuring MAP_q is computed identically to MAP_total.",
    "",
    "## Quartile edges (on |qS − π/2|, equal-count per D-14)",
    "",
    f"- Q1: [{q_edges[0]:.4f}, {q_edges[1]:.4f})  [events nearest ecliptic equator]",
    f"- Q2: [{q_edges[1]:.4f}, {q_edges[2]:.4f})",
    f"- Q3: [{q_edges[2]:.4f}, {q_edges[3]:.4f})",
    f"- Q4: [{q_edges[3]:.4f}, {q_edges[4]:.4f}]  [events furthest from ecliptic equator]",
    "",
    "## Per-quartile MAP_q",
    "",
    "| # | Quartile label | N events | N finite-lk | MAP_q | |MAP_q − MAP_total| | σ | Trigger (ΔMAP > σ)? |",
    "|---|----------------|----------|-------------|-------|---------------------|---|----------------------|",
]

for q in per_quartile:
    map_q_str = "-" if q["MAP_q"] is None else f"{q['MAP_q']:.4f}"
    delta_str = "-" if q["delta_MAP_abs"] is None else f"{q['delta_MAP_abs']:.4f}"
    trig_str = "YES" if q["stage_2_trigger"] else "no"
    n_finite = q.get("n_events_with_finite_lk", "-")
    lines.append(
        f"| {q['quartile']} | {q['label']} | {q['n_events']} "
        f"| {n_finite} | {map_q_str} | {delta_str} | {sigma:.4f} | {trig_str} |"
    )

lines += [
    "",
    "## Verdict",
    "",
    f"**VERIFY-04: {verdict}**",
    "",
]

if any_trigger:
    lines += [
        "Per D-12, this is a Stage-2 trigger for **Phase 42** (Sky-Dependent Injection Campaign),",
        "NOT an abort condition. Phase 40 continues; Phase 41 can still run if its own VERIFY-05",
        "trigger also fires.",
        "",
    ]
else:
    lines += [
        "No quartile shows a >1σ MAP shift. Phase 42 is NOT triggered by anisotropy.",
        "If Phase 41 is also not triggered (VERIFY-05 check), both CAMP-* phases may be",
        "marked `skipped (not triggered)` per ROADMAP.md.",
        "",
    ]

lines += [
    "## Links",
    "",
    f"- CRB CSV source: `{CRB_CSV}`",
    f"- Per-h posterior JSON files: `{POSTERIORS_DIR}/h_*.json`",
    f"- Diagnostic CSV (info only): `{DIAG_CSV}`",
    f"- Driver: `.planning/debug/anisotropy_driver_{TS}.py`",
    f"- Machine-readable: `.planning/debug/anisotropy_audit_{TS}.json`",
    "",
]

report_md.write_text("\n".join(lines))
print(f"\nWrote: {report_md}")

report_json.write_text(
    json.dumps(
        {
            "ts": TS,
            "true_h": TRUE_H,
            "MAP_total": MAP_total,
            "ci_lower": snap.ci_lower,
            "ci_upper": snap.ci_upper,
            "sigma": float(sigma),
            "n_events": snap.n_events,
            "n_events_total_in_posteriors": len(all_event_keys),
            "quartile_edges": [float(e) for e in q_edges],
            "per_quartile": per_quartile,
            "any_stage_2_trigger": any_trigger,
            "verdict": verdict,
            "note_sc3": (
                "MAP_total=0.86 reflects extract_baseline bias (SC-3 FAIL from VERIFY-03). "
                "Quartile comparison is internally consistent: all MAPs and sigma derived "
                "from the same biased posterior."
            ),
            "likelihood_source": "per-h JSON posterior files (not diagnostic CSV)",
        },
        indent=2,
    )
)
print(f"Wrote: {report_json}")

print(f"\nVERIFY-04 verdict: {verdict}")
print(f"  sigma={sigma:.6f}; deltas = {[q['delta_MAP_abs'] for q in per_quartile]}")
