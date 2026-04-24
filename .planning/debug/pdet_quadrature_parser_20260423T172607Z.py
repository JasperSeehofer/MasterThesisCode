"""VERIFY-05 quadrature-weight parser.

Reads STAT-04 WARNING lines from two sources:
  1. Plan 40-03 sweep log: .planning/debug/verify03_sweep_20260423T172607Z.log
     - Contains all 37 non-0.73 h-values (0.6..0.86) delimited by:
       '=== BEGIN h=<value> (<ts>) ===' / '=== END h=<value> ==='
     - WARNING format (stdout from the evaluation run):
       'Event <N>: >5% quadrature weight outside P_det grid — numerator=<num>, denominator=<den>'
  2. Plan 40-02 h=0.73 pre-captured warnings:
     .planning/debug/verify02_quadrature_warnings_20260423T172607Z.log
     - Contains h=0.73 WARNING lines (pre-grepped from verify02_reeval log)

Deviation from plan template (Rule 3 - blocking):
  The original plan assumed STAT-04 WARNINGs would appear in the per-h
  'simulations/master_thesis_code_*_h_0_*.log' files. Inspection shows
  those files do NOT contain the WARNING lines (they use a different formatter
  that captures stdout but not the _LOGGER.warning output in the same way).
  The actual WARNINGs appear in the consolidated sweep log created by the
  40-03 sweep shell script, which captured stderr/stdout and embedded
  '=== BEGIN h=X ===' markers. The parser is adapted accordingly.

Writes: .planning/debug/pdet_quadrature_raw_20260423T172607Z.csv
Columns: h, event_idx, quadrature_weight_outside_grid_numerator,
         quadrature_weight_outside_grid_denominator, source_file
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TS = (ROOT / ".planning" / "debug" / "verify_gate_timestamp.txt").read_text().strip()

# Regex matches STAT-04 WARNING line format from bayesian_statistics.py:1269-1275
# Format: 'Event <N>: >5% quadrature weight outside P_det grid — numerator=<num>, denominator=<den>'
# The em-dash (—) is used in the Python source; it may also appear as '--' in some locales.
WARN_RE = re.compile(
    r"Event (?P<ev>\d+): >5%[^0-9]+"
    r"quadrature weight outside P_det grid"
    r"[^0-9]+"
    r"numerator=(?P<num>[\d.]+)"
    r"[^0-9]+"
    r"denominator=(?P<den>[\d.]+)"
)

# Regex to extract h-value from sweep log section markers
# Format: '=== BEGIN h=0.685 (20260424T094418Z) ==='
BEGIN_RE = re.compile(r"^=== BEGIN h=([\d.]+)")

raw_rows: list[dict[str, object]] = []

# --- Source 1: Plan 40-03 consolidated sweep log ---
# This log has section markers for each h-value. Parse sections sequentially.
sweep_log = ROOT / ".planning" / "debug" / f"verify03_sweep_{TS}.log"
if not sweep_log.exists():
    print(f"ERROR: sweep log not found: {sweep_log}", file=sys.stderr)
    sys.exit(1)

current_h: float | None = None
h_section_count = 0

with sweep_log.open("r", errors="replace") as f:
    for line in f:
        line = line.rstrip("\n")
        # Check for section start
        begin_match = BEGIN_RE.match(line)
        if begin_match:
            current_h = float(begin_match.group(1))
            h_section_count += 1
            continue
        # Check for section end
        if line.startswith("=== END h="):
            current_h = None
            continue
        # Inside a section, look for STAT-04 WARNINGs
        if current_h is not None:
            m = WARN_RE.search(line)
            if m:
                raw_rows.append(
                    {
                        "h": current_h,
                        "event_idx": int(m.group("ev")),
                        "quadrature_weight_outside_grid_numerator": float(m.group("num")),
                        "quadrature_weight_outside_grid_denominator": float(m.group("den")),
                        "source_file": str(sweep_log.relative_to(ROOT)),
                    }
                )

print(
    f"Parsed {len(raw_rows)} WARNING rows across {h_section_count} h-sections from sweep log",
    file=sys.stderr,
)

if h_section_count < 27:
    print(
        f"ERROR: only {h_section_count} h-sections in sweep log; VERIFY-05 needs >= 27",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Source 2: Plan 40-02 pre-captured h=0.73 warnings ---
# This file contains lines like:
# 'Event 2: >5% quadrature weight outside P_det grid — numerator=1.000, denominator=0.397'
pre_captured = ROOT / ".planning" / "debug" / f"verify02_quadrature_warnings_{TS}.log"
h_073_rows_from_pre = 0
if pre_captured.exists():
    with pre_captured.open("r", errors="replace") as f:
        for line in f:
            m = WARN_RE.search(line)
            if not m:
                continue
            raw_rows.append(
                {
                    "h": 0.73,
                    "event_idx": int(m.group("ev")),
                    "quadrature_weight_outside_grid_numerator": float(m.group("num")),
                    "quadrature_weight_outside_grid_denominator": float(m.group("den")),
                    "source_file": str(pre_captured.relative_to(ROOT)),
                }
            )
            h_073_rows_from_pre += 1
    print(
        f"  + {h_073_rows_from_pre} WARNING rows for h=0.73 from verify02 pre-captured file",
        file=sys.stderr,
    )
else:
    print(f"WARN: verify02 quadrature warnings file not found: {pre_captured}", file=sys.stderr)

# --- Write raw CSV ---
out = ROOT / ".planning" / "debug" / f"pdet_quadrature_raw_{TS}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    fieldnames = [
        "h",
        "event_idx",
        "quadrature_weight_outside_grid_numerator",
        "quadrature_weight_outside_grid_denominator",
        "source_file",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(raw_rows)

print(f"Total rows written: {len(raw_rows)}", file=sys.stderr)
print(f"Raw CSV: {out}", file=sys.stderr)

if not raw_rows:
    print(
        "INFO: zero STAT-04 warnings matched — either no event crossed 5% at any h, OR the\n"
        "      WARN_RE regex needs an update. Inspect the sweep log manually.",
        file=sys.stderr,
    )

# Print summary to stdout for the executor
unique_h = sorted({r["h"] for r in raw_rows})
print(f"Parsed {len(raw_rows)} WARNING rows across {len(unique_h)} unique h-values")
print(f"H-values covered: {unique_h}")
print(f"Raw CSV: {out}")
