"""VERIFY-03 per-h summary CSV builder.

Invoked once (no shell loop). Takes the posteriors root dir as sys.argv[1], globs
`h_*.json`, and writes a CSV with columns: h, file_mtime_utc, log_posterior_max, n_detections.

The posterior JSON format stores per-event likelihoods as:
  { "h": <float>, "<event_id>": [<likelihood_at_h>], ... }

log_posterior is computed as sum(log(lk)) over all events (matching load_posteriors logic).
n_detections is the count of event keys (all keys except "h").
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path


def h_from_filename(name: str) -> float | None:
    # h_0_73.json -> 0.73 ; h_0_735.json -> 0.735
    stem = Path(name).stem  # h_0_73
    if not stem.startswith("h_"):
        return None
    suffix = stem[len("h_") :]  # 0_73
    # Replace only the FIRST underscore with a decimal point
    if "_" not in suffix:
        return None
    head, tail = suffix.split("_", 1)
    try:
        return float(f"{head}.{tail}")
    except ValueError:
        return None


def compute_log_posterior(d: dict) -> float | str:
    """Compute log_posterior by summing log(lk) over all events.

    Each event entry is a list of likelihood values [lk_at_h].
    Zero or negative likelihoods are skipped (matching load_posteriors behavior).
    Returns "NA" if no events.
    """
    log_post = 0.0
    n_events = 0
    for key, val in d.items():
        if key == "h":
            continue
        # val is a list of likelihood values at this h
        if isinstance(val, list) and val:
            lk = float(val[0])
            if lk > 0:
                log_post += math.log(lk)
            # Zero or negative: skip (matching evaluation_report.py:164)
        elif isinstance(val, int | float):
            # Fallback: scalar likelihood
            lk = float(val)
            if lk > 0:
                log_post += math.log(lk)
        n_events += 1
    if n_events == 0:
        return "NA"
    return log_post


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: verify03_build_summary.py <posteriors_root> <csv_out>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    csv_out = Path(sys.argv[2])
    files = sorted(root.glob("h_*.json"))
    if not files:
        print(f"ERROR: no h_*.json files under {root}", file=sys.stderr)
        return 1

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["h", "file_mtime_utc", "log_posterior_max", "n_detections"])
        for fp in files:
            h = h_from_filename(fp.name)
            if h is None:
                continue
            mtime = datetime.fromtimestamp(fp.stat().st_mtime, tz=UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            try:
                d = json.loads(fp.read_text())
            except json.JSONDecodeError as e:
                print(f"WARN: cannot parse {fp}: {e}", file=sys.stderr)
                continue
            log_post = compute_log_posterior(d)
            n_det = len([k for k in d if k != "h"])
            w.writerow([f"{h}", mtime, log_post, n_det])
    print(f"Summary CSV written: {csv_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
