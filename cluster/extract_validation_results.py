#!/usr/bin/env python3
"""Extract and compare pre-fix vs post-fix posterior values.

Run from the project root after `quick_validation.sh` completes:
    python3 cluster/extract_validation_results.py

Reads posterior JSON files from simulations/posteriors/ and
simulations/posteriors_with_bh_mass/ and prints a comparison table.
"""

import json
import sys
from pathlib import Path


def load_posterior_sum(path: Path) -> tuple[float, int]:
    """Load a posterior JSON and return (sum of per-detection values, count)."""
    with open(path) as f:
        data = json.load(f)

    vals: list[float] = []
    for key, value in data.items():
        if key in ("h", "additional_galaxies_without_bh_mass", "galaxy_likelihoods"):
            continue
        if isinstance(value, list) and len(value) > 0:
            vals.append(float(value[0]))
        elif isinstance(value, int | float):
            vals.append(float(value))

    return sum(vals), len(vals)


def main() -> None:
    posteriors_dir = Path("simulations/posteriors")
    with_bh_dir = Path("simulations/posteriors_with_bh_mass")

    if not posteriors_dir.exists():
        print(f"ERROR: {posteriors_dir} not found. Run quick_validation.sh first.", file=sys.stderr)
        sys.exit(1)

    h_values = [0.652, 0.678, 0.704, 0.730]

    print("=" * 72)
    print("Quick Validation Results: /(1+z) Fix")
    print("=" * 72)
    print()

    # Without BH mass channel
    print("WITHOUT BH mass (should be UNCHANGED from pre-fix):")
    print(f"{'h':>8s} | {'sum(posteriors)':>16s} | {'n_detections':>12s}")
    print("-" * 45)
    for h in h_values:
        h_str = str(round(h, 3)).replace(".", "_")
        path = posteriors_dir / f"h_{h_str}.json"
        if path.exists():
            total, n = load_posterior_sum(path)
            print(f"{h:8.3f} | {total:16.4f} | {n:12d}")
        else:
            print(f"{h:8.3f} | {'MISSING':>16s} |")

    print()

    # With BH mass channel
    print("WITH BH mass (expect peak to shift from h=0.600 toward h=0.678):")
    print(f"{'h':>8s} | {'sum(posteriors)':>16s} | {'n_detections':>12s}")
    print("-" * 45)
    for h in h_values:
        h_str = str(round(h, 3)).replace(".", "_")
        path = with_bh_dir / f"h_{h_str}.json"
        if path.exists():
            total, n = load_posterior_sum(path)
            print(f"{h:8.3f} | {total:16.6f} | {n:12d}")
        else:
            print(f"{h:8.3f} | {'MISSING':>16s} |")

    print()
    print("PRE-FIX BASELINE (run_v12_validation, 22 detections):")
    print("  Without BH: 0.652->5038.79, 0.678->5052.72, 0.704->5061.78, 0.730->5065.17")
    print("  With BH:    0.652->0.022565, 0.678->0.022246, 0.704->0.021870, 0.730->0.021453")
    print("  With BH was monotonically DECREASING (peak at or below h=0.600)")
    print()
    print("EXPECTED POST-FIX:")
    print("  With BH mass: peak should shift toward h=0.678 (no longer decreasing)")
    print("  Without BH mass: UNCHANGED from pre-fix values")


if __name__ == "__main__":
    main()
