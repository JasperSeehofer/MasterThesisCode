"""Phase 45 Plan 45-02: probe `detection_probability_without_bh_mass_interpolated_zero_fill`
at a fixed (d_L, h) grid and dump to JSON.

Used twice:

  1. Before the Plan 45-02 patch (Task 1): records the *pre-fix* values for
     `before/after` delta comparison.  Output: outputs/phase45/pre_fix_interp_probe.json.
  2. After the Plan 45-02 patch (Task 4): records the *post-fix* values and
     fills the `delta = post_fix - pre_fix` field.  Output:
     outputs/phase45/post_fix_interp_probe.json.

Cross-checks against `pdet_asymptote.json` (Plan T10): at h=0.73, d_L=0.10 the
pre-fix value must equal 0.5444; at d_L=0.001 it must equal 0.7476.

Usage::

    uv run python -m scripts.bias_investigation.probe_interp_values \
        --output-name pre_fix_interp_probe.json
    # ... apply Plan 45-02 patch ...
    uv run python -m scripts.bias_investigation.probe_interp_values \
        --output-name post_fix_interp_probe.json \
        --pre-fix-source pre_fix_interp_probe.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from master_thesis_code.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)

# Plan 45-02 + 45-04 d_L probe grid:
# - Plan 45-02 (8 points): 0.0, 0.001, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30
# - Plan 45-04 adds (3 points): 0.005, 0.025, 0.075 — the midpoints of the
#   [0, 0.05] and [0.05, 0.10] hybrid segments where the linear-interp
#   identity is verified. Total: 11 d_L points × 5 h values = 55 rows.
DL_VALUES_GPC = [0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
H_VALUES = [0.65, 0.70, 0.73, 0.80, 0.85]

INJECTION_DATA_DIR = "simulations/injections"
SNR_THRESHOLD = 20.0
OUTPUT_DIR = Path("scripts/bias_investigation/outputs/phase45")

# Cross-check anchors from Plan 45-01 / T10 (pdet_asymptote.json)
EXPECTED_H073_DL010_PRE_FIX = 0.5444372481704604
EXPECTED_H073_DL001_PRE_FIX = 0.7476494104015313


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-name",
        required=True,
        help="JSON filename (relative to scripts/bias_investigation/outputs/phase45/).",
    )
    parser.add_argument(
        "--pre-fix-source",
        default=None,
        help="If set, load pre-fix values from this JSON and add a `delta` "
        "and `pre_fix_value` field per row (post-fix run).",
    )
    parser.add_argument(
        "--label",
        default="probe",
        help="Free-form description added to JSON output (e.g. 'pre-fix' or 'post-fix').",
    )
    args = parser.parse_args()

    pdet = SimulationDetectionProbability(
        injection_data_dir=INJECTION_DATA_DIR,
        snr_threshold=SNR_THRESHOLD,
    )

    rows: list[dict[str, float]] = []
    for h in H_VALUES:
        pdet._get_or_build_grid(h)  # noqa: SLF001
        _, interp_1d = pdet._grid_cache[h]  # noqa: SLF001
        # Phase 45-aware: grid[0] may be the prepended anchor at 0.0, in which
        # case c_0 (the first histogram bin centre) is grid[1].  Pre-Phase-45
        # the grid had no anchor and c_0 was grid[0].  Identify by inspecting
        # whether grid[0] is exactly 0.0.
        grid_axis = interp_1d.grid[0]
        if float(grid_axis[0]) == 0.0:
            c0 = float(grid_axis[1])
        else:
            c0 = float(grid_axis[0])
        dl_max = float(grid_axis[-1])
        for d_L in DL_VALUES_GPC:
            result = pdet.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_L=d_L, phi=0.0, theta=0.0, h=h
            )
            rows.append(
                {
                    "h": float(h),
                    "d_L_gpc": float(d_L),
                    "value": float(result),
                    "c_0": float(c0),
                    "dl_max": float(dl_max),
                }
            )

    # Optional delta vs pre-fix
    if args.pre_fix_source is not None:
        pre_path = OUTPUT_DIR / args.pre_fix_source
        with pre_path.open() as f:
            pre = json.load(f)
        pre_lookup = {(r["h"], r["d_L_gpc"]): r["value"] for r in pre["rows"]}
        for r in rows:
            pre_v = pre_lookup.get((r["h"], r["d_L_gpc"]))
            if pre_v is None:
                msg = (
                    f"pre-fix probe missing row for (h={r['h']}, d_L={r['d_L_gpc']}). "
                    "Re-run pre-fix probe with the same (d_L, h) grid."
                )
                raise RuntimeError(msg)
            r["pre_fix_value"] = pre_v
            r["delta"] = r["value"] - pre_v

    # Cross-checks (only on pre-fix run; after patch the values change)
    cross_checks: dict[str, float | bool] = {}
    h073_dl010 = next(r for r in rows if r["h"] == 0.73 and r["d_L_gpc"] == 0.10)
    h073_dl001 = next(r for r in rows if r["h"] == 0.73 and r["d_L_gpc"] == 0.001)
    cross_checks["measured_h073_dl010"] = h073_dl010["value"]
    cross_checks["measured_h073_dl001"] = h073_dl001["value"]
    cross_checks["expected_pre_fix_h073_dl010_T10"] = EXPECTED_H073_DL010_PRE_FIX
    cross_checks["expected_pre_fix_h073_dl001_T10"] = EXPECTED_H073_DL001_PRE_FIX
    cross_checks["all_in_unit_interval"] = all(0.0 <= r["value"] <= 1.0 for r in rows)

    out = {
        "label": args.label,
        "description": (
            "Plan 45-02 Task 1/4: probe values of "
            "detection_probability_without_bh_mass_interpolated_zero_fill at "
            "d_L x h grid before / after the empirical-anchor patch."
        ),
        "snr_threshold": SNR_THRESHOLD,
        "injection_data_dir": INJECTION_DATA_DIR,
        "dl_values_gpc": DL_VALUES_GPC,
        "h_values": H_VALUES,
        "rows": rows,
        "cross_checks": cross_checks,
    }

    out_path = OUTPUT_DIR / args.output_name
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {len(rows)} rows ({len(H_VALUES)} h x {len(DL_VALUES_GPC)} d_L) to {out_path}")
    print(f"Cross-check h=0.73, d_L=0.10: {h073_dl010['value']:.10f}")
    print(f"  expected pre-fix (T10):    {EXPECTED_H073_DL010_PRE_FIX:.10f}")
    print(f"Cross-check h=0.73, d_L=0.001: {h073_dl001['value']:.10f}")
    print(f"  expected pre-fix (T10):      {EXPECTED_H073_DL001_PRE_FIX:.10f}")
    print(f"All values in [0,1]: {cross_checks['all_in_unit_interval']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
