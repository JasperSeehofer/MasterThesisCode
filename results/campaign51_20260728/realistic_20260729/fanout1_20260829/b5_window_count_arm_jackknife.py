#!/usr/bin/env python3
"""[WIN B5.1] Arm-level jackknife -- A15 operating-characteristics support.

Charter node B5.1, revision response to the refuter panel's must_fix item
"Add an A15 operating-characteristics paragraph ... give the null
distribution / false-fail rate / detectable effect at this N". Launched
under rows #222/#223, produced by the SAME agent that performed the
gw_window() bug fix + re-run (a DIFFERENT agent from whoever built the
original b5_window_count.py -- standing rule 2).

Append-only, new file: does not edit b5_window_count.py or any banked
artifact. Reuses ``load_fleet_with_truth`` / ``gw_window`` / ``pass_mask``
from the (now geometry-fixed) b5_window_count.py verbatim, imported, not
duplicated.

What this measures: the 24 ``bc_9001XX_work`` arms are 24 independently
seeded fleet runs (per ``PREREGISTRATION_MKER_WGEOM_20260828.md`` and
``wgeom_instrument.py``'s own fleet-arm iteration). Treating each arm's
own aggregate pass-fraction / true-host-retention-fraction as one draw
gives a cheap, honest, DATA-DRIVEN (not assumed) empirical estimate of
arm-to-arm (seed-to-seed) fluctuation at this N -- an approximation to
what a fresh 24-arm fleet at a different base seed would show, NOT a
formal sampling-theory null distribution (the arms are not drawn i.i.d.
from a stated model in this read; they are the 24 seeds the fleet
happens to have). Framed as a lower bound on true fleet-to-fleet spread:
a genuinely independent fleet (different galaxy-catalogue draw, different
detector noise realizations at the population level) could vary more than
these same-catalogue, same-selection-pipeline arms do.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_window_count import (  # noqa: E402
    CONFIGS,
    gw_window,
    load_fleet_with_truth,
    pass_mask,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wgeom_instrument import load_pruned_catalogue, verify_catalogue_pin, verify_fleet_row_counts  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def run() -> dict:
    t0 = time.time()
    verify_catalogue_pin()
    verify_fleet_row_counts()
    cat = load_pruned_catalogue(nrows=None)
    events = load_fleet_with_truth()

    arms = sorted({ev.arm for ev in events})
    per_arm: dict[str, dict] = {arm: {"n_events": 0} for arm in arms}
    for label, _, _ in CONFIGS:
        for arm in arms:
            per_arm[arm][f"pass_fraction__{label}"] = None
            per_arm[arm][f"n_all__{label}"] = 0
            per_arm[arm][f"n_pass__{label}"] = 0
            per_arm[arm][f"truth_valid__{label}"] = 0
            per_arm[arm][f"truth_pass__{label}"] = 0

    for ev in events:
        arm = ev.arm
        per_arm[arm]["n_events"] += 1
        pos = ev.candidate_positions
        gw_windows_by_geom_k: dict[tuple[str, float], tuple[float, float]] = {}
        for label, geometry, k in CONFIGS:
            key = (geometry, k)
            if key not in gw_windows_by_geom_k:
                gw_windows_by_geom_k[key] = gw_window(ev.M_z, ev.M_z_sigma, ev.z_min, ev.z_max, k, geometry)
            gw_lo, gw_hi = gw_windows_by_geom_k[key]
            if pos.size:
                m = cat.bh_mass[pos]
                me = cat.bh_mass_error[pos]
                mask = pass_mask(m, me, gw_lo, gw_hi, k, geometry)
                per_arm[arm][f"n_all__{label}"] += int(pos.size)
                per_arm[arm][f"n_pass__{label}"] += int(mask.sum())
        if ev.in_catalog and ev.host_galaxy_index >= 0 and ev.host_galaxy_index < cat.n_pruned:
            hp = ev.host_galaxy_index
            hm = cat.bh_mass[hp]
            hme = cat.bh_mass_error[hp]
            for label, geometry, k in CONFIGS:
                gw_lo, gw_hi = gw_windows_by_geom_k[(geometry, k)]
                per_arm[arm][f"truth_valid__{label}"] += 1
                ok = bool(pass_mask(np.array([hm]), np.array([hme]), gw_lo, gw_hi, k, geometry)[0])
                if ok:
                    per_arm[arm][f"truth_pass__{label}"] += 1

    for arm in arms:
        for label, _, _ in CONFIGS:
            n_all = per_arm[arm][f"n_all__{label}"]
            per_arm[arm][f"pass_fraction__{label}"] = (
                per_arm[arm][f"n_pass__{label}"] / n_all if n_all else None
            )
            tv = per_arm[arm][f"truth_valid__{label}"]
            per_arm[arm][f"retention_fraction__{label}"] = (
                per_arm[arm][f"truth_pass__{label}"] / tv if tv else None
            )

    summary: dict = {"n_arms": len(arms)}
    for label, _, _ in CONFIGS:
        pf_vals = np.array([per_arm[a][f"pass_fraction__{label}"] for a in arms], dtype=np.float64)
        rf_vals = np.array([per_arm[a][f"retention_fraction__{label}"] for a in arms], dtype=np.float64)
        summary[label] = {
            "pass_fraction_across_arms": {
                "mean": float(np.mean(pf_vals)),
                "std": float(np.std(pf_vals, ddof=1)),
                "min": float(np.min(pf_vals)),
                "max": float(np.max(pf_vals)),
            },
            "retention_fraction_across_arms": {
                "mean": float(np.mean(rf_vals)),
                "std": float(np.std(rf_vals, ddof=1)),
                "min": float(np.min(rf_vals)),
                "max": float(np.max(rf_vals)),
            },
        }

    return {
        "provenance": {
            "launched_under": "rows #222/#223 -- charter node B5.1, A15 revision response",
            "n_arms": len(arms),
            "n_events_total": len(events),
            "elapsed_s": time.time() - t0,
        },
        "per_arm": per_arm,
        "summary_across_arms": summary,
    }


def main() -> None:
    result = run()
    out_path = OUT_DIR / "b5_window_count_arm_jackknife.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"wrote {out_path}")
    print(json.dumps(result["summary_across_arms"], indent=2))


if __name__ == "__main__":
    main()
