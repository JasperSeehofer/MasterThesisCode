#!/usr/bin/env python3
"""[WIN B5.1] Zero-compute candidate-count factor on the banked cone census.

Charter node B5.1, part (A). Launched under rows #222/#223 (author standing
grant, all depths + production changes within the tree).

Reuses the FROZEN ``wgeom_instrument.py`` machinery VERBATIM (catalogue pin,
R&V15 map/prune, fleet loader, candidate-position union) rather than
re-deriving any of it -- per standing rule (2) this script is a NEW read
built on that instrument's shared plumbing, not an edit of it (append-only:
``wgeom_instrument.py`` and ``wgeom_w2_split.py`` are untouched).

Measures, over the reproducible cone-exact fleet census (bc_9001XX arms,
24 x 200 = 4800 event rows, n_all = 2,249,231 candidate rows -- the SAME
basis as ``wgeom_result.json`` P3 and ``wgeom_w2_split.json``, corrected per
CLAIM_WGEO_20260827.md §3.8/CORRECTION NOTE W-1, NOT the stale §3.9 numbers
still hardcoded as ``BANKED_P3A`` inside ``wgeom_instrument.py``):

  (i)   linear,     k=1.5  (current production default) -- GATE: must
        reproduce n_lin/n_all = 0.9577 (the corrected §3.8 number).
  (ii)  log-symmetric, k=1.5
  (iii) log-symmetric, k=3.0  (ratified candidate design, row #221 F-ii;
        eps = 2*Phi(-3) = 0.27%; k=3 on BOTH sides -- GW side and candidate
        side -- matching current same-k convention)
  (iv)  log-symmetric, k=2.5

For each: total pass fraction, per-event candidate-growth factor of (iii)
vs (i) [mean/median/p95/max], and the fraction of TRUE hosts retained
inside the mass window -- read from each event's own
``host_galaxy_index``/``in_catalog`` columns (production truth-labeling
convention, ``main.py:826-830``; unscattered fleet, so
``host_galaxy_index`` indexes ``self.reduced_galaxy_catalog`` directly per
``handler.py:_resolve_host_recovery_position``'s ``not self.scattered``
branch -- verified: no ``observed_realization`` sidecar under any
``bc_9001XX_work`` arm).

Status: BUILDER-RUN zero-compute census, not an independently-verified
registered measurement. Per standing rule (2) (verifier independence), a
DIFFERENT agent should re-run this before any number here is cited as
adopted; this run is presentation-support for the gate package (part B),
not itself a physics-change verification.

REVISION NOTE (2026-08-29, appended, does not edit the text above): a
refuter panel caught a MATERIAL BUG in the run this docstring describes --
``gw_window()`` (then at module scope, no ``geometry`` parameter) ALWAYS
used the linear formula for the GW-side window, even for the three "log"
CONFIGS (ii/iii/iv), contradicting this module's own claim above ("k=3 on
BOTH sides") and §2 of PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md. This
fix (see ``gw_window()``'s own revision note) + the re-run that produced the
corrected ``b5_window_count.json`` were performed by a DIFFERENT AGENT from
whoever authored the module docstring and the original buggy `gw_window`
above (branch worker, charter node B5.1, fanout wave 1, 2026-08-29) --
satisfying standing rule (2)'s verifier-independence requirement for this
correction, not merely a builder smoke-test. The gate-(i) [linear/linear]
result and the 0.9577 reproduction are UNAFFECTED (linear geometry never
touched the buggy branch) and are reproduced identically after the fix
(see the re-run's provenance block in ``b5_window_count.json``). All
config-(ii)/(iii)/(iv) numbers in this file and in §7 of the presentation
doc were REGENERATED after the fix; the pre-fix numbers are superseded and
must not be cited (see the presentation doc's own dated revision notes for
the specific value-by-value diff).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

_WGEOM_DIR = Path(__file__).resolve().parents[1]  # .../realistic_20260729/
sys.path.insert(0, str(_WGEOM_DIR))
from wgeom_instrument import (  # noqa: E402
    FLEET_BASE,
    FleetEvent,
    PrunedCatalogue,
    iter_fleet_arms,
    load_pruned_catalogue,
    verify_catalogue_pin,
    verify_fleet_row_counts,
)
from darksiren_emri.physical_relations import get_redshift_outer_bounds  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
H_MIN, H_MAX, OMEGA_M_MIN, OMEGA_M_MAX, W_0, W_A = 0.6, 0.86, 0.04, 0.5, -1.0, 0.0
REDSHIFT_UPPER_LIMIT = 1.5

# WGEOM-corrected census basis (CLAIM_WGEO_20260827.md CORRECTION NOTE W-1,
# 2026-08-28; the instrument's own hardcoded BANKED_P3A is the SUPERSEDED
# §3.9 number and is deliberately NOT used as the gate target here).
GATE_TARGET_N_LIN_OVER_N_ALL = 0.9577  # 4dp; full precision 0.9576899838211371
GATE_TARGET_SOURCE = (
    "wgeom_result.json p3.ratios.n_lin_over_n_all=0.9576899838211371 "
    "(builder-run 2026-08-28, ledger row #220/#221 W-1 correction) "
    "== CLAIM_WGEO_20260827.md line 199"
)


@dataclass
class TruthFleetEvent(FleetEvent):
    host_galaxy_index: int
    in_catalog: bool


def load_fleet_arm_events_with_truth(arm_dir: str) -> list[TruthFleetEvent]:
    """``load_fleet_arm_events`` (wgeom_instrument.py:633-707) + truth columns.

    Duplicated rather than imported because the frozen function's return type
    (``FleetEvent``) has no truth fields and standing rule (1) forbids editing
    that file's text. Every numeric step below is copy-verbatim from that
    function; the only addition is reading ``host_galaxy_index``/``in_catalog``
    from the same already-open CSV and attaching them per row.
    """
    import glob

    seed_dirs = glob.glob(str(Path(arm_dir) / "seed*"))
    if len(seed_dirs) != 1:
        raise RuntimeError(f"expected exactly one seed dir under {arm_dir}, got {seed_dirs}")
    seed_dir = Path(seed_dirs[0])
    csv_path = seed_dir / "simulations" / "prepared_cramer_rao_bounds.csv"
    json_path = seed_dir / "simulations" / "posteriors_with_bh_mass" / "h_0_73.json"

    df = pd.read_csv(
        csv_path,
        usecols=[
            "M",
            "delta_M_delta_M",
            "luminosity_distance",
            "delta_luminosity_distance_delta_luminosity_distance",
            "z_true",
            "host_galaxy_index",
            "in_catalog",
        ],
    )
    with open(json_path) as f:
        posteriors = json.load(f)
    gl = posteriors.get("galaxy_likelihoods", {})
    add = posteriors.get("additional_galaxies_without_bh_mass", {})

    events: list[TruthFleetEvent] = []
    keys = sorted(set(gl.keys()) | set(add.keys()), key=int)
    arm_name = Path(arm_dir).name
    for k in keys:
        idx = int(k)
        if idx >= len(df):
            continue
        row = df.iloc[idx]
        M_z = float(row["M"])
        M_z_sigma = float(np.sqrt(row["delta_M_delta_M"]))
        d_L = float(row["luminosity_distance"])
        d_L_sigma = float(np.sqrt(row["delta_luminosity_distance_delta_luminosity_distance"]))
        z_min, z_max = get_redshift_outer_bounds(
            distance=d_L,
            distance_error=d_L_sigma,
            h_min=H_MIN,
            h_max=H_MAX,
            Omega_m_min=OMEGA_M_MIN,
            Omega_m_max=OMEGA_M_MAX,
            w_0=W_0,
            w_a=W_A,
            sigma_multiplier=2.0,
        )
        z_max = min(z_max, REDSHIFT_UPPER_LIMIT)

        gl_entries = gl.get(k, [])
        add_entries = add.get(k, [])
        lin_pass_positions = {int(e[0]) for e in gl_entries}
        all_positions = np.array(
            sorted(lin_pass_positions | {int(e[0]) for e in add_entries}), dtype=np.int64
        )

        events.append(
            TruthFleetEvent(
                arm=arm_name,
                event_idx=idx,
                M_z=M_z,
                M_z_sigma=M_z_sigma,
                z_min=float(z_min),
                z_max=float(z_max),
                z_true=float(row["z_true"]),
                candidate_positions=all_positions,
                linear_pass_positions=lin_pass_positions,
                host_galaxy_index=int(row["host_galaxy_index"]),
                in_catalog=bool(row["in_catalog"]),
            )
        )
    return events


def load_fleet_with_truth() -> list[TruthFleetEvent]:
    events: list[TruthFleetEvent] = []
    for arm_dir in iter_fleet_arms(FLEET_BASE, "bc_9001??_work"):
        events.extend(load_fleet_arm_events_with_truth(arm_dir))
    return events


def gw_window(
    M_z: float, M_z_sigma: float, z_min: float, z_max: float, k: float, geometry: str
) -> tuple[float, float]:
    """GW-side mass window, geometry-aware.

    REVISION NOTE (2026-08-29, different agent from the builder, per standing
    rule 2): this function originally ALWAYS used the linear formula
    regardless of ``geometry`` -- a refuter-panel-caught bug (the doc's own
    §2/Code-shape spec requires log/exp on BOTH sides under
    ``mass_filter_geometry="log"``). Fixed here to branch on ``geometry``,
    matching PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md §2 exactly:
    ``sigma_lnM_z = M_z_sigma / M_z`` (guarded against ``M_z <= 0`` -- not
    observed in the fleet, but the near-zero-sigma tail IS observed, per the
    refuter's own materiality check: median fractional error 1.6e-8), then
    ``gw_lo = M_z*exp(-k*sigma_lnM_z)/(1+z_max)``,
    ``gw_hi = M_z*exp(+k*sigma_lnM_z)/(1+z_min)``. This is a crash-safety
    guard + logic fix disclosed per standing rule (2); no other behavior
    changed.
    """
    if geometry == "linear":
        gw_lo = (M_z - k * M_z_sigma) / (1.0 + z_max)
        gw_hi = (M_z + k * M_z_sigma) / (1.0 + z_min)
    elif geometry == "log":
        if M_z <= 0:
            raise ValueError(f"M_z must be > 0 for log geometry, got {M_z!r}")
        sigma_lnM_z = M_z_sigma / M_z
        gw_lo = M_z * np.exp(-k * sigma_lnM_z) / (1.0 + z_max)
        gw_hi = M_z * np.exp(k * sigma_lnM_z) / (1.0 + z_min)
    else:
        raise ValueError(geometry)
    return gw_lo, gw_hi


def pass_mask(
    m: npt.NDArray[np.float64],
    me: npt.NDArray[np.float64],
    gw_lo: float,
    gw_hi: float,
    k: float,
    geometry: str,
) -> npt.NDArray[np.bool_]:
    if geometry == "linear":
        lo = m - k * me
        hi = m + k * me
    elif geometry == "log":
        lo = m * np.exp(-k * (me / m))
        hi = m * np.exp(k * (me / m))
    else:
        raise ValueError(geometry)
    return (gw_lo <= hi) & (lo <= gw_hi)


CONFIGS: list[tuple[str, str, float]] = [
    ("i_linear_k1.5", "linear", 1.5),
    ("ii_log_k1.5", "log", 1.5),
    ("iii_log_k3.0", "log", 3.0),
    ("iv_log_k2.5", "log", 2.5),
]


def run() -> dict[str, Any]:
    t0 = time.time()
    catalogue_md5 = verify_catalogue_pin()
    verify_fleet_row_counts()
    cat: PrunedCatalogue = load_pruned_catalogue(nrows=None)
    events = load_fleet_with_truth()

    # -----------------------------------------------------------------
    # Per-config totals + per-event pass counts (for the growth factor)
    # -----------------------------------------------------------------
    per_event_counts: dict[str, list[int]] = {label: [] for label, _, _ in CONFIGS}
    totals: dict[str, dict[str, int]] = {
        label: {"n_all": 0, "n_pass": 0} for label, _, _ in CONFIGS
    }

    # True-host retention bookkeeping
    truth_valid_n = 0  # in_catalog & host_galaxy_index in-range
    truth_pass: dict[str, int] = {label: 0 for label, _, _ in CONFIGS}
    truth_out_of_range = 0

    for ev in events:
        pos = ev.candidate_positions
        # REVISION NOTE (2026-08-29): keyed by (geometry, k), not just k --
        # the GW window now depends on geometry too (see gw_window()'s
        # revision note above). Previously this cache was keyed by k alone,
        # which was harmless only because gw_window() ignored geometry; now
        # that it doesn't, a geometry-blind cache would silently reuse the
        # wrong-geometry window across configs sharing a k (there are none
        # in CONFIGS today -- k=1.5 appears once as linear, once as log --
        # so this is a forward-safety fix, not a materiality fix for the
        # current CONFIGS list).
        gw_windows_by_geom_k: dict[tuple[str, float], tuple[float, float]] = {}
        for label, geometry, k in CONFIGS:
            cache_key = (geometry, k)
            if cache_key not in gw_windows_by_geom_k:
                gw_windows_by_geom_k[cache_key] = gw_window(
                    ev.M_z, ev.M_z_sigma, ev.z_min, ev.z_max, k, geometry
                )
            gw_lo, gw_hi = gw_windows_by_geom_k[cache_key]

            if pos.size == 0:
                per_event_counts[label].append(0)
            else:
                m = cat.bh_mass[pos]
                me = cat.bh_mass_error[pos]
                mask = pass_mask(m, me, gw_lo, gw_hi, k, geometry)
                n_pass = int(mask.sum())
                per_event_counts[label].append(n_pass)
                totals[label]["n_all"] += int(pos.size)
                totals[label]["n_pass"] += n_pass

        # True-host retention: mass-window-only test on the host's own
        # catalogue mass/error, independent of the sky+redshift cone
        # (isolates the effect of the geometry/k parameter under test).
        if ev.in_catalog and ev.host_galaxy_index >= 0:
            hp = ev.host_galaxy_index
            if hp < cat.n_pruned:
                truth_valid_n += 1
                hm = cat.bh_mass[hp]
                hme = cat.bh_mass_error[hp]
                for label, geometry, k in CONFIGS:
                    gw_lo, gw_hi = gw_windows_by_geom_k[(geometry, k)]
                    ok = bool(pass_mask(np.array([hm]), np.array([hme]), gw_lo, gw_hi, k, geometry)[0])
                    if ok:
                        truth_pass[label] += 1
            else:
                truth_out_of_range += 1

    # -----------------------------------------------------------------
    # Pass-fraction summary + G1-style gate on config (i)
    # -----------------------------------------------------------------
    pass_fraction: dict[str, float] = {}
    for label, _, _ in CONFIGS:
        n_all = totals[label]["n_all"]
        pass_fraction[label] = totals[label]["n_pass"] / n_all if n_all else float("nan")

    gate_i_rel4dp = round(pass_fraction["i_linear_k1.5"], 4)
    gate_passed = gate_i_rel4dp == GATE_TARGET_N_LIN_OVER_N_ALL

    # -----------------------------------------------------------------
    # Per-event growth factor: (iii) log k=3 vs (i) linear k=1.5
    # -----------------------------------------------------------------
    n_i = np.array(per_event_counts["i_linear_k1.5"], dtype=np.float64)
    n_iii = np.array(per_event_counts["iii_log_k3.0"], dtype=np.float64)
    has_denom = n_i > 0
    ratios = n_iii[has_denom] / n_i[has_denom]
    n_zero_to_nonzero = int(np.sum((n_i == 0) & (n_iii > 0)))
    n_zero_to_zero = int(np.sum((n_i == 0) & (n_iii == 0)))

    growth_factor = {
        "n_events_with_nonzero_linear_candidates": int(has_denom.sum()),
        "mean": float(np.mean(ratios)) if ratios.size else None,
        "median": float(np.median(ratios)) if ratios.size else None,
        "p95": float(np.percentile(ratios, 95)) if ratios.size else None,
        "max": float(np.max(ratios)) if ratios.size else None,
        "n_events_zero_linear_candidates_gain_some_under_iii": n_zero_to_nonzero,
        "n_events_zero_under_both": n_zero_to_zero,
    }

    # -----------------------------------------------------------------
    # True-host retention fractions
    # -----------------------------------------------------------------
    true_host_retention = {
        "n_events_total": len(events),
        "n_events_with_valid_catalogued_true_host": truth_valid_n,
        "n_true_host_position_out_of_pruned_range": truth_out_of_range,
        "note": (
            "Mass-window-only test on the true host's own catalogue "
            "BH_MASS/BH_MASS_ERROR vs the observed GW mass window at the "
            "same k -- independent of the sky+redshift cone cut, to isolate "
            "the mass-geometry parameter under test. host_galaxy_index reads "
            "directly as a position in this handler's pruned frame: fleet is "
            "UNSCATTERED (no observed_realization sidecar found under any "
            "bc_9001XX_work arm), so handler.py's "
            "resolve_host_recovery_position 'not self.scattered' branch "
            "(handler.py:718-721) applies -- injection-time position == "
            "evaluation-time position."
        ),
        "fraction_retained": {
            label: (truth_pass[label] / truth_valid_n if truth_valid_n else None)
            for label, _, _ in CONFIGS
        },
        "n_retained": truth_pass,
    }

    result: dict[str, Any] = {
        "provenance": {
            "launched_under": "rows #222/#223 -- charter node B5.1",
            "git_commit": None,  # filled by caller via write_outputs
            "catalogue_md5": catalogue_md5,
            "fleet_base": str(FLEET_BASE),
            "fleet_arm_glob": "bc_9001??_work",
            "n_events_loaded": len(events),
            "elapsed_s": time.time() - t0,
        },
        "configs": [{"label": label, "geometry": geometry, "k": k} for label, geometry, k in CONFIGS],
        "pass_fraction": pass_fraction,
        "totals": totals,
        "gate_i_reproduces_0.9577": {
            "computed_full_precision": pass_fraction["i_linear_k1.5"],
            "computed_4dp": gate_i_rel4dp,
            "target": GATE_TARGET_N_LIN_OVER_N_ALL,
            "target_source": GATE_TARGET_SOURCE,
            "passed": gate_passed,
        },
        "growth_factor_iii_vs_i": growth_factor,
        "true_host_retention": true_host_retention,
    }
    return result


def _git_commit() -> str:
    import subprocess

    repo_root = _WGEOM_DIR.parents[2]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def main() -> None:
    result = run()
    result["provenance"]["git_commit"] = _git_commit()

    out_path = OUT_DIR / "b5_window_count.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"wrote {out_path}")
    print(json.dumps(
        {
            "pass_fraction": result["pass_fraction"],
            "gate_i_reproduces_0.9577": result["gate_i_reproduces_0.9577"],
            "growth_factor_iii_vs_i": result["growth_factor_iii_vs_i"],
            "true_host_retention_fraction": result["true_host_retention"]["fraction_retained"],
            "n_events_with_valid_catalogued_true_host": result["true_host_retention"][
                "n_events_with_valid_catalogued_true_host"
            ],
        },
        indent=2,
        default=str,
    ))


if __name__ == "__main__":
    main()
