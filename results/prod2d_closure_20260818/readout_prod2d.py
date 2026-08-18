"""Pre-committed registered scorer for
PREREGISTRATION_PROD2D_CLOSURE_LANDSCAPE.md Sec 3 (T1+T2 cluster cells).
Updated per the verifier Part VII amendments (P7-1, P7-3, P7-7;
VERIFIER_PRECHECK_PROD2D.md Part D), applied verbatim below.

Committed BEFORE the cluster job returns. The closure/landscape verdict may
use ONLY the statistics this script emits for the T1/T2 legs; any other
read is post-hoc and must be labelled as such in the VERDICT section.

Per cell x truth x channel (top-level block = 1D ``combined_no_bh``-analog,
nested ``mass_channel_2d`` = 2D ``combined_with_bh``-analog -- BOTH channels
are verdict-bearing here, unlike the G-1 campaign): map bias mean +- SE,
map_std, cov50/68/90 +- binomial SE, rail fraction, RMS error
``sqrt(bias**2 + map_std**2)``, an ``undetermined_by_rail`` flag
(rail_fraction > 0.10) with A-PF-1 precedence (every-truth legs are
evaluated over non-rail-flagged truths only; >= 2 scoreable truths
required, else the cell/channel read is itself UNDETERMINED-BY-RAIL), and
an ``undetermined_by_quantization`` flag (N-3 guard: map_std < 1.5*h_step).
**P7-7 (applied verbatim):** when quantization-flagged, ``map_std`` and
``rms_error`` are UPPER BOUNDS -- the block additionally quotes
``sigma_real_upper_bound = max(map_std, 1.5*h_step)`` and
``rms_upper_bound = sqrt(bias**2 + sigma_real_upper_bound**2)`` explicitly
labelled as such (the true realization scatter at good rungs is expected
below the grid's resolution floor).

**P7-1 (applied verbatim): the landscape's 1D legs (H-L1-harness, H-L2) are
scored on the OFF-BASIS cells (``vdeep_off_sz{sigma_z}``) only** -- the
fused-1D read at the same (sigma_z, sigma_m_gal=0.55) is reported
alongside with the (fused - off) INSERTION DELTA explicitly labelled the
venue-scoped asymmetric-insertion class (rows #120-#124, G-2
sigma_z-collapse); the fused-1D read itself is NEVER quoted as a "1D
starves" landscape verdict input. See ``insertion_delta_1d`` in the output
and the ``landscape_1d_reference`` block.

Registered pairs (paired per-realization deltas, on the shared per-venue
seed stream -- every T1/T2 cell at a venue draws from the SAME master
seed, so cross-cell reads at that venue are paired by construction):
  - every T2 grid cell (11 of the 12) MINUS the anchor cell
    (grid_sz0.035_sm0.55_fused) -- the sigma_z/sigma_m_gal landscape read.
  - fused - off at the V-deep anchor (grid_sz0.035_sm0.55_fused vs
    vdeep_anchor_off).
  - fused - off at V-prod (vprod_anchor_fused vs vprod_anchor_off).
  - (P7-1) fused - off-basis at each grid sigma_z rung (0.035, 0.010,
    0.002), sigma_m_gal=0.55 -- the asymmetric-insertion-class deltas.

N-1 continuity (class comparison, NOT byte-identity -- different seed and
grid): the (sigma_z=0.035, sigma_m_gal=0.30) grid cell
(grid_sz0.035_sm0.30_fused) vs
results/pp_coverage_prodcal_20260817/cells/vdeep_1600_production_fused.json,
bias compared within 3x the combined SE, per truth per channel.

**H-T1a engagement precondition (P7-3, applied verbatim):** at the anchor
cell's 2D channel, primary truth 0.72: |bias(anchor)| >= max(0.004, 5*SE)
is required before any class-attribution (both-small toggle collapse) read
is scored; if the precondition fails the read is UNDETERMINED-BY-DESIGN
(unscored -- there is no amplitude to attribute). See
``h_t1a_engagement_precondition`` in the output.

Usage (invocation of record):
    python readout_prod2d.py --registered <cells_dir> \
        --prodcal-cells-dir ../pp_coverage_prodcal_20260817/cells \
        --output readout_prod2d_output.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

LEVELS = ("50", "68", "90")
NOMINAL = {"50": 0.50, "68": 0.68, "90": 0.90}
RAIL_UNDETERMINED_THRESHOLD = 0.10
QUANTIZATION_FACTOR = 1.5

ANCHOR_ID = "grid_sz0.035_sm0.55_fused"
N1_GRID_ID = "grid_sz0.035_sm0.30_fused"
N1_REFERENT_NAME = "vdeep_1600_production_fused"
VDEEP_OFF_ID = "vdeep_anchor_off"
VPROD_FUSED_ID = "vprod_anchor_fused"
VPROD_OFF_ID = "vprod_anchor_off"

GRID_SIGMA_Z = (0.035, 0.010, 0.002)
GRID_SIGMA_M = (0.55, 0.30, 0.10, 0.02)

# H-T1a engagement precondition (P7-3).
H_T1A_PRIMARY_TRUTH = "0.7200"
H_T1A_BIAS_FLOOR = 0.004
H_T1A_SE_MULTIPLE = 5.0

# P7-7: quantization upper-bound factor (matches N-3's 1.5x h_step).
QUANTIZATION_UPPER_BOUND_FACTOR = 1.5


def _grid_cell_id(sigma_z: float, sigma_m: float) -> str:
    return f"grid_sz{sigma_z:.3f}_sm{sigma_m:.2f}_fused"


def _off_basis_cell_id(sigma_z: float) -> str:
    return f"vdeep_off_sz{sigma_z:.3f}"


GRID_CELL_IDS = [
    _grid_cell_id(sz, sm) for sz in GRID_SIGMA_Z for sm in GRID_SIGMA_M
]
# P7-1: the 3 off-basis 1D cells (sigma_m_gal=0.55, selection_cell="off").
OFF_BASIS_CELL_IDS = [_off_basis_cell_id(sz) for sz in GRID_SIGMA_Z]
ALL_CELL_IDS = GRID_CELL_IDS + OFF_BASIS_CELL_IDS + [VDEEP_OFF_ID, VPROD_FUSED_ID, VPROD_OFF_ID]

PAIRS: list[tuple[str, str, str]] = (
    [(f"grid_minus_anchor__{cid}", cid, ANCHOR_ID) for cid in GRID_CELL_IDS if cid != ANCHOR_ID]
    + [
        ("fused_minus_off__vdeep_anchor", ANCHOR_ID, VDEEP_OFF_ID),
        ("fused_minus_off__vprod_anchor", VPROD_FUSED_ID, VPROD_OFF_ID),
    ]
    # P7-1: fused - off-basis "insertion delta" at each grid sigma_z rung,
    # sigma_m_gal=0.55 -- the venue-scoped asymmetric-insertion class.
    + [
        (
            f"insertion_delta_1d__sz{sz:.3f}",
            _grid_cell_id(sz, 0.55),
            _off_basis_cell_id(sz),
        )
        for sz in GRID_SIGMA_Z
    ]
)


def _rail_flagged(rail_fraction: float | None) -> bool:
    return rail_fraction is not None and rail_fraction > RAIL_UNDETERMINED_THRESHOLD


def _rail_validity(rail_flags_by_truth: dict[str, bool]) -> dict[str, Any]:
    """A-PF-1 rail-gate precedence (mirrors readout_g1.py's ``_rail_validity``).

    'at every truth' band legs are evaluated over the non-rail-flagged
    truths only; any adjudication requires >= 2 scoreable truths; a read
    with >= 2 truths UNDETERMINED-BY-RAIL is itself UNDETERMINED-BY-RAIL.
    """
    scoreable = sorted(t for t, flagged in rail_flags_by_truth.items() if not flagged)
    rail_flagged = sorted(t for t, flagged in rail_flags_by_truth.items() if flagged)
    state = "UNDETERMINED-BY-RAIL" if len(scoreable) < 2 else "SCOREABLE"
    return {
        "scoreable_truths": scoreable,
        "rail_flagged_truths": rail_flagged,
        "n_scoreable": len(scoreable),
        "n_rail_flagged": len(rail_flagged),
        "state": state,
    }


def _channel_block(block: dict[str, Any], n_real: int, h_step: float) -> dict[str, Any]:
    """Score one channel block (top-level 1D or nested mass_channel_2d)."""
    rail_fraction = block.get("rail_fraction")
    map_bias = block["map_bias"]
    map_std = block["map_std"]
    rms = math.sqrt(map_bias**2 + map_std**2)
    undetermined_by_quantization = bool(map_std < QUANTIZATION_FACTOR * h_step)
    out: dict[str, Any] = {
        "map_bias": map_bias,
        "map_bias_se": map_std / math.sqrt(n_real),
        "map_mean": block["map_mean"],
        "map_std": map_std,
        "rms_error": rms,
        "rail_fraction": rail_fraction,
        "undetermined_by_rail": bool(_rail_flagged(rail_fraction)),
        "undetermined_by_quantization": undetermined_by_quantization,
        "coverage": {},
    }
    if undetermined_by_quantization:
        # P7-7: quantization-flagged cells quote sigma_real/RMS as UPPER
        # BOUNDS -- the grid's resolution floor, not the true realization
        # scatter (expected below it at good rungs).
        sigma_real_upper_bound = max(map_std, QUANTIZATION_UPPER_BOUND_FACTOR * h_step)
        out["sigma_real_upper_bound"] = sigma_real_upper_bound
        out["rms_upper_bound"] = math.sqrt(map_bias**2 + sigma_real_upper_bound**2)
        out["quantization_note"] = (
            f"map_std/rms_error are UPPER BOUNDS (< max(measured, {QUANTIZATION_UPPER_BOUND_FACTOR}*h_step))"
        )
    for name in LEVELS:
        p = block["coverage"][name]
        out["coverage"][name] = {
            "value": p,
            "binomial_se": math.sqrt(NOMINAL[name] * (1.0 - NOMINAL[name]) / n_real),
            "nominal": NOMINAL[name],
        }
    return out


def score_cell(path: Path) -> dict[str, Any]:
    """Score every truth x channel of one cell file."""
    data = json.loads(path.read_text())
    cfg = data["config"]
    n_real = int(cfg["n_realizations"])
    h_step = float(cfg["h_step"])
    cell: dict[str, Any] = {
        "cell": path.stem,
        "selection_cell": cfg.get("selection_cell"),
        "sigma_z": cfg.get("sigma_z"),
        "sigma_m_gal_frac": cfg.get("sigma_m_gal_frac"),
        "z_support": cfg.get("z_support"),
        "d50_gpc": cfg.get("d50_gpc"),
        "seed": cfg.get("seed"),
        "n_events": cfg.get("n_events"),
        "n_realizations": n_real,
        "h_min": cfg.get("h_min"),
        "h_max": cfg.get("h_max"),
        "h_step": h_step,
        "truths": {},
    }
    rail_flags_1d: dict[str, bool] = {}
    rail_flags_2d: dict[str, bool] = {}
    for truth, block in data["results"].items():
        entry: dict[str, Any] = {"channel_1d": _channel_block(block, n_real, h_step)}
        rail_flags_1d[truth] = bool(entry["channel_1d"]["undetermined_by_rail"])
        block2d = block.get("mass_channel_2d")
        if block2d is not None:
            entry["channel_2d"] = _channel_block(block2d, n_real, h_step)
            rail_flags_2d[truth] = bool(entry["channel_2d"]["undetermined_by_rail"])
        cell["truths"][truth] = entry
    cell["rail_validity_1d"] = _rail_validity(rail_flags_1d)
    if rail_flags_2d:
        cell["rail_validity_2d"] = _rail_validity(rail_flags_2d)
    return cell


def paired_read(path_a: Path, path_b: Path) -> dict[str, Any]:
    """Paired per-realization read for two cells on a shared seed stream.

    Both channels (1D, 2D) are verdict-bearing here -- neither is dropped
    to descriptive-only status (unlike the G-1 campaign's AMENDMENT G1-1).
    """
    da = json.loads(path_a.read_text())
    db = json.loads(path_b.read_text())
    out: dict[str, Any] = {"pair": [path_a.stem, path_b.stem], "truths": {}}
    rail_flags: dict[str, bool] = {}
    for truth in da["results"]:
        if truth not in db["results"]:
            continue
        ra, rb = da["results"][truth], db["results"][truth]
        rail_flagged = _rail_flagged(ra.get("rail_fraction")) or _rail_flagged(rb.get("rail_fraction"))
        rail_flags[truth] = rail_flagged
        entry: dict[str, Any] = {
            "rail_fraction_a": ra.get("rail_fraction"),
            "rail_fraction_b": rb.get("rail_fraction"),
            "rail_flagged": rail_flagged,
        }
        for chan, getter in (
            ("channel_1d", lambda blk: blk.get("maps")),
            ("channel_2d", lambda blk: blk.get("mass_channel_2d", {}).get("maps")),
        ):
            ma, mb = getter(ra), getter(rb)
            if not ma or not mb or len(ma) != len(mb):
                entry[chan] = None  # None = not computable, NEVER silently skipped
                continue
            delta = np.asarray(ma, dtype=float) - np.asarray(mb, dtype=float)
            entry[chan] = {
                "degenerate": bool(np.all(delta == 0.0)),
                "n_nonzero": int(np.count_nonzero(delta)),
                "delta_mean": float(delta.mean()),
                "delta_se": (
                    float(delta.std(ddof=1) / math.sqrt(delta.size)) if delta.size > 1 else None
                ),
                "delta_q25": float(np.quantile(delta, 0.25)),
                "delta_median": float(np.median(delta)),
                "delta_q75": float(np.quantile(delta, 0.75)),
                "n_pairs": int(delta.size),
            }
        out["truths"][truth] = entry
    out["rail_validity"] = _rail_validity(rail_flags)
    return out


def n1_continuity(grid_path: Path, referent_path: Path) -> dict[str, Any]:
    """N-1 continuity: class comparison (different seed + grid), 3x combined-SE."""
    dg = json.loads(grid_path.read_text())
    dr = json.loads(referent_path.read_text())
    ng = int(dg["config"]["n_realizations"])
    nr = int(dr["config"]["n_realizations"])
    out: dict[str, Any] = {
        "grid_cell": grid_path.stem,
        "referent_cell": referent_path.stem,
        "note": "class comparison only -- different seed and h-grid (registered)",
        "truths": {},
    }
    for truth in dg["results"]:
        if truth not in dr["results"]:
            out["truths"][truth] = {"comparable": False, "reason": "truth missing in referent"}
            continue
        bg, br = dg["results"][truth], dr["results"][truth]
        entry: dict[str, Any] = {"comparable": True}
        for chan, key in (("channel_1d", None), ("channel_2d", "mass_channel_2d")):
            blk_g = bg if key is None else bg.get(key)
            blk_r = br if key is None else br.get(key)
            if blk_g is None or blk_r is None:
                entry[chan] = None
                continue
            bias_g, bias_r = blk_g["map_bias"], blk_r["map_bias"]
            se_g = blk_g["map_std"] / math.sqrt(ng)
            se_r = blk_r["map_std"] / math.sqrt(nr)
            combined_se = math.hypot(se_g, se_r)
            diff = bias_g - bias_r
            entry[chan] = {
                "bias_grid": bias_g,
                "bias_referent": bias_r,
                "diff": diff,
                "combined_se": combined_se,
                "n_combined_se": (abs(diff) / combined_se) if combined_se > 0 else float("inf"),
                "within_3_combined_se": bool(abs(diff) <= 3.0 * combined_se),
            }
        out["truths"][truth] = entry
    return out


def h_t1a_engagement_precondition(anchor_path: Path) -> dict[str, Any]:
    """P7-3: engagement precondition for the H-T1a class-attribution read.

    |bias(anchor, 2D, primary truth)| >= max(0.004, 5*SE) is required
    before the both-small-toggle collapse read may be scored; else
    UNDETERMINED-BY-DESIGN (no amplitude to attribute).
    """
    data = json.loads(anchor_path.read_text())
    n_real = int(data["config"]["n_realizations"])
    block = data["results"].get(H_T1A_PRIMARY_TRUTH, {}).get("mass_channel_2d")
    if block is None:
        return {
            "state": "UNDETERMINED-BY-DESIGN",
            "reason": f"anchor cell has no mass_channel_2d block at truth {H_T1A_PRIMARY_TRUTH}",
        }
    bias = block["map_bias"]
    se = block["map_std"] / math.sqrt(n_real)
    floor = max(H_T1A_BIAS_FLOOR, H_T1A_SE_MULTIPLE * se)
    engaged = abs(bias) >= floor
    return {
        "anchor_cell": anchor_path.stem,
        "primary_truth": H_T1A_PRIMARY_TRUTH,
        "bias_anchor_2d": bias,
        "se": se,
        "floor": floor,
        "engaged": bool(engaged),
        "state": "SCOREABLE" if engaged else "UNDETERMINED-BY-DESIGN",
    }


def landscape_1d_reference(cells_dir: Path) -> list[dict[str, Any]]:
    """P7-1: per sigma_z rung, the clean off-basis 1D reference plus the
    fused-1D read and its insertion delta -- grouped for the H-L1-harness/
    H-L2 landscape legs (never quote fused-1D as the "1D starves" input).
    """
    out: list[dict[str, Any]] = []
    for sz in GRID_SIGMA_Z:
        off_path = cells_dir / f"{_off_basis_cell_id(sz)}.json"
        fused_path = cells_dir / f"{_grid_cell_id(sz, 0.55)}.json"
        entry: dict[str, Any] = {"sigma_z": sz, "off_basis_cell": off_path.stem, "fused_cell": fused_path.stem}
        if off_path.exists():
            entry["off_basis_channel_1d"] = score_cell(off_path)
        else:
            entry["off_basis_channel_1d"] = None
        if fused_path.exists():
            entry["fused_channel_1d"] = score_cell(fused_path)
        else:
            entry["fused_channel_1d"] = None
        entry["insertion_delta_pair_name"] = f"insertion_delta_1d__sz{sz:.3f}"
        out.append(entry)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cells_dir_positional",
        nargs="?",
        default=None,
        help="dir containing the registered cell JSONs (positional form of --cells-dir)",
    )
    parser.add_argument("--cells-dir", type=Path, default=None)
    parser.add_argument(
        "--prodcal-cells-dir",
        type=Path,
        default=Path("../pp_coverage_prodcal_20260817/cells"),
        help="dir containing vdeep_1600_production_fused.json (N-1 referent)",
    )
    parser.add_argument(
        "--registered",
        action="store_true",
        help="invocation of record: score every prod2d cell present and exactly "
        "the registered PAIRS/N-1 manifest (missing cells reported, never skipped silently)",
    )
    parser.add_argument("--output", type=Path, default=Path("readout_prod2d_output.json"))
    args = parser.parse_args(argv)

    cells_dir = args.cells_dir
    if cells_dir is None:
        cells_dir = Path(args.cells_dir_positional) if args.cells_dir_positional else Path("cells")

    cells: list[Path] = []
    pairs_out: list[dict[str, Any]] = []
    missing: list[str] = []
    n1_out: dict[str, Any] | None = None

    if args.registered:
        for cid in ALL_CELL_IDS:
            p = cells_dir / f"{cid}.json"
            if p.exists():
                cells.append(p)
            else:
                missing.append(cid)

        for name, cid_a, cid_b in PAIRS:
            pa, pb = cells_dir / f"{cid_a}.json", cells_dir / f"{cid_b}.json"
            if pa.exists() and pb.exists():
                entry = paired_read(pa, pb)
                entry["name"] = name
                pairs_out.append(entry)
            else:
                missing.append(f"{name}: {cid_a} | {cid_b}")

        grid_n1 = cells_dir / f"{N1_GRID_ID}.json"
        referent_n1 = args.prodcal_cells_dir / f"{N1_REFERENT_NAME}.json"
        if grid_n1.exists() and referent_n1.exists():
            n1_out = n1_continuity(grid_n1, referent_n1)
        else:
            missing.append(f"N-1: {N1_GRID_ID} | {referent_n1}")

    anchor_path = cells_dir / f"{ANCHOR_ID}.json"
    h_t1a_out: dict[str, Any] | None = None
    if args.registered:
        if anchor_path.exists():
            h_t1a_out = h_t1a_engagement_precondition(anchor_path)
        else:
            missing.append(f"H-T1a precondition: {ANCHOR_ID}")

    landscape_1d_out: list[dict[str, Any]] | None = None
    if args.registered:
        landscape_1d_out = landscape_1d_reference(cells_dir)

    out = {
        "verdict_scope": "channel_1d and channel_2d are BOTH verdict-bearing (unlike G-1); "
        "P7-1: the landscape's 1D legs (H-L1-harness, H-L2) are scored on the off-basis "
        "cells only -- fused-1D is descriptive + the insertion delta, never a landscape "
        "1D-starves input.",
        "cells": [score_cell(p) for p in cells],
        "pairs": pairs_out,
        "n1_continuity": n1_out,
        "h_t1a_engagement_precondition": h_t1a_out,
        "landscape_1d_reference": landscape_1d_out,
        "registered_missing": missing,
        "mode": "registered" if args.registered else "unscored",
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
