"""Pre-committed scorer for PREREGISTRATION_G2_SPECZ_LIMIT.md Sec 3.

Committed BEFORE any cell is run (A8-v2). The verdict may use ONLY the
statistics this script emits; any other read is post-hoc and must be labelled
as such in the VERDICT section.

Input: nine ``pp_coverage_results.json`` cell files, one per
(rung, selection_cell) combination -- {off, 1d, cat1d} x
{0.035, 0.010, 0.002} -- named ``rung_{sigma_z}_{selection_cell}.json`` in
``--g2-cells-dir`` (produced by ``run_g2.py``), PLUS the existing prodcal
``vdeep_250_production_off`` cell for the N-b continuity check
(``--prodcal-cells-dir``).

Per cell x truth this script emits map_bias mean +- SE, coverage
cov50/68/90 +- binomial SE, rail_fraction (mirrors readout_prodcal.py).
Per rung it emits the two registered pairs Pc(r) = cat1d(r) - off(r) and
Pd(r) = 1d(r) - off(r) (prereg Sec 3): per-realization MAP delta mean +- SE
and quartiles, plus a degeneracy flag. It also emits N-b (rung-1 off cell vs
the prodcal off cell, class comparison at 3x combined SE).

Usage (invocation of record):
    python readout_g2.py --registered --g2-cells-dir cells \
        --prodcal-cells-dir <prodcal>/cells --output readout_g2_output.json
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

RUNGS = (0.035, 0.010, 0.002)
CELLS = ("off", "1d", "cat1d")


def _rail_flagged(rail_fraction: float | None) -> bool:
    return rail_fraction is not None and rail_fraction > RAIL_UNDETERMINED_THRESHOLD


def _rail_validity(rail_flags_by_truth: dict[str, bool]) -> dict[str, Any]:
    """A-PF-1 (verifier Part IV, BLOCKING) rail-gate precedence, both drafts.

    'at every truth' band legs are evaluated over the non-rail-flagged
    truths only; any PASS/FAIL adjudication requires >= 2 scoreable truths;
    a read with >= 2 truths UNDETERMINED-BY-RAIL is itself
    UNDETERMINED-BY-RAIL (unscored). A rail-flagged truth never counts
    toward a "coherent at >= 2 truths" FAIL leg.
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


def cell_id(rung: float, cell: str) -> str:
    return f"rung_{rung:.3f}_{cell}"


# Registered pairs manifest (prereg Sec 3): per rung, Pc = cat1d - off,
# Pd = 1d - off. 6 pairs total.
def build_pairs() -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    for rung in RUNGS:
        pairs.append((f"Pc({rung:.3f})", cell_id(rung, "cat1d"), cell_id(rung, "off")))
        pairs.append((f"Pd({rung:.3f})", cell_id(rung, "1d"), cell_id(rung, "off")))
    return pairs


PAIRS = build_pairs()

PRODCAL_OFF_D = "vdeep_250_production_off"


def _channel_block(block: dict[str, Any], n_real: int) -> dict[str, Any]:
    rail_fraction = block.get("rail_fraction")
    out: dict[str, Any] = {
        "map_bias": block["map_bias"],
        "map_bias_se": block["map_std"] / math.sqrt(n_real),
        "map_mean": block["map_mean"],
        "map_std": block["map_std"],
        "rail_fraction": rail_fraction,
        # PRE-FREEZE AMENDMENT A rail-fraction validity gate.
        "undetermined_by_rail": _rail_flagged(rail_fraction),
        "coverage": {},
    }
    for name in LEVELS:
        p = block["coverage"][name]
        out["coverage"][name] = {
            "value": p,
            "binomial_se": math.sqrt(NOMINAL[name] * (1.0 - NOMINAL[name]) / n_real),
            "nominal": NOMINAL[name],
        }
    return out


def score_cell(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    n_real = int(data["config"]["n_realizations"])
    cell: dict[str, Any] = {
        "cell": path.stem,
        "sigma_z": data["config"].get("sigma_z"),
        "selection_cell": data["config"].get("selection_cell"),
        "n_z_quad": data["config"].get("n_z_quad"),
        "n_realizations": n_real,
        "truths": {},
    }
    rail_flags: dict[str, bool] = {}
    for truth, block in data["results"].items():
        entry = {"channel_1d": _channel_block(block, n_real)}
        if "mass_channel_2d" in block:
            entry["channel_2d"] = _channel_block(block["mass_channel_2d"], n_real)
        cell["truths"][truth] = entry
        rail_flags[truth] = bool(entry["channel_1d"]["undetermined_by_rail"])
    # A-PF-1: this cell's own absolute-read validity (1D channel, the
    # verdict-carrying channel throughout G-1/G-2 per AMENDMENT G1-1).
    cell["rail_validity"] = _rail_validity(rail_flags)
    return cell


def paired_read(path_a: Path, path_b: Path, n_realizations_cap: int | None = None) -> dict[str, Any]:
    """[A2] paired per-realization read for two cells on a shared seed stream."""
    da = json.loads(path_a.read_text())
    db = json.loads(path_b.read_text())
    out: dict[str, Any] = {"pair": [path_a.stem, path_b.stem], "truths": {}}
    rail_flags: dict[str, bool] = {}
    for truth in da["results"]:
        if truth not in db["results"]:
            continue
        ra_rail = da["results"][truth].get("rail_fraction")
        rb_rail = db["results"][truth].get("rail_fraction")
        # A-PF-1: a truth is rail-flagged for this PAIR if EITHER cell's 1D
        # channel is rail-flagged at that truth.
        rail_flagged = _rail_flagged(ra_rail) or _rail_flagged(rb_rail)
        rail_flags[truth] = rail_flagged
        entry: dict[str, Any] = {
            "rail_fraction_a": ra_rail,
            "rail_fraction_b": rb_rail,
            "rail_flagged": rail_flagged,
        }
        for chan, getter in (
            ("channel_1d", lambda blk: blk.get("maps")),
            ("channel_2d", lambda blk: blk.get("mass_channel_2d", {}).get("maps")),
        ):
            ma, mb = getter(da["results"][truth]), getter(db["results"][truth])
            if not ma or not mb:
                entry[chan] = None
                continue
            if n_realizations_cap is not None:
                ma, mb = ma[:n_realizations_cap], mb[:n_realizations_cap]
            if len(ma) != len(mb):
                entry[chan] = None
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
    # A-PF-1: pair-level rail-gate precedence.
    out["rail_validity"] = _rail_validity(rail_flags)
    return out


def n_b_continuity(path_rung1_off: Path, path_prodcal_off: Path) -> dict[str, Any]:
    """N-b: rung-1 off cell absolute bias within 3x combined-SE of prodcal off, per truth.

    A-PF-3 (verifier Part IV): interior grid NODES align between the wide
    and original grids, but interior logL values shift slightly (the
    per-event z-quadrature windows derive from h_grid.min()/max(), so they
    widen ~7% / coarsen ~12% at fixed n_z_quad on the wide grid) -- this is
    far below N-b's 3x combined-SE tolerance, so the class comparison
    stands unchanged. A truth railed on EITHER side is EXCLUDED from the
    comparison and flagged (rail_fraction on the wide-grid rung-1 cell, or
    on the prodcal off cell -- the latter is rail_fraction ~ 0 in practice,
    per the verifier's check, but is still tested rather than assumed).
    """
    da = json.loads(path_rung1_off.read_text())
    db = json.loads(path_prodcal_off.read_text())
    n_a = int(da["config"]["n_realizations"])
    n_b = int(db["config"]["n_realizations"])
    out: dict[str, Any] = {"pair": [path_rung1_off.stem, path_prodcal_off.stem], "truths": {}}
    for truth in da["results"]:
        rb = db["results"].get(truth)
        if rb is None:
            out["truths"][truth] = {"within_band": None, "reason": "truth missing in prodcal cell"}
            continue
        ra = da["results"][truth]
        rail_flagged = _rail_flagged(ra.get("rail_fraction")) or _rail_flagged(rb.get("rail_fraction"))
        if rail_flagged:
            out["truths"][truth] = {
                "within_band": None,
                "reason": "excluded: railed on wide-grid rung-1 and/or prodcal off cell",
                "rail_fraction_a": ra.get("rail_fraction"),
                "rail_fraction_b": rb.get("rail_fraction"),
            }
            continue
        se_a = ra["map_std"] / math.sqrt(n_a)
        se_b = rb["map_std"] / math.sqrt(n_b)
        combined_se = math.sqrt(se_a**2 + se_b**2)
        diff = ra["map_bias"] - rb["map_bias"]
        out["truths"][truth] = {
            "map_bias_a": ra["map_bias"],
            "map_bias_b": rb["map_bias"],
            "diff": diff,
            "combined_se": combined_se,
            "n_combined_se": abs(diff) / combined_se if combined_se > 0 else None,
            "within_band": bool(abs(diff) <= 3.0 * combined_se),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g2-cells-dir", type=Path, default=Path("cells"))
    parser.add_argument("--prodcal-cells-dir", type=Path, required=True)
    parser.add_argument(
        "--registered",
        action="store_true",
        help="invocation of record: score every registered G-2 cell present and "
        "exactly the PAIRS + N-b manifest (missing cells reported, never skipped)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="preflight mode (Sec 3b): cap the N-b prodcal-side comparison to the "
        "probe's own realization count (shared-stream prefix)",
    )
    parser.add_argument("--output", type=Path, default=Path("readout_g2_output.json"))
    args = parser.parse_args(argv)

    cells: list[Path] = []
    pairs_out: list[dict[str, Any]] = []
    missing: list[str] = []

    if args.registered or args.probe:
        for rung in RUNGS:
            for c in CELLS:
                p = args.g2_cells_dir / f"{cell_id(rung, c)}.json"
                if p.exists():
                    cells.append(p)
                else:
                    missing.append(cell_id(rung, c))

        for name, cid_a, cid_b in PAIRS:
            pa = args.g2_cells_dir / f"{cid_a}.json"
            pb = args.g2_cells_dir / f"{cid_b}.json"
            if pa.exists() and pb.exists():
                entry = paired_read(pa, pb)
                entry["name"] = name
                pairs_out.append(entry)
            else:
                missing.append(f"{name}: {cid_a} | {cid_b}")

        rung1_off = args.g2_cells_dir / f"{cell_id(RUNGS[0], 'off')}.json"
        prodcal_off = args.prodcal_cells_dir / f"{PRODCAL_OFF_D}.json"
        n_b_out: dict[str, Any] | None = None
        if rung1_off.exists() and prodcal_off.exists():
            if args.probe:
                cap = int(json.loads(rung1_off.read_text())["config"]["n_realizations"])
                da = json.loads(rung1_off.read_text())
                db = json.loads(prodcal_off.read_text())
                for r in da["results"].values():
                    if r.get("maps"):
                        r["maps"] = r["maps"][:cap]
                for r in db["results"].values():
                    if r.get("maps"):
                        r["maps"] = r["maps"][:cap]
                # map_std/map_bias/n_realizations must reflect the SAME
                # truncated prefix for a fair probe-scale comparison.
                for data in (da, db):
                    for r in data["results"].values():
                        if r.get("maps"):
                            arr = np.asarray(r["maps"], dtype=float)
                            r["map_std"] = float(arr.std())
                            r["map_bias"] = float(arr.mean() - r["h_true"])
                    data["config"]["n_realizations"] = cap
                tmp_a = rung1_off.parent / f".__nb_a_{rung1_off.stem}.json"
                tmp_b = prodcal_off.parent / f".__nb_b_{prodcal_off.stem}.json"
                tmp_a.write_text(json.dumps(da))
                tmp_b.write_text(json.dumps(db))
                n_b_out = n_b_continuity(tmp_a, tmp_b)
                tmp_a.unlink()
                tmp_b.unlink()
            else:
                n_b_out = n_b_continuity(rung1_off, prodcal_off)
        else:
            missing.append(f"N-b: {rung1_off.stem} | {prodcal_off.stem}")

    out = {
        "cells": [score_cell(p) for p in cells],
        "pairs": pairs_out,
        "n_b_continuity": n_b_out,
        "registered_missing": missing,
        "mode": "probe" if args.probe else "registered",
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
