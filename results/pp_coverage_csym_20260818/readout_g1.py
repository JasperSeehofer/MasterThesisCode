"""Pre-committed scorer for PREREGISTRATION_G1_CATLEG_SYMMETRY.md Sec 3,
as superseded by PRE-FREEZE AMENDMENT A (all-local twins, wide science
grid, rail-fraction validity gate, 1D-channel-only scoring).

Committed BEFORE any cell is run (A8-v2). The verdict may use ONLY the
statistics this script emits; any other read is post-hoc and must be
labelled as such in the VERDICT section.

Input: ``pp_coverage_results.json`` cell files, one per registered cell,
named ``<cell_id>.json``, ALL in ``--g1-cells-dir`` (produced by
``run_g1.py``) -- the amendment moves every paired twin to a local,
same-grid, same-seed cell; the on-disk prodcal cells are no longer PAIRS
referents (they remain the Sec 2 anchor SOURCE, descriptive only). N-A
byte-identity still needs two external referents (a local pre-extension
worktree rerun for the V-deep leg, the on-disk prodcal off cell for the
V-prod leg, local-vs-local) -- pass their directories via
``--referent-dir`` (default: ``--g1-cells-dir``, where
``run_g1.py`` writes ``referent_preext_vdeep_250_production_off.json``)
and ``--prodcal-cells-dir``.

Per cell x truth this script emits map_bias mean +- SE, coverage
cov50/68/90 +- binomial SE, rail_fraction, PLUS (amendment item 6) an
``undetermined_by_rail`` flag when 1D rail_fraction > 0.10 (mirrors
readout_prodcal.py's ``_channel_block``, extended). AMENDMENT G1-1: **all
registered reads are scored on the 1D channel (``channel_1d``) only; the 2D
channel is reported descriptively and is never verdict-bearing** (the
output carries an explicit ``verdict_scope`` note).

For every registered pair (amended PAIRS manifest below) it emits the [A2]
paired read: per-realization MAP delta mean +- SE and quartiles, plus a
degeneracy flag (N-B), and the per-cell rail_fraction alongside so a paired
read at a railed truth is legible as such. It ALSO emits the N-A
byte-identity check against the two amended referents.

Usage (invocation of record):
    python readout_g1.py --registered \
        --g1-cells-dir cells --prodcal-cells-dir <prodcal>/cells \
        --output readout_g1_output.json
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

# Science cells (WIDE grid), amendment item 2.
OFF_D_W = "vdeep_250_production_off_w"
FUSED_D_W = "vdeep_250_production_fused_w"
SYM_D = "vdeep_250_production_symmetric_w"
CAT_D = "vdeep_250_production_cat1d_w"
OFF_P_W = "vprod_250_production_off_w"
FUSED_P_W = "vprod_250_production_fused_w"
SYM_P = "vprod_250_production_symmetric_w"

# N-A cells (ORIGINAL grid), amendment item 4.
REP_OFF_D = "vdeep_250_production_off"
REP_OFF_P = "vprod_250_production_off"

REFERENT_OFF_D = "referent_preext_vdeep_250_production_off"  # --referent-dir
REFERENT_OFF_P = "vprod_250_production_off"  # --prodcal-cells-dir (local-vs-local)

SCIENCE_CELLS = (OFF_D_W, FUSED_D_W, SYM_D, CAT_D, OFF_P_W, FUSED_P_W, SYM_P)
NA_CELLS = (REP_OFF_D, REP_OFF_P)

# Amended registered pairs manifest (PRE-FREEZE AMENDMENT A item 3): all
# twins are now same-environment, same-grid, same-seed, all-local.
PAIRS: list[tuple[str, str, str]] = [
    ("P1", SYM_D, FUSED_D_W),
    ("P2", SYM_D, OFF_D_W),
    ("P3", CAT_D, OFF_D_W),
    ("P4", SYM_P, FUSED_P_W),
    ("P5", SYM_P, OFF_P_W),
]


def _rail_flagged(rail_fraction: float | None) -> bool:
    return rail_fraction is not None and rail_fraction > RAIL_UNDETERMINED_THRESHOLD


def _rail_validity(rail_flags_by_truth: dict[str, bool]) -> dict[str, Any]:
    """A-PF-1 (verifier Part IV, BLOCKING) rail-gate precedence, both drafts.

    'at every truth' band legs are evaluated over the non-rail-flagged
    truths only; any PASS/FAIL adjudication requires >= 2 scoreable truths;
    a read with >= 2 truths UNDETERMINED-BY-RAIL is itself
    UNDETERMINED-BY-RAIL (unscored, returns to the author with the rail
    diagnostics). A rail-flagged truth never counts toward a "coherent at
    >= 2 truths" FAIL leg -- i.e. it is excluded from ``scoreable_truths``
    entirely, not merely down-weighted.
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


def _channel_block(block: dict[str, Any], n_real: int) -> dict[str, Any]:
    """Score one channel block (top-level 1D or nested mass_channel_2d)."""
    rail_fraction = block.get("rail_fraction")
    out: dict[str, Any] = {
        "map_bias": block["map_bias"],
        "map_bias_se": block["map_std"] / math.sqrt(n_real),
        "map_mean": block["map_mean"],
        "map_std": block["map_std"],
        "rail_fraction": rail_fraction,
        # PRE-FREEZE AMENDMENT A item 6: any cell x truth with 1D rail
        # fraction > 0.10 on the wide grid is UNDETERMINED-BY-RAIL for
        # absolute reads at that truth.
        "undetermined_by_rail": bool(
            rail_fraction is not None and rail_fraction > RAIL_UNDETERMINED_THRESHOLD
        ),
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
    """Score every truth of one cell file."""
    data = json.loads(path.read_text())
    n_real = int(data["config"]["n_realizations"])
    cell: dict[str, Any] = {
        "cell": path.stem,
        "selection_cell": data["config"].get("selection_cell"),
        "h_min": data["config"].get("h_min"),
        "h_max": data["config"].get("h_max"),
        "n_realizations": n_real,
        "truths": {},
    }
    rail_flags: dict[str, bool] = {}
    for truth, block in data["results"].items():
        entry = {"channel_1d": _channel_block(block, n_real)}
        if "mass_channel_2d" in block:
            # AMENDMENT G1-1: descriptive only, never verdict-bearing.
            entry["channel_2d_descriptive"] = _channel_block(block["mass_channel_2d"], n_real)
        cell["truths"][truth] = entry
        # A-PF-1: rail-gate precedence is scored on the 1D channel only
        # (AMENDMENT G1-1's verdict scope).
        rail_flags[truth] = bool(entry["channel_1d"]["undetermined_by_rail"])
    # A-PF-1: this cell's own absolute-read validity, for "at every truth"
    # bands scored directly off this cell (e.g. H-SYM absolute legs).
    cell["rail_validity"] = _rail_validity(rail_flags)
    return cell


def paired_read(path_a: Path, path_b: Path, n_realizations_cap: int | None = None) -> dict[str, Any]:
    """[A2] paired per-realization read for two cells on a shared seed stream.

    1D-channel-only per AMENDMENT G1-1: the 2D delta is still computed and
    reported (never dropped -- [A2] "never silently skipped"), but is
    labelled descriptive, not verdict-bearing.
    """
    da = json.loads(path_a.read_text())
    db = json.loads(path_b.read_text())
    out: dict[str, Any] = {"pair": [path_a.stem, path_b.stem], "truths": {}}
    rail_flags: dict[str, bool] = {}
    for truth in da["results"]:
        if truth not in db["results"]:
            continue
        ra, rb = da["results"][truth], db["results"][truth]
        # A-PF-1: a truth is rail-flagged for this PAIR if EITHER cell's 1D
        # channel is rail-flagged at that truth (the pair's delta is only as
        # trustworthy as its least-trustworthy leg).
        rail_flagged = _rail_flagged(ra.get("rail_fraction")) or _rail_flagged(rb.get("rail_fraction"))
        rail_flags[truth] = rail_flagged
        entry: dict[str, Any] = {
            "rail_fraction_a": ra.get("rail_fraction"),
            "rail_fraction_b": rb.get("rail_fraction"),
            "rail_flagged": rail_flagged,
        }
        for chan, getter, descriptive in (
            ("channel_1d", lambda blk: blk.get("maps"), False),
            ("channel_2d_descriptive", lambda blk: blk.get("mass_channel_2d", {}).get("maps"), True),
        ):
            ma, mb = getter(ra), getter(rb)
            if not ma or not mb:
                entry[chan] = None  # None = not computable, NEVER silently skipped
                continue
            if n_realizations_cap is not None:
                ma, mb = ma[:n_realizations_cap], mb[:n_realizations_cap]
            if len(ma) != len(mb):
                entry[chan] = None
                continue
            delta = np.asarray(ma, dtype=float) - np.asarray(mb, dtype=float)
            entry[chan] = {
                "descriptive_only": descriptive,
                # N-B engagement null: an identically-zero delta distribution
                # means a silently-inert lever -- instrument suspect, STOP.
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
    # A-PF-1: pair-level rail-gate precedence (1D-channel scoring scope).
    out["rail_validity"] = _rail_validity(rail_flags)
    return out


def byte_identity_check(path_a: Path, path_b: Path) -> dict[str, Any]:
    """N-A: bit-compare MAP arrays of two cell files, per truth, at MATCHING R.

    Comparison-scale clause (resolved finding, registered in PRE-FREEZE
    AMENDMENT A): ``run_coverage`` draws its R per-realization seeds from
    one master RNG stream, sequentially PER TRUTH -- a shorter run's
    realizations only share a longer run's master-stream offset for the
    FIRST truth; every later truth is drawn from a different offset. A
    byte-identity claim is therefore only meaningful between two files with
    the SAME ``n_realizations`` (both sides of every registered N-A pair
    are R=120). This function REFUSES a mismatched-R comparison outright
    (no silent ``min(len)`` truncation, which would produce a
    coincidentally-equal-length but semantically meaningless compare for
    every truth after the first) rather than reporting a false negative.
    """
    da = json.loads(path_a.read_text())
    db = json.loads(path_b.read_text())
    n_real_a = int(da["config"]["n_realizations"])
    n_real_b = int(db["config"]["n_realizations"])
    out: dict[str, Any] = {
        "pair": [path_a.stem, path_b.stem],
        "n_realizations_a": n_real_a,
        "n_realizations_b": n_real_b,
        "truths": {},
    }
    if n_real_a != n_real_b:
        out["bit_exact_all"] = False
        out["reason"] = (
            f"n_realizations mismatch ({n_real_a} vs {n_real_b}): realization streams do "
            "not prefix-match across different R (see docstring) -- comparison refused, "
            "not attempted"
        )
        return out
    for truth in da["results"]:
        ra = da["results"].get(truth)
        rb = db["results"].get(truth)
        if ra is None or rb is None:
            out["truths"][truth] = {"bit_exact": False, "reason": "truth missing in one file"}
            continue
        maps_a = ra.get("maps")
        maps_b = rb.get("maps")
        exact = bool(maps_a and maps_b and len(maps_a) == len(maps_b) and maps_a == maps_b)
        out["truths"][truth] = {
            "bit_exact": exact,
            "n_a": len(maps_a) if maps_a else 0,
            "n_b": len(maps_b) if maps_b else 0,
        }
    out["bit_exact_all"] = all(v["bit_exact"] for v in out["truths"].values())
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-cells-dir", type=Path, default=Path("cells"))
    parser.add_argument(
        "--referent-dir",
        type=Path,
        default=None,
        help="dir containing referent_preext_vdeep_250_production_off.json "
        "(default: --g1-cells-dir)",
    )
    parser.add_argument("--prodcal-cells-dir", type=Path, required=True)
    parser.add_argument(
        "--registered",
        action="store_true",
        help="invocation of record: score every G-1 cell present and exactly "
        "the amended PAIRS/N-A manifest (missing cells reported, never skipped silently)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="preflight mode (Sec 3b): scores PAIRS engagement at probe (R=4) "
        "scale only. N-A byte-identity is NEVER attempted in this mode -- "
        "realization streams do not prefix-match across different R, so an "
        "R=4 probe cannot be validly byte-compared to an R=120 referent "
        "(comparison-scale clause, PRE-FREEZE AMENDMENT A).",
    )
    parser.add_argument("--output", type=Path, default=Path("readout_g1_output.json"))
    args = parser.parse_args(argv)
    referent_dir = args.referent_dir if args.referent_dir is not None else args.g1_cells_dir

    cells: list[Path] = []
    pairs_out: list[dict[str, Any]] = []
    na_out: list[dict[str, Any]] = []
    missing: list[str] = []

    if args.registered or args.probe:
        for cid in SCIENCE_CELLS + NA_CELLS:
            p = args.g1_cells_dir / f"{cid}.json"
            if p.exists():
                cells.append(p)
            else:
                missing.append(cid)

        cap = None
        if args.probe:
            probe_path = args.g1_cells_dir / f"{SYM_D}.json"
            if probe_path.exists():
                cap = int(json.loads(probe_path.read_text())["config"]["n_realizations"])

        for name, cid_a, cid_b in PAIRS:
            pa, pb = args.g1_cells_dir / f"{cid_a}.json", args.g1_cells_dir / f"{cid_b}.json"
            if pa.exists() and pb.exists():
                entry = paired_read(pa, pb, n_realizations_cap=cap)
                entry["name"] = name
                pairs_out.append(entry)
            else:
                missing.append(f"{name}: {cid_a} | {cid_b}")

        # N-A byte-identity: --registered (full R=120) ONLY. Comparison-scale
        # clause (PRE-FREEZE AMENDMENT A): realization streams do not
        # prefix-match across different n_realizations, so this is refused
        # entirely under --probe rather than attempted at R=4 (see
        # byte_identity_check's own R-mismatch guard for the same reasoning,
        # kept as defense in depth).
        if args.registered:
            na_targets = [
                (REP_OFF_D, referent_dir / f"{REFERENT_OFF_D}.json"),
                (REP_OFF_P, args.prodcal_cells_dir / f"{REFERENT_OFF_P}.json"),
            ]
            for cid_g1, referent_path in na_targets:
                pa = args.g1_cells_dir / f"{cid_g1}.json"
                if pa.exists() and referent_path.exists():
                    na_out.append(byte_identity_check(pa, referent_path))
                else:
                    missing.append(f"N-A: {cid_g1} | {referent_path}")

    out = {
        "verdict_scope": "channel_1d only (AMENDMENT G1-1); channel_2d_descriptive is "
        "reported but never verdict-bearing",
        "cells": [score_cell(p) for p in cells],
        "pairs": pairs_out,
        "n_a_byte_identity": na_out,
        "registered_missing": missing,
        "mode": "probe" if args.probe else "registered",
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
