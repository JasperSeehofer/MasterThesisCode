"""Pre-committed scorer for PREREGISTRATION_PRODCAL_LADDER.md (row #120, D-5).

Committed BEFORE any cell is run (A8-v2). The verdict may use ONLY the
statistics this script emits; any other read is post-hoc and must be labelled
as such in the VERDICT section.

Input: one or more ``pp_coverage_results.json`` files produced by
``python -m darksiren_emri.validation.pp_coverage`` (the [A3]-extended
harness), one file per cell, named ``<cell_id>.json``.

Per cell x truth this script emits:
  - map_bias mean +- SE (SE = map_std / sqrt(n_realizations))
  - coverage cov50/cov68/cov90 +- binomial SE
  - rail_fraction
  - the same block for the ``mass_channel_2d`` channel when present

For every registered pair of cells (shared seed stream) it emits the [A2]
paired read: per-realization MAP delta mean +- SE and the delta distribution
quartiles, from the per-realization ``maps`` arrays.

Usage:
    python readout_prodcal.py CELL.json [CELL2.json ...] \
        [--pair CELL_A.json CELL_B.json] ... [--output readout.json]
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

# Registered pair manifest (prereg §3, verifier delta amendment D-3).
# cell_id = "{venue}_{n_events}_{noise_model}_{selection_cell}" per the §7 CLI
# template. The invocation of record is `--registered <cells_dir>`: it scores
# every cell file present and exactly these pairs; `--pair` outside this list
# is exploratory and never verdict-bearing.
PAIRS: list[tuple[str, str]] = [
    # N-5 engagement (fused - off) and (const - production), every venue x n
    *[
        (f"{v}_{n}_{nm}_fused", f"{v}_{n}_{nm}_off")
        for v in ("vdeep", "vctrl")
        for n in ((250, 800, 1600) if v == "vdeep" else (250,))
        for nm in ("production", "const")
        if not (v == "vctrl" and n != 250)
    ],
    ("vdeep_250_model_fused", "vdeep_250_model_off"),
    ("vctrl_250_model_fused", "vctrl_250_model_off"),
    # H-B scored read (Block B n=1600) + descriptive n=250/800 twins:
    # (const - production) under each selection cell
    *[
        (f"vdeep_{n}_const_{sc}", f"vdeep_{n}_production_{sc}")
        for n in (250, 800, 1600)
        for sc in ("fused", "off")
    ],
    ("vctrl_250_const_fused", "vctrl_250_production_fused"),
    ("vctrl_250_const_off", "vctrl_250_production_off"),
    # N-3 (V-ctrl production fused vs off) is the first V-ctrl row above.
    # Entry 19 — AMENDMENT-2 (2026-08-18): the V-flat regime-consistency pair.
    ("vflat_250_production_fused", "vflat_250_production_off"),
    # Entry 20 — AMENDMENT-4 (2026-08-18): the V-prod absolute-calibration pair.
    ("vprod_250_production_fused", "vprod_250_production_off"),
]


def _channel_block(block: dict[str, Any], n_real: int) -> dict[str, Any]:
    """Score one channel block (top-level 1D or nested mass_channel_2d)."""
    out: dict[str, Any] = {
        "map_bias": block["map_bias"],
        "map_bias_se": block["map_std"] / math.sqrt(n_real),
        "map_mean": block["map_mean"],
        "map_std": block["map_std"],
        "rail_fraction": block.get("rail_fraction"),
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
    cell: dict[str, Any] = {"cell": path.stem, "n_realizations": n_real, "truths": {}}
    for truth, block in data["results"].items():
        entry = {"channel_1d": _channel_block(block, n_real)}
        if "mass_channel_2d" in block:
            entry["channel_2d"] = _channel_block(block["mass_channel_2d"], n_real)
        cell["truths"][truth] = entry
    return cell


def paired_read(path_a: Path, path_b: Path) -> dict[str, Any]:
    """[A2] paired per-realization read for two cells on a shared seed stream."""
    da = json.loads(path_a.read_text())
    db = json.loads(path_b.read_text())
    out: dict[str, Any] = {"pair": [path_a.stem, path_b.stem], "truths": {}}
    for truth in da["results"]:
        if truth not in db["results"]:
            continue
        entry: dict[str, Any] = {}
        for chan, getter in (
            ("channel_1d", lambda blk: blk.get("maps")),
            ("channel_2d", lambda blk: blk.get("mass_channel_2d", {}).get("maps")),
        ):
            ma, mb = getter(da["results"][truth]), getter(db["results"][truth])
            if not ma or not mb or len(ma) != len(mb):
                entry[chan] = None  # None = not computable, NEVER silently skipped
                continue
            delta = np.asarray(ma, dtype=float) - np.asarray(mb, dtype=float)
            entry[chan] = {
                # N-5 engagement null: an identically-zero delta distribution
                # means a silently-inert lever — instrument suspect, STOP.
                "degenerate": bool(np.all(delta == 0.0)),
                "n_nonzero": int(np.count_nonzero(delta)),
                "delta_mean": float(delta.mean()),
                "delta_se": float(delta.std(ddof=1) / math.sqrt(delta.size)),
                "delta_q25": float(np.quantile(delta, 0.25)),
                "delta_median": float(np.median(delta)),
                "delta_q75": float(np.quantile(delta, 0.75)),
                "n_pairs": int(delta.size),
            }
        out["truths"][truth] = entry
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", nargs="*", type=Path, default=[])
    parser.add_argument(
        "--registered",
        type=Path,
        default=None,
        metavar="CELLS_DIR",
        help="invocation of record: score every <cell_id>.json in CELLS_DIR and "
        "exactly the PAIRS manifest (missing cells reported, never skipped silently)",
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        type=Path,
        default=[],
        metavar=("CELL_A", "CELL_B"),
        help="EXPLORATORY extra pair — never verdict-bearing (prereg D-3)",
    )
    parser.add_argument("--output", type=Path, default=Path("readout_prodcal_output.json"))
    args = parser.parse_args(argv)

    cells = list(args.cells)
    pairs: list[tuple[Path, Path]] = [(a, b) for a, b in args.pair]
    missing: list[str] = []
    if args.registered is not None:
        cells = sorted(args.registered.glob("*.json"))
        for cid_a, cid_b in PAIRS:
            pa, pb = args.registered / f"{cid_a}.json", args.registered / f"{cid_b}.json"
            if pa.exists() and pb.exists():
                pairs.append((pa, pb))
            else:
                missing.append(f"{cid_a} | {cid_b}")

    out = {
        "cells": [score_cell(p) for p in cells],
        "pairs": [paired_read(a, b) for a, b in pairs],
        "registered_pairs_missing": missing,
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
