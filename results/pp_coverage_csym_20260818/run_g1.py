"""Driver for the registered G-1 cells, PRE-FREEZE AMENDMENT A
(PREREGISTRATION_G1_CATLEG_SYMMETRY.md Sec 3/3b + the amendment section).

Amended design (supersedes the original Sec 3 cell table; venues, seeds,
truths, R, noise unchanged):

Science cells -- WIDE grid h in [0.56, 0.92], h_step=0.004 (grid headroom
for the +-0.04-class edge displacement that railed the original [0.60,0.86]
grid at h_true=0.84):
  OFF-D-W     V-deep  off        seed 20270818
  FUSED-D-W   V-deep  fused      seed 20270818
  SYM-D       V-deep  symmetric  seed 20270818
  CAT-D       V-deep  cat1d      seed 20270818
  OFF-P-W     V-prod  off        seed 20271218
  FUSED-P-W   V-prod  fused      seed 20271218
  SYM-P       V-prod  symmetric  seed 20271218

N-A (byte-identity) cells -- ORIGINAL grid h in [0.60, 0.86] (must match the
referent cells' grid exactly for a bit-comparison to be meaningful):
  REP-OFF-D   V-deep  off  seed 20270818  vs a LOCAL pre-extension rerun
              (the G1-2 environment-control referent,
              cells/referent_preext_vdeep_250_production_off.json --
              computed separately at the pre-freeze HEAD on this machine)
  REP-OFF-P   V-prod  off  seed 20271218  vs the on-disk
              results/pp_coverage_prodcal_20260817/cells/vprod_250_production_off.json
              (local-vs-local, as originally registered -- no environment
              exposure on this leg)

All cells: n_events=250, R=120, truths {0.62, 0.72, 0.84}, h_step=0.004,
noise-model=production (gw_measurement_scatter=False,
sigma_dl_model_in_likelihood=False).

Rail-fraction validity gate (amendment item 6): any science cell x truth
with 1D rail_fraction > 0.10 on the WIDE grid is UNDETERMINED-BY-RAIL for
absolute reads at that truth (readout_g1.py flags it; scored descriptively).
At PROBE scale (R=4) a rail_fraction == 1.0 (every probe MAP railed) is
still a hard preflight STOP -- an all-railed probe cannot even establish
that the wide grid relaxed the pathology.

Comparison-scale clause (resolved finding, now registered in the G-1
prereg's PRE-FREEZE AMENDMENT A section): ``run_coverage`` draws its R
per-realization seeds from one master RNG stream, sequentially PER TRUTH
(all R draws for truth 1, then all R draws for truth 2, ...); a shorter
run's realizations therefore only share the master-stream offset of a
longer run's for the FIRST truth -- every later truth is drawn from a
different offset. An R=4 probe can NEVER be validly byte-compared to an
R=120 cell, at any truth beyond the first. N-A byte-identity is therefore
NOT part of the preflight (R=4) gate at all; it is scored ONLY at full
R=120, by ``readout_g1.py --registered``.

Two modes:
  --preflight   Sec 3b anti-void gate: R=4 probes at the EXACT registered
                (amended) configuration for ALL NINE cells (7 science +
                2 N-A), written to preflight/, never scored. Checks
                completion_fraction/catalogue-bearing-fraction bounds, pair
                engagement (via readout_g1.paired_read), finite/non-railed
                (rail_fraction < 1.0) MAPs, and the S_bar_phi table support.
                Does NOT attempt N-A byte-identity (see comparison-scale
                clause above). Prints READY/STOP; does NOT run the scored
                cells.
  (default)     Runs the nine registered R=120 cells into cells/, skipping
                any that already exist (idempotent re-invocation).

Usage:
    python run_g1.py --preflight
    python run_g1.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from darksiren_emri.validation.pp_coverage import (
    D50_GPC,
    Z_MAX_POP,
    PPCoverageConfig,
    run_coverage,
)

HERE = Path(__file__).parent
TRUTHS = [0.62, 0.72, 0.84]
NOISE_KW = {"gw_measurement_scatter": False, "sigma_dl_model_in_likelihood": False}

VDEEP = {"z_support": 0.40, "sky_frac": 1e-4, "d50_gpc": D50_GPC}
VPROD = {"z_support": 0.75, "sky_frac": 1e-4, "d50_gpc": D50_GPC * 8}

WIDE_GRID = {"h_min": 0.56, "h_max": 0.92}
ORIGINAL_GRID = {"h_min": 0.60, "h_max": 0.86}

# (cell_id, venue, selection_cell, seed) -- science cells, WIDE grid.
SCIENCE_CELLS: list[tuple[str, dict, str, int]] = [
    ("vdeep_250_production_off_w", VDEEP, "off", 20270818),
    ("vdeep_250_production_fused_w", VDEEP, "fused", 20270818),
    ("vdeep_250_production_symmetric_w", VDEEP, "symmetric", 20270818),
    ("vdeep_250_production_cat1d_w", VDEEP, "cat1d", 20270818),
    ("vprod_250_production_off_w", VPROD, "off", 20271218),
    ("vprod_250_production_fused_w", VPROD, "fused", 20271218),
    ("vprod_250_production_symmetric_w", VPROD, "symmetric", 20271218),
]

# (cell_id, venue, selection_cell, seed) -- N-A cells, ORIGINAL grid.
NA_CELLS: list[tuple[str, dict, str, int]] = [
    ("vdeep_250_production_off", VDEEP, "off", 20270818),  # REP-OFF-D
    ("vprod_250_production_off", VPROD, "off", 20271218),  # REP-OFF-P
]

# A-PF-5 (registered execution order): the N-A cells run FIRST so a bit-identity
# break stops the campaign at <= 0.2 CPU-h spent, before any science cell launches.
ALL_CELLS: list[tuple[str, dict, str, int, dict]] = [
    (cid, venue, sc, seed, ORIGINAL_GRID) for cid, venue, sc, seed in NA_CELLS
] + [(cid, venue, sc, seed, WIDE_GRID) for cid, venue, sc, seed in SCIENCE_CELLS]


def _cfg(venue: dict, sc: str, seed: int, n_realizations: int, grid: dict) -> PPCoverageConfig:
    return PPCoverageConfig(
        n_realizations=n_realizations,
        n_events=250,
        injected_truths=TRUTHS,
        seed=seed,
        kernel="volume",
        catalogue_mode=True,
        mixture_mode="absolute",
        z_support=venue["z_support"],
        sky_frac=venue["sky_frac"],
        d50_gpc=venue["d50_gpc"],
        n_galaxies=200_000,
        mass_channel=True,
        mass_horizon_index=0.25,
        selection_cell=sc,  # type: ignore[arg-type]
        h_step=0.004,
        h_min=grid["h_min"],
        h_max=grid["h_max"],
        **NOISE_KW,
    )


def _run_one(cell_id: str, cfg: PPCoverageConfig, out_dir: Path) -> str:
    out = out_dir / f"{cell_id}.json"
    if out.exists():
        return f"{cell_id}: SKIP (exists)"
    t0 = time.perf_counter()
    res = run_coverage(cfg)
    out.write_text(json.dumps(res))
    return f"{cell_id}: done in {time.perf_counter() - t0:.0f}s"


def preflight() -> bool:
    """Sec 3b anti-void gate (amended): R=4 probes, never scored. True if READY."""
    out_dir = HERE / "preflight"
    out_dir.mkdir(exist_ok=True)
    problems: list[str] = []

    probe_paths: dict[str, Path] = {}
    for cell_id, venue, sc, seed, grid in ALL_CELLS:
        cfg = _cfg(venue, sc, seed, n_realizations=4, grid=grid)
        msg = _run_one(cell_id, cfg, out_dir)
        print(f"[probe] {msg}", flush=True)
        probe_paths[cell_id] = out_dir / f"{cell_id}.json"

    # Check 1: completion_fraction and catalogue-bearing fraction bounds.
    for cell_id, venue, _sc, _seed, _grid in ALL_CELLS:
        data = json.loads(probe_paths[cell_id].read_text())
        venue_name = "V-deep" if venue is VDEEP else "V-prod"
        for truth, block in data["results"].items():
            cf = block["completion_fraction"]
            cat_bearing = 1.0 - cf
            if not (0.05 <= cf <= 0.95):
                problems.append(f"{cell_id}@{truth}: completion_fraction={cf:.3f} outside [0.05,0.95]")
            if cat_bearing <= 0.3:
                problems.append(
                    f"{cell_id}@{truth} ({venue_name}): catalogue-bearing fraction="
                    f"{cat_bearing:.3f} <= 0.3"
                )

    # Check 3: finite MAPs, and the amended rail gate. At probe scale (R=4)
    # rail_fraction == 1.0 (EVERY probe MAP railed) is still a hard STOP --
    # the amendment's > 0.10 UNDETERMINED-BY-RAIL threshold is a SCORING
    # flag for R=120 science reads (readout_g1.py), not a probe-engagement
    # criterion; but a fully-railed probe means the wide grid did not, in
    # fact, give this cell/truth any headroom, which the amendment exists
    # to prevent -- so it still blocks the scored run pending diagnosis.
    for cell_id, _venue, _sc, _seed, _grid in ALL_CELLS:
        data = json.loads(probe_paths[cell_id].read_text())
        for truth, block in data["results"].items():
            if not (block["map_mean"] == block["map_mean"]):  # NaN check
                problems.append(f"{cell_id}@{truth}: map_mean is NaN")
            if block["rail_fraction"] >= 1.0:
                problems.append(
                    f"{cell_id}@{truth}: rail_fraction=1.0 (every probe MAP railed on the "
                    f"registered grid -- amendment headroom did not clear this cell/truth)"
                )
            elif block["rail_fraction"] > 0.10:
                print(
                    f"  ~ {cell_id}@{truth}: probe rail_fraction={block['rail_fraction']:.2f} "
                    f"> 0.10 (UNDETERMINED-BY-RAIL at scored scale, not a probe STOP)"
                )
            for m in block["maps"]:
                if not (m == m):
                    problems.append(f"{cell_id}@{truth}: a probe MAP is NaN")

    # Check 4: S_bar_phi table support (structural, unaffected by the h-grid
    # widening: n_z_survival default spans [Z_MIN, Z_MAX_POP], independent
    # of h_min/h_max).
    for _cell_id, venue, _sc, _seed, _grid in ALL_CELLS:
        if venue["z_support"] >= Z_MAX_POP:
            problems.append(f"z_support={venue['z_support']} >= Z_MAX_POP={Z_MAX_POP}: no support margin")

    # Check 2: pair engagement (amended PAIRS manifest, all-local), via the
    # readout scorer's own paired_read.
    import readout_g1 as rg1

    for name, cid_a, cid_b in rg1.PAIRS:
        pa, pb = probe_paths.get(cid_a), probe_paths.get(cid_b)
        if pa is None or pb is None or not pa.exists() or not pb.exists():
            problems.append(f"{name}: probe cell missing ({cid_a} | {cid_b})")
            continue
        entry = rg1.paired_read(pa, pb)
        # Engagement is checked across ALL truths pooled: at R=4 probe scale
        # and h_step=0.004, a SINGLE truth's argmax can coincide by
        # discretization even when the lever is live; the pair is
        # degenerate only if EVERY truth's channel_1d delta is identically
        # zero (or not computable).
        chans = [t.get("channel_1d") for t in entry["truths"].values()]
        chans = [c for c in chans if c is not None]
        if not chans:
            problems.append(f"{name}: channel_1d probe delta not computable at any truth")
        elif all(c["degenerate"] for c in chans):
            problems.append(f"{name}: probe delta IDENTICALLY ZERO at every truth (degenerate, N-B)")

    # N-A byte-identity is NOT checked at probe scale (resolved finding,
    # coordinator 2026-08-18): realization streams do not prefix-match
    # across different n_realizations (run_coverage draws R seeds from the
    # master RNG per truth, so a shorter probe's per-truth seed sequence is
    # only aligned with a longer run's for the FIRST truth -- every later
    # truth's realizations are drawn against a different master-stream
    # offset). An R=4 probe can therefore NEVER be validly byte-compared to
    # an R=120 cell; the earlier "cross-machine" divergence this build
    # reported was this comparison-scale artifact, not an environment
    # difference (decisive same-R=120 test: the local pre-extension
    # referent IS bit-identical to the on-disk cluster cell, all 3 truths,
    # maxabsdiff 0.0). N-A is scored ONLY at full R=120 by readout_g1.py
    # --registered, comparing REP-OFF-D/REP-OFF-P against their referents
    # at matching realization counts.

    print()
    if problems:
        print("PREFLIGHT: STOP")
        for p in problems:
            print(f"  - {p}")
        return False
    print("PREFLIGHT: READY")
    return True


def run_scored() -> None:
    out_dir = HERE / "cells"
    out_dir.mkdir(exist_ok=True)
    for cell_id, venue, sc, seed, grid in ALL_CELLS:
        cfg = _cfg(venue, sc, seed, n_realizations=120, grid=grid)
        msg = _run_one(cell_id, cfg, out_dir)
        print(msg, flush=True)
    print("G1 COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        ready = preflight()
        raise SystemExit(0 if ready else 1)
    run_scored()


if __name__ == "__main__":
    main()
