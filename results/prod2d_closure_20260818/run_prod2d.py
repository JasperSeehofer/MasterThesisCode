"""Cluster driver for the registered T1/T2 cells,
PREREGISTRATION_PROD2D_CLOSURE_LANDSCAPE.md Sec 3 (T1 + T2, cluster).
Updated per the verifier Part VII amendments (P7-1, P7-6;
VERIFIER_PRECHECK_PROD2D.md Part D), applied verbatim below.

18 registered cluster cells, all production noise, catalogue_mode,
kernel="volume", mixture_mode="absolute", n_galaxies=200_000,
mass_channel=True, mass_horizon_index=0.25, sigma_mz_frac=0.10,
n_events=1600, R=120, truths [0.62, 0.72, 0.84], h grid [0.56, 0.92]
step 0.002 (181 nodes), n_z_quad=160:

  T2 grid (12 cells, V-deep, fused, seed 20280411):
    sigma_z in {0.035, 0.010, 0.002} x sigma_m_gal_frac in {0.55, 0.30, 0.10, 0.02}
    cell_id = grid_sz{sigma_z:.3f}_sm{sigma_m_gal:.2f}_fused

  T1 anchor's off twin (1 cell, V-deep, seed 20280411):
    cell_id = vdeep_anchor_off  (selection_cell="off" at the anchor
    (sigma_z=0.035, sigma_m_gal=0.55) -- T1 reuses the four grid corners
    (anchor, photo-z toggle, mass toggle, both-small) directly from the T2
    grid; this is the extra off-twin cell T1 needs for the paired
    selection read.)

  1D off-basis cells (3, P7-1, V-deep, seed 20280411): selection_cell="off"
  at (sigma_z, sigma_m_gal) = (0.035,0.55), (0.010,0.55), (0.002,0.55) --
  the 1D-leg basis for H-L1-harness/H-L2 (the fused-1D read is
  venue-scoped-asymmetric-insertion-contaminated, per rows #120-#124; the
  off-basis 1D read is the clean landscape reference at each sigma_z rung).
    cell_id = vdeep_off_sz{sigma_z:.3f}

  V-prod secondary (2 cells, seed 20280511, paired-only per the registered
  confound -- z_support=0.75, d50_gpc=8*D50_GPC):
    cell_id = vprod_anchor_fused, vprod_anchor_off  (both at
    sigma_z=0.035, sigma_m_gal=0.55)

One master seed per venue (20280411 for all 16 V-deep cells, 20280511 for
both V-prod cells) -- the sigma knobs enter the generative stream
continuously, so every cross-cell read at a shared venue is paired.

Two modes:
  --preflight  Sec 3b anti-void gate (P7-1 amended): LOCAL R=4 probes of 5
               cells (the anchor, both toggle extreme corners (0.002,0.02)
               and (0.035,0.02), one V-prod cell, and the (0.002,0.55) off
               -basis cell), written to preflight/, never scored. Checks
               completion/catalogue-bearing fraction bounds, finite MAPs,
               no 2D probe rail=1.0, 2D engagement (2D maps differ from 1D
               maps), and -- the load-bearing good-corner check (P7-1,
               replaces the earlier fused-1D expectation) -- the
               (0.002,0.55) OFF-BASIS cell's 1D channel must NOT be
               100%-railed at any truth (else the landscape's good-corner
               claim is void). Prints READY/STOP.
  (default)    Runs all 18 registered R=120 cells with a ProcessPoolExecutor
               (parallelism across cells; each cell is single-core).
               Workers = min(len(cells), $SLURM_CPUS_PER_TASK or 18).
               Output dir = $RUN_DIR (or ./cells if unset). Idempotent:
               skips any cell whose JSON already exists. Per-cell timing
               printed (flushed) as each completes.

Usage:
    python run_prod2d.py --preflight
    RUN_DIR=/path/to/workspace/run_prod2d_20260818 python run_prod2d.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from darksiren_emri.validation.pp_coverage import D50_GPC, Z_MAX_POP, PPCoverageConfig, run_coverage

HERE = Path(__file__).parent
TRUTHS = [0.62, 0.72, 0.84]
NOISE_KW = {"gw_measurement_scatter": False, "sigma_dl_model_in_likelihood": False}

VDEEP = {"z_support": 0.40, "sky_frac": 1.0e-4, "d50_gpc": D50_GPC}
VPROD = {"z_support": 0.75, "sky_frac": 1.0e-4, "d50_gpc": D50_GPC * 8}
VDEEP_SEED = 20280411
VPROD_SEED = 20280511

GRID_SIGMA_Z = (0.035, 0.010, 0.002)
GRID_SIGMA_M = (0.55, 0.30, 0.10, 0.02)
ANCHOR_SIGMA_Z = 0.035
ANCHOR_SIGMA_M = 0.55

H_MIN, H_MAX, H_STEP = 0.56, 0.92, 0.002
N_EVENTS = 1600
R_SCORED = 120
R_PREFLIGHT = 4


def _grid_cell_id(sigma_z: float, sigma_m: float) -> str:
    return f"grid_sz{sigma_z:.3f}_sm{sigma_m:.2f}_fused"


def _off_basis_cell_id(sigma_z: float) -> str:
    return f"vdeep_off_sz{sigma_z:.3f}"


def _cfg(
    venue: dict,
    selection_cell: str,
    seed: int,
    n_realizations: int,
    sigma_z: float,
    sigma_m_gal_frac: float,
) -> PPCoverageConfig:
    return PPCoverageConfig(
        n_realizations=n_realizations,
        n_events=N_EVENTS,
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
        sigma_mz_frac=0.10,
        sigma_z=sigma_z,
        sigma_m_gal_frac=sigma_m_gal_frac,
        selection_cell=selection_cell,  # type: ignore[arg-type]
        h_min=H_MIN,
        h_max=H_MAX,
        h_step=H_STEP,
        n_z_quad=160,
        **NOISE_KW,
    )


def build_cells(n_realizations: int) -> dict[str, PPCoverageConfig]:
    """Return every registered cell (16 V-deep + 2 V-prod = 18, P7-1)."""
    cells: dict[str, PPCoverageConfig] = {}
    for sigma_z in GRID_SIGMA_Z:
        for sigma_m in GRID_SIGMA_M:
            cid = _grid_cell_id(sigma_z, sigma_m)
            cells[cid] = _cfg(VDEEP, "fused", VDEEP_SEED, n_realizations, sigma_z, sigma_m)
    cells["vdeep_anchor_off"] = _cfg(
        VDEEP, "off", VDEEP_SEED, n_realizations, ANCHOR_SIGMA_Z, ANCHOR_SIGMA_M
    )
    # P7-1: 3 off-basis 1D cells at sigma_m_gal=0.55 (the H-L1-harness/H-L2
    # clean 1D reference, since fused-1D carries the asymmetric-insertion
    # class). vdeep_off_sz0.035 is config-identical to vdeep_anchor_off
    # (same seed, same sigma_z/sigma_m, selection_cell="off") -- kept as a
    # SEPARATE cell id per the registered manifest rather than aliased, so
    # every registered cell_id has its own on-disk file.
    for sigma_z in GRID_SIGMA_Z:
        cells[_off_basis_cell_id(sigma_z)] = _cfg(
            VDEEP, "off", VDEEP_SEED, n_realizations, sigma_z, ANCHOR_SIGMA_M
        )
    cells["vprod_anchor_fused"] = _cfg(
        VPROD, "fused", VPROD_SEED, n_realizations, ANCHOR_SIGMA_Z, ANCHOR_SIGMA_M
    )
    cells["vprod_anchor_off"] = _cfg(
        VPROD, "off", VPROD_SEED, n_realizations, ANCHOR_SIGMA_Z, ANCHOR_SIGMA_M
    )
    assert len(cells) == 18, f"expected 18 registered cells, built {len(cells)}"
    return cells


# The 5 Sec 3b preflight probes (P7-1 amended): anchor, both toggle extreme
# corners, 1 V-prod cell, and the (0.002, 0.55) off-basis good-sigma_z cell.
PREFLIGHT_PROBE_IDS = (
    _grid_cell_id(ANCHOR_SIGMA_Z, ANCHOR_SIGMA_M),  # anchor (0.035, 0.55)
    _grid_cell_id(0.002, 0.02),  # both-small / fused "good corner"
    _grid_cell_id(0.035, 0.02),  # mass-toggle extreme
    "vprod_anchor_fused",  # 1 V-prod cell
    _off_basis_cell_id(0.002),  # off-basis good-sigma_z (load-bearing, P7-1)
)
GOOD_CORNER_OFF_BASIS_ID = _off_basis_cell_id(0.002)


def _run_one(item: tuple[str, PPCoverageConfig], out_dir: Path) -> str:
    cell_id, cfg = item
    out = out_dir / f"{cell_id}.json"
    if out.exists():
        return f"{cell_id}: SKIP (exists)"
    t0 = time.perf_counter()
    res = run_coverage(cfg)
    out.write_text(json.dumps(res))
    dt = time.perf_counter() - t0
    print(f"{cell_id}: done in {dt:.0f}s", flush=True)
    return f"{cell_id}: done in {dt:.0f}s"


def _run_one_star(args: tuple[tuple[str, PPCoverageConfig], Path]) -> str:
    item, out_dir = args
    return _run_one(item, out_dir)


def preflight() -> bool:
    """Sec 3b anti-void gate: R=4 probes at the exact registered config, never scored."""
    out_dir = HERE / "preflight"
    out_dir.mkdir(exist_ok=True)
    problems: list[str] = []

    all_cells = build_cells(n_realizations=R_PREFLIGHT)
    probe_paths: dict[str, Path] = {}
    for cell_id in PREFLIGHT_PROBE_IDS:
        cfg = all_cells[cell_id]
        msg = _run_one((cell_id, cfg), out_dir)
        print(f"[probe] {msg}", flush=True)
        probe_paths[cell_id] = out_dir / f"{cell_id}.json"

    # Check: completion/catalogue-bearing fraction bounds (venue-dependent).
    for cell_id in PREFLIGHT_PROBE_IDS:
        cfg = all_cells[cell_id]
        is_vprod = cfg.z_support == VPROD["z_support"]
        data = json.loads(probe_paths[cell_id].read_text())
        for truth, block in data["results"].items():
            cf = block["completion_fraction"]
            cat_bearing = 1.0 - cf
            if is_vprod:
                if not (0.05 <= cf < 1.0):
                    problems.append(f"{cell_id}@{truth} (V-prod): completion_fraction={cf:.3f} outside [0.05,1.0)")
            else:
                if not (0.05 <= cf <= 0.95):
                    problems.append(f"{cell_id}@{truth} (V-deep): completion_fraction={cf:.3f} outside [0.05,0.95]")
                if cat_bearing <= 0.3:
                    problems.append(
                        f"{cell_id}@{truth} (V-deep): catalogue-bearing fraction={cat_bearing:.3f} <= 0.3"
                    )

    # Check: finite MAPs, no 2D probe rail=1.0, 2D engagement, good-corner un-rail.
    for cell_id in PREFLIGHT_PROBE_IDS:
        data = json.loads(probe_paths[cell_id].read_text())
        for truth, block in data["results"].items():
            if block["map_mean"] != block["map_mean"]:  # NaN check
                problems.append(f"{cell_id}@{truth}: map_mean (1D) is NaN")
            for m in block["maps"]:
                if m != m:
                    problems.append(f"{cell_id}@{truth}: a 1D probe MAP is NaN")
            block2d = block.get("mass_channel_2d")
            if block2d is None:
                problems.append(f"{cell_id}@{truth}: no mass_channel_2d block present")
                continue
            if block2d["map_mean"] != block2d["map_mean"]:
                problems.append(f"{cell_id}@{truth}: map_mean (2D) is NaN")
            for m in block2d["maps"]:
                if m != m:
                    problems.append(f"{cell_id}@{truth}: a 2D probe MAP is NaN")
            if block2d["rail_fraction"] >= 1.0:
                problems.append(f"{cell_id}@{truth}: 2D rail_fraction=1.0 (every probe MAP railed)")
            # 2D engagement: 2D maps must be non-degenerate vs 1D maps.
            maps_1d = block.get("maps") or []
            maps_2d = block2d.get("maps") or []
            if maps_1d and maps_2d and len(maps_1d) == len(maps_2d):
                if all(a == b for a, b in zip(maps_1d, maps_2d, strict=True)):
                    problems.append(f"{cell_id}@{truth}: 2D maps IDENTICAL to 1D maps (no mass-channel engagement)")
            else:
                problems.append(f"{cell_id}@{truth}: 1D/2D maps not comparable (length mismatch or empty)")
            # Good-corner clause (P7-1, load-bearing): at the OFF-BASIS
            # (0.002, 0.55) cell the 1D channel must UN-rail at every truth
            # (rail_fraction < 1.0), else the landscape's good-corner claim
            # is void. This replaces the earlier fused-1D expectation (the
            # fused-1D read carries the asymmetric-insertion class and is
            # not the clean 1D reference -- see H-L1-harness).
            if cell_id == GOOD_CORNER_OFF_BASIS_ID and block["rail_fraction"] >= 1.0:
                problems.append(
                    f"{cell_id}@{truth}: GOOD-CORNER CLAUSE VIOLATED (P7-1) -- off-basis 1D "
                    f"rail_fraction={block['rail_fraction']:.2f} (100%-railed) at the good sigma_z"
                )

    # Structural: S_bar_phi table support (z_support must stay below Z_MAX_POP).
    for cell_id in PREFLIGHT_PROBE_IDS:
        cfg = all_cells[cell_id]
        if cfg.z_support is not None and cfg.z_support >= Z_MAX_POP:
            problems.append(f"{cell_id}: z_support={cfg.z_support} >= Z_MAX_POP={Z_MAX_POP}: no support margin")

    print()
    if problems:
        print("PREFLIGHT: STOP")
        for p in problems:
            print(f"  - {p}")
        return False
    print("PREFLIGHT: READY")
    return True


def run_scored() -> None:
    run_dir = Path(os.environ["RUN_DIR"]) if "RUN_DIR" in os.environ else HERE / "cells"
    run_dir.mkdir(parents=True, exist_ok=True)
    cells = build_cells(n_realizations=R_SCORED)
    n_cpu = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 18))
    workers = min(len(cells), n_cpu, 18)
    print(f"{len(cells)} registered cells; {workers} workers; output -> {run_dir}", flush=True)
    # Longest-running (most h-nodes shared, so order by n_events -- constant here;
    # order by cell name for determinism) first so the pool tail stays short.
    items = sorted(cells.items())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for msg in pool.map(_run_one_star, [(item, run_dir) for item in items]):
            pass  # _run_one already prints/flushes per-cell timing as it completes
    print("PROD2D COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        ready = preflight()
        raise SystemExit(0 if ready else 1)
    run_scored()


if __name__ == "__main__":
    main()
