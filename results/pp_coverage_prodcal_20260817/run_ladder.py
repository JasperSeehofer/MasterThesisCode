"""Campaign driver for the registered prodcal ladder (prereg §3/§7).

Generates every registered cell from the prereg's design tables and runs them
with a worker pool (parallelism is across cells; each cell is single-core, so
the CPU-h budget is unchanged). Cell naming and seeds follow §3/§7 exactly:

  cell_id = {venue}_{n}_{noise_model}_{selection_cell}
  seed    = 20270818 + 100*venue_index + 10*n_index   (shared across the
            noise x selection axes of a (venue, n) — the pairing discipline)

Block N1 replication cells use the 07-11 seed 20260701 (registered reuse).
V-deep z_support/sky_frac are read from pretuning/CHOSEN.json (the sole
permitted §7 fill-in); refusing to run if it is absent.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from darksiren_emri.validation.pp_coverage import PPCoverageConfig, run_coverage

HERE = Path(__file__).parent
TRUTHS = [0.62, 0.72, 0.84]
BASE_SEED = 20270818
VCTRL = {"z_support": 1.5, "sky_frac": 1e-4}
NOISE_KW = {
    # noise_model name -> config overrides (three-way toggle per §3)
    "production": {"gw_measurement_scatter": False, "sigma_dl_model_in_likelihood": False},
    "const": {"gw_measurement_scatter": True, "sigma_dl_model_in_likelihood": False},
    "model": {"gw_measurement_scatter": True, "sigma_dl_model_in_likelihood": True},
}


def _mass_cfg(venue: dict, n: int, nm: str, sc: str, seed: int, h_step: float) -> PPCoverageConfig:
    return PPCoverageConfig(
        n_realizations=120,
        n_events=n,
        injected_truths=TRUTHS,
        seed=seed,
        kernel="volume",
        catalogue_mode=True,
        mixture_mode="absolute",
        z_support=venue["z_support"],
        sky_frac=venue["sky_frac"],
        n_galaxies=200_000,
        mass_channel=True,
        mass_horizon_index=0.25,
        selection_cell=sc,
        h_step=h_step,
        **NOISE_KW[nm],
    )


def build_cells() -> dict[str, PPCoverageConfig]:
    chosen = json.loads((HERE / "pretuning" / "CHOSEN.json").read_text())
    vdeep = {"z_support": chosen["z_support"], "sky_frac": chosen["sky_frac"]}
    venues = {"vdeep": (0, vdeep), "vctrl": (1, VCTRL)}
    n_index = {250: 0, 800: 1, 1600: 2}
    cells: dict[str, PPCoverageConfig] = {}
    # Block A: both venues, n=250, 3 noise x 2 selection
    for vname, (vi, venue) in venues.items():
        seed = BASE_SEED + 100 * vi + 10 * n_index[250]
        for nm in ("production", "const", "model"):
            for sc in ("off", "fused"):
                cells[f"{vname}_250_{nm}_{sc}"] = _mass_cfg(venue, 250, nm, sc, seed, 0.004)
    # Block B: V-deep, n in {800, 1600}, {production, const} x {off, fused}
    for n, hs in ((800, 0.004), (1600, 0.002)):
        seed = BASE_SEED + 10 * n_index[n]
        for nm in ("production", "const"):
            for sc in ("off", "fused"):
                cells[f"vdeep_{n}_{nm}_{sc}"] = _mass_cfg(vdeep, n, nm, sc, seed, hs)
    # S-1: channel decomposition at V-deep, production noise, n=250, truth 0.72
    seed = BASE_SEED
    for sc in ("1d", "2d"):
        cfg = _mass_cfg(vdeep, 250, "production", sc, seed, 0.004)
        cells[f"vdeep_250_production_{sc}"] = dataclasses.replace(cfg, injected_truths=[0.72])
    # Block N1: 07-11 replication (continuum, mass OFF, seed 20260701)
    for zs in (0.3, 0.5):
        for cell, kw in (
            ("const", {}),
            ("modelpdet", {"sigma_dl_model_in_likelihood": True, "pdet_in_numerator": True}),
        ):
            cells[f"n1_{zs}_{cell}"] = PPCoverageConfig(
                n_realizations=120,
                n_events=250,
                injected_truths=TRUTHS,
                seed=20260701,
                kernel="volume",
                # July deep cells ran --mixture-mode exact (noisemodel RUNBOOK:81-94);
                # the first execution omitted this and fell back to two_branch,
                # reintroducing the pre-260711-117 kernel leak (+0.0235 at zs=0.3).
                mixture_mode="exact",
                z_support=zs,
                sigma_z=0.035,
                h_step=0.004,
                **kw,
            )
    return cells


def _run_one(item: tuple[str, PPCoverageConfig]) -> str:
    cell_id, cfg = item
    out = HERE / "cells" / f"{cell_id}.json"
    if out.exists():
        return f"{cell_id}: SKIP (exists)"
    t0 = time.perf_counter()
    res = run_coverage(cfg)
    out.write_text(json.dumps(res, indent=2))
    return f"{cell_id}: done in {time.perf_counter() - t0:.0f}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dry", action="store_true", help="list cells, run nothing")
    args = parser.parse_args()
    cells = build_cells()
    print(f"{len(cells)} registered cells")
    if args.dry:
        for cid in cells:
            print(" ", cid)
        return
    (HERE / "cells").mkdir(exist_ok=True)
    # Longest cells first so the pool tail is short
    order = sorted(cells.items(), key=lambda kv: -kv[1].n_events * (1 / kv[1].h_step))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for msg in pool.map(_run_one, order):
            print(msg, flush=True)
    print("LADDER COMPLETE")


if __name__ == "__main__":
    main()
