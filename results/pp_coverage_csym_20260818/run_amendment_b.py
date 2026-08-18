"""AMENDMENT B scored cells: SEP-Z + V-flat trio, R=120, wide grid (G-1 AMENDMENT B)."""
import json
import time
from pathlib import Path

from darksiren_emri.validation.pp_coverage import D50_GPC, PPCoverageConfig, run_coverage

HERE = Path(__file__).parent
OUT = HERE / "cells"


def _cfg(sc: str, seed: int, sigma_z: float = 0.035, d50m: float = 1.0,
         n_z_quad: int | None = None) -> PPCoverageConfig:
    kw: dict = dict(
        n_realizations=120, n_events=250, injected_truths=[0.62, 0.72, 0.84], seed=seed,
        kernel="volume", catalogue_mode=True, mixture_mode="absolute",
        z_support=0.40, sky_frac=1e-4, d50_gpc=D50_GPC * d50m, n_galaxies=200_000,
        mass_channel=True, mass_horizon_index=0.25, selection_cell=sc,
        h_min=0.56, h_max=0.92, h_step=0.004, sigma_z=sigma_z,
        gw_measurement_scatter=False, sigma_dl_model_in_likelihood=False,
    )
    if n_z_quad is not None:
        kw["n_z_quad"] = n_z_quad
    return PPCoverageConfig(**kw)


CELLS = {
    "sepz_symmetric_0.002": _cfg("symmetric", 20280311, sigma_z=0.002, n_z_quad=160),
    "vflat_250_production_off_w": _cfg("off", 20271118, d50m=8.0),
    "vflat_250_production_fused_w": _cfg("fused", 20271118, d50m=8.0),
    "vflat_250_production_symmetric_w": _cfg("symmetric", 20271118, d50m=8.0),
}

if __name__ == "__main__":
    for name, c in CELLS.items():
        p = OUT / f"{name}.json"
        if p.exists():
            print(f"{name}: SKIP", flush=True)
            continue
        t0 = time.perf_counter()
        p.write_text(json.dumps(run_coverage(c)))
        print(f"{name}: done in {time.perf_counter() - t0:.0f}s", flush=True)
    print("AMENDMENT-B CELLS COMPLETE", flush=True)
