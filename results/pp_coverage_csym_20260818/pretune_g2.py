"""G-2 registered pretuning fill-in (PREREGISTRATION_G2_SPECZ_LIMIT.md Sec 7),
as superseded by AMENDMENT A-PF-4 (verifier Part IV): the Q* sweep and the
N-c doubling probe run on the WIDE grid, not the narrow one.

Registered recipe (Sec 7, archived, never scored): seed 20280399,
sigma_z=0.002 (rung 3, the smallest/most quadrature-sensitive rung), n=250,
R=8, venue V-deep exactly as G-1/G-2 Sec 3, **h in [0.56, 0.92] (A-PF-4)** --
the wide grid whose ~7% wider / ~12% coarser per-event z-quadrature windows
(h_grid.min()/h_grid.max()-derived, `_completion_numerator_batch`) is
EXACTLY what the convergence gate must bound; a Q* certified on the narrow
grid is void for scoring (A-PF-4). Sweep n_z_quad ASCENDING over
{default=160, 240, 480, 960}. Q* = the SMALLEST value whose per-truth MAP
bias changes by <= 0.0005 vs the NEXT LARGER value. Record the runtime
multiplier kappa vs the default. Exhaustion (960 not converged) => STOP ->
author -> amendment. Only convergence and runtime fields are consulted (no
other statistic from this run is verdict-bearing anywhere in G-1/G-2).

selection_cell="off" is used: n_z_quad is a shared numerical resolution knob
for the per-event z-quadrature (Sec 0: "the only axis varying between rungs
is sigma_z itself" -- n_z_quad convergence is a property of that shared
z-integral machinery, not of any particular selection_cell's numerator
insertion), so the cheapest cell that exercises it (mass channel, off) is
the correct -- and fastest -- convergence probe.

Usage:
    python pretune_g2.py [--output preflight/pretune_g2.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from darksiren_emri.validation.pp_coverage import D50_GPC, PPCoverageConfig, run_coverage

HERE = Path(__file__).parent
TRUTHS = [0.62, 0.72, 0.84]
SWEEP = (160, 240, 480, 960)  # 160 = the harness default
CONVERGENCE_TOL = 0.0005
# A-PF-4 (verifier Part IV, BLOCKING): the wide grid, not the narrow one.
WIDE_H_MIN = 0.56
WIDE_H_MAX = 0.92


def _cfg(n_z_quad: int) -> PPCoverageConfig:
    return PPCoverageConfig(
        n_realizations=8,
        n_events=250,
        injected_truths=TRUTHS,
        seed=20280399,
        kernel="volume",
        catalogue_mode=True,
        mixture_mode="absolute",
        z_support=0.40,
        sky_frac=1e-4,
        d50_gpc=D50_GPC,
        n_galaxies=200_000,
        mass_channel=True,
        mass_horizon_index=0.25,
        selection_cell="off",
        h_step=0.004,
        h_min=WIDE_H_MIN,
        h_max=WIDE_H_MAX,
        sigma_z=0.002,
        n_z_quad=n_z_quad,
        gw_measurement_scatter=False,
        sigma_dl_model_in_likelihood=False,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "preflight" / "pretune_g2.json")
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    runs: dict[int, dict] = {}
    for nzq in SWEEP:
        t0 = time.perf_counter()
        res = run_coverage(_cfg(nzq))
        dt = time.perf_counter() - t0
        runs[nzq] = {
            "runtime_s": dt,
            "map_bias": {t: b["map_bias"] for t, b in res["results"].items()},
        }
        print(f"n_z_quad={nzq}: {dt:.1f}s " + " ".join(f"{t}={b['map_bias']:+.4f}" for t, b in res["results"].items()), flush=True)

    kappa_default = runs[SWEEP[0]]["runtime_s"]
    convergence_table = []
    q_star = None
    for i in range(len(SWEEP) - 1):
        lo, hi = SWEEP[i], SWEEP[i + 1]
        deltas = {
            t: abs(runs[hi]["map_bias"][t] - runs[lo]["map_bias"][t]) for t in runs[lo]["map_bias"]
        }
        max_delta = max(deltas.values())
        converged = max_delta <= CONVERGENCE_TOL
        convergence_table.append(
            {"lo": lo, "hi": hi, "deltas": deltas, "max_delta": max_delta, "converged": converged}
        )
        if converged and q_star is None:
            q_star = lo

    exhausted = q_star is None
    out = {
        "grid": {"h_min": WIDE_H_MIN, "h_max": WIDE_H_MAX},
        "amendment": "A-PF-4 (verifier Part IV, BLOCKING): wide-grid Q*/N-c evidence "
        "-- a narrow-grid ([0.60, 0.86]) sweep is void for scoring",
        "sweep": list(SWEEP),
        "runtimes_s": {nzq: runs[nzq]["runtime_s"] for nzq in SWEEP},
        "kappa_vs_default": {nzq: runs[nzq]["runtime_s"] / kappa_default for nzq in SWEEP},
        "convergence_table": convergence_table,
        "q_star": q_star,
        "exhausted": exhausted,
        "convergence_tol": CONVERGENCE_TOL,
    }
    args.output.write_text(json.dumps(out, indent=2))
    print()
    if exhausted:
        print(f"PRETUNE: STOP (not converged by n_z_quad={SWEEP[-1]}; return to author)")
    else:
        print(f"PRETUNE: Q* = {q_star} (kappa = {out['kappa_vs_default'][q_star]:.2f})")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
