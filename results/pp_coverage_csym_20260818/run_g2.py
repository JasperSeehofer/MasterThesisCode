"""Driver for the registered G-2 cells (PREREGISTRATION_G2_SPECZ_LIMIT.md
Sec 3/3b), as superseded by PRE-FREEZE AMENDMENT A + A-PF-4 (verifier
Part IV): all 9 cells run on the WIDE estimator grid, and Q* must itself
have been certified on that same wide grid (a narrow-grid Q* is void for
scoring -- see ``pretune_g2.py``).

Grid: {off, 1d, cat1d} x sigma_z in {0.035, 0.010, 0.002} = 9 cells, ALL
sharing ONE master seed 20280311 (the generative stream is sigma_z-continuous
across rungs, keeping cross-rung latent draws aligned), ALL on h in
[0.56, 0.92] (the amendment's headroom grid -- the same one the cat1d rail
at h_true=0.84 required for G-1). n_z_quad = Q* is FIXED across every cell
(the single instrument for the whole grid) -- pass it via ``--n-z-quad``
(no default: Q* must come from ``pretune_g2.py``'s registered, WIDE-GRID
fill-in, ``preflight/pretune_g2.json``, read automatically if present and
not overridden; this script refuses a narrow-grid pretune file).

Venue: V-deep exactly as G-1 Sec 3 (z_support=0.40, sky_frac=1e-4, default
d50, n_galaxies=2e5, alpha_M=0.25, catalogue_mode, production noise).
All cells: n_events=250, R=120, truths {0.62, 0.72, 0.84}, h_step=0.004.

Rail-fraction validity gate (amendment, mirrors G-1 item 6): any cell x
truth with 1D rail_fraction > 0.10 on the wide grid is UNDETERMINED-BY-RAIL
for that truth's reads (readout_g2.py flags it). At PROBE scale (R=4) a
rail_fraction == 1.0 is still a hard preflight STOP.

Two modes:
  --preflight   Sec 3b anti-void gate: R=4 probes at the EXACT registered
                (amended, wide-grid) configuration for all 9 cells, written
                to preflight/. Checks completion_fraction/catalogue-bearing
                -fraction bounds, pair engagement at rung 1
                (Pc(0.035)/Pd(0.035)), finite MAPs, rail_fraction < 1.0,
                and the S_bar_phi table support.
  (default)     Runs the nine registered R=120 cells into cells/, skipping
                any that already exist.

Usage:
    python run_g2.py --preflight [--n-z-quad Q]
    python run_g2.py [--n-z-quad Q]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from darksiren_emri.validation.pp_coverage import D50_GPC, PPCoverageConfig, run_coverage

HERE = Path(__file__).parent
TRUTHS = [0.62, 0.72, 0.84]
NOISE_KW = {"gw_measurement_scatter": False, "sigma_dl_model_in_likelihood": False}
MASTER_SEED = 20280311
RUNGS = (0.035, 0.010, 0.002)
CELLS = ("off", "1d", "cat1d")
# PRE-FREEZE AMENDMENT A: all 9 cells on the wide grid.
WIDE_H_MIN = 0.56
WIDE_H_MAX = 0.92


def _resolve_n_z_quad(cli_value: int | None) -> int:
    if cli_value is not None:
        return cli_value
    pretune_path = HERE / "preflight" / "pretune_g2.json"
    if pretune_path.exists():
        data = json.loads(pretune_path.read_text())
        q_star = data.get("q_star")
        grid = data.get("grid") or {}
        # A-PF-4 (BLOCKING): a Q* certified on the narrow grid is void for
        # scoring on the wide grid -- refuse it rather than silently using it.
        if grid.get("h_min") != WIDE_H_MIN or grid.get("h_max") != WIDE_H_MAX:
            raise SystemExit(
                f"pretune_g2.json's Q* was certified on grid {grid} "
                f"(expected wide grid h_min={WIDE_H_MIN}, h_max={WIDE_H_MAX} per A-PF-4): "
                "VOID for scoring. Re-run pretune_g2.py (it now sweeps the wide grid by "
                "default) or pass --n-z-quad explicitly."
            )
        if q_star is not None:
            print(f"n_z_quad = Q* = {q_star} (read from {pretune_path}, wide-grid-certified)")
            return int(q_star)
    raise SystemExit(
        "n_z_quad (Q*) not specified and preflight/pretune_g2.json has no q_star: "
        "run pretune_g2.py first (Sec 7 registered fill-in, wide grid per A-PF-4), "
        "or pass --n-z-quad explicitly."
    )


def cell_id(rung: float, cell: str) -> str:
    return f"rung_{rung:.3f}_{cell}"


def _cfg(rung: float, sc: str, n_z_quad: int, n_realizations: int) -> PPCoverageConfig:
    return PPCoverageConfig(
        n_realizations=n_realizations,
        n_events=250,
        injected_truths=TRUTHS,
        seed=MASTER_SEED,
        kernel="volume",
        catalogue_mode=True,
        mixture_mode="absolute",
        z_support=0.40,
        sky_frac=1e-4,
        d50_gpc=D50_GPC,
        n_galaxies=200_000,
        mass_channel=True,
        mass_horizon_index=0.25,
        selection_cell=sc,  # type: ignore[arg-type]
        h_step=0.004,
        h_min=WIDE_H_MIN,
        h_max=WIDE_H_MAX,
        sigma_z=rung,
        n_z_quad=n_z_quad,
        **NOISE_KW,
    )


def _run_one(cell_id_: str, cfg: PPCoverageConfig, out_dir: Path) -> str:
    out = out_dir / f"{cell_id_}.json"
    if out.exists():
        return f"{cell_id_}: SKIP (exists)"
    t0 = time.perf_counter()
    res = run_coverage(cfg)
    out.write_text(json.dumps(res))
    return f"{cell_id_}: done in {time.perf_counter() - t0:.0f}s"


def preflight(n_z_quad: int) -> bool:
    out_dir = HERE / "preflight"
    out_dir.mkdir(exist_ok=True)
    problems: list[str] = []

    probe_paths: dict[str, Path] = {}
    for rung in RUNGS:
        for sc in CELLS:
            cid = cell_id(rung, sc)
            cfg = _cfg(rung, sc, n_z_quad, n_realizations=4)
            msg = _run_one(cid, cfg, out_dir)
            print(f"[probe] {msg}", flush=True)
            probe_paths[cid] = out_dir / f"{cid}.json"

    # Check 1: completion_fraction / catalogue-bearing fraction bounds.
    for rung in RUNGS:
        for sc in CELLS:
            cid = cell_id(rung, sc)
            data = json.loads(probe_paths[cid].read_text())
            for truth, block in data["results"].items():
                cf = block["completion_fraction"]
                cat_bearing = 1.0 - cf
                if not (0.05 <= cf <= 0.95):
                    problems.append(f"{cid}@{truth}: completion_fraction={cf:.3f} outside [0.05,0.95]")
                if cat_bearing <= 0.3:
                    problems.append(f"{cid}@{truth}: catalogue-bearing fraction={cat_bearing:.3f} <= 0.3")

    # Check 3: finite MAPs, and the amended rail gate. rail_fraction == 1.0
    # at probe scale is still a hard STOP (the wide grid did not, in fact,
    # give this cell/truth headroom); 0.10 < rail_fraction < 1.0 is reported
    # as informational (UNDETERMINED-BY-RAIL at scored scale is
    # readout_g2.py's concern, not a probe-engagement criterion).
    for rung in RUNGS:
        for sc in CELLS:
            cid = cell_id(rung, sc)
            data = json.loads(probe_paths[cid].read_text())
            for truth, block in data["results"].items():
                if not (block["map_mean"] == block["map_mean"]):
                    problems.append(f"{cid}@{truth}: map_mean is NaN")
                if block["rail_fraction"] >= 1.0:
                    problems.append(
                        f"{cid}@{truth}: rail_fraction=1.0 (every probe MAP railed on the "
                        f"registered wide grid -- amendment headroom did not clear this cell/truth)"
                    )
                elif block["rail_fraction"] > 0.10:
                    print(
                        f"  ~ {cid}@{truth}: probe rail_fraction={block['rail_fraction']:.2f} "
                        f"> 0.10 (UNDETERMINED-BY-RAIL at scored scale, not a probe STOP)"
                    )
                for m in block["maps"]:
                    if not (m == m):
                        problems.append(f"{cid}@{truth}: a probe MAP is NaN")

    # Check 2: engagement at rung 1 (Sec 1 N-a) -- Pc(0.035), Pd(0.035).
    import readout_g2 as rg2

    rung1 = RUNGS[0]
    for name, cid_a, cid_b in ((f"Pc({rung1:.3f})", cell_id(rung1, "cat1d"), cell_id(rung1, "off")),
                                (f"Pd({rung1:.3f})", cell_id(rung1, "1d"), cell_id(rung1, "off"))):
        pa, pb = probe_paths[cid_a], probe_paths[cid_b]
        entry = rg2.paired_read(pa, pb)
        chans = [t.get("channel_1d") for t in entry["truths"].values()]
        chans = [c for c in chans if c is not None]
        if not chans:
            problems.append(f"{name}: channel_1d probe delta not computable at any truth")
        elif all(c["degenerate"] for c in chans):
            problems.append(f"{name}: probe delta IDENTICALLY ZERO at every truth (degenerate, N-a)")

    # Check 4: S_bar_phi table support (structural, same as G-1).
    from darksiren_emri.validation.pp_coverage import Z_MAX_POP

    if 0.40 >= Z_MAX_POP:
        problems.append(f"z_support=0.40 >= Z_MAX_POP={Z_MAX_POP}: no support margin")

    print()
    if problems:
        print("PREFLIGHT: STOP")
        for p in problems:
            print(f"  - {p}")
        return False
    print("PREFLIGHT: READY")
    return True


def run_scored(n_z_quad: int) -> None:
    out_dir = HERE / "cells"
    out_dir.mkdir(exist_ok=True)
    for rung in RUNGS:
        for sc in CELLS:
            cid = cell_id(rung, sc)
            cfg = _cfg(rung, sc, n_z_quad, n_realizations=120)
            msg = _run_one(cid, cfg, out_dir)
            print(msg, flush=True)
    print("G2 COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--n-z-quad", type=int, default=None)
    args = parser.parse_args()
    n_z_quad = _resolve_n_z_quad(args.n_z_quad)
    if args.preflight:
        ready = preflight(n_z_quad)
        raise SystemExit(0 if ready else 1)
    run_scored(n_z_quad)


if __name__ == "__main__":
    main()
