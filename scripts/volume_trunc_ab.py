"""Decisive empirical gate for the shallow-venue `volume_trunc` kernel (Part 1).

Runs the production evaluation on the archived seed600 shallow venue (494-event
subsample; hosts at z < 0.12, injection pool z_max = 0.5) twice — once with the
golden ``volume_deconv`` kernel (baseline) and once with ``volume_trunc`` — and
compares the combined 1D and 2D H0 posteriors. ``volume_trunc`` integrates the
in-catalogue numerator over each host's galaxy window (shared with Z_g / D_g,
z-floor at 0) instead of the event-level GW window.

The seed600 shallow venue is the venue-matched harness rung where the calibrated
volume kernel over-corrects (+0.0132 in the 1D mean; ledger [L8]). The A/B asks a
question that cannot be derived: does unifying the numerator support move the 1D
mean from ~0.745 toward the truth 0.73? If it does not, the numerator window is
NOT the +0.013 lever — a real finding to report, not to force.

Uses the SAME 7-point grid as the N-5 / Eddington-in-M driver so the baseline arm
reproduces the established seed600 subsample means (1D ~0.745, 2D ~0.768).

Usage:
    uv run python scripts/volume_trunc_ab.py \
        --crb_dir ~/data-backups/seed600_local_derail_20260702/crux_ws \
        --injections_dir ~/data-backups/seed600_local_derail_20260702/simulations/injections \
        --scratch_dir /tmp/volume_trunc_ab [--workers 24]

Writes .planning/gate/volume_trunc_ab.json.
"""

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
GRID = [0.60, 0.65, 0.70, 0.73, 0.76, 0.80, 0.86]


def summarize(combined: dict) -> dict:
    h = list(map(float, combined["h_values"]))
    p = np.array(combined["posterior"], dtype=float)
    s = float(np.nansum(p))
    i = int(np.nanargmax(p))
    return {
        "MAP": h[i],
        "mean": float(np.nansum(np.array(h) * p) / s) if s > 0 else float("nan"),
        "edge_mass": float((p[0] + p[-1]) / s) if s > 0 else float("nan"),
        "h_values": h,
        "posterior": [float(x) for x in p],
    }


def prepare_scratch(crb_dir: Path, injections_dir: Path, scratch: Path) -> None:
    """Symlink the CRBs (from crux_ws) + the REAL injection pool into scratch.

    The crux_ws ``injections`` symlink is dead (points at /tmp), so the pool is
    supplied separately via injections_dir.
    """
    sim_dst = scratch / "simulations"
    sim_dst.mkdir(parents=True, exist_ok=True)
    crb_src = crb_dir / "simulations"
    for name in ("prepared_cramer_rao_bounds.csv", "cramer_rao_bounds.csv"):
        src, dst = crb_src / name, sim_dst / name
        if not src.exists():
            raise FileNotFoundError(f"required CRB input missing: {src}")
        if not (dst.is_symlink() or dst.exists()):
            dst.symlink_to(src.resolve())
    inj_dst = sim_dst / "injections"
    if not injections_dir.is_dir():
        raise FileNotFoundError(f"real injection pool missing: {injections_dir}")
    if not (inj_dst.is_symlink() or inj_dst.exists()):
        inj_dst.symlink_to(injections_dir.resolve())
    pkg_link = scratch / "master_thesis_code"
    if not (pkg_link.is_symlink() or pkg_link.exists()):
        pkg_link.symlink_to(REPO_ROOT / "master_thesis_code")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crb_dir", default="~/data-backups/seed600_local_derail_20260702/crux_ws")
    parser.add_argument(
        "--injections_dir",
        default="~/data-backups/seed600_local_derail_20260702/simulations/injections",
    )
    parser.add_argument("--scratch_dir", default="/tmp/volume_trunc_ab")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--output_json", default=str(REPO_ROOT / ".planning/gate/volume_trunc_ab.json")
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    crb_dir = Path(os.path.expanduser(args.crb_dir)).resolve()
    injections_dir = Path(os.path.expanduser(args.injections_dir)).resolve()
    scratch = Path(args.scratch_dir).resolve()
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    prepare_scratch(crb_dir, injections_dir, scratch)
    os.chdir(scratch)

    from master_thesis_code.bayesian_inference.bayesian_statistics import BayesianStatistics
    from master_thesis_code.bayesian_inference.posterior_combination import combine_posteriors
    from master_thesis_code.cosmological_model import Model1CrossCheck
    from master_thesis_code.galaxy_catalogue.handler import GalaxyCatalogueHandler

    t0 = time.time()
    rng = np.random.default_rng(0)
    model = Model1CrossCheck(rng=rng)
    print("loading catalogue ...", flush=True)
    catalog = GalaxyCatalogueHandler(
        M_min=model.parameter_space.M.lower_limit,
        M_max=model.parameter_space.M.upper_limit,
        z_max=model.max_redshift,
    )
    print(f"handler ready in {time.time() - t0:.0f}s", flush=True)

    results: dict = {}
    if output_json.exists():
        results = json.loads(output_json.read_text())
        print(f"resuming: {list(results)} already done", flush=True)

    # variant label -> normalization_mode passed to evaluate()
    variants = {"volume_deconv": "volume_deconv", "volume_trunc": "volume_trunc"}
    pdir, wdir = "simulations/posteriors", "simulations/posteriors_with_bh_mass"
    for variant, nmode in variants.items():
        if variant in results:
            continue
        for d in (pdir, wdir):
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        tt = time.time()
        for h in GRID:
            th = time.time()
            BayesianStatistics().evaluate(
                catalog,
                model,
                float(h),
                num_workers=args.workers,
                normalization_mode=nmode,
                # Archived shallow venue: the pool covers the shallow host-draw
                # volume but is shallower than the campaign expected z_max, so the
                # coverage guard must be relaxed (same precedent as the N-5 /
                # Eddington-in-M driver; results/seed600_ab_20260710/ANALYSIS.md).
                allow_low_pdet_coverage=True,
            )
            print(f"[{variant}] h={h} done in {time.time() - th:.0f}s", flush=True)
        entry: dict = {}
        for label, d in (("1d", pdir), ("2d", wdir)):
            combine_posteriors(
                posteriors_dir=d,
                strategy="physics-floor",
                output_dir=d,
                allow_shallow_pool=True,
            )
            entry[label] = summarize(json.load(open(f"{d}/combined_posterior.json")))
        results[variant] = entry
        print(
            f"=== {variant}: 1D MAP={entry['1d']['MAP']} mean={entry['1d']['mean']:.4f} "
            f"edge={entry['1d']['edge_mass']:.3f} | 2D MAP={entry['2d']['MAP']} "
            f"mean={entry['2d']['mean']:.4f} edge={entry['2d']['edge_mass']:.3f} "
            f"({time.time() - tt:.0f}s) ===",
            flush=True,
        )
        output_json.write_text(json.dumps(results, indent=2))

    b, t = results["volume_deconv"], results["volume_trunc"]
    delta = {
        "d_MAP_1d": t["1d"]["MAP"] - b["1d"]["MAP"],
        "d_mean_1d": t["1d"]["mean"] - b["1d"]["mean"],
        "d_MAP_2d": t["2d"]["MAP"] - b["2d"]["MAP"],
        "d_mean_2d": t["2d"]["mean"] - b["2d"]["mean"],
    }
    results["delta"] = delta
    output_json.write_text(json.dumps(results, indent=2))
    print(
        f"VOLUME_TRUNC A/B DONE in {time.time() - t0:.0f}s\n"
        f"  baseline (volume_deconv): 1D mean={b['1d']['mean']:.4f} 2D mean={b['2d']['mean']:.4f}\n"
        f"  volume_trunc:             1D mean={t['1d']['mean']:.4f} 2D mean={t['2d']['mean']:.4f}\n"
        f"  delta (trunc - deconv, truth h=0.73): {delta}",
        flush=True,
    )


if __name__ == "__main__":
    main()
