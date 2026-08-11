"""G3 soundness-gate driver: normalization ablation cube on the seed600 subsample.

Runs the production ``BayesianStatistics.evaluate()`` per normalization mode on
the real 494-event seed600 CRB subsample (the de-rail demonstration data),
completing the {host-z kernel} x {L_cat denominator} cube at HEAD (all cells
carry the unconditional 1/(4pi) completion fix, cb16142):

    mode            kernel   denominator   role
    prod_global     bare     GLOBAL        4pi-only baseline (expect low rail ~0.60)
    volume_global   volume   GLOBAL        NEW cell: fix #1 kernel alone
    local_ratio     bare     LOCAL         fix #2 alone (expect peaked ~0.73)
    volume_deconv   volume   LOCAL         fix #1 + #2 (production; expect ~0.73)
    catonly         bare     LOCAL (f=1)   completion-free control (expect ~0.73)

The pre-4pi cells are not reachable at HEAD (the 1/(4pi) fix is unconditional);
their archived posteriors live in results/commission_20260701/redteam/
posteriors_per_mode/prod (MAP 0.86).

Usage (data dir must contain simulations/prepared_cramer_rao_bounds.csv,
simulations/cramer_rao_bounds.csv and simulations/injections/):
    uv run python scripts/ablation_cube_seed600.py \
        --data_dir ~/data-backups/seed600_local_derail_20260702/crux_ws \
        --scratch_dir /tmp/g3_ablation_cube [--workers 24]

Inputs are copied/symlinked into a fresh scratch dir; the source data dir is
never written. Results: .planning/gate/G3_ablation_cube.json (repo-absolute)
+ per-mode posterior dirs under the scratch dir.
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

# (label, catalog_only, normalization_mode)
MODES = [
    ("prod_global", False, "global"),
    ("volume_global", False, "volume_global"),
    ("local_ratio", False, "local_ratio"),
    ("volume_deconv", False, "volume_deconv"),
    ("catonly", True, "local_ratio"),
]


def summarize(combined: dict) -> dict:
    h = list(map(float, combined["h_values"]))
    p = np.array(combined["posterior"], dtype=float)
    s = float(np.nansum(p))
    i = int(np.nanargmax(p))
    return {
        "MAP": h[i],
        "mean": float(np.nansum(np.array(h) * p) / s) if s > 0 else float("nan"),
        "edge_mass": float((p[0] + p[-1]) / s) if s > 0 else float("nan"),
        "railed": bool(i == 0 or i == len(h) - 1),
        "h_values": h,
        "posterior": [float(x) for x in p],
    }


def prepare_scratch(data_dir: Path, scratch: Path) -> None:
    """Fresh scratch workdir with read-only inputs symlinked from data_dir."""
    sim_src = data_dir / "simulations"
    sim_dst = scratch / "simulations"
    sim_dst.mkdir(parents=True, exist_ok=True)
    for name in ("prepared_cramer_rao_bounds.csv", "cramer_rao_bounds.csv", "injections"):
        src = sim_src / name
        dst = sim_dst / name
        if not src.exists():
            raise FileNotFoundError(f"required input missing: {src}")
        if dst.is_symlink() or dst.exists():
            continue
        dst.symlink_to(src.resolve())
    # REDUCED_CATALOGUE_FILE_PATH (handler.py:24) is cwd-relative
    # ("./darksiren_emri/...") — link the repo package dir into the scratch cwd.
    pkg_link = scratch / "darksiren_emri"
    if not (pkg_link.is_symlink() or pkg_link.exists()):
        pkg_link.symlink_to(REPO_ROOT / "darksiren_emri")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--scratch_dir", default="/tmp/g3_ablation_cube")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--output_json", default=str(REPO_ROOT / ".planning/gate/G3_ablation_cube.json")
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    data_dir = Path(os.path.expanduser(args.data_dir)).resolve()
    scratch = Path(args.scratch_dir).resolve()
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    prepare_scratch(data_dir, scratch)
    os.chdir(scratch)

    # Imports AFTER chdir intentionally NOT needed for path resolution (the
    # package resolves via the venv), but worker re-imports must find the same
    # cwd-relative simulations/ inputs, hence evaluate() runs from the scratch dir.
    from darksiren_emri.bayesian_inference.bayesian_statistics import BayesianStatistics
    from darksiren_emri.bayesian_inference.posterior_combination import combine_posteriors
    from darksiren_emri.cosmological_model import Model1CrossCheck
    from darksiren_emri.galaxy_catalogue.handler import GalaxyCatalogueHandler

    t0 = time.time()
    print("loading catalogue + building BallTrees ...", flush=True)
    rng = np.random.default_rng(0)
    model = Model1CrossCheck(rng=rng)
    catalog = GalaxyCatalogueHandler(
        M_min=model.parameter_space.M.lower_limit,
        M_max=model.parameter_space.M.upper_limit,
        z_max=model.max_redshift,
    )
    print(
        f"handler ready in {time.time() - t0:.0f}s; {len(catalog.reduced_galaxy_catalog)} galaxies",
        flush=True,
    )

    results: dict = {}
    if output_json.exists():
        results = json.loads(output_json.read_text())
        print(f"resuming: {list(results)} already done", flush=True)

    for label, catonly, nmode in MODES:
        if label in results:
            continue
        pdir = "simulations/posteriors"
        wdir = "simulations/posteriors_with_bh_mass"
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
                catalog_only=catonly,
                normalization_mode=nmode,
            )
            print(f"[{label}] h={h} done in {time.time() - th:.0f}s", flush=True)
        combine_posteriors(posteriors_dir=pdir, strategy="physics-floor", output_dir=pdir)
        combined = json.load(open(f"{pdir}/combined_posterior.json"))
        results[label] = summarize(combined)
        save = f"simulations/posteriors_{label}"
        if os.path.isdir(save):
            shutil.rmtree(save)
        shutil.move(pdir, save)
        r = results[label]
        print(
            f"=== {label}: MAP={r['MAP']} mean={r['mean']:.4f} "
            f"edge_mass={r['edge_mass']:.3f} railed={r['railed']} ({time.time() - tt:.0f}s) ===",
            flush=True,
        )
        output_json.write_text(json.dumps(results, indent=2))

    print(f"ABLATION CUBE DONE in {time.time() - t0:.0f}s", flush=True)
    for label, _, _ in MODES:
        if label in results:
            r = results[label]
            print(
                f"  {label:14s} MAP={r['MAP']} mean={r['mean']:.4f} "
                f"edge_mass={r['edge_mass']:.3f} railed={r['railed']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
