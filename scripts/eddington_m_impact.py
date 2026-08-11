"""G7 row 9: empirical H0-level impact of the Eddington-in-M omission.

The 2-D (with-BH-mass) channel marginalises each host's BH mass with the bare
catalogue Gaussian N(M; M_g, sigma_M), omitting the mass population weight
R_eff(M) (per-MBH EMRI rate, log-log slope ~ -0.43/dex). To leading order the
omission equals shifting every catalogue mass by the Eddington correction

    delta ln M = (sigma_M / M)^2 * alpha_g,   alpha_g = dln R_eff / dln M |_{M_g}

(posterior-mean shift of a linear-M Gaussian likelihood against a local
power-law prior). This driver measures the H0-level effect directly: run the
production volume_deconv evaluation on the 494-event seed600 subsample twice
(baseline vs Eddington-shifted catalogue masses) and compare the combined 1D
and 2D posteriors. If |Delta MAP_2D| << sigma_boot, quoting the caveat is
quantitatively justified; if material, the prior-deconvolution must be
implemented before the campaign.

Usage:
    uv run python scripts/eddington_m_impact.py \
        --data_dir ~/data-backups/seed600_local_derail_20260702/crux_ws \
        --scratch_dir /tmp/g7row9_eddington_m [--workers 24]

Writes .planning/gate/G7row9_eddington_m_impact.json.
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


def prepare_scratch(data_dir: Path, scratch: Path) -> None:
    sim_src = data_dir / "simulations"
    sim_dst = scratch / "simulations"
    sim_dst.mkdir(parents=True, exist_ok=True)
    for name in ("prepared_cramer_rao_bounds.csv", "cramer_rao_bounds.csv", "injections"):
        src, dst = sim_src / name, sim_dst / name
        if not src.exists():
            raise FileNotFoundError(f"required input missing: {src}")
        if not (dst.is_symlink() or dst.exists()):
            dst.symlink_to(src.resolve())
    pkg_link = scratch / "darksiren_emri"
    if not (pkg_link.is_symlink() or pkg_link.exists()):
        pkg_link.symlink_to(REPO_ROOT / "darksiren_emri")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--scratch_dir", default="/tmp/g7row9_eddington_m")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--output_json", default=str(REPO_ROOT / ".planning/gate/G7row9_eddington_m_impact.json")
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    data_dir = Path(os.path.expanduser(args.data_dir)).resolve()
    scratch = Path(args.scratch_dir).resolve()
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    prepare_scratch(data_dir, scratch)
    os.chdir(scratch)

    from darksiren_emri.bayesian_inference.bayesian_statistics import BayesianStatistics
    from darksiren_emri.bayesian_inference.posterior_combination import combine_posteriors
    from darksiren_emri.cosmological_model import Model1CrossCheck
    from darksiren_emri.emri_rate import R_eff_per_mbh
    from darksiren_emri.galaxy_catalogue.handler import (
        GalaxyCatalogueHandler,
        InternalCatalogColumns,
    )

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

    for variant in ("baseline", "eddington_shifted"):
        if variant in results:
            continue
        if variant == "eddington_shifted":
            df = catalog.reduced_galaxy_catalog
            M = df[InternalCatalogColumns.BH_MASS].to_numpy(dtype=np.float64)
            Merr = df[InternalCatalogColumns.BH_MASS_ERROR].to_numpy(dtype=np.float64)
            # local log-log slope of R_eff at each catalogue mass (finite difference)
            eps = 0.01
            alpha = (
                np.log(np.asarray(R_eff_per_mbh(M * (1 + eps)), dtype=np.float64))
                - np.log(np.asarray(R_eff_per_mbh(M * (1 - eps)), dtype=np.float64))
            ) / (2 * eps)
            sigma_rel = np.clip(Merr / np.maximum(M, 1e-30), 0.0, 2.0)
            dlnM = sigma_rel**2 * alpha
            shift = np.exp(dlnM)
            df[InternalCatalogColumns.BH_MASS] = M * shift
            catalog.set_max_relative_errors()
            catalog.setup_galaxy_catalog_balltree()
            catalog.setup_4d_galaxy_catalog_balltree()
            results["shift_stats"] = {
                "median_dlnM": float(np.median(dlnM)),
                "p10_dlnM": float(np.percentile(dlnM, 10)),
                "p90_dlnM": float(np.percentile(dlnM, 90)),
                "median_alpha": float(np.median(alpha)),
                "median_sigma_rel": float(np.median(sigma_rel)),
            }
            print(f"applied Eddington shift: {results['shift_stats']}", flush=True)

        pdir, wdir = "simulations/posteriors", "simulations/posteriors_with_bh_mass"
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
                normalization_mode="volume_deconv",
                # This driver targets the ARCHIVED seed600 shallow venue (494-event
                # subsample; events at z < 0.12, injection pool z_max = 0.5). The
                # SimulationDetectionProbability guard compares the pool against the
                # campaign-depth expected_z_max = 1.35 and would raise on this pool,
                # even though it fully covers the shallow host-draw volume. Same
                # deliberate archived-baseline re-run precedent as the seed600 A/B
                # (--allow_low_pdet_coverage; results/seed600_ab_20260710/ANALYSIS.md).
                allow_low_pdet_coverage=True,
            )
            print(f"[{variant}] h={h} done in {time.time() - th:.0f}s", flush=True)
        entry: dict = {}
        for label, d in (("1d", pdir), ("2d", wdir)):
            combine_posteriors(
                posteriors_dir=d,
                strategy="physics-floor",
                output_dir=d,
                # Same archived-shallow-venue rationale as the evaluate() call above:
                # combine_posteriors rebuilds the D(h) survival grid from the same pool.
                allow_shallow_pool=True,
            )
            entry[label] = summarize(json.load(open(f"{d}/combined_posterior.json")))
        results[variant] = entry
        print(
            f"=== {variant}: 1D MAP={entry['1d']['MAP']} mean={entry['1d']['mean']:.4f} | "
            f"2D MAP={entry['2d']['MAP']} mean={entry['2d']['mean']:.4f} "
            f"({time.time() - tt:.0f}s) ===",
            flush=True,
        )
        output_json.write_text(json.dumps(results, indent=2))

    b, e = results["baseline"], results["eddington_shifted"]
    delta = {
        "d_MAP_1d": e["1d"]["MAP"] - b["1d"]["MAP"],
        "d_mean_1d": e["1d"]["mean"] - b["1d"]["mean"],
        "d_MAP_2d": e["2d"]["MAP"] - b["2d"]["MAP"],
        "d_mean_2d": e["2d"]["mean"] - b["2d"]["mean"],
    }
    results["delta"] = delta
    output_json.write_text(json.dumps(results, indent=2))
    print(f"EDDINGTON-M IMPACT DONE in {time.time() - t0:.0f}s: {delta}", flush=True)


if __name__ == "__main__":
    main()
