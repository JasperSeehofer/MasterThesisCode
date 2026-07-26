"""Coverage smoke run for the catalogue / impostor-ball harness mode.

SMOKE ONLY — 200 realizations per (mode, truth) cell is enough to see a
gross coverage/bias failure but NOT enough to certify calibration: the 1-sigma
binomial error on a nominal-68% coverage estimate at n=200 is +-3.3 pp, and the
MAP-bias standard error is map_std/sqrt(200) (~0.002 at the observed spread).
Treat every number below as indicative, not as a gate.

Usage (from the repo root of this worktree):
    .venv/bin/python results/pp_impostor_harness_20260726/run_smoke.py
"""

import dataclasses
import json
import time
from pathlib import Path

from master_thesis_code.validation.pp_coverage import PPCoverageConfig, run_coverage

OUT_DIR = Path(__file__).resolve().parent

BASE = PPCoverageConfig(
    n_realizations=200,
    n_events=120,
    sigma_z=0.035,
    sigma_dl_frac=0.05,
    injected_truths=[0.62, 0.72, 0.84],
    seed=20260726,
    kernel="volume",
    catalogue_mode=True,
    z_support=0.30,
    n_galaxies=200_000,
    sky_frac=2.0e-4,
)


def main() -> None:
    """Run the three catalogue-mode estimators and write one JSON per mode."""
    for mode in ("lcat", "absolute", "generator_marginal"):
        config = dataclasses.replace(BASE, mixture_mode=mode)
        t0 = time.time()
        out = run_coverage(config)
        out["wall_seconds"] = time.time() - t0
        (OUT_DIR / f"smoke_{mode}.json").write_text(json.dumps(out, indent=2))
        for key, r in out["results"].items():
            print(
                f"{mode:20s} h_true={key} cov50={r['coverage']['50']:.3f} "
                f"cov68={r['coverage']['68']:.3f} cov90={r['coverage']['90']:.3f} "
                f"rail={r['rail_fraction']:.3f} MAP={r['map_mean']:.4f} "
                f"bias={r['map_bias']:+.4f} sd={r['map_std']:.4f} "
                f"comp={r['completion_fraction']:.3f} ball={r['mean_ball_size']:.2f} "
                f"imp={r['impostor_fraction']:.3f} hib={r['host_in_ball_fraction']:.3f}"
            )
        print(f"{mode}: {out['wall_seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
