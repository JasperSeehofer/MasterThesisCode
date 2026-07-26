"""z_support sweep: separate the impostor channel from the completion channel.

At fixed injected truth, sweeping the completeness edge moves the completion
fraction from ~0.91 (z_support=0.15) to exactly 0 (z_support >= the detected
population's reach) while the candidate balls get LARGER and MORE
impostor-dominated. If the residual MAP bias tracks the completion fraction and
vanishes at zero completion despite 90%+ impostor balls, the residual is the
B_num completion model, not the estimator's impostor handling.

SMOKE resolution (100 realizations per point). Usage:
    .venv/bin/python results/pp_impostor_harness_20260726/run_zsupport_sweep.py
"""

import dataclasses
import json
from pathlib import Path

from master_thesis_code.validation.pp_coverage import (
    Z_MAX_POP,
    PPCoverageConfig,
    run_coverage,
)

OUT_DIR = Path(__file__).resolve().parent

BASE = PPCoverageConfig(
    n_realizations=100,
    n_events=120,
    injected_truths=[0.72],
    catalogue_mode=True,
    n_galaxies=200_000,
    sky_frac=2.0e-4,
    seed=20260726,
    mixture_mode="generator_marginal",
    z_support=0.30,
)


def main() -> None:
    """Sweep z_support and write the collected results to JSON."""
    collected: dict[str, dict[str, object]] = {}
    for z_support in (0.15, 0.30, 0.60, Z_MAX_POP):
        out = run_coverage(dataclasses.replace(BASE, z_support=z_support))
        r = out["results"]["0.7200"]
        collected[f"{z_support:.2f}"] = r
        print(
            f"z_support={z_support:.2f} completion={r['completion_fraction']:.3f} "
            f"ball={r['mean_ball_size']:.2f} impostor={r['impostor_fraction']:.3f} "
            f"cov68={r['coverage']['68']:.2f} MAP={r['map_mean']:.4f} "
            f"bias={r['map_bias']:+.4f} sd={r['map_std']:.4f}",
            flush=True,
        )
    (OUT_DIR / "zsupport_sweep.json").write_text(
        json.dumps({"config": dataclasses.asdict(BASE), "sweep": collected}, indent=2)
    )


if __name__ == "__main__":
    main()
