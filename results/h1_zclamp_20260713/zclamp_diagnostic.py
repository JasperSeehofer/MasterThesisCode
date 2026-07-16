"""H1 clamp-isolation: is the shallow-venue high bias the z>=0 photo-z clamp?

The pp_coverage harness generates the observed photo-z as
``z_gal = clip(z_host + N(0, sigma_z), Z_MIN, None)`` and infers with the naive
Gaussian kernel ``N(z; z_gal, sigma_z) * w_pop`` (which does NOT model the clamp).
This toggles the generative clamp (config.clamp_zgal) at the shallow seed600-matched
venue (d50=0.23, z_med~0.044, sigma_z=0.035) and a deep control (d50=1.85), volume
kernel, to isolate whether the clamp is the mechanism.

Result (2026-07-13, multi-seed n_real=200, n_events=250):
  deep    : bias -0.0024 both ways (clamp-independent control).
  shallow : clamp ON  bias +0.0240 +/- 0.0022, cov68 ~0.61   <- the [L8] +0.030
            clamp OFF bias -0.0056 +/- 0.0020, cov68 ~0.68   <- vanishes, cov recovers
=> the shallow high bias is the boundary clamp on the OBSERVED photo-z, not the
volume/Eddington correction per se. Fix = model the censored measurement (or use
raw photo-z). Production relevance hinges on whether real low-z photo-z are
clamped/piled near 0 (catalogue min -0.0003, 17/500k <= 0: NOT hard-clamped).

Run: uv run python results/h1_zclamp_20260713/zclamp_diagnostic.py
"""

import numpy as np

from master_thesis_code.validation.pp_coverage import PPCoverageConfig, run_coverage


def _bias(d50: float, clamp: bool, seed: int, sigma_z: float = 0.035) -> tuple[float, float]:
    cfg = PPCoverageConfig(
        injected_truths=[0.73],
        n_realizations=200,
        n_events=250,
        kernel="volume",
        sigma_z=sigma_z,
        d50_gpc=d50,
        w_pdet_gpc=0.162 * d50,
        clamp_zgal=clamp,
        seed=seed,
    )
    r = run_coverage(cfg)["results"]["0.7300"]
    return r["map_bias"], r["coverage"]["68"]


def main() -> None:
    seeds = (20260701, 11, 22, 33)
    for label, d50 in (
        ("DEEP  (d50=1.85, z_med~0.28)", 1.85),
        ("SHALLOW (d50=0.23, z_med~0.044)", 0.23),
    ):
        print(f"=== {label} ===")
        for clamp in (True, False):
            bs = np.array([_bias(d50, clamp, s)[0] for s in seeds])
            cs = np.mean([_bias(d50, clamp, s)[1] for s in seeds])
            print(
                f"  clamp={'ON ' if clamp else 'OFF'}: bias mean={bs.mean():+.4f} "
                f"std={bs.std():.4f}  cov68~{cs:.2f}"
            )


if __name__ == "__main__":
    main()
