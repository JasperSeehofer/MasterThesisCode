"""Gate B, credibility test for the C5(a) edge-artifact surrogate.

attack_c5_rail.py extrapolates each railed in-cat event's peak by fitting a
parabola to the top-K grid points and taking the vertex, finding median implied
peaks h_eff = 0.93-1.05.  That is an extrapolation 0.07-0.19 beyond the grid
edge, so before the number is trusted the extrapolator must be VALIDATED on
events whose peak is known.

Validation design (same data, no new assumptions):
  * take realistic in-cat events whose 1D argmax is INTERIOR, at grid index j;
  * stand off by m grid steps (m = 8, 13, 19 -> 0.08, 0.13, 0.19 in h, bracketing
    the real extrapolation distance) and fit the same top-K parabola to the K
    points ENDING at index j - m;
  * record the recovered vertex minus the true peak.
Same for the idealized runs (sharp peaks at 0.73) as a hard case.

Also reported:
  * smoothness of the railed profiles: sign consistency of the 2nd difference
    over the top 10 grid points (is the -6e-4 curvature signal or roundoff?);
  * implied per-event Gaussian width from the peak height above h=0.73,
    sigma_i = |h_peak - 0.73| / sqrt(2 * dlnp), and the displacement in units of
    that width -- the quantitative form of "is the per-event rail cosmetic?".

Read-only w.r.t. master_thesis_code/.  Run from the repo root with .venv/bin/python.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
REAL = HERE.parent
CAMPAIGN = REAL.parent
SEEDS = (61000, 62000)
REALIZATIONS = (1, 2, 3, 4, 5)
IDEALIZED = {
    61000: CAMPAIGN / "run_seed61000" / "posteriors_fixed",
    62000: CAMPAIGN / "run_seed62000" / "posteriors",
}


def h_of_tag(t: str) -> float:
    return float("0." + t)


def load_grid(post_dir: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    tags = sorted((p.stem.removeprefix("h_0_") for p in post_dir.glob("h_0_*.json")), key=h_of_tag)
    h = np.array([h_of_tag(t) for t in tags])
    pe: dict[int, list[float]] = {}
    for t in tags:
        with open(post_dir / f"h_0_{t}.json") as f:
            j = json.load(f)
        for k, v in j.items():
            if k.isdigit():
                pe.setdefault(int(k), []).append(v[0] if isinstance(v, list) else v)
    return h, {k: np.asarray(v) for k, v in pe.items() if len(v) == len(h)}


def incat_index(seed: int) -> set[int]:
    df = pd.read_csv(REAL / f"seed{seed}" / "prepared_cramer_rao_bounds.csv")
    return set(df.index[df.host_galaxy_index >= 0])


def vertex(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    c = np.polyfit(x, y, 2)
    if c[0] == 0:
        return float("inf"), 0.0
    return float(-c[1] / (2 * c[0])), float(c[0])


def main() -> None:
    val_rows, sm_rows, wid_rows = [], [], []
    for seed in SEEDS:
        incat = incat_index(seed)
        srcs = [("IDEAL", IDEALIZED[seed])] + [
            (f"real_r{r}", REAL / f"seed{seed}" / f"real_r{r}" / "posteriors") for r in REALIZATIONS
        ]
        for tag, d in srcs:
            name = f"seed{seed} {tag}"
            h, pe = load_grid(d)
            iedge = len(h) - 1
            lnp = {k: np.log(v) for k, v in pe.items() if k in incat and np.all(v > 0)}

            # ---- (A) extrapolator validation on interior-peak events -------------
            for m in (8, 13, 19):
                for K in (3, 5, 9):
                    errs = []
                    for k, y in lnp.items():
                        j = int(np.argmax(y))
                        if j <= 0 or j >= iedge:
                            continue  # only interior peaks have a known answer
                        end = j - m
                        if end - K + 1 < 0:
                            continue
                        sl = slice(end - K + 1, end + 1)
                        hp, a2 = vertex(h[sl], y[sl])
                        if a2 < 0 and np.isfinite(hp):
                            errs.append(hp - h[j])
                    if errs:
                        e = np.asarray(errs)
                        val_rows.append(
                            {
                                "name": name,
                                "standoff_m": m,
                                "K": K,
                                "n": len(e),
                                "median_err": float(np.median(e)),
                                "iqr_err": float(np.percentile(e, 75) - np.percentile(e, 25)),
                                "frac_within_0p03": float(np.mean(np.abs(e) < 0.03)),
                                "frac_overshoot": float(np.mean(e > 0)),
                            }
                        )

            # ---- (B) smoothness of railed profiles at the top of the grid --------
            # NOTE: the h grid is NON-UNIFORM (0.005 in [0.65,0.80], 0.01 in the
            # wings), so a plain 2nd difference is only meaningful on the top 7
            # points 0.80..0.86, which are uniformly spaced at 0.01.
            top = np.arange(len(h) - 7, len(h))
            assert np.allclose(np.diff(h[top]), 0.01), np.diff(h[top])
            railed = [k for k, y in lnp.items() if int(np.argmax(y)) == iedge]
            if railed:
                d2s = np.array([np.diff(lnp[k][top], 2) for k in railed])  # (n, 5)
                sign_consistent = np.mean(np.all(d2s < 0, axis=1))
                sm_rows.append(
                    {
                        "name": name,
                        "n_railed": len(railed),
                        "frac_all5_d2_negative": float(sign_consistent),
                        "median_min_abs_d2": float(np.median(np.abs(d2s).min(axis=1))),
                        "median_max_abs_d2": float(np.median(np.abs(d2s).max(axis=1))),
                        "d2_dynamic_range": float(
                            np.median(
                                np.abs(d2s).max(axis=1)
                                / np.maximum(np.abs(d2s).min(axis=1), 1e-300)
                            )
                        ),
                    }
                )

            # ---- (C) implied per-event width and displacement in sigma -----------
            i73 = int(np.argmin(np.abs(h - 0.73)))
            sig, disp = [], []
            for k, y in lnp.items():
                j = int(np.argmax(y))
                dh = h[j] - h[i73]
                dl = y[j] - y[i73]
                if dl > 1e-12 and abs(dh) > 1e-12:
                    s = abs(dh) / np.sqrt(2 * dl)
                    sig.append(s)
                    disp.append(dh / s)
            if sig:
                wid_rows.append(
                    {
                        "name": name,
                        "n": len(sig),
                        "median_sigma_h": float(np.median(sig)),
                        "q25_sigma_h": float(np.percentile(sig, 25)),
                        "q75_sigma_h": float(np.percentile(sig, 75)),
                        "median_disp_sigma": float(np.median(disp)),
                        "q90_disp_sigma": float(np.percentile(disp, 90)),
                        "frac_disp_gt_1sig": float(np.mean(np.asarray(disp) > 1.0)),
                    }
                )

    pd.set_option("display.width", 260, "display.max_columns", 30)
    v = pd.DataFrame(val_rows)
    print("=== (A) EXTRAPOLATOR VALIDATION on interior-peak in-cat events ===")
    print("    fit the same top-K parabola m grid steps BELOW the known peak;")
    print("    median_err = recovered vertex - true peak (in h)\n")
    print(
        v.groupby(["standoff_m", "K"])
        .agg(
            n=("n", "sum"),
            median_err=("median_err", "median"),
            iqr=("iqr_err", "median"),
            frac_within_0p03=("frac_within_0p03", "mean"),
            frac_overshoot=("frac_overshoot", "mean"),
        )
        .to_string(float_format=lambda x: f"{x:.4g}")
    )
    print("\n  per-run (K=5):")
    print(v[v.K == 5].to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print("\n=== (B) smoothness of railed profiles (top 10 grid points) ===")
    print(pd.DataFrame(sm_rows).to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print("\n=== (C) implied per-event width and displacement (in-cat) ===")
    print("    sigma_i = |h_peak - 0.73| / sqrt(2 * [lnp(peak) - lnp(0.73)])")
    print(pd.DataFrame(wid_rows).to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    with open(HERE / "c5_extrap_validation_results.json", "w") as f:
        json.dump({"validation": val_rows, "smoothness": sm_rows, "widths": wid_rows}, f, indent=1)
    print(f"\nwrote {HERE / 'c5_extrap_validation_results.json'}")


if __name__ == "__main__":
    main()
