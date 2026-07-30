"""Campaign #53 realistic-run scoring — pre-registered predictions P1-P6.

Scores the 2 truth seeds x 5 observation realizations produced 2026-07-29
(jobs 6092512-6092531, `absolute_marginal` x `volume_deconv`, observed
catalogues `realizations_20260729/observed_catalogue_seed90000{1..5}.csv`)
against the predictions registered BEFORE evaluation in
`docs/derivations/realistic_host_observation_model.md` section 8.

Inputs are the cluster copies rsynced to `seed{61000,62000}/real_r{1..5}/`
(canonical originals on $WS; see RUNBOOK_NEXT_SESSION_5.md). The idealized
campaign-#51 posteriors (`../run_seed61000/posteriors_fixed`,
`../run_seed62000/posteriors`) are the reference baseline for P4.

Conventions REUSED verbatim from
`../idealization_audit/audit_information_decomposition.py`:
  - per-event 3-point ln-likelihood curvature at h in {0.725, 0.73, 0.735},
    curv_k = ln(L_k(0.73)/L_k(0.725)) + ln(L_k(0.73)/L_k(0.735)), dh = 0.005
  - implied sigma_h = dh / sqrt(sum_k curv_k)
  - in-catalogue events = `host_galaxy_index >= 0` in the prepared CRB table

Read-only w.r.t. master_thesis_code/. CPU-only. Run from the repo root with
.venv/bin/python; all numbers quoted in REALISTIC_READOUT.md come from here.
"""

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

HERE = Path(__file__).parent
CAMPAIGN = HERE.parent
H_TRUE = 0.73
DH = 0.005
SEEDS = (61000, 62000)
REALIZATIONS = (1, 2, 3, 4, 5)

# Idealized campaign-#51 baselines (P4 reference). seed 61000's canonical
# posterior directory is `posteriors_fixed`, seed 62000's is `posteriors` —
# the plain `posteriors/` dir of seed 61000 is the PRE-ec09ed0 backup.
IDEALIZED = {
    61000: CAMPAIGN / "run_seed61000" / "posteriors_fixed",
    62000: CAMPAIGN / "run_seed62000" / "posteriors",
}


def load_per_event(post_dir: Path, tag: str) -> dict[int, float]:
    """Per-event likelihood at one h-value, keyed by event index."""
    with open(post_dir / f"h_0_{tag}.json") as f:
        j = json.load(f)
    return {int(k): (j[k][0] if isinstance(j[k], list) else j[k]) for k in j if k.isdigit()}


def curvature(post_dir: Path) -> dict[int, float]:
    """Per-event 3-point ln-likelihood curvature about h = 0.73."""
    a, b, c = (load_per_event(post_dir, t) for t in ("725", "73", "735"))
    common = [k for k in b if k in a and k in c and a[k] > 0 and b[k] > 0 and c[k] > 0]
    return {k: float(np.log(b[k] / a[k]) + np.log(b[k] / c[k])) for k in common}


def posterior_moments(path: Path) -> dict[str, float]:
    """MAP, mean, sigma and the equal-tailed 68% interval of a combined posterior.

    The h-grid is non-uniform (0.01 at the wings, 0.005 in [0.655, 0.79]), so
    every integral is a trapezoid over the actual node spacing.
    """
    with open(path) as f:
        d = json.load(f)
    h = np.asarray(d["h_values"], dtype=np.float64)
    p = np.asarray(d["posterior"], dtype=np.float64)
    order = np.argsort(h)
    h, p = h[order], p[order]
    p = np.where(np.isfinite(p), p, 0.0)
    norm = float(np.trapezoid(p, h))
    if norm <= 0.0:
        raise ValueError(f"non-normalizable posterior: {path}")
    p = p / norm
    mean = float(np.trapezoid(p * h, h))
    var = float(np.trapezoid(p * (h - mean) ** 2, h))
    cdf = np.concatenate([[0.0], np.cumsum(np.diff(h) * 0.5 * (p[1:] + p[:-1]))])
    lo, hi = (float(np.interp(q, cdf, h)) for q in (0.16, 0.84))
    return {
        "map_h": float(d["map_h"]),
        "mean_h": mean,
        "sigma_h": float(np.sqrt(var)),
        "q16": lo,
        "q84": hi,
        "n_events_used": float(d["n_events_used"]),
        "edge_mass": float(max(p[0], p[-1]) / p.max()),
    }


def golden_events(seed: int, n: int = 3) -> list[int]:
    """The n loudest information carriers, ranked on the IDEALIZED baseline."""
    curv = curvature(IDEALIZED[seed])
    return [k for k, _ in sorted(curv.items(), key=lambda kv: -kv[1])[:n]]


def main() -> None:
    rows: list[dict[str, object]] = []
    incat_flags: dict[int, npt.NDArray[np.bool_]] = {}

    for seed in SEEDS:
        golden = golden_events(seed)
        base_curv = curvature(IDEALIZED[seed])
        base_tot = sum(base_curv.values())
        base_golden = sum(base_curv[k] for k in golden) / base_tot
        print(f"\n=== seed {seed} ===")
        print(
            f"idealized baseline: total curvature {base_tot:.1f}, "
            f"sigma_h {DH / np.sqrt(base_tot):.2e}, "
            f"golden events {golden} carry {100 * base_golden:.0f}%"
        )

        for r in REALIZATIONS:
            run = HERE / f"seed{seed}" / f"real_r{r}"
            mom = posterior_moments(run / "posteriors" / "combined_posterior.json")
            mom2d = posterior_moments(run / "combined_posterior_2d.json")

            # The prepared CRB table is identical across a seed's realizations
            # (only the observed catalogue varies), so only one copy per seed is
            # tracked; the per-realization copies are gitignored bulk mirrors.
            prepared = run / "prepared_cramer_rao_bounds.csv"
            if not prepared.exists():
                prepared = run.parent / "prepared_cramer_rao_bounds.csv"
            df = pd.read_csv(prepared)
            incat = set(df.index[df.host_galaxy_index >= 0])
            incat_flags[seed] = np.asarray(df.host_galaxy_index >= 0)

            curv = curvature(run / "posteriors")
            tot = sum(curv.values())
            ci = sum(v for k, v in curv.items() if k in incat)
            di = tot - ci
            g_share = sum(curv.get(k, 0.0) for k in golden) / tot if tot else float("nan")
            # Conditioning of the split: with the realistic kernel the total is
            # ~1e-1 while the per-event terms are O(1e-3) of either sign, so a
            # ratio-to-total is dominated by cancellation. Track the absolute
            # mass too — that is the well-conditioned statement.
            abs_mass = sum(abs(v) for v in curv.values())
            abs_golden = sum(abs(curv.get(k, 0.0)) for k in golden)
            base_abs_golden = sum(abs(base_curv[k]) for k in golden)

            rows.append(
                {
                    "seed": seed,
                    "real": r,
                    "map_h": mom["map_h"],
                    "mean_h": mom["mean_h"],
                    "sigma_h": mom["sigma_h"],
                    "sigma_H0": 100 * mom["sigma_h"],
                    "pull": (mom["map_h"] - H_TRUE) / mom["sigma_h"],
                    "pull_mean": (mom["mean_h"] - H_TRUE) / mom["sigma_h"],
                    "q16": mom["q16"],
                    "q84": mom["q84"],
                    "edge": mom["edge_mass"],
                    "map_h_2d": mom2d["map_h"],
                    "sigma_H0_2d": 100 * mom2d["sigma_h"],
                    "n_events": int(mom["n_events_used"]),
                    "curv_total": tot,
                    "sigma_h_curv": DH / np.sqrt(tot) if tot > 0 else float("nan"),
                    "dark_share": di / tot if tot else float("nan"),
                    "incat_curv": ci,
                    "dark_curv": di,
                    "abs_mass": abs_mass,
                    "golden_share": g_share,
                    "golden_baseline": base_golden,
                    "golden_curv_abs": abs_golden,
                    "golden_curv_abs_base": base_abs_golden,
                    "golden_retained": abs_golden / base_abs_golden,
                    "pull_2d": (mom2d["map_h"] - H_TRUE) / mom2d["sigma_h"],
                    "edge_2d": mom2d["edge_mass"],
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(HERE / "realistic_scores.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("\n=== per-run summary (1D channel) ===")
    print(
        out[
            [
                "seed",
                "real",
                "map_h",
                "mean_h",
                "sigma_H0",
                "pull",
                "q16",
                "q84",
                "map_h_2d",
                "sigma_H0_2d",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )

    print("\n=== P1  sigma_H0 (expect 1.3-1.7; per-seed range [0.5, 4.0]; < 0.3 FALSIFIES) ===")
    for seed in SEEDS:
        s = out[out.seed == seed]["sigma_H0"]
        print(f"  seed {seed}: {s.min():.3f} - {s.max():.3f} km/s/Mpc (mean {s.mean():.3f})")
    print(f"  all runs: min {out.sigma_H0.min():.3f}, max {out.sigma_H0.max():.3f}")
    print(
        f"  VERDICT falsification (<0.3): "
        f"{'FALSIFIED' if (out.sigma_H0 < 0.3).any() else 'not triggered'}"
    )
    print(
        f"  VERDICT expectation band 1.3-1.7: "
        f"{int(out.sigma_H0.between(1.3, 1.7).sum())}/10 runs inside"
    )

    print("\n=== P2  pulls (|pull| > 2 in >= 6 of 10 FALSIFIES) ===")
    print(
        f"  pull(MAP):  mean {out.pull.mean():+.2f}, sd {out.pull.std(ddof=1):.2f}, "
        f"|pull|>2 in {int((out.pull.abs() > 2).sum())}/10"
    )
    print(
        f"  pull(mean): mean {out.pull_mean.mean():+.2f}, "
        f"sd {out.pull_mean.std(ddof=1):.2f}, "
        f"|pull|>2 in {int((out.pull_mean.abs() > 2).sum())}/10"
    )
    print(f"  VERDICT: {'FALSIFIED' if (out.pull.abs() > 2).sum() >= 6 else 'PASS'}")

    print("\n=== P3  dark-event curvature share (expect [-5%, +5%]) ===")
    for _, row in out.iterrows():
        print(
            f"  seed {int(row.seed)} r{int(row.real)}: dark {100 * row.dark_share:+.1f}% "
            f"(in-cat {row.incat_curv:+.4f} + dark {row.dark_curv:+.4f} = "
            f"{row.curv_total:.4f}; |mass| {row.abs_mass:.3f}; "
            f"sigma_h_curv {row.sigma_h_curv:.2e})"
        )
    inside = out.dark_share.between(-0.05, 0.05).sum()
    print(
        f"  VERDICT: {int(inside)}/10 runs inside [-5%, +5%] -> "
        f"{'PASS' if inside == 10 else 'PREDICTION MISSED'}"
    )
    print(
        f"  conditioning: signed total is {out.curv_total.mean():.3f} vs absolute "
        f"mass {out.abs_mass.mean():.3f} "
        f"(ratio {out.curv_total.mean() / out.abs_mass.mean():.3f}) — shares are "
        f"cancellation-dominated, read the signed sums, not the percentages"
    )

    print("\n=== P4  golden-event demotion (each must lose >= 95% of its curvature) ===")
    for seed in SEEDS:
        s = out[out.seed == seed]
        print(
            f"  seed {seed}: |curvature| of the 3 golden events "
            f"{s.golden_curv_abs_base.iloc[0]:.1f} (idealized) -> "
            f"{s.golden_curv_abs.min():.5f}-{s.golden_curv_abs.max():.5f} "
            f"(retained {100 * s.golden_retained.min():.4f}-"
            f"{100 * s.golden_retained.max():.4f}%)"
        )
    print(
        f"  VERDICT (absolute, >= 95% lost): {int((out.golden_retained < 0.05).sum())}/10 runs PASS"
    )

    print("\n=== 2D channel (with BH mass) ===")
    print(
        f"  MAP range {out.map_h_2d.min():.3f}-{out.map_h_2d.max():.3f} "
        f"(truth {H_TRUE}), pull mean {out.pull_2d.mean():+.2f}, "
        f"|pull|>2 in {int((out.pull_2d.abs() > 2).sum())}/10, "
        f"max edge/peak {out.edge_2d.max():.2e}"
    )

    print("\n=== P5  sigma->0 identity gate ===")
    print("  scored separately by md5 (see REALISTIC_READOUT.md): PASS")

    print("\n=== P6  host-loss rate ===")
    print(
        "  NOT SCORABLE from the delivered artifacts — the evaluate logs carry "
        "no ball-tree candidate/miss counter. See readout."
    )

    print(f"\nwrote {HERE / 'realistic_scores.csv'}")


if __name__ == "__main__":
    main()
