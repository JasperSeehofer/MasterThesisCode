"""Extract the 2026-07-30 deep-venue pp_coverage cells and compare to the
2026-07-10 z_support=1.0 controls. Read-only over results/*.json; prints the
per-cell x truth table and the control comparison used in SUMMARY_GATE_C2.md.

Run from repo root: .venv/bin/python results/campaign51_20260728/realistic_20260729/gate_b_20260730/extract_and_compare.py
"""

import glob
import json
import math


def load(path: str) -> list[dict]:
    d = json.load(open(path))
    zs = d["config"]["z_support"]
    sz = d["config"]["sigma_z"]
    out = []
    for r in d["results"].values():
        out.append(
            {
                "zs": zs,
                "sz": sz,
                "truth": r["h_true"],
                "cov50": r["coverage"]["50"],
                "cov68": r["coverage"]["68"],
                "cov90": r["coverage"]["90"],
                "rail": r["rail_fraction"],
                "map_mean": r["map_mean"],
                "map_bias": r["map_bias"],
                "map_std": r["map_std"],
                "comp_frac": r["completion_fraction"],
            }
        )
    return out


def main() -> None:
    new_files = sorted(glob.glob("results/pp_coverage_deepvenue_20260730/pp_zs*_volume.json"))
    ctrl_files = [
        "results/pp_coverage_deepvenue_20260710/pp_zs1.0_sz0.015_volume.json",
        "results/pp_coverage_deepvenue_20260710/pp_zs1.0_sz0.035_volume.json",
    ]

    rows = []
    for f in new_files:
        rows += load(f)

    ctrl = {}
    for f in ctrl_files:
        for r in load(f):
            ctrl[(r["sz"], r["truth"])] = r

    header = (
        f"{'zs':>5} {'sz':>6} {'truth':>6} {'cov50':>6} {'cov68':>6} {'cov90':>6} "
        f"{'rail':>6} {'map_mean':>9} {'bias':>8} {'comp_frac':>9} | "
        f"{'ctrl_cov68':>10} {'d_map':>8} {'2SEM':>6} {'flag_bias':>9} {'flag_cov':>9}"
    )
    print(header)
    for r in sorted(rows, key=lambda x: (-x["zs"], x["sz"], x["truth"])):
        c = ctrl[(r["sz"], r["truth"])]
        dmap = r["map_mean"] - c["map_mean"]
        sem2 = 2 * r["map_std"] / math.sqrt(120)
        dcov = r["cov68"] - c["cov68"]
        flag_cov = "COLLAPSE" if abs(dcov) > 2 * 0.085 else "ok"
        flag_bias = "FLAG" if abs(dmap) > sem2 else "ok"
        print(
            f"{r['zs']:>5} {r['sz']:>6} {r['truth']:>6} {r['cov50']:>6.3f} "
            f"{r['cov68']:>6.3f} {r['cov90']:>6.3f} {r['rail']:>6.3f} "
            f"{r['map_mean']:>9.4f} {r['map_bias']:>+8.4f} {r['comp_frac']:>9.3f} | "
            f"{c['cov68']:>10.3f} {dmap:>+8.4f} {sem2:>6.4f} {flag_bias:>9} {flag_cov:>9}"
        )


if __name__ == "__main__":
    main()
