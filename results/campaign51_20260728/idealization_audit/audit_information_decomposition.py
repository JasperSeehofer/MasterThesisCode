"""Reproduce the idealization-audit numbers (2026-07-29).

Run from repo root with .venv/bin/python. Produces:
1. The per-event ln-likelihood curvature decomposition (in-catalog vs dark)
   from the seed-61000 production posteriors.
2. The host-row alignment (host_galaxy_index -> reduced catalogue row) and the
   photometric/spectroscopic flag census of the information-carrying hosts.
3. The counterfactual sigma_H0 scenarios (spec-z, PV, actual photo-z widths).

All numbers quoted in IDEALIZATION_LEDGER.md come from this script.
"""

import json

import numpy as np
import pandas as pd

RUN = "results/campaign51_20260728/run_seed61000"
POST = f"{RUN}/posteriors_fixed"
H = 0.73
C_KM_S = 299792.458


def load_posterior(tag: str) -> dict[int, float]:
    with open(f"{POST}/h_0_{tag}.json") as f:
        j = json.load(f)
    return {int(k): (j[k][0] if isinstance(j[k], list) else j[k]) for k in j if k.isdigit()}


def main() -> None:
    # ---- 1. curvature decomposition --------------------------------------
    a, b, c = load_posterior("725"), load_posterior("73"), load_posterior("735")
    df = pd.read_csv(f"{RUN}/prepared_fixed.csv")
    incat_idx = set(df.index[df.host_galaxy_index >= 0])
    common = [k for k in b if k in a and k in c and a[k] > 0 and b[k] > 0 and c[k] > 0]
    curv = {k: np.log(b[k] / a[k]) + np.log(b[k] / c[k]) for k in common}
    ci = sum(v for k, v in curv.items() if k in incat_idx)
    di = sum(v for k, v in curv.items() if k not in incat_idx)
    tot = ci + di
    print(f"events={len(common)}  in-cat={len(incat_idx & set(common))}")
    print(f"curvature (dh=0.005 units): total={tot:.1f} in-cat={ci:.1f} dark={di:.1f}")
    print(f"sigma_h = {0.005 / np.sqrt(tot):.2e}  -> sigma_H0 = {100 * 0.005 / np.sqrt(tot):.3f}")

    # ---- 2. host alignment (replicates handler load/prune, index-exact) ---
    cols = ["RA", "DEC", "BMAG", "Z", "ZERR", "SM", "SMERR", "FLAG"]
    cat = pd.read_csv(
        "master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv",
        names=cols,
        usecols=[3, 4, 5, 6, 7],
    )
    alpha, beta = 7.45 * np.log(10), 1.05
    d_alpha, d_beta, sigma_int = 0.08 * np.log(10), 0.11, 0.24 * np.log(10)
    bh = np.exp(alpha + beta * np.log(cat.SM / 10))
    bh_err = bh * np.sqrt(
        sigma_int**2
        + d_alpha**2
        + (np.log(cat.SM / 10) * d_beta) ** 2
        + (beta / cat.SM * cat.SMERR) ** 2
    )
    cat["BH"], cat["BHERR"] = bh, bh_err
    cat = cat[~cat.BH.isna()]
    cat = cat[
        (cat.BH + cat.BHERR >= 1e4) & (cat.BH - cat.BHERR <= 1e7) & (cat.Z - cat.ZERR <= 1.5)
    ].reset_index(drop=True)

    inc = df[df.host_galaxy_index >= 0].copy()
    sub = cat.iloc[inc.host_galaxy_index.astype(int).values]
    inc["host_z"], inc["host_zerr"], inc["host_flag"] = (
        sub.Z.values,
        sub.ZERR.values,
        sub.FLAG.values,
    )
    ratio = inc.M.values / (sub.BH.values * (1 + sub.Z.values))
    assert (np.abs(ratio - 1) < 1e-3).all(), "host alignment broken"
    print("alignment M_z == M_host*(1+z): OK for all 76")
    print("host flags:", inc.host_flag.value_counts().to_dict(), "(1=photometric, 3=spectroscopic)")

    # ---- 3. counterfactual sigma_H0 ---------------------------------------
    rel_dl = (
        np.sqrt(inc.delta_luminosity_distance_delta_luminosity_distance)
        / inc.luminosity_distance
    )
    z = inc.host_z.values

    def s_h0(rel_eff: np.ndarray) -> float:
        return 100.0 / np.sqrt((1.0 / (H * rel_eff) ** 2).sum())

    print(f"A pipeline (z exact):              {s_h0(rel_dl.values):.3f} km/s/Mpc")
    for name, sz in [
        ("B spec-z 0.0017 + 150 km/s PV", np.sqrt(0.0017**2 + ((1 + z) * 150 / C_KM_S) ** 2)),
        ("C spec-z 0.0017 + 500 km/s PV", np.sqrt(0.0017**2 + ((1 + z) * 500 / C_KM_S) ** 2)),
        ("D actual catalogue z_err (photo)", inc.host_zerr.values),
    ]:
        rel = np.sqrt(rel_dl.values**2 + (sz / z) ** 2)
        print(f"{name}: {s_h0(rel):.3f} km/s/Mpc")


if __name__ == "__main__":
    main()
