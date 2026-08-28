"""[CMEM] Read 2 — paired composition read (PREREGISTRATION_CMEM_READS_20260828.md §3).

Recomputes the true-host outside/inside-cone flag over the 24 banked bc arms with the
handler's OWN catalogue load and cone geometry (chord BallTree metric, radius =
1.5·sqrt(lambda_max) of the Jacobian-scaled Fisher sky block — handler.py, the
get_possible_hosts_from_ball_tree radius code, replicated line-for-line on the CRB
columns), then joins the per-event diagnostics (h = 0.73) and evaluates the registered
reads R2a/R2b/R2c with the within-seed permutation band.

Gates: C-G1 (anchor seed 900121 event 20 chord/radius full-float + census 380/2261
reproduction) STOP-gated before any comparison is read; C-G2 c_i in [0,1] sanity.
RNG: permutations use a fixed seed (20260828), disclosed (the prereg fixed the count,
not the seed).
"""

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from darksiren_emri.galaxy_catalogue.handler import GalaxyCatalogueHandler, _polar_to_cartesian

FLEET = Path(__file__).resolve().parent / "p3_2d_fleet_20260825"
OUT = Path(__file__).resolve().parent / "cmem_work"
K = 1.5
# Anchor: radius full-float (R2.6); chord to R2.6's DISPLAYED precision 1.674660e-03 —
# the MKER-6 census entry's parenthetical "full-float 1.6746585172e-03" is inconsistent
# with R2.6's own display (rounds to 1.674659) and with this instrument (1.67465986e-03);
# recorded as a discrepancy in that entry's quoted chord, tolerance here = 5e-10 vs display.
ANCHOR = ("bc_900121_work", 20, 1.674660e-03, 1.4956979545757095e-03)
BANKED = (380, 2261, 0.1681)

CRB_COLS = [
    "qS",
    "phiS",
    "delta_qS_delta_qS",
    "delta_phiS_delta_phiS",
    "delta_phiS_delta_qS",
    "host_galaxy_index",
    "in_catalog",
    "z_true",
]


def cone_radius(theta: float, phi_var: float, theta_var: float, cov: float) -> float:
    sigma = np.array([[phi_var, cov], [cov, theta_var]])
    jac = np.diag([abs(np.sin(theta)), 1.0])
    lam = float(np.linalg.eigvalsh(jac @ sigma @ jac.T).max())
    return float(K * np.sqrt(max(lam, 0.0)))


def main() -> None:
    # Production prune constants (constants.py M_SOURCE_FRAME_MIN/MAX, handler Z_draw) —
    # the handler load applies R&V15 map, COORD-03 ecliptic rotation, and the prune,
    # exactly as production's cone geometry saw them. Positional-vs-label indexing of
    # host_galaxy_index is settled by the C-G1 anchor gate, not assumed.
    handler = GalaxyCatalogueHandler(1e4, 1e7, 1.5)
    cat = handler.reduced_galaxy_catalog.reset_index(drop=True)
    host_xyz = _polar_to_cartesian(
        cat["THETA_S"].to_numpy(dtype=np.float64), cat["PHI_S"].to_numpy(dtype=np.float64)
    )

    rows = []
    for arm_dir in sorted(glob.glob(str(FLEET / "bc_*_work"))):
        seed_dir = Path(glob.glob(str(Path(arm_dir) / "seed*"))[0])
        seed = seed_dir.name
        crb = pd.read_csv(seed_dir / "simulations" / "prepared_cramer_rao_bounds.csv", usecols=CRB_COLS)
        diag = pd.read_csv(seed_dir / "simulations" / "diagnostics" / "event_likelihoods.csv")
        diag = diag[diag["h"] == 0.73].set_index("event_idx")
        for i, r in crb.iterrows():
            if not bool(r["in_catalog"]):
                continue
            hidx = int(r["host_galaxy_index"])
            ev_xyz = _polar_to_cartesian(np.array([r["qS"]]), np.array([r["phiS"]]))[0]
            chord = float(np.linalg.norm(host_xyz[hidx] - ev_xyz))
            radius = cone_radius(
                float(r["qS"]),
                float(r["delta_phiS_delta_phiS"]),
                float(r["delta_qS_delta_qS"]),
                float(r["delta_phiS_delta_qS"]),
            )
            d = diag.loc[i] if i in diag.index else None
            if d is None:
                # Banked census basis = the posterior-joined subset (2261 rows,
                # CLAIM_WGEO §3.8 note); un-evaluated CRB rows are out of basis.
                continue
            rows.append(
                {
                    "arm": Path(arm_dir).name,
                    "seed": seed,
                    "event_idx": i,
                    "chord": chord,
                    "radius": radius,
                    "outside": chord > radius,
                    "z_true": float(r["z_true"]),
                    "L_cat_no_bh": float(d["L_cat_no_bh"]) if d is not None else np.nan,
                    "B_num": float(d["B_num"]) if d is not None else np.nan,
                    "combined_no_bh": float(d["combined_no_bh"]) if d is not None else np.nan,
                    "D_tilde_phi": float(d["D_tilde_phi"]) if d is not None else np.nan,
                }
            )
    df = pd.DataFrame(rows)

    # ---- Gate C-G1: anchor + census reproduction, STOP on failure ----
    a = df[(df["arm"] == ANCHOR[0]) & (df["event_idx"] == ANCHOR[1])]
    if len(a) != 1:
        raise SystemExit(f"C-G1 STOP: anchor row not found ({len(a)})")
    chord_ok = abs(float(a["chord"].iloc[0]) - ANCHOR[2]) < 5e-10
    radius_ok = abs(float(a["radius"].iloc[0]) - ANCHOR[3]) < 1e-15
    n_out, n_tot = int(df["outside"].sum()), len(df)
    frac = n_out / n_tot
    census_ok = (n_out, n_tot) == (BANKED[0], BANKED[1])
    g1 = {
        "anchor_chord": float(a["chord"].iloc[0]),
        "anchor_radius": float(a["radius"].iloc[0]),
        "chord_ok": chord_ok,
        "radius_ok": radius_ok,
        "n_outside": n_out,
        "n_total": n_tot,
        "fraction": frac,
        "census_ok": census_ok,
        "passed": chord_ok and radius_ok and census_ok,
    }
    print("C-G1:", json.dumps(g1))
    if not g1["passed"]:
        OUT.mkdir(exist_ok=True)
        with open(OUT / "cmem_read2.json", "w") as f:
            json.dump({"verdict": "INSTRUMENT-DEFECT", "c_g1": g1}, f, indent=1)
        raise SystemExit("C-G1 STOP: anchor/census not reproduced")

    # ---- Reads ----
    df = df.dropna(subset=["combined_no_bh", "B_num", "D_tilde_phi"])
    c_share = 1.0 - df["B_num"] / (df["combined_no_bh"] * df["D_tilde_phi"])
    df = df.assign(c_share=c_share)
    in_range = ((df["c_share"] >= -1e-9) & (df["c_share"] <= 1.0 + 1e-9)).mean()
    g2 = {"c_share_in_[0,1]_fraction": float(in_range), "passed": bool(in_range >= 0.999)}

    out_m, in_m = df[df["outside"]], df[~df["outside"]]

    def med_diff(col: str, frame: pd.DataFrame, mask: np.ndarray) -> float:
        return float(np.median(frame[col].to_numpy()[mask]) - np.median(frame[col].to_numpy()[~mask]))

    rng = np.random.default_rng(20260828)

    def perm_p(col: str) -> tuple[float, float]:
        obs = med_diff(col, df, df["outside"].to_numpy())
        seeds = df["seed"].to_numpy()
        labels = df["outside"].to_numpy().copy()
        count = 0
        for _ in range(10_000):
            perm = labels.copy()
            for s in np.unique(seeds):
                m = seeds == s
                perm[m] = rng.permutation(perm[m])
            if abs(med_diff(col, df, perm)) >= abs(obs):
                count += 1
        return obs, count / 10_000

    r2a_obs, r2a_p = perm_p("c_share")
    r2c_obs, r2c_p = perm_p("combined_no_bh")
    r2b = {
        "collapse_rate_outside": float((out_m["L_cat_no_bh"] == 0.0).mean()),
        "collapse_rate_inside": float((in_m["L_cat_no_bh"] == 0.0).mean()),
    }
    result = {
        "c_g1": g1,
        "c_g2": g2,
        "n_outside": len(out_m),
        "n_inside": len(in_m),
        "R2a": {
            "median_c_share_outside": float(np.median(out_m["c_share"])),
            "median_c_share_inside": float(np.median(in_m["c_share"])),
            "mean_c_share_outside": float(np.mean(out_m["c_share"])),
            "mean_c_share_inside": float(np.mean(in_m["c_share"])),
            "median_diff": r2a_obs,
            "perm_p": r2a_p,
            "displaced": bool(r2a_p < 0.01),
        },
        "R2b": r2b,
        "R2c": {
            "median_combined_outside": float(np.median(out_m["combined_no_bh"])),
            "median_combined_inside": float(np.median(in_m["combined_no_bh"])),
            "ratio_outside_over_inside": float(
                np.median(out_m["combined_no_bh"]) / np.median(in_m["combined_no_bh"])
            ),
            "median_diff": r2c_obs,
            "perm_p": r2c_p,
            "displaced": bool(r2c_p < 0.01),
        },
        "covariate_z_true": {
            "median_outside": float(np.median(out_m["z_true"])),
            "median_inside": float(np.median(in_m["z_true"])),
        },
    }
    OUT.mkdir(exist_ok=True)
    with open(OUT / "cmem_read2.json", "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
