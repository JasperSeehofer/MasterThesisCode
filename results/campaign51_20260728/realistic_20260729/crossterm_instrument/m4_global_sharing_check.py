"""M-4 follow-up: GLOBAL sharing census over ALL event pairs (inverted index).

C-4's 279 pairs are an upper bound under its own sky(2sigma-chord-sum)+d_L(2sigma
window) criteria, but the production ball search uses 1.5sigma sky and a 3sigma d_L
z-window — so a sharing pair OUTSIDE the C-4 set is conceivable. This check builds
galaxy -> event inverted indices from the frozeng h_0_73.json ball emits and counts
every event pair sharing >=1 catalogue galaxy (1D and 2D channels), then compares
against the C-4 pair set. Updates m4_results.json in place with a
'global_sharing_check' block per venue. Read-only w.r.t. production data.

Run: cd /home/jasper/Repositories/MasterThesisCode && uv run python results/campaign51_20260728/realistic_20260729/crossterm_instrument/m4_global_sharing_check.py
"""

import json
import os
import time
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

T0 = time.time()
REPO = "/home/jasper/Repositories/MasterThesisCode"
os.chdir(REPO)

OUT = os.path.join(
    REPO, "results/campaign51_20260728/realistic_20260729/crossterm_instrument/m4_results.json"
)
CRB_CSV = os.path.join(
    REPO, "results/run_20260804_postfix/joint_r1/diagnostics/prepared_cramer_rao_bounds.csv"
)
FROZENG = os.path.join(REPO, "results/run_20260804_frozeng")


def c4_pairs():
    df = pd.read_csv(CRB_CSV)
    n = len(df)
    theta = df["qS"].to_numpy()
    phi = df["phiS"].to_numpy()
    s_phi2 = df["delta_phiS_delta_phiS"].to_numpy()
    s_theta2 = df["delta_qS_delta_qS"].to_numpy()
    cov = df["delta_phiS_delta_qS"].to_numpy()
    dl = df["luminosity_distance"].to_numpy()
    s_dl = np.sqrt(df["delta_luminosity_distance_delta_luminosity_distance"].to_numpy())
    r = np.empty(n)
    for i in range(n):
        sig = np.array([[s_phi2[i], cov[i]], [cov[i], s_theta2[i]]])
        jac = np.diag([abs(np.sin(theta[i])), 1.0])
        r[i] = 2.0 * np.sqrt(max(float(np.linalg.eigvalsh(jac @ sig @ jac.T).max()), 0.0))
    st = np.sin(theta)
    xyz = np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=1)
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    iu = np.triu_indices(n, k=1)
    sky = d[iu] <= (r[:, None] + r[None, :])[iu]
    ii, jj = iu[0][sky], iu[1][sky]
    lo, hi = dl - 2 * s_dl, dl + 2 * s_dl
    win = (lo[ii] <= hi[jj]) & (lo[jj] <= hi[ii])
    return set(zip(ii[win].tolist(), jj[win].tolist()))


def global_pairs(json_path):
    d = json.load(open(json_path))
    gl, add = d["galaxy_likelihoods"], d["additional_galaxies_without_bh_mass"]
    inv_1d, inv_2d = defaultdict(list), defaultdict(list)
    for k in gl:
        ev = int(k)
        for r in gl[k]:
            inv_1d[r[0]].append(ev)
            inv_2d[r[0]].append(ev)
        for r in add[k]:
            inv_1d[r[0]].append(ev)
    out = {}
    for name, inv in (("1d", inv_1d), ("2d", inv_2d)):
        pairs = set()
        max_deg = 0
        for g, evs in inv.items():
            if len(evs) > 1:
                max_deg = max(max_deg, len(evs))
                for a, b in combinations(sorted(set(evs)), 2):
                    pairs.add((a, b))
        out[name] = {
            "pairs": pairs,
            "max_events_per_galaxy": max_deg,
            "n_galaxy_rows": sum(len(v) for v in inv.values()),
            "n_distinct_galaxies": len(inv),
        }
    return out


def main():
    c4 = c4_pairs()
    assert len(c4) == 279
    results = json.load(open(OUT))
    for venue in ("joint_r1", "iiib"):
        jp = os.path.join(FROZENG, venue, "posteriors_with_bh_mass/h_0_73.json")
        g = global_pairs(jp)
        block = {"ball_json": jp, "c4_pair_set_size": len(c4)}
        for name in ("1d", "2d"):
            pairs = g[name]["pairs"]
            inside = pairs & c4
            outside = pairs - c4
            block[f"n_sharing_pairs_global_{name}"] = len(pairs)
            block[f"n_sharing_pairs_in_c4_set_{name}"] = len(inside)
            block[f"n_sharing_pairs_OUTSIDE_c4_set_{name}"] = len(outside)
            block[f"outside_pairs_{name}"] = sorted(list(outside))[:200]
            block[f"max_events_per_galaxy_{name}"] = g[name]["max_events_per_galaxy"]
            block[f"n_distinct_ball_galaxies_{name}"] = g[name]["n_distinct_galaxies"]
            print(
                f"[{time.time() - T0:6.1f}s] {venue} {name}: global sharing pairs "
                f"{len(pairs)}; in C-4 set {len(inside)}; OUTSIDE C-4 set {len(outside)}"
            )
        results[venue]["global_sharing_check"] = block
    results["global_check_runtime_s"] = round(time.time() - T0, 1)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print(f"[{time.time() - T0:6.1f}s] updated {OUT}")


if __name__ == "__main__":
    main()
