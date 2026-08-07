"""Cheap integrity reads for the adversarial review (M-2/M-3/M-4 class only).

Reads: CRB CSVs (md5, row counts, filter), event_likelihoods.csv (columns,
h grid), frozeng ball JSONs (M-4 ball sets), staged catalogues (sha256),
and reruns the C-4 census recipe. Does NOT run the instrument.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(
    0,
    "/home/jasper/Repositories/MasterThesisCode/results/campaign51_20260728/"
    "realistic_20260729/crossterm_instrument",
)
from crossterm_instrument import (  # noqa: E402
    c4_pair_census,
    load_ball_sets,
    load_filtered_events,
)

REPO = Path("/home/jasper/Repositories/MasterThesisCode")

# --- 1. CRB CSV integrity -------------------------------------------------
p1 = REPO / "results/run_20260804_postfix/joint_r1/diagnostics/prepared_cramer_rao_bounds.csv"
p2 = REPO / "results/run_20260804_postfix/iiib/diagnostics/prepared_cramer_rao_bounds.csv"
md5_1 = hashlib.md5(p1.read_bytes()).hexdigest()
md5_2 = hashlib.md5(p2.read_bytes()).hexdigest()
df = pd.read_csv(p1)
print(f"CRB md5 joint_r1={md5_1} iiib={md5_2} identical={md5_1 == md5_2}")
print(f"CRB rows={len(df)}")
filt = load_filtered_events(p1)
print(f"filtered rows={len(filt)} (SNR>=20 then rel_dL<0.10)")
snr_only = df[df["SNR"] >= 20.0]
print(f"SNR-only rows={len(snr_only)} (drop {len(df) - len(snr_only)})")

# --- 2. event_likelihoods columns + h grid --------------------------------
for venue in ["iiib", "joint_r1"]:
    el = pd.read_csv(
        REPO / f"results/run_20260804_postfix/{venue}/diagnostics/event_likelihoods.csv"
    )
    hs = np.sort(el["h"].unique())
    need = ["event_idx", "h", "w_G", "L_cat_no_bh", "L_cat_with_bh"]
    print(
        f"{venue}: rows={len(el)} cols_ok={all(c in el.columns for c in need)} "
        f"n_h={len(hs)} h_min={hs[0]:.4f} h_max={hs[-1]:.4f} "
        f"has(0.60,0.73,0.81,0.86)={[bool(np.min(np.abs(hs - x)) < 1e-9) for x in (0.60, 0.73, 0.81, 0.86)]} "
        f"min_spacing={np.min(np.diff(hs)):.4f} "
        f"n_events={el['event_idx'].nunique()}"
    )

# --- 3. C-4 census reproduction ------------------------------------------
pairs, degree = c4_pair_census(df)
ev_touched = {e for p in pairs for e in p}
print(f"census: sky+dL pairs={len(pairs)} events_touched={len(ev_touched)}")

# sky-only census for the 1620/981 numbers
theta = df["qS"].to_numpy()
phi = df["phiS"].to_numpy()
s_phi2 = df["delta_phiS_delta_phiS"].to_numpy()
s_theta2 = df["delta_qS_delta_qS"].to_numpy()
cov = df["delta_phiS_delta_qS"].to_numpy()
n = len(df)
r = np.empty(n)
for k in range(n):
    sig = np.array([[s_phi2[k], cov[k]], [cov[k], s_theta2[k]]])
    jac = np.diag([abs(np.sin(theta[k])), 1.0])
    lam = float(np.linalg.eigvalsh(jac @ sig @ jac.T).max())
    r[k] = 2.0 * np.sqrt(max(lam, 0.0))
st = np.sin(theta)
xyz = np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=1)
d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
iu = np.triu_indices(n, k=1)
sky = d[iu] <= (r[:, None] + r[None, :])[iu]
sky_events = set(iu[0][sky]) | set(iu[1][sky])
print(f"census: sky-only pairs={int(sky.sum())} events_touched={len(sky_events)}")

# --- 4. M-4 ball sets + shared-count distribution over census pairs -------
filtered_idx = set(int(i) for i in filt.index)
pairs_f = [(i, j) for (i, j) in pairs if i in filtered_idx and j in filtered_idx]
print(f"pairs after production filter: {len(pairs_f)}")
for venue in ["iiib", "joint_r1"]:
    b1, b2 = load_ball_sets(REPO / f"results/run_20260804_frozeng/{venue}")
    for ch, ball in (("1d", b1), ("2d", b2)):
        ns = [len(ball.get(i, set()) & ball.get(j, set())) for (i, j) in pairs_f]
        ns = np.array(ns)
        print(
            f"{venue}/{ch}: pairs={len(ns)} n_shared>0: {int((ns > 0).sum())}  "
            f"n_shared>=2: {int((ns >= 2).sum())}  max={ns.max() if len(ns) else 0}  "
            f"median_nonzero={np.median(ns[ns > 0]) if (ns > 0).any() else 0}"
        )

# --- 5. staged catalogue hashes ------------------------------------------
staged = REPO / "results/campaign51_20260728/realistic_20260729/realizations_staged"
for f in ["cluster_parent_reduced_galaxy_catalogue.csv", "observed_catalogue_seed900001.csv"]:
    p = staged / f
    if p.exists():
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"{f}: sha256={h[:16]}... size={p.stat().st_size}")
    else:
        print(f"{f}: MISSING")
# sidecar check for the observed catalogue
for sc in staged.glob("*.json"):
    print("sidecar:", sc.name)
