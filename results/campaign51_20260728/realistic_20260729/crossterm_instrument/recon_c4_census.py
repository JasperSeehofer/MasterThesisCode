"""Recon: reproduce the C-4 census (1620 sky pairs / 981 events; 279 d_L pairs / 385 events)
from prepared_cramer_rao_bounds.csv using the production ball-radius formula
(galaxy_catalogue/handler.py:613-617): r = 2 * sqrt(lambda_max(J Sigma J^T)),
J = diag(|sin theta|, 1), Sigma = [[s_phi^2, C], [C, s_theta^2]] (chord on unit sphere).

Pair predicates tested:
  sky overlap: chord distance between event centers <= r_i + r_j
  d_L window:  intervals [d_L - 2 s_dL, d_L + 2 s_dL] intersect
Read-only. Output: printed census numbers only.
"""

import numpy as np
import pandas as pd

CSV = (
    "/home/jasper/Repositories/MasterThesisCode/results/run_20260804_postfix/"
    "joint_r1/diagnostics/prepared_cramer_rao_bounds.csv"
)

df = pd.read_csv(CSV)
n = len(df)
theta = df["qS"].to_numpy()
phi = df["phiS"].to_numpy()
s_phi2 = df["delta_phiS_delta_phiS"].to_numpy()
s_theta2 = df["delta_qS_delta_qS"].to_numpy()
cov = df["delta_phiS_delta_qS"].to_numpy()
dl = df["luminosity_distance"].to_numpy()
s_dl = np.sqrt(df["delta_luminosity_distance_delta_luminosity_distance"].to_numpy())

# radius per event
r = np.empty(n)
for i in range(n):
    sig = np.array([[s_phi2[i], cov[i]], [cov[i], s_theta2[i]]])
    jac = np.diag([abs(np.sin(theta[i])), 1.0])
    lam = float(np.linalg.eigvalsh(jac @ sig @ jac.T).max())
    r[i] = 2.0 * np.sqrt(max(lam, 0.0))

r_deg = np.degrees(r)  # chord treated as angle in the draft's deg quote
print(f"n_events = {n}")
print(
    f"radius chord->deg: median {np.median(r_deg):.2f}, p90 {np.percentile(r_deg, 90):.2f}, max {r_deg.max():.2f}"
)

# unit vectors
st = np.sin(theta)
xyz = np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=1)
# pairwise chord distances
d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
rsum = r[:, None] + r[None, :]
iu = np.triu_indices(n, k=1)
sky = d[iu] <= rsum[iu]
n_pairs_total = len(iu[0])
n_sky = int(sky.sum())
touched_sky = np.zeros(n, dtype=bool)
ii, jj = iu[0][sky], iu[1][sky]
touched_sky[ii] = True
touched_sky[jj] = True
deg = np.bincount(np.concatenate([ii, jj]), minlength=n)
print(f"sky-overlap pairs: {n_sky} of {n_pairs_total} ({100 * n_sky / n_pairs_total:.3f}%)")
print(
    f"events with >=1 partner: {int(touched_sky.sum())}/{n} "
    f"({100 * touched_sky.mean():.1f}%); degree median {np.median(deg[touched_sky]):.0f}, "
    f"p90 {np.percentile(deg[touched_sky], 90):.0f}, max {deg.max()}"
)

# d_L 2-sigma window intersection
lo = dl - 2 * s_dl
hi = dl + 2 * s_dl
win = (lo[ii] <= hi[jj]) & (lo[jj] <= hi[ii])
n_both = int(win.sum())
touched = np.zeros(n, dtype=bool)
touched[ii[win]] = True
touched[jj[win]] = True
print(
    f"sky+dL(2sig window) pairs: {n_both} ({100 * n_both / n_pairs_total:.3f}%), "
    f"touching {int(touched.sum())}/{n} events ({100 * touched.mean():.1f}%)"
)
