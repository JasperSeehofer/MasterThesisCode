"""Gate C item 4, step 3: catalogue rate-weighted MBH-mass distribution vs the
population mass marginal (mbh_mass_function * R_eff) that the injection pool --
and hence the mass-marginalised p_det used inside beta_G -- is built from."""

import numpy as np
import pandas as pd

from master_thesis_code.constants import M_SOURCE_FRAME_MAX, M_SOURCE_FRAME_MIN
from master_thesis_code.emri_rate import R_eff_per_mbh, mbh_mass_function

OUT = "results/campaign51_20260728/realistic_20260729/gate_b_20260730"
d = np.load(f"{OUT}/g2_zshape.npz")
zg, Mg, w_g = d["zg"], d["Mg"], d["w_g"]

lg = np.log10(np.clip(Mg, 1e-3, None))
edges = np.arange(0.0, 12.01, 0.5)
hw, _ = np.histogram(lg, bins=edges, weights=w_g)
hn, _ = np.histogram(lg, bins=edges)
cen = 0.5 * (edges[1:] + edges[:-1])
# population marginal in log10 M over the injection band
grid = np.linspace(np.log10(M_SOURCE_FRAME_MIN), np.log10(M_SOURCE_FRAME_MAX), 2000)
dens = mbh_mass_function(10**grid) * R_eff_per_mbh(10**grid)
dens /= np.trapezoid(dens, grid)
print("log10M  N_gal        rateW_frac   pop_frac")
for i in range(len(cen)):
    lo, hi = edges[i], edges[i + 1]
    g = np.clip(np.array([lo, hi]), grid[0], grid[-1])
    pf = (
        np.trapezoid(
            np.interp(np.linspace(g[0], g[1], 50), grid, dens), np.linspace(g[0], g[1], 50)
        )
        if hi > grid[0] and lo < grid[-1]
        else 0.0
    )
    if hn[i] > 0 or pf > 1e-4:
        print(f"{lo:5.1f}-{hi:4.1f} {hn[i]:10d}  {hw[i] / hw.sum():10.5f}  {pf:9.5f}")
print()
print(
    f"catalogue rate-weighted median log10 M   = {np.interp(0.5, np.cumsum(np.sort(w_g)[np.argsort(np.argsort(lg))]) / 0, 0) if False else ''}"
)
o = np.argsort(lg)
cw = np.cumsum(w_g[o]) / w_g.sum()
for q in (0.05, 0.25, 0.5, 0.75, 0.95):
    print(f"  catalogue rate-weighted log10 M quantile {q:.2f}: {np.interp(q, cw, lg[o]):.3f}")
cd = np.cumsum(dens) * np.gradient(grid)
cd /= cd[-1]
for q in (0.05, 0.25, 0.5, 0.75, 0.95):
    print(f"  population       log10 M quantile {q:.2f}: {np.interp(q, cd, grid):.3f}")
inband = (Mg >= M_SOURCE_FRAME_MIN) & (Mg <= M_SOURCE_FRAME_MAX)
print(
    f"\ncatalogue rate weight inside the EMRI band [1e4,1e7]: {w_g[inband].sum() / w_g.sum():.5f}"
    f"  (galaxies: {inband.sum()}/{Mg.size} = {inband.mean():.5f})"
)
print(f"catalogue rate weight ABOVE 1e7 : {w_g[Mg > M_SOURCE_FRAME_MAX].sum() / w_g.sum():.5f}")
print(f"catalogue rate weight BELOW 1e4 : {w_g[Mg < M_SOURCE_FRAME_MIN].sum() / w_g.sum():.5f}")

# detected-event masses (detector frame M_z) by class
for seed in (61000, 62000):
    df = pd.read_csv(
        f"results/campaign51_20260728/realistic_20260729/seed{seed}/prepared_cramer_rao_bounds.csv"
    )
    inc = df["host_galaxy_index"].to_numpy() >= 0
    print(
        f"\nseed{seed} detected events: log10 M_z median in-cat {np.median(np.log10(df['M'][inc])):.3f} "
        f"(n={inc.sum()}) vs dark {np.median(np.log10(df['M'][~inc])):.3f} (n={(~inc).sum()})"
    )
    print(
        f"   d_L Gpc median in-cat {np.median(df['luminosity_distance'][inc]):.3f} dark {np.median(df['luminosity_distance'][~inc]):.3f}"
    )
