"""Gate C item 4 step 1: verify the empirical in-catalogue rate and w_G."""

import re

import numpy as np
import pandas as pd

BASE = "results/campaign51_20260728/realistic_20260729"

for seed in (61000, 62000):
    df = pd.read_csv(f"{BASE}/seed{seed}/prepared_cramer_rao_bounds.csv")
    n = len(df)
    idx_incat = df["host_galaxy_index"].to_numpy() >= 0
    flag = df["in_catalog"].to_numpy()
    print(
        f"seed{seed}: rows={n}  host_galaxy_index>=0: {idx_incat.sum()}  in_catalog flag true: {int(np.sum(flag.astype(bool)))}  agree={np.array_equal(idx_incat, flag.astype(bool))}"
    )
    print(f"   empirical in-cat rate = {idx_incat.sum()}/{n} = {idx_incat.sum() / n:.6f}")
    print(
        f"   d_L range Gpc: {df['luminosity_distance'].min():.4f} .. {df['luminosity_distance'].max():.4f}; SNR min {df['SNR'].min():.2f}"
    )
    print(
        f"   d_L in-cat median {np.median(df['luminosity_distance'][idx_incat]):.4f}, dark median {np.median(df['luminosity_distance'][~idx_incat]):.4f}"
    )

# w_G from the log extract
txt = open(f"{BASE}/seed61000/mixture_leg_log_extract.txt").read().splitlines()
D, bG, wG, sg1, sg2 = {}, {}, {}, {}, {}
for line in txt:
    m = re.search(r"D\(h=([\d.]+)\) = ([\d.e+-]+)", line)
    if m:
        D[float(m.group(1))] = float(m.group(2))
    m = re.search(r"beta_Gbar\(h=([\d.]+)\) = ([\d.e+-]+)", line)
    if m:
        bG[float(m.group(1))] = float(m.group(2))
    m = re.search(
        r"h_0_(\d+)\.log.*Partition-norm: w_G=beta_G/D\(h\)=([\d.]+), sum_w_Dg\(no_bh\)=([\d.e+-]+), sum_w_Dg\(with_bh\)=([\d.e+-]+)",
        line,
    )
    if m:
        h = float("0." + m.group(1))
        wG[h] = float(m.group(2))
        sg1[h] = float(m.group(3))
        sg2[h] = float(m.group(4))
print("n h-points:", len(D), len(bG), len(wG))
hs = sorted(D)
for h in (0.60, 0.73, 0.81, 0.86):
    beta_G = D[h] - bG[h]
    print(
        f"h={h}: D={D[h]:.6e} beta_Gbar={bG[h]:.6e} beta_G={beta_G:.6e} w_G(7sf)={beta_G / D[h]:.7f} w_G(log4dp)={wG[h]:.4f} Sglob_nobh={sg1[h]:.4e} Sglob_wbh={sg2[h]:.4e}"
    )
np.save(
    f"{BASE}/gate_b_20260730/legs.npy",
    np.array([[h, D[h], bG[h], wG[h], sg1[h], sg2[h]] for h in hs]),
)
