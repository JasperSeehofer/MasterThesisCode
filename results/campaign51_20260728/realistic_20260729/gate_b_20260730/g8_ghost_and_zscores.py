"""(3) the 0.0697 'loose thread' + tension z-scores before/after the mass-aware fix."""

import json
import re

import numpy as np

BASE = "results/campaign51_20260728/realistic_20260729"
OUT = f"{BASE}/gate_b_20260730"
D, bGbar, Sg1, Sg2 = {}, {}, {}, {}
for line in open(f"{BASE}/seed61000/mixture_leg_log_extract.txt"):
    m = re.search(r"D\(h=([\d.]+)\) = ([\d.e+-]+)", line)
    if m:
        D[round(float(m.group(1)), 4)] = float(m.group(2))
    m = re.search(r"beta_Gbar\(h=([\d.]+)\) = ([\d.e+-]+)", line)
    if m:
        bGbar[round(float(m.group(1)), 4)] = float(m.group(2))
    m = re.search(
        r"h_0_(\d+)\.log.*sum_w_Dg\(no_bh\)=([\d.e+-]+), sum_w_Dg\(with_bh\)=([\d.e+-]+)", line
    )
    if m:
        h = round(float("0." + m.group(1)), 4)
        Sg1[h] = float(m.group(2))
        Sg2[h] = float(m.group(3))
hs = sorted(D)
bG = {h: D[h] - bGbar[h] for h in hs}
S = json.load(open(f"{OUT}/g2_catalogue_summary.json"))
W, Vf73 = S["W_cat"], S["V_f"]
print(" h     w_G=beta_G/D   w_G[gen_marginal]=(Sig_wbh/n_hat_w)/D_gen   w_G[beta_G*r]")
for h in hs[::4] + [0.86]:
    nh = W / (Vf73 * (0.73 / h) ** 3)
    a = Sg2[h] / nh
    r = Sg2[h] / Sg1[h]
    bc = bG[h] * r
    print(
        f"{h:.3f}   {bG[h] / D[h]:.5f}          {a / (a + bGbar[h]):.5f}                       {bc / (bc + bGbar[h]):.5f}"
    )
r73 = Sg2[0.73] / Sg1[0.73]
bc = bG[0.73] * r73
wc = bc / (bc + bGbar[0.73])
print(
    f"\nCorrected membership weight at truth: w_G_corr(0.73) = {wc:.5f} "
    f"(delivered {bG[0.73] / D[0.73]:.5f})"
)
for tag, (n, k) in (("seed61000", (1590, 76)), ("seed62000", (1545, 88)), ("pooled", (3135, 164))):
    for name, p in (("delivered w_G", bG[0.73] / D[0.73]), ("mass-aware w_G", wc)):
        e = n * p
        print(
            f"  {tag:9s} {name:15s}: expected {e:6.1f}, observed {k:3d}, binomial z = {(k - e) / np.sqrt(e * (1 - p)):+6.2f}"
        )
