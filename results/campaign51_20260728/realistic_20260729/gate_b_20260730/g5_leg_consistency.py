"""Gate C item 1: level + h-slope consistency of the two mixture legs about the
same population, and the channel-difference interaction of the w_G mis-calibration."""

import json
import re

import numpy as np
import pandas as pd

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
hs = np.array(sorted(D))
bG = {h: D[h] - bGbar[h] for h in hs}
S = json.load(open(f"{OUT}/g2_catalogue_summary.json"))
W_cat = S["W_cat"]
Vf73 = S["V_f"]
nhat = {h: W_cat / (Vf73 * (0.73 / h) ** 3) for h in hs}  # V_f(h) = V_f(0.73)(0.73/h)^3 exactly

print(
    "GATE C ITEM 1 — do the completeness-model leg and the catalogue leg describe the same population?"
)
print(
    " h     beta_G(model)   Sig_nobh/beta_G  Sig_wbh/beta_G   n_hat_w=W_cat/V_f   A:(S_nobh/bG)/nhat  B:(S_wbh/bG)/nhat"
)
for h in (0.60, 0.66, 0.70, 0.73, 0.76, 0.81, 0.86):
    a = Sg1[h] / bG[h]
    b = Sg2[h] / bG[h]
    print(
        f"{h:.2f}  {bG[h]:.6e}   {a:12.5f}   {b:12.5f}   {nhat[h]:12.5f}   {a / nhat[h]:10.4f}   {b / nhat[h]:10.4f}"
    )
a73 = Sg1[0.73] / bG[0.73]
b73 = Sg2[0.73] / bG[0.73]
print("\n LEVEL at truth h=0.73:")
print(
    f"   mass-BLIND catalogue side vs completeness side : {a73 / nhat[0.73]:.4f}   (agree to {abs(a73 / nhat[0.73] - 1) * 100:.1f} %)"
)
print(
    f"   mass-AWARE catalogue side vs completeness side : {b73 / nhat[0.73]:.4f}   (disagree by factor {nhat[0.73] / b73:.2f})"
)
sl_a = (Sg1[0.86] / bG[0.86] / nhat[0.86]) / (Sg1[0.60] / bG[0.60] / nhat[0.60]) - 1
sl_b = (Sg2[0.86] / bG[0.86] / nhat[0.86]) / (Sg2[0.60] / bG[0.60] / nhat[0.60]) - 1
print(
    f" H-SLOPE 0.60->0.86 of the ratio: mass-blind {sl_a * 100:+.2f} %, mass-aware {sl_b * 100:+.2f} %"
    "   (h^-3 volume Jacobian already removed by n_hat_w ∝ h^3; see exoneration note)"
)

# ---------------- channel-difference interaction --------------------------
df = pd.read_csv(f"{BASE}/seed61000/real_r1/diagnostics/event_likelihoods.csv")
df["h"] = df["h"].round(4)
crb = pd.read_csv(f"{BASE}/seed61000/prepared_cramer_rao_bounds.csv")
incat = set(crb.index[crb.host_galaxy_index >= 0])
H = np.array(sorted(df.h.unique()))
N = df.event_idx.nunique()
r = {h: Sg2[h] / Sg1[h] for h in hs}

piv = {
    c: df.pivot(index="event_idx", columns="h", values=c)
    for c in ("L_cat_no_bh", "L_cat_with_bh", "B_num")
}
idx = piv["L_cat_no_bh"].index.to_numpy()
is_in = np.isin(idx, list(incat))
z1 = (piv["L_cat_no_bh"].to_numpy() == 0).all(axis=1)
z2 = (piv["L_cat_with_bh"].to_numpy() == 0).all(axis=1)
print(
    f"\nCOMPLETION-ONLY events (L_cat == 0 at EVERY h): 1D {z1.sum()}/{N} ({z1.mean():.1%}), "
    f"2D {z2.sum()}/{N} ({z2.mean():.1%})   [dark-only: 1D {(z1 & ~is_in).sum()}, 2D {(z2 & ~is_in).sum()}]"
)
print("   -> for these events the (1-w_G) completion tilt is their ENTIRE h-dependence budget.")


def dnats(col, scale, mask):
    i73 = np.where(np.isclose(H, 0.73))[0][0]
    i81 = np.where(np.isclose(H, 0.81))[0][0]
    tot = 0.0
    for i in (i81, i73):
        h = H[i]
        bgc = bG[h] * (1.0 if scale is None else scale[h])
        pi = (bgc * piv[col].to_numpy()[:, i] + piv["B_num"].to_numpy()[:, i]) / (bgc + bGbar[h])
        tot = np.log(pi) if i == i81 else tot - np.log(pi)
    return float(tot[mask].sum())


print("\n  Delta ln p_i, h=0.73 -> 0.81, per class and channel (nats)")
print(
    "  variant                 |  1D in-cat   1D dark   1D tot |  2D in-cat  2D dark   2D tot | channel diff"
)
for name, sc in (("as delivered", None), ("beta_G -> r(h) beta_G", r)):
    a = [dnats("L_cat_no_bh", sc, is_in), dnats("L_cat_no_bh", sc, ~is_in)]
    b = [dnats("L_cat_with_bh", sc, is_in), dnats("L_cat_with_bh", sc, ~is_in)]
    print(
        f"  {name:23s} | {a[0]:+9.2f} {a[1]:+9.2f} {sum(a):+9.2f} | {b[0]:+9.2f} {b[1]:+9.2f} {sum(b):+9.2f} | {sum(b) - sum(a):+9.2f}"
    )
