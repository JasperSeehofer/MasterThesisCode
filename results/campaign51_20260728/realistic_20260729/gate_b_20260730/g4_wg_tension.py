"""Gate C items 1 + 4.

(a) Verify the mixture-leg identities in the diagnostics CSV against the per-h logs.
(b) Strong-tension test: modelled w_G(0.73) vs the realized detected in-catalogue rate.
(c) Decompose the tension: F (pre-detection, exact by construction) vs the
    detection-efficiency ratio, and test it against the catalogue's own
    mass-aware/mass-blind selection-sum ratio r(h) = Sum_wbh / Sum_nobh.
(d) Counterfactual posteriors with beta_G -> r(h) beta_G (catalogue-anchored,
    mass-aware in-catalogue selection normalisation), both channels.
"""

import json
import re

import numpy as np
import pandas as pd

BASE = "results/campaign51_20260728/realistic_20260729"
OUT = f"{BASE}/gate_b_20260730"

# ---------- per-h legs from the logs ----------
D, bGbar, wGlog, Sg1, Sg2 = {}, {}, {}, {}, {}
for line in open(f"{BASE}/seed61000/mixture_leg_log_extract.txt"):
    m = re.search(r"D\(h=([\d.]+)\) = ([\d.e+-]+)", line)
    if m:
        D[round(float(m.group(1)), 4)] = float(m.group(2))
    m = re.search(r"beta_Gbar\(h=([\d.]+)\) = ([\d.e+-]+)", line)
    if m:
        bGbar[round(float(m.group(1)), 4)] = float(m.group(2))
    m = re.search(
        r"h_0_(\d+)\.log.*w_G=beta_G/D\(h\)=([\d.]+), sum_w_Dg\(no_bh\)=([\d.e+-]+), sum_w_Dg\(with_bh\)=([\d.e+-]+)",
        line,
    )
    if m:
        h = round(float("0." + m.group(1)), 4)
        wGlog[h] = float(m.group(2))
        Sg1[h] = float(m.group(3))
        Sg2[h] = float(m.group(4))
hs = np.array(sorted(D))
betaG = {h: D[h] - bGbar[h] for h in hs}
wG = {h: betaG[h] / D[h] for h in hs}
r = {h: Sg2[h] / Sg1[h] for h in hs}

# ---------- diagnostics CSV ----------
df = pd.read_csv(f"{BASE}/seed61000/real_r1/diagnostics/event_likelihoods.csv")
df["h"] = df["h"].round(4)
N = df["event_idx"].nunique()
H = np.array(sorted(df["h"].unique()))
print(f"diagnostics: {len(df)} rows, {N} events x {len(H)} h")
assert set(H) <= set(hs), set(H) - set(hs)

# (a) identity checks at h=0.73
s = df[df.h == 0.73]
bg, d, bb = betaG[0.73], D[0.73], bGbar[0.73]
rec = (bg * s.L_cat_no_bh + s.B_num) / d
print(
    f"(a) max rel err combined_no_bh  vs (beta_G L_cat + B_num)/D : {np.max(np.abs(rec / s.combined_no_bh - 1)):.3e}"
)
rec2 = (bg * s.L_cat_with_bh + s.B_num) / d
print(
    f"    max rel err combined_with_bh                            : {np.max(np.abs(rec2 / s.combined_with_bh - 1)):.3e}"
)
print(
    f"    max rel err L_comp vs B_num/beta_Gbar                    : {np.max(np.abs((s.B_num / bb) / s.L_comp - 1)):.3e}"
)
print(
    f"    w_G csv {s.w_G.iloc[0]:.9f} vs beta_G/D {wG[0.73]:.9f}   (rel {s.w_G.iloc[0] / wG[0.73] - 1:.2e})"
)

# (b) tension
crb = pd.read_csv(f"{BASE}/seed61000/prepared_cramer_rao_bounds.csv")
crb2 = pd.read_csv(f"{BASE}/seed62000/prepared_cramer_rao_bounds.csv")
n1, k1 = len(crb), int((crb.host_galaxy_index >= 0).sum())
n2, k2 = len(crb2), int((crb2.host_galaxy_index >= 0).sum())
p = wG[0.73]
for tag, (n, k) in (
    ("seed61000", (n1, k1)),
    ("seed62000", (n2, k2)),
    ("pooled", (n1 + n2, k1 + k2)),
):
    exp = n * p
    z = (k - exp) / np.sqrt(exp * (1 - p))
    print(
        f"(b) {tag}: {k}/{n} = {k / n:.5f}   modelled w_G(0.73) = {p:.6f} -> expected {exp:.1f}, binomial z = {z:+.2f}"
    )

# (c) decomposition
F = json.load(open(f"{OUT}/g2_catalogue_summary.json"))["F"]
oddsF = F / (1 - F)
oddsM = p / (1 - p)
print(f"\n(c) F (pre-detection in-cat fraction, h=0.73) = {F:.6f}, odds {oddsF:.6f}")
print(
    f"    modelled detected odds w_G/(1-w_G)        = {oddsM:.6f}  -> model eps_cat/eps_dark = {oddsM / oddsF:.4f}"
)
for tag, (n, k) in (
    ("seed61000", (n1, k1)),
    ("seed62000", (n2, k2)),
    ("pooled", (n1 + n2, k1 + k2)),
):
    oddsE = k / (n - k)
    print(
        f"    {tag}: realized odds {oddsE:.6f} -> eps_cat/eps_dark = {oddsE / oddsF:.4f}"
        f" -> eps_cat/eps_hat_cat = {(oddsE / oddsF) / (oddsM / oddsF):.4f} +- {(oddsE / oddsM) / np.sqrt(k):.4f}"
    )
print(
    f"    catalogue mass-aware/mass-blind selection ratio r(0.73) = Sum_wbh/Sum_nobh = {r[0.73]:.5f}"
)
print("    r(h): " + " ".join(f"{h:.3f}:{r[h]:.4f}" for h in (0.60, 0.66, 0.73, 0.78, 0.81, 0.86)))


# (d) counterfactual posteriors
def combine(col, scale=None):
    """log posterior over h from per-event p_i, optionally rescaling beta_G by scale(h)."""
    lp = np.zeros(len(H))
    for i, h in enumerate(H):
        sh = df[df.h == h]
        if scale is None:
            pi = sh[col].to_numpy()
        else:
            bgc = betaG[h] * scale[h]
            pi = (
                bgc * sh[("L_cat_no_bh" if col == "combined_no_bh" else "L_cat_with_bh")].to_numpy()
                + sh.B_num.to_numpy()
            ) / (bgc + bGbar[h])
        lp[i] = np.sum(np.log(pi))
    return lp - lp.max()


def summarize(name, lp):
    w = np.exp(lp)
    w /= np.trapezoid(w, H)
    maph = H[np.argmax(lp)]
    mean = np.trapezoid(H * w, H)
    print(
        f"    {name:34s} MAP {maph:.3f}  mean {mean:.4f}  "
        f"lnP(0.81)-lnP(0.73) = {lp[np.isclose(H, 0.81)][0] - lp[np.isclose(H, 0.73)][0]:+8.2f}"
    )
    return maph, mean


print("\n(d) counterfactual: beta_G -> r(h)*beta_G   (catalogue-anchored, mass-aware)")
res = {}
for col in ("combined_no_bh", "combined_with_bh"):
    res[col + "_base"] = summarize(col + "  [as delivered]", combine(col))
    res[col + "_corr"] = summarize(col + "  [beta_G*r(h)]", combine(col, scale=r))


# amplifier isolation: N*dln(1-w_G)
def tilt(scale=None):
    out = {}
    for h in (0.73, 0.81):
        bg = betaG[h] * (1 if scale is None else scale[h])
        out[h] = 1 - bg / (bg + bGbar[h])
    return N * np.log(out[0.81] / out[0.73])


print(f"\n    N*Dln(1-w_G) 0.73->0.81, as delivered : {tilt():+8.2f} nats")
print(
    f"    N*Dln(1-w_G) 0.73->0.81, beta_G*r(h)  : {tilt(r):+8.2f} nats   "
    f"(difference {tilt(r) - tilt():+.2f} nats -> Dh ~ {(tilt(r) - tilt()) * 4.9e-3:+.4f})"
)
json.dump({k: list(v) for k, v in res.items()}, open(f"{OUT}/g4_results.json", "w"), indent=1)

# --- (e) posterior widths + curves for the adjudicator ---
print("\n(e) posterior sd and curves")
curves = {}
for col in ("combined_no_bh", "combined_with_bh"):
    for tag, sc in (("base", None), ("corr", r)):
        lp = combine(col, sc)
        w = np.exp(lp)
        w /= np.trapezoid(w, H)
        mu = np.trapezoid(H * w, H)
        sd = np.sqrt(np.trapezoid((H - mu) ** 2 * w, H))
        print(f"    {col:17s} {tag}: MAP {H[np.argmax(lp)]:.3f} mean {mu:.4f} sd {sd:.4f}")
        curves[f"{col}_{tag}"] = lp.tolist()
curves["h"] = H.tolist()
json.dump(curves, open(f"{OUT}/g4_posterior_curves.json", "w"))
