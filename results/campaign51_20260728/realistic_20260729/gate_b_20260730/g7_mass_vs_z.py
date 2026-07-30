"""Why P(in-cat|detected,z) collapses above z~0.17: catalogue Malmquist selection
anti-correlates with EMRI detectability (higher-z catalogue galaxies are brighter,
hence heavier BHs, hence WORSE LISA EMRI hosts)."""

import numpy as np

OUT = "results/campaign51_20260728/realistic_20260729/gate_b_20260730"
d = np.load(f"{OUT}/g2_zshape.npz")
zg, Mg, w = d["zg"], d["Mg"], d["w_g"]
lg = np.log10(Mg)
edges = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.8, 1.5]
print(" z bin        N_gal    rate-weighted log10 M_BH quantiles (0.25/0.50/0.75)   frac(M>1e7)")
for i in range(len(edges) - 1):
    m = (zg >= edges[i]) & (zg < edges[i + 1])
    n = int(m.sum())
    if n < 10:
        print(f" {edges[i]:5.3f}-{edges[i + 1]:4.2f} {n:9d}   (too few)")
        continue
    lv = lg[m]
    ww = w[m]
    o = np.argsort(lv)
    cw = np.cumsum(ww[o]) / ww.sum()
    q = [np.interp(x, cw, lv[o]) for x in (0.25, 0.5, 0.75)]
    print(
        f" {edges[i]:5.3f}-{edges[i + 1]:4.2f} {n:9d}        {q[0]:6.3f}  {q[1]:6.3f}  {q[2]:6.3f}"
        f"                {ww[Mg[m] > 1e7].sum() / ww.sum():8.4f}"
    )
