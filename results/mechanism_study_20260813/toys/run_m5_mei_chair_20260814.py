import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m5_toy as t

t.load_sigd()
H = np.arange(0.630, 0.8301, 0.005)  # 41-point grid
SIG = 0.035


def mei_bias(K, seeds, n_ev=120):
    biases = []
    for s in seeds:
        R = t.draw(s, n_ev, K, SIG, scatter=False)  # all exact
        rng = np.random.default_rng(10_000 + s)
        z_obs = R["z_cand"].copy()
        z_obs[n_ev:] += SIG * rng.standard_normal(z_obs.size - n_ev)  # impostors only
        R["z_obs"] = z_obs
        ln = t.lnpost(R, H, SIG)
        biases.append(t.argmax_refined(H, ln) - t.H_TRUE)
    b = np.array(biases)
    return b.mean(), b.std(ddof=1) / np.sqrt(len(b))


for K in (50, 1216):
    m, se = mei_bias(K, range(8))
    print(f"K={K:5d}  MEI toy bias = {m:+.4f} +/- {se:.4f}")
