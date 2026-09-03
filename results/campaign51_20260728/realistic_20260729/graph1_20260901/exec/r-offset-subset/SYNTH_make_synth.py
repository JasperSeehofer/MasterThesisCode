import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset")

# ---- covariate table (blind), 8 synthetic events ----
event_idx = list(range(8))
# S (iiib_2d) = {0,1,2}: constructed to have HIGH z_gw and be truth-hosted (C1) sometimes.
c1 = [True, False, False, True, False, False, True, False]           # in_catalog
c2 = [True, True, False, False, False, True, False, False]           # hosted_exact
c3 = [True, True, True, False, False, True, False, False]            # hosted_rel
c3c = [-1.0, -0.5, -2.0, -6.0, -7.0, -1.5, -8.0, -9.0]                # log10_f_cat
c4 = [0.90, 0.85, 0.80, 0.30, 0.25, 0.20, 0.15, 0.10]                 # z_gw -- S high, B low
c5 = [1.2, 1.1, 1.0, 0.5, 0.4, 0.6, 0.3, 0.2]                         # log10_sky_area
c6 = [0.9, 0.8, np.nan, 0.5, 0.6, 0.4, np.nan, 0.3]                   # mass_window_retention
c7 = [1.0, 0.9, 0.8, 0.3, 0.2, 0.4, 0.1, 0.05]                        # log10_n_cand_1d
c8 = [False, np.nan, np.nan, True, np.nan, np.nan, True, np.nan]      # cone_outside (C1 only)
c10 = [5.5, 5.4, 5.3, 5.0, 4.9, 5.1, 4.8, 4.7]                        # log10_M
c10b = [False] * 8                                                    # n<10 -> NOT-TESTED gate
c11 = [1.5, 1.4, 1.3, 0.9, 0.8, 1.0, 0.7, 0.6]                        # log10_snr, reported-only

df = pd.DataFrame(
    {
        "event_idx": event_idx,
        "C1": c1, "C2": c2, "C3": c3, "C3c": c3c, "C4": c4, "C5": c5,
        "C6": c6, "C7": c7, "C8": c8, "C10": c10, "C10b": c10b, "C11": c11,
    }
)
table_path = OUT / "SYNTH_covariate_table_blind.csv"
df.to_csv(table_path, index=False)
table_sha256 = hashlib.sha256(table_path.read_bytes()).hexdigest()
print("table sha256:", table_sha256)

# ---- influence vectors: 4 families' d_e/rank/in_S + primary logL_h* columns ----
h_grid = np.array([0.60, 0.665, 0.73, 0.795, 0.86])

# primary family iiib_2d: S = {0,1,2} (k=3). Rank by influence descending: 0>1>2>rest.
d_e_iiib2d = [0.02, 0.015, 0.01, 0.004, 0.003, 0.002, 0.001, 0.0005]
in_s_iiib2d = [True, True, True, False, False, False, False, False]

# replicate iiib_1d: k=3, SAME direction as primary for C4 (consistency)
d_e_iiib1d = [0.018, 0.013, 0.009, 0.005, 0.002, 0.001, 0.0008, 0.0003]
in_s_iiib1d = [True, True, True, False, False, False, False, False]

# replicate jr1_2d: k=3, also consistent
d_e_jr12d = [0.016, 0.011, 0.008, 0.004, 0.003, 0.0015, 0.0009, 0.0004]
in_s_jr12d = [True, True, True, False, False, False, False, False]

# replicate jr1_1d: k=2, only events 0,1 in S -- still same direction (subset)
d_e_jr11d = [0.02, 0.017, 0.003, 0.002, 0.0018, 0.001, 0.0007, 0.0002]
in_s_jr11d = [True, True, False, False, False, False, False, False]

infl = pd.DataFrame(
    {
        "event_idx": event_idx,
        "iiib_2d_d_e": d_e_iiib2d, "iiib_2d_rank": np.argsort(np.argsort(-np.array(d_e_iiib2d))) + 1,
        "iiib_2d_in_S": in_s_iiib2d,
        "iiib_1d_d_e": d_e_iiib1d, "iiib_1d_rank": np.argsort(np.argsort(-np.array(d_e_iiib1d))) + 1,
        "iiib_1d_in_S": in_s_iiib1d,
        "jr1_2d_d_e": d_e_jr12d, "jr1_2d_rank": np.argsort(np.argsort(-np.array(d_e_jr12d))) + 1,
        "jr1_2d_in_S": in_s_jr12d,
        "jr1_1d_d_e": d_e_jr11d, "jr1_1d_rank": np.argsort(np.argsort(-np.array(d_e_jr11d))) + 1,
        "jr1_1d_in_S": in_s_jr11d,
    }
)

# Primary-family (iiib_2d) per-event log-likelihood at each h-grid node, constructed so
# that events {0,1,2} (S, high z_gw) pull the posterior toward LOW h and the bulk toward
# h=0.73 -- i.e. removing the {0,1,2} stratum should move mean_h UP toward truth 0.73.
rng = np.random.default_rng(20260904)
logl = np.zeros((8, 5))
truth_idx = 2  # h=0.73
for e in range(8):
    if e in (0, 1, 2):
        # peaked at h=0.60 (index 0) -- these events pull the sample off-truth
        base = np.array([3.0, 1.0, -1.0, -3.0, -5.0])
    else:
        # peaked at h=0.73 (truth) -- the bulk
        base = np.array([-2.0, 1.0, 3.0, 1.0, -2.0])
    logl[e] = base + rng.normal(0, 0.05, size=5)
for i, col in enumerate(h_grid):
    infl[f"logL_h{col:.6f}"] = logl[:, i]

infl_path = OUT / "SYNTH_influence_vectors.csv"
infl.to_csv(infl_path, index=False)

print("wrote", table_path, infl_path)
print(json.dumps({"table_sha256": table_sha256}, indent=2))
