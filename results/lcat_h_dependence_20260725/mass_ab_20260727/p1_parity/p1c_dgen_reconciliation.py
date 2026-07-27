"""P1 parity audit, part C — D_gen-axis reconciliation against the B-cell.

The B‴−B″ diagnostics give the flag's effect on the ACTUAL generator-stack
normalisation per event: shift(h) = −Δ_flag ln D_gen(h), constant across all
3454 events at 1e-14 (measured in this audit). Combined with the cluster
Σ_glob_wbh tables (flag off/on), that yields the MEASURED in-catalogue
share P_cat(h) = Δ_flag ln D_gen(h) / Δ_flag ln Σ(h) of the production
B-stack, to be compared with the z3 table assembly's P_cat(h) =
(Σ/n̂_w)/D_gen built from generator_norm_Dgen_table.json n̂_w +
z2_results.json β_Ḡ(zres). The comparison decides whether the −6.5 gate's
D_gen-axis arithmetic (not just its A-cell transfer) was miscalibrated.

Writes p1c_results.json. Cheap: no kernel builds, tables only.
"""

import json
import sys

import numpy as np

REPO = "/home/jasper/Repositories/MasterThesisCode"
BASE = f"{REPO}/results/lcat_h_dependence_20260725"
HERE = f"{BASE}/mass_ab_20260727/p1_parity"
sys.path.insert(0, REPO)

N_EVENTS = 3454
H_VENUE = [0.60, 0.65, 0.70, 0.73, 0.76, 0.80, 0.86]

# cluster-measured Σ tables (provenance: p1a header)
SIG_OFF = {0.60: 2.700808e8, 0.65: 2.759635e8, 0.70: 2.815376e8, 0.73: 2.847445e8,
           0.76: 2.878574e8, 0.80: 2.918739e8, 0.86: 2.976373e8}
SIG_ON = {0.60: 2.767679e8, 0.65: 2.826148e8, 0.70: 2.885021e8, 0.73: 2.920316e8,
          0.76: 2.955726e8, 0.80: 3.003153e8, 0.86: 3.074504e8}

# B‴−B″ measured per-event ln shifts, extracted from the diagnostics
# (identical for the 1D and 2D channels and constant across all 3454
# events at std < 1e-13 — asserted below).
import pandas as pd  # noqa: E402

_b = pd.read_csv(f"{BASE}/mass_ab_20260727/cellBpp/simulations/diagnostics/event_likelihoods.csv")
_f = pd.read_csv(f"{BASE}/mass_ab_20260727/zmzBpp/simulations/diagnostics/event_likelihoods.csv")
_m = _b.merge(_f, on=["event_idx", "h"], suffixes=("_b", "_f"))
SHIFT_B = {}
for _h in H_VENUE:
    _mm = _m[np.isclose(_m.h, _h)]
    _d1 = np.log(_mm.combined_no_bh_f / _mm.combined_no_bh_b)
    _d2 = np.log(_mm.combined_with_bh_f / _mm.combined_with_bh_b)
    assert float(np.std(_d1)) < 1e-12 and float(np.std(_d2)) < 1e-12
    assert abs(float(np.median(_d1) - np.median(_d2))) < 1e-12
    SHIFT_B[_h] = float(np.median(_d1))

tabj = json.load(open(f"{BASE}/generator_norm_Dgen_table.json"))
h_grid = np.array(tabj["h"])
n_hat_w = np.array(tabj["n_hat_w"])
z2res = json.load(open(f"{BASE}/zres_survival/z2_results.json"))
bg_zres = np.array(z2res["beta_Gbar_zres_table"])
z3res = json.load(open(f"{BASE}/zres_survival/z3_results.json"))

hv_idx = {h: int(np.argmin(np.abs(h_grid - h))) for h in H_VENUE}

rows = {}
for h in H_VENUE:
    j = hv_idx[h]
    dln_sig_flag = float(np.log(SIG_ON[h] / SIG_OFF[h]))
    dln_dgen_meas = -SHIFT_B[h]
    p_cat_meas = dln_dgen_meas / dln_sig_flag
    # z3-style table assembly share at this h, on BOTH arms
    p_cat_tab_off = (SIG_OFF[h] / n_hat_w[j]) / (SIG_OFF[h] / n_hat_w[j] + bg_zres[j])
    p_cat_tab_on = (SIG_ON[h] / n_hat_w[j]) / (SIG_ON[h] / n_hat_w[j] + bg_zres[j])
    # exact table-assembly flag shift of ln D_gen
    dln_dgen_tab = float(
        np.log((SIG_ON[h] / n_hat_w[j] + bg_zres[j]) / (SIG_OFF[h] / n_hat_w[j] + bg_zres[j]))
    )
    rows[h] = {
        "dln_sigma_flag": dln_sig_flag,
        "dln_dgen_measured_B": dln_dgen_meas,
        "dln_dgen_table_assembly": dln_dgen_tab,
        "P_cat_measured_B": p_cat_meas,
        "P_cat_table_off": float(p_cat_tab_off),
        "P_cat_table_on": float(p_cat_tab_on),
    }

# increments 0.73 -> 0.86 on the B (D_gen) axis
inc_meas = -N_EVENTS * (rows[0.86]["dln_dgen_measured_B"] - rows[0.73]["dln_dgen_measured_B"])
inc_tab = -N_EVENTS * (rows[0.86]["dln_dgen_table_assembly"] - rows[0.73]["dln_dgen_table_assembly"])

res = {
    "per_h": {str(h): rows[h] for h in H_VENUE},
    "Bcell_increment_073_086_measured": float(inc_meas),
    "Bcell_increment_073_086_table_assembly_with_production_sigmas": float(inc_tab),
    "z3_gate_increment_probe_sigmas": z3res["production_axis_gaps"][
        "increment_Mzonly_to_shrunk_joint"
    ],
    "note": (
        "inc_measured is the actual B-cell 1D flag increment (readout +0.45 = "
        "-N*(dln_dgen_meas(0.86)-dln_dgen_meas(0.73)) with signs as stored); "
        "inc_table uses the SAME production Sigma tables but the z2/z3 "
        "n_hat_w + beta_Gbar(zres) assembly — the difference is pure "
        "assembly miscalibration of the D_gen axis, independent of any "
        "probe-vs-production Sigma parity question."
    ),
}
with open(f"{HERE}/p1c_results.json", "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1))
