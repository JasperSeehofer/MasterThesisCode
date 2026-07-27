"""S1b — three-component mixtures a + cat + flat_um (issue #51 sizing).

Rationale (from s1 results): cat-component buys catalogue-weighted w_bar,
flat_um buys the grid-wide reachable ESS floor, measure a keeps the pool's
population marginal legs cheap (stratified sub-pool). Scan a few (alpha_a,
alpha_cat, alpha_flat) at the s1 N-scan. Same conventions as s1_sizing.py.
"""

import json
import sys

import numpy as np

sys.path.insert(0, "/home/jasper/Repositories/MasterThesisCode")
sys.path.insert(0, "/home/jasper/Repositories/MasterThesisCode/results/"
                   "lcat_h_dependence_20260725/campaign_sizing_20260728")

import s1_sizing as s1  # noqa: E402  (reuses density grids + machinery)

MIX3 = {
    "mix3_50_25_25": (0.50, 0.25, 0.25),
    "mix3_60_20_20": (0.60, 0.20, 0.20),
    "mix3_40_20_40": (0.40, 0.20, 0.40),
}

out = {}
for name, (aa, ac, af) in MIX3.items():
    rho = aa * s1.rho_a + ac * s1.rho_cat + af * s1.rho_flat
    print(f"=== {name} ===", flush=True)
    lg_s, z_s = s1.draw(rho, s1.N_DRAW)
    u_s = np.log1p(z_s)
    m_s = lg_s + np.log10(1.0 + z_s)
    mrec = {"alphas": {"a": aa, "cat": ac, "flat": af}, "by_N": {}}
    for n in s1.N_SCAN:
        u, m = u_s[:n], m_s[:n]
        ess, su, sm = s1.ess_grid(u, m, s1.U_NODES, s1.M_NODES)
        rec = {
            "sigma_u": su, "sigma_m": sm,
            "grid_min_ESS_reachable": float(ess[s1.node_reach].min()),
            "grid_frac_reachable_ESS_lt_100": float(
                (ess[s1.node_reach] < 100).mean()),
            "grid_frac_reachable_ESS_lt_500": float(
                (ess[s1.node_reach] < 500).mean()),
            "catalogue": s1.cat_metrics(ess, s1.U_NODES, s1.M_NODES),
        }
        mrec["by_N"][str(n)] = rec
        c = rec["catalogue"]["reachable"]
        print(f"  N={n}: wbar_reach={c['wbar']:.4f} med={c['median_ESS']:.0f} "
              f"minESS={rec['grid_min_ESS_reachable']:.1f} "
              f"fr<500={rec['grid_frac_reachable_ESS_lt_500']:.3f}", flush=True)
    out[name] = mrec

with open(f"{s1.OUT}/sizing_results_mix3.json", "w") as f:
    json.dump(out, f, indent=1)
print("done -> sizing_results_mix3.json")
