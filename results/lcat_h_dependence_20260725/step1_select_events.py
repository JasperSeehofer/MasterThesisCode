"""Step 1 — select ~10 representative HOST-FOUND events for the L_cat(h) decomposition.

Selection basis: the shipped EXP-40 per-event diagnostics
(results/campaign_phase2_runs/run_20260719_seed1000_exp40/simulations/diagnostics/
event_likelihoods.csv, 3454 events x 41 h) — L_cat_no_bh(h) per event — plus the
injected redshift z_inj = dist_to_redshift(d_L, h=0.73) from the venue's prepared
CRB CSV.

Criteria:
- host-found events only (L_cat_no_bh > 0 at all h)
- span injected z bins
- include strong rail contributors (large positive Delta log L_cat(0.60 -> 0.86))
  and a few near-neutral / anti-rail events.

Writes selected_events.json with per-event metadata and the shipped L_cat(h) curves
(the validation target for the instrumented recomputation).
"""

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jasper/Repositories/MasterThesisCode")
from master_thesis_code.physical_relations import dist_to_redshift  # noqa: E402

OUT = "/home/jasper/Repositories/MasterThesisCode/results/lcat_h_dependence_20260725"
RUN = "/home/jasper/Repositories/MasterThesisCode/results/campaign_phase2_runs/run_20260719_seed1000_exp40"
VENUE = "/home/jasper/Repositories/MasterThesisCode/results/campaign_phase2_runs/run_20260703_seed1000"

diag = pd.read_csv(f"{RUN}/simulations/diagnostics/event_likelihoods.csv")
crb = pd.read_csv(f"{VENUE}/simulations/prepared_cramer_rao_bounds.csv")

h_grid = sorted(diag["h"].unique())
assert len(h_grid) == 41, len(h_grid)

# pivot: L_cat_no_bh per event across h
piv = diag.pivot_table(index="event_idx", columns="h", values="L_cat_no_bh")
piv = piv[sorted(piv.columns)]
host_found = piv[(piv > 0).all(axis=1)]
print(f"host-found events (L_cat>0 at all h): {len(host_found)}")

logL = np.log(host_found.values)
h_arr = np.array(sorted(piv.columns))
i060, i073, i086 = 0, int(np.argmin(np.abs(h_arr - 0.73))), len(h_arr) - 1
dlog_rail = logL[:, i060] - logL[:, i086]  # >0 == rail-driving (low tilt)
argmax_h = h_arr[np.argmax(logL, axis=1)]

d_L = crb["luminosity_distance"].values
sig = np.sqrt(crb["delta_luminosity_distance_delta_luminosity_distance"].values)
z_inj_all = np.array([dist_to_redshift(d, h=0.73) for d in d_L])

ev = pd.DataFrame(
    {
        "event_idx": host_found.index,
        "dlog_rail": dlog_rail,
        "dlog_060_073": logL[:, i060] - logL[:, i073],
        "argmax_h": argmax_h,
        "z_inj": z_inj_all[host_found.index],
        "d_L": d_L[host_found.index],
        "d_L_relerr": (sig / d_L)[host_found.index],
    }
).set_index("event_idx")

print(ev["dlog_rail"].describe())
print("frac argmax at 0.60:", (ev["argmax_h"] == h_arr[0]).mean())

# --- selection ---
sel: dict[int, str] = {}
zbins = [(0.0, 0.15), (0.15, 0.30), (0.30, 0.45), (0.45, 1.5)]
# strongest rail drivers per z bin (up to 1 each + 2 extra from the global top)
for zlo, zhi in zbins:
    sub = ev[(ev.z_inj >= zlo) & (ev.z_inj < zhi)]
    if len(sub):
        idx = sub["dlog_rail"].idxmax()
        sel[int(idx)] = f"strong_rail z[{zlo},{zhi})"
for idx in ev["dlog_rail"].nlargest(6).index:
    if int(idx) not in sel and len([k for k, v in sel.items() if "strong" in v]) < 6:
        sel[int(idx)] = "strong_rail global_top"
# median-tilt events per z bin
for zlo, zhi in zbins[:3]:
    sub = ev[(ev.z_inj >= zlo) & (ev.z_inj < zhi) & (~ev.index.isin(sel))]
    if len(sub):
        med = sub["dlog_rail"].median()
        idx = (sub["dlog_rail"] - med).abs().idxmin()
        sel[int(idx)] = f"median_tilt z[{zlo},{zhi})"
# near-neutral / anti-rail
neutral = ev[~ev.index.isin(sel)].reindex(
    ev[~ev.index.isin(sel)]["dlog_rail"].abs().nsmallest(2).index
)
for idx in neutral.index:
    sel[int(idx)] = "near_neutral"
anti = ev[~ev.index.isin(sel)]["dlog_rail"].idxmin()
sel[int(anti)] = "anti_rail_extreme"

out = {
    "h_grid": [float(x) for x in h_arr],
    "n_host_found": int(len(host_found)),
    "events": {},
}
for idx, role in sorted(sel.items()):
    row = ev.loc[idx]
    out["events"][str(idx)] = {
        "role": role,
        "z_inj": float(row.z_inj),
        "d_L": float(row.d_L),
        "d_L_relerr": float(row.d_L_relerr),
        "argmax_h": float(row.argmax_h),
        "dlog_rail_060_086": float(row.dlog_rail),
        "shipped_L_cat_no_bh": [float(v) for v in host_found.loc[idx].values],
    }
    print(
        f"event {idx:5d}  {role:28s} z_inj={row.z_inj:.3f} d_L={row.d_L:.3f} "
        f"argmax={row.argmax_h:.2f} dlogR={row.dlog_rail:+9.2f}"
    )

with open(f"{OUT}/selected_events.json", "w") as f:
    json.dump(out, f, indent=1)
print(f"\nwrote {OUT}/selected_events.json ({len(sel)} events)")
