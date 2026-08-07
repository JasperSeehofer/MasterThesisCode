# ruff: noqa: B023
"""M-3: h-slope of f_k's contribution to L_cat under the D1 pool substitution.

Spec: CLAIM_HITCHHIKER_INDEPENDENCE_20260805.md.DRAFT lines 562-567.

Measures, per venue (iiib, joint_r1), the substitution-induced per-event movement

    Delta_i(h) = ln L_cat^{A2}(i, h) - ln L_cat^{A1}(i, h)

from results/run_20260805_d1/{a1,a2}_{venue}/diagnostics/event_likelihoods.csv,
and reports the H-SLOPE of that movement: the per-event chord

    chord_i(h_a, h_b) = Delta_i(h_b) - Delta_i(h_a)

over the grid endpoints (0.60 -> 0.86) and all pairs among {0.60, 0.73, 0.81, 0.86}.
The known 62-69% cell-movement number is a LEVEL statement; only a nonzero chord
(h-dependence of Delta) can tilt a posterior. Decision rule (pre-registered):
if the chord is <~ 1 nat/event, H-3 is bounded-null without any new run.

Channels: 1D (L_cat_no_bh, primary per spec) and 2D (L_cat_with_bh, supplementary).
Also computed as a cross-check: the same chords on combined_no_bh/combined_with_bh
(the actual per-event likelihoods entering the posterior), which are zero-free.

Read-only. No production instrument run. Output: m3_results.json in this directory.
"""

import json
import os

import numpy as np
import pandas as pd

BASE = "/home/jasper/Repositories/MasterThesisCode/results/run_20260805_d1"
OUT = (
    "/home/jasper/Repositories/MasterThesisCode/results/campaign51_20260728/"
    "realistic_20260729/crossterm_instrument/m3_results.json"
)

VENUES = ["iiib", "joint_r1"]
H_PROBE = [0.60, 0.73, 0.81, 0.86]
CHANNELS = {
    "1d_Lcat": "L_cat_no_bh",
    "2d_Lcat": "L_cat_with_bh",
    "1d_combined": "combined_no_bh",
    "2d_combined": "combined_with_bh",
}


def load_arm(venue: str, arm: str) -> pd.DataFrame:
    path = f"{BASE}/{arm}_{venue}/diagnostics/event_likelihoods.csv"
    df = pd.read_csv(path)
    assert len(df) == 65108, (path, len(df))
    return df


def pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """event_idx x h matrix of the column."""
    return df.pivot(index="event_idx", columns="h", values=col)


def stats(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    a = np.abs(x)
    return {
        "n": int(x.size),
        "median_abs": float(np.median(a)),
        "p90_abs": float(np.percentile(a, 90)),
        "max_abs": float(np.max(a)),
        "median_signed": float(np.median(x)),
        "mean_signed": float(np.mean(x)),
        "class_sum_signed": float(np.sum(x)),
        "class_sum_abs": float(np.sum(a)),
        "frac_gt_1nat": float(np.mean(a > 1.0)),
        "n_gt_1nat": int(np.sum(a > 1.0)),
    }


def nearest_h(h_grid: np.ndarray, h: float) -> float:
    i = int(np.argmin(np.abs(h_grid - h)))
    hv = float(h_grid[i])
    assert abs(hv - h) < 1e-9, (h, hv)
    return hv


results: dict = {
    "measurement": "M-3 h-slope of f_k contribution to L_cat under D1 pool substitution (A2 vs A1)",
    "spec": "CLAIM_HITCHHIKER_INDEPENDENCE_20260805.md.DRAFT lines 562-567",
    "definition": (
        "Delta_i(h) = ln(col^A2/col^A1); chord_i(ha,hb) = Delta_i(hb) - Delta_i(ha). "
        "Decision rule: chord <~ 1 nat/event => H-3 bounded-null."
    ),
    "inputs": {},
    "venues": {},
}

for venue in VENUES:
    a1 = load_arm(venue, "a1")
    a2 = load_arm(venue, "a2")
    results["inputs"][venue] = {
        "a1_csv": f"{BASE}/a1_{venue}/diagnostics/event_likelihoods.csv",
        "a2_csv": f"{BASE}/a2_{venue}/diagnostics/event_likelihoods.csv",
        "n_rows_each": [int(len(a1)), int(len(a2))],
        "n_events": int(a1["event_idx"].nunique()),
        "n_h": int(a1["h"].nunique()),
    }
    vres: dict = {"channels": {}}
    h_grid = np.sort(a1["h"].unique())
    h_lo, h_hi = float(h_grid[0]), float(h_grid[-1])
    probe = [nearest_h(h_grid, h) for h in H_PROBE if np.min(np.abs(h_grid - h)) < 1e-9]
    vres["h_grid_endpoints"] = [h_lo, h_hi]
    vres["h_probe_present"] = probe

    for ch_name, col in CHANNELS.items():
        m1 = pivot(a1, col)
        m2 = pivot(a2, col)
        assert m1.shape == m2.shape == (1588, 41), (ch_name, m1.shape)
        assert (m1.index == m2.index).all() and np.allclose(m1.columns, m2.columns)

        with np.errstate(divide="ignore", invalid="ignore"):
            delta = np.log(m2.values) - np.log(m1.values)  # events x h
        # defined only where both arms > 0
        valid = (m1.values > 0) & (m2.values > 0)
        delta = np.where(valid, delta, np.nan)

        hcols = np.asarray(m1.columns, dtype=float)

        def col_of(hval: float) -> np.ndarray:
            j = int(np.argmin(np.abs(hcols - hval)))
            return delta[:, j]

        ch: dict = {"column": col}
        # zero bookkeeping at endpoints
        j_lo = int(np.argmin(np.abs(hcols - h_lo)))
        j_hi = int(np.argmin(np.abs(hcols - h_hi)))
        ch["n_events_zero_either_arm_at_h_lo"] = int((~valid[:, j_lo]).sum())
        ch["n_events_zero_either_arm_at_h_hi"] = int((~valid[:, j_hi]).sum())

        # level of Delta at truth h=0.73 for context (level, NOT the decision quantity)
        ch["level_delta_at_h0.73"] = stats(col_of(0.73))

        # endpoint chord
        chord_ep = col_of(h_hi) - col_of(h_lo)
        ch["chord_endpoints_0.60_to_0.86"] = stats(chord_ep)

        # full-grid max range of Delta_i(h) across all 41 h (worst-case tilt)
        with np.errstate(invalid="ignore"):
            rng = np.nanmax(delta, axis=1) - np.nanmin(delta, axis=1)
        n_valid_h = valid.sum(axis=1)
        rng = np.where(n_valid_h >= 2, rng, np.nan)
        ch["max_range_over_full_grid"] = stats(rng)

        # pairwise chords among probe h values
        pair_chords = {}
        for i in range(len(probe)):
            for j in range(i + 1, len(probe)):
                ha, hb = probe[i], probe[j]
                pair_chords[f"{ha:.2f}->{hb:.2f}"] = stats(col_of(hb) - col_of(ha))
        ch["pairwise_chords"] = pair_chords

        vres["channels"][ch_name] = ch

    results["venues"][venue] = vres

# Decision: primary channel 1d_Lcat, endpoint chord + full-grid range, both venues.
decision = {}
for venue in VENUES:
    c = results["venues"][venue]["channels"]["1d_Lcat"]
    decision[venue] = {
        "endpoint_chord_median_abs": c["chord_endpoints_0.60_to_0.86"]["median_abs"],
        "endpoint_chord_p90_abs": c["chord_endpoints_0.60_to_0.86"]["p90_abs"],
        "endpoint_chord_max_abs": c["chord_endpoints_0.60_to_0.86"]["max_abs"],
        "full_grid_range_median": c["max_range_over_full_grid"]["median_abs"],
        "full_grid_range_p90": c["max_range_over_full_grid"]["p90_abs"],
        "full_grid_range_max": c["max_range_over_full_grid"]["max_abs"],
    }
results["decision_inputs_primary_1d_Lcat"] = decision

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results["decision_inputs_primary_1d_Lcat"], indent=2))
print("wrote", OUT)
