"""Gate (vii) proper read — post-fix dark-class catalogue-leg channel difference.

Task-queue item 1 of RUNBOOK_NEXT_SESSION_7.md §1. Target object:

    Sigma_dark Delta ln(L_cat^2D / L_cat^1D), 0.73 -> 0.81

evaluated under the `cellb_readout.py` / `attack_c3_c4.py` conventions
(results/campaign51_20260728/realistic_20260729/) over the NEW post-fix
41-h diagnostics CSVs:

  - run_20260804_postfix_iiib      (idealized venue)
  - run_20260804_postfix_joint_r1  (joint/observed venue)

COMPUTE ONLY. No interpretation added here.

Conventions extracted from attack_c3_c4.py (the C3/C4 provenance-repair
script) and cellb_readout.py:

  - CATALOGUE-LEG likelihood columns: `L_cat_no_bh` (1D catalogue leg) and
    `L_cat_with_bh` (2D catalogue leg). These are NOT `combined_no_bh` /
    `combined_with_bh` (the whole-mixture columns = catalogue leg + B_num
    completion leg, mixed by w_G) — the runbook explicitly flags the
    whole-mixture read (+6.45 joint / +8.15 idealized) as a DIFFERENT,
    cruder object. This script uses the catalogue-leg-only columns.
  - Class definition: `incat` = event_idx values with
    `host_galaxy_index >= 0` in prepared_cramer_rao_bounds.csv (row index
    == event_idx, per attack_c3_c4.py). `dark` = complement.
  - Object construction (attack_c3_c4.py C4 block): restrict to events with
    BOTH L_cat_no_bh > 0 AND L_cat_with_bh > 0 at BOTH h=0.73 and h=0.81
    (exclude any event with a zero catalogue-leg likelihood at either h in
    either channel -- these are events with no catalogue-consistent host
    at that h, log-ratio undefined). For survivors:
        ratio(h) = ln(L_cat_with_bh(h) / L_cat_no_bh(h))
        Sigma_class Delta = sum_class ratio(0.81) - sum_class ratio(0.73)
  - Per-h profile (this script's extension, cheap): the same log-ratio
    summed over the class at EACH of the 41 h, plus its value relative to
    h=0.73, using the SAME per-event survivor mask (nonzero in both
    channels at ALL 41 h) so the profile is on a fixed, consistent event
    set across h -- otherwise the survivor set would change with h and the
    profile would not be a clean function.

Run: .venv/bin/python results/run_20260804_postfix/gate_vii/compute_gate_vii.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent

VENUES = {
    "iiib_idealized": REPO / "results/run_20260804_postfix/iiib/diagnostics",
    "joint_r1_observed": REPO / "results/run_20260804_postfix/joint_r1/diagnostics",
}

H_LO = 0.73
H_HI = 0.81
H_REPORT = (0.60, 0.73, 0.78, 0.81, 0.86)


def load_incat(crb_path: Path) -> set[int]:
    crb = pd.read_csv(crb_path)
    return set(crb.index[crb.host_galaxy_index >= 0])


def compute_venue(name: str, d: Path) -> dict[str, Any]:
    ev_path = d / "event_likelihoods.csv"
    crb_path = d / "prepared_cramer_rao_bounds.csv"
    ev = pd.read_csv(ev_path)

    fingerprint = {
        "csv_path": str(ev_path),
        "n_rows": int(len(ev)),
        "n_events": int(ev.event_idx.nunique()),
        "n_h_values": int(ev.h.unique().size),
        "columns": list(ev.columns),
    }

    incat = load_incat(crb_path)
    n_events_total = ev.event_idx.nunique()
    all_events = set(ev.event_idx.unique())
    incat_present = incat & all_events
    dark_present = all_events - incat

    # --- Fixed survivor mask: nonzero L_cat_no_bh AND L_cat_with_bh at
    #     EVERY h (so the per-h profile is over a consistent event set).
    g = ev.groupby("event_idx").agg(
        min_no_bh=("L_cat_no_bh", "min"),
        min_with_bh=("L_cat_with_bh", "min"),
        any_nan_no_bh=("L_cat_no_bh", lambda s: bool(s.isna().any())),
        any_nan_with_bh=("L_cat_with_bh", lambda s: bool(s.isna().any())),
    )
    survivors_allh = g.index[
        (g.min_no_bh > 0)
        & (g.min_with_bh > 0)
        & (~g.any_nan_no_bh)
        & (~g.any_nan_with_bh)
    ]
    survivors_allh = set(survivors_allh)
    excluded_allh = all_events - survivors_allh
    n_excluded_allh = len(excluded_allh)
    n_excluded_allh_incat = len(excluded_allh & incat)
    n_excluded_allh_dark = len(excluded_allh & dark_present)

    # --- Headline object: 0.73 -> 0.81 only, attack_c3_c4.py's own (looser)
    #     survivor definition -- nonzero in both channels at 0.73 AND 0.81
    #     only (not necessarily at all 41 h). Computed alongside the
    #     fixed-survivor version for cross-check; both reported.
    at_lo = ev[np.isclose(ev.h, H_LO)].set_index("event_idx")
    at_hi = ev[np.isclose(ev.h, H_HI)].set_index("event_idx")
    common_idx = at_lo.index.intersection(at_hi.index)
    at_lo = at_lo.loc[common_idx]
    at_hi = at_hi.loc[common_idx]

    ok_pairwise = (
        (at_lo.L_cat_with_bh > 0)
        & (at_lo.L_cat_no_bh > 0)
        & (at_hi.L_cat_with_bh > 0)
        & (at_hi.L_cat_no_bh > 0)
        & at_lo.L_cat_with_bh.notna()
        & at_lo.L_cat_no_bh.notna()
        & at_hi.L_cat_with_bh.notna()
        & at_hi.L_cat_no_bh.notna()
    )
    ratio_lo_pairwise = np.log(at_lo.L_cat_with_bh[ok_pairwise] / at_lo.L_cat_no_bh[ok_pairwise])
    ratio_hi_pairwise = np.log(at_hi.L_cat_with_bh[ok_pairwise] / at_hi.L_cat_no_bh[ok_pairwise])
    isin_pairwise = ratio_lo_pairwise.index.isin(incat)

    def class_result(ratio_lo: pd.Series, ratio_hi: pd.Series, isin: np.ndarray) -> dict[str, Any]:
        out = {}
        for label, mask in (("dark", ~isin), ("incat", isin)):
            s_lo = float(ratio_lo[mask].sum())
            s_hi = float(ratio_hi[mask].sum())
            out[label] = {
                "n_events": int(mask.sum()),
                f"sum_ln_ratio_at_{H_LO}": s_lo,
                f"sum_ln_ratio_at_{H_HI}": s_hi,
                "delta": s_hi - s_lo,
            }
        return out

    headline_pairwise_survivors = class_result(ratio_lo_pairwise, ratio_hi_pairwise, isin_pairwise)

    # --- Fixed-survivor (all-41-h) version of the same headline object ---
    fixed_idx = sorted(survivors_allh & set(common_idx))
    at_lo_fixed = at_lo.loc[fixed_idx]
    at_hi_fixed = at_hi.loc[fixed_idx]
    ratio_lo_fixed = np.log(at_lo_fixed.L_cat_with_bh / at_lo_fixed.L_cat_no_bh)
    ratio_hi_fixed = np.log(at_hi_fixed.L_cat_with_bh / at_hi_fixed.L_cat_no_bh)
    isin_fixed = np.array([k in incat for k in fixed_idx])
    headline_fixed_survivors = class_result(ratio_lo_fixed, ratio_hi_fixed, isin_fixed)

    # --- Full per-h profile on the fixed survivor set ---
    hs = np.sort(ev.h.unique())
    piv_no_bh = ev.pivot(index="event_idx", columns="h", values="L_cat_no_bh").loc[fixed_idx]
    piv_with_bh = ev.pivot(index="event_idx", columns="h", values="L_cat_with_bh").loc[fixed_idx]
    ln_ratio_all_h = np.log(piv_with_bh) - np.log(piv_no_bh)  # events x h

    isin_fixed_series = pd.Series(isin_fixed, index=fixed_idx)
    profile = {}
    for label, mask in (("dark", ~isin_fixed_series.values), ("incat", isin_fixed_series.values)):
        s = ln_ratio_all_h.loc[np.array(fixed_idx)[mask]].sum(axis=0)
        s = s.reindex(hs)
        base_073 = float(s.iloc[int(np.argmin(np.abs(hs - 0.73)))])
        profile[label] = {
            "h": [float(x) for x in hs],
            "sum_ln_ratio": [float(x) for x in s.values],
            "delta_rel_h073": [float(x - base_073) for x in s.values],
        }
        argmax_h = float(s.idxmax())
        profile[label]["argmax_h"] = argmax_h
        profile[label]["value_at_argmax"] = float(s.max())

    # report-h snapshot table
    report_table = {}
    for label in ("dark", "incat"):
        s_h = profile[label]["h"]
        s_v = profile[label]["sum_ln_ratio"]
        s0 = dict(zip(s_h, s_v))
        row = {}
        base = s0[min(s_h, key=lambda x: abs(x - 0.73))]
        for hv in H_REPORT:
            nearest = min(s_h, key=lambda x: abs(x - hv))
            row[f"h={hv}"] = {
                "nearest_grid_h": nearest,
                "sum_ln_ratio": s0[nearest],
                "delta_rel_073": s0[nearest] - base,
            }
        report_table[label] = row

    return {
        "venue": name,
        "fingerprint": fingerprint,
        "class_counts": {
            "n_events_total": int(n_events_total),
            "n_incat": int(len(incat_present)),
            "n_dark": int(len(dark_present)),
        },
        "exclusions": {
            "n_excluded_allh_fixed_survivor_def": n_excluded_allh,
            "n_excluded_allh_incat": n_excluded_allh_incat,
            "n_excluded_allh_dark": n_excluded_allh_dark,
            "n_excluded_pairwise_073_081_def": int((~ok_pairwise).sum()),
            "note": (
                "'excluded' = event has L_cat_no_bh<=0 or L_cat_with_bh<=0 or NaN "
                "at the relevant h grid points (no catalogue-consistent host at "
                "that h for one or both channels); these events contribute 0 to "
                "the log-ratio sum by construction (undefined log(0) or log(x/0))."
            ),
        },
        "headline_0p73_to_0p81": {
            "pairwise_survivors_073_081_only": headline_pairwise_survivors,
            "fixed_survivors_all_41h": headline_fixed_survivors,
        },
        "per_h_profile_fixed_survivors": profile,
        "report_h_snapshot": report_table,
    }


def main() -> None:
    results = {}
    for name, d in VENUES.items():
        print(f"=== {name} ===")
        r = compute_venue(name, d)
        results[name] = r
        fp = r["fingerprint"]
        print(f"  rows={fp['n_rows']} events={fp['n_events']} h_values={fp['n_h_values']}")
        print(f"  columns={fp['columns']}")
        cc = r["class_counts"]
        print(f"  classes: total={cc['n_events_total']} incat={cc['n_incat']} dark={cc['n_dark']}")
        exc = r["exclusions"]
        print(
            f"  exclusions: fixed-survivor {exc['n_excluded_allh_fixed_survivor_def']} "
            f"(incat {exc['n_excluded_allh_incat']}, dark {exc['n_excluded_allh_dark']}); "
            f"pairwise-73-81 {exc['n_excluded_pairwise_073_081_def']}"
        )
        hp = r["headline_0p73_to_0p81"]["pairwise_survivors_073_081_only"]
        hf = r["headline_0p73_to_0p81"]["fixed_survivors_all_41h"]
        print(
            f"  HEADLINE dark Sigma Delta ln(L_cat_2D/L_cat_1D) 0.73->0.81 "
            f"(pairwise-survivor def): {hp['dark']['delta']:.7g} nats "
            f"(N={hp['dark']['n_events']})"
        )
        print(
            f"  HEADLINE dark Sigma Delta ln(L_cat_2D/L_cat_1D) 0.73->0.81 "
            f"(fixed-all-41h-survivor def): {hf['dark']['delta']:.7g} nats "
            f"(N={hf['dark']['n_events']})"
        )
        print(
            f"  in-cat analogue (pairwise-survivor def): {hp['incat']['delta']:.7g} nats "
            f"(N={hp['incat']['n_events']})"
        )
        print(
            f"  in-cat analogue (fixed-all-41h-survivor def): {hf['incat']['delta']:.7g} nats "
            f"(N={hf['incat']['n_events']})"
        )
        prof = r["per_h_profile_fixed_survivors"]
        print(
            f"  dark profile argmax: h={prof['dark']['argmax_h']:.3f}, "
            f"value={prof['dark']['value_at_argmax']:.7g}"
        )
        print(
            f"  incat profile argmax: h={prof['incat']['argmax_h']:.3f}, "
            f"value={prof['incat']['value_at_argmax']:.7g}"
        )
        print()

    out_path = OUT_DIR / "gate_vii_readout.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
