"""Mechanical data extraction for the frozen-g_frac counterfactual visualization.

COMPUTE ONLY. No interpretation added here.

Produces results/run_20260804_postfix/gate_vii/viz_data.json with, per venue
("iiib", "joint_r1"):

  1. h_grid                    -- sorted 41 h values (shared across venues).
  2. g_frac(h)                 -- per-h median g_frac curve + per-h min/max
                                   (near-scalar-across-events verification).
  3. logpost_2d(h)             -- Sigma_events ln(combined_with_bh), Delta rel max.
  4. logpost_2d_frozen(h)      -- frozen-g_frac counterfactual: replace g_frac(h)
                                   by g_frac(h=0.73) in the completion leg only,
                                   recombine, Sigma ln, Delta rel max.
  5. exp-normalized posteriors of (3) and (4) on the shared h grid.
  6. logpost_1d(h)             -- Sigma_events ln(combined_no_bh), Delta rel max.
  7. dark_catleg_delta(h)      -- fixed-41h-survivor Sigma_dark
                                   ln(L_cat_with_bh/L_cat_no_bh)(h) - same(0.73),
                                   reused from gate_vii_readout.json.
  8. Scalar summary block: N_events, N_dark, MAPs (full/frozen/1D), g_frac(0.73),
     g_frac range over the grid, Delta ln g_frac(0.60 -> 0.86).

Class split: dark = host_galaxy_index < 0 in
results/run_20260804_postfix/<venue>/diagnostics/prepared_cramer_rao_bounds.csv
(row index == event_idx).

Run: uv run python results/run_20260804_postfix/gate_vii/viz_data.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
READOUT_PATH = OUT_DIR / "gate_vii_readout.json"
OUT_PATH = OUT_DIR / "viz_data.json"

VENUES = {
    "iiib": REPO / "results/run_20260804_postfix/iiib/diagnostics",
    "joint_r1": REPO / "results/run_20260804_postfix/joint_r1/diagnostics",
}
# maps our short venue key -> the key used inside gate_vii_readout.json
READOUT_VENUE_KEY = {
    "iiib": "iiib_idealized",
    "joint_r1": "joint_r1_observed",
}

H_FROZEN = 0.73
SIG = 6  # significant figures for rounding


def _round(x: float, sig: int = SIG) -> float:
    if x == 0 or not np.isfinite(x):
        return float(x)
    from math import floor, log10

    d = sig - int(floor(log10(abs(x)))) - 1
    return float(round(x, d))


def _round_list(xs: list[float], sig: int = SIG) -> list[float]:
    return [_round(x, sig) for x in xs]


def load_dark_mask(crb_path: Path) -> set[int]:
    """dark = host_galaxy_index < 0; row index == event_idx (repo convention)."""
    crb = pd.read_csv(crb_path)
    return set(crb.index[crb.host_galaxy_index < 0])


def exp_normalize(delta_ln: np.ndarray) -> list[float]:
    """Sigma exp(delta_ln) normalized to 1 on the grid (already max-subtracted)."""
    w = np.exp(delta_ln)
    w = w / w.sum()
    return [float(x) for x in w]


def compute_venue(short_name: str, d: Path, readout: dict[str, Any]) -> dict[str, Any]:
    ev_path = d / "event_likelihoods.csv"
    crb_path = d / "prepared_cramer_rao_bounds.csv"
    ev = pd.read_csv(ev_path)

    n_events_total = int(ev.event_idx.nunique())
    dark_idx = load_dark_mask(crb_path)
    all_events = set(ev.event_idx.unique())
    dark_present = dark_idx & all_events
    n_dark = len(dark_present)

    hs = np.sort(ev.h.unique())
    n_h = len(hs)

    # ------------------------------------------------------------------
    # 2. g_frac(h): per-h median + min/max across events (near-scalar check)
    # ------------------------------------------------------------------
    g_by_h = ev.groupby("h")["g_frac"].agg(["median", "min", "max"]).reindex(hs)
    g_frac_median = g_by_h["median"].to_numpy()
    g_frac_min = g_by_h["min"].to_numpy()
    g_frac_max = g_by_h["max"].to_numpy()

    # g_frac(h=0.73), used to build the frozen counterfactual (per-event value,
    # not the median -- the counterfactual freezes each event's OWN g_frac at
    # its own h=0.73 value).
    h073_nearest = hs[np.argmin(np.abs(hs - H_FROZEN))]
    ev_073 = ev[np.isclose(ev.h, h073_nearest)].set_index("event_idx")
    g_frac_073_per_event = ev_073["g_frac"]  # Series indexed by event_idx

    g_frac_073_median = float(g_by_h.loc[h073_nearest, "median"])

    # ------------------------------------------------------------------
    # 3./4. Build the frozen combined_with_bh counterfactual, per event per h:
    #   combined_frozen = w_tilde_G * L_cat_with_bh + (1 - w_tilde_G) * L_comp * g_frac(h=0.73)
    # (replace ONLY the g_frac factor inside the completion leg by its
    #  h=0.73 value; everything else -- w_tilde_G, L_cat_with_bh, L_comp --
    #  stays at its own h.)
    # ------------------------------------------------------------------
    ev2 = ev.set_index("event_idx", drop=False)
    g073_aligned = g_frac_073_per_event.reindex(ev2.index).to_numpy()
    combined_frozen = (
        ev2["w_tilde_G"].to_numpy() * ev2["L_cat_with_bh"].to_numpy()
        + (1.0 - ev2["w_tilde_G"].to_numpy()) * ev2["L_comp"].to_numpy() * g073_aligned
    )
    ev2 = ev2.assign(combined_frozen=combined_frozen)

    # ------------------------------------------------------------------
    # ln(0) guard: drop events with combined==0 at ANY h, symmetrically for
    # both the full (combined_with_bh) and frozen curves, and for combined_no_bh.
    # ------------------------------------------------------------------
    bad_full = ev2.loc[ev2["combined_with_bh"] <= 0, "event_idx"].unique()
    bad_frozen = ev2.loc[ev2["combined_frozen"] <= 0, "event_idx"].unique()
    bad_2d = set(bad_full) | set(bad_frozen)
    n_dropped_2d = len(bad_2d)

    bad_1d = set(ev2.loc[ev2["combined_no_bh"] <= 0, "event_idx"].unique())
    n_dropped_1d = len(bad_1d)

    keep_2d_mask = ~ev2["event_idx"].isin(bad_2d)
    keep_1d_mask = ~ev2["event_idx"].isin(bad_1d)

    ev_2d = ev2.loc[keep_2d_mask]
    ev_1d = ev2.loc[keep_1d_mask]

    # Sigma ln over surviving events, per h
    logpost_2d_raw = (
        np.log(ev_2d["combined_with_bh"]).groupby(ev_2d["h"]).sum().reindex(hs).to_numpy()
    )
    logpost_2d_frozen_raw = (
        np.log(ev_2d["combined_frozen"]).groupby(ev_2d["h"]).sum().reindex(hs).to_numpy()
    )
    logpost_1d_raw = (
        np.log(ev_1d["combined_no_bh"]).groupby(ev_1d["h"]).sum().reindex(hs).to_numpy()
    )

    def to_delta_and_map(raw: np.ndarray) -> tuple[np.ndarray, float, float]:
        imax = int(np.argmax(raw))
        map_h = float(hs[imax])
        delta = raw - raw[imax]
        return delta, map_h, float(raw[imax])

    delta_2d, map_2d, val_2d_max = to_delta_and_map(logpost_2d_raw)
    delta_2d_frozen, map_2d_frozen, val_2d_frozen_max = to_delta_and_map(logpost_2d_frozen_raw)
    delta_1d, map_1d, val_1d_max = to_delta_and_map(logpost_1d_raw)

    post_2d = exp_normalize(delta_2d)
    post_2d_frozen = exp_normalize(delta_2d_frozen)

    # ------------------------------------------------------------------
    # 7. dark_catleg_delta(h): reuse from gate_vii_readout.json if present.
    # ------------------------------------------------------------------
    readout_key = READOUT_VENUE_KEY[short_name]
    dark_catleg_delta_h: list[float] | None = None
    dark_catleg_delta_vals: list[float] | None = None
    source = "recomputed"
    if readout and readout_key in readout:
        prof = readout[readout_key].get("per_h_profile_fixed_survivors", {}).get("dark")
        if prof is not None:
            dark_catleg_delta_h = prof["h"]
            dark_catleg_delta_vals = prof["delta_rel_h073"]
            source = "reused_from_gate_vii_readout"

    if dark_catleg_delta_h is None:
        # Fallback recompute: fixed-survivor Sigma_dark ln(L_cat_with_bh/L_cat_no_bh)(h)
        # minus the same at h=0.73, using events with nonzero L_cat_no_bh AND
        # L_cat_with_bh at every h (mirrors compute_gate_vii.py's convention).
        g = ev.groupby("event_idx").agg(
            min_no_bh=("L_cat_no_bh", "min"),
            min_with_bh=("L_cat_with_bh", "min"),
        )
        survivors = set(g.index[(g.min_no_bh > 0) & (g.min_with_bh > 0)])
        fixed_idx = sorted(survivors & dark_present)
        piv_no_bh = ev.pivot(index="event_idx", columns="h", values="L_cat_no_bh").loc[fixed_idx]
        piv_with_bh = ev.pivot(index="event_idx", columns="h", values="L_cat_with_bh").loc[
            fixed_idx
        ]
        ln_ratio = np.log(piv_with_bh) - np.log(piv_no_bh)
        s = ln_ratio.sum(axis=0).reindex(hs)
        base = float(s.iloc[int(np.argmin(np.abs(hs - H_FROZEN)))])
        dark_catleg_delta_h = [float(x) for x in hs]
        dark_catleg_delta_vals = [float(x - base) for x in s.values]

    # ------------------------------------------------------------------
    # 8. summary
    # ------------------------------------------------------------------
    g_frac_grid_min = float(np.min(g_frac_min))
    g_frac_grid_max = float(np.max(g_frac_max))
    h_lo_nearest = hs[np.argmin(np.abs(hs - 0.60))]
    h_hi_nearest = hs[np.argmin(np.abs(hs - 0.86))]
    g_median_lo = float(g_by_h.loc[h_lo_nearest, "median"])
    g_median_hi = float(g_by_h.loc[h_hi_nearest, "median"])
    delta_ln_g_frac_060_086 = float(np.log(g_median_hi) - np.log(g_median_lo))

    return {
        "n_events": n_events_total,
        "n_dark": n_dark,
        "n_dropped_2d_logL0": int(n_dropped_2d),
        "n_dropped_1d_logL0": int(n_dropped_1d),
        "h_grid": _round_list([float(x) for x in hs]),
        "g_frac": {
            "median": _round_list([float(x) for x in g_frac_median]),
            "min": _round_list([float(x) for x in g_frac_min]),
            "max": _round_list([float(x) for x in g_frac_max]),
        },
        "logpost_2d": {
            "delta_rel_max": _round_list([float(x) for x in delta_2d]),
            "map_h": map_2d,
            "value_at_max": _round(val_2d_max),
        },
        "logpost_2d_frozen": {
            "delta_rel_max": _round_list([float(x) for x in delta_2d_frozen]),
            "map_h": map_2d_frozen,
            "value_at_max": _round(val_2d_frozen_max),
        },
        "posterior_2d_expnorm": _round_list(post_2d),
        "posterior_2d_frozen_expnorm": _round_list(post_2d_frozen),
        "logpost_1d": {
            "delta_rel_max": _round_list([float(x) for x in delta_1d]),
            "map_h": map_1d,
            "value_at_max": _round(val_1d_max),
        },
        "dark_catleg_delta": {
            "h": _round_list(dark_catleg_delta_h),
            "delta_rel_h073": _round_list(dark_catleg_delta_vals),
            "source": source,
        },
        "summary": {
            "N_events": n_events_total,
            "N_dark": n_dark,
            "map_h_full_2d": map_2d,
            "map_h_frozen_2d": map_2d_frozen,
            "map_h_1d": map_1d,
            "g_frac_at_h073_median": _round(g_frac_073_median),
            "g_frac_grid_min": _round(g_frac_grid_min),
            "g_frac_grid_max": _round(g_frac_grid_max),
            "delta_ln_g_frac_060_to_086_median": _round(delta_ln_g_frac_060_086),
        },
    }


def main() -> None:
    readout: dict[str, Any] = {}
    if READOUT_PATH.exists():
        with open(READOUT_PATH) as f:
            readout = json.load(f)

    out: dict[str, Any] = {}
    for short_name, d in VENUES.items():
        print(f"=== {short_name} ===")
        r = compute_venue(short_name, d, readout)
        out[short_name] = r
        s = r["summary"]
        print(f"  N_events={s['N_events']} N_dark={s['N_dark']}")
        print(
            f"  dropped (logL==0 guard): 2d={r['n_dropped_2d_logL0']} "
            f"1d={r['n_dropped_1d_logL0']}"
        )
        print(f"  MAP full 2D: h={s['map_h_full_2d']}")
        print(f"  MAP frozen 2D: h={s['map_h_frozen_2d']}")
        print(f"  MAP 1D: h={s['map_h_1d']}")
        print(
            f"  g_frac(0.73) median={s['g_frac_at_h073_median']}, "
            f"grid range=[{s['g_frac_grid_min']}, {s['g_frac_grid_max']}]"
        )
        print(f"  dark_catleg_delta source: {r['dark_catleg_delta']['source']}")
        print()

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"wrote {OUT_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
