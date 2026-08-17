"""Mechanical readout for the pre-registered fusion counterfactual (item 4, rows #117-#118).

Scores the four retrieved runs (off/fused x iiib/joint_r1) against
PREREGISTRATION_FUSION_COUNTERFACTUAL.md: M-1 (2D-channel tilt = [P1]),
M-2 (1D-channel tilt = [P2]), M-3 (channel MAPs/posteriors), M-4 (mixture skew),
NULL-1 (metadata), NULL-2 handled qualitatively in the verdict (off twin vs run
of record across the ratified 08-12 divergence classes).

No interpretation is performed here; numbers only, written to readout.json.
Run from repo root: uv run python results/run_20260817_fusion_counterfactual/readout.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
VENUES = ("iiib", "joint_r1")
CELLS = ("off", "fused")

EXPECTED_H_COUNT = 41
EXPECTED_EVENT_COUNT = 1588


def load_csv(cell: str, venue: str) -> pd.DataFrame:
    return pd.read_csv(
        BASE / f"{cell}_{venue}" / "simulations" / "diagnostics" / "event_likelihoods.csv"
    )


def metadata_null(cell: str, venue: str) -> dict:
    out: dict[str, object] = {"tasks_checked": 0, "cell_ok": True, "freeze_ok": True, "commits": set()}
    for p in sorted((BASE / f"{cell}_{venue}").glob("run_metadata_*.json")):
        m = json.load(open(p))
        a = m["cli_args"]
        out["tasks_checked"] = int(out["tasks_checked"]) + 1
        if a.get("selection_in_completion_numerator") != cell:
            out["cell_ok"] = False
        if a.get("freeze_g_frac_ref_h") is not None:
            out["freeze_ok"] = False
        out["commits"].add(m["git_commit"][:8])  # type: ignore[union-attr]
    out["commits"] = sorted(out["commits"])  # type: ignore[arg-type]
    return out


def merged(venue: str) -> pd.DataFrame:
    m = load_csv("fused", venue).merge(
        load_csv("off", venue), on=["event_idx", "h"], suffixes=("_fused", "_off"), how="inner"
    )
    for ch in ("combined_no_bh", "combined_with_bh"):
        for c in ("fused", "off"):
            m[f"ln_{ch}_{c}"] = np.log(m[f"{ch}_{c}"].astype(float).clip(lower=1e-300))
        m[f"delta_ln_{ch}"] = m[f"ln_{ch}_fused"] - m[f"ln_{ch}_off"]
    return m


def channel_tilt(m: pd.DataFrame, ch: str) -> dict:
    """Sigma-ln tilt of channel `ch` (fused - off): full-grid chord + central diff @0.73."""
    d = m.groupby("h")[f"delta_ln_{ch}"].sum().sort_index()
    h = d.index.to_numpy(dtype=float)
    h_lo, h_hi = float(h.min()), float(h.max())
    chord = float((d.loc[h_hi] - d.loc[h_lo]) / (h_hi - h_lo))
    hs = np.sort(h)
    i73 = int(np.argmin(np.abs(hs - 0.73)))
    h_m, h_p = hs[i73 - 1], hs[i73 + 1]
    central = float((d.loc[h_p] - d.loc[h_m]) / (h_p - h_m))
    return {
        "chord_nats_per_h": chord,
        "central_diff_nats_per_h_at_073": central,
        "level_at_073": float(d.loc[hs[i73]]),
        "sum_delta_ln_by_h": {str(k): float(v) for k, v in d.items()},
    }


def channel_posterior(df: pd.DataFrame, ch: str) -> dict:
    """MAP + trapezoid moments of the Sigma-ln channel posterior."""
    g = (
        df.groupby("h")[ch]
        .apply(lambda s: float(np.log(s.astype(float).clip(lower=1e-300)).sum()))
        .sort_index()
    )
    h = g.index.to_numpy(dtype=float)
    ln = g.to_numpy(dtype=float)
    w = np.exp(ln - ln.max())
    z = np.trapezoid(w, h)
    mean = float(np.trapezoid(w * h, h) / z)
    var = float(np.trapezoid(w * (h - mean) ** 2, h) / z)
    map_h = float(h[np.argmax(ln)])
    railed_low = bool(np.isclose(map_h, h.min(), atol=1e-9))
    railed_high = bool(np.isclose(map_h, h.max(), atol=1e-9))
    # rail hardness: delta-ln from the MAP grid edge to its neighbor
    order = np.argsort(h)
    ln_s, h_s = ln[order], h[order]
    i_map = int(np.argmin(np.abs(h_s - map_h)))
    neighbor = i_map + 1 if i_map == 0 else i_map - 1
    return {
        "map_h": map_h,
        "railed_low": railed_low,
        "railed_high": railed_high,
        "rail_hardness_nats": float(ln_s[neighbor] - ln_s[i_map]),
        "mean_h": mean,
        "sigma_h": float(np.sqrt(var)),
    }


def mixture_skew(m: pd.DataFrame) -> dict:
    """M-4: 1D catalogue-vs-completion share shift, off -> fused (G3 corrected direction).

    share_cat = A_cat / (A_cat + B_num) with A_cat = w_tilde_G * L_cat_no_bh * D_tilde_phi
    reconstructed per row is NOT available directly; the diagnostics carry the
    assembled pieces, so the share is computed from the mixture identity
    combined = (A_cat + B_num)/D_tilde_phi via A_cat = combined*D_tilde_phi - B_num.
    """
    out: dict[str, object] = {}
    for c in ("off", "fused"):
        a_cat = (
            m[f"combined_no_bh_{c}"].astype(float) * m[f"D_tilde_phi_{c}"].astype(float)
            - m[f"B_num_{c}"].astype(float)
        ).clip(lower=0.0)
        share = a_cat / (a_cat + m[f"B_num_{c}"].astype(float)).replace(0.0, np.nan)
        m[f"share_cat_{c}"] = share
    d = m["share_cat_fused"] - m["share_cat_off"]
    at073 = m[np.isclose(m["h"], 0.73)]
    d73 = at073["share_cat_fused"] - at073["share_cat_off"]
    out["delta_share_cat_all_h"] = {
        "mean": float(d.mean()),
        "median": float(d.median()),
        "q05": float(d.quantile(0.05)),
        "q95": float(d.quantile(0.95)),
        "frac_positive": float((d > 0).mean()),
    }
    out["delta_share_cat_at_073"] = {
        "mean": float(d73.mean()),
        "median": float(d73.median()),
        "max": float(d73.max()),
        "frac_positive": float((d73 > 0).mean()),
    }
    out["share_cat_off_at_073"] = {
        "mean": float(at073["share_cat_off"].mean()),
        "median": float(at073["share_cat_off"].median()),
    }
    # h-dependence of the mean skew (the G/Gbar reweighting the author rules on)
    out["mean_delta_share_by_h"] = {
        str(k): float(v)
        for k, v in (m.groupby("h").apply(lambda g: (g["share_cat_fused"] - g["share_cat_off"]).mean(), include_groups=False)).items()
    }
    return out


def selection_side_nulls(m: pd.DataFrame) -> dict:
    """Selection-side objects must be bit-identical between cells (no leak)."""
    cols = ["w_G", "w_tilde_G", "alpha_G_phi", "r_Malm", "D_tilde_phi"]
    return {
        c: int((m[f"{c}_fused"].astype(float) != m[f"{c}_off"].astype(float)).sum()) for c in cols
    }


def main() -> None:
    result: dict[str, object] = {}
    for venue in VENUES:
        m = merged(venue)
        fused_df, off_df = load_csv("fused", venue), load_csv("off", venue)
        result[venue] = {
            "fingerprints": {
                "rows_merged": int(len(m)),
                "n_h": int(m["h"].nunique()),
                "n_events": int(m["event_idx"].nunique()),
                "expected": EXPECTED_H_COUNT * EXPECTED_EVENT_COUNT,
            },
            "NULL_1_metadata": {c: metadata_null(c, venue) for c in CELLS},
            "NULL_selection_side_differing_cells": selection_side_nulls(m),
            "M1_2d_channel_tilt_P1": channel_tilt(m, "combined_with_bh"),
            "M2_1d_channel_tilt_P2": channel_tilt(m, "combined_no_bh"),
            "M3_posteriors": {
                ch: {c: channel_posterior(df, ch) for c, df in (("off", off_df), ("fused", fused_df))}
                for ch in ("combined_no_bh", "combined_with_bh")
            },
            "M4_mixture_skew": mixture_skew(m),
        }
    out_path = BASE / "readout.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(f"written: {out_path}")
    for venue in VENUES:
        r = result[venue]
        print(f"\n== {venue} ==")
        print("  M1 2D tilt: chord {chord_nats_per_h:+.3f}  central@0.73 {central_diff_nats_per_h_at_073:+.3f} nats/h".format(**r["M1_2d_channel_tilt_P1"]))  # type: ignore[index]
        print("  M2 1D tilt: chord {chord_nats_per_h:+.3f}  central@0.73 {central_diff_nats_per_h_at_073:+.3f} nats/h".format(**r["M2_1d_channel_tilt_P2"]))  # type: ignore[index]
        for ch in ("combined_no_bh", "combined_with_bh"):
            p = r["M3_posteriors"][ch]  # type: ignore[index]
            print(f"  M3 {ch}: off MAP {p['off']['map_h']:.4f} -> fused MAP {p['fused']['map_h']:.4f} (sigma {p['off']['sigma_h']:.4f} -> {p['fused']['sigma_h']:.4f})")
        s = r["M4_mixture_skew"]["delta_share_cat_at_073"]  # type: ignore[index]
        print(f"  M4 skew@0.73: mean Δshare_cat {s['mean']:+.3e} (frac>0 {s['frac_positive']:.3f})")
        print(f"  NULL sel-side differing cells: {r['NULL_selection_side_differing_cells']}")  # type: ignore[index]


if __name__ == "__main__":
    main()
