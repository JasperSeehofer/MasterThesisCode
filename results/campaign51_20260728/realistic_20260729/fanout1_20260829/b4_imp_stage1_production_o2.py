#!/usr/bin/env python3
"""B4.1 [IMP] part 2 -- stage-1 free read 3: the O2 construction (L_cat_no_bh == 0)
applied to the PRODUCTION HEAD readout diagnostics (zero evaluate()).

launched under rows #222/#223 -- charter node B4.1 [IMP] part 2

Substrate: results/campaign51_20260728/realistic_20260729/headreadout_20260827/{iiib,joint_r1,off_iiib}/
event_likelihoods.csv (row #213 HEAD readout, commit d04d9dc9, fused/phi; off_iiib = the off
completion cell on iiib). 1588 events x H_GRID_41.

Same assembly identity + GATE I as the mirror read (7-s.f. storage tolerance 2e-6), same
corrected combine (physics_floor + trapezoid, row #146). Production has NO per-event truth in the
diagnostics; the catalogue class (in_catalog) and d_L are joined from the realization's
prepared_cramer_rao_bounds.csv ONLY IF the event count matches, and are tagged as an
ASSUMPTION-JOIN (event_idx == CRB row order) in the output.

This is a stage-1 forecast input (A1/A12 free re-read): no band, no verdict. A registered form
must be run by a different agent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from darksiren_emri.validation.correspondence_1d import (
    H_GRID_41,
    H_TRUE,
    R_LOW_THRESHOLD,
    _hpd_contains,
    combine_log_likelihood,
    moment_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = REPO_ROOT / "results/campaign51_20260728/realistic_20260729/fanout1_20260829"
HEAD = REPO_ROOT / "results/campaign51_20260728/realistic_20260729/headreadout_20260827"
CRB_CANDIDATES = {
    "iiib": REPO_ROOT / "results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv",
    "off_iiib": REPO_ROOT / "results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv",
    "joint_r1": REPO_ROOT / "results/campaign51_20260728/realistic_20260729/seed62000/prepared_cramer_rao_bounds.csv",
}
GATE_I_TOL = 2.0e-6


def piv(df: pd.DataFrame, col: str, grid: np.ndarray) -> np.ndarray:
    return df.pivot_table(index="event_idx", columns="h", values=col, aggfunc="first").reindex(columns=grid).to_numpy(np.float64)


def moments(vals: np.ndarray, grid: np.ndarray) -> dict:
    sll = combine_log_likelihood(vals, "physics_floor")
    w = moment_weights(grid, "trapezoid")
    post = np.exp(sll - sll.max())
    post_n = post / float((post * w).sum())
    mean_h = float((post_n * grid * w).sum())
    var = float((post_n * (grid - mean_h) ** 2 * w).sum())
    map_h = float(grid[int(np.argmax(sll))])
    target = int(np.nonzero(np.isclose(grid, H_TRUE))[0][0])
    # mass on the floor node (censoring disclosure)
    floor_mass = float(post_n[0] * w[0])
    return {"mean_h": mean_h, "map_h": map_h, "sigma_h": float(np.sqrt(max(var, 0.0))),
            "c68": bool(_hpd_contains(post_n, w, target, 0.68)), "r_low": bool(map_h <= R_LOW_THRESHOLD),
            "floor_node_mass": floor_mass, "n_excluded_physics_floor": int((~(vals > 0).any(axis=1)).sum())}


def scores(vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    i_lo = int(np.nonzero(np.isclose(grid, 0.725))[0][0]); i_hi = int(np.nonzero(np.isclose(grid, 0.735))[0][0])
    lo, hi = vals[:, i_lo], vals[:, i_hi]
    out = np.full(vals.shape[0], np.nan); ok = (lo > 0) & (hi > 0)
    out[ok] = (np.log(hi[ok]) - np.log(lo[ok])) / 0.01
    return out


def main() -> int:
    grid = np.array(H_GRID_41, dtype=np.float64)
    out: dict = {}
    for venue in ("iiib", "joint_r1", "off_iiib"):
        p = HEAD / venue / "event_likelihoods.csv"
        df = pd.read_csv(p)
        hs = sorted(df["h"].unique())
        d = df[np.isin(df["h"].to_numpy(np.float64), grid)].copy()
        beta = d["alpha_G_phi"].to_numpy(np.float64) / d["r_Malm"].to_numpy(np.float64)
        d["cat"] = beta * d["L_cat_no_bh"].to_numpy(np.float64) / d["D_tilde_phi"].to_numpy(np.float64)
        d["comp"] = d["B_num"].to_numpy(np.float64) / d["D_tilde_phi"].to_numpy(np.float64)
        full = piv(d, "combined_no_bh", grid); cat = piv(d, "cat", grid); comp = piv(d, "comp", grid)
        gate_i = float(np.nanmax(np.abs(cat + comp - full) / np.maximum(np.abs(full), np.finfo(float).tiny)))
        pure = np.clip(full - cat, 0.0, None)
        idx = sorted(d["event_idx"].unique().tolist())
        m_full, m_pure = moments(full, grid), moments(pure, grid)
        s_full, s_pure = scores(full, grid), scores(pure, grid)
        s_imp = s_full - s_pure
        at = d[np.isclose(d["h"], 0.73)].set_index("event_idx").reindex(idx)
        b73 = at["alpha_G_phi"].to_numpy(np.float64) / at["r_Malm"].to_numpy(np.float64)
        share = b73 * at["L_cat_no_bh"].to_numpy(np.float64) / (b73 * at["L_cat_no_bh"].to_numpy(np.float64) + at["B_num"].to_numpy(np.float64))
        active = at["L_cat_no_bh"].to_numpy(np.float64) > 0
        rec: dict = {
            "n_events": len(idx), "n_h_in_csv": len(hs), "gate_i_max_rel": gate_i, "gate_i_pass": gate_i <= GATE_I_TOL,
            "full": m_full, "pure": m_pure, "delta_mean_h": m_pure["mean_h"] - m_full["mean_h"],
            "score_full_mean": float(np.nanmean(s_full)), "score_pure_mean": float(np.nanmean(s_pure)),
            "score_imp_mean": float(np.nanmean(s_imp)), "score_imp_sem_pooled": float(np.nanstd(s_imp, ddof=1) / np.sqrt(np.isfinite(s_imp).sum())),
            "frac_active_073": float(active.mean()), "score_imp_mean_active": float(np.nanmean(s_imp[active])),
            "share_073_median_active": float(np.median(share[active])), "share_073_mean_all": float(share.mean()),
        }
        # share-quartile localization (all events; share==0 for inactive)
        q_edges = np.quantile(share[active], [0.25, 0.5, 0.75])
        g = np.searchsorted(q_edges, share, side="right")
        tot = float(np.nansum(s_imp))
        rec["share_quartiles_active"] = [{"q": k + 1, "n": int(((g == k) & active).sum()),
                                          "mean_s_imp": float(np.nanmean(s_imp[(g == k) & active])),
                                          "share_of_total": float(np.nansum(s_imp[(g == k) & active]) / tot)} for k in range(4)]
        # ASSUMPTION-JOIN to the realization CRB (event_idx == row order)
        crb_p = CRB_CANDIDATES[venue]
        if crb_p.is_file():
            crb = pd.read_csv(crb_p)
            rec["crb_path"] = str(crb_p.relative_to(REPO_ROOT)); rec["crb_rows"] = int(len(crb))
            if max(idx) < len(crb):
                sub = crb.iloc[idx]
                dl = sub["luminosity_distance"].to_numpy(np.float64)
                rec["join"] = "ASSUMPTION-JOIN event_idx == CRB row order (unvalidated; secondary only)"
                if "in_catalog" in sub.columns:
                    ic = sub["in_catalog"].to_numpy()
                    ic = np.array([str(v).lower() == "true" or v is True or v == 1 for v in ic])
                    rec["in_catalog_frac"] = float(ic.mean())
                    rec["score_imp_mean_dark"] = float(np.nanmean(s_imp[~ic])); rec["score_imp_mean_incat"] = float(np.nanmean(s_imp[ic]))
                    rec["share_of_total_s_imp_dark"] = float(np.nansum(s_imp[~ic]) / tot)
                    # dark-only pure arm: remove the catalogue leg ONLY for dark events
                    pure_dark = full.copy(); pure_dark[~ic] = pure[~ic]
                    rec["pure_dark_only"] = moments(pure_dark, grid)
                    rec["delta_mean_h_dark_only"] = rec["pure_dark_only"]["mean_h"] - m_full["mean_h"]
                dq = np.quantile(dl, [0.25, 0.5, 0.75])
                gd = np.searchsorted(dq, dl, side="right")
                rec["dL_quartiles"] = [{"q": k + 1, "n": int((gd == k).sum()), "dL_median_gpc": float(np.median(dl[gd == k])),
                                        "mean_s_imp": float(np.nanmean(s_imp[gd == k])), "share_of_total": float(np.nansum(s_imp[gd == k]) / tot)} for k in range(4)]
        out[venue] = rec
        print(venue, json.dumps({k: v for k, v in rec.items() if k not in ("share_quartiles_active", "dL_quartiles")}, indent=1))
        print(venue, "share quartiles", rec["share_quartiles_active"])
        print(venue, "dL quartiles", rec.get("dL_quartiles"))
    (OUT_DIR / "b4_imp_stage1_production_o2.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
