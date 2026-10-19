#!/usr/bin/env python3
"""B4.1 [IMP] part 2 -- stage-1 information-forecast FREE READ (zero ``evaluate()``).

launched under rows #222/#223 -- charter node B4.1 [IMP] part 2

Purpose (research-cycle stage 1, amendments A1 + A12): before naming the depth-2
decisive read, exhaust the banked B-SEL diagnostics for what a perfect analysis
of the existing data already says about the impostor-drag REMAINDER -- its size
on the fused (production) basis and on the fused+twin basis, and whether the
per-event impostor-leg SCORE at truth is diffuse or localized in the covariates
that ARE banked (true-host d_L/z, SNR, candidate-ball size, catalogue share).

This is a forecast input, NOT a registered measurement: no band is attached to
any number here, nothing below is a verdict, and any registered version of a
statistic printed here must be run by a different agent (standing rule 2).

Substrate (three arms of the same 12 realizations, seeds 900101..900112):
  bsel  -- off completion basis, coded catalogue leg (the O2 substrate, row #149):
           results/prod2d_closure_20260818/arm_event_likelihoods/bsel_seed*/...
  fc    -- fused completion basis, coded catalogue leg (row #173 FC):
           results/campaign51_20260728/realistic_20260729/p3_work/fc_*_work/...
  ft    -- fused completion basis, twin (phi) catalogue leg (row #173 FT):
           .../p3_work/ft_*_work/...
All three carry the assembly identity (bayesian_statistics.py p_Di assembly,
re-derived in decompose_impostor_leg.py):
    combined_no_bh = (alpha_G_phi/r_Malm * L_cat_no_bh + B_num) / D_tilde_phi
which is gated here (GATE I, 2e-6 = the 7-s.f. CSV storage bound) before use.
Pure arm := L_cat_no_bh == 0 by exact subtraction (the O2 construction).

Statistics (per event i, with the central difference over h = 0.725/0.735):
    s_full,i = d ln combined_no_bh / dh ;  s_pure,i = d ln pure / dh
    s_imp,i  = s_full,i - s_pure,i          (the impostor leg's score contribution;
                                             identically 0 for L_cat == 0 events)
Covariates: d_L (prepared_cramer_rao_bounds.csv, FT work dirs, joined on
event_idx), z_true = dist_to_redshift(d_L, h=0.73) (fiducial inversion; the
mirror observation is noiseless so d_L is the injected value), SNR, the
candidate-ball size from the "possible hosts found" log lines (part-1 method,
global-count consistency check required), and the beta-convention catalogue
share at h = 0.73.

Outputs: b4_imp_stage1_forecast.json, b4_imp_stage1_events.csv (this directory).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from darksiren_emri.physical_relations import dist_to_redshift
from darksiren_emri.validation.correspondence_1d import (
    H_GRID_41,
    H_GRID_FULL,
    H_TRUE,
    R_LOW_THRESHOLD,
    _hpd_contains,
    combine_log_likelihood,
    compute_seed_statistics,
    moment_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = REPO_ROOT / "results/campaign51_20260728/realistic_20260729/fanout1_20260829"
BSEL_ROOT = REPO_ROOT / "results/prod2d_closure_20260818/arm_event_likelihoods"
P3_WORK = REPO_ROOT / "results/campaign51_20260728/realistic_20260729/p3_work"
SEEDS = [900101 + i for i in range(12)]
GATE_I_TOL = 2.0e-6
O2_OF_RECORD = 0.07918832458493741  # decompose_impostor_leg_output.json delta_bias
POSSIBLE_HOSTS_RE = re.compile(r"possible hosts found (\d+)/(\d+)\.\.\.")


def diag_path(arm: str, seed: int) -> Path:
    if arm == "bsel":
        return BSEL_ROOT / f"bsel_seed{seed}" / f"seed{seed}" / "simulations/diagnostics/event_likelihoods.csv"
    return P3_WORK / f"{arm}_{seed}_work" / f"seed{seed}" / "simulations/diagnostics/event_likelihoods.csv"


def crb_path(seed: int) -> Path:
    return P3_WORK / f"ft_{seed}_work" / f"seed{seed}" / "simulations/prepared_cramer_rao_bounds.csv"


def log_path(arm: str, seed: int) -> Path | None:
    if arm == "bsel":
        return None
    return P3_WORK / f"{arm}_{seed}.log"


def pivot(df: pd.DataFrame, col: str, grid: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return (
        df.pivot_table(index="event_idx", columns="h", values=col, aggfunc="first")
        .reindex(columns=grid)
        .to_numpy(dtype=np.float64)
    )


def matrices(df: pd.DataFrame, grid: npt.NDArray[np.float64]) -> tuple[Any, Any, float, list[int]]:
    d = df[np.isin(df["h"].to_numpy(np.float64), grid)].copy()
    beta = d["alpha_G_phi"].to_numpy(np.float64) / d["r_Malm"].to_numpy(np.float64)
    d["cat_term"] = beta * d["L_cat_no_bh"].to_numpy(np.float64) / d["D_tilde_phi"].to_numpy(np.float64)
    d["comp_term"] = d["B_num"].to_numpy(np.float64) / d["D_tilde_phi"].to_numpy(np.float64)
    full = pivot(d, "combined_no_bh", grid)
    cat = pivot(d, "cat_term", grid)
    comp = pivot(d, "comp_term", grid)
    scale = np.maximum(np.abs(full), np.finfo(float).tiny)
    gate_i = float(np.nanmax(np.abs(cat + comp - full) / scale))
    pure = np.clip(full - cat, 0.0, None)
    idx = sorted(d["event_idx"].unique().tolist())
    return full, pure, gate_i, idx


def moments(vals: npt.NDArray[np.float64], grid: npt.NDArray[np.float64]) -> dict[str, Any]:
    sum_log_l = combine_log_likelihood(vals, "physics_floor")
    weights = moment_weights(grid, "trapezoid")
    lp = sum_log_l - sum_log_l.max()
    post = np.exp(lp)
    norm = float((post * weights).sum())
    post_n = post / norm
    mean_h = float((post_n * grid * weights).sum())
    var = float((post_n * (grid - mean_h) ** 2 * weights).sum())
    map_h = float(grid[int(np.argmax(sum_log_l))])
    target = int(np.nonzero(np.isclose(grid, H_TRUE))[0][0])
    return {
        "mean_h": mean_h,
        "map_h": map_h,
        "sigma_h": float(np.sqrt(max(var, 0.0))),
        "c68": bool(_hpd_contains(post_n, weights, target, 0.68)),
        "r_low": bool(map_h <= R_LOW_THRESHOLD),
    }


def per_event_scores(vals: npt.NDArray[np.float64], grid: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    i_lo = int(np.nonzero(np.isclose(grid, 0.725))[0][0])
    i_hi = int(np.nonzero(np.isclose(grid, 0.735))[0][0])
    lo, hi = vals[:, i_lo], vals[:, i_hi]
    out = np.full(vals.shape[0], np.nan)
    ok = (lo > 0.0) & (hi > 0.0)
    out[ok] = (np.log(hi[ok]) - np.log(lo[ok])) / (0.735 - 0.725)
    return out


def candidate_counts(arm: str, seed: int, df: pd.DataFrame, idx: list[int]) -> tuple[dict[int, tuple[int, int]], str]:
    """Part-1 method: log-line order == first-h-block CSV order for non-zero-host events."""
    lp = log_path(arm, seed)
    if lp is None or not lp.is_file():
        return {}, "no log for this arm"
    counts: list[tuple[int, int]] = []
    with open(lp) as f:
        for line in f:
            if "possible hosts found" in line:
                m = POSSIBLE_HOSTS_RE.search(line)
                if m:
                    counts.append((int(m.group(1)), int(m.group(2))))
    h_values = sorted(df["h"].unique())
    first = df[df["h"] == h_values[0]].sort_values("event_idx").reset_index(drop=True)
    zero = (first["L_cat_no_bh"] == 0.0) & (first["L_cat_with_bh"] == 0.0)
    n_print = int((~zero).sum())
    if len(counts) != n_print * len(h_values):
        return {}, f"ALIGNMENT FAIL: {len(counts)} log lines vs {n_print}x{len(h_values)}"
    block = counts[:n_print]
    printed_idx = first.loc[~zero, "event_idx"].tolist()
    out = {int(e): (0, 0) for e in first["event_idx"].tolist()}
    for e, c in zip(printed_idx, block, strict=True):
        out[int(e)] = c
    return out, ""


def eta2(y: npt.NDArray[np.float64], g: npt.NDArray[np.int64]) -> float:
    ok = np.isfinite(y) & (g >= 0)
    y, g = y[ok], g[ok]
    if y.size < 4:
        return float("nan")
    gm = y.mean()
    ss_tot = float(((y - gm) ** 2).sum())
    ss_b = 0.0
    for k in np.unique(g):
        yk = y[g == k]
        ss_b += yk.size * (yk.mean() - gm) ** 2
    return float(ss_b / ss_tot) if ss_tot > 0 else float("nan")


def quartile_groups(x: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.int64], list[float]]:
    ok = np.isfinite(x)
    edges = np.quantile(x[ok], [0.25, 0.5, 0.75]).tolist()
    g = np.full(x.shape, -1, dtype=np.int64)
    g[ok] = np.searchsorted(np.array(edges), x[ok], side="right")
    return g, edges


def main() -> int:
    grid41 = np.array(H_GRID_41, dtype=np.float64)
    gridfull = np.array(H_GRID_FULL, dtype=np.float64)
    out: dict[str, Any] = {"arms": {}, "notes": []}
    events_rows: list[dict[str, Any]] = []
    lcat_fc_vs_bsel: list[float] = []

    for arm in ("bsel", "fc", "ft"):
        arm_rec: dict[str, Any] = {"per_seed": [], "gate_i": [], "gate_moment": [], "count_warnings": []}
        for seed in SEEDS:
            p = diag_path(arm, seed)
            if not p.is_file():
                arm_rec["count_warnings"].append(f"{arm} {seed}: MISSING {p}")
                continue
            df = pd.read_csv(p)
            full41, pure41, gi41, idx = matrices(df, grid41)
            fullF, pureF, giF, _ = matrices(df, gridfull)
            arm_rec["gate_i"].append({"seed": seed, "max_rel_41": gi41, "max_rel_full": giF, "pass": max(gi41, giF) <= GATE_I_TOL})
            ref = compute_seed_statistics(p, seed)
            m_full = moments(full41, grid41)
            arm_rec["gate_moment"].append({"seed": seed, "delta": abs(m_full["mean_h"] - ref.mean_h)})
            m_pure = moments(pure41, grid41)
            mF_full = moments(fullF, gridfull)
            mF_pure = moments(pureF, gridfull)

            s_full = per_event_scores(full41, grid41)
            s_pure = per_event_scores(pure41, grid41)
            s_imp = s_full - s_pure
            # L_cat == 0 events: pure == full exactly -> s_imp = 0 by construction
            at73 = df[np.isclose(df["h"], 0.73)].set_index("event_idx").reindex(idx)
            beta73 = at73["alpha_G_phi"].to_numpy(np.float64) / at73["r_Malm"].to_numpy(np.float64)
            lcat73 = at73["L_cat_no_bh"].to_numpy(np.float64)
            bnum73 = at73["B_num"].to_numpy(np.float64)
            share73 = beta73 * lcat73 / (beta73 * lcat73 + bnum73)
            active = lcat73 > 0.0

            crb = pd.read_csv(crb_path(seed)) if crb_path(seed).is_file() else None
            counts, warn = candidate_counts(arm, seed, df, idx)
            if warn:
                arm_rec["count_warnings"].append(f"{arm} {seed}: {warn}")

            if arm == "fc":
                # alignment check bsel vs fc: catalogue leg must be bit-identical (fused
                # touches the completion numerator only)
                pb = diag_path("bsel", seed)
                if pb.is_file():
                    db = pd.read_csv(pb)
                    a = db[np.isclose(db["h"], 0.73)].set_index("event_idx").reindex(idx)["L_cat_no_bh"].to_numpy(np.float64)
                    sc = np.maximum(np.abs(lcat73), np.finfo(float).tiny)
                    lcat_fc_vs_bsel.append(float(np.nanmax(np.abs(a - lcat73) / sc)))

            for k, e in enumerate(idx):
                dl = float(crb.iloc[e]["luminosity_distance"]) if crb is not None else float("nan")
                snr = float(crb.iloc[e]["SNR"]) if crb is not None else float("nan")
                n1, n2 = counts.get(e, (np.nan, np.nan)) if counts else (np.nan, np.nan)
                events_rows.append(
                    {
                        "arm": arm, "seed": seed, "event_idx": e, "d_L_gpc": dl,
                        "z_true": dist_to_redshift(dl, H_TRUE) if np.isfinite(dl) else float("nan"),
                        "SNR": snr, "n_cand_no_bh": n1, "n_cand_with_bh": n2,
                        "active_073": bool(active[k]), "share_073": float(share73[k]),
                        "s_full": float(s_full[k]), "s_pure": float(s_pure[k]), "s_imp": float(s_imp[k]),
                    }
                )

            arm_rec["per_seed"].append(
                {
                    "seed": seed, "n_events": len(idx), "n_active_073": int(active.sum()),
                    "full_41": m_full, "pure_41": m_pure, "delta_41": m_pure["mean_h"] - m_full["mean_h"],
                    "full_FULL": mF_full, "pure_FULL": mF_pure, "delta_FULL": mF_pure["mean_h"] - mF_full["mean_h"],
                    "mean_s_full": float(np.nanmean(s_full)), "mean_s_pure": float(np.nanmean(s_pure)),
                    "mean_s_imp": float(np.nanmean(s_imp)),
                    "sem_s_imp": float(np.nanstd(s_imp, ddof=1) / np.sqrt(np.isfinite(s_imp).sum())),
                }
            )
        ps = arm_rec["per_seed"]
        if ps:
            d41 = np.array([r["delta_41"] for r in ps]); dF = np.array([r["delta_FULL"] for r in ps])
            f41 = np.array([r["full_41"]["mean_h"] for r in ps]); p41 = np.array([r["pure_41"]["mean_h"] for r in ps])
            fF = np.array([r["full_FULL"]["mean_h"] for r in ps]); pF = np.array([r["pure_FULL"]["mean_h"] for r in ps])
            si = np.array([r["mean_s_imp"] for r in ps]); sf = np.array([r["mean_s_full"] for r in ps]); sp = np.array([r["mean_s_pure"] for r in ps])
            n = len(ps)
            arm_rec["fleet"] = {
                "n_seeds": n,
                "bias_full_41": float(f41.mean() - H_TRUE), "bias_pure_41": float(p41.mean() - H_TRUE),
                "delta_41_mean": float(d41.mean()), "delta_41_sd": float(d41.std(ddof=1)), "delta_41_sem": float(d41.std(ddof=1) / np.sqrt(n)),
                "delta_41_n_positive": int((d41 > 0).sum()),
                "bias_full_FULL": float(fF.mean() - H_TRUE), "bias_pure_FULL": float(pF.mean() - H_TRUE),
                "delta_FULL_mean": float(dF.mean()), "delta_FULL_sd": float(dF.std(ddof=1)), "delta_FULL_sem": float(dF.std(ddof=1) / np.sqrt(n)),
                "r_low_full_41": int(sum(r["full_41"]["r_low"] for r in ps)), "r_low_pure_41": int(sum(r["pure_41"]["r_low"] for r in ps)),
                "score_full_mean": float(sf.mean()), "score_pure_mean": float(sp.mean()),
                "score_imp_mean": float(si.mean()), "score_imp_seed_sd": float(si.std(ddof=1)), "score_imp_seed_sem": float(si.std(ddof=1) / np.sqrt(n)),
                "gate_i_pass_all": all(g["pass"] for g in arm_rec["gate_i"]),
                "gate_moment_max_delta": max(g["delta"] for g in arm_rec["gate_moment"]),
            }
        out["arms"][arm] = arm_rec

    out["o2_reproduction"] = {
        "delta_41_bsel": out["arms"]["bsel"]["fleet"]["delta_41_mean"],
        "of_record": O2_OF_RECORD,
        "abs_dev": abs(out["arms"]["bsel"]["fleet"]["delta_41_mean"] - O2_OF_RECORD),
    }
    out["lcat_fc_vs_bsel_max_rel"] = max(lcat_fc_vs_bsel) if lcat_fc_vs_bsel else None

    ev = pd.DataFrame(events_rows)
    ev.to_csv(OUT_DIR / "b4_imp_stage1_events.csv", index=False)

    # covariate resolution of s_imp, per arm, fleet-pooled quartiles; per-seed stratification
    cov_out: dict[str, Any] = {}
    for arm in ("bsel", "fc", "ft"):
        e = ev[ev["arm"] == arm].copy()
        y = e["s_imp"].to_numpy(np.float64)
        rec: dict[str, Any] = {"n_events": int(len(e)), "n_active": int(e["active_073"].sum()),
                               "mean_s_imp_all": float(np.nanmean(y)),
                               "sem_s_imp_all_pooled": float(np.nanstd(y, ddof=1) / np.sqrt(np.isfinite(y).sum())),
                               "mean_s_imp_active": float(np.nanmean(y[e["active_073"].to_numpy()])),
                               "frac_active": float(e["active_073"].mean())}
        total = float(np.nansum(y))
        for cov, lab in (("z_true", "z_true"), ("SNR", "SNR"), ("share_073", "share_073"), ("n_cand_no_bh", "log_n_cand_no_bh")):
            x = e[cov].to_numpy(np.float64)
            if cov == "n_cand_no_bh":
                x = np.log10(np.maximum(x, 0.5))  # zero-ball -> log10(0.5)
            if not np.isfinite(x).any():
                rec[lab] = "unavailable"
                continue
            g, edges = quartile_groups(x)
            q: list[dict[str, Any]] = []
            for k in range(4):
                m = (g == k) & np.isfinite(y)
                yk = y[m]
                q.append({"q": k + 1, "n": int(m.sum()), "mean_s_imp": float(yk.mean()) if yk.size else None,
                          "sem": float(yk.std(ddof=1) / np.sqrt(yk.size)) if yk.size > 1 else None,
                          "share_of_total_s_imp": float(yk.sum() / total) if total != 0 else None,
                          "cov_median": float(np.nanmedian(x[m])) if m.any() else None})
            rec[lab] = {"edges": edges, "eta2": eta2(y, g), "quartiles": q,
                        "pearson_r": float(np.corrcoef(x[np.isfinite(x) & np.isfinite(y)], y[np.isfinite(x) & np.isfinite(y)])[0, 1])}
        # active-only resolution by z (the drag lives only on active events)
        ea = e[e["active_073"]]
        ya = ea["s_imp"].to_numpy(np.float64); xa = ea["z_true"].to_numpy(np.float64)
        if np.isfinite(xa).any():
            g, edges = quartile_groups(xa)
            rec["z_true_active_only"] = {"edges": edges, "eta2": eta2(ya, g),
                                         "quartiles": [{"q": k + 1, "n": int(((g == k)).sum()),
                                                        "mean_s_imp": float(ya[g == k].mean()),
                                                        "sem": float(ya[g == k].std(ddof=1) / np.sqrt((g == k).sum()))} for k in range(4)]}
        cov_out[arm] = rec
    out["covariates"] = cov_out
    (OUT_DIR / "b4_imp_stage1_forecast.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=1, default=float))
    for arm in ("bsel", "fc", "ft"):
        print(arm, json.dumps(out["arms"][arm]["fleet"], indent=1, default=float))
        print(arm, "warnings:", out["arms"][arm]["count_warnings"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
