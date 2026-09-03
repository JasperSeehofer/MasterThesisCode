"""rd-2d-bootstrap-jackknife (Branch K) — verdict-free frequentist read.

Reproduces row #302 (gradient-trapezoid moment convention, physics-floor
zero handling — the "frozen T0 convention" quoted in
exec/m-head-rebaseline/READOUT_RECORD.md and defined in
results/prod2d_closure_20260818/tier0_bootstrap_jackknife.py), then computes:
  (a) bootstrap-over-events SE(mean_h) at N=1588 (B=2000, seed=20260903),
      both venues, both channels; ratio to sigma_h=0.018475 (iiib 2D, row #302)
  (b) full leave-one-out jackknife influence of every event on mean_h
  (c) minimal-subset fraction (events ranked by |jackknife influence|,
      cumulative removal) that brings mean_h within 1 sigma_h of H_TRUE=0.73
  (d) bootstrap width-vs-N table: N in {100,200,400,800,1588}, 200 draws
      each, sampling with replacement from the full 1588-event set.

No verdict is rendered here — only numbers, per the row #325 grant.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/jasper/Repositories/darksiren-emri")
NODE_DIR = REPO / "results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/rd-2d-bootstrap-jackknife"

VENUE_PATHS = {
    "iiib": REPO
    / "results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved"
    / "run_20260902_graph1_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv",
    "joint_r1": REPO
    / "results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved"
    / "run_20260902_graph1_headrebaseline_joint_r1/simulations/diagnostics/event_likelihoods.csv",
}

CHANNELS = ("combined_no_bh", "combined_with_bh")
CHANNEL_LABEL = {"combined_no_bh": "1D", "combined_with_bh": "2D"}
TRUTH = 0.73
SIGMA_H_ROW302 = 0.018475  # iiib 2D, row #302 — the decisive comparand for the ratio
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20260903
WIDTH_N_LIST = [100, 200, 400, 800, 1588]
WIDTH_DRAWS = 200

# row #302 reproduction targets (to 1e-5)
ROW302_TARGETS = {
    ("iiib", "combined_with_bh"): {"map_h": 0.665, "mean_h": 0.665854},
    ("iiib", "combined_no_bh"): {"map_h": 0.665, "mean_h": 0.666987},
    ("joint_r1", "combined_with_bh"): {"map_h": 0.665, "mean_h": 0.667127},
    ("joint_r1", "combined_no_bh"): {"map_h": 0.665, "mean_h": 0.667032},
}


def physics_floor_apply(likelihoods: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row: zeros -> row's own min nonzero value; all-zero row -> excluded."""
    result = likelihoods.copy()
    n = result.shape[0]
    exclude = np.zeros(n, dtype=bool)
    for i in range(n):
        row = result[i]
        zero = row == 0.0
        if not zero.any():
            continue
        nz = row[~zero]
        if nz.size == 0:
            exclude[i] = True
        else:
            result[i, zero] = float(nz.min())
    return result, exclude


def load_matrix(venue: str, channel: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    df = pd.read_csv(VENUE_PATHS[venue])
    h_grid = np.sort(df["h"].unique())
    piv = df.pivot(index="event_idx", columns="h", values=channel).reindex(columns=h_grid)
    if piv.isna().any().any():
        raise ValueError(f"{venue}/{channel}: ragged CSV, missing (event,h) cells")
    event_idx = piv.index.to_numpy()
    L = piv.to_numpy(dtype=np.float64)
    L_floored, exclude = physics_floor_apply(L)
    n_excluded = int(exclude.sum())
    if n_excluded:
        L_floored = L_floored[~exclude]
        event_idx = event_idx[~exclude]
    logL = np.log(L_floored)
    return h_grid, event_idx, logL, n_excluded


def moments(logpost: np.ndarray, h_grid: np.ndarray, weights: np.ndarray):
    lp = logpost - logpost.max(axis=-1, keepdims=True)
    post = np.exp(lp)
    norm = (post * weights).sum(axis=-1)
    post_n = post / norm[..., None]
    mean_h = (post_n * h_grid * weights).sum(axis=-1)
    var = (post_n * (h_grid - mean_h[..., None]) ** 2 * weights).sum(axis=-1)
    sigma_h = np.sqrt(np.clip(var, 0.0, None))
    map_h = h_grid[np.argmax(logpost, axis=-1)]
    return mean_h, sigma_h, map_h, post_n


def run_one(venue: str, channel: str, rng: np.random.Generator) -> dict:
    h_grid, event_idx, logL, n_excluded = load_matrix(venue, channel)
    weights = np.gradient(h_grid)
    n_events, n_h = logL.shape

    # --- full-sample posterior --------------------------------------------
    logpost_full = logL.sum(axis=0)
    mean_arr, sigma_arr, map_arr, _pn = moments(logpost_full[None, :], h_grid, weights)
    mean_h_full, sigma_h_full, map_h_full = float(mean_arr[0]), float(sigma_arr[0]), float(map_arr[0])

    target = ROW302_TARGETS[(venue, channel)]
    reproduces = (
        abs(mean_h_full - target["mean_h"]) < 1e-5 and abs(map_h_full - target["map_h"]) < 1e-9
    )

    # --- (a) bootstrap-over-events (multinomial-weighted, N=n_events) -----
    counts = rng.multinomial(n_events, np.full(n_events, 1.0 / n_events), size=BOOTSTRAP_B)
    logpost_boot = counts.astype(np.float64) @ logL
    mean_h_boot, _s, map_h_boot, _p = moments(logpost_boot, h_grid, weights)
    se_mean_h = float(mean_h_boot.std(ddof=1))
    se_map_h = float(map_h_boot.std(ddof=1))
    rail_fraction = float(np.mean((map_h_boot == h_grid[0]) | (map_h_boot == h_grid[-1])))

    # --- (b) full leave-one-out jackknife influence on mean_h -------------
    loo_logpost = logpost_full[None, :] - logL
    loo_mean_h, _s2, _m2, _p2 = moments(loo_logpost, h_grid, weights)
    influence = mean_h_full - loo_mean_h  # positive: removing event i pulls mean_h down (event i was pulling it up)

    # --- (c) minimal-subset fraction that carries the offset --------------
    # Directional ranking: mean_h_full is below TRUTH in every observed
    # channel/venue here, so the events whose removal moves mean_h TOWARD
    # truth are the ones with the most-NEGATIVE influence (their presence
    # pulls the mean down; ``influence = mean_full - loo_mean``, so
    # influence < 0 <=> loo_mean > mean_full <=> removing that event raises
    # the mean). Ranking by signed influence in the direction that reduces
    # the offset (not by |influence|) is the informative statistic here —
    # ranking by |influence| mixes in events that make the offset WORSE
    # when removed and produces a non-monotone curve.
    sign_toward_truth = -1.0 if (mean_h_full - TRUTH) < 0 else 1.0
    directional_influence = sign_toward_truth * influence
    order = np.argsort(-directional_influence)  # most-helpful-to-remove first
    cum_curve = []
    minimal_k = None
    for k in range(0, n_events + 1):
        if k == 0:
            logpost_k = logpost_full
        else:
            dropped = order[:k]
            logpost_k = logpost_full - logL[dropped].sum(axis=0)
        mean_k_arr, _s3, _m3, _p3 = moments(logpost_k[None, :], h_grid, weights)
        mean_k = float(mean_k_arr[0])
        if k in (0, 1, 2, 5, 10, 20, 50, 100, 200, 400, 800) or k == n_events:
            cum_curve.append({"k": k, "mean_h": mean_k, "abs_dev_from_truth": abs(mean_k - TRUTH)})
        if minimal_k is None and abs(mean_k - TRUTH) <= sigma_h_full:
            minimal_k = k
    if minimal_k is None:
        minimal_k = n_events  # never reached within 1 sigma_h even removing everything

    minimal_fraction = minimal_k / n_events

    # Also report the |influence|-ranked (undirected) curve's minimal_k for
    # transparency/comparison — this was the first (wrong-direction-mixed)
    # attempt and is retained only as a cross-check, never as the decisive
    # number.
    order_abs = np.argsort(-np.abs(influence))
    minimal_k_abs_ranked = None
    for k in range(0, n_events + 1):
        if k == 0:
            logpost_k = logpost_full
        else:
            dropped = order_abs[:k]
            logpost_k = logpost_full - logL[dropped].sum(axis=0)
        mean_k_arr, _s5, _m5, _p5 = moments(logpost_k[None, :], h_grid, weights)
        if abs(float(mean_k_arr[0]) - TRUTH) <= sigma_h_full:
            minimal_k_abs_ranked = k
            break
    if minimal_k_abs_ranked is None:
        minimal_k_abs_ranked = n_events

    # --- (d) bootstrap width-vs-N table (with replacement, 200 draws) -----
    width_table = {}
    for N in WIDTH_N_LIST:
        idx = rng.integers(0, n_events, size=(WIDTH_DRAWS, N))
        logpost_draws = logL[idx].sum(axis=1)  # (WIDTH_DRAWS, n_h)
        mean_draws, _s4, _m4, _p4 = moments(logpost_draws, h_grid, weights)
        width_table[N] = {
            "mean_of_mean_h": float(mean_draws.mean()),
            "std_mean_h": float(mean_draws.std(ddof=1)),
        }
    # fill in the sqrt(N) scaling prediction after computing all, anchored at N=1588 draws
    ref_std_1588 = width_table[1588]["std_mean_h"]
    for N in WIDTH_N_LIST:
        width_table[N]["predicted_from_1588_scaling"] = float(ref_std_1588 * np.sqrt(1588.0 / N))

    return {
        "venue": venue,
        "channel": channel,
        "channel_label": CHANNEL_LABEL[channel],
        "n_events": int(n_events),
        "n_excluded_physics_floor": n_excluded,
        "full_sample": {
            "map_h": map_h_full,
            "mean_h": mean_h_full,
            "sigma_h": sigma_h_full,
        },
        "row302_reproduction": {
            "target_map_h": target["map_h"],
            "target_mean_h": target["mean_h"],
            "computed_map_h": map_h_full,
            "computed_mean_h": mean_h_full,
            "abs_diff_mean_h": abs(mean_h_full - target["mean_h"]),
            "reproduces_to_1e-5": reproduces,
        },
        "bootstrap": {
            "B": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "se_mean_h": se_mean_h,
            "se_map_h": se_map_h,
            "sigma_h_row302_iiib2D": SIGMA_H_ROW302,
            "ratio_se_mean_h_to_sigma_h_row302": se_mean_h / SIGMA_H_ROW302,
            "ratio_se_mean_h_to_own_sigma_h": se_mean_h / sigma_h_full,
            "g_censoring_rail_fraction_map_at_grid_edge": rail_fraction,
        },
        "jackknife": {
            "influence_summary": {
                "mean": float(influence.mean()),
                "std": float(influence.std(ddof=1)),
                "min": float(influence.min()),
                "max": float(influence.max()),
                "abs_mean": float(np.abs(influence).mean()),
                "n_events": n_events,
            },
            "top10_events_by_abs_influence": [
                {"event_idx": int(event_idx[i]), "influence": float(influence[i])}
                for i in order[:10]
            ],
        },
        "minimal_subset": {
            "sigma_h_used": sigma_h_full,
            "truth": TRUTH,
            "full_abs_offset_from_truth": abs(mean_h_full - TRUTH),
            "ranking": "signed influence in the direction that reduces |mean_h - truth| (see docstring note)",
            "minimal_k_events_removed": int(minimal_k),
            "minimal_fraction_of_events": minimal_fraction,
            "curve_sample": cum_curve,
            "cross_check_abs_influence_ranked_minimal_k": int(minimal_k_abs_ranked),
            "cross_check_abs_influence_ranked_minimal_fraction": minimal_k_abs_ranked / n_events,
        },
        "width_vs_N": width_table,
    }


def main() -> None:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results = []
    for venue in ("iiib", "joint_r1"):
        for channel in CHANNELS:
            r = run_one(venue, channel, rng)
            results.append(r)
            print(
                f"{venue:9s} {CHANNEL_LABEL[channel]:2s} "
                f"map={r['full_sample']['map_h']:.4f} "
                f"mean={r['full_sample']['mean_h']:.6f} "
                f"sigma={r['full_sample']['sigma_h']:.6f} "
                f"reproduces={r['row302_reproduction']['reproduces_to_1e-5']} "
                f"diff={r['row302_reproduction']['abs_diff_mean_h']:.2e} "
                f"SE_boot={r['bootstrap']['se_mean_h']:.6f} "
                f"ratio(to_iiib2D_sigma)={r['bootstrap']['ratio_se_mean_h_to_sigma_h_row302']:.4f} "
                f"minfrac={r['minimal_subset']['minimal_fraction_of_events']:.4f}"
            )

    all_reproduce = all(r["row302_reproduction"]["reproduces_to_1e-5"] for r in results)

    out = {
        "node": "rd-2d-bootstrap-jackknife",
        "branch": "K",
        "convention": "gradient-trapezoid weights (np.gradient(h_grid)), physics-floor zero handling"
        " -- the frozen T0 convention (results/prod2d_closure_20260818/tier0_bootstrap_jackknife.py,"
        " quoted verbatim in exec/m-head-rebaseline/READOUT_RECORD.md)",
        "truth": TRUTH,
        "sigma_h_row302_iiib2D_reference": SIGMA_H_ROW302,
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "width_N_list": WIDTH_N_LIST,
        "width_draws": WIDTH_DRAWS,
        "all_row302_targets_reproduced_to_1e-5": all_reproduce,
        "results": results,
    }

    out_path = NODE_DIR / "rd_2d_bootstrap_jackknife_output.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    print(f"ALL ROW302 TARGETS REPRODUCED TO 1e-5: {all_reproduce}")


if __name__ == "__main__":
    main()
