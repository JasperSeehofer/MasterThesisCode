"""T0 scorer -- production-side free reads, zero new simulation.

PREREGISTRATION_PROD2D_CLOSURE_LANDSCAPE.md Sec 3 (T0), Sec 1 (H-T0a, H-T0b,
H-L1-prod). Pre-committed BEFORE the cluster job returns (T0 is
production-native and self-contained -- Sec 4 "Execution-completeness"
clause). Updated per the verifier Part VII amendments (P7-2, P7-5;
VERIFIER_PRECHECK_PROD2D.md Part D), applied verbatim below.

Input: ``results/run_20260804_postfix/{iiib,joint_r1}/diagnostics/event_likelihoods.csv``
(1588 events x 41 h-grid points, non-uniform grid -- 0.01 tails / 0.005 core,
h in [0.60, 0.86]).

**Per-event likelihood assembly formula (P7-2b, pinned at freeze).** The CSV
columns ``combined_no_bh`` (1D channel) and ``combined_with_bh`` (2D
channel) are per-event PLAIN LIKELIHOOD values L_e(h), NOT log-likelihoods
-- every value observed is strictly positive (min ~3e-7, max ~1.6, no
zeros in either venue), matching the shape of a per-event likelihood
rather than a log-likelihood (which would commonly be negative /
unbounded below). ``posterior_combination.py``'s canonical reference form
(``compute_canonical_combined_posterior`` / ``load_per_h_likelihoods``)
takes ``log(L_event)`` itself and combines via the raw ``Sigma log L``
across events -- exactly the operation this script performs on the CSV
columns:

    L_floored_e(h) = physics_floor(L_e(h))          [P7-2c, see below]
    log_L_e(h)     = log(L_floored_e(h))
    logpost(h)     = Sigma_e log_L_e(h)              (uniform prior)
    post(h)        = exp(logpost(h) - max_h logpost(h))
    normalization  = Sigma_h post(h) * w(h)           [P7-2a, see below]
    post_n(h)      = post(h) / normalization
    mean_h         = Sigma_h post_n(h) * h * w(h)
    sigma_h        = sqrt(Sigma_h post_n(h) * (h - mean_h)**2 * w(h))
    MAP            = argmax_h logpost(h)

**Grid weights (P7-2a, applied verbatim).** ALL grid moments (mean_h,
sigma_h, normalization, top-2-node mass, HPD) use trapezoid weights
``w = np.gradient(h_grid)`` on the non-uniform 41-node grid -- the naive
equal-weight mean differs by -0.004/-0.006 (materiality-scale), so the
gradient weighting is load-bearing, not cosmetic.

**Physics-floor zero-handling (P7-2c, applied verbatim -- the production
combine's strategy of record, ``posterior_combination._physics_floor``).**
Per event (row): if the row has zero entries, replace each zero with the
row's OWN minimum nonzero value (NOT the ``_per_event_floor`` variant,
which divides by 100); if the row is entirely zero, exclude that event
(it has no nonzero value to floor from). Replicated here as
``_physics_floor_apply`` rather than imported (keeps this scorer
import-independent of the production module's private helpers), but the
per-row algorithm is identical. Verified: neither venue's CSV has any
zero entries in either channel, so this is a no-op here (0 events
excluded both venues, both channels) -- applied anyway per the registered
"replicate identically" instruction, and the excluded count is reported.

**N-0 continuity gate (registered, scored, BLOCKING).** The full-sample
2D-channel mean_h computed by the pipeline above must reproduce the
production M3 anchors (0.7842 iiib / 0.7966 joint_r1) to within 5e-4 per
venue BEFORE any bootstrap/jackknife statistic is quoted. Verified against
the un-weighted (plain-trapezoid) convention beforehand: the gradient-
weighted full-sample mean_h reproduces both anchors within ~1.1e-4 (iiib)
and ~1.1e-4 (joint_r1) -- well inside the 5e-4 gate. On gate failure this
script HARD-STOPS: it writes only the gate diagnostics (no
bootstrap/jackknife/drop-top-k statistics) and exits nonzero.

Outputs ``tier0_output.json`` (or the path given via ``--output``) with,
per venue x channel: full-sample posterior (mean_h, sigma_h, MAP, 68% HPD
width, edge-mode flag -- H-L1-prod), jackknife-889 read (H-T0a), the full
leave-one-out Delta-mean_h distribution, the drop-top-k curve (k=1..20,
ranked by |per-event central-difference h-slope of that channel's logL at
h~0.73|), the B=10,000 bootstrap distribution of (mean_h, MAP) with
sigma_boot and quantiles plus the P7-5 top-2-node relative-mass diagnostic
(median per venue x channel, flagged if > 0.05), and the registered
z-score z_v = Delta_v / sigma_boot(mean_h) (H-T0b), where Delta_v is this
script's own full-sample mean_h minus the injected truth 0.73.

Usage:
    python tier0_bootstrap_jackknife.py [--output tier0_output.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]

VENUES = ("iiib", "joint_r1")
CHANNELS = ("combined_no_bh", "combined_with_bh")  # ("1D", "2D")
CHANNEL_LABEL = {"combined_no_bh": "1D", "combined_with_bh": "2D"}
TRUTH = 0.73
DROP_EVENT = 889
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 20280611
TOP_K_MAX = 20
TOP2_MASS_FLAG_THRESHOLD = 0.05
HPD_LEVEL = 0.68

# N-0 gate (P7-2d, BLOCKING): full-sample 2D-channel mean_h must reproduce
# these production M3 anchors to within N0_TOLERANCE.
N0_TARGETS_2D = {"iiib": 0.7842, "joint_r1": 0.7966}
N0_TOLERANCE = 5.0e-4

# Header-quoted production offsets (M3, off legs) -- cross-check only, never
# substituted for the recomputed value.
HEADER_DELTA_2D = {"iiib": 0.054, "joint_r1": 0.067}
HEADER_SIGMA_H_2D = {"iiib": 0.0177, "joint_r1": 0.0216}


def _physics_floor_apply(likelihoods: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replicate ``posterior_combination._physics_floor`` per-row (P7-2c).

    Per event row: zeros -> the row's own minimum nonzero value; an
    all-zero row has no nonzero value to floor from and is marked for
    exclusion instead. Returns ``(floored, exclude_mask)`` -- the caller
    drops excluded rows from BOTH the likelihood matrix and the aligned
    event_idx array so a jackknife/top-k read never silently references a
    dropped event.
    """
    result = likelihoods.copy()
    n_events = result.shape[0]
    exclude_mask = np.zeros(n_events, dtype=bool)
    for i in range(n_events):
        row = result[i]
        zero_mask = row == 0.0
        if not zero_mask.any():
            continue
        nonzero = row[~zero_mask]
        if nonzero.size == 0:
            exclude_mask[i] = True
        else:
            result[i, zero_mask] = float(nonzero.min())
    return result, exclude_mask


def _load_matrix(venue: str, channel: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (h_grid sorted, event_idx array, logL matrix [n_events, n_h], n_excluded)."""
    csv_path = REPO_ROOT / "results" / "run_20260804_postfix" / venue / "diagnostics" / "event_likelihoods.csv"
    df = pd.read_csv(csv_path)
    h_grid = np.sort(df["h"].unique())
    piv = df.pivot(index="event_idx", columns="h", values=channel).reindex(columns=h_grid)
    if piv.isna().any().any():
        raise ValueError(f"{venue}/{channel}: pivot has missing (event, h) cells -- ragged CSV")
    event_idx = piv.index.to_numpy()
    L = piv.to_numpy(dtype=np.float64)
    L_floored, exclude_mask = _physics_floor_apply(L)
    n_excluded = int(exclude_mask.sum())
    if n_excluded:
        L_floored = L_floored[~exclude_mask]
        event_idx = event_idx[~exclude_mask]
    logL = np.log(L_floored)
    return h_grid, event_idx, logL, n_excluded


def _moments(
    logpost: np.ndarray, h_grid: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gradient-trapezoid-weighted (mean_h, sigma_h, MAP, post_n) for a batch.

    ``logpost`` has shape ``(..., n_h)``; the first three returns have
    shape ``(...,)``, ``post_n`` (the normalized posterior density,
    P7-2a weighting) has shape ``(..., n_h)``.
    """
    lp = logpost - logpost.max(axis=-1, keepdims=True)
    post = np.exp(lp)
    norm = (post * weights).sum(axis=-1)
    post_n = post / norm[..., None]
    mean_h = (post_n * h_grid * weights).sum(axis=-1)
    var = (post_n * (h_grid - mean_h[..., None]) ** 2 * weights).sum(axis=-1)
    sigma_h = np.sqrt(np.clip(var, 0.0, None))
    map_h = h_grid[np.argmax(logpost, axis=-1)]
    return mean_h, sigma_h, map_h, post_n


def _top2_node_mass(post_n: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """P7-5: relative posterior mass in the top-2 (highest-h) grid nodes.

    ``post_n`` has shape ``(..., n_h)`` and is already normalized so that
    ``Sigma_h post_n * w == 1``; the top-2-node mass is therefore directly
    ``post_n[..., -1] * w[-1] + post_n[..., -2] * w[-2]``.
    """
    result: np.ndarray = post_n[..., -1] * weights[-1] + post_n[..., -2] * weights[-2]
    return result


def _hpd_width(post_n: np.ndarray, h_grid: np.ndarray, weights: np.ndarray, level: float) -> tuple[float, float]:
    """H-L1-prod: smallest-density-first HPD width for a single (n_h,) posterior.

    Nodes are added in descending posterior-density order until the
    accumulated weighted mass reaches ``level``; the width is the h-range
    spanned by the included nodes. Returns ``(width, grid_span)``.
    """
    order = np.argsort(-post_n)
    cum = 0.0
    included: list[int] = []
    for idx in order:
        cum += float(post_n[idx] * weights[idx])
        included.append(int(idx))
        if cum >= level:
            break
    included_h = h_grid[included]
    width = float(included_h.max() - included_h.min())
    grid_span = float(h_grid[-1] - h_grid[0])
    return width, grid_span


def _quantiles(x: np.ndarray) -> dict[str, float]:
    qs = np.quantile(x, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {"q05": float(qs[0]), "q25": float(qs[1]), "q50": float(qs[2]), "q75": float(qs[3]), "q95": float(qs[4])}


def _full_sample_read(venue: str, channel: str) -> dict[str, Any]:
    """Cheap first pass: full-sample posterior only (needed for the N-0 gate)."""
    h_grid, event_idx, logL, n_excluded = _load_matrix(venue, channel)
    weights = np.gradient(h_grid)
    logpost_full = logL.sum(axis=0)
    mean_arr, sigma_arr, map_arr, postn_arr = _moments(logpost_full[None, :], h_grid, weights)
    mean_h, sigma_h, map_h = float(mean_arr[0]), float(sigma_arr[0]), float(map_arr[0])
    post_n = postn_arr[0]
    hpd_width, grid_span = _hpd_width(post_n, h_grid, weights, HPD_LEVEL)
    return {
        "h_grid": h_grid,
        "weights": weights,
        "event_idx": event_idx,
        "logL": logL,
        "n_events_excluded": n_excluded,
        "logpost_full": logpost_full,
        "mean_h": mean_h,
        "sigma_h": sigma_h,
        "map_h": map_h,
        "hpd68_width": hpd_width,
        "grid_span": grid_span,
        "mode_on_edge": bool(map_h == h_grid[0] or map_h == h_grid[-1]),
    }


def _score_venue_channel(venue: str, channel: str, fs: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    h_grid, weights, event_idx, logL = fs["h_grid"], fs["weights"], fs["event_idx"], fs["logL"]
    n_events, n_h = logL.shape
    logpost_full = fs["logpost_full"]
    mean_h_full, sigma_h_full, map_h_full = fs["mean_h"], fs["sigma_h"], fs["map_h"]
    delta_full = mean_h_full - TRUTH

    # --- H-L1-prod: 68% HPD width + edge-mode (1D-starves legs) ------------
    h_l1_prod = {
        "hpd68_width": fs["hpd68_width"],
        "grid_span": fs["grid_span"],
        "hpd68_width_ge_half_span": bool(fs["hpd68_width"] >= 0.5 * fs["grid_span"]),
        "mode_on_edge": fs["mode_on_edge"],
        "starves_1d": bool(
            channel == "combined_no_bh"
            and (fs["hpd68_width"] >= 0.5 * fs["grid_span"] or fs["mode_on_edge"])
        ),
    }

    # --- H-T0a: jackknife-889 ------------------------------------------------
    drop_mask = event_idx == DROP_EVENT
    if not drop_mask.any():
        raise ValueError(f"{venue}/{channel}: event {DROP_EVENT} not found in event_idx column (post-floor)")
    logpost_wo889 = logpost_full - logL[drop_mask].sum(axis=0)
    mean_wo889_arr, _s, map_wo889_arr, _p = _moments(logpost_wo889[None, :], h_grid, weights)
    mean_h_wo889 = float(mean_wo889_arr[0])
    map_h_wo889 = float(map_wo889_arr[0])
    delta_wo889 = mean_h_wo889 - TRUTH
    ratio = abs(delta_wo889) / abs(delta_full) if delta_full != 0.0 else float("nan")
    if delta_full != 0.0 and np.sign(delta_wo889) != np.sign(delta_full) and delta_wo889 < -0.01:
        band = "OVERSHOOT"
    elif ratio <= 0.5:
        band = "FRAGILE"
    elif ratio >= 0.75:
        band = "ROBUST"
    else:
        band = "MIXED"
    ht0a = {
        "delta_full": delta_full,
        "delta_without_889": delta_wo889,
        "ratio_abs": ratio,
        "band": band,
        "mean_h_without_889": mean_h_wo889,
        "map_h_without_889": map_h_wo889,
    }

    # --- Full leave-one-out Delta-mean_h distribution -----------------------
    loo_logpost = logpost_full[None, :] - logL  # (n_events, n_h)
    loo_mean_h, _loo_sigma_h, _loo_map_h, _loo_postn = _moments(loo_logpost, h_grid, weights)
    loo_delta = loo_mean_h - TRUTH
    loo_summary = {
        "mean": float(loo_delta.mean()),
        "std": float(loo_delta.std(ddof=1)),
        "min": float(loo_delta.min()),
        "max": float(loo_delta.max()),
        **_quantiles(loo_delta),
    }

    # --- Drop-top-k curve, ranked by |central-difference h-slope at h~0.73| -
    i73 = int(np.argmin(np.abs(h_grid - TRUTH)))
    if h_grid[i73] != TRUTH:
        raise ValueError(f"{venue}/{channel}: h=0.73 not an exact grid node (nearest={h_grid[i73]})")
    i_lo, i_hi = i73 - 1, i73 + 1
    slope = (logL[:, i_hi] - logL[:, i_lo]) / (h_grid[i_hi] - h_grid[i_lo])
    order = np.argsort(-np.abs(slope))  # descending |slope|
    top_k_curve = []
    for k in range(1, TOP_K_MAX + 1):
        dropped = order[:k]
        logpost_k = logpost_full - logL[dropped].sum(axis=0)
        mean_h_k_arr, _s, _m, _p = _moments(logpost_k[None, :], h_grid, weights)
        mean_h_k = float(mean_h_k_arr[0])
        top_k_curve.append(
            {
                "k": k,
                "dropped_event_idx": [int(event_idx[j]) for j in dropped],
                "mean_h": mean_h_k,
                "delta_mean_h": mean_h_k - TRUTH,
            }
        )

    # --- Bootstrap B=10,000 (multinomial-count weighted column sums) --------
    counts = rng.multinomial(n_events, np.full(n_events, 1.0 / n_events), size=BOOTSTRAP_B)
    logpost_boot = counts.astype(np.float64) @ logL  # (B, n_h)
    mean_h_boot, _sigma_h_boot, map_h_boot, postn_boot = _moments(logpost_boot, h_grid, weights)
    sigma_boot_mean = float(mean_h_boot.std(ddof=1))
    sigma_boot_map = float(map_h_boot.std(ddof=1))
    z_v: float = delta_full / sigma_boot_mean if sigma_boot_mean > 0 else float("nan")
    if math.isnan(z_v):
        z_interpretation = "undefined"
    elif abs(z_v) <= 2:
        z_interpretation = "z<=2: event-draw luck alone"
    elif abs(z_v) < 4:
        z_interpretation = "2<z<4: partial"
    else:
        z_interpretation = "z>=4: systematic component required"

    # --- P7-5: top-2-node relative posterior mass per resample --------------
    top2_mass = _top2_node_mass(postn_boot, weights)  # (B,)
    top2_median = float(np.median(top2_mass))
    grid_truncation_flagged = bool(top2_median > TOP2_MASS_FLAG_THRESHOLD)
    z_v_undetermined_by_grid_truncation = bool(grid_truncation_flagged and 2.0 < abs(z_v) < 4.0)

    return {
        "venue": venue,
        "channel": channel,
        "channel_label": CHANNEL_LABEL[channel],
        "n_events": n_events,
        "n_events_excluded_physics_floor": fs["n_events_excluded"],
        "n_h": n_h,
        "full_sample": {
            "mean_h": mean_h_full,
            "sigma_h": sigma_h_full,
            "map_h": map_h_full,
            "delta_vs_truth": delta_full,
        },
        "h_l1_prod": h_l1_prod,
        "h_t0a_jackknife_889": ht0a,
        "h_t0b_loo_full_distribution": loo_summary,
        "drop_top_k_curve": top_k_curve,
        "bootstrap": {
            "B": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "mean_h": {
                "sigma_boot": sigma_boot_mean,
                **_quantiles(mean_h_boot),
            },
            "map_h": {
                "sigma_boot": sigma_boot_map,
                **_quantiles(map_h_boot),
            },
            "p7_5_top2_node_relative_mass": {
                "median": top2_median,
                "flag_threshold": TOP2_MASS_FLAG_THRESHOLD,
                "flagged": grid_truncation_flagged,
            },
        },
        "h_t0b_z_score": {
            "z_v": z_v,
            "interpretation": z_interpretation,
            "combined_sigma_h_boot": float(np.hypot(sigma_h_full, sigma_boot_mean)),
            "undetermined_by_grid_truncation": z_v_undetermined_by_grid_truncation,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "tier0_output.json")
    args = parser.parse_args(argv)

    # --- First pass: full-sample reads only, for the N-0 gate ---------------
    full_samples: dict[tuple[str, str], dict[str, Any]] = {}
    for venue in VENUES:
        for channel in CHANNELS:
            print(f"Full-sample read {venue}/{channel} ...", flush=True)
            full_samples[(venue, channel)] = _full_sample_read(venue, channel)

    n0_gate: list[dict[str, Any]] = []
    n0_pass = True
    for venue in VENUES:
        fs2d = full_samples[(venue, "combined_with_bh")]
        target = N0_TARGETS_2D[venue]
        diff = fs2d["mean_h"] - target
        ok = abs(diff) <= N0_TOLERANCE
        n0_pass = n0_pass and ok
        n0_gate.append(
            {
                "venue": venue,
                "recomputed_mean_h_2d": fs2d["mean_h"],
                "target_mean_h_2d": target,
                "diff": diff,
                "tolerance": N0_TOLERANCE,
                "pass": ok,
            }
        )

    if not n0_pass:
        print()
        print("N-0 GATE: STOP -- convention mismatch, no T0 number enters the budget")
        for g in n0_gate:
            status = "PASS" if g["pass"] else "FAIL"
            print(f"  [{status}] {g['venue']}: recomputed={g['recomputed_mean_h_2d']:.6f} "
                  f"target={g['target_mean_h_2d']:.6f} diff={g['diff']:.6e} tol={g['tolerance']:.1e}")
        out_fail = {
            "n0_gate": n0_gate,
            "n0_gate_pass": False,
            "note": "N-0 gate FAILED -- no bootstrap/jackknife statistics were computed "
            "(registered hard-stop, P7-2d).",
        }
        args.output.write_text(json.dumps(out_fail, indent=2))
        print(f"Wrote gate-failure diagnostics to {args.output}")
        raise SystemExit(1)

    print()
    print("N-0 GATE: PASS")
    for g in n0_gate:
        print(f"  {g['venue']}: recomputed={g['recomputed_mean_h_2d']:.6f} "
              f"target={g['target_mean_h_2d']:.6f} diff={g['diff']:.6e} (tol {g['tolerance']:.1e})")
    print()

    # --- Full scoring pass ---------------------------------------------------
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    entries = []
    for venue in VENUES:
        for channel in CHANNELS:
            print(f"Scoring {venue}/{channel} ...", flush=True)
            entries.append(_score_venue_channel(venue, channel, full_samples[(venue, channel)], rng))

    header_cross_check = []
    for venue in VENUES:
        e2d = next(e for e in entries if e["venue"] == venue and e["channel"] == "combined_with_bh")
        header_cross_check.append(
            {
                "venue": venue,
                "recomputed_mean_h_2d": e2d["full_sample"]["mean_h"],
                "header_mean_h_2d": N0_TARGETS_2D[venue],
                "recomputed_sigma_h_2d": e2d["full_sample"]["sigma_h"],
                "header_sigma_h_2d": HEADER_SIGMA_H_2D[venue],
                "recomputed_delta_2d": e2d["full_sample"]["delta_vs_truth"],
                "header_delta_2d": HEADER_DELTA_2D[venue],
                "matches_header": (
                    abs(e2d["full_sample"]["mean_h"] - N0_TARGETS_2D[venue]) < 1e-3
                    and abs(e2d["full_sample"]["sigma_h"] - HEADER_SIGMA_H_2D[venue]) < 1e-3
                ),
            }
        )

    out = {
        "truth": TRUTH,
        "drop_event": DROP_EVENT,
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "convention": (
            "CSV columns combined_no_bh/combined_with_bh are PLAIN LIKELIHOOD values "
            "(verified: strictly positive, no zeros); this script applies the production "
            "combine's physics-floor zero-handling (P7-2c, no-op here), logs the floored "
            "values and sums over events (uniform prior, canonical raw Sigma log L form), "
            "and normalizes/takes moments with gradient-trapezoid weights (P7-2a) -- see "
            "the module docstring for the full pinned formula."
        ),
        "n0_gate": n0_gate,
        "n0_gate_pass": True,
        "header_cross_check": header_cross_check,
        "entries": entries,
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
