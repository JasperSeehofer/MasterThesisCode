"""A-7 re-score: recover the correspondence campaign's real numbers.

The mirror harness floors a zero per-event likelihood **in log space** at
``-1.0e300`` (``darksiren_emri/validation/correspondence_1d.py:1965`` and the
identical ``:2479``).  In float64 ``-1e300 + (-358.62) == -1e300`` exactly, so a
single zero-likelihood event annihilates the whole ensemble's log-likelihood at
that h-node.  This script re-scores every banked arm-seed from its retained
per-event ``event_likelihoods.csv`` using production's registered zero-handling
strategies (``bayesian_inference/posterior_combination.py``) instead.

Registered in ``PREREGISTRATION_1D_CORRESPONDENCE.md`` as AMENDMENT A-7.
GATE R-0 runs first and, on failure, aborts before any downstream number is read.

Usage:
    python results/prod2d_closure_20260818/rescore_sentinel.py [--out <path>]
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

# Verbatim from correspondence_1d.py:326-335 -- do NOT re-derive.
H_GRID_41: tuple[float, ...] = (
    0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.655, 0.66, 0.665, 0.67, 0.675, 0.68,
    0.685, 0.69, 0.695, 0.7, 0.705, 0.71, 0.715, 0.72, 0.725, 0.73, 0.735,
    0.74, 0.745, 0.75, 0.755, 0.76, 0.765, 0.77, 0.775, 0.78, 0.785, 0.79,
    0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86,
)  # fmt: skip
H_TRUE: float = 0.73
R_LOW_THRESHOLD: float = 0.605
SENTINEL: float = -1.0e300

REPO_ROOT = Path(__file__).resolve().parents[2]
ARMS_DIR = REPO_ROOT / "results/prod2d_closure_20260818/correspondence_arms"
CSV_ROOT = REPO_ROOT / "results/prod2d_closure_20260818/arm_event_likelihoods"

GATE_R0_TOL = 1.0e-9


def event_matrix(csv_path: Path) -> tuple[npt.NDArray[np.float64], list[str]]:
    """Pivot one run's per-event likelihoods to an (event, h) matrix on H_GRID_41.

    Returns the matrix and a list of structural anomalies found while pivoting
    (A-7 amendment E: the pivot is a load-bearing stage, so it reports rather
    than silently absorbing).
    """
    df = pd.read_csv(csv_path)
    grid = np.array(H_GRID_41, dtype=np.float64)
    anomalies: list[str] = []

    df = df[np.isin(df["h"].to_numpy(dtype=np.float64), grid)]
    dupes = int(df.duplicated(subset=["event_idx", "h"]).sum())
    if dupes:
        anomalies.append(f"{dupes} duplicated (event_idx, h) rows -- aggfunc='first' hides them")

    piv = df.pivot_table(
        index="event_idx", columns="h", values="combined_no_bh", aggfunc="first"
    ).reindex(columns=grid)
    vals = piv.to_numpy(dtype=np.float64)

    n_nan = int(np.isnan(vals).sum())
    if n_nan:
        anomalies.append(f"{n_nan} missing (event, h) cells -- events not present at every node")
    n_neg = int((vals < 0.0).sum())
    if n_neg:
        anomalies.append(f"{n_neg} negative likelihood values")
    return vals, anomalies


# ── zero-handling strategies ─────────────────────────────────────────────────


def _as_run(vals: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The defect, reproduced verbatim from correspondence_1d.py:1963-1965."""
    with np.errstate(divide="ignore", invalid="ignore"):
        log_l = np.where(vals > 0.0, np.log(vals, where=vals > 0.0), -np.inf)
    return np.nansum(np.where(np.isfinite(log_l), log_l, SENTINEL), axis=0)


def _physics_floor(vals: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """posterior_combination._physics_floor: per-event min-nonzero; drop all-zero events."""
    out = vals.copy()
    keep = np.ones(out.shape[0], dtype=bool)
    for i, row in enumerate(out):
        zero = row == 0.0
        if not zero.any():
            continue
        nonzero = row[~zero & ~np.isnan(row)]
        if nonzero.size == 0:
            keep[i] = False
        else:
            out[i, zero] = float(nonzero.min())
    return np.log(out[keep]).sum(axis=0)


def _per_event_floor(vals: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """posterior_combination._per_event_floor: min-nonzero/100; all-zero -> tiny."""
    out = vals.copy()
    for i, row in enumerate(out):
        zero = row == 0.0
        if not zero.any():
            continue
        nonzero = row[~zero & ~np.isnan(row)]
        out[i, zero] = np.finfo(float).tiny if nonzero.size == 0 else float(nonzero.min()) / 100.0
    return np.log(out).sum(axis=0)


def _exclude(vals: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """posterior_combination._exclude_zero_events: drop any event with any zero."""
    keep = ~(vals == 0.0).any(axis=1)
    if not keep.any():
        return np.full(vals.shape[1], np.nan)
    return np.log(vals[keep]).sum(axis=0)


def _clip_1e300(vals: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The house pattern used throughout validation/pp_coverage.py."""
    return np.log(np.clip(vals, 1.0e-300, None)).sum(axis=0)


STRATEGIES: dict[str, Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]] = {
    "as_run_sentinel": _as_run,
    "physics_floor": _physics_floor,
    "per_event_floor": _per_event_floor,
    "exclude": _exclude,
    "clip_1e-300": _clip_1e300,
}
PRIMARY = "physics_floor"


# ── statistics (verbatim convention of compute_seed_statistics) ──────────────


def _hpd_contains(
    post_n: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    target_idx: int,
    level: float,
) -> bool:
    order = np.argsort(-post_n)
    cum = 0.0
    for idx in order:
        cum += float(post_n[idx] * weights[idx])
        if idx == target_idx:
            return True
        if cum >= level:
            return False
    return False


def summarize(sum_log_l: npt.NDArray[np.float64], n_events: int) -> dict[str, Any]:
    grid = np.array(H_GRID_41, dtype=np.float64)
    weights = np.gradient(grid)
    if not np.isfinite(sum_log_l).any():
        return {
            "n_events": n_events,
            "mean_h": None,
            "map_h": None,
            "sigma_h": None,
            "c50": None,
            "c68": None,
            "c90": None,
            "r_low": None,
            "span_nats": None,
        }
    lp = sum_log_l - np.nanmax(sum_log_l)
    post = np.exp(lp)
    norm = float((post * weights).sum())
    post_n = post / norm if norm > 0 else post
    mean_h = float((post_n * grid * weights).sum())
    var = float((post_n * (grid - mean_h) ** 2 * weights).sum())
    sigma_h = float(np.sqrt(max(var, 0.0)))
    map_h = float(grid[int(np.nanargmax(sum_log_l))])
    target = int(np.nonzero(np.isclose(grid, H_TRUE))[0][0])
    finite = sum_log_l[np.isfinite(sum_log_l)]
    return {
        "n_events": n_events,
        "mean_h": mean_h,
        "map_h": map_h,
        "sigma_h": sigma_h,
        "c50": _hpd_contains(post_n, weights, target, 0.50),
        "c68": _hpd_contains(post_n, weights, target, 0.68),
        "c90": _hpd_contains(post_n, weights, target, 0.90),
        "r_low": map_h <= R_LOW_THRESHOLD,
        "span_nats": float(finite.max() - finite.min()) if finite.size else None,
    }


# ── ORIGIN sub-measurement (BAND O) ──────────────────────────────────────────


def origin_stats(csv_path: Path, vals: npt.NDArray[np.float64]) -> dict[str, Any]:
    """Underflow vs structural zeros, per BAND O."""
    df = pd.read_csv(csv_path)
    tiny = float(np.finfo(float).tiny)
    nz = vals[(vals > 0.0) & np.isfinite(vals)]
    zero_rows = df["combined_no_bh"] == 0.0
    comp_cols = [c for c in ("L_cat_no_bh", "B_num") if c in df.columns]
    comps_zero = (
        bool((df.loc[zero_rows, comp_cols] == 0.0).all().all())
        if comp_cols and zero_rows.any()
        else None
    )
    return {
        "n_zero_cells": int((vals == 0.0).sum()),
        "n_cells": int(vals.size),
        "events_with_any_zero": int((vals == 0.0).any(axis=1).sum()),
        "events_all_zero": int((vals == 0.0).all(axis=1).sum()),
        "nonzero_min": float(nz.min()) if nz.size else None,
        "n_subnormal_nonzero": int((nz < tiny).sum()) if nz.size else 0,
        "n_below_1e-200": int((nz < 1e-200).sum()) if nz.size else 0,
        "zero_rows_have_zero_components": comps_zero,
        "component_columns_checked": comp_cols,
    }


# ── driver ───────────────────────────────────────────────────────────────────


def csv_for(arm: str, seed: int) -> Path:
    return (
        CSV_ROOT
        / f"{arm}_seed{seed}"
        / f"seed{seed}"
        / "simulations/diagnostics/event_likelihoods.csv"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "results/prod2d_closure_20260818/rescore_sentinel_output.json"),
    )
    args = ap.parse_args()

    banked: list[dict[str, Any]] = []
    for p in sorted(ARMS_DIR.glob("*.json")):
        banked.append(json.loads(p.read_text()))

    grid46 = None
    per_seed: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    gate_r0a_rows: list[dict[str, Any]] = []

    for rec in banked:
        arm, seed = rec["arm"], int(rec["seed"])
        path = csv_for(arm, seed)
        if not path.is_file():
            per_seed.append({"arm": arm, "seed": seed, "status": "NO_CSV"})
            continue
        vals, anomalies = event_matrix(path)

        # sentinel contamination on the registered 41-node grid
        g46 = np.asarray(rec["h_grid"], dtype=np.float64)
        lp46 = np.asarray(rec["log_posterior"], dtype=np.float64)
        sel = np.isin(np.round(g46, 6), np.round(np.array(H_GRID_41), 6))
        lp41 = lp46[sel]
        k = np.round(lp41 / SENTINEL).astype(int)
        k[np.abs(lp41) < 1e100] = 0
        n_bad = int((k > 0).sum())
        grid46 = g46.size

        # GATE R-1: pairing provenance. run_arm_seed is idempotent (correspondence_1d.py:2549)
        # while work-roots persist across resubmissions, so a retrieved CSV is not guaranteed
        # to come from the same execution as its banked JSON.
        n_events_csv = int(vals.shape[0])
        paired = n_events_csv == int(rec["n_eff"])

        row: dict[str, Any] = {
            "arm": arm,
            "seed": seed,
            "status": "OK" if paired else "PAIRING-UNVERIFIED",
            "n_events_csv": n_events_csv,
            "n_eff_banked": int(rec["n_eff"]),
            "n_sentinel_nodes_41": n_bad,
            "pivot_anomalies": anomalies,
            "banked": {
                kk: rec[kk]
                for kk in ("mean_h", "map_h", "sigma_h", "c50", "c68", "c90", "r_low", "n_eff")
            },
            "origin": origin_stats(path, vals),
            "rescored": {},
        }
        for name, fn in STRATEGIES.items():
            row["rescored"][name] = summarize(fn(vals), vals.shape[0])
        per_seed.append(row)

        # GATE R-0a: reproduce the AS-RUN defective path on all 123 banked seeds. This is a
        # can-fail provenance control covering exactly the seeds R-0b cannot reach (the
        # contaminated ones), at zero compute.
        a = row["rescored"]["as_run_sentinel"]
        r0a = {
            kk: (None if a[kk] is None else abs(a[kk] - rec[kk])) for kk in ("mean_h", "sigma_h")
        }
        gate_r0a_rows.append(
            {
                "arm": arm,
                "seed": seed,
                "deltas": r0a,
                "map_match": a["map_h"] == rec["map_h"],
                "flags_match": all(a[f] == rec[f] for f in ("c50", "c68", "c90", "r_low")),
                "pass": all(d is not None and d <= GATE_R0_TOL for d in r0a.values())
                and a["map_h"] == rec["map_h"]
                and all(a[f] == rec[f] for f in ("c50", "c68", "c90", "r_low")),
            }
        )

        if n_bad == 0 and paired:
            r = row["rescored"][PRIMARY]
            deltas = {
                kk: (None if r[kk] is None else abs(r[kk] - rec[kk]))
                for kk in ("mean_h", "map_h", "sigma_h")
            }
            gate_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "deltas": deltas,
                    "pass": all(d is not None and d <= GATE_R0_TOL for d in deltas.values()),
                }
            )

    gate_pass = bool(gate_rows) and all(g["pass"] for g in gate_rows)
    unpaired = [
        {
            "arm": r["arm"],
            "seed": r["seed"],
            "n_events_csv": r["n_events_csv"],
            "n_eff_banked": r["n_eff_banked"],
        }
        for r in per_seed
        if r.get("status") == "PAIRING-UNVERIFIED"
    ]
    r0a_fail = [g for g in gate_r0a_rows if not g["pass"]]
    out = {
        "gate_r0a_as_run_provenance": {
            "n_seeds": len(gate_r0a_rows),
            "n_fail": len(r0a_fail),
            "verdict": "PASS" if not r0a_fail else "FAIL",
            "failures": r0a_fail,
        },
        "gate_r0b_noop_identity": {
            "n_zero_sentinel_seeds": len(gate_rows),
            "tolerance": GATE_R0_TOL,
            "verdict": "PASS" if gate_pass else "FAIL",
            "rows": gate_rows,
        },
        "gate_r1": {
            "n_pairing_unverified": len(unpaired),
            "verdict": "PASS" if not unpaired else "PARTIAL",
            "excluded": unpaired,
        },
        "h_grid_full_nodes": grid46,
        "primary_strategy": PRIMARY,
        "per_seed": per_seed,
    }
    # Per-arm aggregation under every strategy (BAND S + BAND V inputs).
    by_arm: dict[str, dict[str, Any]] = {}
    for r in per_seed:
        if r.get("status") == "NO_CSV":
            continue
        by_arm.setdefault(r["arm"], {"n": 0, "published": [], "strategies": {}})
        by_arm[r["arm"]]["n"] += 1
        by_arm[r["arm"]]["published"].append(r["banked"]["mean_h"])
        for name in STRATEGIES:
            by_arm[r["arm"]]["strategies"].setdefault(name, {"mean_h": [], "c68": [], "r_low": []})
            s = r["rescored"][name]
            by_arm[r["arm"]]["strategies"][name]["mean_h"].append(s["mean_h"])
            by_arm[r["arm"]]["strategies"][name]["c68"].append(s["c68"])
            by_arm[r["arm"]]["strategies"][name]["r_low"].append(s["r_low"])
    summary: dict[str, Any] = {}
    for arm, d in sorted(by_arm.items()):
        pub = float(np.mean(d["published"]))
        ent: dict[str, Any] = {"n": d["n"], "published_mean_h": pub, "published_bias": pub - H_TRUE}
        for name, s in d["strategies"].items():
            m = np.array([x for x in s["mean_h"] if x is not None], dtype=np.float64)
            ent[name] = {
                "mean_h": float(m.mean()) if m.size else None,
                "bias": float(m.mean() - H_TRUE) if m.size else None,
                "sem": float(m.std(ddof=1) / np.sqrt(m.size)) if m.size > 1 else None,
                "c68_frac": float(np.mean([bool(x) for x in s["c68"] if x is not None])),
                "r_low_frac": float(np.mean([bool(x) for x in s["r_low"] if x is not None])),
            }
        prim = ent[PRIMARY]["mean_h"]
        ent["delta_vs_published"] = None if prim is None else prim - pub
        spread = [
            ent[n]["mean_h"]
            for n in ("physics_floor", "per_event_floor", "exclude", "clip_1e-300")
            if ent[n]["mean_h"] is not None
        ]
        ent["strategy_spread"] = float(max(spread) - min(spread)) if len(spread) > 1 else None
        summary[arm] = ent
    out["per_arm"] = summary
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(
        json.dumps(
            {
                "gate_r0a_as_run_provenance": out["gate_r0a_as_run_provenance"]["verdict"],
                "n_r0a_fail": len(r0a_fail),
                "gate_r0b_noop_identity": out["gate_r0b_noop_identity"]["verdict"],
                "n_r0b_seeds": len(gate_rows),
                "gate_r1": out["gate_r1"]["verdict"],
                "n_pairing_unverified": len(unpaired),
                "n_seeds": len(per_seed),
                "out": args.out,
            },
            indent=2,
        )
    )
    if not gate_pass or r0a_fail:
        print("GATE R-0 FAILED -- no downstream number may be read.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
