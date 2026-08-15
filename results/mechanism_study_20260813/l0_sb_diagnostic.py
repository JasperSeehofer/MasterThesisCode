"""L0-SB sandwich/score-balance diagnostic (PROPOSAL_STAGE3_20260815.md, ledger row #105).

Tests H-SB -- "the residual displacement is the misspecification (score-balance)
mechanism": under a misspecified likelihood (White 1982; Kleijn & van der Vaart
2012) the MAP converges to the pseudo-true (KL-argmin) value and the posterior
spread is sandwich-form A^-1 B A^-1, not the inverse-information A^-1. All
quantities below are recomputed from the raw per-seed ``ln_post_1d``/``ln_post_2d``
grid vectors and per-seed ``map_*``/``post_sd_*`` fields in the committed result
JSONs. Never reads a file's ``aggregate`` block.

Two independent local estimators are used, both grid-based, no smoothing:

* **Score (first derivative) at truth**, ``T`` -- the same central-difference
  geometry as ``m6r_l0_decomposition._slope_at_truth_per_seed`` /
  ``darksiren_emri.validation.venue_transfer._slope_at_truth`` (grid neighbours
  either side of ``h_true``), reused verbatim so the T-values here are directly
  comparable to the previously verified 0.749 +/- 0.046 closure.
* **Curvature (second derivative)**, ``A`` -- an *independent* local quadratic
  least-squares fit ``ln_post(h) ~= c0 + c1*(h-h0) + c2*(h-h0)^2`` over a small
  window of grid neighbours around a target point ``h0``; ``A = -2*c2``. Used at
  two locations per seed: at the seed's own grid-argmax (MAP curvature, Section
  1) and at ``h_true`` (Section 2's per-cell mean curvature Abar). The fit's own
  ``c1`` at ``h0=h_true`` gives a second, cross-check estimate of the score,
  reported alongside the central-difference T but not used in the headline
  ratios (kept separate so the T-values match the previously verified number
  exactly).

Sections:

1. Per-seed information width: A (curvature at MAP) -> sigma_A = A^-1/2,
   compared to the stored post_sd (grid-moment sd of the full posterior).
2. Per-cell sandwich width: B = Var_seeds[score at truth], Abar = mean curvature
   at truth (both channels) -> sigma_SW = sqrt(B)/Abar; predicted overconfidence
   sigma_SW / sigma_A (cell-median sigma_A from Section 1).
3. H-SB's three parameter-free, quantitative predictions:
   (a) bias ~= T/Abar (pseudo-true displacement) vs the measured MAP bias, and
       whether replacing 1/sigma^2_post (the M6R closure route, factor ~0.749)
       with Abar closes the ratio to ~1;
   (b) does sigma_SW predict the observed bias/post_sd ~8.5x overconfidence
       scale, i.e. is post_sd ~= sigma_A << sigma_SW;
   (c) counterexample flag: any cell where either prediction is off by >3x.
4. Verdict per the falsification-first default.

Output: ``L0_SB_output.json`` next to this file. Report:
``L0_SB_DIAGNOSTIC_20260815.md`` (PRESENTED, NOT ADJUDICATED).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

RESULTS_DIR = Path(__file__).parent
VENUE_TRANSFER_DIR = RESULTS_DIR.parent / "venue_transfer_20260811"

# Known-degenerate S-cells (f_h = 0, near-delta posterior -- host resolved
# exactly in >=59/60 seeds; post_sd ~ 0 identically). Reported, excluded from
# headline cell-level statistics, exactly as in m6r_l0_decomposition.py
# Sections 3-4.
DEGENERATE_FH0_CELLS = {"S00", "S01", "S02", "S03"}
# f_i = 0 column: near-degenerate on the other axis (M6R Section 4 caveat) --
# reported, flagged, excluded from headline statistics.
EDGE_FI0_CELLS = {"S10", "S20", "S30"}


def _is_degenerate(d: dict[str, Any], cid: str) -> bool:
    """A cell/arm is degenerate if it is a known f_h=0 S-cell, or if its
    stored post_sd is exactly 0 for every seed in either channel (near-delta
    posterior -- host resolved exactly, e.g. MEI, venue-transfer T0). Detected
    from the data, not just the naming convention, so arms outside the S-cell
    dose grid (MEI, T0, ...) are caught too.
    """
    if cid in DEGENERATE_FH0_CELLS:
        return True
    for ch in ("1d", "2d"):
        sds = np.array([ps[f"post_sd_{ch}"] for ps in d["per_seed"]], dtype=np.float64)
        if np.all(sds == 0.0):
            return True
    return False


def _load_json(path: str) -> dict[str, Any]:
    with open(path) as fh:
        result: dict[str, Any] = json.load(fh)
    return result


def _load_mechanism_study_files() -> dict[str, dict[str, Any]]:
    """All 22 committed mechanism-study result files (16 S-cells + 6 arms)."""
    files = sorted(glob.glob(str(RESULTS_DIR / "*_h0p730_results_seeds*.json")))
    cells: dict[str, dict[str, Any]] = {}
    for f in files:
        d = _load_json(f)
        cells[d["config"]["cell"]] = d
    return cells


def _load_venue_transfer_files() -> dict[str, dict[str, Any]]:
    """Venue-transfer h0p730 per-seed records, merged across seed-batch files
    per cell (T0, Ta, Tb, Tc). Returns {} if the directory is not present on
    this machine -- committed data only, no raw aggregate.
    """
    if not VENUE_TRANSFER_DIR.is_dir():
        return {}
    files = sorted(glob.glob(str(VENUE_TRANSFER_DIR / "*_h0p730_results_seeds*.json")))
    if not files:
        return {}
    merged: dict[str, dict[str, Any]] = {}
    for f in files:
        d = _load_json(f)
        cid = d["config"]["cell"]
        if cid not in merged:
            merged[cid] = d
        else:
            merged[cid]["per_seed"] = merged[cid]["per_seed"] + d["per_seed"]
    return merged


def _mean_se(x: npt.NDArray[np.float64]) -> tuple[float, float]:
    n = x.size
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return mean, se


def _slope_at_truth_per_seed(
    d: dict[str, Any],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-seed central-difference d(ln_post)/dh at truth, both channels.

    Verbatim geometry match to ``venue_transfer._slope_at_truth`` /
    ``m6r_l0_decomposition._slope_at_truth_per_seed``: grid neighbours of
    ``h_true``, central difference, applied to the raw per-seed vectors.
    """
    hg = np.asarray(d["config"]["h_grid"], dtype=np.float64)
    h_true = float(d["config"]["h_true"])
    i_true = int(np.argmin(np.abs(hg - h_true)))
    lo, hi = i_true - 1, i_true + 1
    dh = hg[hi] - hg[lo]
    s1 = np.array(
        [
            (np.asarray(ps["ln_post_1d"])[hi] - np.asarray(ps["ln_post_1d"])[lo]) / dh
            for ps in d["per_seed"]
        ]
    )
    s2 = np.array(
        [
            (np.asarray(ps["ln_post_2d"])[hi] - np.asarray(ps["ln_post_2d"])[lo]) / dh
            for ps in d["per_seed"]
        ]
    )
    return s1, s2


def _local_quadratic_fit(
    h_grid: npt.NDArray[np.float64],
    ln_post: npt.NDArray[np.float64],
    center_idx: int,
    half_window: int = 2,
) -> tuple[float, float]:
    """Local quadratic least-squares fit around ``h_grid[center_idx]``.

    Fits ``ln_post(h) ~= c0 + c1*(h-h0) + c2*(h-h0)^2`` over up to
    ``2*half_window + 1`` grid neighbours (fewer near an edge or where
    ``ln_post`` is non-finite). Returns ``(score, curvature)`` where
    ``score = c1`` (d(ln_post)/dh at h0) and ``curvature = -2*c2``
    (-d^2(ln_post)/dh^2 at h0). Returns ``(nan, nan)`` if fewer than 3
    finite points are available.
    """
    n = h_grid.size
    lo = max(0, center_idx - half_window)
    hi = min(n, center_idx + half_window + 1)
    xs = h_grid[lo:hi]
    ys = ln_post[lo:hi]
    finite = np.isfinite(ys)
    xs, ys = xs[finite], ys[finite]
    if xs.size < 3:
        return float("nan"), float("nan")
    h0 = h_grid[center_idx]
    coeffs = np.polyfit(xs - h0, ys, 2)
    c2, c1 = float(coeffs[0]), float(coeffs[1])
    return c1, -2.0 * c2


def _map_curvature_per_seed(d: dict[str, Any], channel: str) -> npt.NDArray[np.float64]:
    """Per-seed A = -d^2(ln_post)/dh^2 at the seed's own grid-argmax MAP."""
    hg = np.asarray(d["config"]["h_grid"], dtype=np.float64)
    out = []
    for ps in d["per_seed"]:
        lp = np.asarray(ps[f"ln_post_{channel}"], dtype=np.float64)
        finite = np.isfinite(lp)
        if not np.any(finite):
            out.append(float("nan"))
            continue
        idx = int(np.argmax(np.where(finite, lp, -np.inf)))
        _, curv = _local_quadratic_fit(hg, lp, idx)
        out.append(curv)
    return np.asarray(out, dtype=np.float64)


def _truth_score_and_curvature_per_seed(
    d: dict[str, Any], channel: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-seed (score, curvature) at h_true from the local quadratic fit
    (cross-check estimator, independent of the central-difference T)."""
    hg = np.asarray(d["config"]["h_grid"], dtype=np.float64)
    h_true = float(d["config"]["h_true"])
    i_true = int(np.argmin(np.abs(hg - h_true)))
    scores, curvs = [], []
    for ps in d["per_seed"]:
        lp = np.asarray(ps[f"ln_post_{channel}"], dtype=np.float64)
        s, c = _local_quadratic_fit(hg, lp, i_true)
        scores.append(s)
        curvs.append(c)
    return np.asarray(scores, dtype=np.float64), np.asarray(curvs, dtype=np.float64)


def _cell_bias_and_post_sd(
    d: dict[str, Any], channel: str
) -> tuple[float, float, npt.NDArray[np.float64]]:
    """Mean MAP bias, median post_sd, and per-seed post_sd for one channel."""
    h_true = float(d["config"]["h_true"])
    maps = np.array([ps[f"map_{channel}"] for ps in d["per_seed"]], dtype=np.float64)
    sds = np.array([ps[f"post_sd_{channel}"] for ps in d["per_seed"]], dtype=np.float64)
    bias = float(np.mean(maps - h_true))
    med_sd = float(np.median(sds))
    return bias, med_sd, sds


def _nanmedian_or_nan(x: npt.NDArray[np.float64]) -> float:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


# ---------------------------------------------------------------------------
# Section 1 -- per-seed information width A -> sigma_A vs stored post_sd
# ---------------------------------------------------------------------------


def section1_information_width(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cid, d in cells.items():
        dose = d["config"].get("dose_scales")
        entry: dict[str, Any] = {
            "dose_scales": dose,
            "n_seeds": len(d["per_seed"]),
            "degenerate_fh0": _is_degenerate(d, cid),
            "edge_fi0": cid in EDGE_FI0_CELLS,
        }
        for ch in ("1d", "2d"):
            a_vals = _map_curvature_per_seed(d, ch)
            # sigma_A defined only where the local fit is concave (A > 0).
            valid = np.isfinite(a_vals) & (a_vals > 0)
            sigma_a = np.full_like(a_vals, np.nan)
            sigma_a[valid] = 1.0 / np.sqrt(a_vals[valid])
            _, med_sd, sds = _cell_bias_and_post_sd(d, ch)
            entry[f"A_median_{ch}"] = _nanmedian_or_nan(a_vals)
            entry[f"sigma_A_median_{ch}"] = _nanmedian_or_nan(sigma_a)
            entry[f"frac_nonconcave_or_invalid_{ch}"] = float(np.mean(~valid))
            entry[f"post_sd_median_{ch}"] = med_sd
            ratio = (
                entry[f"sigma_A_median_{ch}"] / med_sd
                if med_sd > 0 and np.isfinite(entry[f"sigma_A_median_{ch}"])
                else float("nan")
            )
            entry[f"ratio_sigmaA_over_post_sd_{ch}"] = ratio
        out[cid] = entry
    return out


# ---------------------------------------------------------------------------
# Section 2 -- per-cell sandwich width sigma_SW and predicted overconfidence
# ---------------------------------------------------------------------------


def section2_sandwich_width(
    cells: dict[str, dict[str, Any]], section1: dict[str, Any]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cid, d in cells.items():
        dose = d["config"].get("dose_scales")
        entry: dict[str, Any] = {
            "dose_scales": dose,
            "degenerate_fh0": _is_degenerate(d, cid),
            "edge_fi0": cid in EDGE_FI0_CELLS,
        }
        s1, s2 = _slope_at_truth_per_seed(d)  # central-difference T, verified geometry
        for ch, t_cd in (("1d", s1), ("2d", s2)):
            t_mean_cd, t_se_cd = _mean_se(t_cd)
            score_qfit, curv_qfit = _truth_score_and_curvature_per_seed(d, ch)
            valid = np.isfinite(score_qfit) & np.isfinite(curv_qfit)
            B = float(np.var(score_qfit[valid], ddof=1)) if valid.sum() > 1 else float("nan")
            Abar = _nanmedian_or_nan(curv_qfit) if valid.sum() > 0 else float("nan")
            Abar_mean = float(np.mean(curv_qfit[valid])) if valid.sum() > 0 else float("nan")
            sigma_sw = (
                float(np.sqrt(B) / Abar_mean)
                if np.isfinite(B) and np.isfinite(Abar_mean) and Abar_mean > 0
                else float("nan")
            )
            sigma_a_med = section1[cid][f"sigma_A_median_{ch}"]
            overconf_pred = (
                sigma_sw / sigma_a_med
                if np.isfinite(sigma_sw) and np.isfinite(sigma_a_med) and sigma_a_med > 0
                else float("nan")
            )
            entry[f"T_central_diff_mean_{ch}"] = t_mean_cd
            entry[f"T_central_diff_se_{ch}"] = t_se_cd
            entry[f"T_qfit_mean_{ch}"] = (
                float(np.mean(score_qfit[valid])) if valid.sum() > 0 else float("nan")
            )
            entry[f"B_score_var_{ch}"] = B
            entry[f"Abar_median_{ch}"] = Abar
            entry[f"Abar_mean_{ch}"] = Abar_mean
            entry[f"sigma_SW_{ch}"] = sigma_sw
            entry[f"sigma_A_median_{ch}"] = sigma_a_med
            entry[f"overconfidence_predicted_sigmaSW_over_sigmaA_{ch}"] = overconf_pred
        out[cid] = entry
    return out


# ---------------------------------------------------------------------------
# Section 3 -- H-SB's parameter-free quantitative predictions
# ---------------------------------------------------------------------------


def section3_predictions(
    cells: dict[str, dict[str, Any]], section2: dict[str, Any]
) -> dict[str, Any]:
    rows = []
    for cid, d in cells.items():
        dose = d["config"].get("dose_scales")
        row: dict[str, Any] = {
            "cell": cid,
            "dose_scales": dose,
            "degenerate_fh0": _is_degenerate(d, cid),
            "edge_fi0": cid in EDGE_FI0_CELLS,
        }
        for ch in ("1d", "2d"):
            bias, med_sd, _ = _cell_bias_and_post_sd(d, ch)
            T = section2[cid][f"T_central_diff_mean_{ch}"]
            Abar = section2[cid][f"Abar_mean_{ch}"]
            sigma_sw = section2[cid][f"sigma_SW_{ch}"]
            sigma_a = section2[cid][f"sigma_A_median_{ch}"]

            # (a) bias ~= T/Abar (pseudo-true displacement), vs the M6R route
            # bias ~= 0.749 * T * post_sd^2 (i.e. using 1/post_sd^2 as an
            # implied curvature).
            pred_bias_Abar = T / Abar if np.isfinite(Abar) and Abar != 0 else float("nan")
            pred_bias_sigmapost2 = T * med_sd**2
            ratio_a = (
                bias / pred_bias_Abar
                if pred_bias_Abar not in (0, float("nan")) and np.isfinite(pred_bias_Abar)
                else float("nan")
            )
            ratio_sigmapost = (
                bias / pred_bias_sigmapost2 if pred_bias_sigmapost2 != 0 else float("nan")
            )
            # does Abar reproduce the implied curvature 1/post_sd^2?
            implied_curv_from_post_sd = 1.0 / med_sd**2 if med_sd > 0 else float("nan")
            abar_over_implied = (
                Abar / implied_curv_from_post_sd
                if np.isfinite(Abar) and np.isfinite(implied_curv_from_post_sd)
                else float("nan")
            )

            # (b) overconfidence: predicted sigma_SW/sigma_A vs observed
            # |bias|/post_sd.
            observed_overconf = abs(bias) / med_sd if med_sd > 0 else float("nan")
            predicted_overconf = sigma_sw / sigma_a if sigma_a > 0 else float("nan")
            ratio_overconf = (
                predicted_overconf / observed_overconf
                if observed_overconf not in (0, float("nan"))
                and np.isfinite(observed_overconf)
                and np.isfinite(predicted_overconf)
                else float("nan")
            )

            counterexample = bool(
                (np.isfinite(ratio_a) and (ratio_a > 3.0 or ratio_a < 1.0 / 3.0))
                or (
                    np.isfinite(ratio_overconf)
                    and (ratio_overconf > 3.0 or ratio_overconf < 1.0 / 3.0)
                )
            )

            row[f"bias_{ch}"] = bias
            row[f"post_sd_median_{ch}"] = med_sd
            row[f"T_{ch}"] = T
            row[f"Abar_{ch}"] = Abar
            row[f"predicted_bias_T_over_Abar_{ch}"] = pred_bias_Abar
            row[f"ratio_measured_over_predicted_Abar_{ch}"] = ratio_a
            row[f"predicted_bias_T_sigmapost2_{ch}"] = pred_bias_sigmapost2
            row[f"ratio_measured_over_predicted_sigmapost2_{ch}"] = ratio_sigmapost
            row[f"Abar_over_implied_1_over_post_sd2_{ch}"] = abar_over_implied
            row[f"sigma_SW_{ch}"] = sigma_sw
            row[f"sigma_A_{ch}"] = sigma_a
            row[f"overconfidence_predicted_{ch}"] = predicted_overconf
            row[f"overconfidence_observed_bias_over_post_sd_{ch}"] = observed_overconf
            row[f"ratio_predicted_over_observed_overconf_{ch}"] = ratio_overconf
            row[f"counterexample_gt3x_{ch}"] = counterexample
        rows.append(row)
    rows.sort(key=lambda r: str(r["cell"]))

    def _clean(cells_set: set[str], edge_set: set[str]) -> list[dict[str, Any]]:
        return [r for r in rows if r["cell"] not in cells_set and r["cell"] not in edge_set]

    headline_rows = [r for r in rows if not r["degenerate_fh0"] and not r["edge_fi0"]]

    def _ratio_stats(rows_: list[dict[str, Any]], key: str) -> dict[str, float]:
        vals = np.array([r[key] for r in rows_ if np.isfinite(r[key])], dtype=np.float64)
        if vals.size == 0:
            return {"n": 0, "mean": float("nan"), "sd": float("nan")}
        return {
            "n": int(vals.size),
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan"),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    return {
        "all_rows": rows,
        "headline_rows_excl_degenerate_and_edge": headline_rows,
        "ratio_stats_Abar_1d": _ratio_stats(headline_rows, "ratio_measured_over_predicted_Abar_1d"),
        "ratio_stats_Abar_2d": _ratio_stats(headline_rows, "ratio_measured_over_predicted_Abar_2d"),
        "ratio_stats_sigmapost2_1d": _ratio_stats(
            headline_rows, "ratio_measured_over_predicted_sigmapost2_1d"
        ),
        "ratio_stats_sigmapost2_2d": _ratio_stats(
            headline_rows, "ratio_measured_over_predicted_sigmapost2_2d"
        ),
        "abar_over_implied_stats_1d": _ratio_stats(
            headline_rows, "Abar_over_implied_1_over_post_sd2_1d"
        ),
        "abar_over_implied_stats_2d": _ratio_stats(
            headline_rows, "Abar_over_implied_1_over_post_sd2_2d"
        ),
        "overconf_ratio_stats_1d": _ratio_stats(
            headline_rows, "ratio_predicted_over_observed_overconf_1d"
        ),
        "overconf_ratio_stats_2d": _ratio_stats(
            headline_rows, "ratio_predicted_over_observed_overconf_2d"
        ),
        "counterexamples": [
            {
                "cell": r["cell"],
                "1d": r["counterexample_gt3x_1d"],
                "2d": r["counterexample_gt3x_2d"],
            }
            for r in rows
            if r["counterexample_gt3x_1d"] or r["counterexample_gt3x_2d"]
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ms_cells = _load_mechanism_study_files()
    vt_cells = _load_venue_transfer_files()
    all_cells = {**ms_cells, **vt_cells}

    section1 = section1_information_width(all_cells)
    section2 = section2_sandwich_width(all_cells, section1)
    section3 = section3_predictions(all_cells, section2)

    n_ms = len(ms_cells)
    n_vt = len(vt_cells)
    n_vt_seeds = sum(len(d["per_seed"]) for d in vt_cells.values())
    n_ms_seeds = sum(len(d["per_seed"]) for d in ms_cells.values())

    output = {
        "provenance": {
            "mechanism_study_cells_loaded": sorted(ms_cells.keys()),
            "mechanism_study_n_cells": n_ms,
            "mechanism_study_n_seeds": n_ms_seeds,
            "venue_transfer_present_on_this_machine": n_vt > 0,
            "venue_transfer_cells_loaded": sorted(vt_cells.keys()),
            "venue_transfer_n_cells": n_vt,
            "venue_transfer_n_seeds": n_vt_seeds,
        },
        "section1_information_width": section1,
        "section2_sandwich_width": section2,
        "section3_predictions": section3,
    }

    out_path = RESULTS_DIR / "L0_SB_output.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2, sort_keys=False)
    print(f"wrote {out_path}")
    print(
        f"mechanism-study cells: {n_ms} ({n_ms_seeds} seeds); "
        f"venue-transfer cells: {n_vt} ({n_vt_seeds} seeds, "
        f"{'present' if n_vt else 'NOT FOUND on this machine'})"
    )


if __name__ == "__main__":
    main()
