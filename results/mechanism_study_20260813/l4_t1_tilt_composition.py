"""L4-T1 post-repair tilt composition (author-approved, ledger row #108).

Follow-on to the A-JREN joint-repair Stage-3 readout (``STAGE3_READOUT.md``): the joint
repair (missing Jacobian restored + kernel-mass renormalization) leaves +0.0178 of the
original +0.0373 1D bias unaccounted for, attributed provisionally to "the alpha-tilt
(correct physics, still uncancelled) plus T_res" (``PROPOSAL_STAGE4_20260815.md``, item
L4-T1). This script measures the aggregate d(ln_post)/dh at truth for the new A-JREN arm
and decomposes that remaining tilt against three independent references, all built from
data already committed under ``results/mechanism_study_20260813/``:

1. The analytic alpha-tilt +1393.6 nats/h (1.036*N/h, N=982, h=0.730, prereg Sec.2). If
   T(AJREN) equals this within errors, T_res(full dose) ~= 0 post-repair and the entire
   remaining bias is the uncancelled alpha term.
2. T(AJREN) - T(AM2P) (J-only arm, both full dose): isolates the REN tilt actually present
   on the instrument, for comparison to the L0-REN-B toy's dose-1.0 read (+99.52 nats/h,
   seed-scatter std 52.13, SE 18.43 over n=8 seeds -- production-population transfer
   check; the toy's z_median caveat is on record in ``L0_REN_B_TOY_RESULTS_20260815.md``).
3. The L0-SB displacement law bias ~= T/Abar (``l0_sb_diagnostic.py`` Sections 2-3; the
   confirmed headline ratio measured-over-predicted is 1.147 +/- 0.132 (1D) / 1.164 +/-
   0.134 (2D) on the 16 S-cell + prior-arm interior set) applied out-of-sample to AJREN,
   using AJREN's OWN local quadratic-fit curvature Abar at truth -- a parameter-free
   closure test, not a refit.
4. The same three numbers for the 2D channel, given the Stage-3 2D-only sub-additivity
   finding (+0.0027, ~3.8 sigma) -- is there a corresponding 2D-only tilt excess in T
   itself, or does 2D track 1D within its own SE budget?

Recomputes everything from the raw per-seed ``ln_post_1d``/``ln_post_2d`` grid vectors and
per-seed ``map_*``/``post_sd_*`` fields in the committed AJREN and AM2P result JSONs.
Never reads a file's ``aggregate`` block (that quantity is the Sigma ln(L2/L1) gfrac slope,
a different object -- see ``m6r_l0_decomposition.py`` module docstring for the same
caveat).

Method precedent (verbatim geometry, three times reused across this thread):
``results/mechanism_study_20260813/m6r_l0_decomposition.py`` /
``darksiren_emri.validation.venue_transfer._slope_at_truth`` (central difference at the
grid neighbours of h_true) for T; ``l0_sb_diagnostic.py``'s local quadratic least-squares
fit (``ln_post(h) ~= c0 + c1*(h-h0) + c2*(h-h0)^2``, ``Abar = -2*c2`` at h_true) for the
curvature used in the displacement-law closure.

Output: ``L4_T1_output.json`` next to this file. Report:
``L4_T1_TILT_COMPOSITION_20260815.md`` (PRESENTED, NOT ADJUDICATED).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

RESULTS_DIR = Path(__file__).parent

# ---- registered/derived constants (PREREGISTRATION_M2PRIME_ABLATION.md Sec.2) ----
ALPHA_N = 982
ALPHA_H = 0.730
ALPHA_SLOPE_COEFF = 1.036
ALPHA_TILT_NATS_PER_H = ALPHA_SLOPE_COEFF * ALPHA_N / ALPHA_H  # = +1393.6 nats/h

# L0-REN-B toy, full dose (f_i = 1.0), production-scaled to N=982 (L0_REN_B_toy_output.json
# / L0_REN_B_TOY_RESULTS_20260815.md Sec.2).
TOY_REN_TILT_MEAN = 99.52
TOY_REN_TILT_SEED_STD = 52.13
TOY_REN_TILT_SE = 18.43
TOY_REN_TILT_N_SEEDS = 8

# L0-SB displacement-law headline ratio (bias / (T/Abar)), 16 S-cells + prior arms,
# headline set excluding degenerate f_h=0 and edge f_i=0 cells (L0_SB_output.json
# section3_predictions ratio_stats_Abar_1d/2d; L0_SB_DIAGNOSTIC_20260815.md line 103).
L0_SB_RATIO_1D_MEAN = 1.147
L0_SB_RATIO_1D_SE = 0.132
L0_SB_RATIO_2D_MEAN = 1.164
L0_SB_RATIO_2D_SE = 0.134

# Stage-3 2D-only sub-additivity finding (STAGE3_READOUT.md Sec.2), reported for context only.
STAGE3_2D_SUBADDITIVE_NATS = 0.0027


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        result: dict[str, Any] = json.load(fh)
    return result


def _slope_at_truth_per_seed(
    d: dict[str, Any],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-seed central-difference d(ln_post)/dh at truth, both channels.

    Verbatim geometry match to ``venue_transfer._slope_at_truth`` /
    ``m6r_l0_decomposition._slope_at_truth_per_seed`` / ``l0_sb_diagnostic``'s
    ``T_central_diff``: grid neighbours of ``h_true``, central difference, applied
    directly to the raw ``ln_post_1d``/``ln_post_2d`` per-seed vectors.
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

    Verbatim copy of ``l0_sb_diagnostic._local_quadratic_fit``: fits
    ``ln_post(h) ~= c0 + c1*(h-h0) + c2*(h-h0)^2`` over up to ``2*half_window + 1``
    grid neighbours (fewer near an edge or where ``ln_post`` is non-finite). Returns
    ``(score, curvature)`` where ``score = c1`` and ``curvature = -2*c2``. Returns
    ``(nan, nan)`` if fewer than 3 finite points are available.
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


def _truth_score_and_curvature_per_seed(
    d: dict[str, Any], channel: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-seed (score, curvature) at h_true from the local quadratic fit."""
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


def _mean_se(x: npt.NDArray[np.float64]) -> tuple[float, float]:
    n = x.size
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return mean, se


def _cell_bias(d: dict[str, Any], channel: str) -> tuple[float, float]:
    """Mean MAP bias and its SE for one channel."""
    h_true = float(d["config"]["h_true"])
    maps = np.array([ps[f"map_{channel}"] for ps in d["per_seed"]], dtype=np.float64)
    delta = maps - h_true
    m, se = _mean_se(delta)
    return m, se


# ---------------------------------------------------------------------------
# Section 1 -- T(AJREN), both channels, seed-averaged, SE; residual vs alpha
# ---------------------------------------------------------------------------


def section1_tilt_vs_alpha(ajren: dict[str, Any]) -> dict[str, Any]:
    s1, s2 = _slope_at_truth_per_seed(ajren)
    t1, se1 = _mean_se(s1)
    t2, se2 = _mean_se(s2)

    resid1 = t1 - ALPHA_TILT_NATS_PER_H
    resid2 = t2 - ALPHA_TILT_NATS_PER_H

    def _sigma(resid: float, se: float) -> float:
        return abs(resid) / se if se > 0 else float("nan")

    return {
        "n_seeds": int(s1.size),
        "T1_mean": t1,
        "T1_se": se1,
        "T2_mean": t2,
        "T2_se": se2,
        "alpha_tilt_nats_per_h": ALPHA_TILT_NATS_PER_H,
        "residual_1d_T_minus_alpha": resid1,
        "residual_1d_se": se1,
        "residual_1d_sigma": _sigma(resid1, se1),
        "residual_2d_T_minus_alpha": resid2,
        "residual_2d_se": se2,
        "residual_2d_sigma": _sigma(resid2, se2),
    }


# ---------------------------------------------------------------------------
# Section 2 -- instrument REN tilt: T(AJREN) - T(AM2P), vs the L0-REN-B toy
# ---------------------------------------------------------------------------


def section2_ren_tilt_vs_toy(ajren: dict[str, Any], am2p: dict[str, Any]) -> dict[str, Any]:
    s1_j, s2_j = _slope_at_truth_per_seed(ajren)
    s1_m, s2_m = _slope_at_truth_per_seed(am2p)

    t1_j, se1_j = _mean_se(s1_j)
    t2_j, se2_j = _mean_se(s2_j)
    t1_m, se1_m = _mean_se(s1_m)
    t2_m, se2_m = _mean_se(s2_m)

    ren1 = t1_j - t1_m
    ren2 = t2_j - t2_m
    se_ren1 = float(np.hypot(se1_j, se1_m))
    se_ren2 = float(np.hypot(se2_j, se2_m))

    def _diff_sigma(diff: float, se: float, ref: float, ref_se: float) -> float:
        denom = float(np.hypot(se, ref_se))
        return abs(diff - ref) / denom if denom > 0 else float("nan")

    return {
        "AJREN": {
            "T1_mean": t1_j,
            "T1_se": se1_j,
            "T2_mean": t2_j,
            "T2_se": se2_j,
            "n": int(s1_j.size),
        },
        "AM2P": {
            "T1_mean": t1_m,
            "T1_se": se1_m,
            "T2_mean": t2_m,
            "T2_se": se2_m,
            "n": int(s1_m.size),
        },
        "instrument_ren_tilt_1d_nats_per_h": ren1,
        "instrument_ren_tilt_1d_se": se_ren1,
        "instrument_ren_tilt_2d_nats_per_h": ren2,
        "instrument_ren_tilt_2d_se": se_ren2,
        "toy_ren_tilt_full_dose_mean": TOY_REN_TILT_MEAN,
        "toy_ren_tilt_full_dose_seed_std": TOY_REN_TILT_SEED_STD,
        "toy_ren_tilt_full_dose_se": TOY_REN_TILT_SE,
        "toy_ren_tilt_n_seeds": TOY_REN_TILT_N_SEEDS,
        "toy_caveat": (
            "L0-REN-B is a 500-event toy at the toy's own z_median, not the production "
            "n(z); production-population transfer check only, per "
            "L0_REN_B_TOY_RESULTS_20260815.md on-record caveat."
        ),
        "instrument_minus_toy_1d_using_toy_se": _diff_sigma(
            ren1, se_ren1, TOY_REN_TILT_MEAN, TOY_REN_TILT_SE
        ),
        "instrument_minus_toy_2d_using_toy_se": _diff_sigma(
            ren2, se_ren2, TOY_REN_TILT_MEAN, TOY_REN_TILT_SE
        ),
        "instrument_minus_toy_1d_using_toy_seed_std": _diff_sigma(
            ren1, se_ren1, TOY_REN_TILT_MEAN, TOY_REN_TILT_SEED_STD
        ),
        "instrument_minus_toy_2d_using_toy_seed_std": _diff_sigma(
            ren2, se_ren2, TOY_REN_TILT_MEAN, TOY_REN_TILT_SEED_STD
        ),
    }


# ---------------------------------------------------------------------------
# Section 3 -- displacement-law closure on AJREN (out-of-sample, parameter-free)
# ---------------------------------------------------------------------------


def section3_displacement_law(ajren: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ch, label in (("1d", "1"), ("2d", "2")):
        bias, bias_se = _cell_bias(ajren, ch)
        score_qfit, curv_qfit = _truth_score_and_curvature_per_seed(ajren, ch)
        valid = np.isfinite(score_qfit) & np.isfinite(curv_qfit)
        abar_mean = float(np.mean(curv_qfit[valid])) if valid.sum() > 0 else float("nan")
        abar_median = float(np.median(curv_qfit[valid])) if valid.sum() > 0 else float("nan")
        t_qfit_mean = float(np.mean(score_qfit[valid])) if valid.sum() > 0 else float("nan")

        pred_bias = (
            t_qfit_mean / abar_mean
            if abar_mean not in (0.0,) and np.isfinite(abar_mean)
            else float("nan")
        )
        ratio = (
            bias / pred_bias if pred_bias not in (0.0,) and np.isfinite(pred_bias) else float("nan")
        )

        ref_mean = L0_SB_RATIO_1D_MEAN if label == "1" else L0_SB_RATIO_2D_MEAN
        ref_se = L0_SB_RATIO_1D_SE if label == "1" else L0_SB_RATIO_2D_SE

        out[f"channel_{ch}"] = {
            "bias_mean": bias,
            "bias_se": bias_se,
            "T_qfit_mean_at_truth": t_qfit_mean,
            "Abar_mean_at_truth": abar_mean,
            "Abar_median_at_truth": abar_median,
            "n_valid_seeds": int(valid.sum()),
            "predicted_bias_T_over_Abar": pred_bias,
            "ratio_measured_over_predicted": ratio,
            "reference_headline_ratio_mean": ref_mean,
            "reference_headline_ratio_se": ref_se,
            "ratio_minus_reference": ratio - ref_mean if np.isfinite(ratio) else float("nan"),
            "within_reference_se": (
                bool(abs(ratio - ref_mean) <= ref_se) if np.isfinite(ratio) else False
            ),
        }
    return out


def main() -> None:
    ajren = _load_json(RESULTS_DIR / "AJREN_h0p730_results_seeds0_25.json")
    am2p = _load_json(RESULTS_DIR / "AM2P_h0p730_results_seeds0_25.json")

    section1 = section1_tilt_vs_alpha(ajren)
    section2 = section2_ren_tilt_vs_toy(ajren, am2p)
    section3 = section3_displacement_law(ajren)

    output = {
        "note": "results/mechanism_study_20260813/L4_T1_TILT_COMPOSITION_20260815.md",
        "parent": "results/mechanism_study_20260813/STAGE3_READOUT.md",
        "prereg_source": "results/mechanism_study_20260813/PROPOSAL_STAGE4_20260815.md item L4-T1",
        "data_source": (
            "AJREN_h0p730_results_seeds0_25.json + AM2P_h0p730_results_seeds0_25.json, both "
            "committed under results/mechanism_study_20260813/; L0-REN-B toy and L0-SB "
            "reference numbers taken from L0_REN_B_TOY_RESULTS_20260815.md / "
            "L0_SB_DIAGNOSTIC_20260815.md (committed, not recomputed here)."
        ),
        "method_note": (
            "All quantities recomputed from raw per-seed ln_post_1d/ln_post_2d vectors and "
            "per-seed map_*/post_sd_* fields; the per-file 'aggregate' block was never read. "
            "T is the grid-neighbour central difference at h_true (verbatim "
            "venue_transfer._slope_at_truth / m6r_l0_decomposition geometry); the "
            "displacement-law closure additionally uses the local quadratic-fit score/curvature "
            "at h_true (verbatim l0_sb_diagnostic._local_quadratic_fit geometry, half_window=2)."
        ),
        "constants": {
            "ALPHA_N": ALPHA_N,
            "ALPHA_H": ALPHA_H,
            "ALPHA_SLOPE_COEFF": ALPHA_SLOPE_COEFF,
            "ALPHA_TILT_NATS_PER_H": ALPHA_TILT_NATS_PER_H,
            "TOY_REN_TILT_MEAN": TOY_REN_TILT_MEAN,
            "TOY_REN_TILT_SEED_STD": TOY_REN_TILT_SEED_STD,
            "TOY_REN_TILT_SE": TOY_REN_TILT_SE,
            "L0_SB_RATIO_1D_MEAN": L0_SB_RATIO_1D_MEAN,
            "L0_SB_RATIO_1D_SE": L0_SB_RATIO_1D_SE,
            "L0_SB_RATIO_2D_MEAN": L0_SB_RATIO_2D_MEAN,
            "L0_SB_RATIO_2D_SE": L0_SB_RATIO_2D_SE,
            "STAGE3_2D_SUBADDITIVE_NATS": STAGE3_2D_SUBADDITIVE_NATS,
        },
        "section1_tilt_vs_alpha": section1,
        "section2_ren_tilt_vs_toy": section2,
        "section3_displacement_law": section3,
    }

    out_path = RESULTS_DIR / "L4_T1_output.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2, sort_keys=False)
    print(f"wrote {out_path}")
    print(
        json.dumps(
            {
                "T1_AJREN": section1["T1_mean"],
                "residual_1d_vs_alpha": section1["residual_1d_T_minus_alpha"],
                "residual_1d_sigma": section1["residual_1d_sigma"],
                "instrument_ren_tilt_1d": section2["instrument_ren_tilt_1d_nats_per_h"],
                "displacement_ratio_1d": section3["channel_1d"]["ratio_measured_over_predicted"],
                "displacement_ratio_2d": section3["channel_2d"]["ratio_measured_over_predicted"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
