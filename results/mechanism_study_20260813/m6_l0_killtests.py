"""M6-L0 registered kill tests (PREREGISTRATION_M2PRIME_ABLATION.md §2, "M6/M7 L0
obligations").

Recomputes everything from the raw per-seed ``ln_post_1d``/``ln_post_2d`` vectors
and per-seed ``map_*``/``post_sd_*`` fields in the 20 committed result JSONs under
``results/mechanism_study_20260813/``. Never reads the per-file ``aggregate`` block.

Three registered, two-sided kill tests on the M6 composite (sigma_z-blind aggregate
log-posterior tilt x dose-controlled curvature):

(i)   tilt dose-invariance: per-cell aggregate d(ln post)/dh at truth (central
      difference on the h_true=0.730 grid neighbours, matching
      ``darksiren_emri.validation.venue_transfer._slope_at_truth`` verbatim),
      averaged over seeds; KILL if not dose-invariant within +/-10% across all
      f_h > 0 cells.
(ii)  bias/sigma^2_post constancy: per interior cell (f_h > 0 and f_i > 0, the 9
      S-cells), mean MAP bias / median(post_sd)^2; KILL if not constant within a
      factor 2 across the 9 cells.
(iii) alpha-share: alpha's registered tilt contribution is +1.036*N/h with
      N=982, h=0.730 (=+1393.6 nats/h); its share of the measured total tilt
      from (i); KILL if outside 52.7% +/- 5pp.

Falsification-first operationalization note (registered ambiguity, resolved in
the direction most favorable to KILLING M6): "across all f_h > 0 cells" in (i)
is read as the full 12-cell set {S10..S13, S20..S23, S30..S33} (dose_scales[0] >
0, no restriction on dose_scales[1]) -- the literal, most inclusive reading of
the registered wording, and the one most likely to expose non-invariance. A
narrower reading (varying f_h only at fixed f_i=1.0, i.e. the 3-cell column
{S13, S23, S33}) is also reported for comparison against the commission's
reference measurement, which used that narrower column.
"""

import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

RESULTS_DIR = Path(__file__).parent
ALPHA_N = 982
ALPHA_H = 0.730
ALPHA_SLOPE_COEFF = 1.036
ALPHA_TILT_NATS_PER_H = ALPHA_SLOPE_COEFF * ALPHA_N / ALPHA_H  # = +1393.6 nats/h

TILT_TOL_FRAC = 0.10  # (i) +/-10%
CONST_TOL_FACTOR = 2.0  # (ii) factor of 2
ALPHA_SHARE_TARGET_PCT = 52.7  # (iii)
ALPHA_SHARE_TOL_PP = 5.0


def _load_s_cells() -> dict[str, dict[str, Any]]:
    """Load the 16 dose-grid S-cell result JSONs, keyed by cell id."""
    files = sorted(glob.glob(str(RESULTS_DIR / "S*_h0p730_results_seeds*.json")))
    cells: dict[str, dict[str, Any]] = {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        cells[d["config"]["cell"]] = d
    return cells


def _slope_at_truth_per_seed(
    d: dict[str, Any],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-seed central-difference d(ln_post)/dh at truth, both channels.

    Matches ``venue_transfer._slope_at_truth`` verbatim: grid neighbours of
    ``h_true``, central difference. Computed directly from the raw
    ``ln_post_1d``/``ln_post_2d`` per-seed vectors, never from the file's
    ``aggregate`` block or the precomputed ``sum_dlog_gfrac_dh`` field (a
    different quantity -- the Sigma ln(L2/L1) gfrac slope, not the channel's
    own log-posterior tilt).
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


def _cell_mean_slope(d: dict[str, Any]) -> tuple[float, float]:
    s1, s2 = _slope_at_truth_per_seed(d)
    return float(np.mean(s1)), float(np.mean(s2))


def test_i_tilt_dose_invariance(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """(i) tilt dose-invariance across all f_h > 0 cells, +/-10%, both channels."""
    group = sorted(c for c in cells if cells[c]["config"]["dose_scales"][0] > 0)
    per_cell_1d: dict[str, float] = {}
    per_cell_2d: dict[str, float] = {}
    for c in group:
        m1, m2 = _cell_mean_slope(cells[c])
        per_cell_1d[c] = m1
        per_cell_2d[c] = m2

    def _invariance(per_cell: dict[str, float]) -> dict[str, Any]:
        vals = np.array(list(per_cell.values()))
        mean = float(np.mean(vals))
        dev = {c: (v - mean) / mean for c, v in per_cell.items()}
        max_abs_dev = max(abs(x) for x in dev.values())
        return {
            "per_cell_mean_slope": per_cell,
            "grand_mean": mean,
            "fractional_deviation": dev,
            "max_abs_fractional_deviation": max_abs_dev,
            "verdict": "KILL" if max_abs_dev > TILT_TOL_FRAC else "SURVIVE",
        }

    result_1d = _invariance(per_cell_1d)
    result_2d = _invariance(per_cell_2d)

    # Comparison-only alternative operationalization: fixed f_i=1.0 column,
    # varying f_h -- matches the commission's reference measurement window.
    fi1_col = sorted(
        c
        for c in cells
        if cells[c]["config"]["dose_scales"][0] > 0 and cells[c]["config"]["dose_scales"][1] == 1.0
    )
    fi1_1d = {c: per_cell_1d[c] for c in fi1_col}
    fi1_2d = {c: per_cell_2d[c] for c in fi1_col}

    interior9 = sorted(
        c
        for c in cells
        if cells[c]["config"]["dose_scales"][0] > 0 and cells[c]["config"]["dose_scales"][1] > 0
    )
    interior_1d = {c: per_cell_1d[c] for c in interior9}
    interior_2d = {c: per_cell_2d[c] for c in interior9}

    return {
        "registered_group": group,
        "channel_1d": result_1d,
        "channel_2d": result_2d,
        "verdict": "KILL"
        if (result_1d["verdict"] == "KILL" or result_2d["verdict"] == "KILL")
        else "SURVIVE",
        "comparison_only": {
            "note": (
                "Not the registered set. Fixed f_i=1.0 column {S13,S23,S33} "
                "reproduces the commission's 2625-2720 nats/h reference band; "
                "the interior-9 set is shown for reference against test (ii)/(iii)."
            ),
            "fi1_column_1d": fi1_1d,
            "fi1_column_2d": fi1_2d,
            "fi1_column_1d_range": [min(fi1_1d.values()), max(fi1_1d.values())],
            "fi1_column_2d_range": [min(fi1_2d.values()), max(fi1_2d.values())],
            "interior9_1d": interior_1d,
            "interior9_2d": interior_2d,
            "interior9_1d_range": [min(interior_1d.values()), max(interior_1d.values())],
            "interior9_2d_range": [min(interior_2d.values()), max(interior_2d.values())],
        },
    }


def test_ii_bias_over_sigma2_constancy(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """(ii) mean MAP bias / median(post_sd)^2, constant within factor 2, 9 interior cells."""
    interior9 = sorted(
        c
        for c in cells
        if cells[c]["config"]["dose_scales"][0] > 0 and cells[c]["config"]["dose_scales"][1] > 0
    )

    def _ratios(channel: str) -> dict[str, Any]:
        map_key = f"map_{channel}"
        sd_key = f"post_sd_{channel}"
        per_cell: dict[str, dict[str, float]] = {}
        for c in interior9:
            d = cells[c]
            h_true = float(d["config"]["h_true"])
            maps = np.array([ps[map_key] for ps in d["per_seed"]], dtype=np.float64)
            sds = np.array([ps[sd_key] for ps in d["per_seed"]], dtype=np.float64)
            bias = float(np.mean(maps - h_true))
            med_sd = float(np.median(sds))
            ratio = bias / med_sd**2
            per_cell[c] = {"mean_bias": bias, "median_post_sd": med_sd, "ratio": ratio}
        ratios = np.array([v["ratio"] for v in per_cell.values()])
        all_positive = bool(np.all(ratios > 0))
        max_over_min = float(np.max(ratios) / np.min(ratios)) if all_positive else float("inf")
        return {
            "per_cell": per_cell,
            "all_same_sign": all_positive,
            "max_over_min_ratio": max_over_min,
            "verdict": "KILL" if (max_over_min > CONST_TOL_FACTOR) else "SURVIVE",
        }

    result_1d = _ratios("1d")
    result_2d = _ratios("2d")
    return {
        "interior9_cells": interior9,
        "channel_1d": result_1d,
        "channel_2d": result_2d,
        "verdict": "KILL"
        if (result_1d["verdict"] == "KILL" or result_2d["verdict"] == "KILL")
        else "SURVIVE",
    }


def test_iii_alpha_share(test_i_result: dict[str, Any]) -> dict[str, Any]:
    """(iii) alpha's share of the measured total tilt from (i), 52.7% +/- 5pp."""
    total_1d = test_i_result["channel_1d"]["grand_mean"]
    total_2d = test_i_result["channel_2d"]["grand_mean"]
    share_1d = ALPHA_TILT_NATS_PER_H / total_1d * 100.0
    share_2d = ALPHA_TILT_NATS_PER_H / total_2d * 100.0
    lo, hi = (
        ALPHA_SHARE_TARGET_PCT - ALPHA_SHARE_TOL_PP,
        ALPHA_SHARE_TARGET_PCT + ALPHA_SHARE_TOL_PP,
    )

    def _verdict(share: float) -> str:
        return "SURVIVE" if lo <= share <= hi else "KILL"

    # Comparison-only: share using the fi1-column measured total (matches the
    # commission's narrower operationalization).
    fi1_1d = test_i_result["comparison_only"]["fi1_column_1d"]
    fi1_2d = test_i_result["comparison_only"]["fi1_column_2d"]
    fi1_mean_1d = float(np.mean(list(fi1_1d.values())))
    fi1_mean_2d = float(np.mean(list(fi1_2d.values())))
    share_fi1_1d = ALPHA_TILT_NATS_PER_H / fi1_mean_1d * 100.0
    share_fi1_2d = ALPHA_TILT_NATS_PER_H / fi1_mean_2d * 100.0

    return {
        "alpha_tilt_nats_per_h": ALPHA_TILT_NATS_PER_H,
        "target_pct": ALPHA_SHARE_TARGET_PCT,
        "tolerance_pp": ALPHA_SHARE_TOL_PP,
        "window_pct": [lo, hi],
        "measured_total_tilt_1d": total_1d,
        "measured_total_tilt_2d": total_2d,
        "alpha_share_pct_1d": share_1d,
        "alpha_share_pct_2d": share_2d,
        "verdict_1d": _verdict(share_1d),
        "verdict_2d": _verdict(share_2d),
        "verdict": "KILL"
        if (_verdict(share_1d) == "KILL" or _verdict(share_2d) == "KILL")
        else "SURVIVE",
        "comparison_only": {
            "note": "alpha-share using the fi1=1.0-column total tilt from test (i)'s comparison block.",
            "fi1_column_mean_total_1d": fi1_mean_1d,
            "fi1_column_mean_total_2d": fi1_mean_2d,
            "alpha_share_pct_fi1_1d": share_fi1_1d,
            "alpha_share_pct_fi1_2d": share_fi1_2d,
        },
    }


def main() -> None:
    cells = _load_s_cells()
    assert len(cells) == 16, f"expected 16 S-cells (4x4 dose grid), found {len(cells)}"

    result_i = test_i_tilt_dose_invariance(cells)
    result_ii = test_ii_bias_over_sigma2_constancy(cells)
    result_iii = test_iii_alpha_share(result_i)

    commission_reference = {
        "source": "results/commission_research_20260814/REPORT.md",
        "gradient_nats_per_h_band": [2625, 2720],
        "gradient_claimed_dose_invariant": True,
        "alpha_share_predicted_pct": 52.7,
        "alpha_share_measured_pct": 53.3,
    }
    prereg_section2_prediction = {
        "source": "results/mechanism_study_20260813/PREREGISTRATION_M2PRIME_ABLATION.md §2",
        "total_up_tilt_nats_per_h": 2738.8,
        "alpha_share_pct_predicted": 52.7,
        "missing_J_share_pct_predicted": 49.1,
    }

    output = {
        "preregistration": "results/mechanism_study_20260813/PREREGISTRATION_M2PRIME_ABLATION.md §2",
        "data_source": "the 20 committed *_h0p730_results_seeds*.json files (16 S-cell dose grid used for the kill tests)",
        "method_note": (
            "All quantities recomputed from raw per-seed ln_post_1d/ln_post_2d "
            "vectors and per-seed map_*/post_sd_* fields; the per-file "
            "'aggregate' block was never read."
        ),
        "test_i_tilt_dose_invariance": result_i,
        "test_ii_bias_sigma2_constancy": result_ii,
        "test_iii_alpha_share": result_iii,
        "commission_reference": commission_reference,
        "prereg_section2_prediction": prereg_section2_prediction,
        "overall_verdicts": {
            "i_tilt_dose_invariance": result_i["verdict"],
            "ii_bias_sigma2_constancy": result_ii["verdict"],
            "iii_alpha_share": result_iii["verdict"],
        },
    }

    out_path = RESULTS_DIR / "M6_L0_killtests_output.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2, sort_keys=False)

    print(f"wrote {out_path}")
    print(json.dumps(output["overall_verdicts"], indent=2))


if __name__ == "__main__":
    main()
