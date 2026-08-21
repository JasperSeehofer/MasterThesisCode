"""C-SG v3 pilot band-setter (registered PRE-pilot-data; prereg section 6 mandate).

Reads the 4 C-SG-F pilot JSONs (seeds 910101-910104, job 6415588) and derives
every band from sigma_hat via the REGISTERED FORMULAS below. Committed before
any pilot JSON was read. The numeric thresholds this script prints are the
bands of record for the remaining 42 seeds; the false-fail table it emits is
published with them (A15).

Registered inputs (computed pre-pilot, zero compute, from banked B-SEL data):

    S_REF   = -0.1932   B-SEL matched-channel per-event score at truth
                        (fleet mean over 12 seeds; per-seed sd 0.0914,
                        SEM 0.0264) -- the INTERNAL-DEFECT reference scale.
    B_REF   = -0.0846   B-SEL matched-channel mean_h bias (O3 verdict).

Registered statistics (matched channel ONLY, per the v3 design change):

    S_bar   = mean over F seeds of the per-seed mean per-event score at h_gen
              (score_at_h_gen["matched"] in each banked JSON).
    bias    = mean over F seeds of matched mean_h - 0.73.
    sigma_hat_score / sigma_hat_seed = per-seed sd of those two quantities
              over the 4 pilot seeds (3 dof).

Registered band FORMULAS (numbers filled by this script, then frozen):

    SE15        = sigma_hat_score / sqrt(15)
    SELF-CONSISTENT : |S_bar_15| <= 3*SE15           (nominal false-fail 0.27%,
                      Gaussian; 3-dof sd caveat published: the true sd may be
                      0.57x-3.73x the estimate at 95% confidence -- the
                      false-fail table states the band's rate at both edges)
    INTERNAL-DEFECT : S_bar_15 <= S_REF/2 = -0.0966  AND |S_bar_15| > 3*SE15
    MIXED           : anything else.
    The same tri-band applies to bias with B_REF/2 = -0.0423 and
    SE15_bias = sigma_hat_seed/sqrt(15). The SCORE band is primary (prereg
    pre-check O1); the bias band is confirmatory. Disagreement => MIXED.

    GATE S (regression, prereg section 6): fit mean_h(seed) = b + s*h_gen over
    all F + delta seeds when they exist; CONTROL-VALID if |s_hat - 1| <=
    3*SE(s_hat); CONTROL-INERT if the CI on s_hat contains 0 (STOP).
    Power note published now: with sigma_seed ~ sigma_hat_seed and arms
    {15@0.73, 8@0.68, 8@0.78}, SE(s_hat) = sigma_seed / sqrt(sum_i (h_i-h_bar)^2
    over seed-arm assignments) -- the script prints the realized value.

    BAND R (F vs E): DECLARED INDEPENDENT (v2 section 6: pairing is not
    achievable without RNG-stream surgery; the sigma draw desynchronizes the
    stream). Threshold: |mean_F - mean_E| <= 3*sqrt(2)*sigma_hat_seed/sqrt(15),
    false-fail 0.27% nominal at the pilot sigma_hat.

    N-adequacy check (A15 power): the design is adequate if half the reference
    effect is detectable at >= 5 sigma: |S_REF/2| / SE15 >= 5. If it fails,
    the script prints the required N and STOPS the fleet (orchestrator returns
    to the author -- N changes are not silently made).

    GATE V roll-up: any pilot seed failing gate_v (span < 5 nats on H_GRID_41
    or sigma_h > 0.5*sigma_prior) is reported; 2+ of 4 failing = STOP.

Usage:
    uv run python results/prod2d_closure_20260818/csg_pilot_bands.py --pilot-dir <dir>
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

S_REF = -0.1932
S_REF_SEM = 0.0264
B_REF = -0.0846
H_GEN_F = 0.73
N_F = 15
PILOT_SEEDS = (910101, 910102, 910103, 910104)

# 95% CI multipliers on a 3-dof sd estimate: sqrt(3/chi2_{0.975,3}), sqrt(3/chi2_{0.025,3}).
SD_CI_LO_MULT = 0.5665
SD_CI_HI_MULT = 3.7297


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", required=True, help="Directory holding csgf_seed9101??.json")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "csg_pilot_bands_output.json"),
    )
    args = ap.parse_args()
    pilot_dir = Path(args.pilot_dir)

    recs: list[dict[str, Any]] = []
    for seed in PILOT_SEEDS:
        p = pilot_dir / f"csgf_seed{seed}.json"
        if not p.is_file():
            raise SystemExit(f"missing pilot output {p} -- bands cannot be set")
        recs.append(json.loads(p.read_text()))

    scores = np.array(
        [r["score_at_h_gen"]["matched"]["mean_score"] for r in recs], dtype=np.float64
    )
    means = np.array([r["channel_scores"]["matched"]["mean_h"] for r in recs], dtype=np.float64)
    sigma_hs = np.array([r["channel_scores"]["matched"]["sigma_h"] for r in recs], dtype=np.float64)
    gate_v_fail = [
        {"seed": int(r["seed"]), "gate_v": r["gate_v"]}
        for r in recs
        if not bool(r["gate_v"].get("pass", False))
    ]

    sigma_hat_score = float(scores.std(ddof=1))
    sigma_hat_seed = float(means.std(ddof=1))
    se15_score = sigma_hat_score / np.sqrt(N_F)
    se15_bias = sigma_hat_seed / np.sqrt(N_F)

    half_ref_sig = abs(S_REF / 2.0) / se15_score if se15_score > 0 else np.inf
    n_required = int(np.ceil((5.0 * sigma_hat_score / abs(S_REF / 2.0)) ** 2))
    n_adequate = bool(half_ref_sig >= 5.0)

    # GATE S realized SE(s_hat) at the registered seed allocation.
    h_alloc = np.array([0.73] * 15 + [0.68] * 8 + [0.78] * 8)
    sxx = float(((h_alloc - h_alloc.mean()) ** 2).sum())
    se_slope = sigma_hat_seed / np.sqrt(sxx)

    out = {
        "registered_in": "csg_pilot_bands.py (committed pre-pilot-data) + prereg section 6",
        "pilot_seeds": list(PILOT_SEEDS),
        "per_seed": [
            {
                "seed": int(r["seed"]),
                "matched_mean_h": float(m),
                "matched_sigma_h": float(sh),
                "matched_mean_score_at_h_gen": float(s),
            }
            for r, m, sh, s in zip(recs, means, sigma_hs, scores, strict=True)
        ],
        "sigma_hat_score": sigma_hat_score,
        "sigma_hat_seed": sigma_hat_seed,
        "median_sigma_h": float(np.median(sigma_hs)),
        "sd_3dof_ci_multipliers": [SD_CI_LO_MULT, SD_CI_HI_MULT],
        "bands_of_record": {
            "score_self_consistent_abs_max": 3.0 * se15_score,
            "score_internal_defect_max": S_REF / 2.0,
            "bias_self_consistent_abs_max": 3.0 * se15_bias,
            "bias_internal_defect_max": B_REF / 2.0,
            "band_r_abs_max": 3.0 * np.sqrt(2.0) * sigma_hat_seed / np.sqrt(N_F),
        },
        "false_fail_table": {
            "nominal_gaussian_3sigma": 0.0027,
            "if_true_sd_is_hi_edge_(3.73x)": float(2.0 * (1.0 - _phi(3.0 / SD_CI_HI_MULT))),
            "if_true_sd_is_lo_edge_(0.57x)": float(2.0 * (1.0 - _phi(3.0 / SD_CI_LO_MULT))),
            "note": "rate the 3*SE15 band false-fails under the null if the pilot sd mis-estimates the true sd at the 95% CI edges (3 dof)",
        },
        "power": {
            "half_reference_effect": S_REF / 2.0,
            "significance_at_half_reference_sigma": float(half_ref_sig),
            "n_required_for_5sigma_at_half_reference": n_required,
            "n_adequate_at_registered_15": n_adequate,
        },
        "gate_s_projection": {"se_slope_at_registered_allocation": float(se_slope)},
        "gate_v_failures": gate_v_fail,
        "stop_conditions": {
            "n_inadequate": not n_adequate,
            "gate_v_2plus_failures": len(gate_v_fail) >= 2,
        },
        "fleet_may_launch": bool(n_adequate and len(gate_v_fail) < 2),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if out["fleet_may_launch"] else 1


def _phi(x: float) -> float:
    """Standard normal CDF."""
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


if __name__ == "__main__":
    raise SystemExit(main())
