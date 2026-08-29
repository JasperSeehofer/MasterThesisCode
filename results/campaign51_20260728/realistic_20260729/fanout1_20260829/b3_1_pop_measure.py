"""B3.1 [POP] -- zero-compute measure-first read.

Does row #138's population-mismatch prediction (M1-vs-comoving dark-class
score-at-truth) survive on the fused HEAD diagnostics (headreadout_20260827),
per z-bin, paired?

Launched under rows #222/#223 -- charter node B3.1.

No package code is modified. This script only reads banked CSVs + calls
existing, unmodified library functions (dist, dist_to_redshift,
comoving_volume_element, Model1CrossCheck.dN_dz_of_mass, Model1CrossCheck.R_emri)
to *re-derive* the predicted profile -- it does not copy any published number.

Run: uv run python b3_1_pop_measure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from darksiren_emri.physical_relations import (  # noqa: E402
    comoving_volume_element,
    dist,
    dist_to_redshift,
)
from darksiren_emri.cosmological_model import Model1CrossCheck  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
H_TRUE = 0.73

CRB_PATH = REPO_ROOT / "results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv"
CRB_MD5_EXPECTED = "9a1f2a14384a9281c97ca3be312ddaab"

VENUES = {
    "iiib": REPO_ROOT
    / "results/campaign51_20260728/realistic_20260729/headreadout_20260827/iiib/event_likelihoods.csv",
    "joint_r1": REPO_ROOT
    / "results/campaign51_20260728/realistic_20260729/headreadout_20260827/joint_r1/event_likelihoods.csv",
}

# Row #138's historical dark-class 1D ensemble score at truth, for the "did HEAD
# move" comparison (item 4). Source: BIAS_HISTORY_LEDGER.md:1347-1348 (row #137,
# quoted again in row #138's memo) and hier_provenance_stamps_20260826.md:150.
HISTORICAL_SCORE = {
    "iiib": (-0.635, 0.017),
    "joint_r1": (-0.565, 0.020),
}

# REGISTERED BEFORE LOOKING AT ANY HEAD NUMBER (per task instruction):
# z-bin edges = the memo's own fixed edges (population_mismatch_dark_score.md
# Table, iiib dark class), reused verbatim rather than re-quantiled on the HEAD
# data, so every bin is a paired comparison against a previously published
# number, not a fresh partition chosen after seeing the result.
REGISTERED_BIN_EDGES = [0.075, 0.392, 0.559, 0.659, 0.753, 1.018]

# REGISTERED class split (before looking): "dark" = C-C per
# PREREG_COMPLETION_CLASS_DECOMPOSITION.md / wgeo_s0_coupling_20260827.md --
# L_cat_no_bh == 0 at *every* h node. "matched" = the complement (C-A union
# C-B: has >=1 h node with L_cat_no_bh > 0), i.e. any catalogue support at all.
# This conflates C-A (true host in catalogue) and C-B (impostor-only) into one
# "matched" bucket -- disclosed as a caveat, per row #141's finding that C-A
# alone pulls the *opposite* sign.


def md5sum(path: Path) -> str:
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def w_true_of_z(z: np.ndarray, n_mass: int = 400) -> np.ndarray:
    """Marginal-in-mass M1 injection rate density in z, up to an overall
    z-independent normalisation (only the *shape* is used downstream).

    w_true(z) ∝ INTEGRAL d(log10 M)  dN/dz|_mass(M, z) * R_EMRI(M)

    dN_dz_of_mass and R_emri are Model1CrossCheck's own staticmethods (the
    same functions that define the injected (M, z) sampling density via
    _log_probability / emri_distribution) -- called directly, not
    reimplemented, and not touching package code.

    Integration measure, checked against the sampler (not assumed): emcee's
    `log_probability(x)` in `setup_emri_events_sampler` walks x = (log10(M),
    z) and returns `_log_probability(10**x[0], x[1])` -- i.e.
    `emri_distribution(M, z)` is used AS THE DENSITY IN THE SAMPLED
    COORDINATES (log10 M, z), with no extra `dM/d(log10 M) = M ln10`
    Jacobian applied anywhere in that call chain. `merger_distribution_coefficients`
    is also literally a fit indexed by log10(M) anchor points (mass_bin
    4.5/5.0/5.5/6.0/6.25), consistent with dN/dz being a per-decade-in-mass
    density already. The marginal used by the injected sampler is therefore
    the plain integral over log10(M), not over M -- verified by a direct
    A/B check (`/tmp/wtrue_check.py`, this session): the M-weighted variant
    changes the z=0.1->1.0 growth factor by ~25% (39x vs 29.5x) but not the
    qualitative shape; the JSON output flags this convention explicitly.
    """
    m_lo, m_hi = 1.0e4, 1.0e7  # M_SOURCE_FRAME_MIN / MAX (constants.py)
    logm = np.linspace(np.log10(m_lo), np.log10(m_hi), n_mass)
    m_grid = 10.0**logm
    out = np.empty_like(z, dtype=np.float64)
    for i, zi in enumerate(z):
        dndz = np.array([Model1CrossCheck.dN_dz_of_mass(m, float(zi)) for m in m_grid])
        r = np.array([Model1CrossCheck.R_emri(m) for m in m_grid])
        integrand = dndz * r  # density in (log10 M, z) space, per sampler convention
        out[i] = np.trapezoid(integrand, logm)  # integrate over log10(M) directly
    return out


def w_model_of_z(z: np.ndarray, h: float = H_TRUE) -> np.ndarray:
    """Estimator's dark-class population weight: constant comoving density,
    dVc/dz / (1+z) (bayesian_statistics.py precompute_completion_denominator
    integrand, population part only -- p_det factored out since it appears
    identically in both numerator and denominator of the score-at-truth
    saddle-point ratio and cancels; only the SHAPE of w matters, per the
    memo's own §2 note)."""
    dvc = np.asarray(comoving_volume_element(z, h=h), dtype=np.float64)
    return dvc / (1.0 + z)


def dz_star_dh(z: np.ndarray, h: float = H_TRUE, dz: float = 1.0e-4) -> np.ndarray:
    """dz*/dh at the saddle z*=z (score-at-truth: the saddle sits exactly at
    the injected z when h = h_true), via a(z*)=h*d_L(z*,h) -- and since
    d_L(z,h) = d_L(z,1)/h exactly (H_0 propto h, all else fixed), a(z) =
    dist(z, h=1) is h-independent, computed EXACTLY rather than approximated.

    dz*/dh = a(z*) / (h * da/dz(z*))
    """
    a_of_z = lambda zz: np.asarray([dist(float(x), h=1.0) for x in np.atleast_1d(zz)])
    a_z = a_of_z(z)
    da_dz = (a_of_z(z + dz) - a_of_z(z - dz)) / (2.0 * dz)
    return a_z / (h * da_dz)


def w_true_of_z_mass_weighted(z: np.ndarray, n_mass: int = 400) -> np.ndarray:
    """Alternative (REJECTED) integration convention: INTEGRAL dM instead of
    d(log10 M) -- kept only for the disclosed A/B check in the record (item 1
    caveat). NOT used by predicted_delta_score."""
    m_lo, m_hi = 1.0e4, 1.0e7
    logm = np.linspace(np.log10(m_lo), np.log10(m_hi), n_mass)
    m_grid = 10.0**logm
    out = np.empty_like(z, dtype=np.float64)
    for i, zi in enumerate(z):
        dndz = np.array([Model1CrossCheck.dN_dz_of_mass(m, float(zi)) for m in m_grid])
        r = np.array([Model1CrossCheck.R_emri(m) for m in m_grid])
        integrand_dlnm = dndz * r * m_grid
        out[i] = np.trapezoid(integrand_dlnm, logm * np.log(10.0))
    return out


def predicted_delta_score(
    z: np.ndarray, h: float = H_TRUE, dz: float = 1.0e-4, mass_weighted: bool = False
) -> np.ndarray:
    """Δscore(z) ≈ [d ln(w_model/w_true)/dz](z) * dz*/dh(z), memo §2."""
    w_true_fn = w_true_of_z_mass_weighted if mass_weighted else w_true_of_z
    ln_ratio = lambda zz: np.log(w_model_of_z(zz, h=h)) - np.log(w_true_fn(zz))
    dln_ratio_dz = (ln_ratio(z + dz) - ln_ratio(z - dz)) / (2.0 * dz)
    return dln_ratio_dz * dz_star_dh(z, h=h)


def measured_score_central_diff(
    df: pd.DataFrame, h_lo: float = 0.72, h_hi: float = 0.74
) -> pd.Series:
    """Per-event central finite-difference score at truth on the h-grid,
    step = h_hi - h_lo = 0.02 (nearest grid neighbours either side of
    h_true=0.73 on the 41-node, 0.01-spaced production grid)."""
    lo = df[np.isclose(df["h"], h_lo)].set_index("event_idx")["combined_no_bh"]
    hi = df[np.isclose(df["h"], h_hi)].set_index("event_idx")["combined_no_bh"]
    common = lo.index.intersection(hi.index)
    lo, hi = lo.loc[common], hi.loc[common]
    with np.errstate(divide="ignore"):
        score = (np.log(hi) - np.log(lo)) / (h_hi - h_lo)
    return score


def main() -> dict:
    crb_md5 = md5sum(CRB_PATH)
    crb_ok = crb_md5 == CRB_MD5_EXPECTED
    crb = pd.read_csv(CRB_PATH)
    z_true_all = np.array(
        [dist_to_redshift(float(d), h=H_TRUE) for d in crb["luminosity_distance"]]
    )
    crb = crb.copy()
    crb["z_true"] = z_true_all
    crb["event_idx"] = np.arange(len(crb))

    result: dict = {
        "node": "B3.1_POP",
        "authorization": "rows #222/#223 -- charter node B3.1",
        "crb_path": str(CRB_PATH.relative_to(REPO_ROOT)),
        "crb_md5": crb_md5,
        "crb_md5_matches_run_metadata": crb_ok,
        "n_crb_rows": int(len(crb)),
        "registered_bin_edges": REGISTERED_BIN_EDGES,
        "finite_difference": {"h_lo": 0.72, "h_hi": 0.74, "step": 0.02, "h_true": H_TRUE},
        "venues": {},
    }

    for venue, path in VENUES.items():
        df = pd.read_csv(path)
        n_events = df["event_idx"].nunique()
        n_h = df["h"].nunique()

        # class membership: dark (C-C) = L_cat_no_bh == 0 at every h node
        zero_at_every_h = df.groupby("event_idx")["L_cat_no_bh"].apply(lambda s: bool((s == 0.0).all()))
        dark_idx = set(zero_at_every_h[zero_at_every_h].index)
        matched_idx = set(zero_at_every_h[~zero_at_every_h].index)

        # non-positive / non-finite combined_no_bh check at the three nodes used
        sub = df[np.isclose(df["h"], 0.72) | np.isclose(df["h"], 0.73) | np.isclose(df["h"], 0.74)]
        n_nonpos = int((sub["combined_no_bh"] <= 0).sum())

        score = measured_score_central_diff(df)
        finite_mask = np.isfinite(score)
        n_nonfinite = int((~finite_mask).sum())
        score = score[finite_mask]

        merged = crb.set_index("event_idx")[["z_true"]].loc[score.index]
        merged["score"] = score
        merged["dark"] = merged.index.map(lambda i: i in dark_idx)

        dark_df = merged[merged["dark"]]
        matched_df = merged[~merged["dark"]]

        z_true_dark = dark_df["z_true"].to_numpy()
        pred_dark = predicted_delta_score(z_true_dark, h=H_TRUE)
        dark_df = dark_df.copy()
        dark_df["predicted"] = pred_dark

        # ensemble numbers (dark class, 1D channel) -- item 4
        ens_mean = float(dark_df["score"].mean())
        ens_sem = float(dark_df["score"].std(ddof=1) / np.sqrt(len(dark_df)))
        ens_pred = float(dark_df["predicted"].mean())
        ens_coverage = ens_pred / ens_mean if ens_mean != 0 else float("nan")

        hist_mean, hist_sem = HISTORICAL_SCORE[venue]
        moved = abs(ens_mean - hist_mean)
        moved_sigma = moved / np.sqrt(hist_sem**2 + ens_sem**2)

        # per-bin table
        edges = REGISTERED_BIN_EDGES
        bins_out = []
        for i in range(len(edges) - 1):
            lo_e, hi_e = edges[i], edges[i + 1]
            in_bin = (dark_df["z_true"] >= lo_e) & (dark_df["z_true"] < hi_e if i < len(edges) - 2 else dark_df["z_true"] <= hi_e)
            b = dark_df[in_bin]
            n_b = len(b)
            if n_b == 0:
                bins_out.append(
                    {"z_lo": lo_e, "z_hi": hi_e, "n": 0, "measured_mean": None, "measured_sem": None, "predicted_mean": None, "ratio": None}
                )
                continue
            m_mean = float(b["score"].mean())
            m_sem = float(b["score"].std(ddof=1) / np.sqrt(n_b)) if n_b > 1 else float("nan")
            p_mean = float(b["predicted"].mean())
            ratio = p_mean / m_mean if m_mean != 0 else float("nan")
            bins_out.append(
                {"z_lo": lo_e, "z_hi": hi_e, "n": n_b, "measured_mean": m_mean, "measured_sem": m_sem, "predicted_mean": p_mean, "ratio": ratio}
            )
        # robustness: coverage excluding bin 1 (0.075-0.392), where measured
        # and predicted are both near zero and OPPOSITE in sign (uninformative
        # denominator -- see item 2 caveat)
        b25 = [b for b in bins_out if b["n"] > 0][1:]
        n_b25 = sum(b["n"] for b in b25)
        if n_b25 > 0:
            meas_b25 = sum(b["n"] * b["measured_mean"] for b in b25) / n_b25
            pred_b25 = sum(b["n"] * b["predicted_mean"] for b in b25) / n_b25
            coverage_b25 = pred_b25 / meas_b25 if meas_b25 != 0 else float("nan")
        else:
            meas_b25 = pred_b25 = coverage_b25 = float("nan")

        # disclosed A/B: alternate (rejected) integration measure for w_true
        # (integral dM instead of d(log10 M)) -- see predicted_delta_score
        # docstring for why d(log10 M) is the one that matches the sampler.
        pred_mass_weighted = predicted_delta_score(z_true_dark, h=H_TRUE, mass_weighted=True)
        ens_pred_mass_weighted = float(np.mean(pred_mass_weighted))

        # events beyond the top registered edge (memo pool had max 1.018;
        # HEAD z_true may extend further under the current campaign depth)
        overflow = dark_df[dark_df["z_true"] > edges[-1]]
        underflow = dark_df[dark_df["z_true"] < edges[0]]

        # matched-class ensemble (item 2/3 context; opposite-sign sanity check
        # per row #141's C-A finding)
        matched_mean = float(matched_df["score"].mean()) if len(matched_df) else float("nan")
        matched_sem = (
            float(matched_df["score"].std(ddof=1) / np.sqrt(len(matched_df))) if len(matched_df) > 1 else float("nan")
        )

        result["venues"][venue] = {
            "n_events_scored_csv": int(n_events),
            "n_h_nodes": int(n_h),
            "n_events_zero_or_nonpositive_combined_no_bh_near_truth": n_nonpos,
            "n_events_score_nonfinite": n_nonfinite,
            "n_dark": int(len(dark_idx)),
            "n_matched": int(len(matched_idx)),
            "dark_ensemble": {
                "measured_mean": ens_mean,
                "measured_sem": ens_sem,
                "predicted_mean": ens_pred,
                "coverage_fraction": ens_coverage,
                "n": int(len(dark_df)),
                "predicted_mean_ALT_mass_weighted_integration_REJECTED_CONVENTION": ens_pred_mass_weighted,
            },
            "dark_ensemble_bins2to5_only_robustness": {
                "measured_mean": meas_b25,
                "predicted_mean": pred_b25,
                "coverage_fraction": coverage_b25,
                "n": int(n_b25),
                "note": "excludes bin 1 (0.075-0.392), where measured and predicted are both near zero and opposite in sign",
            },
            "matched_ensemble": {
                "measured_mean": matched_mean,
                "measured_sem": matched_sem,
                "n": int(len(matched_df)),
            },
            "historical_row138": {"mean": hist_mean, "sem": hist_sem},
            "head_vs_historical": {
                "abs_diff": float(moved),
                "diff_in_combined_sigma": float(moved_sigma),
            },
            "bins": bins_out,
            "n_overflow_above_top_edge": int(len(overflow)),
            "n_underflow_below_bottom_edge": int(len(underflow)),
            "z_true_max_dark": float(dark_df["z_true"].max()) if len(dark_df) else None,
            "z_true_min_dark": float(dark_df["z_true"].min()) if len(dark_df) else None,
        }

    return result


if __name__ == "__main__":
    res = main()
    out_path = OUT_DIR / "b3_pop_prediction.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
