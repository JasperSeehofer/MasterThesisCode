"""End-of-fan-out verifier pass, item 4/20 -- B3 [POP] independent re-derivation.

Falsification brief: re-derive the decisive numbers FROM SOURCE (CSV/code), not
from any record restating them. This script is written independently of
`b3_1_pop_measure.py` (different helper structure, computed from scratch) as a
cross-check, not a re-run of the registered script.

Three independent reads, matching the item's charge:

(1) Structural: confirm the production dark-host draw law at commit 03cfe80
    (`dark_siren_injection.py:328`) is the SAME functional form as the
    estimator's completion-leg population integrand
    (`bayesian_statistics.precompute_completion_denominator` docstring, S1).
    This is a `git show` + source-read check, done in bash outside this
    script and reported alongside it (both funcs reduce to `dVc/dz/(1+z)`).

(2) The HEAD dark-class 1D score-at-truth divergence from row #138's
    historical baseline (7.16 sigma / 5.95 sigma claim).

(3) The five-bin chair-recompute (113.1%/125.9%) vs the record's flagged
    mislabel ("all 5 bins" 114.3%/129.9%, which silently includes 1-2
    sub-bottom-edge events).

No production code is modified. Read-only on all CSV/JSON inputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
CRB_PATH = (
    REPO_ROOT
    / "results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv"
)
CRB_MD5_EXPECTED = "9a1f2a14384a9281c97ca3be312ddaab"

VENUE_PATHS = {
    "iiib": REPO_ROOT
    / "results/campaign51_20260728/realistic_20260729/headreadout_20260827/iiib/event_likelihoods.csv",
    "joint_r1": REPO_ROOT
    / "results/campaign51_20260728/realistic_20260729/headreadout_20260827/joint_r1/event_likelihoods.csv",
}

HISTORICAL = {
    # row #137/#138, BIAS_HISTORY_LEDGER.md:1347-1348
    "iiib": (-0.635, 0.017),
    "joint_r1": (-0.565, 0.020),
}

REGISTERED_BIN_EDGES = [0.075, 0.392, 0.559, 0.659, 0.753, 1.018]


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def score_at_truth(df: pd.DataFrame) -> pd.Series:
    """Independent implementation: central finite difference of
    ln(combined_no_bh) between the two production h-grid nodes flanking
    h_true=0.73 by one native 0.01 step each side (h=0.72, h=0.74)."""
    piv = df.pivot_table(index="event_idx", columns="h", values="combined_no_bh")
    # nearest columns to 0.72 / 0.74
    cols = piv.columns.to_numpy()
    h_lo = cols[np.argmin(np.abs(cols - 0.72))]
    h_hi = cols[np.argmin(np.abs(cols - 0.74))]
    assert abs(h_lo - 0.72) < 1e-9 and abs(h_hi - 0.74) < 1e-9, (h_lo, h_hi)
    lo = piv[h_lo]
    hi = piv[h_hi]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (np.log(hi) - np.log(lo)) / (h_hi - h_lo)
    return s


def dark_class_mask(df: pd.DataFrame) -> pd.Series:
    """dark (C-C) = L_cat_no_bh == 0 at EVERY h node for that event."""
    g = df.groupby("event_idx")["L_cat_no_bh"]
    return g.apply(lambda s: bool((s.to_numpy() == 0.0).all()))


def main() -> dict:
    out: dict = {"crb": {}, "venues": {}}

    # ---- CRB dataset pin ----
    crb_md5 = md5sum(CRB_PATH)
    out["crb"]["path"] = str(CRB_PATH.relative_to(REPO_ROOT))
    out["crb"]["md5"] = crb_md5
    out["crb"]["md5_matches_expected"] = crb_md5 == CRB_MD5_EXPECTED
    crb = pd.read_csv(CRB_PATH)
    out["crb"]["n_rows"] = int(len(crb))
    if "in_catalog" in crb.columns:
        out["crb"]["n_dark_in_catalog_false"] = int((~crb["in_catalog"]).sum())
        out["crb"]["n_in_catalog_true"] = int((crb["in_catalog"]).sum())

    for venue, path in VENUE_PATHS.items():
        df = pd.read_csv(path)
        n_events = int(df["event_idx"].nunique())
        n_h = int(df["h"].nunique())

        dark_mask = dark_class_mask(df)
        dark_idx = set(dark_mask[dark_mask].index)

        score = score_at_truth(df)
        finite = score[np.isfinite(score)]
        n_dropped_nonfinite = int(len(score) - len(finite))

        dark_score = finite[finite.index.isin(dark_idx)]
        ens_mean = float(dark_score.mean())
        ens_n = int(len(dark_score))
        ens_sem = float(dark_score.std(ddof=1) / np.sqrt(ens_n))

        hist_mean, hist_sem = HISTORICAL[venue]
        abs_diff = abs(ens_mean - hist_mean)
        combined_sigma = abs_diff / np.sqrt(hist_sem**2 + ens_sem**2)

        # z_true for binning (need CRB join on event_idx -> luminosity_distance)
        # dist_to_redshift re-derived independently here (bisection on the
        # package's own `dist` function would require importing package code;
        # instead cross-check against the package directly to avoid a parallel
        # re-implementation of cosmology).
        import sys

        sys.path.insert(0, str(REPO_ROOT))
        from darksiren_emri.physical_relations import dist_to_redshift  # noqa: E402

        z_true = crb["luminosity_distance"].apply(lambda d: dist_to_redshift(float(d), h=0.73))
        z_true.index = crb.index  # event_idx == row index by construction

        dark_df = pd.DataFrame({"score": dark_score})
        dark_df["z_true"] = z_true.loc[dark_df.index]

        edges = REGISTERED_BIN_EDGES
        bin_n = []
        for i in range(len(edges) - 1):
            lo_e, hi_e = edges[i], edges[i + 1]
            if i < len(edges) - 2:
                m = (dark_df["z_true"] >= lo_e) & (dark_df["z_true"] < hi_e)
            else:
                m = (dark_df["z_true"] >= lo_e) & (dark_df["z_true"] <= hi_e)
            bin_n.append(int(m.sum()))

        n_in_5_bins = sum(bin_n)
        n_underflow = int((dark_df["z_true"] < edges[0]).sum())
        n_overflow = int((dark_df["z_true"] > edges[-1]).sum())
        n_all_dark_events = ens_n

        out["venues"][venue] = {
            "n_events_csv": n_events,
            "n_h_nodes": n_h,
            "n_dark_class": len(dark_idx),
            "n_score_nonfinite_dropped": n_dropped_nonfinite,
            "dark_ensemble_score_mean": ens_mean,
            "dark_ensemble_score_sem": ens_sem,
            "dark_ensemble_n": ens_n,
            "historical_mean": hist_mean,
            "historical_sem": hist_sem,
            "abs_diff_vs_historical": abs_diff,
            "combined_sigma_vs_historical": combined_sigma,
            "bin_counts_1to5": bin_n,
            "n_in_5_registered_bins": n_in_5_bins,
            "n_underflow_below_bottom_edge": n_underflow,
            "n_overflow_above_top_edge": n_overflow,
            "n_all_dark_events": n_all_dark_events,
            "n_all_minus_n_5bins": n_all_dark_events - n_in_5_bins,
        }

    return out


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))
    out_path = Path(__file__).resolve().parent / "item4_rederive_output.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
