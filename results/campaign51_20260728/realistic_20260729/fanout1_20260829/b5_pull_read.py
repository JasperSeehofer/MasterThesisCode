#!/usr/bin/env python3
"""[P3 B5.2-pre] True-host mass PULL distribution on the mirror (P3-2D) fleet.

Charter node B5.2-pre. Launched under rows #222/#223 -- charter node
B5.2-pre. Zero-compute: reuses the SAME banked fleet + pruned-catalogue
census as ``b5_window_count.py`` (append-only: that file and
``wgeom_instrument.py`` are imported, never edited).

Question this answers (SYNTHESIS_DOCKET_1_20260829.md sec.2 B5 pre-read
paragraph): a "3 sigma" log mass window (k=3) retains only 78.9% of true
hosts (b5_window_count.json), far short of the 99.7% a correctly-budgeted
log-normal sigma_lnM should retain at k=3. This script computes, per event
with a valid catalogued true host,

    x = ln( M_z / (1+z_host) ) - ln( BH_MASS_host )

and the pull p = x / sigma_lnM under two candidate definitions of
sigma_lnM (CV = BH_MASS_ERROR/BH_MASS off the host's own catalogue row):

    def1 ("ratio"):    sigma_lnM = CV                       (== BH_MASS_ERROR/BH_MASS,
                        which L9's code read -- see B5_2_PULL_READ_20260829.md sec.1 --
                        establishes IS ALREADY the ln-space sigma the R&V15 formula
                        computed before multiplying back onto BH_mass; "the ln-space
                        sigma directly" and "BH_MASS_ERROR/BH_MASS" are therefore the
                        SAME definition, not two of the task's three candidates)
    def2 ("ln1p"):      sigma_lnM = ln(1+CV)                 (the "small-error
                        correspondence" applied naively to a NOT-small CV)

plus a GW-side pull using the window's own GW-side small-error-correspondence sigma
(sigma_lnM,z = M_z_sigma/M_z, exactly what ``gw_window()`` uses) against the SAME
numerator x, to show how much of the pull's spread is host-side vs GW-side.

z_host is read from the host's OWN catalogue row (``cat.redshift[hp]``), not the
event's (kernel-smeared) drawn ``z_true`` -- the task's formula names ``z_host``, and the
mirror's ``_draw_kernel_survival_redshifts`` intentionally smears ``z_true`` away from
the host's listed z by the photo-z kernel (``correspondence_1d.py:_B0i2DLatents``
docstring). ``|z_host - z_true|`` is reported as a diagnostic, not substituted in.

Status: BUILDER-RUN zero-compute read, not independently verified (standing rule 2)
-- a different agent should re-run before any number here is cited as adopted.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))
from b5_window_count import TruthFleetEvent, load_fleet_with_truth  # noqa: E402

_WGEOM_DIR = _DIR.parent
sys.path.insert(0, str(_WGEOM_DIR))
from wgeom_instrument import (  # noqa: E402
    PrunedCatalogue,
    load_pruned_catalogue,
    verify_catalogue_pin,
    verify_fleet_row_counts,
)

OUT_DIR = _DIR

# Gaussian two-sided reference fractions for |p| <= {1.5, 2.5, 3} (standard normal CDF).
GAUSSIAN_REF = {"1.5": 0.866386, "2.5": 0.987581, "3.0": 0.997300}
K_THRESHOLDS = [1.5, 2.5, 3.0]


def _arm_tag(arm: str) -> str:
    """``bc_900101_work`` -> ``900101``."""
    return arm.replace("bc_", "").replace("_work", "")


def _fractions(pull: np.ndarray) -> dict[str, float]:
    absp = np.abs(pull)
    return {str(k): float(np.mean(absp <= k)) for k in K_THRESHOLDS}


def _stats_block(pull: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(pull.size),
        "mean": float(np.mean(pull)) if pull.size else float("nan"),
        "sd": float(np.std(pull, ddof=1)) if pull.size > 1 else float("nan"),
        "median": float(np.median(pull)) if pull.size else float("nan"),
        "fraction_abs_le": _fractions(pull) if pull.size else {},
    }


def run() -> dict[str, Any]:
    t0 = time.time()
    catalogue_md5 = verify_catalogue_pin()
    verify_fleet_row_counts()
    cat: PrunedCatalogue = load_pruned_catalogue(nrows=None)
    events: list[TruthFleetEvent] = load_fleet_with_truth()

    # Per-event arrays (only events with a valid catalogued true host --
    # SAME filter as b5_window_count.run()'s true-host-retention loop).
    rows: list[dict[str, Any]] = []
    n_total = len(events)
    n_out_of_range = 0
    n_nonpositive_mass = 0

    for ev in events:
        if not (ev.in_catalog and ev.host_galaxy_index >= 0):
            continue
        hp = ev.host_galaxy_index
        if hp >= cat.n_pruned:
            n_out_of_range += 1
            continue
        M_host = float(cat.bh_mass[hp])
        M_host_err = float(cat.bh_mass_error[hp])
        z_host = float(cat.redshift[hp])
        if M_host <= 0 or ev.M_z <= 0:
            n_nonpositive_mass += 1
            continue

        CV = M_host_err / M_host
        sigma1 = CV  # BH_MASS_ERROR/BH_MASS == ln-space sigma directly (L9 resolution)
        sigma2 = np.log1p(CV)  # ln(1+CV)
        sigma_gw = ev.M_z_sigma / ev.M_z  # small-error correspondence, GW side

        x = np.log(ev.M_z / (1.0 + z_host)) - np.log(M_host)

        rows.append(
            {
                "arm": _arm_tag(ev.arm),
                "event_idx": ev.event_idx,
                "M_z": ev.M_z,
                "M_z_sigma": ev.M_z_sigma,
                "z_true": ev.z_true,
                "z_host": z_host,
                "abs_dz_host_minus_true": abs(z_host - ev.z_true),
                "M_host": M_host,
                "M_host_err": M_host_err,
                "CV": CV,
                "x": x,
                "sigma_def1_ratio": sigma1,
                "sigma_def2_ln1p": sigma2,
                "sigma_gw": sigma_gw,
                "pull_def1": x / sigma1,
                "pull_def2": x / sigma2,
                "pull_gw": x / sigma_gw,
            }
        )

    n_valid = len(rows)

    def _arr(key: str, subset: list[dict[str, Any]]) -> np.ndarray:
        return np.array([r[key] for r in subset], dtype=np.float64)

    pooled_pull1 = _arr("pull_def1", rows)
    pooled_pull2 = _arr("pull_def2", rows)
    pooled_pull_gw = _arr("pull_gw", rows)
    pooled_cv = _arr("CV", rows)
    pooled_dz = _arr("abs_dz_host_minus_true", rows)

    pooled = {
        "n_events": n_valid,
        "CV_median": float(np.median(pooled_cv)),
        "CV_p10_p90": [float(np.percentile(pooled_cv, 10)), float(np.percentile(pooled_cv, 90))],
        "abs_dz_host_minus_ztrue_median": float(np.median(pooled_dz)),
        "abs_dz_host_minus_ztrue_p95": float(np.percentile(pooled_dz, 95)),
        "pull_def1_ratio_BHMASSERR_over_BHMASS": _stats_block(pooled_pull1),
        "pull_def2_ln1p_CV": _stats_block(pooled_pull2),
        "pull_gw_side_Mzsigma_over_Mz": _stats_block(pooled_pull_gw),
    }

    arms = sorted({r["arm"] for r in rows})
    per_arm: dict[str, Any] = {}
    for a in arms:
        subset = [r for r in rows if r["arm"] == a]
        per_arm[a] = {
            "n_events": len(subset),
            "pull_def1": _stats_block(_arr("pull_def1", subset)),
            "pull_def2": _stats_block(_arr("pull_def2", subset)),
            "pull_gw": _stats_block(_arr("pull_gw", subset)),
        }

    # Cross-check against b5_window_count.json's true-host retention numbers:
    # fraction |pull_def1| <= k should be IN THE VICINITY of (not identical to,
    # since that test uses the z_min/z_max cosmology-prior box + the GW-side
    # sigma too, not a bare pull-at-z_host threshold) the reported retention
    # at that k for the log geometry.
    cross_check = {
        "note": (
            "b5_window_count.json true_host_retention.fraction_retained (log geometry) "
            "for comparison; NOT expected to match exactly (that test uses the "
            "z_min/z_max cosmology-prior box from d_L +/- sigma_dL over H0 in "
            "[0.6,0.86], not a point z_host, and folds in the GW-side sigma too)."
        ),
        "fraction_abs_pull_def1_le_k": _fractions(pooled_pull1),
        "fraction_abs_pull_def2_le_k": _fractions(pooled_pull2),
        "b5_window_count_log_retention": {
            "k1.5": 0.7001326846528085,
            "k2.5": 0.7682441397611677,
            "k3.0": 0.7890314020344981,
        },
        "b5_window_count_linear_retention_k1.5": 0.9566563467492261,
    }

    out = {
        "provenance": {
            "launched_under": "rows #222/#223 -- charter node B5.2-pre",
            "git_commit": _git_commit(),
            "catalogue_md5": catalogue_md5,
            "n_events_loaded": n_total,
            "n_events_valid_true_host": n_valid,
            "n_events_out_of_range": n_out_of_range,
            "n_events_nonpositive_mass_excluded": n_nonpositive_mass,
            "elapsed_s": time.time() - t0,
        },
        "gaussian_reference_fraction_abs_le_k": GAUSSIAN_REF,
        "pooled": pooled,
        "per_arm": per_arm,
        "cross_check_vs_b5_window_count": cross_check,
    }
    return out


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_WGEOM_DIR), text=True
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    result = run()
    out_path = OUT_DIR / "b5_pull_read.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}")
    print(json.dumps(result["pooled"], indent=2))
    print(json.dumps(result["cross_check_vs_b5_window_count"], indent=2))
