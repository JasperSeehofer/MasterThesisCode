"""Generator for Chapter 4 — "The Universe Only Shows You Its Loud Half".

Produces the two data files behind the chapter's interactives.

``book/site/data/ch04_denominator.json``  (I4.1 "Delete the Denominator")
    The real ``seed61000/real_r1`` event set as a per-h **sum of log
    likelihoods**, plus the run's own selection normalisation ``D(h)``.  The
    browser reconstructs the two live states in log space:

        denominator ON   :  S(h)  =  sum_i ln p_i(h)
        denominator OFF  :  S(h)  +  N ln D(h)          (multiply D back in)

    because each per-event likelihood is a single ratio carrying ``D(h)`` in
    its denominator (``bayesian_statistics.py:3006-3009``), and ``D(h)`` is
    the *same* number for every event at a given h.

    The widget's third state — the historical **local-window** denominator —
    is NOT reconstructible from this run: that estimator was replaced in
    Phase 32, three code eras before campaign #51/#53.  It is carried as a
    **recorded** measurement only (MAP 0.60, bias -17.8% -> 0.0%;
    ``BIAS_HISTORY_LEDGER.md`` row #9, ``docs/H0_BIAS_RESOLUTION.md:1980``)
    and is never recomputed or blended with the live curves.

``book/site/data/ch04_horizon.json``  (I4.2 "The Horizon Breather")
    The detection-horizon survival ``p_det(d_L)`` evaluated with the
    project's OWN estimator object (``SimulationDetectionProbability``) on
    the production injection pool, and the "visible volume" integrand of

        D(h) = INTEGRAL p_det(d_L(z,h)) (dV_c/dz dOmega) dz/(1+z)  [Mpc^3/sr]

    sampled per h on a display grid.  The *level* of the plotted integral is
    a pooled recomputation; the *authoritative* ``D(h)`` shown beside it is
    the run's own (see PROVENANCE below).

PROVENANCE — D(h)
-----------------
``D(h)`` is taken from the run's own log lines
(``seed61000/mixture_leg_log_extract.txt``, 41 h-values, 7 s.f.), which are
committed.  A fresh call to ``precompute_completion_denominator`` on the
staged pool lands 4-7% ABOVE them, because production passes a
``completeness`` object which switches D(h) onto the *sky-aware* path (per
ecliptic-latitude band, weighted by equal-area HEALPix pixel counts —
``bayesian_statistics.py:1074-1088``) while the plain call uses the pooled
isotropic survival.  The book therefore quotes the run's D(h) as the number
and labels the recomputed integrand as a *shape*.  Both are emitted, with the
ratio, so the page can state the difference instead of hiding it.
Recorded in ``book/design/flags/ch04_FLAGS.md``.

DATA AVAILABILITY
-----------------
Everything I4.1 needs is git-tracked and present in any checkout of this
branch.  The 200k-row injection pool is **not** tracked (it lives in the
working tree of the main checkout only), so the horizon step resolves it
from, in order: this repo root, then a sibling ``MasterThesisCode`` checkout
next to this one.  If neither is present the already-committed
``ch04_horizon.json`` is left untouched and a NOTICE is printed — the
generator never fails a build over an untracked artifact, and never writes a
partial or silently-degraded file.

Determinism: no RNG anywhere; every number is read or recomputed from
committed artifacts.  Read-only outside ``book/``.

Run as::

    /home/jasper/Repositories/MasterThesisCode/.venv/bin/python \\
        book/generators/gen_ch04.py
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from master_thesis_code.bayesian_inference.posterior_combination import (  # noqa: E402
    build_likelihood_array,
    load_posterior_jsons,
)
from master_thesis_code.constants import H as H_TRUE  # noqa: E402
from master_thesis_code.constants import SNR_THRESHOLD  # noqa: E402

# --- repo-relative artifact paths (§4.2 rule 7; never absolute) ------------
CAMPAIGN_REL = Path("results/campaign51_20260728/realistic_20260729")
SEED_REL = CAMPAIGN_REL / "seed61000"
POSTERIORS_REL = SEED_REL / "real_r1" / "posteriors"
CRB_REL = SEED_REL / "prepared_cramer_rao_bounds.csv"
RUN_LOG_REL = SEED_REL / "mixture_leg_log_extract.txt"
POOL_REL = CAMPAIGN_REL / "gate_b_20260730" / "injection_pool_mix200k_20260728"

OUT_DIR = Path(__file__).resolve().parent.parent / "site" / "data"
OUT_DENOM = OUT_DIR / "ch04_denominator.json"
OUT_HORIZON = OUT_DIR / "ch04_horizon.json"

EVENT_889 = 889  # the book's running example (pedagogy B4)

# REALISTIC_READOUT.md §1, row "61000 | r1" — the published values this
# generator must reproduce, or stop.
READOUT_MAP = 0.740
READOUT_MEAN = 0.7321

# Display sampling for the horizon widget (presentation grids, not physics).
N_Z_DISPLAY = 71
N_DL_DISPLAY = 121
Z_DISPLAY_MAX = 1.40


def _r(x: Any, sig: int = 8) -> float:
    """Round to `sig` significant digits (JSON size hygiene; every displayed
    quantity is quoted to at most 6 s.f.)."""
    v = float(x)
    if v == 0.0 or not np.isfinite(v):
        return v
    return float(f"%.{sig}g" % v)


def _rl(a: Any, sig: int = 8) -> list[float]:
    return [_r(v, sig) for v in np.asarray(a, dtype=np.float64).ravel()]


def _pool_dir() -> Path | None:
    """Locate the (untracked) production injection pool without hardcoding a
    machine path: this checkout first, then a sibling ``MasterThesisCode``."""
    for root in (REPO_ROOT, REPO_ROOT.parent / "MasterThesisCode"):
        candidate = root / POOL_REL
        if candidate.is_dir() and any(candidate.glob("injection_h_*_task_*.csv")):
            return candidate
    return None


# ---------------------------------------------------------------------------
# D(h) from the run's own log lines
# ---------------------------------------------------------------------------
_LOG_RE = re.compile(
    r"D\(h=(?P<h>[0-9.]+)\)\s*=\s*(?P<D>[0-9.eE+-]+)\s*\[z_max=(?P<zmax>[0-9.]+),"
    r"\s*dl_max=(?P<dlmax>[0-9.]+)\s*Gpc\]"
)


def read_run_log_D() -> tuple[dict[float, float], dict[float, float], float]:
    """Parse ``D(h)``, ``z_max(h)`` and ``dl_max`` out of the run's log extract."""
    text = (REPO_ROOT / RUN_LOG_REL).read_text()
    d_table: dict[float, float] = {}
    zmax: dict[float, float] = {}
    dlmax: set[float] = set()
    for m in _LOG_RE.finditer(text):
        d_table[float(m.group("h"))] = float(m.group("D"))
        zmax[float(m.group("h"))] = float(m.group("zmax"))
        dlmax.add(float(m.group("dlmax")))
    if not d_table:
        msg = f"No D(h) log lines found in {RUN_LOG_REL}"
        raise RuntimeError(msg)
    if len(dlmax) != 1:
        # The h-invariance of the detection horizon is a load-bearing claim of
        # this chapter — if the run log ever contradicts it, stop.
        msg = f"dl_max is not h-invariant in the run log: {sorted(dlmax)}"
        raise RuntimeError(msg)
    return d_table, zmax, dlmax.pop()


def posterior_from_logsum(log_sum: np.ndarray, h_grid: np.ndarray) -> np.ndarray:
    y = np.exp(log_sum - log_sum.max())
    return y / np.trapezoid(y, h_grid)


# ---------------------------------------------------------------------------
# Step 1 — I4.1
# ---------------------------------------------------------------------------
def build_denominator_payload() -> dict[str, Any]:
    h_values, event_likelihoods = load_posterior_jsons(REPO_ROOT / POSTERIORS_REL)
    likelihoods, detection_indices = build_likelihood_array(h_values, event_likelihoods)
    h_grid = np.asarray(h_values, dtype=np.float64)
    n_total = int(len(detection_indices))

    complete = ~np.any(np.isnan(likelihoods), axis=1)
    likelihoods = likelihoods[complete, :]
    n_zero = int((likelihoods <= 0.0).sum())
    if n_zero:
        msg = f"{n_zero} non-positive likelihood cells — zero-handling would be needed; stop."
        raise RuntimeError(msg)
    n_ev = int(likelihoods.shape[0])

    d_table, zmax_table, dl_max = read_run_log_D()
    missing = [float(h) for h in h_grid if float(h) not in d_table]
    if missing:
        msg = f"run log is missing D(h) for {missing}"
        raise RuntimeError(msg)
    d_h = np.array([d_table[float(h)] for h in h_grid])
    z_max = np.array([zmax_table[float(h)] for h in h_grid])
    log_d = np.log(d_h)

    log_sum = np.log(likelihoods).sum(axis=0)
    post_on = posterior_from_logsum(log_sum, h_grid)
    post_off = posterior_from_logsum(log_sum + n_ev * log_d, h_grid)

    def summarize(y: np.ndarray) -> dict[str, float]:
        return {
            "map": _r(h_grid[int(np.argmax(y))], 6),
            "mean": _r(float(np.trapezoid(y * h_grid, h_grid)), 6),
            "edge_low_over_peak": _r(float(y[0] / y.max()), 4),
            "edge_high_over_peak": _r(float(y[-1] / y.max()), 4),
        }

    s_on, s_off = summarize(post_on), summarize(post_off)

    # Fidelity gate: the "denominator ON" state IS the published 1D posterior.
    if abs(s_on["map"] - READOUT_MAP) > 1e-9 or abs(s_on["mean"] - READOUT_MEAN) > 5e-5:
        msg = (
            "Recomputed r1 posterior disagrees with REALISTIC_READOUT.md §1 "
            f"(MAP {s_on['map']} vs {READOUT_MAP}, mean {s_on['mean']} vs "
            f"{READOUT_MEAN}) — STOP and flag; do not reconcile silently."
        )
        raise RuntimeError(msg)

    i_map = int(np.argmax(post_on))
    tilt = {
        "log_D_060_minus_086": _r(float(log_d[0] - log_d[-1]), 6),
        "denominator_nats_060_over_086": _r(float(n_ev * (log_d[0] - log_d[-1])), 6),
        "data_nats_map_over_060": _r(float(log_sum[i_map] - log_sum[0]), 6),
        "data_nats_map_over_086": _r(float(log_sum[i_map] - log_sum[-1]), 6),
        "note": (
            "denominator_nats = N [ln D(0.60) - ln D(0.86)] — the monotone pull that "
            "deleting D(h) injects across the grid. data_nats = the same-span swing of "
            "Sigma_i ln p_i(h) with D(h) in place. No nats->h conversion is quoted "
            "anywhere in this chapter (sources map §7.20)."
        ),
    }

    crb = pd.read_csv(REPO_ROOT / CRB_REL)
    row = crb.iloc[EVENT_889]
    dl889 = float(row["luminosity_distance"])
    snr889 = float(row["SNR"])
    # The CRB column is a variance in Gpc^2, so its square root is the ABSOLUTE
    # sigma_dL in Gpc — it is not a fraction.  Shipping it under the key
    # ``sigma_dL_over_dL`` was the book-wide units slip corrected on 2026-07-31
    # (REVISION_WORKLIST.md §A-D1): the key now names what it holds, and the
    # fractional precision is emitted separately.
    sigma_dl_gpc = float(np.sqrt(float(row["delta_luminosity_distance_delta_luminosity_distance"])))
    ev889 = {
        "index": EVENT_889,
        "SNR": _r(snr889, 6),
        "d_L_Gpc": _r(dl889, 6),
        "d_L_Mpc": _r(dl889 * 1000.0, 4),
        "M_Msun": _r(float(row["M"]), 6),
        "mu_Msun": _r(float(row["mu"]), 4),
        "host_galaxy_index": int(row["host_galaxy_index"]),
        "in_catalog": bool(row["in_catalog"]),
        "sigma_dL_Gpc": _r(sigma_dl_gpc, 3),
        "sigma_dL_over_dL": _r(sigma_dl_gpc / dl889, 3),
        "d_horizon_Gpc": _r(snr889 * dl889 / SNR_THRESHOLD, 6),
        "horizon_over_distance": _r(snr889 / SNR_THRESHOLD, 6),
        "is_loudest_of": int(len(crb)),
        "snr_is_max": bool(int(crb["SNR"].idxmax()) == EVENT_889),
    }

    return {
        "chapter": "ch04",
        "h_grid": _rl(h_grid, 6),
        "h_true": float(H_TRUE),
        "n_events": n_ev,
        "n_events_total": n_total,
        # log-space native (pedagogy interaction principle 5)
        "log_like_sum": _rl(log_sum - log_sum.max(), 10),
        "log_D": _rl(log_d, 10),
        "D_h": _rl(d_h, 8),
        "z_max": _rl(z_max, 6),
        "dl_max_Gpc": dl_max,
        "posteriors": {"on": _rl(post_on, 6), "off": _rl(post_off, 6)},
        "summary": {"on": s_on, "off": s_off},
        "recorded_local_window": {
            "map": 0.60,
            "bias_in_h_pct": -17.8,
            "map_after_full_volume": 0.73,
            "bias_after_pct": 0.0,
            "era": "Phase 32",
            "provenance": "BIAS_HISTORY_LEDGER.md #9; docs/H0_BIAS_RESOLUTION.md:1980",
            "note": (
                "Recorded measurement from a different venue and code era. The "
                "local-window estimator does not exist in campaign #51/#53 and is "
                "NOT recomputed here."
            ),
        },
        "checks": {
            "readout_map": READOUT_MAP,
            "readout_mean": READOUT_MEAN,
            "readout_source": "REALISTIC_READOUT.md §1, row seed 61000 / r1",
            "readout_match": True,
            "n_nonpositive_cells": n_zero,
        },
        "tilt": tilt,
        "event889": ev889,
        "classes": {
            "n_rows": int(len(crb)),
            "n_in_catalog": int((crb["host_galaxy_index"] >= 0).sum()),
            "n_dark": int((crb["host_galaxy_index"] < 0).sum()),
        },
        "source": {
            "run": str(POSTERIORS_REL.parent),
            "posteriors": str(POSTERIORS_REL),
            "run_log": str(RUN_LOG_REL),
            "channel": "1D (posteriors/, without host BH mass)",
        },
    }


# ---------------------------------------------------------------------------
# Step 2 — I4.2 (needs the untracked injection pool)
# ---------------------------------------------------------------------------
def build_horizon_payload(pool_dir: Path) -> dict[str, Any]:
    from master_thesis_code.bayesian_inference.bayesian_statistics import (
        precompute_completion_denominator,
    )
    from master_thesis_code.bayesian_inference.simulation_detection_probability import (
        SimulationDetectionProbability,
    )
    from master_thesis_code.constants import OMEGA_DE, OMEGA_M
    from master_thesis_code.physical_relations import comoving_volume_element, dist_vectorized

    files = sorted(glob.glob(str(pool_dir / "injection_h_*_task_*.csv")))
    inj = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    strata = inj["stratum"].fillna("a").astype(str)
    a_rows = inj[strata == "a"]
    census = {
        "n_files": len(files),
        "n_data_rows": int(len(inj)),
        "n_lines_with_headers": int(len(inj) + len(files)),
        "strata": {s: int((strata == s).sum()) for s in sorted(strata.unique())},
        "z_cut": _r(float(inj["z_cut"].max()), 4),
        "stratum_a_detected_fraction": _r(
            float((a_rows["SNR"].to_numpy() >= SNR_THRESHOLD).mean()), 4
        ),
        "stratum_a_max_horizon_Gpc": _r(
            float(
                (
                    a_rows["SNR"].to_numpy()
                    * a_rows["luminosity_distance"].to_numpy()
                    / SNR_THRESHOLD
                ).max()
            ),
            6,
        ),
        "code_revisions": sorted(inj["code_rev"].astype(str).unique().tolist()),
    }

    dp = SimulationDetectionProbability(
        injection_data_dir=str(pool_dir), snr_threshold=SNR_THRESHOLD
    )
    fingerprints = {f"{h:.2f}": _r(float(dp.get_dl_max(h)), 8) for h in (0.60, 0.73, 0.86)}
    if len(set(fingerprints.values())) != 1:
        msg = f"dl_max is not h-invariant: {fingerprints}"
        raise RuntimeError(msg)

    d_table, zmax_table, dl_max_log = read_run_log_D()
    h_grid = np.array(sorted(d_table))
    d_h = np.array([d_table[float(h)] for h in h_grid])
    z_max = np.array([zmax_table[float(h)] for h in h_grid])

    dl_grid = np.linspace(0.02, float(dp.get_dl_max(float(H_TRUE))), N_DL_DISPLAY)
    zeros = np.zeros_like(dl_grid)
    p_det_dl = np.asarray(
        dp.detection_probability_without_bh_mass_interpolated(
            dl_grid, zeros, zeros, h=float(H_TRUE)
        ),
        dtype=np.float64,
    )
    p_det_dl_low = np.asarray(
        dp.detection_probability_without_bh_mass_interpolated(dl_grid, zeros, zeros, h=0.60),
        dtype=np.float64,
    )
    survival_invariance = _r(float(np.max(np.abs(p_det_dl - p_det_dl_low))), 3)

    z_disp = np.linspace(0.0, Z_DISPLAY_MAX, N_Z_DISPLAY)
    z_zeros = np.zeros_like(z_disp)
    p_det_z: list[list[float]] = []
    integrand: list[list[float]] = []
    dl_of_z: list[list[float]] = []
    for j, h in enumerate(h_grid):
        d_of_z = np.asarray(dist_vectorized(z_disp, h=float(h)), dtype=np.float64)
        pz = np.asarray(
            dp.detection_probability_without_bh_mass_interpolated(
                d_of_z, z_zeros, z_zeros, h=float(h)
            ),
            dtype=np.float64,
        )
        pz = np.where(z_disp <= z_max[j], pz, 0.0)  # the run's own selection-domain cap
        dvc = np.asarray(comoving_volume_element(z_disp, h=float(h)), dtype=np.float64)
        p_det_z.append(_rl(pz, 5))
        dl_of_z.append(_rl(d_of_z, 5))
        integrand.append(_rl(pz * dvc / (1.0 + z_disp), 5))

    d_pool_table = precompute_completion_denominator(
        [float(h) for h in h_grid], dp, OMEGA_M, OMEGA_DE, z_max_cap=float(np.max(z_max))
    )
    d_pool = np.array([d_pool_table[float(h)] for h in h_grid])
    ratio = d_pool / d_h
    # Shape agreement after removing the overall level (the widget's claim).
    shape_spread = _r(float(ratio.max() / ratio.min() - 1.0), 3)

    dl889 = float(pd.read_csv(REPO_ROOT / CRB_REL).iloc[EVENT_889]["luminosity_distance"])
    p_det_889 = float(
        np.asarray(
            dp.detection_probability_without_bh_mass_interpolated(
                np.array([dl889]), np.zeros(1), np.zeros(1), h=float(H_TRUE)
            ),
            dtype=np.float64,
        )[0]
    )

    return {
        "chapter": "ch04",
        "h_grid": _rl(h_grid, 6),
        "h_true": float(H_TRUE),
        "snr_threshold": float(SNR_THRESHOLD),
        "dl_grid_Gpc": _rl(dl_grid, 6),
        "p_det_of_dl": _rl(p_det_dl, 5),
        "survival_h_invariance_max_abs_diff": survival_invariance,
        "dl_max_Gpc_by_h": fingerprints,
        "dl_max_run_log_Gpc": dl_max_log,
        "z_grid": _rl(z_disp, 5),
        "z_max_by_h": _rl(z_max, 6),
        "p_det_of_z": p_det_z,
        "dl_of_z_Gpc": dl_of_z,
        "visible_volume_integrand": integrand,
        "integrand_units": "Mpc^3 sr^-1 per unit z",
        "D_h_production": _rl(d_h, 8),
        "D_h_pooled_recompute": _rl(d_pool, 8),
        "D_level_ratio_pooled_over_production": {
            "min": _r(float(ratio.min()), 5),
            "max": _r(float(ratio.max()), 5),
            "shape_spread": shape_spread,
            "note": (
                "Production evaluates D(h) per ecliptic-latitude band and averages over "
                "equal-area sky pixels (bayesian_statistics.py:1074-1088, sky-aware path); "
                "this recomputation uses the pooled isotropic survival. The levels differ "
                "by this ratio; across the whole h-grid the ratio itself varies by only "
                "shape_spread, so the SHAPE is the same object. The widget plots the "
                "recomputed integrand as a shape and quotes the production D(h) as the number."
            ),
        },
        "D_fall_across_grid": {
            "production_ratio_060_over_086": _r(float(d_h[0] / d_h[-1]), 5),
            "pooled_ratio_060_over_086": _r(float(d_pool[0] / d_pool[-1]), 5),
        },
        "pool": {
            "dir": str(POOL_REL),
            "n_used_by_survival": int(dp._n_inj),  # noqa: SLF001 — provenance readout
            "n_sky_bands": int(dp._n_sky_bands),  # noqa: SLF001
            "z_resolved": bool(dp.z_resolved),
            **census,
        },
        "event889": {
            "index": EVENT_889,
            "d_L_Gpc": _r(dl889, 6),
            "p_det_at_own_distance": _r(p_det_889, 4),
        },
        "source": {
            "estimator": "master_thesis_code/bayesian_inference/simulation_detection_probability.py",
            "run_log": str(RUN_LOG_REL),
        },
    }


def main() -> None:
    for rel in (POSTERIORS_REL, CRB_REL, RUN_LOG_REL):
        if not (REPO_ROOT / rel).exists():
            msg = f"Required tracked artifact missing: {rel}"
            raise FileNotFoundError(msg)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    denom = build_denominator_payload()
    with OUT_DENOM.open("w") as f:
        json.dump(denom, f, separators=(",", ":"))
    print(f"Wrote {OUT_DENOM} ({OUT_DENOM.stat().st_size / 1024:.1f} KB)")
    print(
        f"  {denom['n_events']}/{denom['n_events_total']} events · "
        f"MAP on={denom['summary']['on']['map']} off={denom['summary']['off']['map']} · "
        f"mean on={denom['summary']['on']['mean']} off={denom['summary']['off']['mean']} · "
        f"readout gate PASS"
    )

    pool_dir = _pool_dir()
    if pool_dir is None:
        if OUT_HORIZON.exists():
            print(
                "NOTICE: injection pool not found (untracked artifact — expected only in "
                "the main checkout's working tree). Keeping the committed "
                f"{OUT_HORIZON.name} unchanged."
            )
            return
        msg = (
            "Injection pool not found and no committed ch04_horizon.json to keep. "
            f"Expected at <repo>/{POOL_REL} or ../MasterThesisCode/{POOL_REL}."
        )
        raise FileNotFoundError(msg)

    horizon = build_horizon_payload(pool_dir)
    with OUT_HORIZON.open("w") as f:
        json.dump(horizon, f, separators=(",", ":"))
    print(f"Wrote {OUT_HORIZON} ({OUT_HORIZON.stat().st_size / 1024:.1f} KB)")
    print(
        f"  pool {horizon['pool']['n_data_rows']} data rows in "
        f"{horizon['pool']['n_files']} files (+headers = "
        f"{horizon['pool']['n_lines_with_headers']}), strata {horizon['pool']['strata']}, "
        f"survival uses {horizon['pool']['n_used_by_survival']}"
    )
    print(
        f"  dl_max h-invariance {horizon['dl_max_Gpc_by_h']} · survival max|diff| "
        f"{horizon['survival_h_invariance_max_abs_diff']} · D pooled/production "
        f"{horizon['D_level_ratio_pooled_over_production']['min']}-"
        f"{horizon['D_level_ratio_pooled_over_production']['max']}"
    )


if __name__ == "__main__":
    main()
