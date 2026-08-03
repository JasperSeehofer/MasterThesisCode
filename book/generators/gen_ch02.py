"""Generator for Chapter 2 — "Bayes, Once and For All".

Produces the three data files behind the chapter's interactives and its
quantitative prose.  Everything is read from committed campaign-#51/#53
artifacts; there is no RNG except one fixed-seed permutation used as the
"random stacking order" (``ORDER_SEED``), so the output is bit-reproducible.

``book/site/data/ch02_stacker.json``   (I2.1 "The Event Stacker")
    Cumulative log-likelihood sums  ``S_N(h) = sum_{i<=N} ln L_i(h)``  for two
    stacking orders (a fixed random permutation, and loudest-SNR-first) in two
    venues:

      idealized  campaign #51, ``run_seed61000/zoom/posteriors``      (41-point
                 1e-4 grid over [0.728, 0.732] — the only grid on which this
                 run's peak is resolved at all: sigma_h = 3.0e-4)
      realistic  campaign #53, ``seed61000/real_r1/posteriors``       (the
                 production 41-point non-uniform grid over [0.60, 0.86])

    plus the idealized run's combined posterior on the *production* grid, which
    is the delta-spike that motivates the zoom hook (pedagogy Q2.4).

    The browser only exponentiates and trapezoid-normalises; the combination
    rule is the production one (``combine_log_space``: a plain product of
    per-event likelihoods, no beta(h)^N — Loredo 2004; Mandel, Farr & Gair 2019
    arXiv:1809.02063 sec. 3).

``book/site/data/ch02_information.json``  (the lurch, quantified)
    Per-event information budget under the project's OWN statistic, reused
    verbatim from ``realistic_20260729/score_realistic.py:14-21`` (which in turn
    reuses ``idealization_audit/audit_information_decomposition.py``):

        curv_k = ln(L_k(0.73)/L_k(0.725)) + ln(L_k(0.73)/L_k(0.735)),  dh = 0.005
        implied sigma_h = dh / sqrt(sum_k curv_k)

    This is the metric under which ``IDEALIZED_BASELINE_READOUT.md:42-47``'s
    "76 of 1588 carry 100%, the 3 loudest carry 46%" reproduces exactly; the
    generator asserts it (see GATES).  The realistic-venue shares are emitted as
    SIGNED SUMS ONLY with ``quotable_ratios: false`` — ``REALISTIC_READOUT.md``
    sec. 4 forbids quoting the ratios ("cancellation-dominated ... dark share
    reaches 140% and one run's golden share goes to -159%").

``book/site/data/ch02_runs.json``   (the bias / scatter / coverage anchor)
    The ten realistic runs (2 truth seeds x 5 observation realizations) with
    MAP, mean, sigma_h and the pull ``(MAP - h_true)/sigma_h`` recomputed from
    each run's ``combined_posterior.json`` with ``score_realistic.py``'s own
    moment code, gated against the published table in ``REALISTIC_READOUT.md``
    sec. 1.  Used as the *real* anchor beside I2.2's toy trainer, so the toy is
    never mistaken for data.

GATES (the generator stops rather than shipping a silently-different number)
--------------------------------------------------------------------------
 G1  idealized golden share  == 0.46 +/- 0.005            (readout: "46 %")
 G2  idealized in-catalogue share  == 1.00 +/- 0.02       (readout: "100 %")
 G3  idealized implied sigma_h in [2.9e-4, 3.4e-4]        (readout: 0.00030)
 G4  realistic r1 combined MAP == 0.740, mean == 0.7321   (REALISTIC_READOUT sec. 1)
 G5  all ten runs' recomputed MAP == the published MAP column
 G6  the three venues carry the identical 1588-event index set
 G7  r1's signed/absolute curvature ratio == the 52 % printed in sec. 4
     (the readout's 62 % is the ENSEMBLE figure, mean 0.076 / mean 0.123 --
     expert-A review M5; the page now prints both, each with its scope)
 G8  the Ch 3 census figures quoted in Q2.5 still match the census Ch 3
     ships (``ch03_candidates.json``).  Advisory when that file is absent
     (cold clone); a hard failure when it is present and has drifted --
     this gate exists because ch02 asserted "tens of thousands" for three
     screens against a measured median of six (mara BLOCKER-1).

FLAGS
-----
``book/design/flags/ch02_FLAGS.md`` — F-ch02-1 (RESOLVED 2026-07-31 by author
mandate: sigma_dL/d_L = 8.98e-4 is the spec value book-wide; the retired 8.0e-5
was the absolute sigma_dL in Gpc under a fractional label, and ships only inside
this file's erratum block), F-ch02-2 (the "46 %" is metric-dependent and the
metric is now pinned), F-ch02-3 (realistic shares are not quotable).

Run as::

    /home/jasper/Repositories/MasterThesisCode/.venv/bin/python \\
        book/generators/gen_ch02.py
"""

from __future__ import annotations

import json
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

# --- repo-relative artifact paths (BOOK_DESIGN sec. 4.2 rule 7) -------------
CAMPAIGN_REL = Path("results/campaign51_20260728")
REALISTIC_REL = CAMPAIGN_REL / "realistic_20260729"
SEED_REL = REALISTIC_REL / "seed61000"
CRB_REL = SEED_REL / "prepared_cramer_rao_bounds.csv"

# sec. 4.2 rule 1 — the canonical directories.  seed 61000's IDEALIZED baseline
# is `posteriors_fixed` (plain `posteriors/` is the stale pre-ec09ed0 backup);
# the realistic per-realization dirs are canonical as `posteriors/`.
IDEAL_PROD_REL = CAMPAIGN_REL / "run_seed61000" / "posteriors_fixed"
IDEAL_ZOOM_REL = CAMPAIGN_REL / "run_seed61000" / "zoom" / "posteriors"
REAL_R1_REL = SEED_REL / "real_r1" / "posteriors"

OUT_DIR = Path(__file__).resolve().parent.parent / "site" / "data"
OUT_STACKER = OUT_DIR / "ch02_stacker.json"
OUT_INFO = OUT_DIR / "ch02_information.json"
OUT_RUNS = OUT_DIR / "ch02_runs.json"

EVENT_889 = 889  # the book's running example (pedagogy beat B4)

ORDER_SEED = 20260731  # the one and only RNG use; fixed => reproducible
DH_CURV = 0.005  # the curvature metric's step (score_realistic.py:35)

# Display hygiene: the shipped log arrays are clipped this far below their own
# maximum.  exp(-60) ~ 9e-27 — far below anything a plot or a table can show,
# and it keeps the JSON small.  Every SUMMARY number (MAP, mean, sigma, edge
# ratio) is computed at full precision BEFORE the clip.
LOG_CLIP = 60.0

# Published values this generator must reproduce, or stop.
READOUT_R1_MAP = 0.740
READOUT_R1_MEAN = 0.7321
READOUT_MAPS = {  # REALISTIC_READOUT.md sec. 1, the "MAP h" column
    (61000, 1): 0.740, (61000, 2): 0.725, (61000, 3): 0.730,
    (61000, 4): 0.725, (61000, 5): 0.740,
    (62000, 1): 0.715, (62000, 2): 0.700, (62000, 3): 0.710,
    (62000, 4): 0.710, (62000, 5): 0.710,
}
# The r1 signed/absolute curvature ratio as PRINTED on the page (sec. 4).
# Expert-A M5: the page used to print the readout's ensemble 62 % beside r1's
# own pair, which does not divide.  Both now ship, each scoped.
PAGE_R1_SIGNED_FRACTION = 0.52  # r1: 0.0851 / 0.1650
READOUT_ENSEMBLE_SIGNED_FRACTION = 0.62  # REALISTIC_READOUT sec. 4: 0.076 / 0.123

# The Ch 3 census figures this chapter quotes in Q2.5 (worklist §C-ch02 P0,
# consumed from ch03's REGENERATED census at the production radius n_sigma =
# 1.5).  Gate G8 re-reads ch03_candidates.json and refuses to ship if any of
# them has drifted.
CH03_CENSUS_QUOTED = {
    "median_in_ball": 888,
    "median_after_window": 6,
    "p95_after_window": 2725,
    "max_after_window": 245334,
    "n_zero_candidate": 607,
    "n_events": 1590,
    "event889_n_cand": 2,
}

# REALISTIC_READOUT.md sec. 6 — the 2D channel row, carried verbatim as a
# recorded measurement (this chapter never recomputes a 2D number; RATIFY-M6
# designates the 2D pairing a CANDIDATE, see BOOK_SOURCES_MAP sec. 7 item 18).
# DISPLAY SCOPE: Ch 8 and later.  Ch 2 prints the phenomenon only -- the
# magnitude is Ch 8's reveal (REVISION_WORKLIST §D4; pedagogy B1 / M9).
TWOD_RECORDED = {
    "map_range": [0.780, 0.820],
    "pull_range": [3.4, 4.5],
    "pull_mean": 4.04,
    "n_runs_pull_gt_2": 10,
    "n_runs": 10,
    "source": "REALISTIC_READOUT.md §6",
    "badge": "CANDIDATE",
    "badge_note": "RATIFY-M6 designates the 2D pairing necessary, not established sufficient",
    "display_scope": (
        "Ch 8 and later. Chapter 2 names the phenomenon (a coherent tilt makes the pull "
        "grow as sqrt(N)) and prints none of these values: they are Ch 8's reveal "
        "(REVISION_WORKLIST.md §D4)."
    ),
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _r(x: Any, sig: int = 8) -> float:
    """Round to `sig` significant digits (JSON size hygiene)."""
    v = float(x)
    if v == 0.0 or not np.isfinite(v):
        return v
    return float(f"%.{sig}g" % v)


def _rl(a: Any, sig: int = 8) -> list[float]:
    return [_r(v, sig) for v in np.asarray(a, dtype=np.float64).ravel()]


def _rd(a: Any, nd: int = 4) -> list[float]:
    """Round to `nd` decimals — used for the clipped log arrays."""
    return [round(float(v), nd) for v in np.asarray(a, dtype=np.float64).ravel()]


def _moments(h: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """MAP / mean / sigma / 68% interval on the NON-UNIFORM h-grid.

    Trapezoid over the actual node spacing, exactly as
    ``score_realistic.py:posterior_moments`` does it (the grid is 0.01-spaced on
    [0.60, 0.65] and [0.80, 0.86] and 0.005-spaced between; a uniform rule would
    be wrong at the seams).
    """
    p = np.where(np.isfinite(p), p, 0.0)
    norm = float(np.trapezoid(p, h))
    p = p / norm
    mean = float(np.trapezoid(p * h, h))
    var = float(np.trapezoid(p * (h - mean) ** 2, h))
    cdf = np.concatenate([[0.0], np.cumsum(np.diff(h) * 0.5 * (p[1:] + p[:-1]))])
    lo, hi = (float(np.interp(q, cdf, h)) for q in (0.16, 0.84))
    return {
        "map": float(h[int(np.argmax(p))]),
        "mean": mean,
        "sigma": float(np.sqrt(var)),
        "q16": lo,
        "q84": hi,
        "edge_over_peak": float(max(p[0], p[-1]) / p.max()),
    }


def _load_log(rel: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(h_grid, per-event ln L array, detection indices) for a posteriors dir."""
    h_values, event_likelihoods = load_posterior_jsons(REPO_ROOT / rel)
    arr, indices = build_likelihood_array(h_values, event_likelihoods)
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        msg = f"{rel}: non-positive or missing likelihood entries — refusing to log()"
        raise ValueError(msg)
    return np.asarray(h_values, dtype=np.float64), np.log(arr), np.asarray(indices)


# ---------------------------------------------------------------------------
# the project's own per-event information statistic
# ---------------------------------------------------------------------------
def _curvature(rel: Path) -> dict[int, float]:
    """Per-event 3-point ln-likelihood curvature about h = 0.73.

    Reused verbatim from ``realistic_20260729/score_realistic.py:57-62`` — this
    generator does not define an information measure of its own.
    """

    def _at(tag: str) -> dict[int, float]:
        with open(REPO_ROOT / rel / f"h_0_{tag}.json") as fh:
            j = json.load(fh)
        return {int(k): (j[k][0] if isinstance(j[k], list) else j[k]) for k in j if k.isdigit()}

    a, b, c = (_at(t) for t in ("725", "73", "735"))
    common = [k for k in b if k in a and k in c and a[k] > 0 and b[k] > 0 and c[k] > 0]
    return {k: float(np.log(b[k] / a[k]) + np.log(b[k] / c[k])) for k in common}


# ---------------------------------------------------------------------------
# I2.1 — the Event Stacker
# ---------------------------------------------------------------------------
def _n_ladder(n_total: int, pin: list[int]) -> list[int]:
    """A stacking ladder: every N up to 24, then coarsening — plus a dense
    window around every pinned rank (where a golden event enters), because the
    lurch is a one-event step and a coarse ladder would step straight over it."""
    steps: set[int] = set(range(1, 25))
    steps.update(range(25, 101, 5))
    steps.update(range(120, 401, 40))
    steps.update(range(450, n_total, 150))
    steps.add(n_total)
    for r in pin:
        steps.update(k for k in (r - 2, r - 1, r, r + 1, r + 2) if 1 <= k <= n_total)
    return sorted(steps)


def _stack(
    h: np.ndarray, log_l: np.ndarray, order: np.ndarray, ladder: list[int]
) -> dict[str, Any]:
    """Cumulative log-sums along `order`, sampled at `ladder`."""
    logs: list[list[float]] = []
    summary: dict[str, list[float]] = {"map": [], "mean": [], "sigma": [], "edge": []}
    cum = np.zeros(len(h))
    prev = 0
    for n in ladder:
        cum = cum + log_l[order[prev:n]].sum(axis=0)
        prev = n
        shifted = cum - cum.max()
        m = _moments(h, np.exp(np.maximum(shifted, -700.0)))
        for key, val in (("map", m["map"]), ("mean", m["mean"]),
                         ("sigma", m["sigma"]), ("edge", m["edge_over_peak"])):
            summary[key].append(val)
        logs.append(_rd(np.maximum(shifted, -LOG_CLIP)))
    return {
        "log": logs,
        "map": _rl(summary["map"], 6),
        "mean": _rl(summary["mean"], 8),
        "sigma": _rl(summary["sigma"], 6),
        "edge_over_peak": _rl(summary["edge"], 4),
    }


def build_stacker() -> dict[str, Any]:
    h_zoom, log_zoom, idx_zoom = _load_log(IDEAL_ZOOM_REL)
    h_prod, log_real, idx_real = _load_log(REAL_R1_REL)
    h_prod_i, log_ideal_prod, idx_ideal = _load_log(IDEAL_PROD_REL)

    # G6 — the three venues must be the same 1588 events, or "the same events,
    # two universes" is not the comparison the chapter claims to be making.
    if not (np.array_equal(idx_zoom, idx_real) and np.array_equal(idx_zoom, idx_ideal)):
        msg = "G6 FAILED: the idealized and realistic venues carry different event sets"
        raise ValueError(msg)
    if not np.allclose(h_prod, h_prod_i):
        msg = "G6 FAILED: idealized and realistic production h-grids differ"
        raise ValueError(msg)

    n_events = len(idx_zoom)
    crb = pd.read_csv(REPO_ROOT / CRB_REL)
    snr = crb["SNR"].to_numpy(dtype=np.float64)[idx_zoom]

    # The golden set, defined on the IDEALIZED baseline exactly as
    # score_realistic.py:golden_events does (top 3 by 3-point curvature).
    curv_ideal = _curvature(IDEAL_PROD_REL)
    golden = [k for k, _ in sorted(curv_ideal.items(), key=lambda kv: -kv[1])[:3]]

    rng = np.random.default_rng(ORDER_SEED)
    order_random = rng.permutation(n_events)
    order_snr = np.argsort(-snr, kind="stable")

    pos_in = {int(idx_zoom[p]): i + 1 for i, p in enumerate(order_random)}
    ranks_random = [pos_in[g] for g in golden]
    pos_snr = {int(idx_zoom[p]): i + 1 for i, p in enumerate(order_snr)}
    ranks_snr = [pos_snr[g] for g in golden]

    ladder = _n_ladder(n_events, sorted(ranks_random) + sorted(ranks_snr))

    venues = {
        "idealized": {
            "grid": "zoom",
            "label": "idealized #51 baseline · zoom grid",
            "run": "campaign51 run_seed61000/zoom/posteriors",
            "orders": {
                "random": _stack(h_zoom, log_zoom, order_random, ladder),
                "snr": _stack(h_zoom, log_zoom, order_snr, ladder),
            },
        },
        "realistic": {
            "grid": "prod",
            "label": "realistic #53 · seed61000/real_r1",
            "run": "campaign51_20260728/realistic_20260729/seed61000/real_r1/posteriors",
            "orders": {
                "random": _stack(h_prod, log_real, order_random, ladder),
                "snr": _stack(h_prod, log_real, order_snr, ladder),
            },
        },
    }

    # G4 — the realistic full-stack must reproduce the published row.
    full = venues["realistic"]["orders"]["random"]
    if abs(full["map"][-1] - READOUT_R1_MAP) > 1e-9:
        msg = f"G4 FAILED: r1 MAP {full['map'][-1]} != published {READOUT_R1_MAP}"
        raise ValueError(msg)
    if abs(full["mean"][-1] - READOUT_R1_MEAN) > 5e-5:
        msg = f"G4 FAILED: r1 mean {full['mean'][-1]} != published {READOUT_R1_MEAN}"
        raise ValueError(msg)
    # Order-independence of the endpoint is the chapter's own claim about
    # log-additivity; assert it rather than assert it in prose only.
    if abs(full["map"][-1] - venues["realistic"]["orders"]["snr"]["map"][-1]) > 1e-12:
        msg = "endpoint depends on stacking order — log-additivity violated"
        raise ValueError(msg)

    # The idealized run on the PRODUCTION grid: the delta spike of Q2.4.
    spike_log = log_ideal_prod.sum(axis=0)
    spike_log = spike_log - spike_log.max()
    spike_mom = _moments(h_prod, np.exp(np.maximum(spike_log, -700.0)))

    e889 = int(np.where(idx_zoom == EVENT_889)[0][0])
    row889 = crb.iloc[EVENT_889]
    sigma_dl_gpc = float(np.sqrt(row889["delta_luminosity_distance_delta_luminosity_distance"]))
    dl_gpc = float(row889["luminosity_distance"])

    return {
        "meta": {
            "chapter": 2,
            "generator": "book/generators/gen_ch02.py",
            "combination_rule": (
                "S_N(h) = sum_{i<=N} ln L_i(h); plain product of per-event "
                "likelihoods, no beta(h)^N — combine_log_space(), "
                "posterior_combination.py:284-330 (Loredo 2004; Mandel, Farr & "
                "Gair 2019 arXiv:1809.02063 §3)"
            ),
            "log_clip_below_max": LOG_CLIP,
            "order_seed": ORDER_SEED,
            "grid_note": (
                "the production grid is NON-uniform (0.01 on [0.60,0.65] and "
                "[0.80,0.86], 0.005 between); every integral here is a "
                "trapezoid over the actual node spacing and nothing is "
                "differentiated across a seam"
            ),
        },
        "h_true": _r(H_TRUE, 6),
        "n_events": n_events,
        "grids": {"zoom": _rl(h_zoom, 8), "prod": _rl(h_prod, 6)},
        "n_steps": ladder,
        "golden": {
            "indices": [int(g) for g in golden],
            "snr": _rl([float(crb["SNR"].iloc[g]) for g in golden], 6),
            "rank_random": ranks_random,
            "rank_snr": ranks_snr,
        },
        "venues": venues,
        "idealized_production_grid": {
            "log": _rd(np.maximum(spike_log, -LOG_CLIP)),
            "map": _r(spike_mom["map"], 6),
            "mean": _r(spike_mom["mean"], 8),
            "sigma": _r(spike_mom["sigma"], 4),
            "edge_over_peak": _r(spike_mom["edge_over_peak"], 4),
            "note": (
                "the same 1588 events on the production grid: sigma_h = 3.0e-4 is "
                "~15x narrower than the grid's 0.005 step, so the posterior is a "
                "single-node spike (IDEALIZED_BASELINE_READOUT.md:36-39)"
            ),
        },
        "event889": {
            "index": EVENT_889,
            "snr": _r(row889["SNR"], 7),
            "d_L_Gpc": _r(dl_gpc, 7),
            "d_L_Mpc": _r(dl_gpc * 1000.0, 7),
            "sigma_dL_Gpc": _r(sigma_dl_gpc, 6),
            "sigma_dL_Mpc": _r(sigma_dl_gpc * 1000.0, 6),
            "sigma_dL_over_dL": _r(sigma_dl_gpc / dl_gpc, 6),
            "sigma_dL_over_dL_times_snr": _r(sigma_dl_gpc / dl_gpc * float(row889["SNR"]), 6),
            "erratum": {
                "status": "RESOLVED 2026-07-31 — sigma_dL/d_L = 8.98e-4 is the spec value",
                "note": (
                    "Erratum: the spec card carried sigma_dL/dL = 8.0e-5 — that is the "
                    "absolute sigma_dL in Gpc under a fractional label. Corrected "
                    "book-wide 2026-07-31; record: ch01 flag F1 / BUILD_REPORT §5.1 "
                    "item 1. The retired figure is kept here only as history."
                ),
                "retired_spec_fraction": 8.0e-5,
            },
            "flag": "book/design/flags/ch02_FLAGS.md F-ch02-1 (resolved)",
            "host_galaxy_index": int(row889["host_galaxy_index"]),
            "L_zoom": _rl(np.exp(log_zoom[e889]), 6),
            "L_prod_realistic": _rl(np.exp(log_real[e889]), 6),
            "L_prod_idealized": _rl(np.exp(log_ideal_prod[e889]), 6),
            "map_zoom": _r(h_zoom[int(np.argmax(log_zoom[e889]))], 6),
            "map_prod_realistic": _r(h_prod[int(np.argmax(log_real[e889]))], 6),
        },
        "snr_median_frac_times_snr": _r(
            float(
                np.median(
                    np.sqrt(crb["delta_luminosity_distance_delta_luminosity_distance"].to_numpy())
                    / crb["luminosity_distance"].to_numpy()
                    * crb["SNR"].to_numpy()
                )
            ),
            6,
        ),
    }


# ---------------------------------------------------------------------------
# G8 — the Ch 3 census figures this chapter quotes
# ---------------------------------------------------------------------------
def check_ch03_census() -> dict[str, Any]:
    """Re-read Ch 3's shipped census and gate the figures Q2.5 prints.

    Chapter 2 asserted "tens of thousands of candidates" three times, and
    graded a self-check answer on it, against a Ch 3 census that measures a
    median of six (mara BLOCKER-1; the search radius itself was wrong, see
    worklist §A-D2).  Ch 2 now quotes Ch 3's numbers, so it must break when
    they move.  ``ch03_candidates.json`` is Ch 3's own output; if it is not
    built yet (cold clone, gen_ch02 runs before gen_ch03) the check is an
    advisory rather than a failure.
    """
    path = OUT_DIR / "ch03_candidates.json"
    quoted = dict(CH03_CENSUS_QUOTED)
    if not path.exists():
        print(f"  G8 SKIPPED: {path.name} not built yet — Q2.5's census figures unverified")
        return {"gated": False, "quoted": quoted, "why": "ch03_candidates.json absent"}

    with open(path) as fh:
        c = json.load(fh)
    measured = {
        "median_in_ball": int(c["n_ball"]["percentiles"]["50"]),
        "median_after_window": int(c["n_cand"]["percentiles"]["50"]),
        "p95_after_window": int(round(c["n_cand"]["percentiles"]["95"])),
        "max_after_window": int(c["n_cand"]["max"]),
        "n_zero_candidate": int(c["n_cand"]["n_zero"]),
        "n_events": int(c["n_events"]),
        "event889_n_cand": int(c["featured"]["889"]["n_cand"]),
    }
    drift = {k: (quoted[k], measured[k]) for k in quoted if quoted[k] != measured[k]}
    if drift:
        msg = (
            "G8 FAILED: ch02 quotes Ch 3 census figures that Ch 3 no longer measures "
            f"(quoted, measured): {drift}. Update Q2.5 and CH03_CENSUS_QUOTED together."
        )
        raise ValueError(msg)
    return {
        "gated": True,
        "source": "book/site/data/ch03_candidates.json (Ch 3's own census)",
        "ball_rule": c["meta"]["ball_rule"],
        "quoted": quoted,
    }


# ---------------------------------------------------------------------------
# the information budget
# ---------------------------------------------------------------------------
def build_information() -> dict[str, Any]:
    crb = pd.read_csv(REPO_ROOT / CRB_REL)
    in_cat = set(int(i) for i in crb.index[crb["host_galaxy_index"] >= 0])

    curv = _curvature(IDEAL_PROD_REL)
    keys = sorted(curv)
    vals = np.array([curv[k] for k in keys])
    total = float(vals.sum())
    golden = [k for k, _ in sorted(curv.items(), key=lambda kv: -kv[1])[:3]]
    incat_total = float(sum(v for k, v in curv.items() if k in in_cat))
    golden_sum = float(sum(curv[g] for g in golden))
    golden_share = golden_sum / total
    golden_share_incat = golden_sum / incat_total
    incat_share = incat_total / total
    sigma_implied = DH_CURV / np.sqrt(total)

    # --- GATES against IDEALIZED_BASELINE_READOUT.md:42-47 -----------------
    # The readout's "46 %" is the share of the IN-CATALOGUE budget (0.4641);
    # the share of the SIGNED TOTAL is 0.4700.  They differ because the dark
    # class contributes -1.3%, so the two denominators differ by 1.3%.  Both
    # are emitted and both are printed on the page — see F-ch02-2.
    if abs(golden_share_incat - 0.46) > 0.005:
        msg = (
            f"G1 FAILED: golden share of the in-catalogue budget "
            f"{golden_share_incat:.4f} != 0.46 (readout '46 %')"
        )
        raise ValueError(msg)
    if not (0.46 <= golden_share <= 0.48):
        msg = f"G1b FAILED: golden share of the signed total {golden_share:.4f} outside [0.46, 0.48]"
        raise ValueError(msg)
    if abs(incat_share - 1.0) > 0.02:
        msg = f"G2 FAILED: in-catalogue share {incat_share:.4f} != 1.00 (readout '100 %')"
        raise ValueError(msg)
    if not (2.9e-4 <= sigma_implied <= 3.4e-4):
        msg = f"G3 FAILED: implied sigma_h {sigma_implied:.3e} outside [2.9e-4, 3.4e-4]"
        raise ValueError(msg)

    order = np.argsort(-vals)
    cum = np.cumsum(vals[order]) / total
    ks = sorted(set(list(range(1, 41)) + list(range(45, 121, 5)) + list(range(140, 401, 20))
                    + list(range(450, len(vals) + 1, 50)) + [len(vals)]))
    positives = vals[vals > 0]
    participation = float(positives.sum() ** 2 / np.sum(positives**2))

    # --- the realistic venue: SIGNED SUMS ONLY (REALISTIC_READOUT sec. 4) --
    curv_r1 = _curvature(REAL_R1_REL)
    v_r1 = np.array([curv_r1[k] for k in sorted(curv_r1)])
    incat_signed = float(sum(v for k, v in curv_r1.items() if k in in_cat))
    total_r1 = float(v_r1.sum())
    abs_mass_r1 = float(np.abs(v_r1).sum())
    signed_fraction_r1 = total_r1 / abs_mass_r1

    # G7 — the page prints "the signed total is only 52% of the absolute
    # curvature mass" beside r1's own two numbers.  A reader who does the
    # division must land on the printed figure (expert-A M5: it used to print
    # the readout's ENSEMBLE 62% beside r1's pair).
    if abs(signed_fraction_r1 - PAGE_R1_SIGNED_FRACTION) > 0.005:
        msg = (
            f"G7 FAILED: r1 signed/absolute curvature ratio {signed_fraction_r1:.4f} "
            f"!= the {PAGE_R1_SIGNED_FRACTION:.2f} printed in ch02 §4"
        )
        raise ValueError(msg)

    return {
        "meta": {
            "generator": "book/generators/gen_ch02.py",
            "statistic": (
                "curv_k = ln(L_k(0.73)/L_k(0.725)) + ln(L_k(0.73)/L_k(0.735)), dh = 0.005; "
                "implied sigma_h = dh / sqrt(sum_k curv_k)"
            ),
            "statistic_source": (
                "results/campaign51_20260728/realistic_20260729/score_realistic.py:14-21, "
                "reused verbatim from idealization_audit/audit_information_decomposition.py"
            ),
            "flag": "book/design/flags/ch02_FLAGS.md F-ch02-2",
        },
        "ch03_census_quoted": check_ch03_census(),
        "idealized": {
            "run": "campaign51 run_seed61000/posteriors_fixed (canonical; NOT posteriors/)",
            "n_events": len(vals),
            "n_in_catalogue": len(in_cat),
            "total_curvature": _r(total, 7),
            "sigma_h_implied": _r(sigma_implied, 5),
            "golden_indices": [int(g) for g in golden],
            "golden_share_of_signed_total": _r(golden_share, 5),
            "golden_share_of_in_catalogue": _r(golden_share_incat, 5),
            "golden_share_quoted_by_readout": "of_in_catalogue",
            "in_catalogue_curvature": _r(incat_total, 7),
            "in_catalogue_share": _r(incat_share, 5),
            "dark_share": _r(1.0 - incat_share, 5),
            "n_positive": int((vals > 0).sum()),
            "n_negative": int((vals < 0).sum()),
            "participation_ratio": _r(participation, 5),
            "cumulative": {"k": ks, "share": _rl([cum[k - 1] for k in ks], 6)},
            "top_curvatures": _rl(vals[order][:24], 6),
            "top_indices": [int(keys[i]) for i in order[:24]],
        },
        "realistic_r1": {
            "run": "seed61000/real_r1/posteriors",
            "quotable_ratios": False,
            "why_not": (
                "REALISTIC_READOUT.md §4: the signed total is only ~62% of the absolute "
                "curvature mass, so shares are cancellation-dominated — 'dark share' "
                "reaches 140% and one run's golden share goes to -159%. Quote the signed "
                "sums, never the ratios."
            ),
            "signed_total": _r(total_r1, 5),
            "absolute_mass": _r(abs_mass_r1, 5),
            "signed_over_absolute": _r(signed_fraction_r1, 4),
            "signed_over_absolute_printed": PAGE_R1_SIGNED_FRACTION,
            "signed_over_absolute_readout_ensemble": READOUT_ENSEMBLE_SIGNED_FRACTION,
            "signed_over_absolute_scope": (
                "0.52 is r1's own ratio (0.0851/0.1650); 0.62 is REALISTIC_READOUT §4's "
                "ENSEMBLE figure (mean 0.076 / mean 0.123 over the ten runs). The page "
                "prints both, each labelled with its scope (expert-A review M5)."
            ),
            "in_catalogue_signed": _r(incat_signed, 5),
            "dark_signed": _r(total_r1 - incat_signed, 5),
            "sigma_h_implied": _r(DH_CURV / np.sqrt(total_r1), 5) if total_r1 > 0 else None,
            "recorded_P4": (
                "the 3 golden events retained 0.045% of their idealized curvature "
                "(REALISTIC_READOUT.md §2, P4 — PASS by ~3 orders of magnitude)"
            ),
        },
    }


# ---------------------------------------------------------------------------
# the ten realistic runs (bias / scatter / coverage anchor)
# ---------------------------------------------------------------------------
def build_runs() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in (61000, 62000):
        for r in (1, 2, 3, 4, 5):
            path = REALISTIC_REL / f"seed{seed}" / f"real_r{r}" / "posteriors" / "combined_posterior.json"
            with open(REPO_ROOT / path) as fh:
                d = json.load(fh)
            h = np.asarray(d["h_values"], dtype=np.float64)
            p = np.asarray(d["posterior"], dtype=np.float64)
            srt = np.argsort(h)
            m = _moments(h[srt], p[srt])
            pull = (m["map"] - H_TRUE) / m["sigma"]
            published = READOUT_MAPS[(seed, r)]
            if abs(m["map"] - published) > 1e-9:
                msg = (
                    f"G5 FAILED: seed{seed}/r{r} recomputed MAP {m['map']} != "
                    f"published {published} (REALISTIC_READOUT.md §1)"
                )
                raise ValueError(msg)
            rows.append({
                "seed": seed,
                "realization": r,
                "map": _r(m["map"], 5),
                "mean": _r(m["mean"], 6),
                "sigma_h": _r(m["sigma"], 5),
                "sigma_H0": _r(m["sigma"] * 100.0, 4),
                "pull": _r(pull, 4),
                "q16": _r(m["q16"], 5),
                "q84": _r(m["q84"], 5),
                "n_events_used": int(d["n_events_used"]),
            })

    pulls = np.array([row["pull"] for row in rows])
    maps = np.array([row["map"] for row in rows])
    covered = int(sum(1 for row in rows if row["q16"] <= H_TRUE <= row["q84"]))
    return {
        "meta": {
            "generator": "book/generators/gen_ch02.py",
            "source": "campaign #53 realistic runs; gated against REALISTIC_READOUT.md §1",
            "pull_definition": "(MAP - h_true) / sigma_h  (score_realistic.py:152)",
        },
        "h_true": _r(H_TRUE, 6),
        "runs": rows,
        "summary_1d": {
            "map_min": _r(maps.min(), 5),
            "map_max": _r(maps.max(), 5),
            "map_mean": _r(maps.mean(), 6),
            "pull_mean": _r(pulls.mean(), 4),
            "pull_sd": _r(pulls.std(ddof=1), 4),
            "max_abs_pull": _r(np.abs(pulls).max(), 4),
            "n_pull_gt_2": int((np.abs(pulls) > 2).sum()),
            "n_68_intervals_containing_truth": covered,
            "n_runs": len(rows),
            "caveat": (
                "REALISTIC_READOUT.md §3: the ten runs share TWO truth universes, so the "
                "pulls are strongly correlated within a seed — two effective degrees of "
                "freedom, not ten. pull sd = 0.58 must NOT be read as over-conservative sigma."
            ),
        },
        "summary_2d_recorded": TWOD_RECORDED,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, payload in (
        (OUT_STACKER, build_stacker()),
        (OUT_INFO, build_information()),
        (OUT_RUNS, build_runs()),
    ):
        with open(path, "w") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        print(f"wrote {path.relative_to(path.parents[3])}  ({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
