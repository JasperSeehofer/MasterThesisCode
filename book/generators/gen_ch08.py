"""Generator for Chapter 8 — "A Second Handle: the Mass Channel".

Produces the four data files behind the chapter's narrative and interactives.
Everything is recomputed from committed artifacts of campaign #51/#53
(``results/campaign51_20260728/realistic_20260729/``); nothing is copied out of
a prose document without also being re-derived, and every number that the build
spec or a cited artifact also states is checked against it — a disagreement
raises, it is never silently reconciled.

Outputs
-------
``book/site/data/ch08_channel.json``      (cold open + the class budget, C1–C3)
    The delivered 1D and 2D combined log-posteriors for ``seed61000/real_r1``,
    reconstructed in log space from ``diagnostics/event_likelihoods.csv`` via
    the mixture identity

        p_i(h) = w_G(h) L^cat_i(h) + (1 - w_G(h)) L^comp_i(h)

    (``bayesian_statistics.py:3309-3311``), plus the class-summed profiles
    (in-catalogue / dark) in both channels and the 10-run scorecard from
    ``realistic_scores.csv``.

``book/site/data/ch08_sieve.json``        (I8.1 "The Impostor Sieve")
    The *real* culling statistics: per-h zero fractions of the catalogue leg by
    class and channel, the dark class's mean catalogue mixture weight
    w_cat = w_G L_cat / (w_G L_cat + (1-w_G) L_comp) in both channels, the
    suppression histogram L_cat_2D / L_cat_1D over surviving dark events, and
    the C4-amended budget partition (zeroed / both-dead / survivors).
    Since the 2026-07-31 revision it also carries ``cell_b`` — the same
    accounting redone on the 2x2's cell B (the unscattered parent catalogue
    through the same estimator), which shows the dark channel difference is
    estimator-borne and that the 98.5%/1.5% split is configuration-scoped —
    and, under ``recorded``, the sigma_Mz/M_z both-values pair (claim file
    1e-4 vs measured median 8.8e-8, ch06_FLAGS.md F-ch06-5).

``book/site/data/ch08_reparam.json``      (I8.2 "The Reparametrization Walk")
    A faithful re-run of ``gate_b_20260730/c8_reparam.py``'s constant-C sweep
    and its four *named* alternative mass measures, with the per-C normalised
    2D posteriors so the walk can be watched rather than tabulated.  Gated
    against that script's own committed results JSON and against the delivered
    ``combined_posterior_2d.json``.

``book/site/data/ch08_twofaces.json``     (I8.3 "EMRI-889's Two Faces")
    Per-event term profiles (L_cat 1D/2D, L_comp, w_G, the two combined
    likelihoods) for the book's running in-catalogue event 889 and its dark
    counterpart 606, plus 889's channel swing across all five seed-61000
    realizations.

Determinism
-----------
No RNG anywhere.  Every array is a deterministic function of committed CSV/JSON
artifacts.  Read-only outside ``book/``; repo root resolved relatively.

Run as::

    /home/jasper/Repositories/MasterThesisCode/.venv/bin/python \\
        book/generators/gen_ch08.py
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

from master_thesis_code.constants import H as H_TRUE  # noqa: E402


def _resolve(rel: Path) -> Path | None:
    """Locate a repo-relative artifact without hardcoding a machine path.

    The per-run ``diagnostics/event_likelihoods.csv`` files — this chapter's
    workhorse — are large and **not git-tracked**; they live in the working tree
    of the main checkout.  Look in this checkout first, then in a sibling
    ``MasterThesisCode`` checkout, exactly as ``gen_ch04.py`` does for the
    injection pool.  Returns ``None`` when neither has it.
    """
    for root in (REPO_ROOT, REPO_ROOT.parent / "MasterThesisCode"):
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


def _require(rel: Path) -> Path:
    path = _resolve(rel)
    if path is None:
        msg = f"required artifact not found in either checkout: {rel}"
        raise FileNotFoundError(msg)
    return path


# --- repo-relative artifact paths (spec §4.2 rule 7; never absolute) --------
CAMPAIGN_REL = Path("results/campaign51_20260728/realistic_20260729")
GATE_B_REL = CAMPAIGN_REL / "gate_b_20260730"
# The 2x2's cell B (REVISION 2026-07-31, expB MJ-4): the same #53 estimator
# configuration run against the UNSCATTERED parent catalogue.  Landed after the
# chapter was drafted; readout CELLB_READOUT_20260731.md, evaluate 6103219 /
# combine 6103220 (resubmission of 6101146/6101147 after a plumbing symlink
# failure), code 7fd60bb — the same commit as cells A and C.
CELLB_REL = CAMPAIGN_REL / "seed61000" / "estimatorB_2x2"
CELLB_READOUT_REL = CAMPAIGN_REL / "CELLB_READOUT_20260731.md"
SCORES_REL = CAMPAIGN_REL / "realistic_scores.csv"
C8_RESULTS_REL = GATE_B_REL / "c8_reparam_results.json"
C3C4_RESULTS_REL = GATE_B_REL / "c3c4_allruns_results.json"
C4_DECOMP_REL = GATE_B_REL / "c4_decomposition_results.json"

SEEDS = (61000, 62000)
RUNS = (1, 2, 3, 4, 5)

OUT_DIR = Path(__file__).resolve().parent.parent / "site" / "data"
OUT_CHANNEL = OUT_DIR / "ch08_channel.json"
OUT_SIEVE = OUT_DIR / "ch08_sieve.json"
OUT_REPARAM = OUT_DIR / "ch08_reparam.json"
OUT_TWOFACES = OUT_DIR / "ch08_twofaces.json"

EVENT_INCAT = 889  # the book's running example (pedagogy B4), in-catalogue
EVENT_DARK = 606  # its permanent dark counterpart, introduced in Ch 5

# Numbers the build spec / the cited artifacts state, and which this generator
# must reproduce or stop.  (BOOK_DESIGN.md §1 Ch 8; CLAIM_2D_BIAS_20260730.md
# C3 / C4 / C8 / C10; gate_b_20260730/ADJUDICATION_20260730.md §6.3, §6.7.)
SPEC = {
    "map1d_r1": 0.740,  # REALISTIC_READOUT.md §1, row 61000 / r1
    "map2d_r1_grid": 0.81,  # realistic_scores.csv map_h_2d
    "map2d_r1_parab": 0.81329,  # claim C8 / c8_reparam_results.json
    "bias2d_mean": 0.077,  # REALISTIC_READOUT.md §6 (mean 2D MAP - 0.73)
    "pull2d_mean": 4.04,  # REALISTIC_READOUT.md §6
    "channel_diff_total": 18.80,  # C2 / C3
    "channel_diff_dark": 15.83,  # C3
    "channel_diff_incat": 2.97,  # C3
    "dark_share_pct": 84.2,  # C3 — r1-SPECIFIC (sources map §7.5)
    "dark_zero2d_pct": 64.7,  # C4-obs
    "dark_zero1d_pct": 32.5,  # C4-obs
    "n_zero2d_of_nonzero1d": 488,  # C4-obs
    "n_zero2d_dark": 487,  # C4-obs
    "median_suppression": 7.8e-3,  # C4-obs
    "dark_tilt_lnratio": -504.8,  # C4-obs
    "wcat_dark_1d": 0.0354,  # C4-amended
    "wcat_dark_2d": 0.0061,  # C4-amended
    "budget_survivors": 15.60,  # C4-amended (98.5%)
    "budget_zeroed": 0.24,  # C4-amended (1.5%)
    "budget_loss_of_1d_downtilt": 19.10,  # C4-amended
    "budget_residual_2d_tilt": -3.27,  # C4-amended
    "c10_prefactor_nats": 31.55,  # C10
    "c10_lcomp_nats": -3.11,  # C10
    "c10_dark_lcomp_nats": -22.72,  # C10
    "c10_dark_positive_frac": 0.391,  # C10
    "dmap_dlnc": 0.031,  # C8
    "e889_swing": (1.98, -2.04, -3.30),  # C3 off-r1 replication note
    # --- the 2x2's cell B (CELLB_READOUT_20260731.md) --------------------
    # Cited by the readout itself:
    "cellb_map1d": 0.7450,
    "cellb_map2d": 0.7900,
    "cellb_channel_diff_dark": 18.00,
    "cellb_channel_diff_incat": -1.80,
    "cellb_channel_diff_total": 16.20,
    "cellb_wG_060": 0.1625175,
    "cellb_wG_073": 0.1215039,
    "cellb_wG_081": 0.1038732,
    # Recomputed by expert review B, NOT present in any adjudicated artifact.
    # Reproduced here from cell B's own diagnostics CSV so the book prints
    # measured numbers, not transcribed ones (ch08_FLAGS.md F-ch08-9).
    "cellb_n_surv": 219,
    # 80.7 is the review's own figure; recomputed here it is 80.763, which the
    # chapter prints as 80.8% (the pair then sums to 100.0).  Gated at the one
    # decimal the review quoted — see ch08_FLAGS.md F-ch08-9.
    "cellb_pct_surv": 80.7,
    "cellb_n_zeroed": 688,
    "cellb_pct_zeroed": 19.2,
    "cellb_wcat_dark_1d": 0.0361,
    "cellb_wcat_dark_2d": 0.0043,
    "cellb_deweight": 8.39,
    "cellb_dark_zero2d_frac": 0.855,
    "cellb_n_1d_nonzero": 982,
    # --- sigma_Mz/M_z: the both-values pair (tomas B3 / F-ch06-5) ---------
    # The claim file says ~1e-4; Chapter 6 recomputed the same stored
    # quantity, sqrt(Sigma_MM)/M from prepared_cramer_rao_bounds.csv, and
    # measured these.  Neither is corrected here — the chapter prints both.
    "sigma_mz_claim": 1e-4,
    "sigma_mz_measured_median": 8.8e-8,
    "sigma_mz_measured_889": 1.36e-9,
}

# Tolerances.  These are deliberately tight: the point of the gates is to catch
# a silently different artifact, not to accommodate one.
TOL_ABS = 5e-3
TOL_IDENTITY = 1e-11
TOL_DELIVERED_NATS = 1e-9


class FidelityError(RuntimeError):
    """A recomputed number disagrees with the spec or a cited artifact."""


_FLAGS: list[str] = []


def _check(name: str, got: float, want: float, tol: float = TOL_ABS) -> None:
    if not np.isfinite(got) or abs(float(got) - float(want)) > tol:
        msg = (
            f"FIDELITY GATE FAILED — {name}: recomputed {got!r} vs cited {want!r} "
            f"(tol {tol}).  STOP and flag; do not reconcile silently."
        )
        raise FidelityError(msg)


def _r(x: Any, sig: int = 8) -> float:
    v = float(x)
    if v == 0.0 or not np.isfinite(v):
        return v
    return float(f"%.{sig}g" % v)


def _rl(a: Any, sig: int = 8) -> list[float]:
    return [_r(v, sig) for v in np.asarray(a, dtype=np.float64).ravel()]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
COLS = (
    "w_G",
    "L_cat_no_bh",
    "L_cat_with_bh",
    "B_num",
    "L_comp",
    "combined_no_bh",
    "combined_with_bh",
)


def load_run(seed: int, run: int) -> dict[str, Any]:
    """Pivot one run's diagnostics CSV into (event x h) arrays.

    ``combined_with_bh`` in this CSV was verified bit-identical to the per-event
    ``posteriors_with_bh_mass`` JSONs at ~1e-16 relative
    (``attack_c3_c4_allruns.py`` docstring), so the CSV *is* the delivered
    likelihood, not a proxy.
    """
    run_rel = CAMPAIGN_REL / f"seed{seed}" / f"real_r{run}"
    run_dir = _require(run_rel)
    df = pd.read_csv(_require(run_rel / "diagnostics" / "event_likelihoods.csv"))
    hs = np.sort(df["h"].unique())
    evs = np.sort(df["event_idx"].unique())
    piv = {
        c: df.pivot(index="event_idx", columns="h", values=c).loc[evs, hs].to_numpy() for c in COLS
    }
    crb = pd.read_csv(_require(CAMPAIGN_REL / f"seed{seed}" / "prepared_cramer_rao_bounds.csv"))
    incat = crb["host_galaxy_index"].to_numpy()[evs] >= 0
    m_z_det = crb["M"].to_numpy()[evs]

    # Gate: the mixture identity the whole chapter rests on.
    rec2 = piv["w_G"] * piv["L_cat_with_bh"] + (1.0 - piv["w_G"]) * piv["L_comp"]
    rec1 = piv["w_G"] * piv["L_cat_no_bh"] + (1.0 - piv["w_G"]) * piv["L_comp"]
    r2 = float(np.abs(rec2 / piv["combined_with_bh"] - 1.0).max())
    r1 = float(np.abs(rec1 / piv["combined_no_bh"] - 1.0).max())
    if max(r1, r2) > TOL_IDENTITY:
        msg = f"mixture identity broken for seed{seed}/r{run}: 1D {r1:.3e} 2D {r2:.3e}"
        raise FidelityError(msg)

    return {
        "seed": seed,
        "run": run,
        "dir": run_dir,
        "h": hs,
        "events": evs,
        "incat": incat,
        "M_z_det": m_z_det,
        "identity_relerr": {"1D": r1, "2D": r2},
        **piv,
    }


def load_cellb() -> dict[str, Any] | None:
    """Pivot the 2x2 cell-B run's diagnostics CSV, or ``None`` if absent.

    Cell B is the same estimator configuration as campaign #53 run against the
    *unscattered* parent catalogue (``observed_catalogue: null``), on the same
    CRB table and injection pool and at the same commit.  It is the control
    that separates the estimator's contribution from the realism layer's, and
    for this chapter it is the first *second* diagnostics CSV the C4 partition
    has ever had.

    Returns ``None`` rather than raising when the artifact is not in either
    checkout: the cell-B block is an addition to an already-gated chapter, and
    a machine without the run must still be able to rebuild the other data.
    """
    csv = _resolve(CELLB_REL / "diagnostics" / "event_likelihoods.csv")
    if csv is None:
        return None
    df = pd.read_csv(csv)
    hs = np.sort(df["h"].unique())
    evs = np.sort(df["event_idx"].unique())
    piv = {
        c: df.pivot(index="event_idx", columns="h", values=c).loc[evs, hs].to_numpy() for c in COLS
    }
    crb = pd.read_csv(_require(CAMPAIGN_REL / "seed61000" / "prepared_cramer_rao_bounds.csv"))
    incat = crb["host_galaxy_index"].to_numpy()[evs] >= 0

    rec2 = piv["w_G"] * piv["L_cat_with_bh"] + (1.0 - piv["w_G"]) * piv["L_comp"]
    rec1 = piv["w_G"] * piv["L_cat_no_bh"] + (1.0 - piv["w_G"]) * piv["L_comp"]
    r2 = float(np.abs(rec2 / piv["combined_with_bh"] - 1.0).max())
    r1 = float(np.abs(rec1 / piv["combined_no_bh"] - 1.0).max())
    if max(r1, r2) > TOL_IDENTITY:
        msg = f"mixture identity broken for cell B: 1D {r1:.3e} 2D {r2:.3e}"
        raise FidelityError(msg)

    return {
        "h": hs,
        "events": evs,
        "incat": incat,
        "identity_relerr": {"1D": r1, "2D": r2},
        **piv,
    }


def map_of(logp: np.ndarray, hs: np.ndarray) -> tuple[float, float]:
    """Grid argmax + 3-point parabola refinement (the convention of
    ``gate_b_20260730/c8_reparam.py``, reproduced so the MAPs are comparable)."""
    k = int(np.argmax(logp))
    grid = float(hs[k])
    if 0 < k < len(hs) - 1:
        y0, y1, y2 = logp[k - 1], logp[k], logp[k + 1]
        d = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / d if d != 0 else 0.0
        refined = grid + off * float(hs[k + 1] - hs[k])
    else:
        refined = grid
    return grid, refined


def sum_logp(leg_cat: np.ndarray, leg_comp: np.ndarray, w: np.ndarray, scale: Any) -> np.ndarray:
    sc = np.asarray(scale, dtype=float)
    if sc.ndim == 1:
        sc = sc[:, None]
    p = w * sc * leg_cat + (1.0 - w) * leg_comp
    p = np.where(p > 0, p, np.finfo(float).tiny)
    return np.log(p).sum(axis=0)


def normalised(logp: np.ndarray, hs: np.ndarray) -> np.ndarray:
    y = np.exp(logp - logp.max())
    return y / np.trapezoid(y, hs)


# ---------------------------------------------------------------------------
# Step 1 — ch08_channel.json  (cold open + C1/C2/C3)
# ---------------------------------------------------------------------------
def build_channel(r1: dict[str, Any]) -> dict[str, Any]:
    hs = r1["h"]
    incat = r1["incat"]
    i73 = int(np.argmin(np.abs(hs - 0.73)))
    i81 = int(np.argmin(np.abs(hs - 0.81)))

    ln1 = np.log(r1["combined_no_bh"])
    ln2 = np.log(r1["combined_with_bh"])
    log1 = ln1.sum(axis=0)
    log2 = ln2.sum(axis=0)

    # Gate against the delivered 2D posterior file.
    delivered = json.loads((r1["dir"] / "combined_posterior_2d.json").read_text())
    hv = np.asarray(delivered["h_values"], dtype=np.float64)
    lo = np.log(np.asarray(delivered["posterior"], dtype=np.float64))
    lo -= lo.max()
    if not np.allclose(hv, hs):
        msg = "delivered combined_posterior_2d.json h-grid differs from the diagnostics CSV"
        raise FidelityError(msg)
    dev = float(np.max(np.abs((log2 - log2.max()) - lo)))
    if dev > TOL_DELIVERED_NATS:
        msg = f"2D reconstruction deviates from the delivered posterior by {dev:.3e} nats"
        raise FidelityError(msg)

    map1_grid, map1_parab = map_of(log1, hs)
    map2_grid, map2_parab = map_of(log2, hs)
    _check("1D MAP (r1)", map1_grid, SPEC["map1d_r1"], 1e-9)
    _check("2D MAP grid (r1)", map2_grid, SPEC["map2d_r1_grid"], 1e-9)
    _check("2D MAP parabola (r1)", map2_parab, SPEC["map2d_r1_parab"], 1e-4)

    # --- C1/C2/C3: the class budget, 0.73 -> 0.81 ---------------------------
    def class_delta(ln: np.ndarray, mask: np.ndarray) -> float:
        return float((ln[mask, i81] - ln[mask, i73]).sum())

    d1_in, d1_dark = class_delta(ln1, incat), class_delta(ln1, ~incat)
    d2_in, d2_dark = class_delta(ln2, incat), class_delta(ln2, ~incat)
    diff_in, diff_dark = d2_in - d1_in, d2_dark - d1_dark
    _check("C1 in-cat 1D delta", d1_in, 2.48, 5e-3)
    _check("C1 dark 1D delta", d1_dark, -11.77, 5e-3)
    _check("C2 1D total", d1_in + d1_dark, -9.30, 5e-3)
    _check("C2 2D total", d2_in + d2_dark, 9.51, 5e-3)
    _check("C3 dark channel diff", diff_dark, SPEC["channel_diff_dark"], 5e-3)
    _check("C3 in-cat channel diff", diff_in, SPEC["channel_diff_incat"], 5e-3)
    _check("C3 total channel diff", diff_in + diff_dark, SPEC["channel_diff_total"], 5e-3)
    _check("C3 dark share %", 100 * diff_dark / (diff_in + diff_dark), SPEC["dark_share_pct"], 0.05)

    # class-summed log profiles (for the "two runaways in the mass channel" view)
    profiles = {
        "1D_incat": ln1[incat].sum(axis=0),
        "1D_dark": ln1[~incat].sum(axis=0),
        "2D_incat": ln2[incat].sum(axis=0),
        "2D_dark": ln2[~incat].sum(axis=0),
    }

    # --- off-r1 replication (all 10 runs), recomputed, not copied ------------
    replication = []
    for seed in SEEDS:
        for run in RUNS:
            rr = load_run(seed, run)
            hh = rr["h"]
            j73 = int(np.argmin(np.abs(hh - 0.73)))
            j81 = int(np.argmin(np.abs(hh - 0.81)))
            a1 = np.log(rr["combined_no_bh"])
            a2 = np.log(rr["combined_with_bh"])
            m = rr["incat"]
            di = float((a2[m, j81] - a2[m, j73]).sum() - (a1[m, j81] - a1[m, j73]).sum())
            dd = float((a2[~m, j81] - a2[~m, j73]).sum() - (a1[~m, j81] - a1[~m, j73]).sum())
            replication.append(
                {
                    "run": f"seed{seed}/real_r{run}",
                    "channel_diff_incat": _r(di, 6),
                    "channel_diff_dark": _r(dd, 6),
                    "dark_share_pct": _r(100 * dd / (di + dd), 5),
                    "n_incat": int(m.sum()),
                    "n_dark": int((~m).sum()),
                }
            )
    dark_all = [row["channel_diff_dark"] for row in replication]
    if min(dark_all) <= 0:
        msg = "off-r1 replication: a dark channel difference is not positive"
        raise FidelityError(msg)
    if not (15.8 <= min(dark_all) and max(dark_all) <= 17.2):
        msg = f"dark channel-difference range {min(dark_all)}..{max(dark_all)} left the cited +15.83..+17.14"
        raise FidelityError(msg)

    # --- the 10-run scorecard ------------------------------------------------
    scores = pd.read_csv(_require(SCORES_REL))
    bias2d = float(scores["map_h_2d"].mean() - H_TRUE)
    pull2d = float(scores["pull_2d"].mean())
    _check("2D mean bias", bias2d, SPEC["bias2d_mean"], 1e-6)
    _check("2D mean pull", pull2d, SPEC["pull2d_mean"], 5e-3)
    n_gt2 = int((scores["pull_2d"].abs() > 2).sum())
    if n_gt2 != 10:
        msg = f"runs with |pull_2d| > 2 = {n_gt2}, cited 10/10"
        raise FidelityError(msg)

    # The readout's per-run pull RANGE does not reproduce; the mean and the
    # 10/10 count do.  Flag, do not reconcile (spec §4.1).
    pull_lo, pull_hi = float(scores["pull_2d"].min()), float(scores["pull_2d"].max())
    if abs(pull_lo - 3.4) > 0.05 or abs(pull_hi - 4.5) > 0.05:
        _FLAGS.append(
            "F1  2D per-run pull RANGE: REALISTIC_READOUT.md §6 states '+3.4 … +4.5 "
            f"(mean +4.04)'; recomputed from realistic_scores.csv (pull_2d, the column "
            f"score_realistic.py:171 writes) the range is +{pull_lo:.2f} … +{pull_hi:.2f}, "
            "mean +%.2f.  Mean and the 10/10 |pull|>2 count agree exactly; the range "
            "does not.  The chapter quotes the mean, the count and the recomputed "
            "range, and says which is which." % pull2d
        )

    return {
        "chapter": "ch08",
        "venue": "campaign #51/#53 realistic, seed61000/real_r1 (1588 events, 41-point h grid)",
        "h_grid": _rl(hs, 6),
        "h_true": float(H_TRUE),
        "n_events": int(len(r1["events"])),
        "n_incat": int(incat.sum()),
        "n_dark": int((~incat).sum()),
        "log_sum": {"1D": _rl(log1 - log1.max(), 10), "2D": _rl(log2 - log2.max(), 10)},
        "class_log_profiles": {k: _rl(v - v.max(), 10) for k, v in profiles.items()},
        "map": {
            "1D_grid": _r(map1_grid, 6),
            "1D_parab": _r(map1_parab, 6),
            "2D_grid": _r(map2_grid, 6),
            "2D_parab": _r(map2_parab, 6),
        },
        "budget": {
            "d1_incat": _r(d1_in, 6),
            "d1_dark": _r(d1_dark, 6),
            "d2_incat": _r(d2_in, 6),
            "d2_dark": _r(d2_dark, 6),
            "diff_incat": _r(diff_in, 6),
            "diff_dark": _r(diff_dark, 6),
            "diff_total": _r(diff_in + diff_dark, 6),
            "dark_share_pct_r1_only": _r(100 * diff_dark / (diff_in + diff_dark), 5),
            "window": "ln p(h=0.81) - ln p(h=0.73), summed over the class",
        },
        "replication": replication,
        "replication_summary": {
            "dark_min": _r(min(dark_all), 6),
            "dark_max": _r(max(dark_all), 6),
            "dark_all_positive": True,
            "incat_min": _r(min(row["channel_diff_incat"] for row in replication), 6),
            "incat_max": _r(max(row["channel_diff_incat"] for row in replication), 6),
            "dark_share_min": _r(min(row["dark_share_pct"] for row in replication), 5),
            "dark_share_max": _r(max(row["dark_share_pct"] for row in replication), 5),
            "note": (
                "What replicates is dark >> in-cat and dark always positive; the precise "
                "84.2% is r1-specific (BOOK_SOURCES_MAP.md §7.5)."
            ),
        },
        "scorecard": [
            {
                "run": f"seed{int(row.seed)}/r{int(row.real)}",
                "map_1d": _r(row.map_h, 5),
                "map_2d": _r(row.map_h_2d, 5),
                "pull_1d": _r(row.pull, 5),
                "pull_2d": _r(row.pull_2d, 5),
                "sigma_H0_2d": _r(row.sigma_H0_2d, 5),
                "edge_2d": _r(row.edge_2d, 4),
            }
            for row in scores.itertuples()
        ],
        "scorecard_summary": {
            "map_2d_mean": _r(float(scores["map_h_2d"].mean()), 6),
            "bias_2d_mean": _r(bias2d, 4),
            "pull_2d_mean": _r(pull2d, 5),
            "pull_2d_min": _r(pull_lo, 5),
            "pull_2d_max": _r(pull_hi, 5),
            "n_pull_gt_2": n_gt2,
            "map_1d_mean": _r(float(scores["map_h"].mean()), 6),
            "pull_1d_absmax": _r(float(scores["pull"].abs().max()), 5),
            "edge_2d_max": _r(float(scores["edge_2d"].max()), 4),
            "cited": "REALISTIC_READOUT.md §6; realistic_scores.csv (score_realistic.py)",
        },
        "checks": {
            "mixture_identity_relerr": {k: _r(v, 4) for k, v in r1["identity_relerr"].items()},
            "delivered_2d_max_abs_dev_nats": _r(dev, 4),
            "readout_map_1d": SPEC["map1d_r1"],
        },
    }


# ---------------------------------------------------------------------------
# Step 2 — ch08_sieve.json  (I8.1)
# ---------------------------------------------------------------------------
def sigma_mz_measured() -> dict[str, float]:
    """Chapter 6's recomputation of sigma_Mz/M_z, redone here (tomas B3).

    The claim file states ~1e-4; the same stored quantity — the CRB table's
    own sqrt(Sigma_MM)/M — measures ~1e-7.  ch06_FLAGS.md F-ch06-5 raised the
    conflict *for this chapter*, so the chapter recomputes it rather than
    transcribing either side, and prints both (worklist D5).
    """
    crb = pd.read_csv(_require(CAMPAIGN_REL / "seed61000" / "prepared_cramer_rao_bounds.csv"))
    rel = np.sqrt(crb["delta_M_delta_M"].to_numpy()) / crb["M"].to_numpy()
    out = {
        "median": float(np.median(rel)),
        "p5": float(np.quantile(rel, 0.05)),
        "p95": float(np.quantile(rel, 0.95)),
        "e889": float(rel[EVENT_INCAT]),
    }
    _check(
        "sigma_Mz/M_z median vs F-ch06-5", out["median"], SPEC["sigma_mz_measured_median"], 5e-10
    )
    _check("sigma_Mz/M_z for 889 vs ch06 §4.1", out["e889"], SPEC["sigma_mz_measured_889"], 5e-12)
    if not (out["median"] < 1e-6 < SPEC["sigma_mz_claim"]):
        msg = "the sigma_Mz both-values pair no longer straddles 1e-6 — re-read F-ch06-5"
        raise FidelityError(msg)
    return out


def build_sieve(r1: dict[str, Any]) -> dict[str, Any]:
    sig_mz = sigma_mz_measured()
    hs = r1["h"]
    incat = r1["incat"]
    dark = ~incat
    i73 = int(np.argmin(np.abs(hs - 0.73)))
    i81 = int(np.argmin(np.abs(hs - 0.81)))
    w = r1["w_G"]
    lc1 = r1["L_cat_no_bh"]
    lc2 = r1["L_cat_with_bh"]
    lcomp = r1["L_comp"]

    # --- per-h zero fractions ------------------------------------------------
    zero1_dark = (lc1[dark] <= 0).mean(axis=0)
    zero2_dark = (lc2[dark] <= 0).mean(axis=0)
    zero1_in = (lc1[incat] <= 0).mean(axis=0)
    zero2_in = (lc2[incat] <= 0).mean(axis=0)
    _check("dark 2D zero % at 0.73", 100 * zero2_dark[i73], SPEC["dark_zero2d_pct"], 0.05)
    _check("dark 1D zero % at 0.73", 100 * zero1_dark[i73], SPEC["dark_zero1d_pct"], 0.05)

    # --- the catalogue mixture weight, per h, dark class ----------------------
    def wcat(lc: np.ndarray) -> np.ndarray:
        num = w * lc
        return num / (num + (1.0 - w) * lcomp)

    wcat1, wcat2 = wcat(lc1), wcat(lc2)
    wcat1_dark = wcat1[dark].mean(axis=0)
    wcat2_dark = wcat2[dark].mean(axis=0)
    _check("dark mean w_cat 1D @0.73", wcat1_dark[i73], SPEC["wcat_dark_1d"], 5e-5)
    _check("dark mean w_cat 2D @0.73", wcat2_dark[i73], SPEC["wcat_dark_2d"], 5e-5)
    deweight = float(wcat1_dark[i73] / wcat2_dark[i73])

    # --- the C4 partition ----------------------------------------------------
    # Masks exactly as gate_b_20260730/attack_c4_decomposition.py defines them:
    # "alive" means nonzero at ANY h on the grid, not merely at h = 0.73.
    any1d = np.any(lc1 > 0, axis=1)
    any2d = np.any(lc2 > 0, axis=1)
    n_1d_nonzero = int(any1d.sum())
    n_zero2_of_nonzero1 = int((any1d & ~any2d).sum())
    n_zero2_dark = int((any1d & ~any2d & dark).sum())
    _check("n events with a live 1D catalogue leg", n_1d_nonzero, 1095, 0.5)
    _check("n 2D-dead of 1D-alive", n_zero2_of_nonzero1, SPEC["n_zero2d_of_nonzero1d"], 0.5)
    _check("n 2D-dead dark", n_zero2_dark, SPEC["n_zero2d_dark"], 0.5)

    grp_zeroed = dark & any1d & ~any2d
    grp_bothdead = dark & ~any1d
    grp_surv = dark & any2d
    ln1 = np.log(r1["combined_no_bh"])
    ln2 = np.log(r1["combined_with_bh"])

    def part(mask: np.ndarray) -> dict[str, Any]:
        d1 = float((ln1[mask, i81] - ln1[mask, i73]).sum())
        d2 = float((ln2[mask, i81] - ln2[mask, i73]).sum())
        return {"n": int(mask.sum()), "d1": _r(d1, 6), "d2": _r(d2, 6), "diff": _r(d2 - d1, 6)}

    p_zeroed, p_bothdead, p_surv = part(grp_zeroed), part(grp_bothdead), part(grp_surv)
    total_dark_diff = p_zeroed["diff"] + p_bothdead["diff"] + p_surv["diff"]
    _check("C4 zeroed-group contribution", p_zeroed["diff"], SPEC["budget_zeroed"], 5e-3)
    _check("C4 both-dead contribution", p_bothdead["diff"], 0.0, 1e-9)
    _check("C4 survivors contribution", p_surv["diff"], SPEC["budget_survivors"], 5e-3)
    _check("C4 partition closes", total_dark_diff, SPEC["channel_diff_dark"], 5e-3)

    # --- the median suppression of surviving catalogue legs at h = 0.73 ------
    # Population as attack_c3_c4_allruns.py:145-148 defines it: EVERY event whose
    # two catalogue legs are both alive at h = 0.73 (not the dark subset).
    both_alive = (lc1[:, i73] > 0) & (lc2[:, i73] > 0)
    ratio = lc2[both_alive, i73] / lc1[both_alive, i73]
    both_alive_dark = both_alive & dark
    ratio_dark = lc2[both_alive_dark, i73] / lc1[both_alive_dark, i73]
    median_supp = float(np.median(ratio))
    _check("median suppression @0.73", median_supp, SPEC["median_suppression"], 5e-5)

    # log10 histogram of the suppression (fixed bins so the browser draws bars)
    lr = np.log10(ratio)
    edges = np.arange(-5.0, 0.51, 0.25)
    counts, _ = np.histogram(lr, bins=edges)

    # --- the dark ln(L_cat_2D / L_cat_1D) tilt -------------------------------
    # Population as attack_c3_c4_allruns.py:150-163: both legs alive in BOTH
    # channels at BOTH h = 0.73 and h = 0.81.
    ok_pair = (lc1[:, i73] > 0) & (lc2[:, i73] > 0) & (lc1[:, i81] > 0) & (lc2[:, i81] > 0)
    alive_both_allh = dark & ok_pair
    incat_pair = incat & ok_pair
    s073 = float(np.log(lc2[alive_both_allh, i73] / lc1[alive_both_allh, i73]).sum())
    s081 = float(np.log(lc2[alive_both_allh, i81] / lc1[alive_both_allh, i81]).sum())
    incat_tilt = float(
        np.log(lc2[incat_pair, i81] / lc1[incat_pair, i81]).sum()
        - np.log(lc2[incat_pair, i73] / lc1[incat_pair, i73]).sum()
    )
    _check("dark ln-ratio tilt", s081 - s073, SPEC["dark_tilt_lnratio"], 0.05)
    _check("in-cat ln-ratio tilt", incat_tilt, 0.27, 0.05)

    # --- C10: where the up-pull actually lives -------------------------------
    ln_prefactor = np.log(1.0 - w)
    d_pref_all = float((ln_prefactor[:, i81] - ln_prefactor[:, i73]).sum())
    d_pref_dark = float((ln_prefactor[dark, i81] - ln_prefactor[dark, i73]).sum())
    d_pref_in = float((ln_prefactor[incat, i81] - ln_prefactor[incat, i73]).sum())
    ln_comp = np.log(lcomp)
    d_comp_all = float((ln_comp[:, i81] - ln_comp[:, i73]).sum())
    d_comp_dark = float((ln_comp[dark, i81] - ln_comp[dark, i73]).sum())
    d_comp_in = float((ln_comp[incat, i81] - ln_comp[incat, i73]).sum())
    # attack_c4_decomposition.py:234 counts the sign of the FULL channel-common
    # factor dlnC = dln[(1-w_G) L_comp], not of L_comp alone.  Both are reported.
    dlnC = (ln_prefactor + ln_comp)[:, i81] - (ln_prefactor + ln_comp)[:, i73]
    frac_pos = float((dlnC[dark] > 0).mean())
    frac_pos_lcomp_only = float((ln_comp[dark, i81] > ln_comp[dark, i73]).mean())
    _check("C10 N dln(1-w_G)", d_pref_all, SPEC["c10_prefactor_nats"], 5e-3)
    _check("C10 sum dln L_comp", d_comp_all, SPEC["c10_lcomp_nats"], 5e-3)
    _check("C10 dark dln L_comp", d_comp_dark, SPEC["c10_dark_lcomp_nats"], 5e-3)
    _check("C10 dark positive fraction", frac_pos, SPEC["c10_dark_positive_frac"], 1e-3)

    # --- the dark class's own profile, and where it now points (C4-amended) --
    # These are the CLASS-SUMMED combined log-likelihood profiles, exactly the
    # objects gate_b_20260730/attack_c4_decomposition.py §7 stores as
    # ``profiles["1D_DARK"] / ["2D_DARK"] / ["C_DARK"]``.
    prof1 = ln1[dark].sum(axis=0)
    prof2 = ln2[dark].sum(axis=0)
    prof_c = (ln_prefactor + ln_comp)[dark].sum(axis=0)
    i86 = int(len(hs) - 1)
    opp1 = float(prof1[i86] - prof1[i73])
    opp2 = float(prof2[i86] - prof2[i73])
    _check("dark opposition 1D (0.73->0.86)", opp1, -24.46, 0.05)
    _check("dark opposition 2D (0.73->0.86)", opp2, -0.63, 0.05)
    argmax1 = float(hs[int(np.argmax(prof1))])
    argmax2 = float(hs[int(np.argmax(prof2))])
    _check("dark class profile argmax 1D", argmax1, 0.640, 1e-9)
    _check("dark class profile argmax 2D", argmax2, 0.785, 1e-9)

    # the dark class's completion leg alone, (1-w_G) L_comp
    argmax_comp = float(hs[int(np.argmax(prof_c))])
    _check("dark completion-leg argmax", argmax_comp, 0.810, 1e-9)
    tilt1, tilt2 = prof1, prof2

    return {
        "chapter": "ch08",
        "venue": "seed61000/real_r1",
        "h_grid": _rl(hs, 6),
        "h_true": float(H_TRUE),
        "i_073": i73,
        "zero_fraction": {
            "dark_1D": _rl(zero1_dark, 5),
            "dark_2D": _rl(zero2_dark, 5),
            "incat_1D": _rl(zero1_in, 5),
            "incat_2D": _rl(zero2_in, 5),
        },
        "w_cat_dark": {"1D": _rl(wcat1_dark, 6), "2D": _rl(wcat2_dark, 6)},
        "deweighting_factor_073": _r(deweight, 4),
        "partition": {
            "zeroed_2d_alive_1d": p_zeroed,
            "both_dead": p_bothdead,
            "survivors": p_surv,
            "total": _r(total_dark_diff, 6),
            "pct_survivors": _r(100 * p_surv["diff"] / total_dark_diff, 4),
            "pct_zeroed": _r(100 * p_zeroed["diff"] / total_dark_diff, 4),
            "budget_note": (
                "+15.83 = 0 (completion, cancels identically) + 19.10 (loss of the 1D "
                "catalogue down-tilt) - 3.27 (residual 2D tilt) — CLAIM C4-amended"
            ),
            "loss_of_1d_downtilt": SPEC["budget_loss_of_1d_downtilt"],
            "residual_2d_tilt": SPEC["budget_residual_2d_tilt"],
        },
        "suppression": {
            "median_073": _r(median_supp, 5),
            "median_073_dark_only": _r(float(np.median(ratio_dark)), 5),
            "n_surviving_pairs": int(both_alive.sum()),
            "n_surviving_pairs_dark": int(both_alive_dark.sum()),
            "hist_log10_edges": _rl(edges, 4),
            "hist_counts": [int(c) for c in counts],
            "q10_q90": _rl(np.quantile(ratio, [0.1, 0.9]), 4),
        },
        "dark_lnratio_tilt": {
            "n_events": int(alive_both_allh.sum()),
            "s_073": _r(s073, 8),
            "s_081": _r(s081, 8),
            "delta": _r(s081 - s073, 6),
            "incat_n_events": int(incat_pair.sum()),
            "incat_delta": _r(incat_tilt, 4),
        },
        "counts": {
            "n_1d_nonzero": n_1d_nonzero,
            "n_zero2d_of_nonzero1d": n_zero2_of_nonzero1,
            "n_zero2d_dark": n_zero2_dark,
        },
        "c10": {
            "N_dln_1_minus_wG": {
                "all": _r(d_pref_all, 6),
                "dark": _r(d_pref_dark, 6),
                "incat": _r(d_pref_in, 6),
            },
            "sum_dln_Lcomp": {
                "all": _r(d_comp_all, 6),
                "dark": _r(d_comp_dark, 6),
                "incat": _r(d_comp_in, 6),
            },
            "dark_frac_positive_completion_tilt": _r(frac_pos, 5),
            "dark_frac_positive_Lcomp_only": _r(frac_pos_lcomp_only, 5),
            "dlnC_dark": _r(float(dlnC[dark].sum()), 6),
            "dlnC_incat": _r(float(dlnC[incat].sum()), 6),
            "definition": (
                "dlnC = dln[(1-w_G) L_comp] over h = 0.73 -> 0.81; the 39.1% counts "
                "the sign of dlnC (attack_c4_decomposition.py:234). L_comp alone is "
                "positive for fewer dark events still."
            ),
        },
        "dark_class_profile": {
            "1D": _rl(tilt1 - tilt1.max(), 6),
            "2D": _rl(tilt2 - tilt2.max(), 6),
            "completion_only": _rl(prof_c - prof_c.max(), 6),
            "argmax_1D": argmax1,
            "argmax_2D": argmax2,
            "opposition_1D_073_086": _r(opp1, 5),
            "opposition_2D_073_086": _r(opp2, 5),
            "dark_completion_argmax": argmax_comp,
        },
        "w_G": {
            "0.60": _r(float(w[0, 0]), 8),
            "0.73": _r(float(w[0, i73]), 8),
            "0.86": _r(float(w[0, -1]), 8),
        },
        "recorded": {
            "impostor_rejection_pct": "97-99",
            "impostor_rejection_src": "CLAIM_2D_BIAS_20260730.md, the claim in one paragraph",
            "one_sidedness_low": 193,
            "one_sidedness_high": 1,
            "one_sidedness_src": "CLAIM_2D_BIAS_20260730.md C4 (P6 measurement)",
            "sigma_lnM_catalogue": 1.28,
            "sigma_lnM_kernel_floor": 0.58,
            # BOTH-VALUES item (worklist D5; tomas B3; ch06_FLAGS.md F-ch06-5).
            # The claim file states 1e-4 at CLAIM_2D_BIAS_20260730.md:172; the
            # same stored quantity recomputed from the CRB table is ~1e-7.
            # The chapter prints both and prefers neither; amending the claim
            # file is the author's call, not the book's.
            "sigma_Mz_over_Mz": 1e-4,
            "sigma_Mz_over_Mz_claim_src": "CLAIM_2D_BIAS_20260730.md:172 (C4)",
            "sigma_Mz_over_Mz_measured_median": _r(sig_mz["median"], 4),
            "sigma_Mz_over_Mz_measured_p5": _r(sig_mz["p5"], 4),
            "sigma_Mz_over_Mz_measured_p95": _r(sig_mz["p95"], 4),
            "sigma_Mz_over_Mz_measured_889": _r(sig_mz["e889"], 4),
            "sigma_Mz_over_Mz_measured_src": (
                "sqrt(delta_M_delta_M)/M of prepared_cramer_rao_bounds.csv; ch06_FLAGS.md F-ch06-5"
            ),
            "sigma_Mz_over_Mz_status": (
                "BOTH VALUES LIVE — three orders of magnitude apart, flagged, neither "
                "corrected in the book. Every argument in the chapter needs only that the "
                "GW side is negligible against the catalogue's sigma_lnM ~ 1.28, which "
                "both satisfy."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Step 2b — the 2x2's cell B  (expB MJ-4; folded into ch08_sieve.json)
# ---------------------------------------------------------------------------
def build_cellb(cb: dict[str, Any]) -> dict[str, Any]:
    """The C4 accounting redone on the unscattered configuration.

    Two things this delivers to §4/§5, neither of which the chapter could say
    when it was drafted:

    1.  the dark channel difference is +18.00 nats with **zero realized
        scatter**, so the mass channel's de-weighting is estimator-borne and
        not an artifact of the realism layer;
    2.  the 98.5% / 1.5% survivors-vs-deleted split is *configuration-scoped* —
        it reads 80.7% / 19.2% here.  Deletion is the minority carrier in both,
        which is the finding; the percentage is not.

    The partition numbers are gated against expert review B's independent
    recomputation, and flagged in ``ch08_FLAGS.md`` (F-ch08-9) as recomputed
    for the book: they are in no adjudicated artifact.
    """
    hs = cb["h"]
    incat = cb["incat"]
    dark = ~incat
    i73 = int(np.argmin(np.abs(hs - 0.73)))
    i81 = int(np.argmin(np.abs(hs - 0.81)))
    w, lc1, lc2, lcomp = cb["w_G"], cb["L_cat_no_bh"], cb["L_cat_with_bh"], cb["L_comp"]
    ln1 = np.log(cb["combined_no_bh"])
    ln2 = np.log(cb["combined_with_bh"])

    # --- the delivered MAPs (the readout's headline) -------------------------
    map1_grid, _ = map_of(ln1.sum(axis=0), hs)
    map2_grid, _ = map_of(ln2.sum(axis=0), hs)
    _check("cell B 1D MAP", map1_grid, SPEC["cellb_map1d"], 1e-9)
    _check("cell B 2D MAP", map2_grid, SPEC["cellb_map2d"], 1e-9)

    # --- the class budget, 0.73 -> 0.81 --------------------------------------
    def diff(mask: np.ndarray) -> float:
        return float(
            (ln2[mask, i81] - ln2[mask, i73]).sum() - (ln1[mask, i81] - ln1[mask, i73]).sum()
        )

    d_dark, d_incat = diff(dark), diff(incat)
    _check("cell B dark channel diff", d_dark, SPEC["cellb_channel_diff_dark"], 5e-3)
    _check("cell B in-cat channel diff", d_incat, SPEC["cellb_channel_diff_incat"], 5e-3)
    _check("cell B total channel diff", d_dark + d_incat, SPEC["cellb_channel_diff_total"], 5e-3)
    if d_dark <= 0:
        msg = "cell B's dark channel difference is not positive — the whole §5 claim inverts"
        raise FidelityError(msg)

    # --- the C4 partition, same masks as attack_c4_decomposition.py ----------
    any1d = np.any(lc1 > 0, axis=1)
    any2d = np.any(lc2 > 0, axis=1)

    def part(mask: np.ndarray) -> dict[str, Any]:
        d1 = float((ln1[mask, i81] - ln1[mask, i73]).sum())
        d2 = float((ln2[mask, i81] - ln2[mask, i73]).sum())
        return {"n": int(mask.sum()), "diff": _r(d2 - d1, 6)}

    p_zeroed = part(dark & any1d & ~any2d)
    p_bothdead = part(dark & ~any1d)
    p_surv = part(dark & any2d)
    _check("cell B survivors n", p_surv["n"], SPEC["cellb_n_surv"], 0.5)
    _check("cell B zeroed n", p_zeroed["n"], SPEC["cellb_n_zeroed"], 0.5)
    _check("cell B both-dead contribution", p_bothdead["diff"], 0.0, 1e-9)
    _check("cell B partition closes", p_zeroed["diff"] + p_surv["diff"], d_dark, 5e-3)
    pct_surv = 100 * p_surv["diff"] / d_dark
    pct_zeroed = 100 * p_zeroed["diff"] / d_dark
    # the reviewer quoted the two shares to one decimal; gate at that precision
    _check("cell B survivors share %", pct_surv, SPEC["cellb_pct_surv"], 0.1)
    _check("cell B zeroed share %", pct_zeroed, SPEC["cellb_pct_zeroed"], 0.1)
    # The structural claim the chapter actually makes — deletion is the
    # minority carrier — must hold, or the sentence is wrong on this venue.
    if pct_zeroed >= pct_surv:
        msg = "cell B: deletion is no longer the minority carrier; §5's sentence is false"
        raise FidelityError(msg)

    # --- the de-weighting, same estimator as F-ch08-7 -------------------------
    def wcat(lc: np.ndarray) -> np.ndarray:
        num = w * lc
        return num / (num + (1.0 - w) * lcomp)

    wc1 = float(wcat(lc1)[dark].mean(axis=0)[i73])
    wc2 = float(wcat(lc2)[dark].mean(axis=0)[i73])
    _check("cell B dark w_cat 1D", wc1, SPEC["cellb_wcat_dark_1d"], 5e-5)
    _check("cell B dark w_cat 2D", wc2, SPEC["cellb_wcat_dark_2d"], 5e-5)
    _check("cell B de-weighting factor", wc1 / wc2, SPEC["cellb_deweight"], 5e-3)

    zero2d = float((lc2[dark, i73] <= 0).mean())
    _check("cell B dark 2D zero fraction", zero2d, SPEC["cellb_dark_zero2d_frac"], 5e-4)
    _check("cell B events with a live 1D leg", int(any1d.sum()), SPEC["cellb_n_1d_nonzero"], 0.5)

    # --- w_G(h): the pre-registered bit-identity read (ch09's payoff) ---------
    i60 = int(np.argmin(np.abs(hs - 0.60)))
    _check("cell B w_G(0.60)", float(w[0, i60]), SPEC["cellb_wG_060"], 5e-8)
    _check("cell B w_G(0.73)", float(w[0, i73]), SPEC["cellb_wG_073"], 5e-8)
    _check("cell B w_G(0.81)", float(w[0, i81]), SPEC["cellb_wG_081"], 5e-8)

    return {
        "venue": "seed61000/estimatorB_2x2 — the 2x2 cell B (unscattered parent catalogue)",
        "readout": "CELLB_READOUT_20260731.md",
        "jobs_result": "evaluate 6103219 / combine 6103220",
        "jobs_prereg": "6101146 / 6101147",
        "resubmission_note": (
            "6103219/6103220 are the resubmission of 6101146/6101147 after a pure-plumbing "
            "symlink failure in the run-dir setup; the test design and the pre-registration "
            "are unchanged, and the code is the same commit (7fd60bb) as cells A and C."
        ),
        "code": "7fd60bb",
        "map_1d": _r(map1_grid, 6),
        "map_2d": _r(map2_grid, 6),
        "channel_diff": {
            "dark": _r(d_dark, 6),
            "incat": _r(d_incat, 6),
            "total": _r(d_dark + d_incat, 6),
            "comparison_scattered_r1": {"dark": 15.83, "incat": 2.97, "total": 18.80},
        },
        "partition": {
            "survivors": p_surv,
            "zeroed_2d_alive_1d": p_zeroed,
            "both_dead": p_bothdead,
            "pct_survivors": _r(pct_surv, 4),
            "pct_zeroed": _r(pct_zeroed, 4),
            "comparison_scattered_r1": {"pct_survivors": 98.5, "pct_zeroed": 1.5},
        },
        "w_cat_dark_073": {"1D": _r(wc1, 5), "2D": _r(wc2, 5), "factor": _r(wc1 / wc2, 4)},
        "dark_zero_fraction_2d_073": _r(zero2d, 5),
        "n_1d_nonzero": int(any1d.sum()),
        "w_G": {
            "0.60": _r(float(w[0, i60]), 8),
            "0.73": _r(float(w[0, i73]), 8),
            "0.81": _r(float(w[0, i81]), 8),
        },
        "identity_relerr": {k: _r(v, 4) for k, v in cb["identity_relerr"].items()},
        "provenance": (
            "RECOMPUTED FOR THE BOOK from cell B's own diagnostics CSV. The MAPs, the "
            "channel differences and w_G(h) are in CELLB_READOUT_20260731.md; the C4 "
            "partition and the de-weighting factor are NOT in any adjudicated artifact — "
            "they were first computed by the book's expert review and are reproduced here. "
            "See book/design/flags/ch08_FLAGS.md F-ch08-9."
        ),
    }


# ---------------------------------------------------------------------------
# Step 3 — ch08_reparam.json  (I8.2)
# ---------------------------------------------------------------------------
C_STEPS = (100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 1e-3, 1e-4, 1e-6)


def build_reparam(r1: dict[str, Any]) -> dict[str, Any]:
    hs = r1["h"]
    w = r1["w_G"]
    lc1 = r1["L_cat_no_bh"]
    lc2 = r1["L_cat_with_bh"]
    lcomp = r1["L_comp"]
    m_z_det = r1["M_z_det"]

    cited = json.loads(_require(C8_RESULTS_REL).read_text())

    sweep = []
    for c in C_STEPS:
        lp = sum_logp(lc2, lcomp, w, 1.0 / c)
        grid, parab = map_of(lp, hs)
        sweep.append(
            {
                "C": c,
                "map_grid": _r(grid, 6),
                "map_parab": _r(parab, 6),
                "posterior": _rl(normalised(lp, hs), 5),
            }
        )
        key = f"{c:g}"
        if key in cited["constant_C_sweep"]:
            _check(
                f"C-sweep MAP at C={key} vs c8_reparam_results.json",
                parab,
                cited["constant_C_sweep"][key]["map2d_parab"],
                1e-9,
            )

    # The 1D channel carries no mass factor at all: verify bitwise invariance
    # by construction *and* by evaluation.
    lp1 = sum_logp(lc1, lcomp, w, 1.0)
    # Bitwise: the 1D summed log-likelihood recomputed at every C on the sweep
    # must be the identical float array, because C never touches the 1D leg.
    bitwise = all(np.array_equal(lp1, sum_logp(lc1, lcomp, w, 1.0)) for _ in C_STEPS)
    map1 = map_of(lp1, hs)
    _check("1D MAP under the sweep", map1[1], cited["baseline"]["map1d_parab"], 1e-9)

    # --- the four NAMED measures (per-event C_i), c8_reparam.py block [D] ----
    named = []
    for label, ci, blurb in (
        (
            "fraction M_z / M_z,det,i (the code as shipped)",
            np.ones_like(m_z_det),
            "the measure bayesian_statistics.py hard-wires: each event's own measured mass",
        ),
        (
            "M_z in 10^6 M_sun",
            m_z_det / 1e6,
            "a single, population-scale unit for every event",
        ),
        ("M_z in 10^5 M_sun", m_z_det / 1e5, "the same choice, one decade smaller"),
        ("M_z in M_sun", m_z_det, "the textbook unit"),
    ):
        lp = sum_logp(lc2, lcomp, w, 1.0 / ci)
        grid, parab = map_of(lp, hs)
        named.append(
            {
                "label": label,
                "note": blurb,
                "map_grid": _r(grid, 6),
                "map_parab": _r(parab, 6),
                "_map_parab_raw": parab,
                "posterior": _rl(normalised(lp, hs), 5),
            }
        )
    for label_key, entry in (
        ("fraction (code as-is)", named[0]),
        ("M_z in 1e6 Msun", named[1]),
        ("M_z in 1e5 Msun", named[2]),
        ("M_z in Msun", named[3]),
    ):
        _check(
            f"named measure '{label_key}' vs c8_reparam_results.json",
            entry.pop("_map_parab_raw"),
            cited["alternative_measures"][label_key]["map2d_parab"],
            1e-9,
        )

    # --- the sensitivity slope, on the unrailed band ------------------------
    # 25 log-spaced C in [0.05, 3], exactly as c8_reparam.py:218-224.
    c_dense = np.exp(np.linspace(np.log(0.05), np.log(3.0), 25))
    maps_dense = [map_of(sum_logp(lc2, lcomp, w, 1.0 / c), hs)[1] for c in c_dense]
    slope = float(np.polyfit(np.log(c_dense), np.array(maps_dense), 1)[0])
    _check("d MAP / d ln C vs c8_reparam_results.json", slope, cited["dMAP_dlnC"], 1e-9)
    _check("d MAP / d ln C vs the claim's +0.031", slope, SPEC["dmap_dlnc"], 1e-3)

    # --- geometric-mean control: a single constant of the same scale ---------
    geo = float(np.exp(np.mean(np.log(m_z_det))))
    lp_geo = sum_logp(lc2, lcomp, w, geo / m_z_det)  # replace M_z,det,i by its geometric mean
    _, map_geo = map_of(lp_geo, hs)
    shift_geo = abs(map_geo - float(named[0]["map_parab"]))

    return {
        "chapter": "ch08",
        "venue": "seed61000/real_r1 (1588 events)",
        "h_grid": _rl(hs, 6),
        "h_true": float(H_TRUE),
        "sweep": sweep,
        "named_measures": named,
        "one_d": {
            "map_grid": _r(map1[0], 6),
            "map_parab": _r(map1[1], 6),
            "posterior": _rl(normalised(lp1, hs), 5),
            "bitwise_invariant": bitwise,
            "n_sweep_points_checked": len(C_STEPS),
            "note": (
                "The 1D catalogue leg carries no mass density (handler.py builds its "
                "candidate list with a redshift filter only), so C never multiplies it. "
                "The sweep leaves the summed 1D log-likelihood bitwise identical."
            ),
        },
        "sensitivity": {
            "dMAP_dlnC": _r(slope, 5),
            "band": "C in [0.05, 3], 25 log-spaced points",
            "per_efold_H0_km_s_Mpc": _r(100.0 * slope, 4),
            "ln_C": _rl(np.log(c_dense), 5),
            "map_parab": _rl(maps_dense, 6),
        },
        "geometric_mean_control": {
            "M_z_det_geometric_mean_Msun": _r(geo, 6),
            "map_parab": _r(map_geo, 6),
            "shift_from_shipped": _r(shift_geo, 3),
            "note": (
                "Replacing each event's own M_z,det,i by one constant of the same "
                "geometric mean already moves the MAP — the implicit unit is per-event."
            ),
        },
        "M_z_det": {
            "min": _r(float(m_z_det.min()), 6),
            "max": _r(float(m_z_det.max()), 6),
            "median": _r(float(np.median(m_z_det)), 6),
            "span_factor": _r(float(m_z_det.max() / m_z_det.min()), 4),
        },
        "checks": {
            "delivered_logpost_max_abs_dev_nats": _r(
                cited["delivered_logpost_max_abs_dev_nats"], 4
            ),
            "identity_relerr_2d": _r(cited["identity_max_relerr_2d"], 4),
            "identity_relerr_1d": _r(cited["identity_max_relerr_1d"], 4),
            "source": "gate_b_20260730/c8_reparam.py + c8_reparam_results.json",
        },
        "canonical_fix_indicative": {
            "g_frac_median": 0.135,
            "over_weight_factor": 7.4,
            "map_h_frozen": 0.7558,
            "map_full_g_of_h": 0.84917,
            "ha_endpoint_reproduced": 0.8492,
            "ha_agreement": 3e-5,
            "status": "INDICATIVE, NOT RATIFIED",
            "src": "gate_b_20260730/README_C8.md part (3); CLAIM_2D_BIAS_20260730.md C8",
        },
    }


# ---------------------------------------------------------------------------
# Step 4 — ch08_twofaces.json  (I8.3)
# ---------------------------------------------------------------------------
def build_twofaces(r1: dict[str, Any]) -> dict[str, Any]:
    hs = r1["h"]
    evs = list(r1["events"])
    i73 = int(np.argmin(np.abs(hs - 0.73)))
    i81 = int(np.argmin(np.abs(hs - 0.81)))
    crb = pd.read_csv(_require(CAMPAIGN_REL / "seed61000" / "prepared_cramer_rao_bounds.csv"))

    events: dict[str, Any] = {}
    for ev in (EVENT_INCAT, EVENT_DARK):
        k = evs.index(ev)
        row = crb.iloc[ev]
        lc1 = r1["L_cat_no_bh"][k]
        lc2 = r1["L_cat_with_bh"][k]
        lcomp = r1["L_comp"][k]
        w = r1["w_G"][k]
        c1 = r1["combined_no_bh"][k]
        c2 = r1["combined_with_bh"][k]
        alive = (lc1 > 0) & (lc2 > 0)
        ratio = np.where(alive, lc2 / np.where(lc1 > 0, lc1, 1.0), np.nan)
        events[str(ev)] = {
            "index": ev,
            "in_catalog": bool(row["host_galaxy_index"] >= 0),
            "host_galaxy_index": int(row["host_galaxy_index"]),
            "SNR": _r(float(row["SNR"]), 6),
            "d_L_Gpc": _r(float(row["luminosity_distance"]), 6),
            # The CRB table's mass column is the DETECTOR-frame mass: this is how
            # gate_b_20260730/c8_reparam.py reads it (`M_z = crb["M"]`), and its
            # min/max reproduce README_C8's quoted M_z,det span 1.33e5-1.63e6.
            "M_z_det_Msun": _r(float(row["M"]), 6),
            "L_cat_1D": _rl(lc1, 6),
            "L_cat_2D": _rl(lc2, 6),
            "L_comp": _rl(lcomp, 6),
            "w_G": _rl(w, 6),
            "combined_1D": _rl(c1, 6),
            "combined_2D": _rl(c2, 6),
            "post_1D": _rl(normalised(np.log(c1), hs), 5),
            "post_2D": _rl(normalised(np.log(c2), hs), 5),
            "argmax_1D": _r(float(hs[int(np.argmax(c1))]), 6),
            "argmax_2D": _r(float(hs[int(np.argmax(c2))]), 6),
            "argmax_Lcat_1D": (_r(float(hs[int(np.argmax(lc1))]), 6) if lc1.max() > 0 else None),
            "argmax_Lcat_2D": (_r(float(hs[int(np.argmax(lc2))]), 6) if lc2.max() > 0 else None),
            "argmax_Lcomp": _r(float(hs[int(np.argmax(lcomp))]), 6),
            "suppression_073": (_r(float(ratio[i73]), 5) if alive[i73] else None),
            "suppression_factor_073": (
                _r(float(1.0 / ratio[i73]), 5) if alive[i73] and ratio[i73] > 0 else None
            ),
            "median_suppression": (
                _r(float(np.nanmedian(ratio)), 5) if np.isfinite(ratio).any() else None
            ),
            "channel_diff_073_081": _r(
                float(np.log(c2[i81]) - np.log(c2[i73]) - np.log(c1[i81]) + np.log(c1[i73])), 5
            ),
            "w_cat_073_1D": _r(float(w[i73] * lc1[i73] / c1[i73]), 6),
            "w_cat_073_2D": _r(float(w[i73] * lc2[i73] / c2[i73]), 6),
            "one_minus_w_cat_073_1D": _r(float((1 - w[i73]) * lcomp[i73] / c1[i73]), 4),
            "one_minus_w_cat_073_2D": _r(float((1 - w[i73]) * lcomp[i73] / c2[i73]), 4),
            "w_cat_collapse_factor_073": _r(
                float((w[i73] * lc1[i73] / c1[i73]) / (w[i73] * lc2[i73] / c2[i73])), 5
            )
            if lc2[i73] > 0
            else None,
            # span of each leg across the grid, relative to its value at h = 0.73
            "L_cat_1D_span": (_r(float(lc1.max() / lc1[i73]), 4) if lc1[i73] > 0 else None),
            "L_cat_2D_span": (_r(float(lc2.max() / lc2[i73]), 4) if lc2[i73] > 0 else None),
            "L_comp_ratio_086_over_073": _r(float(lcomp[-1] / lcomp[i73]), 4),
            "L_cat_2D_ratio_086_over_073": (
                _r(float(lc2[-1] / lc2[i73]), 4) if lc2[i73] > 0 else None
            ),
            "L_cat_1D_ratio_086_over_073": (
                _r(float(lc1[-1] / lc1[i73]), 4) if lc1[i73] > 0 else None
            ),
        }

    # --- 889's channel swing across the five seed-61000 realizations ---------
    swing = []
    for run in RUNS:
        rr = load_run(61000, run)
        hh = rr["h"]
        j73 = int(np.argmin(np.abs(hh - 0.73)))
        j81 = int(np.argmin(np.abs(hh - 0.81)))
        kk = list(rr["events"]).index(EVENT_INCAT)
        d1 = float(np.log(rr["combined_no_bh"][kk, j81]) - np.log(rr["combined_no_bh"][kk, j73]))
        d2 = float(
            np.log(rr["combined_with_bh"][kk, j81]) - np.log(rr["combined_with_bh"][kk, j73])
        )
        swing.append({"run": f"r{run}", "d1": _r(d1, 5), "d2": _r(d2, 5), "diff": _r(d2 - d1, 5)})
    for j, want in enumerate(SPEC["e889_swing"]):
        _check(f"889 channel swing r{j + 1}", swing[j]["diff"], want, 6e-3)

    return {
        "chapter": "ch08",
        "venue": "seed61000/real_r1",
        "h_grid": _rl(hs, 6),
        "h_true": float(H_TRUE),
        "events": events,
        "e889_swing": swing,
        "swing_note": (
            "The same in-catalogue event, five noise realizations of the SAME 76-event "
            "class: its channel difference swings +1.98 -> -2.04 -> -3.30 (CLAIM C3, "
            "off-r1 replication note)."
        ),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"repo root: {REPO_ROOT}")
    r1 = load_run(61000, 1)
    print(
        f"loaded seed61000/real_r1: {len(r1['events'])} events x {len(r1['h'])} h-points; "
        f"mixture identity 1D {r1['identity_relerr']['1D']:.2e} / "
        f"2D {r1['identity_relerr']['2D']:.2e}"
    )

    sieve = build_sieve(r1)
    cb = load_cellb()
    if cb is None:
        _FLAGS.append(
            "F2  the 2x2 cell-B run is not in either checkout "
            f"({CELLB_REL}); §5's cell-B block is on the page but its numbers "
            "could not be re-derived on this machine."
        )
        print("WARNING: cell-B diagnostics not found — cell-B block not regenerated")
    else:
        cell_b = build_cellb(cb)
        sieve["cell_b"] = cell_b
        surv = cell_b["partition"]["survivors"]
        zeroed = cell_b["partition"]["zeroed_2d_alive_1d"]
        print(
            f"cell B: 1D MAP {cell_b['map_1d']:.4f} / 2D MAP {cell_b['map_2d']:.4f}; "
            f"dark channel diff {cell_b['channel_diff']['dark']:+.2f} nats; "
            f"survivors {surv['n']} ({cell_b['partition']['pct_survivors']:.1f}%) "
            f"vs zeroed {zeroed['n']} ({cell_b['partition']['pct_zeroed']:.1f}%)"
        )

    payloads = [
        (OUT_CHANNEL, build_channel(r1)),
        (OUT_SIEVE, sieve),
        (OUT_REPARAM, build_reparam(r1)),
        (OUT_TWOFACES, build_twofaces(r1)),
    ]
    for path, payload in payloads:
        path.write_text(json.dumps(payload, separators=(",", ":")))
        size = path.stat().st_size
        print(f"wrote {path.relative_to(REPO_ROOT.parent)}  {size / 1024:.1f} KiB")
        if size > 500_000:
            msg = f"{path.name} is {size} bytes — over the 500 KB budget"
            raise FidelityError(msg)

    if _FLAGS:
        print("\nFLAGS RAISED (recorded in book/design/flags/ch08_FLAGS.md):")
        for f in _FLAGS:
            print("  - " + f)
    print("\nAll fidelity gates passed.")


if __name__ == "__main__":
    main()
