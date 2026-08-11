"""Generator for Chapter 3 — "Which Galaxy?".

Produces the three data files behind the chapter's interactives.  Everything
here is a *measurement* on committed artifacts plus the project's own code;
nothing is modelled, fitted or invented.

``book/site/data/ch03_candidates.json``   (the opening hook — the census)
    How many catalogue galaxies actually sit inside each detected event's
    localization ball, measured for all 1590 rows of the
    ``seed61000`` Cramer-Rao table against the reduced GLADE+ catalogue,
    using the production ball radius
    ``r = sigma_multiplier * sqrt(lambda_max(J Sigma J^T))``
    (``galaxy_catalogue/handler.py:558``, ``:617``) with the PRODUCTION
    ``sigma_multiplier = 1.5`` (``bayesian_statistics.py:2838``) and the
    production candidate redshift window ``get_redshift_outer_bounds``
    (``physical_relations.py:546``).

``book/site/data/ch03_skyball.json``      (I3.1 "Sky-Ball Explorer")
    Two real events -- EMRI-889 (the book's running example: the loudest
    event of the run, a ball 0.0126 deg across holding *two* galaxies) and
    event 1121 (SNR 35, a wide ball holding tens of thousands) -- with their
    real candidate galaxies, real rate weights ``w_g = R_eff(M_g)/(1+z_g)``
    (``bayesian_statistics.py:879``), the event's real 3-D Fisher Gaussian,
    and the exact per-h aggregate of the **point-kernel** numerator
    (Gray et al. 2020, arXiv:1908.06050, Eq. A.9) over *every* candidate.

``book/site/data/ch03_ratio.json``        (I3.2 "Ratio of Sums vs Sum of Ratios")
    The two algebraic forms evaluated on those same real candidate sets, plus
    the per-galaxy ``N_g`` / ``D_g`` that the project's OWN
    ``single_host_likelihood`` (``bayesian_statistics.py:3615``) returns for
    EMRI-889's two candidates, plus the run's own measured ``L_cat`` leg for
    event 889 from ``real_r1/diagnostics/event_likelihoods.csv``.

VENUE — read this before quoting any number from these files
------------------------------------------------------------
The seed-61000 *evaluation* ran against an OBSERVED-catalogue realization
(``absolute_marginal`` x ``volume_deconv``, observed catalogues
``realizations_20260729/observed_catalogue_seed90000{1..5}.csv``;
``REALISTIC_READOUT.md:1-11``).  Those realization CSVs are **not** in this
checkout.  The *injection* side -- which is where ``host_galaxy_index`` comes
from -- used the baseline reduced catalogue (campaign #53 convention (A):
the catalogue is TRUTH, the observed realization is what the estimator sees;
``realistic_host_observation_model.md`` §1.2, and ``main.py`` refuses to pair
an observed catalogue with any generative stage).

So the reconstruction in these files is on the **truth** catalogue: the same
catalogue 889's host was drawn from, and therefore the right venue for
"which galaxy could this have been".  It is **not** the catalogue the run's
own ``L_cat`` was computed against, and the two are *not* reconciled here.
Both are emitted, each labelled, and the difference is recorded in
``book/design/flags/ch03_FLAGS.md``.

Determinism: one fixed seed (``DISPLAY_SEED``) for the display subsample of
the large candidate set; every other number is a deterministic recomputation.
Read-only outside ``book/``.

Run as::

    /home/jasper/Repositories/MasterThesisCode/.venv/bin/python \\
        book/generators/gen_ch03.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WORKTREE_ROOT = Path(__file__).resolve().parents[2]


def _resolve_source_root() -> Path:
    """Pick the checkout that carries BOTH the current package and the data.

    This book lives in a git worktree of the main repository; the worktree's
    branch predates several helpers this generator calls
    (``_mass_redshift_prune_mask``), and the large artifacts it reads (the
    1.7 GB reduced catalogue, ``results/campaign51_20260728/...``) are
    working-tree-only files that exist in the main checkout.  Prefer a sibling
    ``MasterThesisCode`` checkout, fall back to this worktree, and probe for the
    capability rather than assuming it — no absolute paths, no silent import of
    a stale package.
    """
    here = Path(__file__).resolve().parents[2]
    for root in (here.parent / "MasterThesisCode", here):
        handler = root / "darksiren_emri" / "galaxy_catalogue" / "handler.py"
        if handler.is_file() and "_mass_redshift_prune_mask" in handler.read_text():
            return root
    raise RuntimeError(
        "gen_ch03: no checkout with a current darksiren_emri found "
        "(need galaxy_catalogue/handler.py:_mass_redshift_prune_mask)"
    )


REPO_ROOT = _resolve_source_root()
sys.path.insert(0, str(REPO_ROOT))

import astropy.units as u  # noqa: E402
from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import darksiren_emri.bayesian_inference.bayesian_statistics as bs  # noqa: E402
from darksiren_emri.constants import (  # noqa: E402
    HOST_DRAW_Z_MAX,
    M_SOURCE_FRAME_MAX,
    M_SOURCE_FRAME_MIN,
    OMEGA_M,
    SNR_THRESHOLD,
)
from darksiren_emri.constants import (  # noqa: E402
    H as H_TRUE,
)
from darksiren_emri.datamodels.detection import Detection  # noqa: E402
from darksiren_emri.emri_rate import R_eff_per_mbh  # noqa: E402
from darksiren_emri.galaxy_catalogue.handler import (  # noqa: E402
    HostGalaxy,
    InternalCatalogColumns,
    _empiric_stellar_mass_to_BH_mass_relation,
    _mass_redshift_prune_mask,
    _polar_to_cartesian,
    _reduced_catalog_column_names,
)
from darksiren_emri.physical_relations import (  # noqa: E402
    dist_to_redshift,
    dist_vectorized,
)

# --- repo-relative artifact paths (BOOK_DESIGN §4.2 rule 7; never absolute) ---
CATALOGUE_REL = Path("darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv")
CAMPAIGN_REL = Path("results/campaign51_20260728/realistic_20260729")
SEED_REL = CAMPAIGN_REL / "seed61000"
CRB_REL = SEED_REL / "prepared_cramer_rao_bounds.csv"
RUN_LOG_REL = SEED_REL / "mixture_leg_log_extract.txt"
DIAG_REL = SEED_REL / "real_r1" / "diagnostics" / "event_likelihoods.csv"
POOL_REL = CAMPAIGN_REL / "gate_b_20260730" / "injection_pool_mix200k_20260728"

OUT_DIR = WORKTREE_ROOT / "book" / "site" / "data"
OUT_CANDIDATES = OUT_DIR / "ch03_candidates.json"
OUT_SKYBALL = OUT_DIR / "ch03_skyball.json"
OUT_RATIO = OUT_DIR / "ch03_ratio.json"

# The two featured events.  889 = the book's running example (pedagogy B4).
# 1121 = the in-catalogue event with the largest candidate ball in this run,
# chosen by measurement (see `main`'s console summary), not by taste.
EVENT_GOLDEN = 889
EVENT_CROWDED = 1121
# Third event, ratio-block only: the candidate set with the largest measured
# spread in the per-galaxy selection factor D_g anywhere in this run.  Ratio of
# sums and mean of ratios can only differ where D_g differs, so this is the
# event on which the algebra is worth looking at.  Selected by measurement
# (max_g p_det - min_g p_det over all events with >= D_SPREAD_MIN_CAND
# candidates, at h = 0.60), not taste — and RE-VERIFIED at the production
# radius by `d_spread_scan` below, which raises if the pick is not the argmax.
EVENT_DSPREAD = 676
D_SPREAD_MIN_CAND = 50

# Production ball / window conventions, quoted from the code sites.
#
# REVISION 2026-07-31 (worklist D2): was `2`, the *signature default* of
# `handler.get_possible_hosts_from_ball_tree` (handler.py:568) — which production
# never uses.  The only production ball-search call site is
# `bayesian_statistics.py:2838` → `sigma_multiplier=1.5`.  The `2.0` in the call
# immediately above (`:2823`) is an argument to `get_redshift_outer_bounds`: a
# different multiplier for a different cut (the candidate z-window), which is why
# the two got crossed.  Do NOT read the handler signature default as production.
SIGMA_MULTIPLIER = 1.5  # bayesian_statistics.py:2838 (production call site)
H_PRIOR_MIN, H_PRIOR_MAX = 0.6, 0.86  # physical_relations.get_redshift_outer_bounds
NUMERATOR_SIGMA_MULTIPLIER = 4.0  # single_host_likelihood integration window

DISPLAY_MAX = 2000  # display subsample cap for the crowded event
DISPLAY_SEED = 20260731  # the generator's ONLY RNG use

# The production posterior grid is 41 points and NON-uniform (0.01 on
# [0.60,0.65] u [0.80,0.86], 0.005 between).  EMRI-889's fractional distance
# precision is 9.0e-4, so ONE 0.005 h-step slides its distance shell by ~7.6
# sigma_dL: on the production grid the point-kernel numerator of a single
# galaxy is invisible between samples.  The Sky-Ball Explorer therefore draws
# its own uniform display grid; every number that is compared with the pipeline
# still lives on the production grid.
DENSE_H_MIN, DENSE_H_MAX, DENSE_H_STEP = 0.60, 0.86, 0.0005
PER_GALAXY_CURVE_MAX = 8  # ship per-galaxy curves only for tiny candidate sets

# d_L(z; h) = d_L(z; h=1)/h exactly (c/H_0 prefactor); table for the browser.
DL_TABLE_N = 321
DL_TABLE_ZMAX = 1.6

# Recorded measurement (never recomputed here): the ratio-of-sums fix.
LEDGER_26 = {
    "commit": "816f904",
    "date": "2026-06-19/20",
    "map_1d_before": 0.750,
    "map_1d_after": 0.740,
    "map_2d_before": 0.7375,
    "map_2d_after": 0.7350,
    "artifact": "H0R:1119-1139",
    "note": (
        "L_cat departed from Gray A.9/A.10 in TWO ways at once: a spurious p_det in the "
        "numerator, and mean-of-ratios instead of ratio-of-sums. The joint fix halved the "
        "1D bias."
    ),
}


def _r(x: Any, sig: int = 7) -> float:
    """Round to `sig` significant digits (JSON size hygiene)."""
    v = float(x)
    if v == 0.0 or not np.isfinite(v):
        return v if np.isfinite(v) else 0.0
    return float(f"%.{sig}g" % v)


def _rl(a: Any, sig: int = 7) -> list[float]:
    return [_r(v, sig) for v in np.asarray(a, dtype=np.float64).ravel()]


_DATA_ROOTS = (REPO_ROOT, WORKTREE_ROOT, WORKTREE_ROOT.parent / "MasterThesisCode")


def _pool_dir() -> Path | None:
    """Locate the (untracked) production injection pool without a machine path."""
    for root in _DATA_ROOTS:
        candidate = root / POOL_REL
        if candidate.is_dir() and any(candidate.glob("injection_h_*_task_*.csv")):
            return candidate
    return None


def _catalogue_path() -> Path | None:
    for root in _DATA_ROOTS:
        candidate = root / CATALOGUE_REL
        if candidate.is_file():
            return candidate
    return None


def _data_file(rel: Path) -> Path:
    for root in _DATA_ROOTS:
        candidate = root / rel
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"gen_ch03: required artifact not found in any checkout: {rel}")


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    """Wrap an angle difference into (-pi, pi]."""
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Catalogue: one streaming pass, production prune, production ordering
# ---------------------------------------------------------------------------
def load_pruned_catalogue(path: Path) -> dict[str, np.ndarray]:
    """Load the reduced catalogue and apply the production prune, in file order.

    Mirrors ``GalaxyCatalogueHandler.__init__``'s pipeline for the columns this
    chapter needs: stellar mass -> BH mass (``_map_stellar_masses_to_BH_masses``),
    drop rows without mass (``_remove_galaxies_without_mass_information``), then
    the mass/redshift prune (``_get_pruned_galaxy_catalog`` via the shared
    predicate ``_mass_redshift_prune_mask``).  Row ORDER is preserved, so the
    resulting index is exactly the ``reset_index`` position that
    ``Detection.host_galaxy_index`` refers to.

    Sky angles stay in the on-disk EQUATORIAL frame here; the rotation to
    ecliptic is applied only to the small per-event patches (astropy on 22.6 M
    rows is neither necessary nor cheap, and angular separations are
    rotation-invariant).
    """
    names = _reduced_catalog_column_names()
    ra_parts, dec_parts, z_parts, ze_parts = [], [], [], []
    m_parts, me_parts, flag_parts = [], [], []
    for chunk in pd.read_csv(path, names=names, chunksize=4_000_000):
        bh, bh_err = _empiric_stellar_mass_to_BH_mass_relation(
            chunk[InternalCatalogColumns.BH_MASS], chunk[InternalCatalogColumns.BH_MASS_ERROR]
        )
        chunk[InternalCatalogColumns.BH_MASS] = bh
        chunk[InternalCatalogColumns.BH_MASS_ERROR] = bh_err
        chunk = chunk[chunk[InternalCatalogColumns.BH_MASS].notna()]
        keep = _mass_redshift_prune_mask(
            chunk[InternalCatalogColumns.BH_MASS],
            chunk[InternalCatalogColumns.BH_MASS_ERROR],
            chunk[InternalCatalogColumns.REDSHIFT],
            chunk[InternalCatalogColumns.REDSHIFT_ERROR],
            M_SOURCE_FRAME_MIN,
            M_SOURCE_FRAME_MAX,
            HOST_DRAW_Z_MAX,
        )
        chunk = chunk[keep]
        ra_parts.append(chunk["RIGHT_ASCENSION"].to_numpy(np.float64))
        dec_parts.append(chunk["DECLINATION"].to_numpy(np.float64))
        z_parts.append(chunk[InternalCatalogColumns.REDSHIFT].to_numpy(np.float64))
        ze_parts.append(chunk[InternalCatalogColumns.REDSHIFT_ERROR].to_numpy(np.float64))
        m_parts.append(chunk[InternalCatalogColumns.BH_MASS].to_numpy(np.float64))
        me_parts.append(chunk[InternalCatalogColumns.BH_MASS_ERROR].to_numpy(np.float64))
        flag_parts.append(chunk[InternalCatalogColumns.REDSHIFT_FLAG].to_numpy(np.int8))
    return {
        "ra": np.concatenate(ra_parts),
        "dec": np.concatenate(dec_parts),
        "z": np.concatenate(z_parts),
        "z_err": np.concatenate(ze_parts),
        "M": np.concatenate(m_parts),
        "M_err": np.concatenate(me_parts),
        "flag": np.concatenate(flag_parts),
    }


def event_sky_equatorial(crb: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the events' ECLIPTIC (phi, theta) into equatorial RA/Dec (deg).

    The CRB rows are stamped ``ecliptic_BarycentricTrue_J2000`` (COORD-03); the
    catalogue on disk is equatorial ICRS.  Rotating 1590 event positions is the
    cheap direction of the same astropy transform the handler applies to the
    catalogue (``_rotate_equatorial_to_ecliptic``), and angular separations are
    invariant under it.
    """
    theta = crb["qS"].to_numpy(np.float64)
    phi = crb["phiS"].to_numpy(np.float64)
    coord = SkyCoord(
        lon=(np.degrees(phi) % 360.0) * u.deg,
        lat=(90.0 - np.degrees(theta)) * u.deg,
        frame=BarycentricTrueEcliptic(equinox="J2000"),
    ).transform_to("icrs")
    return coord.ra.to(u.deg).value % 360.0, coord.dec.to(u.deg).value


def ball_radii(crb: pd.DataFrame) -> np.ndarray:
    """Production sky-search radius per event (chord length on the unit sphere).

    ``r = sigma_multiplier * sqrt(lambda_max(J Sigma J^T))``,
    ``J = diag(|sin theta|, 1)``, ``Sigma`` the 2x2 Fisher sky block --
    ``galaxy_catalogue/handler.py:519`` (COORD-04).
    """
    s_phi2 = crb["delta_phiS_delta_phiS"].to_numpy(np.float64)
    s_theta2 = crb["delta_qS_delta_qS"].to_numpy(np.float64)
    c = crb["delta_phiS_delta_qS"].to_numpy(np.float64)
    j = np.abs(np.sin(crb["qS"].to_numpy(np.float64)))
    a = j * j * s_phi2
    b = j * c
    d = s_theta2
    lam = 0.5 * (a + d) + np.sqrt(np.maximum(0.25 * (a - d) ** 2 + b * b, 0.0))
    return SIGMA_MULTIPLIER * np.sqrt(np.maximum(lam, 0.0))


def candidate_z_window(d_l: float, sigma_d_l: float) -> tuple[float, float]:
    """The production candidate redshift window for one event.

    ``get_redshift_outer_bounds`` (``physical_relations.py:546``) maps
    ``d_L -+ 3 sigma`` through the EXTREMES of the h prior, so the candidate list
    is built once and is the same for every trial h; the h-sweep then re-weights
    the members, it does not re-select them.  ``bayesian_statistics.py:2816-2826``
    additionally caps ``z_max`` at the analysis' redshift limit.
    """
    lo = d_l - 3.0 * sigma_d_l
    z_min = 0.0 if lo < 0 else float(dist_to_redshift(lo, H_PRIOR_MIN))
    z_max = float(dist_to_redshift(d_l + 3.0 * sigma_d_l, H_PRIOR_MAX))
    return z_min, min(z_max, HOST_DRAW_Z_MAX)


# ---------------------------------------------------------------------------
# Step 1 — the census
# ---------------------------------------------------------------------------
def build_census(
    cat: dict[str, np.ndarray], crb: pd.DataFrame
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Count real candidates per event and return the per-event index lists."""
    xyz = _polar_to_cartesian(np.radians(90.0 - cat["dec"]), np.radians(cat["ra"]))
    tree = cKDTree(xyz, balanced_tree=False, compact_nodes=False)
    ra_e, dec_e = event_sky_equatorial(crb)
    ev_xyz = _polar_to_cartesian(np.radians(90.0 - dec_e), np.radians(ra_e))
    radii = ball_radii(crb)
    d_l = crb["luminosity_distance"].to_numpy(np.float64)
    s_dl = np.sqrt(crb["delta_luminosity_distance_delta_luminosity_distance"].to_numpy(np.float64))

    n_ball = np.zeros(len(crb), dtype=np.int64)
    n_cand = np.zeros(len(crb), dtype=np.int64)
    cand_lists: list[np.ndarray] = []
    for i in range(len(crb)):
        idx = np.asarray(tree.query_ball_point(ev_xyz[i], radii[i]), dtype=np.int64)
        n_ball[i] = idx.size
        if idx.size:
            z_min, z_max = candidate_z_window(float(d_l[i]), float(s_dl[i]))
            keep = (z_min <= cat["z"][idx] + cat["z_err"][idx]) & (
                z_max >= cat["z"][idx] - cat["z_err"][idx]
            )
            idx = idx[keep]
        n_cand[i] = idx.size
        cand_lists.append(idx)

    def _hist(vals: np.ndarray) -> dict[str, list[float]]:
        # log-spaced bins from 1; the zero bin is reported separately.
        edges = np.concatenate([[0.5], np.logspace(0.0, np.log10(max(vals.max(), 10)) + 0.2, 22)])
        counts, _ = np.histogram(vals[vals > 0], bins=edges)
        return {"edges": _rl(edges, 5), "counts": [int(c) for c in counts]}

    pcts = [5, 25, 50, 75, 90, 95, 99]
    census: dict[str, Any] = {
        "meta": {
            "what": "candidate galaxies per detected event, measured",
            "catalogue": "darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv "
            "(baseline reduced GLADE+, production prune)",
            "crb": "results/.../realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv",
            "n_catalogue_rows_pruned": int(len(cat["z"])),
            "ball_rule": "handler.py:617 r = 1.5*sqrt(lambda_max(J Sigma J^T)), chord on unit "
            "sphere; n_sigma = 1.5 from the production call site "
            "bayesian_statistics.py:2838 (the handler signature default 2 is NOT production)",
            "z_window_rule": "physical_relations.py:546 get_redshift_outer_bounds, h in [0.60, 0.86]",
            "venue_note": "baseline (truth) catalogue; the run's evaluation used an "
            "observed-catalogue realization not present in this checkout",
        },
        "n_events": int(len(crb)),
        "n_ball": {
            "hist": _hist(n_ball),
            "n_zero": int((n_ball == 0).sum()),
            "percentiles": {str(p): _r(np.percentile(n_ball, p), 6) for p in pcts},
            "mean": _r(n_ball.mean(), 6),
            "max": int(n_ball.max()),
        },
        "n_cand": {
            "hist": _hist(n_cand),
            "n_zero": int((n_cand == 0).sum()),
            "percentiles": {str(p): _r(np.percentile(n_cand, p), 6) for p in pcts},
            "mean": _r(n_cand.mean(), 6),
            "max": int(n_cand.max()),
        },
        "scatter": {
            "snr": _rl(crb["SNR"].to_numpy(), 5),
            "radius_deg": _rl(np.degrees(2.0 * np.arcsin(np.clip(radii / 2.0, 0, 1))), 5),
            "n_cand": [int(v) for v in n_cand],
            "in_catalog": [bool(v) for v in crb["in_catalog"].to_numpy()],
        },
        "featured": {
            str(EVENT_GOLDEN): {
                "n_ball": int(n_ball[EVENT_GOLDEN]),
                "n_cand": int(n_cand[EVENT_GOLDEN]),
            },
            str(EVENT_CROWDED): {
                "n_ball": int(n_ball[EVENT_CROWDED]),
                "n_cand": int(n_cand[EVENT_CROWDED]),
            },
        },
        "n_in_catalog": int(crb["in_catalog"].sum()),
    }
    return census, cand_lists


def concentration_census(
    cat: dict[str, np.ndarray],
    crb: pd.DataFrame,
    cand_lists: list[np.ndarray],
    h_eval: float,
) -> dict[str, Any]:
    """How concentrated is each event's weighted numerator, at h = h_eval?

    For every event with at least one candidate, evaluate the Gray (A.9)
    point-kernel numerator on ALL of its candidates and report (a) the largest
    single galaxy's share of ``sum_g w_g N_g`` and (b) the participation ratio
    ``(sum x)^2 / sum x^2`` -- the effective number of galaxies carrying the
    sum.  This is the measurement behind "a single event rarely identifies a
    host": it says how often one galaxy actually dominates.
    """
    top_share: list[float] = []
    n_eff: list[float] = []
    idx_ok: list[int] = []
    for i, cand_idx in enumerate(cand_lists):
        if cand_idx.size == 0:
            continue
        row = crb.iloc[i]
        try:
            det = Detection(row)
            _, cov_inv, log_norm = gaussian_3d(det)
        except (RuntimeError, np.linalg.LinAlgError):
            continue
        coord = SkyCoord(
            ra=cat["ra"][cand_idx] * u.deg, dec=cat["dec"][cand_idx] * u.deg, frame="icrs"
        ).transform_to(BarycentricTrueEcliptic(equinox="J2000"))
        phi_g = np.radians(coord.lon.to(u.deg).value % 360.0)
        theta_g = -(np.radians(coord.lat.to(u.deg).value) - np.pi / 2.0)
        z_g = cat["z"][cand_idx]
        w_g = np.asarray(R_eff_per_mbh(cat["M"][cand_idx]), dtype=np.float64) / (1.0 + z_g)
        log_n = point_kernel_log_numerator(det, cov_inv, log_norm, phi_g, theta_g, z_g, h_eval)
        log_wn = np.log(w_g) + log_n
        m = log_wn.max()
        x = np.exp(log_wn - m)
        s = x.sum()
        top_share.append(float(x.max() / s))
        n_eff.append(float(s * s / np.sum(x * x)))
        idx_ok.append(i)
    ts = np.asarray(top_share)
    ne = np.asarray(n_eff)
    return {
        "h_eval": _r(h_eval, 4),
        "n_events_with_candidates": int(ts.size),
        "frac_top_share_above_half": _r(float(np.mean(ts > 0.5)), 4),
        "frac_top_share_above_90pct": _r(float(np.mean(ts > 0.9)), 4),
        "top_share_percentiles": {str(p): _r(np.percentile(ts, p), 4) for p in (5, 25, 50, 75, 95)},
        "n_eff_percentiles": {str(p): _r(np.percentile(ne, p), 5) for p in (5, 25, 50, 75, 95)},
        "n_eff_median": _r(np.median(ne), 5),
        "note": "Gray (A.9) point-kernel numerator on the truth catalogue, evaluated at the "
        "injected h; the participation ratio is (sum x)^2 / sum x^2 over x_g = w_g N_g",
    }


# ---------------------------------------------------------------------------
# Step 2 — the two featured events
# ---------------------------------------------------------------------------
def gaussian_3d(det: Detection) -> tuple[np.ndarray, np.ndarray, float]:
    """The event's 3-D GW Gaussian in fraction coordinates (phi, theta, d_L/d_L^det).

    Transcribed from the production assembly in ``bayesian_statistics.evaluate``
    (lines 2393-2465): the covariance is the Fisher sky block plus the distance
    row/column divided by ``d_L`` so the third coordinate is the *fraction*
    ``u = d_L/d_L^det`` with mean 1.
    """
    cov = np.array(
        [
            [det.phi_error**2, det.theta_phi_covariance, det.d_L_phi_covariance / det.d_L],
            [det.theta_phi_covariance, det.theta_error**2, det.d_L_theta_covariance / det.d_L],
            [
                det.d_L_phi_covariance / det.d_L,
                det.d_L_theta_covariance / det.d_L,
                det.d_L_uncertainty**2 / det.d_L**2,
            ],
        ]
    )
    cov_inv = np.linalg.pinv(cov)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise RuntimeError("non-positive-definite 3-D covariance — event unusable")
    log_norm = -0.5 * (3.0 * np.log(2.0 * np.pi) + logdet)
    return cov, cov_inv, float(log_norm)


def point_kernel_log_numerator(
    det: Detection,
    cov_inv: np.ndarray,
    log_norm: float,
    phi_g: np.ndarray,
    theta_g: np.ndarray,
    z_g: np.ndarray,
    h: float,
) -> np.ndarray:
    """log N_g in the DELTA-kernel (point) limit of Gray (2020) Eq. (A.9).

    ``N_g = N_3((phi_g, theta_g, d_L(z_g,h)/d_L^det); (phi_det, theta_det, 1), Sigma_3)``
    -- the galaxy's catalogue redshift taken at face value.  This is exactly the
    production ``generator_marginal`` numerator (``bayesian_statistics.py:3697``
    ``_use_generator_point``); the production default ``volume_deconv`` replaces
    the delta by an integral over a host-z kernel, which is Chapter 7's subject.
    """
    d_l = dist_vectorized(z_g, h=h)
    delta = np.stack(
        [
            _wrap_pi(phi_g - det.phi),
            theta_g - det.theta,
            d_l / det.d_L - 1.0,
        ],
        axis=1,
    )
    quad = np.einsum("ni,ij,nj->n", delta, cov_inv, delta)
    return log_norm - 0.5 * quad


def build_event_block(
    cat: dict[str, np.ndarray],
    crb: pd.DataFrame,
    idx_event: int,
    cand_idx: np.ndarray,
    h_grid: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    row = crb.iloc[idx_event]
    det = Detection(row)
    cov, cov_inv, log_norm = gaussian_3d(det)
    radius = float(ball_radii(crb)[idx_event])
    z_min, z_max = candidate_z_window(det.d_L, det.d_L_uncertainty)

    # Candidate galaxies, rotated into the ecliptic frame the Gaussian lives in.
    coord = SkyCoord(
        ra=cat["ra"][cand_idx] * u.deg, dec=cat["dec"][cand_idx] * u.deg, frame="icrs"
    ).transform_to(BarycentricTrueEcliptic(equinox="J2000"))
    phi_g = np.radians(coord.lon.to(u.deg).value % 360.0)
    theta_g = -(np.radians(coord.lat.to(u.deg).value) - np.pi / 2.0)
    z_g = cat["z"][cand_idx]
    ze_g = cat["z_err"][cand_idx]
    m_g = cat["M"][cand_idx]
    flag_g = cat["flag"][cand_idx]
    w_g = np.asarray(R_eff_per_mbh(m_g), dtype=np.float64) / (1.0 + z_g)

    host_row = int(row["host_galaxy_index"])
    is_host = cand_idx == host_row

    # h at which each galaxy's point-kernel numerator peaks: the h that puts the
    # measured distance exactly on that galaxy.  d_L(z; h) = d_L(z; 1)/h exactly,
    # so h*(g) = d_L(z_g; 1) / d_L^det.  "Which h does this galaxy vote for?"
    h_star = dist_vectorized(z_g, h=1.0) / det.d_L
    # ... and how wide that vote is, from the galaxy's own redshift error alone
    # (half-width, evaluated the same way) versus from the GW distance error.
    h_star_sigma = (
        dist_vectorized(np.maximum(z_g + ze_g, 0.0), h=1.0)
        - dist_vectorized(np.maximum(z_g - ze_g, 0.0), h=1.0)
    ) / (2.0 * det.d_L)
    h_star_sigma_gw = h_star * (det.d_L_uncertainty / det.d_L)

    # ---- exact per-h aggregates over EVERY candidate ----------------------
    log_sum_wn, n_eff, n_in_window, z_shell = [], [], [], []
    top_share = []
    per_galaxy_log_n: list[np.ndarray] | None = (
        [] if cand_idx.size <= PER_GALAXY_CURVE_MAX else None
    )
    for h in h_grid:
        log_n = point_kernel_log_numerator(det, cov_inv, log_norm, phi_g, theta_g, z_g, float(h))
        if per_galaxy_log_n is not None:
            per_galaxy_log_n.append(log_n.copy())
        log_wn = np.log(w_g) + log_n
        m = log_wn.max()
        wn = np.exp(log_wn - m)
        s = wn.sum()
        log_sum_wn.append(m + np.log(s))
        n_eff.append(float(s * s / np.sum(wn * wn)))
        top_share.append(float(wn.max() / s))
        d_l = dist_vectorized(z_g, h=float(h))
        n_in_window.append(
            int(np.sum(np.abs(d_l - det.d_L) <= NUMERATOR_SIGMA_MULTIPLIER * det.d_L_uncertainty))
        )
        z_shell.append(float(dist_to_redshift(det.d_L, float(h))))

    # ---- display subsample (seeded; the aggregates above are exact) -------
    n_cand = int(cand_idx.size)
    if n_cand > DISPLAY_MAX:
        pick = np.sort(rng.choice(n_cand, size=DISPLAY_MAX, replace=False))
        if is_host.any() and not is_host[pick].any():
            pick[0] = int(np.flatnonzero(is_host)[0])
            pick = np.sort(pick)
    else:
        pick = np.arange(n_cand)

    # Sky offsets in arcmin, with the |sin theta| great-circle rescaling on phi
    # (the same Jacobian the ball radius uses) so the panel is not distorted.
    d_phi = _wrap_pi(phi_g[pick] - det.phi) * abs(np.sin(det.theta))
    d_theta = theta_g[pick] - det.theta

    return {
        "idx": idx_event,
        "label": f"EMRI-{idx_event}",
        "snr": _r(row["SNR"], 6),
        "d_L_Gpc": _r(det.d_L, 7),
        "sigma_d_L_Gpc": _r(det.d_L_uncertainty, 6),
        "sigma_d_L_over_d_L": _r(det.d_L_uncertainty / det.d_L, 6),
        "phi": _r(det.phi, 9),
        "theta": _r(det.theta, 9),
        "sigma_phi": _r(det.phi_error, 6),
        "sigma_theta": _r(det.theta_error, 6),
        "M_z": _r(det.M, 7),
        "in_catalog": bool(row["in_catalog"]),
        "host_row": host_row,
        "host_found_in_ball": bool(is_host.any()),
        "radius_chord": _r(radius, 7),
        "radius_deg": _r(np.degrees(2.0 * np.arcsin(min(radius / 2.0, 1.0))), 6),
        "radius_arcmin": _r(60.0 * np.degrees(2.0 * np.arcsin(min(radius / 2.0, 1.0))), 6),
        "solid_angle_deg2": _r(
            np.degrees(1.0) ** 2 * np.pi * (2.0 * np.arcsin(min(radius / 2.0, 1.0))) ** 2, 5
        ),
        "z_window": [_r(z_min, 6), _r(z_max, 6)],
        "n_cand": n_cand,
        "cov_inv_3d": [_rl(r_, 9) for r_ in cov_inv],
        "cov_3d": [_rl(r_, 9) for r_ in cov],
        "log_norm_3d": _r(log_norm, 10),
        "display": {
            "n_shown": int(pick.size),
            "seeded_subsample": bool(n_cand > DISPLAY_MAX),
            "d_phi_arcmin": _rl(np.degrees(d_phi) * 60.0, 6),
            "d_theta_arcmin": _rl(np.degrees(d_theta) * 60.0, 6),
            "phi": _rl(phi_g[pick], 9),
            "theta": _rl(theta_g[pick], 9),
            "z": _rl(z_g[pick], 6),
            "z_err": _rl(ze_g[pick], 5),
            "M": _rl(m_g[pick], 5),
            "w": _rl(w_g[pick], 6),
            "spec_z": [int(v) for v in (flag_g[pick] == 3).astype(int)],
            "is_host": [int(v) for v in is_host[pick].astype(int)],
            "h_star": _rl(h_star[pick], 7),
            "h_star_sigma_from_z": _rl(h_star_sigma[pick], 5),
            "h_star_sigma_from_GW": _rl(h_star_sigma_gw[pick], 5),
            "sep_arcmin": _rl(60.0 * np.degrees(np.sqrt(d_phi**2 + d_theta**2)), 5),
        },
        "per_galaxy_log_N": (
            [_rl(row_, 8) for row_ in np.asarray(per_galaxy_log_n).T]
            if per_galaxy_log_n is not None
            else None
        ),
        "z_hist": _z_hist(z_g, w_g),
        "agg": {
            "log_sum_wN": _rl(log_sum_wn, 8),
            "n_eff": _rl(n_eff, 5),
            "top_share": _rl(top_share, 5),
            "n_in_window": [int(v) for v in n_in_window],
            "z_shell": _rl(z_shell, 7),
        },
    }


def _z_hist(z_g: np.ndarray, w_g: np.ndarray) -> dict[str, list[float]]:
    """Rate-weighted redshift histogram of the candidate set (exact, all members)."""
    if z_g.size == 0:
        return {"edges": [], "counts": [], "weight": []}
    lo, hi = float(z_g.min()), float(z_g.max())
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, min(61, max(4, z_g.size + 1)))
    counts, _ = np.histogram(z_g, bins=edges)
    weight, _ = np.histogram(z_g, bins=edges, weights=w_g)
    return {"edges": _rl(edges, 7), "counts": [int(c) for c in counts], "weight": _rl(weight, 5)}


# ---------------------------------------------------------------------------
# Step 3 — ratio of sums vs sum of ratios, on the real candidate sets
# ---------------------------------------------------------------------------
D_FLOORS = [0.0, 1e-6, 1e-4, 1e-2]

# Deliberate, labelled distortion of the MEASURED per-galaxy selection factors:
#   ln D_g(lambda) = <ln D> + lambda * (ln D_g - <ln D>),  <.> = w-weighted mean.
# lambda = 1 is the measured set.  lambda = 0 makes every D_g equal, where the
# two algebraic forms are identical by construction -- so the widget can *prove*
# the identity rather than assert it.  lambda > 1 exaggerates the real spread.
# Anything other than lambda = 1 is chipped `toy` on the page.
D_SPREAD_LAMBDAS = [0.0, 0.5, 1.0, 2.0, 3.0]


def build_ratio_block(
    cat: dict[str, np.ndarray],
    crb: pd.DataFrame,
    idx_event: int,
    cand_idx: np.ndarray,
    h_grid: np.ndarray,
    dp: Any,
) -> dict[str, Any]:
    """The two forms, in the Gray (A.9) point-kernel limit, on real galaxies.

    ``N_g`` is the point-evaluated GW Gaussian; ``D_g = p_det(d_L(z_g, h))`` is
    the same delta-kernel limit of the per-host selection integral that
    ``precompute_global_catalog_selection`` uses
    (``bayesian_statistics.py:493-656``, docstring 524).  Both forms use the
    identical ``N_g``, ``D_g``, ``w_g``; only the algebra differs.
    """
    row = crb.iloc[idx_event]
    det = Detection(row)
    _, cov_inv, log_norm = gaussian_3d(det)
    coord = SkyCoord(
        ra=cat["ra"][cand_idx] * u.deg, dec=cat["dec"][cand_idx] * u.deg, frame="icrs"
    ).transform_to(BarycentricTrueEcliptic(equinox="J2000"))
    phi_g = np.radians(coord.lon.to(u.deg).value % 360.0)
    theta_g = -(np.radians(coord.lat.to(u.deg).value) - np.pi / 2.0)
    z_g = cat["z"][cand_idx]
    w_g = np.asarray(R_eff_per_mbh(cat["M"][cand_idx]), dtype=np.float64) / (1.0 + z_g)

    log_ros: list[float] = []
    log_mor: dict[str, list[float]] = {f"{f:g}": [] for f in D_FLOORS}
    lam_ros: dict[str, list[float]] = {f"{lam:g}": [] for lam in D_SPREAD_LAMBDAS}
    lam_mor: dict[str, list[float]] = {f"{lam:g}": [] for lam in D_SPREAD_LAMBDAS}
    d_min, d_max, d_frac_zero = [], [], []
    for h in h_grid:
        log_n = point_kernel_log_numerator(det, cov_inv, log_norm, phi_g, theta_g, z_g, float(h))
        d_l = dist_vectorized(z_g, h=float(h))
        d_g = np.asarray(
            dp.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_l, phi_g, theta_g, h=float(h)
            ),
            dtype=np.float64,
        )
        log_w = np.log(w_g)
        m = (log_w + log_n).max()
        wn = np.exp(log_w + log_n - m)
        wd = w_g * d_g
        # ratio of sums (Gray A.9/A.10, production `weighted_ratio_of_sums`)
        log_ros.append(m + np.log(wn.sum()) - np.log(wd.sum()) if wd.sum() > 0 else float("-inf"))
        # sum (mean) of ratios — the refuted form, per floor on D_g
        for f in D_FLOORS:
            dd = np.maximum(d_g, f)
            ok = dd > 0
            if not ok.any():
                log_mor[f"{f:g}"].append(float("-inf"))
                continue
            terms = np.exp(log_w[ok] + log_n[ok] - m) / dd[ok]
            log_mor[f"{f:g}"].append(m + np.log(terms.sum()) - np.log(w_g[ok].sum()))
        # lambda-scaled spread of the measured D_g (see D_SPREAD_LAMBDAS)
        safe = np.maximum(d_g, 1e-12)
        ln_d = np.log(safe)
        ln_d_bar = float(np.sum(w_g * ln_d) / np.sum(w_g))
        for lam in D_SPREAD_LAMBDAS:
            d_lam = np.exp(ln_d_bar + lam * (ln_d - ln_d_bar))
            lam_ros[f"{lam:g}"].append(m + np.log(wn.sum()) - np.log(np.sum(w_g * d_lam)))
            terms = np.exp(log_w + log_n - m) / d_lam
            lam_mor[f"{lam:g}"].append(m + np.log(terms.sum()) - np.log(w_g.sum()))

        d_min.append(float(d_g.min()))
        d_max.append(float(d_g.max()))
        d_frac_zero.append(float(np.mean(d_g <= 0.0)))

    ros_arr = np.asarray(log_ros)
    mor_arr = np.asarray(log_mor["0"])
    diff = ros_arr - mor_arr
    return {
        "idx": idx_event,
        "n_cand": int(cand_idx.size),
        "log_ratio_of_sums": _rl(log_ros, 8),
        "log_mean_of_ratios": {k: _rl(v, 8) for k, v in log_mor.items()},
        "lambda_ratio_of_sums": {k: _rl(v, 8) for k, v in lam_ros.items()},
        "lambda_mean_of_ratios": {k: _rl(v, 8) for k, v in lam_mor.items()},
        # The two log-legs run down to -2.9e4 in the wings of a sharp event, so
        # 8-significant-figure storage cannot resolve their O(1e-3) DIFFERENCE.
        # Emit the difference itself, at full working precision, and let the
        # page read it from here instead of subtracting two rounded arrays.
        "log_form_difference": {
            k: _rl(np.asarray(lam_ros[k]) - np.asarray(lam_mor[k]), 6) for k in lam_ros
        },
        "lambda_identity_residual": _r(
            np.max(np.abs(np.asarray(lam_ros["0"]) - np.asarray(lam_mor["0"]))), 3
        ),
        "D_g_min": _rl(d_min, 5),
        "D_g_max": _rl(d_max, 5),
        "D_g_frac_zero": _rl(d_frac_zero, 5),
        "summary": {
            # The two forms are identical when every D_g is equal, so what
            # matters is the h-DEPENDENCE of their ratio, not its level.
            "ln_form_ratio_spread": _r(diff.max() - diff.min(), 4),
            "ln_form_ratio_spread_by_lambda": {
                k: _r(
                    (np.asarray(lam_ros[k]) - np.asarray(lam_mor[k])).max()
                    - (np.asarray(lam_ros[k]) - np.asarray(lam_mor[k])).min(),
                    4,
                )
                for k in lam_ros
            },
            "argmax_ratio_of_sums": _r(h_grid[int(np.argmax(ros_arr))], 4),
            "argmax_mean_of_ratios": _r(h_grid[int(np.argmax(mor_arr))], 4),
            "D_g_spread_at_h060": _r(max(d_max[0] - d_min[0], 0.0), 4),
            "D_g_ratio_at_h060": _r(d_max[0] / d_min[0] if d_min[0] > 0 else float("inf"), 4),
        },
    }


def d_spread_scan(
    cat: dict[str, np.ndarray],
    cand_lists: list[np.ndarray],
    dp: Any,
    h: float = 0.60,
) -> list[tuple[int, float, float, float, int]]:
    """Rank every event by the spread of its per-galaxy selection factor D_g.

    Ratio-of-sums and mean-of-ratios are algebraically identical when the D_g
    are equal, so the event worth looking at is the one whose candidates differ
    most in detectability.  This makes the page's "selected by measurement"
    claim checkable *at the production radius* rather than inherited from an
    earlier, wider ball.  Returns (event, max-min, min, max, n_cand) sorted by
    spread, descending.
    """
    out: list[tuple[int, float, float, float, int]] = []
    for i, cand_idx in enumerate(cand_lists):
        if cand_idx.size < D_SPREAD_MIN_CAND:
            continue
        coord = SkyCoord(
            ra=cat["ra"][cand_idx] * u.deg, dec=cat["dec"][cand_idx] * u.deg, frame="icrs"
        ).transform_to(BarycentricTrueEcliptic(equinox="J2000"))
        phi_g = np.radians(coord.lon.to(u.deg).value % 360.0)
        theta_g = -(np.radians(coord.lat.to(u.deg).value) - np.pi / 2.0)
        d_l = dist_vectorized(cat["z"][cand_idx], h=h)
        d_g = np.asarray(
            dp.detection_probability_without_bh_mass_interpolated_zero_fill(
                d_l, phi_g, theta_g, h=h
            ),
            dtype=np.float64,
        )
        lo, hi = float(d_g.min()), float(d_g.max())
        out.append((i, hi - lo, lo, hi, int(cand_idx.size)))
    out.sort(key=lambda t: (-t[1], t[0]))
    return out


def global_catalogue_denominator(h_grid: np.ndarray) -> np.ndarray:
    """Sigma_global(h) = sum over the WHOLE catalogue of w_g p_det(d_L(z_g,h)).

    Read from the run's own log lines (``bayesian_statistics.py:2335``,
    ``sum_w_Dg(no_bh)``, 5 s.f.) rather than recomputed: it is the denominator
    the run actually divided by, and recomputing it would silently substitute a
    different catalogue (the run's was the observed realization).
    """
    import re

    text = _data_file(RUN_LOG_REL).read_text()
    table: dict[float, float] = {}
    pattern = re.compile(r"h_(0_\d+)\.log.*sum_w_Dg\(no_bh\)=([0-9.eE+-]+)")
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            table[round(float(m.group(1).replace("_", ".")), 6)] = float(m.group(2))
    missing = [h for h in h_grid if round(float(h), 6) not in table]
    if missing:
        raise RuntimeError(f"Sigma_global missing for h = {missing}")
    return np.array([table[round(float(h), 6)] for h in h_grid])


def production_legs_889(
    cat: dict[str, np.ndarray],
    crb: pd.DataFrame,
    cand_idx: np.ndarray,
    h_grid: np.ndarray,
    dp: Any,
    measured_l_cat: np.ndarray,
) -> dict[str, Any]:
    """Per-galaxy N_g / D_g from the project's OWN `single_host_likelihood`.

    The production entry point is a module-global-driven worker; we install the
    globals through the shipped initializer ``child_process_init`` with a
    one-detection table, then call ``single_host_likelihood`` exactly as
    ``p_Di`` does, in the run's own mode (``absolute_marginal``, which carries
    the ``volume_deconv`` host-z kernel — ``REALISTIC_READOUT.md:1-11``).
    """
    row = crb.iloc[EVENT_GOLDEN]
    det = Detection(row)
    _, cov_inv, log_norm = gaussian_3d(det)
    bs.child_process_init(
        1e-6,
        HOST_DRAW_Z_MAX,
        M_SOURCE_FRAME_MIN,
        M_SOURCE_FRAME_MAX,
        dp,
        np.array([[det.phi, det.theta, 1.0]]),
        cov_inv[None, :, :],
        np.array([log_norm]),
        np.zeros((1, 4)),
        np.zeros((1, 4, 4)),
        np.zeros(1),
        {EVENT_GOLDEN: 0},
        np.zeros(1),
        np.zeros((1, 3)),
        np.array([det.d_L]),
        np.array([det.d_L_uncertainty]),
        np.array([det.M]),
        np.array([det.phi]),
        np.array([det.theta]),
    )
    coord = SkyCoord(
        ra=cat["ra"][cand_idx] * u.deg, dec=cat["dec"][cand_idx] * u.deg, frame="icrs"
    ).transform_to(BarycentricTrueEcliptic(equinox="J2000"))
    phi_g = np.radians(coord.lon.to(u.deg).value % 360.0)
    theta_g = -(np.radians(coord.lat.to(u.deg).value) - np.pi / 2.0)
    hosts = [
        HostGalaxy.from_attributes(
            phiS=float(phi_g[k]),
            qS=float(theta_g[k]),
            z=float(cat["z"][cand_idx[k]]),
            z_error=float(cat["z_err"][cand_idx[k]]),
            M=float(cat["M"][cand_idx[k]]),
            M_error=float(cat["M_err"][cand_idx[k]]),
            catalog_index=int(cand_idx[k]),
        )
        for k in range(cand_idx.size)
    ]
    weights = [float(bs._rate_weight(g)) for g in hosts]
    n_g = np.zeros((len(hosts), h_grid.size))
    d_g = np.zeros((len(hosts), h_grid.size))
    for j, h in enumerate(h_grid):
        for k, g in enumerate(hosts):
            res = bs.single_host_likelihood(
                g.phiS,
                g.qS,
                g.z,
                g.z_error,
                g.M,
                g.M_error,
                EVENT_GOLDEN,
                float(h),
                False,
                normalization_mode="absolute_marginal",
            )
            n_g[k, j] = res[0]
            d_g[k, j] = res[1]
    w = np.asarray(weights)[:, None]
    ros = (w * n_g).sum(axis=0) / (w * d_g).sum(axis=0)
    mor = (w * n_g / np.maximum(d_g, 1e-300)).sum(axis=0) / w.sum()
    # The RUN's own assembly: absolute_marginal divides the SAME local numerator
    # sum by the GLOBAL catalogue selection sum (bayesian_statistics.py:3190),
    # not by the local one — G2c §4.1 vs §4.2.
    sigma_global = global_catalogue_denominator(h_grid)
    l_cat_global = (w * n_g).sum(axis=0) / sigma_global
    ratio_to_measured = np.where(measured_l_cat > 0, l_cat_global / measured_l_cat, np.nan)
    wings = np.r_[0:8, 33:41]  # grid points away from the peak region
    return {
        "mode": "absolute_marginal (volume_deconv host-z kernel) — the run's own mode",
        "sigma_global": _rl(sigma_global, 6),
        "L_cat_absolute_marginal": _rl(l_cat_global, 7),
        "vs_measured": {
            "ratio": _rl(ratio_to_measured, 5),
            "argmax_reconstruction": _r(h_grid[int(np.argmax(l_cat_global))], 4),
            "argmax_measured": _r(h_grid[int(np.argmax(measured_l_cat))], 4),
            "wing_ratio_min": _r(np.nanmin(ratio_to_measured[wings]), 4),
            "wing_ratio_max": _r(np.nanmax(ratio_to_measured[wings]), 4),
            "note": "reconstruction on the TRUTH catalogue vs the run's own leg on its "
            "observed-catalogue realization; not reconciled (ch03_FLAGS.md F-ch03-3)",
        },
        "rows": [int(i) for i in cand_idx],
        "is_host": [int(i == int(row["host_galaxy_index"])) for i in cand_idx],
        "z": _rl([cat["z"][i] for i in cand_idx], 7),
        "z_err": _rl([cat["z_err"][i] for i in cand_idx], 5),
        "spec_z": [int(cat["flag"][i] == 3) for i in cand_idx],
        "M": _rl([cat["M"][i] for i in cand_idx], 6),
        "w": _rl(weights, 6),
        "N_g": [_rl(rr, 6) for rr in n_g],
        "D_g": [_rl(rr, 6) for rr in d_g],
        "L_cat_ratio_of_sums": _rl(ros, 7),
        "L_cat_mean_of_ratios": _rl(mor, 7),
    }


# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cat_path = _catalogue_path()
    if cat_path is None:
        print(
            "NOTICE: reduced_galaxy_catalogue.csv not found (untracked, ~1.7 GB) — "
            "leaving the committed ch03_*.json untouched.",
            flush=True,
        )
        return

    crb = pd.read_csv(_data_file(CRB_REL))
    diag = pd.read_csv(_data_file(DIAG_REL))
    h_grid = np.sort(diag["h"].unique())

    t0 = time.monotonic()
    cat = load_pruned_catalogue(cat_path)
    print(f"  catalogue: {len(cat['z']):,} pruned rows in {time.monotonic() - t0:.1f}s", flush=True)

    t0 = time.monotonic()
    census, cand_lists = build_census(cat, crb)
    print(
        f"  census: median ball {census['n_ball']['percentiles']['50']}, "
        f"median candidates {census['n_cand']['percentiles']['50']}, "
        f"max {census['n_cand']['max']} in {time.monotonic() - t0:.1f}s",
        flush=True,
    )

    # Sanity gate: EMRI-889's host must be recoverable in the baseline frame.
    host_889 = int(crb.iloc[EVENT_GOLDEN]["host_galaxy_index"])
    m_lift = cat["M"][host_889] * (1.0 + cat["z"][host_889])
    rel = abs(m_lift - crb.iloc[EVENT_GOLDEN]["M"]) / crb.iloc[EVENT_GOLDEN]["M"]
    if rel > 1e-5:
        raise RuntimeError(
            f"host-frame gate FAILED: pruned row {host_889} lifts to M_z={m_lift:.6g}, "
            f"CRB says {crb.iloc[EVENT_GOLDEN]['M']:.6g} (rel {rel:.2e})"
        )
    print(f"  host-frame gate: M_g*(1+z_g) reproduces M_z to {rel:.2e}", flush=True)

    t0 = time.monotonic()
    census["concentration"] = concentration_census(cat, crb, cand_lists, float(H_TRUE))
    print(
        f"  concentration: one galaxy carries >50% of the numerator for "
        f"{100 * census['concentration']['frac_top_share_above_half']:.1f}% of events, "
        f"median n_eff {census['concentration']['n_eff_median']} "
        f"({time.monotonic() - t0:.1f}s)",
        flush=True,
    )

    census["host_frame_gate"] = {
        "event": EVENT_GOLDEN,
        "host_row": host_889,
        "M_g": _r(cat["M"][host_889], 8),
        "z_g": _r(cat["z"][host_889], 8),
        "M_g_lifted": _r(m_lift, 8),
        "M_z_from_CRB": _r(crb.iloc[EVENT_GOLDEN]["M"], 8),
        "relative_difference": _r(rel, 3),
    }
    OUT_CANDIDATES.write_text(json.dumps(census, separators=(",", ":")))

    rng = np.random.default_rng(DISPLAY_SEED)
    z_tab = np.linspace(0.0, DL_TABLE_ZMAX, DL_TABLE_N)
    dense_h = np.round(
        np.arange(DENSE_H_MIN, DENSE_H_MAX + 0.5 * DENSE_H_STEP, DENSE_H_STEP),
        6,
    )
    skyball: dict[str, Any] = {
        "meta": {
            "h_true": H_TRUE,
            "Omega_m": OMEGA_M,
            "snr_threshold": SNR_THRESHOLD,
            "kernel": "Gray et al. (2020) arXiv:1908.06050 Eq. (A.9) point (delta) limit",
            "venue": "baseline (truth) reduced GLADE+ catalogue + seed61000 CRB row; the "
            "run's own L_cat used an observed-catalogue realization not in this checkout",
            "numerator_window_sigma": NUMERATOR_SIGMA_MULTIPLIER,
        },
        "h_grid": _rl(dense_h, 6),
        "production_h_grid": _rl(h_grid, 5),
        "dl_table": {"z": _rl(z_tab, 6), "dl_h1_Gpc": _rl(dist_vectorized(z_tab, h=1.0), 7)},
        "events": {},
    }
    for idx_event in (EVENT_GOLDEN, EVENT_CROWDED):
        skyball["events"][str(idx_event)] = build_event_block(
            cat, crb, idx_event, cand_lists[idx_event], dense_h, rng
        )
    OUT_SKYBALL.write_text(json.dumps(skyball, separators=(",", ":")))

    # ---- ratio block (needs the untracked injection pool for p_det) --------
    pool = _pool_dir()
    ratio: dict[str, Any] = {
        "meta": {
            "what": "ratio of sums vs mean of ratios on real candidate sets",
            "forms": {
                "ratio_of_sums": "sum_g w_g N_g / sum_g w_g D_g  (Gray A.9/A.10; "
                "bayesian_statistics.py:804 weighted_ratio_of_sums)",
                "mean_of_ratios": "sum_g w_g (N_g/D_g) / sum_g w_g  (the refuted form; ledger #26)",
            },
            "venue": skyball["meta"]["venue"],
        },
        "h_grid": _rl(h_grid, 5),
        "ledger_26": LEDGER_26,
        "measured_889": {
            "source": "seed61000/real_r1/diagnostics/event_likelihoods.csv",
            "L_cat_no_bh": _rl(
                diag[diag.event_idx == EVENT_GOLDEN].sort_values("h")["L_cat_no_bh"].to_numpy(), 7
            ),
            "note": "the RUN's own catalogue leg for event 889, computed against its "
            "observed-catalogue realization — a different catalogue from the "
            "reconstruction above; the two are not reconciled here (ch03_FLAGS.md F-ch03-3)",
        },
        "events": {},
    }
    if pool is None:
        print(
            "NOTICE: injection pool not found (untracked) — ch03_ratio.json written "
            "without the p_det-dependent legs.",
            flush=True,
        )
        ratio["meta"]["degraded"] = "no injection pool: D_g legs omitted"
    else:
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            SimulationDetectionProbability,
        )

        t0 = time.monotonic()
        dp = SimulationDetectionProbability(
            injection_data_dir=str(pool), snr_threshold=SNR_THRESHOLD
        )
        print(f"  p_det built in {time.monotonic() - t0:.1f}s", flush=True)

        # Re-verify the featured D_g-spread event AT THE PRODUCTION RADIUS.
        t0 = time.monotonic()
        rank = d_spread_scan(cat, cand_lists, dp)
        top = rank[:5]
        print(
            f"  D_g-spread ranking (h=0.60, >={D_SPREAD_MIN_CAND} candidates): "
            + " | ".join(f"{ev}: {lo:.3f}->{hi:.3f} (n={n})" for ev, _, lo, hi, n in top)
            + f"  ({time.monotonic() - t0:.1f}s)",
            flush=True,
        )
        if top and top[0][0] != EVENT_DSPREAD:
            raise RuntimeError(
                f"EVENT_DSPREAD = {EVENT_DSPREAD} is no longer the measured argmax of the "
                f"D_g spread (now event {top[0][0]}, spread {top[0][1]:.4f} vs "
                f"{next((t[1] for t in rank if t[0] == EVENT_DSPREAD), float('nan')):.4f}). "
                "Update EVENT_DSPREAD and every number the page quotes for it, or drop the "
                "'selected by measurement' claim — do not ship the stale pick."
            )
        ratio["meta"]["d_spread_selection"] = {
            "criterion": "max_g p_det - min_g p_det at h = 0.60 over events with "
            f">= {D_SPREAD_MIN_CAND} candidates, at the production ball radius",
            "n_events_scanned": len(rank),
            "top5": [
                {
                    "event": ev,
                    "spread": _r(sp, 4),
                    "p_det_min": _r(lo, 4),
                    "p_det_max": _r(hi, 4),
                    "n_cand": n,
                }
                for ev, sp, lo, hi, n in top
            ],
        }
        for idx_event in (EVENT_GOLDEN, EVENT_CROWDED, EVENT_DSPREAD):
            ratio["events"][str(idx_event)] = build_ratio_block(
                cat, crb, idx_event, cand_lists[idx_event], h_grid, dp
            )
            ratio["events"][str(idx_event)]["snr"] = _r(crb.iloc[idx_event]["SNR"], 5)
            ratio["events"][str(idx_event)]["d_L_Gpc"] = _r(
                crb.iloc[idx_event]["luminosity_distance"], 6
            )
            ratio["events"][str(idx_event)]["z_range"] = [
                _r(cat["z"][cand_lists[idx_event]].min(), 5),
                _r(cat["z"][cand_lists[idx_event]].max(), 5),
            ]
        ratio["production_889"] = production_legs_889(
            cat,
            crb,
            cand_lists[EVENT_GOLDEN],
            h_grid,
            dp,
            np.asarray(ratio["measured_889"]["L_cat_no_bh"], dtype=np.float64),
        )
    OUT_RATIO.write_text(json.dumps(ratio, separators=(",", ":")))

    for path in (OUT_CANDIDATES, OUT_SKYBALL, OUT_RATIO):
        print(f"  wrote {path.name}: {path.stat().st_size / 1024:.1f} KB", flush=True)
    print(f"gen_ch03 done in {time.monotonic() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
