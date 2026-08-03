"""Generator for Chapter 6 — "Opening the Black Box: What the Waveform Actually Measures".

Produces the two data files behind the chapter's interactives, plus the
running-example dossier numbers.

``book/site/data/ch06_fisher.json``   (I6.1 "The Fisher Ellipse Forge")
    Everything the ellipse widget needs, from the run's OWN Cramer-Rao table
    (``seed61000/prepared_cramer_rao_bounds.csv``, 1590 rows):

    * the population-level evidence for the *fraction* coordinate: the
      distribution of ``sigma_u * SNR``, ``sigma_phi * SNR``,
      ``sigma_theta * SNR`` and ``sigma_Mz/Mz * SNR`` over all 1590 events.
      ``sigma_u = sigma_dL/dL`` collapses onto 1/SNR (median 1.04, p95/p5 =
      1.29); the sky widths do NOT (p95/p5 = 5.4 and 9.2).  That contrast is
      the chapter's argument for fraction coordinates, measured rather than
      asserted, AND the honest bound on the widget's own SNR rescaling.
    * the conditioning of the **14x14** covariance -- the gate's actual
      operand.  ``FISHER_CONDITION_NUMBER_MAX = 1e14`` is applied to the
      14x14 *Fisher* at simulation time, and for a symmetric positive
      definite matrix ``cond_2(Gamma^-1) = cond_2(Gamma)`` exactly, so the
      stored covariance's condition number IS the gated quantity (up to the
      inversion's own error).  Measured, not copied: the reviewer's expected
      values were median 2.6e9 / p95 1.4e10 / max 3.9e12 (tomas M8).
    * the measured correlation structure of the stored 3x3 covariance
      (r_theta_phi, r_phi_u, r_theta_u), including the Spearman rank
      correlation of |r| against SNR.
    * per-event blocks for four real events: 889 (SNR 1425, in-catalogue, the
      book's running example), 361 (SNR 168, in-catalogue, the most
      sky-correlated event in the run, r_theta_phi = +0.975), 606 (SNR 43,
      dark, the Ch 5 counterpart) and 555 (SNR 28.3, the population median).
      Each carries the full 3x3 fraction covariance assembled EXACTLY as
      ``bayesian_statistics.py:2389-2437`` assembles it, the Bishop 2.81-2.82
      conditional (``sigma_cond``, ``proj``) exactly as ``:2495-2510``, the
      production BallTree search radius
      ``1.5 * sqrt(lambda_max(J Sigma_sky J^T))`` (``handler.py:575-578``),
      and a real GLADE+ sky patch around the event.
    * for each event, a 40-point SNR grid on which the browser reads off
      *measured* candidate counts: the number of catalogue galaxies inside the
      search ball, with and without the sky-sky covariance ``C_theta_phi``,
      before and after the redshift window; plus the total-variation
      displacement of the candidate weight when the covariance is factorized.
      The counts are exact over the whole 4.5-degree cone; the plotted scatter
      is the nearest 2000 galaxies (a display sample, labelled as such).

``book/site/data/ch06_dt2.json``      (I6.2 "The dt^2 Switch")
    The detected-population redshift histogram under the two inner-product
    conventions, computed on the production injection pool
    (``gate_b_20260730/injection_pool_mix200k_20260728``, stratum ``a`` = the
    99,014 rows that carry the population measure).  Because the pre-fix code
    returned ``<h1|h2>/dt^2`` exactly (G8 section 1.2, verified to machine
    precision on three independent references), the pre-fix *selection* is
    exactly ``SNR/10 >= 20`` on the same injections at ``dt = 10 s`` — a
    counterfactual on one pool, not a re-simulation, and the page says so.

PROVENANCE / SCOPE NOTES
------------------------
1. ``sigma_dL/dL`` for event 889.  **RESOLVED 2026-07-31** by author mandate
   (``REVISION_WORKLIST.md`` section A-D1): the measured fraction
   ``8.98e-4`` is now the book-wide spec value.  The CRB row gives
   ``sigma_dL = 7.984e-5 **Gpc**`` at ``d_L = 0.0888792 Gpc``, i.e. a
   *fraction* of ``8.983e-4``; the old spec figure ``8.0e-5`` was that
   absolute Gpc value wearing a fractional label.  This generator is
   unchanged: it always emitted the two as separate, correctly-named keys
   (``sigma_dL_Gpc`` in Gpc and ``sigma_u`` dimensionless), which are two
   quantities in two units, not two candidate values for one quantity.  The
   page now prints the corrected fraction plus a one-line erratum; the
   history stays in ``book/design/flags/ch06_FLAGS.md`` (F-ch06-1, F-ch06-2).
2. The galaxy patch is cut from the committed baseline
   ``reduced_galaxy_catalogue.csv``.  Per ``BOOK_DESIGN.md`` section 4.2 rule
   5 that file differs from the campaign-#53 realization parent in exactly
   one column, ``z_error`` — so it is licensed for positions and redshifts
   (which is all the counts here need at the geometry level) but is NOT the
   production candidate list, and the page states this.
3. ``physical_relations.get_redshift_outer_bounds`` accepts a
   ``sigma_multiplier`` argument that its body never uses (it hardcodes
   ``3 *``, ``physical_relations.py:563-566``), so production's
   ``sigma_multiplier=1.5``/``2.0`` call sites get 3 sigma.  This generator
   calls the function itself, so it inherits the real behaviour.

DATA AVAILABILITY
-----------------
The CRB table is git-tracked and present in any checkout of this branch, so
the population + per-event Fisher blocks always rebuild.  The 1.7 GB reduced
galaxy catalogue and the 200k-row injection pool are NOT tracked (they live
in the main checkout's working tree), so those steps resolve them from, in
order: this repo root, then a sibling ``MasterThesisCode`` checkout.  If a
source is missing the already-committed JSON is left untouched and a NOTICE
is printed — the generator never fails a build over an untracked artifact and
never writes a partial or silently-degraded file.

Determinism: no RNG anywhere.  Every number is read or recomputed from
committed artifacts with the production package's own functions.  Read-only
outside ``book/``.

Run as::

    /home/jasper/Repositories/MasterThesisCode/.venv/bin/python \\
        book/generators/gen_ch06.py
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from master_thesis_code.constants import (  # noqa: E402
    HOST_DRAW_Z_MAX,
    M_SOURCE_FRAME_MAX,
    M_SOURCE_FRAME_MIN,
    SNR_THRESHOLD,
)
from master_thesis_code.constants import (  # noqa: E402
    H as H_TRUE,
)
from master_thesis_code.galaxy_catalogue.handler import (  # noqa: E402
    _empiric_stellar_mass_to_BH_mass_relation,
    _polar_to_cartesian,
    _reduced_catalog_column_names,
)
from master_thesis_code.physical_relations import (  # noqa: E402
    dist,
    get_redshift_outer_bounds,
)

try:  # newer package revisions expose the prune predicate as a helper
    from master_thesis_code.galaxy_catalogue.handler import (  # noqa: E402
        _mass_redshift_prune_mask,
    )
except ImportError:  # pragma: no cover - depends on the checked-out package revision

    def _mass_redshift_prune_mask(  # type: ignore[misc]
        bh_mass: pd.Series,
        bh_mass_error: pd.Series,
        redshift: pd.Series,
        redshift_error: pd.Series,
        M_min: float,
        M_max: float,
        z_max: float,
    ) -> pd.Series:
        """Verbatim transcription of ``GalaxyCatalogueHandler._get_pruned_galaxy_catalog``
        for package revisions that still inline the predicate (identical
        expression, identical inclusive boundaries)."""
        return (
            (bh_mass + bh_mass_error >= M_min)
            & (bh_mass - bh_mass_error <= M_max)
            & (redshift - redshift_error <= z_max)
        )

# --- repo-relative artifact paths (BOOK_DESIGN section 4.2 rule 7) ---------
CAMPAIGN_REL = Path("results/campaign51_20260728/realistic_20260729")
SEED_REL = CAMPAIGN_REL / "seed61000"
CRB_REL = SEED_REL / "prepared_cramer_rao_bounds.csv"
POOL_REL = CAMPAIGN_REL / "gate_b_20260730" / "injection_pool_mix200k_20260728"
CATALOGUE_REL = Path("master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv")

OUT_DIR = Path(__file__).resolve().parent.parent / "site" / "data"
OUT_FISHER = OUT_DIR / "ch06_fisher.json"
OUT_DT2 = OUT_DIR / "ch06_dt2.json"

# The four real events the widget offers. 889 = the book's running example
# (loudest, in-catalogue); 361 = the most sky-correlated in-catalogue event in
# the run (r_theta_phi = +0.975); 606 = the Ch 5 dark counterpart; 555 = the
# population-median SNR event.
EVENT_IDS = [889, 361, 606, 555]

# Production BallTree call: bayesian_statistics.py:2838 passes
# sigma_multiplier=1.5 explicitly, and it is the ONLY production ball-search
# call site.  Do NOT use handler.get_possible_hosts_from_ball_tree's signature
# default of 2 (handler.py:568) -- production never uses it, and the 2.0 at
# :2823 is a different multiplier for a different cut (the redshift window,
# get_redshift_outer_bounds).  gen_ch03 was measured at the signature default
# until 2026-07-31; see ch03_FLAGS.md F-ch03-2 and REVISION_WORKLIST.md §A-D2.
SIGMA_MULTIPLIER = 1.5
# Production redshift cap: REDSHIFT_UPPER_LIMIT = cosmological_model.max_redshift
# (bayesian_statistics.py:2194, :2826) = HOST_DRAW_Z_MAX.
REDSHIFT_UPPER_LIMIT = HOST_DRAW_Z_MAX

# Sky patch cut from the raw catalogue: a true angular cone of this radius
# (deg). Must exceed the widest search radius any event reaches on the SNR
# grid (largest is event 606 at SNR 20: 3.67 deg); the generator asserts this
# and refuses to emit counts it cannot support.
PATCH_RADIUS_DEG = 4.5
# Nearest-N galaxies shipped for the scatter display (counts stay exact over
# the whole patch).
PATCH_SCATTER_N = 2000
# SNR grid the slider indexes (log-spaced, endpoints pinned).
SNR_GRID = [
    20.0, 22.0, 24.5, 27.0, 30.0, 33.5, 37.0, 41.5, 46.0, 51.5, 57.0, 64.0,
    71.0, 79.5, 88.0, 99.0, 110.0, 123.0, 137.0, 153.0, 171.0, 191.0, 214.0,
    239.0, 267.0, 299.0, 334.0, 373.0, 417.0, 466.0, 521.0, 582.0, 651.0,
    728.0, 813.0, 909.0, 1016.0, 1136.0, 1270.0, 1424.7236072062765,
]

# The 14 waveform parameters, in the CRB table's own column order.  The
# stored lower triangle is named ``delta_<later>_delta_<earlier>``, so this
# order is what reassembles the full 14x14 covariance.
CRB_PARAMS_14 = [
    "M", "mu", "a", "p0", "e0", "x0", "luminosity_distance",
    "qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0",
]

# Babak et al. (2017), arXiv:1703.09722 -- this project's own EMRI population
# reference (it is already cited by G8 evidence line L5 for the SNR-20
# horizon).  Its Fisher forecasts quote fractional redshifted-mass precisions
# of order 1e-5 to 1e-6 at comparable SNR.  This is a LITERATURE FIGURE, not a
# recomputation: it is carried as a citation so the page can price its own
# most extreme number against a published one, and the verdict the page draws
# is explicitly "not tested here".
BABAK_2017_MASS_PRECISION = {
    "reference": "Babak et al. (2017), arXiv:1703.09722",
    "quantity": "fractional redshifted-mass precision Delta(ln M_z) from EMRI Fisher forecasts",
    "range_low": 1e-6,
    "range_high": 1e-5,
    "kind": "literature citation, not recomputed here",
    "verdict": (
        "This run is 10-100x better than the published forecasts at comparable SNR. "
        "Whether that is the AK-vs-numerical-derivative difference, the mission "
        "duration, or optimism of the high-SNR Fisher approximation is NOT TESTED HERE."
    ),
}

# G8 evidence line L5: the measured PRE-fix detected population (seed600,
# 500 events, nominal SNR >= 20). Carried as a RECORDED measurement to sit
# beside this generator's counterfactual — never blended with it.
G8_L5_RECORDED = {
    "venue": "seed600, 500 events, nominal SNR >= 20 (pre-dt^2 code era)",
    "z_median": 0.046,
    "z_p90": 0.074,
    "z_max": 0.109,
    "d_L_max_Gpc": 0.485,
    "source": "G8_dt2_inner_product_derivation.md, evidence line L5",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _r(x: Any, sig: int = 7) -> float:
    """Round to `sig` significant digits (JSON size hygiene)."""
    v = float(x)
    if v == 0.0 or not np.isfinite(v):
        return 0.0 if v == 0.0 else float(v)
    return float(round(v, sig - 1 - int(math.floor(math.log10(abs(v))))))


def _quantiles(a: npt_like) -> dict[str, float]:  # type: ignore[valid-type]
    q = np.percentile(np.asarray(a, dtype=float), [5, 25, 50, 75, 95])
    return {
        "p5": _r(q[0]),
        "p25": _r(q[1]),
        "median": _r(q[2]),
        "p75": _r(q[3]),
        "p95": _r(q[4]),
        "spread_p95_over_p5": _r(q[4] / q[0]) if q[0] != 0 else None,
    }


npt_like = Any  # tiny alias so the annotation above stays readable


def _resolve(rel: Path) -> Path | None:
    """Find an artifact in this repo root, else in a sibling MasterThesisCode."""
    here = REPO_ROOT / rel
    if here.exists():
        return here
    sibling = REPO_ROOT.parent / "MasterThesisCode" / rel
    if sibling.exists():
        return sibling
    return None


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


# --------------------------------------------------------------------------
# I6.1 — the Fisher / CRB block
# --------------------------------------------------------------------------
def _cov3_fraction(row: pd.Series) -> np.ndarray:
    """The 3x3 fraction-coordinate covariance, assembled exactly as
    ``bayesian_statistics.py:2389-2409`` assembles it from the CRB row."""
    d_L = float(row["luminosity_distance"])
    s_phi2 = float(row["delta_phiS_delta_phiS"])
    s_th2 = float(row["delta_qS_delta_qS"])
    s_dl2 = float(row["delta_luminosity_distance_delta_luminosity_distance"])
    c_tp = float(row["delta_phiS_delta_qS"])
    c_dp = float(row["delta_phiS_delta_luminosity_distance"])
    c_dt = float(row["delta_qS_delta_luminosity_distance"])
    return np.array(
        [
            [s_phi2, c_tp, c_dp / d_L],
            [c_tp, s_th2, c_dt / d_L],
            [c_dp / d_L, c_dt / d_L, s_dl2 / d_L**2],
        ]
    )


def _cov4_fraction(row: pd.Series) -> np.ndarray:
    """The 4x4 fraction covariance (``bayesian_statistics.py:2411-2437``)."""
    d_L = float(row["luminosity_distance"])
    M = float(row["M"])
    c3 = _cov3_fraction(row)
    c_mp = float(row["delta_phiS_delta_M"]) / M
    c_mt = float(row["delta_qS_delta_M"]) / M
    c_md = float(row["delta_luminosity_distance_delta_M"]) / d_L / M
    s_m2 = float(row["delta_M_delta_M"]) / M**2
    cov = np.zeros((4, 4))
    cov[:3, :3] = c3
    cov[3, :3] = cov[:3, 3] = [c_mp, c_mt, c_md]
    cov[3, 3] = s_m2
    return cov


def _cov14(crb: pd.DataFrame) -> np.ndarray:
    """The full 14x14 covariances, stacked, exactly as the CRB table stores
    them: the lower triangle lives in ``delta_<i>_delta_<j>`` columns over
    ``CRB_PARAMS_14``, in the table's own parameter order and units (masses in
    solar masses, distance in Gpc, angles in radians).  No rescaling: the
    condition number of the *stored* matrix is what the pipeline's gate sees.

    Returns an ``(n, 14, 14)`` array; raises if a column is missing, so a
    schema change fails the build instead of silently degrading the number.
    """
    n = len(crb)
    out = np.zeros((n, 14, 14))
    for i, a in enumerate(CRB_PARAMS_14):
        for j, b in enumerate(CRB_PARAMS_14[: i + 1]):
            col = f"delta_{a}_delta_{b}"
            if col not in crb.columns:
                raise SystemExit(f"gen_ch06: CRB table is missing column {col!r}")
            v = crb[col].to_numpy(dtype=float)
            out[:, i, j] = v
            out[:, j, i] = v
    return out


def _conditioning_stats(crb: pd.DataFrame) -> dict[str, Any]:
    """Condition numbers of the stored 14x14 covariance and of the assembled
    3D and 4D blocks derived from it, against the two gates the pipeline
    actually applies:

    * ``FISHER_CONDITION_NUMBER_MAX = 1e14`` on the 14x14 **Fisher**, at
      simulation time (``parameter_estimation.py:441-452``; ledger #11,
      ``d17230d``) — an event failing it never reaches this table at all.
      Sigma = Gamma^-1 and both are symmetric positive definite, so in exact
      arithmetic ``cond_2(Sigma_14) == cond_2(Gamma_14)``: the 14x14 number
      below IS the gated quantity, which is why it is the one worth showing;
    * ``fisher_cond_threshold = 1e16`` on the assembled 3D/4D **covariance**,
      at inference time (``bayesian_statistics.py:1964, :2443-2444``).
    """
    c3, c4 = [], []
    for _, row in crb.iterrows():
        c3.append(float(np.linalg.cond(_cov3_fraction(row))))
        c4.append(float(np.linalg.cond(_cov4_fraction(row))))
    a3 = np.array(c3)
    a4 = np.array(c4)

    cov14 = _cov14(crb)
    a14 = np.linalg.cond(cov14)
    eig14 = np.linalg.eigvalsh(cov14)
    # float64 carries ~16 significant digits; a condition number kappa costs
    # roughly log10(kappa) of them in the inverse.
    digits_worst = 16.0 - math.log10(float(a14.max()))

    return {
        "cond14": _quantiles(a14),
        "cond14_max": _r(float(a14.max()), 5),
        "cond14_min": _r(float(a14.min()), 5),
        "cond14_n_above_1e14": int((a14 > 1e14).sum()),
        "cond14_equals_fisher_cond": (
            "Sigma = Gamma^-1, both symmetric positive definite, so "
            "cond_2(Sigma_14) = cond_2(Gamma_14) in exact arithmetic"
        ),
        "cond14_all_positive_definite": bool((eig14.min(axis=1) > 0).all()),
        "cond14_float64_digits_left_worst_case": _r(digits_worst, 3),
        "cond3": _quantiles(a3),
        "cond4": _quantiles(a4),
        "cond3_max": _r(float(a3.max()), 5),
        "cond4_max": _r(float(a4.max()), 5),
        "n_excluded_at_1e16": int(((a3 > 1e16) | (a4 > 1e16)).sum()),
        "inference_threshold": 1e16,
        "simulation_fisher_threshold": 1e14,
    }


def _sky_search_radius(row: pd.Series, drop_corr: bool, snr_scale: float = 1.0) -> float:
    """``sigma_multiplier * sqrt(lambda_max(J Sigma_sky J^T))``
    (``handler.py:575-578``).  ``snr_scale`` = SNR_ref / SNR multiplies every
    1-sigma width (Fisher ~ SNR^2 under an amplitude-only rescaling)."""
    s_phi = math.sqrt(float(row["delta_phiS_delta_phiS"])) * snr_scale
    s_th = math.sqrt(float(row["delta_qS_delta_qS"])) * snr_scale
    c_tp = 0.0 if drop_corr else float(row["delta_phiS_delta_qS"]) * snr_scale**2
    theta = float(row["qS"])
    sigma = np.array([[s_phi**2, c_tp], [c_tp, s_th**2]])
    jac = np.diag([abs(math.sin(theta)), 1.0])
    lam = float(np.linalg.eigvalsh(jac @ sigma @ jac.T).max())
    return SIGMA_MULTIPLIER * math.sqrt(max(lam, 0.0))


def _load_patch(catalogue: Path, targets: list[tuple[float, float]]) -> list[pd.DataFrame]:
    """One streaming pass over the raw reduced catalogue, cutting a true
    angular cone of radius ``PATCH_RADIUS_DEG`` around each (ra0, dec0) target
    in the file's own equatorial frame.  A rotation preserves angles, so the
    cone is exactly a cone in the ecliptic frame too."""
    names = _reduced_catalog_column_names()
    keep: list[list[pd.DataFrame]] = [[] for _ in targets]
    unit = [
        np.array(
            [
                math.cos(math.radians(dec0)) * math.cos(math.radians(ra0)),
                math.cos(math.radians(dec0)) * math.sin(math.radians(ra0)),
                math.sin(math.radians(dec0)),
            ]
        )
        for ra0, dec0 in targets
    ]
    cos_r = math.cos(math.radians(PATCH_RADIUS_DEG))
    n_rows = 0
    for chunk in pd.read_csv(catalogue, names=names, header=None, chunksize=2_000_000):
        n_rows += len(chunk)
        ra = np.radians(chunk["RIGHT_ASCENSION"].to_numpy())
        dec = np.radians(chunk["DECLINATION"].to_numpy())
        xyz = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
        for i, vec in enumerate(unit):
            mask = xyz @ vec >= cos_r
            if mask.any():
                keep[i].append(chunk[mask])
    print(f"    catalogue rows scanned: {n_rows:,}")
    return [pd.concat(parts, ignore_index=True) for parts in keep]


def _prepare_patch(raw: pd.DataFrame) -> pd.DataFrame:
    """Run the raw patch through the production catalogue pipeline:
    stellar->BH mass, equatorial->ecliptic rotation, polar mapping, drop
    massless rows, mass/redshift prune (``handler.py:346-352``)."""
    from astropy import units as u  # noqa: PLC0415
    from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord  # noqa: PLC0415

    df = raw.copy()
    bh_mass, bh_mass_error = _empiric_stellar_mass_to_BH_mass_relation(
        df["STELLAR_MASS"], df["STELLAR_MASS_ABSOULTE_ERROR"]
    )
    df["BH_MASS"] = bh_mass
    df["BH_MASS_ERROR"] = bh_mass_error

    coord = SkyCoord(
        ra=df["RIGHT_ASCENSION"].to_numpy() * u.deg,
        dec=df["DECLINATION"].to_numpy() * u.deg,
        frame="icrs",
    ).transform_to(BarycentricTrueEcliptic(equinox="J2000"))
    lon = coord.lon.to(u.deg).value % 360.0
    lat = coord.lat.to(u.deg).value
    df["PHI_S"] = np.radians(lon)
    df["THETA_S"] = (np.radians(lat) - math.pi / 2.0) * (-1.0)

    df = df[~df["BH_MASS"].isna()]
    mask = _mass_redshift_prune_mask(
        df["BH_MASS"],
        df["BH_MASS_ERROR"],
        df["REDSHIFT"],
        df["REDSHIFT_MEASUREMENT_ERROR"],
        M_SOURCE_FRAME_MIN,
        M_SOURCE_FRAME_MAX,
        HOST_DRAW_Z_MAX,
    )
    return df[mask].reset_index(drop=True)


def _mvn_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """log N(x; mean, cov) for a stack of row vectors (mirrors
    ``bayesian_statistics._mvn_pdf``: pinv + slogdet)."""
    inv = np.linalg.pinv(cov)
    _sign, logdet = np.linalg.slogdet(cov)
    d = x - mean
    quad = np.einsum("ij,jk,ik->i", d, inv, d)
    return -0.5 * (x.shape[1] * math.log(2 * math.pi) + logdet + quad)


def build_fisher(crb: pd.DataFrame, catalogue: Path | None) -> dict[str, Any]:
    d_L = crb["luminosity_distance"].to_numpy()
    snr = crb["SNR"].to_numpy()
    s_phi = np.sqrt(crb["delta_phiS_delta_phiS"].to_numpy())
    s_th = np.sqrt(crb["delta_qS_delta_qS"].to_numpy())
    s_u = np.sqrt(crb["delta_luminosity_distance_delta_luminosity_distance"].to_numpy()) / d_L
    s_mz = np.sqrt(crb["delta_M_delta_M"].to_numpy()) / crb["M"].to_numpy()
    c_tp = crb["delta_phiS_delta_qS"].to_numpy()
    d_omega = 2 * math.pi * np.abs(np.sin(crb["qS"].to_numpy())) * np.sqrt(
        np.clip(s_phi**2 * s_th**2 - c_tp**2, 0.0, None)
    )
    r_tp = c_tp / (s_phi * s_th)
    r_pu = (crb["delta_phiS_delta_luminosity_distance"].to_numpy() / d_L) / (s_phi * s_u)
    r_tu = (crb["delta_qS_delta_luminosity_distance"].to_numpy() / d_L) / (s_th * s_u)
    r_sky_u = np.hypot(r_pu, r_tu)
    in_cat = crb["in_catalog"].to_numpy().astype(bool)

    # Downsampled (SNR, sigma_u) scatter for the "1/SNR" panel: every event,
    # but rounded — 1590 pairs is ~30 KB, cheap and complete (no sampling).
    scatter = {
        "snr": [_r(v, 5) for v in snr],
        "sigma_u": [_r(v, 5) for v in s_u],
        "in_catalog": [bool(v) for v in in_cat],
    }

    payload: dict[str, Any] = {
        "meta": {
            "run": "campaign51_20260728/realistic_20260729/seed61000",
            "crb_file": str(CRB_REL),
            "n_rows": int(len(crb)),
            "n_in_catalog": int(in_cat.sum()),
            "snr_threshold": float(SNR_THRESHOLD),
            "h_true": float(H_TRUE),
            "sigma_multiplier": SIGMA_MULTIPLIER,
            "coord_frame": str(crb["_coord_frame"].iloc[0]),
            "cov_frame": str(crb["_cov_frame"].iloc[0]),
            "mission_T_yr": _r(crb["T"].iloc[0], 4),
            "dt_s": _r(crb["dt"].iloc[0], 4),
        },
        "population": {
            "snr": {
                "min": _r(snr.min()),
                "p5": _r(np.percentile(snr, 5)),
                "median": _r(np.median(snr)),
                "p95": _r(np.percentile(snr, 95)),
                "max": _r(snr.max()),
            },
            "scaling": {
                "sigma_u_x_snr": _quantiles(s_u * snr),
                "sigma_phi_x_snr": _quantiles(s_phi * snr),
                "sigma_theta_x_snr": _quantiles(s_th * snr),
                "sqrt_dOmega_x_snr": _quantiles(np.sqrt(d_omega) * snr),
                "sigma_Mz_frac_x_snr": _quantiles(s_mz * snr),
            },
            "correlations": {
                "abs_r_theta_phi": _quantiles(np.abs(r_tp)),
                "abs_r_phi_u": _quantiles(np.abs(r_pu)),
                "abs_r_theta_u": _quantiles(np.abs(r_tu)),
                "r_sky_u": _quantiles(r_sky_u),
                "frac_abs_r_theta_phi_gt_0p3": _r(float((np.abs(r_tp) > 0.3).mean()), 4),
                "max_abs_r_theta_phi": _r(float(np.abs(r_tp).max()), 4),
                "max_r_sky_u": _r(float(r_sky_u.max()), 4),
                "spearman_snr_vs_r_sky_u": _r(_spearman(snr, r_sky_u), 3),
                "spearman_snr_vs_abs_r_theta_phi": _r(_spearman(snr, np.abs(r_tp)), 3),
                "r_sky_u_median_in_catalog": _r(float(np.median(r_sky_u[in_cat])), 4),
                "r_sky_u_median_dark": _r(float(np.median(r_sky_u[~in_cat])), 4),
            },
            "d_omega_deg2": _quantiles(d_omega * (180.0 / math.pi) ** 2),
            "sigma_u": _quantiles(s_u),
            # The chapter's most extreme number, priced against the published
            # literature -- the chapter's own "measure before you generalize"
            # discipline applied to itself (tomas M8).
            "mass_precision_plausibility": {
                "measured_sigma_Mz_frac_median": _r(float(np.median(s_mz)), 4),
                "measured_sigma_Mz_frac_x_snr_median": _r(float(np.median(s_mz * snr)), 4),
                "implied_at_snr_20": _r(float(np.median(s_mz * snr)) / 20.0, 4),
                "event_889": _r(
                    math.sqrt(float(crb.loc[889, "delta_M_delta_M"]))
                    / float(crb.loc[889, "M"]),
                    4,
                ),
                "literature": BABAK_2017_MASS_PRECISION,
                "ratio_vs_literature_low": _r(
                    BABAK_2017_MASS_PRECISION["range_low"]  # type: ignore[operator]
                    / (float(np.median(s_mz * snr)) / 20.0),
                    3,
                ),
                "ratio_vs_literature_high": _r(
                    BABAK_2017_MASS_PRECISION["range_high"]  # type: ignore[operator]
                    / (float(np.median(s_mz * snr)) / 20.0),
                    3,
                ),
            },
            "conditioning": _conditioning_stats(crb),
            "scatter": scatter,
        },
        "snr_grid": [_r(v, 6) for v in SNR_GRID],
        # d_L(z) at the mock truth h = 0.73, tabulated so the browser can map a
        # candidate's catalogue redshift onto the fraction coordinate
        # u = d_L(z; h) / d_hat_L without carrying a cosmology of its own.
        # Values come from physical_relations.dist — the production function.
        "dl_table": {
            "h": float(H_TRUE),
            "z": [_r(float(z), 5) for z in np.linspace(0.0, 1.6, 161)],
            "d_L_Gpc": [
                _r(float(dist(float(z), H_TRUE)), 7) for z in np.linspace(0.0, 1.6, 161)
            ],
        },
        "events": {},
    }

    # ---- per-event blocks -------------------------------------------------
    targets: list[tuple[float, float]] = []
    if catalogue is not None:
        from astropy import units as u  # noqa: PLC0415
        from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord  # noqa: PLC0415

        for eid in EVENT_IDS:
            row = crb.loc[eid]
            lon = math.degrees(float(row["phiS"])) % 360.0
            lat = math.degrees(math.pi / 2.0 - float(row["qS"]))
            icrs = SkyCoord(
                lon=lon * u.deg,
                lat=lat * u.deg,
                frame=BarycentricTrueEcliptic(equinox="J2000"),
            ).transform_to("icrs")
            targets.append((float(icrs.ra.deg), float(icrs.dec.deg)))
        print("    cutting sky patches (one streaming pass over the catalogue)...")
        patches = _load_patch(catalogue, targets)
        patches = [_prepare_patch(p) for p in patches]
    else:
        patches = [None] * len(EVENT_IDS)  # type: ignore[list-item]

    cond14_by_event = {
        eid: float(np.linalg.cond(_cov14(crb.loc[[eid]])[0])) for eid in EVENT_IDS
    }

    for k, eid in enumerate(EVENT_IDS):
        row = crb.loc[eid]
        cov3 = _cov3_fraction(row)
        cov4 = _cov4_fraction(row)
        sd = np.sqrt(np.diag(cov3))
        corr3 = cov3 / np.outer(sd, sd)

        # Bishop (2006) PRML Eq. 2.81-2.82, as bayesian_statistics.py:2495-2510
        cov_obs = cov4[:3, :3]
        cov_cross = cov4[3, :3]
        cov_obs_inv = np.linalg.pinv(cov_obs)
        proj = cov_cross @ cov_obs_inv
        sigma2_cond = float(cov4[3, 3] - cov_cross @ cov_obs_inv @ cov_cross)

        theta = float(row["qS"])
        dl = float(row["luminosity_distance"])
        sdl = math.sqrt(float(row["delta_luminosity_distance_delta_luminosity_distance"]))
        snr_e = float(row["SNR"])

        block: dict[str, Any] = {
            "idx": int(eid),
            "snr": _r(snr_e, 7),
            "d_L_Gpc": _r(dl, 7),
            "d_L_Mpc": _r(dl * 1000.0, 6),
            "M_z_Msun": _r(float(row["M"]), 7),
            "mu_Msun": _r(float(row["mu"]), 4),
            "in_catalog": bool(row["in_catalog"]),
            "host_galaxy_index": int(row["host_galaxy_index"]),
            "phi": _r(float(row["phiS"]), 9),
            "theta": _r(theta, 9),
            "sigma_phi": _r(sd[0], 6),
            "sigma_theta": _r(sd[1], 6),
            "sigma_u": _r(sd[2], 6),
            "sigma_dL_Gpc": _r(sdl, 6),
            "sigma_dL_Mpc": _r(sdl * 1000.0, 6),
            "sigma_Mz_frac": _r(math.sqrt(cov4[3, 3]), 6),
            "cov3": [[_r(v, 8) for v in r_] for r_ in cov3],
            "corr3": [[_r(v, 5) for v in r_] for r_ in corr3],
            "cond3": _r(float(np.linalg.cond(cov3)), 5),
            "cond4": _r(float(np.linalg.cond(cov4)), 5),
            "cond14": _r(cond14_by_event[eid], 5),
            "sigma_cond": _r(math.sqrt(max(sigma2_cond, 0.0)), 6),
            "proj": [_r(v, 6) for v in proj],
            "d_omega_sr": _r(
                2 * math.pi * abs(math.sin(theta)) * math.sqrt(
                    max(sd[0] ** 2 * sd[1] ** 2 - cov3[0, 1] ** 2, 0.0)
                ),
                6,
            ),
            "radius_full_rad": _r(_sky_search_radius(row, drop_corr=False), 6),
            "radius_nocorr_rad": _r(_sky_search_radius(row, drop_corr=True), 6),
        }
        block["d_omega_deg2"] = _r(block["d_omega_sr"] * (180.0 / math.pi) ** 2, 6)

        # z window (physical_relations.py:546-567 — note the body hardcodes 3 sigma)
        z_lo, z_hi = get_redshift_outer_bounds(dl, sdl, 0.6, 0.86, 0.04, 0.5)
        block["z_window"] = [_r(z_lo, 6), _r(min(z_hi, REDSHIFT_UPPER_LIMIT), 6)]

        # Index of the grid point nearest this event's OWN measured SNR — the
        # widget's default position, so the reader starts on real data and
        # every other position is an explicit counterfactual.
        block["own_snr_index"] = int(
            np.argmin(np.abs(np.asarray(SNR_GRID) - snr_e))
        )

        patch = patches[k]
        if patch is not None:
            block["patch"] = _patch_block(row, patch, snr_e)
        payload["events"][str(eid)] = block

    return payload


def _patch_block(row: pd.Series, patch: pd.DataFrame, snr_ref: float) -> dict[str, Any]:
    """Measured candidate counts on the SNR grid over a real GLADE+ patch."""
    phi0 = float(row["phiS"])
    theta0 = float(row["qS"])
    dl = float(row["luminosity_distance"])
    sdl = math.sqrt(float(row["delta_luminosity_distance_delta_luminosity_distance"]))

    query = _polar_to_cartesian(np.array([theta0]), np.array([phi0]))[0]
    pts = _polar_to_cartesian(patch["THETA_S"].to_numpy(), patch["PHI_S"].to_numpy())
    chord = np.linalg.norm(pts - query, axis=1)  # the BallTree metric
    z_g = patch["REDSHIFT"].to_numpy()
    z_err = patch["REDSHIFT_MEASUREMENT_ERROR"].to_numpy()

    counts: dict[str, list[int]] = {
        "full": [], "full_zwin": [], "nocorr": [], "nocorr_zwin": [],
    }
    radius_deg: list[float] = []
    zwin_lo: list[float] = []
    zwin_hi: list[float] = []
    tv_move: list[float | None] = []       # total-variation weight displacement
    ess_full: list[float | None] = []      # 1 / sum p^2  (effective #candidates)
    ess_diag: list[float | None] = []
    top_changed: list[bool | None] = []
    radius_ok = True

    cov3 = _cov3_fraction(row)
    cone_chord = 2.0 * math.sin(math.radians(PATCH_RADIUS_DEG) / 2.0)
    for snr in SNR_GRID:
        scale = snr_ref / snr  # every 1-sigma width x scale (Fisher ~ SNR^2)
        r_full = _sky_search_radius(row, drop_corr=False, snr_scale=scale)
        r_none = _sky_search_radius(row, drop_corr=True, snr_scale=scale)
        if max(r_full, r_none) > cone_chord:
            radius_ok = False
        z_lo, z_hi = get_redshift_outer_bounds(dl, sdl * scale, 0.6, 0.86, 0.04, 0.5)
        z_hi = min(z_hi, REDSHIFT_UPPER_LIMIT)
        in_z = (z_g + z_err >= z_lo) & (z_g - z_err <= z_hi)
        m_full = chord <= r_full
        m_none = chord <= r_none
        counts["full"].append(int(m_full.sum()))
        counts["nocorr"].append(int(m_none.sum()))
        counts["full_zwin"].append(int((m_full & in_z).sum()))
        counts["nocorr_zwin"].append(int((m_none & in_z).sum()))
        radius_deg.append(_r(math.degrees(2 * math.asin(min(r_full / 2.0, 1.0))), 6))
        zwin_lo.append(_r(z_lo, 6))
        zwin_hi.append(_r(z_hi, 6))

        # "Assume sky and distance are independent": how much of the candidate
        # weight moves when the covariance is factorized. Scale-free measure:
        # normalise both weight vectors over the SAME candidate set and take
        # the total-variation distance, 0.5 * sum |p_full - p_diag| (in [0,1] =
        # "this fraction of the candidate weight was redistributed"). Absolute
        # densities are NOT compared: their ratio is dominated by det(Sigma)
        # and by how far into the tail a lone candidate sits, neither of which
        # is what "mis-weights every candidate" means.
        # The GW factor alone, evaluated at each candidate's catalogue
        # redshift (the sigma_z -> 0 point limit; the kernel is Ch 7's subject).
        sel = m_full & in_z
        n_sel = int(sel.sum())
        if n_sel < 2:
            tv_move.append(None)
            ess_full.append(float(n_sel) if n_sel else None)
            ess_diag.append(float(n_sel) if n_sel else None)
            top_changed.append(None)
            continue
        cov_s = cov3 * scale**2
        u_g = np.array([dist(float(z), H_TRUE) for z in z_g[sel]]) / dl
        x = np.column_stack(
            [patch["PHI_S"].to_numpy()[sel], patch["THETA_S"].to_numpy()[sel], u_g]
        )
        mean = np.array([phi0, theta0, 1.0])
        lp_full = _mvn_logpdf(x, mean, cov_s)
        lp_diag = _mvn_logpdf(x, mean, np.diag(np.diag(cov_s)))
        p_full = np.exp(lp_full - lp_full.max())
        p_diag = np.exp(lp_diag - lp_diag.max())
        s_f, s_d = p_full.sum(), p_diag.sum()
        if s_f <= 0 or s_d <= 0:
            tv_move.append(None)
            ess_full.append(None)
            ess_diag.append(None)
            top_changed.append(None)
            continue
        p_full /= s_f
        p_diag /= s_d
        tv_move.append(_r(0.5 * float(np.abs(p_full - p_diag).sum()), 4))
        ess_full.append(_r(1.0 / float((p_full**2).sum()), 5))
        ess_diag.append(_r(1.0 / float((p_diag**2).sum()), 5))
        top_changed.append(bool(int(np.argmax(p_full)) != int(np.argmax(p_diag))))

    # Display sample: the nearest PATCH_SCATTER_N galaxies, in local
    # tangent-plane offsets (arcmin), great-circle-correct in azimuth.
    order = np.argsort(chord)[:PATCH_SCATTER_N]
    d_phi = ((patch["PHI_S"].to_numpy()[order] - phi0 + math.pi) % (2 * math.pi)) - math.pi
    d_theta = patch["THETA_S"].to_numpy()[order] - theta0
    arcmin = 180.0 * 60.0 / math.pi
    return {
        "catalogue_file": str(CATALOGUE_REL),
        "catalogue_note": (
            "committed baseline reduced catalogue; differs from the campaign-#53 "
            "realization parent in exactly one column, z_error "
            "(BOOK_DESIGN.md section 4.2 rule 5)"
        ),
        "cone_radius_deg": PATCH_RADIUS_DEG,
        "cone_fully_contains_every_search_ball": bool(radius_ok),
        "n_after_prune": int(len(patch)),
        "n_scatter": int(len(order)),
        "scatter_r_max_arcmin": _r(
            math.degrees(2 * math.asin(min(float(chord[order].max()) / 2.0, 1.0))) * 60.0, 5
        ),
        "scatter_x_arcmin": [_r(v * math.sin(theta0) * arcmin, 4) for v in d_phi],
        "scatter_y_arcmin": [_r(v * arcmin, 4) for v in d_theta],
        "scatter_z": [_r(v, 4) for v in z_g[order]],
        "counts": counts,
        "radius_deg": radius_deg,
        "radius_arcmin": [_r(v * 60.0, 5) for v in radius_deg],
        "zwin_lo": zwin_lo,
        "zwin_hi": zwin_hi,
        "weight_moved_by_factorizing": tv_move,
        "ess_full": ess_full,
        "ess_diagonal": ess_diag,
        "top_candidate_changed": top_changed,
    }


# --------------------------------------------------------------------------
# I6.2 — the dt^2 switch
# --------------------------------------------------------------------------
def build_dt2(pool_dir: Path) -> dict[str, Any]:
    files = sorted(glob.glob(str(pool_dir / "injection_h_0p73_task_*.csv")))
    frames = [pd.read_csv(f) for f in files]
    pool = pd.concat(frames, ignore_index=True)
    strata = {str(k): int(v) for k, v in pool["stratum"].value_counts().items()}
    # Only stratum "a" carries the population measure
    # (simulation_detection_probability.py:370-400).
    a = pool[pool["stratum"] == "a"]
    z = a["z"].to_numpy()
    snr = a["SNR"].to_numpy()
    d_l = a["luminosity_distance"].to_numpy()

    edges = np.linspace(0.0, 1.5, 61)
    states: dict[str, Any] = {}
    for key, eff_snr, label in (
        ("with_dt2", snr, "dt^2 present (physical SNR)"),
        ("without_dt2", snr / 10.0, "dt^2 absent (code returned <h|h>/dt^2)"),
    ):
        det = eff_snr >= SNR_THRESHOLD
        zz = z[det]
        hist = np.histogram(zz, bins=edges)[0]
        states[key] = {
            "label": label,
            "n_detected": int(det.sum()),
            "detected_fraction": _r(float(det.mean()), 5),
            "z_hist": [int(v) for v in hist],
            "z_median": _r(float(np.median(zz)), 5) if det.any() else None,
            "z_p90": _r(float(np.percentile(zz, 90)), 5) if det.any() else None,
            "z_max": _r(float(zz.max()), 5) if det.any() else None,
            "d_L_median_Gpc": _r(float(np.median(d_l[det])), 5) if det.any() else None,
            "d_L_max_Gpc": _r(float(d_l[det].max()), 5) if det.any() else None,
            "horizon_max_Gpc": _r(float((eff_snr * d_l / SNR_THRESHOLD).max()), 6),
        }

    return {
        "meta": {
            "pool": str(POOL_REL),
            "n_files": len(files),
            "n_data_rows": int(len(pool)),
            "n_lines_with_headers": int(len(pool) + len(files)),
            "strata": strata,
            "population_measure_stratum": "a",
            "n_population_measure": int(len(a)),
            "z_cut": _r(float(pool["z_cut"].iloc[0]), 4),
            "h_inj": _r(float(pool["h_inj"].iloc[0]), 4),
            "snr_threshold": float(SNR_THRESHOLD),
            "dt_s": 10.0,
            "counterfactual_note": (
                "The pre-fix state is a COUNTERFACTUAL on the post-fix pool, not a "
                "re-simulation: G8 section 1.2 shows the old code returned exactly "
                "<h1|h2>/dt^2, so at dt = 10 s the pre-fix detection criterion is "
                "exactly SNR/10 >= 20 on these same injections."
            ),
        },
        "z_edges": [_r(v, 5) for v in edges],
        "states": states,
        "recorded_prefix_population": G8_L5_RECORDED,
        "fisher_consequence": {
            "snr_factor": 0.1,
            "fisher_factor": 0.01,
            "sigma_factor": 10.0,
            "source": "G8_dt2_inner_product_derivation.md section 1 (claim) + G7 row 1",
        },
    }


# --------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    crb_path = _resolve(CRB_REL)
    if crb_path is None:
        raise SystemExit(f"gen_ch06: required CRB table not found: {CRB_REL}")
    crb = pd.read_csv(crb_path)
    print(f"    CRB rows: {len(crb)} ({int(crb['in_catalog'].sum())} in-catalogue)")

    catalogue = _resolve(CATALOGUE_REL)
    if catalogue is None:
        print(
            "    NOTICE: reduced_galaxy_catalogue.csv not found in this repo or a "
            "sibling MasterThesisCode checkout — the I6.1 candidate-count blocks "
            "are not rebuilt; any committed ch06_fisher.json is left untouched."
        )
        if OUT_FISHER.exists():
            fisher = None
        else:
            fisher = build_fisher(crb, None)
    else:
        fisher = build_fisher(crb, catalogue)

    if fisher is not None:
        OUT_FISHER.write_text(json.dumps(fisher, separators=(",", ":")) + "\n")
        print(f"    wrote {OUT_FISHER.name}  ({OUT_FISHER.stat().st_size / 1024:.1f} KB)")

    pool = _resolve(POOL_REL)
    if pool is None:
        print(
            "    NOTICE: injection pool not found in this repo or a sibling "
            "MasterThesisCode checkout — I6.2 not rebuilt; any committed "
            "ch06_dt2.json is left untouched."
        )
    else:
        dt2 = build_dt2(pool)
        OUT_DT2.write_text(json.dumps(dt2, separators=(",", ":")) + "\n")
        print(f"    wrote {OUT_DT2.name}  ({OUT_DT2.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
