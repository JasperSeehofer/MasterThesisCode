"""Generator for Chapter 5 — "The Galaxy You Cannot See".

Produces the two data files behind the chapter's interactives.

``book/site/data/ch05_mixture.json``   (cold open, I5.1, I5.2, the dossier)
    Everything derived from the campaign's own delivered per-event legs,
    ``seed61000/real_r1/diagnostics/event_likelihoods.csv`` (1588 events x 41
    h-values), plus the run's own selection integrals from its logs
    (``seed61000/mixture_leg_log_extract.txt``: ``D(h)`` and
    ``beta_Gbar(h)`` at 7 s.f., 41 h-values).

    The master equation is

        p_i(h) = w_G(h) L_cat,i(h) + (1 - w_G(h)) L_comp,i(h),
        w_G = beta_G / D,  beta_G = D - beta_Gbar
        (bayesian_statistics.py:3006-3009, :1042-1048; G2c section 1)

    and the CSV ships all four columns, so the generator can (a) VERIFY the
    identity on all 65,108 rows and (b) re-mix the SAME delivered legs under a
    counterfactual mixture weight without re-running any physics.  Nothing here
    re-derives a likelihood; the only thing ever altered is the scalar
    w_G(h), and every altered state is labelled as a counterfactual.

    The counterfactual dial is an ODDS scale on the shipped weight,

        w_kappa(h) / (1 - w_kappa(h)) = kappa * w_G(h) / (1 - w_G(h)),

    which is monotone, stays inside (0, 1) for every finite kappa > 0, and has
    the two sanity limits of Q5.3 as its endpoints: kappa -> 0 gives the empty
    catalogue (p_i -> L_comp) and kappa -> infinity gives the complete
    catalogue (p_i -> L_cat).  It is emitted as a precomputed family of summed
    log-posteriors so the browser never has to hold 1588 x 41 numbers.

``book/site/data/ch05_completeness.json``   (section 1, Trap 5.B)
    The catalogue completeness f(z, Omega) itself, evaluated with the
    project's own frozen estimator and its frozen m_th map
    (``galaxy_catalogue/pixel_completeness.py``, ``m_th_map_nside32.npy``;
    Gray-Messenger-Veitch 2022, arXiv:2111.04629, Eqs. (2)(3)(5)).  Emitted:
    the sky-average f_bar(z) (GMV Eq. 3, equal-area pixels) and three example
    per-pixel curves at the 10th/50th/90th percentile of m_th, plus the
    measured h-invariance of f_bar across the chapter's whole h-grid.

PROVENANCE NOTES
----------------
* ``w_G``: the CSV column is full double precision and is what the browser
  uses; the log-derived ``1 - beta_Gbar/D`` is emitted alongside it as an
  INDEPENDENT cross-check (the two agree to 5.5e-7, the 7-s.f. rounding of the
  log line).  The 4-d.p. ``w_G`` log line (``bayesian_statistics.py:2335``) is
  never used — sources map section 7.19(e).
* The h-grid is NON-UNIFORM (0.01 on [0.60,0.65] and [0.80,0.86], 0.005
  between).  Every integral in this file uses ``np.trapezoid`` on the actual
  grid; no second difference is ever taken.
* ``in_catalog`` (class membership) is the GENERATOR's truth flag from
  ``prepared_cramer_rao_bounds.csv`` -- whether the injected host was drawn
  from the catalogue -- and is NOT the same thing as "the estimator found a
  candidate".  Both counts are emitted and the chapter states the difference.

DATA AVAILABILITY
-----------------
``mixture_leg_log_extract.txt`` and the frozen ``m_th`` map are git-tracked and
present in any checkout of this branch.  The two per-event CSVs
(``diagnostics/event_likelihoods.csv``, ``prepared_cramer_rao_bounds.csv``) are
**not** tracked -- they live in the working tree of the main checkout only -- so
they are resolved from, in order: this repo root, then a sibling
``MasterThesisCode`` checkout next to this one.  If neither has them, the
already-committed ``ch05_mixture.json`` is left untouched and a NOTICE is
printed: the generator never fails a build over an untracked artifact and never
writes a partial or silently-degraded file.  (Same contract as
``gen_ch04.py``'s injection-pool step.)

Determinism: no RNG anywhere.  Read-only outside ``book/``.

Run as::

    /home/jasper/Repositories/MasterThesisCode/.venv/bin/python \\
        book/generators/gen_ch05.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from master_thesis_code.constants import H as H_TRUE  # noqa: E402
from master_thesis_code.galaxy_catalogue.pixel_completeness import (  # noqa: E402
    M_TH_CACHE_PATH,
    NSIDE,
    PixelCompleteness,
)
from master_thesis_code.physical_relations import dist_to_redshift  # noqa: E402

OUT_DIR = REPO_ROOT / "book" / "site" / "data"
RUN_REL = Path("results/campaign51_20260728/realistic_20260729/seed61000")
R1_REL = RUN_REL / "real_r1"


def resolve(rel: Path) -> Path | None:
    """Resolve a data path relative to this checkout, then a sibling one.

    Never hardcodes a machine path (BOOK_DESIGN.md section 4.3 item 6).
    """
    for root in (REPO_ROOT, REPO_ROOT.parent / "MasterThesisCode"):
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


LOG_EXTRACT = resolve(RUN_REL / "mixture_leg_log_extract.txt")
DIAG_CSV = resolve(R1_REL / "diagnostics" / "event_likelihoods.csv")
CRB_CSV = resolve(R1_REL / "prepared_cramer_rao_bounds.csv")

# The two protagonists (BOOK_DESIGN.md section 1, Ch 5 "Running example").
EVENT_889 = 889
EVENT_606 = 606

# Counterfactual odds-scale grid for I5.1.  `null` on the JSON side means the
# limit kappa -> infinity (a complete catalogue, w_G == 1 identically).
KAPPA_GRID: list[float | None] = [
    0.0,
    0.05,
    0.1,
    0.2,
    0.35,
    0.5,
    0.7,
    1.0,  # <- the shipped catalogue
    1.4,
    2.0,
    3.0,
    5.0,
    8.0,
    15.0,
    30.0,
    60.0,
    120.0,
    250.0,
    500.0,
    1000.0,
    3000.0,
    None,  # <- complete catalogue
]


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------
def _round(a: Any, n: int = 10) -> Any:
    """Round for compact JSON without losing anything the page can see."""
    arr = np.asarray(a, dtype=np.float64)
    return [float(x) for x in np.round(arr, n)]


def _posterior_summary(
    log_sum: npt.NDArray[np.float64], h_grid: npt.NDArray[np.float64]
) -> dict[str, float]:
    """MAP (grid argmax), posterior mean, and the peak-to-edge log depths."""
    shifted = log_sum - float(np.max(log_sum))
    dens = np.exp(shifted)
    norm = float(np.trapezoid(dens, h_grid))
    dens = dens / norm
    imax = int(np.argmax(log_sum))
    return {
        "map": float(h_grid[imax]),
        "mean": float(np.trapezoid(dens * h_grid, h_grid)),
        "depth_low_nats": float(log_sum[imax] - log_sum[0]),
        "depth_high_nats": float(log_sum[imax] - log_sum[-1]),
        "edge_low_over_peak": float(np.exp(log_sum[0] - log_sum[imax])),
        "edge_high_over_peak": float(np.exp(log_sum[-1] - log_sum[imax])),
    }


# ----------------------------------------------------------------------
# 1. the run's own selection integrals, from its own logs
# ----------------------------------------------------------------------
def read_selection_logs() -> dict[str, Any]:
    """Parse D(h), beta_Gbar(h) and z_max(h) out of the committed log extract.

    Code sites: ``bayesian_statistics.py:1013`` / ``:1145`` (D(h)) and
    ``:1170`` / ``:1297`` (beta_Gbar).  beta_G = D - beta_Gbar
    (``bayesian_statistics.py`` line 910 per G2c section 2).
    """
    assert LOG_EXTRACT is not None
    txt = LOG_EXTRACT.read_text()
    d_h: dict[float, float] = {}
    z_max: dict[float, float] = {}
    b_gbar: dict[float, float] = {}
    for m in re.finditer(
        r"D\(h=([\d.]+)\) = ([\d.eE+-]+)\s+\[z_max=([\d.]+), dl_max=([\d.]+) Gpc\]", txt
    ):
        d_h[float(m.group(1))] = float(m.group(2))
        z_max[float(m.group(1))] = float(m.group(3))
    for m in re.finditer(
        r"beta_Gbar\(h=([\d.]+)\) = ([\d.eE+-]+)\s+\[z_max=([\d.]+)\]", txt
    ):
        b_gbar[float(m.group(1))] = float(m.group(2))
    dl_max = sorted(
        {
            float(m.group(4))
            for m in re.finditer(
                r"D\(h=([\d.]+)\) = ([\d.eE+-]+)\s+\[z_max=([\d.]+), dl_max=([\d.]+) Gpc\]",
                txt,
            )
        }
    )
    return {"D": d_h, "beta_Gbar": b_gbar, "z_max": z_max, "dl_max_values": dl_max}


# ----------------------------------------------------------------------
# 2. the mixture file
# ----------------------------------------------------------------------
def build_mixture() -> dict[str, Any]:
    assert DIAG_CSV is not None and CRB_CSV is not None
    diag = pd.read_csv(DIAG_CSV)
    crb = pd.read_csv(CRB_CSV)

    h_grid = np.sort(diag["h"].unique())
    n_h = len(h_grid)

    def pivot(col: str) -> npt.NDArray[np.float64]:
        return (
            diag.pivot(index="event_idx", columns="h", values=col)
            .reindex(columns=h_grid)
            .to_numpy(dtype=np.float64)
        )

    combined = pivot("combined_no_bh")
    l_cat = pivot("L_cat_no_bh")
    l_comp = pivot("L_comp")
    b_num = pivot("B_num")

    event_idx = np.sort(diag["event_idx"].unique())
    n_events = len(event_idx)

    # w_G is event-INDEPENDENT (G2c section 2 row w_G): assert it.
    per_h_unique = diag.groupby("h")["w_G"].nunique().to_numpy()
    assert (per_h_unique == 1).all(), "w_G is not event-independent in this CSV"
    w_g = diag.groupby("h")["w_G"].first().reindex(h_grid).to_numpy(dtype=np.float64)

    # --- the identity, verified on every delivered cell ------------------
    remixed = w_g[None, :] * l_cat + (1.0 - w_g[None, :]) * l_comp
    identity_max_rel = float(np.max(np.abs(remixed - combined) / np.abs(combined)))

    # --- independent cross-check of w_G from the run's own logs ----------
    logs = read_selection_logs()
    d_arr = np.array([logs["D"][round(float(h), 5)] for h in h_grid])
    bgbar_arr = np.array([logs["beta_Gbar"][round(float(h), 5)] for h in h_grid])
    zmax_arr = np.array([logs["z_max"][round(float(h), 5)] for h in h_grid])
    beta_g_arr = d_arr - bgbar_arr
    w_g_from_logs = beta_g_arr / d_arr
    w_g_log_max_abs_diff = float(np.max(np.abs(w_g_from_logs - w_g)))

    # --- classes: the GENERATOR's truth flag -----------------------------
    in_catalog = crb["in_catalog"].to_numpy(dtype=bool)[event_idx]
    n_incat = int(in_catalog.sum())
    n_dark = int((~in_catalog).sum())

    # --- the estimator's own view: which events have a catalogue leg -----
    has_cat_leg = (l_cat > 0.0).all(axis=1)
    only_zero_at_some_h = ((l_cat > 0.0).any(axis=1)) & (~has_cat_leg)
    n_no_cat_leg = int((~has_cat_leg).sum())
    n_no_cat_leg_but_incat = int(in_catalog[~has_cat_leg].sum())

    log_combined = np.log(combined)
    log_all = log_combined.sum(axis=0)
    log_incat = log_combined[in_catalog].sum(axis=0)
    log_dark = log_combined[~in_catalog].sum(axis=0)

    # --- the two branch legs, alone --------------------------------------
    log_leg_cat = np.log(l_cat[has_cat_leg]).sum(axis=0)
    log_leg_comp = np.log(l_comp).sum(axis=0)

    # --- I5.1: counterfactual mixture weights ----------------------------
    odds = w_g / (1.0 - w_g)
    log_by_kappa: list[list[float]] = []
    w_by_kappa: list[list[float]] = []
    n_zero_by_kappa: list[int] = []
    summary_by_kappa: list[dict[str, float]] = []
    for kappa in KAPPA_GRID:
        if kappa is None:
            w_k = np.ones(n_h)
        elif kappa == 0.0:
            w_k = np.zeros(n_h)
        else:
            ok = kappa * odds
            w_k = ok / (1.0 + ok)
        mixed = w_k[None, :] * l_cat + (1.0 - w_k[None, :]) * l_comp
        # Events whose likelihood is identically zero across the grid make no
        # contribution: they are exactly the events the pre-8db6c6e code
        # SILENTLY dropped (C4).  Count them and exclude them, which is what
        # that code did -- the page says so out loud.
        alive = (mixed > 0.0).all(axis=1)
        n_zero_by_kappa.append(int((~alive).sum()))
        log_sum = np.log(mixed[alive]).sum(axis=0)
        log_by_kappa.append(_round(log_sum, 6))
        w_by_kappa.append(_round(w_k, 10))
        summary_by_kappa.append(_posterior_summary(log_sum, h_grid))

    # "flatten the slope": w_G held at its h = 0.73 value.
    i73 = int(np.where(np.isclose(h_grid, 0.73))[0][0])
    w_flat = np.full(n_h, w_g[i73])
    mixed_flat = w_flat[None, :] * l_cat + (1.0 - w_flat[None, :]) * l_comp
    log_flat = np.log(mixed_flat).sum(axis=0)

    # --- C10's prefactor budget, recomputed ------------------------------
    i81 = int(np.where(np.isclose(h_grid, 0.81))[0][0])
    nats_prefactor = float(
        n_events * (np.log(1.0 - w_g[i81]) - np.log(1.0 - w_g[i73]))
    )

    # --- the two protagonists --------------------------------------------
    pos = {int(e): k for k, e in enumerate(event_idx)}

    # The sky-averaged completeness at each protagonist's own redshift — the
    # number Trap 5.B is dismantled with.  Same frozen estimator as section 1.
    completeness = PixelCompleteness(np.load(M_TH_CACHE_PATH))

    def event_block(idx: int) -> dict[str, Any]:
        k = pos[idx]
        d_l_gpc = float(crb["luminosity_distance"].iloc[idx])
        z_ev = float(dist_to_redshift(d_l_gpc, h=H_TRUE))
        return {
            "event_idx": idx,
            "in_catalog": bool(crb["in_catalog"].iloc[idx]),
            "host_galaxy_index": int(crb["host_galaxy_index"].iloc[idx]),
            "snr": float(crb["SNR"].iloc[idx]),
            "d_L_Gpc": d_l_gpc,
            "d_L_Mpc": d_l_gpc * 1e3,
            "z_at_h_true": z_ev,
            "f_bar_at_z": float(completeness.f_bar(np.array([z_ev]), H_TRUE)[0]),
            "has_catalogue_leg": bool(has_cat_leg[k]),
            "L_cat": _round(l_cat[k], 8),
            "L_comp": _round(l_comp[k], 10),
            "B_num": _round(b_num[k], 4),
            "combined": _round(combined[k], 10),
        }

    return {
        "_provenance": {
            "run": "campaign51_20260728/realistic_20260729/seed61000/real_r1",
            "diagnostics": "diagnostics/event_likelihoods.csv",
            "selection_integrals": "seed61000/mixture_leg_log_extract.txt",
            "class_flag": "prepared_cramer_rao_bounds.csv:in_catalog",
            "master_equation": "bayesian_statistics.py:3006-3009, :1042-1048",
            "mapping": "G2c section 1 + section 2 rows beta_G/beta_Gbar/w_G",
        },
        "h_grid": _round(h_grid, 6),
        "h_true": H_TRUE,
        "n_events": n_events,
        "n_rows": int(n_events * n_h),
        "n_incat": n_incat,
        "n_dark": n_dark,
        "n_no_cat_leg": n_no_cat_leg,
        "n_no_cat_leg_but_incat": n_no_cat_leg_but_incat,
        "n_partial_cat_leg": int(only_zero_at_some_h.sum()),
        "n_with_cat_leg": int(has_cat_leg.sum()),
        "identity_max_rel_err": identity_max_rel,
        "w_G": _round(w_g, 12),
        "w_G_from_logs": _round(w_g_from_logs, 12),
        "w_G_log_max_abs_diff": w_g_log_max_abs_diff,
        "D_h": _round(d_arr, 1),
        "beta_G": _round(beta_g_arr, 1),
        "beta_Gbar": _round(bgbar_arr, 1),
        "z_max": _round(zmax_arr, 5),
        "dl_max_Gpc": logs["dl_max_values"],
        "log_post_all": _round(log_all, 6),
        "log_post_incat": _round(log_incat, 6),
        "log_post_dark": _round(log_dark, 6),
        "summary_all": _posterior_summary(log_all, h_grid),
        "summary_incat": _posterior_summary(log_incat, h_grid),
        "summary_dark": _posterior_summary(log_dark, h_grid),
        "log_leg_cat_only": _round(log_leg_cat, 6),
        "log_leg_comp_only": _round(log_leg_comp, 6),
        "summary_leg_cat_only": _posterior_summary(log_leg_cat, h_grid),
        "summary_leg_comp_only": _posterior_summary(log_leg_comp, h_grid),
        "kappa_grid": KAPPA_GRID,
        "kappa_shipped_index": KAPPA_GRID.index(1.0),
        "log_post_by_kappa": log_by_kappa,
        "w_by_kappa": w_by_kappa,
        "n_zero_by_kappa": n_zero_by_kappa,
        "summary_by_kappa": summary_by_kappa,
        "log_post_flat_w": _round(log_flat, 6),
        "w_flat": float(w_g[i73]),
        "summary_flat_w": _posterior_summary(log_flat, h_grid),
        "nats_prefactor_073_to_081": nats_prefactor,
        "events": {"e889": event_block(EVENT_889), "e606": event_block(EVENT_606)},
    }


# ----------------------------------------------------------------------
# 3. the completeness file
# ----------------------------------------------------------------------
def build_completeness(h_grid: list[float]) -> dict[str, Any]:
    m_th = np.load(M_TH_CACHE_PATH)
    completeness = PixelCompleteness(m_th)
    valid = np.isfinite(m_th)

    z_grid = np.concatenate(
        [
            np.linspace(0.001, 0.05, 30, endpoint=False),
            np.linspace(0.05, 0.6, 111, endpoint=False),
            np.linspace(0.6, 1.5, 46),
        ]
    )
    f_bar = completeness.f_bar(z_grid, H_TRUE)

    # f_bar's h-invariance is a documented property of the estimator
    # (pixel_completeness.py: the +5 log10 h in M_* cancels the distance
    # modulus's h).  Measure it rather than assert it.
    f_bar_dev = 0.0
    for h in (float(h_grid[0]), float(h_grid[-1])):
        f_bar_dev = max(
            f_bar_dev, float(np.max(np.abs(completeness.f_bar(z_grid, h) - f_bar)))
        )

    m_valid = m_th[valid]
    pct = {p: float(np.percentile(m_valid, p)) for p in (10, 50, 90)}
    pixel_curves = {}
    for p, m_val in pct.items():
        k = int(np.arange(len(m_th))[valid][np.argmin(np.abs(m_valid - m_val))])
        pixel_curves[f"p{p}"] = {
            "pixel": k,
            "m_th": float(m_th[k]),
            "f": _round(completeness.f_k(z_grid, k, H_TRUE), 6),
        }

    # Where the sky-averaged completeness crosses a few round numbers.
    def z_at_f(target: float) -> float:
        below = np.where(f_bar <= target)[0]
        if len(below) == 0:
            return float("nan")
        j = int(below[0])
        if j == 0:
            return float(z_grid[0])
        z0, z1 = float(z_grid[j - 1]), float(z_grid[j])
        f0, f1 = float(f_bar[j - 1]), float(f_bar[j])
        return z0 + (target - f0) * (z1 - z0) / (f1 - f0)

    return {
        "_provenance": {
            "estimator": "galaxy_catalogue/pixel_completeness.py",
            "m_th_map": Path(M_TH_CACHE_PATH).name,
            "reference": "Gray, Messenger & Veitch 2022, arXiv:2111.04629, Eqs. (2)(3)(5)",
            "sky_average": "GMV Eq. (3), equal-area HEALPix pixels",
        },
        "nside": int(NSIDE),
        "npix": int(len(m_th)),
        "n_valid_pixels": int(valid.sum()),
        "n_empty_pixels": int((~valid).sum()),
        "m_th_percentiles": pct,
        "z_grid": _round(z_grid, 6),
        "f_bar": _round(f_bar, 6),
        "f_bar_h_max_abs_dev": f_bar_dev,
        "pixel_curves": pixel_curves,
        "z_at_f_half": z_at_f(0.5),
        "z_at_f_tenth": z_at_f(0.1),
        "z_at_f_one_pct": z_at_f(0.01),
        "f_bar_at": {
            f"{z:g}": float(completeness.f_bar(np.array([z]), H_TRUE)[0])
            for z in (0.005, 0.0205, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8)
        },
    }


# ----------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # The completeness leg needs only git-tracked artifacts and always runs.
    h_grid_fallback = [0.6, 0.86]
    completeness = build_completeness(h_grid_fallback)
    path = OUT_DIR / "ch05_completeness.json"
    path.write_text(json.dumps(completeness, separators=(",", ":"), allow_nan=False))
    print(f"  wrote {path.relative_to(REPO_ROOT)}  ({path.stat().st_size:,} bytes)")

    if DIAG_CSV is None or CRB_CSV is None or LOG_EXTRACT is None:
        missing = [
            str(rel)
            for rel, got in (
                (R1_REL / "diagnostics" / "event_likelihoods.csv", DIAG_CSV),
                (R1_REL / "prepared_cramer_rao_bounds.csv", CRB_CSV),
                (RUN_REL / "mixture_leg_log_extract.txt", LOG_EXTRACT),
            )
            if got is None
        ]
        print(
            "  NOTICE: ch05_mixture.json NOT regenerated — untracked input(s) absent:\n"
            + "".join(f"    - {p}\n" for p in missing)
            + "    Expected at <repo>/<path> or ../MasterThesisCode/<path>.\n"
            "    The committed ch05_mixture.json is left untouched."
        )
        return

    mixture = build_mixture()
    path = OUT_DIR / "ch05_mixture.json"
    path.write_text(json.dumps(mixture, separators=(",", ":"), allow_nan=False))
    print(f"  wrote {path.relative_to(REPO_ROOT)}  ({path.stat().st_size:,} bytes)")

    # ---- console gates: everything the chapter quotes, printed ----------
    m = mixture
    print("\n  --- gates -------------------------------------------------")
    print(f"  events                     : {m['n_events']}  ({m['n_rows']} rows)")
    print(f"  in-catalogue / dark        : {m['n_incat']} / {m['n_dark']}")
    print(f"  no catalogue leg at any h  : {m['n_no_cat_leg']} "
          f"(of which in_catalog=True: {m['n_no_cat_leg_but_incat']})")
    print(f"  partial (zero at some h)   : {m['n_partial_cat_leg']}")
    print(f"  mixture identity max relerr: {m['identity_max_rel_err']:.3e}")
    print(f"  w_G(0.60) / (0.73) / (0.86): {m['w_G'][0]:.7f} / "
          f"{m['w_G'][m['h_grid'].index(0.73)]:.7f} / {m['w_G'][-1]:.7f}")
    print(f"  |w_G(csv) - w_G(logs)|max  : {m['w_G_log_max_abs_diff']:.3e}")
    print(f"  N dln(1-w_G), 0.73->0.81   : {m['nats_prefactor_073_to_081']:+.5f} nats")
    print(f"  MAP all / in-cat / dark    : {m['summary_all']['map']} / "
          f"{m['summary_incat']['map']} / {m['summary_dark']['map']}")
    print(f"  mean all / in-cat / dark   : {m['summary_all']['mean']:.4f} / "
          f"{m['summary_incat']['mean']:.4f} / {m['summary_dark']['mean']:.4f}")
    print(f"  catalogue leg alone  MAP   : {m['summary_leg_cat_only']['map']} "
          f"(mean {m['summary_leg_cat_only']['mean']:.4f})")
    print(f"  completion leg alone MAP   : {m['summary_leg_comp_only']['map']} "
          f"(mean {m['summary_leg_comp_only']['mean']:.4f})")
    print(f"  kappa=0 (empty cat)   MAP  : {m['summary_by_kappa'][0]['map']}")
    print(f"  kappa=inf (complete)  MAP  : {m['summary_by_kappa'][-1]['map']} "
          f"({m['n_zero_by_kappa'][-1]} events silenced)")
    print(f"  flat w_G = {m['w_flat']:.7f}  MAP  : {m['summary_flat_w']['map']}")
    c = completeness
    print(f"  f_bar(0.0205) / (0.1) / (0.3): {c['f_bar_at']['0.0205']:.4f} / "
          f"{c['f_bar_at']['0.1']:.4f} / {c['f_bar_at']['0.3']:.4f}")
    print(f"  f_bar h-invariance max dev  : {c['f_bar_h_max_abs_dev']:.3e}")
    print(f"  z where f_bar = 0.5 / 0.1   : {c['z_at_f_half']:.4f} / "
          f"{c['z_at_f_tenth']:.4f}")
    print(f"  889: SNR {m['events']['e889']['snr']:.1f}, "
          f"d_L {m['events']['e889']['d_L_Mpc']:.1f} Mpc, "
          f"z(h=0.73) {m['events']['e889']['z_at_h_true']:.5f}, "
          f"f_bar {m['events']['e889']['f_bar_at_z']:.4f}, "
          f"in_cat {m['events']['e889']['in_catalog']}")
    print(f"  606: SNR {m['events']['e606']['snr']:.1f}, "
          f"d_L {m['events']['e606']['d_L_Gpc']:.4f} Gpc, "
          f"z(h=0.73) {m['events']['e606']['z_at_h_true']:.5f}, "
          f"f_bar {m['events']['e606']['f_bar_at_z']:.4f}, "
          f"in_cat {m['events']['e606']['in_catalog']}")
    print(f"  889 L_cat 0.60/0.73/0.86  : {m['events']['e889']['L_cat'][0]:.4g} / "
          f"{m['events']['e889']['L_cat'][m['h_grid'].index(0.73)]:.4g} / "
          f"{m['events']['e889']['L_cat'][-1]:.4g}")
    print(f"  606 L_comp 0.60/0.73/0.86 : {m['events']['e606']['L_comp'][0]:.4g} / "
          f"{m['events']['e606']['L_comp'][m['h_grid'].index(0.73)]:.4g} / "
          f"{m['events']['e606']['L_comp'][-1]:.4g}")
    print(f"  606 L_cat 0.60/0.73/0.86  : {m['events']['e606']['L_cat'][0]:.4g} / "
          f"{m['events']['e606']['L_cat'][m['h_grid'].index(0.73)]:.4g} / "
          f"{m['events']['e606']['L_cat'][-1]:.4g}")


if __name__ == "__main__":
    main()
