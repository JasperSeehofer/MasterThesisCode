"""Bridge-the-closure shared library.

The closure harness (``.planning/derivation-partition-norm/closure_harness.py``)
recovers the injected H0 unbiased on a *synthetic* population. The real seed-600
pipeline rails to the upper grid edge. This library lets us start from the
closure and swap in ONE real-pipeline ingredient at a time (a "rung"), re-running
the same partition-norm MAP recovery, to find which ingredient introduces the
H0 bias.

Design: every rung is a :class:`BridgeConfig`; :func:`run_bridge` executes it and
returns the MAP, bias, and per-h log-posterior. The per-event likelihood and the
three Task-A precomputes are the SAME real production functions the closure used
(``precompute_completion_denominator`` etc.), so the only thing that changes
between rungs is the ingredient under test.

Ground-truth facts established for the seed-600 railing dataset (2026-06-29):
  * Selection is on the OPTIMAL (true) SNR>=20, NOT a noisy observable
    (``parameter_estimation.py:455``, ``main.py:539``) -> classic Malmquist is
    structurally impossible; the live effect is a sigma^2/distance-scatter
    curvature bias + the in-catalogue density structure.
  * Measurement scatter is unbiased: (meas-true)/true mean +0.0007, std 0.040.
  * Real sigma_dL/d_L ~ 3.7% (range 0.3-10%).
  * 99.2% of detected events are IN-CATALOGUE (nearby, GLADE ~complete) -> the
    bias lives in the in-catalogue term beta_G*L_cat/D(h), not the completion.

Run rungs via the ``rung_*.py`` scripts; they import from here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.integrate import fixed_quad
from scipy.stats import norm

from master_thesis_code.bayesian_inference.bayesian_statistics import (
    precompute_completion_denominator,
    precompute_global_catalog_selection,
    precompute_missing_completion_denominator,
)
from master_thesis_code.emri_rate import R_eff_per_mbh, mbh_mass_function
from master_thesis_code.galaxy_catalogue.handler import InternalCatalogColumns
from master_thesis_code.physical_relations import (
    comoving_volume_element,
    dist,
    dist_to_redshift,
    dist_vectorized,
)

# --- paths -----------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
OUTPUTS = _HERE / "outputs"
OUTPUTS.mkdir(exist_ok=True)

_SEED600 = Path("/tmp/seed600_local/simulations")
_PREPARED_CRB = _SEED600 / "prepared_cramer_rao_bounds.csv"  # measured
_RAW_CRB = _SEED600 / "cramer_rao_bounds.csv"  # true injected

# --- closure constants (mirrors closure_harness.py) ------------------------
_Z_MIN = 1e-3
_Z_MAX = 0.5
_M_MIN = 1.0e4
_M_MAX = 1.0e7
_D_HOR_GPC = 1.2  # mock detection horizon (Gpc)

_OMEGA_M = 0.25
_OMEGA_DE = 0.75
TRUE_H = 0.73  # seed-600 injected H0/100


# ---------------------------------------------------------------------------
# Mock detection probability + completeness (closure defaults)
# ---------------------------------------------------------------------------
def _p_det_of_dl(d_L: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Smooth mock detection probability vs d_L (Gpc), -> 0 past the horizon."""
    d = np.asarray(d_L, dtype=np.float64)
    p = 1.0 / (1.0 + np.exp((d - 0.7 * _D_HOR_GPC) / (0.08 * _D_HOR_GPC)))
    return np.where(d > _D_HOR_GPC, 0.0, p)


class MockPdet:
    """Closure's sky-independent, h-invariant smooth p_det(d_L)."""

    def get_dl_max(self, h: float) -> float:
        return _D_HOR_GPC

    def detection_probability_without_bh_mass_interpolated_zero_fill(
        self,
        d_L: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
        *,
        h: float,
    ) -> npt.NDArray[np.float64]:
        return _p_det_of_dl(d_L)

    def detection_probability_with_bh_mass_interpolated(
        self,
        d_L: npt.NDArray[np.float64],
        M_z: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
        *,
        h: float,
    ) -> npt.NDArray[np.float64]:
        return _p_det_of_dl(d_L)


class ZCompleteness:
    """Completeness f(z) (H0-independent), shared by injection and inference."""

    def __init__(self, f_func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]) -> None:
        self._f = f_func

    def get_completeness_at_redshift(
        self, z: npt.NDArray[np.float64], h: float | None = None, **kw: Any
    ) -> npt.NDArray[np.float64]:
        return np.clip(self._f(np.asarray(z, dtype=np.float64)), 0.0, 1.0)

    # sky-flat shims so the missing-completion precompute (which calls f_bar) works
    def f_bar(self, z: npt.NDArray[np.float64], h: float) -> npt.NDArray[np.float64]:
        return self.get_completeness_at_redshift(z, h)

    def f_k(self, z: npt.NDArray[np.float64], pixel: int, h: float) -> npt.NDArray[np.float64]:
        return self.get_completeness_at_redshift(z, h)


class _ClosureCatalog:
    """Minimal catalog handler exposing reduced_galaxy_catalog (z, M columns)."""

    def __init__(self, z: npt.NDArray[np.float64], M: npt.NDArray[np.float64]) -> None:
        self.reduced_galaxy_catalog = pd.DataFrame(
            {InternalCatalogColumns.REDSHIFT: z, InternalCatalogColumns.BH_MASS: M}
        )


# completeness shapes ---------------------------------------------------------
def f_const(value: float) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    return lambda z: value + 0.0 * z


def f_declining(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.clip(1.0 - z / 0.35, 0.05, 1.0)


def f_one(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.ones_like(z)


# ---------------------------------------------------------------------------
# Synthetic population + injection (closure)
# ---------------------------------------------------------------------------
def sample_population(
    rng: np.random.Generator, n_gal: int, h_true: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """z ~ (1/(1+z)) dVc/dz; log10 M ~ phi_MBH * R_eff."""
    zg = np.linspace(_Z_MIN, _Z_MAX, 4000)
    wz = np.asarray(comoving_volume_element(zg, h=h_true), dtype=np.float64) / (1.0 + zg)
    cdf = np.cumsum(wz)
    cdf /= cdf[-1]
    z = np.interp(rng.uniform(size=n_gal), cdf, zg)

    lmg = np.linspace(np.log10(_M_MIN), np.log10(_M_MAX), 2000)
    wm = np.asarray(mbh_mass_function(10.0**lmg), dtype=np.float64) * np.asarray(
        R_eff_per_mbh(10.0**lmg), dtype=np.float64
    )
    cdfm = np.cumsum(wm)
    cdfm /= cdfm[-1]
    M = 10.0 ** np.interp(rng.uniform(size=n_gal), cdfm, lmg)
    return z, M


def sample_real_nz_population(
    rng: np.random.Generator, n_gal: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Synthetic population whose REDSHIFT density follows the real GLADE n(z).

    Resamples z from the real catalogue redshifts (capturing GLADE's peaked /
    incompleteness-declining n(z) shape) and assigns masses from the synthetic
    mass function. Because L_cat is a normalised ratio of sums, only the SHAPE of
    n(z) matters -- not the 2.2M galaxy count -- so a modest n_gal reproduces the
    density-gradient effect cheaply (no sky, fast candidate sums).
    """
    cat_z, _cat_M, _h = load_real_catalog()
    z = rng.choice(cat_z, size=n_gal, replace=True)
    # jitter within the resampling bin width to avoid exact z duplicates
    z = z + rng.normal(0.0, 1e-4, size=n_gal)
    z = np.clip(z, _Z_MIN, None)
    lmg = np.linspace(np.log10(_M_MIN), np.log10(_M_MAX), 2000)
    wm = np.asarray(mbh_mass_function(10.0**lmg), dtype=np.float64) * np.asarray(
        R_eff_per_mbh(10.0**lmg), dtype=np.float64
    )
    cdfm = np.cumsum(wm)
    cdfm /= cdfm[-1]
    M = 10.0 ** np.interp(rng.uniform(size=n_gal), cdfm, lmg)
    return z.astype(np.float64), M


def inject_events_synthetic(
    rng: np.random.Generator,
    z_all: npt.NDArray[np.float64],
    M_all: npt.NDArray[np.float64],
    in_catalog_mask: npt.NDArray[np.bool_],
    h_true: float,
    n_target: int,
    sigma_fracs: npt.NDArray[np.float64] | float,
) -> list[dict]:
    """Rate-weighted hosts -> synthetic d_L_meas (Gaussian scatter) -> p_det filter.

    ``sigma_fracs`` may be a scalar (fixed fractional error) or an array to draw
    per-event fractional errors from (the real-sigma rung passes the empirical
    sigma_dL/d_L pool).
    """
    w = np.asarray(R_eff_per_mbh(M_all), dtype=np.float64) / (1.0 + z_all)
    p = w / w.sum()
    pool = np.atleast_1d(np.asarray(sigma_fracs, dtype=np.float64))
    events: list[dict] = []
    tries = 0
    while len(events) < n_target and tries < 400 * n_target:
        tries += 1
        g = int(rng.choice(len(z_all), p=p))
        z_host = float(z_all[g])
        d_true = float(dist(z_host, h=h_true))  # Gpc
        sigma_frac = float(pool[rng.integers(len(pool))]) if pool.size > 1 else float(pool[0])
        sigma_dL = sigma_frac * d_true
        d_meas = d_true + sigma_dL * rng.standard_normal()
        if d_meas <= 0:
            continue
        if rng.uniform() < float(_p_det_of_dl(np.asarray([d_meas]))[0]):
            events.append(
                {
                    "d_meas": d_meas,
                    "sigma_dL": sigma_dL,
                    "z_host": z_host,
                    "M_host": float(M_all[g]),
                    "in_catalog": bool(in_catalog_mask[g]),
                }
            )
    return events


# ---------------------------------------------------------------------------
# Per-event partition-norm likelihood (1-D d_L channel; mirrors closure)
# ---------------------------------------------------------------------------
def event_log_likelihood(
    event: dict,
    catalog_z: npt.NDArray[np.float64],
    catalog_M: npt.NDArray[np.float64],
    completeness: ZCompleteness,
    h: float,
    D_h: float,
    beta_G: float,
    global_denom: float,
    *,
    sorted_z: bool = False,
) -> float:
    """log p_i(h) = log[(beta_G * L_cat + B_num) / D(h)] for one event.

    If ``sorted_z`` the catalogue arrays are pre-sorted ascending in z and the
    candidate window is sliced with ``searchsorted`` (O(log N)) -- required for
    the 2.2M-galaxy real catalogue.
    """
    d_meas = event["d_meas"]
    sigma_dL = event["sigma_dL"]

    def p_gw_of_z(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        d_model = np.asarray(dist_vectorized(z, h=h), dtype=np.float64)
        return norm.pdf(d_model, loc=d_meas, scale=sigma_dL)

    # L_cat over candidate catalog galaxies in an H0-independent 5-sigma window
    z_lo = dist_to_redshift(max(d_meas - 5.0 * sigma_dL, 1e-4), h=0.60)
    z_hi = dist_to_redshift(d_meas + 5.0 * sigma_dL, h=0.80)
    if sorted_z:
        i0 = int(np.searchsorted(catalog_z, z_lo, side="left"))
        i1 = int(np.searchsorted(catalog_z, z_hi, side="right"))
        zc = catalog_z[i0:i1]
        Mc = catalog_M[i0:i1]
    else:
        cand = (catalog_z >= z_lo) & (catalog_z <= z_hi)
        zc = catalog_z[cand]
        Mc = catalog_M[cand]
    cat_num_sum = 0.0
    if zc.size:
        wc = np.asarray(R_eff_per_mbh(Mc), dtype=np.float64) / (1.0 + zc)
        cat_num_sum = float(np.sum(wc * p_gw_of_z(zc)))
    L_cat = cat_num_sum / global_denom if global_denom > 0 else 0.0

    # B_num = INTEGRAL (1-f(z)) p_GW(z) (1/(1+z)) dVc/dz over the event window
    bz_lo = max(dist_to_redshift(max(d_meas - 4.0 * sigma_dL, 1e-4), h=h), 1e-6)
    bz_hi = dist_to_redshift(d_meas + 4.0 * sigma_dL, h=h)

    def b_integrand(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        f_z = completeness.get_completeness_at_redshift(z, h)
        dVc = np.asarray(comoving_volume_element(z, h=h), dtype=np.float64)
        return (1.0 - f_z) * p_gw_of_z(z) * dVc / (1.0 + z)

    B_num = float(fixed_quad(b_integrand, bz_lo, bz_hi, n=50)[0])

    p_i = (beta_G * L_cat + B_num) / D_h if D_h > 0 else 0.0
    return float(np.log(p_i)) if p_i > 0 else -1e30


# ---------------------------------------------------------------------------
# Real-data loaders
# ---------------------------------------------------------------------------
def load_real_detections(apply_cuts: bool = True) -> dict[str, npt.NDArray[np.float64]]:
    """Load the seed-600 detections: measured & true d_L, sigma_dL, SNR, in_catalog.

    Returns arrays aligned by event. ``apply_cuts`` applies the inference's
    SNR>=20 and relative-distance-error<0.10 filters.
    """
    prep = pd.read_csv(_PREPARED_CRB)
    raw = pd.read_csv(_RAW_CRB)
    d_meas = prep["luminosity_distance"].to_numpy(dtype=np.float64)
    d_true = raw["luminosity_distance"].to_numpy(dtype=np.float64)
    sigma_dL = np.sqrt(
        prep["delta_luminosity_distance_delta_luminosity_distance"].to_numpy(dtype=np.float64)
    )
    snr = prep["SNR"].to_numpy(dtype=np.float64)
    in_cat = raw["in_catalog"].to_numpy()
    phi = prep["phiS"].to_numpy(dtype=np.float64)
    theta = prep["qS"].to_numpy(dtype=np.float64)
    keep = np.ones(len(prep), dtype=bool)
    if apply_cuts:
        keep = (snr >= 20.0) & (sigma_dL / d_meas < 0.10)
    return {
        "d_meas": d_meas[keep],
        "d_true": d_true[keep],
        "sigma_dL": sigma_dL[keep],
        "snr": snr[keep],
        "in_catalog": in_cat[keep].astype(bool),
        "phi": phi[keep],
        "theta": theta[keep],
    }


def real_sigma_frac_pool(apply_cuts: bool = True) -> npt.NDArray[np.float64]:
    """Empirical sigma_dL/d_L pool from the real detections (for the real-sigma rung)."""
    d = load_real_detections(apply_cuts=apply_cuts)
    return d["sigma_dL"] / d["d_meas"]


def real_events_from_crb(apply_cuts: bool = True) -> list[dict]:
    """Build event dicts straight from the real CRBs (measured d_L + real sigma)."""
    d = load_real_detections(apply_cuts=apply_cuts)
    z_true = np.asarray(dist_to_redshift_vec(d["d_true"], TRUE_H))
    return [
        {
            "d_meas": float(d["d_meas"][i]),
            "sigma_dL": float(d["sigma_dL"][i]),
            "z_host": float(z_true[i]),
            "M_host": float("nan"),
            "in_catalog": bool(d["in_catalog"][i]),
        }
        for i in range(len(d["d_meas"]))
    ]


def dist_to_redshift_vec(d_L: npt.NDArray[np.float64], h: float) -> npt.NDArray[np.float64]:
    return np.array([dist_to_redshift(float(x), h=h) for x in np.atleast_1d(d_L)])


_REAL_CATALOG_CACHE: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Any] | None = None


def load_real_catalog() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Any]:
    """Instantiate the REAL GalaxyCatalogueHandler exactly as main.py does.

    Returns (z, M, handler). ``M`` is the catalog BH/stellar mass column the real
    rate weight uses. Cached across rungs (the load is heavy).
    """
    global _REAL_CATALOG_CACHE
    if _REAL_CATALOG_CACHE is not None:
        return _REAL_CATALOG_CACHE
    from master_thesis_code.cosmological_model import Model1CrossCheck
    from master_thesis_code.galaxy_catalogue.handler import GalaxyCatalogueHandler

    cm = Model1CrossCheck(rng=np.random.default_rng(0))
    handler = GalaxyCatalogueHandler(
        M_min=cm.parameter_space.M.lower_limit,
        M_max=cm.parameter_space.M.upper_limit,
        z_max=cm.max_redshift,
    )
    cat = handler.reduced_galaxy_catalog
    z = cat[InternalCatalogColumns.REDSHIFT].to_numpy(dtype=np.float64)
    M = cat[InternalCatalogColumns.BH_MASS].to_numpy(dtype=np.float64)
    good = np.isfinite(z) & np.isfinite(M) & (z > 0)
    _REAL_CATALOG_CACHE = (z[good], M[good], handler)
    return _REAL_CATALOG_CACHE


# ---------------------------------------------------------------------------
# Config-driven bridge runner
# ---------------------------------------------------------------------------
@dataclass
class BridgeConfig:
    name: str
    # ingredients (synthetic defaults = closure baseline)
    catalog: str = "synthetic"  # "synthetic" | "real"
    events: str = "synthetic"  # "synthetic" | "real"
    sigma_model: str = "fixed"  # "fixed" | "real_dist"
    sigma_frac: float = 0.05
    completeness: str = "const"  # "const" | "declining" | "one"
    f_value: float = 0.6
    # scale
    n_gal: int = 8000
    n_events: int = 300
    h_true: float = TRUE_H
    seed: int = 0
    h_grid: list[float] = field(default_factory=lambda: list(np.round(np.arange(0.60, 0.8701, 0.01), 4)))


def _make_completeness(cfg: BridgeConfig) -> ZCompleteness:
    if cfg.completeness == "one":
        return ZCompleteness(f_one)
    if cfg.completeness == "declining":
        return ZCompleteness(f_declining)
    return ZCompleteness(f_const(cfg.f_value))


def run_bridge(cfg: BridgeConfig, verbose: bool = True) -> dict:
    rng = np.random.default_rng(cfg.seed)
    hs = [float(h) for h in cfg.h_grid]
    completeness = _make_completeness(cfg)
    pdet = MockPdet()

    # --- catalogue ---------------------------------------------------------
    if cfg.catalog == "real":
        cat_z, cat_M, handler = load_real_catalog()
        catalog_obj: Any = handler
        # full population for synthetic injection = the real catalogue itself
        pop_z, pop_M = cat_z, cat_M
        in_cat_mask = np.ones(len(cat_z), dtype=bool)
    else:
        if cfg.catalog == "real_nz":
            pop_z, pop_M = sample_real_nz_population(rng, cfg.n_gal)
        else:
            pop_z, pop_M = sample_population(rng, cfg.n_gal, cfg.h_true)
        f_at = np.asarray(
            completeness.get_completeness_at_redshift(pop_z), dtype=np.float64
        )
        in_cat_mask = rng.uniform(size=len(pop_z)) < f_at
        cat_z, cat_M = pop_z[in_cat_mask], pop_M[in_cat_mask]
        catalog_obj = _ClosureCatalog(cat_z, cat_M)

    # --- sigma model -------------------------------------------------------
    if cfg.sigma_model == "real_dist":
        sigma_fracs: npt.NDArray[np.float64] | float = real_sigma_frac_pool()
    else:
        sigma_fracs = cfg.sigma_frac

    # --- events ------------------------------------------------------------
    if cfg.events == "real":
        events = real_events_from_crb()
    else:
        events = inject_events_synthetic(
            rng, pop_z, pop_M, in_cat_mask, cfg.h_true, cfg.n_events, sigma_fracs
        )
    n_in = int(sum(e["in_catalog"] for e in events))

    # --- precomputes (real production functions) ---------------------------
    D_tab = precompute_completion_denominator(hs, pdet, Omega_m=_OMEGA_M, Omega_DE=_OMEGA_DE)
    bGbar_tab = precompute_missing_completion_denominator(hs, pdet, completeness=completeness)
    gdenom_tab = precompute_global_catalog_selection(hs, catalog_obj, pdet, with_bh_mass=False)

    # sort catalogue ascending in z for the searchsorted candidate slice
    order = np.argsort(cat_z)
    cat_z_s, cat_M_s = cat_z[order], cat_M[order]

    # --- posterior ---------------------------------------------------------
    logpost = np.zeros(len(hs))
    for i, h in enumerate(hs):
        D_h = D_tab[h]
        beta_G = D_h - bGbar_tab[h]
        gd = gdenom_tab[h]
        total = 0.0
        for ev in events:
            total += event_log_likelihood(
                ev, cat_z_s, cat_M_s, completeness, h, D_h, beta_G, gd, sorted_z=True
            )
        logpost[i] = total

    result = extract_map(hs, logpost, cfg.h_true)
    result.update(
        {
            "name": cfg.name,
            "config": cfg.__dict__,
            "n_events": len(events),
            "n_in_catalog": n_in,
            "n_catalog_galaxies": int(len(cat_z)),
        }
    )
    if verbose:
        print(
            f"[{cfg.name}] n_events={len(events)} in_cat={n_in} "
            f"MAP(grid)={result['h_map']} MAP(parab)={result['h_refined']:.4f} "
            f"bias={result['bias']:+.4f}"
        )
    return result


def extract_map(hs: list[float], logpost: npt.NDArray[np.float64], h_true: float) -> dict:
    lp = logpost - logpost.max()
    i_map = int(np.argmax(lp))
    h_map = hs[i_map]
    if 0 < i_map < len(hs) - 1:
        y0, y1, y2 = lp[i_map - 1], lp[i_map], lp[i_map + 1]
        denom = y0 - 2 * y1 + y2
        dh = hs[1] - hs[0]
        h_refined = h_map + 0.5 * (y0 - y2) / denom * dh if denom != 0 else h_map
    else:
        h_refined = h_map  # rail: grid edge
    return {
        "h_true": h_true,
        "h_map": float(h_map),
        "h_refined": float(h_refined),
        "bias": float(h_refined - h_true),
        "railed": bool(i_map == 0 or i_map == len(hs) - 1),
        "hs": [float(x) for x in hs],
        "logpost": [float(x) for x in lp],
    }


def save_result(result: dict, filename: str) -> Path:
    path = OUTPUTS / filename
    path.write_text(json.dumps(result, indent=2))
    return path
