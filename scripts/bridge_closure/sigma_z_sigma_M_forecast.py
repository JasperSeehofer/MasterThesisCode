"""sigma_z / sigma_M precision-forecast heatmap (LISA EMRI dark-siren H0).

This is the *forecast* engine for the paper. The photo-z railing investigation
proved the in-catalogue PHOTOMETRIC channel at GLADE's regime (sigma_z ~ 0.035,
sigma_z/z ~ 0.7) is information-starved. The forecast asks the constructive
question: *what redshift / host-mass measurement precision (sigma_z, sigma_M)
makes a LISA dark-siren H0 useful, and where is it futile?*

Method (Route A of HANDOFF-SIGMAZ-SIGMAM-FORECAST-20260630.md): use the
SELF-CONSISTENT closure (unbiased by construction) as the engine, so the
posterior WIDTH is a clean measure of information content. We extend the
bridge ``rung_I`` closure with

  1. posterior-width reporting (sigma(h), plus RMSE-around-truth),
  2. a sigma_M (host BH-mass error) axis = the WITH-BH-MASS (2-D) channel,
     mirroring the production ``single_host_likelihood`` mass term, and
  3. a multiprocessing grid sweep over (sigma_z, sigma_M) -> heatmap.

The 2-D channel adds an h-INDEPENDENT redshift anchor: the GW measures the
EMRI detector-frame mass M_z precisely, and the host's source-frame mass M_g
gives  1 + z = M_z / M_g  (precision ~ sigma_M * (1+z)). At trial h the
h-dependent GW-distance redshift z*(h) must agree with this mass anchor ->
the mass channel constrains H0 directly, with an effective redshift precision
~ min(sigma_z, sigma_M*(1+z)). Hence the hypothesis: the 2-D channel tolerates
larger sigma_z than the 1-D channel.

Production-fidelity of the mass term (verified against bayesian_statistics.py
single_host_likelihood, with_bh_mass path):
  * source-frame host mass M_g, observer-frame hypothesis M_g*(1+z) (H3 fix);
  * numerator Gaussian product  N(M_z_meas; M_g*(1+z), sigma_Mz^2 + (sigma_M*M_g*(1+z))^2);
  * the mock p_det is mass-blind, so the selection denominator is identical to
    the 1-D channel -> the mass channel is a pure numerator information gain
    (the cleanest possible test of "does adding the mass channel narrow H0").

Run:
  uv run python scripts/bridge_closure/sigma_z_sigma_M_forecast.py --smoke
  uv run python scripts/bridge_closure/sigma_z_sigma_M_forecast.py --sweep
  uv run python scripts/bridge_closure/sigma_z_sigma_M_forecast.py --plot
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

logging.disable(logging.WARNING)
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import _bridge_lib as B  # noqa: E402,N812

from darksiren_emri.emri_rate import R_eff_per_mbh  # noqa: E402
from darksiren_emri.physical_relations import (  # noqa: E402
    dist,
    dist_to_redshift,
    dist_vectorized,
)

_RESULTS_JSON = B.OUTPUTS / "sigma_z_sigma_M_forecast.json"

# Reference operating points (host redshift error sigma_z), for figure overlays.
SIGMA_Z_GLADE_PHOTO = 0.035  # GLADE+ flag-1 photometric median (Dalya+ 2022)
SIGMA_Z_GLADE_SPEC = 0.0017  # GLADE+ flag-3 spectroscopic median


# ---------------------------------------------------------------------------
# Configuration of one forecast grid
# ---------------------------------------------------------------------------
@dataclass
class ForecastConfig:
    """A (sigma_z x sigma_M) grid swept with multi-seed closures."""

    h_true: float = B.TRUE_H
    # grids (log-spaced; chosen to resolve the convergence frontier sigma_M~sigma_z
    # seen in the smoke, plus GLADE photo-z 0.035 and the realistic large-sigma_M end)
    sigma_z_grid: list[float] = field(
        default_factory=lambda: [5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.5e-2, 2.5e-2, 5e-2]
    )
    # host BH-mass error, FRACTIONAL (sigma_M * M_g), matches production's linear
    # Gaussian. Spans the frontier (small) and the realistic intrinsic-scatter end
    # (~0.3-0.5 dex ~ fractional ~1-2, where the mass channel is useless).
    sigma_M_grid: list[float] = field(
        default_factory=lambda: [5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1.0]
    )
    seeds: list[int] = field(default_factory=lambda: list(range(8)))
    # population n(z) shape: "synthetic" (smooth dVc/(1+z)) or "real_nz" (resample
    # the real GLADE redshift density; a robustness check that the convergence
    # frontier does not depend on the n(z) shape).
    population: str = "synthetic"
    # scale (width ~ 1/sqrt(n_events); MAP location is N-independent)
    n_gal: int = 12000
    n_events: int = 400
    sigma_dL_frac: float = 0.05  # GW luminosity-distance fractional error
    sigma_Mz_frac: float = 1e-3  # GW detector-frame MBH-mass error (EMRIs: very tight)
    # widened H0 grid so a flat (uninformative) posterior gives a LARGE width and
    # rails sit clearly at the edges (HANDOFF pitfall: avoid the [0.60,0.87]
    # midpoint=0.735~truth artifact).
    h_lo: float = 0.50
    h_hi: float = 0.96
    h_step: float = 0.01


def _h_grid(cfg: ForecastConfig) -> list[float]:
    return [float(h) for h in np.round(np.arange(cfg.h_lo, cfg.h_hi + 1e-9, cfg.h_step), 4)]


# ---------------------------------------------------------------------------
# Posterior summary metrics
# ---------------------------------------------------------------------------
def _posterior_metrics(
    hs: list[float], logpost: npt.NDArray[np.float64], h_true: float
) -> dict[str, float | bool]:
    """Width, centred-ness, and rail diagnostics from a log-posterior."""
    hs_a = np.asarray(hs, dtype=np.float64)
    lp = logpost - np.max(logpost)
    P = np.exp(lp)
    Z = float(P.sum())
    if not np.isfinite(Z) or Z <= 0:
        return {
            "E_h": float("nan"),
            "width": float("nan"),
            "rmse_truth": float("nan"),
            "h_map": float("nan"),
            "bias": float("nan"),
            "railed": True,
            "edge_mass": 1.0,
        }
    P = P / Z
    E_h = float(np.sum(hs_a * P))
    width = float(np.sqrt(np.sum((hs_a - E_h) ** 2 * P)))
    rmse_truth = float(np.sqrt(np.sum((hs_a - h_true) ** 2 * P)))
    i_map = int(np.argmax(lp))
    h_map = hs[i_map]
    railed = bool(i_map == 0 or i_map == len(hs) - 1)
    # fraction of posterior mass within 1 step of either grid edge (flat/rail flag)
    edge_mass = float(P[0] + P[1] + P[-1] + P[-2])
    return {
        "E_h": E_h,
        "width": width,
        "rmse_truth": rmse_truth,
        "h_map": float(h_map),
        "bias": float(E_h - h_true),
        "railed": railed,
        "edge_mass": edge_mass,
    }


# ---------------------------------------------------------------------------
# Core: one self-consistent closure cell
#
# Two exact optimisations make the grid sweep cheap:
#   (1) d_L(z, h) = dist(z, 1)/h  -- the luminosity distance scales EXACTLY as
#       1/h (the E(z) integral is h-independent; physical_relations.dist:70-76).
#       So we precompute g(z)=dist(z,1) once per event; no per-h distance calls.
#   (2) only the GW-distance factor depends on h. The photo-z kernel and the
#       host-mass factor are h-INDEPENDENT, so per event we collapse the
#       candidate-galaxy dimension into an h-independent vector v[z] ONCE, and
#       the per-h numerator is a single matmul  num(h) = dz * (gw(h) @ v).
# ---------------------------------------------------------------------------
@dataclass
class _EventPre:
    """Per-event h-independent precompute (shared by the 1-D and all 2-D passes)."""

    d_meas: float
    sigma_dL: float
    M_z_meas: float
    sigma_Mz_abs: float
    g_grid: npt.NDArray[np.float64]  # dist(z_grid, h=1) -> d_L(z,h)=g_grid/h
    z_grid: npt.NDArray[np.float64]
    dz: float
    i0: int
    i1: int
    nm: npt.NDArray[np.float64]  # photo-z kernel N(z_grid; z_cat_g, sigma_z), (n_cand, nz)
    wg: npt.NDArray[np.float64]  # rate weights of candidate galaxies, (n_cand,)


def _precompute_event(
    ev: dict,
    zc: npt.NDArray[np.float64],
    wc: npt.NDArray[np.float64],
    sigma_z: float,
    *,
    h_lo: float,
    h_hi: float,
) -> _EventPre | None:
    sig_z = max(sigma_z, 1e-4)
    d_meas = ev["d_meas"]
    sdL = ev["sigma_dL"]
    z_lo = max(dist_to_redshift(max(d_meas - 5 * sdL, 1e-4), h=h_lo) - 4 * sig_z, 1e-5)
    z_hi = dist_to_redshift(d_meas + 5 * sdL, h=h_hi) + 4 * sig_z
    i0 = int(np.searchsorted(zc, z_lo))
    i1 = int(np.searchsorted(zc, z_hi))
    if i1 <= i0:
        return None
    zg = zc[i0:i1]
    wg = wc[i0:i1]
    ngrid = int(np.clip((z_hi - z_lo) / (0.4 * max(sig_z, 2e-3)), 120, 500))
    z_grid = np.linspace(z_lo, z_hi, ngrid)
    g_grid = np.asarray(dist_vectorized(z_grid, h=1.0), dtype=np.float64)  # d_L(z,h)=g/h
    nm = np.exp(-0.5 * ((z_grid[None, :] - zg[:, None]) / sig_z) ** 2) / (
        np.sqrt(2 * np.pi) * sig_z
    )
    return _EventPre(
        d_meas,
        sdL,
        ev["M_z_meas"],
        ev["sigma_Mz_abs"],
        g_grid,
        z_grid,
        float(z_grid[1] - z_grid[0]),
        i0,
        i1,
        nm,
        wg,
    )


def _accumulate(
    logpost: npt.NDArray[np.float64],
    pre: _EventPre,
    v: npt.NDArray[np.float64],
    gw: npt.NDArray[np.float64],
    gdenom_arr: npt.NDArray[np.float64],
) -> None:
    """logpost[h] += log( dz * (gw(h) @ v) / D(h) ), with safe -1e30 floor."""
    num = pre.dz * (gw @ v)  # (n_h,)
    L = np.where(gdenom_arr > 0, num / gdenom_arr, 0.0)
    logpost += np.where(L > 0, np.log(np.where(L > 0, L, 1.0)), -1e30)


def run_cell(
    sigma_z: float,
    sigma_M_grid: list[float],
    *,
    h_true: float,
    n_gal: int,
    n_events: int,
    sigma_dL_frac: float,
    sigma_Mz_frac: float,
    seed: int,
    hs: list[float],
    population: str = "synthetic",
) -> dict:
    """One (sigma_z, seed) closure: returns the 1-D metrics and the 2-D metrics
    for EVERY sigma_M in the grid (population/events/denominator are reused)."""
    rng = np.random.default_rng((seed + 1) * 1_000_003 + int(round(sigma_z * 1e6)))
    h_lo, h_hi = hs[0], hs[-1]

    # --- self-consistent population (smooth synthetic, or real GLADE n(z)) --
    if population == "real_nz":
        z_true, M_true = B.sample_real_nz_population(rng, n_gal)
    else:
        z_true, M_true = B.sample_population(rng, n_gal, h_true)
    # catalogue reports a NOISY redshift; mass noise drawn once (scaled per sigma_M)
    z_cat = np.clip(z_true + rng.normal(0.0, sigma_z, n_gal), 1e-3, None)
    e_mass = rng.standard_normal(n_gal)  # shared host-mass noise pattern (scaled by sigma_M)

    # --- rate-weighted injection at TRUE (z, M); measure noisy d_L & M_z ----
    w_true = np.asarray(R_eff_per_mbh(M_true), dtype=np.float64) / (1.0 + z_true)
    p = w_true / w_true.sum()
    events: list[dict] = []
    tries = 0
    while len(events) < n_events and tries < 400 * n_events:
        tries += 1
        g = int(rng.choice(n_gal, p=p))
        d_true = float(dist(z_true[g], h=h_true))
        sdL = sigma_dL_frac * d_true
        d_meas = d_true + sdL * rng.standard_normal()
        if d_meas <= 0:
            continue
        if rng.uniform() >= float(B._p_det_of_dl(np.asarray([d_meas]))[0]):
            continue
        M_z_true = float(M_true[g]) * (1.0 + float(z_true[g]))  # detector-frame source mass
        sMz = sigma_Mz_frac * M_z_true
        M_z_meas = M_z_true + sMz * rng.standard_normal()
        events.append(
            {
                "d_meas": d_meas,
                "sigma_dL": sdL,
                "M_z_meas": M_z_meas,
                "sigma_Mz_abs": sMz,
                "g_true": g,
            }
        )

    # --- sorted catalogue arrays (searchsorted candidate slice) -------------
    order = np.argsort(z_cat)
    zc = z_cat[order]
    Mt = M_true[order]
    em = e_mass[order]
    wc = np.asarray(R_eff_per_mbh(Mt), dtype=np.float64) / (1.0 + zc)

    # --- selection denominator (mass-blind mock p_det -> shared by both channels)
    pdet = B.MockPdet()
    catalog = B._ClosureCatalog(zc, Mt)
    gdenom = B.precompute_global_catalog_selection(hs, catalog, pdet, with_bh_mass=False)
    gdenom_arr = np.asarray([gdenom[h] for h in hs], dtype=np.float64)
    hs_arr = np.asarray(hs, dtype=np.float64)

    # noisy host masses per sigma_M (shared noise pattern em, scaled)
    Mcat_by_sM = {sM: np.clip(Mt * (1.0 + sM * em), 1.0e3, None) for sM in sigma_M_grid}

    lp_1d = np.zeros(len(hs))
    lp_2d = {sM: np.zeros(len(hs)) for sM in sigma_M_grid}

    for ev in events:
        pre = _precompute_event(ev, zc, wc, sigma_z, h_lo=h_lo, h_hi=h_hi)
        if pre is None:
            lp_1d += -1e30
            for sM in sigma_M_grid:
                lp_2d[sM] += -1e30
            continue
        # GW distance factor p_GW(z;h) = N(g(z)/h; d_meas, sigma_dL), (n_h, nz) -- ONCE
        d_model = pre.g_grid[None, :] / hs_arr[:, None]
        gw = norm.pdf(d_model, loc=pre.d_meas, scale=pre.sigma_dL)
        # 1-D: collapse candidates with the photo-z kernel only
        v1 = (pre.wg[:, None] * pre.nm).sum(axis=0)
        _accumulate(lp_1d, pre, v1, gw, gdenom_arr)
        # 2-D: add the host-mass factor m_g(z) = N(M_z_meas; M_g*(1+z), sMz^2+(sM*M_g*(1+z))^2)
        one_pz = 1.0 + pre.z_grid[None, :]
        for sM in sigma_M_grid:
            Mg = Mcat_by_sM[sM][pre.i0 : pre.i1]
            mu = Mg[:, None] * one_pz  # observer-frame host mass at trial z (n_cand, nz)
            sig2 = pre.sigma_Mz_abs**2 + (sM * mu) ** 2
            mm = np.exp(-0.5 * (pre.M_z_meas - mu) ** 2 / sig2) / np.sqrt(2 * np.pi * sig2)
            v2 = (pre.wg[:, None] * pre.nm * mm).sum(axis=0)
            _accumulate(lp_2d[sM], pre, v2, gw, gdenom_arr)

    m1 = _posterior_metrics(hs, lp_1d, h_true)
    twod = {f"{sM:.4g}": _posterior_metrics(hs, lp_2d[sM], h_true) for sM in sigma_M_grid}

    return {
        "sigma_z": sigma_z,
        "seed": seed,
        "n_events": len(events),
        "oned": m1,
        "twod": twod,
    }


# ---------------------------------------------------------------------------
# Sweep + aggregation
# ---------------------------------------------------------------------------
def _aggregate(cells: list[dict], cfg: ForecastConfig) -> dict:
    """Median-over-seeds aggregation into 2-D arrays for the heatmap."""
    sz = cfg.sigma_z_grid
    sM = cfg.sigma_M_grid
    sz_keys = [f"{x:.6g}" for x in sz]

    by_sz: dict[str, list[dict]] = {k: [] for k in sz_keys}
    for c in cells:
        by_sz[f"{c['sigma_z']:.6g}"].append(c)

    def med(vals: list[float]) -> float:
        v = [x for x in vals if np.isfinite(x)]
        return float(np.median(v)) if v else float("nan")

    # 1-D arrays (function of sigma_z only)
    oned = {
        "width": [med([c["oned"]["width"] for c in by_sz[k]]) for k in sz_keys],
        "rmse_truth": [med([c["oned"]["rmse_truth"] for c in by_sz[k]]) for k in sz_keys],
        "bias": [med([c["oned"]["bias"] for c in by_sz[k]]) for k in sz_keys],
        "rail_frac": [
            float(np.mean([c["oned"]["railed"] for c in by_sz[k]])) if by_sz[k] else float("nan")
            for k in sz_keys
        ],
    }
    # 2-D arrays (sigma_z x sigma_M)
    W = np.full((len(sz), len(sM)), np.nan)
    R = np.full((len(sz), len(sM)), np.nan)
    Bz = np.full((len(sz), len(sM)), np.nan)
    Rf = np.full((len(sz), len(sM)), np.nan)
    for i, k in enumerate(sz_keys):
        for j, m in enumerate(sM):
            mk = f"{m:.4g}"
            W[i, j] = med([c["twod"][mk]["width"] for c in by_sz[k]])
            R[i, j] = med([c["twod"][mk]["rmse_truth"] for c in by_sz[k]])
            Bz[i, j] = med([c["twod"][mk]["bias"] for c in by_sz[k]])
            Rf[i, j] = (
                float(np.mean([c["twod"][mk]["railed"] for c in by_sz[k]])) if by_sz[k] else np.nan
            )
    return {
        "config": asdict(cfg),
        "sigma_z_grid": sz,
        "sigma_M_grid": sM,
        "oned": oned,
        "twod": {
            "width": W.tolist(),
            "rmse_truth": R.tolist(),
            "bias": Bz.tolist(),
            "rail_frac": Rf.tolist(),
        },
        "n_seeds": len(cfg.seeds),
    }


def sweep(cfg: ForecastConfig, *, workers: int = 12, out: Path = _RESULTS_JSON) -> dict:
    hs = _h_grid(cfg)
    jobs = [(sz, seed) for sz in cfg.sigma_z_grid for seed in cfg.seeds]
    print(
        f"[sweep] {len(jobs)} (sigma_z x seed) cells x {len(cfg.sigma_M_grid)} sigma_M "
        f"| pop={cfg.population} n_events={cfg.n_events} n_gal={cfg.n_gal} | {len(hs)}-pt "
        f"h-grid [{hs[0]},{hs[-1]}] | workers={workers}",
        flush=True,
    )
    if cfg.population == "real_nz":  # load real catalogue ONCE so forked workers inherit the cache
        t = time.time()
        z, _M, _h = B.load_real_catalog()
        print(
            f"[sweep] real GLADE n(z) loaded: {len(z)} galaxies ({time.time() - t:.0f}s)",
            flush=True,
        )
    t0 = time.time()
    cells: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                run_cell,
                sz,
                cfg.sigma_M_grid,
                h_true=cfg.h_true,
                n_gal=cfg.n_gal,
                n_events=cfg.n_events,
                sigma_dL_frac=cfg.sigma_dL_frac,
                sigma_Mz_frac=cfg.sigma_Mz_frac,
                seed=seed,
                hs=hs,
                population=cfg.population,
            )
            for (sz, seed) in jobs
        ]
        for n, fut in enumerate(futs, 1):
            cells.append(fut.result())
            if n % max(1, len(futs) // 20) == 0 or n == len(futs):
                print(f"  {n}/{len(futs)} cells ({time.time() - t0:.0f}s)", flush=True)
    agg = _aggregate(cells, cfg)
    agg["raw_cells"] = cells
    agg["elapsed_s"] = time.time() - t0
    out.write_text(json.dumps(agg, indent=2))
    print(f"[sweep] done in {agg['elapsed_s']:.0f}s -> {out}", flush=True)
    return agg


# ---------------------------------------------------------------------------
# Smoke test (a few cells, single seed; sanity gate)
# ---------------------------------------------------------------------------
def smoke() -> None:
    print("=== SMOKE: self-consistent forecast closure ===", flush=True)
    hs = _h_grid(ForecastConfig())
    sM_grid = [0.02, 0.1, 0.4]
    for sz in [1e-3, 1e-2, 3.5e-2]:
        r = run_cell(
            sz,
            sM_grid,
            h_true=B.TRUE_H,
            n_gal=12000,
            n_events=400,
            sigma_dL_frac=0.05,
            sigma_Mz_frac=1e-3,
            seed=0,
            hs=hs,
        )
        o = r["oned"]
        print(f"\nsigma_z={sz:.4f}  (n_events={r['n_events']})", flush=True)
        print(
            f"  1-D : width={o['width']:.4f}  rmse_truth={o['rmse_truth']:.4f}  "
            f"E[h]={o['E_h']:.4f} bias={o['bias']:+.4f} railed={o['railed']} "
            f"edge_mass={o['edge_mass']:.2f}",
            flush=True,
        )
        for sM in sM_grid:
            t = r["twod"][f"{sM:.4g}"]
            print(
                f"  2-D sigma_M={sM:.2f}: width={t['width']:.4f}  rmse_truth={t['rmse_truth']:.4f}  "
                f"E[h]={t['E_h']:.4f} bias={t['bias']:+.4f} railed={t['railed']} "
                f"edge_mass={t['edge_mass']:.2f}",
                flush=True,
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="quick sanity gate (a few cells)")
    ap.add_argument("--sweep", action="store_true", help="run the full multi-seed grid sweep")
    ap.add_argument("--plot", action="store_true", help="render the heatmap from cached results")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n_events", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--population", choices=["synthetic", "real_nz"], default="synthetic")
    ap.add_argument("--out", type=str, default=str(_RESULTS_JSON))
    args = ap.parse_args()
    if args.smoke:
        smoke()
    if args.sweep:
        cfg = ForecastConfig(
            n_events=args.n_events, seeds=list(range(args.seeds)), population=args.population
        )
        sweep(cfg, workers=args.workers, out=Path(args.out))
    if args.plot:
        from _forecast_plot import plot_heatmap  # noqa: PLC0415

        plot_heatmap(json.loads(Path(args.out).read_text()))
    if not (args.smoke or args.sweep or args.plot):
        ap.print_help()


if __name__ == "__main__":
    main()
