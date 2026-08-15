"""M7-L0 toy: A/B frozen-edge harness isolating the boundary-layer flux.

Registered in ``M7_L0_DERIVATION_20260815.md`` §3 (ledger row #103). Adapts the
committed ``toys/m3_toy.py`` A/B estimator mirror (same GL-50 quadrature, same
bootstrapped sigma_dL from the CRB CSV) but replaces m3_toy's flat
``SIG_Z = 0.042`` with the production GLADE-empirical sigma_z mix, reused
directly from ``darksiren_emri.validation.venue_transfer`` (the VT-D3
z-decile sampler built against the same pruned production catalogue the
venue uses), rather than reimplemented.

Arm A: domain = [max(z_lo(h), z_obs-5*sigma_z), min(z_hi(h), z_obs+5*sigma_z)]
       (h-moving edges, "as coded")
Arm B: identical, but z_lo, z_hi FROZEN at their h_true values for every h
       (kills both M3's interior-clip channel and M7's boundary-layer channel
       together; the term cancels exactly between arms per the note's
       False-read note, §3)

Population: per event, K candidates (1 exact host + K-1 impostors) are drawn
by TRUE z; membership is fixed at h_true (uniform-in-comoving-volume inside
the ball window built at h_true from d_obs, h-independent thereafter,
matching the production fixed-K ball). z_obs = z_true + sigma_z * eps for
scattered candidates; sigma_z is drawn per candidate from the GLADE-empirical
z-decile mix (production VT-D3 recipe). Two host variants:

  - "scattered" (full-dose venue, primary/registered): every candidate,
    including the host, is z-scattered -- this matches the null-arm venue
    the note registers the read against.
  - "exact" (reported if cheap, at full dose only): host z_obs = z_true (no
    noise); impostors still scattered.

Dose f_i in {0.25, 0.5, 1.0} scales sigma_z for IMPOSTORS ONLY (the paper's
dose axis); the host's own scatter, when applied, is never dosed.

Reported per (dose, host-variant): the stacked slope
d/dh[sum_i(lnL_i^A - lnL_i^B)] at h_true, scaled to 982 events; the toy's own
joint (arm-A) posterior curvature -> sigma_post^2 (Laplace); S_need =
Delta/sigma_post^2 with Delta = +0.0192 (the A-M2' residual,
STAGE2_READOUT.md); the implied MAP shift = slope_982 * sigma_post^2, both
combined (M3+M7, what A-B actually measures) and with the M3 background
(~6e-7 in h, from the M3 note) subtracted to isolate M7 alone; seed scatter
over >= 8 seeds. The realized boundary-layer population fraction (candidates
with z_obs within 1 sigma_z of an edge of the h_true window, and the fraction
fully outside it) is reported alongside.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm

from darksiren_emri.physical_relations import comoving_volume_element, dist_vectorized
from darksiren_emri.validation import closed_loop_gfrac as cl
from darksiren_emri.validation import venue_transfer as vt

H_TRUE = 0.73
N_EV = 500
K = 400
NQ = 50
KERN_WINDOW = 5.0  # _IMPOSTOR_KERNEL_WINDOW (prereg convention, matches m3_toy)
SIGMA_WINDOW = 4.0  # ball half-width in sigma_d
DOSES: tuple[float, ...] = (0.25, 0.5, 1.0)
SEEDS: tuple[int, ...] = (101, 102, 103, 104, 105, 106, 107, 108)
DH = 0.005
N_PRODUCTION = 982
DELTA_RESIDUAL = 0.0192  # A-M2' residual, STAGE2_READOUT.md (+0.019200)
M3_BACKGROUND = 6.0e-7  # M3's own implied MAP shift in h (M3_truncation_window.md), CLOSED

_x, _w = leggauss(NQ)

# ---- z <-> d table (h=1 baseline grid) and comoving-volume CDF -------------
# The CDF shape is h-invariant (w_pop(z,h) = v(z,h=1)/h^3, a multiplicative
# constant that cancels from any conditional draw inside a fixed z-window).
_ZT = np.linspace(1e-6, 3.0, 20001)
_VT_H1 = np.asarray(comoving_volume_element(_ZT, h=1.0), dtype=np.float64) / (1.0 + _ZT)
_VC_H1 = np.concatenate([[0.0], np.cumsum(0.5 * (_VT_H1[1:] + _VT_H1[:-1]) * np.diff(_ZT))])


def vcdf(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Comoving-volume CDF shape (h-invariant), evaluated at z."""
    return np.asarray(np.interp(z, _ZT, _VC_H1), dtype=np.float64)


def vinv(c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Inverse of :func:`vcdf`."""
    return np.asarray(np.interp(c, _VC_H1, _ZT), dtype=np.float64)


def dist_at(z: npt.NDArray[np.float64], h: float) -> npt.NDArray[np.float64]:
    """Luminosity distance (Gpc) at the given h, floored to avoid z=0 singularities."""
    return np.asarray(dist_vectorized(np.maximum(z, 1e-8), h=h), dtype=np.float64)


def dist_table(h: float) -> npt.NDArray[np.float64]:
    """d(z) on the shared _ZT grid at the given h, for interpolation/inversion."""
    return dist_at(_ZT, h)


# ---- GLADE-empirical sigma_z mix (VT-D3 z-decile sampler, reused) ----------


def load_glade_sigma_mix() -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]]:
    """Build the production z-decile sigma_z sampler tables against the pruned catalogue.

    Reuses ``venue_transfer.load_pruned_z_sigma`` + ``venue_transfer.build_sigma_sampler``
    verbatim (the VT-D3 recipe) rather than reimplementing the mix.
    """
    z_cat, sz_cat = vt.load_pruned_z_sigma(vt.PRUNED_CATALOGUE_CSV)
    edges, pools = vt.build_sigma_sampler(z_cat, sz_cat)
    return edges, pools


def draw_sigma_z(
    z_true: npt.NDArray[np.float64],
    edges: npt.NDArray[np.float64],
    pools: list[npt.NDArray[np.float64]],
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Per-candidate sigma_z, z-decile matched -- mirrors venue_transfer.draw_member_sigma_z."""
    dec = np.searchsorted(edges, z_true, side="right")
    out = np.empty(z_true.size, dtype=np.float64)
    for b in range(len(pools)):
        m = dec == b
        if np.any(m):
            pool = pools[b]
            out[m] = pool[rng.integers(0, pool.size, size=int(m.sum()))]
    return out


# ---- event / candidate population ------------------------------------------


@dataclass
class EventSet:
    """One realised population: N_EV events, K candidates each (col 0 = host)."""

    d_obs: npt.NDArray[np.float64]  # (N_EV,)
    sig_d: npt.NDArray[np.float64]  # (N_EV,) fractional sigma_dL/d_L
    zlo0: npt.NDArray[np.float64]  # (N_EV,) ball window edges built at h_true
    zhi0: npt.NDArray[np.float64]
    z_cand: npt.NDArray[np.float64]  # (N_EV, K) true z, membership fixed at h_true
    sigma_z: npt.NDArray[np.float64]  # (N_EV, K) undosed per-candidate kernel width


def build_events(
    seed: int,
    sig_d_pool: npt.NDArray[np.float64],
    edges: npt.NDArray[np.float64],
    pools: list[npt.NDArray[np.float64]],
) -> EventSet:
    """Draw one population: membership on TRUE z, uniform-in-volume inside the h_true window."""
    rng = np.random.default_rng(seed)
    z_true_host = vinv(rng.random(N_EV) * vcdf(_ZT[-1]))
    sig_d = sig_d_pool[rng.integers(0, sig_d_pool.size, N_EV)]
    d_true = dist_at(z_true_host, H_TRUE)
    d_obs = d_true * (1.0 + sig_d * rng.standard_normal(N_EV))

    dt_htrue = dist_table(H_TRUE)
    zlo0 = np.maximum(
        np.interp(d_obs * (1 - SIGMA_WINDOW * sig_d), dt_htrue, _ZT).astype(np.float64), 1e-6
    )
    zhi0 = np.minimum(
        np.interp(d_obs * (1 + SIGMA_WINDOW * sig_d), dt_htrue, _ZT).astype(np.float64), _ZT[-1]
    )
    f_lo, f_hi = vcdf(zlo0), vcdf(zhi0)
    u = rng.random((N_EV, K - 1))
    z_imp = vinv(f_lo[:, None] + (f_hi - f_lo)[:, None] * u)
    z_cand = np.concatenate([z_true_host[:, None], z_imp], axis=1)

    sigma_z = draw_sigma_z(z_cand.reshape(-1), edges, pools, rng).reshape(N_EV, K)
    return EventSet(d_obs=d_obs, sig_d=sig_d, zlo0=zlo0, zhi0=zhi0, z_cand=z_cand, sigma_z=sigma_z)


def scatter_rng(seed: int, dose: float, variant: str) -> np.random.Generator:
    """Deterministic RNG stream for the noise draw, independent of the membership draw."""
    variant_code = 0 if variant == "scattered" else 1
    return np.random.default_rng([seed, int(round(dose * 1000)), variant_code])


def make_z_obs(
    ev: EventSet, seed: int, dose: float, host_variant: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """z_obs and the effective (dosed) per-candidate kernel width for one configuration."""
    rng = scatter_rng(seed, dose, host_variant)
    eff_sigma = ev.sigma_z.copy()
    eff_sigma[:, 1:] *= dose  # dose scales IMPOSTORS only; host (col 0) is never dosed
    eps = rng.standard_normal(ev.z_cand.shape)
    z_obs = ev.z_cand + eff_sigma * eps
    if host_variant == "exact":
        z_obs = z_obs.copy()
        z_obs[:, 0] = ev.z_cand[:, 0]
    return z_obs, eff_sigma


# ---- estimator mirror (GL-50, m3_toy's A/B kernel-and-window integral) ----


def ln_l(
    h: float,
    ev: EventSet,
    z_obs: npt.NDArray[np.float64],
    so: npt.NDArray[np.float64],
    edge_mode: str,
) -> npt.NDArray[np.float64]:
    """Per-event log-likelihood at h, arm A (edges move with h) or B (edges frozen at h_true)."""
    if edge_mode == "A":
        dt_h = dist_table(h)
        zlo = np.maximum(
            np.interp(ev.d_obs * (1 - SIGMA_WINDOW * ev.sig_d), dt_h, _ZT).astype(np.float64),
            1e-6,
        )
        zhi = np.minimum(
            np.interp(ev.d_obs * (1 + SIGMA_WINDOW * ev.sig_d), dt_h, _ZT).astype(np.float64),
            _ZT[-1],
        )
    elif edge_mode == "B":
        zlo = ev.zlo0
        zhi = ev.zhi0
    else:
        raise ValueError(f"unknown edge_mode {edge_mode!r}")

    a = np.maximum(zlo[:, None], z_obs - KERN_WINDOW * so)
    b = np.minimum(zhi[:, None], z_obs + KERN_WINDOW * so)
    valid = b > a
    half = 0.5 * (b - a)
    mid = 0.5 * (b + a)
    zn = mid[..., None] + half[..., None] * _x
    dn = dist_at(zn.reshape(-1), h).reshape(zn.shape)
    frac = dn / ev.d_obs[:, None, None]
    pgw = norm.pdf(frac, loc=1.0, scale=ev.sig_d[:, None, None])
    kern = norm.pdf(zn, loc=z_obs[..., None], scale=so[..., None])
    c = half * ((kern * pgw) @ _w)
    c = np.where(valid, c, 0.0)
    ell = c.sum(axis=1) / K
    return np.asarray(np.where(ell > 0, np.log(np.where(ell > 0, ell, 1.0)), -745.0))


# ---- per-configuration measurement -----------------------------------------


def boundary_layer_fractions(
    ev: EventSet, z_obs: npt.NDArray[np.float64], so: npt.NDArray[np.float64]
) -> tuple[float, float]:
    """Fraction of candidates within 1 sigma_z of an h_true-window edge, and fully outside it."""
    zlo = ev.zlo0[:, None]
    zhi = ev.zhi0[:, None]
    near_edge = (np.abs(z_obs - zlo) <= so) | (np.abs(z_obs - zhi) <= so)
    outside = (z_obs < zlo) | (z_obs > zhi)
    n_total = float(z_obs.size)
    return float(np.sum(near_edge)) / n_total, float(np.sum(outside)) / n_total


def measure_one(ev: EventSet, seed: int, dose: float, host_variant: str) -> dict[str, float]:
    """One (seed, dose, host_variant) configuration: slope, curvature, implied shift, fractions."""
    z_obs, so = make_z_obs(ev, seed, dose, host_variant)
    hs = (H_TRUE - DH, H_TRUE, H_TRUE + DH)
    ln_a = [ln_l(h, ev, z_obs, so, "A") for h in hs]
    ln_b_lo = ln_l(hs[0], ev, z_obs, so, "B")
    ln_b_hi = ln_l(hs[2], ev, z_obs, so, "B")

    diff_lo = ln_a[0] - ln_b_lo
    diff_hi = ln_a[2] - ln_b_hi
    slope_per_ev = float((diff_hi - diff_lo).sum()) / (2 * DH) / N_EV
    slope_982 = slope_per_ev * N_PRODUCTION

    total_lo, total_mid, total_hi = (float(x.sum()) for x in ln_a)
    curvature_per_ev = -(total_hi - 2.0 * total_mid + total_lo) / DH**2 / N_EV
    curvature_982 = curvature_per_ev * N_PRODUCTION
    sigma_post2 = 1.0 / curvature_982 if curvature_982 > 0 else float("nan")
    s_need = DELTA_RESIDUAL / sigma_post2 if sigma_post2 == sigma_post2 else float("nan")

    implied_shift_combined = slope_982 * sigma_post2  # what A-B actually measures (M3+M7)
    implied_shift_m7 = implied_shift_combined - M3_BACKGROUND

    frac_near, frac_outside = boundary_layer_fractions(ev, z_obs, so)

    return {
        "slope_982": slope_982,
        "curvature_982": curvature_982,
        "sigma_post": float(np.sqrt(sigma_post2)) if sigma_post2 == sigma_post2 else float("nan"),
        "s_need": s_need,
        "implied_shift_combined": implied_shift_combined,
        "implied_shift_m7": implied_shift_m7,
        "frac_boundary_layer": frac_near,
        "frac_outside_window": frac_outside,
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    """Mean and seed-scatter (std) over a list of per-seed measure_one() dicts."""
    out: dict[str, float] = {}
    for key in rows[0]:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


def read_for(implied_shift_mean: float) -> str:
    """The registered §3 read, applied to the full-dose (f_i=1.0), scattered-host result."""
    if implied_shift_mean > 1.0e-3:
        return "M7-LIVE"
    if implied_shift_mean < -1.0e-3:
        return "M7-REFUTED-ON-SIGN"
    return "M7-CLOSED"


def main() -> None:
    sig_d_pool = cl.load_sigma_triples(cl.DEFAULT_CRB_CSV)[:, 0]
    edges, pools = load_glade_sigma_mix()

    results: dict[str, Any] = {"config": {}, "per_dose": {}, "registered_read": {}}
    results["config"] = {
        "H_TRUE": H_TRUE,
        "N_EV": N_EV,
        "K": K,
        "NQ": NQ,
        "SEEDS": list(SEEDS),
        "DOSES": list(DOSES),
        "DH": DH,
        "N_PRODUCTION": N_PRODUCTION,
        "DELTA_RESIDUAL": DELTA_RESIDUAL,
        "M3_BACKGROUND": M3_BACKGROUND,
        "sigma_z_mix": "GLADE-empirical (VT-D3 z-decile sampler, venue_transfer.py)",
    }

    # Membership (population, sig_d, sigma_z-kernel-widths) is drawn once per
    # seed and reused across every dose/host-variant -- only the noise draw
    # (make_z_obs) differs, since dose/host-variant scale/relocate the same
    # underlying population rather than redraw it.
    events_by_seed = {seed: build_events(seed, sig_d_pool, edges, pools) for seed in SEEDS}

    configs: list[tuple[str, float]] = [("scattered", dose) for dose in DOSES]
    configs.append(("exact", 1.0))

    for host_variant, dose in configs:
        print(f"--- host={host_variant} dose={dose} ---", flush=True)
        rows = []
        for seed in SEEDS:
            row = measure_one(events_by_seed[seed], seed, dose, host_variant)
            rows.append(row)
            print(
                f"  seed={seed} slope_982={row['slope_982']:.4e} "
                f"implied_shift_m7={row['implied_shift_m7']:+.4e} "
                f"frac_boundary={row['frac_boundary_layer']:.3f}",
                flush=True,
            )
        summary = summarize(rows)
        key = f"{host_variant}_f{dose:.2f}"
        results["per_dose"][key] = {
            "host_variant": host_variant,
            "dose": dose,
            "n_seeds": len(SEEDS),
            **summary,
        }
        if host_variant == "scattered" and dose == 1.0:
            results["registered_read"] = {
                "read": read_for(summary["implied_shift_m7_mean"]),
                "implied_shift_m7_mean": summary["implied_shift_m7_mean"],
                "implied_shift_m7_std": summary["implied_shift_m7_std"],
                "rule": (
                    "M7-CLOSED in [-1e-3, +1e-3]; M7-LIVE if > +1e-3; "
                    "M7-REFUTED-ON-SIGN if < -1e-3 (M7_L0_DERIVATION_20260815.md Sec 3)"
                ),
            }

    out_path = Path(__file__).resolve().parent.parent / "M7_L0_toy_output.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")
    print(json.dumps(results["registered_read"], indent=2))


if __name__ == "__main__":
    main()
