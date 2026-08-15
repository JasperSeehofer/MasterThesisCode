"""L0-REN-B toy: A/B renormalization harness isolating the retained-kernel-mass tilt.

Registered in ``L0_REN_A_DERIVATION_20260815.md`` Sec 4 (ledger row #105 item 1).
Adapts the committed ``toys/m7_ab_toy.py`` A/B estimator mirror (same population
construction: membership fixed on TRUE z inside the h_true window, then scattered;
production GLADE-empirical sigma_z mix via ``venue_transfer``; CRB-bootstrapped
sigma_dL via ``closed_loop_gfrac``; K=400 candidates/event, ~500 events, >=8 seeds)
but changes the arm definitions and the dose convention:

Arm A: estimator as coded -- kernel-branch integral c_1k = int_a^b N(z; z_obs_k,
       sigma_k) * p_gw dz with a = max(z_lo(h), z_obs-5*sigma_k),
       b = min(z_hi(h), z_obs+5*sigma_k) (h-moving edges), truncated, NEVER
       divided by the retained kernel mass.
Arm B: identical -- same a, b at the same h, same draws -- except each
       candidate's integral is divided by its retained kernel mass
       W_k(h) = Phi((b-zo)/sigma_k) - Phi((a-zo)/sigma_k), using the SAME a, b
       used for the integral at that h (W is h-dependent through a, b and is
       recomputed at every h evaluation).

Both arms therefore share the h-moving window edges (unlike M7's frozen-edge B);
only the renormalization by W_k(h) differs.

Dose f_i in {0.25, 0.5, 1.0} scales sigma_z for EVERY candidate, host included
(the "full-dose venue configuration" the note calls for -- unlike M7's
impostor-only dosing).

Reported per dose: the stacked tilt T_REN = d/dh[sum_i(lnL_i^A - lnL_i^B)] at
h_true (central difference, same dh geometry as m7_ab_toy), scaled to the
982-event production population, with seed-scatter (std over >=8 seeds); the
implied MAP shift under BOTH conversions -- the toy's own arm-A joint-posterior
curvature (Laplace second difference), and the fixed production
sigma_post = 0.004386 (the M3-note / M7-addendum convention, squared and
multiplied by the slope directly, no toy curvature involved); and the
double-clipped / single-clipped / unclipped population fractions (candidates
whose retained kernel window is clipped on both, one, or neither box edge) at
h_true.
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
KERN_WINDOW = 5.0  # _IMPOSTOR_KERNEL_WINDOW (prereg convention, matches m7_ab_toy)
SIGMA_WINDOW = 4.0  # ball half-width in sigma_d
DOSES: tuple[float, ...] = (0.25, 0.5, 1.0)
SEEDS: tuple[int, ...] = (101, 102, 103, 104, 105, 106, 107, 108)
DH = 0.005
N_PRODUCTION = 982
SIGMA_POST_PRODUCTION = 0.004386  # M3-note / M7-addendum fixed production conversion

# Pre-stated reads (L0_REN_A_DERIVATION_20260815.md Sec 4)
R1_BAND = 1.0e-3
T_RES_STEPS = (-550.0, -212.0)  # measured T_res(0.25->0.5), T_res(0.5->1.0), nats/h
R2_TIGHT = 150.0
R2_WRONG = 300.0
T_RES_FULL_DOSE = -62.0
R3_BAND = 150.0

_x, _w = leggauss(NQ)

# ---- z <-> d table (h=1 baseline grid) and comoving-volume CDF -------------
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


def scatter_rng(seed: int, dose: float) -> np.random.Generator:
    """Deterministic RNG stream for the noise draw, independent of the membership draw."""
    return np.random.default_rng([seed, int(round(dose * 1000))])


def make_z_obs(
    ev: EventSet, seed: int, dose: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """z_obs and the effective (dosed) per-candidate kernel width for one configuration.

    Unlike m7_ab_toy, dose scales EVERY candidate's sigma_z -- host included -- the
    "full-dose venue configuration" this note's toy is registered against.
    """
    rng = scatter_rng(seed, dose)
    eff_sigma = ev.sigma_z * dose
    eps = rng.standard_normal(ev.z_cand.shape)
    z_obs = ev.z_cand + eff_sigma * eps
    return z_obs, eff_sigma


# ---- estimator mirror (GL-50, A = as-coded, B = A + per-candidate W_k(h)) -


def window_edges(h: float, ev: EventSet) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Box window edges [z_lo(h), z_hi(h)] -- h-moving, "as coded" (both arms share these)."""
    dt_h = dist_table(h)
    zlo = np.maximum(
        np.interp(ev.d_obs * (1 - SIGMA_WINDOW * ev.sig_d), dt_h, _ZT).astype(np.float64),
        1e-6,
    )
    zhi = np.minimum(
        np.interp(ev.d_obs * (1 + SIGMA_WINDOW * ev.sig_d), dt_h, _ZT).astype(np.float64),
        _ZT[-1],
    )
    return zlo, zhi


def ln_l_ab(
    h: float,
    ev: EventSet,
    z_obs: npt.NDArray[np.float64],
    so: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-event log-likelihood at h for arm A (unrenormalized) and arm B (/ W_k(h))."""
    zlo, zhi = window_edges(h, ev)

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

    # Arm A: unrenormalized (as coded).
    ell_a = c.sum(axis=1) / K
    ln_a = np.asarray(np.where(ell_a > 0, np.log(np.where(ell_a > 0, ell_a, 1.0)), -745.0))

    # Arm B: divide each candidate by its retained kernel mass W_k(h), SAME a, b.
    w_k = norm.cdf((b - z_obs) / so) - norm.cdf((a - z_obs) / so)
    w_k = np.where(valid, np.maximum(w_k, 1e-300), 1.0)
    c_b = np.where(valid, c / w_k, 0.0)
    ell_b = c_b.sum(axis=1) / K
    ln_b = np.asarray(np.where(ell_b > 0, np.log(np.where(ell_b > 0, ell_b, 1.0)), -745.0))

    return ln_a, ln_b


def clip_fractions(
    h: float, ev: EventSet, z_obs: npt.NDArray[np.float64], so: npt.NDArray[np.float64]
) -> tuple[float, float, float]:
    """Double-clipped / single-clipped / unclipped fractions of candidates at h."""
    zlo, zhi = window_edges(h, ev)
    kern_lo = z_obs - KERN_WINDOW * so
    kern_hi = z_obs + KERN_WINDOW * so
    clip_lo = zlo[:, None] > kern_lo
    clip_hi = zhi[:, None] < kern_hi
    double_clip = clip_lo & clip_hi
    single_clip = clip_lo ^ clip_hi
    unclipped = (~clip_lo) & (~clip_hi)
    n_total = float(z_obs.size)
    return (
        float(np.sum(double_clip)) / n_total,
        float(np.sum(single_clip)) / n_total,
        float(np.sum(unclipped)) / n_total,
    )


# ---- per-configuration measurement -----------------------------------------


def measure_one(ev: EventSet, seed: int, dose: float) -> dict[str, float]:
    """One (seed, dose) configuration: T_REN slope, both curvature conversions, fractions."""
    z_obs, so = make_z_obs(ev, seed, dose)
    hs = (H_TRUE - DH, H_TRUE, H_TRUE + DH)

    ln_a_lo, ln_b_lo = ln_l_ab(hs[0], ev, z_obs, so)
    ln_a_mid, ln_b_mid = ln_l_ab(hs[1], ev, z_obs, so)
    ln_a_hi, ln_b_hi = ln_l_ab(hs[2], ev, z_obs, so)

    diff_lo = float((ln_a_lo - ln_b_lo).sum())
    diff_hi = float((ln_a_hi - ln_b_hi).sum())
    t_ren_per_ev = (diff_hi - diff_lo) / (2 * DH) / N_EV
    t_ren_982 = t_ren_per_ev * N_PRODUCTION

    total_lo, total_mid, total_hi = (float(x.sum()) for x in (ln_a_lo, ln_a_mid, ln_a_hi))
    curvature_per_ev = -(total_hi - 2.0 * total_mid + total_lo) / DH**2 / N_EV
    curvature_982 = curvature_per_ev * N_PRODUCTION
    sigma_post2_toy = 1.0 / curvature_982 if curvature_982 > 0 else float("nan")

    implied_shift_toy = (
        t_ren_982 * sigma_post2_toy if sigma_post2_toy == sigma_post2_toy else float("nan")
    )
    implied_shift_prod = t_ren_982 * SIGMA_POST_PRODUCTION**2

    frac_double, frac_single, frac_unclipped = clip_fractions(H_TRUE, ev, z_obs, so)

    return {
        "t_ren_982": t_ren_982,
        "curvature_982": curvature_982,
        "sigma_post_toy": float(np.sqrt(sigma_post2_toy))
        if sigma_post2_toy == sigma_post2_toy
        else float("nan"),
        "implied_shift_toy_curvature": implied_shift_toy,
        "implied_shift_production_curvature": implied_shift_prod,
        "frac_double_clip": frac_double,
        "frac_single_clip": frac_single,
        "frac_unclipped": frac_unclipped,
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    """Mean and seed-scatter (std) over a list of per-seed measure_one() dicts."""
    out: dict[str, float] = {}
    for key in rows[0]:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


def apply_reads(per_dose: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Apply the four pre-stated reads (Sec 4 of the derivation note) mechanically."""
    t = {d: per_dose[f"f{d:.2f}"]["t_ren_982_mean"] for d in DOSES}
    shift_full = per_dose["f1.00"]["implied_shift_production_curvature_mean"]

    # R1: full-dose implied MAP shift, production conversion.
    if -R1_BAND <= shift_full <= R1_BAND:
        r1 = "CLOSED"
    else:
        r1 = "LIVE"

    # R2: dose-shape steps vs measured T_res steps (-550, -212).
    step1 = t[0.5] - t[0.25]
    step2 = t[1.0] - t[0.5]
    diff1 = abs(step1 - T_RES_STEPS[0])
    diff2 = abs(step2 - T_RES_STEPS[1])
    if diff1 <= R2_TIGHT and diff2 <= R2_TIGHT:
        r2 = "OWNS-SHAPE"
    elif diff1 > R2_WRONG or diff2 > R2_WRONG:
        r2 = "WRONG-SHAPE"
    else:
        r2 = "PARTIAL-SHAPE"

    # R3: T_REN(1.0) vs measured T_res(1.0) = -62 +- 150.
    t_full = t[1.0]
    r3 = "CONSISTENT" if abs(t_full - T_RES_FULL_DOSE) <= R3_BAND else "BUDGET-TENSION"

    return {
        "R1_magnitude": {
            "read": r1,
            "implied_shift_full_dose_production": shift_full,
            "band": [-R1_BAND, R1_BAND],
        },
        "R2_dose_shape": {
            "read": r2,
            "step_0.25_to_0.5": step1,
            "step_0.5_to_1.0": step2,
            "target_steps": list(T_RES_STEPS),
            "diff_step1": diff1,
            "diff_step2": diff2,
            "tight_tol": R2_TIGHT,
            "wrong_tol": R2_WRONG,
        },
        "R3_budget": {
            "read": r3,
            "t_ren_full_dose": t_full,
            "target": T_RES_FULL_DOSE,
            "band": R3_BAND,
        },
        "R_sign": {
            "note": "reported only, not read (net sign not pre-stated, Sec 2)",
            "t_ren_by_dose": {f"{d:.2f}": t[d] for d in DOSES},
            "sign": "positive" if t[1.0] > 0 else ("negative" if t[1.0] < 0 else "zero"),
        },
    }


def main() -> None:
    sig_d_pool = cl.load_sigma_triples(cl.DEFAULT_CRB_CSV)[:, 0]
    edges, pools = load_glade_sigma_mix()

    results: dict[str, Any] = {"config": {}, "per_dose": {}, "reads": {}}
    results["config"] = {
        "H_TRUE": H_TRUE,
        "N_EV": N_EV,
        "K": K,
        "NQ": NQ,
        "SEEDS": list(SEEDS),
        "DOSES": list(DOSES),
        "DH": DH,
        "N_PRODUCTION": N_PRODUCTION,
        "SIGMA_POST_PRODUCTION": SIGMA_POST_PRODUCTION,
        "sigma_z_mix": "GLADE-empirical (VT-D3 z-decile sampler, venue_transfer.py)",
        "dose_convention": "scales sigma_z for ALL candidates, host included (full-dose venue)",
    }

    events_by_seed = {seed: build_events(seed, sig_d_pool, edges, pools) for seed in SEEDS}

    for dose in DOSES:
        print(f"--- dose={dose} ---", flush=True)
        rows = []
        for seed in SEEDS:
            row = measure_one(events_by_seed[seed], seed, dose)
            rows.append(row)
            print(
                f"  seed={seed} t_ren_982={row['t_ren_982']:+.4e} "
                f"implied_shift_prod={row['implied_shift_production_curvature']:+.4e} "
                f"frac_double={row['frac_double_clip']:.3f} frac_single={row['frac_single_clip']:.3f}",
                flush=True,
            )
        summary = summarize(rows)
        key = f"f{dose:.2f}"
        results["per_dose"][key] = {"dose": dose, "n_seeds": len(SEEDS), **summary}

    results["reads"] = apply_reads(results["per_dose"])

    out_path = Path(__file__).resolve().parent.parent / "L0_REN_B_toy_output.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")
    print(json.dumps(results["reads"], indent=2))


if __name__ == "__main__":
    main()
