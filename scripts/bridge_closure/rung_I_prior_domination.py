"""Rung I — is the railing a NORMALISATION bug or fundamental photo-z prior-domination?

User's hypothesis (correct intuition): a properly-normalised Bayesian inference with
large host-redshift error should be UNBIASED but WIDER. A monotonically-increasing
(non-peaked) posterior is the signature of an UNCANCELLED redshift prior
n(z) ∝ dVc/dz (rising with z) -> i.e. a normalisation bug, not a fundamental limit.

This is a FULLY SELF-CONSISTENT synthetic closure (correct normalisation by
construction): galaxies ~ dVc/(1+z); the catalogue reports a NOISY z_g = z_true + N(0,σ_z);
events injected at the galaxy's TRUE z; the inference convolves the SAME σ_z around the
reported z_g (single_host_likelihood form). Pure in-catalogue (f=1). No sky (clean).

We sweep σ_z. If MAP stays ~h_true (just wider) -> the user is right, the real pipeline's
railing is a normalisation INCONSISTENCY (fixable). If it rails monotonically up -> the
dVc prior dominates and the partition-norm selection does not cancel it (the bug to fix).

We ALSO test a candidate fix: normalise the selection denominator's redshift prior to be
CONSISTENT with the numerator (so the dVc/(1+z) rise cancels) — global_denom convolved
with the same kernel — to see if recovery returns.

Run: uv run python scripts/bridge_closure/rung_I_prior_domination.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

logging.disable(logging.WARNING)
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import _bridge_lib as B  # noqa: E402
from _plot_style import OK_COLOR, RAIL_COLOR, TRUTH_COLOR, plt  # noqa: E402
from master_thesis_code.emri_rate import R_eff_per_mbh  # noqa: E402
from master_thesis_code.physical_relations import (  # noqa: E402
    dist,
    dist_to_redshift,
    dist_vectorized,
)


def run_closure_photoz(h_true: float, sigma_z: float, *, n_gal: int = 30000,
                       n_events: int = 800, sigma_dL_frac: float = 0.05, seed: int = 0,
                       consistent_denom: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    hs = [float(h) for h in np.round(np.arange(0.60, 0.8701, 0.01), 4)]
    # true galaxy population ~ dVc/(1+z)
    z_true_g, M = B.sample_population(rng, n_gal, h_true)
    # catalogue reports a NOISY redshift (genuine measurement error)
    z_cat = np.clip(z_true_g + rng.normal(0.0, sigma_z, n_gal), 1e-3, None)
    w_true = np.asarray(R_eff_per_mbh(M), float) / (1.0 + z_true_g)

    # inject events at the galaxy's TRUE z (rate-weighted), with GW distance noise + p_det
    p = w_true / w_true.sum()
    events = []
    tries = 0
    while len(events) < n_events and tries < 400 * n_events:
        tries += 1
        g = int(rng.choice(n_gal, p=p))
        d_true = float(dist(z_true_g[g], h=h_true))
        sdL = sigma_dL_frac * d_true
        d_meas = d_true + sdL * rng.standard_normal()
        if d_meas <= 0:
            continue
        if rng.uniform() < float(B._p_det_of_dl(np.asarray([d_meas]))[0]):
            events.append((d_meas, sdL))

    # inference catalogue uses the REPORTED z_g (sorted for searchsorted)
    order = np.argsort(z_cat)
    zc = z_cat[order]
    wc = np.asarray(R_eff_per_mbh(M[order]), float) / (1.0 + zc)
    pdet = B.MockPdet()
    catalog = B._ClosureCatalog(zc, M[order])
    D_tab = B.precompute_completion_denominator(hs, pdet, Omega_m=B._OMEGA_M, Omega_DE=B._OMEGA_DE)
    gdenom = B.precompute_global_catalog_selection(hs, catalog, pdet, with_bh_mass=False)

    logpost = np.zeros(len(hs))
    for i, h in enumerate(hs):
        gd = gdenom[h]
        total = 0.0
        for d_meas, sdL in events:
            zlo = max(dist_to_redshift(max(d_meas - 5 * sdL, 1e-4), h=0.60) - 4 * sigma_z, 1e-5)
            zhi = dist_to_redshift(d_meas + 5 * sdL, h=0.87) + 4 * sigma_z
            i0 = int(np.searchsorted(zc, zlo)); i1 = int(np.searchsorted(zc, zhi))
            zg = zc[i0:i1]; wg = wc[i0:i1]
            if zg.size == 0:
                total += -1e30
                continue
            ngrid = int(np.clip((zhi - zlo) / (0.4 * max(sigma_z, 2e-3)), 120, 500))
            zgrid = np.linspace(zlo, zhi, ngrid)
            dzg = zgrid[1] - zgrid[0]
            gw = norm.pdf(np.asarray(dist_vectorized(zgrid, h=h), float), loc=d_meas, scale=sdL)
            nm = np.exp(-0.5 * ((zgrid[None, :] - zg[:, None]) / max(sigma_z, 1e-4)) ** 2) / (
                np.sqrt(2 * np.pi) * max(sigma_z, 1e-4)
            )
            N_g = nm @ (gw * dzg)
            if consistent_denom:
                # candidate fix: selection denominator convolved with the SAME kernel,
                # over the SAME candidates -> num/denom share the photo-z-smoothed prior.
                pdet_grid = B._p_det_of_dl(np.asarray(dist_vectorized(zgrid, h=h), float))
                D_g = nm @ (pdet_grid * dzg)
                denom = float(np.sum(wg * D_g))
            else:
                denom = gd
            L_cat = float(np.sum(wg * N_g)) / denom if denom > 0 else 0.0
            total += np.log(L_cat) if L_cat > 0 else -1e30
        logpost[i] = total
    res = B.extract_map(hs, logpost, h_true)
    res["sigma_z"] = sigma_z
    res["consistent_denom"] = consistent_denom
    res["n_events"] = len(events)
    return res


def main() -> None:
    h_true = 0.73
    print("=== self-consistent closure, STANDARD pipeline normalisation ===", flush=True)
    std = []
    for sz in [0.002, 0.035]:
        r = run_closure_photoz(h_true, sz, seed=1)
        std.append(r)
        print(f"  sigma_z={sz:.3f}: MAP={r['h_refined']:.4f} bias={r['bias']:+.4f} "
              f"railed={r['railed']} n_ev={r['n_events']}", flush=True)
    print("=== candidate fix: photo-z-CONSISTENT selection denominator ===", flush=True)
    fix = []
    for sz in [0.001, 0.035]:
        r = run_closure_photoz(h_true, sz, seed=1, consistent_denom=True)
        fix.append(r)
        print(f"  sigma_z={sz:.3f} [consistent denom]: MAP={r['h_refined']:.4f} "
              f"bias={r['bias']:+.4f} railed={r['railed']}", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for r in std:
        hs = np.array(r["hs"]); post = np.exp(np.array(r["logpost"]))
        c = RAIL_COLOR if abs(r["bias"]) > 0.02 else OK_COLOR
        ax[0].plot(hs, post / post.max(), color=c, label=f"σz={r['sigma_z']} → {r['h_refined']:.3f}")
    ax[0].axvline(h_true, color=TRUTH_COLOR, ls="--", lw=1.2, label="truth 0.73")
    ax[0].set(xlabel=r"$h=H_0/100$", ylabel="normalised posterior",
              title="(a) self-consistent closure: does large σz rail?")
    ax[0].legend(fontsize=8)
    allr = std + fix
    labels = [f"{'fix ' if r['consistent_denom'] else ''}σz={r['sigma_z']}" for r in allr]
    biases = [r["bias"] for r in allr]
    cols = [RAIL_COLOR if abs(b) > 0.02 else OK_COLOR for b in biases]
    ax[1].axhline(0, color=TRUTH_COLOR, ls="--", lw=1)
    ax[1].bar(range(len(biases)), biases, color=cols)
    ax[1].set_xticks(range(len(labels))); ax[1].set_xticklabels(labels, rotation=30, ha="right")
    ax[1].set(ylabel=r"MAP bias $\hat h-0.73$", title="(b) standard vs consistent-denominator")
    fig.tight_layout()
    out = B.OUTPUTS / "rungI_prior_domination.pdf"
    fig.savefig(out); plt.close(fig); print(f"  wrote {out}", flush=True)

    (B.OUTPUTS / "rungI_results.json").write_text(json.dumps(
        {"standard": [{k: r[k] for k in ("sigma_z", "h_refined", "bias", "railed")} for r in std],
         "consistent_denom": [{k: r[k] for k in ("sigma_z", "h_refined", "bias", "railed")} for r in fix]},
        indent=2))
    print(f"\n>>> STANDARD: {[(r['sigma_z'], round(r['h_refined'],3)) for r in std]}", flush=True)
    print(f">>> CONSISTENT-DENOM FIX: {[(r['sigma_z'], round(r['h_refined'],3)) for r in fix]}", flush=True)


if __name__ == "__main__":
    main()
