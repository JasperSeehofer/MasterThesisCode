"""Rung H / EXP-2 — sim<->inference consistency is the real lever.

EXP-1 (comoving-volume kernel regulariser) is a no-op. The workflow verified the
seed-600 events are injected AT the exact catalogue redshift (no real photo-z
scatter), while the inference convolves sigma_z=0.035 -> a sim<->inference
inconsistency. delta-z recovering (0.725) is the consistent case where the
inference treats the catalogue z as exact.

This tests the OTHER consistent direction: synthesise events from the REAL
catalogue WITH genuine photo-z scatter (true_z = z_g + N(0, sigma_z_g)), so the
inference's sigma_z convolution is justified. Prediction (literature: Echoes
2509.18243, photometric catalogues unbiased / variance-only): the conv inference
recovers ~0.73 on the CONSISTENT set, while the INCONSISTENT set (true_z = z_g,
no scatter) rails -- isolating the inconsistency as the cause.

Run: uv run python scripts/bridge_closure/rung_H_exp2.py [n_events]
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.disable(logging.WARNING)
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import _bridge_lib as B  # noqa: E402
import _bridge_sky as S  # noqa: E402
from _plot_style import OK_COLOR, RAIL_COLOR, TRUTH_COLOR, plt  # noqa: E402
from master_thesis_code.physical_relations import dist  # noqa: E402


def synth_events(cat: S.SkyCatalog, n: int, *, scatter_photoz: bool, seed: int = 0,
                 sigma_dL_frac: float = 0.037, sigma_sky: float = 0.01) -> list[dict]:
    """Draw n hosts from the real catalogue and build sky-aware events.

    scatter_photoz=True  -> true_z = z_g + N(0, sigma_z_g)  (CONSISTENT with the
                            inference's sigma_z convolution around the catalogue z_g).
    scatter_photoz=False -> true_z = z_g (the current INCONSISTENT injection).
    """
    rng = np.random.default_rng(seed)
    p = cat.w / cat.w.sum()
    idx = rng.choice(len(cat.z), size=n, p=p)
    events = []
    for g in idx:
        z_g = float(cat.z[g]); sz = float(max(cat.zerr[g], 1e-4))
        true_z = z_g + (rng.normal(0.0, sz) if scatter_photoz else 0.0)
        true_z = max(true_z, 1e-3)
        d_true = float(dist(true_z, h=B.TRUE_H))
        s_dL = sigma_dL_frac * d_true
        d_meas = d_true + rng.normal(0.0, s_dL)
        if d_meas <= 0:
            continue
        phi_m = float(cat.phi[g] + rng.normal(0.0, sigma_sky))
        the_m = float(cat.theta[g] + rng.normal(0.0, sigma_sky))
        events.append({
            "phi": phi_m, "theta": the_m, "d_meas": d_meas, "sigma_dL": s_dL,
            "phi2": sigma_sky**2, "the2": sigma_sky**2, "phi_the": 0.0,
            "phi_dL": 0.0, "the_dL": 0.0, "in_catalog": True,
        })
    return events


def main(n_events: int = 1000) -> None:
    cat = S.SkyCatalog(shuffle_sky=False)
    rp = S.make_real_pdet(); rc = S.make_real_completeness()
    common = dict(pdet_obj=rp, completeness_obj=rc, include_bnum=True, sigma_mult=1.5, mode="conv")
    print(f"catalogue {len(cat.z)} gals, median sigma_z={np.median(cat.zerr):.4f}", flush=True)

    results = []
    for label, scatter in [("INCONSISTENT (true_z=z_g)", False),
                           ("CONSISTENT (true_z=z_g+N(0,sigma_z))", True)]:
        ev = synth_events(cat, n_events, scatter_photoz=scatter)
        print(f"{label}: {len(ev)} events", flush=True)
        r = S.run_sky_rung(label, cat, ev, **common)
        r["scatter"] = scatter
        results.append(r)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for r in results:
        hs = np.array(r["hs"]); post = np.exp(np.array(r["logpost"]))
        color = RAIL_COLOR if abs(r["bias"]) > 0.03 else OK_COLOR
        ax.plot(hs, post / post.max(), color=color, label=f"{r['name']} → {r['h_refined']:.3f}")
    ax.axvline(B.TRUE_H, color=TRUTH_COLOR, ls="--", lw=1.2, label="truth 0.73")
    ax.set(xlabel=r"$h=H_0/100$", ylabel="normalised posterior",
           title="EXP-2: sim↔inference photo-z consistency")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = B.OUTPUTS / "rungH_exp2.pdf"
    fig.savefig(out); plt.close(fig); print(f"  wrote {out}", flush=True)

    (B.OUTPUTS / "rungH_results.json").write_text(json.dumps(
        {"results": [{k: r[k] for k in ("name", "scatter", "h_refined", "bias", "railed")}
                     for r in results]}, indent=2))
    print(f"\n>>> VERDICT: {[(r['name'][:12], round(r['h_refined'],4)) for r in results]}", flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    main(n)
