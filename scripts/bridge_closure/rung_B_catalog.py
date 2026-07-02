"""Rung B — the catalogue side: does the real GLADE redshift density rail H0?

The measurement side (rung A) does NOT rail. The remaining ingredient absent
from the synthetic baseline is the real catalogue. This rung isolates the
REDSHIFT-DENSITY SHAPE n(z): it feeds the closure a synthetic catalogue whose z
follows the real GLADE n(z) (peaked at low z, declining from incompleteness)
while keeping everything else closure-synthetic (no sky, 1-D likelihood, mock
p_det). L_cat is a normalised ratio, so the shape -- not the 2.2M count -- is
what matters.

  C0: synthetic dVc/dz catalogue (baseline)   -- expected: recovers 0.73
  C1: real GLADE n(z) catalogue                -- does the density gradient rail?

Run: uv run python scripts/bridge_closure/rung_B_catalog.py
Outputs: outputs/{rungB_nz_comparison.pdf, rungB_posteriors.pdf, rungB_results.json}
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.disable(logging.WARNING)
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import _bridge_lib as B  # noqa: E402
from _plot_style import OK_COLOR, RAIL_COLOR, TRUTH_COLOR, plt  # noqa: E402


def run() -> dict:
    rng = np.random.default_rng(0)
    # n(z) comparison data
    cat_z, _, _ = B.load_real_catalog()
    synth_z, _ = B.sample_population(rng, 200000, B.TRUE_H)
    realnz_z = rng.choice(cat_z, size=200000, replace=True)

    configs = [
        ("C0 synthetic dVc/dz", dict(catalog="synthetic")),
        ("C1 real GLADE n(z)", dict(catalog="real_nz")),
    ]
    results = []
    for label, kw in configs:
        # median over a few seeds (N=2000 each) to suppress finite-N variance
        seeds = []
        for s in range(3):
            t = time.time()
            cfg = B.BridgeConfig(name=f"{label}_s{s}", completeness="declining",
                                 n_gal=40000, n_events=2000, seed=s, **kw)
            r = B.run_bridge(cfg, verbose=False)
            seeds.append(r)
            print(f"  {label:22s} seed{s}: MAP={r['h_refined']:.4f} "
                  f"bias={r['bias']:+.4f} railed={r['railed']} ({time.time()-t:.0f}s)", flush=True)
        med = float(np.median([r["h_refined"] for r in seeds]))
        railed_any = any(r["railed"] for r in seeds)
        results.append({"name": label, "median_map": med, "median_bias": med - B.TRUE_H,
                        "railed_any": railed_any, "seeds": seeds})
        print(f"  >>> {label}: median MAP={med:.4f} bias={med-B.TRUE_H:+.4f} railed_any={railed_any}",
              flush=True)

    # --- figure 1: n(z) comparison ---
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, 0.3, 80)
    ax.hist(synth_z[synth_z < 0.3], bins=bins, density=True, histtype="step",
            color=OK_COLOR, lw=2, label="synthetic  dVc/dz")
    ax.hist(realnz_z[realnz_z < 0.3], bins=bins, density=True, histtype="step",
            color=RAIL_COLOR, lw=2, label="real GLADE  n(z)")
    ax.set(xlabel="redshift z", ylabel="normalised density",
           title="Catalogue redshift density: synthetic vs real GLADE")
    ax.legend()
    fig.tight_layout()
    out1 = B.OUTPUTS / "rungB_nz_comparison.pdf"
    fig.savefig(out1); plt.close(fig); print(f"  wrote {out1}", flush=True)

    # --- figure 2: posteriors ---
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for res in results:
        r0 = res["seeds"][0]
        hs = np.array(r0["hs"]); post = np.exp(np.array(r0["logpost"]))
        color = RAIL_COLOR if res["railed_any"] else OK_COLOR
        ax.plot(hs, post / post.max(), color=color,
                label=f"{res['name']} (median MAP {res['median_map']:.3f})")
    ax.axvline(B.TRUE_H, color=TRUTH_COLOR, ls="--", lw=1.2, label="truth 0.73")
    ax.set(xlabel=r"$h=H_0/100$", ylabel="normalised posterior",
           title="Rung B — real GLADE n(z) vs synthetic (seed 0 curve)")
    ax.legend()
    fig.tight_layout()
    out2 = B.OUTPUTS / "rungB_posteriors.pdf"
    fig.savefig(out2); plt.close(fig); print(f"  wrote {out2}", flush=True)

    summary = {"results": [{k: r[k] for k in ("name", "median_map", "median_bias", "railed_any")}
                           for r in results]}
    (B.OUTPUTS / "rungB_results.json").write_text(json.dumps(
        {"summary": summary,
         "curves": [{"name": r["name"], "hs": r["seeds"][0]["hs"],
                     "logpost": r["seeds"][0]["logpost"]} for r in results]}, indent=2))
    print(f"\n>>> VERDICT: {[(r['name'], round(r['median_map'],4), r['railed_any']) for r in results]}",
          flush=True)
    return summary


if __name__ == "__main__":
    run()
