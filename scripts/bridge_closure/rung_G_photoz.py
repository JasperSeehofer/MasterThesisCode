"""Rung G (root cause) — catalogue photo-z error sigma_z drives the railing.

The catalogue redshift error sigma_z ~= 0.035 is ~14x the GW distance precision
(sigma_dL/d_L ~ 3.7%). The real single_host_likelihood convolves each candidate's
GW contribution with norm(z_g, sigma_z), so the in-catalogue likelihood is
photo-z DOMINATED: the sharp GW distance information is washed out and the
catalogue density gradient drives the H0 trend -> railing.

This sweeps sigma_z (scaling the catalogue z-error) with the fully faithful
ingredients (real catalogue + sky + 3-D MVN + host-z convolution + real p_det +
real pixelated f_k + B_num + the pipeline's 1.5-sigma candidate radius):

  delta-z (no convolution)  -> recovers 0.73            (why the bridge recovered)
  sigma_z x 0.05 (spec-z)   -> recovers 0.73            (the fix)
  sigma_z x 1.0  (real)     -> rails                    (reproduces the pipeline)

Run: uv run python scripts/bridge_closure/rung_G_photoz.py [n_events]
Outputs: outputs/{rungG_photoz.pdf, rungG_results.json}
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


def main(n_events: int | None = None) -> None:
    events = S.load_real_events_with_sky(apply_cuts=True)
    if n_events:
        events = events[:n_events]
    print(f"events: {len(events)}", flush=True)
    cat = S.SkyCatalog(shuffle_sky=False)
    real_pdet = S.make_real_pdet()
    real_comp = S.make_real_completeness()
    print(f"catalogue {len(cat.z)}; median sigma_z = {np.median(cat.zerr):.4f}", flush=True)

    common = dict(pdet_obj=real_pdet, completeness_obj=real_comp, include_bnum=True, sigma_mult=1.5)
    results = []
    # reference: no photo-z convolution (delta z) -> why the bridge recovered
    r = S.run_sky_rung("delta-z (no photo-z)", cat, events, mode="mvn", **common)
    r["sigma_z_scale"] = 0.0
    results.append(r)
    # photo-z convolution at increasing sigma_z
    for sc in [0.05, 0.25, 0.5, 1.0]:
        r = S.run_sky_rung(f"photo-z x{sc}", cat, events, mode="conv", zerr_scale=sc, **common)
        r["sigma_z_scale"] = sc
        results.append(r)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for r in results:
        hs = np.array(r["hs"]); post = np.exp(np.array(r["logpost"]))
        color = RAIL_COLOR if abs(r["bias"]) > 0.03 else OK_COLOR
        lbl = "delta-z" if r["sigma_z_scale"] == 0 else f"σz×{r['sigma_z_scale']} (≈{r['sigma_z_scale']*0.035:.3f})"
        ax[0].plot(hs, post / post.max(), color=color, label=f"{lbl} → {r['h_refined']:.3f}")
    ax[0].axvline(B.TRUE_H, color=TRUTH_COLOR, ls="--", lw=1.2, label="truth 0.73")
    ax[0].set(xlabel=r"$h=H_0/100$", ylabel="normalised posterior",
              title="(a) catalogue photo-z error rails the inference")
    ax[0].legend(fontsize=8)
    conv = [r for r in results if r["sigma_z_scale"] > 0]
    xs = [r["sigma_z_scale"] * 0.035 for r in conv]
    bs = [r["bias"] for r in conv]
    colors = [RAIL_COLOR if abs(b) > 0.03 else OK_COLOR for b in bs]
    ax[1].axhline(0, color=TRUTH_COLOR, ls="--", lw=1)
    ax[1].axhline(results[0]["bias"], color="#888", ls=":", lw=1, label="delta-z (recovers)")
    ax[1].plot(xs, bs, "-", color="#888")
    for x, b, c in zip(xs, bs, colors):
        ax[1].plot([x], [b], "o", color=c, ms=10)
    ax[1].axvline(0.035, color=RAIL_COLOR, ls=":", lw=1.2)
    ax[1].text(0.0355, 0.0, "real σz≈0.035", color=RAIL_COLOR, fontsize=8, rotation=90, va="bottom")
    ax[1].set(xlabel=r"catalogue redshift error $\sigma_z$", ylabel=r"MAP bias $\hat h-0.73$",
              title="(b) bias vs photo-z error (the fix: small σz)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    out = B.OUTPUTS / "rungG_photoz.pdf"
    fig.savefig(out); plt.close(fig); print(f"  wrote {out}", flush=True)

    (B.OUTPUTS / "rungG_results.json").write_text(json.dumps(
        {"results": [{k: r[k] for k in ("name", "sigma_z_scale", "h_refined", "bias", "railed",
                                        "n_events")} for r in results],
         "curves": [{"name": r["name"], "sigma_z_scale": r["sigma_z_scale"],
                     "hs": r["hs"], "logpost": r["logpost"]} for r in results]}, indent=2))
    print(f"\n>>> VERDICT: {[(r['name'], round(r['h_refined'],4)) for r in results]}", flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(n)
