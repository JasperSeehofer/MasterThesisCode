"""Rung G2 (G6 gate) — starvation post-mortem: the volume prior de-rails rung G.

Reconciles the a8cbab0 "photo-z information starvation" verdict with the
commission's de-rail result ON THE SAME HARNESS. Rung G showed the bare
photo-z convolution rails at sigma_z x 1.0 (MAP 0.857). The D_sm candidate
(global smeared denominator) was falsified — but the working fix is a
DIFFERENT estimator: the volume-consistent host-z prior (EXP-1
``regularise_photoz``: p_red(z|z_g) = N(z; z_g, sigma_z) (dVc/dz)/(1+z) / Z_g,
= the production ``volume_deconv`` kernel), which rung G never swept.

This runs the identical sigma_z sweep with the volume prior ON, plus the
N-scaling check that D_sm failed (posterior width must SHRINK with n_events
if the channel carries information — "flat/multimodal, std does not shrink"
was the starvation signature).

Note: this harness keeps the GLOBAL selection denominator, i.e. it is the
analogue of the 'volume_global' ablation-cube cell (real-data MAP 0.76);
the residual global-denominator tilt is quantified in G1/G3
(.planning/gate/G1_beta_g_check.md, G3_ablation_cube.json).

Run: uv run python scripts/bridge_closure/rung_G2_volume_prior.py [n_events]
Outputs: outputs/{rungG2_volume_prior.pdf, rungG2_results.json}
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

import _bridge_lib as B  # noqa: E402, N812
import _bridge_sky as S  # noqa: E402, N812
from _plot_style import OK_COLOR, RAIL_COLOR, TRUTH_COLOR, plt  # noqa: E402


def _width68(hs: np.ndarray, logpost: np.ndarray) -> float:
    """68% highest-density credible-interval width of the gridded posterior."""
    p = np.exp(logpost - logpost.max())
    p /= p.sum()
    order = np.argsort(p)[::-1]
    csum = np.cumsum(p[order])
    inside = np.zeros_like(p, dtype=bool)
    inside[order[: int(np.searchsorted(csum, 0.68)) + 1]] = True
    dh = hs[1] - hs[0]
    return float(inside.sum() * dh)


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
    # (a) bare conv at real sigma_z — the rung-G rail (reference)
    r = S.run_sky_rung("bare photo-z x1.0", cat, events, mode="conv", zerr_scale=1.0, **common)
    r["variant"] = "bare"
    r["sigma_z_scale"] = 1.0
    results.append(r)
    # (b) volume prior ON across the sigma_z sweep
    for sc in [0.25, 0.5, 1.0]:
        r = S.run_sky_rung(
            f"volume-prior photo-z x{sc}",
            cat,
            events,
            mode="conv",
            zerr_scale=sc,
            regularise_photoz=True,
            **common,
        )
        r["variant"] = "volume"
        r["sigma_z_scale"] = sc
        results.append(r)

    # (c) N-scaling at real sigma_z with the volume prior: width must shrink ~1/sqrt(N)
    nscale = []
    for frac in (0.25, 0.5, 1.0):
        n_sub = max(int(len(events) * frac), 10)
        r = S.run_sky_rung(
            f"volume-prior x1.0 N={n_sub}",
            cat,
            events[:n_sub],
            mode="conv",
            zerr_scale=1.0,
            regularise_photoz=True,
            **common,
        )
        hs = np.array(r["hs"])
        w = _width68(hs, np.array(r["logpost"]))
        nscale.append(
            {"n_events": n_sub, "width68": w, "MAP": r["h_refined"], "railed": r["railed"]}
        )
        print(f"  N-scaling: n={n_sub} width68={w:.4f} MAP={r['h_refined']:.4f}", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for r in results:
        hs = np.array(r["hs"])
        post = np.exp(np.array(r["logpost"]) - np.max(r["logpost"]))
        color = RAIL_COLOR if r["variant"] == "bare" else OK_COLOR
        ls = "--" if r["variant"] == "bare" else "-"
        ax[0].plot(
            hs, post / post.max(), ls, color=color, label=f"{r['name']} → {r['h_refined']:.3f}"
        )
    ax[0].axvline(B.TRUE_H, color=TRUTH_COLOR, ls="--", lw=1.2, label="truth 0.73")
    ax[0].set(
        xlabel=r"$h=H_0/100$",
        ylabel="normalised posterior",
        title="(a) volume prior de-rails the photo-z channel",
    )
    ax[0].legend(fontsize=8)
    ns = [d["n_events"] for d in nscale]
    ws = [d["width68"] for d in nscale]
    ax[1].loglog(ns, ws, "o-", color=OK_COLOR, label="volume prior, σz×1.0")
    ref = ws[0] * np.sqrt(ns[0] / np.asarray(ns, float))
    ax[1].loglog(ns, ref, ":", color="#888", label=r"$\propto 1/\sqrt{N}$")
    ax[1].set(
        xlabel="n_events",
        ylabel="68% HDI width",
        title="(b) information accumulates (starvation ⇒ flat width)",
    )
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    out = B.OUTPUTS / "rungG2_volume_prior.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}", flush=True)

    (B.OUTPUTS / "rungG2_results.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        k: r[k]
                        for k in (
                            "name",
                            "variant",
                            "sigma_z_scale",
                            "h_refined",
                            "bias",
                            "railed",
                            "n_events",
                        )
                    }
                    for r in results
                ],
                "n_scaling": nscale,
                "curves": [
                    {"name": r["name"], "hs": r["hs"], "logpost": r["logpost"]} for r in results
                ],
            },
            indent=2,
        )
    )
    print(
        f"\n>>> VERDICT: {[(r['name'], round(r['h_refined'], 4), r['railed']) for r in results]}",
        flush=True,
    )
    print(f">>> N-SCALING: {[(d['n_events'], round(d['width68'], 4)) for d in nscale]}", flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(n)
