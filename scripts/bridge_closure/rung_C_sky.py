"""Rung C — sky + 3-D MVN over the real catalogue (the remaining suspect).

Ablation set (selection normalisation held at the SAME mock p_det as rungs A/B,
so any railing is attributable to the in-catalogue NUMERATOR):

  C-real : real GLADE sky + 3-D MVN (full Fisher cov)   -> reproduce the rail?
  C-1d   : sky-cone candidate selection, 1-D d_L only   -> is it the MVN sky term?
  C-diag : 3-D MVN, diagonal cov (no d_L-sky corr)      -> is it the correlations?
  C-iso  : real catalogue, sky positions shuffled       -> is it sky-z clustering?

Run: uv run python scripts/bridge_closure/rung_C_sky.py [n_events]
Outputs: outputs/{rungC_posteriors.pdf, rungC_results.json}
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
    print(f"events: {len(events)} (in_cat {sum(e['in_catalog'] for e in events)})", flush=True)

    print("building real sky catalogue ...", flush=True)
    cat = S.SkyCatalog(shuffle_sky=False)
    print(f"  catalogue: {len(cat.z)} galaxies", flush=True)

    results = []
    results.append(S.run_sky_rung("C-real (3D MVN full cov)", cat, events, mode="mvn"))
    results.append(S.run_sky_rung("C-1d (sky select, 1D dL)", cat, events, mode="1d"))
    results.append(S.run_sky_rung("C-diag (3D MVN diag cov)", cat, events, mode="mvn_diag"))

    print("building sky-shuffled catalogue ...", flush=True)
    cat_iso = S.SkyCatalog(shuffle_sky=True, seed=1)
    results.append(S.run_sky_rung("C-iso (sky shuffled, 3D MVN)", cat_iso, events, mode="mvn"))

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for r in results:
        hs = np.array(r["hs"]); post = np.exp(np.array(r["logpost"]))
        color = RAIL_COLOR if r["railed"] else OK_COLOR
        ax[0].plot(hs, post / post.max(), color=color,
                   label=f"{r['name']} (MAP {r['h_refined']:.3f})")
    ax[0].axvline(B.TRUE_H, color=TRUTH_COLOR, ls="--", lw=1.2, label="truth 0.73")
    ax[0].set(xlabel=r"$h=H_0/100$", ylabel="normalised posterior",
              title="(a) Rung C — sky + 3-D MVN ablations")
    ax[0].legend(fontsize=8)
    biases = [r["bias"] for r in results]
    labels = [r["name"].split(" ")[0] for r in results]
    colors = [RAIL_COLOR if r["railed"] else OK_COLOR for r in results]
    ax[1].axhline(0, color=TRUTH_COLOR, ls="--", lw=1)
    ax[1].bar(range(len(biases)), biases, color=colors)
    ax[1].set_xticks(range(len(labels))); ax[1].set_xticklabels(labels, rotation=20, ha="right")
    ax[1].set(ylabel=r"MAP bias $\hat h - 0.73$", title="(b) which ingredient rails?")
    fig.tight_layout()
    out = B.OUTPUTS / "rungC_posteriors.pdf"
    fig.savefig(out); plt.close(fig); print(f"  wrote {out}", flush=True)

    (B.OUTPUTS / "rungC_results.json").write_text(json.dumps(
        {"results": [{k: r[k] for k in ("name", "mode", "h_refined", "bias", "railed",
                                        "n_events")} for r in results],
         "curves": [{"name": r["name"], "hs": r["hs"], "logpost": r["logpost"]} for r in results]},
        indent=2))
    print(f"\n>>> VERDICT: {[(r['name'], round(r['h_refined'],4), r['railed']) for r in results]}",
          flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(n)
