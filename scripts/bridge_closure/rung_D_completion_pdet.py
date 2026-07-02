"""Rung D/E/F — add the two ingredients Rung C omitted, one at a time.

C-real (real sky + 3-D MVN, mock p_det, NO completion) recovers 0.73. The real
pipeline adds to EVERY event: (i) the completion term B_num with the real
pixelated f_k (<1 even for in-cat events), and (ii) the real survival p_det in
the selection D(h)/beta_Gbar/global_denom. This rung adds each:

  C-real : mock p_det, toy f, no B_num            -> recovers (reference)
  D      : mock p_det, REAL f_k, B_num ON         -> does the completion rail?
  E      : REAL p_det, f=1, no B_num              -> does the selection rail?
  F      : REAL p_det, REAL f_k, B_num ON         -> fully faithful (reproduce?)

Run: uv run python scripts/bridge_closure/rung_D_completion_pdet.py [n_events]
Outputs: outputs/{rungD_posteriors.pdf, rungD_results.json}
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
    print("building real sky catalogue ...", flush=True)
    cat = S.SkyCatalog(shuffle_sky=False)
    print(f"  catalogue: {len(cat.z)} galaxies", flush=True)
    print("building real p_det + real completeness ...", flush=True)
    real_pdet = S.make_real_pdet()
    real_comp = S.make_real_completeness()
    print("  built.", flush=True)

    results = []
    results.append(S.run_sky_rung("C-real (mock pdet, no Bnum)", cat, events, mode="mvn"))
    results.append(S.run_sky_rung("D (mock pdet, REAL f_k + Bnum)", cat, events, mode="mvn",
                                  completeness_obj=real_comp, include_bnum=True))
    results.append(S.run_sky_rung("E (REAL pdet, f=1, no Bnum)", cat, events, mode="mvn",
                                  completeness="one", pdet_obj=real_pdet))
    results.append(S.run_sky_rung("F (REAL pdet, REAL f_k + Bnum)", cat, events, mode="mvn",
                                  pdet_obj=real_pdet, completeness_obj=real_comp, include_bnum=True))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for r in results:
        hs = np.array(r["hs"]); post = np.exp(np.array(r["logpost"]))
        color = RAIL_COLOR if abs(r["bias"]) > 0.03 else OK_COLOR
        ax[0].plot(hs, post / post.max(), color=color,
                   label=f"{r['name']} (MAP {r['h_refined']:.3f})")
    ax[0].axvline(B.TRUE_H, color=TRUTH_COLOR, ls="--", lw=1.2, label="truth 0.73")
    ax[0].set(xlabel=r"$h=H_0/100$", ylabel="normalised posterior",
              title="(a) Rung D/E/F — completion + real p_det")
    ax[0].legend(fontsize=8)
    biases = [r["bias"] for r in results]
    labels = ["C-real", "D:Bnum", "E:pdet", "F:both"]
    colors = [RAIL_COLOR if abs(b) > 0.03 else OK_COLOR for b in biases]
    ax[1].axhline(0, color=TRUTH_COLOR, ls="--", lw=1)
    ax[1].bar(range(len(biases)), biases, color=colors)
    ax[1].set_xticks(range(len(labels))); ax[1].set_xticklabels(labels)
    ax[1].set(ylabel=r"MAP bias $\hat h - 0.73$", title="(b) which omitted ingredient rails?")
    fig.tight_layout()
    out = B.OUTPUTS / "rungD_posteriors.pdf"
    fig.savefig(out); plt.close(fig); print(f"  wrote {out}", flush=True)

    (B.OUTPUTS / "rungD_results.json").write_text(json.dumps(
        {"results": [{k: r[k] for k in ("name", "h_refined", "bias", "railed", "n_events",
                                        "include_bnum")} for r in results],
         "curves": [{"name": r["name"], "hs": r["hs"], "logpost": r["logpost"]} for r in results]},
        indent=2))
    print(f"\n>>> VERDICT: {[(r['name'], round(r['h_refined'],4)) for r in results]}", flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(n)
