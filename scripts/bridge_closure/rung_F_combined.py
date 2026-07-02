"""Rung F (combined) — tight radius + B_num completion reproduces the railing.

Mechanism hypothesis: the 1.5-sigma candidate ball drops the true host for ~15.6%
of events (host scattered outside the search radius). With the B_num completion
term ON and the REAL pixelated f_k (f_bar~0.71 even nearby -> large (1-f)), a
dropped-host in-catalogue event is RE-SCORED as a dark event by B_num, whose
intrinsic h-trend biases up. Neither ingredient alone rails (Rung F@4sigma
recovered; the no-B_num radius sweep recovered). This sweeps the radius WITH the
full real completion ON -> the combination should rail at 1.5-sigma and recover
at large radius, reproducing the real pipeline.

Run: uv run python scripts/bridge_closure/rung_F_combined.py [n_events]
Outputs: outputs/{rungF_combined.pdf, rungF_results.json}
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
    print("built real pdet + completeness", flush=True)

    results = []
    for sm in [1.5, 2.0, 3.0, 4.5]:
        r = S.run_sky_rung(f"radius={sm}σ +Bnum", cat, events, mode="mvn",
                           pdet_obj=real_pdet, completeness_obj=real_comp,
                           include_bnum=True, sigma_mult=sm)
        r["sigma_mult"] = sm
        results.append(r)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for r in results:
        hs = np.array(r["hs"]); post = np.exp(np.array(r["logpost"]))
        color = RAIL_COLOR if abs(r["bias"]) > 0.03 else OK_COLOR
        ax[0].plot(hs, post / post.max(), color=color,
                   label=f"{r['sigma_mult']}σ (MAP {r['h_refined']:.3f})")
    ax[0].axvline(B.TRUE_H, color=TRUTH_COLOR, ls="--", lw=1.2, label="truth 0.73")
    ax[0].set(xlabel=r"$h=H_0/100$", ylabel="normalised posterior",
              title="(a) tight radius + real completion = railing")
    ax[0].legend(fontsize=9)
    sms = [r["sigma_mult"] for r in results]; biases = [r["bias"] for r in results]
    colors = [RAIL_COLOR if abs(b) > 0.03 else OK_COLOR for b in biases]
    ax[1].axhline(0, color=TRUTH_COLOR, ls="--", lw=1)
    ax[1].plot(sms, biases, "-", color="#888")
    for sm, b, c in zip(sms, biases, colors):
        ax[1].plot([sm], [b], "o", color=c, ms=10)
    ax[1].axvline(1.5, color=RAIL_COLOR, ls=":", lw=1.2)
    ax[1].text(1.55, max(biases) * 0.6 + 0.01, "pipeline (1.5σ)", color=RAIL_COLOR, fontsize=8)
    ax[1].set(xlabel="sky search radius (×σ)", ylabel=r"MAP bias $\hat h-0.73$",
              title="(b) bias vs radius (B_num + real f_k ON)")
    fig.tight_layout()
    out = B.OUTPUTS / "rungF_combined.pdf"
    fig.savefig(out); plt.close(fig); print(f"  wrote {out}", flush=True)

    (B.OUTPUTS / "rungF_results.json").write_text(json.dumps(
        {"results": [{k: r[k] for k in ("sigma_mult", "h_refined", "bias", "railed", "n_events")}
                     for r in results],
         "curves": [{"sigma_mult": r["sigma_mult"], "hs": r["hs"], "logpost": r["logpost"]}
                    for r in results]}, indent=2))
    print(f"\n>>> VERDICT: {[(r['sigma_mult'], round(r['h_refined'],4), abs(r['bias'])>0.03) for r in results]}",
          flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(n)
