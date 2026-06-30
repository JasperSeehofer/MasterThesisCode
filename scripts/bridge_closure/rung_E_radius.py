"""Rung E (decisive) — the candidate search radius reproduces the railing.

The real p_D selects in-catalogue candidates in a 1.5-sigma sky ball around the
MEASURED (MVN-scattered) sky (bayesian_statistics.py:1185). 15.6% of seed-600
events have their TRUE host outside that ball -> the host is dropped from the
candidate set -> the event behaves like C-iso (random galaxies) -> bias up.

This rung sweeps the bridge's sky search radius. Prediction: tight (1.5-sigma,
the pipeline) rails; generous (>=3-sigma) recovers 0.73.

Run: uv run python scripts/bridge_closure/rung_E_radius.py [n_events]
Outputs: outputs/{rungE_radius.pdf, rungE_results.json}
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
    print(f"catalogue: {len(cat.z)} galaxies", flush=True)

    results = []
    for sm in [1.5, 2.0, 3.0, 4.0, 6.0]:
        r = S.run_sky_rung(f"radius={sm}sigma", cat, events, mode="mvn", sigma_mult=sm)
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
              title="(a) sky candidate radius sweep")
    ax[0].legend(fontsize=9)
    sms = [r["sigma_mult"] for r in results]
    biases = [r["bias"] for r in results]
    colors = [RAIL_COLOR if abs(b) > 0.03 else OK_COLOR for b in biases]
    ax[1].axhline(0, color=TRUTH_COLOR, ls="--", lw=1)
    ax[1].axvline(1.5, color=RAIL_COLOR, ls=":", lw=1.2)
    ax[1].text(1.55, max(biases) * 0.8, "pipeline\n(1.5σ)", color=RAIL_COLOR, fontsize=8)
    ax[1].plot(sms, biases, "o-", color=OK_COLOR)
    for sm, b, c in zip(sms, biases, colors):
        ax[1].plot([sm], [b], "o", color=c, ms=9)
    ax[1].set(xlabel="sky search radius (×σ)", ylabel=r"MAP bias $\hat h-0.73$",
              title="(b) tighter radius drops the host -> rails")
    fig.tight_layout()
    out = B.OUTPUTS / "rungE_radius.pdf"
    fig.savefig(out); plt.close(fig); print(f"  wrote {out}", flush=True)

    (B.OUTPUTS / "rungE_results.json").write_text(json.dumps(
        {"results": [{k: r[k] for k in ("sigma_mult", "h_refined", "bias", "railed", "n_events")}
                     for r in results],
         "curves": [{"sigma_mult": r["sigma_mult"], "hs": r["hs"], "logpost": r["logpost"]}
                    for r in results]}, indent=2))
    print(f"\n>>> VERDICT: {[(r['sigma_mult'], round(r['h_refined'],4)) for r in results]}", flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(n)
