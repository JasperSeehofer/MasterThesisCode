"""F4 (money figure) — render the spec-z vs photo-z host decomposition.

Renders ``docs/figures/f4_specz_decomposition.{pdf,png}`` from the Phase-2
decomposition outputs (``f4_specz_decomposition.py`` and its physically-correct
sigma_z-aware companion ``f4_specz_decomposition_conv.py``).

WHAT THE FIGURE SHOWS (the honest result — the hypothesis was tested and refuted).
The original F4 premise was that the informative shape of the stacked H0 posterior is
carried ENTIRELY by events whose localisation cone contains a spectroscopic host,
while photo-z-only events rail. Run on the real seed-600 detections + real GLADE, the
premise does NOT hold: spectroscopic hosts (0.56% of GLADE) never carry the majority
of the rate-weighted in-catalogue likelihood, so no spec-z subset carries the shape.
The figure is therefore the *inverse* proof of photo-z information starvation:

  (A) single-event H0 posteriors (sigma_z-aware ``conv`` channel), coloured by whether
      a spectroscopic host is present in the LISA localisation cone. Every posterior
      rails toward a grid edge REGARDLESS of spec-z presence -> spec-z presence does
      not produce a peak at truth.
  (B) the stacked posterior split three ways (all / spec-z-present cone /
      photo-z-only cone). All three coincide in railing to the upper grid edge -> the
      spec-z-present subset does NOT recover the informative shape.
  (C) the smoking gun: per-event spectroscopic likelihood-weight fraction. Even where a
      spec-z host is present, it carries <= 8.7% (median ~0%) of the sigma_z-broadened,
      rate-weighted candidate contribution -- never the >= 50% needed to dominate.

This is an ADDITIVE, read-only analysis: it only reads the two decomposition JSONs and
re-sums existing per-event log-posteriors. Nothing in the H0 computation is touched.

Run:  uv run python scripts/bridge_closure/f4_plot.py
Out:  docs/figures/f4_specz_decomposition.pdf  (committable)
      docs/figures/f4_specz_decomposition.png  (gitignored)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from _plot_style import TRUTH_COLOR, plt

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_OUT = _HERE / "outputs"
_FIG_DIR = _REPO / "docs" / "figures"
_FIG_DIR.mkdir(parents=True, exist_ok=True)

# colour roles ---------------------------------------------------------------
SPECZ_COLOR = "#2c7fb8"   # informative: a spectroscopic host is present
PHOTOZ_COLOR = "#9aa0a6"  # muted: photometric hosts only
ALL_COLOR = "#c0392b"     # the full stacked posterior
DOM_THRESH = 0.5          # spec-z "domination" threshold used by the conv classifier


def _load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((_OUT / name).read_text())
    return data


def _density(hs: npt.NDArray[np.float64], logpost: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalise a (log-)posterior on the H0 grid to unit area (trapezoid)."""
    lp = np.asarray(logpost, dtype=np.float64)
    lp = lp - np.max(lp[np.isfinite(lp)])
    p = np.where(np.isfinite(lp), np.exp(lp), 0.0)
    area = float(np.trapezoid(p, hs))
    return p / area if area > 0 else p


def _stack_density(
    hs: npt.NDArray[np.float64], logposts: npt.NDArray[np.float64], mask: npt.NDArray[np.bool_]
) -> tuple[npt.NDArray[np.float64], float]:
    """Sum per-event log-posteriors over ``mask`` and return (unit-area density, MAP h)."""
    total = logposts[mask].sum(axis=0)
    total = total - np.max(total[np.isfinite(total)])
    return _density(hs, total), float(hs[int(np.argmax(total))])


def make_figure() -> tuple[Path, Path]:
    conv = _load("f4_specz_decomposition_conv.json")
    lit = _load("f4_specz_decomposition.json")

    hs = np.asarray(conv["meta"]["h_grid"], dtype=np.float64)
    h_true = float(conv["meta"]["h_true"])
    ev = conv["events"]
    n_ev = len(ev)

    present = np.array([bool(e["specz_present"]) for e in ev])
    frac = np.array([float(e["specz_weight_frac"]) for e in ev])
    lp_conv = np.array([e["logpost_conv"] for e in ev], dtype=np.float64)

    n_present = int(present.sum())
    n_absent = int((~present).sum())
    n_dominated = int(conv["summary"]["n_specz_dominated"])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)
    axA, axB, axC = axes

    # --- Panel A: single-event posteriors, coloured by spec-z presence -------
    for i in range(n_ev):
        p = _density(hs, lp_conv[i])
        c = SPECZ_COLOR if present[i] else PHOTOZ_COLOR
        axA.plot(hs, p, color=c, alpha=0.5, lw=1.1)
    # legend proxies
    axA.plot([], [], color=SPECZ_COLOR, lw=2, label=f"spec-z host in cone  (n={n_present})")
    axA.plot([], [], color=PHOTOZ_COLOR, lw=2, label=f"photo-z hosts only  (n={n_absent})")
    axA.axvline(h_true, color=TRUTH_COLOR, ls="--", lw=1.4, label=rf"truth $h={h_true}$")
    axA.set_xlabel(r"$H_0/100\;\;(h)$")
    axA.set_ylabel("single-event posterior density")
    axA.set_title("(A) single-event posteriors\ncoloured by spec-z presence")
    axA.set_xlim(hs[0], hs[-1])
    axA.set_ylim(bottom=0.0)
    axA.legend(loc="upper left")
    axA.annotate(
        "every posterior rails to a grid edge — spec-z presence does not peak it",
        xy=(0.5, 0.02), xycoords="axes fraction", ha="center", va="bottom",
        fontsize=8.5, color="#555555",
    )

    # --- Panel B: three stacked posteriors -----------------------------------
    d_all, map_all = _stack_density(hs, lp_conv, np.ones(n_ev, dtype=bool))
    d_pre, map_pre = _stack_density(hs, lp_conv, present)
    d_abs, map_abs = _stack_density(hs, lp_conv, ~present)
    axB.plot(hs, d_all, color=ALL_COLOR, lw=2.4, label=f"all events (n={n_ev}), MAP={map_all:.2f}")
    axB.plot(hs, d_pre, color=SPECZ_COLOR, lw=2.0, ls="-",
             label=f"spec-z-present cone (n={n_present}), MAP={map_pre:.2f}")
    axB.plot(hs, d_abs, color=PHOTOZ_COLOR, lw=2.0, ls="--",
             label=f"photo-z-only cone (n={n_absent}), MAP={map_abs:.2f}")
    axB.axvline(h_true, color=TRUTH_COLOR, ls="--", lw=1.4, label=rf"truth $h={h_true}$")
    axB.set_xlabel(r"$H_0/100\;\;(h)$")
    axB.set_ylabel("stacked posterior density")
    axB.set_title("(B) stacked posterior,\nspec-z / photo-z decomposition")
    axB.set_xlim(hs[0], hs[-1])
    axB.legend(loc="upper left")
    axB.annotate(
        "all three stacks rail together to the upper edge",
        xy=(0.5, 0.02), xycoords="axes fraction", ha="center", va="bottom",
        fontsize=8.5, color="#555555",
    )

    # --- Panel C: per-event spec-z likelihood-weight fraction ----------------
    order = np.argsort(frac)[::-1]
    xs = np.arange(n_ev)
    bar_colors = [SPECZ_COLOR if present[i] else PHOTOZ_COLOR for i in order]
    axC.bar(xs, frac[order] * 100.0, color=bar_colors, width=0.9)
    axC.axhline(DOM_THRESH * 100.0, color=ALL_COLOR, ls="--", lw=1.5,
                label=f"domination threshold ({DOM_THRESH:.0%})")
    axC.set_ylim(0, DOM_THRESH * 100.0 * 1.08)
    axC.set_xlabel("event (sorted by spec-z weight)")
    axC.set_ylabel("spec-z likelihood-weight fraction  [%]")
    axC.set_title("(C) spec-z weight per event\n(never reaches domination)")
    axC.legend(loc="upper right")
    axC.grid(axis="x", visible=False)
    axC.annotate(
        f"max = {frac.max() * 100:.1f}%   (median ~0%)\n"
        f"{n_dominated}/{n_ev} events spec-z-dominated",
        xy=(0.5, 0.55), xycoords="axes fraction", ha="center", va="center",
        fontsize=9.5, color="#333333",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.9),
    )

    # --- figure-level caption -------------------------------------------------
    map_conv = float(conv["stacked"]["conv"]["all"]["h_refined"])
    map_lit = float(lit["stacked"]["all"]["h_refined"])
    n_lit = int(lit["summary"]["n_events"])
    n_cat = int(conv["meta"]["n_catalog_galaxies"])
    n_specz = int(conv["meta"]["n_catalog_specz"])
    fig.suptitle(
        r"F4 — spec-z vs photo-z host decomposition of the LISA EMRI dark-siren "
        rf"$H_0$ posterior (seed 600, GLADE+).  "
        rf"Spectroscopic hosts: {n_specz:,}/{n_cat:,} = {n_specz / n_cat:.2%} of catalogue.  "
        rf"Stacked MAP rails to $h={map_conv:.2f}$ (conv) / ${map_lit:.2f}$ "
        rf"(1-D, N={n_lit}) vs truth $h={h_true}$.",
        fontsize=10,
    )

    pdf = _FIG_DIR / "f4_specz_decomposition.pdf"
    png = _FIG_DIR / "f4_specz_decomposition.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"  wrote {pdf}")
    print(f"  wrote {png}")
    return pdf, png


if __name__ == "__main__":
    make_figure()
