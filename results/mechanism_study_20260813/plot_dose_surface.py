"""Money chart for the mechanism-isolation campaign readout (A7).

Builds, from the RAW per-seed records only (no ``aggregate`` block is read):

* panel A -- the per-seed MAP distribution of every cell/arm against the truth
  marker h_true = 0.730 (the template's rule 3: a distribution against truth,
  never a table of means);
* panel B -- the 16-cell dose surface bias(f_host, f_imp), annotated per cell.

Analysis-only. Reads committed campaign JSONs, writes one figure and one JSON
of the numbers it plotted. Touches no registered document and no production
module.

Run with the project interpreter:  ``.venv/bin/python plot_dose_surface.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from darksiren_emri.plotting import _colors
from darksiren_emri.plotting._helpers import get_figure, save_figure
from darksiren_emri.plotting._style import apply_style

RUN_DIR = Path(__file__).resolve().parent
H_TRUE = 0.730

FRACS = (0.0, 0.25, 0.5, 1.0)
CELLS = {
    f"S{h}{i}": (FRACS[h], FRACS[i]) for h in range(4) for i in range(4)
}
ARMS = {
    "MN0": "N-0 (all dosed, N=15)",
    "MEH": "E1-host (host only, N=15)",
    "MEI": "E1-imp (impostors only, N=15)",
    "MN0X": "A1 null (all dosed, N=100)",
}


def _load(cell: str) -> list[dict]:
    """Return the raw per-seed records of *cell* (the only input trusted)."""
    matches = sorted(RUN_DIR.glob(f"{cell}_h0p730_results_seeds*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one JSON for {cell}, got {matches}")
    with matches[0].open() as fh:
        return json.load(fh)["per_seed"]


def _bias(records: list[dict], channel: str = "1d") -> tuple[np.ndarray, float, float]:
    """Per-seed MAP bias, its mean and its standard error (ddof=1)."""
    maps = np.array([r[f"map_{channel}"] for r in records], dtype=float)
    bias = maps - H_TRUE
    se = float(np.std(bias, ddof=1) / np.sqrt(bias.size)) if bias.size > 1 else 0.0
    return maps, float(bias.mean()), se


def main() -> None:
    apply_style()

    data: dict[str, dict] = {}
    for name in list(CELLS) + list(ARMS):
        recs = _load(name)
        maps, bias, se = _bias(recs)
        data[name] = {"n": len(recs), "maps": maps, "bias": bias, "se": se}

    fig, axes = get_figure(
        1, 2, figsize=(11.0, 6.2), gridspec_kw={"width_ratios": (1.35, 1.0)}
    )
    ax_strip, ax_surf = axes

    # --- panel A: per-seed MAP distributions, one row per cell/arm ----------
    rows = [f"S{h}{i}" for h in range(4) for i in range(4)] + list(ARMS)
    labels = [f"{r}  ({CELLS[r][0]:g}, {CELLS[r][1]:g})" if r in CELLS else r for r in rows]
    rng = np.random.default_rng(20260814)
    for y, name in enumerate(rows):
        maps = data[name]["maps"]
        jitter = rng.uniform(-0.28, 0.28, size=maps.size)
        colour = _colors.CYCLE[4] if name in CELLS else _colors.CYCLE[5]
        ax_strip.scatter(
            maps,
            np.full_like(maps, y, dtype=float) + jitter,
            s=11,
            alpha=0.55,
            color=colour,
            edgecolors="none",
            zorder=2,
        )
        ax_strip.plot(
            [H_TRUE + data[name]["bias"]],
            [y],
            marker="|",
            markersize=13,
            color=_colors.MEAN,
            zorder=3,
        )
    ax_strip.axvline(H_TRUE, color=_colors.TRUTH, lw=1.4, zorder=1, label="$h_{true}=0.730$")
    ax_strip.set_yticks(range(len(rows)))
    ax_strip.set_yticklabels(labels, fontsize=7)
    ax_strip.invert_yaxis()
    ax_strip.set_xlabel("per-seed MAP $h$ (grid argmax, 1D channel)")
    ax_strip.set_title("A  per-seed MAPs vs truth (325 scan seeds + 145 arm seeds)", fontsize=9)
    ax_strip.legend(loc="lower right", fontsize=7, frameon=False)
    ax_strip.grid(axis="x", alpha=0.25)

    # --- panel B: the 16-cell surface --------------------------------------
    surface = np.array(
        [[data[f"S{h}{i}"]["bias"] for i in range(4)] for h in range(4)], dtype=float
    )
    im = ax_surf.imshow(surface, cmap=_colors.SEQUENTIAL_BLUES, origin="upper", aspect="auto")
    for h in range(4):
        for i in range(4):
            cell = f"S{h}{i}"
            val = surface[h, i]
            txt = f"{val:+.6f}\n±{data[cell]['se']:.6f}"
            ax_surf.text(
                i,
                h,
                txt,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if val > 0.5 * surface.max() else _colors.EDGE,
            )
    ax_surf.set_xticks(range(4), [f"{f:g}" for f in FRACS])
    ax_surf.set_yticks(range(4), [f"{f:g}" for f in FRACS])
    ax_surf.set_xlabel(r"$f_{imp}$ — impostor dose (fraction of each candidate's GLADE $\sigma_z$)")
    ax_surf.set_ylabel(r"$f_{host}$ — host dose")
    ax_surf.set_title("B  bias($f_{host}$, $f_{imp}$): host gate × impostor amplifier", fontsize=9)
    cbar = fig.colorbar(im, ax=ax_surf, fraction=0.046, pad=0.03)
    cbar.set_label("MAP bias in $h$ (1D)")

    fig.suptitle(
        "Mechanism-isolation dose surface — recomputed from raw per-seed records",
        fontsize=10,
    )
    out = RUN_DIR / "fig_dose_surface_20260814"
    save_figure(fig, str(out), formats=("png", "pdf"))

    dump = {
        name: {"n": d["n"], "bias_1d": d["bias"], "se_1d": d["se"]}
        for name, d in data.items()
    }
    with (RUN_DIR / "fig_dose_surface_20260814.json").open("w") as fh:
        json.dump(dump, fh, indent=2, sort_keys=True)
    for name in rows:
        print(f"{name:5s} N={data[name]['n']:3d} bias={data[name]['bias']:+.6f} SE={data[name]['se']:.6f}")


if __name__ == "__main__":
    main()
