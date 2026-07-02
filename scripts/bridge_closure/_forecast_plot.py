"""Render the sigma_z / sigma_M precision-forecast heatmap.

Reads the aggregated sweep produced by ``sigma_z_sigma_M_forecast.sweep`` and
writes ``docs/figures/sigma_z_sigma_M_precision_heatmap*.png``. Three panels:
  (A) 1-D channel (without BH mass) -- function of sigma_z only (mass-blind),
      shown as a heatmap broadcast across sigma_M for direct comparison;
  (B) 2-D channel (with BH mass)    -- (sigma_z, sigma_M) heatmap;
  (C) the gain      (1-D) / (2-D)   -- how much the host-mass channel helps.

Primary metric = RMSE-to-truth  sqrt(<(h-h_true)^2>)/h_true (the honest forecast
accuracy: large for a wide OR a railed/biased posterior, small only for a
narrow+centred one). The pure posterior WIDTH is misleading here because a sharp
rail at the wrong grid edge has a small width.

Overlays: GLADE photometric (sigma_z=0.035) and spectroscopic (0.0017) operating
points; the predicted convergence frontier sigma_M ~ sigma_z (where the
mass-implied redshift precision sigma_M*(1+z) matches the photo-z); target
accuracy contours (2%, 5%).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from _plot_style import plt  # noqa: E402
from matplotlib.colors import LogNorm

_REPO = Path(__file__).resolve().parents[2]
_FIG_DIR = _REPO / "docs" / "figures"
_FIG_DIR.mkdir(parents=True, exist_ok=True)

SIGMA_Z_GLADE_PHOTO = 0.035
SIGMA_Z_GLADE_SPEC = 0.0017

_METRIC_LABEL = {
    "rmse_truth": r"$\sigma_{\rm eff}(H_0)/H_0 = \langle(h-h_{\rm true})^2\rangle^{1/2}/h_{\rm true}$",
    "width": r"$\sigma(H_0)/H_0$ (posterior width)",
}


def _log_edges(centers: list[float]) -> np.ndarray:
    """Geometric-midpoint cell edges for a (roughly) log-spaced axis."""
    c = np.asarray(centers, dtype=np.float64)
    lo = c[0] ** 2 / c[1]
    hi = c[-1] ** 2 / c[-2]
    mids = np.sqrt(c[:-1] * c[1:])
    return np.concatenate([[lo], mids, [hi]])


def plot_heatmap(agg: dict, *, metric: str = "rmse_truth", out_name: str | None = None) -> Path:
    sz = np.asarray(agg["sigma_z_grid"], dtype=np.float64)
    sM = np.asarray(agg["sigma_M_grid"], dtype=np.float64)
    h_true = float(agg["config"]["h_true"])
    n_seeds = int(agg.get("n_seeds", 0))
    n_events = int(agg["config"]["n_events"])

    oned_1d = np.asarray(agg["oned"][metric], dtype=np.float64) / h_true  # (n_sz,)
    twod = np.asarray(agg["twod"][metric], dtype=np.float64) / h_true     # (n_sz, n_sM)
    oned_grid = np.repeat(oned_1d[:, None], len(sM), axis=1)
    gain = oned_grid / np.where(twod > 0, twod, np.nan)                  # >1 => 2-D helps

    zx = _log_edges(list(sz))
    my = _log_edges(list(sM))
    finite = np.concatenate([twod[np.isfinite(twod)], oned_1d[np.isfinite(oned_1d)]])
    vmin = max(1e-3, float(np.nanpercentile(finite, 2)))
    vmax = float(np.nanpercentile(finite, 99))
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), constrained_layout=True)

    def _decorate(ax, *, ylabel: bool, frontier: bool) -> None:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"host redshift error $\sigma_z$")
        if ylabel:
            ax.set_ylabel(r"host BH-mass error $\sigma_M / M$")
        for x, lab, c in [
            (SIGMA_Z_GLADE_PHOTO, "GLADE photo-z", "#c0392b"),
            (SIGMA_Z_GLADE_SPEC, "GLADE spec-z", "#27ae60"),
        ]:
            if zx[0] <= x <= zx[-1]:
                ax.axvline(x, color=c, ls="--", lw=1.3, alpha=0.95)
                ax.text(x, my[-1], f" {lab}", color=c, rotation=90, va="top", ha="left", fontsize=8)
        if frontier:
            # mass-channel-engages frontier: sigma_M*(1+z) ~ sigma_z, i.e.
            # sigma_M ~ sigma_z/(1+z) with the detected-event typical z ~ 0.15.
            one_pz = 1.146  # median(1+z) of detected events
            zs = np.array([zx[0], zx[-1]])
            ax.plot(zs, zs / one_pz, color="white", ls=":", lw=1.6, alpha=0.9)
            xe = zx[-1]
            ax.text(xe, xe / one_pz, r" $\sigma_M\!=\!\sigma_z/(1{+}z)$", color="white",
                    fontsize=8, va="center", ha="left")
        ax.set_xlim(zx[0], zx[-1])
        ax.set_ylim(my[0], my[-1])

    # --- Panel A: 1-D channel -------------------------------------------------
    axA = axes[0]
    pcA = axA.pcolormesh(zx, my, oned_grid.T, norm=norm, cmap="viridis_r", shading="flat")
    _decorate(axA, ylabel=True, frontier=False)
    axA.set_title("(A) without BH mass (1-D)")
    fig.colorbar(pcA, ax=axA, fraction=0.046, pad=0.02)

    # --- Panel B: 2-D channel + frontier + target contours -------------------
    axB = axes[1]
    pcB = axB.pcolormesh(zx, my, twod.T, norm=norm, cmap="viridis_r", shading="flat")
    _decorate(axB, ylabel=False, frontier=True)
    axB.set_title("(B) with BH mass (2-D)")
    ZZ, MM = np.meshgrid(sz, sM, indexing="ij")
    try:
        cs = axB.contour(ZZ, MM, twod, levels=[0.02, 0.05], colors="white", linewidths=1.4)
        axB.clabel(cs, fmt=lambda v: f"{v:.0%}", fontsize=8)
    except Exception:
        pass
    fig.colorbar(pcB, ax=axB, fraction=0.046, pad=0.02, label=_METRIC_LABEL[metric])

    # --- Panel C: gain = (1-D) / (2-D) ---------------------------------------
    axC = axes[2]
    g_finite = gain[np.isfinite(gain)]
    gmax = float(np.nanpercentile(g_finite, 98)) if g_finite.size else 5.0
    pcC = axC.pcolormesh(zx, my, gain.T, cmap="magma", shading="flat", vmin=1.0, vmax=max(1.5, gmax))
    _decorate(axC, ylabel=False, frontier=True)
    axC.set_title("(C) accuracy gain from BH mass  (1-D)/(2-D)")
    fig.colorbar(pcC, ax=axC, fraction=0.046, pad=0.02)

    fig.suptitle(
        rf"LISA EMRI dark-siren $H_0$ feasibility — self-consistent closure, "
        rf"$N_{{\rm events}}={n_events}$, {n_seeds} seeds "
        rf"($\sigma\!\propto\!N^{{-1/2}}$).  Colour: {_METRIC_LABEL[metric]}",
        fontsize=11,
    )
    out = _FIG_DIR / (out_name or f"sigma_z_sigma_M_precision_heatmap_{metric}.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}", flush=True)
    return out


if __name__ == "__main__":
    import json
    import sys

    _here = Path(__file__).resolve().parent
    sys.path.insert(0, str(_here))
    data = json.loads((_here / "outputs" / "sigma_z_sigma_M_forecast.json").read_text())
    plot_heatmap(data, metric="rmse_truth", out_name="sigma_z_sigma_M_precision_heatmap.png")
    plot_heatmap(data, metric="width")
