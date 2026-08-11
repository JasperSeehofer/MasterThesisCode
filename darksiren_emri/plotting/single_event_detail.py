"""Detailed single-event multi-panel figure (Phase D).

For one selected EMRI detection, this module plots how the BH-mass channel
reshapes the marginalisation over candidate host galaxies. Six panels in a
2×3 grid:

  (1,1) Per-host weights without the BH-mass cut (sorted descending).
  (1,2) Per-host weights with the BH-mass cut (sorted descending).
  (1,3) Cross-comparison scatter w_no vs w_with for galaxies appearing in
        both pools.
  (2,1) L(h) per event without BH mass (from the per-h posterior JSONs).
  (2,2) L(h) per event with BH mass.
  (2,3) L(h) overlay (peak-normalised).

Data sources:
  - ``posteriors_with_bh_mass/h_*.json`` files carry per-event scalar
    likelihoods (the 2D channel result) in their integer-keyed entries.
  - The h=0.73 JSON additionally carries ``galaxy_likelihoods`` and
    ``additional_galaxies_without_bh_mass`` lists with per-host numerator
    /denominator features used to derive per-galaxy weights.
  - ``posteriors/h_*.json`` files carry per-event 1D likelihoods.

Note on data availability: the cluster Phase 48 (1473-event) 2D posteriors
were stripped of ``galaxy_likelihoods`` to keep the rsync footprint small.
This figure therefore works against the local 417-event Phase 45 R2 dataset
(``simulations/posteriors_with_bh_mass/``) by default; a follow-up will
selectively re-sync ``galaxy_likelihoods`` for 3–5 representative event IDs
from the cluster so the figure can also point at the production dataset.
"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure

from darksiren_emri.plotting._colors import (
    EDGE,
    TRUTH,
    VARIANT_NO_MASS,
    VARIANT_WITH_MASS,
)
from darksiren_emri.plotting._helpers import get_figure

_logger = logging.getLogger(__name__)

# Number of top-weighted hosts to display in the per-host bar plots.
_TOP_HOSTS = 20


def _safe_ratio(num: float, den: float) -> float:
    """Return ``num / den`` if ``den > 0``, else 0."""
    return num / den if den > 0 else 0.0


def extract_galaxy_weights(
    posteriors_with_mass_dir: Path,
    event_id: int,
    *,
    h_value: float = 0.73,
) -> pd.DataFrame:
    """Extract per-host weights for one event at one h-value.

    Reads the ``h_*.json`` for *h_value* from ``posteriors_with_bh_mass`` and
    builds a DataFrame of all candidate hosts (BH-mass-cut survivors plus
    additional 1D-only hosts). Per-host likelihoods are computed as
    ``num/den`` from the stored numerator/denominator features.

    Parameters
    ----------
    posteriors_with_mass_dir:
        Directory of ``h_*.json`` files for the 2D channel (must contain
        the file at *h_value*).
    event_id:
        Integer event index (the per-event JSON key).
    h_value:
        The h grid value at which to extract weights (default 0.73).

    Returns
    -------
    DataFrame with columns:
        ``galaxy_id`` (int), ``in_with_mass_pool`` (bool),
        ``L_no`` (float, raw L without BH mass),
        ``L_with`` (float, raw L with BH mass; 0 for galaxies that fail mass cut),
        ``w_no`` (float, normalised weight in [0, 1]),
        ``w_with`` (float, normalised weight in [0, 1]).

    Raises
    ------
    FileNotFoundError
        When the requested h-value JSON does not exist.
    KeyError
        When the requested event_id is not present in the JSON.
    """
    # Resolve by reading the stored "h" key — robust to filename precision
    # changes (3-decimal legacy vs. 4-decimal post-fix).
    target = None
    for f in sorted(posteriors_with_mass_dir.glob("h_*.json")):
        with open(f) as fh:
            preview = json.load(fh)
        if "h" in preview and abs(float(preview["h"]) - h_value) < 1e-6:
            target = f
            data = preview
            break
    if target is None:
        raise FileNotFoundError(f"No h_*.json with h={h_value} in {posteriors_with_mass_dir}")

    ev_key = str(event_id)
    if ev_key not in data.get("galaxy_likelihoods", {}) and ev_key not in data.get(
        "additional_galaxies_without_bh_mass", {}
    ):
        raise KeyError(f"Event {event_id} not present in {target.name}")

    # Hosts that pass the BH-mass cut (6 features each).
    rows: list[dict[str, Any]] = []
    for gal_id, feats in data.get("galaxy_likelihoods", {}).get(ev_key, []):
        num_no, den_no, num_w, den_w = (
            float(feats[0]),
            float(feats[1]),
            float(feats[2]),
            float(feats[3]),
        )
        rows.append(
            {
                "galaxy_id": int(gal_id),
                "in_with_mass_pool": True,
                "L_no": _safe_ratio(num_no, den_no),
                "L_with": _safe_ratio(num_w, den_w),
            }
        )
    # Additional hosts that fail the BH-mass cut (4 features each).
    for gal_id, feats in data.get("additional_galaxies_without_bh_mass", {}).get(ev_key, []):
        num_no, den_no = float(feats[0]), float(feats[1])
        rows.append(
            {
                "galaxy_id": int(gal_id),
                "in_with_mass_pool": False,
                "L_no": _safe_ratio(num_no, den_no),
                "L_with": 0.0,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    sum_no = float(df["L_no"].sum())
    sum_w = float(df["L_with"].sum())
    df["w_no"] = df["L_no"] / sum_no if sum_no > 0 else 0.0
    df["w_with"] = df["L_with"] / sum_w if sum_w > 0 else 0.0
    return df


def _load_event_likelihood_curve(
    posteriors_dir: Path, event_id: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Return ``(h_grid, L_event_at_h)`` for one event across all h-values.

    Aggregates the scalar entry under integer key ``event_id`` from each
    ``h_*.json`` file. Returns ``None`` when the directory or event is
    missing.
    """
    files = sorted(posteriors_dir.glob("h_*.json"))
    if not files:
        return None
    h_values: list[float] = []
    likelihoods: list[float] = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        if "h" not in d:
            continue
        ev_key = str(event_id)
        if ev_key not in d:
            continue
        val = d[ev_key]
        if isinstance(val, list):
            if len(val) == 0:
                continue
            scalar = float(val[0])
        else:
            scalar = float(val)
        h_values.append(float(d["h"]))
        likelihoods.append(scalar)
    if not h_values:
        return None
    order = np.argsort(h_values)
    h_arr = np.asarray(h_values, dtype=np.float64)[order]
    L_arr = np.asarray(likelihoods, dtype=np.float64)[order]
    return h_arr, L_arr


def plot_single_event_detail(
    data_dir: Path,
    event_id: int,
    *,
    h_true: float = 0.73,
    h_eval: float = 0.73,
    top_hosts: int = _TOP_HOSTS,
) -> tuple[Figure, npt.NDArray[np.object_]]:
    """Six-panel detailed single-event diagnostic.

    Parameters
    ----------
    data_dir:
        Directory containing ``posteriors/`` and ``posteriors_with_bh_mass/``
        subdirectories with the per-h JSON files. The 2D file at
        ``h_eval`` must carry ``galaxy_likelihoods`` (i.e. the data must
        be unstripped).
    event_id:
        Integer event index (the JSON key) to visualise.
    h_true:
        Truth h-value to draw on the L(h) panels.
    h_eval:
        h-value at which per-host weights are extracted (default 0.73).
    top_hosts:
        Maximum number of hosts shown in the per-host bar plots.

    Returns
    -------
    (Figure, ndarray of Axes)
    """
    weights_df = extract_galaxy_weights(
        data_dir / "posteriors_with_bh_mass", event_id, h_value=h_eval
    )
    if weights_df.empty:
        raise ValueError(f"No candidate hosts found for event {event_id} at h={h_eval}")

    fig, axes = get_figure(nrows=2, ncols=3, figsize=(11.0, 5.6))

    # --- Top row: per-host weights ---
    ax_no, ax_with, ax_xy = axes[0, 0], axes[0, 1], axes[0, 2]

    # (1,1) Top-N hosts without BH mass
    df_no = weights_df.sort_values("w_no", ascending=False).reset_index(drop=True)
    n_no = min(top_hosts, len(df_no))
    ax_no.bar(
        np.arange(n_no),
        df_no["w_no"].iloc[:n_no].to_numpy(),
        color=VARIANT_NO_MASS,
        edgecolor=EDGE,
        linewidth=0.4,
    )
    ax_no.set_title(f"Per-host weights without $M_z$ (top {n_no})", fontsize="medium")
    ax_no.set_xlabel("Host rank")
    ax_no.set_ylabel(r"$w_g$")
    n_active_no = int((weights_df["L_no"] > 0).sum())
    ax_no.text(
        0.98,
        0.95,
        f"{n_active_no} hosts with L>0\nof {len(weights_df)} candidates",
        transform=ax_no.transAxes,
        va="top",
        ha="right",
        fontsize=7,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8, "edgecolor": EDGE},
    )

    # (1,2) Top-N hosts with BH mass
    df_w = weights_df.sort_values("w_with", ascending=False).reset_index(drop=True)
    n_w = min(top_hosts, len(df_w))
    ax_with.bar(
        np.arange(n_w),
        df_w["w_with"].iloc[:n_w].to_numpy(),
        color=VARIANT_WITH_MASS,
        edgecolor=EDGE,
        linewidth=0.4,
    )
    ax_with.set_title(f"Per-host weights with $M_z$ (top {n_w})", fontsize="medium")
    ax_with.set_xlabel("Host rank")
    ax_with.set_ylabel(r"$w_g$")
    n_active_w = int((weights_df["L_with"] > 0).sum())
    ax_with.text(
        0.98,
        0.95,
        f"{n_active_w} hosts with L>0\nof {weights_df['in_with_mass_pool'].sum()} pass mass cut",
        transform=ax_with.transAxes,
        va="top",
        ha="right",
        fontsize=7,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8, "edgecolor": EDGE},
    )

    # (1,3) Scatter w_no vs w_with
    mask = (weights_df["w_no"] > 0) | (weights_df["w_with"] > 0)
    pts = weights_df[mask]
    ax_xy.scatter(
        pts["w_no"],
        pts["w_with"],
        s=14,
        c=EDGE,
        alpha=0.65,
        edgecolor="none",
    )
    upper = float(
        max(
            pts["w_no"].max() if not pts.empty else 0.0,
            pts["w_with"].max() if not pts.empty else 0.0,
            1e-4,
        )
    )
    ax_xy.plot([0, upper], [0, upper], color=TRUTH, linewidth=0.8, linestyle=":")
    ax_xy.set_xlabel(r"$w_g$ without $M_z$")
    ax_xy.set_ylabel(r"$w_g$ with $M_z$")
    ax_xy.set_title(r"Per-host weight comparison", fontsize="medium")
    ax_xy.set_xlim(-0.02 * upper, 1.05 * upper)
    ax_xy.set_ylim(-0.02 * upper, 1.05 * upper)

    # --- Bottom row: L(h) curves ---
    ax_lh_no, ax_lh_w, ax_lh_xy = axes[1, 0], axes[1, 1], axes[1, 2]
    curve_no = _load_event_likelihood_curve(data_dir / "posteriors", event_id)
    curve_w = _load_event_likelihood_curve(data_dir / "posteriors_with_bh_mass", event_id)

    def _plot_curve(ax: Any, data: Any, color: str, title: str) -> None:
        if data is None:
            ax.text(0.5, 0.5, "data not available", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize="medium")
            return
        h_arr, L_arr = data
        if L_arr.max() > 0:
            L_norm = L_arr / L_arr.max()
        else:
            L_norm = L_arr.copy()
        ax.plot(h_arr, L_norm, color=color, linewidth=1.4)
        ax.axvline(h_true, color=TRUTH, linewidth=0.8, linestyle="--")
        ax.set_xlabel(r"$h$")
        ax.set_ylabel(r"$L(h)/L_{\max}$")
        ax.set_ylim(-0.05, 1.10)
        ax.set_title(title, fontsize="medium")

    _plot_curve(ax_lh_no, curve_no, VARIANT_NO_MASS, r"$L(h)$ without $M_z$")
    _plot_curve(ax_lh_w, curve_w, VARIANT_WITH_MASS, r"$L(h)$ with $M_z$")

    # (2,3) Overlay
    if curve_no is not None and curve_w is not None:
        h_arr_n, L_n = curve_no
        h_arr_w, L_w = curve_w
        if L_n.max() > 0:
            L_n_norm = L_n / L_n.max()
        else:
            L_n_norm = L_n
        if L_w.max() > 0:
            L_w_norm = L_w / L_w.max()
        else:
            L_w_norm = L_w
        ax_lh_xy.plot(
            h_arr_n, L_n_norm, color=VARIANT_NO_MASS, linewidth=1.4, label=r"without $M_z$"
        )
        ax_lh_xy.plot(
            h_arr_w,
            L_w_norm,
            color=VARIANT_WITH_MASS,
            linewidth=1.4,
            linestyle="--",
            label=r"with $M_z$",
        )
        ax_lh_xy.axvline(h_true, color=TRUTH, linewidth=0.8, linestyle="--", label="truth")
        ax_lh_xy.set_xlabel(r"$h$")
        ax_lh_xy.set_ylabel(r"$L(h)/L_{\max}$")
        ax_lh_xy.set_ylim(-0.05, 1.10)
        ax_lh_xy.legend(fontsize="small", loc="best")
        ax_lh_xy.set_title(r"$L(h)$ overlay", fontsize="medium")
    else:
        ax_lh_xy.text(0.5, 0.5, "Need both posteriors", ha="center", va="center")
        ax_lh_xy.set_xticks([])
        ax_lh_xy.set_yticks([])
        ax_lh_xy.set_title(r"$L(h)$ overlay", fontsize="medium")

    fig.suptitle(f"Event {event_id} — single-event marginalisation detail", fontsize="large")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig, axes


def select_representative_event_id(
    data_dir: Path,
    *,
    percentile: float = 0.50,
    h_value: float = 0.73,
    require_both_channels: bool = True,
) -> int:
    """Pick a representative event_id by per-event L(h) width.

    Enumerates events from the *h_value* JSON file (so events without data
    at all h-values are still considered) and ranks them by the FWHM of
    their per-event L(h) curve. Restricts to events where both channels
    have non-zero per-host weight when ``require_both_channels=True`` so
    the bar panels are informative.

    Falls back to the first event with non-zero per-host weights, then to
    event 0 if nothing can be ranked.
    """
    with_mass_dir = data_dir / "posteriors_with_bh_mass"
    # Discover event ids from the h_eval JSON (most complete enumeration).
    target_file = None
    for f in sorted(with_mass_dir.glob("h_*.json")):
        with open(f) as fh:
            blob = json.load(fh)
        if "h" in blob and abs(float(blob["h"]) - h_value) < 1e-6:
            target_file = blob
            break
    if target_file is None:
        return 0
    candidate_ids: list[int] = []
    for key in target_file:
        try:
            candidate_ids.append(int(key))
        except (TypeError, ValueError):
            continue
    candidate_ids.sort()

    # Build per-event FWHM list. Slow path: read each posteriors/ JSON to
    # gather per-event L(h) curves. Cached implicitly by os filesystem.
    widths: list[tuple[int, float]] = []
    for ev_idx in candidate_ids:
        curve = _load_event_likelihood_curve(data_dir / "posteriors", ev_idx)
        if curve is None:
            continue
        h_arr, L_arr = curve
        if L_arr.max() <= 0 or len(h_arr) < 3:
            continue
        L_norm = L_arr / L_arr.max()
        mask = L_norm >= 0.5
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        widths.append((ev_idx, float(h_arr[idx.max()] - h_arr[idx.min()])))
    if not widths:
        return candidate_ids[0] if candidate_ids else 0

    if require_both_channels:
        valid: list[tuple[int, float]] = []
        for ev_idx, w in widths:
            try:
                df = extract_galaxy_weights(with_mass_dir, ev_idx, h_value=h_value)
            except (FileNotFoundError, KeyError):
                continue
            if df.empty:
                continue
            if df["w_no"].sum() > 0 and df["w_with"].sum() > 0:
                valid.append((ev_idx, w))
        if valid:
            widths = valid

    widths.sort(key=lambda t: t[1])
    target = int(round(percentile * (len(widths) - 1)))
    return widths[target][0]


# ---------------------------------------------------------------------------
# Module-level smoke test (executed when run as a script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import sys

    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "simulations")
    event_id = int(sys.argv[2]) if len(sys.argv) > 2 else select_representative_event_id(data_dir)
    fig, _ = plot_single_event_detail(data_dir, event_id)
    out = data_dir / "figures" / f"fig17_single_event_{event_id}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")
