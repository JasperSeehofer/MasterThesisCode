"""Factory functions for physical relations plots.

Extracted from ``physical_relations.visualize()``.
"""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from darksiren_emri.plotting._colors import EDGE, SEQUENTIAL_CMAP
from darksiren_emri.plotting._helpers import _fig_from_ax, get_figure
from darksiren_emri.plotting._labels import LABELS


def _resolve_cmap(name: str) -> Colormap:
    """Resolve a palette colormap token to a registered ``Colormap`` object.

    The Atlas tokens in ``_colors`` use bare ``cmcrameri`` names (e.g.
    ``"batlow"``), but ``cmcrameri`` registers them under a ``cmc.`` prefix.
    Try the prefixed name first, then the bare name (covers the built-in
    fallback such as ``"cividis"`` when ``cmcrameri`` is absent).
    """
    for candidate in (f"cmc.{name}", name):
        try:
            return plt.get_cmap(candidate)
        except (KeyError, ValueError):
            continue
    return plt.get_cmap(name)


# ``dist_vectorized`` (and ``dist``) return luminosity distance in **Gpc**
# (see ``physical_relations.dist`` docstring; elsewhere the code multiplies by
# ``GPC_TO_MPC`` when Mpc is wanted).  The shared ``LABELS["d_L"]`` token is in
# Mpc because the (d_L, M) detection heatmaps consume Mpc-scale data; here the
# curve is in Gpc, so we use a Gpc-labelled axis to avoid the historic
# "d_L peaks ~28 Mpc" mislabel bug (the data was always ~28 Gpc at z=3).
_D_L_GPC_LABEL = r"$d_L \, [\mathrm{Gpc}]$"


def plot_distance_redshift(
    redshifts: npt.NDArray[np.float64],
    distances: npt.NDArray[np.float64],
    *,
    h0_values: list[float] | None = None,
    distance_fn: Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]] | None = None,
    label: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    r"""Plot luminosity distance :math:`d_L(z)` vs redshift for several :math:`h`.

    The luminosity distance is in **Gpc** (the convention of
    :func:`physical_relations.dist_vectorized`).  When *h0_values* is given the
    curves form an ordered family, so they are coloured along the sequential
    Atlas colormap (low :math:`h` -> short distance) and *direct-labelled* at
    their right endpoints instead of carrying a legend.

    Parameters
    ----------
    redshifts:
        Redshift array shared by every curve.
    distances:
        Luminosity distances (Gpc) for the primary curve, evaluated at the
        fiducial ``h``.  Only drawn when *h0_values* is ``None`` (otherwise the
        per-``h`` family from *distance_fn* supersedes it).
    h0_values:
        Optional ascending list of dimensionless Hubble parameters ``h`` for the
        comparison family.  Requires *distance_fn*.
    distance_fn:
        Callable ``(redshifts, h) -> distances`` (Gpc) producing each family
        member.
    label:
        Label for the single primary curve (only used when *h0_values* is
        ``None``).
    ax:
        Optional pre-existing Axes.
    """
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    if h0_values is not None:
        if distance_fn is None:
            msg = "distance_fn must be provided when h0_values is set"
            raise ValueError(msg)

        # Order the family so colour encodes magnitude (low h -> larger d_L).
        ordered = sorted(h0_values)
        cmap = _resolve_cmap(SEQUENTIAL_CMAP)
        n = len(ordered)
        # Sample the colormap on [0.12, 0.78] to dodge near-white / near-black so
        # even the lightest curve and its inline label stay legible.
        frac = np.linspace(0.12, 0.78, n) if n > 1 else np.array([0.45])

        colors = [cmap(float(frac[i])) for i in range(n)]
        endpoints = np.empty(n, dtype=np.float64)
        for i, h0 in enumerate(ordered):
            d = np.asarray(distance_fn(redshifts, h0), dtype=np.float64)
            ax.plot(redshifts, d, color=colors[i], linewidth=1.5, zorder=3)
            endpoints[i] = float(d[-1])

        # The endpoint d_L values cluster within a few Gpc, far closer than a
        # label's height, so anchor the inline labels to an evenly-spaced ladder
        # (in endpoint order) centred on the endpoint cluster.  A leader line
        # ties each label back to its curve.
        order = np.argsort(endpoints)  # ascending d_L (high h -> low d_L)
        d_lo, d_hi = float(endpoints.min()), float(endpoints.max())
        d_mid = 0.5 * (d_lo + d_hi)
        full_range = d_hi - d_lo if d_hi > d_lo else 1.0
        ladder_half = max(0.9 * full_range, 0.0)
        ladder = np.linspace(d_mid - ladder_half, d_mid + ladder_half, n)
        label_y = np.empty(n, dtype=np.float64)
        for rank, idx in enumerate(order):
            label_y[idx] = ladder[rank]

        x_text = float(redshifts[-1])
        for i, h0 in enumerate(ordered):
            ax.annotate(
                rf"$h={h0:g}$",
                xy=(x_text, endpoints[i]),
                xytext=(x_text + 0.05 * x_text, float(label_y[i])),
                textcoords="data",
                va="center",
                ha="left",
                fontsize=7,
                color=colors[i],
                annotation_clip=False,
                arrowprops={"arrowstyle": "-", "color": colors[i], "lw": 0.6},
            )
        # Leave head-room on the right for the inline labels and the upper ladder.
        ax.set_xlim(float(redshifts[0]), x_text + 0.22 * (x_text - float(redshifts[0])))
    else:
        ax.plot(
            redshifts,
            distances,
            color=EDGE,
            linewidth=1.5,
            label=label,
            zorder=3,
        )
        if label is not None:
            ax.legend()

    ax.set_xlabel(LABELS["z"])
    ax.set_ylabel(_D_L_GPC_LABEL)
    ax.margins(y=0.05)
    return fig, ax
