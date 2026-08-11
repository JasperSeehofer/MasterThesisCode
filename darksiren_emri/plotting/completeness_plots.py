"""Per-pixel HEALPix catalog-completeness plots (Change 5, GMV-2022).

Visualises the magnitude-threshold completeness :class:`PixelCompleteness`: the
frozen per-pixel median apparent-B-magnitude map ``m_th``, the resulting per-pixel
completeness ``f_k(z, Omega)``, and the sky-averaged ``f_bar(z)``.

All sky maps are in the BarycentricTrueEcliptic(J2000) frame -- the SAME frame the
``m_th`` map and the catalog BallTree use (see ``.planning/FRAME-AUDIT.md``); the
axes are labelled accordingly so the frame is never ambiguous.
"""

from typing import Any, Literal

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy_healpix import HEALPix
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from darksiren_emri.constants import HOST_DRAW_Z_MAX, H
from darksiren_emri.galaxy_catalogue.pixel_completeness import PixelCompleteness
from darksiren_emri.plotting._colors import REFERENCE, SEQUENTIAL_CMAP, VARIANT_NO_MASS
from darksiren_emri.plotting._helpers import _fig_from_ax, get_figure


def _resolve_cmap(name: str, *, reverse: bool = False) -> Any:
    """Resolve a colormap object robustly (cmcrameri registers as ``cmc.<name>``)."""
    import matplotlib.pyplot as plt

    for candidate in (name, f"cmc.{name}", "cividis"):
        try:
            cmap = plt.get_cmap(candidate)
        except (ValueError, KeyError):
            continue
        return cmap.reversed() if reverse else cmap
    return plt.get_cmap("cividis")


def _pixel_value_grid(
    comp: PixelCompleteness,
    values: npt.NDArray[np.float64],
    n_lon: int = 720,
    n_lat: int = 360,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Sample a per-pixel ``values[npix]`` array onto an ecliptic (lon, lat) grid.

    Returns ``(lon_deg, lat_deg, grid)`` with ``grid`` shape ``(n_lat, n_lon)``,
    ``lon_deg`` in ``[-180, 180)`` (for a Mollweide projection).
    """
    hp = HEALPix(nside=comp.nside, order=comp.order)
    lon = np.linspace(-180.0, 180.0, n_lon, endpoint=False)
    lat = np.linspace(-90.0, 90.0, n_lat)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    pix = np.asarray(
        hp.lonlat_to_healpix((lon_grid.ravel() % 360.0) * u.deg, lat_grid.ravel() * u.deg)
    )
    grid = values[pix].reshape(lat_grid.shape).astype(np.float64)
    return lon, lat, grid


def plot_completeness_sky_map(
    comp: PixelCompleteness,
    *,
    quantity: Literal["m_th", "f_k"] = "f_k",
    z: float = 0.05,
    h: float = H,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    r"""HEALPix sky map (Mollweide, ecliptic) of the per-pixel completeness inputs.

    ``quantity="m_th"`` shows the frozen per-pixel median apparent B magnitude
    (catalog depth); ``quantity="f_k"`` shows the completeness ``f_k(z, Omega)`` at
    redshift ``z``. Empty / Zone-of-Avoidance pixels (no catalog galaxies) are
    rendered in light grey -- there ``f_k = 0`` (pure completion).
    """
    if quantity == "m_th":
        values = np.where(np.isfinite(comp.m_th), comp.m_th, np.nan)
        title = "Per-pixel catalog depth: median apparent B mag $m_{\\mathrm{th},k}$"
        cbar_label = r"$m_{\mathrm{th},k}\,[\mathrm{mag}]$"
        cmap_obj = _resolve_cmap(SEQUENTIAL_CMAP, reverse=True)  # brighter (smaller mag) = deeper
        vmin = vmax = None
    else:
        fk = comp.f_map(z, h)
        values = np.where(comp.m_th > -np.inf, fk, np.nan)  # mask empty pixels
        title = rf"Per-pixel completeness $f_k(z={z:g},\,\Omega)$ (GMV 2022)"
        cbar_label = r"$f_k$"
        cmap_obj = _resolve_cmap(SEQUENTIAL_CMAP)
        vmin, vmax = 0.0, 1.0

    lon, lat, grid = _pixel_value_grid(comp, values)
    masked = np.ma.masked_invalid(grid)

    if ax is None:
        fig, ax = get_figure(preset="double", subplot_kw={"projection": "mollweide"})
    else:
        fig = _fig_from_ax(ax)

    cmap_obj = cmap_obj.copy()
    cmap_obj.set_bad("0.82")  # ZoA / empty pixels in light grey
    mesh = ax.pcolormesh(
        np.radians(lon),
        np.radians(lat),
        masked,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        rasterized=True,
    )
    ax.grid(True, alpha=0.3)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Ecliptic longitude (BarycentricTrueEcliptic J2000)")
    ax.set_ylabel("Ecliptic latitude")
    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.05, pad=0.07)
    cbar.set_label(cbar_label)
    return fig, ax


def plot_sky_averaged_completeness(
    comp: PixelCompleteness,
    *,
    h: float = H,
    z_max: float = HOST_DRAW_Z_MAX,
    n_z: int = 120,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    r"""Sky-averaged completeness ``f_bar(z) = (1/N_pix) sum_k f_k(z)`` vs redshift.

    This is the Omega-marginalised completeness the run-level selection integrals
    (``beta_Gbar``) and the injection fraction ``F`` consume (GMV 2022, Eq. 3).
    """
    z = np.linspace(1e-3, z_max, n_z)
    fbar = np.asarray(comp.f_bar(z, h), dtype=np.float64)

    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    ax.plot(z, fbar, color=VARIANT_NO_MASS, linewidth=1.6, label=r"$\bar f(z)$ (per-pixel HEALPix)")
    ax.axhline(0.5, color=REFERENCE, linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(r"Redshift $z$")
    ax.set_ylabel(r"Sky-averaged completeness $\bar f(z)$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0.0, z_max)
    ax.legend(fontsize="small", loc="best")
    return fig, ax


def empty_pixel_fraction(comp: PixelCompleteness) -> float:
    """Fraction of HEALPix pixels with no catalog galaxies (``f_k == 0``; ZoA/empty)."""
    n_empty = int(np.sum(~np.isfinite(comp.m_th)))
    return float(n_empty / comp.npix)
