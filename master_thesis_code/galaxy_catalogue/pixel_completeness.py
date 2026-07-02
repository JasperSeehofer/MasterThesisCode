r"""Per-HEALPix-pixel GLADE+ catalog completeness ``f_k(z, Omega, h)`` (Change 5a).

The magnitude-threshold (Schechter-luminosity-function) completeness estimator of
Gray, Messenger & Veitch (2022), arXiv:2111.04629 (GMV), Eqs. (2)(3)(5), built on
Gray et al. (2020), arXiv:1908.06050, Appendix A.2.  Each equal-area HEALPix pixel
``k`` carries a single data-estimated scalar -- the median apparent B magnitude of
its catalog galaxies ``m_th,k`` -- and the completeness is the closed-form Schechter
luminosity ratio above that flux threshold:

.. math::

    f_k(z, h) = \frac{\Gamma(\alpha+2,\, x_{\mathrm{th},k}(z,h))}
                     {\Gamma(\alpha+2,\, x_{\dim})},
    \qquad
    x_{\mathrm{th},k}(z,h) = 10^{\,0.4\,(M_* - M_{\mathrm{th},k}(z,h))},
    \quad x_{\dim} = 10^{-3},

with the pixel absolute-magnitude horizon (Gray-2020 App. A.2)

.. math::

    M_{\mathrm{th},k}(z,h) = m_{\mathrm{th},k} - 25 - 5\log_{10}\!\bigl(d_L(z,h)/\mathrm{Mpc}\bigr) - K(z),
    \qquad K(z) = 0,

and the B-band Schechter parameters ``alpha = -1.07``, ``M_* = -19.7 + 5 log10 h``,
faint cutoff ``M_dim = -12.2 + 5 log10 h`` (so ``x_dim = 10^{0.4(M_*-M_dim)} = 1e-3``,
H0-independent).  ``scipy.special.gammaincc(s, x)`` is the *regularized* upper
incomplete gamma ``Gamma(s,x)/Gamma(s)``, so the ``Gamma(s)`` cancels in the ratio.
Empty / Zone-of-Avoidance pixels (fewer than ``EMPTY_PIXEL_MIN_GALAXIES`` non-null-B
galaxies) self-set ``m_th,k -> -inf`` so ``f_k == 0`` there (pure completion), with NO
separate hard mask.

The sky-average ``f_bar(z,h) = (1/N_pix) sum_k f_k(z,h)`` (GMV Eq. 3, equal-area
pixels) is the Omega-marginalized completeness used by the run-level selection
integrals ``beta_Gbar`` and the injection fraction ``F``; the per-pixel ``f_k`` is
used by the per-event completion numerator ``B_num`` (at the event pixel) and by the
joint dark-host sampler ``W_k`` (over all pixels).

HARD CONSISTENCY REQUIREMENT (C1): ONE frozen cached ``m_th[npix]`` ``.npy`` map is the
SOLE source of ``f`` and is loaded byte-identically by BOTH the EMRI injection and the
H0 inference.  Any divergence in realization / nside / null-policy gives
``f^inj != f^inf`` and a DIRECT H0 bias.  See
``.planning/derivation-change5-healpix-estimator/{DERIVATION,PHYSICS-CHANGE-PROTOCOL}.md``.

Frame convention: all ``(phi, theta)`` are BarycentricTrueEcliptic(J2000) -- ``phi`` is
the ecliptic azimuth (longitude, rad) and ``theta`` the polar angle (colatitude,
rad in ``[0, pi]``), the SAME frame the catalog handler maps galaxies and detections
into (handler._rotate_equatorial_to_ecliptic).  ``astropy_healpix`` parametrizes pixels
by ``(lon, lat)`` with ``lat = pi/2 - theta``.

References
----------
Gray, Messenger & Veitch (2022), arXiv:2111.04629, Eqs. (2)(3)(5), Sec. V.
Gray et al. (2020), arXiv:1908.06050, Eqs. (12)(13)(32)(33), Appendix A.2.
Schechter (1976), ApJ 203, 297 (luminosity function).
Dalya et al. (2022), arXiv:2110.06184 (GLADE+ completeness).
"""

import logging
import os
from typing import Any, Protocol

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord
from astropy_healpix import HEALPix
from scipy.special import gammaincc

from master_thesis_code.constants import GPC_TO_MPC, OMEGA_DE, OMEGA_M, H
from master_thesis_code.galaxy_catalogue.handler import (
    REDUCED_CATALOGUE_FILE_PATH,
    CatalogueColumns,
    _reduced_catalog_column_names,
)
from master_thesis_code.physical_relations import dist_vectorized

_LOGGER = logging.getLogger()

# ── Change 5.0: Schechter B-band luminosity-function constants (GMV-2022 Sec. 4) ──
# alpha + 2 = 0.93 > 0 keeps scipy.special.gammaincc valid (the number-weighted
# s = alpha + 1 = -0.07 form is mathematically finite but unusable in scipy).
SCHECHTER_ALPHA: float = -1.07  # B-band faint-end slope
SCHECHTER_M_STAR_0: float = -19.7  # M_* before the +5 log10 h cosmology term
SCHECHTER_M_DIM_0: float = -12.2  # faint absolute-magnitude cutoff (before +5 log10 h)
LF_S: float = SCHECHTER_ALPHA + 2.0  # = 0.93, the incomplete-gamma order
# x_dim = 10^{0.4 (M_* - M_dim)} = 10^{0.4 (-7.5)} = 1e-3 (the +5 log10 h cancels).
X_DIM: float = 10.0 ** (0.4 * (SCHECHTER_M_STAR_0 - SCHECHTER_M_DIM_0))
K_CORR: float = 0.0  # K-correction (GMV-faithful: none)

# ── Change 5.0 / 5.7: HEALPix build parameters ──
NSIDE: int = 32  # 12_288 equal-area pixels, 3.36 deg^2 (GMV GLADE balance)
HEALPIX_ORDER: str = "ring"  # pixel ordering (fixed; both sides must agree)
EMPTY_PIXEL_MIN_GALAXIES: int = 10  # < this many non-null-B galaxies => f_k == 0 (ZoA)

# Frozen cached per-pixel median apparent-B-magnitude map (the SOLE source of f; C1).
# Resolved relative to THIS package file (committed package data), so it is found
# byte-identically regardless of the process working directory (eval temp dirs, the
# cluster job CWD, tests) -- the same frozen map on both the injection and inference
# sides is the C1 requirement.
M_TH_CACHE_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), f"m_th_map_nside{NSIDE}.npy"
)

# Memory bound for the f_bar / W_k vectorized reductions (chunk sizes).
_FBAR_Z_CHUNK: int = 256  # redshift rows per f_bar chunk
_WK_PIX_CHUNK: int = 1024  # pixels per W_k chunk

# Regularized-gamma denominator Gamma(s, x_dim)/Gamma(s); h-independent constant.
_F_DENOM: float = float(gammaincc(LF_S, X_DIM))


class CompletenessModel(Protocol):
    """Structural type shared by both completeness estimators.

    Satisfied by :class:`PixelCompleteness` (per-pixel Schechter) and by
    :class:`~master_thesis_code.galaxy_catalogue.glade_completeness.GladeCatalogCompleteness`
    (Omega-independent Dalya curve, via shim methods).  Both are called
    byte-identically by injection and inference.
    """

    def f_bar(
        self, z: float | npt.NDArray[np.floating[Any]], h: float = ...
    ) -> float | npt.NDArray[np.floating[Any]]:
        """Sky-averaged completeness ``f_bar(z, h)``."""
        ...

    def f_k(
        self, z: float | npt.NDArray[np.floating[Any]], k: int, h: float = ...
    ) -> float | npt.NDArray[np.floating[Any]]:
        """Per-pixel completeness ``f_k(z, h)`` at pixel ``k``."""
        ...

    def ang2pix(self, phi: float, theta: float) -> int:
        """Ecliptic ``(phi, theta_colatitude)`` -> HEALPix pixel index."""
        ...

    def get_completeness_at_redshift(
        self, z: float | npt.NDArray[np.floating[Any]], h: float = ...
    ) -> float | npt.NDArray[np.floating[Any]]:
        """Backward-compatible sky-averaged completeness (alias of ``f_bar``)."""
        ...


class PixelCompleteness:
    """Per-HEALPix-pixel magnitude-threshold completeness (GMV-2022).

    Parameters
    ----------
    m_th:
        Per-pixel median apparent B magnitude, shape ``(npix,)``.  Empty / ZoA
        pixels carry ``-inf`` (``f_k == 0`` there).  ``npix`` fixes ``nside``.
    nside:
        HEALPix resolution; must satisfy ``npix == 12 nside^2``.
    order:
        HEALPix pixel ordering (``"ring"`` or ``"nested"``); must match the map
        that produced ``m_th``.

    Notes
    -----
    ``m_th`` is h-independent (Sec. 1.5 of DERIVATION.md: the ``+5 log10 h`` in
    ``M_*`` cancels the ``-5 log10 h`` in the distance modulus), so the catalog
    pass runs ONCE and only the distance modulus carries ``(z, h)``.
    """

    def __init__(
        self,
        m_th: npt.NDArray[np.floating[Any]],
        nside: int = NSIDE,
        order: str = HEALPIX_ORDER,
    ) -> None:
        self.m_th: npt.NDArray[np.float64] = np.asarray(m_th, dtype=np.float64)
        self.nside = nside
        self.order = order
        self._healpix = HEALPix(nside=nside, order=order)
        if self.m_th.shape != (self._healpix.npix,):
            raise ValueError(
                f"m_th map has shape {self.m_th.shape}, expected ({self._healpix.npix},) "
                f"for nside={nside}."
            )
        # Non-empty pixels (finite median); empty pixels contribute f_k == 0.
        self._valid: npt.NDArray[np.bool_] = np.isfinite(self.m_th)
        self._m_th_valid: npt.NDArray[np.float64] = self.m_th[self._valid]

    @property
    def npix(self) -> int:
        return int(self._healpix.npix)

    # ------------------------------------------------------------------
    # Core estimator
    # ------------------------------------------------------------------

    @staticmethod
    def _distance_modulus(z: npt.NDArray[np.float64], h: float) -> npt.NDArray[np.float64]:
        r"""``25 + 5 log10(d_L(z,h)/Mpc)`` (with K-correction); shape ``(Z,)``.

        ``dist_vectorized`` returns Gpc; ``GPC_TO_MPC`` converts to Mpc.
        """
        d_L_mpc = (
            np.asarray(
                dist_vectorized(z, h=h, Omega_m=OMEGA_M, Omega_de=OMEGA_DE),
                dtype=np.float64,
            )
            * GPC_TO_MPC
        )
        return 25.0 + 5.0 * np.log10(d_L_mpc) + K_CORR

    def _f_from_mth(
        self,
        m_th_values: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        h: float,
    ) -> npt.NDArray[np.float64]:
        r"""Schechter completeness for finite ``m_th_values``; shape ``(Z, P)``.

        ``M_th = m_th - distance_modulus``; ``x_th = 10^{0.4 (M_* - M_th)}``;
        ``f = gammaincc(s, x_th) / gammaincc(s, x_dim)``, clipped to ``[0, 1]``.
        Caller is responsible for empty pixels (``m_th = -inf``, ``f = 0``).
        """
        # M_* carries the +5 log10 h that cancels the -5 log10 h of the distance
        # modulus (DERIVATION Sec. 1.5): f_k depends on h only through the weak
        # Omega_m shape of the dimensionless distance, NOT through h itself.
        m_star = SCHECHTER_M_STAR_0 + 5.0 * np.log10(h)
        dist_mod = self._distance_modulus(z, h)  # (Z,)
        # M_th_k(z) = m_th,k - 25 - 5 log10(d_L/Mpc) - K. Gray-2020 App. A.2.
        m_th_abs = m_th_values[None, :] - dist_mod[:, None]  # (Z, P)
        x_th = 10.0 ** (0.4 * (m_star - m_th_abs))  # (Z, P), dimensionless L/L_*
        # Eq. (12)/(13) in Gray et al. (2020); GMV-2022 Eq. (2). Regularized
        # upper incomplete gamma => Gamma(s) cancels in the ratio.
        f = np.asarray(gammaincc(LF_S, x_th), dtype=np.float64) / _F_DENOM
        return np.clip(f, 0.0, 1.0)

    def f_k(
        self,
        z: float | npt.NDArray[np.floating[Any]],
        k: int,
        h: float = H,
    ) -> float | npt.NDArray[np.float64]:
        r"""Per-pixel completeness ``f_k(z, h)`` at pixel ``k``.

        Returns a scalar if ``z`` is scalar, else an array of the same shape.
        Empty / ZoA pixels (``m_th = -inf``) return exactly ``0``.
        """
        z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
        if not bool(self._valid[k]):
            out = np.zeros_like(z_arr)
        else:
            out = self._f_from_mth(np.asarray([self.m_th[k]], dtype=np.float64), z_arr, h)[:, 0]
        if np.ndim(z) == 0:
            return float(out[0])
        return out

    def f_map(
        self,
        z: float,
        h: float = H,
    ) -> npt.NDArray[np.float64]:
        r"""Full per-pixel completeness map ``f_k(z, h)`` over all ``npix`` pixels.

        Convenience reduction (no new physics): returns the same ``f_k`` values as
        :meth:`f_k` but for every pixel at once, with empty/ZoA pixels = 0. Used for
        whole-sky visualisation and any all-pixel reduction. Shape ``(npix,)``.
        """
        out = np.zeros(self.npix, dtype=np.float64)
        if np.any(self._valid):
            z_arr = np.asarray([float(z)], dtype=np.float64)
            out[self._valid] = self._f_from_mth(self._m_th_valid, z_arr, h)[0]
        return out

    def f_bar(
        self,
        z: float | npt.NDArray[np.floating[Any]],
        h: float = H,
    ) -> float | npt.NDArray[np.float64]:
        r"""Sky-averaged completeness ``f_bar(z,h) = (1/N_pix) sum_k f_k(z,h)``.

        Empty pixels contribute ``0``; the average runs over ALL ``N_pix`` pixels
        (GMV-2022 Eq. 3, equal area).  Chunked over ``z`` to bound memory.
        Returns a scalar if ``z`` is scalar, else an array of the same shape.
        """
        z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
        out = np.empty(z_arr.shape, dtype=np.float64)
        n_valid_sum = self._m_th_valid
        for start in range(0, z_arr.size, _FBAR_Z_CHUNK):
            z_chunk = z_arr[start : start + _FBAR_Z_CHUNK]
            f_chunk = self._f_from_mth(n_valid_sum, z_chunk, h)  # (chunk, P_valid)
            out[start : start + _FBAR_Z_CHUNK] = f_chunk.sum(axis=1) / self.npix
        if np.ndim(z) == 0:
            return float(out[0])
        return out

    # Backward-compatible alias: the Omega-marginalized completeness. Lets the
    # PixelCompleteness object drop into any caller that previously used the
    # all-sky GladeCatalogCompleteness.get_completeness_at_redshift.
    def get_completeness_at_redshift(
        self,
        z: float | npt.NDArray[np.floating[Any]],
        h: float = H,
    ) -> float | npt.NDArray[np.float64]:
        return self.f_bar(z, h)

    # ------------------------------------------------------------------
    # Sky <-> pixel (BarycentricTrueEcliptic J2000)
    # ------------------------------------------------------------------

    def ang2pix(self, phi: float, theta: float) -> int:
        r"""Ecliptic ``(phi azimuth, theta colatitude)`` [rad] -> pixel index.

        ``astropy_healpix`` takes ``(lon, lat)`` with ``lat = pi/2 - theta``.
        """
        lon = float(phi) * u.rad
        lat = (np.pi / 2.0 - float(theta)) * u.rad
        return int(self._healpix.lonlat_to_healpix(lon, lat))

    def pixel_centers(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Ecliptic centres of every HEALPix pixel: ``(phi_k, theta_k)`` [rad].

        Pure geometry (the inverse of :meth:`ang2pix`; mirrors
        :meth:`sample_sky_in_pixels` with the cell-centre offset ``dx=dy=0.5``,
        the ``astropy_healpix`` default).  ``phi_k`` is the ecliptic azimuth
        (longitude) and ``theta_k = pi/2 - lat_k`` the ecliptic colatitude, the
        SAME BarycentricTrueEcliptic(J2000) frame the response and the catalog
        share.  No physical value is computed here (Change 6, software).

        Returns
        -------
        (phi_k, theta_k) : tuple of ndarray, each shape ``(npix,)``
            Pixel-centre ecliptic azimuth and colatitude in radians, indexed by
            HEALPix pixel ``k = 0 .. npix-1``.
        """
        lon, lat = self._healpix.healpix_to_lonlat(np.arange(self.npix))
        phi_k = np.asarray(lon.to(u.rad).value, dtype=np.float64)
        theta_k = np.asarray(np.pi / 2.0 - lat.to(u.rad).value, dtype=np.float64)
        return phi_k, theta_k

    def f_pixels(
        self,
        z: float | npt.NDArray[np.floating[Any]],
        h: float = H,
    ) -> npt.NDArray[np.float64]:
        r"""Per-pixel completeness for an array of redshifts: shape ``(Z, npix)``.

        Vectorised generalisation of :meth:`f_map` over a redshift array (no new
        physics: identical ``f_k`` values, empty/ZoA pixels = 0).  Used by the
        sky-resolved missing-completion selection integral ``beta_Gbar`` to form
        the per-band incompleteness sums ``sum_{k in band}(1 - f_k(z))``.

        Parameters
        ----------
        z : float or ndarray
            Redshift(s).
        h : float
            Dimensionless Hubble parameter.

        Returns
        -------
        ndarray, shape ``(Z, npix)``
            ``f_k(z, h)`` for every pixel, with empty/ZoA pixels exactly ``0``.
        """
        z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
        out = np.zeros((z_arr.size, self.npix), dtype=np.float64)
        if np.any(self._valid):
            out[:, self._valid] = self._f_from_mth(self._m_th_valid, z_arr, h)
        return out

    def sample_sky_in_pixels(
        self, pix: npt.NDArray[np.int_], rng: np.random.Generator
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Uniform-in-pixel sky draw: ``pix -> (phi azimuth, theta colatitude)`` [rad].

        ``(dx, dy)`` uniform on ``[0, 1]^2`` map to a uniform position within the
        equal-area HEALPix cell (area-preserving projection).
        """
        n = int(pix.size)
        dx = rng.uniform(0.0, 1.0, size=n)
        dy = rng.uniform(0.0, 1.0, size=n)
        lon, lat = self._healpix.healpix_to_lonlat(pix, dx=dx, dy=dy)
        phi = np.asarray(lon.to(u.rad).value, dtype=np.float64)
        theta = np.asarray(np.pi / 2.0 - lat.to(u.rad).value, dtype=np.float64)
        return phi, theta

    # ------------------------------------------------------------------
    # Joint dark-host sampler support (FIX-A, Change 5.5)
    # ------------------------------------------------------------------

    def pixel_dark_weights(
        self,
        z_grid: npt.NDArray[np.float64],
        p_pop: npt.NDArray[np.float64],
        h: float,
    ) -> npt.NDArray[np.float64]:
        r"""Per-pixel dark weight ``W_k = INTEGRAL (1 - f_k(z)) p_pop(z) dz``.

        Shape ``(npix,)``.  Empty pixels (``f_k == 0``) get the maximal weight
        ``INTEGRAL p_pop dz`` (dark hosts concentrate in the ZoA).  Chunked over
        pixels to bound memory.  ``W_k`` enters only the dimensionless pixel
        selection ratio ``W_k / sum_k W_k`` (DERIVATION Sec. 3).
        """
        z_grid = np.asarray(z_grid, dtype=np.float64)
        p_pop = np.asarray(p_pop, dtype=np.float64)
        w_full = float(np.trapezoid(p_pop, z_grid))  # empty-pixel weight (f_k == 0)
        weights = np.full(self.npix, w_full, dtype=np.float64)

        valid_idx = np.flatnonzero(self._valid)
        for start in range(0, valid_idx.size, _WK_PIX_CHUNK):
            chunk_idx = valid_idx[start : start + _WK_PIX_CHUNK]
            f_chunk = self._f_from_mth(self.m_th[chunk_idx], z_grid, h)  # (Zg, chunk)
            integrand = (1.0 - f_chunk) * p_pop[:, None]  # (Zg, chunk)
            weights[chunk_idx] = np.trapezoid(integrand, z_grid, axis=0)
        return weights


# ======================================================================
# Map builder + cache (Change 5.7)
# ======================================================================


def build_m_th_map(
    catalog_path: str = REDUCED_CATALOGUE_FILE_PATH,
    nside: int = NSIDE,
    order: str = HEALPIX_ORDER,
    min_galaxies: int = EMPTY_PIXEL_MIN_GALAXIES,
    chunksize: int = 2_000_000,
) -> npt.NDArray[np.float64]:
    r"""Build the per-pixel median apparent-B-magnitude map ``m_th[npix]``.

    Reads the on-disk reduced catalog (already filtered to redshift flag in
    ``{1, 3}``; columns RA, Dec [equatorial deg], apparent B mag, ...), rotates
    RA/Dec to BarycentricTrueEcliptic(J2000), ``ang2pix`` at ``nside``, and takes
    the per-pixel median of the non-null apparent B magnitudes.  Pixels with
    fewer than ``min_galaxies`` non-null-B galaxies get ``m_th = -inf`` (``f_k == 0``;
    Zone of Avoidance / empty), with NO separate mask.

    The completeness is catalog DEPTH (mass-independent), so this builds from the
    FULL flag-{1,3} ~22.6M catalog, NOT the mass-pruned subset (DERIVATION Sec. 8).

    Returns
    -------
    ndarray, shape (12 nside^2,)
        Per-pixel median apparent B magnitude (``-inf`` for empty/ZoA pixels).
    """
    import pandas as pd

    healpix = HEALPix(nside=nside, order=order)
    # On-disk column order (handler.read_reduced_galaxy_catalog): RA, Dec,
    # APPARENT_B_MAG, z, ..., with the RETAINED redshift flag as trailing column.
    # Use the shared source of truth so positional alignment survives schema edits.
    on_disk_names = _reduced_catalog_column_names()
    use_cols = [
        CatalogueColumns.RIGHT_ASCENSION.name,
        CatalogueColumns.DECLINATION.name,
        CatalogueColumns.APPARENT_B_MAG.name,
    ]

    pix_parts: list[npt.NDArray[np.int64]] = []
    bmag_parts: list[npt.NDArray[np.float64]] = []
    n_rows = 0
    n_bmag = 0
    reader = pd.read_csv(
        catalog_path,
        header=None,
        names=on_disk_names,
        usecols=use_cols,
        chunksize=chunksize,
    )
    for chunk in reader:
        n_rows += len(chunk)
        chunk = chunk.dropna(subset=[CatalogueColumns.APPARENT_B_MAG.name])
        if len(chunk) == 0:
            continue
        ra = chunk[CatalogueColumns.RIGHT_ASCENSION.name].to_numpy(dtype=np.float64)
        dec = chunk[CatalogueColumns.DECLINATION.name].to_numpy(dtype=np.float64)
        bmag = chunk[CatalogueColumns.APPARENT_B_MAG.name].to_numpy(dtype=np.float64)
        # Equatorial ICRS J2000 -> ecliptic SSB (same rotation as handler).
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        ecl = coord.transform_to(BarycentricTrueEcliptic(equinox="J2000"))
        pix = np.asarray(healpix.lonlat_to_healpix(ecl.lon, ecl.lat), dtype=np.int64)
        pix_parts.append(pix)
        bmag_parts.append(bmag)
        n_bmag += len(bmag)
        _LOGGER.info(
            "build_m_th_map: %d rows scanned (%d with non-null B mag).",
            n_rows,
            n_bmag,
        )

    m_th = np.full(healpix.npix, -np.inf, dtype=np.float64)
    if not pix_parts:
        _LOGGER.warning("build_m_th_map: no non-null-B galaxies found; all pixels empty.")
        return m_th

    all_pix = np.concatenate(pix_parts)
    all_bmag = np.concatenate(bmag_parts)
    grouped = pd.DataFrame({"pix": all_pix, "b": all_bmag}).groupby("pix")["b"]
    median = grouped.median()
    count = grouped.count()
    keep = count >= min_galaxies
    kept_pix = median.index[keep].to_numpy()
    m_th[kept_pix] = median[keep].to_numpy()
    _LOGGER.info(
        "build_m_th_map: nside=%d, %d/%d pixels populated (>=%d B-galaxies); "
        "%d total rows, %d with B mag; median per-pixel m_th=%.3f.",
        nside,
        int(keep.sum()),
        healpix.npix,
        min_galaxies,
        n_rows,
        n_bmag,
        float(np.median(m_th[np.isfinite(m_th)])) if np.any(np.isfinite(m_th)) else float("nan"),
    )
    return m_th


def from_cache_or_build(
    cache_path: str = M_TH_CACHE_PATH,
    catalog_path: str = REDUCED_CATALOGUE_FILE_PATH,
    nside: int = NSIDE,
    order: str = HEALPIX_ORDER,
    min_galaxies: int = EMPTY_PIXEL_MIN_GALAXIES,
) -> PixelCompleteness:
    r"""Load the frozen ``m_th`` cache, building (and caching) it on first use.

    The cache is the SOLE source of ``f`` (C1): the SAME ``.npy`` file is loaded
    byte-identically by injection and inference.
    """
    if os.path.exists(cache_path):
        m_th = np.load(cache_path)
        _LOGGER.info("Loaded cached m_th map from %s (nside=%d).", cache_path, nside)
    else:
        _LOGGER.info("No m_th cache at %s; building from %s.", cache_path, catalog_path)
        m_th = build_m_th_map(
            catalog_path=catalog_path, nside=nside, order=order, min_galaxies=min_galaxies
        )
        np.save(cache_path, m_th)
        _LOGGER.info("Saved m_th map to %s.", cache_path)
    return PixelCompleteness(m_th, nside=nside, order=order)
