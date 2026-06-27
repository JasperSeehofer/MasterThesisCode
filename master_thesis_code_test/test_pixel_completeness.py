"""Unit tests for the per-HEALPix-pixel Schechter completeness (Change 5a).

Covers the estimator's analytical limits, the sky-average identity
``f_bar = mean_k f_k``, the exact H0-independence (the ``+5 log10 h``
cancellation), the ecliptic ang2pix / uniform-in-pixel round trip, the empty/ZoA
pixel rule, the m_th map builder on a synthetic catalog, and the C1 byte-identity
of the cached map.  All CPU-only and synthetic (no GLADE+ file, no GPU).

References:
    Gray, Messenger & Veitch (2022), arXiv:2111.04629, Eqs. (2)(3)(5).
    Gray et al. (2020), arXiv:1908.06050, Eqs. (12)(13), Appendix A.2.
"""

import numpy as np
import pytest

from master_thesis_code.galaxy_catalogue.glade_completeness import GladeCatalogCompleteness
from master_thesis_code.galaxy_catalogue.pixel_completeness import (
    EMPTY_PIXEL_MIN_GALAXIES,
    NSIDE,
    X_DIM,
    PixelCompleteness,
    build_m_th_map,
    from_cache_or_build,
)

_H = 0.73


def _uniform_map(m_th_value: float, nside: int = 4) -> PixelCompleteness:
    """A PixelCompleteness whose every pixel shares one finite ``m_th``."""
    npix = 12 * nside * nside
    return PixelCompleteness(np.full(npix, m_th_value, dtype=np.float64), nside=nside)


def _mixed_map(nside: int = 4) -> PixelCompleteness:
    """A map with a spread of finite m_th plus one empty (-inf) pixel."""
    npix = 12 * nside * nside
    rng = np.random.default_rng(0)
    m_th = rng.uniform(16.0, 21.0, size=npix)
    m_th[0] = -np.inf  # empty / ZoA pixel
    return PixelCompleteness(m_th, nside=nside)


# ----------------------------------------------------------------------
# Estimator analytical limits
# ----------------------------------------------------------------------


def test_x_dim_is_one_milli() -> None:
    """x_dim = 10^{0.4 (M_*-M_dim)} = 10^{0.4 (-7.5)} = 1e-3, H0-independent."""
    assert X_DIM == pytest.approx(1e-3, rel=1e-9)


def test_f_k_goes_to_one_at_zero_redshift() -> None:
    """z -> 0 => d_L -> 0 => x_th -> 0 => f_k -> 1 (complete nearby)."""
    pc = _uniform_map(19.0)
    f = pc.f_k(1e-6, k=10, h=_H)
    assert f == pytest.approx(1.0, abs=1e-6)


def test_f_k_decreases_to_zero_at_high_redshift() -> None:
    """f_k is monotone decreasing in z and -> 0 at large z."""
    pc = _uniform_map(19.0)
    z = np.array([0.01, 0.05, 0.1, 0.2, 0.4, 0.8])
    f = np.asarray(pc.f_k(z, k=10, h=_H))
    assert np.all(np.diff(f) <= 1e-12), f"f_k must be non-increasing in z: {f}"
    assert f[0] > 0.9
    assert f[-1] < 0.05


def test_f_k_in_unit_interval() -> None:
    """0 <= f_k <= 1 for every pixel (valid and empty) over a z range."""
    pc = _mixed_map()
    z = np.linspace(1e-4, 1.0, 50)
    for k in range(pc.npix):
        f = np.asarray(pc.f_k(z, k=k, h=_H))
        assert np.all(f >= 0.0) and np.all(f <= 1.0)


def test_empty_pixel_is_pure_completion() -> None:
    """An empty/ZoA pixel (m_th = -inf) has f_k == 0 for all z (pure completion)."""
    pc = _mixed_map()
    z = np.linspace(1e-4, 0.5, 20)
    assert np.all(np.asarray(pc.f_k(z, k=0, h=_H)) == 0.0)
    assert pc.f_k(0.1, k=0, h=_H) == 0.0


# ----------------------------------------------------------------------
# Sky-average identity and limiting case (a) (Omega-independent)
# ----------------------------------------------------------------------


def test_f_bar_equals_mean_of_f_k() -> None:
    """f_bar(z) == (1/N_pix) sum_k f_k(z) to machine precision (GMV Eq. 3)."""
    pc = _mixed_map()
    z = np.array([0.02, 0.08, 0.2, 0.45])
    f_bar = np.asarray(pc.f_bar(z, _H))
    f_mean = np.mean(np.array([pc.f_k(z, k, _H) for k in range(pc.npix)]), axis=0)
    np.testing.assert_allclose(f_bar, f_mean, rtol=0, atol=1e-13)


def test_uniform_map_collapses_to_scalar_completeness() -> None:
    """All m_th equal => f_bar == f_k(any k) == a single f(z) (limiting case a)."""
    pc = _uniform_map(19.5)
    z = np.array([0.03, 0.1, 0.25])
    f_bar = np.asarray(pc.f_bar(z, _H))
    f_k0 = np.asarray(pc.f_k(z, 0, _H))
    f_k7 = np.asarray(pc.f_k(z, 7, _H))
    np.testing.assert_allclose(f_bar, f_k0, rtol=0, atol=1e-13)
    np.testing.assert_allclose(f_k0, f_k7, rtol=0, atol=1e-13)


def test_completeness_is_exactly_h_independent() -> None:
    """The +5 log10 h in M_* exactly cancels the distance-modulus -5 log10 h.

    DERIVATION Sec. 1.5: f_k depends on H0 only through the dimensionless
    distance shape (h-independent), so f_k(h1) == f_k(h2) to machine precision.
    Exposes the spurious h-dependence of the old Dalya interpolation (OQ-A).
    """
    pc = _mixed_map()
    z = np.array([0.05, 0.15, 0.3])
    f_low = np.asarray(pc.f_bar(z, 0.60))
    f_high = np.asarray(pc.f_bar(z, 0.86))
    np.testing.assert_allclose(f_low, f_high, rtol=1e-12, atol=0)


def test_scalar_and_array_inputs_agree() -> None:
    """Scalar z returns a float equal to the corresponding array element."""
    pc = _mixed_map()
    z = 0.123
    assert pc.f_bar(z, _H) == pytest.approx(float(np.asarray(pc.f_bar(np.array([z]), _H))[0]))
    assert pc.f_k(z, 5, _H) == pytest.approx(float(np.asarray(pc.f_k(np.array([z]), 5, _H))[0]))
    assert isinstance(pc.f_bar(z, _H), float)
    assert isinstance(pc.f_k(z, 5, _H), float)


# ----------------------------------------------------------------------
# Sky <-> pixel (ecliptic, astropy_healpix)
# ----------------------------------------------------------------------


def test_sample_in_pixel_then_ang2pix_round_trips() -> None:
    """Uniform-in-pixel draws map back to their own pixel via ang2pix."""
    pc = _uniform_map(19.0, nside=NSIDE)
    rng = np.random.default_rng(1)
    pix = rng.integers(0, pc.npix, size=5000)
    phi, theta = pc.sample_sky_in_pixels(pix, rng)
    assert np.all((phi >= 0.0) & (phi < 2.0 * np.pi + 1e-9))
    assert np.all((theta >= 0.0) & (theta <= np.pi + 1e-9))
    back = np.array([pc.ang2pix(float(phi[i]), float(theta[i])) for i in range(pix.size)])
    assert np.array_equal(back, pix)


def test_constructor_rejects_wrong_npix() -> None:
    """A map whose length is not 12 nside^2 is rejected."""
    with pytest.raises(ValueError):
        PixelCompleteness(np.zeros(100), nside=4)  # 12*16 = 192 != 100


# ----------------------------------------------------------------------
# Joint dark-host weights (FIX-A support)
# ----------------------------------------------------------------------


def test_pixel_dark_weights_uniform_map_are_equal() -> None:
    """Omega-independent (uniform) map => every W_k equal (uniform pixel pick)."""
    pc = _uniform_map(19.0)
    z_grid = np.linspace(1e-4, 0.5, 512)
    p_pop = (1.0 + z_grid) ** -1  # any positive density
    w = pc.pixel_dark_weights(z_grid, p_pop, _H)
    assert w.shape == (pc.npix,)
    np.testing.assert_allclose(w, w[0], rtol=1e-12)


def test_pixel_dark_weights_empty_pixel_is_maximal() -> None:
    """An empty pixel (f_k == 0) gets the maximal dark weight INTEGRAL p_pop dz."""
    pc = _mixed_map()
    z_grid = np.linspace(1e-4, 0.5, 1024)
    p_pop = np.ones_like(z_grid)
    w = pc.pixel_dark_weights(z_grid, p_pop, _H)
    w_full = float(np.trapezoid(p_pop, z_grid))
    assert w[0] == pytest.approx(w_full, rel=1e-9)  # pixel 0 is empty
    assert np.all(w[1:] <= w[0] + 1e-9)  # populated pixels have (1-f) < 1


# ----------------------------------------------------------------------
# m_th map builder + C1 byte-identity of the cache
# ----------------------------------------------------------------------


def test_build_m_th_map_synthetic_catalog(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Builder takes per-pixel medians and applies the <min-galaxies => -inf rule."""
    import pandas as pd

    nside = 1  # 12 pixels
    # 12 galaxies at one sky position (one pixel) + 3 at another (below the floor).
    b_full = np.arange(10.0, 22.0)  # 12 values, median = 15.5
    rows = []
    for b in b_full:
        rows.append((10.0, 20.0, b, 0.05, 0.001, 0.3, 0.1))  # same (RA, Dec)
    for b in [12.0, 13.0, 14.0]:
        rows.append((200.0, -40.0, b, 0.05, 0.001, 0.3, 0.1))  # different pixel
    csv = tmp_path / "synthetic_reduced.csv"
    pd.DataFrame(rows).to_csv(csv, header=False, index=False)

    m_th = build_m_th_map(catalog_path=str(csv), nside=nside, min_galaxies=10)
    assert m_th.shape == (12,)
    finite = np.isfinite(m_th)
    assert int(finite.sum()) == 1, "only the >=10-galaxy pixel should be populated"
    assert float(m_th[finite][0]) == pytest.approx(np.median(b_full))


def test_from_cache_or_build_is_byte_identical(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """C1: the cached map reloads byte-identically and yields identical f (both sides)."""
    import pandas as pd

    nside = 1
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(40):
        rows.append(
            (rng.uniform(0, 360), rng.uniform(-80, 80), rng.uniform(15, 21), 0.05, 0.001, 0.3, 0.1)
        )
    csv = tmp_path / "synthetic_reduced.csv"
    pd.DataFrame(rows).to_csv(csv, header=False, index=False)
    cache = tmp_path / "m_th.npy"

    # First call builds + caches; second loads the frozen cache.
    pc_inj = from_cache_or_build(cache_path=str(cache), catalog_path=str(csv), nside=nside)
    assert cache.exists()
    pc_inf = from_cache_or_build(cache_path=str(cache), catalog_path=str(csv), nside=nside)

    np.testing.assert_array_equal(pc_inj.m_th, pc_inf.m_th)  # byte-identical map
    z = np.array([0.02, 0.1, 0.3])
    np.testing.assert_array_equal(np.asarray(pc_inj.f_bar(z, _H)), np.asarray(pc_inf.f_bar(z, _H)))
    for k in range(pc_inj.npix):
        np.testing.assert_array_equal(
            np.asarray(pc_inj.f_k(z, k, _H)), np.asarray(pc_inf.f_k(z, k, _H))
        )


# ----------------------------------------------------------------------
# GladeCatalogCompleteness Omega-independent shims (regression / case a)
# ----------------------------------------------------------------------


def test_glade_shims_match_all_sky_curve() -> None:
    """GladeCatalogCompleteness.f_bar / f_k equal its all-sky curve; ang2pix == 0."""
    gc = GladeCatalogCompleteness()
    z = np.array([0.02, 0.1, 0.3])
    base = np.asarray(gc.get_completeness_at_redshift(z, _H))
    np.testing.assert_array_equal(np.asarray(gc.f_bar(z, _H)), base)
    np.testing.assert_array_equal(np.asarray(gc.f_k(z, 5, _H)), base)
    np.testing.assert_array_equal(np.asarray(gc.f_k(z, 999, _H)), base)
    assert gc.ang2pix(1.0, 0.5) == 0


def test_default_constants_match_protocol() -> None:
    """The locked Change 5.0 constants match the protocol."""
    from master_thesis_code.galaxy_catalogue import pixel_completeness as pcmod

    assert pcmod.SCHECHTER_ALPHA == -1.07
    assert pcmod.SCHECHTER_M_STAR_0 == -19.7
    assert pcmod.SCHECHTER_M_DIM_0 == -12.2
    assert pcmod.LF_S == pytest.approx(0.93)
    assert pcmod.NSIDE == 32
    assert EMPTY_PIXEL_MIN_GALAXIES == 10
