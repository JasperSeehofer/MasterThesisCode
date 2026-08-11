"""Gate tests for CHANGE 4b: out-of-catalog (dark) EMRI host injection.

These tests pin the self-consistent dark-event injection that augments the
in-catalog rate-weighted host draw with a fraction ``1-F`` of out-of-catalog
("dark") hosts, so the injected population matches the inference mixture
``f*L_cat + (1-f)*L_comp`` (Gray et al. 2020, arXiv:1908.06050, Eq. 9; Chen et
al. 2024, arXiv:2212.08694, self-consistency).

Covered (all CPU-only, synthetic, deterministic under a fixed seed):

* ``draw_dark_host`` / ``draw_dark_hosts`` sampling:
    - redshift ``z ∝ (1-f(z))/(1+z) * dVc/dz`` (KS vs target CDF);
    - mass ``log10 M ∝ phi_MBH(M) * R_eff(M)`` (KS vs target CDF);
    - isotropic sky (``phiS`` uniform, ``cos qS`` uniform);
    - reproducibility under a fixed seed;
    - every drawn ``z`` lies in ``(z_min, z_max)``;
    - dark host carries ``catalog_index = -1``.
* Global in-catalog fraction ``F``:
    - limiting cases ``f=1 -> F=1`` and ``f=0 -> F=0``;
    - constant ``f=c -> F=c`` (known value).
* Bernoulli(F) mixture split:
    - realised in-catalog fraction over many draws ``≈ F``;
    - ``F=1 ->`` all in-catalog, ``F=0 ->`` all dark.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
from scipy.integrate import cumulative_trapezoid
from scipy.stats import kstest

from darksiren_emri.constants import HOST_DRAW_Z_MAX, H
from darksiren_emri.dark_siren_injection import (
    DARK_HOST_CATALOG_INDEX,
    compute_global_catalog_fraction,
    draw_dark_host,
    draw_dark_hosts,
    draw_mixture_hosts,
)
from darksiren_emri.emri_rate import R_eff_per_mbh, mbh_mass_function
from darksiren_emri.galaxy_catalogue.glade_completeness import GladeCatalogCompleteness
from darksiren_emri.galaxy_catalogue.handler import HostGalaxy
from darksiren_emri.galaxy_catalogue.pixel_completeness import PixelCompleteness
from darksiren_emri.physical_relations import comoving_volume_element

_M_MIN: float = 1.0e4
_M_MAX: float = 1.0e7
_Z_MIN: float = 1.0e-6


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _constant_completeness(fraction_percent: float) -> GladeCatalogCompleteness:
    """A completeness model with a constant ``f(z) = fraction_percent/100``."""
    return GladeCatalogCompleteness(
        distance=[0.0, 1.0e7],
        completeness=[fraction_percent, fraction_percent],
    )


def _expected_redshift_cdf(
    completeness: GladeCatalogCompleteness,
    h: float,
    z_min: float,
    z_max: float,
    n: int = 20000,
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Target dark-host redshift CDF ``∝ (1-f(z))/(1+z) * dVc/dz``."""
    z = np.linspace(z_min, z_max, n, dtype=np.float64)
    f = np.clip(np.asarray(completeness.get_completeness_at_redshift(z, h), dtype=np.float64), 0, 1)
    dVc = np.asarray(comoving_volume_element(z, h=h), dtype=np.float64)
    weight = (1.0 - f) / (1.0 + z) * dVc
    cdf = cumulative_trapezoid(weight, z, initial=0.0)
    cdf /= cdf[-1]

    def _cdf(query: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(np.interp(query, z, cdf), dtype=np.float64)

    return _cdf


def _expected_log_mass_cdf(
    m_min: float, m_max: float, n: int = 20000
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Target dark-host ``log10 M`` CDF ``∝ phi_MBH(M) * R_eff(M)`` (per dex)."""
    log_m = np.linspace(np.log10(m_min), np.log10(m_max), n, dtype=np.float64)
    mass = 10.0**log_m
    weight = np.asarray(mbh_mass_function(mass), dtype=np.float64) * np.asarray(
        R_eff_per_mbh(mass), dtype=np.float64
    )
    cdf = cumulative_trapezoid(weight, log_m, initial=0.0)
    cdf /= cdf[-1]

    def _cdf(query: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(np.interp(query, log_m, cdf), dtype=np.float64)

    return _cdf


class _FakeCatalog:
    """Lightweight in-catalog host source for the mixture-split tests.

    Structurally matches the members of ``GalaxyCatalogueHandler`` used by
    ``draw_mixture_hosts`` (``M_min``, ``M_max``, ``draw_rate_weighted_hosts``)
    without touching the multi-GB GLADE file. Returns hosts with a non-negative
    ``catalog_index`` so they are flagged in-catalog.
    """

    def __init__(self, m_min: float = _M_MIN, m_max: float = _M_MAX) -> None:
        self.M_min = m_min
        self.M_max = m_max

    def draw_rate_weighted_hosts(
        self,
        number_of_hosts: int,
        rng: np.random.Generator,
        z_max: float = HOST_DRAW_Z_MAX,
    ) -> list[HostGalaxy]:
        return [
            HostGalaxy.from_attributes(
                phiS=0.0,
                qS=1.0,
                z=0.1,
                z_error=0.001,
                M=1.0e5,
                M_error=1.0e4,
                catalog_index=i,
            )
            for i in range(number_of_hosts)
        ]


# ----------------------------------------------------------------------------
# Dark host representation
# ----------------------------------------------------------------------------


def test_dark_host_has_catalog_index_minus_one() -> None:
    """A drawn dark host is flagged with catalog_index = -1."""
    assert DARK_HOST_CATALOG_INDEX == -1
    rng = np.random.default_rng(0)
    host = draw_dark_host(rng, GladeCatalogCompleteness(), _M_MIN, _M_MAX, h=H)
    assert host.catalog_index == -1
    # Drawn (not catalog) quantities are carried straight through.
    assert _Z_MIN < host.z < HOST_DRAW_Z_MAX
    assert _M_MIN <= host.M <= _M_MAX
    assert 0.0 <= host.phiS <= 2.0 * np.pi
    assert 0.0 <= host.qS <= np.pi
    # Peculiar-velocity floor + finite mass error set up a well-formed object.
    assert host.z_error == pytest.approx(0.0015)
    assert host.M_error > 0.0


def test_from_attributes_threads_catalog_index() -> None:
    """HostGalaxy.from_attributes carries an explicit catalog_index (-1 for dark)."""
    host = HostGalaxy.from_attributes(
        phiS=0.1, qS=1.0, z=0.2, z_error=0.001, M=1e5, M_error=1e4, catalog_index=-1
    )
    assert host.catalog_index == -1


# ----------------------------------------------------------------------------
# draw_dark_host(s): bounds + reproducibility
# ----------------------------------------------------------------------------


def test_dark_redshifts_within_bounds() -> None:
    """Every drawn dark-host redshift lies strictly inside (z_min, z_max)."""
    rng = np.random.default_rng(1)
    hosts = draw_dark_hosts(5000, rng, GladeCatalogCompleteness(), _M_MIN, _M_MAX, h=H)
    z = np.array([host.z for host in hosts])
    assert np.all(z < HOST_DRAW_Z_MAX)
    assert np.all(z > 0.0)


def test_dark_masses_within_bounds() -> None:
    """Every drawn dark-host mass lies inside [M_min, M_max]."""
    rng = np.random.default_rng(2)
    hosts = draw_dark_hosts(5000, rng, GladeCatalogCompleteness(), _M_MIN, _M_MAX, h=H)
    masses = np.array([host.M for host in hosts])
    assert np.all(masses >= _M_MIN)
    assert np.all(masses <= _M_MAX)


def test_dark_draw_reproducible_under_fixed_seed() -> None:
    """Two fixed-seed generators yield identical dark-host draws."""
    completeness = GladeCatalogCompleteness()
    hosts_a = draw_dark_hosts(200, np.random.default_rng(42), completeness, _M_MIN, _M_MAX, h=H)
    hosts_b = draw_dark_hosts(200, np.random.default_rng(42), completeness, _M_MIN, _M_MAX, h=H)
    z_a = np.array([h.z for h in hosts_a])
    z_b = np.array([h.z for h in hosts_b])
    m_a = np.array([h.M for h in hosts_a])
    m_b = np.array([h.M for h in hosts_b])
    phi_a = np.array([h.phiS for h in hosts_a])
    phi_b = np.array([h.phiS for h in hosts_b])
    q_a = np.array([h.qS for h in hosts_a])
    q_b = np.array([h.qS for h in hosts_b])
    np.testing.assert_array_equal(z_a, z_b)
    np.testing.assert_array_equal(m_a, m_b)
    np.testing.assert_array_equal(phi_a, phi_b)
    np.testing.assert_array_equal(q_a, q_b)


# ----------------------------------------------------------------------------
# draw_dark_host(s): distribution shapes (KS tests, fixed seed -> deterministic)
# ----------------------------------------------------------------------------


def test_dark_redshift_distribution_matches_target() -> None:
    """Dark-host redshifts follow (1-f(z))/(1+z) * dVc/dz (KS test)."""
    rng = np.random.default_rng(7)
    hosts = draw_dark_hosts(8000, rng, GladeCatalogCompleteness(), _M_MIN, _M_MAX, h=H)
    z = np.array([host.z for host in hosts])
    cdf = _expected_redshift_cdf(GladeCatalogCompleteness(), H, _Z_MIN, HOST_DRAW_Z_MAX)
    statistic, pvalue = kstest(z, cdf)
    assert statistic < 0.025, f"KS statistic {statistic:.4f} too large (p={pvalue:.4g})"


def test_dark_mass_distribution_matches_target() -> None:
    """Dark-host masses follow the per-dex phi_MBH(M)*R_eff(M) marginal (KS test)."""
    rng = np.random.default_rng(8)
    hosts = draw_dark_hosts(8000, rng, GladeCatalogCompleteness(), _M_MIN, _M_MAX, h=H)
    log_m = np.log10(np.array([host.M for host in hosts]))
    cdf = _expected_log_mass_cdf(_M_MIN, _M_MAX)
    statistic, pvalue = kstest(log_m, cdf)
    assert statistic < 0.025, f"KS statistic {statistic:.4f} too large (p={pvalue:.4g})"


def test_dark_sky_is_isotropic() -> None:
    """Dark-host sky positions are isotropic: phiS uniform, cos(qS) uniform."""
    rng = np.random.default_rng(9)
    hosts = draw_dark_hosts(8000, rng, GladeCatalogCompleteness(), _M_MIN, _M_MAX, h=H)
    phiS = np.array([host.phiS for host in hosts])
    cos_qS = np.cos(np.array([host.qS for host in hosts]))
    stat_phi, _ = kstest(phiS, "uniform", args=(0.0, 2.0 * np.pi))
    stat_cos, _ = kstest(cos_qS, "uniform", args=(-1.0, 2.0))
    assert stat_phi < 0.025, f"phiS not uniform: KS={stat_phi:.4f}"
    assert stat_cos < 0.025, f"cos(qS) not uniform: KS={stat_cos:.4f}"


# ----------------------------------------------------------------------------
# Global in-catalog fraction F
# ----------------------------------------------------------------------------


def test_global_fraction_full_completeness_is_one() -> None:
    """f(z) = 1 everywhere -> F = 1 (recovers the pure in-catalog draw)."""
    F = compute_global_catalog_fraction(_constant_completeness(100.0), h=H)
    assert F == pytest.approx(1.0, abs=1e-9)


def test_global_fraction_zero_completeness_is_zero() -> None:
    """f(z) = 0 everywhere -> F = 0 (every host dark)."""
    F = compute_global_catalog_fraction(_constant_completeness(0.0), h=H)
    assert F == pytest.approx(0.0, abs=1e-9)


def test_global_fraction_constant_completeness_equals_value() -> None:
    """f(z) = c (constant) -> F = c, independent of the population weighting."""
    F = compute_global_catalog_fraction(_constant_completeness(37.0), h=H)
    assert F == pytest.approx(0.37, abs=1e-6)


def test_global_fraction_default_completeness_in_unit_interval() -> None:
    """The realistic GLADE+ completeness gives a strictly interior F in (0, 1)."""
    F = compute_global_catalog_fraction(GladeCatalogCompleteness(), h=H)
    assert 0.0 < F < 1.0


# ----------------------------------------------------------------------------
# Bernoulli(F) mixture split
# ----------------------------------------------------------------------------


def test_realized_in_catalog_fraction_matches_F() -> None:
    """Over many draws the realised in-catalog fraction tracks F."""
    completeness = GladeCatalogCompleteness()
    F = compute_global_catalog_fraction(completeness, h=H)
    rng = np.random.default_rng(2024)
    n = 20000
    hosts = draw_mixture_hosts(n, rng, _FakeCatalog(), completeness, F, h=H)
    realized = np.mean([host.catalog_index != -1 for host in hosts])
    assert realized == pytest.approx(F, abs=0.02)


def test_mixture_full_fraction_is_all_in_catalog() -> None:
    """F = 1 -> every host is in-catalog (recovers CHANGE 3)."""
    rng = np.random.default_rng(11)
    hosts = draw_mixture_hosts(300, rng, _FakeCatalog(), _constant_completeness(100.0), 1.0, h=H)
    assert all(host.catalog_index != -1 for host in hosts)


def test_mixture_zero_fraction_is_all_dark() -> None:
    """F = 0 -> every host is dark (catalog_index = -1)."""
    rng = np.random.default_rng(12)
    hosts = draw_mixture_hosts(300, rng, _FakeCatalog(), GladeCatalogCompleteness(), 0.0, h=H)
    assert all(host.catalog_index == -1 for host in hosts)
    # And the dark draws are well-formed (z in bounds).
    z = np.array([host.z for host in hosts])
    assert np.all((z > 0.0) & (z < HOST_DRAW_Z_MAX))


# ----------------------------------------------------------------------------
# FIX-A: end-to-end PIXELATED dark draw (Change 5.5, _draw_dark_hosts_pixelated)
#
# The tests above all pass an Omega-INDEPENDENT completeness, which takes the
# legacy isotropic fallback branch of draw_dark_hosts. These exercise the
# PRODUCTION per-pixel W_k joint sampler -- the branch main.py actually runs --
# by passing a real PixelCompleteness, and assert it reproduces the joint
# p(z, k) proportional to (1 - f_k(z)) p_pop(z): pixel pick frequency tracks
# W_k = INTEGRAL (1 - f_k) p_pop dz (so dark hosts CLUSTER in low-completeness /
# Zone-of-Avoidance directions -- the load-bearing FIX-A correction, DERIVATION
# Sec. 3), and z|k* follows that pixel's incompleteness-weighted population.
# ----------------------------------------------------------------------------


def _pixel_completeness_gradient() -> PixelCompleteness:
    """A 48-pixel (nside=2) map: 3-level completeness gradient + 2 empty pixels."""
    npix = 12 * 2 * 2
    m_th = np.empty(npix, dtype=np.float64)
    third = npix // 3
    m_th[:third] = 21.0  # most complete (high f_k)
    m_th[third : 2 * third] = 19.5
    m_th[2 * third :] = 17.5  # least complete (low f_k)
    m_th[7] = -np.inf  # empty / ZoA (f_k == 0, maximal dark weight)
    m_th[31] = -np.inf
    return PixelCompleteness(m_th, nside=2)


def test_pixelated_dark_draw_takes_pixel_branch_and_is_wellformed() -> None:
    """A real PixelCompleteness routes draw_dark_hosts through the W_k sampler."""
    from darksiren_emri.dark_siren_injection import _PixelDarkSampler

    pc = _pixel_completeness_gradient()
    # Dispatch precondition (assert on a fresh instance so pc keeps its concrete type).
    assert isinstance(_pixel_completeness_gradient(), _PixelDarkSampler)
    rng = np.random.default_rng(101)
    hosts = draw_dark_hosts(3000, rng, pc, _M_MIN, _M_MAX, h=H)
    assert len(hosts) == 3000
    assert all(host.catalog_index == DARK_HOST_CATALOG_INDEX for host in hosts)
    z = np.array([host.z for host in hosts])
    phi = np.array([host.phiS for host in hosts])
    q = np.array([host.qS for host in hosts])
    m = np.array([host.M for host in hosts])
    assert np.all((z > 0.0) & (z < HOST_DRAW_Z_MAX))
    assert np.all((phi >= 0.0) & (phi < 2.0 * np.pi + 1e-9))
    assert np.all((q >= 0.0) & (q <= np.pi + 1e-9))
    assert np.all((m >= _M_MIN) & (m <= _M_MAX))


def test_pixelated_dark_draw_reproducible_under_fixed_seed() -> None:
    """The W_k joint sampler is deterministic under a fixed seed."""
    pc = _pixel_completeness_gradient()
    a = draw_dark_hosts(500, np.random.default_rng(7), pc, _M_MIN, _M_MAX, h=H)
    b = draw_dark_hosts(500, np.random.default_rng(7), pc, _M_MIN, _M_MAX, h=H)
    np.testing.assert_array_equal([h.z for h in a], [h.z for h in b])
    np.testing.assert_array_equal([h.phiS for h in a], [h.phiS for h in b])
    np.testing.assert_array_equal([h.qS for h in a], [h.qS for h in b])


def test_pixelated_dark_draw_pixel_frequency_tracks_W_k() -> None:
    """Dark-host pixel occupancy is proportional to W_k = INT (1-f_k) p_pop dz.

    This is the JOINT (z, Omega) correlation FIX-A introduces (the load-bearing
    correction, DERIVATION Sec. 3): dark hosts cluster where the catalog is
    incomplete. An isotropic draw (occupancy proportional to sky area = uniform
    over equal-area pixels) would NOT track W_k.
    """
    pc = _pixel_completeness_gradient()
    rng = np.random.default_rng(202)
    n = 40000
    # Explicit shallow window: the sampler is depth-agnostic, but the W_k
    # contrast between pixels lives at low z where f_k differs. At the
    # campaign depth (HOST_DRAW_Z_MAX = 1.5) the volume integral is dominated
    # by the f ~ 0 shell, W_k becomes near-uniform, and the occupancy/W_k
    # Pearson r drowns in multinomial noise — a power issue, not a sampler
    # bug. z_max = 0.5 preserves the discriminating power of this check.
    z_max = 0.5
    hosts = draw_dark_hosts(n, rng, pc, _M_MIN, _M_MAX, h=H, z_max=z_max)
    pix = np.array([pc.ang2pix(float(h.phiS), float(h.qS)) for h in hosts])
    observed = np.bincount(pix, minlength=pc.npix).astype(np.float64)

    # Expected per-pixel weight (the categorical the sampler draws from).
    z_grid = np.linspace(1e-6, z_max, 4096)
    dVc = np.asarray(comoving_volume_element(z_grid, h=H), dtype=np.float64)
    p_pop = dVc / (1.0 + z_grid)
    w_k = pc.pixel_dark_weights(z_grid, p_pop, H)

    # Pixel pick frequency must track W_k (equal-area pixels -> occupancy ~ W_k).
    # This is the exact, map-agnostic statement of FIX-A's joint correlation.
    r = float(np.corrcoef(observed, w_k)[0, 1])
    assert r > 0.95, f"dark-host pixel occupancy should track W_k (Pearson r={r:.3f})"

    # Empty / ZoA pixels carry the MAXIMAL per-pixel weight (W_k = INT p_pop, f_k=0),
    # so their mean per-pixel occupancy exceeds the most-complete (m_th=21) third's.
    empty = ~np.isfinite(pc.m_th)
    high_f = np.zeros(pc.npix, dtype=bool)
    high_f[: pc.npix // 3] = True
    high_f &= np.isfinite(pc.m_th)
    assert observed[empty].mean() > observed[high_f].mean(), (
        f"empty pixels must out-attract complete pixels per pixel: "
        f"{observed[empty].mean():.0f} vs {observed[high_f].mean():.0f}"
    )


def test_pixelated_dark_draw_redshift_given_pixel_matches_target() -> None:
    """For a chosen pixel, z|k* follows (1 - f_k(z)) p_pop(z) (KS test)."""
    from scipy.integrate import cumulative_trapezoid
    from scipy.stats import kstest

    pc = _pixel_completeness_gradient()
    # First pixel of the least-complete third (m_th=17.5): attracts many hosts.
    k = 2 * (12 * 2 * 2) // 3
    rng = np.random.default_rng(303)
    hosts = draw_dark_hosts(60000, rng, pc, _M_MIN, _M_MAX, h=H)
    pix = np.array([pc.ang2pix(float(h.phiS), float(h.qS)) for h in hosts])
    z_k = np.array([h.z for h in hosts])[pix == k]
    assert z_k.size > 500, "need enough hosts in the test pixel"

    z_grid = np.linspace(1e-6, HOST_DRAW_Z_MAX, 20000)
    f_k = np.clip(np.asarray(pc.f_k(z_grid, k, H), dtype=np.float64), 0.0, 1.0)
    dVc = np.asarray(comoving_volume_element(z_grid, h=H), dtype=np.float64)
    weight = (1.0 - f_k) * dVc / (1.0 + z_grid)
    cdf = cumulative_trapezoid(weight, z_grid, initial=0.0)
    cdf /= cdf[-1]

    def target_cdf(query: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(np.interp(query, z_grid, cdf), dtype=np.float64)

    statistic, _ = kstest(z_k, target_cdf)
    assert statistic < 0.05, f"z|k* KS statistic {statistic:.4f} too large"
