r"""Dark (out-of-catalog) EMRI host injection for self-consistent H0 inference.

The simulation loop draws every injected EMRI host from the GLADE+ galaxy
catalog (``GalaxyCatalogueHandler.draw_rate_weighted_hosts``). The dark-siren
inference, however, evaluates a *mixture* likelihood per event,

.. math::

    \mathcal{L}_i = f(z_i)\,\mathcal{L}_\mathrm{cat}
                  + \bigl(1 - f(z_i)\bigr)\,\mathcal{L}_\mathrm{comp},

where :math:`f(z)` is the GLADE+ catalog completeness (Gray et al. 2020,
arXiv:1908.06050, Eq. 9) and :math:`\mathcal{L}_\mathrm{comp}` is the
out-of-catalog ("completion"/dark) term. To make the *injected* population
match the population the inference assumes (Chen, Fishbach & Holz 2024,
arXiv:2212.08694, self-consistency), a fraction ``1 - F`` of injected hosts
must come from galaxies that are NOT in the catalog.

This module provides

* :func:`compute_global_catalog_fraction` -- the run-level in-catalog fraction
  ``F`` obtained by marginalising the per-event mixing weight ``f(z)`` over the
  source-frame population prior in redshift;
* :func:`draw_dark_host` / :func:`draw_dark_hosts` -- draws of out-of-catalog
  hosts whose ``(z, M, sky)`` follow the missing-galaxy population, returned as
  :class:`~master_thesis_code.galaxy_catalogue.handler.HostGalaxy` objects with
  ``catalog_index = -1`` (no catalog snap, no catalog lookup);
* :func:`draw_mixture_hosts` -- the per-event ``Bernoulli(F)`` in/out-of-catalog
  split used by the simulation refill.

Population model (source frame, mass-integrated). The shared EMRI population
density is :func:`master_thesis_code.emri_rate.R_EMRI`. With the default
``p0 = 1`` surrogate it is redshift independent, so the *redshift* population
prior is purely cosmological,

.. math::

    p_\mathrm{pop}(z) \propto \frac{1}{1+z}\,\frac{dV_c}{dz},

and the mass-integrated rate cancels in every ratio formed here (it appears
identically in numerator and denominator of ``F``). The *mass* marginal is the
per-dex EMRI rate weight
``mbh_mass_function(M) * R_eff_per_mbh(M)`` (Babak et al. 2017,
arXiv:1703.09722); because ``mbh_mass_function`` is ``dn/dlog10 M`` this is a
density in ``log10 M``, which is exactly the per-MBH weight that
``draw_rate_weighted_hosts`` applies to catalog galaxies, so the dark and
in-catalog mass marginals are mutually self-consistent.

References:
    Gray et al. (2020), arXiv:1908.06050, Eq. (9) (completeness mixing weight),
        Appendix A.2 (completion term volume prior).
    Chen, Fishbach & Holz (2024), arXiv:2212.08694 (in/out-of-catalog mixture
        self-consistency).
    Babak et al. (2017), arXiv:1703.09722 (per-MBH EMRI rate; see
        :mod:`master_thesis_code.emri_rate`).
"""

import logging
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from master_thesis_code.constants import HOST_DRAW_Z_MAX, H
from master_thesis_code.emri_rate import R_eff_per_mbh, mbh_mass_function
from master_thesis_code.galaxy_catalogue.handler import HostGalaxy
from master_thesis_code.physical_relations import comoving_volume_element

_LOGGER = logging.getLogger()

# Catalog index flagging an out-of-catalog (dark) host. Real catalog rows carry
# a non-negative positional index (handler resets the index at load time), so
# -1 is an unambiguous dark-host sentinel.
DARK_HOST_CATALOG_INDEX: int = -1

# Peculiar-velocity redshift floor (GLADE+ convention; handler.py fills missing
# REDSHIFT_PECULIAR_VELOCITY_ERROR with this value). A dark host's z_error only
# sets up the injection bookkeeping -- the inference never reads a dark host's
# catalog redshift -- but a physically sensible floor keeps the object valid.
_DARK_HOST_Z_ERROR: float = 0.0015

# Fractional MBH-mass uncertainty assigned to a dark host. Like z_error this is
# never consumed by the inference for a dark host; it only makes the HostGalaxy
# a well-formed object (finite, positive error).
_DARK_HOST_FRACTIONAL_M_ERROR: float = 0.1

# Lower redshift bound of the population integrals. dVc/dz ∝ z^2 → 0 as z → 0,
# so the precise value is immaterial; 1e-6 matches the inference completion
# integral (bayesian_statistics.D(h), z_min = 1e-6).
_DEFAULT_Z_MIN: float = 1e-6

# Grid resolutions for the 1-D inverse-CDF samplers / the F quadrature. The
# completeness curve is piecewise linear, so a fine trapezoid grid is both
# accurate and robust (more so than Gauss-Legendre on a non-smooth integrand).
_DEFAULT_Z_GRID_POINTS: int = 4096
_DEFAULT_M_GRID_POINTS: int = 2048


class _CompletenessModel(Protocol):
    """Structural type for the catalog-completeness object ``f(z)``.

    :class:`~master_thesis_code.galaxy_catalogue.glade_completeness.GladeCatalogCompleteness`
    satisfies this protocol. Declaring it structurally lets the samplers accept
    any object exposing ``get_completeness_at_redshift`` (e.g. a constant-``f``
    stub in tests).
    """

    def get_completeness_at_redshift(
        self,
        z: float | npt.NDArray[np.floating[Any]],
        h: float = ...,
    ) -> float | npt.NDArray[np.floating[Any]]: ...

    def f_bar(
        self,
        z: float | npt.NDArray[np.floating[Any]],
        h: float = ...,
    ) -> float | npt.NDArray[np.floating[Any]]:
        """Sky-averaged completeness ``f_bar(z, h)`` (Change 5.4)."""
        ...


class _RateWeightedHostSource(Protocol):
    """Structural type for the in-catalog rate-weighted host source.

    :class:`~master_thesis_code.galaxy_catalogue.handler.GalaxyCatalogueHandler`
    satisfies this protocol. Only the members used by :func:`draw_mixture_hosts`
    are required, which keeps the mixture split testable with a lightweight
    synthetic catalog.
    """

    M_min: float
    M_max: float

    def draw_rate_weighted_hosts(
        self,
        number_of_hosts: int,
        rng: np.random.Generator,
        z_max: float = ...,
    ) -> list[HostGalaxy]: ...


def _redshift_population_weight(
    z_grid: npt.NDArray[np.float64], h: float
) -> npt.NDArray[np.float64]:
    r"""Source-frame redshift population weight ``(1/(1+z)) dVc/dz`` on a grid.

    The mass-integrated EMRI rate is redshift independent (``p0 = 1`` surrogate)
    and cancels in every ratio, so it is omitted here.

    Args:
        z_grid: Redshift grid (must be > 0).
        h: Dimensionless Hubble parameter for the comoving volume element.

    Returns:
        ``p_pop(z) ∝ dVc/dz / (1 + z)`` evaluated on ``z_grid`` (arbitrary
        overall normalisation).
    """
    dVc_dz = np.asarray(comoving_volume_element(z_grid, h=h), dtype=np.float64)
    return dVc_dz / (1.0 + z_grid)


def _inverse_cdf_sample(
    rng: np.random.Generator,
    grid: npt.NDArray[np.float64],
    density: npt.NDArray[np.float64],
    size: int,
) -> npt.NDArray[np.float64]:
    r"""Draw ``size`` samples on ``grid`` by inverse-CDF of an unnormalised density.

    The CDF is built by trapezoidal cumulative integration of ``density`` over
    the (possibly non-uniform) ``grid``; samples are obtained by linear
    interpolation of the inverse CDF at uniform deviates. This is the standard
    1-D inverse-transform sampler for an arbitrary positive density tabulated on
    a grid.

    Args:
        rng: Seeded generator (threads run-wide ``--seed`` reproducibility).
        grid: Strictly increasing sample-support grid, shape ``(n,)``.
        density: Non-negative unnormalised density at each grid node, shape
            ``(n,)``.
        size: Number of samples to draw.

    Returns:
        Array of ``size`` samples in ``[grid[0], grid[-1]]``.

    Raises:
        ValueError: If the total integrated density is non-positive (degenerate
            distribution).
    """
    density = np.clip(density, 0.0, None)
    spacing = np.diff(grid)
    increments = 0.5 * (density[1:] + density[:-1]) * spacing
    cdf = np.concatenate(([0.0], np.cumsum(increments)))
    total = float(cdf[-1])
    if not (total > 0.0):
        raise ValueError(
            "Inverse-CDF density integrates to a non-positive total "
            f"({total}); cannot draw samples."
        )
    cdf /= total
    deviates = rng.uniform(0.0, 1.0, size=size)
    samples: npt.NDArray[np.float64] = np.interp(deviates, cdf, grid)
    return samples


def compute_global_catalog_fraction(
    completeness: _CompletenessModel,
    h: float = H,
    z_min: float = _DEFAULT_Z_MIN,
    z_max: float = HOST_DRAW_Z_MAX,
    n_grid: int = _DEFAULT_Z_GRID_POINTS,
) -> float:
    r"""Global in-catalog fraction ``F`` of the injected EMRI population.

    .. math::

        F = \frac{\int_{z_\mathrm{min}}^{z_\mathrm{max}} f(z)\,p_\mathrm{pop}(z)\,dz}
                 {\int_{z_\mathrm{min}}^{z_\mathrm{max}} p_\mathrm{pop}(z)\,dz},
        \qquad p_\mathrm{pop}(z) \propto \frac{1}{1+z}\,\frac{dV_c}{dz},

    where ``f_bar(z, h) = completeness.f_bar(z, h)`` is the SKY-AVERAGED
    completeness the inference's ``beta_Gbar`` uses (Gray et al. 2020, Eq. 9;
    Gray-Messenger-Veitch 2022, Eq. 3). The mass-integrated EMRI rate is redshift independent and
    cancels between numerator and denominator, so it is omitted from
    ``p_pop(z)``. ``F`` is precomputed once per run and used as the per-event
    ``Bernoulli(F)`` in/out-of-catalog split probability.

    Limiting cases:
        * ``f(z) ≡ 1`` (fully complete) → ``F = 1`` → every host in-catalog
          (recovers the rate-weighted catalog draw exactly).
        * ``f(z) ≡ 0`` (fully incomplete) → ``F = 0`` → every host dark.
        * ``f(z) ≡ c`` (constant) → ``F = c``.

    Args:
        completeness: Catalog completeness model exposing
            ``get_completeness_at_redshift(z, h)``.
        h: Dimensionless Hubble parameter (injection cosmology).
        z_min: Lower redshift bound of the population integral.
        z_max: Upper redshift bound (default
            :data:`~master_thesis_code.constants.HOST_DRAW_Z_MAX`).
        n_grid: Number of trapezoid grid nodes.

    Returns:
        In-catalog fraction ``F`` in ``[0, 1]``.

    Raises:
        ValueError: If the population normalisation integral is non-positive.

    References:
        Gray et al. (2020), arXiv:1908.06050, Eq. (9).
        Chen, Fishbach & Holz (2024), arXiv:2212.08694.
    """
    z_grid = np.linspace(z_min, z_max, n_grid, dtype=np.float64)
    # Change 5.4: F is the direction-AND-population-averaged in-catalog fraction, so
    # it uses the sky-averaged completeness f_bar(z,h) = (1/Npix) sum_k f_k (Gray
    # 2020 Eq. 9; Gray-Messenger-Veitch 2022 Eq. 3). f_bar is the SAME object the
    # inference's beta_Gbar uses (bayesian_statistics), closing the sim/inference loop.
    f_z = np.asarray(completeness.f_bar(z_grid, h), dtype=np.float64)
    f_z = np.clip(f_z, 0.0, 1.0)
    p_pop = _redshift_population_weight(z_grid, h)

    numerator = float(np.trapezoid(f_z * p_pop, z_grid))
    denominator = float(np.trapezoid(p_pop, z_grid))
    if not (denominator > 0.0):
        raise ValueError(
            "Redshift population normalisation integral is non-positive "
            f"({denominator}); cannot form the global catalog fraction F."
        )
    fraction = numerator / denominator
    # Numerical guard: clamp to [0, 1] against trapezoid round-off.
    return float(min(max(fraction, 0.0), 1.0))


def _draw_dark_redshifts(
    rng: np.random.Generator,
    completeness: _CompletenessModel,
    h: float,
    z_min: float,
    z_max: float,
    size: int,
    n_grid: int = _DEFAULT_Z_GRID_POINTS,
) -> npt.NDArray[np.float64]:
    r"""Sample dark-host redshifts ``∝ (1 - f(z)) / (1 + z) * dVc/dz``.

    This is the missing-galaxy (out-of-catalog) redshift density: the source
    population prior ``p_pop(z) ∝ dVc/dz / (1+z)`` weighted by the
    *incompleteness* ``1 - f(z)``.
    """
    z_grid = np.linspace(z_min, z_max, n_grid, dtype=np.float64)
    f_z = np.asarray(completeness.get_completeness_at_redshift(z_grid, h), dtype=np.float64)
    f_z = np.clip(f_z, 0.0, 1.0)
    density = (1.0 - f_z) * _redshift_population_weight(z_grid, h)
    return _inverse_cdf_sample(rng, z_grid, density, size)


def _draw_dark_masses(
    rng: np.random.Generator,
    M_min: float,
    M_max: float,
    size: int,
    n_grid: int = _DEFAULT_M_GRID_POINTS,
) -> npt.NDArray[np.float64]:
    r"""Sample dark-host MBH masses from the per-dex EMRI-rate mass marginal.

    The marginal density in ``log10 M`` is
    ``mbh_mass_function(M) * R_eff_per_mbh(M)`` (Babak et al. 2017): the
    per-dex MBH number density times the effective per-MBH EMRI rate. Sampling
    is performed on a uniform ``log10 M`` grid because ``mbh_mass_function`` is
    ``dn/dlog10 M``; this matches the per-MBH weight applied to catalog galaxies
    in :meth:`GalaxyCatalogueHandler.draw_rate_weighted_hosts`, making the dark
    and in-catalog mass marginals self-consistent.
    """
    log_grid = np.linspace(np.log10(M_min), np.log10(M_max), n_grid, dtype=np.float64)
    mass_grid = 10.0**log_grid
    # Per-dex EMRI-rate weight (density in log10 M).
    density = np.asarray(mbh_mass_function(mass_grid), dtype=np.float64) * np.asarray(
        R_eff_per_mbh(mass_grid), dtype=np.float64
    )
    log_samples = _inverse_cdf_sample(rng, log_grid, density, size)
    return np.asarray(10.0**log_samples, dtype=np.float64)


def _draw_isotropic_sky(
    rng: np.random.Generator, size: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Sample isotropic sky positions ``(phiS, qS)`` on the sphere.

    ``phiS`` is uniform on ``[0, 2pi)`` and the polar angle ``qS`` follows
    ``qS = arccos(U)`` with ``U`` uniform on ``[-1, 1]`` (uniform in
    ``cos qS``), i.e. a uniform distribution on the unit sphere.
    """
    phiS = rng.uniform(0.0, 2.0 * np.pi, size=size)
    qS = np.arccos(rng.uniform(-1.0, 1.0, size=size))
    return phiS, qS


def draw_dark_hosts(
    number_of_hosts: int,
    rng: np.random.Generator,
    completeness: _CompletenessModel,
    M_min: float,
    M_max: float,
    h: float = H,
    z_min: float = _DEFAULT_Z_MIN,
    z_max: float = HOST_DRAW_Z_MAX,
    z_grid_points: int = _DEFAULT_Z_GRID_POINTS,
    m_grid_points: int = _DEFAULT_M_GRID_POINTS,
) -> list[HostGalaxy]:
    r"""Draw ``number_of_hosts`` out-of-catalog (dark) EMRI hosts.

    Each host's redshift, source-frame MBH mass, and sky position are drawn
    independently from the missing-galaxy population (``z`` and ``M`` factorise
    because the population density is mass-redshift separable under the
    ``p0 = 1`` surrogate):

    * ``z ∝ (1 - f(z)) / (1 + z) * dVc/dz`` over ``[z_min, z_max]`` (1-D
      inverse-CDF), where ``f(z)`` is the catalog completeness;
    * ``log10 M ∝ mbh_mass_function(M) * R_eff_per_mbh(M)`` over
      ``[M_min, M_max]`` (1-D inverse-CDF, per-dex marginal);
    * sky isotropic (``phiS`` uniform on ``[0, 2pi)``,
      ``qS = arccos(U[-1, 1])``).

    There is NO catalog snap and NO catalog lookup -- the returned values are
    the drawn ones. Each host is a
    :class:`~master_thesis_code.galaxy_catalogue.handler.HostGalaxy` with
    ``catalog_index = -1`` (:data:`DARK_HOST_CATALOG_INDEX`), a
    peculiar-velocity redshift-error floor, and a finite fractional mass error;
    these errors only make the object well-formed -- the inference never reads a
    dark host's catalog redshift.

    Args:
        number_of_hosts: Number of dark hosts to draw (``<= 0`` returns ``[]``).
        rng: Seeded generator (reproducible under ``--seed``).
        completeness: Catalog completeness model ``f(z)``.
        M_min: Lower MBH mass bound (solar masses).
        M_max: Upper MBH mass bound (solar masses).
        h: Dimensionless Hubble parameter (injection cosmology).
        z_min: Lower redshift bound.
        z_max: Upper redshift bound.
        z_grid_points: Redshift inverse-CDF grid resolution.
        m_grid_points: Mass inverse-CDF grid resolution.

    Returns:
        List of ``number_of_hosts`` dark :class:`HostGalaxy` objects.

    References:
        Gray et al. (2020), arXiv:1908.06050, Appendix A.2 (completion prior).
        Babak et al. (2017), arXiv:1703.09722 (per-MBH EMRI rate).
    """
    if number_of_hosts <= 0:
        return []

    redshifts = _draw_dark_redshifts(
        rng, completeness, h, z_min, z_max, number_of_hosts, z_grid_points
    )
    masses = _draw_dark_masses(rng, M_min, M_max, number_of_hosts, m_grid_points)
    phiS, qS = _draw_isotropic_sky(rng, number_of_hosts)

    hosts: list[HostGalaxy] = []
    for index in range(number_of_hosts):
        mass = float(masses[index])
        hosts.append(
            HostGalaxy.from_attributes(
                phiS=float(phiS[index]),
                qS=float(qS[index]),
                z=float(redshifts[index]),
                z_error=_DARK_HOST_Z_ERROR,
                M=mass,
                M_error=_DARK_HOST_FRACTIONAL_M_ERROR * mass,
                catalog_index=DARK_HOST_CATALOG_INDEX,
            )
        )
    return hosts


def draw_dark_host(
    rng: np.random.Generator,
    completeness: _CompletenessModel,
    M_min: float,
    M_max: float,
    h: float = H,
    z_min: float = _DEFAULT_Z_MIN,
    z_max: float = HOST_DRAW_Z_MAX,
    z_grid_points: int = _DEFAULT_Z_GRID_POINTS,
    m_grid_points: int = _DEFAULT_M_GRID_POINTS,
) -> HostGalaxy:
    """Draw a single out-of-catalog (dark) EMRI host.

    Thin singular wrapper over :func:`draw_dark_hosts`; see that function for the
    sampling distributions and dark-host representation.

    Returns:
        One dark :class:`HostGalaxy` with ``catalog_index = -1``.
    """
    return draw_dark_hosts(
        1,
        rng,
        completeness,
        M_min,
        M_max,
        h=h,
        z_min=z_min,
        z_max=z_max,
        z_grid_points=z_grid_points,
        m_grid_points=m_grid_points,
    )[0]


def draw_mixture_hosts(
    number_of_hosts: int,
    rng: np.random.Generator,
    galaxy_catalog: _RateWeightedHostSource,
    completeness: _CompletenessModel,
    global_catalog_fraction: float,
    h: float = H,
    z_min: float = _DEFAULT_Z_MIN,
    z_max: float = HOST_DRAW_Z_MAX,
) -> list[HostGalaxy]:
    r"""Build a host batch via a per-event ``Bernoulli(F)`` in/out-of-catalog split.

    Each of ``number_of_hosts`` hosts is independently in-catalog with
    probability ``F = global_catalog_fraction`` (drawn via
    ``galaxy_catalog.draw_rate_weighted_hosts``) or out-of-catalog/dark with
    probability ``1 - F`` (drawn via :func:`draw_dark_hosts`). All randomness
    uses the single seeded ``rng`` for reproducibility. The returned list keeps
    the random in/dark ordering of the Bernoulli mask; each host's
    :attr:`HostGalaxy.catalog_index` records its origin (``-1`` for dark), so
    the realised in-catalog fraction is recoverable downstream.

    Limiting cases:
        * ``F = 1`` → every host in-catalog (recovers the rate-weighted catalog
          draw exactly);
        * ``F = 0`` → every host dark.

    Args:
        number_of_hosts: Batch size (e.g. the simulation refill size).
        rng: Seeded generator.
        galaxy_catalog: In-catalog rate-weighted host source (provides
            ``M_min``/``M_max`` and ``draw_rate_weighted_hosts``).
        completeness: Catalog completeness model ``f(z)``.
        global_catalog_fraction: Precomputed ``F`` from
            :func:`compute_global_catalog_fraction`.
        h: Dimensionless Hubble parameter (injection cosmology).
        z_min: Lower redshift bound for the dark draw.
        z_max: Upper redshift bound for both draws.

    Returns:
        List of ``number_of_hosts`` :class:`HostGalaxy` objects in Bernoulli-mask
        order.

    References:
        Gray et al. (2020), arXiv:1908.06050, Eq. (9).
        Chen, Fishbach & Holz (2024), arXiv:2212.08694.
    """
    if number_of_hosts <= 0:
        return []

    in_catalog_mask: npt.NDArray[np.bool_] = rng.random(number_of_hosts) < global_catalog_fraction
    n_in_catalog = int(np.count_nonzero(in_catalog_mask))
    n_dark = number_of_hosts - n_in_catalog

    in_catalog_hosts: list[HostGalaxy] = (
        galaxy_catalog.draw_rate_weighted_hosts(n_in_catalog, rng=rng, z_max=z_max)
        if n_in_catalog > 0
        else []
    )
    dark_hosts: list[HostGalaxy] = (
        draw_dark_hosts(
            n_dark,
            rng,
            completeness,
            galaxy_catalog.M_min,
            galaxy_catalog.M_max,
            h=h,
            z_min=z_min,
            z_max=z_max,
        )
        if n_dark > 0
        else []
    )

    in_catalog_iter = iter(in_catalog_hosts)
    dark_iter = iter(dark_hosts)
    return [
        next(in_catalog_iter) if is_in_catalog else next(dark_iter)
        for is_in_catalog in in_catalog_mask
    ]
