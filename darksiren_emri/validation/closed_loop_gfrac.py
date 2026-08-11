r"""Closed-loop two-channel calibration harness for the completion-leg mass factor.

**What this instrument is.** A synthetic-universe calibration test that answers
one question and no other:

    Is the 2D (with-BH-mass) channel CALIBRATED when the universe actually
    follows the estimator's own generative assumptions?

It is the deciding measurement pre-registered in
``.planning/derivation-gfrac-20260805/GFRAC_DERIVATION_PACKAGE.md`` §9
(bands frozen in ``results/closed_loop_gfrac_20260805/PREREGISTRATION.md``),
as amended by ``.planning/derivation-gfrac-20260805/GATEB_REFUTATION_REPORT.md``
and ``docs/RESEARCH_CYCLE.md`` stage 4 amendment **A3**.

**Why it is a separate module from** :mod:`darksiren_emri.validation.pp_coverage`.
The two harnesses are different instrument classes and must not be merged:

* ``pp_coverage`` is deliberately **production-independent** — it re-implements
  the dark-siren estimator from scratch, and that independence is its whole
  scientific value (it can catch a coding error in the production estimator).
* This harness is deliberately **production-dependent** — it imports the
  estimator's *own* population and selection objects so that the loop **closes**:
  the universe is generated from exactly the density the estimator assumes, so
  any residual displacement of the recovered ``h`` is an estimator defect and
  cannot be a population mis-specification.

``pp_coverage.py`` is therefore not modified by this work.

**The closed-loop guarantee — production objects imported and used verbatim**

===========================================================  ==================================
object                                                        role in the loop
===========================================================  ==================================
:func:`~darksiren_emri.bayesian_inference.bayesian_statistics.dark_mass_density_per_mass`
                                                              source-mass draw AND ``g_i``;
                                                              carries the ``kappa_cap`` kink at
                                                              ``M = 1e5`` via
                                                              :func:`darksiren_emri.emri_rate.R_eff_per_mbh`
                                                              (GATEB amendment 2: the kink is
                                                              ACTIVE in the real data)
:func:`~darksiren_emri.bayesian_inference.bayesian_statistics.completion_mass_factor_g`
                                                              the 2D completion leg's
                                                              ``g_i(z;h)``, called verbatim and
                                                              **recomputed at every h**
                                                              (A3(i): never frozen)
:func:`~darksiren_emri.bayesian_inference.bayesian_statistics.precompute_phi_marginal_survival`
                                                              ``S_bar_phi(z;h)`` -> the shared
                                                              normalisation ``alpha(h)``
``SimulationDetectionProbability.detection_probability_with_bh_mass_interpolated``
                                                              ``S_4D`` — the **generator's**
                                                              detection rule (deterministic
                                                              horizon survival)
:func:`~darksiren_emri.physical_relations.dist_vectorized`,
:func:`~darksiren_emri.physical_relations.comoving_volume_element`
                                                              flat-LCDM ladder and
                                                              ``w_pop = dV_c/dz /(1+z)``
``_HOST_QUAD_N`` (50), ``_G_I_HERMITE_NODES`` (64)             identical quadrature convention
===========================================================  ==================================

It does **not** call :class:`~darksiren_emri.bayesian_inference.bayesian_statistics.BayesianStatistics`
— 200 seeds x 1500 events x 41 h is unaffordable through the full pipeline. The
completion-leg math is re-implemented compactly here (see
:func:`log_channel_posteriors`), mirroring ``single_host_likelihood``'s
completion branch; fidelity is enforced by *calling* the production ``g`` and
``S_bar_phi`` rather than re-coding them.

**Explicitly OUT OF SCOPE — stated rather than pretended.**
``docs/RESEARCH_CYCLE.md`` A3 lists three acceptance criteria for the stage-4
harness. This instrument satisfies (i) genuinely 2-channel with ``g``
recomputed per ``h`` and (ii) production ``N``. It does **NOT** satisfy
(iii) **multi-candidate host balls**: it is a single-host, catalogue-leg-off
harness, exactly as §9 specifies ("scored by the 2D completion leg alone
(catalogue leg off, which §6.6 shows is nearly the production configuration
anyway)"). A one-candidate-per-event harness structurally cannot exercise the
impostor-ball mechanism, and **no claim about that mechanism may be drawn from
this run**. The optional constant-completeness catalogue leg (``--f-cat``,
default ``0.0``) gives each catalogued event its *true* host redshift exactly,
which is a bright-siren caricature, not an impostor ball; it exists for
limiting-case tests only.

Neither GLADE+, real ``n(z)``, photo-z scatter, nor the completeness map enter
here. This harness cannot and does not speak to the photo-z rail mechanism
(``[[h0-railing-rootcause-photoz]]``).

CPU-only. No cupy import, direct or transitive.

References:
    Mandel, Farr & Gair (2019), arXiv:1809.02063, Eqs. (5)-(7) — the
        hierarchical selection identity the loop closes on.
    Gray et al. (2020), arXiv:1908.06050, Eq. (32) — completion numerator.
    Babak et al. (2017), arXiv:1703.09722, Eqs. (5), (23), (26)-(27), (30)-(31),
        (34) — the ``phi`` this harness draws masses from.
"""

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import roots_legendre
from scipy.stats import norm

from darksiren_emri.bayesian_inference.bayesian_statistics import (
    _G_I_HERMITE_NODES,
    _HOST_QUAD_N,
    completion_mass_factor_g,
    dark_mass_density_per_mass,
    precompute_phi_marginal_survival,
)
from darksiren_emri.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)
from darksiren_emri.constants import (
    M_SOURCE_FRAME_MAX,
    M_SOURCE_FRAME_MIN,
    SNR_THRESHOLD,
)
from darksiren_emri.physical_relations import (
    comoving_volume_element,
    dist_vectorized,
)

_LOGGER = logging.getLogger(__name__)

# ── Canonical production h grid (41 points) ──────────────────────────────────
# Read off results/run_20260804_postfix/iiib/diagnostics/event_likelihoods.csv:
# coarse 0.01 spacing on the wings, refined 0.005 spacing across the peak.
CANONICAL_H_GRID: tuple[float, ...] = (
    0.600, 0.610, 0.620, 0.630, 0.640, 0.650,
    0.655, 0.660, 0.665, 0.670, 0.675, 0.680, 0.685, 0.690, 0.695,
    0.700, 0.705, 0.710, 0.715, 0.720, 0.725, 0.730, 0.735, 0.740, 0.745,
    0.750, 0.755, 0.760, 0.765, 0.770, 0.775, 0.780, 0.785, 0.790,
    0.800, 0.810, 0.820, 0.830, 0.840, 0.850, 0.860,
)  # fmt: skip

DEFAULT_INJECTION_DIR: str = (
    "results/campaign51_20260728/realistic_20260729/gate_b_20260730/injection_pool_mix200k_20260728"
)
DEFAULT_CRB_CSV: str = (
    "results/run_20260804_postfix/iiib/diagnostics/prepared_cramer_rao_bounds.csv"
)
DEFAULT_BASE_SEED: int = 20260805
DEFAULT_N_SEEDS: int = 200
DEFAULT_N_EVENTS: int = 1500

_Z_TABLE_POINTS: int = 4000
_M_TABLE_POINTS: int = 3000
_SIGMA_WINDOW: float = 4.0  # the estimator's +-4 sigma z window (production)


@dataclass(frozen=True)
class ClosedLoopConfig:
    """Frozen configuration of one closed-loop sweep.

    Attributes:
        injection_data_dir: Injection pool defining ``S_4D`` (the generator's
            detection rule and the estimator's selection normalisation).
        crb_reference_csv: Production prepared-CRB CSV whose fractional
            ``(sigma_dL/d_L, sigma_Mz/M_z, rho)`` triples are bootstrap-resampled
            as the observation error model.
        n_events: Detected events per synthetic universe (production venue N).
        h_true: The injected Hubble parameter.
        h_grid: The h grid both channels are evaluated on.
        f_cat: Fraction of detected events whose host redshift is known exactly
            (bright-siren caricature; 0.0 = §9's completion-leg-only registered
            configuration).
        numerator_pdet: ``"off"`` reproduces the shipped estimator (no selection
            factor inside the numerator); ``"on"`` is the GATEB N-2 diagnostic
            variant, NOT part of the registered readout.
        snr_threshold: SNR threshold defining the detection horizons.
        n_quad: Gauss-Legendre order of the completion quadrature.
        n_hermite: Gauss-Hermite order inside ``g_i``.
    """

    injection_data_dir: str = DEFAULT_INJECTION_DIR
    crb_reference_csv: str = DEFAULT_CRB_CSV
    n_events: int = DEFAULT_N_EVENTS
    h_true: float = 0.73
    h_grid: tuple[float, ...] = CANONICAL_H_GRID
    f_cat: float = 0.0
    numerator_pdet: str = "off"
    snr_threshold: float = SNR_THRESHOLD
    n_quad: int = _HOST_QUAD_N
    n_hermite: int = _G_I_HERMITE_NODES


@dataclass
class ClosedLoopContext:
    """Per-process shared, seed-independent tables (built once, forked to workers).

    Every field is derived from a production object; nothing here is a
    re-implementation.
    """

    config: ClosedLoopConfig
    detection: SimulationDetectionProbability
    sigma_triples: npt.NDArray[np.float64]  # (n_rows, 3): sigma_dL, sigma_Mz, rho
    z_max_true: float
    gen_z_nodes: npt.NDArray[np.float64]
    gen_z_cdf: npt.NDArray[np.float64]
    gen_log10_M_nodes: npt.NDArray[np.float64]
    gen_M_cdf: npt.NDArray[np.float64]
    # per-h tables, aligned with config.h_grid
    z_of_dl_tables: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = field(
        default_factory=list
    )
    log_alpha: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    s_phi_tables: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = field(
        default_factory=list
    )
    gl_nodes: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    gl_weights: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )


@dataclass
class SyntheticUniverse:
    """One realised, detected, noisy event set."""

    z_true: npt.NDArray[np.float64]
    M_true: npt.NDArray[np.float64]
    d_L_true: npt.NDArray[np.float64]
    d_L_obs: npt.NDArray[np.float64]
    M_z_obs: npt.NDArray[np.float64]
    sigma_dL: npt.NDArray[np.float64]
    sigma_Mz: npt.NDArray[np.float64]
    rho: npt.NDArray[np.float64]
    in_catalogue: npt.NDArray[np.bool_]
    n_drawn: int


# ── Error model ──────────────────────────────────────────────────────────────


def load_sigma_triples(csv_path: str) -> npt.NDArray[np.float64]:
    r"""Fractional ``(sigma_dL/d_L, sigma_Mz/M_z, rho)`` triples from a CRB CSV.

    These are exactly the ``(2,2), (3,3), (2,3)`` entries of the production
    ``cov_4d`` (``bayesian_statistics.py`` "Build 4D covariance"), i.e. the
    fractional-coordinate 2x2 block the 2D channel conditions on.

    Args:
        csv_path: Path to a ``prepared_cramer_rao_bounds.csv``.

    Returns:
        Array of shape ``(n_rows, 3)``: ``sigma_dL/d_L``, ``sigma_Mz/M_z``,
        ``rho``. Rows with a non-finite or non-positive sigma are dropped.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    d_L = np.asarray(df["luminosity_distance"], dtype=np.float64)
    M = np.asarray(df["M"], dtype=np.float64)
    s_d = (
        np.sqrt(
            np.asarray(df["delta_luminosity_distance_delta_luminosity_distance"], dtype=np.float64)
        )
        / d_L
    )
    s_m = np.sqrt(np.asarray(df["delta_M_delta_M"], dtype=np.float64)) / M
    cov = np.asarray(df["delta_luminosity_distance_delta_M"], dtype=np.float64) / d_L / M
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = cov / (s_d * s_m)
    ok = np.isfinite(s_d) & np.isfinite(s_m) & np.isfinite(rho)
    ok &= (s_d > 0.0) & (s_m > 0.0) & (np.abs(rho) < 1.0)
    triples = np.column_stack([s_d[ok], s_m[ok], rho[ok]]).astype(np.float64)
    if triples.shape[0] == 0:
        raise ValueError(f"No usable CRB error triples in '{csv_path}'")
    return triples


# ── Context construction ─────────────────────────────────────────────────────


def _w_pop(z: npt.NDArray[np.float64], h: float) -> npt.NDArray[np.float64]:
    r"""The estimator's population measure ``w_pop(z;h) = (dV_c/dz)/(1+z)``.

    Per-steradian; the isotropic sky factor is a per-event, h-independent
    constant and cancels from the posterior shape.

    Args:
        z: Redshifts (any shape).
        h: Dimensionless Hubble parameter.

    Returns:
        ``w_pop`` at ``z``, same shape.
    """
    dVc = np.asarray(comoving_volume_element(z, h=h), dtype=np.float64)
    return np.asarray(dVc / (1.0 + np.asarray(z, dtype=np.float64)), dtype=np.float64)


def _z_of_dl_table(
    h: float, z_max: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Monotone ``(d_L, z)`` table for fast inversion of the distance ladder.

    Replaces per-event ``dist_to_redshift`` root-finding (123k fsolve calls per
    seed would dominate the runtime); the ladder is smooth and monotone, so
    linear interpolation on a 4000-node table is exact to ~1e-8 in z.

    Args:
        h: Dimensionless Hubble parameter.
        z_max: Upper redshift of the table.

    Returns:
        ``(d_L_nodes, z_nodes)``, both strictly increasing.
    """
    z_nodes = np.linspace(1e-8, z_max, _Z_TABLE_POINTS, dtype=np.float64)
    d_L_nodes = np.asarray(dist_vectorized(z_nodes, h=h), dtype=np.float64)
    return d_L_nodes, z_nodes


def build_context(config: ClosedLoopConfig) -> ClosedLoopContext:
    """Build every seed-independent table exactly once.

    All population/selection content comes from production objects; this
    function only tabulates them.

    Args:
        config: The frozen sweep configuration.

    Returns:
        A :class:`ClosedLoopContext` ready for :func:`run_seed`.
    """
    detection = SimulationDetectionProbability(
        injection_data_dir=config.injection_data_dir,
        snr_threshold=config.snr_threshold,
    )
    sigma_triples = load_sigma_triples(config.crb_reference_csv)

    h_grid = list(config.h_grid)
    # z_max(h) exactly as the estimator defines it (get_dl_max -> ladder inverse).
    z_max_per_h: list[float] = []
    z_of_dl_tables: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
    for h in h_grid:
        dl_max = detection.get_dl_max(h)
        # Invert with a table rather than fsolve: build generously, then trim.
        d_L_nodes, z_nodes = _z_of_dl_table(h, 5.0)
        z_max = float(np.interp(dl_max, d_L_nodes, z_nodes))
        z_max_per_h.append(z_max)
        z_of_dl_tables.append(_z_of_dl_table(h, z_max))

    dl_max_true = detection.get_dl_max(config.h_true)
    d_L_nodes_t, z_nodes_t = _z_of_dl_table(config.h_true, 5.0)
    z_max_true = float(np.interp(dl_max_true, d_L_nodes_t, z_nodes_t))

    # ── generator tables ────────────────────────────────────────────────────
    gen_z_nodes = np.linspace(1e-6, z_max_true, _Z_TABLE_POINTS, dtype=np.float64)
    w = _w_pop(gen_z_nodes, config.h_true)
    gen_z_cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(gen_z_nodes))])
    gen_z_cdf /= gen_z_cdf[-1]

    gen_log10_M = np.linspace(
        math.log10(M_SOURCE_FRAME_MIN), math.log10(M_SOURCE_FRAME_MAX), _M_TABLE_POINTS
    ).astype(np.float64)
    M_nodes = 10.0**gen_log10_M
    # phi is a density in M; the density in log10 M is phi(M) * M * ln10.
    phi_log10 = dark_mass_density_per_mass(M_nodes) * M_nodes * math.log(10.0)
    gen_M_cdf = np.concatenate(
        [[0.0], np.cumsum(0.5 * (phi_log10[1:] + phi_log10[:-1]) * np.diff(gen_log10_M))]
    )
    gen_M_cdf /= gen_M_cdf[-1]

    # ── alpha(h) = INTEGRAL w_pop(z;h) S_bar_phi(z;h) dz (the shared normalisation) ──
    s_phi_table = precompute_phi_marginal_survival(h_grid, detection)
    log_alpha = np.empty(len(h_grid), dtype=np.float64)
    s_phi_tables: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
    for i, h in enumerate(h_grid):
        z_grid, s_phi = s_phi_table[h]
        alpha = float(np.trapezoid(_w_pop(z_grid, h) * s_phi, z_grid))
        if not (alpha > 0.0):
            raise ValueError(f"alpha(h={h}) = {alpha} is non-positive")
        log_alpha[i] = math.log(alpha)
        s_phi_tables.append(
            (np.asarray(z_grid, dtype=np.float64), np.asarray(s_phi, dtype=np.float64))
        )

    gl_nodes, gl_weights = roots_legendre(config.n_quad)

    return ClosedLoopContext(
        config=config,
        detection=detection,
        sigma_triples=sigma_triples,
        z_max_true=z_max_true,
        gen_z_nodes=gen_z_nodes,
        gen_z_cdf=gen_z_cdf,
        gen_log10_M_nodes=gen_log10_M,
        gen_M_cdf=gen_M_cdf,
        z_of_dl_tables=z_of_dl_tables,
        log_alpha=log_alpha,
        s_phi_tables=s_phi_tables,
        gl_nodes=np.asarray(gl_nodes, dtype=np.float64),
        gl_weights=np.asarray(gl_weights, dtype=np.float64),
    )


# ── Generator ────────────────────────────────────────────────────────────────


def draw_universe(
    ctx: ClosedLoopContext,
    rng: np.random.Generator,
    *,
    batch: int = 4096,
) -> SyntheticUniverse:
    r"""Draw one synthetic universe: population -> selection -> noise.

    The three steps use, in order, the estimator's ``w_pop``, the estimator's
    ``phi`` (kink included), and the production ``S_4D``. Detection is a
    Bernoulli draw with probability ``S_4D(d_L, M_z)``: the survival object is
    already marginalised over the nuisance parameters (inclination, spins,
    phases), so at fixed ``(d_L, M_z)`` detection is genuinely random — this is
    the generative statement the estimator's ``alpha(h)`` integrates.

    Args:
        ctx: The shared context.
        rng: Seeded generator.
        batch: Proposal batch size for the accept/reject loop.

    Returns:
        A :class:`SyntheticUniverse` with exactly ``ctx.config.n_events``
        detections.
    """
    cfg = ctx.config
    z_acc: list[npt.NDArray[np.float64]] = []
    M_acc: list[npt.NDArray[np.float64]] = []
    n_have = 0
    n_drawn = 0
    while n_have < cfg.n_events:
        u_z = rng.random(batch)
        z = np.interp(u_z, ctx.gen_z_cdf, ctx.gen_z_nodes)
        u_m = rng.random(batch)
        M = 10.0 ** np.interp(u_m, ctx.gen_M_cdf, ctx.gen_log10_M_nodes)
        d_L = np.asarray(dist_vectorized(z, h=cfg.h_true), dtype=np.float64)
        M_z = M * (1.0 + z)
        p_det = np.asarray(
            ctx.detection.detection_probability_with_bh_mass_interpolated(
                d_L, M_z, 0.0, 0.0, h=cfg.h_true
            ),
            dtype=np.float64,
        )
        keep = rng.random(batch) < p_det
        n_drawn += batch
        if np.any(keep):
            z_acc.append(z[keep])
            M_acc.append(M[keep])
            n_have += int(keep.sum())
    z_true = np.concatenate(z_acc)[: cfg.n_events]
    M_true = np.concatenate(M_acc)[: cfg.n_events]
    d_L_true = np.asarray(dist_vectorized(z_true, h=cfg.h_true), dtype=np.float64)
    M_z_true = M_true * (1.0 + z_true)

    idx = rng.integers(0, ctx.sigma_triples.shape[0], size=cfg.n_events)
    sigma_dL = ctx.sigma_triples[idx, 0]
    sigma_Mz = ctx.sigma_triples[idx, 1]
    rho = ctx.sigma_triples[idx, 2]

    # Correlated fractional noise on (d_L, M_z).
    e1 = rng.standard_normal(cfg.n_events)
    e2 = rng.standard_normal(cfg.n_events)
    frac_d = sigma_dL * e1
    frac_m = sigma_Mz * (rho * e1 + np.sqrt(np.maximum(1.0 - rho**2, 0.0)) * e2)
    d_L_obs = d_L_true * (1.0 + frac_d)
    M_z_obs = M_z_true * (1.0 + frac_m)

    in_catalogue = rng.random(cfg.n_events) < cfg.f_cat

    return SyntheticUniverse(
        z_true=z_true,
        M_true=M_true,
        d_L_true=d_L_true,
        d_L_obs=d_L_obs,
        M_z_obs=M_z_obs,
        sigma_dL=sigma_dL,
        sigma_Mz=sigma_Mz,
        rho=rho,
        in_catalogue=in_catalogue,
        n_drawn=n_drawn,
    )


# ── Estimator ────────────────────────────────────────────────────────────────


def _g_at_nodes(
    ctx: ClosedLoopContext,
    universe: SyntheticUniverse,
    z_nodes: npt.NDArray[np.float64],
    d_L_frac: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Evaluate the production ``g_i(z;h)`` for every event at its own nodes.

    Calls :func:`completion_mass_factor_g` verbatim, once per event, with the
    event's own ``(det_M_z, proj, sigma_cond)`` — the same Bishop (2006)
    Eqs. 2.81-2.82 conditional the production code builds from ``cov_4d``.

    Args:
        ctx: Shared context (supplies the Gauss-Hermite order).
        universe: The event set (supplies ``M_z^obs`` and the 2x2 block).
        z_nodes: ``(n_events, n_quad)`` quadrature redshifts.
        d_L_frac: ``(n_events, n_quad)`` values of ``d_L(z;h)/d_L^obs_i``.

    Returns:
        ``g_i`` at the nodes, shape ``(n_events, n_quad)``, units ``1/x_M``.
    """
    n = z_nodes.shape[0]
    s_dd = universe.sigma_dL**2
    s_dm = universe.rho * universe.sigma_dL * universe.sigma_Mz
    s_mm = universe.sigma_Mz**2
    proj = np.where(s_dd > 0.0, s_dm / np.maximum(s_dd, 1e-300), 0.0)
    sigma_cond = np.sqrt(np.maximum(s_mm - proj * s_dm, 1e-30))
    out = np.empty_like(z_nodes)
    for i in range(n):
        out[i] = completion_mass_factor_g(
            z_nodes[i],
            d_L_frac[i],
            float(universe.M_z_obs[i]),
            float(proj[i]),
            float(sigma_cond[i]),
            n_hermite=ctx.config.n_hermite,
        )
    return out


def log_channel_posteriors(
    ctx: ClosedLoopContext,
    universe: SyntheticUniverse,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Unnormalised log posteriors of both channels on the h grid.

    Mirrors ``single_host_likelihood``'s completion branch (``B_num`` /
    ``B_num_wbh``) with ``g`` recomputed at every ``h`` (A3(i)):

    .. math::

        B^{1D}_i(h) &= \int_{z_-}^{z_+}\! w_\mathrm{pop}(z;h)\,
            \mathcal{N}\!\bigl(d_L(z;h)/d_L^{obs}_i;1,\sigma_{d_L,i}\bigr)\,\mathrm{d}z \\
        B^{2D}_i(h) &= \int_{z_-}^{z_+}\! w_\mathrm{pop}(z;h)\,
            \mathcal{N}(\cdot)\,g_i(z;h)\,\mathrm{d}z \\
        \ln P(h) &= \sum_i \bigl[\ln B_i(h) - \ln\alpha(h)\bigr]

    with ``alpha(h) = INTEGRAL w_pop S_bar_phi dz`` shared by both channels (the
    derivation package's (T2): ``alpha`` is a property of the population and the
    detector, not of which observables the analyst uses), and the z window
    ``d_L^obs (1 -+ 4 sigma)`` capped at ``z_max(h)`` — the production domain.

    The isotropic sky factor ``sin(theta)/(4 pi)`` is omitted: it is a per-event,
    h-independent constant that cancels from the posterior shape.

    Args:
        ctx: Shared context.
        universe: The event set.

    Returns:
        ``(ln_post_1d, ln_post_2d, sum_dlog_g_dh)``: two arrays of length
        ``len(h_grid)``, plus the grid-differenced
        ``sum_i d ln g_frac_i/dh`` at ``h_true`` (a scalar array of size 1) for
        the §9 MIXED-branch comparison against the production +243.5 nats/h.
    """
    cfg = ctx.config
    n_h = len(cfg.h_grid)
    ln1 = np.zeros(n_h, dtype=np.float64)
    ln2 = np.zeros(n_h, dtype=np.float64)
    ln_gfrac = np.zeros(n_h, dtype=np.float64)

    x = ctx.gl_nodes
    w_gl = ctx.gl_weights
    d_obs = universe.d_L_obs
    sig = universe.sigma_dL

    for k, h in enumerate(cfg.h_grid):
        d_L_nodes, z_tab = ctx.z_of_dl_tables[k]
        z_hi = np.interp(d_obs * (1.0 + _SIGMA_WINDOW * sig), d_L_nodes, z_tab)
        z_lo = np.interp(d_obs * (1.0 - _SIGMA_WINDOW * sig), d_L_nodes, z_tab)
        z_lo = np.maximum(z_lo, 1e-6)
        z_hi = np.minimum(z_hi, z_tab[-1])
        valid = z_hi > z_lo
        half = 0.5 * (z_hi - z_lo)
        mid = 0.5 * (z_hi + z_lo)
        z_nodes = mid[:, None] + half[:, None] * x[None, :]  # (n, n_quad)

        d_L_n = np.asarray(dist_vectorized(z_nodes.reshape(-1), h=h), dtype=np.float64).reshape(
            z_nodes.shape
        )
        d_L_frac = d_L_n / d_obs[:, None]
        p_gw = norm.pdf(d_L_frac, loc=1.0, scale=sig[:, None])
        base = _w_pop(z_nodes, h) * p_gw

        if cfg.numerator_pdet == "on":
            z_s, s_phi = ctx.s_phi_tables[k]
            base_1d = base * np.interp(z_nodes, z_s, s_phi)
        else:
            base_1d = base

        g = _g_at_nodes(ctx, universe, z_nodes, d_L_frac)
        if cfg.numerator_pdet == "on":
            # sigma_cond ~ 1e-8 in production, so g_i is a point evaluation at
            # mu_cond; inserting S_4D at that point is exact to that order.
            proj = np.where(
                universe.sigma_dL > 0.0,
                universe.rho * universe.sigma_Mz / universe.sigma_dL,
                0.0,
            )
            mu_cond = 1.0 + proj[:, None] * (d_L_frac - 1.0)
            s4 = np.asarray(
                ctx.detection.detection_probability_with_bh_mass_interpolated(
                    d_L_n.reshape(-1),
                    (mu_cond * universe.M_z_obs[:, None]).reshape(-1),
                    0.0,
                    0.0,
                    h=h,
                ),
                dtype=np.float64,
            ).reshape(z_nodes.shape)
            base_2d = base * g * s4
        else:
            base_2d = base * g

        b1 = half * (base_1d @ w_gl)
        b2 = half * (base_2d @ w_gl)

        if cfg.f_cat > 0.0:
            # Bright-siren caricature leg: the host redshift is known exactly.
            cat = universe.in_catalogue
            if np.any(cat):
                z_g = universe.z_true[cat]
                d_g = np.asarray(dist_vectorized(z_g, h=h), dtype=np.float64)
                frac_g = d_g / d_obs[cat]
                p_g = norm.pdf(frac_g, loc=1.0, scale=sig[cat])
                w_g = _w_pop(z_g, h) * p_g
                b1[cat] = w_g
                g_g = _g_at_nodes(
                    ctx,
                    _subset(universe, cat),
                    z_g[:, None],
                    frac_g[:, None],
                )[:, 0]
                b2[cat] = w_g * g_g

        ok = valid & (b1 > 0.0) & (b2 > 0.0) & np.isfinite(b1) & np.isfinite(b2)
        ln1[k] = float(np.sum(np.log(b1[ok]))) - float(ok.sum()) * ctx.log_alpha[k]
        ln2[k] = float(np.sum(np.log(b2[ok]))) - float(ok.sum()) * ctx.log_alpha[k]
        ln_gfrac[k] = float(np.sum(np.log(b2[ok] / b1[ok])))

    # Central difference of sum_i ln g_frac,i at h_true.
    h_arr = np.asarray(cfg.h_grid, dtype=np.float64)
    i_true = int(np.argmin(np.abs(h_arr - cfg.h_true)))
    lo = max(i_true - 1, 0)
    hi = min(i_true + 1, n_h - 1)
    slope = (ln_gfrac[hi] - ln_gfrac[lo]) / (h_arr[hi] - h_arr[lo])
    return ln1, ln2, np.asarray([slope], dtype=np.float64)


def _subset(universe: SyntheticUniverse, mask: npt.NDArray[np.bool_]) -> SyntheticUniverse:
    """Return the sub-universe selected by ``mask`` (used by the catalogue leg).

    Args:
        universe: The full event set.
        mask: Boolean selector of length ``n_events``.

    Returns:
        A :class:`SyntheticUniverse` restricted to the masked events.
    """
    return SyntheticUniverse(
        z_true=universe.z_true[mask],
        M_true=universe.M_true[mask],
        d_L_true=universe.d_L_true[mask],
        d_L_obs=universe.d_L_obs[mask],
        M_z_obs=universe.M_z_obs[mask],
        sigma_dL=universe.sigma_dL[mask],
        sigma_Mz=universe.sigma_Mz[mask],
        rho=universe.rho[mask],
        in_catalogue=universe.in_catalogue[mask],
        n_drawn=universe.n_drawn,
    )


# ── Readout ──────────────────────────────────────────────────────────────────


def posterior_readout(
    h_grid: npt.NDArray[np.float64],
    ln_post: npt.NDArray[np.float64],
) -> dict[str, float]:
    """Grid MAP, parabolic-refined MAP, posterior mean, and rail flags.

    Args:
        h_grid: The h grid.
        ln_post: Unnormalised log posterior on that grid.

    Returns:
        ``{"map", "map_refined", "mean", "railed_low", "railed_high"}``.
    """
    i = int(np.argmax(ln_post))
    p = np.exp(ln_post - ln_post[i])
    norm_p = float(np.trapezoid(p, h_grid))
    mean = float(np.trapezoid(p * h_grid, h_grid) / norm_p) if norm_p > 0.0 else float("nan")
    map_grid = float(h_grid[i])
    map_ref = map_grid
    if 0 < i < len(h_grid) - 1:
        # Parabolic vertex through the three points (unequal spacing safe).
        x0, x1, x2 = h_grid[i - 1], h_grid[i], h_grid[i + 1]
        y0, y1, y2 = ln_post[i - 1], ln_post[i], ln_post[i + 1]
        d1 = (y1 - y0) / (x1 - x0)
        d2 = (y2 - y1) / (x2 - x1)
        curv = (d2 - d1) / (0.5 * (x2 - x0))
        if curv < 0.0:
            map_ref = float(0.5 * (x0 + x1) - d1 / curv)
            map_ref = float(np.clip(map_ref, h_grid[0], h_grid[-1]))
    return {
        "map": map_grid,
        "map_refined": map_ref,
        "mean": mean,
        "railed_low": float(i == 0),
        "railed_high": float(i == len(h_grid) - 1),
    }


# ── Per-seed driver ──────────────────────────────────────────────────────────

_CTX: ClosedLoopContext | None = None


def _worker_init(config: ClosedLoopConfig) -> None:
    """Build (or inherit) the shared context in a worker process."""
    global _CTX
    if _CTX is None:
        _CTX = build_context(config)


def run_seed(seed: int, ctx: ClosedLoopContext | None = None) -> dict[str, Any]:
    """Run one synthetic universe end to end.

    Args:
        seed: The universe's random seed.
        ctx: Shared context; falls back to the process-global one.

    Returns:
        A JSON-serialisable per-seed record.
    """
    context = ctx if ctx is not None else _CTX
    if context is None:
        raise RuntimeError("closed-loop context not initialised")
    rng = np.random.default_rng(seed)
    universe = draw_universe(context, rng)
    ln1, ln2, slope = log_channel_posteriors(context, universe)
    h_arr = np.asarray(context.config.h_grid, dtype=np.float64)
    r1 = posterior_readout(h_arr, ln1)
    r2 = posterior_readout(h_arr, ln2)
    return {
        "seed": int(seed),
        "n_events": int(context.config.n_events),
        "n_proposed": int(universe.n_drawn),
        "z_median": float(np.median(universe.z_true)),
        "M_source_median": float(np.median(universe.M_true)),
        "frac_below_kink": float(np.mean(universe.M_true < 1.0e5)),
        "map_1d": r1["map"],
        "map_2d": r2["map"],
        "map_1d_refined": r1["map_refined"],
        "map_2d_refined": r2["map_refined"],
        "mean_1d": r1["mean"],
        "mean_2d": r2["mean"],
        "railed_low_1d": r1["railed_low"],
        "railed_high_1d": r1["railed_high"],
        "railed_low_2d": r2["railed_low"],
        "railed_high_2d": r2["railed_high"],
        "sum_dlog_gfrac_dh": float(slope[0]),
        "ln_post_1d": [float(v) for v in ln1],
        "ln_post_2d": [float(v) for v in ln2],
    }


# ── Sweep + aggregation ──────────────────────────────────────────────────────


def _quantiles(values: npt.NDArray[np.float64]) -> dict[str, float]:
    """Return the §5 quantile set of a sample.

    Args:
        values: The sample.

    Returns:
        Quantiles keyed by percentage.
    """
    qs = np.percentile(values, [0, 5, 25, 50, 75, 95, 100])
    return {f"q{p}": float(v) for p, v in zip([0, 5, 25, 50, 75, 95, 100], qs, strict=True)}


def score_against_bands(maps_2d: npt.NDArray[np.float64], h_true: float) -> dict[str, Any]:
    """Score the 2D MAP distribution against the frozen §9 bands.

    CONFIRM if ``|<MAP> - h_true| <= 0.010``; REFUTE if ``<MAP> - h_true >=
    +0.030``; MIXED otherwise (a first-class outcome — §9: "report the split;
    do not force a branch").

    Args:
        maps_2d: Per-seed 2D grid-argmax MAPs.
        h_true: The injected value.

    Returns:
        ``{"verdict", "displacement", "mc_error"}``.
    """
    mean = float(np.mean(maps_2d))
    disp = mean - h_true
    mc = float(np.std(maps_2d, ddof=1) / math.sqrt(len(maps_2d))) if len(maps_2d) > 1 else 0.0
    if disp >= 0.030:
        verdict = "REFUTE"
    elif abs(disp) <= 0.010:
        verdict = "CONFIRM"
    else:
        verdict = "MIXED"
    return {"verdict": verdict, "displacement": disp, "mc_error": mc, "mean_map_2d": mean}


def aggregate(records: list[dict[str, Any]], config: ClosedLoopConfig) -> dict[str, Any]:
    """Aggregate per-seed records into the pre-registered readout.

    Args:
        records: Per-seed records from :func:`run_seed`.
        config: The sweep configuration.

    Returns:
        The aggregate block of the results JSON.
    """
    m1 = np.asarray([r["map_1d"] for r in records], dtype=np.float64)
    m2 = np.asarray([r["map_2d"] for r in records], dtype=np.float64)
    m1r = np.asarray([r["map_1d_refined"] for r in records], dtype=np.float64)
    m2r = np.asarray([r["map_2d_refined"] for r in records], dtype=np.float64)
    slopes = np.asarray([r["sum_dlog_gfrac_dh"] for r in records], dtype=np.float64)
    n = len(records)
    return {
        "n_seeds": n,
        "map_1d": {
            "mean": float(np.mean(m1)),
            "displacement": float(np.mean(m1) - config.h_true),
            "mc_error": float(np.std(m1, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
            "quantiles": _quantiles(m1),
            "railed_low_frac": float(np.mean([r["railed_low_1d"] for r in records])),
            "railed_high_frac": float(np.mean([r["railed_high_1d"] for r in records])),
            "mean_refined": float(np.mean(m1r)),
        },
        "map_2d": {
            "mean": float(np.mean(m2)),
            "displacement": float(np.mean(m2) - config.h_true),
            "mc_error": float(np.std(m2, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
            "quantiles": _quantiles(m2),
            "railed_low_frac": float(np.mean([r["railed_low_2d"] for r in records])),
            "railed_high_frac": float(np.mean([r["railed_high_2d"] for r in records])),
            "mean_refined": float(np.mean(m2r)),
        },
        "posterior_mean_1d": float(np.mean([r["mean_1d"] for r in records])),
        "posterior_mean_2d": float(np.mean([r["mean_2d"] for r in records])),
        "sum_dlog_gfrac_dh": {
            "mean": float(np.mean(slopes)),
            "quantiles": _quantiles(slopes),
            "production_reference_nats_per_h": 243.5,
        },
        "scoring": score_against_bands(m2, config.h_true),
    }


def run_sweep(
    config: ClosedLoopConfig,
    seeds: list[int],
    workers: int,
) -> dict[str, Any]:
    """Run the sweep and assemble the results document.

    Args:
        config: The frozen configuration.
        seeds: Seeds to run.
        workers: Worker processes (``<= 1`` runs in-process).

    Returns:
        The full results dict (written to JSON by :func:`main`).
    """
    if workers > 1:
        ctx_mp = mp.get_context("fork")
        with ctx_mp.Pool(processes=workers, initializer=_worker_init, initargs=(config,)) as pool:
            records = pool.map(run_seed, seeds, chunksize=1)
    else:
        _worker_init(config)
        records = [run_seed(s) for s in seeds]
    return {
        "instrument": "closed_loop_gfrac",
        "preregistration": "results/closed_loop_gfrac_20260805/PREREGISTRATION.md",
        "git_commit": _git_commit(),
        "config": asdict(config),
        "seeds": seeds,
        "aggregate": aggregate(records, config),
        "per_seed": records,
    }


def _git_commit() -> str:
    """Return the current git commit, or ``"unknown"``."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    p.add_argument("--smoke", action="store_true", help="3 seeds, reduced N — sanity run")
    p.add_argument("--sweep", action="store_true", help="full pre-registered sweep")
    p.add_argument("--out", type=str, default="closed_loop_results.json")
    p.add_argument("--workers", type=int, default=max(mp.cpu_count() - 2, 1))
    p.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    p.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    p.add_argument("--n-events", type=int, default=DEFAULT_N_EVENTS)
    p.add_argument("--f-cat", type=float, default=0.0)
    p.add_argument("--numerator-pdet", choices=("off", "on"), default="off")
    p.add_argument("--injection-dir", type=str, default=DEFAULT_INJECTION_DIR)
    p.add_argument("--crb-csv", type=str, default=DEFAULT_CRB_CSV)
    p.add_argument("--log-level", type=str, default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    n_seeds = 3 if args.smoke and not args.sweep else args.n_seeds
    n_events = min(args.n_events, 300) if args.smoke and not args.sweep else args.n_events
    config = ClosedLoopConfig(
        injection_data_dir=args.injection_dir,
        crb_reference_csv=args.crb_csv,
        n_events=n_events,
        f_cat=args.f_cat,
        numerator_pdet=args.numerator_pdet,
    )
    seeds = [args.base_seed + i for i in range(n_seeds)]
    workers = 1 if args.smoke and not args.sweep else args.workers
    results = run_sweep(config, seeds, workers)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    agg = results["aggregate"]
    _LOGGER.info(
        "closed-loop: n=%d  1D MAP mean=%.4f (disp %+0.4f)  2D MAP mean=%.4f "
        "(disp %+0.4f, MC %.4f)  verdict=%s",
        agg["n_seeds"],
        agg["map_1d"]["mean"],
        agg["map_1d"]["displacement"],
        agg["map_2d"]["mean"],
        agg["map_2d"]["displacement"],
        agg["scoring"]["mc_error"],
        agg["scoring"]["verdict"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
