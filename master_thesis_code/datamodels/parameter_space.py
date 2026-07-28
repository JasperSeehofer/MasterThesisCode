from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from master_thesis_code.constants import M_SOURCE_FRAME_MAX, M_SOURCE_FRAME_MIN
from master_thesis_code.exceptions import ParameterOutOfBoundsError
from master_thesis_code.galaxy_catalogue.handler import HostGalaxy
from master_thesis_code.physical_relations import dist, redshifted_mass


def uniform(lower_limit: float, upper_limit: float, rng: np.random.Generator) -> float:
    return float(rng.uniform(lower_limit, upper_limit))


def log_uniform(lower_limit: float, upper_limit: float, rng: np.random.Generator) -> float:
    lower_limit = np.log10(lower_limit)
    upper_limit = np.log10(upper_limit)
    uniform_log = uniform(lower_limit, upper_limit, rng)
    return float(10**uniform_log)


def polar_angle_distribution(
    lower_limit: float, upper_limit: float, rng: np.random.Generator
) -> float:
    return float(np.arccos(rng.uniform(-1.0, 1.0)))


@dataclass
class Parameter:
    """Main class for parameters."""

    symbol: str
    unit: str
    lower_limit: float
    upper_limit: float
    value: float = 0.0
    derivative_epsilon: float = 1e-6
    is_fixed: bool = False
    randomize_by_distribution: Callable[[float, float, np.random.Generator], float] = uniform


@dataclass
class ParameterSpace:
    """
    Dataclass to manage the parameter space of a simulation.
    """

    # Per-parameter derivative_epsilon: Vallisneri (2008) arXiv:gr-qc/0703086 Eq. (A11)
    # Optimal step size for 5-point stencil (p=4): h* ≈ ε_machine^(1/4) × |x| ≈ 3.3e-4 × |x|
    # Each epsilon is chosen to be ~3e-4 × (representative parameter value).
    # Vallisneri (2008) arXiv:gr-qc/0703086 Eq. (A11) — per-param epsilon

    M: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="M",
            unit="solar masses",
            # Babak et al. (2017) arXiv:1703.09722 valid band — single source of
            # truth in constants.py (issue #51). Model1CrossCheck lifts the upper
            # limit to the detector-frame image M_SOURCE_FRAME_MAX*(1+max_redshift)
            # at construction (parameter_space.M holds M_z in production).
            lower_limit=M_SOURCE_FRAME_MIN,
            upper_limit=M_SOURCE_FRAME_MAX,
            randomize_by_distribution=log_uniform,
            # A tiny ABSOLUTE step (1 M_sun) on a ~1e5-1e6 M_sun mass: the EMRI phase is
            # extremely M-sensitive, so the finite-difference step must keep ∂Φ/∂M·(2ε)
            # well under a radian — the Vallisneri ε_mach^(1/4)·|x| heuristic (~60-100
            # M_sun here) assumes f varies on scale |x|, which is false for an
            # oscillatory waveform. (Prior comment mis-stated the log-uniform midpoint
            # as ~3e3 M_sun; the [1e4,1e7] geometric midpoint is 10^5.5 ≈ 3e5. Any change
            # to this value needs a Fisher step-halving convergence study + /physics-change
            # — review PHY-09.)
            derivative_epsilon=1.0,
        )
    )  # mass of the MBH (massive black hole) in solar masses

    mu: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="mu",
            unit="solar masses",
            lower_limit=1,
            upper_limit=1e2,
            derivative_epsilon=0.01,  # ~3e-4 × 30 SM (midpoint ~30 SM)
        )
    )  # mass of the CO (compact object) in solar masses
    a: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="a",
            unit="dimensionless",
            lower_limit=0.0,
            upper_limit=1,
            derivative_epsilon=1e-3,  # ~3e-4 × 0.5 (dimensionless [0, 1])
        )
    )  # dimensionless spin of the MBH
    p0: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="p0",
            unit="dimensionless",
            # [PHYSICS] SNAPSHOT-mode bounds only (--snapshot_ics archaeology
            # path): [10, 16] is few's documented Pn5AAK input domain, adopted
            # as a prior in 2023 and RETIRED as the production convention on
            # 2026-07-28 (HIGHM_AUDIT.md item 1). Production draws p0 via the
            # plunge-window convention (plunge_window.py: t_plunge ~ U[0, T],
            # p0 = root of t_insp(p0) = t_plunge), which OVERWRITES this value
            # after the detector-frame mass is set; the plunge-window domain is
            # p0 >= p_sep(a, e0, x0) + 0.05 with no upper clamp
            # (docs/derivations/plunge_window_initial_conditions.md).
            # Babak et al. (2017), arXiv:1703.09722, SS III C/D.
            lower_limit=10.0,
            upper_limit=16.0,
            derivative_epsilon=1e-3,  # ~3e-4 × 13 (midpoint; dimensionless semi-latus rectum)
        )
    )  # Kepler-orbit parameter: separation (semi-latus rectum, units of G M / c^2)
    e0: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="e0",
            unit="dimensionless",
            lower_limit=0.05,
            upper_limit=0.7,
            derivative_epsilon=1e-4,  # ~3e-4 × 0.35 ≈ 1e-4 (dimensionless [0.05, 0.7])
        )
    )  # Kepler-orbit parameter: eccentricity
    x0: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="x0",
            unit="dimensionless",
            lower_limit=-1.0,
            upper_limit=1.0,
            derivative_epsilon=1e-4,  # symmetric around 0; use half-range scale 1e-4
        )
    )  # Kepler-orbit parameter: x_I0=cosI (I is the inclination)
    luminosity_distance: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="luminosity_distance",
            unit="Gpc",
            lower_limit=0.0,
            # dist(HOST_DRAW_Z_MAX=1.5, h=H_MIN/100=0.60) = 13.0015 Gpc — the
            # campaign population reach at the lowest grid h. Model1CrossCheck
            # recomputes this exactly; the literal here protects bare
            # ParameterSpace() constructions from a sub-horizon cap (the old
            # 7 Gpc default silently rejected z >~ 0.9 events).
            upper_limit=13.1,
            derivative_epsilon=1e-4,  # ~3e-4 × 1 Gpc ≈ 3e-4; use 1e-4 Gpc (= 0.1 Mpc)
        )
    )  # luminosity distance
    qS: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="qS",
            unit="radian",
            lower_limit=0.0,
            upper_limit=np.pi,
            randomize_by_distribution=polar_angle_distribution,
            derivative_epsilon=1e-4,  # ~3e-4 × π/2 ≈ 5e-4; use 1e-4 rad
        )
    )  # Sky location polar angle in ecliptic coordinates.
    phiS: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="phiS",
            unit="radian",
            lower_limit=0.0,
            upper_limit=2 * np.pi,
            derivative_epsilon=1e-4,  # ~3e-4 × π ≈ 1e-3; use 1e-4 rad
        )
    )  # Sky location azimuthal angle in ecliptic coordinates.
    qK: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="qK",
            unit="radian",
            lower_limit=0.0,
            upper_limit=np.pi,
            randomize_by_distribution=polar_angle_distribution,
            derivative_epsilon=1e-4,  # same as qS
        )
    )  # Initial BH spin polar angle in ecliptic coordinates.
    phiK: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="phiK",
            unit="radian",
            lower_limit=0.0,
            upper_limit=2 * np.pi,
            derivative_epsilon=1e-4,  # same as phiS
        )
    )  # Initial BH spin azimuthal angle in ecliptic coordinates.
    Phi_phi0: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="Phi_phi0",
            unit="radian",
            lower_limit=0.0,
            upper_limit=2 * np.pi,
            derivative_epsilon=1e-4,  # ~3e-4 × π ≈ 1e-3; use 1e-4 rad
        )
    )  # initial azimuthal phase
    Phi_theta0: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="Phi_theta0",
            unit="radian",
            lower_limit=0.0,
            upper_limit=2 * np.pi,
            derivative_epsilon=1e-4,  # same as Phi_phi0
        )
    )  # initial polar phase
    Phi_r0: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="Phi_r0",
            unit="radian",
            lower_limit=0.0,
            upper_limit=2 * np.pi,
            derivative_epsilon=1e-4,  # same as Phi_phi0
        )
    )  # initial radial phase

    # Plunge-window bookkeeping (NOT one of the 14 waveform parameters): the
    # drawn plunge time in observer years, set by
    # plunge_window.draw_plunge_window_initial_conditions and recorded in the
    # injection/CRB CSVs for provenance. NaN in snapshot mode (--snapshot_ics)
    # and before any plunge-window draw.
    t_plunge_yr: float = float("nan")

    def randomize_parameter(self, parameter: Parameter, rng: np.random.Generator) -> None:
        parameter.value = parameter.randomize_by_distribution(
            parameter.lower_limit, parameter.upper_limit, rng
        )
        setattr(self, parameter.symbol, parameter)

    def randomize_parameters(self, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()
        for parameter in vars(self).values():
            if isinstance(parameter, Parameter) and not parameter.is_fixed:
                self.randomize_parameter(parameter=parameter, rng=rng)
        # Reset plunge-window provenance: stale t_plunge from a previous event
        # must not survive into a snapshot-mode row or a failed re-draw.
        self.t_plunge_yr = float("nan")
        self._check_separatrix_guard()

    def _check_separatrix_guard(self) -> None:
        """Reject draws too close to the plunge separatrix (G9 gate guard).

        The Schwarzschild separatrix is p_sep(e) = 6 + 2e (conservative for
        prograde Kerr, where p_sep is smaller); FEW waveforms are unphysical for
        p0 near/below it. Current bounds (p0 >= 10, e0 <= 0.7) satisfy this with
        margin >= 2.6, so this never fires today -- it protects against future
        bound changes silently entering the plunge regime.
        Stein & Warburton (2020), arXiv:1912.07609 (separatrix).

        NOTE (plunge-window convention, 2026-07-28): this guard checks the
        SNAPSHOT draw only (it runs inside randomize_parameters, before the
        plunge-window overwrite). A plunge-window p0 legitimately lies below
        6 + 2 e0 + 0.5 for prograde Kerr orbits; its validity is enforced by
        construction instead (brentq bracket [p_sep_Kerr + 0.05, p_up] in
        plunge_window.draw_plunge_window_initial_conditions).
        """
        p_sep = 6.0 + 2.0 * self.e0.value
        if self.p0.value < p_sep + 0.5:
            raise ParameterOutOfBoundsError(
                f"p0={self.p0.value:.3f} within 0.5 of the separatrix "
                f"p_sep(e0={self.e0.value:.3f})={p_sep:.3f}; adjust parameter bounds."
            )

    def set_host_galaxy_parameters(self, host_galaxy: HostGalaxy, h: float) -> None:
        # FEW (Pn5AAKWaveform) expects the DETECTOR-FRAME (redshifted) mass
        # M_z = M_source·(1+z) in the M slot; redshift enters the mass, luminosity
        # distance enters the amplitude.  The GLADE-derived catalog mass
        # host_galaxy.M is source-frame, so it must be lifted by (1+z) before the
        # waveform call so the stored CRB "M" column genuinely holds M_z (which the
        # Bayesian inference assumes: det.M = M_z, bayesian_statistics.py:1335).
        # Maggiore (2008) GW Vol. 1 §4.1.4; Babak et al. (2017) arXiv:1703.09722.
        self.M.value = redshifted_mass(host_galaxy.M, host_galaxy.z)  # M_z = M·(1+z)
        self.phiS.value = host_galaxy.phiS
        self.qS.value = host_galaxy.qS
        # h_inj threaded explicitly per PE-01 (Phase 37); dark siren PE self-consistency at h_inj
        # (Gray et al. 2020, Laghi et al. 2021). h has no default — calling without h raises TypeError (SC-2).
        self.luminosity_distance.value = dist(host_galaxy.z, h=h)  # SC-1: h_inj threaded

    def _parameters_to_dict(self) -> dict:
        return {
            "M": self.M.value,
            "mu": self.mu.value,
            "a": self.a.value,
            "p0": self.p0.value,
            "e0": self.e0.value,
            "x0": self.x0.value,
            "luminosity_distance": self.luminosity_distance.value,
            "qS": self.qS.value,
            "phiS": self.phiS.value,
            "qK": self.qK.value,
            "phiK": self.phiK.value,
            "Phi_phi0": self.Phi_phi0.value,
            "Phi_theta0": self.Phi_theta0.value,
            "Phi_r0": self.Phi_r0.value,
        }
