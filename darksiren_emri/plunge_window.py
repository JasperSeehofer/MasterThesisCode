"""Plunge-window initial-condition draw (author-ratified convention, 2026-07-28).

Replaces the historical snapshot draw p0 ~ U[10, 16] (a 2023 adoption of the
few Pn5AAK *input-domain* as an astrophysical prior) with the plunge-window
convention of the population model this pipeline samples: Babak et al. (2017),
arXiv:1703.09722 SS III C/D draw plunge times uniformly within the observation
window, so an "event" is an EMRI that PLUNGES during the mission.

Convention (docs/derivations/plunge_window_initial_conditions.md):
    t_plunge ~ U[0, T_mission]   (observer/detector-frame years)
    p0 = root of  t_insp(p0; M_z, mu, a, e0, x0) = t_plunge
computed with few's PN5 trajectory (the SAME trajectory that generates the
Pn5AAK waveform, so the realized plunge time is self-consistent by
construction). All masses are DETECTOR-frame (M_z = M_source*(1+z)); the PN5
trajectory clock is then automatically observer time, matching T_mission.

The domain rule is the trajectory's own: p0 >= p_sep(a, e0, x0) + 0.05 (few
ODEBase.separatrix_buffer_dist). No snapshot [10, 16] clamp survives; there is
no upper p0 clamp either (low-M_z events that plunge in-window legitimately
start at p0 >> 16, where the PN5 flux is MORE accurate). C0 continuity: at
t_plunge -> 0 the draw approaches the separatrix start continuously; at
t_plunge -> T it matches the longest in-window inspiral.

References:
    Babak et al. (2017), arXiv:1703.09722, SS III C/D (plunge-window population).
    Colpi et al. (2024), arXiv:2402.07571 (T_mission = 4.5 yr science operations).
    Peters (1964), Phys. Rev. 136, B1224 (upper-bracket seed, Eq. 5.10).
"""

import logging
from typing import Any

import numpy as np

from darksiren_emri.datamodels.parameter_space import ParameterSpace

_LOGGER = logging.getLogger()

# Root-find tolerances (MEASURED, results/campaign51_20260728/plunge_window/):
# xtol = 1e-3 on p0 and integrator err = 1e-8 give a realized plunge-time
# accuracy |t_end - t_plunge|/t_plunge <= 2.8e-4 when the waveform is generated
# with few's default integrator tolerance (1e-11), at a median draw cost of
# ~0.33 s/call on the dev CPU. Both are numerical knobs, not physics: the
# convention is exact up to these documented tolerances.
_P0_XTOL: float = 1e-3
_TRAJ_ERR: float = 1e-8

_traj_module: Any = None


def _get_trajectory() -> Any:
    """Lazy singleton PN5 trajectory module (the Pn5AAK generating trajectory)."""
    global _traj_module
    if _traj_module is None:
        from few.trajectory.inspiral import EMRIInspiral  # noqa: PLC0415
        from few.trajectory.ode import PN5  # noqa: PLC0415

        _traj_module = EMRIInspiral(func=PN5)
    return _traj_module


def _peters_p_upper_seed(t_plunge_years: float, M: float, mu: float) -> float:
    """Peters (1964) circular quadrupole inspiral-time inversion, used ONLY to
    seed the upper root bracket (the root-find itself uses the full PN5 flux).

    Eq. (5.10) of Peters (1964) with a = p*G*M/c^2, e = 0:
        t_insp = (5/256) * p^4 * (G M / c^3) * (M / mu)
    inverted for p. Dimensionless p; t in observer years. The true PN5/Kerr
    inspiral from the same p is FASTER than Peters quadrupole (stronger
    strong-field fluxes), so 2x this seed is a safe upper bracket; the bracket
    is verified and geometrically enlarged below if ever insufficient.
    """
    from few.utils.constants import MTSUN_SI, YRSID_SI  # noqa: PLC0415

    return float((256.0 / 5.0 * (t_plunge_years * YRSID_SI) / (MTSUN_SI * M) * (mu / M)) ** 0.25)


def draw_plunge_window_initial_conditions(
    parameter_space: ParameterSpace,
    rng: np.random.Generator,
    mission_duration_years: float,
) -> float:
    """Draw t_plunge ~ U[0, T_mission] and set p0 so the inspiral plunges then.

    Must be called AFTER the detector-frame mass M_z has been set on
    ``parameter_space`` (set_host_galaxy_parameters or the injection-loop
    M assignment) — the time-to-plunge map depends on M_z.

    Sets ``parameter_space.p0.value`` and ``parameter_space.t_plunge_yr``;
    returns the drawn t_plunge in years.

    Raises whatever the few trajectory/brentq machinery raises for pathological
    parameter combinations (ValueError "Brent...", ZeroDivisionError from the
    Y->xI map near polar orbits) — the same exception classes, handled by the
    same per-event skip handlers, as the subsequent waveform generation.
    """
    from few.utils.constants import YRSID_SI  # noqa: PLC0415
    from few.utils.utility import get_p_at_t  # noqa: PLC0415

    traj = _get_trajectory()

    M_z = parameter_space.M.value
    mu = parameter_space.mu.value
    a = parameter_space.a.value
    e0 = parameter_space.e0.value
    x0 = parameter_space.x0.value

    # t_plunge ~ U[0, T]: Babak et al. (2017) arXiv:1703.09722 SS III D
    # ("Plunge times are taken to be uniform"), window = the nominal science
    # observation span T_mission (Colpi et al. 2024, arXiv:2402.07571).
    t_plunge = float(rng.uniform(0.0, mission_duration_years))

    # few 2.0 wart: ODEBase.min_p ignores its `a` kwarg and reads self.a, which
    # is only set by add_fixed_parameters (normally called inside a trajectory
    # run). Set it explicitly so the separatrix lower bound is well-defined.
    traj.func.add_fixed_parameters(M_z, mu, a)
    p_lo = float(traj.func.min_p(e0, x0))  # p_sep(a, e0, x0) + 0.05 buffer

    # Upper bracket: 2x Peters seed, verified against the actual PN5 trajectory
    # (t(p_up) >= t_plunge) and enlarged geometrically if needed. few's default
    # max_p is +inf, which brentq cannot take.
    p_up = max(2.0 * _peters_p_upper_seed(t_plunge, M_z, mu), p_lo + 1.0)
    for _ in range(30):
        out = traj(M_z, mu, a, p_up, e0, x0, T=1.05 * max(t_plunge, 1e-6), err=_TRAJ_ERR)
        if out[0][-1] >= t_plunge * YRSID_SI:
            break
        p_up *= 1.5
    else:
        raise ValueError(
            f"Brent root solver does not converge: no p0 upper bracket found "
            f"(M_z={M_z:.3e}, t_plunge={t_plunge:.3f} yr, p_up={p_up:.1f})."
        )

    p0 = float(
        get_p_at_t(
            traj,
            t_plunge,
            [M_z, mu, a, e0, x0],
            bounds=[p_lo, p_up],
            xtol=_P0_XTOL,
            traj_kwargs={"err": _TRAJ_ERR},
        )
    )

    # Domain rule by construction: brentq confines p0 to [p_sep + 0.05, p_up].
    if not (p_lo - 1e-9 <= p0 <= p_up + 1e-9):
        raise ValueError(f"plunge-window p0={p0} escaped bracket [{p_lo}, {p_up}]")

    parameter_space.p0.value = p0
    parameter_space.t_plunge_yr = t_plunge
    _LOGGER.debug(
        "Plunge-window draw: t_plunge=%.4f yr -> p0=%.4f (M_z=%.3e, e0=%.3f, x0=%.3f)",
        t_plunge,
        p0,
        M_z,
        e0,
        x0,
    )
    return t_plunge
