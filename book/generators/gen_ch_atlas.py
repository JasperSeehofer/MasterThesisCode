"""Generator for the Atlas formula explorer (cross-chapter reference page).

Produces two data files:

``book/site/data/atlas_curves.json``
    Six small curve families backing the Atlas's "formula explorer" widgets,
    each independently provenanced as REAL (a repo function evaluated
    read-only) or TOY (a closed-form transcription used for illustration
    where a live read-only call is not practical from this generator):

    1. ``psd`` — REAL. ``LisaTdiConfiguration.power_spectral_density_a_channel``
       (``master_thesis_code/LISA_configuration.py``), A-channel PSD with and
       without the galactic confusion foreground. The confusion-alone curve
       is the *difference* of two live calls (``include_confusion_noise=True``
       minus ``False``) rather than a private-method call, so it is exactly
       the term the total curve actually adds — no separate formula to drift
       out of sync with the class.

    2. ``dlz`` — REAL. ``physical_relations.dist`` (flat LCDM luminosity
       distance), evaluated on a linear z grid for four h values (the true
       h_true = 0.73 plus three illustrative values spanning the book's H0
       tension range).

    3. ``zkernel`` — REAL. The host-z volume-deconvolved kernel shape used by
       Pipeline B's production host-z marginalisation
       (``bayesian_inference/bayesian_statistics.py``, see ``_w_pop_eff`` /
       the ``p_g(z) = N(z; z_g, sigma_z) * w_pop(z) / Z_g`` comment block
       above line 4794), recomputed here directly from
       ``physical_relations.comoving_volume_element`` (the same function the
       production kernel calls) rather than imported from the inference
       module, which requires the galaxy catalogue and multiprocessing
       machinery to construct. ``w_pop(z) = dVc/dz/dOmega / (1+z)``; the
       "bare" curve is the plain Gaussian N(z; z_g, sigma_z) with no volume
       weighting (the pre-C7 delta-kernel comparison point).

    4. ``rv`` — REAL. Reines & Volonteri (2015) stellar-to-BH mass relation
       constants (alpha=7.45, beta=1.05, intrinsic scatter 0.24 dex) as
       docstringed in ``galaxy_catalogue/handler.py`` lines 33-44:
       ``log10(M_BH/Msun) = 7.45 + 1.05 * log10(M_*/1e11 Msun)``, transcribed
       directly in the documented log10/1e11-pivot form (the module's own
       runtime code re-expresses this in natural-log units pivoted at a
       different mass scale for its error-propagation machinery; the two are
       algebraically identical -- this generator uses the docstring form
       because it is what the module itself asserts as the physical
       statement, and it is trivially checked against the module's alpha,
       beta constants below).

    5. ``pdet`` — REAL. ``validation.pp_coverage.detection_probability``, the
       P-P/coverage harness's smooth Malmquist erfc detection-probability
       model, called directly (pure numpy/scipy, importable CPU-only with no
       simulation-stack dependency).

    6. ``stencil`` — TOY. A closed-form finite-difference-error demonstration
       on the analytic test function f(x) = sin(x) at x=1 (exact derivative
       cos(1) is known in closed form), illustrating the O(eps) forward-
       difference vs O(eps^4) five-point-stencil error scaling that motivates
       ``parameter_estimation.py``'s ``use_five_point_stencil=True`` default
       (Vallisneri 2008, arXiv:gr-qc/0703086, Appendix A). Not a call into
       the Fisher-matrix code itself (that requires a full EMRI waveform
       evaluation); the scaling law demonstrated is the same one that
       motivates the stencil choice there.

Determinism: no RNG anywhere. Every REAL section calls the repo's own
functions on a fixed grid; the TOY sections are closed-form. Read-only
outside ``book/``.

``book/site/data/atlas_journey.json``
    Per-event "journey" data for three real events from the run's own
    Cramer-Rao table (889 loud/in-catalogue, 606 medium/dark, 555
    faint/dark — the same three carried by ``ch06_fisher.json``), each
    section independently provenanced REAL or TOY:

    * ``waveform`` — REAL. ``few.waveform.GenerateEMRIWaveform`` with
      ``waveform_class="Pn5AAKWaveform"`` (the production wave generator,
      ``waveform_generator.py`` ``WaveGeneratorType.PN5_AAK``), called
      directly with the event's own 14 CRB parameters and ``T`` = 48 hours
      (so the segment is the exact head of the production trajectory), pre
      the LISA TDI response (which needs ``fastlisaresponse`` + orbits and
      is not called here).
    * ``ftrack`` — REAL. ``few.trajectory.inspiral.EMRIInspiral`` with
      ``func=PN5`` (the ODE the Pn5AAK waveform module uses internally),
      over the full mission span or until plunge, whichever is sooner;
      ``f_phi`` from a cubic spline of ``Phi_phi(t)``, reported as
      ``2*f_phi`` (the dominant quadrupole harmonic).
    * ``spectrum`` — REAL. Welch ASD (``scipy.signal.welch``) of the 48-hour
      waveform snapshot above, ``hc = sqrt(f*S_h)``.
    * ``corr14`` — REAL. Correlation matrix of the event's own stored 14x14
      Cramer-Rao covariance (``delta_X_delta_Y`` columns), the same table
      ``gen_ch06.py`` assembles for its conditioning statistics.
    * ``csv_row`` — REAL. The event's own CRB row headline columns.
    * ``ball`` — REAL. Catalogue candidate count within the event's own
      production BallTree search radius (``ch06_fisher.json``
      ``radius_full_rad``) and redshift window (``z_window``), over the
      same committed reduced catalogue and the same two-stage angular-cut +
      ``handler`` ecliptic-transform + mass/redshift-prune pipeline
      ``gen_ch06.py`` uses to build its sky patches (imported and reused
      directly, not re-derived).
    * ``geolike`` — TOY. A closed-form single-host geometric H0
      reconstruction: ``lnL(h) = log INT N(d_L(z;h); d_L_meas, sigma_eff) *
      w(z) dz`` over the event's own z_window, ``w(z)`` the same comoving-
      volume prior as ``zkernel`` above; ``sigma_eff`` folds in the host
      photo-z error (sigma_z=0.035, propagated through dd_L/dz at h=0.73)
      only for the in-catalogue event.
    * ``kernelz`` — REAL. The same bare-vs-volume host-z kernel shape as
      ``zkernel`` above, recentered on this event's own z_window.

    Plus a ``shared`` block: the injection-pool (logM, z) 2D histogram and
    z marginals, reused directly from the already-committed
    ``ch09_factory.json`` (not re-derived from the raw pool).

Determinism: no RNG anywhere. Read-only outside ``book/``.

Run as::

    /home/jasper/Repositories/MasterThesisCode/.venv/bin/python \\
        book/generators/gen_ch_atlas.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
# gen_ch06.py lives in this same directory (the script's own dir is on
# sys.path[0] when run directly); imported for its sky-patch pipeline
# (_load_patch / _prepare_patch / _polar_to_cartesian / _resolve), reused
# verbatim below rather than re-derived.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_ch06 as _ch06  # noqa: E402

# galaxy_catalogue.handler's alpha/beta/sigma_int module constants, imported
# read-only so the transcribed docstring formula below is checked against the
# module's own numbers rather than typed independently.
from master_thesis_code.constants import LISA_MISSION_DURATION_YEARS  # noqa: E402
from master_thesis_code.galaxy_catalogue import handler as _rv_handler  # noqa: E402
from master_thesis_code.LISA_configuration import LisaTdiConfiguration  # noqa: E402
from master_thesis_code.physical_relations import (  # noqa: E402
    comoving_volume_element,
    dist,
)
from master_thesis_code.validation.pp_coverage import (  # noqa: E402
    D50_GPC,
    W_PDET_GPC,
    detection_probability,
)

_RV_ALPHA_LN = _rv_handler.alpha
_RV_BETA = _rv_handler.beta
_RV_SIGMA_LN = _rv_handler.sigma_int

OUT_DIR = Path(__file__).resolve().parent.parent / "site" / "data"
OUT_FILE = OUT_DIR / "atlas_curves.json"
OUT_JOURNEY = OUT_DIR / "atlas_journey.json"
CH06_FISHER_FILE = OUT_DIR / "ch06_fisher.json"
CH09_FACTORY_FILE = OUT_DIR / "ch09_factory.json"

N_PTS = 120

# The three events the journey widget offers -- 889 (loud, in-catalogue, the
# book's running example), 606 (medium, dark, the Ch 5 counterpart) and 555
# (faint, dark, population-median SNR). Subset of gen_ch06.py's EVENT_IDS,
# same CRB row indices (row position in prepared_cramer_rao_bounds.csv).
JOURNEY_EVENT_IDS = [889, 606, 555]
JOURNEY_ROLES = {889: "loud", 606: "medium", 555: "faint"}

SEC_PER_YEAR = 365.25 * 24.0 * 3600.0
HOST_PHOTOZ_SIGMA_Z = 0.035  # same value zkernel/ch06 use throughout


def _r(x: Any, sig: int = 6) -> float:
    """Round to `sig` significant digits — JSON size hygiene."""
    v = float(x)
    if v == 0.0 or not np.isfinite(v):
        return v
    return float(np.round(v, sig - 1 - int(np.floor(np.log10(abs(v))))))


def _rlist(arr: np.ndarray, sig: int = 6) -> list[float]:
    return [_r(v, sig) for v in np.asarray(arr).ravel()]


def _fail(msg: str) -> None:
    raise SystemExit(f"gen_ch_atlas: HARD GATE FAILED — {msg}")


# ---------------------------------------------------------------------------
# 1. LISA A-channel PSD, with/without confusion noise (REAL)
# ---------------------------------------------------------------------------
def build_psd() -> dict[str, Any]:
    f = np.logspace(-5, 0, N_PTS)
    cfg_total = LisaTdiConfiguration(include_confusion_noise=True)
    cfg_bare = LisaTdiConfiguration(include_confusion_noise=False)
    sn_total = cfg_total.power_spectral_density_a_channel(f)
    sn_no_confusion = cfg_bare.power_spectral_density_a_channel(f)
    confusion = sn_total - sn_no_confusion

    if not np.all(np.isfinite(sn_total)) or np.any(sn_total <= 0):
        _fail("PSD not finite/positive everywhere")
    if np.any(confusion < -1e-40):  # allow tiny negative roundoff, not real negativity
        _fail("confusion term negative beyond roundoff")

    return {
        "f": _rlist(f, 8),
        "sn_total": _rlist(sn_total, 8),
        "sn_no_confusion": _rlist(sn_no_confusion, 8),
        "confusion": _rlist(np.clip(confusion, 0.0, None), 8),
        "t_obs_years": float(cfg_total.t_obs_years),
        "prov": "real: LISA_configuration.py LisaTdiConfiguration.power_spectral_density_a_channel "
        "(A/E channel; confusion = total - instrumental-only, both live calls)",
    }


# ---------------------------------------------------------------------------
# 2. d_L(z; h) flat LCDM (REAL)
# ---------------------------------------------------------------------------
def build_dlz() -> dict[str, Any]:
    from master_thesis_code.constants import OMEGA_DE, OMEGA_M

    z = np.linspace(0.0, 1.5, N_PTS)
    h_values = [0.60, 0.704, 0.73, 0.86]
    curves = {}
    for h in h_values:
        curves[f"{h:g}"] = [_r(dist(zi, h=h, Omega_m=OMEGA_M, Omega_de=OMEGA_DE), 7) for zi in z]

    if abs(dist(0.0)) > 1e-10:
        _fail(f"dist(0.0) = {dist(0.0)!r} not ~0")

    return {
        "z": _rlist(z, 6),
        "curves": curves,
        "omega_m": float(OMEGA_M),
        "omega_de": float(OMEGA_DE),
        "units": "Gpc",
        "prov": "real: physical_relations.dist (flat LCDM, fiducial Omega_m/Omega_de)",
    }


# ---------------------------------------------------------------------------
# 3. Host-z kernel: bare Gaussian vs volume-deconvolved (REAL)
# ---------------------------------------------------------------------------
def build_zkernel() -> dict[str, Any]:
    from master_thesis_code.constants import OMEGA_DE, OMEGA_M
    from master_thesis_code.constants import H as H_TRUE

    z_g = 0.40
    sigma_z = 0.035
    z = np.linspace(0.2, 0.6, N_PTS)

    bare_unnorm = np.exp(-0.5 * ((z - z_g) / sigma_z) ** 2) / (sigma_z * np.sqrt(2.0 * np.pi))
    # normalize the bare Gaussian over the display grid (it is already
    # ~unit-normalized analytically; grid re-normalization keeps it exactly
    # comparable with the volume-weighted curve, which is normalized the same
    # way per bayesian_statistics's Z_g).
    z_fine = np.linspace(max(0.0, z_g - 8 * sigma_z), z_g + 8 * sigma_z, 4001)
    bare_fine = np.exp(-0.5 * ((z_fine - z_g) / sigma_z) ** 2) / (sigma_z * np.sqrt(2.0 * np.pi))
    bare_norm = float(np.trapezoid(bare_fine, z_fine))
    bare = bare_unnorm / bare_norm

    # w_pop(z) = dVc/dz/dOmega / (1+z); same function production's _w_pop_eff
    # calls (bayesian_statistics.py line ~4850).
    def w_pop(zz: np.ndarray) -> np.ndarray:
        return np.asarray(
            comoving_volume_element(zz, h=H_TRUE, Omega_m=OMEGA_M, Omega_de=OMEGA_DE),
            dtype=np.float64,
        ) / (1.0 + zz)

    gauss_fine = np.exp(-0.5 * ((z_fine - z_g) / sigma_z) ** 2) / (sigma_z * np.sqrt(2.0 * np.pi))
    w_fine = w_pop(z_fine)
    Z_g = float(np.trapezoid(gauss_fine * w_fine, z_fine))
    gauss = np.exp(-0.5 * ((z - z_g) / sigma_z) ** 2) / (sigma_z * np.sqrt(2.0 * np.pi))
    volume = gauss * w_pop(z) / Z_g

    if Z_g <= 0.0 or not np.isfinite(Z_g):
        _fail("Z_g normalization non-positive/non-finite")
    if abs(float(np.trapezoid(bare_fine / bare_norm, z_fine)) - 1.0) > 1e-3:
        _fail("bare Gaussian kernel does not normalize to 1")

    return {
        "z": _rlist(z, 6),
        "bare": _rlist(bare, 7),
        "volume": _rlist(volume, 7),
        "z_g": z_g,
        "sigma_z": sigma_z,
        "Z_g": _r(Z_g, 6),
        "prov": "real: bayesian_statistics host-z volume kernel shape "
        "(p_g(z) = N(z;z_g,sigma_z) * w_pop(z) / Z_g, w_pop = dVc/dz/(1+z)), "
        "recomputed from physical_relations.comoving_volume_element",
    }


# ---------------------------------------------------------------------------
# 4. Reines & Volonteri (2015) stellar-to-BH mass relation (REAL constants)
# ---------------------------------------------------------------------------
def build_rv() -> dict[str, Any]:
    # Module constants are natural-log/1e10-pivot form; recover the docstring's
    # log10/1e11-pivot numbers to check against literature values before use.
    alpha_log10 = float(_RV_ALPHA_LN / np.log(10.0))  # expect 7.45
    beta_check = float(_RV_BETA)  # expect 1.05
    sigma_log10 = float(_RV_SIGMA_LN / np.log(10.0))  # expect 0.24

    if abs(alpha_log10 - 7.45) > 1e-6:
        _fail(f"R&V15 alpha mismatch: module gives {alpha_log10} log10, expected 7.45")
    if abs(beta_check - 1.05) > 1e-6:
        _fail(f"R&V15 beta mismatch: module gives {beta_check}, expected 1.05")
    if abs(sigma_log10 - 0.24) > 1e-6:
        _fail(f"R&V15 intrinsic scatter mismatch: module gives {sigma_log10} dex, expected 0.24")

    log_mstar = np.linspace(8.5, 12.0, N_PTS)
    log_mbh = alpha_log10 + beta_check * (log_mstar - 11.0)

    return {
        "log_mstar": _rlist(log_mstar, 6),
        "log_mbh": _rlist(log_mbh, 6),
        "log_mbh_lo": _rlist(log_mbh - sigma_log10, 6),
        "log_mbh_hi": _rlist(log_mbh + sigma_log10, 6),
        "alpha_log10": alpha_log10,
        "beta": beta_check,
        "sigma_dex": sigma_log10,
        "prov": "real: galaxy_catalogue/handler.py R&V15 constants "
        "(alpha, beta, sigma_int module constants, checked against literature "
        "7.45/1.05/0.24 before use; log10(M_BH/Msun) = alpha + beta*(log10(M*/Msun) - 11))",
    }


# ---------------------------------------------------------------------------
# 5. Detection probability p_det(d_L) (REAL)
# ---------------------------------------------------------------------------
def build_pdet() -> dict[str, Any]:
    d_l = np.linspace(0.0, 2.0 * D50_GPC, N_PTS)
    p = detection_probability(d_l, D50_GPC, W_PDET_GPC)

    if not np.all((p >= 0.0) & (p <= 1.0)):
        _fail("p_det out of [0,1]")
    p_at_d50 = float(detection_probability(np.array([D50_GPC]), D50_GPC, W_PDET_GPC)[0])
    if abs(p_at_d50 - 0.5) > 1e-9:
        _fail(f"p_det(d50) = {p_at_d50} != 0.5")

    return {
        "d_l": _rlist(d_l, 6),
        "p": _rlist(p, 6),
        "params": {"d50_gpc": float(D50_GPC), "w_pdet_gpc": float(W_PDET_GPC)},
        "prov": "real: validation/pp_coverage.py detection_probability "
        "(smooth Malmquist erfc model, commission-venue defaults)",
    }


# ---------------------------------------------------------------------------
# 6. Finite-difference stencil error scaling (TOY, analytic demo)
# ---------------------------------------------------------------------------
def build_stencil() -> dict[str, Any]:
    x0 = 1.0
    exact = np.cos(x0)  # d/dx sin(x) at x=1
    eps = np.logspace(-8, -1, N_PTS)

    forward = (np.sin(x0 + eps) - np.sin(x0)) / eps
    err_forward = np.abs(forward - exact)

    five_pt = (
        -np.sin(x0 + 2 * eps) + 8 * np.sin(x0 + eps) - 8 * np.sin(x0 - eps) + np.sin(x0 - 2 * eps)
    ) / (12.0 * eps)
    err_5pt = np.abs(five_pt - exact)

    return {
        "eps": _rlist(eps, 6),
        "err_forward": _rlist(np.clip(err_forward, 1e-20, None), 6),
        "err_5pt": _rlist(np.clip(err_5pt, 1e-20, None), 6),
        "x0": x0,
        "f": "sin(x)",
        "exact_derivative": _r(exact, 8),
        "prov": "toy: analytic sin(x) demo of Vallisneri (2008) arXiv:gr-qc/0703086 "
        "Appendix A stencil error scaling; not a call into parameter_estimation.py "
        "(that requires a full EMRI waveform evaluation)",
    }


# ---------------------------------------------------------------------------
# atlas_journey.json -- per-event journey data (889 loud, 606 medium, 555 faint)
# ---------------------------------------------------------------------------
# The 14 CRB table parameters, in the table's own column order -- matches
# gen_ch06.py's CRB_PARAMS_14 and datamodels/parameter_space.py's
# _parameters_to_dict() (which is also the exact positional order the
# production waveform/response generators are called with).
CRB_PARAMS_14 = [
    "M",
    "mu",
    "a",
    "p0",
    "e0",
    "x0",
    "luminosity_distance",
    "qS",
    "phiS",
    "qK",
    "phiK",
    "Phi_phi0",
    "Phi_theta0",
    "Phi_r0",
]

CRB_REL = _ch06.CRB_REL


def _corr14(row: pd.Series) -> np.ndarray:
    """The 14x14 correlation matrix of this event's own stored Cramer-Rao
    covariance, assembled from the ``delta_<a>_delta_<b>`` columns exactly
    as ``gen_ch06.py``'s ``_cov14`` assembles the covariance (same columns,
    same order), then normalized to a correlation matrix."""
    n = len(CRB_PARAMS_14)
    cov = np.zeros((n, n))
    for i, a in enumerate(CRB_PARAMS_14):
        for j, b in enumerate(CRB_PARAMS_14[: i + 1]):
            v = float(row[f"delta_{a}_delta_{b}"])
            cov[i, j] = v
            cov[j, i] = v
    sd = np.sqrt(np.diag(cov))
    return np.asarray(cov / np.outer(sd, sd))


def _waveform_48h(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Real ``few`` Pn5AAK waveform, the production wave generator
    (``waveform_generator.py`` ``WaveGeneratorType.PN5_AAK``: same class,
    same ``inspiral_kwargs``/``sum_kwargs``, same ``frame="detector"``),
    called directly (pre-TDI-response) on this event's own 14 CRB
    parameters, with ``T`` = 48 hours so the segment is the exact head of
    the production trajectory (same initial conditions -> identical first
    48 h). Returns (t_hr full-res, hp full-res, hp downsample-striped, dt).
    """
    from few.waveform import GenerateEMRIWaveform  # noqa: PLC0415

    inspiral_kwargs = {"DENSE_STEPPING": 0, "max_init_len": int(1e6)}
    sum_kwargs = {"pad_output": True}
    gen = GenerateEMRIWaveform(
        waveform_class="Pn5AAKWaveform",
        inspiral_kwargs=inspiral_kwargs,
        sum_kwargs=sum_kwargs,
        frame="detector",
        force_backend=None,
    )
    dt = 10.0
    T_yr = 48.0 / 24.0 / 365.25
    args = [float(row[k]) for k in CRB_PARAMS_14]
    h = gen(*args, dt=dt, T=T_yr)
    h = np.asarray(h)
    if not np.all(np.isfinite(h)):
        _fail("waveform not finite")
    hp = h.real
    max_hp = float(np.max(np.abs(hp)))
    if not (1e-24 <= max_hp <= 1e-19):
        _fail(f"max|hp| = {max_hp!r} outside gate range [1e-24, 1e-19]")
    t_hr = np.arange(len(hp)) * dt / 3600.0
    return t_hr, hp, hp, dt


def _ftrack(row: pd.Series) -> tuple[np.ndarray, np.ndarray, float]:
    """Real ``few`` trajectory: ``EMRIInspiral(func=PN5)``, the ODE the
    Pn5AAK waveform module runs internally (``few.waveform.waveform.PN5``,
    set in ``Pn5AAKWaveform.__init__``), over the mission span or until
    plunge, whichever is sooner. ``f_phi`` from a cubic spline of
    ``Phi_phi(t)`` (smooth derivative, avoids raw-gradient noise on the
    trajectory's sparse adaptive steps); reported as the dominant
    quadrupole harmonic ``2*f_phi``."""
    from few.trajectory.inspiral import EMRIInspiral  # noqa: PLC0415
    from few.waveform.waveform import PN5  # noqa: PLC0415
    from scipy.interpolate import CubicSpline  # noqa: PLC0415

    traj = EMRIInspiral(func=PN5)
    args = [float(row[k]) for k in ("M", "mu", "a", "p0", "e0", "x0")]
    out = traj(*args, T=LISA_MISSION_DURATION_YEARS, dt=10.0)
    t, phi_phi = out[0], out[4]
    if len(t) < 4:
        _fail("trajectory too short to interpolate")
    cs = CubicSpline(t, phi_phi)
    t_grid = np.linspace(float(t[0]), float(t[-1]), 200)
    dphidt = cs(t_grid, 1)  # rad/s
    f_mhz = 2.0 * (dphidt / (2.0 * np.pi)) * 1000.0
    tol = 1e-6 * max(1.0, float(np.max(np.abs(f_mhz))))
    if np.any(np.diff(f_mhz) < -tol):
        _fail("f_phi track is not monotonically increasing (chirp gate)")
    return t_grid / SEC_PER_YEAR, f_mhz, float(t[-1]) / SEC_PER_YEAR


def _spectrum(hp: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Real Welch ASD (``scipy.signal.welch``) of the 48-hour waveform
    snapshot; ``hc = sqrt(f*S_h)`` characteristic strain, downsampled onto
    a log-spaced frequency grid via log-log interpolation."""
    from scipy.signal import welch  # noqa: PLC0415

    fs = 1.0 / dt
    nperseg = min(4096, len(hp))
    f, pxx = welch(hp, fs=fs, nperseg=nperseg)
    mask = f > 0
    f, pxx = f[mask], np.clip(pxx[mask], 0.0, None)
    hc = np.sqrt(f * pxx)
    f_log = np.logspace(np.log10(f[0]), np.log10(f[-1]), 150)
    hc_log = np.exp(np.interp(np.log(f_log), np.log(f), np.log(np.clip(hc, 1e-40, None))))
    return f_log, hc_log


def _geolike(
    d_l_meas: float, sigma_dl: float, z_window: list[float], *, in_catalog: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    """TOY single-host geometric H0 reconstruction:
    ``lnL(h) = log INT N(d_L(z;h); d_L_meas, sigma_eff) * w(z) dz`` over the
    event's own z_window, ``w(z) = dVc/dz/(1+z)`` the same comoving-volume
    prior ``zkernel`` above uses. ``sigma_eff`` folds in the host photo-z
    error (sigma_z = 0.035, propagated through dd_L/dz at h=0.73) only for
    the in-catalogue event -- dark events carry no host-z measurement, so
    only their own d_L uncertainty and the (wide) window enter. Normalized
    so the max of lnL is 0."""
    from master_thesis_code.constants import OMEGA_DE, OMEGA_M

    z_lo, z_hi = z_window
    z_c = 0.5 * (z_lo + z_hi)
    sigma_photoz_dl = 0.0
    if in_catalog:
        eps = 1e-4
        dd_dz = (
            dist(z_c + eps, h=0.73, Omega_m=OMEGA_M, Omega_de=OMEGA_DE)
            - dist(z_c - eps, h=0.73, Omega_m=OMEGA_M, Omega_de=OMEGA_DE)
        ) / (2.0 * eps)
        sigma_photoz_dl = abs(dd_dz) * HOST_PHOTOZ_SIGMA_Z
    sigma_eff = math.sqrt(sigma_dl**2 + sigma_photoz_dl**2)

    h_grid = np.linspace(0.60, 0.86, 63)
    z_fine = np.linspace(z_lo, z_hi, 400)
    w = comoving_volume_element(z_fine, h=0.73, Omega_m=OMEGA_M, Omega_de=OMEGA_DE) / (1.0 + z_fine)

    ln_l = np.empty_like(h_grid)
    for i, h_val in enumerate(h_grid):
        d_l_z = np.array(
            [dist(float(z), h=float(h_val), Omega_m=OMEGA_M, Omega_de=OMEGA_DE) for z in z_fine]
        )
        integrand = np.exp(-0.5 * ((d_l_z - d_l_meas) / sigma_eff) ** 2) * w
        val = float(np.trapezoid(integrand, z_fine))
        ln_l[i] = math.log(max(val, 1e-300))
    ln_l = ln_l - ln_l.max()
    return h_grid, ln_l, sigma_eff


def _kernelz(z_window: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Real bare-vs-volume host-z kernel, same math as ``build_zkernel``
    above, recentered on this event's own z_window (z_c = mean(z_window),
    sigma_z = 0.035)."""
    from master_thesis_code.constants import OMEGA_DE, OMEGA_M
    from master_thesis_code.constants import H as H_TRUE

    z_c = 0.5 * (z_window[0] + z_window[1])
    sigma_z = HOST_PHOTOZ_SIGMA_Z
    z = np.linspace(max(0.0, z_c - 5 * sigma_z), z_c + 5 * sigma_z, 120)

    bare_unnorm = np.exp(-0.5 * ((z - z_c) / sigma_z) ** 2) / (sigma_z * np.sqrt(2.0 * np.pi))
    z_fine = np.linspace(max(0.0, z_c - 8 * sigma_z), z_c + 8 * sigma_z, 4001)
    bare_fine = np.exp(-0.5 * ((z_fine - z_c) / sigma_z) ** 2) / (sigma_z * np.sqrt(2.0 * np.pi))
    bare_norm = float(np.trapezoid(bare_fine, z_fine))
    bare = bare_unnorm / bare_norm

    def w_pop(zz: np.ndarray) -> np.ndarray:
        return np.asarray(
            comoving_volume_element(zz, h=H_TRUE, Omega_m=OMEGA_M, Omega_de=OMEGA_DE),
            dtype=np.float64,
        ) / (1.0 + zz)

    w_fine = w_pop(z_fine)
    z_g_norm = float(np.trapezoid(bare_fine * w_fine, z_fine))
    if z_g_norm <= 0.0 or not np.isfinite(z_g_norm):
        _fail("kernelz normalization non-positive/non-finite")
    gauss = np.exp(-0.5 * ((z - z_c) / sigma_z) ** 2) / (sigma_z * np.sqrt(2.0 * np.pi))
    volume = gauss * w_pop(z) / z_g_norm
    return z, bare, volume, z_c


def _ball_counts(crb: pd.DataFrame, catalogue: Path, ch06_events: dict[str, Any]) -> dict[int, int]:
    """Catalogue candidate count within each event's own production
    BallTree search radius (``ch06_fisher.json`` ``radius_full_rad``, a
    chord length on the unit-sphere embedding -- see
    ``gen_ch06.py::_patch_block``) and redshift window (``z_window``).
    Reuses ``gen_ch06.py``'s own two-stage pipeline verbatim: a cheap
    angular cone cut on the raw ICRS catalogue (``_load_patch``), then the
    production stellar-BH-mass + equatorial->ecliptic + mass/redshift-prune
    pipeline (``_prepare_patch``) on the small culled patch."""
    from astropy import units as u  # noqa: PLC0415
    from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord  # noqa: PLC0415

    targets: list[tuple[float, float]] = []
    for eid in JOURNEY_EVENT_IDS:
        row = crb.loc[eid]
        lon = math.degrees(float(row["phiS"])) % 360.0
        lat = math.degrees(math.pi / 2.0 - float(row["qS"]))
        icrs = SkyCoord(
            lon=lon * u.deg, lat=lat * u.deg, frame=BarycentricTrueEcliptic(equinox="J2000")
        ).transform_to("icrs")
        targets.append((float(icrs.ra.deg), float(icrs.dec.deg)))

    raw_patches = _ch06._load_patch(catalogue, targets)
    patches = [_ch06._prepare_patch(p) for p in raw_patches]

    counts: dict[int, int] = {}
    for k, eid in enumerate(JOURNEY_EVENT_IDS):
        row = crb.loc[eid]
        patch = patches[k]
        query = _ch06._polar_to_cartesian(
            np.array([float(row["qS"])]), np.array([float(row["phiS"])])
        )[0]
        pts = _ch06._polar_to_cartesian(patch["THETA_S"].to_numpy(), patch["PHI_S"].to_numpy())
        chord = np.linalg.norm(pts - query, axis=1)
        radius = float(ch06_events[str(eid)]["radius_full_rad"])
        z_lo, z_hi = ch06_events[str(eid)]["z_window"]
        z_g = patch["REDSHIFT"].to_numpy()
        z_err = patch["REDSHIFT_MEASUREMENT_ERROR"].to_numpy()
        in_z = (z_g + z_err >= z_lo) & (z_g - z_err <= z_hi)
        mask = (chord <= radius) & in_z
        counts[eid] = int(mask.sum())
    return counts


def _pool_shared() -> dict[str, Any]:
    """The injection-pool (logM, z) 2D histogram and z marginals, reused
    directly from the already-committed ``ch09_factory.json`` (its
    ``logm_edges``/``z_edges``/``pool.hist_drawn``/``pool.hist_detected``,
    themselves built from the production injection pool) -- not
    re-derived from the raw pool files."""
    if not CH09_FACTORY_FILE.exists():
        return {
            "pool_hist": None,
            "z_drawn": None,
            "z_detected": None,
            "pool_note": f"{CH09_FACTORY_FILE.name} not found; no pool reuse possible",
        }
    factory = json.loads(CH09_FACTORY_FILE.read_text())
    logm_edges = np.asarray(factory["logm_edges"], dtype=float)
    z_edges = np.asarray(factory["z_edges"], dtype=float)
    hist_drawn = np.asarray(factory["pool"]["hist_drawn"], dtype=float)
    hist_detected = np.asarray(factory["pool"]["hist_detected"], dtype=float)

    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    z_width = np.diff(z_edges)

    def _marginal(hist2d: np.ndarray) -> dict[str, Any]:
        counts_z = hist2d.sum(axis=0)  # sum over logM axis -> per-z counts
        total = float(counts_z.sum())
        density = counts_z / (total * z_width) if total > 0 else counts_z
        return {"z": _rlist(z_centers, 5), "density": _rlist(density, 5)}

    return {
        "pool_hist": {
            "logm_edges": _rlist(logm_edges, 5),
            "z_edges": _rlist(z_edges, 5),
            "counts": [[int(v) for v in row] for row in hist_drawn],
            "prov": "real: ch09_factory.json pool.hist_drawn (production injection pool, "
            f"{factory['pool']['dir']})",
        },
        "z_drawn": _marginal(hist_drawn),
        "z_detected": _marginal(hist_detected),
        "pool_note": None,
    }


def build_journey() -> dict[str, Any]:
    print("gen_ch_atlas: building per-event journey data (889/606/555)")

    crb_path = _ch06._resolve(CRB_REL)
    if crb_path is None:
        raise SystemExit(f"gen_ch_atlas: required CRB table not found: {CRB_REL}")
    crb = pd.read_csv(crb_path)

    if not CH06_FISHER_FILE.exists():
        raise SystemExit(
            f"gen_ch_atlas: {CH06_FISHER_FILE.name} not found; run gen_ch06.py first "
            "(atlas_journey.json reuses its per-event radius_full_rad/z_window)"
        )
    ch06 = json.loads(CH06_FISHER_FILE.read_text())
    ch06_events = ch06["events"]

    catalogue = _ch06._resolve(_ch06.CATALOGUE_REL)
    if catalogue is None:
        print(
            "    NOTICE: reduced_galaxy_catalogue.csv not found in this repo or a "
            "sibling MasterThesisCode checkout -- ball counts will be omitted."
        )
        ball_counts: dict[int, int] = {}
    else:
        print("    cutting sky patches for the ball counts (reused gen_ch06 pipeline)...")
        ball_counts = _ball_counts(crb, catalogue, ch06_events)

    events: dict[str, Any] = {}
    max_hp_by_event: dict[int, float] = {}
    for eid in JOURNEY_EVENT_IDS:
        row = crb.loc[eid]
        block = ch06_events[str(eid)]
        z_window = block["z_window"]
        in_catalog = bool(block["in_catalog"])

        print(f"    event {eid}: waveform (few, ~48h)...")
        t_hr_full, hp_full, hp_for_spectrum, dt = _waveform_48h(row)
        target_n = 1300
        stride = max(1, len(hp_full) // target_n)
        max_hp = float(np.max(np.abs(hp_full)))
        max_hp_by_event[eid] = max_hp

        print(f"    event {eid}: trajectory (few EMRIInspiral, PN5)...")
        t_yr, f_mhz, t_plunge_yr = _ftrack(row)

        print(f"    event {eid}: spectrum (Welch)...")
        f_hz, hc = _spectrum(hp_for_spectrum, dt)

        corr14 = _corr14(row)

        d_l = float(row["luminosity_distance"])
        sigma_dl = float(
            math.sqrt(float(row["delta_luminosity_distance_delta_luminosity_distance"]))
        )
        h_grid, ln_l, sigma_eff = _geolike(d_l, sigma_dl, z_window, in_catalog=in_catalog)

        z_k, bare_k, volume_k, z_c = _kernelz(z_window)

        events[str(eid)] = {
            "role": JOURNEY_ROLES[eid],
            "waveform": {
                "t_hr": _rlist(t_hr_full[::stride], 6),
                "hp": _rlist(hp_full[::stride], 5),
                "prov": "real: few GenerateEMRIWaveform(waveform_class='Pn5AAKWaveform'), "
                "the production wave generator, called directly (dt=10 s, T=48 h), "
                "first 48 h, source-frame h_plus (pre-TDI)",
                "note": (
                    "Same initial conditions as the full production run -> this is exactly "
                    "the first 48 hours of that trajectory, not a separate short run."
                ),
                "max_abs_hp": _r(max_hp, 5),
                "dt_s": dt,
            },
            "ftrack": {
                "t_yr": _rlist(t_yr, 6),
                "f_mHz": _rlist(f_mhz, 6),
                "prov": "real: few EMRIInspiral(func=PN5) trajectory, "
                "f_phi = d(Phi_phi)/dt / (2 pi) via cubic-spline derivative, "
                "dominant harmonic reported as 2*f_phi",
                "t_plunge_yr": _r(t_plunge_yr, 5),
            },
            "spectrum": {
                "f_Hz": _rlist(f_hz, 6),
                "hc": _rlist(hc, 6),
                "prov": "real: Welch ASD (scipy.signal.welch) of the 48-hour few snapshot, "
                "hc = sqrt(f*S_h)",
            },
            "corr14": [[_r(v, 3) for v in r_] for r_ in corr14],
            "csv_row": {
                k: (f"{float(row[k]):.6g}" if k != "SNR" else f"{float(row['SNR']):.6g}")
                for k in [*CRB_PARAMS_14, "SNR"]
            },
            "ball": {
                "n_candidates": ball_counts.get(eid),
                "method": "chord distance <= radius_full_rad (ch06_fisher.json, production "
                "BallTree metric) AND redshift inside z_window, over the production "
                "stellar-BH-mass-pruned ecliptic-frame patch",
                "prov": "real: reduced catalogue count (gen_ch06.py sky-patch pipeline, reused)"
                if catalogue is not None
                else "unavailable: reduced_galaxy_catalogue.csv not found locally",
            },
            "geolike": {
                "h": _rlist(h_grid, 5),
                "lnL": _rlist(ln_l, 6),
                "sigma_eff_Gpc": _r(sigma_eff, 5),
                "prov": "toy: geometric reconstruction, lnL(h) = log INT N(d_L(z;h); "
                "d_L_meas, sigma_eff) * w(z) dz over z_window, w(z) the same comoving-"
                "volume prior as atlas_curves.json's zkernel section"
                + (
                    "; sigma_eff includes the host photo-z term (sigma_z=0.035, propagated "
                    "through dd_L/dz at h=0.73) because this event is in-catalogue"
                    if in_catalog
                    else "; dark event -- sigma_eff is d_L uncertainty only"
                ),
            },
            "kernelz": {
                "z": _rlist(z_k, 6),
                "bare": _rlist(bare_k, 7),
                "volume": _rlist(volume_k, 7),
                "z_c": _r(z_c, 6),
                "prov": "real: bayesian_statistics host-z volume kernel shape, recentered "
                "on this event's own z_window (same math as atlas_curves.json's zkernel)",
            },
        }

    if max_hp_by_event[889] <= max(max_hp_by_event[606], max_hp_by_event[555]):
        _fail(
            "waveform amplitude ordering gate failed: event 889 (highest SNR, closest "
            f"d_L) must have the largest max|hp|; got {max_hp_by_event!r}"
        )
    if ball_counts:
        if ball_counts.get(889, 0) >= 50:
            _fail(f"ball count gate failed: event 889 count {ball_counts[889]} not < 50")
        if not (ball_counts.get(555, 0) > ball_counts.get(889, 0)):
            _fail(
                "ball count gate failed: event 555 count must exceed event 889's; "
                f"got {ball_counts!r}"
            )

    shared = _pool_shared()
    if shared["z_detected"] is None:
        # Fallback: derive the detected-z density directly from the CRB table's
        # own distances, inverting physical_relations.dist at h=0.73 by
        # bisection on the monotonic d_L(z) relation.
        from master_thesis_code.constants import OMEGA_DE, OMEGA_M

        d_l_all = crb["luminosity_distance"].to_numpy()
        z_grid = np.linspace(1e-4, 3.0, 4000)
        d_l_grid = np.array(
            [dist(float(z), h=0.73, Omega_m=OMEGA_M, Omega_de=OMEGA_DE) for z in z_grid]
        )
        z_of_dl = np.interp(d_l_all, d_l_grid, z_grid)
        edges = np.linspace(0.0, float(np.percentile(z_of_dl, 99)), 31)
        hist, edges = np.histogram(z_of_dl, bins=edges, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        shared["z_detected"] = {"z": _rlist(centers, 5), "density": _rlist(hist, 5)}
        shared["pool_note"] = (
            shared["pool_note"] or ""
        ) + " z_detected derived from CRB CSV distances (dist inverted at h=0.73), not the pool."

    return {
        "meta": {
            "generator": "gen_ch_atlas.py::build_journey",
            "note": "per-event journey data for the Atlas (889 loud, 606 medium, 555 faint)",
            "prov": {
                "waveform": "real",
                "ftrack": "real",
                "spectrum": "real",
                "corr14": "real",
                "csv_row": "real",
                "ball": "real" if ball_counts else "unavailable",
                "geolike": "toy",
                "kernelz": "real",
                "shared.pool_hist": "real" if shared["pool_hist"] is not None else "unavailable",
            },
        },
        "params14_order": list(CRB_PARAMS_14),
        "events": events,
        "shared": shared,
    }


def main() -> None:
    print("gen_ch_atlas: building Atlas formula-explorer data")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "meta": {
            "generator": "gen_ch_atlas.py",
            "note": "deterministic curve data for the Atlas formula explorer",
            "prov": {
                "psd": "real",
                "dlz": "real",
                "zkernel": "real",
                "rv": "real",
                "pdet": "real",
                "stencil": "toy",
            },
        },
        "psd": build_psd(),
        "dlz": build_dlz(),
        "zkernel": build_zkernel(),
        "rv": build_rv(),
        "pdet": build_pdet(),
        "stencil": build_stencil(),
    }

    OUT_FILE.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print(f"  wrote {OUT_FILE.relative_to(REPO_ROOT)} ({OUT_FILE.stat().st_size:,} bytes)")
    print(
        "  gates: PSD finite/positive; dist(0)==0; kernel normalizations OK; "
        "R&V15 constants match literature (7.45/1.05/0.24); p_det(d50)==0.5"
    )

    journey = build_journey()
    OUT_JOURNEY.write_text(json.dumps(journey, separators=(",", ":")) + "\n")
    print(f"  wrote {OUT_JOURNEY.relative_to(REPO_ROOT)} ({OUT_JOURNEY.stat().st_size:,} bytes)")
    print(
        "  gates: waveform finite + max|hp| in [1e-24,1e-19] + 889 largest amplitude; "
        "ftrack monotonically increasing (chirp); kernelz normalization OK; "
        "889 ball count < 50 and < 555's"
    )


if __name__ == "__main__":
    main()
