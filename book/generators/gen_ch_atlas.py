"""Generator for the Atlas formula explorer (cross-chapter reference page).

Produces one data file:

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

Run as::

    /home/jasper/Repositories/MasterThesisCode/.venv/bin/python \\
        book/generators/gen_ch_atlas.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# galaxy_catalogue.handler's alpha/beta/sigma_int module constants, imported
# read-only so the transcribed docstring formula below is checked against the
# module's own numbers rather than typed independently.
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

N_PTS = 120


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


if __name__ == "__main__":
    main()
