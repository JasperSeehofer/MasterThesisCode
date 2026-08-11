"""Generator for Chapter 7 — "A Redshift Is Not a Number".

Produces four data files under ``book/site/data/``:

``ch07_eddington.json``
    Cosmology tables (from the project's own ``physical_relations``) for the two
    venues the chapter uses, plus the ratified G2b measured points/laws.  Powers
    **I7.1 The Eddington Machine** stages 1-2: the reader builds
    ``p_g(z) ∝ N(z; z_g, σ_z)·w_pop(z)`` live and reads the induced ``Δh`` off the
    exact ``d_L``-anchored mapping ``h_eff/h_true = f(z_g)/f(z_true)``.

``ch07_c7.json``
    The delivered Gate-B C7 measurement (per-host kernel peaks vs σ_z/z, the
    production confrontation, the σ_z→0 gate) — a RECORDED artifact, re-served
    for the browser.  Powers I7.1 stage 3.

``ch07_speczrescue.json``
    I7.2 — the recorded ledger #42 verdict plus the in-venue σ_z/z distribution
    of the 76 in-catalogue hosts that makes the verdict legible.

``ch07_volumetrunc.json``
    The museum interlude: real Gauss-Legendre nodes/weights (including the
    production ``_GL_NODES_50``) so the browser can *re-run* the aliasing that
    falsified ``volume_trunc``, plus the recorded FINDING.md numbers.

PROVENANCE / FIDELITY RULES OBSERVED HERE
-----------------------------------------
* Nothing is re-derived.  Every formula evaluated below is quoted from a
  ratified packet and cited in-line: G2b Eqs. (1.3)-(1.5), (2.1)-(2.3); the
  ``d_L``-anchored ``h_eff/h_true = f(z_eff)/f(z_true)`` mapping of
  ``gate_b_20260730/C7_README.md`` §5.
* Two venues are kept strictly separate, because the artifacts are:
    - ``g2b``        : Ω_m = 0.3, Ω_Λ = 0.7, h_true = 0.72 — the commission
                       synthetic in which the four Δh points were MEASURED
                       (``G2b_host_z_volume_prior.md`` §2.3).
    - ``production`` : Ω_m = 0.2726, h_true = 0.73 — ``constants.py`` (the
                       Barausse-M1-consistent design choice, G11).
* Three reproduction gates are executed at generation time and printed; they
  re-derive nothing, they re-measure published tables:
    G1  G2b §2.3 exact posterior-mean shift table at z_g = 0.05
    G2  G2b §2.3 amplitude table C(z̄) = h·s(z̄)·dln f/dz
    G3  the C7 corrected law [1+√(1+12ε²)]/2 against the delivered per-host peaks
  A fourth gate (G4) re-computes the observed in-catalogue ball-numerator tilt
  from ``real_r1/diagnostics/event_likelihoods.csv`` when that file is present
  (it is not tracked in every worktree); its verdict is printed, never written,
  so the JSON stays byte-deterministic across checkouts.

Two DISAGREEMENTS with the build spec were found — see
``book/design/flags/ch07_FLAGS.md``.  **FLAG-1** (σ_dL/dL of EMRI-889) was
RESOLVED book-wide on 2026-07-31 by the author's D1 mandate: the spec figure
``8.0e-5`` was the *absolute* σ_dL in Gpc under a *fractional* label, so the
JSON now carries the measured pair (``sigma_dL_Gpc`` + ``rel_dL_recomputed``)
plus a single ``rel_dL_erratum`` string, and no live bare spec value.
**FLAG-2** (the 0.256 rail threshold) stays open, with both values side by side.

The C7 decider — **the 2×2 cell B** — LANDED 2026-07-31 (evaluate 6103219 /
combine 6103220, the resubmission of the pre-registered 6101146/6101147 after a
pure-plumbing symlink failure).  ``ch07_c7.json`` carries the landed values in
``conflict.decider`` and ``hosts.resolved_by_cellB``; cell B settles C7's
magnitude and attribution, **not** the G2b↔C7 collision.

Run (read-only against ``darksiren_emri/``, ``docs/`` and ``results/``):

    /home/jasper/Repositories/MasterThesisCode/.venv/bin/python \\
        book/generators/gen_ch07.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scipy.special import roots_legendre  # noqa: E402

from darksiren_emri.bayesian_inference.bayesian_statistics import (  # noqa: E402
    _GL_NODES_50,
    _GL_WEIGHTS_50,
)
from darksiren_emri.constants import OMEGA_DE as OMEGA_DE_PRODUCTION  # noqa: E402
from darksiren_emri.constants import OMEGA_M as OMEGA_M_PRODUCTION  # noqa: E402
from darksiren_emri.constants import H as H_PRODUCTION  # noqa: E402
from darksiren_emri.physical_relations import (  # noqa: E402
    comoving_volume_element,
    dist_vectorized,
    lambda_cdm_analytic_distance,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "site" / "data"

CAMPAIGN = REPO_ROOT / "results" / "campaign51_20260728" / "realistic_20260729"
GATE_B = CAMPAIGN / "gate_b_20260730"
SEED = CAMPAIGN / "seed61000"

# --------------------------------------------------------------------------- #
# Venues.  Ω_m = 0.3 / h = 0.72 is the commission synthetic in which G2b's four
# Δh points were measured (G2b §2.3 header); Ω_m = 0.2726 / h = 0.73 is
# constants.py (design choice G11, Barausse 2012 M1 consistency).
# --------------------------------------------------------------------------- #
VENUES: dict[str, dict[str, Any]] = {
    "g2b": {
        "label": "G2b / commission synthetic",
        "omega_m": 0.3,
        "omega_de": 0.7,
        "h_true": 0.72,
        "chip": "G2b §2.3",
        "note": (
            "The venue in which the four Δh points were measured: flat ΛCDM, "
            "Ω_m = 0.3, h_true = 0.72, single-host clean coverage test."
        ),
    },
    "production": {
        "label": "this pipeline (campaign #51/#53)",
        "omega_m": float(OMEGA_M_PRODUCTION),
        "omega_de": float(OMEGA_DE_PRODUCTION),
        "h_true": float(H_PRODUCTION),
        "chip": "constants.py",
        "note": (
            "The mock universe of this book: Ω_m = 0.2726, h_true = 0.73 "
            "(Barausse 2012 M1 consistency — a design choice, G11, not a fit)."
        ),
    },
}

# z table: fine where the hosts are, coarse in the tail.  Linear interpolation
# of these smooth functions is well below the σ_z→0 numerical floor (checked by
# gate G1, which reproduces G2b's own exact table from the interpolated form).
Z_FINE_STEP = 0.001
Z_FINE_MAX = 0.60
Z_COARSE_STEP = 0.01
Z_MAX = 2.40

# The σ_z values the commission measured at (G2b §2.3, table rows).
G2B_SIGMA_Z = [0.005, 0.015, 0.035, 0.050]
G2B_DELTA_H = [-0.0016, -0.0064, -0.023, -0.046]
G2B_FLOOR_SUBTRACTED = [+0.0004, -0.0044, -0.021, -0.044]
G2B_C_MEAS = [None, 19.6, 17.1, 17.6]
G2B_VOLUME_FLOOR = -0.002  # the σ_z-independent VOLUME-estimator residual

# G2b §2.3 amplitude table, quoted, to be re-measured by gate G2.
G2B_AMPLITUDE_TABLE = [
    {"z_bar": 0.20, "s": 8.14, "dlnf_dz": 5.58, "C": 32.7},
    {"z_bar": 0.25, "s": 6.15, "dlnf_dz": 4.55, "C": 20.1},
    {"z_bar": 0.26, "s": 5.84, "dlnf_dz": 4.39, "C": 18.4},
    {"z_bar": 0.30, "s": 4.82, "dlnf_dz": 3.85, "C": 13.4},
    {"z_bar": 0.357, "s": 3.77, "dlnf_dz": 3.28, "C": 8.9},
]

# G2b §2.3 exact-vs-leading-order table at z_g = 0.05, quoted, re-measured by G1.
G2B_EXACT_SHIFT_TABLE = [
    {"sigma_z": 0.005, "exact": 0.00094, "leading": 0.00095},
    {"sigma_z": 0.015, "exact": 0.0079, "leading": 0.0086},
    {"sigma_z": 0.035, "exact": 0.0325, "leading": 0.0467},
    {"sigma_z": 0.050, "exact": 0.0535, "leading": 0.0953},
]

HOST_PRESETS = [
    {"z_g": 0.05, "label": "z_g = 0.05 — a GLADE-like low-z host (G2b §2.3)"},
    {"z_g": 0.10, "label": "z_g = 0.10"},
    {"z_g": 0.15, "label": "z_g = 0.15"},
    {"z_g": 0.20, "label": "z_g = 0.20"},
    {"z_g": 0.25, "label": "z_g = 0.25 — the effective host z̄ implied by C_meas"},
    {"z_g": 0.26, "label": "z_g = 0.26 — G2b's z̄_eff"},
    {"z_g": 0.30, "label": "z_g = 0.30 — the synthetic's median detected z"},
    {"z_g": 0.357, "label": "z_g = 0.357 — z(D_50 = 1.85 Gpc)"},
]

# Museum interlude: the falsified `volume_trunc` window, from FINDING.md §Mechanism.
VT_WINDOW = (0.0, 0.182)  # z_g = 0.05, σ_z = 0.033 → [z_g−4σ, z_g+4σ] clipped at 0
VT_PEAK_SIGMA = 0.0004  # a narrow GW peak in z (toy: FINDING quotes "~0.003" wide)
VT_NODE_COUNTS = [8, 16, 32, 50, 100, 200, 400, 800]

VT_FINDING_TABLE = [
    {"h": 0.60, "gw_window_n50": 0.0003, "host_window_n50": 0.0000, "host_window_exact": 0.2417},
    {"h": 0.73, "gw_window_n50": 0.0005, "host_window_n50": 0.0000, "host_window_exact": 0.4314},
    {"h": 0.86, "gw_window_n50": 0.0007, "host_window_n50": 0.0000, "host_window_exact": 0.6537},
]


# --------------------------------------------------------------------------- #
# Cosmology helpers — thin wrappers over the project's own functions.
# --------------------------------------------------------------------------- #
def _e_of_z(z: np.ndarray, om: float, ode: float) -> np.ndarray:
    """E(z) = sqrt(Ω_m(1+z)³ + Ω_Λ) — G2b Eq. (2.2) context, flat ΛCDM."""
    return np.sqrt(om * (1.0 + z) ** 3 + ode)


def _z_table() -> np.ndarray:
    fine = np.arange(0.0, Z_FINE_MAX, Z_FINE_STEP)
    coarse = np.arange(Z_FINE_MAX, Z_MAX + 0.5 * Z_COARSE_STEP, Z_COARSE_STEP)
    return np.unique(np.round(np.concatenate([fine, coarse]), 6))


def _venue_tables(om: float, ode: float) -> dict[str, list[float]]:
    """Tabulate the four z-functions the browser needs, from project code.

    w_pop(z) = (dV_c/dz)/(1+z)                      — G2b Eq. (1.3)
    f(z)     = d_L(z; h=1) = (1+z)∫dz'/E(z')·c/100  — the h-independent factor
               of d_L (C7_README §5: d_L(z;h) = f(z)/h)
    s(z)     = d ln w_pop/dz                        — G2b Eq. (2.2)
    dlnf/dz  = 1/(1+z) + 1/(I E)                    — G2b Eq. (2.3)
    """
    z = _z_table()
    vol = np.asarray(comoving_volume_element(z, h=1.0, Omega_m=om, Omega_de=ode), float)
    w_pop = vol / (1.0 + z)
    # w_pop(0) = 0 exactly; normalize by the maximum on the table (an overall
    # constant cancels between numerator and Z_g — G2b §1.2 dimensional note).
    w_shape = w_pop / float(np.max(w_pop))

    f_gpc = np.asarray(dist_vectorized(z, h=1.0, Omega_m=om, Omega_de=ode), float)

    i_of_z = np.array([lambda_cdm_analytic_distance(float(x), om, ode) for x in z])
    e_of_z = _e_of_z(z, om, ode)
    e_prime = 3.0 * om * (1.0 + z) ** 2 / (2.0 * e_of_z)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = 2.0 / (i_of_z * e_of_z) - e_prime / e_of_z - 1.0 / (1.0 + z)
        dlnf = 1.0 / (1.0 + z) + 1.0 / (i_of_z * e_of_z)
    # z = 0 is a coordinate singularity of s and dln f/dz (both ~ 2/z, 1/z).
    s[0] = float(s[1])
    dlnf[0] = float(dlnf[1])

    return {
        "z": [round(float(v), 6) for v in z],
        "w_pop_shape": [float(f"{v:.7g}") for v in w_shape],
        "f_gpc": [float(f"{v:.7g}") for v in f_gpc],
        "s_of_z": [float(f"{v:.6g}") for v in s],
        "dlnf_dz": [float(f"{v:.6g}") for v in dlnf],
    }


def _exact_mean_shift(z_g: float, sigma_z: float, om: float, ode: float) -> float:
    """δz = ⟨z⟩ − z_g under p_g(z) ∝ N(z; z_g, σ_z)·w_pop(z) — G2b Eq. (1.4)/(2.1).

    Evaluated by dense quadrature on ±8σ (the code itself uses 50-point
    Gauss-Legendre on ±4σ, `bayesian_statistics.py:4120-4123`; the wider, denser
    grid here is for the *reference* value G2b's table quotes).
    """
    lo = max(z_g - 8.0 * sigma_z, 1e-9)
    hi = z_g + 8.0 * sigma_z
    z = np.linspace(lo, hi, 20001)
    vol = np.asarray(comoving_volume_element(z, h=1.0, Omega_m=om, Omega_de=ode), float)
    p = np.exp(-0.5 * ((z - z_g) / sigma_z) ** 2) * vol / (1.0 + z)
    return float(np.trapezoid(z * p, z) / np.trapezoid(p, z) - z_g)


def _delta_h_from_shift(z_g: float, delta_z: float, om: float, ode: float, h_true: float) -> float:
    """Δh for a bare-Gaussian kernel that assumes ẑ = z_g when the truth is z_g+δz.

    C7_README §5 / G2b §2.2: d_L(z;h) = f(z)/h with f h-independent, so matching
    the same measured d_L gives h_eff/h_true = f(ẑ)/f(z_true) exactly.
    """
    f = np.asarray(
        dist_vectorized(np.array([z_g, z_g + delta_z]), h=1.0, Omega_m=om, Omega_de=ode), float
    )
    return float(h_true * (f[0] / f[1] - 1.0))


# --------------------------------------------------------------------------- #
# Reproduction gates
# --------------------------------------------------------------------------- #
def _gate_g1() -> list[dict[str, float]]:
    """Re-measure G2b §2.3's exact posterior-mean shift table at z_g = 0.05."""
    om, ode = VENUES["g2b"]["omega_m"], VENUES["g2b"]["omega_de"]
    i_of_z = lambda_cdm_analytic_distance(0.05, om, ode)
    e_z = float(_e_of_z(np.array([0.05]), om, ode)[0])
    e_p = 3.0 * om * 1.05**2 / (2.0 * e_z)
    s_005 = 2.0 / (i_of_z * e_z) - e_p / e_z - 1.0 / 1.05
    rows = []
    for quoted in G2B_EXACT_SHIFT_TABLE:
        s_z = float(quoted["sigma_z"])
        rows.append(
            {
                "sigma_z": s_z,
                "quoted_exact": float(quoted["exact"]),
                "measured_exact": round(_exact_mean_shift(0.05, s_z, om, ode), 6),
                "quoted_leading": float(quoted["leading"]),
                "measured_leading": round(s_z**2 * s_005, 6),
            }
        )
    return rows


def _gate_g2() -> list[dict[str, float]]:
    """Re-measure G2b §2.3's amplitude table C(z̄) = h·s(z̄)·dln f/dz."""
    om, ode = VENUES["g2b"]["omega_m"], VENUES["g2b"]["omega_de"]
    h_true = VENUES["g2b"]["h_true"]
    rows = []
    for quoted in G2B_AMPLITUDE_TABLE:
        z_b = float(quoted["z_bar"])
        i_of_z = lambda_cdm_analytic_distance(z_b, om, ode)
        e_z = float(_e_of_z(np.array([z_b]), om, ode)[0])
        e_p = 3.0 * om * (1.0 + z_b) ** 2 / (2.0 * e_z)
        s_z = 2.0 / (i_of_z * e_z) - e_p / e_z - 1.0 / (1.0 + z_b)
        dlnf = 1.0 / (1.0 + z_b) + 1.0 / (i_of_z * e_z)
        rows.append(
            {
                "z_bar": z_b,
                "quoted_s": float(quoted["s"]),
                "measured_s": round(float(s_z), 3),
                "quoted_dlnf_dz": float(quoted["dlnf_dz"]),
                "measured_dlnf_dz": round(float(dlnf), 3),
                "quoted_C": float(quoted["C"]),
                "measured_C": round(float(h_true * s_z * dlnf), 2),
            }
        )
    return rows


def c7_law(eps: float) -> float:
    """h_eff/h_true = [1 + sqrt(1 + 12 ε²)]/2 — the CORRECTED C7 law.

    `CLAIM_2D_BIAS_20260730.md` C7 as amended 2026-07-30 / `C7_README.md` §1.
    NEVER the claim's own superseded [1+sqrt(1+8ε²)]/2 form.
    """
    return 0.5 * (1.0 + math.sqrt(1.0 + 12.0 * eps * eps))


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def build_eddington() -> dict[str, Any]:
    venues = {}
    for key, meta in VENUES.items():
        tables = _venue_tables(meta["omega_m"], meta["omega_de"])
        venues[key] = {**{k: v for k, v in meta.items()}, "tables": tables}

    om, ode = VENUES["g2b"]["omega_m"], VENUES["g2b"]["omega_de"]
    h_true = VENUES["g2b"]["h_true"]

    # The chapter's headline reproduction: the exact Δh curve at z̄ = 0.25 lands
    # on the FLOOR-SUBTRACTED measured points.  Computed here so the page can
    # state it without the browser having to be trusted with it.
    landing = []
    for i, s_z in enumerate(G2B_SIGMA_Z):
        d_z = _exact_mean_shift(0.25, s_z, om, ode)
        landing.append(
            {
                "sigma_z": s_z,
                "measured_delta_h": G2B_DELTA_H[i],
                "measured_floor_subtracted": G2B_FLOOR_SUBTRACTED[i],
                "reproduced_delta_h_at_zbar_0p25": round(
                    _delta_h_from_shift(0.25, d_z, om, ode, h_true), 5
                ),
            }
        )

    return {
        "_provenance": {
            "equations": [
                "G2b Eq. (1.3): w_pop(z) = (dV_c/dz)/(1+z)",
                "G2b Eq. (1.4): p_g(z) = N(z_g; z, sigma_z) w_pop(z) / Z_g",
                "G2b Eq. (2.1): delta_z_Edd = sigma_z^2 d ln w_pop/dz",
                "G2b Eq. (2.3): Delta h = -h (dln f/dz) sigma_z^2 s(zbar) = -C sigma_z^2",
                "C7_README section 5: h_eff/h_true = f(z_eff)/f(z_true), d_L(z;h) = f(z)/h",
            ],
            "code": [
                "physical_relations.py:571 comoving_volume_element (Hogg 1999 Eq. 28)",
                "physical_relations.py:132 dist / :226 dist_vectorized",
                "bayesian_statistics.py:3712 galaxy_redshift_prior_pdf",
                "bayesian_statistics.py:4190-4199 w_pop construction, :4202 _z_prior_pdf_at",
            ],
            "artifacts": [
                "docs/derivations/G2b_host_z_volume_prior.md:229-237 (measured points)",
                "docs/derivations/G2b_host_z_volume_prior.md:413-436 (VERDICT: CONFIRMED)",
                "BIAS_HISTORY_LEDGER.md row 47 (bias -0.024 -> -0.002, coverage ~0% -> nominal)",
            ],
        },
        "venues": venues,
        "host_presets": HOST_PRESETS,
        "measured": {
            "sigma_z": G2B_SIGMA_Z,
            "delta_h": G2B_DELTA_H,
            "floor_subtracted": G2B_FLOOR_SUBTRACTED,
            "C_meas": G2B_C_MEAS,
            "volume_floor": G2B_VOLUME_FLOOR,
            "C_range": [17.0, 20.0],
            "scaling_ratios_measured": [1.0, 4.8, 10.0],
            "scaling_ratios_sigma_sq": [1.0, 5.44, 11.1],
            "ledger_47": {
                "bias_before": -0.024,
                "bias_after": -0.002,
                "coverage_before": "~0%",
                "coverage_after": "nominal",
            },
        },
        "gate_g1_exact_shift_table": _gate_g1(),
        "gate_g2_amplitude_table": _gate_g2(),
        "landing_check_zbar_0p25": landing,
    }


def _d_l_889_gpc() -> float:
    """d_L of CRB row 889 (Gpc), read from the tracked seed61000 CRB.

    Used only by D1's erratum arithmetic: the retired spec figure
    ``8.0e-5`` was the ABSOLUTE ``sigma_dL`` in Gpc under a fractional
    label, and ``sigma_dL = rel_dL * d_L`` has to be shown, not asserted.
    Pure stdlib csv — no pandas at module import time.
    """
    import csv

    path = SEED / "prepared_cramer_rao_bounds.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if i == 889:
                return float(row["luminosity_distance"])
    raise RuntimeError(f"row 889 not present in {path}")


def build_c7() -> dict[str, Any]:
    km = json.loads((GATE_B / "c7_kernel_measure_results.json").read_text())
    vp = json.loads((GATE_B / "c7_vs_production_results.json").read_text())

    eps_grid = [r["eps"] for r in km["legA"]]
    leg_a = []
    for r in km["legA"]:
        eps = float(r["eps"])
        leg_a.append(
            {
                "eps": eps,
                "measured_peak_h": round(float(r["median_peak_deconv"]), 4),
                "measured_frac_shift": round(float(r["median_frac_shift"]), 4),
                "law_frac_shift": round(c7_law(eps) - 1.0, 4),
                "superseded_8eps2_form": round(
                    0.5 * (1.0 + math.sqrt(1.0 + 8.0 * eps * eps)) - 1.0, 4
                ),
                "frac_peak_above_086": round(float(r["frac_peak_above_086"]), 4),
            }
        )

    eps_ind = np.asarray(km["eps_indicative"], float)
    z_true = np.asarray(km["z_true"], float)
    peak_ind = np.asarray(km["per_host_peak"]["indicative"], float)
    rel_dl = np.asarray(km["rel_dL"], float)

    # EMRI-889 is CRB row 889 of seed61000; its position in the 76-host in-cat
    # list is fixed by the driver's `crb[host_galaxy_index >= 0]` ordering.
    idx_889 = 41
    assert abs(float(rel_dl[idx_889]) - 8.983284023774961e-04) < 1e-12, "889 row moved"

    # D1: the ABSOLUTE sigma_dL, in Gpc — the quantity the retired spec figure
    # actually was.  Read from the tracked CRB so the erratum's own arithmetic
    # is a measurement, not a transcription.
    d_l_889 = _d_l_889_gpc()

    observed = np.asarray(vp["observed_incat_dln"], float)

    # FLAG-2: the artifacts state the rail threshold as sigma_z/z > 0.256; solving
    # the artifacts' OWN corrected law for h_eff = 0.86 at h_true = 0.73 gives
    # 0.2644.  Both are carried; neither is silently preferred.
    target = 0.86 / 0.73
    eps_solved = math.sqrt(((2.0 * target - 1.0) ** 2 - 1.0) / 12.0)

    return {
        "_provenance": {
            "artifacts": [
                "gate_b_20260730/c7_kernel_measure_results.json (the measurement)",
                "gate_b_20260730/c7_vs_production_results.json (the confrontation)",
                "gate_b_20260730/C7_README.md (verdict, corrected law, caveats)",
                "CLAIM_2D_BIAS_20260730.md C7 (adjudicated FINDING (MEASURED), 2026-07-30)",
                "docs/derivations/G2b_host_z_volume_prior.md:413-436 (the other side)",
            ],
            "code": ["bayesian_statistics.py:4190-4207 (w_pop_num, _z_prior_pdf_at)"],
            "law": "h_eff/h_true = [1 + sqrt(1 + 12 (sigma_z/z)^2)]/2 -> 1 + 3 (sigma_z/z)^2",
            "superseded_law": "[1 + sqrt(1 + 8 eps^2)]/2 — the claim's own form, UNDERSTATES by 1.35-1.5x",
        },
        "h_true": float(H_PRODUCTION),
        "prior_edges": [0.60, 0.86],
        "eps_grid": eps_grid,
        "leg_a": leg_a,
        "leg_b": [
            {
                "eps": float(r["eps"]),
                "median_peak": round(float(r["median_peak"]), 4),
                "frac_above_086": round(float(r["frac_above_086"]), 4),
            }
            for r in km["legB"]
        ],
        "leg_a_prime_median_peak": round(float(km["legA_prime"]["median_peak"]), 4),
        "sigma0_gate": {
            "loglog_slope_full_range": round(float(km["loglog_slope"]), 3),
            "loglog_slope_last_decade": 1.99,
            "coefficient_limit": 3.0,
            "verdict": "PASSES",
            "chip": "C7_README §2",
        },
        "rail_threshold": {
            "artifact_value": 0.256,
            "artifact_chip": "C7_README §1 / CLAIM C7 / ADJUDICATION:148",
            "recomputed_from_quoted_law": round(eps_solved, 4),
            "note": (
                "FLAG-2: solving the artifacts' own corrected law for h_eff = 0.86 at "
                "h_true = 0.73 gives 0.2644, not 0.256. Recorded, not reconciled — see "
                "book/design/flags/ch07_FLAGS.md."
            ),
        },
        "hosts": {
            "n": int(len(z_true)),
            "z_true": [round(float(v), 6) for v in z_true],
            "eps_indicative": [round(float(v), 5) for v in eps_ind],
            "peak_indicative": [round(float(v), 5) for v in peak_ind],
            "rel_dL": [float(f"{v:.5g}") for v in rel_dl],
            "point_peak": [round(float(v), 6) for v in km["point_peak"]],
            "quartiles_eps": [round(float(v), 4) for v in np.percentile(eps_ind, [25, 50, 75])],
            "median_rel_dL": round(float(np.median(rel_dl)), 6),
            "frac_eps_above_threshold": round(float(np.mean(eps_ind > 0.256)), 4),
            "frac_peak_above_086": round(float(np.mean(peak_ind > 0.86)), 4),
            "median_peak_indicative": round(float(np.median(peak_ind)), 4),
            "staleness_caveat": (
                "INDICATIVE ONLY: the local reduced_galaxy_catalogue.csv is not the #53 "
                "realization parent — it differs in exactly the z_error column (#40b PV "
                "width). The 2×2 cell B is the staleness-free magnitude check — and it "
                "ran; see resolved_by_cellB below."
            ),
            # MN-2 (expert B): the caveat used to end on the question.  It now
            # carries the answer, because the noscript reader gets this file's
            # numbers and nothing else.
            "resolved_by_cellB": {
                "date": "2026-07-31",
                "lcat_rail_frac": 0.907,
                "n": "68/75",
                "comparison_scattered": 0.892,
                "comparison_idealised_estimator": 0.053,
                "combined_rail_B": 0.697,
                "combined_rail_C": 0.579,
                "incat_class_argmax": 0.860,
                "jobs": "evaluate 6103219 / combine 6103220",
                "chip": "CELLB_READOUT_20260731.md",
                "honest_nuance": (
                    "The indicative (stale) z_error column predicted 75/76 = 98.7% of "
                    "hosts peaking above 0.86; the staleness-free measurement gives "
                    "90.7%. These are NOT the same statistic (reconstructed unclipped "
                    "single-host peak vs delivered clipped L_cat argmax), so it is not "
                    "a contradiction — the honest reading is that the staleness caveat "
                    "resolves in the confirming direction with the delivered rail "
                    "somewhat weaker than the stale column implied, never '98.7% "
                    "confirmed'."
                ),
            },
        },
        "event_889": {
            "crb_row": 889,
            "incat_index": idx_889,
            "z_true": round(float(z_true[idx_889]), 6),
            "rel_dL_recomputed": float(f"{rel_dl[idx_889]:.5g}"),
            # D1 (author mandate, 2026-07-31): the spec figure was the ABSOLUTE
            # sigma_dL in Gpc carried under a fractional label.  The book now
            # prints one corrected value; the retired figure survives only
            # inside the erratum string below (and in the flag file).
            "sigma_dL_Gpc": float(f"{rel_dl[idx_889] * d_l_889:.5g}"),
            "d_L_Gpc": float(f"{d_l_889:.6g}"),
            "rel_dL_erratum": (
                "Erratum: the spec card carried σ_dL/dL = 8.0×10⁻⁵ — that is the absolute "
                "σ_dL in Gpc under a fractional label. Corrected book-wide 2026-07-31; "
                "record: ch01 flag F1 / BUILD_REPORT §5.1 item 1. Measured on CRB row 889 "
                "of seed61000 (identical in all six copies, and equal to the rel_dL the "
                "project's own C7 driver stores for this host): σ_dL = 7.98e-05 Gpc, "
                "σ_dL/d_L = 8.98e-04. ch07 FLAG-1 is thereby RESOLVED."
            ),
            "eps_indicative": round(float(eps_ind[idx_889]), 4),
            "eps_rank_desc": int((eps_ind > eps_ind[idx_889]).sum()) + 1,
            "peak_indicative": round(float(peak_ind[idx_889]), 4),
            "point_peak": round(float(km["point_peak"][idx_889]), 6),
        },
        "production_confrontation": {
            "sigma_glob_dln_073_086": round(float(vp["sigma_glob_dln_073_086"]), 6),
            "observed_incat_dln": [round(float(v), 5) for v in observed],
            "observed_median": round(float(np.median(observed)), 4),
            "observed_iqr": [round(float(v), 4) for v in np.percentile(observed, [25, 75])],
            "observed_frac_positive": round(float(np.mean(observed > 0)), 4),
            "predicted": [
                {
                    "eps": float(r["eps"]),
                    "median": round(float(r["median"]), 4),
                    "p25": round(float(r["p25"]), 4),
                    "p75": round(float(r["p75"]), 4),
                    "frac_positive": round(float(r["frac_positive"]), 4),
                }
                for r in vp["predicted"]
            ],
            "point_kernel_median_nats": -408,
            "point_kernel_frac_positive": 0.0,
            "point_kernel_p5_p95": [-4064, -10],
            "chip": "C7_README §3",
        },
        "orientation": {
            "lcat_incat_median_argmax": 0.860,
            "lcat_incat_frac_at_086": 0.892,
            "lcat_incat_n": "66/74",
            "lcat_share_of_incat_mixture_at_073": 0.963,
            "lcat_dark_median_argmax": 0.600,
            "chip": "C7_README §0",
        },
        "g2b_side": {
            "verdict": "CONFIRMED",
            "statement": (
                "w_pop = (dV_c/dz)/(1+z) without p_det is the unique population weight "
                "consistent with the project's own rate model and with every selection "
                "integral; exactly h-invariant (Z_g ∝ h⁻³ to 1e-15); reduces to the point "
                "kernel as σ_z → 0."
            ),
            "gate": "h-independence protected by a binding regression gate",
            "chip": "G2b:413-436 / ledger #75",
        },
        "conflict": {
            "name": "G2b ↔ C7",
            "register_item": "BOOK_SOURCES_MAP.md §7 item 1",
            # MN-1 (expert B): this string is read by a future grepper, so it
            # carries the landed values, not the pre-registration's tense.
            # Job-ID split rule (worklist D3): the pre-registration keeps
            # 6101146/6101147; the RESULT cites the resubmission 6103219/6103220.
            "decider": (
                "the 2×2 cell B (PREREGISTRATION_2x2_cellB.md, registered 2026-07-30 "
                "as jobs 6101146/6101147) — LANDED 2026-07-31, reported from evaluate "
                "6103219 / combine 6103220: catalogue-leg per-event argmax at the top "
                "of the prior for 68/75 in-catalogue events (90.7%) with EXACT host "
                "redshifts, against 66/74 (89.2%) scattered and 5.3% under the "
                "idealised estimator; in-catalogue class argmax 0.860 as registered; "
                "combined-leg rail 69.7% (B) vs 57.9% (C). CELLB_READOUT_20260731.md."
            ),
            "decider_scope": (
                "Cell B settles C7's MAGNITUDE and ATTRIBUTION, not the G2b↔C7 "
                "collision: that is a derivation-level conflict no posterior can "
                "settle. G2b's CONFIRMED verdict is untouched."
            ),
            "post_cellB_constraint": (
                "New constraint delivered by the readout (CELLB_READOUT_20260731.md "
                "§Next steps 1b): the C7 kernel fix must explicitly supersede G2b AND "
                "must not be the historically-exonerated 'p_det inside the numerator "
                "alone' form. The fix stays author-gated under /physics-change."
            ),
            "binding_rule": (
                "A C7 fix must explicitly supersede G2b and must not silently contradict "
                "it. Neither side may be presented as settled without the other."
            ),
            "historical_opposite_sign": (
                "the measured historical failure mode of the deconvolution at large σ_z/z "
                "was OVER-correction (ledger #62/#68) — the opposite sign to the direction "
                "a C7 fix pushes"
            ),
        },
    }


def build_specz_rescue() -> dict[str, Any]:
    km = json.loads((GATE_B / "c7_kernel_measure_results.json").read_text())
    eps_ind = np.asarray(km["eps_indicative"], float)
    return {
        "_provenance": {
            "artifacts": [
                "BIAS_HISTORY_LEDGER.md row 42 (REFUTED)",
                "docs/F4_SPECZ_DECOMPOSITION.md:8-15",
                "H0R:1268",
                "results/campaign51_20260728/IDEALIZED_BASELINE_READOUT.md:50-52",
            ]
        },
        "ledger_42": {
            "hypothesis": "Spec-z host subsets carry the informative posterior shape",
            "verdict": "REFUTED",
            "specz_fraction_of_glade": 0.0056,
            "specz_share_of_rateweighted_incat_likelihood_max": 0.087,
            "specz_share_median": 0.0,
            "inference_side_cut_result_map": 0.870,
            "cut_description": "inference-side flag == 3 (spec-z only) cut",
        },
        "venue_median_eps_readout": 0.49,
        "venue_median_eps_readout_chip": "IDEALIZED_BASELINE_READOUT.md:50-52",
        "venue_median_eps_c7": round(float(np.median(eps_ind)), 4),
        "venue_median_eps_c7_chip": "c7_kernel_measure (binned-median z_error/z relation)",
        "n_hosts": int(len(eps_ind)),
        "n_hosts_below_threshold": int((eps_ind <= 0.256).sum()),
        "eps_indicative": [round(float(v), 5) for v in eps_ind],
        "widths": {
            "sigma_v_corrected_km_s": 150,
            "sigma_v_uncorrected_km_s": 500,
            "chip": "hostz_pv_photoz_kernel.md §3.1 (RATIFY-1/RATIFY-2, RATIFIED 2026-07-26)",
            "pv_vs_gw_ratio": 2.3,
            "zfloor_vs_gw_ratio": 4.9,
            "retained_gw_sigma_dl_over_dl": 0.0054,
            "pv_median": 0.0125,
            "zfloor_median": 0.0267,
            "golden_width_degradation_pv": 3.3,
            "golden_width_degradation_pv_plus_floor": 6.8,
            "dominance_chip": "hostz_pv_photoz_kernel.md §0",
        },
        "point_kernel_licence": {
            "statement": "The δ-kernel is not 'optimistic' under scatter, it is wrong.",
            "chip": "realistic_host_observation_model.md §3.1 (RATIFY-R3)",
            "leverage_1d": 0.853,
            "leverage_2d": 0.867,
            "leverage_chip": "ledger #88 / THREEWAY_AB_READOUT.md:19-56",
        },
        "anti_anchor": {
            "claim": "sigma_z = 0.013(1+z)^3 attributed to Gray et al.",
            "verdict": "REJECTED — it exactly matches this repo's own dead code",
            "code": "datamodels/galaxy.py:66 (GitHub #7)",
            "chip": "hostz_pv_photoz_kernel.md §2 (anti-anchor)",
        },
    }


def build_volume_trunc() -> dict[str, Any]:
    nodes: dict[str, list[float]] = {}
    weights: dict[str, list[float]] = {}
    for n in VT_NODE_COUNTS:
        if n == 50:
            x, w = np.asarray(_GL_NODES_50, float), np.asarray(_GL_WEIGHTS_50, float)
        else:
            x, w = roots_legendre(n)
        nodes[str(n)] = [float(f"{v:.9g}") for v in x]
        weights[str(n)] = [float(f"{v:.9g}") for v in w]

    return {
        "_provenance": {
            "artifacts": [
                "results/volume_trunc_ab_20260712/FINDING.md:1-58 (FALSIFIED)",
                "BIAS_HISTORY_LEDGER.md row 70",
                "realistic_host_observation_model.md §3.2 (kernel table: volume_trunc FALSIFIED)",
            ],
            "code": ["bayesian_statistics.py — the production 50-node Gauss-Legendre rule"],
            "note": (
                "nodes['50']/weights['50'] ARE the production _GL_NODES_50/_GL_WEIGHTS_50 "
                "imported from bayesian_statistics; the other orders come from "
                "scipy.special.roots_legendre."
            ),
        },
        "window": list(VT_WINDOW),
        "peak_sigma": VT_PEAK_SIGMA,
        "peak_centre_range": [0.030, 0.090],
        "node_counts": VT_NODE_COUNTS,
        "nodes": nodes,
        "weights": weights,
        "recorded": {
            "finding_table": VT_FINDING_TABLE,
            "ab_result": {
                "1d_deconv_mean": 0.7450,
                "1d_trunc_mean": 0.8000,
                "2d_deconv_mean": 0.7681,
                "2d_trunc_mean": 0.8000,
                "delta_1d": 0.0549,
                "delta_2d": 0.0319,
                "residual_targeted": 0.013,
                "wrong_way_factor": 4,
                "n_events": 494,
                "venue": "seed600 shallow venue, 494-event subsample",
            },
            "two_causes": [
                "fixed_quad(n=50) aliases the narrow GW peak inside the wide host window "
                "(0.0000 vs exact 0.24-0.65)",
                "even the EXACT host-window numerator tilts monotonically high in h "
                "(0.24 -> 0.65 over h = 0.60 -> 0.86)",
            ],
            "status": "experimental / FALSIFIED mode; not wired into the CLI; do not revive",
        },
    }


# --------------------------------------------------------------------------- #
def _gate_g3(c7: dict[str, Any]) -> tuple[bool, float]:
    """Corrected C7 law vs the delivered per-host median peaks (Leg A)."""
    worst = 0.0
    for row in c7["leg_a"]:
        if row["eps"] > 0.5:  # C7_README claims <1% agreement up to eps = 0.5
            continue
        worst = max(worst, abs(row["law_frac_shift"] - row["measured_frac_shift"]))
    return worst < 0.005, worst


def _gate_g4() -> str:
    """Optional: recompute the observed in-cat tilt from the diagnostics CSV.

    The diagnostics directory is not tracked in every worktree, so this gate is
    reported but never written into the JSON (determinism).
    """
    rel = Path("results/campaign51_20260728/realistic_20260729/seed61000")
    # The diagnostics/ directory is untracked churn: it exists in the primary
    # checkout but not in every worktree.  Look in this repo first, then in a
    # sibling checkout of the same repo (relative, never an absolute path).
    candidates = [REPO_ROOT / rel]
    if REPO_ROOT.name.endswith("-book"):
        candidates.append(REPO_ROOT.parent / REPO_ROOT.name[: -len("-book")] / rel)
    seed_dir = next((c for c in candidates if (c / "real_r1" / "diagnostics").is_dir()), None)
    if seed_dir is None:
        return "G4 observed-tilt recomputation: SKIPPED (diagnostics CSV not in this worktree)"
    csv = seed_dir / "real_r1" / "diagnostics" / "event_likelihoods.csv"
    import pandas as pd

    crb = pd.read_csv(seed_dir / "prepared_cramer_rao_bounds.csv")
    incat = set(crb.index[crb.host_galaxy_index >= 0])
    df = pd.read_csv(csv)
    piv = df.pivot(index="event_idx", columns="h", values="L_cat_no_bh")
    sub = piv.loc[[i for i in sorted(incat) if i in piv.index]]
    a, b = sub[0.73].to_numpy(), sub[0.86].to_numpy()
    mask = (a > 0) & (b > 0)
    vp = json.loads((GATE_B / "c7_vs_production_results.json").read_text())
    dln = np.log(b[mask]) - np.log(a[mask]) + vp["sigma_glob_dln_073_086"]
    stored = np.asarray(vp["observed_incat_dln"], float)
    if len(dln) != len(stored):
        return f"G4: length mismatch ({len(dln)} vs {len(stored)}) — INVESTIGATE"
    err = float(np.abs(np.sort(dln) - np.sort(stored)).max())
    return (
        f"G4 observed-tilt recomputation from diagnostics: max|Δ| = {err:.2e} "
        f"(median {np.median(dln):+.4f}, frac>0 {np.mean(dln > 0):.4f}) — "
        f"{'PASS' if err < 1e-12 else 'FAIL'}"
    )


def _write(name: str, payload: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=False))
    print(f"  wrote {path.relative_to(REPO_ROOT)}  ({path.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    print("gen_ch07: Chapter 7 — A Redshift Is Not a Number")

    edd = build_eddington()
    c7 = build_c7()
    spec = build_specz_rescue()
    vt = build_volume_trunc()

    # ---- reproduction gates -------------------------------------------------
    print("\n  reproduction gates (re-measuring published tables, not re-deriving):")
    g1_ok = True
    for row in edd["gate_g1_exact_shift_table"]:
        rel = abs(row["measured_exact"] - row["quoted_exact"]) / max(row["quoted_exact"], 1e-12)
        g1_ok &= rel < 0.01
        print(
            f"    G1 sigma_z={row['sigma_z']:.3f}  exact quoted {row['quoted_exact']:+.5f}"
            f"  measured {row['measured_exact']:+.5f}  ({rel:.1%})"
        )
    print(f"    G1 (G2b §2.3 exact-shift table at z_g=0.05): {'PASS' if g1_ok else 'FAIL'}")

    g2_ok = True
    for row in edd["gate_g2_amplitude_table"]:
        rel = abs(row["measured_C"] - row["quoted_C"]) / row["quoted_C"]
        g2_ok &= rel < 0.01
    print(f"    G2 (G2b §2.3 amplitude table C(z_bar)): {'PASS' if g2_ok else 'FAIL'}")

    g3_ok, worst = _gate_g3(c7)
    print(
        f"    G3 (C7 corrected law vs delivered peaks, worst |Δ| = {worst:.4f}): "
        f"{'PASS' if g3_ok else 'FAIL'}"
    )
    print("    " + _gate_g4())

    print("\n  flags — book/design/flags/ch07_FLAGS.md:")
    ev = c7["event_889"]
    print(
        f"    FLAG-1 sigma_dL(EMRI-889): RESOLVED by the D1 mandate 2026-07-31 — "
        f"absolute {ev['sigma_dL_Gpc']:.3g} Gpc / d_L {ev['d_L_Gpc']:.4g} Gpc "
        f"=> fractional {ev['rel_dL_recomputed']:.3g} (the retired spec figure was "
        f"the absolute value under a fractional label)"
    )
    print(
        f"    FLAG-2 rail threshold: artifact 0.256 vs law-solved "
        f"{c7['rail_threshold']['recomputed_from_quoted_law']} — still OPEN, "
        f"not reconciled"
    )
    print(
        f"    cell B (2×2) LANDED 2026-07-31: catalogue-leg rail "
        f"{c7['hosts']['resolved_by_cellB']['lcat_rail_frac']:.3f} "
        f"({c7['hosts']['resolved_by_cellB']['n']}) vs "
        f"{c7['hosts']['resolved_by_cellB']['comparison_scattered']:.3f} scattered"
    )

    print("\n  outputs:")
    _write("ch07_eddington.json", edd)
    _write("ch07_c7.json", c7)
    _write("ch07_speczrescue.json", spec)
    _write("ch07_volumetrunc.json", vt)


if __name__ == "__main__":
    main()
