"""B8.1 [CAL] — F5 information floor at the PRODUCTION venue (1D + 2D).

Direct Fisher/Cramer-Rao floor computation using the ACTUAL production per-event
Cramer-Rao-bound data (seed61000/prepared_cramer_rao_bounds.csv, the CRB/event set
shared -- verified md5 -- by both HEAD-readout venues iiib and joint_r1;
MEASUREMENT_HEAD_READOUT_20260827.md section 1.1). N=1588 events after the SAME
production filters (SNR>=20, fractional distance error <10%; bayesian_statistics.py
:3998, :386).

Two independent routes, both reported:
  (A) NUMERIC route (primary): per-event marginal log-likelihood L_i(h) is built by
      numerically integrating the exact (non-linearized) single-true-host model over
      a fine z-grid around the event's own redshift, using the EXACT production
      distance law d_L(z,h)=dist(z,1)/h (physical_relations.dist_vectorized, the
      production ΛCDM: OMEGA_M=0.2726, OMEGA_DE=0.7274 -- constants.py). The photo-z
      kernel N(z; z_true_i, sigma_z) and (for 2D) the host-mass anchor kernel
      N(M_z_meas_i; M_g_i*(1+z), sigma_Mz_i^2+(sigma_M*M_g_i*(1+z))^2) mirror
      bayesian_statistics.py's with_bh_mass numerator term and the F5 engine's
      _accumulate() (sigma_z_sigma_M_forecast.py:329-337) EXACTLY, restricted to the
      single injected host (no impostor catalog, no selection denominator -- see
      caveats). Fisher info I_i = -d^2 lnL_i/dh^2 at h_true via a 3-point stencil.
  (B) CLOSED-FORM route (cross-check / the presentation-gate derivation): the local
      linear-Gaussian marginalization of the same model, an algebraic Schur-complement
      Fisher matrix (see B8_1_CAL_FLOOR_RECORD.md section 1). Uses a numeric dD/dz via
      central finite difference on the same dist_vectorized (no extra dependency).

Both use each event's ACTUAL (sigma_dL, SNR, M_z, sigma_Mz) straight from the CRB
Fisher matrix -- nothing about the GW-side measurement is assumed or drawn; only the
host photo-z error sigma_z and host photo-mass error sigma_M are the "realistic draw"
external inputs (GLADE / Reines&Volonteri15 literature values, not measured per-event
in this codebase).

Builder note (standing rule 2 -- verifier independence): this script is the
INSTRUMENT. It is smoke-tested here (a builder may smoke-test only) but the
registered numbers in B8_1_CAL_FLOOR_RECORD.md must be reproduced by re-running this
script, not by trusting this docstring.

Run:
  uv run python results/campaign51_20260728/realistic_20260729/fanout1_20260829/b8_information_floor.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from darksiren_emri.constants import H, OMEGA_DE, OMEGA_M, SNR_THRESHOLD  # noqa: E402
from darksiren_emri.physical_relations import dist_vectorized  # noqa: E402

H_TRUE = H  # 0.73, constants.py -- the injected truth for this production venue
FRAC_DL_ERR_THRESHOLD = 0.10  # bayesian_statistics.py:386, FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD

CRB_PATH = (
    REPO_ROOT
    / "results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv"
)
OUT_DIR = Path(__file__).resolve().parent

# --- realistic operating points (literature-sourced, not fit here) ---------
SIGMA_Z_GLADE_PHOTO = 0.035  # GLADE+ flag-1 photometric median (sigma_z_sigma_M_forecast.py:72)
SIGMA_Z_GLADE_SPEC = 0.0017  # GLADE+ flag-3 spectroscopic median (…:73)

# host BH-mass fractional error sigma_M (linear CV), from
# docs/MASS_RELATION_ASSESSMENT.md section 2 (Reines & Volonteri 2015 sec 4.1 scatter):
SIGMA_M_TABLE = {
    "F5_threshold_2pct": 0.02,  # the F5 "useful" boundary at GLADE sigma_z (informational anchor)
    "code_fit_only_0.08dex": 0.19,  # current code: fit-parameter error only (under-estimate)
    "intrinsic_floor_0.24dex": 0.60,  # R&V15 intrinsic scatter floor
    "measurement_0.50dex": 1.66,  # R&V15 virial measurement error
    "total_predictive_0.55dex": 1.99,  # R&V15 total predictive rms (the realistic number)
}


def load_events() -> dict[str, np.ndarray]:
    df = pd.read_csv(CRB_PATH)
    d_meas = df["luminosity_distance"].to_numpy(dtype=np.float64)
    sigma_dL = np.sqrt(
        df["delta_luminosity_distance_delta_luminosity_distance"].to_numpy(dtype=np.float64)
    )
    Mz_meas = df["M"].to_numpy(dtype=np.float64)
    sigma_Mz = np.sqrt(df["delta_M_delta_M"].to_numpy(dtype=np.float64))
    snr = df["SNR"].to_numpy(dtype=np.float64)
    frac_err = sigma_dL / d_meas
    mask = (snr >= SNR_THRESHOLD) & (frac_err < FRAC_DL_ERR_THRESHOLD)
    n_before = len(df)
    out = {
        "d_meas": d_meas[mask],
        "sigma_dL": sigma_dL[mask],
        "Mz_meas": Mz_meas[mask],
        "sigma_Mz": sigma_Mz[mask],
        "snr": snr[mask],
        "frac_dL_err": frac_err[mask],
    }
    print(
        f"[load] {n_before} rows -> {mask.sum()} events after SNR>={SNR_THRESHOLD} "
        f"& frac_dL_err<{FRAC_DL_ERR_THRESHOLD} (production cuts)",
        flush=True,
    )
    return out


def d_to_z(d_gpc: np.ndarray, *, h: float, z_max: float = 3.0, n: int = 200_000) -> np.ndarray:
    """Vectorized distance->redshift inversion by monotone-table interpolation."""
    z_table = np.linspace(1e-5, z_max, n)
    d_table = dist_vectorized(z_table, h=h)
    return np.interp(d_gpc, d_table, z_table)


def numeric_fisher(
    ev: dict[str, np.ndarray],
    z_i: np.ndarray,
    *,
    sigma_z: float,
    sigma_M: float | None,
    dh: float = 0.005,
    n_z: int = 400,
    n_sigma_window: float = 7.0,
) -> np.ndarray:
    """Route A: per-event I_i = -d^2 lnL_i/dh^2 at h_true, exact nonlinear z-marginal.

    Data are fixed at the noiseless self-consistent point (d_data=dist(z_i,h_true),
    M_z data = ev['Mz_meas']); only the ACTUAL per-event (sigma_dL, sigma_Mz) and the
    stipulated (sigma_z, sigma_M) set the widths. No impostor catalog, no selection
    denominator -- see caveats in the companion .md.
    """
    N = len(z_i)
    d_data = dist_vectorized(z_i, h=H_TRUE)
    Mg_i = ev["Mz_meas"] / (1.0 + z_i)

    zmin = np.maximum(z_i - n_sigma_window * sigma_z, 1e-6)
    zmax = z_i + n_sigma_window * sigma_z
    t = np.linspace(0.0, 1.0, n_z)
    zg = zmin[:, None] + (zmax - zmin)[:, None] * t[None, :]
    dz = (zmax - zmin) / (n_z - 1)
    Dshape = dist_vectorized(zg.ravel(), h=1.0).reshape(zg.shape)  # dist(z,1)=D(z), d_L(z,h)=D(z)/h

    photo_term = -0.5 * ((zg - z_i[:, None]) / sigma_z) ** 2
    if sigma_M is not None:
        mu = Mg_i[:, None] * (1.0 + zg)
        sig2 = ev["sigma_Mz"][:, None] ** 2 + (sigma_M * mu) ** 2
        mass_term = -0.5 * (ev["Mz_meas"][:, None] - mu) ** 2 / sig2 - 0.5 * np.log(sig2)
    else:
        mass_term = 0.0

    hs = (H_TRUE - dh, H_TRUE, H_TRUE + dh)
    logL = np.zeros((N, 3))
    for k, h in enumerate(hs):
        dL_model = Dshape / h
        gw_term = -0.5 * ((d_data[:, None] - dL_model) / ev["sigma_dL"][:, None]) ** 2
        integrand = photo_term + gw_term + mass_term
        m = np.max(integrand, axis=1, keepdims=True)
        L = np.sum(np.exp(integrand - m), axis=1) * dz
        logL[:, k] = np.log(np.maximum(L, 1e-300)) + m[:, 0]

    d2 = (logL[:, 2] - 2.0 * logL[:, 1] + logL[:, 0]) / dh**2
    I_i = np.clip(-d2, 0.0, None)
    return I_i


def closed_form_fisher(
    ev: dict[str, np.ndarray],
    z_i: np.ndarray,
    *,
    sigma_z: float,
    sigma_M: float | None,
    eps: float = 1e-4,
) -> np.ndarray:
    """Route B: local linear-Gaussian Schur-complement Fisher info (analytic form).

    I_i = d_data_i^2 / [ H_TRUE^2 * sigma_dL_i^2 + (dD/dz|_{z_i})^2 * sigma_z_eff_i^2 ]
    with 1/sigma_z_eff_i^2 = 1/sigma_z^2 + 1/(sigma_M*(1+z_i))^2 (2D) or = 1/sigma_z^2 (1D).
    See B8_1_CAL_FLOOR_RECORD.md section 1 for the derivation (F_hh, F_hz, F_zz Schur
    complement of the exact nonlinear model d_L(z,h)=D(z)/h -- only the sigma_dL^2 term
    picks up the H_TRUE^2 Jacobian factor; an earlier draft of this function multiplied
    BOTH terms by H_TRUE^2, overstating info by up to 1/H_TRUE^2 ~= 1.9x in the
    photo-z-dominated regime -- fixed before this was registered).
    """
    d_data = dist_vectorized(z_i, h=H_TRUE)
    dDdz = (dist_vectorized(z_i + eps, h=1.0) - dist_vectorized(z_i - eps, h=1.0)) / (2 * eps)
    inv_sz2 = 1.0 / sigma_z**2
    if sigma_M is not None:
        sigma_z_mass = sigma_M * (1.0 + z_i)
        inv_sz2 = inv_sz2 + 1.0 / sigma_z_mass**2
    sigma_z_eff2 = 1.0 / inv_sz2
    denom = (H_TRUE**2) * ev["sigma_dL"] ** 2 + dDdz**2 * sigma_z_eff2
    return d_data**2 / denom


def summarize(I: np.ndarray, label: str) -> dict:
    """Aggregate + concentration diagnostics.

    The naive aggregate floor sigma_h_floor = (sum I_i)^-1/2 is reported, but per-event
    Fisher info I_i for these production CRBs is extremely heavy-tailed (a handful of
    very-nearby, very-loud events carry orders of magnitude more single-host info than
    a typical event -- see 'frac_top10'/'n_eff' below). n_eff = (sum I)^2/sum(I^2) is the
    number of *equally-informative typical events* the aggregate is worth (Kish effective
    sample size); 'sigma_h_floor_typical' = (N * median(I))^-1/2 rescales the MEDIAN
    event's info to the full N as a robust-to-outliers companion number.
    """
    I_total = float(np.sum(I))
    sigma_floor = float(1.0 / np.sqrt(I_total)) if I_total > 0 else float("inf")
    N = len(I)
    med_I = float(np.median(I))
    sigma_floor_typical = float(1.0 / np.sqrt(N * med_I)) if med_I > 0 else float("inf")
    order = np.argsort(-I)
    sum_I2 = float(np.sum(I**2))
    n_eff = float(I_total**2 / sum_I2) if sum_I2 > 0 else float("nan")
    return {
        "label": label,
        "n_events": int(N),
        "I_total": I_total,
        "sigma_h_floor": sigma_floor,
        "sigma_h_floor_over_h": sigma_floor / H_TRUE,
        "median_event_I": med_I,
        "sigma_h_floor_typical": sigma_floor_typical,
        "n_eff_events": n_eff,
        "frac_I_from_top10": float(np.sum(I[order[:10]]) / I_total) if I_total > 0 else float("nan"),
        "frac_I_from_top50": float(np.sum(I[order[:50]]) / I_total) if I_total > 0 else float("nan"),
    }


def main() -> None:
    t0 = time.time()
    ev = load_events()
    z_i = d_to_z(ev["d_meas"], h=H_TRUE)
    print(
        f"[z] median z={np.median(z_i):.4f}  p10={np.percentile(z_i, 10):.4f}  "
        f"p90={np.percentile(z_i, 90):.4f}  max={np.max(z_i):.4f}",
        flush=True,
    )
    print(
        f"[dL frac err] median={np.median(ev['frac_dL_err']):.4f}  "
        f"p10={np.percentile(ev['frac_dL_err'], 10):.4f}  "
        f"p90={np.percentile(ev['frac_dL_err'], 90):.4f}",
        flush=True,
    )
    frac_Mz = ev["sigma_Mz"] / ev["Mz_meas"]
    print(
        f"[GW mass frac err, sigma_Mz/Mz] median={np.median(frac_Mz):.3e}  "
        f"p90={np.percentile(frac_Mz, 90):.3e}  (negligible vs host-mass sigma_M below)",
        flush=True,
    )

    results: dict = {
        "meta": {
            "crb_path": str(CRB_PATH.relative_to(REPO_ROOT)),
            "h_true": H_TRUE,
            "omega_m": OMEGA_M,
            "omega_de": OMEGA_DE,
            "snr_threshold": SNR_THRESHOLD,
            "frac_dL_err_threshold": FRAC_DL_ERR_THRESHOLD,
            "n_events": int(len(z_i)),
            "z_median": float(np.median(z_i)),
            "z_p10": float(np.percentile(z_i, 10)),
            "z_p90": float(np.percentile(z_i, 90)),
            "frac_dL_err_median": float(np.median(ev["frac_dL_err"])),
        },
        "sigma_z_grid": {"GLADE_photo": SIGMA_Z_GLADE_PHOTO, "GLADE_spec": SIGMA_Z_GLADE_SPEC},
        "sigma_M_table": SIGMA_M_TABLE,
        "oneD": {},
        "twoD": {},
        "dh_convergence": {},
    }

    for sz_name, sz in results["sigma_z_grid"].items():
        I_num = numeric_fisher(ev, z_i, sigma_z=sz, sigma_M=None)
        I_cf = closed_form_fisher(ev, z_i, sigma_z=sz, sigma_M=None)
        results["oneD"][sz_name] = {
            "sigma_z": sz,
            "numeric": summarize(I_num, f"1D numeric sigma_z={sz}"),
            "closed_form": summarize(I_cf, f"1D closed-form sigma_z={sz}"),
        }
        num = results["oneD"][sz_name]["numeric"]
        cf = results["oneD"][sz_name]["closed_form"]
        print(
            f"[1D {sz_name} sz={sz}] numeric floor={num['sigma_h_floor']:.6f} "
            f"typical={num['sigma_h_floor_typical']:.6f} n_eff={num['n_eff_events']:.1f} "
            f"top10={num['frac_I_from_top10']:.2f} | closed_form floor={cf['sigma_h_floor']:.6f} "
            f"typical={cf['sigma_h_floor_typical']:.6f} n_eff={cf['n_eff_events']:.1f}",
            flush=True,
        )

    for sz_name, sz in results["sigma_z_grid"].items():
        results["twoD"][sz_name] = {}
        for sm_name, sm in SIGMA_M_TABLE.items():
            I_num = numeric_fisher(ev, z_i, sigma_z=sz, sigma_M=sm)
            I_cf = closed_form_fisher(ev, z_i, sigma_z=sz, sigma_M=sm)
            results["twoD"][sz_name][sm_name] = {
                "sigma_z": sz,
                "sigma_M": sm,
                "numeric": summarize(I_num, f"2D numeric sigma_z={sz} sigma_M={sm}"),
                "closed_form": summarize(I_cf, f"2D closed-form sigma_z={sz} sigma_M={sm}"),
            }
            num = results["twoD"][sz_name][sm_name]["numeric"]
            cf = results["twoD"][sz_name][sm_name]["closed_form"]
            print(
                f"[2D {sz_name}/{sm_name} sz={sz} sM={sm}] numeric floor={num['sigma_h_floor']:.6f} "
                f"typical={num['sigma_h_floor_typical']:.6f} | closed_form floor={cf['sigma_h_floor']:.6f} "
                f"typical={cf['sigma_h_floor_typical']:.6f}",
                flush=True,
            )

    # dh convergence check on the headline cell (1D, GLADE photo-z) and (2D, total predictive)
    for dh in (0.002, 0.005, 0.01, 0.02):
        I_1d = numeric_fisher(ev, z_i, sigma_z=SIGMA_Z_GLADE_PHOTO, sigma_M=None, dh=dh)
        I_2d = numeric_fisher(
            ev,
            z_i,
            sigma_z=SIGMA_Z_GLADE_PHOTO,
            sigma_M=SIGMA_M_TABLE["total_predictive_0.55dex"],
            dh=dh,
        )
        results["dh_convergence"][str(dh)] = {
            "1D_sigma_h_floor": summarize(I_1d, "conv")["sigma_h_floor"],
            "2D_sigma_h_floor": summarize(I_2d, "conv")["sigma_h_floor"],
        }
        print(
            f"[dh={dh}] 1D floor={results['dh_convergence'][str(dh)]['1D_sigma_h_floor']:.6f}  "
            f"2D floor={results['dh_convergence'][str(dh)]['2D_sigma_h_floor']:.6f}",
            flush=True,
        )

    # Measured HEAD-readout posterior summaries, hard-coded with citation (A11): these are
    # NOT recomputed here -- they are quoted verbatim from
    # head_readout_extraction_20260827.md section "Results table" (rows iiib/joint_r1,
    # 1D/2D), the mechanical extraction of the same run_metadata_21.json evaluate sweep
    # this script's CRB input feeds.
    results["measured_head_readout_20260827"] = {
        "source": "results/campaign51_20260728/realistic_20260729/head_readout_extraction_20260827.md#results-table",
        "iiib": {
            "1D": {"mean_h": 0.6077, "sigma_h": 0.00845, "offset_mean_minus_truth": -0.1223, "map": 0.600},
            "2D": {"mean_h": 0.6634, "sigma_h": 0.01833, "offset_mean_minus_truth": -0.0666, "map": 0.6629},
        },
        "joint_r1": {
            "1D": {"mean_h": 0.6143, "sigma_h": 0.01147, "offset_mean_minus_truth": -0.1157, "map": 0.600},
            "2D": {"mean_h": 0.6630, "sigma_h": 0.01861, "offset_mean_minus_truth": -0.0670, "map": 0.6616},
        },
    }

    results["elapsed_s"] = time.time() - t0
    out_path = OUT_DIR / "b8_information_floor.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[done] {results['elapsed_s']:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
