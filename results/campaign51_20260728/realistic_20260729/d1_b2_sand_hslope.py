#!/usr/bin/env python
"""B2 measurement-before-gate instrument for the D1 investigation
(`CLAIM_D1_P0WINDOW_20260805.md.DRAFT`, stage-1 §B2).

Re-runs the exact selection-scalar computation of `cand_b_joint_selection.py`
(`.planning/derivation-2dbias-fix-20260803/fixb_x15_attribution/`) -- same two
derived injection pools (SNR-kept `S` vs `SNR:=0`-on-p0-reject `S_and`), same
production `SimulationDetectionProbability` machinery, same class definitions
(in-catalogue leg via the reduced GLADE+ catalogue mass-density sum; dark leg
via the phi-mass-function quadrature) -- but swept over h instead of pinned at
h=0.73, to answer: is s_G/s_D h-dependent?

Read-only w.r.t. the source tree. Does NOT modify `cand_b_joint_selection.py`.
Writes only to this directory (`d1_b2_sand_hslope.json`) and to a scratch pool
cache under `$CAND_B_SCRATCH` (default `/tmp/cand_b_pools_b2`, kept separate
from the original instrument's `/tmp/cand_b_pools` cache to avoid any chance of
cross-contamination).
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

from master_thesis_code.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)
from master_thesis_code.constants import (
    HOST_DRAW_Z_MAX,
    M_SOURCE_FRAME_MAX,
    M_SOURCE_FRAME_MIN,
    SNR_THRESHOLD,
)
from master_thesis_code.dark_siren_injection import _redshift_population_weight
from master_thesis_code.emri_rate import R_eff_per_mbh, mbh_mass_function
from master_thesis_code.galaxy_catalogue.handler import (
    REDUCED_CATALOGUE_FILE_PATH,
    CatalogueColumns,
    InternalCatalogColumns,
    _empiric_stellar_mass_to_BH_mass_relation,
    _mass_redshift_prune_mask,
    _reduced_catalog_column_names,
)
from master_thesis_code.galaxy_catalogue.pixel_completeness import from_cache_or_build
from master_thesis_code.physical_relations import dist_to_redshift, dist_vectorized

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/jasper/Repositories/MasterThesisCode"
POOL = (
    "results/campaign51_20260728/realistic_20260729/gate_b_20260730/"
    "injection_pool_mix200k_20260728"
)
SCRATCH = os.environ.get("CAND_B_SCRATCH", "/tmp/cand_b_pools_b2")
P0_LO, P0_HI = 10.0 + 2e-3, 16.0 - 2e-3
Z_MAX_PRUNE = 1.5
CHUNK = 2_000_000

# Pin-reproduction check value (C4 of the claim file, verbatim).
PIN_H = 0.73
PIN_S_G = 0.2286246597604769
PIN_S_D = 0.3129747690740832
PIN_RATIO = 0.7304891075943567
PIN_DL_MAX = 9.164987215485882

# The canonical 41-point h grid used throughout the post-fix gate-vii campaign
# (`results/run_20260804_postfix/gate_vii/gate_vii_readout.json`
# `/iiib_idealized/per_h_profile_fixed_survivors/dark/h`), non-uniform
# 0.01 / 0.005 (0.65-0.695, 0.795 skipped) / 0.01 spacing.
H_GRID_41 = [
    0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.655, 0.66, 0.665, 0.67, 0.675, 0.68,
    0.685, 0.69, 0.695, 0.7, 0.705, 0.71, 0.715, 0.72, 0.725, 0.73, 0.735,
    0.74, 0.745, 0.75, 0.755, 0.76, 0.765, 0.77, 0.775, 0.78, 0.785, 0.79,
    0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86,
]
H_PROBE = [0.60, 0.73, 0.86]

FLAGS = dict(
    snr_threshold=SNR_THRESHOLD,
    dl_bins=60,
    mass_bins=40,
    estimator="local_linear",
    expected_z_max=HOST_DRAW_Z_MAX,
    pdet_z_resolved=True,
    pdet_wbh_z_resolved=False,
)


def build_pools() -> tuple[str, str, int, int]:
    """Build (or reuse cached) S / S_and derived injection pools. Identical
    logic to `cand_b_joint_selection.py:69-95` (same P0 window, same 60-file
    no-p0-column drop rule)."""
    os.makedirs(SCRATCH, exist_ok=True)
    dir_S = os.path.join(SCRATCH, "pool_p0kept")
    dir_A = os.path.join(SCRATCH, "pool_p0window")
    n_in = n_out = 0
    if os.path.isdir(dir_S) and os.path.isdir(dir_A) and os.listdir(dir_S):
        for f in glob.glob(os.path.join(dir_S, "*.csv")):
            n_in += len(pd.read_csv(f, usecols=["p0"]))
        return dir_S, dir_A, n_in, -1  # n_out unknown on cache reuse, not needed downstream
    for d in (dir_S, dir_A):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    for f in sorted(glob.glob(os.path.join(ROOT, POOL, "*.csv"))):
        df = pd.read_csv(f)
        if "p0" not in df.columns:
            continue
        df = df[df.p0.notna()]
        if len(df) == 0:
            continue
        n_in += len(df)
        df.to_csv(os.path.join(dir_S, os.path.basename(f)), index=False)
        bad = (df.p0 < P0_LO) | (df.p0 > P0_HI)
        n_out += int(bad.sum())
        df2 = df.copy()
        df2.loc[bad, "SNR"] = 0.0
        df2.to_csv(os.path.join(dir_A, os.path.basename(f)), index=False)
    print(f"[pools] rows kept {n_in}, p0-rejected {n_out} ({n_out / n_in:.4%})")
    return dir_S, dir_A, n_in, n_out


def S4(dp: SimulationDetectionProbability, dL: np.ndarray, Mz: np.ndarray, h: float) -> np.ndarray:
    z0 = np.zeros_like(dL)
    return np.asarray(
        dp.detection_probability_with_bh_mass_interpolated(dL, Mz, z0, z0, h=h), dtype=np.float64
    )


def load_catalogue() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int]:
    """Load + prune the reduced GLADE+ catalogue once (h-independent mass/z
    prune mask -- identical to `cand_b_joint_selection.py:166-206`).
    Returns (z_all, M_all, w_all, W_cat, n_raw, n_mass)."""
    names = _reduced_catalog_column_names()
    zs, Ms = [], []
    n_raw = n_mass = 0
    t0 = time.time()
    for chunk in pd.read_csv(REDUCED_CATALOGUE_FILE_PATH, names=names, chunksize=CHUNK):
        chunk = chunk.rename(
            columns={
                CatalogueColumns.RIGHT_ASCENSION.name: InternalCatalogColumns.PHI_S,
                CatalogueColumns.DECLINATION.name: InternalCatalogColumns.THETA_S,
            }
        )
        n_raw += len(chunk)
        M, dM = _empiric_stellar_mass_to_BH_mass_relation(
            chunk[InternalCatalogColumns.BH_MASS], chunk[InternalCatalogColumns.BH_MASS_ERROR]
        )
        chunk[InternalCatalogColumns.BH_MASS] = M
        chunk[InternalCatalogColumns.BH_MASS_ERROR] = dM
        chunk = chunk[~chunk[InternalCatalogColumns.BH_MASS].isna()]
        n_mass += len(chunk)
        keep = _mass_redshift_prune_mask(
            chunk[InternalCatalogColumns.BH_MASS],
            chunk[InternalCatalogColumns.BH_MASS_ERROR],
            chunk[InternalCatalogColumns.REDSHIFT],
            chunk[InternalCatalogColumns.REDSHIFT_ERROR],
            M_SOURCE_FRAME_MIN,
            M_SOURCE_FRAME_MAX,
            Z_MAX_PRUNE,
        )
        chunk = chunk[keep]
        if len(chunk) == 0:
            continue
        zs.append(chunk[InternalCatalogColumns.REDSHIFT].to_numpy(dtype=np.float64))
        Ms.append(chunk[InternalCatalogColumns.BH_MASS].to_numpy(dtype=np.float64))
    z_all = np.concatenate(zs)
    M_all = np.concatenate(Ms)
    w_all = np.asarray(R_eff_per_mbh(M_all), dtype=np.float64) / (1.0 + z_all)
    print(f"[cat] raw={n_raw} with-mass={n_mass} pruned={z_all.size}  ({time.time() - t0:.1f}s)")
    draw_elig = z_all < HOST_DRAW_Z_MAX
    W_cat = float(np.sum(w_all[draw_elig]))
    return z_all, M_all, w_all, W_cat, n_raw, n_mass


def measure_at_h(
    h: float,
    dp_S: SimulationDetectionProbability,
    dp_A: SimulationDetectionProbability,
    Mg: np.ndarray,
    lg: np.ndarray,
    phi: np.ndarray,
    comp,
    z_all: np.ndarray,
    M_all: np.ndarray,
    w_all: np.ndarray,
) -> dict:
    dp_S._get_or_build_grid(h)
    dp_A._get_or_build_grid(h)
    dl_max_S = float(dp_S.get_dl_max(h))
    dl_max_A = float(dp_A.get_dl_max(h))

    # --- dark leg (phi quadrature) -- identical structure to lines 132-163 ---
    z_max = min(dist_to_redshift(dl_max_S, h=h), HOST_DRAW_Z_MAX)
    zq = np.linspace(1e-6, z_max, 1200)
    dLq = np.asarray(dist_vectorized(zq, h=h), dtype=np.float64)
    ZZ, MM = np.meshgrid(zq, Mg, indexing="ij")
    DD = np.repeat(dLq[:, None], Mg.size, axis=1)
    Mz_flat = (MM * (1.0 + ZZ)).ravel()
    S_flat = S4(dp_S, DD.ravel(), Mz_flat, h).reshape(DD.shape)
    A_flat = S4(dp_A, DD.ravel(), Mz_flat, h).reshape(DD.shape)
    Sbar = np.trapezoid(S_flat * phi[None, :], lg, axis=1)
    Abar = np.trapezoid(A_flat * phi[None, :], lg, axis=1)
    ppq = np.asarray(_redshift_population_weight(zq, h), dtype=np.float64)
    fbq = np.clip(np.asarray(comp.f_bar(zq, h), dtype=np.float64), 0.0, 1.0)
    beta_phi = float(np.trapezoid((1.0 - fbq) * Sbar * ppq, zq))
    beta_phi_and = float(np.trapezoid((1.0 - fbq) * Abar * ppq, zq))
    s_D = beta_phi_and / beta_phi

    # --- in-cat leg -- identical structure to lines 205-250 ------------------
    elig = z_all < min(dist_to_redshift(dl_max_S, h=h), Z_MAX_PRUNE)
    idx = np.flatnonzero(elig)
    S_sum = A_sum = 0.0
    for start in range(0, idx.size, CHUNK):
        sl = idx[start : start + CHUNK]
        z_g, M_g, w_g = z_all[sl], M_all[sl], w_all[sl]
        d_L = np.asarray(dist_vectorized(z_g, h=h), dtype=np.float64)
        Mz = M_g * (1.0 + z_g)
        p_s = S4(dp_S, d_L, Mz, h)
        p_a = S4(dp_A, d_L, Mz, h)
        S_sum += float(np.sum(w_g * p_s))
        A_sum += float(np.sum(w_g * p_a))
    s_G = A_sum / S_sum
    ratio = s_G / s_D

    return dict(
        h=h,
        dl_max_S=dl_max_S,
        dl_max_and=dl_max_A,
        beta_Gbar_4D_phi=beta_phi,
        beta_Gbar_4D_phi_and=beta_phi_and,
        s_dark=s_D,
        Sigma4D=S_sum,
        Sigma4D_and=A_sum,
        s_incat=s_G,
        ratio_sG_sD=ratio,
        ln_ratio=float(np.log(ratio)),
    )


def main() -> None:
    t_start = time.time()
    dir_S, dir_A, n_in, n_out = build_pools()

    dp_S = SimulationDetectionProbability(injection_data_dir=dir_S, **FLAGS)
    dp_A = SimulationDetectionProbability(injection_data_dir=dir_A, **FLAGS)

    comp = from_cache_or_build()
    lg = np.linspace(np.log10(M_SOURCE_FRAME_MIN), np.log10(M_SOURCE_FRAME_MAX), 400)
    Mg = 10.0**lg
    phi = np.asarray(mbh_mass_function(Mg), dtype=np.float64) * np.asarray(
        R_eff_per_mbh(Mg), dtype=np.float64
    )
    phi = phi / np.trapezoid(phi, lg)

    z_all, M_all, w_all, W_cat, n_raw, n_mass = load_catalogue()

    # ---------------------------------------------------------- pin check ---
    pin = measure_at_h(PIN_H, dp_S, dp_A, Mg, lg, phi, comp, z_all, M_all, w_all)
    pin_check = dict(
        h=PIN_H,
        measured_s_G=pin["s_incat"],
        pin_s_G=PIN_S_G,
        rel_dev_s_G=pin["s_incat"] / PIN_S_G - 1.0,
        measured_s_D=pin["s_dark"],
        pin_s_D=PIN_S_D,
        rel_dev_s_D=pin["s_dark"] / PIN_S_D - 1.0,
        measured_ratio=pin["ratio_sG_sD"],
        pin_ratio=PIN_RATIO,
        rel_dev_ratio=pin["ratio_sG_sD"] / PIN_RATIO - 1.0,
        measured_dl_max_S=pin["dl_max_S"],
        pin_dl_max=PIN_DL_MAX,
        rel_dev_dl_max=pin["dl_max_S"] / PIN_DL_MAX - 1.0,
    )
    print("\n[PIN CHECK]", json.dumps(pin_check, indent=1))
    pin_reproduced = (
        abs(pin_check["rel_dev_ratio"]) < 1e-3 and abs(pin_check["rel_dev_dl_max"]) < 1e-6
    )
    if not pin_reproduced:
        print("\n*** PIN NOT REPRODUCED -- STOPPING per task instructions ***")
        out = dict(
            status="PIN_MISMATCH_STOPPED",
            pin_check=pin_check,
            pool_fingerprint=dict(n_files=len(glob.glob(os.path.join(ROOT, POOL, "*.csv"))), n_rows_kept=n_in),
        )
        json.dump(out, open(os.path.join(HERE, "d1_b2_sand_hslope.json"), "w"), indent=1)
        raise SystemExit(1)

    # -------------------------------------------------------- probe h's -----
    probe_results = [
        measure_at_h(h, dp_S, dp_A, Mg, lg, phi, comp, z_all, M_all, w_all)
        if h != PIN_H
        else pin
        for h in H_PROBE
    ]
    print("\n[PROBE 3-h]")
    for r in probe_results:
        print(f"  h={r['h']:.2f}  s_G={r['s_incat']:.10f}  s_D={r['s_dark']:.10f}  "
              f"ratio={r['ratio_sG_sD']:.10f}  ln_ratio={r['ln_ratio']:.10f}")

    dln_dh_probe = (probe_results[-1]["ln_ratio"] - probe_results[0]["ln_ratio"]) / (
        probe_results[-1]["h"] - probe_results[0]["h"]
    )

    # -------------------------------------------------------- full 41-grid --
    grid_results = []
    for h in H_GRID_41:
        if h == PIN_H:
            grid_results.append(pin)
            continue
        match = next((r for r in probe_results if r["h"] == h), None)
        if match is not None:
            grid_results.append(match)
            continue
        r = measure_at_h(h, dp_S, dp_A, Mg, lg, phi, comp, z_all, M_all, w_all)
        grid_results.append(r)
        print(f"  h={r['h']:.3f}  s_G={r['s_incat']:.10f}  s_D={r['s_dark']:.10f}  "
              f"ratio={r['ratio_sG_sD']:.10f}  ln_ratio={r['ln_ratio']:.10f}")

    ln_ratios = np.array([r["ln_ratio"] for r in grid_results])
    hs = np.array([r["h"] for r in grid_results])
    # finite-difference slope, central differences on the non-uniform grid
    dln_dh_fd = np.gradient(ln_ratios, hs)
    # end-to-end slope across the whole grid
    dln_dh_endpoints = (ln_ratios[-1] - ln_ratios[0]) / (hs[-1] - hs[0])
    delta_ln_ratio_full_grid = float(ln_ratios.max() - ln_ratios.min())
    delta_ln_ratio_endpoints = float(ln_ratios[-1] - ln_ratios[0])

    out = dict(
        status="OK",
        wall_time_s=time.time() - t_start,
        pool_fingerprint=dict(
            n_files=len(glob.glob(os.path.join(ROOT, POOL, "*.csv"))),
            n_rows_kept_p0_present=n_in,
            n_rows_dropped_p0_reject=n_out,
        ),
        pin_check=pin_check,
        pin_reproduced=bool(pin_reproduced),
        probe_h=[0.60, 0.73, 0.86],
        probe_results=probe_results,
        dln_ratio_dh_probe_60_86=float(dln_dh_probe),
        grid41_results=grid_results,
        grid41_dln_ratio_dh_fd=dln_dh_fd.tolist(),
        grid41_dln_ratio_dh_endpoints=float(dln_dh_endpoints),
        grid41_delta_ln_ratio_max_minus_min=delta_ln_ratio_full_grid,
        grid41_delta_ln_ratio_endpoints_60_to_86=delta_ln_ratio_endpoints,
        band_0p5pct_in_ln=0.005,
        h_flat_verdict_full_grid=bool(delta_ln_ratio_full_grid < 0.005),
        h_flat_verdict_endpoints=bool(abs(delta_ln_ratio_endpoints) < 0.005),
    )
    json.dump(out, open(os.path.join(HERE, "d1_b2_sand_hslope.json"), "w"), indent=1)
    print(f"\nwrote d1_b2_sand_hslope.json  (wall time {out['wall_time_s']:.1f}s)")
    print(f"\n[VERDICT] delta ln(ratio) over full 41-h grid = {delta_ln_ratio_full_grid:.6e} "
          f"(band 0.005) -> h-flat: {out['h_flat_verdict_full_grid']}")
    print(f"[VERDICT] delta ln(ratio) endpoints 0.60->0.86 = {delta_ln_ratio_endpoints:.6e} "
          f"-> h-flat: {out['h_flat_verdict_endpoints']}")


if __name__ == "__main__":
    main()
