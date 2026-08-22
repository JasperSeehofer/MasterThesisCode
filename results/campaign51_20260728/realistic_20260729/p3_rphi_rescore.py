r"""[P3-RPHI] slot-correction rescore -- zero-evaluate(), registered in CLAIM_P3_RPHI_20260822.md.

``cat_term_corrected(e,h) = cat_term_off(e,h) / r_phi(h)`` with r_phi = Sigma^phi/Sigma^3D from
the committed leaves at every H_GRID_41 node; mixture reassembled via the verified identity;
scored by the committed ``compute_seed_statistics`` (trapezoid).
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from darksiren_emri.bayesian_inference.bayesian_statistics import (
    precompute_global_catalog_selection,
    precompute_phi_marginal_survival,
    precompute_phi_selection_integrals,
)
from darksiren_emri.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)
from darksiren_emri.constants import SNR_THRESHOLD
from darksiren_emri.galaxy_catalogue.pixel_completeness import from_cache_or_build
from darksiren_emri.validation.correspondence_1d import (
    H_GRID_41,
    HOST_DRAW_Z_MAX,
    INJECTION_POOL_DIR,
    REDUCED_CATALOGUE_PATH,
    _load_galaxy_catalog_handler,
    compute_seed_statistics,
)

BASE = Path(__file__).parent
OUT_PATH = BASE / "p3_rphi_rescore_output.json"
REGISTRATION_SECTION = (
    "results/campaign51_20260728/realistic_20260729/CLAIM_P3_RPHI_20260822.md, "
    "Stage 2 -- RESCORE REGISTRATION (2026-08-22)"
)
SEEDS = list(range(900101, 900113))
GATE_IS_TOL = 2e-6
GATE_TC_TOL = 2e-6
HEADLINE_BIAS_ANCHOR = -0.108302
GATE_BS_TOL = 1e-5


def _a22_stamp() -> dict[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "darksiren_emri/", str(Path(__file__))],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"git_commit_at_start": commit, "tree_dirty_incl_instrument": dirty or "clean"}


def _banked_csv(seed: int) -> Path:
    return Path(
        "results/prod2d_closure_20260818/arm_event_likelihoods/"
        f"bsel_seed{seed}/seed{seed}/simulations/diagnostics/event_likelihoods.csv"
    )


def _gate_is(df: pd.DataFrame, label: str) -> float:
    recon = (
        df["alpha_G_phi"].to_numpy()
        / df["r_Malm"].to_numpy()
        * df["L_cat_no_bh"].to_numpy()
        / df["D_tilde_phi"].to_numpy()
        + df["B_num"].to_numpy() / df["D_tilde_phi"].to_numpy()
    )
    ref = df["combined_no_bh"].to_numpy()
    rel = np.abs(recon - ref) / np.maximum(np.abs(ref), np.finfo(float).tiny)
    max_rel = float(np.nanmax(rel))
    if max_rel > GATE_IS_TOL:
        raise SystemExit(f"GATE I-S FAILED ({label}): {max_rel:.3e} -- fail closed (A17(f))")
    return max_rel


def main() -> int:
    stamp = _a22_stamp()
    print("A22 stamp:", stamp)
    h_values = [float(h) for h in H_GRID_41]

    handler = _load_galaxy_catalog_handler(REDUCED_CATALOGUE_PATH)
    completeness = from_cache_or_build()
    det = SimulationDetectionProbability(
        injection_data_dir=INJECTION_POOL_DIR,
        snr_threshold=SNR_THRESHOLD,
        dl_bins=60,
        mass_bins=40,
        estimator="local_linear",
        expected_z_max=HOST_DRAW_Z_MAX,
        allow_shallow_pool=True,
        pdet_z_resolved=True,
    )
    phi_table = precompute_phi_marginal_survival(
        h_values=h_values, detection_probability_obj=det, z_max_cap=HOST_DRAW_Z_MAX
    )
    _, beta_gbar_phi = precompute_phi_selection_integrals(
        h_values=h_values, phi_survival_table=phi_table, completeness=completeness
    )
    sigma_3d = precompute_global_catalog_selection(
        h_values=h_values,
        galaxy_catalog=handler,
        detection_probability_obj=det,
        with_bh_mass=False,
        z_max_cap=HOST_DRAW_Z_MAX,
        smear_sigma_z=False,
    )
    sigma_phi = precompute_global_catalog_selection(
        h_values=h_values,
        galaxy_catalog=handler,
        detection_probability_obj=det,
        with_bh_mass=False,
        z_max_cap=HOST_DRAW_Z_MAX,
        smear_sigma_z=False,
        phi_survival_table=phi_table,
    )
    r_phi = {h: sigma_phi[h] / sigma_3d[h] for h in h_values}

    # GATE S-R + rebuild-consistency (T-C form).
    vals = [r_phi[h] for h in h_values]
    if not all(0.8 < v < 1.0 for v in vals) or not all(
        vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1)
    ):
        raise SystemExit(f"GATE S-R FAILED: r_phi range [{min(vals)}, {max(vals)}]")
    tc_max = 0.0
    df0 = pd.read_csv(_banked_csv(SEEDS[0]))
    one = df0[df0["event_idx"] == df0["event_idx"].iloc[0]]
    for _, row in one.iterrows():
        h = float(row["h"])
        if h not in beta_gbar_phi:
            continue
        col = float(row["D_tilde_phi"] - row["alpha_G_phi"])
        tc_max = max(tc_max, abs(beta_gbar_phi[h] - col) / abs(col))
    if tc_max > GATE_TC_TOL:
        raise SystemExit(f"REBUILD-CONSISTENCY FAILED: {tc_max:.3e}")
    print(f"gates: S-R PASS (r_phi [{min(vals):.6f}, {max(vals):.6f}]), T-C PASS ({tc_max:.3e})")

    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        banked = pd.read_csv(_banked_csv(seed))
        _gate_is(banked, f"banked {seed}")
        w = (
            banked["alpha_G_phi"].to_numpy()
            / banked["r_Malm"].to_numpy()
            / banked["D_tilde_phi"].to_numpy()
        )
        cat_off = w * banked["L_cat_no_bh"].to_numpy()
        inv_r = banked["h"].map(lambda h: 1.0 / r_phi.get(float(h), np.nan)).to_numpy(np.float64)
        keep = ~np.isnan(inv_r)
        combined_corr = np.where(
            keep,
            banked["combined_no_bh"].to_numpy() - cat_off + cat_off * inv_r,
            banked["combined_no_bh"].to_numpy(),
        )
        patched = banked.copy()
        patched["combined_no_bh"] = combined_corr
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "event_likelihoods.csv"
            patched.to_csv(p, index=False)
            st_c = compute_seed_statistics(str(p), seed, h_grid=H_GRID_41)
        st_b = compute_seed_statistics(str(_banked_csv(seed)), seed, h_grid=H_GRID_41)
        per_seed.append(
            {
                "seed": seed,
                "mean_h_corrected": float(st_c.mean_h),
                "mean_h_banked_trapezoid": float(st_b.mean_h),
                "delta_s": float(st_c.mean_h) - float(st_b.mean_h),
            }
        )
        print(f"seed {seed}: delta_s = {per_seed[-1]['delta_s']:+.6f}")

    base = np.array([r["mean_h_banked_trapezoid"] for r in per_seed])
    if abs(float(base.mean() - 0.73) - HEADLINE_BIAS_ANCHOR) > GATE_BS_TOL:
        raise SystemExit("baseline gate FAILED")
    d = np.array([r["delta_s"] for r in per_seed])
    out = {
        "registered_in": REGISTRATION_SECTION,
        "a22_stamp": stamp,
        "r_phi": {str(h): r_phi[h] for h in h_values},
        "sigma_phi": {str(h): sigma_phi[h] for h in h_values},
        "sigma_3d": {str(h): sigma_3d[h] for h in h_values},
        "rebuild_consistency_max_rel": tc_max,
        "primary": {
            "delta_bar": float(d.mean()),
            "sd": float(d.std(ddof=1)),
            "sem_paired": float(d.std(ddof=1) / np.sqrt(len(d))),
            "n_positive": int((d > 0).sum()),
            "per_seed": per_seed,
            "reference": (
                f"{REGISTRATION_SECTION}; subtracts the banked trapezoid mean_h "
                f"(gated on the headline anchor {HEADLINE_BIAS_ANCHOR}, disclosed form)"
            ),
        },
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(
        f"RPHI: delta_bar = {d.mean():+.6f} +/- {d.std(ddof=1) / np.sqrt(len(d)):.6f} "
        f"(sd {d.std(ddof=1):.6f}, {int((d > 0).sum())}/12 positive)"
    )
    print("wrote", OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
