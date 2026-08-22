r"""[P3-IMP] COMPLETED-PAIRING arm -- zero-evaluate() rescore of the registered candidate.

Registered in ``PREREGISTRATION_P3_TWIN_20260822.md`` "COMPLETED-PAIRING ARM --
REGISTRATION" (2026-08-22, row #166 item 2(i)). Per event and node:
``cat_term_completed = cat_term_phi * R(h)`` with ``R = beta_G / beta_G_phi``,
both betas from the COMMITTED leaf ``precompute_phi_selection_integrals``
called twice (real S_bar_phi table vs an S_bar==1 table on the same grids).
The venue objects are rebuilt exactly as ``BayesianStatistics.evaluate()``
builds them internally (the ``build_bsel_selection_objects`` construction,
h-list generalized); GATE T-C anchors the rebuild on the banked columns'
``D_tilde_phi - alpha_G_phi`` per h. First-order completion: the Sigma-chain
(n_hat_w, r_Malm, Sigma ratios) is held invariant, per the registration.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from darksiren_emri.bayesian_inference.bayesian_statistics import (
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
    compute_seed_statistics,
)

BASE = Path(__file__).parent
OUT_PATH = BASE / "p3_completed_rescore_output.json"
REGISTRATION_SECTION = (
    "results/campaign51_20260728/realistic_20260729/PREREGISTRATION_P3_TWIN_20260822.md, "
    "COMPLETED-PAIRING ARM -- REGISTRATION (2026-08-22, row #166 item 2(i))"
)
SEEDS = list(range(900101, 900113))
GATE_TC_TOL = 2e-6
HEADLINE_BIAS_ANCHOR = -0.108302  # amendment-8-discharged baseline gate form (disclosed)
GATE_BS_TOL = 1e-5
TWIN = 0.015524133
SHAPE = 0.000570
LEVEL = 0.014954577


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


def _phi_csv(seed: int) -> Path:
    return (
        BASE
        / "p3_work"
        / f"phi_{seed}_work"
        / f"seed{seed}"
        / "simulations"
        / "diagnostics"
        / "event_likelihoods.csv"
    )


def _build_betas(
    h_values: list[float],
) -> tuple[dict[float, float], dict[float, float], dict[float, float], dict[float, float]]:
    """(beta_G_phi, beta_Gbar_phi, beta_G, beta_Gbar) via the committed leaves.

    Venue objects rebuilt exactly as evaluate() builds them internally (the
    build_bsel_selection_objects construction, h-list generalized).
    """
    completeness = from_cache_or_build()
    detection_probability = SimulationDetectionProbability(
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
        h_values=h_values,
        detection_probability_obj=detection_probability,
        z_max_cap=HOST_DRAW_Z_MAX,
    )
    ones_table = {h: (z, np.ones_like(s)) for h, (z, s) in phi_table.items()}
    beta_g_phi, beta_gbar_phi = precompute_phi_selection_integrals(
        h_values=h_values, phi_survival_table=phi_table, completeness=completeness
    )
    beta_g, beta_gbar = precompute_phi_selection_integrals(
        h_values=h_values, phi_survival_table=ones_table, completeness=completeness
    )
    return beta_g_phi, beta_gbar_phi, beta_g, beta_gbar


def main() -> int:
    stamp = _a22_stamp()
    print("A22 stamp:", stamp)
    h_values = [float(h) for h in H_GRID_41]
    beta_g_phi, beta_gbar_phi, beta_g, beta_gbar = _build_betas(h_values)

    # GATE T-C: rebuilt beta_Gbar_phi(h) vs banked D_tilde - alpha per h, all seeds.
    tc_max = 0.0
    for seed in SEEDS:
        df = pd.read_csv(_banked_csv(seed))
        one = df[df["event_idx"] == df["event_idx"].iloc[0]]
        for _, row in one.iterrows():
            h = float(row["h"])
            if h not in beta_gbar_phi:
                continue
            col = float(row["D_tilde_phi"] - row["alpha_G_phi"])
            rel = abs(beta_gbar_phi[h] - col) / max(abs(col), np.finfo(float).tiny)
            tc_max = max(tc_max, rel)
    if tc_max > GATE_TC_TOL:
        raise SystemExit(f"GATE T-C FAILED: max rel {tc_max:.3e} > {GATE_TC_TOL} -- STOP (A21)")
    print(f"GATE T-C PASS: max rel {tc_max:.3e}")

    # GATE S-C: R(h) > 1, catalogue fraction in (0,1); bank the full vectors.
    r_of_h = {h: beta_g[h] / beta_g_phi[h] for h in h_values}
    for h in h_values:
        frac = beta_g[h] / (beta_g[h] + beta_gbar[h])
        if not (r_of_h[h] > 1.0 and 0.0 < frac < 1.0):
            raise SystemExit(f"GATE S-C FAILED at h={h}: R={r_of_h[h]}, frac={frac}")
    print(f"GATE S-C PASS: R(h) in [{min(r_of_h.values()):.4f}, {max(r_of_h.values()):.4f}]")

    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        banked = pd.read_csv(_banked_csv(seed))
        phi = pd.read_csv(_phi_csv(seed))
        m = banked.merge(
            phi[["event_idx", "h", "L_cat_no_bh"]],
            on=["event_idx", "h"],
            suffixes=("", "_phi"),
            how="inner",
            validate="one_to_one",
        )
        if len(m) != len(banked):
            raise SystemExit(f"seed {seed}: key mismatch")
        w = m["alpha_G_phi"].to_numpy() / m["r_Malm"].to_numpy() / m["D_tilde_phi"].to_numpy()
        cat_off = w * m["L_cat_no_bh"].to_numpy()
        r_vec = m["h"].map(lambda h: r_of_h.get(float(h), np.nan)).to_numpy(dtype=np.float64)
        if np.isnan(r_vec).any():
            keep = ~np.isnan(r_vec)
        else:
            keep = np.ones(len(m), dtype=bool)
        cat_completed = w * m["L_cat_no_bh_phi"].to_numpy() * r_vec
        combined_completed = np.where(
            keep,
            m["combined_no_bh"].to_numpy() - cat_off + cat_completed,
            m["combined_no_bh"].to_numpy(),
        )
        patched = banked.copy()
        patched["combined_no_bh"] = combined_completed
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "event_likelihoods.csv"
            patched.to_csv(p, index=False)
            st_c = compute_seed_statistics(str(p), seed, h_grid=H_GRID_41)
        st_b = compute_seed_statistics(str(_banked_csv(seed)), seed, h_grid=H_GRID_41)
        per_seed.append(
            {
                "seed": seed,
                "mean_h_completed": float(st_c.mean_h),
                "mean_h_banked_trapezoid": float(st_b.mean_h),
                "delta_s": float(st_c.mean_h) - float(st_b.mean_h),
            }
        )
        print(f"seed {seed}: delta_s = {per_seed[-1]['delta_s']:+.6f}")

    base = np.array([r["mean_h_banked_trapezoid"] for r in per_seed])
    fleet_base_bias = float(base.mean() - 0.73)
    if abs(fleet_base_bias - HEADLINE_BIAS_ANCHOR) > GATE_BS_TOL:
        raise SystemExit(f"baseline gate FAILED: {fleet_base_bias:.6f}")
    d = np.array([r["delta_s"] for r in per_seed])
    out: dict[str, Any] = {
        "registered_in": REGISTRATION_SECTION,
        "a22_stamp": stamp,
        "gate_tc_max_rel": tc_max,
        "r_of_h": {str(h): r_of_h[h] for h in h_values},
        "primary": {
            "delta_bar": float(d.mean()),
            "sd": float(d.std(ddof=1)),
            "sem_paired": float(d.std(ddof=1) / np.sqrt(len(d))),
            "n_positive": int((d > 0).sum()),
            "per_seed": per_seed,
            "reference": (
                f"{REGISTRATION_SECTION}; subtracts the banked trapezoid mean_h "
                f"(baseline gated on the headline anchor {HEADLINE_BIAS_ANCHOR}, "
                "the amendment-8-discharged form, disclosed)"
            ),
        },
        "decomposition_report": {
            "twin": TWIN,
            "shape": SHAPE,
            "level": LEVEL,
            "completed": float(d.mean()),
            "reference": f"{REGISTRATION_SECTION}, 'Decomposition report'",
        },
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(
        f"COMPLETED: delta_bar = {d.mean():+.6f} +/- {d.std(ddof=1) / np.sqrt(len(d)):.6f} "
        f"(sd {d.std(ddof=1):.6f}, {int((d > 0).sum())}/12 positive)"
    )
    print("wrote", OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
