r"""[P3-IMP] SHAPE-ONLY arm -- h_ref-anchored zero-evaluate() rescore.

Registered in ``PREREGISTRATION_P3_TWIN_20260822.md`` "SHAPE-ONLY ARM --
REGISTRATION" (2026-08-22, row #163 item 3). Per event and node:
``L_cat_shape(h) = L_cat_phi(h) * [L_cat_off(h_ref)/L_cat_phi(h_ref)]`` with
h_ref = 0.73; the mixture is reassembled via the verified decompose identity
and scored through the COMMITTED ``compute_seed_statistics`` (trapezoid, the
convention of record per A20 amendment 4). Gates I-S / N-S / B-S; A22 stamp.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from darksiren_emri.validation.correspondence_1d import H_GRID_41, compute_seed_statistics

BASE = Path(__file__).parent
OUT_PATH = BASE / "p3_shape_rescore_output.json"
REGISTRATION_SECTION = (
    "results/campaign51_20260728/realistic_20260729/PREREGISTRATION_P3_TWIN_20260822.md, "
    "SHAPE-ONLY ARM -- REGISTRATION (2026-08-22, row #163 item 3)"
)
SEEDS = list(range(900101, 900113))
H_REF = 0.73
H_REF_SENSITIVITY = (0.70, 0.76)
GATE_IS_TOL = 2e-6
GATE_NS_TOL = 1e-12
HEADLINE_BIAS_ANCHOR = -0.108302  # A20 amendment 4: banked trapezoid fleet bias
GATE_BS_TOL = 1e-5
DELTA_PHI_REREFERENCED = 0.015524  # A20 amendment 4 primary of record


def _a22_stamp() -> dict[str, str]:
    """A22: provenance stamped at run START."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "darksiren_emri/"], capture_output=True, text=True
    ).stdout.strip()
    return {"git_commit_at_start": commit, "estimator_tree_dirty": dirty or "clean"}


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


def _cat_term(df: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        df["alpha_G_phi"].to_numpy()
        / df["r_Malm"].to_numpy()
        * df["L_cat_no_bh"].to_numpy()
        / df["D_tilde_phi"].to_numpy()
    )


def _gate_is(df: pd.DataFrame, label: str) -> float:
    """GATE I-S: combined_no_bh == cat_term + B_num/D_tilde to <= 2e-6 rel."""
    recon = _cat_term(df) + df["B_num"].to_numpy() / df["D_tilde_phi"].to_numpy()
    ref = df["combined_no_bh"].to_numpy()
    rel = np.abs(recon - ref) / np.maximum(np.abs(ref), np.finfo(float).tiny)
    max_rel = float(np.nanmax(rel))
    if max_rel > GATE_IS_TOL:
        raise SystemExit(f"GATE I-S FAILED ({label}): max rel {max_rel:.3e} > {GATE_IS_TOL}")
    return max_rel


def rescore_seed(seed: int, h_ref: float) -> dict[str, Any]:
    banked = pd.read_csv(_banked_csv(seed))
    phi = pd.read_csv(_phi_csv(seed))
    is_b = _gate_is(banked, f"banked {seed}")
    is_p = _gate_is(phi, f"phi {seed}")

    m = banked.merge(
        phi[["event_idx", "h", "L_cat_no_bh"]],
        on=["event_idx", "h"],
        suffixes=("", "_phi"),
        how="inner",
        validate="one_to_one",
    )
    if len(m) != len(banked):
        raise SystemExit(f"seed {seed}: key mismatch banked {len(banked)} vs merged {len(m)}")

    ref_rows = m[np.isclose(m["h"], h_ref)][["event_idx", "L_cat_no_bh", "L_cat_no_bh_phi"]]
    ref = ref_rows.set_index("event_idx")
    off_ref = ref["L_cat_no_bh"].reindex(m["event_idx"]).to_numpy()
    phi_ref = ref["L_cat_no_bh_phi"].reindex(m["event_idx"]).to_numpy()

    zero_ref = phi_ref <= 0.0
    if not np.array_equal(zero_ref, off_ref <= 0.0):
        raise SystemExit(f"seed {seed}: phi/off zero sets differ at h_ref (GATE N-S premise)")
    factor = np.where(zero_ref, 0.0, off_ref / np.where(zero_ref, 1.0, phi_ref))
    l_shape = m["L_cat_no_bh_phi"].to_numpy() * factor

    # GATE N-S: at h_ref the shape column equals the off column exactly.
    at_ref = np.isclose(m["h"].to_numpy(), h_ref)
    ns = np.abs(l_shape[at_ref] - m.loc[at_ref, "L_cat_no_bh"].to_numpy())
    ns_max = float(np.max(ns / np.maximum(m.loc[at_ref, "L_cat_no_bh"].to_numpy(), 1e-300)))
    if ns_max > GATE_NS_TOL:
        raise SystemExit(f"GATE N-S FAILED seed {seed}: {ns_max:.3e}")

    cat_off = _cat_term(m)
    cat_shape = (
        m["alpha_G_phi"].to_numpy() / m["r_Malm"].to_numpy() * l_shape / m["D_tilde_phi"].to_numpy()
    )
    combined_shape = m["combined_no_bh"].to_numpy() - cat_off + cat_shape

    patched = banked.copy()
    patched["combined_no_bh"] = combined_shape
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "event_likelihoods.csv"
        patched.to_csv(p, index=False)
        st_shape = compute_seed_statistics(str(p), seed, h_grid=H_GRID_41)
    st_base = compute_seed_statistics(str(_banked_csv(seed)), seed, h_grid=H_GRID_41)
    return {
        "seed": seed,
        "h_ref": h_ref,
        "gate_is_max_rel": max(is_b, is_p),
        "gate_ns_max_rel": ns_max,
        "mean_h_shape": float(st_shape.mean_h),
        "mean_h_banked_trapezoid": float(st_base.mean_h),
        "delta_s": float(st_shape.mean_h) - float(st_base.mean_h),
    }


def main() -> int:
    stamp = _a22_stamp()
    print("A22 stamp:", stamp)
    results: dict[str, Any] = {"h_ref_primary": {}, "h_ref_sensitivity": {}}

    primary_rows = [rescore_seed(s, H_REF) for s in SEEDS]
    base = np.array([r["mean_h_banked_trapezoid"] for r in primary_rows])
    fleet_base_bias = float(base.mean() - 0.73)
    if abs(fleet_base_bias - HEADLINE_BIAS_ANCHOR) > GATE_BS_TOL:
        raise SystemExit(
            f"GATE B-S FAILED: baseline fleet bias {fleet_base_bias:.6f} != "
            f"headline anchor {HEADLINE_BIAS_ANCHOR}"
        )
    d = np.array([r["delta_s"] for r in primary_rows])
    primary = {
        "per_seed": primary_rows,
        "delta_bar": float(d.mean()),
        "sd": float(d.std(ddof=1)),
        "sem_paired": float(d.std(ddof=1) / np.sqrt(len(d))),
        "n_positive": int((d > 0).sum()),
        "gate_bs_fleet_base_bias": fleet_base_bias,
        "reference": (
            f"{REGISTRATION_SECTION}; subtracts the banked TRAPEZOID mean_h "
            "(A17(e): re-derived through compute_seed_statistics and gated on "
            f"the headline anchor {HEADLINE_BIAS_ANCHOR})"
        ),
    }
    results["h_ref_primary"] = primary
    results["decomposition_report"] = {
        "delta_bar_phi_rereferenced": DELTA_PHI_REREFERENCED,
        "delta_bar_shape": primary["delta_bar"],
        "delta_bar_level_implied": DELTA_PHI_REREFERENCED - primary["delta_bar"],
        "reference": f"{REGISTRATION_SECTION}, 'Decomposition report'",
    }
    for h_ref in H_REF_SENSITIVITY:
        rows = [rescore_seed(s, h_ref) for s in SEEDS]
        dd = np.array([r["delta_s"] for r in rows])
        results["h_ref_sensitivity"][str(h_ref)] = {
            "delta_bar": float(dd.mean()),
            "sem_paired": float(dd.std(ddof=1) / np.sqrt(len(dd))),
        }
    results["a22_stamp"] = stamp
    results["registered_in"] = REGISTRATION_SECTION
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(
        f"SHAPE: delta_bar = {primary['delta_bar']:+.6f} +/- {primary['sem_paired']:.6f} "
        f"(sd {primary['sd']:.6f}, {primary['n_positive']}/12 positive)"
    )
    print(
        "level implied =",
        f"{results['decomposition_report']['delta_bar_level_implied']:+.6f}",
    )
    for k, v in results["h_ref_sensitivity"].items():
        print(f"h_ref {k}: delta_bar {v['delta_bar']:+.6f} +/- {v['sem_paired']:.6f}")
    print("wrote", OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
