"""Test 21 (Tier 3): D(h) audit — sensitivity to integration grid + integrand shape.

Empirical observation from A7-redux + A4: at both h_true=0.65 and h_true=0.73,
ΔΣ log L (truth → MAP) is small and pulls toward truth, while Δ(−N log D)
≈ +7.6 and pushes MAP away. The structural bias lives in D(h).

Naive scaling: D(h) ∝ ∫ P_det(d_L(z,h)) · dV_c/dz · dz, with dV_c/dz ∝ 1/h³
and z_max(h) ≈ h-independent (since dl_max(h) ∝ 1/h cancels d_L's 1/h
scaling). So the naive expectation is dlog D/dh ≈ −3/h ≈ −4 to −5 in our
range. But the cluster-computed D(h) shows dlog D/dh ≈ −0.7 to −1.8,
much shallower. Something compensates: presumably P_det(d_L(z,h)) dropping
to zero before z_max contributes most to the integrand. **What we want to
know: is the actual D(h) shape correct, or biased by integration error or
P_det grid pathology at high d_L?**

Audits:
  3b. Quad-order convergence: recompute D(h) at quad_n ∈ {50, 100, 200, 400}
      for h ∈ {0.625, 0.65, 0.675, 0.730, 0.755}. If D(h) is unstable, the
      integration grid is the bug.
  3c. dV_c/dz scaling check: log dV_c/dz vs log h at fixed z; expect slope −3.
  3d. Integrand shape: plot P_det · dV_c/dz vs z for h ∈ {0.65, 0.67, 0.73, 0.755}.
      Identify where the mass is concentrated.
  3e. d_L(z, h) round-trip: dist(dist_to_redshift(d, h), h) ≈ d.

Output: scripts/bias_investigation/outputs/phase45/dh_audit.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import fixed_quad

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from darksiren_emri.bayesian_inference.simulation_detection_probability import (  # noqa: E402
    SimulationDetectionProbability,
)
from darksiren_emri.physical_relations import (  # noqa: E402
    comoving_volume_element,
    dist,
    dist_to_redshift,
    dist_vectorized,
)

INJECTION_DIR = PROJECT_ROOT / "simulations" / "injections"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "outputs" / "phase45"

H_VALUES_AUDIT = [0.625, 0.650, 0.670, 0.700, 0.730, 0.755]
QUAD_ORDERS = [50, 100, 200, 400, 800]
N_PLOT_POINTS = 200


def compute_dh_with_quad_order(
    sdp: SimulationDetectionProbability, h: float, n: int
) -> tuple[float, float, float]:
    """Recompute D(h) at quadrature order n. Returns (D, z_max, dl_max)."""
    dl_max = sdp.get_dl_max(h)
    z_max = dist_to_redshift(dl_max, h=h)
    z_min = 1e-6

    def integrand(z: np.ndarray, _h: float = h) -> np.ndarray:
        d_L = np.asarray(dist_vectorized(z, h=_h), dtype=np.float64)
        phi = np.zeros_like(z)
        theta = np.zeros_like(z)
        p_det = sdp.detection_probability_without_bh_mass_interpolated_zero_fill(
            d_L, phi, theta, h=_h
        )
        dVc = np.atleast_1d(np.asarray(comoving_volume_element(z, h=_h), dtype=np.float64))
        return np.asarray(p_det, dtype=np.float64) * dVc

    D, _ = fixed_quad(integrand, z_min, z_max, n=n)
    return float(D), float(z_max), float(dl_max)


def integrand_shape(
    sdp: SimulationDetectionProbability, h: float, n_pts: int = N_PLOT_POINTS
) -> dict[str, Any]:
    """Sample integrand at n_pts redshift values for diagnostic plotting."""
    dl_max = sdp.get_dl_max(h)
    z_max = dist_to_redshift(dl_max, h=h)
    zs = np.geomspace(1e-4, z_max, n_pts)
    d_L = dist_vectorized(zs, h=h)
    phi = np.zeros_like(zs)
    theta = np.zeros_like(zs)
    p_det = sdp.detection_probability_without_bh_mass_interpolated_zero_fill(d_L, phi, theta, h=h)
    dVc = np.atleast_1d(np.asarray(comoving_volume_element(zs, h=h), dtype=np.float64))
    integrand = np.asarray(p_det) * dVc
    # Where is the integrand mass?
    cumulative = np.cumsum(integrand) * (zs[1] - zs[0])  # rough; not used for D itself
    # Quantiles of integrand mass
    weights = integrand * np.gradient(zs)
    weights_norm = weights / weights.sum()
    cum_w = np.cumsum(weights_norm)
    z_q = {}
    for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
        idx = int(np.searchsorted(cum_w, q))
        z_q[f"q{int(q * 100)}"] = float(zs[min(idx, len(zs) - 1)])
    return {
        "h": h,
        "z_max": float(z_max),
        "dl_max": float(dl_max),
        "z_grid": zs.tolist(),
        "d_L_grid": d_L.tolist(),
        "p_det_grid": p_det.tolist() if hasattr(p_det, "tolist") else list(p_det),
        "dVc_grid": dVc.tolist(),
        "integrand": integrand.tolist(),
        "integrand_quantile_z": z_q,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("TIER 3 — D(h) audit (integration order, integrand shape, scaling)")
    print("=" * 70)

    sdp = SimulationDetectionProbability(
        injection_data_dir=str(INJECTION_DIR),
        snr_threshold=20.0,
    )
    print(f"Pooled injections: {len(sdp._pooled_df)}")

    summary: dict[str, Any] = {
        "h_values_audit": H_VALUES_AUDIT,
        "quad_orders": QUAD_ORDERS,
    }

    # Audit 3b: quad-order convergence
    print("\n--- 3b: quad-order convergence ---")
    print(f"{'h':>6} {'n=50':>14} {'n=100':>14} {'n=200':>14} {'n=400':>14} {'n=800':>14}")
    quad_table: dict[str, dict[str, float]] = {}
    log_d_table: dict[str, dict[int, float]] = {}
    for h in H_VALUES_AUDIT:
        row: dict[int, float] = {}
        for n in QUAD_ORDERS:
            D, _, _ = compute_dh_with_quad_order(sdp, h, n)
            row[n] = D
        quad_table[f"{h:.3f}"] = {f"n={n}": v for n, v in row.items()}
        log_d_table[f"{h:.3f}"] = {n: float(np.log(v)) for n, v in row.items()}
        vals = [row[n] for n in QUAD_ORDERS]
        rel_change_50_to_800 = abs(vals[-1] - vals[0]) / max(abs(vals[-1]), 1e-30)
        print(
            f"{h:>6.3f} "
            + " ".join(f"{v:>14.4e}" for v in vals)
            + f"   rel(50→800)={rel_change_50_to_800:.3e}"
        )

    # Adapt quad-order verdict
    rel_changes = []
    for h_str, row in quad_table.items():
        v50 = row["n=50"]
        v800 = row["n=800"]
        rel = abs(v800 - v50) / max(abs(v800), 1e-30)
        rel_changes.append(rel)
    max_rel_change = float(max(rel_changes))
    if max_rel_change > 1e-3:
        verdict_3b = (
            f"FLAG quad order: max relative D shift n=50→n=800 = {max_rel_change:.2e} "
            "(>1e-3); the production n=100 may be biasing D(h)"
        )
    else:
        verdict_3b = (
            f"PASS quad order: max relative D shift n=50→n=800 = {max_rel_change:.2e} "
            "(<1e-3); n=100 is well-converged"
        )
    print(f"\n>>> 3b verdict: {verdict_3b}")
    summary["3b_quad_table"] = quad_table
    summary["3b_log_d_table"] = log_d_table
    summary["3b_max_rel_change_50_to_800"] = max_rel_change
    summary["3b_verdict"] = verdict_3b

    # 3c: dV_c/dz scaling check at fixed z
    print("\n--- 3c: dV_c/dz scaling check at fixed z ---")
    z_test = 0.1
    dVc_by_h: dict[str, float] = {}
    for h in H_VALUES_AUDIT:
        dVc = float(comoving_volume_element(z_test, h=h))
        dVc_by_h[f"{h:.3f}"] = dVc
    # Fit log(dVc) = a + b log(h); expect b=-3
    h_arr = np.array(H_VALUES_AUDIT)
    dVc_arr = np.array([dVc_by_h[f"{h:.3f}"] for h in H_VALUES_AUDIT])
    slope, intercept = np.polyfit(np.log(h_arr), np.log(dVc_arr), 1)
    print(f"  dVc/dz at z={z_test} for h ∈ {H_VALUES_AUDIT}:")
    for h in H_VALUES_AUDIT:
        print(f"    h={h:.3f}: dVc = {dVc_by_h[f'{h:.3f}']:.4e}")
    print(f"  fit log(dVc) ~ {slope:.4f} log(h) + {intercept:.4f}; expected slope -3.0")
    if abs(slope + 3.0) < 0.05:
        verdict_3c = "PASS dV_c/dz scaling: slope = expected -3 within tol"
    else:
        verdict_3c = f"FLAG dV_c/dz scaling: slope = {slope:.3f}, expected -3"
    print(f"  >>> 3c verdict: {verdict_3c}")
    summary["3c_dVc_at_z01"] = dVc_by_h
    summary["3c_log_slope"] = float(slope)
    summary["3c_verdict"] = verdict_3c

    # 3d: integrand shape at multiple h
    print("\n--- 3d: integrand shape (where is the mass concentrated) ---")
    integrand_data: dict[str, Any] = {}
    for h in [0.625, 0.65, 0.675, 0.700, 0.730, 0.755]:
        shape = integrand_shape(sdp, h, n_pts=N_PLOT_POINTS)
        integrand_data[f"{h:.3f}"] = shape
        z_q = shape["integrand_quantile_z"]
        z_max = shape["z_max"]
        print(
            f"  h={h:.3f}: z_max={z_max:.3f}, "
            f"integrand mass quantiles z[10,25,50,75,90] = "
            f"[{z_q['q10']:.2f}, {z_q['q25']:.2f}, {z_q['q50']:.2f}, "
            f"{z_q['q75']:.2f}, {z_q['q90']:.2f}]"
        )
    summary["3d_integrand_shape"] = integrand_data

    # 3d follow-up: P_det evaluated at d_L(z_max(h), h) — is it 0 there?
    print("\n  P_det at top of integration range (z_max, dl_max):")
    p_det_at_zmax: dict[str, float] = {}
    for h in H_VALUES_AUDIT:
        dl_max = sdp.get_dl_max(h)
        d_L = np.array([dl_max])
        phi = np.zeros(1)
        theta = np.zeros(1)
        p = sdp.detection_probability_without_bh_mass_interpolated_zero_fill(d_L, phi, theta, h=h)
        p_det_at_zmax[f"{h:.3f}"] = float(p[0])
        print(f"    h={h:.3f}: P_det(d_L=dl_max={dl_max:.4f}) = {p[0]:.4f}")
    summary["3d_p_det_at_dl_max"] = p_det_at_zmax

    # 3e: d_L round-trip via dist_to_redshift
    print("\n--- 3e: d_L round-trip ---")
    rt_errors: dict[str, float] = {}
    for h in H_VALUES_AUDIT:
        # Pick a few d_L probes spanning the grid
        dl_probes = [0.5, 2.0, 5.0, 10.0]
        max_err = 0.0
        for dl in dl_probes:
            try:
                z = dist_to_redshift(dl, h=h)
                dl_back = dist(z, h=h)
                err = abs(dl_back - dl) / max(dl, 1e-9)
                max_err = max(max_err, err)
            except Exception as e:  # noqa: BLE001
                print(f"    h={h:.3f} dl={dl}: error {e}")
        rt_errors[f"{h:.3f}"] = max_err
        print(f"  h={h:.3f}: max round-trip rel err = {max_err:.3e}")
    summary["3e_round_trip_errors"] = rt_errors

    # Save
    out_path = OUTPUT_DIR / "dh_audit.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
