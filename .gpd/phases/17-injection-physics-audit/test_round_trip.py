#!/usr/bin/env python3
"""Round-trip test: z -> dist(z, h) -> dist_to_redshift(d_L, h) -> z_recovered.

Tests numerical accuracy of the d_L-to-z inversion for the 7 h-values used
in the injection campaign, across z in [0.001, 0.5].

Phase 17-01 Task 2, deliverable deliv-roundtrip-test.
"""

import sys

import numpy as np

from master_thesis_code.physical_relations import dist, dist_to_redshift

# --- Configuration ---
h_values = [0.60, 0.65, 0.70, 0.73, 0.80, 0.85, 0.90]
z_test = np.linspace(0.001, 0.5, 100)
THRESHOLD = 1e-4

# Speed of light in km/s for Hubble law check
C_KM_S = 299792.458


def main() -> None:
    print("=" * 72)
    print("d_L round-trip accuracy test: z -> dist(z,h) -> dist_to_redshift -> z_rec")
    print("=" * 72)

    all_pass = True
    worst_global_error = 0.0
    worst_global_z = 0.0
    worst_global_h = 0.0

    # --- Test 1: Limiting case dist(z=0, h) = 0 ---
    print("\n--- Test 1: dist(z=0, h) = 0 for all h ---")
    for h in h_values:
        d0 = dist(0.0, h=h)
        status = "PASS" if abs(d0) < 1e-15 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  h={h:.2f}: dist(0, h) = {d0:.2e}  [{status}]")

    # --- Test 2: Round-trip accuracy ---
    print(f"\n--- Test 2: Round-trip accuracy (threshold = {THRESHOLD:.0e}) ---")
    print(f"  {'h':>6s}  {'max_rel_err':>12s}  {'worst_z':>8s}  {'status':>6s}")
    print(f"  {'------':>6s}  {'------------':>12s}  {'--------':>8s}  {'------':>6s}")

    for h in h_values:
        max_err = 0.0
        worst_z = 0.0
        for z in z_test:
            d_L = dist(z, h=h)
            z_rec = dist_to_redshift(d_L, h=h)
            rel_err = abs(z - z_rec) / z
            if rel_err > max_err:
                max_err = rel_err
                worst_z = z

        status = "PASS" if max_err < THRESHOLD else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {h:6.2f}  {max_err:12.2e}  {worst_z:8.4f}  {status:>6s}")

        if max_err > worst_global_error:
            worst_global_error = max_err
            worst_global_z = worst_z
            worst_global_h = h

    print(
        f"\n  Global worst: rel_error = {worst_global_error:.2e} "
        f"at z = {worst_global_z:.4f}, h = {worst_global_h:.2f}"
    )

    # --- Test 3: Edge cases ---
    print("\n--- Test 3: Edge cases ---")

    # z=0.001 (fsolve convergence at low z)
    for h in h_values:
        z_low = 0.001
        d_L = dist(z_low, h=h)
        z_rec = dist_to_redshift(d_L, h=h)
        rel_err = abs(z_low - z_rec) / z_low
        status = "PASS" if rel_err < THRESHOLD else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  z=0.001, h={h:.2f}: rel_error = {rel_err:.2e}  [{status}]")

    # z=0.5 (upper bound of injection range)
    print()
    for h in h_values:
        z_high = 0.5
        d_L = dist(z_high, h=h)
        z_rec = dist_to_redshift(d_L, h=h)
        rel_err = abs(z_high - z_rec) / z_high
        status = "PASS" if rel_err < THRESHOLD else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  z=0.500, h={h:.2f}: d_L = {d_L:.4f} Gpc, rel_error = {rel_err:.2e}  [{status}]")

    # --- Test 4: Low-z Hubble law scaling ---
    # d_L(z) ~ c*z / H0 = c*z / (100*h) [in km/s / (km/s/Mpc)] = c*z/(100*h) Mpc
    # Convert to Gpc: d_L ~ c*z / (100*h) / 1000 Gpc = c*z / (1e5 * h) Gpc
    print("\n--- Test 4: Low-z Hubble law scaling ---")
    print("  d_L(z=0.001, h) vs c*z/(1e5*h) Gpc (Hubble law)")
    z_hubble = 0.001
    for h in h_values:
        d_L_actual = dist(z_hubble, h=h)
        d_L_hubble = C_KM_S * z_hubble / (1e5 * h)  # Gpc
        frac_diff = abs(d_L_actual - d_L_hubble) / d_L_hubble
        status = "PASS" if frac_diff < 0.05 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(
            f"  h={h:.2f}: actual={d_L_actual:.6f} Gpc, Hubble={d_L_hubble:.6f} Gpc, "
            f"frac_diff={frac_diff:.4f}  [{status}]"
        )

    # --- Overall result ---
    print("\n" + "=" * 72)
    if all_pass:
        print("OVERALL: PASS -- all tests within thresholds")
    else:
        print("OVERALL: FAIL -- one or more tests exceeded threshold")
    print("=" * 72)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
