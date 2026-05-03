"""Test 17 (Audit A7 — lean closure test): rescale cluster CRB to h_true=0.65.

Per the user's observation: the simulation can be transformed to any h_true via
the distance relation. Source-frame EMRI parameters (M, μ, a, p_0, e_0, x_0,
sky/Kerr angles, phases) are h-independent; only d_L and SNR scale with h.
The Fisher matrix in (log d_L, source-frame X) coordinates is also
h-independent — so the covariance entries transform as:

  Cov(d_L, d_L)_new   = (h_old/h_new)² · Cov(d_L, d_L)_old
  Cov(d_L, X)_new     = (h_old/h_new)   · Cov(d_L, X)_old   (X ≠ d_L)
  Cov(X, Y)_new       =                    Cov(X, Y)_old    (X, Y ≠ d_L)

  d_L_new   = (h_old/h_new) · d_L_old
  SNR_new   = (h_new/h_old) · SNR_old

After rescaling we drop events with SNR_new < SNR_THRESHOLD (some events
detected at h=0.73 are below threshold at h=0.65 — this is correct, they're
not part of the h_true=0.65 detection set).

Output: a rescaled raw `cramer_rao_bounds.csv` ready to be fed to
`prepare_detections.py --workdir <closure_h065_workspace> --seed 201` to
produce a `prepared_cramer_rao_bounds.csv` for closure-test inference.

Usage:
    uv run python scripts/bias_investigation/test_17_rescale_crb_to_h065.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_CRB = PROJECT_ROOT / "simulations" / "cluster_run_phase45_20260501" / "cramer_rao_bounds.csv"
OUTPUT_DIR = PROJECT_ROOT / "simulations" / "closure_h065"
OUTPUT_CRB = OUTPUT_DIR / "simulations" / "cramer_rao_bounds.csv"

H_OLD = 0.73  # injection h of the cluster simulation
H_NEW = 0.65  # closure-test target truth
SNR_THRESHOLD = 20.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "simulations").mkdir(exist_ok=True)
    # Symlink the injection campaign so SimulationDetectionProbability finds
    # the same pooled data as the cluster did. The injection campaign is
    # h_inj-tagged per-file; SNR rescaling internal to the class handles the
    # h_target adjustment, so the SAME injection set is correct for any
    # h_true.
    inj_link = OUTPUT_DIR / "simulations" / "injections"
    if not inj_link.exists():
        inj_link.symlink_to(PROJECT_ROOT / "simulations" / "injections")

    print("=" * 70)
    print(f"AUDIT A7 (lean) — rescaling cluster CRB from h={H_OLD} → h={H_NEW}")
    print("=" * 70)

    df = pd.read_csv(INPUT_CRB)
    print(f"Loaded raw CRB: {len(df)} events")
    n_initial_snr20 = int((df["SNR"] >= SNR_THRESHOLD).sum())
    print(f"Events with SNR ≥ {SNR_THRESHOLD} at h={H_OLD}: {n_initial_snr20}")

    scale = H_OLD / H_NEW  # d_L_new / d_L_old
    print(f"\nScale factor h_old/h_new = {scale:.6f}")
    print(f"  d_L_new = scale · d_L_old             ({scale:.4f}× larger)")
    print(f"  SNR_new = (1/scale) · SNR_old         ({1 / scale:.4f}×)")
    print(f"  Cov(d_L, d_L)_new = scale² · Cov(d_L, d_L)_old   ({scale**2:.4f}×)")
    print(f"  Cov(d_L, X)_new = scale · Cov(d_L, X)_old (X ≠ d_L)   ({scale:.4f}×)")

    df_new: pd.DataFrame = df.copy()
    df_new["luminosity_distance"] = df_new["luminosity_distance"] * scale
    df_new["SNR"] = df_new["SNR"] / scale

    # Identify Fisher-covariance columns. These are named
    # ``delta_<X>_delta_<Y>`` and store Cov(X, Y).
    DL_KEY = "luminosity_distance"
    fisher_cols = [c for c in df.columns if c.startswith("delta_") and "_delta_" in c]
    print(f"\nIdentified {len(fisher_cols)} Fisher-covariance columns")

    # Cov(dL, dL)
    col_dL_dL = f"delta_{DL_KEY}_delta_{DL_KEY}"
    if col_dL_dL in df_new.columns:
        df_new[col_dL_dL] = df_new[col_dL_dL] * scale**2
    else:
        msg = f"Expected column {col_dL_dL!r} not found in CRB CSV"
        raise KeyError(msg)

    # Cov(dL, X) for X != dL
    n_cross_scaled = 0
    for col in fisher_cols:
        # Strip "delta_" and split into two parameter names by "_delta_".
        body = col.removeprefix("delta_")
        params = body.split("_delta_")
        if len(params) != 2:
            continue
        a, b = params
        if a == DL_KEY and b == DL_KEY:
            continue  # already scaled above
        if a == DL_KEY or b == DL_KEY:
            df_new[col] = df_new[col] * scale
            n_cross_scaled += 1
    print(f"Scaled {n_cross_scaled} Cov(d_L, X) cross-covariance columns by {scale:.4f}")

    # Apply SNR threshold at the new h_true and drop events that wouldn't
    # be detected there.
    n_dropped = int((df_new["SNR"] < SNR_THRESHOLD).sum())
    df_new = df_new[df_new["SNR"] >= SNR_THRESHOLD].copy()
    print(
        f"\nAfter SNR ≥ {SNR_THRESHOLD} filter at h={H_NEW}: "
        f"{len(df_new)} events ({n_dropped} dropped)"
    )
    print(
        f"  d_L range: [{df_new['luminosity_distance'].min():.4f}, "
        f"{df_new['luminosity_distance'].max():.4f}] Gpc"
    )
    print(f"  SNR range: [{df_new['SNR'].min():.2f}, {df_new['SNR'].max():.2f}]")

    df_new.to_csv(OUTPUT_CRB, index=False)
    print(f"\nWrote rescaled raw CRB to {OUTPUT_CRB}")

    # Sanity check: a single event before/after.
    sample = df.iloc[0]
    sample_new = df_new.iloc[0] if len(df_new) > 0 else df.iloc[0]
    print("\nSanity check (event 0):")
    print(
        f"  d_L:        {sample['luminosity_distance']:.6f} → "
        f"{sample_new['luminosity_distance']:.6f} Gpc"
    )
    print(f"  SNR:        {sample['SNR']:.3f} → {sample_new['SNR']:.3f}")
    print(f"  Cov(dL,dL): {sample[col_dL_dL]:.6e} → {sample_new[col_dL_dL]:.6e}")
    sigma_dL_old = float(np.sqrt(sample[col_dL_dL]))
    sigma_dL_new = float(np.sqrt(sample_new[col_dL_dL]))
    print(
        f"  σ_dL:       {sigma_dL_old:.6f} → {sigma_dL_new:.6f} Gpc "
        f"(σ_dL/d_L: {sigma_dL_old / sample['luminosity_distance']:.4f} → "
        f"{sigma_dL_new / sample_new['luminosity_distance']:.4f})"
    )

    print("\nNext steps:")
    print(
        f"  uv run python scripts/prepare_detections.py "
        f"--workdir {OUTPUT_DIR.relative_to(PROJECT_ROOT)} --seed 201 --force"
    )
    print(
        "  Then run --evaluate at multiple h values (use cluster's "
        "evaluate.sbatch with extended H_VALUES grid down to 0.55)."
    )


if __name__ == "__main__":
    main()
