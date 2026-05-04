"""Test 23 — Generalized rescaling of cluster CRB to an arbitrary h_true.

Successor to test_17 (h_true=0.65 hardcoded). Rescales the cluster's raw
``cramer_rao_bounds.csv`` (h_inj=0.73) to any target h_true via the
distance-relation transformation:

  d_L_new   = (h_old/h_new) · d_L_old
  SNR_new   = (h_new/h_old) · SNR_old
  Cov(d_L, d_L)_new = (h_old/h_new)² · Cov(d_L, d_L)_old
  Cov(d_L, X)_new   = (h_old/h_new)   · Cov(d_L, X)_old   (X ≠ d_L)
  Cov(X, Y)_new     =                    Cov(X, Y)_old    (X, Y ≠ d_L)

Drops events whose post-rescale SNR falls below ``SNR_THRESHOLD``.

Usage:
    uv run python scripts/bias_investigation/test_23_rescale_crb_to_h_true.py \\
        --h-true 0.70 \\
        --workdir simulations/closure_h070

The output workspace is structured for direct ingestion by
``scripts/prepare_detections.py`` and the cluster
``cluster/evaluate_closure_h_true_finegrid.sbatch`` array job.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT_CRB = (
    PROJECT_ROOT / "simulations" / "cluster_run_phase45_20260501" / "cramer_rao_bounds.csv"
)
H_OLD = 0.73
SNR_THRESHOLD = 20.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    p.add_argument(
        "--h-true",
        type=float,
        required=True,
        help="Target injection h for the rescaled closure-test simulation.",
    )
    p.add_argument(
        "--workdir",
        type=Path,
        required=True,
        help="Output workspace directory (relative to project root or absolute).",
    )
    p.add_argument(
        "--input-crb",
        type=Path,
        default=DEFAULT_INPUT_CRB,
        help="Path to the source cluster cramer_rao_bounds.csv (defaults to "
        "the Phase 45 cluster run).",
    )
    p.add_argument(
        "--snr-threshold",
        type=float,
        default=SNR_THRESHOLD,
        help="SNR detection threshold to apply post-rescale (default 20).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    h_new = float(args.h_true)
    workdir = (
        args.workdir if args.workdir.is_absolute() else PROJECT_ROOT / args.workdir
    ).resolve()

    sims_dir = workdir / "simulations"
    sims_dir.mkdir(parents=True, exist_ok=True)
    inj_link = sims_dir / "injections"
    if not inj_link.exists():
        inj_link.symlink_to(PROJECT_ROOT / "simulations" / "injections")

    output_crb = sims_dir / "cramer_rao_bounds.csv"

    print("=" * 70)
    print(f"Rescaling cluster CRB from h={H_OLD} → h={h_new}")
    print("=" * 70)
    print(f"  Input:    {args.input_crb}")
    print(f"  Output:   {output_crb}")

    df = pd.read_csv(args.input_crb)
    print(f"\nLoaded raw CRB: {len(df)} events")
    n_initial_snr20 = int((df["SNR"] >= args.snr_threshold).sum())
    print(f"Events with SNR ≥ {args.snr_threshold} at h={H_OLD}: {n_initial_snr20}")

    scale = H_OLD / h_new
    print(f"\nScale factor h_old/h_new = {scale:.6f}")

    df_new: pd.DataFrame = df.copy()
    df_new["luminosity_distance"] = df_new["luminosity_distance"] * scale
    df_new["SNR"] = df_new["SNR"] / scale

    DL_KEY = "luminosity_distance"
    fisher_cols = [c for c in df.columns if c.startswith("delta_") and "_delta_" in c]
    col_dL_dL = f"delta_{DL_KEY}_delta_{DL_KEY}"
    if col_dL_dL not in df_new.columns:
        msg = f"Expected column {col_dL_dL!r} not found in CRB CSV"
        raise KeyError(msg)
    df_new[col_dL_dL] = df_new[col_dL_dL] * scale**2

    n_cross_scaled = 0
    for col in fisher_cols:
        body = col.removeprefix("delta_")
        params = body.split("_delta_")
        if len(params) != 2:
            continue
        a, b = params
        if a == DL_KEY and b == DL_KEY:
            continue
        if a == DL_KEY or b == DL_KEY:
            df_new[col] = df_new[col] * scale
            n_cross_scaled += 1
    print(f"Scaled Cov(d_L, d_L) by {scale**2:.4f} and {n_cross_scaled} cross-covs by {scale:.4f}")

    n_dropped = int((df_new["SNR"] < args.snr_threshold).sum())
    df_new = df_new[df_new["SNR"] >= args.snr_threshold].copy()
    print(
        f"\nAfter SNR ≥ {args.snr_threshold} filter at h={h_new}: "
        f"{len(df_new)} events ({n_dropped} dropped)"
    )
    print(
        f"  d_L range: [{df_new['luminosity_distance'].min():.4f}, "
        f"{df_new['luminosity_distance'].max():.4f}] Gpc"
    )
    print(f"  SNR range: [{df_new['SNR'].min():.2f}, {df_new['SNR'].max():.2f}]")

    df_new.to_csv(output_crb, index=False)
    print(f"\nWrote rescaled raw CRB to {output_crb}")

    sample = df.iloc[0]
    sample_new = df_new.iloc[0] if len(df_new) > 0 else df.iloc[0]
    sigma_dL_old = float(np.sqrt(sample[col_dL_dL]))
    sigma_dL_new = float(np.sqrt(sample_new[col_dL_dL]))
    print("\nSanity check (event 0):")
    print(
        f"  d_L: {sample['luminosity_distance']:.6f} → "
        f"{sample_new['luminosity_distance']:.6f} Gpc; "
        f"σ_dL/d_L: {sigma_dL_old / sample['luminosity_distance']:.4f} → "
        f"{sigma_dL_new / sample_new['luminosity_distance']:.4f}"
    )

    print("\nNext steps:")
    print(
        f"  uv run python scripts/prepare_detections.py "
        f"--workdir {workdir.relative_to(PROJECT_ROOT)} --seed 201 --force"
    )
    print(
        f"  Push to cluster RUN_DIR and submit: "
        f"sbatch --array=0-3 --export=ALL,RUN_DIR=...,H_TRUE={h_new} "
        "cluster/evaluate_closure_h_true_finegrid.sbatch"
    )


if __name__ == "__main__":
    main()
