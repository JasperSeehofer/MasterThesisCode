"""Test 28: Production fine-grid sweep analyzer (Phase 48).

Analyzes the 63-point non-uniform h-sweep at h=0.73 produced by
``cluster/evaluate_production_h0p73_dense.sbatch`` (Phase 48 plan at
``~/.claude/plans/please-look-at-the-velvety-quail.md``).

The grid:
  - 41 dense points at Δh=0.001 across [0.710, 0.750] (truth ± 0.02)
  - 11 left wing points at Δh=0.010 across [0.600, 0.700]
  - 11 right wing points at Δh=0.010 across [0.760, 0.860]
  - truth h=0.730 lies on the dense core

Goal: paper-grade MAP and σ_boot at h=0.73, plus a Δh-sensitivity
diagnostic confirming Δh is not the resolution-limiting factor.

Reuses ``load_per_h_likelihoods`` and ``parabolic_refine`` from
``test_24_multi_truth_bias_sweep.py`` (the helpers are non-uniform-grid
safe: they sort by h-value and parabolic_refine uses a 3-point local
stencil at the discrete argmax).

Outputs the verdict JSON to a path of the user's choosing.

Usage:

    uv run python scripts/bias_investigation/test_28_production_finegrid_analyze.py \\
        --posteriors-dir simulations/cluster_run_production_h0p73_<DATE> \\
        --output scripts/bias_investigation/outputs/phase46_merged/h3_production_sweep_verdict.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "bias_investigation"))

from test_24_multi_truth_bias_sweep import (  # noqa: E402
    load_per_h_likelihoods,
    parabolic_refine,
)

H_TRUTH = 0.73
N_BOOT = 1000
RNG_SEED = 42

# Δh-sensitivity sub-grids: filter the 63-point grid down to coarser
# uniform-stride approximations to test whether MAP is stable under
# coarser sampling near the peak.  Each entry: (label, Δh, restrict_range)
# where restrict_range is None (full) or (lo, hi) inclusive.
SENSITIVITY_SUBGRIDS = [
    ("full_63pt", None, None),  # the full 63-point grid
    ("dense_core_only", None, (0.710, 0.750)),  # 41 points, dense core only
    ("delta_0p005", 0.005, None),  # ~52 points if we sub-sample to Δh=0.005
    ("delta_0p010", 0.010, None),  # ~27 points uniform Δh=0.01
]


def filter_grid(
    h_grid: npt.NDArray[np.float64],
    log_L: npt.NDArray[np.float64],
    delta_h: float | None,
    restrict_range: tuple[float, float] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return a sub-grid + the matching log-L columns.

    delta_h: if provided, keep only h-values at integer multiples of
        delta_h (within numerical tolerance), preserving the truth
        h=0.73 anchor.
    restrict_range: if provided, clip to [lo, hi] inclusive.
    """
    mask = np.ones(len(h_grid), dtype=bool)
    if restrict_range is not None:
        lo, hi = restrict_range
        mask &= (h_grid >= lo - 1e-9) & (h_grid <= hi + 1e-9)
    if delta_h is not None:
        # Snap each h-value to nearest delta_h gridline anchored at 0;
        # keep if it lies on the gridline.  Truth h=0.73 lies on every
        # standard delta_h ∈ {0.001, 0.005, 0.010} so it survives.
        ratio = h_grid / delta_h
        on_gridline = np.abs(ratio - np.round(ratio)) < 1e-6
        mask &= on_gridline
    return h_grid[mask], log_L[:, mask]


def map_with_sigma_boot(
    h_grid: npt.NDArray[np.float64],
    log_L: npt.NDArray[np.float64],
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """Compute discrete MAP, continuous MAP (parabolic refine), σ_boot."""
    n_events = log_L.shape[0]
    L_term = log_L.sum(axis=0)
    discrete_argmax = int(np.argmax(L_term))
    discrete_map = float(h_grid[discrete_argmax])
    continuous_map = parabolic_refine(h_grid, L_term)
    boundary_rail = discrete_argmax == 0 or discrete_argmax == len(h_grid) - 1

    boot_maps: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_events, size=n_events)
        boot_logL = log_L[idx].sum(axis=0)  # noqa: N806
        boot_maps.append(parabolic_refine(h_grid, boot_logL))
    sigma_boot = float(np.std(boot_maps))
    bias = continuous_map - H_TRUTH
    z = bias / sigma_boot if sigma_boot > 0 else float("nan")
    pos_frac = float(
        np.mean([parabolic_refine(h_grid, log_L[i]) > H_TRUTH for i in range(n_events)])
    )
    return {
        "n_events": n_events,
        "n_h": len(h_grid),
        "h_grid_min": float(h_grid.min()),
        "h_grid_max": float(h_grid.max()),
        "discrete_map": discrete_map,
        "continuous_map": continuous_map,
        "bias": bias,
        "sigma_boot": sigma_boot,
        "z": z,
        "pos_frac": pos_frac,
        "boundary_rail": boundary_rail,
    }


def analyze_channel(
    posteriors_dir: Path, label: str, rng: np.random.Generator
) -> dict[str, Any]:
    h_values, log_L = load_per_h_likelihoods(posteriors_dir)
    if not h_values:
        return {"error": f"no posteriors at {posteriors_dir}"}
    h_grid = np.asarray(h_values)

    # Primary: full grid
    primary = map_with_sigma_boot(h_grid, log_L, rng)
    primary["label"] = label
    primary["posteriors_dir"] = str(posteriors_dir.relative_to(PROJECT_ROOT))

    # Δh-sensitivity: re-compute MAP on each sub-grid
    delta_h_scan: dict[str, dict[str, Any]] = {}
    for sub_label, dh, rng_range in SENSITIVITY_SUBGRIDS:
        sub_h, sub_logL = filter_grid(h_grid, log_L, dh, rng_range)  # noqa: N806
        if len(sub_h) < 5:
            delta_h_scan[sub_label] = {
                "n_h": len(sub_h),
                "skipped": "too few points (<5)",
            }
            continue
        sub_rng = np.random.default_rng(RNG_SEED)  # fresh RNG per sub-grid
        sub_result = map_with_sigma_boot(sub_h, sub_logL, sub_rng, n_boot=500)
        delta_h_scan[sub_label] = {
            "n_h": int(sub_result["n_h"]),
            "discrete_map": sub_result["discrete_map"],
            "continuous_map": sub_result["continuous_map"],
            "sigma_boot": sub_result["sigma_boot"],
            "bias": sub_result["bias"],
            "z": sub_result["z"],
        }

    # Spread across sub-grids (excluding skipped)
    valid_maps = [v["continuous_map"] for v in delta_h_scan.values() if "continuous_map" in v]
    if len(valid_maps) >= 2:
        primary["delta_h_sensitivity_max_minus_min"] = float(
            max(valid_maps) - min(valid_maps)
        )
    else:
        primary["delta_h_sensitivity_max_minus_min"] = float("nan")
    primary["delta_h_scan"] = delta_h_scan
    return primary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    p.add_argument(
        "--posteriors-dir",
        type=Path,
        required=True,
        help="Directory containing posteriors/ and posteriors_with_bh_mass/ "
        "subdirectories produced by the production-sweep sbatch.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path for the verdict.",
    )
    p.add_argument(
        "--commit",
        type=str,
        default="f01595c",
        help="Git commit corresponding to the H3 fix (default: f01595c).",
    )
    args = p.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    base = args.posteriors_dir.resolve()

    print("=" * 72)
    print("TEST 28 — Production fine-grid h-sweep at h=0.73 analyzer")
    print("=" * 72)
    print(f"\nPosteriors base dir: {base}")
    print("  posteriors/         (1D channel)")
    print("  posteriors_with_bh_mass/  (2D channel)")
    print(f"Truth h: {H_TRUTH}")
    print(f"Bootstrap N: {N_BOOT}")
    print()

    results: dict[str, Any] = {
        "h_truth": H_TRUTH,
        "commit": args.commit,
        "phase": "Phase 48 — production fine-grid sweep",
        "posteriors_root": str(base.relative_to(PROJECT_ROOT)),
        "n_boot": N_BOOT,
        "rng_seed": RNG_SEED,
        "channels": {},
    }

    for label, subdir in [("1D", "posteriors"), ("2D", "posteriors_with_bh_mass")]:
        posteriors_dir = base / subdir
        if not posteriors_dir.exists():
            print(f"{label}: posteriors dir missing — {posteriors_dir}")
            continue
        ch = analyze_channel(posteriors_dir, label, rng)
        if "error" in ch:
            print(f"{label}: {ch['error']}")
            continue
        results["channels"][label] = ch
        print(f"--- {label} channel (full {ch['n_h']}-pt grid) ---")
        print(f"  N events       : {ch['n_events']}")
        print(f"  h-grid range   : [{ch['h_grid_min']:.4f}, {ch['h_grid_max']:.4f}]")
        print(f"  Discrete MAP   : {ch['discrete_map']:.4f}")
        print(f"  Continuous MAP : {ch['continuous_map']:.4f}")
        print(f"  σ_boot         : {ch['sigma_boot']:.4f}")
        print(f"  Bias           : {ch['bias']:+.4f}")
        print(f"  z              : {ch['z']:+.2f}σ")
        print(f"  pos_frac       : {ch['pos_frac']:.3f}")
        print(f"  Boundary rail  : {ch['boundary_rail']}")
        if not np.isnan(ch["delta_h_sensitivity_max_minus_min"]):
            print(
                f"  Δh-spread      : {ch['delta_h_sensitivity_max_minus_min']:.5f} "
                f"(max − min continuous MAP across sub-grids)"
            )
            for sub_label, sub in ch["delta_h_scan"].items():
                if "continuous_map" in sub:
                    print(
                        f"    {sub_label:20s} N_h={sub['n_h']:3d} "
                        f"MAP={sub['continuous_map']:.4f} "
                        f"σ={sub['sigma_boot']:.4f} z={sub['z']:+.2f}"
                    )
                else:
                    print(f"    {sub_label:20s} {sub.get('skipped', '?')}")
        print()

    # Info-monotonicity verdict
    if "1D" in results["channels"] and "2D" in results["channels"]:
        b1 = abs(results["channels"]["1D"]["bias"])
        b2 = abs(results["channels"]["2D"]["bias"])
        info_mono = b2 <= b1 + 0.0005  # tolerance for floating-point parity
        results["info_monotonicity_pass"] = info_mono
        print(
            f"Info monotonicity: |2D bias|={b2:.4f} vs |1D bias|={b1:.4f} → "
            f"{'PASS ✓' if info_mono else 'FAIL ✗'}"
        )

    output_abs = args.output.resolve()
    output_abs.parent.mkdir(parents=True, exist_ok=True)
    with open(output_abs, "w") as f:
        json.dump(results, f, indent=2, default=float)
    try:
        rel = output_abs.relative_to(PROJECT_ROOT)
        print(f"\nWrote {rel}")
    except ValueError:
        print(f"\nWrote {output_abs}")


if __name__ == "__main__":
    main()
