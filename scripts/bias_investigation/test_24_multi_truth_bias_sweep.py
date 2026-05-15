"""Test 24 — Multi-truth bias-vs-h_true sweep analyzer.

Aggregates fine-grid closure runs at multiple h_true values (post-Tier-3 fix)
and tests whether the residual bias is purely statistical or carries a
sub-percent structural component.

Pre-registered hypothesis (under H0: residual is purely statistical):
  - mean bias across all h_true values is consistent with 0 (within error)
  - sign of per-truth bias is approximately balanced across truths
  - bias magnitudes scatter as σ_boot, not larger

Pre-registered alternative (residual structural systematic remains):
  - mean bias > 0 or < 0 by > 2 · σ_mean across the truth panel
  - all biases share the same sign
  - magnitude scatter > σ_boot (would suggest h-dependent systematic)

CAVEAT — shared injection set
-----------------------------
All truths in the panel reuse the *same* Phase 45 injection campaign
(rescaling reuses the cluster CRB; only ``prepare_detections --seed``
varies, which decorrelates the observed-d_L noise but not the underlying
sky positions, masses, or true redshifts).  The bootstrap σ_boot resamples
events at fixed truth and therefore captures statistical scatter in the
per-event d_L draw, but it does *not* capture correlation through the
shared injection campaign.  If a coherent positive (or negative) bias
appears across the panel, two interpretations remain:
  (a) genuine structural residual in the inference pipeline, or
  (b) idiosyncratic pull from the shared injection set.

The per-event diagnostic ``per_event_bias_pos_frac_*`` reports, for each
truth, what fraction of individual events have positive per-event bias.
If those fractions are similar across truths, the same events are pulling
MAP in the same direction at every truth — a signature of (b).  A fraction
that drifts smoothly with h_true (e.g. monotonic with truth) is more
consistent with (a).

Inputs: rsync'd posteriors at
  ``simulations/cluster_run_closure_h{HHH}_finegrid/posteriors{,_with_bh_mass}/``
where HHH = e.g. 0p60, 0p65, 0p70, 0p73, 0p75, 0p80, 0p85.

Output:
  scripts/bias_investigation/outputs/phase45/multi_truth_sweep.json
  scripts/bias_investigation/outputs/phase45/multi_truth_sweep.png

Run from project root after rsyncing all per-truth posteriors:
    uv run python scripts/bias_investigation/test_24_multi_truth_bias_sweep.py \\
        --truths 0.60 0.65 0.70 0.73 0.75 0.80 0.85
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from master_thesis_code.bayesian_inference.simulation_detection_probability import (  # noqa: E402
    SimulationDetectionProbability,
)

INJECTION_DIR = PROJECT_ROOT / "simulations" / "injections"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "outputs" / "phase45"

DEFAULT_TRUTHS = [0.60, 0.65, 0.70, 0.73, 0.75, 0.80, 0.85]
N_BOOTSTRAP = 1000
RNG_SEED = 20260504


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    p.add_argument(
        "--truths",
        nargs="+",
        type=float,
        default=DEFAULT_TRUTHS,
        help="Space-separated list of h_true values to aggregate.",
    )
    p.add_argument(
        "--posteriors-root",
        type=Path,
        default=PROJECT_ROOT / "simulations",
        help="Parent directory containing cluster_run_closure_h{HHH}_finegrid/ folders.",
    )
    p.add_argument("--no-plot", action="store_true", help="Skip matplotlib plot.")
    return p.parse_args()


def truth_to_dirname(h: float) -> str:
    """0.65 → 'cluster_run_closure_h0p65_finegrid'; 0.7 → 'h0p70'."""
    s = f"{h:.2f}".replace(".", "p")
    return f"cluster_run_closure_h{s}_finegrid"


# Re-export the canonical implementations now lifted to
# `master_thesis_code.bayesian_inference.posterior_combination`. Other bias-
# investigation scripts (and test_28) import these from this module for
# back-compat; the lifted versions are byte-equivalent (same algorithm, same
# numerics) and now also feed the plotting pipeline.
from master_thesis_code.bayesian_inference.posterior_combination import (
    _h_from_filename,  # noqa: F401
    load_per_h_likelihoods,
)
from master_thesis_code.bayesian_inference.posterior_combination import (
    parabolic_refine_map as parabolic_refine,
)


def analyze_one_truth(
    h_truth: float,
    posteriors_dir: Path,
    label: str,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    h_values, log_L = load_per_h_likelihoods(posteriors_dir)
    if not h_values:
        print(f"  [{label}] no posteriors at {posteriors_dir} — skipping")
        return None

    h_grid = np.asarray(h_values)
    n_events = log_L.shape[0]

    # Joint posterior post-Tier-3 fix: just Σ log L_i (no outer −N log D).
    L_term = log_L.sum(axis=0)
    discrete_argmax = int(np.argmax(L_term))
    discrete_map = float(h_grid[discrete_argmax])
    continuous_map = parabolic_refine(h_grid, L_term)

    # Boundary-rail flag: if the discrete MAP sits at either edge of the
    # grid, the parabolic refinement returns the boundary itself and the
    # underlying posterior peak likely lies outside the window.  Surface
    # this so the panel verdict can flag the truth instead of trusting a
    # truncated estimate.
    boundary_rail = discrete_argmax == 0 or discrete_argmax == len(h_grid) - 1

    # Bootstrap σ_boot (event resample with replacement) — captures
    # statistical scatter at fixed injection set; does NOT see correlation
    # through the shared injection campaign across truths (see module
    # docstring caveat).
    boot_maps = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.choice(n_events, size=n_events, replace=True)
        boot_maps[b] = parabolic_refine(h_grid, log_L[idx].sum(axis=0))
    sigma_boot = float(np.std(boot_maps, ddof=1))
    boot_q = np.percentile(boot_maps, [5, 16, 50, 84, 95]).tolist()

    # Per-event MAP and bias: argmax_h log_L_i(h) for each event i.  The
    # distribution of (h_event_map_i - h_truth) across events at fixed
    # truth probes whether individual injections systematically pull the
    # MAP in one direction.  If pos_frac is similar across truths, the
    # same injections drive the per-truth bias — a shared-injection-set
    # signature (interpretation b in the docstring caveat).
    per_event_map = np.array([parabolic_refine(h_grid, log_L[i]) for i in range(n_events)])
    per_event_bias = per_event_map - h_truth
    per_event_pos_frac = float(np.mean(per_event_bias > 0))
    per_event_median_bias = float(np.median(per_event_bias))

    bias = continuous_map - h_truth
    z = bias / sigma_boot if sigma_boot > 0 else float("inf")

    rail_flag = " [RAIL]" if boundary_rail else ""
    print(
        f"  [{label}] h_truth={h_truth:.3f}  N={n_events:>3}  "
        f"MAP={continuous_map:.4f}  bias={bias:+.4f}  σ_boot={sigma_boot:.4f}  "
        f"z={z:+.2f}  pos_frac={per_event_pos_frac:.2f}{rail_flag}"
    )

    return {
        "h_truth": h_truth,
        "n_events": int(n_events),
        "discrete_map": discrete_map,
        "continuous_map": continuous_map,
        "boundary_rail": bool(boundary_rail),
        "h_grid_min": float(h_grid.min()),
        "h_grid_max": float(h_grid.max()),
        "bias": bias,
        "sigma_boot": sigma_boot,
        "z_score": z,
        "bootstrap_quantiles": {
            "q05": boot_q[0],
            "q16": boot_q[1],
            "q50": boot_q[2],
            "q84": boot_q[3],
            "q95": boot_q[4],
        },
        "per_event_bias_pos_frac": per_event_pos_frac,
        "per_event_bias_median": per_event_median_bias,
        "h_grid": h_values,
    }


def panel_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Test H0: panel of biases is drawn from N(0, σ_boot²) at each truth.

    Compute:
      - inverse-variance-weighted mean bias and its σ
      - z of (weighted mean) vs zero
      - sign concordance: fraction of biases with same sign
      - reduced χ² of biases against zero
      - boundary-rail flag: any per-truth MAP at the grid edge
      - per-event pos-fraction stability: std across truths of the
        per-truth fraction of events with positive per-event bias
        (small std + |mean - 0.5| > ~0.05 ⇒ shared-injection-set pull)
    """
    if not rows:
        return {"verdict": "INSUFFICIENT_DATA"}
    biases = np.array([r["bias"] for r in rows])
    sigmas = np.array([r["sigma_boot"] for r in rows])
    weights = 1.0 / sigmas**2
    weighted_mean = float(np.sum(biases * weights) / np.sum(weights))
    weighted_mean_sigma = float(np.sqrt(1.0 / np.sum(weights)))
    z_panel = weighted_mean / weighted_mean_sigma if weighted_mean_sigma > 0 else float("inf")
    chi2 = float(np.sum((biases / sigmas) ** 2))
    n_dof = max(len(biases) - 1, 1)
    chi2_red = chi2 / n_dof
    pos_frac = float(np.mean(biases > 0))

    # Sign-concordance binomial p-value (two-sided): P(|n_pos - N/2| ≥ obs)
    from math import comb

    n = len(biases)
    n_pos = int(np.sum(biases > 0))
    extreme = max(n_pos, n - n_pos)
    p_sign_two_sided = sum(comb(n, k) for k in range(extreme, n + 1)) / 2 ** (n - 1)
    p_sign_two_sided = min(p_sign_two_sided, 1.0)

    # Boundary-rail flag: any truth where MAP railed at the grid edge.
    railed_truths = [r["h_truth"] for r in rows if r.get("boundary_rail")]

    # Per-event pos-fraction stability across truths.
    pe_pos_fracs = np.array([r["per_event_bias_pos_frac"] for r in rows])
    pe_mean = float(np.mean(pe_pos_fracs))
    pe_std = float(np.std(pe_pos_fracs, ddof=1)) if len(pe_pos_fracs) > 1 else 0.0
    # Heuristic: a stable pos_frac far from 0.5 signals a shared-injection
    # pull.  Flag when std < 0.05 and |mean - 0.5| > 0.05 (~10% imbalance).
    if pe_std < 0.05 and abs(pe_mean - 0.5) > 0.05 and len(pe_pos_fracs) >= 3:
        v_inj = (
            f"FLAG — per-event pos_frac stable across truths "
            f"(mean={pe_mean:.2f}, std={pe_std:.2f}); shared injection set may be pulling MAP"
        )
    else:
        v_inj = (
            f"PASS — per-event pos_frac dispersion not suspicious "
            f"(mean={pe_mean:.2f}, std={pe_std:.2f})"
        )

    if abs(z_panel) <= 2:
        v_mean = "PASS — weighted mean bias consistent with 0 (|z| ≤ 2)"
    elif abs(z_panel) <= 3:
        v_mean = "MARGINAL — weighted mean bias 2-3σ from 0"
    else:
        v_mean = "FAIL — weighted mean bias > 3σ from 0; structural residual"

    if pos_frac == 1.0 or pos_frac == 0.0:
        v_sign = f"FLAG — all {n} biases have the same sign (binomial p={p_sign_two_sided:.3f})"
    elif p_sign_two_sided < 0.10:
        v_sign = f"FLAG — sign concordance binomial p={p_sign_two_sided:.3f} < 0.10"
    else:
        v_sign = f"PASS — sign distribution consistent with random (p={p_sign_two_sided:.2f})"

    if railed_truths:
        v_rail = (
            f"FLAG — boundary-rail at h_truth={railed_truths!r}; "
            "discrete MAP sits at grid edge — widen window or trust with caution"
        )
    else:
        v_rail = "PASS — no boundary-rail; all MAPs lie in the interior"

    return {
        "weighted_mean_bias": weighted_mean,
        "weighted_mean_sigma": weighted_mean_sigma,
        "z_panel": z_panel,
        "chi2": chi2,
        "chi2_reduced": chi2_red,
        "n_dof": n_dof,
        "positive_fraction": pos_frac,
        "binomial_p_sign_concordance_2sided": p_sign_two_sided,
        "per_event_pos_frac_mean": pe_mean,
        "per_event_pos_frac_std": pe_std,
        "railed_truths": railed_truths,
        "verdict_mean": v_mean,
        "verdict_sign_concordance": v_sign,
        "verdict_boundary_rail": v_rail,
        "verdict_shared_injection_pull": v_inj,
    }


def maybe_plot(channel_results: dict[str, list[dict[str, Any]]], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"1D": "C0", "2D": "C1"}
    offsets = {"1D": -0.003, "2D": +0.003}
    for ch, rows in channel_results.items():
        if not rows:
            continue
        truths = np.array([r["h_truth"] for r in rows])
        biases = np.array([r["bias"] for r in rows])
        sigmas = np.array([r["sigma_boot"] for r in rows])
        ax.errorbar(
            truths + offsets.get(ch, 0),
            biases,
            yerr=sigmas,
            fmt="o",
            color=colors.get(ch, "gray"),
            label=f"{ch} channel",
            capsize=3,
        )
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel(r"$h_{\rm true}$")
    ax.set_ylabel(r"bias = $\hat{h}_{\rm MAP} - h_{\rm true}$")
    ax.set_title(
        "Multi-truth bias sweep (post-Tier-3 fix)\n"
        r"$\sigma_{\rm boot}$ = event resample at fixed truth — does not capture shared-injection correlation"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    print("=" * 70)
    print(f"Multi-truth bias-vs-h_true sweep ({len(args.truths)} truths)")
    print("=" * 70)

    # SDP not actually needed unless we recompute D(h); kept for parity with test_20.
    sdp = SimulationDetectionProbability(injection_data_dir=str(INJECTION_DIR), snr_threshold=20.0)
    print(f"Pooled injections: {len(sdp._pooled_df)}\n")

    results_1d: list[dict[str, Any]] = []
    results_2d: list[dict[str, Any]] = []
    for h_truth in args.truths:
        run_dir = args.posteriors_root / truth_to_dirname(h_truth)
        print(f"--- h_truth={h_truth:.3f} ({run_dir.name}) ---")
        r1d = analyze_one_truth(h_truth, run_dir / "posteriors", "1D", rng)
        r2d = analyze_one_truth(h_truth, run_dir / "posteriors_with_bh_mass", "2D", rng)
        if r1d is not None:
            results_1d.append(r1d)
        if r2d is not None:
            results_2d.append(r2d)

    print("\n=== Panel verdicts ===")
    panel_1d = panel_verdict(results_1d)
    panel_2d = panel_verdict(results_2d)
    for ch, panel in [("1D", panel_1d), ("2D", panel_2d)]:
        print(
            f"\n  {ch} channel ({len([r for r in (results_1d if ch == '1D' else results_2d)])} truths):"
        )
        for k, v in panel.items():
            print(f"    {k}: {v}")

    summary = {
        "rng_seed": RNG_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "truths_requested": args.truths,
        "channels": {
            "1D": {"per_truth": results_1d, "panel": panel_1d},
            "2D": {"per_truth": results_2d, "panel": panel_2d},
        },
    }
    out_json = OUTPUT_DIR / "multi_truth_sweep.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")

    if not args.no_plot:
        maybe_plot(
            {"1D": results_1d, "2D": results_2d},
            OUTPUT_DIR / "multi_truth_sweep.png",
        )


if __name__ == "__main__":
    main()
