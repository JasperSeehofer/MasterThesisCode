"""Test 20 — Fine-grid closure-test analysis at h_true=0.65 with bootstrap σ_boot.

Successor to test_18: same RUN_DIR (rescaled CRBs at h_true=0.65), evaluated on
an 11-point fine grid Δh=0.005 in [0.625, 0.675] — mirrors the production h=0.73
grid resolution at which the +0.025 residual was discovered.

After the Tier 3 fix (2026-05-04), the joint posterior is just Σ log L_i (no
outer −N · log D correction); D(h) enters via L_comp = num/D inside each
per-event likelihood (Gray Eq. 31).  The D(h) decomposition diagnostic below
is reported for traceability but is no longer the joint-posterior driver.

Adds three things test_18 did NOT have:

  1. Bootstrap σ_boot via event resample (B=1000), exactly as T08 does for h=0.73.
  2. Parabolic 3-point refinement around the discrete peak.
  3. Pre-registered gate using bootstrap-derived σ_boot:
       PASS:        |MAP - 0.65| ≤ 3·σ_boot
       MARGINAL:    3·σ_boot < |MAP - 0.65| ≤ 5·σ_boot
       FAIL:        |MAP - 0.65| > 5·σ_boot

Run from project root after rsync:
    rsync -avz bwunicluster:/pfs/.../run_closure_h065_20260503/simulations/posteriors* \\
        simulations/cluster_run_closure_h065_20260503_finegrid/
    uv run python scripts/bias_investigation/test_20_closure_finegrid_analyze.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from darksiren_emri.bayesian_inference.bayesian_statistics import (  # noqa: E402
    precompute_completion_denominator,
)
from darksiren_emri.bayesian_inference.simulation_detection_probability import (  # noqa: E402
    SimulationDetectionProbability,
)
from darksiren_emri.constants import OMEGA_M  # noqa: E402

CLOSURE_DIR = PROJECT_ROOT / "simulations" / "cluster_run_closure_h065_20260503_finegrid"
POSTERIORS_DIR = CLOSURE_DIR / "posteriors"
POSTERIORS_BH_DIR = CLOSURE_DIR / "posteriors_with_bh_mass"
INJECTION_DIR = PROJECT_ROOT / "simulations" / "injections"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "outputs" / "phase45"

H_TRUTH = 0.65
N_BOOTSTRAP = 1000
RNG_SEED = 20260504


def _h_from_filename(path: Path) -> float:
    m = re.match(r"h_(\d+)_(\d+)\.json", path.name)
    if m is None:
        return float("nan")
    return float(f"{m.group(1)}.{m.group(2)}")


def _load_per_h_likelihoods(directory: Path) -> tuple[list[float], np.ndarray]:
    """Load per-h per-event likelihoods. Returns (sorted h_values, log_L[n_events, n_h])."""
    if not directory.exists():
        return [], np.empty((0, 0))
    files = sorted(directory.glob("h_*.json"), key=_h_from_filename)
    h_values: list[float] = [_h_from_filename(f) for f in files]
    event_indices: set[int] = set()
    per_h_data: list[dict[int, float]] = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        per_h: dict[int, float] = {}
        for k, v in d.items():
            try:
                ev = int(k)
            except (TypeError, ValueError):
                continue
            if isinstance(v, list):
                if len(v) == 0:
                    continue
                val = v[0]
            else:
                val = v
            event_indices.add(ev)
            per_h[ev] = float(val)
        per_h_data.append(per_h)

    events_sorted = sorted(event_indices)
    log_L = np.full((len(events_sorted), len(h_values)), np.nan)
    for j, per_h in enumerate(per_h_data):
        for i, ev in enumerate(events_sorted):
            if ev in per_h:
                log_L[i, j] = float(np.log(max(per_h[ev], 1e-300)))

    # Drop events with any NaN (incomplete h coverage)
    full_mask = ~np.isnan(log_L).any(axis=1)
    n_dropped = int((~full_mask).sum())
    if n_dropped:
        print(f"  Dropping {n_dropped} events with incomplete h coverage")
    log_L = log_L[full_mask]

    print(f"  Loaded {log_L.shape[0]} events × {len(h_values)} h-values from {directory.name}")
    return h_values, log_L


def _parabolic_refine(h_grid: np.ndarray, log_post: np.ndarray) -> float:
    """3-point parabolic interpolation around discrete argmax for sub-grid MAP."""
    i = int(np.argmax(log_post))
    if i <= 0 or i >= len(h_grid) - 1:
        return float(h_grid[i])
    h0, h1, h2 = h_grid[i - 1], h_grid[i], h_grid[i + 1]
    y0, y1, y2 = log_post[i - 1], log_post[i], log_post[i + 1]
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-12:
        return float(h1)
    h_max = h1 - 0.5 * (h2 - h0) * (y2 - y0) / (2 * denom)
    return float(h_max)


def _bootstrap_map(
    log_L: np.ndarray, log_D: np.ndarray, h_grid: np.ndarray, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """B bootstrap resamples (with replacement) → array of MAPs (continuous via parabolic).

    Joint posterior = Σ log L_i (Tier 3 fix 2026-05-04: no outer −N log D).
    log_D retained for the diagnostic decomposition only.
    """
    del log_D  # unused after Tier 3 fix; argument kept for API stability
    n_events = log_L.shape[0]
    maps = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n_events, size=n_events, replace=True)
        sum_log_L = log_L[idx].sum(axis=0)
        maps[b] = _parabolic_refine(h_grid, sum_log_L)
    return maps


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    print("=" * 70)
    print(f"AUDIT A7 redux — fine-grid closure analysis (h_true = {H_TRUTH})")
    print("=" * 70)

    summary: dict[str, Any] = {
        "audit": "A7 redux — fine-grid closure test (11 h-values, Δh=0.005, bootstrap σ_boot)",
        "h_truth": H_TRUTH,
        "n_bootstrap": N_BOOTSTRAP,
        "rng_seed": RNG_SEED,
    }

    sdp = SimulationDetectionProbability(
        injection_data_dir=str(INJECTION_DIR),
        snr_threshold=20.0,
    )
    print(
        f"\nLocal pooled injections: {len(sdp._pooled_df)} events "
        f"(h_inj groups: {sdp._h_values_found})"
    )

    for label, posteriors_dir in [
        ("without_bh_mass (1D)", POSTERIORS_DIR),
        ("with_bh_mass (2D)", POSTERIORS_BH_DIR),
    ]:
        print(f"\n=== {label} ===")
        h_values, log_L = _load_per_h_likelihoods(posteriors_dir)
        if not h_values:
            print(f"  No per-h JSONs in {posteriors_dir}")
            continue

        h_grid = np.asarray(h_values)
        n_events = log_L.shape[0]
        L_term = log_L.sum(axis=0)

        D_h_table = precompute_completion_denominator(
            h_values=list(h_values),
            detection_probability_obj=sdp,
            Omega_m=OMEGA_M,
            Omega_DE=1 - OMEGA_M,
        )
        D_h_arr = np.array([D_h_table[h] for h in h_values])
        log_D = np.log(D_h_arr)
        # Joint posterior post-Tier-3-fix: just Σ log L_i (D enters inside L_comp).
        joint_log = L_term
        # Diagnostic: legacy D_term shows what the (now-removed) outer correction would do.
        D_term = -n_events * log_D

        discrete_map = float(h_grid[int(np.argmax(joint_log))])
        continuous_map = _parabolic_refine(h_grid, joint_log)
        legacy_continuous_map = _parabolic_refine(h_grid, L_term + D_term)

        print(f"  N events: {n_events}")
        print(f"  h-grid:                  {[f'{h:.4f}' for h in h_values]}")
        print(f"  Σ log L per h:           {[f'{v:.2f}' for v in L_term]}")
        print(f"  joint log p(h) (post-fix): {[f'{v:.2f}' for v in joint_log]}")
        print(f"  Discrete MAP:                 h = {discrete_map:.4f}")
        print(f"  Continuous MAP (post-fix):    h = {continuous_map:.4f}")
        print(f"  Continuous MAP (legacy +D):   h = {legacy_continuous_map:.4f}  (diagnostic only)")
        print(f"  Bias vs h_true=0.65:          Δh = {continuous_map - H_TRUTH:+.4f}")

        # Bootstrap σ_boot via event resample (matches T08).
        print(f"  Running B={N_BOOTSTRAP} bootstrap...")
        boot_maps = _bootstrap_map(log_L, log_D, h_grid, N_BOOTSTRAP, rng)
        sigma_boot = float(np.std(boot_maps, ddof=1))
        boot_q = np.percentile(boot_maps, [5, 16, 50, 84, 95]).tolist()
        print(f"  Bootstrap σ_boot:           {sigma_boot:.4f}")
        print(f"  Bootstrap MAP quantiles:    [5,16,50,84,95] = {[f'{q:.4f}' for q in boot_q]}")

        # Pre-registered gate
        bias = continuous_map - H_TRUTH
        z = bias / sigma_boot if sigma_boot > 0 else float("inf")
        if abs(z) <= 3:
            verdict = "PASS — pipeline closure-validated (|MAP - h_true| ≤ 3·σ_boot)"
        elif abs(z) <= 5:
            verdict = "MARGINAL — bias 3-5·σ_boot; investigate D(h) systematic"
        else:
            verdict = "FAIL — bias > 5·σ_boot; structural systematic, escalate to Tier 3 D(h) audit"
        print(f"  z = bias / σ_boot = {z:+.2f}")
        print(f"  >>> Verdict: {verdict}")

        # A4-style decomposition: where does the bias come from?
        truth_idx = int(np.argmin(np.abs(h_grid - H_TRUTH)))
        map_idx = int(np.argmax(joint_log))
        delta_L = float(L_term[map_idx] - L_term[truth_idx])
        delta_D_legacy = float(D_term[map_idx] - D_term[truth_idx])
        print("  Δ(MAP vs truth) decomposition:")
        print(f"    Δ Σ log L:               {delta_L:+.3f}  (drives MAP post-Tier-3-fix)")
        print(f"    Δ -N log D (legacy):     {delta_D_legacy:+.3f}  (no longer applied)")

        summary[label] = {
            "h_values": h_values,
            "n_events": int(n_events),
            "L_term_per_h": L_term.tolist(),
            "D_term_per_h_legacy": D_term.tolist(),
            "joint_log_per_h": joint_log.tolist(),
            "discrete_map": discrete_map,
            "continuous_map": continuous_map,
            "legacy_continuous_map_with_outer_D": legacy_continuous_map,
            "bias_vs_truth": bias,
            "sigma_boot": sigma_boot,
            "z_score": z,
            "bootstrap_quantiles": {
                "q05": boot_q[0],
                "q16": boot_q[1],
                "q50": boot_q[2],
                "q84": boot_q[3],
                "q95": boot_q[4],
            },
            "delta_L_truth_to_map": delta_L,
            "delta_D_truth_to_map_legacy": delta_D_legacy,
            "verdict": verdict,
        }

    out_json = OUTPUT_DIR / "closure_h065_finegrid.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
