"""Test 18 (Audit A7 — closure-test analysis): inspect h_true=0.65 closure run.

Reads per-h posterior JSONs from the cluster's closure RUN_DIR (rsynced to
``simulations/cluster_run_closure_h065_20260503/posteriors/``) and computes:

  1. Per-h Σ log L_i(h) over the 251 closure events.
  2. D(h) at each h via local ``precompute_completion_denominator`` (uses
     the same pooled injection campaign as the cluster).
  3. Joint posterior log p(h) = Σ log L − N log D(h) on the sparse
     closure h-grid (4 values).
  4. Discrete MAP and pre-registered gate verdict.
  5. A4-style decomposition: at the MAP h, what fraction of preference for
     MAP vs h_truth=0.65 comes from per-event L vs −N log D(h)?

Pre-registered closure gates:
  G7a  MAP ∈ [0.635, 0.665]   pipeline unbiased at h_true=0.65
  G7b  MAP ≈ 0.665–0.685     h-independent additive systematic (~+0.025)
  G7c  MAP ≈ 0.668–0.680     multiplicative bias matching h=0.73 case
  G7d  MAP ≈ 0.720–0.740     pipeline tuned to h=0.73 — HALT paper

Run from project root after rsync:
    rsync -avz bwunicluster:/pfs/.../run_closure_h065_20260503/simulations/posteriors* \\
        simulations/cluster_run_closure_h065_20260503/
    uv run python scripts/bias_investigation/test_18_closure_analyze.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from darksiren_emri.bayesian_inference.bayesian_statistics import (  # noqa: E402
    precompute_completion_denominator,
)
from darksiren_emri.bayesian_inference.simulation_detection_probability import (  # noqa: E402
    SimulationDetectionProbability,
)
from darksiren_emri.constants import OMEGA_M  # noqa: E402

CLOSURE_DIR = PROJECT_ROOT / "simulations" / "cluster_run_closure_h065_20260503"
POSTERIORS_DIR = CLOSURE_DIR / "posteriors"
POSTERIORS_BH_DIR = CLOSURE_DIR / "posteriors_with_bh_mass"
INJECTION_DIR = PROJECT_ROOT / "simulations" / "injections"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "outputs" / "phase45"

H_TRUTH = 0.65

# Pre-registered gate windows (with half-step tolerance for floating-point endpoints).
GATE_G7A = (0.635, 0.665)
GATE_G7B = (0.665, 0.685)
GATE_G7C = (0.668, 0.680)
GATE_G7D = (0.720, 0.740)


def _h_from_filename(path: Path) -> float:
    m = re.match(r"h_(\d+)_(\d+)\.json", path.name)
    if m is None:
        return float("nan")
    return float(f"{m.group(1)}.{m.group(2)}")


def _load_per_h_likelihoods(directory: Path) -> tuple[list[float], np.ndarray]:
    """Load per-h likelihoods from cluster JSONs.

    Each JSON is `{"<event_idx>": [likelihood_value], ...}`.

    Returns:
        h_values (sorted), log_L matrix shape (n_events, n_h)
    """
    if not directory.exists():
        return [], np.empty((0, 0))
    files = sorted(directory.glob("h_*.json"))
    h_values: list[float] = [_h_from_filename(f) for f in files]
    # Build event index union
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
                continue  # metadata key (e.g. "h"); skip
            if isinstance(v, list):
                if len(v) == 0:
                    continue  # event with no host match: drop
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
                val = per_h[ev]
                log_L[i, j] = np.log(max(val, 1e-300))

    print(f"  Loaded {len(events_sorted)} events × {len(h_values)} h-values from {directory.name}")
    return h_values, log_L


def _classify_gate(map_h: float) -> str:
    eps = 0.005  # half closure h-step
    if (GATE_G7A[0] - eps) <= map_h <= (GATE_G7A[1] + eps):
        return f"G7a — pipeline UNBIASED at h_true={H_TRUTH}"
    if (GATE_G7C[0] - eps) <= map_h <= (GATE_G7C[1] + eps):
        return "G7c — multiplicative bias (h_true=0.73 case scaled)"
    if (GATE_G7B[0] - eps) <= map_h <= (GATE_G7B[1] + eps):
        return "G7b — h-INDEPENDENT additive systematic (~+0.025)"
    if (GATE_G7D[0] - eps) <= map_h <= (GATE_G7D[1] + eps):
        return "G7d — PIPELINE TUNED TO 0.73; HALT PAPER"
    return f"G7-OTHER — bias = {map_h - H_TRUTH:+.4f}; investigate"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"AUDIT A7 — closure-test analysis (h_true = {H_TRUTH})")
    print("=" * 70)

    summary: dict[str, Any] = {
        "audit": "A7 — lean closure test at h_true=0.65",
        "h_truth": H_TRUTH,
        "preregistered_gates": {
            "G7a_unbiased": list(GATE_G7A),
            "G7b_additive": list(GATE_G7B),
            "G7c_multiplicative": list(GATE_G7C),
            "G7d_tuned_to_0p73": list(GATE_G7D),
        },
    }

    # Build SimulationDetectionProbability for D(h) computation.
    # Note: this uses the SAME pooled injection campaign that the cluster
    # used (project simulations/injections), since the closure RUN_DIR's
    # injections symlink points there.
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

        # Σ log L_i(h) per h
        L_term = np.nansum(log_L, axis=0)

        # D(h) per h via local computation (cluster's combined_posterior.json
        # would also be valid if we had it; D(h) is event-independent).
        D_h_table = precompute_completion_denominator(
            h_values=list(h_values),
            detection_probability_obj=sdp,
            Omega_m=OMEGA_M,
            Omega_DE=1 - OMEGA_M,
        )
        D_h_arr = np.array([D_h_table[h] for h in h_values])
        log_D_h = np.log(D_h_arr)
        n_events = log_L.shape[0]
        D_term = -n_events * log_D_h

        joint_log = L_term + D_term
        map_idx = int(np.argmax(joint_log))
        L_only_idx = int(np.argmax(L_term))

        print(f"  N events: {n_events}")
        print(f"  h-grid: {[f'{h:.3f}' for h in h_values]}")
        print(f"  Σ log L per h:      {[f'{v:.2f}' for v in L_term]}")
        print(f"  -N log D(h) per h:  {[f'{v:.2f}' for v in D_term]}")
        print(f"  Joint log p per h:  {[f'{v:.2f}' for v in joint_log]}")
        print(f"  Σ log L peaks at h = {h_values[L_only_idx]:.4f}")
        print(f"  Joint MAP at h    = {h_values[map_idx]:.4f}")

        # Fine-grid continuous MAP via cubic spline on the 4 sparse points.
        h_fine = np.linspace(min(h_values), max(h_values), 1001)
        cs_joint = CubicSpline(h_values, joint_log)
        joint_log_fine = cs_joint(h_fine)
        cs_L = CubicSpline(h_values, L_term)
        L_term_fine = cs_L(h_fine)
        continuous_map = float(h_fine[int(np.argmax(joint_log_fine))])
        continuous_L_only_map = float(h_fine[int(np.argmax(L_term_fine))])
        print(f"  Continuous MAP (cubic spline, fine grid): h = {continuous_map:.4f}")
        print(f"  Continuous Σ log L peaks at:               h = {continuous_L_only_map:.4f}")
        print(
            f"  Bias vs h_true={H_TRUTH}:                       Δh = {continuous_map - H_TRUTH:+.4f}"
        )

        gate = _classify_gate(continuous_map)
        print(f"  >>> {gate}")

        # A4-style decomposition: at MAP vs truth, separate L pull from D pull
        truth_idx = int(np.argmin(np.abs(np.array(h_values) - H_TRUTH)))
        delta_L = float(L_term[map_idx] - L_term[truth_idx])
        delta_D = float(D_term[map_idx] - D_term[truth_idx])
        print("  Δ(MAP vs truth) decomposition:")
        print(f"    Δ Σ log L:  {delta_L:+.3f}")
        print(f"    Δ -N log D: {delta_D:+.3f}")
        print(f"    Δ total:    {delta_L + delta_D:+.3f}")

        summary[label] = {
            "h_values": h_values,
            "n_events": int(n_events),
            "L_term_per_h": L_term.tolist(),
            "D_term_per_h": D_term.tolist(),
            "joint_log_per_h": joint_log.tolist(),
            "L_only_peak_h_discrete": float(h_values[L_only_idx]),
            "joint_map_h_discrete": float(h_values[map_idx]),
            "joint_map_h_continuous": continuous_map,
            "L_only_peak_h_continuous": continuous_L_only_map,
            "bias_vs_truth": continuous_map - H_TRUTH,
            "delta_L_truth_to_map": delta_L,
            "delta_D_truth_to_map": delta_D,
            "gate_verdict": gate,
        }

    out_json = OUTPUT_DIR / "closure_h065.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
