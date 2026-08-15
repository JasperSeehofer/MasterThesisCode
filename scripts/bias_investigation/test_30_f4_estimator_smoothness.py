"""F4 estimator smoothness — efficient loop order (outer: h, inner: queries)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/jasper/Repositories/darksiren-emri")
sys.path.insert(0, str(REPO))

from darksiren_emri.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)
from darksiren_emri.constants import INJECTION_DATA_DIR, SNR_THRESHOLD

H_MIN, H_MAX, DH = 0.730, 0.745, 0.0005
H_GRID = np.arange(H_MIN, H_MAX + 1e-9, DH)
QUERIES = [
    (dl, m)
    for dl in (0.08, 0.12, 0.18, 0.25, 0.32, 0.38, 0.45, 0.52)
    for m in (1.5e5, 2.5e5, 4.0e5, 5.5e5, 7.0e5, 9.0e5)
]

print(f"Loading SDP from {REPO / INJECTION_DATA_DIR}", flush=True)
t0 = time.time()
sdp = SimulationDetectionProbability(
    injection_data_dir=str(REPO / INJECTION_DATA_DIR),
    snr_threshold=SNR_THRESHOLD,
)
print(f"  load: {time.time() - t0:.1f}s; pooled events: {len(sdp._pooled_df)}", flush=True)

# OUTER h, INNER queries
print(f"Probing {len(QUERIES)} queries × {len(H_GRID)} h-values (outer h)…", flush=True)
p_matrix = np.zeros((len(QUERIES), len(H_GRID)), dtype=np.float64)
for j, h in enumerate(H_GRID):
    t1 = time.time()
    # One grid build for this h
    _ = sdp._get_or_build_grid(float(h))
    # Bump LRU max to 31 to avoid eviction during the loop
    # (no API for this; instead we rely on _MAX_CACHE_SIZE=20 and accept rebuilds
    # but they'll be cached for the duration of this h's probing)
    for i, (dl_q, M_q) in enumerate(QUERIES):
        p = float(
            sdp.detection_probability_with_bh_mass_interpolated(
                d_L=dl_q,
                M_z=M_q,
                phi=0.0,
                theta=0.0,
                h=float(h),
            )
        )
        p_matrix[i, j] = p
    print(f"  h={h:.4f}: build+probe {time.time() - t1:.2f}s", flush=True)

# Σ(Δp)² and max-step per query
diffs = np.diff(p_matrix, axis=1)
sigma_dp_sq_per_query = np.sum(diffs**2, axis=1)
max_step_per_query = np.max(np.abs(diffs), axis=1)
total = float(np.sum(sigma_dp_sq_per_query))
worst = float(np.max(max_step_per_query))
median = float(np.median(max_step_per_query))

result = {
    "h_min": H_MIN,
    "h_max": H_MAX,
    "h_step": DH,
    "n_queries": len(QUERIES),
    "total_sigma_dp_sq": total,
    "worst_query_max_step": worst,
    "median_query_max_step": median,
}
OUT = REPO / "scripts/bias_investigation/outputs/phase46_merged/test_30_f4_smoothness.json"
OUT.write_text(json.dumps(result, indent=2))
print("\n=== F4 KERNEL ESTIMATOR SMOOTHNESS ===")
print(f"  total Σ(Δp_det)² over 48 queries × {len(H_GRID) - 1} Δh-steps : {total:.4f}")
print(f"  worst single-step |Δp_det|                       : {worst:.4f}")
print(f"  median per-query max-step                        : {median:.4f}")
print("\n  Pre-F4 reference (test_29): Σ(Δp_A)²+Σ(Δp_B)² = 1.5434, max ≈ 0.05")
print(f"  F4 reduction factor                              : {1.5434 / max(total, 1e-12):.1f}×")
print(f"\nReport: {OUT}")
