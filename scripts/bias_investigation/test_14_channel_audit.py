"""DEPRECATED 2026-05-05: this Phase-14 audit measures the 1D vs 2D anchor
asymmetry that motivated Plan 45-06/45-07 (lift the 1D Wilson anchor; extend
hybrid to 2D).  Both channels were superseded by the principled
monotonic-asymptotic extrapolation scheme on 2026-05-05; the anchor
constants imported below (``_P_MAX_EMPIRICAL_ANCHOR``,
``_D_INTERMEDIATE_ANCHOR_GPC``, ``_P_INTERMEDIATE_EMPIRICAL``) no longer
exist.  See ``.planning/2D-CHANNEL-AUDIT-20260505.md``.  The script will
fail with ImportError until rewritten or removed.

Original docstring follows.

Test 14 (Audit A2): 1D vs 2D channel structural audit.

Resolves concern #5 (with_bh_mass MAP=0.7450 closer to truth than the
patched without_bh_mass MAP=0.7550) by measuring the effective `[0, c_0]`
lift each channel applies. Determines whether Plans 45-06 (raise d_L=0
anchor) and 45-07 (extend hybrid to 2D) are signed in the right direction.

Both 1D and 2D channels use scipy.interpolate.RegularGridInterpolator with
``fill_value=None``, which performs **linear extrapolation** (not nearest
neighbour, contrary to a stale docstring claim now corrected). The
structural difference is:

* 1D ``_build_grid_1d``: prepends two anchors (0, 0.7931) and (0.05, 1.0)
  before the histogram bins; the interpolator therefore reads
  approximately ``0.7931 + 4.138·d_L`` on [0, 0.05] and
  ``1.0 + (p̂(c_0) − 1.0)/(c_0 − 0.05) · (d_L − 0.05)`` on [0.05, c_0].
* 2D ``_build_grid_2d``: no anchors. Histogram bins are ``(d_L, M)``;
  the interpolator linearly extrapolates through bins 0 and 1 in d_L for
  d_L < c_0, separately per M-bin.

This script:
  1. Builds SimulationDetectionProbability for h ∈ {0.70, 0.73, 0.75, 0.77}.
  2. Reads ``c_0(h)`` from quality_flags['dl_edges'][1] / 2.
  3. Computes the unanchored 1D first-bin estimate ``p̂_1D(c_0; h)`` as
     the M-marginal of the 2D quality flags histogram.
  4. Probes the anchored 1D interpolator at a fine d_L grid in [0, c_0(h)]
     to compute the anchored window-average lift.
  5. For 2D, probes ``interp_2d(d_L, M)`` at the same fine d_L grid for a
     range of M values spanning the EMRI mass distribution; reports the
     effective d_L<c_0 lift per M and the M-marginal mean weighted by
     uniform-log-M (proxy for GLADE host mass distribution).
  6. Reports whether 1D's anchor provides MORE or LESS lift than 2D's
     natural extrapolation on [0, c_0].

Pre-registered gates (set BEFORE running):
  G2a: 2D mean lift on [0, c_0] >= 1D anchored mean lift  →  1D
       under-anchors. 45-06 (raise to 0.8873) defensible after A3.
  G2b: 2D mean lift on [0, c_0] noticeably LESS than 1D anchored lift,
       yet 2D MAP=0.745 is closer to truth than 1D MAP=0.755
       →  1D over-anchors. REJECT 45-06 and 45-07.
  G2c: lift comparison inconclusive (within ±0.05 absolute)
       →  channel difference is unrelated to anchor; 45-07 risky.

Run from project root:
    uv run python scripts/bias_investigation/test_14_channel_audit.py
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from darksiren_emri.bayesian_inference.simulation_detection_probability import (
    _D_INTERMEDIATE_ANCHOR_GPC,
    _P_INTERMEDIATE_EMPIRICAL,
    _P_MAX_EMPIRICAL_ANCHOR,
    SimulationDetectionProbability,
)

INJECTION_DIR = PROJECT_ROOT / "simulations" / "injections"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "outputs" / "phase45"
SNR_THRESHOLD = 20.0

H_VALUES_PROBE = [0.70, 0.73, 0.75, 0.77]
H_FOCUS = 0.73  # h at which we report headline numbers
N_FINE = 51  # fine-grid points in [0, c_0]

# Pre-registered tolerance for G2c (inconclusive band)
G2C_INCONCLUSIVE_TOL = 0.05


def _unanchored_1d_phat_first_bin(qf: dict) -> tuple[float, int, int]:
    """M-marginal of 2D quality_flags first d_L bin.

    `_build_grid_1d` is the M-marginal histogram of the same dl_edges as
    the 2D builder. So the unanchored 1D first-bin p_det estimate is
    sum_M n_detected[0, :] / sum_M n_total[0, :].
    """
    n_total_first = float(np.asarray(qf["n_total"])[0, :].sum())
    n_det_first = float(np.asarray(qf["n_detected"])[0, :].sum())
    if n_total_first == 0:
        return 0.0, 0, 0
    return n_det_first / n_total_first, int(n_det_first), int(n_total_first)


def _anchored_1d_window_mean(
    interp_1d: RegularGridInterpolator,
    d_lo: float,
    d_hi: float,
    n_fine: int = N_FINE,
) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    d_grid = np.linspace(d_lo, d_hi, n_fine)
    p_grid = np.clip(interp_1d(d_grid), 0.0, 1.0)
    mean = float(np.trapezoid(p_grid, d_grid) / (d_hi - d_lo))
    return mean, d_grid, p_grid


def _2d_window_mean_per_M(
    interp_2d: RegularGridInterpolator,
    d_lo: float,
    d_hi: float,
    M_values: npt.NDArray[np.float64],
    n_fine: int = N_FINE,
    M_weights: npt.NDArray[np.float64] | None = None,
    M_reliable: npt.NDArray[np.bool_] | None = None,
) -> dict[str, Any]:
    """Per-M window-average of 2D interpolator over [d_lo, d_hi].

    If M_weights is provided, the M-marginal is the weighted mean (matching
    the actual local injection density). If M_reliable is provided, only
    M-bins with reliable statistics (n_total ≥ 10) are included in the
    marginal.
    """
    d_grid = np.linspace(d_lo, d_hi, n_fine)
    means: list[float] = []
    edges_lo: list[float] = []
    edges_hi: list[float] = []
    for M in M_values:
        pts = np.column_stack([d_grid, np.full_like(d_grid, M)])
        p_grid = np.clip(interp_2d(pts), 0.0, 1.0)
        means.append(float(np.trapezoid(p_grid, d_grid) / (d_hi - d_lo)))
        edges_lo.append(float(p_grid[0]))
        edges_hi.append(float(p_grid[-1]))
    means_arr = np.asarray(means, dtype=np.float64)

    # Uniform-log marginal (original)
    marginal_uniform = float(np.mean(means_arr))

    # Weighted marginal (if weights given)
    if M_weights is not None:
        w = np.asarray(M_weights, dtype=np.float64)
        if w.sum() > 0:
            marginal_weighted = float(np.sum(means_arr * w) / np.sum(w))
        else:
            marginal_weighted = float("nan")
    else:
        marginal_weighted = marginal_uniform

    # Reliable-only marginal (n_total ≥ 10 in first d_L bin)
    if M_reliable is not None:
        rel = np.asarray(M_reliable, dtype=bool)
        if rel.any():
            if M_weights is not None:
                w_rel = np.asarray(M_weights, dtype=np.float64)[rel]
                if w_rel.sum() > 0:
                    marginal_reliable = float(np.sum(means_arr[rel] * w_rel) / np.sum(w_rel))
                else:
                    marginal_reliable = float("nan")
            else:
                marginal_reliable = float(np.mean(means_arr[rel]))
        else:
            marginal_reliable = float("nan")
    else:
        marginal_reliable = marginal_uniform

    return {
        "M_values": [float(m) for m in M_values],
        "window_means": means,
        "values_at_dL_lo": edges_lo,
        "values_at_dL_hi": edges_hi,
        "M_marginal_uniform_log": marginal_uniform,
        "M_marginal_density_weighted": marginal_weighted,
        "M_marginal_reliable_only": marginal_reliable,
    }


def _classify_gate(lift_2d: float, lift_1d: float) -> str:
    diff = lift_2d - lift_1d
    if abs(diff) <= G2C_INCONCLUSIVE_TOL:
        return f"G2c (inconclusive; |Δlift|={abs(diff):.4f} ≤ {G2C_INCONCLUSIVE_TOL})"
    if diff > G2C_INCONCLUSIVE_TOL:
        return f"G2a (1D under-anchors; 2D mean lift {lift_2d:.4f} > 1D {lift_1d:.4f})"
    return f"G2b (1D over-anchors; 2D mean lift {lift_2d:.4f} < 1D {lift_1d:.4f})"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("AUDIT A2 — 1D vs 2D CHANNEL STRUCTURAL AUDIT")
    print("=" * 60)
    print(
        f"Anchors: _P_MAX_EMPIRICAL_ANCHOR={_P_MAX_EMPIRICAL_ANCHOR}, "
        f"intermediate=({_D_INTERMEDIATE_ANCHOR_GPC}, {_P_INTERMEDIATE_EMPIRICAL})"
    )
    print(f"Loading injections from {INJECTION_DIR} (SNR≥{SNR_THRESHOLD})…")

    sdp = SimulationDetectionProbability(
        injection_data_dir=str(INJECTION_DIR),
        snr_threshold=SNR_THRESHOLD,
    )
    print(f"Pooled {len(sdp._pooled_df)} injections (h_inj groups: {sdp._h_values_found})")

    # Build all grids first (warms cache and quality flags)
    for h in H_VALUES_PROBE:
        sdp._get_or_build_grid(h)

    # Infer M centers from the focus-h quality flags (same M edges across h)
    qf_focus = sdp.quality_flags(H_FOCUS)
    M_edges = np.asarray(qf_focus["M_edges"], dtype=np.float64)
    M_centers = np.sqrt(M_edges[:-1] * M_edges[1:])  # geometric mean (log-spaced)
    # Probe at every M_center so we can reweight by actual density.
    M_probe = M_centers

    summary: dict[str, Any] = {
        "audit": "A2 — 1D vs 2D channel structural audit",
        "anchors": {
            "P_max_empirical": _P_MAX_EMPIRICAL_ANCHOR,
            "intermediate_dL": _D_INTERMEDIATE_ANCHOR_GPC,
            "intermediate_p": _P_INTERMEDIATE_EMPIRICAL,
        },
        "snr_threshold": SNR_THRESHOLD,
        "h_focus": H_FOCUS,
        "M_probe_values": [float(m) for m in M_probe],
        "per_h": {},
    }

    print("\nPer-h structural metrics:")
    print(
        f"  {'h':>5}  {'c_0':>8}  {'p̂_1D(c_0)':>10}  {'1D@0.05':>9}  "
        f"{'1D[0,c_0]':>10}  {'2D[0,c_0]':>10}  {'gate':>}"
    )

    for h in H_VALUES_PROBE:
        interp_2d, interp_1d = sdp._get_or_build_grid(h)
        qf = sdp.quality_flags(h)
        dl_edges = np.asarray(qf["dl_edges"], dtype=np.float64)
        c_0 = float(dl_edges[1] / 2.0)  # first-bin midpoint

        # Unanchored 1D first-bin p̂(c_0)
        phat_1d, n_det_first, n_total_first = _unanchored_1d_phat_first_bin(qf)

        # Anchored 1D evaluations
        p_at_zero = float(np.clip(interp_1d(np.array([0.0])), 0.0, 1.0)[0])
        p_at_005 = float(np.clip(interp_1d(np.array([0.05])), 0.0, 1.0)[0])
        p_at_c0 = float(np.clip(interp_1d(np.array([c_0])), 0.0, 1.0)[0])
        # Anchored 1D window mean on [0, c_0]
        lift_1d_window, d_grid, p_grid_1d = _anchored_1d_window_mean(interp_1d, 0.0, c_0)

        # 2D per-M window means
        # Density weights = total injections in first d_L bin per M-bin
        n_total_2d = np.asarray(qf["n_total"], dtype=np.float64)
        density_weights = n_total_2d[0, :]  # first d_L bin, per M
        # Reliable mask: M-bins with ≥10 events in first d_L bin (Wilson floor)
        reliable_mask = density_weights >= 10
        window_2d = _2d_window_mean_per_M(
            interp_2d,
            0.0,
            c_0,
            M_probe,
            M_weights=density_weights,
            M_reliable=reliable_mask,
        )
        # Headline 2D marginal: density-weighted (matches what 1D channel
        # marginalizes over implicitly; M-marginalised by injection density).
        lift_2d_marginal = window_2d["M_marginal_density_weighted"]
        lift_2d_uniform = window_2d["M_marginal_uniform_log"]
        lift_2d_reliable = window_2d["M_marginal_reliable_only"]
        lift_2d_min = float(min(window_2d["window_means"]))
        lift_2d_max = float(max(window_2d["window_means"]))
        n_reliable_M = int(reliable_mask.sum())

        gate = _classify_gate(lift_2d_marginal, lift_1d_window)

        print(
            f"  {h:>5.2f}  {c_0:>8.5f}  {phat_1d:>10.4f}  {p_at_005:>9.4f}  "
            f"{lift_1d_window:>10.4f}  {lift_2d_marginal:>10.4f}  "
            f"(rel-only={lift_2d_reliable:.3f}, n_rel_M={n_reliable_M}/{len(M_probe)})"
        )

        summary["per_h"][f"{h:.2f}"] = {
            "c_0_Gpc": c_0,
            "n_total_first_bin_2D_M_marginal": n_total_first,
            "n_detected_first_bin_2D_M_marginal": n_det_first,
            "n_reliable_M_bins_first_dL_bin": n_reliable_M,
            "n_total_M_bins": int(len(M_probe)),
            "phat_1D_unanchored_at_c0": phat_1d,
            "anchored_1D_at_dL_0": p_at_zero,
            "anchored_1D_at_dL_0p05": p_at_005,
            "anchored_1D_at_dL_c0": p_at_c0,
            "anchored_1D_window_mean_0_to_c0": lift_1d_window,
            "_2D_window_mean_per_M": window_2d,
            "_2D_window_mean_density_weighted": lift_2d_marginal,
            "_2D_window_mean_uniform_log": lift_2d_uniform,
            "_2D_window_mean_reliable_only": lift_2d_reliable,
            "_2D_window_mean_M_min_max": [lift_2d_min, lift_2d_max],
            "lift_difference_2D_minus_1D_density_weighted": lift_2d_marginal - lift_1d_window,
            "gate_verdict": gate,
        }

    # Headline gate from h=0.73
    focus = summary["per_h"][f"{H_FOCUS:.2f}"]
    print(f"\n>>> Headline gate (h={H_FOCUS}): {focus['gate_verdict']}\n")
    print(f"Channel-comparison details for h={H_FOCUS}:")
    print(f"  c_0 = {focus['c_0_Gpc']:.5f} Gpc")
    print(
        f"  Unanchored 1D p̂(c_0) = {focus['phat_1D_unanchored_at_c0']:.4f} "
        f"({focus['n_detected_first_bin_2D_M_marginal']}/{focus['n_total_first_bin_2D_M_marginal']})"
    )
    print(f"  Anchored 1D at d_L=0:    {focus['anchored_1D_at_dL_0']:.4f}")
    print(f"  Anchored 1D at d_L=0.05: {focus['anchored_1D_at_dL_0p05']:.4f}")
    print(f"  Anchored 1D at d_L=c_0:  {focus['anchored_1D_at_dL_c0']:.4f}")
    print(f"  Anchored 1D window mean [0, c_0]: {focus['anchored_1D_window_mean_0_to_c0']:.4f}")
    print(
        f"  2D window mean [0, c_0] (density-weighted M): {focus['_2D_window_mean_density_weighted']:.4f}"
    )
    print(
        f"  2D window mean [0, c_0] (uniform-log M):     {focus['_2D_window_mean_uniform_log']:.4f}"
    )
    print(
        f"  2D window mean [0, c_0] (reliable M only, n_rel={focus['n_reliable_M_bins_first_dL_bin']}/"
        f"{focus['n_total_M_bins']}): {focus['_2D_window_mean_reliable_only']:.4f}"
    )
    print(
        f"  2D window mean range across M:   "
        f"[{focus['_2D_window_mean_M_min_max'][0]:.4f}, {focus['_2D_window_mean_M_min_max'][1]:.4f}]"
    )
    print(
        f"  Lift difference (2D density-weighted - 1D): "
        f"{focus['lift_difference_2D_minus_1D_density_weighted']:+.4f}"
    )

    summary["headline_gate_local"] = focus["gate_verdict"]
    summary["preregistered_inconclusive_tol"] = G2C_INCONCLUSIVE_TOL

    # ---------------------------------------------------------------
    # Analytical cluster-scale projection.
    # Local injection set has dl_max ≈ 2.77 Gpc → c_0 = 0.025 Gpc; the
    # cluster has dl_max ≈ 12 Gpc → c_0 = 0.10 Gpc. The first bin width
    # changes the unanchored p̂(c_0):
    #   local:   first bin [0, 0.05], p̂ = 7/7 = 1.0  (all close events detected)
    #   cluster: first bin [0, 0.20], p̂ = 0.544     (mix of detected/undetected)
    # The 1D anchor (d_L=0)=0.7931, intermediate (d_L=0.05)=1.0 are
    # h-INDEPENDENT scalars baked into the code. So the anchored 1D
    # interpolator at cluster c_0=0.10 reads:
    #   d_L=0    -> 0.7931
    #   d_L=0.05 -> 1.0
    #   d_L=0.10 -> 0.544 (cluster p̂(c_0))
    # 2D unanchored at cluster c_0 reads p̂(c_0)=0.544 (no anchor); linear
    # extrapolation through bins 0 and 1 (slope ≈ -(p̂_bin1-p̂_bin0)/Δd_L).
    # We approximate p̂_bin1 ≈ 0.45 from monotonic decline in P_det with
    # d_L; sensitivity tested below.
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ANALYTICAL CLUSTER-SCALE PROJECTION (h=0.73)")
    print("=" * 60)
    print("Inputs (from Phase 45 T10 cluster docs):")
    cluster_c0 = 0.10
    cluster_phat_c0 = 0.544  # from outputs/phase45/pdet_asymptote.json
    cluster_phat_bin1 = 0.45  # ~ assumption; sensitivity below
    print(f"  cluster c_0 = {cluster_c0} Gpc")
    print(f"  cluster p̂(c_0) = {cluster_phat_c0} (Phase 45 T10)")
    print(f"  cluster p̂(bin 1, ≈3·c_0) ≈ {cluster_phat_bin1} (assumed; sensitivity below)")

    # 1D anchored window mean on [0, c_0_cluster]
    # Segment A: [0, 0.05] linear from 0.7931 to 1.0; mean = (0.7931+1.0)/2
    # Segment B: [0.05, 0.10] linear from 1.0 to p̂_cluster(c_0); mean = (1.0+p̂)/2
    seg_a_width = _D_INTERMEDIATE_ANCHOR_GPC - 0.0
    seg_b_width = cluster_c0 - _D_INTERMEDIATE_ANCHOR_GPC
    seg_a_mean = 0.5 * (_P_MAX_EMPIRICAL_ANCHOR + _P_INTERMEDIATE_EMPIRICAL)
    seg_b_mean = 0.5 * (_P_INTERMEDIATE_EMPIRICAL + cluster_phat_c0)
    lift_1d_cluster = (seg_a_mean * seg_a_width + seg_b_mean * seg_b_width) / cluster_c0

    # 2D unanchored window mean on [0, c_0_cluster] (linear extrapolation
    # through bins 0,1; slope = (p̂_bin1 - p̂_bin0) / (Δd_L_bin)). Bin0 center
    # at c_0, bin1 center at 3·c_0 → Δ = 2·c_0.
    dl_bin1_center = 3.0 * cluster_c0
    slope_2d = (cluster_phat_bin1 - cluster_phat_c0) / (dl_bin1_center - cluster_c0)
    p_2d_at_zero = cluster_phat_c0 + slope_2d * (0.0 - cluster_c0)
    p_2d_at_005 = cluster_phat_c0 + slope_2d * (0.05 - cluster_c0)
    lift_2d_cluster = 0.5 * (p_2d_at_zero + cluster_phat_c0)  # mean of linear

    print("\nProjected cluster-scale lifts on [0, 0.10]:")
    print(f"  1D anchored window mean: {lift_1d_cluster:.4f}")
    print(
        f"    (segment [0, 0.05] mean = {seg_a_mean:.4f}; "
        f"segment [0.05, 0.10] mean = {seg_b_mean:.4f})"
    )
    print(f"  2D unanchored window mean: {lift_2d_cluster:.4f}")
    print(f"    (extrap to d_L=0: {p_2d_at_zero:.4f}; bin0 center: {cluster_phat_c0:.4f})")
    print(f"  Lift difference (2D - 1D): {lift_2d_cluster - lift_1d_cluster:+.4f}")

    cluster_gate = _classify_gate(lift_2d_cluster, lift_1d_cluster)
    print(f"\n>>> Cluster-projection gate: {cluster_gate}")

    # Sensitivity sweep over plausible p̂_bin1 values
    print("\nSensitivity to p̂(bin1) assumption:")
    for phat1 in [0.40, 0.45, 0.50, 0.55]:
        slope = (phat1 - cluster_phat_c0) / (dl_bin1_center - cluster_c0)
        p0 = cluster_phat_c0 + slope * (0 - cluster_c0)
        lift_alt = 0.5 * (p0 + cluster_phat_c0)
        gate_alt = _classify_gate(lift_alt, lift_1d_cluster)
        print(f"  p̂_bin1={phat1:.2f}: 2D lift = {lift_alt:.4f} → gate: {gate_alt.split(' ')[0]}")

    summary["cluster_projection"] = {
        "note": (
            "Analytical projection using known cluster scale c_0=0.10 Gpc and "
            "p̂(c_0)=0.544 from Phase 45 T10 (pdet_asymptote.json). Local "
            "audit cannot directly measure cluster behaviour because local "
            "dl_max ≈ 2.77 Gpc vs cluster ≈ 12 Gpc."
        ),
        "cluster_c0_Gpc": cluster_c0,
        "cluster_phat_c0": cluster_phat_c0,
        "assumed_phat_bin1": cluster_phat_bin1,
        "anchored_1D_window_mean_cluster": lift_1d_cluster,
        "_2D_unanchored_window_mean_cluster": lift_2d_cluster,
        "lift_difference_2D_minus_1D_cluster": lift_2d_cluster - lift_1d_cluster,
        "gate_verdict_cluster_projection": cluster_gate,
    }
    summary["headline_gate_cluster_projection"] = cluster_gate

    out_json = OUTPUT_DIR / "channel_audit.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
