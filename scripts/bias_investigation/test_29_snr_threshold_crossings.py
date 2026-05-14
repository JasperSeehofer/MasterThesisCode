"""Test 29: verify whether residual post-F1 p_det spikes correlate with
SNR-threshold integer crossings of individual injections.

Hypothesis (from `.planning/HANDOFF-PHASE49-MECHANISM-VERIFY-20260514.md`):

  F1 fixed `dl_edges` so that the histogram support no longer drifts with h.
  But two h-dependent processes still change per-bin (detected/total) counts:

    (A) d_L motion across fixed bin edges
        Each injection's `d_L_target(z_inj, h) = dist(z_inj, h)` changes
        with h.  When it crosses a fixed dl_edge, the (i_dL) bin its
        contribution enters changes, shifting BOTH `n_total` and (if
        detected) `n_detected` in two bins by ±1.

    (B) SNR-threshold crossings
        Each injection's `SNR(h) = SNR_raw * d_L(z, h_inj) / d_L(z, h)`
        changes with h.  When `SNR(h)` crosses the threshold (20) at some
        h*, only `n_detected` in that injection's bin flips by ±1.

  Because `M_edges` is built from h-independent `M_source · (1 + z_inj)`
  (since the injection campaign's `z_inj` is fixed) and `dl_edges` is now
  h-stable, the bin a *fixed query* (d_L_q, M_z_q) falls in is constant
  across h.  Bin-content jumps therefore come ONLY from (A) and (B).

Each spike `|Δp_det / Δh|` at a fixed query can therefore be attributed
to a specific count of A-events and B-events between consecutive h
values, and the expected `Δp_det` magnitude is
``(Δn_det × n_total_old − Δn_total × n_det_old) / (n_total_old · n_total_new)``.

Decision tree (handoff §"Expected output"):

  * (B) dominates each spike, magnitudes match → SNR-threshold mechanism
    CONFIRMED → write `.planning/PHASE-49-F4-PLAN.md` and engage
    `/physics-change` for the Farr 2019 refactor.
  * (A) dominates → F1 did not fully fix what it was supposed to fix;
    re-open `gsd-debug` for a new look at bin-edge construction.
  * Neither explains the spikes → third mechanism (interpolation knot
    discontinuity, floating-point precision, ...); re-open `gsd-debug`.
  * Mixed → write F4 plan with note that F4 closes (B) but (A) needs
    a separate decision.

The script is CPU-only.  Loading injections + 31 h-evaluations × 105k
events ≈ a few seconds; writing the report ≈ 10s total wall.

Output: `outputs/phase46_merged/test_29_snr_threshold_crossings.json`
        + summary table on stdout.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from master_thesis_code.bayesian_inference.simulation_detection_probability import (  # noqa: E402
    _DL_PADDING_FACTOR,
    SimulationDetectionProbability,
)
from master_thesis_code.constants import INJECTION_DATA_DIR, SNR_THRESHOLD  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "outputs" / "phase46_merged"
OUTPUT_FILE = OUTPUT_DIR / "test_29_snr_threshold_crossings.json"

# h-grid: Δh=0.0005 across the post-F1 MAP region [0.730, 0.745].
# Twice the resolution of the Phase 48 production grid (Δh=0.001 dense core)
# so threshold crossings between successive trial values are resolved
# rather than hidden inside one step.
H_MIN: float = 0.730
H_MAX: float = 0.745
H_STEP: float = 0.0005

# Spike threshold: |Δp_det| > 0.002 between consecutive h.  Sized to
# catch a single integer flip in a bin with ~500 injections (1/500=0.002)
# — the SNR-threshold mechanism predicts such 1/N_bin jumps.  A coarser
# threshold of 0.01 (the original handoff guess) only triggers in
# bins with ≤100 injections; the production bins around the MAP have
# 500–2000.
SPIKE_THRESHOLD: float = 0.002

# Whether to print the full per-h trace for every query.
VERBOSE_TRACE: bool = False

# Query points targeted at where DETECTIONS actually live, per a
# direct probe of the injection pool at h ∈ [0.730, 0.745]:
#
#   detected d_L range ≈ [0.03, 0.54] Gpc (median 0.20)
#   detected M_z range ≈ [180k, 920k] M⊙ (median 445k)
#   n_det grows monotonically from 326 → 342 over Δh=0.015
#     ⇒ +16 detections / Δh=0.015 ≈ +1 per Δh=0.001 step.
#     This is the smoking-gun signature of SNR-threshold integer
#     crossings: each crossing flips one injection's detected status,
#     adding ~1/N_bin to p_det in that bin (handoff §"Hypothesis").
#
# Each (d_L, M_z) pair indexes a single (i_dL, j_M) bin in the production
# grid; multiple queries sample different bins so we see whether the
# mechanism is universal or bin-specific.
QUERY_POINTS: list[tuple[float, float]] = [
    (dl, m)
    for dl in [0.08, 0.12, 0.18, 0.25, 0.32, 0.38, 0.45, 0.52]
    for m in [1.5e5, 2.5e5, 4.0e5, 5.5e5, 7.0e5, 9.0e5]
]


def find_bin_index(value: float, edges: npt.NDArray[np.float64]) -> int:
    """Return the bin index ``i`` such that ``edges[i] <= value < edges[i+1]``.

    Returns -1 if ``value`` is below the first edge, ``len(edges)-1`` if
    above the last (these are out-of-grid sentinels, matching the
    semantics of ``np.digitize`` minus 1).
    """
    idx = int(np.digitize(value, edges)) - 1
    return idx


def compute_per_h_state(
    sdp: SimulationDetectionProbability,
    h: float,
    dl_edges: npt.NDArray[np.float64],
    M_edges: npt.NDArray[np.float64],  # noqa: N803
) -> dict[str, npt.NDArray[Any]]:
    """For trial h, compute the per-injection (d_L_target, SNR, dl_bin, M_bin,
    detected) state used to attribute mechanism (A) vs (B).

    Returns a dict with arrays of shape (N_inj,).
    """
    d_L_target, snr_rescaled = sdp._rescale_snr(h)  # noqa: SLF001
    # M_z is observer-frame M_source · (1 + z_inj) — h-independent because
    # both M_source and z_inj come from the injection campaign metadata.
    M_z = sdp._M_arr * (1.0 + sdp._z_arr)  # noqa: N806, SLF001

    dl_bin = np.digitize(d_L_target, dl_edges) - 1
    M_bin = np.digitize(M_z, M_edges) - 1  # noqa: N806
    detected = snr_rescaled >= sdp._snr_threshold  # noqa: SLF001

    return {
        "d_L_target": d_L_target,
        "snr": snr_rescaled,
        "dl_bin": dl_bin.astype(np.int32),
        "M_bin": M_bin.astype(np.int32),
        "detected": detected.astype(np.bool_),
    }


def attribute_spike(
    state_prev: dict[str, npt.NDArray[Any]],
    state_curr: dict[str, npt.NDArray[Any]],
    i_dl: int,
    j_M: int,  # noqa: N803
) -> dict[str, Any]:
    """For the transition h_prev -> h_curr, count A-events and B-events
    affecting bin (i_dl, j_M).

    A-event: injection's `dl_bin` changes such that either prev or curr
             lies in the queried bin.  Counted by the change in `n_total`
             and `n_detected` contributed by injections whose bin
             membership flipped on either side of the queried bin.

    B-event: injection's `dl_bin` and `M_bin` stay equal to the queried
             bin in both prev and curr, but its `detected` status flipped.
    """
    in_bin_prev = (state_prev["dl_bin"] == i_dl) & (state_prev["M_bin"] == j_M)
    in_bin_curr = (state_curr["dl_bin"] == i_dl) & (state_curr["M_bin"] == j_M)

    # Injections that exited the bin (prev in, curr out) — A-events
    exited = in_bin_prev & ~in_bin_curr
    # Injections that entered the bin (prev out, curr in) — A-events
    entered = ~in_bin_prev & in_bin_curr
    # Injections that stayed in the bin — only their detection can flip
    stayed = in_bin_prev & in_bin_curr

    det_prev = state_prev["detected"]
    det_curr = state_curr["detected"]

    # B-events: stayed in bin but detection flipped
    b_flip_up = stayed & ~det_prev & det_curr  # gained detection
    b_flip_down = stayed & det_prev & ~det_curr  # lost detection
    b_events = int(np.sum(b_flip_up)) + int(np.sum(b_flip_down))

    n_total_prev = int(np.sum(in_bin_prev))
    n_total_curr = int(np.sum(in_bin_curr))
    n_det_prev = int(np.sum(in_bin_prev & det_prev))
    n_det_curr = int(np.sum(in_bin_curr & det_curr))

    # Decompose Δn_det into A-contribution and B-contribution
    delta_n_det_from_A = int(np.sum(entered & det_curr)) - int(np.sum(exited & det_prev))
    delta_n_det_from_B = int(np.sum(b_flip_up)) - int(np.sum(b_flip_down))
    delta_n_total_from_A = int(np.sum(entered)) - int(np.sum(exited))

    # Sanity check: contributions sum to observed delta
    assert delta_n_det_from_A + delta_n_det_from_B == n_det_curr - n_det_prev, (
        "decomposition consistency check failed"
    )
    assert delta_n_total_from_A == n_total_curr - n_total_prev, "n_total decomposition check failed"

    return {
        "n_total_prev": n_total_prev,
        "n_total_curr": n_total_curr,
        "n_det_prev": n_det_prev,
        "n_det_curr": n_det_curr,
        "a_entered": int(np.sum(entered)),
        "a_exited": int(np.sum(exited)),
        "a_entered_detected": int(np.sum(entered & det_curr)),
        "a_exited_detected": int(np.sum(exited & det_prev)),
        "b_flip_up": int(np.sum(b_flip_up)),
        "b_flip_down": int(np.sum(b_flip_down)),
        "b_events": b_events,
        "delta_n_total_from_A": delta_n_total_from_A,
        "delta_n_det_from_A": delta_n_det_from_A,
        "delta_n_det_from_B": delta_n_det_from_B,
        # injection-index snapshots (for forensic reporting; cap at 20 per category)
        "b_flip_indices": np.where(b_flip_up | b_flip_down)[0][:20].tolist(),
        "a_entered_indices": np.where(entered)[0][:20].tolist(),
        "a_exited_indices": np.where(exited)[0][:20].tolist(),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SimulationDetectionProbability from", INJECTION_DATA_DIR)
    sdp = SimulationDetectionProbability(
        injection_data_dir=INJECTION_DATA_DIR,
        snr_threshold=SNR_THRESHOLD,
    )
    n_inj = len(sdp._z_arr)  # noqa: SLF001
    print(f"  loaded {n_inj} injections, h_inj values: {sdp._h_values_found}")

    # Build the h-stable edges directly (F1 logic, mirrored from
    # `_build_grid_2d` / `_build_grid_1d`)
    dl_global_max = sdp._dl_global_max  # noqa: SLF001
    dl_edges = np.linspace(0.0, dl_global_max, sdp._dl_bins + 1)  # noqa: SLF001
    # M_z is observer-frame M_source · (1 + z_inj)
    M_arr_obs = sdp._M_arr * (1.0 + sdp._z_arr)  # noqa: N806, SLF001
    M_min = float(np.min(M_arr_obs)) * 0.9  # noqa: N806
    M_max = float(np.max(M_arr_obs)) * 1.1  # noqa: N806
    M_edges = np.geomspace(M_min, M_max, sdp._mass_bins + 1)  # noqa: N806, SLF001

    # Audit: F1's dl_max chosen at h_min, padded by _DL_PADDING_FACTOR
    # Confirms in stdout.
    h_min_audit = sdp._h_prior_min  # noqa: SLF001
    print(f"  dl_global_max = {dl_global_max:.6f} Gpc")
    print(f"  expected:     = max(dist(z, h={h_min_audit:.4f})) * {_DL_PADDING_FACTOR}")
    print(f"  M_edges range = [{M_edges[0]:.3e}, {M_edges[-1]:.3e}]")

    h_grid = np.arange(H_MIN, H_MAX + 1e-9, H_STEP)
    print(f"  h-grid: {len(h_grid)} points from {h_grid[0]:.4f} to {h_grid[-1]:.4f}")

    # Pre-compute per-h state for every injection
    print("Pre-computing per-h injection state...")
    states: list[dict[str, npt.NDArray[Any]]] = []
    for h in h_grid:
        states.append(compute_per_h_state(sdp, float(h), dl_edges, M_edges))
    print(f"  done ({len(states)} h-values × {n_inj} injections)")

    # For each query point, identify the (i_dl, j_M) bin and walk the h-grid
    all_query_reports: list[dict[str, Any]] = []
    print("\nProbing query points:")
    for d_L_q, M_z_q in QUERY_POINTS:
        i_dl = find_bin_index(d_L_q, dl_edges)
        j_M = find_bin_index(M_z_q, M_edges)
        if i_dl < 0 or i_dl >= sdp._dl_bins:  # noqa: SLF001
            print(f"  [SKIP] query ({d_L_q:.3f}, {M_z_q:.2e}) out of d_L grid")
            continue
        if j_M < 0 or j_M >= sdp._mass_bins:  # noqa: SLF001
            print(f"  [SKIP] query ({d_L_q:.3f}, {M_z_q:.2e}) out of M grid")
            continue

        bin_lo_dl = dl_edges[i_dl]
        bin_hi_dl = dl_edges[i_dl + 1]
        bin_lo_M = M_edges[j_M]  # noqa: N806
        bin_hi_M = M_edges[j_M + 1]  # noqa: N806
        print(
            f"  query ({d_L_q:.3f} Gpc, {M_z_q:.2e} M⊙) → "
            f"bin (i_dL={i_dl}, j_M={j_M}); "
            f"d_L∈[{bin_lo_dl:.4f},{bin_hi_dl:.4f}], "
            f"M_z∈[{bin_lo_M:.3e},{bin_hi_M:.3e}]"
        )

        # Walk the h-grid and compute p_det at the bin
        h_trace: list[dict[str, Any]] = []
        for i, h in enumerate(h_grid):
            in_bin = (states[i]["dl_bin"] == i_dl) & (states[i]["M_bin"] == j_M)
            n_total = int(np.sum(in_bin))
            n_det = int(np.sum(in_bin & states[i]["detected"]))
            p_det = n_det / n_total if n_total > 0 else 0.0
            h_trace.append({"h": float(h), "n_total": n_total, "n_det": n_det, "p_det": p_det})

        # Identify spikes
        spikes: list[dict[str, Any]] = []
        for k in range(1, len(h_grid)):
            d_pdet = h_trace[k]["p_det"] - h_trace[k - 1]["p_det"]
            if abs(d_pdet) < SPIKE_THRESHOLD:
                continue
            attribution = attribute_spike(states[k - 1], states[k], i_dl, j_M)
            spikes.append(
                {
                    "h_prev": h_trace[k - 1]["h"],
                    "h_curr": h_trace[k]["h"],
                    "p_det_prev": h_trace[k - 1]["p_det"],
                    "p_det_curr": h_trace[k]["p_det"],
                    "delta_p_det": d_pdet,
                    "attribution": attribution,
                }
            )

        if VERBOSE_TRACE:
            print(
                "    per-h: "
                + ", ".join(
                    f"{t['h']:.4f}:{t['n_det']}/{t['n_total']}={t['p_det']:.4f}"
                    for t in h_trace[::4]  # every 4th h to keep terse
                )
            )
            n_total_uniq = sorted({t["n_total"] for t in h_trace})
            n_det_uniq = sorted({t["n_det"] for t in h_trace})
            print(f"    unique n_total across h: {n_total_uniq}")
            print(f"    unique n_det across h:   {n_det_uniq}")

        print(f"    {len(spikes)} spike(s) with |Δp_det| > {SPIKE_THRESHOLD}")
        for s in spikes:
            a = s["attribution"]
            print(
                f"      h: {s['h_prev']:.4f}→{s['h_curr']:.4f}  "
                f"p_det: {s['p_det_prev']:.4f}→{s['p_det_curr']:.4f}  "
                f"Δ={s['delta_p_det']:+.4f}  "
                f"|  A: total {a['delta_n_total_from_A']:+d}, det {a['delta_n_det_from_A']:+d}  "
                f"|  B: {a['b_events']} flips (det {a['delta_n_det_from_B']:+d})"
            )

        all_query_reports.append(
            {
                "query": {"d_L_Gpc": d_L_q, "M_z_Msun": M_z_q},
                "bin_indices": {"i_dl": int(i_dl), "j_M": int(j_M)},
                "bin_edges": {
                    "dl_lo": float(bin_lo_dl),
                    "dl_hi": float(bin_hi_dl),
                    "M_lo": float(bin_lo_M),
                    "M_hi": float(bin_hi_M),
                },
                "h_trace": h_trace,
                "spikes": spikes,
                "n_spikes": len(spikes),
            }
        )

    # Aggregate stats: how often is each mechanism dominant?
    #
    # Decompose the observed Δp_det into A-only and B-only contributions:
    #   p_curr - p_prev = n_det_curr/n_total_curr - n_det_prev/n_total_prev
    #
    # A-only effect (only n_total motion, no detection flips inside bin):
    #   Δp_A = (n_det_prev + δn_det_A) / (n_total_prev + δn_total_A)
    #          − n_det_prev / n_total_prev
    #
    # B-only effect (only SNR-threshold flips inside the bin, no d_L motion):
    #   Δp_B = (n_det_prev + δn_det_B) / n_total_prev
    #          − n_det_prev / n_total_prev
    #        = δn_det_B / n_total_prev
    #
    # NB δn_total_A may be 0 with δn_det_A != 0 (A-flip stayed in bin but its
    # detection status was already opposite of what it would have at new h --
    # treated as A-driven since the underlying cause is bin membership).
    total_spikes = sum(r["n_spikes"] for r in all_query_reports)
    b_dominant = 0
    a_dominant = 0
    mixed = 0
    pure_B = 0
    pure_A = 0
    total_dp_from_A_sq = 0.0
    total_dp_from_B_sq = 0.0
    for r in all_query_reports:
        for s in r["spikes"]:
            a = s["attribution"]
            n_p = max(a["n_total_prev"], 1)
            n_a = max(a["n_total_curr"], 1)  # for the new denominator
            dp_a = (a["n_det_prev"] + a["delta_n_det_from_A"]) / n_a - a["n_det_prev"] / n_p
            dp_b = a["delta_n_det_from_B"] / n_p
            a["dp_A_only"] = float(dp_a)
            a["dp_B_only"] = float(dp_b)
            total_dp_from_A_sq += dp_a * dp_a
            total_dp_from_B_sq += dp_b * dp_b
            adp_a = abs(dp_a)
            adp_b = abs(dp_b)
            # "Pure" = the other channel contributes < 10% of this spike
            tot = adp_a + adp_b
            tot = max(tot, 1e-30)
            if adp_b / tot > 0.9:
                pure_B += 1
                b_dominant += 1
            elif adp_a / tot > 0.9:
                pure_A += 1
                a_dominant += 1
            elif adp_b > adp_a:
                b_dominant += 1
                mixed += 1
            elif adp_a > adp_b:
                a_dominant += 1
                mixed += 1
            else:
                mixed += 1

    print("\n=== AGGREGATE ===")
    print(f"  total spikes (across {len(all_query_reports)} queries): {total_spikes}")
    print(f"    pure B (SNR-threshold only, >90% of |Δp|): {pure_B}")
    print(f"    pure A (d_L bin crossing only, >90% of |Δp|): {pure_A}")
    print(f"    B-dominant (mixed): {b_dominant - pure_B}")
    print(f"    A-dominant (mixed): {a_dominant - pure_A}")
    print(f"    other / tied: {mixed - (b_dominant - pure_B) - (a_dominant - pure_A)}")
    if total_dp_from_A_sq + total_dp_from_B_sq > 0:
        frac_A = total_dp_from_A_sq / (total_dp_from_A_sq + total_dp_from_B_sq)
        frac_B = total_dp_from_B_sq / (total_dp_from_A_sq + total_dp_from_B_sq)
        print(f"  Σ(Δp)² decomposition: A = {frac_A:.1%}, B = {frac_B:.1%}")
    else:
        frac_A = frac_B = float("nan")

    # Verdict logic — uses the Σ(Δp)² decomposition, which is the
    # right summary statistic for the noise variance the SUM Σ_i log L_i
    # picks up: if Var(p_det) at fixed query splits 70/30 between
    # mechanisms A and B, so does the variance the production posterior
    # inherits, by linearity of variance for independent A and B events.
    if total_spikes == 0:
        verdict = "NO_SPIKES_FOUND"
        decision = (
            "Spike threshold not reached at any query.  "
            "Either threshold too high or spikes are at queries we did not sample.  "
            "Re-run with finer query grid or lower SPIKE_THRESHOLD."
        )
    elif frac_B > 0.7:
        verdict = "SNR_THRESHOLD_MECHANISM_CONFIRMED"
        decision = (
            f">70% of Σ(Δp_det)² noise variance ({frac_B:.0%}) comes from "
            "SNR-threshold crossings (B).  "
            "F4 (Farr 2019 reweighting) is the right next move.  "
            "Write `.planning/PHASE-49-F4-PLAN.md`; engage `/physics-change`."
        )
    elif frac_A > 0.7:
        verdict = "DL_BIN_MECHANISM_DOMINANT"
        decision = (
            f">70% of Σ(Δp_det)² noise variance ({frac_A:.0%}) comes from "
            "d_L bin crossings (A) — F1's stated fix did NOT eliminate the "
            "bin-crossing mechanism for *injection motion across fixed edges* "
            "(distinct from F1's bin-edge-drift target).  "
            "F4 still closes both A and B; if F4 is infeasible, F2 "
            "(histogram smoothing along d_L) or F3 (denser injections) addresses A."
        )
    elif frac_B > frac_A:
        verdict = "BOTH_MECHANISMS_B_DOMINANT"
        decision = (
            f"Both mechanisms contribute; SNR-threshold (B) modestly dominant "
            f"({frac_B:.0%} of Σ(Δp)² vs A={frac_A:.0%}).  "
            "F4 closes both A and B and is the principled fix.  "
            "Proceed with F4 plan."
        )
    else:
        verdict = "BOTH_MECHANISMS_A_DOMINANT"
        decision = (
            f"Both mechanisms contribute; d_L motion (A) modestly dominant "
            f"({frac_A:.0%} of Σ(Δp)² vs B={frac_B:.0%}).  "
            "F1 left A intact (motion across fixed edges).  "
            "F4 closes both A and B and is the principled fix."
        )

    print("\n=== VERDICT ===")
    print(f"  {verdict}")
    print(f"  {decision}")

    report = {
        "metadata": {
            "git_commit": _git_head(),
            "h_grid": h_grid.tolist(),
            "h_step": H_STEP,
            "snr_threshold": SNR_THRESHOLD,
            "spike_threshold": SPIKE_THRESHOLD,
            "dl_global_max": float(dl_global_max),
            "dl_bins": int(sdp._dl_bins),  # noqa: SLF001
            "M_bins": int(sdp._mass_bins),  # noqa: SLF001
            "n_injections": int(n_inj),
            "h_inj_values": [float(h) for h in sdp._h_values_found],  # noqa: SLF001
        },
        "queries": all_query_reports,
        "aggregate": {
            "total_spikes": total_spikes,
            "pure_B_snr_threshold": pure_B,
            "pure_A_dl_bin": pure_A,
            "b_dominant_total": b_dominant,
            "a_dominant_total": a_dominant,
            "mixed": mixed,
            "sum_dp_A_sq": float(total_dp_from_A_sq),
            "sum_dp_B_sq": float(total_dp_from_B_sq),
            "frac_variance_A": float(frac_A) if total_spikes > 0 else float("nan"),
            "frac_variance_B": float(frac_B) if total_spikes > 0 else float("nan"),
        },
        "verdict": verdict,
        "decision": decision,
    }
    with OUTPUT_FILE.open("w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nReport written to: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")


def _git_head() -> str:
    """Return short HEAD SHA for the report metadata."""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
