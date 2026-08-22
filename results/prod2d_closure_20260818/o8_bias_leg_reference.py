r"""O8 -- FUSED BIAS-LEG REFERENCE (registered in ``PREREGISTRATION_SELFGEN_CONTROL.md``,
"O8 -- FUSED BIAS-LEG REFERENCE: REGISTRATION", 2026-08-22, author-approved rows #163 item 1
+ morning "Run now"; A21/A22 in force).

Question (the O7 amendment-1 gap): what is the fused-cell fleet **bias**
(``mean_h - h_gen`` on the matched channel), the BAND-C leg O7 could not
close?

**Construction (zero-``evaluate()`` transfer, the O6/O7 pattern extended to
the full grid):** per C-SG-F seed (:data:`o4_pairing_test.F_SEEDS`), the
harness replica of the fused numerator
(:func:`o6_reference_derivation.r_prod_b_num`, UNMODIFIED import) is
evaluated at EVERY node of :data:`darksiren_emri.validation.correspondence_1d.
H_GRID_41` (not just the 0.725/0.735 pair O6/O7 used), with
:func:`o4_pairing_test.build_aligned_tables`-PATTERN ``S_bar_phi``/
``beta_Gbar_phi`` tables built for the full grid by calling the SAME pinned
production leaf functions (:func:`precompute_phi_marginal_survival`,
:func:`precompute_phi_selection_integrals`) with the 41-node ``h_values``
list instead of ``[H_LO, H_HI]`` -- a generalization of the two-node call,
not a reimplementation of either leaf.

The per-event ``matched`` posterior (``B_num_replica(h) / beta_Gbar_phi(h)``)
is combined into ``mean_h_fused_ref(seed)`` by the COMMITTED
:func:`~darksiren_emri.validation.correspondence_1d.seed_statistics_from_matrix`
reduction -- the SAME combine :func:`~darksiren_emri.validation.
selfgen_control.csg_channel_scores` uses for every C-SG shard's ``matched``
channel (``csg_channel_matrices`` builds ``matched = B_num / beta_Gbar_phi``
from a diagnostics CSV pivot; this instrument builds the identical linear
matrix directly from the in-memory replica, then hands it to the SAME
reduction function). ``seed_statistics_from_matrix``'s own docstring is
explicit that its ``vals`` argument is "per-event likelihoods (linear, not
log)" (``correspondence_1d.py:2119``) -- the matched matrix built here is
linear (``B_num/beta_Gbar_phi``, no log taken before the call), matching that
contract exactly; the function takes the log internally
(``combine_log_likelihood``) before forming the posterior mean. No
hand-rolled combine is used.

Zero ``BayesianStatistics.evaluate()`` calls by construction, exactly as O6/
O7: tables from the pinned production leaf functions, event sets from the
deterministic ``draw_csg_realization`` redraw (O4 GATE R4 bit-exactness).

**FLAGGED SPEC AMBIGUITY (A21 -- disclosed, not silently resolved):** the O8
registration says the matched posterior is "combined by the COMMITTED
``seed_statistics_from_matrix``-equivalent path (``csg_channel_matrices``/
``compute`` conventions -- the same combine every C-SG shard used)" but does
not name a single callable. Two committed paths exist:
(a) :func:`csg_channel_scores` (CSV-to-matrix wrapper -> ``seed_statistics_
from_matrix``), which this instrument cannot use directly because it has no
diagnostics CSV to pivot (this is a zero-``evaluate()`` in-memory replica);
(b) :func:`seed_statistics_from_matrix` itself, the core reduction
``csg_channel_scores`` delegates to, callable on any ``(n_events, n_nodes)``
linear matrix. This instrument uses (b) directly on the in-memory
``B_num_replica/beta_Gbar_phi`` matrix -- the only one of the two callables
that can consume a matrix that was never written to a CSV -- and considers
this the "``seed_statistics_from_matrix``-equivalent path" the registration
names verbatim. This choice cannot silently soften GATE M8: M8 compares this
instrument's ``mean_h`` against the BANKED end-to-end records' ``channel_
scores.matched.mean_h``, which were themselves produced via path (a) on the
SAME underlying reduction (b) -- so a wrong combine choice here would show up
as an M8 failure, not a silent pass.

Usage:
    # Full registered run (all 15 F seeds x 41 h-nodes; ~30-60 min, <=6 GB):
    uv run python results/prod2d_closure_20260818/o8_bias_leg_reference.py

    # Smoke mode (2 h-nodes, 1 seed, gates skipped -- plumbing proof only):
    uv run python results/prod2d_closure_20260818/o8_bias_leg_reference.py --smoke
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import o4_pairing_test as o4  # noqa: E402
import o6_reference_derivation as o6r  # noqa: E402

from darksiren_emri.bayesian_inference.bayesian_statistics import (  # noqa: E402
    precompute_phi_marginal_survival,
    precompute_phi_selection_integrals,
)
from darksiren_emri.bayesian_inference.simulation_detection_probability import (  # noqa: E402
    SimulationDetectionProbability,
)
from darksiren_emri.validation import correspondence_1d as c1d  # noqa: E402
from darksiren_emri.validation.correspondence_1d import CRB_CSV_PATH, H_GRID_41  # noqa: E402
from darksiren_emri.validation.selfgen_control import (  # noqa: E402
    CsgCompletenessModel,
    CsgDetectionProbabilityModel,
    build_csg_selection_objects,
    draw_csg_realization,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results/prod2d_closure_20260818"
OUT_PATH = RESULTS_DIR / "o8_bias_leg_reference_output.json"
SMOKE_OUT_PATH = RESULTS_DIR / "o8_smoke_output.json"

REGISTRATION_SECTION = (
    "results/prod2d_closure_20260818/PREREGISTRATION_SELFGEN_CONTROL.md, "
    "O8 -- FUSED BIAS-LEG REFERENCE: REGISTRATION (2026-08-22, rows #163 item 1 "
    "+ morning 'Run now')"
)

N_EVENTS: int = 200
ARM: str = "csgf"

# GATE T8 (registration verbatim): O4-banked full-precision beta_Gbar_phi at
# the two-node pair, table-construction identity to 1e-9 relative.
GATE_T8_BETA_725: float = 893324861.1081496
GATE_T8_BETA_735: float = 883510508.7955135
GATE_T8_TOL: float = 1.0e-9

# GATE M8 (the transfer anchor): banked end-to-end fused records, matched
# channel mean_h, for the three anchor seeds.
ANCHOR_SEEDS: tuple[int, ...] = (910101, 910105, 910113)
ANCHOR_RECORD_PATHS: dict[int, Path] = {
    910101: RESULTS_DIR / "o6_work" / "f6_out" / "csgf_seed910101.json",
    910105: RESULTS_DIR / "o7_work" / "s7_910105_out" / "csgf_seed910105.json",
    910113: RESULTS_DIR / "o7_work" / "s7_910113_out" / "csgf_seed910113.json",
}
GATE_M8_TOL: float = 2.0e-4

# Primary band (registration verbatim): the frozen C-SG bias SELF-CONSISTENT
# edge, ported per A17.
BAND_BIAS_LEG: float = 0.0209
# Banked off-cell bias leg (C-SG readout), reported alongside for the paired
# delta -- NOT subtracted from bias_fused_15, only displayed.
OFF_CELL_BIAS_BANKED: float = -0.0665

H_GEN: float = o4.H_GEN  # 0.73, == correspondence_1d.H_TRUE


def _a22_stamp() -> dict[str, str]:
    """A22: provenance stamped at run START (copied pattern from
    ``results/campaign51_20260728/realistic_20260729/p3_shape_rescore.py:38-46``).
    """
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "darksiren_emri/"], capture_output=True, text=True
    ).stdout.strip()
    return {"git_commit_at_start": commit, "estimator_tree_dirty": dirty or "clean"}


def build_aligned_tables_full_grid(
    completeness: CsgCompletenessModel,
    detection_probability: CsgDetectionProbabilityModel,
    h_values: list[float],
) -> tuple[
    dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    dict[float, float],
]:
    """``S_bar_phi``/``beta_Gbar_phi`` tables at an ARBITRARY h-list, via the
    SAME pinned production leaf functions
    :func:`o4_pairing_test.build_aligned_tables` calls (no reimplementation);
    this is that function's ``h_values`` generalized from the hardcoded
    ``[H_LO, H_HI]`` pair to the full :data:`H_GRID_41`, per the O8
    registration's "S_bar_phi/beta_Gbar_phi tables built for the full grid"
    construction line.
    """
    detection_probability_concrete = cast(SimulationDetectionProbability, detection_probability)
    phi_table = precompute_phi_marginal_survival(
        h_values, detection_probability_concrete, z_max_cap=o4.REDSHIFT_UPPER_LIMIT
    )
    _beta_g_phi, beta_gbar_phi = precompute_phi_selection_integrals(
        h_values, phi_table, completeness
    )
    return phi_table, beta_gbar_phi


def mean_h_fused_ref_for_seed(
    seed: int,
    h_grid: list[float],
    grid_arr: npt.NDArray[np.float64],
    completeness: CsgCompletenessModel,
    detection_probability: CsgDetectionProbabilityModel,
    donor_rows: pd.DataFrame,
    phi_table: dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    beta_gbar_phi: dict[float, float],
) -> tuple[float, npt.NDArray[np.float64]]:
    """One seed's ``mean_h_fused_ref``: the deterministic redraw + per-event,
    per-h-node ``B_num`` replica (:func:`o6_reference_derivation.
    r_prod_b_num`, UNMODIFIED), assembled into the linear ``matched`` matrix
    and reduced by :func:`~darksiren_emri.validation.correspondence_1d.
    seed_statistics_from_matrix` (the committed combine; see module
    docstring's A21 note).

    Returns ``(mean_h, matched_matrix)``.
    """
    rows, _diag = draw_csg_realization(
        seed, ARM, N_EVENTS, completeness, detection_probability, donor_rows
    )
    geos = o4.event_geometries(rows, completeness)

    n_events = len(geos)
    n_nodes = len(h_grid)
    b_num = np.empty((n_events, n_nodes), dtype=np.float64)
    for j, h in enumerate(h_grid):
        z_grid, s_grid = phi_table[h]
        b_num[:, j] = [o6r.r_prod_b_num(g, h, completeness, z_grid, s_grid) for g in geos]

    beta_vec = np.array([beta_gbar_phi[h] for h in h_grid], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        matched = b_num / beta_vec[np.newaxis, :]

    stats = c1d.seed_statistics_from_matrix(matched, seed, grid_arr, h_true=H_GEN)
    return stats.mean_h, matched


def gate_t8(beta_gbar_phi: dict[float, float]) -> dict[str, Any]:
    beta_725 = beta_gbar_phi[o4.H_LO]
    beta_735 = beta_gbar_phi[o4.H_HI]
    rel_725 = abs(beta_725 - GATE_T8_BETA_725) / abs(GATE_T8_BETA_725)
    rel_735 = abs(beta_735 - GATE_T8_BETA_735) / abs(GATE_T8_BETA_735)
    passed = rel_725 <= GATE_T8_TOL and rel_735 <= GATE_T8_TOL
    return {
        "beta_gbar_phi_0.725_computed": beta_725,
        "beta_gbar_phi_0.725_registered": GATE_T8_BETA_725,
        "rel_0.725": rel_725,
        "beta_gbar_phi_0.735_computed": beta_735,
        "beta_gbar_phi_0.735_registered": GATE_T8_BETA_735,
        "rel_0.735": rel_735,
        "tol": GATE_T8_TOL,
        "pass": passed,
    }


def gate_m8(mean_h_by_seed: dict[int, float]) -> dict[str, Any]:
    per_anchor: dict[str, Any] = {}
    all_pass = True
    for seed in ANCHOR_SEEDS:
        path = ANCHOR_RECORD_PATHS[seed]
        if not path.is_file():
            per_anchor[str(seed)] = {"pass": False, "reason": f"banked record not found: {path}"}
            all_pass = False
            continue
        record = json.loads(path.read_text())
        banked_mean_h = float(record["channel_scores"]["matched"]["mean_h"])
        computed = mean_h_by_seed[seed]
        diff = abs(computed - banked_mean_h)
        ok = diff <= GATE_M8_TOL
        all_pass = all_pass and ok
        per_anchor[str(seed)] = {
            "banked_record": str(path),
            "banked_mean_h": banked_mean_h,
            "computed_mean_h_fused_ref": computed,
            "abs_diff": diff,
            "tol": GATE_M8_TOL,
            "pass": ok,
        }
    return {"per_anchor": per_anchor, "tol": GATE_M8_TOL, "pass": all_pass}


def run(smoke: bool) -> dict[str, Any]:
    stamp = _a22_stamp()
    print("A22 stamp:", stamp)

    seeds: tuple[int, ...]
    if smoke:
        h_grid = [o4.H_LO, o4.H_HI]
        seeds = (o4.F_SEEDS[0],)
    else:
        h_grid = list(H_GRID_41)
        seeds = o4.F_SEEDS
    grid_arr = np.array(sorted(h_grid), dtype=np.float64)

    completeness, detection_probability = build_csg_selection_objects(h_gen=H_GEN)
    donor_rows = pd.read_csv(CRB_CSV_PATH)

    t0 = time.time()
    phi_table, beta_gbar_phi = build_aligned_tables_full_grid(
        completeness, detection_probability, h_grid
    )
    table_elapsed = time.time() - t0

    per_seed: list[dict[str, Any]] = []
    mean_h_by_seed: dict[int, float] = {}
    t1 = time.time()
    for seed in seeds:
        mean_h, _matched = mean_h_fused_ref_for_seed(
            seed,
            h_grid,
            grid_arr,
            completeness,
            detection_probability,
            donor_rows,
            phi_table,
            beta_gbar_phi,
        )
        mean_h_by_seed[seed] = mean_h
        per_seed.append({"seed": seed, "mean_h_fused_ref": mean_h})
        print(f"seed {seed}: mean_h_fused_ref = {mean_h:.6f}", flush=True)
    seeds_elapsed = time.time() - t1

    output: dict[str, Any] = {
        "registered_in": REGISTRATION_SECTION,
        "instrument": "o8_bias_leg_reference.py",
        "smoke": smoke,
        "a22_stamp": stamp,
        "arm": ARM,
        "h_gen": H_GEN,
        "h_grid_n_nodes": len(h_grid),
        "h_grid": h_grid,
        "n_events": N_EVENTS,
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "table_build_elapsed_s": table_elapsed,
        "per_seed_elapsed_s": seeds_elapsed,
        "zero_evaluate_note": (
            "No BayesianStatistics.evaluate() call; tables from the pinned "
            "production leaf functions (precompute_phi_marginal_survival, "
            "precompute_phi_selection_integrals) at the full h-grid, event "
            "sets from the deterministic draw_csg_realization redraw (O4 "
            "GATE R4 bit-exactness); matched-channel combine via "
            "correspondence_1d.seed_statistics_from_matrix, the SAME "
            "reduction csg_channel_scores/csg_channel_matrices use for every "
            "banked C-SG shard's matched channel (see module docstring A21 "
            "note)."
        ),
        "per_seed": per_seed,
    }

    if smoke:
        output["note"] = (
            "SMOKE MODE: 2 h-nodes (0.725/0.735), 1 seed, GATE T8/M8 SKIPPED "
            "-- plumbing proof only, not a registered statistic."
        )
        return output

    gt8 = gate_t8(beta_gbar_phi)
    gm8 = gate_m8(mean_h_by_seed)
    output["gate_t8"] = gt8
    output["gate_m8"] = gm8

    if not gt8["pass"]:
        output["stop_reason"] = "GATE T8 FAILED"
        out_path = OUT_PATH
        out_path.write_text(json.dumps(output, indent=2))
        print(f"GATE T8 FAILED: {json.dumps(gt8, indent=2)}", file=sys.stderr)
        print(f"wrote {out_path} (partial, gate failure)", file=sys.stderr)
        sys.exit(f"A21 STOP: GATE T8 failed -- {gt8}")

    if not gm8["pass"]:
        output["stop_reason"] = "GATE M8 FAILED"
        out_path = OUT_PATH
        out_path.write_text(json.dumps(output, indent=2))
        print(f"GATE M8 FAILED: {json.dumps(gm8, indent=2)}", file=sys.stderr)
        print(f"wrote {out_path} (partial, gate failure)", file=sys.stderr)
        sys.exit(f"A21 STOP: GATE M8 failed -- {gm8}")

    mean_h_vec = np.array([mean_h_by_seed[s] for s in seeds], dtype=np.float64)
    bias_fused_15 = float(mean_h_vec.mean() - H_GEN)
    band = "BIAS-LEG-CLOSED" if abs(bias_fused_15) <= BAND_BIAS_LEG else "BIAS-LEG-OPEN"

    output["primary"] = {
        "bias_fused_15": bias_fused_15,
        "per_seed_mean_h_fused_ref": mean_h_vec.tolist(),
        "n_seeds": len(seeds),
        "typed": "BANKED, by transfer",
        "anchor_fraction": f"{len(ANCHOR_SEEDS)}/{len(seeds)} end-to-end anchors (GATE M8)",
        "band": band,
        "band_threshold": BAND_BIAS_LEG,
        "off_cell_bias_banked_reported_alongside": OFF_CELL_BIAS_BANKED,
        "reference": (
            "PREREGISTRATION_SELFGEN_CONTROL.md O8 -- FUSED BIAS-LEG "
            "REFERENCE, 'Primary [registered wording]': bias_fused(15) = "
            "mean_s[mean_h_fused_ref(s)] - 0.73; bands: BIAS-LEG-CLOSED iff "
            "|bias_fused(15)| <= 0.0209 else BIAS-LEG-OPEN (two-sided, no "
            "materiality commentary)."
        ),
    }

    out_path = OUT_PATH
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nGATE T8: {'PASS' if gt8['pass'] else 'FAIL'}")
    print(f"GATE M8: {'PASS' if gm8['pass'] else 'FAIL'}")
    print(f"bias_fused_15 = {bias_fused_15:+.6f}  -> {band} (threshold {BAND_BIAS_LEG})")
    print(f"off-cell bias (banked, reported alongside) = {OFF_CELL_BIAS_BANKED:+.6f}")
    print(f"wrote {out_path}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: 2 h-nodes, 1 seed, gates skipped -- plumbing proof only.",
    )
    args = parser.parse_args()

    output = run(smoke=args.smoke)
    if args.smoke:
        SMOKE_OUT_PATH.write_text(json.dumps(output, indent=2))
        print(f"wrote {SMOKE_OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
