r"""[P3-IMP] arms P / PILOT / F-phi / K-flat + scorer.

Registered in ``PREREGISTRATION_P3_TWIN_20260822.md`` (binding; AMENDMENT 1 --
A21: no deviation from arm/gate text). Templates for structure/idioms:
``results/prod2d_closure_20260818/o6_fused_seed_test.py`` (log capture,
idempotent-skip refusal, gate scoring) and ``o7_spot_check.py`` (module-level
seed patch pattern). ``decompose_impostor_leg.py`` supplies secondary 1's
impostor-decomposition formula (reused by import, not reimplemented).

Question (prereg §1): does the catalogue leg's missing per-host S_bar_phi
factor (paired exactly with what its own normalizer, beta_G_phi, already
integrates -- ``bayesian_statistics.py:2065``) own a material share of the
banked B-SEL fleet's H0 bias, when measured as a REAL end-to-end
``BayesianStatistics.evaluate()`` twin cell (``catalogue_numerator_survival
="phi"``) against the byte-identical-to-production ``"off"`` baseline?

**Arms** (verbatim from the prereg's Design table, §3):

- **P (replica gate)**: seed 900101, flag omitted/"off" (byte-identical to
  the banked baseline) -- GATE R-P3.
- **PILOT**: seeds 900101, 900102 (registry order) under "phi" -- realized
  paired sigma-hat for band freezing + costing.
- **F-phi (primary)**: all 12 banked B-SEL seeds under "phi".
- **K-flat (kill test, REPORTED-ONLY)**: seed 900101 under ``"phi_flat"``
  (no registered-faithful override hook exists; see that function's
  docstring for the audited reason).

**Stages** (``--stage {p,pilot,fleet,kflat,score}``):

- ``p``: regenerate the off replica (GATE R-P3: bit-exact/<=1e-12 relative on
  ``L_cat_no_bh``/``B_num``/``combined_no_bh``, wall > 60 s).
- ``pilot``: regenerate 900101/900102 under "phi"; print paired
  Delta_s = mean_h(phi) - mean_h(banked) per seed + realized scatter proxy.
- ``fleet``: regenerate all 12 banked seeds under "phi" -- seeds already
  produced by ``pilot`` are REUSED (disclosed), not re-run.
- ``kflat``: seed 900101 under ``"phi_flat"`` (constant-table kill arm).
- ``score``: applies GATE E-P3 (amended, AMENDMENT 1) and GATE L-P3, then the
  primary Delta-bar(12) with paired SEM, and secondaries 1-4 (prereg §5).

**HARD CONSTRAINTS (launch task, mirrors o4/o6/o7):**

1. Never end a turn to wait on an untracked process -- every ``evaluate()``
   call below is synchronous/blocking (``run_mirror_seed_inprocess``).
2. Every load-bearing claim cites file:line where practical.
3. ``--jobs`` (default 2) is accepted for CLI-shape compatibility with the
   prereg's costing line, but this implementation runs seeds SEQUENTIALLY
   regardless of its value (no subprocess/process-pool fan-out) --
   ``run_mirror_seed_inprocess`` monkeypatches module-level state
   (``_bs_mod.from_cache_or_build``) for its duration, which is not
   safely shareable across concurrent in-process calls in one Python
   process; true 2-wide parallelism would require separate OS processes,
   out of scope for this instrument. Wall-time consequence disclosed in
   the module docstring's costing line (prereg §8: ~3 h wall for F-phi
   local-sequential vs. the registered "2-wide" ~3 h estimate -- i.e. this
   implementation's sequential wall time is expected to run roughly 2x the
   prereg's 2-wide estimate, a cost overrun disclosed here, not hidden).

Usage:
    uv run python results/campaign51_20260728/realistic_20260729/p3_twin_test.py \
        --stage p
    uv run python .../p3_twin_test.py --stage pilot
    uv run python .../p3_twin_test.py --stage fleet
    uv run python .../p3_twin_test.py --stage score
"""

import argparse
import contextlib
import json
import logging
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prod2d_closure_20260818"))
import decompose_impostor_leg as o2  # noqa: E402

from darksiren_emri.validation import correspondence_1d as c1d  # noqa: E402
from darksiren_emri.validation.correspondence_1d import (  # noqa: E402
    H_GRID_41,
    H_GRID_FULL,
    H_TRUE,
    combine_log_likelihood,
    compute_seed_statistics,
    moment_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
BANKED_JSON_DIR = REPO_ROOT / "results/prod2d_closure_20260818/correspondence_arms"
BANKED_CSV_ROOT = REPO_ROOT / "results/prod2d_closure_20260818/arm_event_likelihoods"

REGISTRATION_SECTION: str = (
    "results/campaign51_20260728/realistic_20260729/"
    "PREREGISTRATION_P3_TWIN_20260822.md (AMENDMENT 1, 2026-08-22)"
)

BSEL_SEEDS: tuple[int, ...] = tuple(range(900101, 900113))
PILOT_SEEDS: tuple[int, ...] = (900101, 900102)
P_SEED: int = 900101

H_GEN: float = 0.73
H_LO: float = 0.725
H_HI: float = 0.735

# GATE R-P3 (prereg §4): bit-exact, or <=1e-12 relative under the registered
# multiprocessing-float-order fallback (same convention as GATE R4/D6).
GATE_RP3_COLUMNS: tuple[str, ...] = ("L_cat_no_bh", "B_num", "combined_no_bh")
GATE_RP3_RTOL: float = 1.0e-12
GATE_RP3_MIN_WALL_S: float = 60.0

# GATE E-P3 (AMENDMENT 1's amended form).
GATE_EP3_MOVE_RTOL: float = 1.0e-6
GATE_EP3_MIN_FRACTION: float = 0.10
BATCH_ENGAGEMENT_LOG_SUBSTRING: str = "ENGAGED in the batch host path"

# GATE L-P3.
COUNTERFACTUAL_LOG_SUBSTRING: str = "COUNTERFACTUAL: catalogue_numerator_survival='phi'"


@contextlib.contextmanager
def _capture_root_log(log_path: Path) -> Iterator[None]:
    """Attach a ``logging.FileHandler`` to the ROOT logger (o6_fused_seed_test.py
    hard constraint 3 precedent: the bare root logger, not a
    ``"darksiren_emri"``-named logger, is the only placement under which the
    engagement/counterfactual log-line gates can pass -- see that module's
    docstring for the propagation-direction argument, reused verbatim here).
    """
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()
        root.setLevel(old_level)


def _banked_csv_path(seed: int) -> Path:
    return (
        BANKED_CSV_ROOT
        / f"bsel_seed{seed}"
        / f"seed{seed}"
        / "simulations/diagnostics/event_likelihoods.csv"
    )


def _banked_json(seed: int) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((BANKED_JSON_DIR / f"bsel_seed{seed}.json").read_text())
    return loaded


def _run_bsel_seed(
    seed: int,
    catalogue_numerator_survival: str,
    out_root: Path,
    subdir: str,
) -> dict[str, Any]:
    """Regenerate one B-SEL realization AND evaluate it end-to-end -- the exact
    call pattern ``run_arm_seed``'s ``bsel``/``bself`` branch uses
    (``correspondence_1d.py:2739-2765``), with ONLY
    ``catalogue_numerator_survival`` added on top (per the launch task's "P3
    arms must replicate it exactly, adding only catalogue_numerator_survival"
    instruction) -- fresh work root, root-logger capture, anti-idempotent-skip
    refusal (o6 precedent).
    """
    meta_path = out_root / f"{subdir}_meta.json"
    if meta_path.is_file():
        sys.exit(
            f"REFUSED: {meta_path} already exists -- use a fresh --out-root "
            "or remove this file if a genuine regeneration is intended (A21: "
            "the registered arm may not silently substitute a cached run)."
        )
    work_root = out_root / f"{subdir}_work"
    work_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / f"{subdir}.log"

    sigma_z_scale, area_scale = c1d.ARM_SPECS["bsel"]
    catalogue_pin_ok = c1d.check_reduced_catalogue_pin()
    cfg = c1d.CorrespondenceConfig(sigma_z_scale=sigma_z_scale, area_scale=area_scale)
    gen = c1d.MirrorUniverseGenerator(cfg)
    host_pool, _observed_path, handler = gen.host_pool_for_sigma_scale(
        work_root / "catalogue", seed, sigma_z_scale=sigma_z_scale
    )
    completeness_obj, phi_survival_table = c1d.build_bsel_selection_objects()
    events = gen.draw_realization(
        seed,
        host_pool=host_pool,
        host_mode="population_selected",
        completeness=completeness_obj,
        phi_survival_table=phi_survival_table,
    )

    t0 = time.time()
    with _capture_root_log(log_path):
        diag_csv, elapsed = c1d.run_mirror_seed_inprocess(
            work_root / f"seed{seed}",
            events,
            seed,
            galaxy_catalog=handler,
            # GATE R-P3 diagnosis fix (prereg AMENDMENT 2): the canonical bsel
            # branch evaluates over H_GRID_FULL; H_GRID_41 narrowed the h-prior
            # lower limit (0.6 vs 0.5) and with it the candidate z-window,
            # dropping low-z candidates (18 events fully, 110 partially).
            h_values=H_GRID_FULL,
            selection_in_completion_numerator=c1d.ARM_SELECTION_CELL["bsel"],
            completion_event_measure=c1d.ARM_EVENT_MEASURE["bsel"],
            catalogue_numerator_survival=catalogue_numerator_survival,
        )
    wall_time_s = time.time() - t0
    stats = compute_seed_statistics(diag_csv, seed, h_grid=H_GRID_41)

    meta: dict[str, Any] = {
        "subdir": subdir,
        "seed": seed,
        "catalogue_numerator_survival": catalogue_numerator_survival,
        "work_root": str(work_root),
        "diagnostics_csv": str(diag_csv),
        "log_path": str(log_path),
        "wall_time_s": wall_time_s,
        "elapsed_evaluate_s": elapsed,
        "catalogue_pin_ok": catalogue_pin_ok,
        "mean_h": stats.mean_h,
        "map_h": stats.map_h,
        "sigma_h": stats.sigma_h,
        "r_low": stats.r_low,
        "n_events": stats.n_events,
        "git_commit": c1d._git_commit(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "diagnostics_csv"}, indent=2))
    return meta


def _compare_columns(
    fresh: pd.DataFrame, banked_csv: Path, columns: tuple[str, ...], rtol: float
) -> dict[str, Any]:
    """Bit-exact-or-<=rtol-relative multi-column comparison (o6 GATE D6/o4
    GATE R4 pattern, generalized to :data:`GATE_RP3_COLUMNS`).
    """
    if not banked_csv.is_file():
        return {"pass": False, "reason": f"banked diagnostics not found: {banked_csv}"}
    banked = pd.read_csv(banked_csv)
    merged = fresh.merge(
        banked[["event_idx", "h", *columns]],
        on=["event_idx", "h"],
        suffixes=("_fresh", "_banked"),
        how="outer",
        indicator=True,
    )
    key_mismatch = bool((merged["_merge"] != "both").any())
    per_column: dict[str, Any] = {}
    all_ok = not key_mismatch
    for col in columns:
        a = merged[f"{col}_fresh"].to_numpy(dtype=np.float64)
        b = merged[f"{col}_banked"].to_numpy(dtype=np.float64)
        exact = bool(np.array_equal(a, b))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(a - b) / np.maximum(np.abs(b), np.finfo(float).tiny)
        max_rel = float(np.nanmax(rel)) if rel.size else float("nan")
        ok = exact or max_rel <= rtol
        all_ok = all_ok and ok
        per_column[col] = {"bit_exact": exact, "max_rel_err": max_rel, "pass": ok}
    return {
        "pass": all_ok,
        "key_mismatch": key_mismatch,
        "tol": rtol,
        "n_rows_compared": int(len(merged)),
        "per_column": per_column,
        "banked_csv": str(banked_csv),
        "fallback_justification": (
            "production evaluate() dispatches per-host likelihood terms "
            "through a multiprocessing pool (_starmap_host_batches, "
            "bayesian_statistics.py:4692-4721) whose float-summation order "
            f"is not guaranteed run-to-run identical, so the registered "
            f"<= {rtol:.0e} relative fallback is used per GATE R-P3."
        ),
    }


def stage_p(out_root: Path) -> dict[str, Any]:
    """GATE R-P3 arm: off replica, fresh work root."""
    meta = _run_bsel_seed(P_SEED, "off", out_root, "p_900101")
    fresh = pd.read_csv(meta["diagnostics_csv"])
    gate = _compare_columns(fresh, _banked_csv_path(P_SEED), GATE_RP3_COLUMNS, GATE_RP3_RTOL)
    gate["gate"] = "GATE_R-P3"
    gate["wall_time_s"] = meta["wall_time_s"]
    gate["wall_time_min_s"] = GATE_RP3_MIN_WALL_S
    gate["wall_time_pass"] = meta["wall_time_s"] > GATE_RP3_MIN_WALL_S
    gate["pass"] = bool(gate["pass"]) and gate["wall_time_pass"]
    gate["reference"] = f"{REGISTRATION_SECTION}, §4 GATE R-P3"
    (out_root / "p_gate_result.json").write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2))
    return gate


def stage_pilot(out_root: Path) -> dict[str, Any]:
    """PILOT arm: 900101/900102 under "phi"; paired Delta_s + realized scatter."""
    rows: list[dict[str, Any]] = []
    for seed in PILOT_SEEDS:
        meta = _run_bsel_seed(seed, "phi", out_root, f"phi_{seed}")
        banked = _banked_json(seed)
        delta_s = float(meta["mean_h"]) - float(banked["mean_h"])
        rows.append(
            {
                "seed": seed,
                "mean_h_phi": meta["mean_h"],
                "mean_h_banked": banked["mean_h"],
                "delta_s": delta_s,
            }
        )
    deltas = np.array([r["delta_s"] for r in rows], dtype=np.float64)
    scatter = float(deltas.std(ddof=1)) if deltas.size > 1 else None
    out = {
        "registered_in": f"{REGISTRATION_SECTION}, §3 PILOT",
        "rows": rows,
        "mean_delta_s": float(deltas.mean()),
        "realized_paired_scatter_proxy": scatter,
        "note": (
            "n=2: this scatter is a costing/band-freezing INPUT (prereg §5's "
            "'realized paired sigma-hat' precedent), not itself a band -- "
            "the band NUMBERS are frozen by the orchestrator from this "
            "output, per prereg §5"
        ),
    }
    (out_root / "pilot_output.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def stage_fleet(out_root: Path, seeds: list[int] | None = None) -> dict[str, Any]:
    """F-phi arm: the 12 banked seeds under "phi" -- seeds already produced by
    ``pilot`` (same ``phi_<seed>_meta.json`` naming) are REUSED, not re-run.
    ``seeds`` (from ``--seeds``) restricts this invocation to a subset so two
    driver processes can split the fleet 2-wide across disjoint halves
    (disclosed operational split; the score stage always reads all 12).
    """
    reused: list[int] = []
    ran: list[int] = []
    for seed in seeds if seeds is not None else BSEL_SEEDS:
        meta_path = out_root / f"phi_{seed}_meta.json"
        if meta_path.is_file():
            reused.append(seed)
            print(f"seed {seed}: REUSING existing phi_{seed}_meta.json (disclosed)")
            continue
        _run_bsel_seed(seed, "phi", out_root, f"phi_{seed}")
        ran.append(seed)
    summary = {"reused_from_pilot": reused, "freshly_ran": ran}
    print(json.dumps(summary, indent=2))
    return summary


def stage_kflat(out_root: Path) -> dict[str, Any]:
    r"""K-flat kill test (prereg \u00a73): seed 900101 under ``"phi_flat"``.

    The originally-flagged blocker (no table-override hook without flattening
    the normalizer legs) was resolved by the orchestrator ON THE BRANCH with a
    third flag value ``"phi_flat"`` (bayesian_statistics.py evaluate(): the
    catalogue consumer's PER-CALL table slice is replaced by the real table's
    grid-mean constant; the normalizer legs keep the real table object, so the
    prereg \u00a76 invariant "the phi-survival table construction" holds). The
    audit trail of the rejected monkeypatch route is preserved in git history.
    """
    meta = _run_bsel_seed(900101, "phi_flat", out_root, "kflat_900101")
    banked = _banked_json(900101)
    delta = float(meta["mean_h"]) - float(banked["mean_h"])
    out = {
        "stage": "kflat",
        "seed": 900101,
        "mean_h_phi_flat": meta["mean_h"],
        "mean_h_banked": banked["mean_h"],
        "delta_vs_banked": delta,
        "reference": (
            "PREREGISTRATION_P3_TWIN_20260822.md \u00a73 K-flat + \u00a75 secondary 5: "
            "subtracts the banked coded mean_h; expected within the pilot noise "
            "floor if the z-slope is the mechanism"
        ),
    }
    (out_root / "kflat_result.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def _catalogue_bearing_mask(banked_csv: Path, h_gen: float) -> npt.NDArray[np.bool_]:
    df = pd.read_csv(banked_csv)
    at = df[np.isclose(df["h"].to_numpy(dtype=np.float64), h_gen)].sort_values("event_idx")
    return np.asarray(at["L_cat_no_bh"].to_numpy(dtype=np.float64) > 0.0, dtype=np.bool_)


def _column_at_h(csv_path: Path, column: str, h_gen: float) -> npt.NDArray[np.float64]:
    df = pd.read_csv(csv_path)
    at = df[np.isclose(df["h"].to_numpy(dtype=np.float64), h_gen)].sort_values("event_idx")
    return np.asarray(at[column].to_numpy(dtype=np.float64), dtype=np.float64)


def _gate_e_p3(phi_metas: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """GATE E-P3, AMENDED form (AMENDMENT 1, replacing prereg §4's unsatisfiable
    original -- see the module PREREGISTRATION's Amendment text).

    (a) engagement magnitude: >=10% of catalogue-bearing events' L_cat_no_bh
        move >=1e-6 relative vs banked, at h = H_GEN. **FLAGGED AMBIGUITY
        (A21):** the amendment text does not say whether this fraction is
        per-seed or fleet-pooled; this gate applies it FLEET-POOLED (all
        catalogue-bearing events across all 12 seeds combined) and separately
        reports the per-seed fractions for the orchestrator's review, rather
        than silently picking the per-seed reading.
    (b) batch-path engagement log line present in every "phi" run's log.
    (c)/(d) code-audit disclosures (zero-compute, static; see the audit
        strings below) -- NOT independently re-derived from run data, exactly
        as the amendment scopes them ("explicitly in scope for the A20
        review", not a runtime check this scorer can perform).
    """
    moved_flags: list[npt.NDArray[np.bool_]] = []
    per_seed_fraction: dict[int, float] = {}
    log_b_pass: dict[int, bool] = {}
    for seed, meta in phi_metas.items():
        banked_csv = _banked_csv_path(seed)
        mask = _catalogue_bearing_mask(banked_csv, H_GEN)
        banked_lcat = _column_at_h(banked_csv, "L_cat_no_bh", H_GEN)
        phi_lcat = _column_at_h(Path(meta["diagnostics_csv"]), "L_cat_no_bh", H_GEN)
        n = min(mask.size, banked_lcat.size, phi_lcat.size)
        mask, banked_lcat, phi_lcat = mask[:n], banked_lcat[:n], phi_lcat[:n]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(phi_lcat - banked_lcat) / np.maximum(
                np.abs(banked_lcat), np.finfo(float).tiny
            )
        moved = (rel >= GATE_EP3_MOVE_RTOL) & mask
        moved_flags.append(moved[mask])
        per_seed_fraction[seed] = float(moved[mask].mean()) if mask.any() else 0.0

        log_text = Path(meta["log_path"]).read_text()
        log_b_pass[seed] = BATCH_ENGAGEMENT_LOG_SUBSTRING in log_text

    pooled = np.concatenate(moved_flags) if moved_flags else np.array([], dtype=np.bool_)
    pooled_fraction = float(pooled.mean()) if pooled.size else 0.0
    a_pass = pooled_fraction >= GATE_EP3_MIN_FRACTION
    b_pass = all(log_b_pass.values())

    audit_c = (
        "the scalar twin (single_host_likelihood's catalogue-leg branch, "
        "bayesian_statistics.py:5765-5769) applies the SAME expression on "
        "the SAME table input as the batch path (:6621-6625) -- verified by "
        "source read (grep-confirmed, not runtime-measured: production has "
        "no runtime call site of the scalar path, per AMENDMENT 1's own "
        "finding)"
    )
    audit_d = (
        "both _starmap_host_batches call sites (bayesian_statistics.py:"
        "4692-4721) receive _cat_surv/_cat_surv_table -- the with-BH host "
        "batch's r[0] no-BH numerator ALSO feeds L_cat_no_bh via the "
        "caller's all_results_without_bh concatenation, per AMENDMENT 1(d)"
    )
    return {
        "gate": "GATE_E-P3 (AMENDED, AMENDMENT 1)",
        "pass": a_pass and b_pass,
        "a_engagement_magnitude": {
            "pooled_fraction_moved": pooled_fraction,
            "per_seed_fraction_moved": per_seed_fraction,
            "min_fraction": GATE_EP3_MIN_FRACTION,
            "move_rtol": GATE_EP3_MOVE_RTOL,
            "pass": a_pass,
            "ambiguity_flag": (
                "A21: fleet-pooled convention chosen, per-seed fractions "
                "also reported -- see docstring"
            ),
        },
        "b_batch_log_present": {"per_seed": log_b_pass, "pass": b_pass},
        "c_scalar_code_audit": {"statement": audit_c, "pass": True},
        "d_both_batches_code_audit": {"statement": audit_d, "pass": True},
        "reference": f"{REGISTRATION_SECTION}, AMENDMENT 1 amended GATE E-P3",
    }


def _gate_l_p3(p_meta: dict[str, Any], phi_metas: dict[int, dict[str, Any]]) -> dict[str, Any]:
    p_log = Path(p_meta["log_path"]).read_text()
    p_absent = COUNTERFACTUAL_LOG_SUBSTRING not in p_log
    phi_present = {
        seed: COUNTERFACTUAL_LOG_SUBSTRING in Path(m["log_path"]).read_text()
        for seed, m in phi_metas.items()
    }
    ok = p_absent and all(phi_present.values())
    return {
        "gate": "GATE_L-P3",
        "pass": ok,
        "p_log_counterfactual_absent": p_absent,
        "phi_logs_counterfactual_present": phi_present,
        "reference": f"{REGISTRATION_SECTION}, §4 GATE L-P3",
    }


def _sec1_impostor_decomposition(phi_metas: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Secondary 1: O2-style impostor decomposition on the F-phi diagnostics
    (``decompose_impostor_leg.py``'s ``load_matrices``/``moments``, reused by
    import verbatim -- the same beta_G_phi assembly identity, just applied to
    phi-cell CSVs instead of off-cell ones).
    """
    per_seed: list[dict[str, Any]] = []
    for seed, meta in sorted(phi_metas.items()):
        df = pd.read_csv(meta["diagnostics_csv"])
        full_vals, pure_vals, gate_i, n_events = o2.load_matrices(df)
        m_full = o2.moments(full_vals)
        m_pure = o2.moments(pure_vals)
        per_seed.append(
            {
                "seed": seed,
                "gate_i_identity_max_rel": gate_i,
                "n_events": n_events,
                "full_mean_h": m_full["mean_h"],
                "pure_mean_h": m_pure["mean_h"],
                "delta_mean_h": (
                    None
                    if m_full["mean_h"] is None or m_pure["mean_h"] is None
                    else m_pure["mean_h"] - m_full["mean_h"]
                ),
            }
        )
    full_means = np.array(
        [r["full_mean_h"] for r in per_seed if r["full_mean_h"] is not None], dtype=np.float64
    )
    pure_means = np.array(
        [r["pure_mean_h"] for r in per_seed if r["pure_mean_h"] is not None], dtype=np.float64
    )
    bias_full = float(full_means.mean() - H_TRUE) if full_means.size else None
    bias_pure = float(pure_means.mean() - H_TRUE) if pure_means.size else None
    delta_bias = None if bias_full is None or bias_pure is None else bias_pure - bias_full
    return {
        "reference": (
            "results/prod2d_closure_20260818/decompose_impostor_leg.py "
            "load_matrices()/moments(), applied to the phi-cell CSVs "
            "(REPORTED-ONLY, prereg §5 secondary 1)"
        ),
        "per_seed": per_seed,
        "bias_full_phi": bias_full,
        "bias_pure_phi": bias_pure,
        "delta_bias_phi": delta_bias,
    }


def _sec2_paired_event_delta_ln(
    phi_metas: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Secondary 2: per-event paired Delta ln(combined_no_bh) at H_GEN, phi vs
    banked, pooled across seeds/events (distribution, not just the mean --
    prereg §5 secondary 2 / [A2]).
    """
    all_delta: list[float] = []
    per_seed_mean: dict[int, float] = {}
    for seed, meta in sorted(phi_metas.items()):
        banked = _column_at_h(_banked_csv_path(seed), "combined_no_bh", H_GEN)
        phi = _column_at_h(Path(meta["diagnostics_csv"]), "combined_no_bh", H_GEN)
        n = min(banked.size, phi.size)
        banked, phi = banked[:n], phi[:n]
        ok = (banked > 0.0) & (phi > 0.0)
        if not ok.any():
            continue
        delta_ln = np.log(phi[ok]) - np.log(banked[ok])
        all_delta.extend(delta_ln.tolist())
        per_seed_mean[seed] = float(delta_ln.mean())
    arr = np.array(all_delta, dtype=np.float64)
    if arr.size == 0:
        return {"reference": f"{REGISTRATION_SECTION}, §5 secondary 2", "n": 0}
    return {
        "reference": f"{REGISTRATION_SECTION}, §5 secondary 2",
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "per_seed_mean": per_seed_mean,
    }


def _sec3_score_at_truth(phi_metas: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Secondary 3: score-at-truth (A12), central difference at 0.725/0.735,
    "full" (all events) and "matched" (banked catalogue-bearing events only)
    channels, on the phi-cell ``combined_no_bh`` column
    (``decompose_impostor_leg.score_at_truth``, reused verbatim).
    """
    per_seed: list[dict[str, Any]] = []
    for seed, meta in sorted(phi_metas.items()):
        df = pd.read_csv(meta["diagnostics_csv"])
        full_vals, _pure_vals, _gi, _n = o2.load_matrices(df)
        mask = _catalogue_bearing_mask(_banked_csv_path(seed), H_GEN)
        n = min(mask.size, full_vals.shape[0])
        full_score = o2.score_at_truth(full_vals[:n])
        matched_score = o2.score_at_truth(full_vals[:n][mask[:n]])
        per_seed.append({"seed": seed, "full": full_score, "matched": matched_score})
    return {
        "reference": (
            "decompose_impostor_leg.score_at_truth(), full (all events) vs "
            "matched (banked L_cat_no_bh>0 mask) channels, prereg §5 "
            "secondary 3"
        ),
        "per_seed": per_seed,
    }


def _sec4_rail_read(phi_metas: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Secondary 4: rail read -- r_low count and floor-node ("floor" = the
    lowest H_GRID_41 node, 0.6) posterior mass, phi vs banked. Expected NULL
    per prereg §5 secondary 4 ("the rail is photo-z territory").
    """
    grid = np.array(H_GRID_41, dtype=np.float64)
    weights = moment_weights(grid, "trapezoid")

    def floor_mass(vals: npt.NDArray[np.float64]) -> float | None:
        sum_log_l = combine_log_likelihood(vals, "physics_floor")
        if not np.isfinite(sum_log_l).any():
            return None
        lp = sum_log_l - sum_log_l.max()
        post = np.exp(lp)
        norm = float((post * weights).sum())
        post_n = post / norm if norm > 0 else post
        return float(post_n[0] * weights[0])

    rows: list[dict[str, Any]] = []
    for seed, meta in sorted(phi_metas.items()):
        banked = _banked_json(seed)
        phi_df = pd.read_csv(meta["diagnostics_csv"])
        phi_full_vals, _pure, _gi, _n = o2.load_matrices(phi_df)
        rows.append(
            {
                "seed": seed,
                "r_low_banked": banked["r_low"],
                "r_low_phi": meta["r_low"],
                "floor_node_mass_phi": floor_mass(phi_full_vals),
            }
        )
    n_r_low_banked = sum(1 for r in rows if r["r_low_banked"])
    n_r_low_phi = sum(1 for r in rows if r["r_low_phi"])
    return {
        "reference": f"{REGISTRATION_SECTION}, §5 secondary 4",
        "expected": "NULL (photo-z territory, not the twin cell)",
        "n_r_low_banked": n_r_low_banked,
        "n_r_low_phi": n_r_low_phi,
        "per_seed": rows,
    }


def stage_score(out_root: Path) -> dict[str, Any]:
    p_meta_path = out_root / "p_900101_meta.json"
    p_gate_path = out_root / "p_gate_result.json"
    if not p_meta_path.is_file() or not p_gate_path.is_file():
        sys.exit(f"REFUSED: run --stage p first (missing {p_meta_path} or {p_gate_path}).")
    p_meta = json.loads(p_meta_path.read_text())
    gate_rp3 = json.loads(p_gate_path.read_text())

    phi_metas: dict[int, dict[str, Any]] = {}
    missing: list[int] = []
    for seed in BSEL_SEEDS:
        meta_path = out_root / f"phi_{seed}_meta.json"
        if not meta_path.is_file():
            missing.append(seed)
            continue
        phi_metas[seed] = json.loads(meta_path.read_text())
    if missing:
        sys.exit(
            f"REFUSED: missing phi run(s) for seeds {missing} -- run --stage "
            "fleet (after --stage pilot) first."
        )

    gate_e = _gate_e_p3(phi_metas)
    gate_l = _gate_l_p3(p_meta, phi_metas)
    gates = {"GATE_R-P3": gate_rp3, "GATE_E-P3": gate_e, "GATE_L-P3": gate_l}
    all_gates_pass = all(bool(g.get("pass")) for g in gates.values())

    per_seed_delta: list[dict[str, Any]] = []
    for seed in BSEL_SEEDS:
        banked = _banked_json(seed)
        phi = phi_metas[seed]
        delta_s = float(phi["mean_h"]) - float(banked["mean_h"])
        per_seed_delta.append(
            {
                "seed": seed,
                "mean_h_phi": phi["mean_h"],
                "mean_h_banked": banked["mean_h"],
                "delta_s": delta_s,
            }
        )
    deltas = np.array([r["delta_s"] for r in per_seed_delta], dtype=np.float64)
    delta_bar = float(deltas.mean())
    sem_paired = float(deltas.std(ddof=1) / np.sqrt(deltas.size)) if deltas.size > 1 else None

    secondaries = {
        "1_impostor_decomposition": _sec1_impostor_decomposition(phi_metas),
        "2_paired_event_delta_ln": _sec2_paired_event_delta_ln(phi_metas),
        "3_score_at_truth": _sec3_score_at_truth(phi_metas),
        "4_rail_read": _sec4_rail_read(phi_metas),
    }

    output: dict[str, Any] = {
        "registered_in": REGISTRATION_SECTION,
        "seeds": list(BSEL_SEEDS),
        "h_gen": H_GEN,
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "primary": {
            "statistic": "Delta_bar(12) = mean_s[mean_h(F-phi,s) - mean_h(coded banked,s)]",
            "value": delta_bar if all_gates_pass else None,
            "sem_paired": sem_paired if all_gates_pass else None,
            "per_seed": per_seed_delta,
            "reference": (
                f"{REGISTRATION_SECTION}, §5 'Primary' -- paired per-seed "
                "delta, fleet mean with paired SEM (the C-SG BAND R lesson)"
            ),
            "band_note": (
                "bands (TWIN-MATERIAL/TWIN-IMMATERIAL/REPORT-BOUND) are "
                "applied by the orchestrator against the pilot-frozen "
                "numbers, per prereg §5 -- this scorer prints Delta_bar/SEM "
                "only, no band is computed here"
            ),
        },
        "secondaries_reported_only": secondaries,
        "verdict": "GATES-FAILED -- primary/secondaries MAY NOT BE READ"
        if not all_gates_pass
        else "GATES-PASS",
    }

    out_path = THIS_DIR / "p3_twin_test_output.json"
    out_path.write_text(json.dumps(output, indent=2))

    print("=== [P3-IMP] twin test score ===")
    for name, g in gates.items():
        print(f"  {name}: pass={g.get('pass')}")
    print(f"all_gates_pass = {all_gates_pass}")
    print(f"Delta_bar(12) = {delta_bar!r}  SEM_paired = {sem_paired!r}")
    print(f"verdict = {output['verdict']}")
    print(f"wrote {out_path}")
    return output


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("p", "pilot", "fleet", "kflat", "score"), required=True)
    ap.add_argument(
        "--seeds", type=str, default=None, help="fleet only: comma-separated seed subset"
    )
    ap.add_argument(
        "--out-root",
        type=str,
        default=str(THIS_DIR / "p3_work"),
        help="Root scratch/output directory for fresh work roots/logs/metadata.",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=2,
        help=(
            "Accepted for CLI-shape compatibility only -- this implementation "
            "runs seeds sequentially regardless (see module docstring, hard "
            "constraint 3)."
        ),
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.stage == "p":
        stage_p(out_root)
        return 0
    if args.stage == "pilot":
        stage_pilot(out_root)
        return 0
    if args.stage == "fleet":
        stage_fleet(out_root, [int(x) for x in args.seeds.split(",")] if args.seeds else None)
        return 0
    if args.stage == "kflat":
        stage_kflat(out_root)
        return 0
    result = stage_score(out_root)
    return 0 if result["verdict"] == "GATES-PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
