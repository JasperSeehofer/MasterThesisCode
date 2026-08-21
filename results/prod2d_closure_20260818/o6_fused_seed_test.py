r"""CONFIRMATION RUN O6 driver/scorer
(``PREREGISTRATION_SELFGEN_CONTROL.md``, "CONFIRMATION RUN O6 -- REGISTRATION",
2026-08-21, ledger row #157 item 2; A21 -- the executed configuration may not
deviate from the registration's arm table).

Question: does the REAL ``fused`` cell -- ``selection_in_completion_numerator
="fused"``, executed end-to-end inside ``BayesianStatistics.evaluate()``
(both legs, not a numerator patch) -- null the matched-channel violation for
C-SG-F seed 910101, as the identified mechanism (off-cell ``S_bar_phi``
omission, O4/A20) predicts?

**Arms (exact text, A21):**

- **D6 (off replica, gate)** -- ``run_csg_arm_seed(work_root=FRESH_DIR,
  "csgf", 910101, ...)`` with ``selection_in_completion_numerator=None``
  (pinned production ``"off"``), fresh work root.
- **F6 (fused, primary)** -- same call with
  ``selection_in_completion_numerator="fused"``, fresh work root.

**Stages** (``--stage {d6,f6,score,all}``, default ``all``):

- ``d6``: regenerate the off replica; GATE D6 (bit-exact/<=1e-12 relative
  B_num vs the banked ``csgf_seed910101`` diagnostics, wall time > 60 s).
- ``f6``: regenerate the fused cell (same shape, different flag).
- ``score``: reads both records + logs + ``o6_reference_derivation_output.json``
  and applies GATE L6 (cell-identity, zero-compute), GATE T6 (normalizer
  invariance), GATE V6 (anti-vacuity), then the primary statistic
  ``S(F6) = f6_record["score_at_h_gen"]["matched"]["mean_score"]``, identity
  delta ``S(F6) - r_prod(910101)``, and the registered bands.

The only code change O6 authorizes is the ``selection_in_completion_numerator``
passthrough on :func:`~darksiren_emri.validation.selfgen_control.run_csg_arm_seed`
-- already present in-tree (``selfgen_control.py:1367-1478``); this script
makes NO further production edits, calling that function verbatim.

**HARD CONSTRAINTS (launch task, mirrors o4_pairing_test.py):**

1. Never end a turn to wait on an untracked process -- every call below is a
   synchronous, blocking ``run_csg_arm_seed`` invocation.
2. Every load-bearing claim cites file:line where practical.
3. GATE L6's log-content checks require capturing the log lines
   ``bayesian_statistics.py`` emits at ``_LOGGER = logging.getLogger()``
   (bayesian_statistics.py:73) -- the BARE ROOT LOGGER, not a
   ``"darksiren_emri"``-named logger. A handler attached only to the
   ``"darksiren_emri"`` logger would NOT see those records (propagation
   flows child -> parent, never parent -> child, and the root logger is an
   ancestor of, not a descendant of, ``"darksiren_emri"``). **FLAGGED SPEC
   AMBIGUITY (A21 -- disclosed, not silently resolved):** the launch task
   said to attach the ``logging.FileHandler`` "capturing the darksiren_emri
   logger tree"; this script instead attaches it to the root logger
   (``logging.getLogger()``), which is a strict superset (it also captures
   everything the ``"darksiren_emri"`` tree would, since that tree
   propagates upward to root by default) and is the only placement under
   which GATE L6 can pass at all for a genuine fused run. Recorded here
   rather than silently substituted.

Usage:
    uv run python results/prod2d_closure_20260818/o6_fused_seed_test.py --stage all
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import o4_pairing_test as o4  # noqa: E402

from darksiren_emri.validation import selfgen_control as csg  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results/prod2d_closure_20260818"

ARM: str = "csgf"
SEED: int = 910101
H_GEN: float = 0.73
H_LO: float = 0.725
H_HI: float = 0.735
N_EVENTS: int = 200

REGISTRATION_SECTION: str = (
    "results/prod2d_closure_20260818/PREREGISTRATION_SELFGEN_CONTROL.md, "
    "CONFIRMATION RUN O6 -- REGISTRATION (2026-08-21, ledger row #157 item 2)"
)

# GATE D6 (registration "Validity gates"): bit-exact, or <=1e-12 relative
# under the registered multiprocessing-float-order fallback (same fallback
# GATE R4 uses, o4_pairing_test.py GATE_R4_RTOL).
GATE_D6_RTOL: float = 1.0e-12
GATE_D6_MIN_WALL_S: float = 60.0

# GATE T6 (registration): 2e-6 relative -- the same GATE-T-family tolerance
# as GATE T4 (o4_pairing_test.py GATE_T4_TOL, decompose_matched_channel.py:87
# GATE_T_TOL).
GATE_T6_TOL: float = 2.0e-6

# GATE V6 (registration): F6's B_num column must differ from D6's on > 99%
# of merged rows.
GATE_V6_MIN_DIFFER_FRACTION: float = 0.99
# Floating-point noise floor for "differs" (below GATE R4/D6's own 1e-12
# fallback, so genuine floating-point-only agreement is not miscounted as a
# "difference").
GATE_V6_DIFFER_RTOL: float = 1.0e-9

# Primary band (registration "Bands" table).
PRIMARY_BAND_TOL: float = 1.0e-4

# Secondary, REPORTED-ONLY (registration: "explicitly NOT a band") -- the
# frozen SELF-CONSISTENT edge quoted verbatim in the O6 registration text
# (PREREGISTRATION_SELFGEN_CONTROL.md line ~669/719: "0.037339"), distinct
# from the rounded 0.0373 used elsewhere in the prereg.
FROZEN_SELF_CONSISTENT_EDGE: float = 0.037339

OFF_CELL_LOG_SUBSTRING: str = "selection_in_completion_numerator='off'"
FUSED_CELL_LOG_SUBSTRING: str = "selection fusion ACTIVE"


@contextlib.contextmanager
def _capture_root_log(log_path: Path) -> Iterator[None]:
    """Attach a ``logging.FileHandler`` to the ROOT logger at INFO level for
    the duration of the ``with`` block (see module docstring, hard
    constraint 3, for why the root logger rather than the
    ``"darksiren_emri"`` named logger). Removed and closed on exit so a
    second call (e.g. the F6 stage right after D6 in ``--stage all``) does
    not leak the previous stage's lines into the new log file.
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


def _refuse_if_idempotent_skip(out_dir: Path) -> None:
    """GATE D6/F6 anti-idempotent-skip guard (registration: "REFUSE to run if
    the out dir already contains csgf_seed910101.json" -- the O4/A20
    3/15-cached-artifact finding this run is designed to preclude).
    """
    existing = out_dir / f"{ARM}_seed{SEED}.json"
    if existing.is_file():
        sys.exit(
            f"REFUSED: {existing} already exists -- run_csg_arm_seed is "
            "idempotent (selfgen_control.py:1414-1421) and would return the "
            "cached artifact without regenerating (the O4/A20 3/15-cached "
            "finding). Use a fresh --out-root or remove this file if a "
            "genuine regeneration is intended (A21: the registered arm may "
            "not silently substitute a cached run)."
        )


def _compare_b_num(
    fresh: pd.DataFrame, banked_csv: Path, gate_name: str, rtol: float
) -> dict[str, Any]:
    """Bit-exact-or-<=rtol-relative B_num comparison, the same merge/compare
    logic as :func:`o4_pairing_test.run_gate_r4` (including its
    ``fallback_justification`` wording), generalized to an arbitrary gate
    name and banked CSV.
    """
    if not banked_csv.is_file():
        return {
            "gate": gate_name,
            "pass": False,
            "reason": f"banked diagnostics not found: {banked_csv}",
        }
    banked = pd.read_csv(banked_csv)
    merged = fresh.merge(
        banked[["event_idx", "h", "B_num"]],
        on=["event_idx", "h"],
        suffixes=("_fresh", "_banked"),
        how="outer",
        indicator=True,
    )
    key_mismatch = bool((merged["_merge"] != "both").any())
    b_fresh = merged["B_num_fresh"].to_numpy(dtype=np.float64)
    b_banked = merged["B_num_banked"].to_numpy(dtype=np.float64)
    exact = bool(np.array_equal(b_fresh, b_banked))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(b_fresh - b_banked) / np.maximum(np.abs(b_banked), np.finfo(float).tiny)
    max_rel = float(np.nanmax(rel)) if rel.size else float("nan")
    ok = (not key_mismatch) and (exact or max_rel <= rtol)
    return {
        "gate": gate_name,
        "pass": ok,
        "key_mismatch": key_mismatch,
        "bit_exact": exact,
        "max_rel_err": max_rel,
        "tol": rtol,
        "n_rows_compared": int(len(merged)),
        "fresh_diagnostics_csv_rows": int(len(fresh)),
        "banked_csv": str(banked_csv),
        "fallback_justification": (
            None
            if exact
            else (
                "not bit-exact (max_rel_err="
                f"{max_rel:.3e}); production evaluate() dispatches per-host "
                "likelihood terms through a multiprocessing pool "
                "(_starmap_host_batches, bayesian_statistics.py:4636-4662) "
                "whose float-summation order is not guaranteed run-to-run "
                f"identical, so the registered <= {rtol:.0e} relative "
                f"fallback is used per the O6 {gate_name} spec."
            )
        ),
    }


def _run_arm(
    out_root: Path,
    subdir: str,
    selection_in_completion_numerator: str | None,
) -> dict[str, Any]:
    """Run one O6 arm (D6 or F6): fresh work root/out dir, log capture, timed
    ``run_csg_arm_seed`` call, anti-idempotent-skip refusal.

    Returns a metadata dict (``record``, ``diagnostics_csv``, ``log_path``,
    ``wall_time_s``, ``selection_in_completion_numerator``).
    """
    work_root = out_root / f"{subdir}_work"
    out_dir = out_root / f"{subdir}_out"
    work_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    _refuse_if_idempotent_skip(out_dir)

    log_path = out_root / f"{subdir}.log"
    t0 = time.time()
    with _capture_root_log(log_path):
        record_path = csg.run_csg_arm_seed(
            work_root,
            ARM,
            SEED,
            out_dir,
            n_events=N_EVENTS,
            selection_in_completion_numerator=selection_in_completion_numerator,
        )
    wall_time_s = time.time() - t0
    record = json.loads(record_path.read_text())

    meta: dict[str, Any] = {
        "subdir": subdir,
        "work_root": str(work_root),
        "out_dir": str(out_dir),
        "record_path": str(record_path),
        "record": record,
        "diagnostics_csv": str(record["diagnostics_csv"]),
        "log_path": str(log_path),
        "wall_time_s": wall_time_s,
        "selection_in_completion_numerator": selection_in_completion_numerator,
    }
    (out_root / f"{subdir}_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def stage_d6(out_root: Path) -> dict[str, Any]:
    """GATE D6 arm: off replica, fresh work root. Bit-exact/<=1e-12 B_num vs
    banked, and wall time > 60 s (anti-idempotent-skip check).
    """
    meta = _run_arm(out_root, "d6", None)
    fresh = pd.read_csv(meta["diagnostics_csv"])
    banked_csv = o4.BANKED_DIAG_DIR / f"{ARM}_seed{SEED}" / "event_likelihoods.csv"
    gate = _compare_b_num(fresh, banked_csv, "GATE_D6", GATE_D6_RTOL)
    gate["wall_time_s"] = meta["wall_time_s"]
    gate["wall_time_min_s"] = GATE_D6_MIN_WALL_S
    gate["wall_time_pass"] = meta["wall_time_s"] > GATE_D6_MIN_WALL_S
    gate["pass"] = bool(gate["pass"]) and gate["wall_time_pass"]
    gate["reference"] = f"{REGISTRATION_SECTION}, GATE D6"
    gate_path = out_root / "d6_gate_result.json"
    gate_path.write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2))
    return gate


def stage_f6(out_root: Path) -> dict[str, Any]:
    """F6 arm: fused cell, fresh work root. No GATE D6 comparison here (F6's
    checks -- GATE T6, GATE V6, GATE L6 -- are applied in the score stage,
    once both arms' records/logs exist).
    """
    meta = _run_arm(out_root, "f6", "fused")
    print(json.dumps({k: v for k, v in meta.items() if k != "record"}, indent=2))
    return meta


def _gate_l6(d6_log_text: str, f6_log_text: str, f6_selection_kwarg: str | None) -> dict[str, Any]:
    """GATE L6 (cell identity, zero compute): D6's log carries the off-cell
    counterfactual line; F6's log carries the fused-cell line and does NOT
    carry the off-cell line; F6's run metadata records
    ``selection_in_completion_numerator="fused"``.

    ``run_csg_arm_seed``'s own JSON record (``selfgen_control.py:1468-1490``)
    does not itself carry a ``selection_in_completion_numerator`` key -- the
    kwarg this script's ``_run_arm`` passed (persisted in
    ``<subdir>_meta.json``) is the authoritative record of what was actually
    requested, so that value (not the record) is checked here.
    """
    d6_has_off_line = OFF_CELL_LOG_SUBSTRING in d6_log_text
    f6_has_fused_line = FUSED_CELL_LOG_SUBSTRING in f6_log_text
    f6_has_off_line = OFF_CELL_LOG_SUBSTRING in f6_log_text
    f6_flag_recorded = f6_selection_kwarg == "fused"
    ok = d6_has_off_line and f6_has_fused_line and (not f6_has_off_line) and f6_flag_recorded
    return {
        "gate": "GATE_L6",
        "pass": ok,
        "d6_log_has_off_cell_line": d6_has_off_line,
        "f6_log_has_fused_line": f6_has_fused_line,
        "f6_log_has_off_cell_line": f6_has_off_line,
        "f6_selection_flag_recorded_as_fused": f6_flag_recorded,
        "off_cell_substring": OFF_CELL_LOG_SUBSTRING,
        "fused_cell_substring": FUSED_CELL_LOG_SUBSTRING,
        "reference": f"{REGISTRATION_SECTION}, GATE L6",
    }


def _gate_t6(f6_diag: pd.DataFrame, banked_off_diag: pd.DataFrame) -> dict[str, Any]:
    """GATE T6 (normalizer invariance): F6's column-derived
    ``D_tilde_phi - alpha_G_phi`` at h in {H_LO, H_HI} equals the SAME
    quantity from the banked OFF-cell diagnostics to 2e-6 relative -- the
    cell switch must not move the normalizer leg (built unconditionally
    under ``absolute_marginal``, bayesian_statistics.py:3800-3821).
    """
    per_h: dict[str, Any] = {}
    ok_all = True
    for h in (H_LO, H_HI):
        f6_rows = f6_diag[(f6_diag["event_idx"] == 0) & np.isclose(f6_diag["h"], h)]
        banked_rows = banked_off_diag[
            (banked_off_diag["event_idx"] == 0) & np.isclose(banked_off_diag["h"], h)
        ]
        if f6_rows.empty or banked_rows.empty:
            per_h[str(h)] = {"pass": False, "reason": f"h={h} not present in one of the CSVs"}
            ok_all = False
            continue
        f6_val = float(f6_rows["D_tilde_phi"].iloc[0] - f6_rows["alpha_G_phi"].iloc[0])
        banked_val = float(banked_rows["D_tilde_phi"].iloc[0] - banked_rows["alpha_G_phi"].iloc[0])
        rel = abs(f6_val - banked_val) / max(abs(banked_val), float(np.finfo(float).tiny))
        ok = rel <= GATE_T6_TOL
        ok_all = ok_all and ok
        per_h[str(h)] = {
            "f6_D_tilde_minus_alpha_G": f6_val,
            "banked_off_D_tilde_minus_alpha_G": banked_val,
            "rel_err": rel,
            "tol": GATE_T6_TOL,
            "pass": ok,
        }
    return {
        "gate": "GATE_T6",
        "pass": ok_all,
        "per_h": per_h,
        "reference": f"{REGISTRATION_SECTION}, GATE T6",
    }


def _gate_v6(f6_diag: pd.DataFrame, d6_diag: pd.DataFrame) -> dict[str, Any]:
    """GATE V6 (anti-vacuity): F6's B_num column differs from D6's on > 99%
    of merged rows -- guards a silent fall-through to the off dispatch.
    """
    merged = f6_diag.merge(
        d6_diag[["event_idx", "h", "B_num"]],
        on=["event_idx", "h"],
        suffixes=("_f6", "_d6"),
        how="outer",
        indicator=True,
    )
    key_mismatch = bool((merged["_merge"] != "both").any())
    b_f6 = merged["B_num_f6"].to_numpy(dtype=np.float64)
    b_d6 = merged["B_num_d6"].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(b_f6 - b_d6) / np.maximum(np.abs(b_d6), np.finfo(float).tiny)
    differs = rel > GATE_V6_DIFFER_RTOL
    differ_fraction = float(np.mean(differs)) if differs.size else 0.0
    ok = (not key_mismatch) and differ_fraction > GATE_V6_MIN_DIFFER_FRACTION
    return {
        "gate": "GATE_V6",
        "pass": ok,
        "key_mismatch": key_mismatch,
        "differ_fraction": differ_fraction,
        "min_differ_fraction": GATE_V6_MIN_DIFFER_FRACTION,
        "differ_rtol": GATE_V6_DIFFER_RTOL,
        "n_rows_compared": int(len(merged)),
        "reference": f"{REGISTRATION_SECTION}, GATE V6",
    }


def stage_score(out_root: Path) -> dict[str, Any]:
    """Score stage: load both arms' records/logs + the registered reference,
    apply GATE L6/T6/V6, then read the primary statistic and identity delta
    and apply the registered bands.
    """
    d6_meta_path = out_root / "d6_meta.json"
    f6_meta_path = out_root / "f6_meta.json"
    d6_gate_path = out_root / "d6_gate_result.json"
    ref_path = RESULTS_DIR / "o6_reference_derivation_output.json"
    for p in (d6_meta_path, f6_meta_path, d6_gate_path, ref_path):
        if not p.is_file():
            sys.exit(
                f"REFUSED: required input missing: {p}. Run --stage d6 and "
                "--stage f6 (and o6_reference_derivation.py) first."
            )

    d6_meta = json.loads(d6_meta_path.read_text())
    f6_meta = json.loads(f6_meta_path.read_text())
    gate_d6 = json.loads(d6_gate_path.read_text())
    reference = json.loads(ref_path.read_text())

    d6_log_text = Path(d6_meta["log_path"]).read_text()
    f6_log_text = Path(f6_meta["log_path"]).read_text()
    gate_l6 = _gate_l6(d6_log_text, f6_log_text, f6_meta["selection_in_completion_numerator"])

    f6_diag = pd.read_csv(f6_meta["diagnostics_csv"])
    d6_diag = pd.read_csv(d6_meta["diagnostics_csv"])
    banked_off_diag = pd.read_csv(
        o4.BANKED_DIAG_DIR / f"{ARM}_seed{SEED}" / "event_likelihoods.csv"
    )
    gate_t6 = _gate_t6(f6_diag, banked_off_diag)
    gate_v6 = _gate_v6(f6_diag, d6_diag)

    gates = {"GATE_D6": gate_d6, "GATE_L6": gate_l6, "GATE_T6": gate_t6, "GATE_V6": gate_v6}
    all_gates_pass = all(bool(g.get("pass")) for g in gates.values())

    s_f6 = f6_meta["record"]["score_at_h_gen"]["matched"]["mean_score"]
    r_prod = reference["r_prod_910101"]["mean_score"]

    delta: float | None
    verdict: str
    if not all_gates_pass:
        delta = None
        verdict = "VOID"
    elif s_f6 is None or r_prod is None:
        delta = None
        verdict = "VOID"
    else:
        delta = float(s_f6) - float(r_prod)
        verdict = "MECHANISM-CONFIRMED" if abs(delta) <= PRIMARY_BAND_TOL else "REPLICA-BROKEN"

    secondary_abs_s_f6 = abs(float(s_f6)) if s_f6 is not None else None

    output: dict[str, Any] = {
        "registered_in": REGISTRATION_SECTION,
        "seed": SEED,
        "arm": ARM,
        "h_gen": H_GEN,
        "h_lo": H_LO,
        "h_hi": H_HI,
        "o6_run_metadata": {
            "d6_selection_in_completion_numerator_kwarg": d6_meta[
                "selection_in_completion_numerator"
            ],
            "f6_selection_in_completion_numerator_kwarg": f6_meta[
                "selection_in_completion_numerator"
            ],
            "d6_wall_time_s": d6_meta["wall_time_s"],
            "f6_wall_time_s": f6_meta["wall_time_s"],
            "d6_log_path": d6_meta["log_path"],
            "f6_log_path": f6_meta["log_path"],
        },
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "S_F6": {
            "value": s_f6,
            "reference": (
                "PREREGISTRATION_SELFGEN_CONTROL.md CONFIRMATION RUN O6, "
                "'Primary statistic' -- f6 record['score_at_h_gen']"
                "['matched']['mean_score']"
            ),
        },
        "primary": {
            "statistic": "S(F6) - r_prod(910101)",
            "value": delta,
            "band_tol": PRIMARY_BAND_TOL,
            "subtracts": "r_prod(910101)",
            "reference": (
                "PREREGISTRATION_SELFGEN_CONTROL.md CONFIRMATION RUN O6, A18 note: "
                "the primary subtracts r_prod(910101), provenance this "
                "registration + o6_reference_derivation_output.json"
            ),
        },
        "secondary_reported_only": {
            "statistic": "|S(F6)|",
            "value": secondary_abs_s_f6,
            "frozen_self_consistent_edge": FROZEN_SELF_CONSISTENT_EDGE,
            "subtracts": 0,
            "note": (
                "REPORTED-ONLY, explicitly NOT a band (registration text): a "
                "single-seed realization statement, not a fleet null."
            ),
            "reference": (
                "PREREGISTRATION_SELFGEN_CONTROL.md CONFIRMATION RUN O6, A18 note: "
                "the reported-only secondary subtracts 0 and names the frozen "
                "SELF-CONSISTENT edge 0.037339"
            ),
        },
        "verdict": verdict,
    }

    out_path = RESULTS_DIR / "o6_fused_seed_test_output.json"
    out_path.write_text(json.dumps(output, indent=2))

    print(f"=== O6 score, seed {SEED} ===")
    print(f"all_gates_pass = {all_gates_pass}")
    for name, g in gates.items():
        print(f"  {name}: pass={g.get('pass')}")
    print(f"S(F6)   = {s_f6}")
    print(f"r_prod  = {r_prod}")
    print(f"delta   = {delta}")
    print(f"VERDICT = {verdict}")
    print(f"wrote {out_path}")
    return output


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("d6", "f6", "score", "all"), default="all")
    ap.add_argument(
        "--out-root",
        type=str,
        default=str(RESULTS_DIR / "o6_work"),
        help="Root scratch/output directory for O6's fresh work/out dirs and logs.",
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.stage in ("d6", "all"):
        stage_d6(out_root)
    if args.stage in ("f6", "all"):
        stage_f6(out_root)
    if args.stage in ("score", "all"):
        result = stage_score(out_root)
        return 0 if result["verdict"] == "MECHANISM-CONFIRMED" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
