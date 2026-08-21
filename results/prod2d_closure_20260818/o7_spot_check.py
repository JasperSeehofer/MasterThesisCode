r"""O7 arm S7 -- the two registered fused spot-check seeds (910105, 910113).

Registered in ``PREREGISTRATION_SELFGEN_CONTROL.md`` "O7 -- FLEET TRANSFER CLOSE"
(2026-08-22, row #159 item 1). Runs each registered seed end-to-end under the
``fused`` cell via the COMMITTED O6 driver's ``_run_arm`` (imported, not
reimplemented; the module-level ``SEED`` constant is set per seed -- the one
deviation from verbatim reuse, disclosed here and in the output), then applies
the registered gates and bands:

- **GATE L7** (zero compute): the fused log line present, the off-cell
  counterfactual line absent (O6 GATE L6's F6-side conditions).
- **GATE V7**: fused ``B_num`` differs from the banked off-cell CSV on > 99% of
  rows (O6 GATE V6's anti-vacuity, against the banked column since no off
  replica is re-run -- registered justification in the O7 arm table).
- **Band**: TRANSFER-HOLDS iff ``|S_fused(seed) - r_prod(seed)| <= 1e-4`` with
  ``r_prod`` read from the banked ``o7_reference_fleet_output.json``; else
  TRANSFER-BROKEN; any gate failure => VOID for that seed.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import o4_pairing_test as o4  # noqa: E402
import o6_fused_seed_test as o6t  # noqa: E402

BASE = Path(__file__).parent
OUT_PATH = BASE / "o7_spot_check_output.json"
R7_PATH = BASE / "o7_reference_fleet_output.json"
REGISTRATION_SECTION = (
    "results/prod2d_closure_20260818/PREREGISTRATION_SELFGEN_CONTROL.md, "
    "O7 -- FLEET TRANSFER CLOSE: REGISTRATION (2026-08-22, row #159 item 1), arm S7"
)
SPOT_SEEDS: tuple[int, ...] = (910105, 910113)
BAND_TOL = 1e-4
FUSED_LINE = "selection fusion ACTIVE"
OFF_LINE = "selection_in_completion_numerator='off'"


def run_seed(seed: int, out_root: Path) -> dict[str, Any]:
    """Run one registered spot-check seed under ``fused`` and score it."""
    o6t.SEED = seed  # disclosed module-constant patch; _run_arm reused verbatim
    meta = o6t._run_arm(out_root, f"s7_{seed}", "fused")
    record = meta["record"]

    log_text = Path(meta["log_path"]).read_text()
    gate_l7 = {
        "gate": "GATE_L7",
        "fused_line_present": FUSED_LINE in log_text,
        "off_cell_line_absent": OFF_LINE not in log_text,
        "reference": f"{REGISTRATION_SECTION}, gates",
    }
    gate_l7["pass"] = gate_l7["fused_line_present"] and gate_l7["off_cell_line_absent"]

    fresh = pd.read_csv(meta["diagnostics_csv"])
    banked = pd.read_csv(o4.BANKED_DIAG_DIR / f"csgf_seed{seed}" / "event_likelihoods.csv")
    merged = fresh.merge(
        banked[["event_idx", "h", "B_num"]],
        on=["event_idx", "h"],
        suffixes=("_fused", "_off"),
        how="outer",
        indicator=True,
    )
    key_mismatch = bool((merged["_merge"] != "both").any())
    a = merged["B_num_fused"].to_numpy(dtype=np.float64)
    b = merged["B_num_off"].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        differ = np.abs(a - b) > 1e-9 * np.maximum(np.abs(b), np.finfo(float).tiny)
    differ_fraction = float(np.mean(differ)) if differ.size else 0.0
    gate_v7 = {
        "gate": "GATE_V7",
        "key_mismatch": key_mismatch,
        "differ_fraction": differ_fraction,
        "min_differ_fraction": 0.99,
        "n_rows_compared": int(len(merged)),
        "pass": (not key_mismatch) and differ_fraction > 0.99,
        "reference": f"{REGISTRATION_SECTION}, gates",
    }

    r7 = json.loads(R7_PATH.read_text())
    r_prod = next(s["r_prod"]["mean_score"] for s in r7["per_seed"] if s["seed"] == seed)
    s_fused = float(record["score_at_h_gen"]["matched"]["mean_score"])
    delta = s_fused - r_prod
    gates_pass = bool(gate_l7["pass"] and gate_v7["pass"])
    if not gates_pass:
        band = "VOID"
    elif abs(delta) <= BAND_TOL:
        band = "TRANSFER-HOLDS"
    else:
        band = "TRANSFER-BROKEN"

    return {
        "seed": seed,
        "gates": {"GATE_L7": gate_l7, "GATE_V7": gate_v7},
        "S_fused": s_fused,
        "primary": {
            "statistic": f"S_fused({seed}) - r_prod({seed})",
            "value": delta,
            "band_tol": BAND_TOL,
            "subtracts": f"r_prod({seed}) = {r_prod!r}",
            "reference": (
                f"{REGISTRATION_SECTION}; r_prod from the banked "
                "o7_reference_fleet_output.json (A18)"
            ),
        },
        "band": band,
        "wall_time_s": meta["wall_time_s"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=BASE / "o7_work")
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    results = [run_seed(seed, args.out_root) for seed in SPOT_SEEDS]
    both_hold = all(r["band"] == "TRANSFER-HOLDS" for r in results)
    r7 = json.loads(R7_PATH.read_text())
    fleet_claim: dict[str, Any] = {
        "closes": both_hold,
        "statement": (
            "S_bar_15(fused) = r_bar_prod(15) +/- SEM(15) = "
            f"{r7['fleet']['r_prod_fleet_mean']:+.6f} +/- "
            f"{r7['fleet']['r_prod_fleet_sem']:.6f} by measured transfer"
            if both_hold
            else "DOES NOT CLOSE -- zero-compute audit before any interpretation "
            "(registered); full 15-seed fleet returns to the author (D1 option B)"
        ),
        "reference": f"{REGISTRATION_SECTION}, 'The fleet claim (registered wording)'",
    }
    output = {
        "registered_in": REGISTRATION_SECTION,
        "spot_seeds": list(SPOT_SEEDS),
        "seed_selection_criterion": (
            "registered pre-data: the two extremes of the banked off-cell score "
            "range (910105 most negative -0.2648; 910113 sign-flip outlier +0.0828)"
        ),
        "results": results,
        "fleet_claim": fleet_claim,
        "reuse_disclosure": (
            "o6_fused_seed_test._run_arm reused verbatim; its module SEED "
            "constant is set per spot-check seed (the one disclosed deviation)."
        ),
    }
    OUT_PATH.write_text(json.dumps(output, indent=2))
    print("=== O7 S7 spot-check ===")
    for r in results:
        print(
            f"seed {r['seed']}: S_fused = {r['S_fused']:+.6f}  "
            f"delta = {r['primary']['value']:+.3e}  band = {r['band']}  "
            f"(L7 {r['gates']['GATE_L7']['pass']}, V7 {r['gates']['GATE_V7']['pass']})"
        )
    print(f"fleet claim closes: {both_hold}")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
