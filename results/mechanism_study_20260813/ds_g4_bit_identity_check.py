#!/usr/bin/env python
"""DS-G4 bit-identity check (prereg §4): full 41-point grid on >=2 arm seeds.

Reproduces the ``a_full`` (not ``a_full_gsel``) variant locally on the AFULL2D
arm's own config (crb_reference_csv, frozeng_emit_json, pruned_catalogue_csv,
injection_data_dir, chunk_pairs, h_grid, etc -- read straight out of the
retrieved combined arm JSON), draws the same seed realization via
``_draw_seed_realization``, and calls ``run_seed_venue`` with
``estimator_variant="a_full"`` for the full 41-point h_grid. Compares the
resulting ``ln_post_1d`` vector against the arm's retrieved (a_full_gsel)
``ln_post_1d`` for the same seed -- prereg says these must be bit-identical
(the 1D channel does not depend on the gsel/base 2D-selection machinery).

Run for seeds 20315108 and 20315120 (2 of the 25 arm seeds, prereg's ">= 2").
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent
REPO_ROOT = RESULTS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from darksiren_emri.validation import venue_transfer as vt

ARM_JSON = RESULTS_DIR / "AFULL2D_h0p730_results_seeds0_25.json"
CHECK_SEEDS = [20315108, 20315120]


def main() -> int:
    arm = json.loads(ARM_JSON.read_text())
    cfg = arm["config"]

    vcfg = vt.VenueConfig(
        cell=cfg["cell"],
        h_true=cfg["h_true"],
        balls=cfg["balls"],
        sigma_mode=cfg["sigma_mode"],
        flat_sigma_z=cfg["flat_sigma_z"],
        lambda_poisson=cfg["lambda_poisson"],
        dose_target=cfg["dose_target"],
        dose_scales=cfg["dose_scales"],
        crb_reference_csv=cfg["crb_reference_csv"],
        frozeng_emit_json=cfg["frozeng_emit_json"],
        pruned_catalogue_csv=cfg["pruned_catalogue_csv"],
        injection_data_dir=cfg["injection_data_dir"],
        n_events_cap=cfg["n_events_cap"],
        chunk_pairs=cfg["chunk_pairs"],
        h_grid=cfg["h_grid"],
        # estimator_variant overridden per-call below; base context value unused
        estimator_variant=vt.ESTIMATOR_VARIANT_A_FULL,
    )
    print("building venue context (full dose, arm's own config) ...", flush=True)
    vctx = vt.build_venue_context(vcfg)

    per_seed_by_seed = {rec["seed"]: rec for rec in arm["per_seed"]}

    results = {}
    overall_pass = True
    for seed in CHECK_SEEDS:
        if seed not in per_seed_by_seed:
            print(f"SEED {seed} NOT FOUND IN ARM JSON -- aborting")
            return 1
        arm_ln1 = np.asarray(per_seed_by_seed[seed]["ln_post_1d"], dtype=np.float64)

        print(f"seed {seed}: drawing realization + running a_full over full 41-point grid ...", flush=True)
        universe, ball, sigma_pairs = vt._draw_seed_realization(seed, vctx)
        ln1_local, ln2_local, _slope = vt.log_channel_posteriors_ball_sigma_vector(
            vctx.gctx,
            universe,
            ball,
            sigma_pairs,
            chunk_pairs=vctx.vcfg.chunk_pairs,
            estimator_variant=vt.ESTIMATOR_VARIANT_A_FULL,
        )

        diff = arm_ln1 - ln1_local
        max_abs_diff = float(np.max(np.abs(diff)))
        n_points = len(arm_ln1)
        results[seed] = {
            "n_points": n_points,
            "max_abs_diff": max_abs_diff,
            "all_zero": bool(max_abs_diff == 0.0),
            "arm_ln1_sample": arm_ln1[:3].tolist(),
            "local_a_full_ln1_sample": ln1_local[:3].tolist(),
        }
        print(f"  n_points={n_points} max|diff|={max_abs_diff:.3e} -> "
              f"{'BIT-IDENTICAL' if max_abs_diff == 0.0 else 'DIFFERS'}")
        overall_pass = overall_pass and (max_abs_diff == 0.0)

    out = {
        "check": "DS-G4 1D-channel bit-identity: arm's ln_post_1d (a_full_gsel) vs local a_full, full 41-point grid",
        "seeds_checked": CHECK_SEEDS,
        "results": results,
        "overall_pass": overall_pass,
    }
    out_path = RESULTS_DIR / "ds_g4_bit_identity_check_output.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nOVERALL DS-G4: {'PASS (bit-identical)' if overall_pass else 'FAIL'}")
    print(f"wrote {out_path}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
