"""A-FULL-2D pre-submission gate (PREREGISTRATION_A_FULL_2D.md §6 item 1).

Reproduces the mirror pre-measurement (``l6_der2_gsel_premeasure.py``) with the INSTALLED
``ESTIMATOR_VARIANT_A_FULL_GSEL`` code path on seed 20310808 (the premeasure's first row) at
k=20 (h=0.725) and k=22 (h=0.735), full dose, and checks:

  1. ln1 (1D channel): installed ``a_full_gsel`` must be BIT-IDENTICAL to installed ``a_full``
     on the same draw (prereg §3: "1D channel byte-identical to a_full").
  2. ln2 (2D channel): installed ``a_full_gsel`` vs the premeasure's own mirror ``ln2["gsel"]``
     for this seed, |Δ| < 1e-6 at both k.

A failure blocks submission -- no tuning, the discrepancy returns to the author (prereg §6
item 1). Writes ``gate_afull2d_premeasure_check_output.json`` next to this file.
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent
sys.path.insert(0, str(RESULTS_DIR))

from l4_t2_audit import build_population_context  # noqa: E402
from l6_der2_gsel_premeasure import K_HI, K_LO, channel_terms_all_configs  # noqa: E402

from darksiren_emri.validation import venue_transfer as vt  # noqa: E402

SEED = 20310808


def main() -> None:
    print("building context (full dose) ...", flush=True)
    vctx, a_lo, a_hi = build_population_context()
    assert (a_lo, a_hi) == (K_LO, K_HI)

    print(f"drawing seed {SEED} realization ...", flush=True)
    universe, ball, sigma_pairs = vt._draw_seed_realization(SEED, vctx)

    print("premeasure mirror (base/afull/gsel, k=20,22) ...", flush=True)
    ln1_lo, ln2_lo = channel_terms_all_configs(vctx, universe, ball, sigma_pairs, K_LO)
    ln1_hi, ln2_hi = channel_terms_all_configs(vctx, universe, ball, sigma_pairs, K_HI)

    # NOTE: log_channel_posteriors_ball_sigma_vector sweeps the FULL 41-point
    # h_grid; the pre-submission gate (prereg §6 item 1) only needs k=20,22,
    # so this calls the per-h body (_channel_terms_at_h) directly at those
    # two indices -- ~20x cheaper than the full sweep for the same check.
    print("installed a_full direct call (k=20,22, per-h body only) ...", flush=True)
    ln1_afull_lo, _, _ = vt._channel_terms_at_h(
        vctx.gctx,
        universe,
        ball,
        sigma_pairs,
        K_LO,
        chunk_pairs=vctx.vcfg.chunk_pairs,
        estimator_variant=vt.ESTIMATOR_VARIANT_A_FULL,
    )
    ln1_afull_hi, _, _ = vt._channel_terms_at_h(
        vctx.gctx,
        universe,
        ball,
        sigma_pairs,
        K_HI,
        chunk_pairs=vctx.vcfg.chunk_pairs,
        estimator_variant=vt.ESTIMATOR_VARIANT_A_FULL,
    )

    print("installed a_full_gsel direct call (k=20,22, per-h body only) ...", flush=True)
    ln1_gsel_lo, ln2_gsel_lo, _ = vt._channel_terms_at_h(
        vctx.gctx,
        universe,
        ball,
        sigma_pairs,
        K_LO,
        chunk_pairs=vctx.vcfg.chunk_pairs,
        estimator_variant=vt.ESTIMATOR_VARIANT_A_FULL_GSEL,
    )
    ln1_gsel_hi, ln2_gsel_hi, _ = vt._channel_terms_at_h(
        vctx.gctx,
        universe,
        ball,
        sigma_pairs,
        K_HI,
        chunk_pairs=vctx.vcfg.chunk_pairs,
        estimator_variant=vt.ESTIMATOR_VARIANT_A_FULL_GSEL,
    )

    # gate 1: ln1 bit-identity, installed a_full_gsel vs installed a_full
    diff_ln1_lo = float(ln1_gsel_lo - ln1_afull_lo)
    diff_ln1_hi = float(ln1_gsel_hi - ln1_afull_hi)
    max_diff_ln1 = max(abs(diff_ln1_lo), abs(diff_ln1_hi))

    # gate 1b: installed a_full_gsel's ln1 also bit-identical to the premeasure's own
    # afull mirror (cross-check: two independent a_full re-derivations agree)
    diff_ln1_vs_mirror_lo = float(ln1_gsel_lo - ln1_lo["afull"])
    diff_ln1_vs_mirror_hi = float(ln1_gsel_hi - ln1_hi["afull"])
    max_diff_ln1_vs_mirror = max(abs(diff_ln1_vs_mirror_lo), abs(diff_ln1_vs_mirror_hi))

    # gate 2: ln2 vs premeasure's own gsel mirror for this seed
    diff_ln2_lo = float(ln2_gsel_lo - ln2_lo["gsel"])
    diff_ln2_hi = float(ln2_gsel_hi - ln2_hi["gsel"])
    max_diff_ln2 = max(abs(diff_ln2_lo), abs(diff_ln2_hi))

    result = {
        "seed": SEED,
        "k_lo": K_LO,
        "k_hi": K_HI,
        "ln1_installed_afull": {"lo": float(ln1_afull_lo), "hi": float(ln1_afull_hi)},
        "ln1_installed_gsel": {"lo": float(ln1_gsel_lo), "hi": float(ln1_gsel_hi)},
        "ln1_premeasure_mirror_afull": {"lo": float(ln1_lo["afull"]), "hi": float(ln1_hi["afull"])},
        "ln2_installed_gsel": {"lo": float(ln2_gsel_lo), "hi": float(ln2_gsel_hi)},
        "ln2_premeasure_mirror_gsel": {"lo": float(ln2_lo["gsel"]), "hi": float(ln2_hi["gsel"])},
        "gate_1_ln1_bit_identity_installed_gsel_vs_installed_afull": {
            "max_abs_diff": max_diff_ln1,
            "pass": bool(max_diff_ln1 == 0.0),
        },
        "gate_1b_ln1_installed_gsel_vs_premeasure_mirror_afull": {
            "max_abs_diff": max_diff_ln1_vs_mirror,
            "pass": bool(max_diff_ln1_vs_mirror == 0.0),
        },
        "gate_2_ln2_installed_gsel_vs_premeasure_mirror_gsel": {
            "max_abs_diff": max_diff_ln2,
            "threshold": 1e-6,
            "pass": bool(max_diff_ln2 < 1e-6),
        },
    }
    overall_pass = (
        result["gate_1_ln1_bit_identity_installed_gsel_vs_installed_afull"]["pass"]
        and result["gate_1b_ln1_installed_gsel_vs_premeasure_mirror_afull"]["pass"]
        and result["gate_2_ln2_installed_gsel_vs_premeasure_mirror_gsel"]["pass"]
    )
    result["overall_pass"] = bool(overall_pass)

    out_path = RESULTS_DIR / "gate_afull2d_premeasure_check_output.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)

    print("\n=== A-FULL-2D pre-submission gate (prereg §6 item 1) ===")
    print(f"seed={SEED} k_lo={K_LO} (h={vctx.gctx.cl_ctx.config.h_grid[K_LO]:.3f}) "
          f"k_hi={K_HI} (h={vctx.gctx.cl_ctx.config.h_grid[K_HI]:.3f})")
    print(f"gate 1 (ln1 installed gsel vs installed afull): max|diff|={max_diff_ln1:.3e} "
          f"-> {'PASS' if result['gate_1_ln1_bit_identity_installed_gsel_vs_installed_afull']['pass'] else 'FAIL'}")
    print(f"gate 1b (ln1 installed gsel vs premeasure afull mirror): max|diff|={max_diff_ln1_vs_mirror:.3e} "
          f"-> {'PASS' if result['gate_1b_ln1_installed_gsel_vs_premeasure_mirror_afull']['pass'] else 'FAIL'}")
    print(f"gate 2 (ln2 installed gsel vs premeasure gsel mirror): max|diff|={max_diff_ln2:.3e} "
          f"(threshold 1e-6) -> {'PASS' if result['gate_2_ln2_installed_gsel_vs_premeasure_mirror_gsel']['pass'] else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print(f"wrote {out_path}")

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
