# N-5 — seed600 494-event 2D subsample under current code (post-D_g-fix) — VERDICT (2026-07-12)

**Provenance:** handoff item **N-5** (`.planning/HANDOFF-DEEP-BIAS-MECHANISM-20260710.md` §N-5;
optional 2D-channel subsample-dependence check). Driver `scripts/eddington_m_impact.py` (threaded
`allow_low_pdet_coverage=True` / `allow_shallow_pool=True` for the archived shallow venue — the
`evaluate()` AND `combine_posteriors()` calls both build a `SimulationDetectionProbability` and
both guard the campaign-depth pool). Data: 494-event seed600 "local derail" subsample CRBs
(`~/data-backups/seed600_local_derail_20260702/crux_ws`) + the real 81-file injection pool
(`~/data-backups/seed600_local_derail_20260702/simulations/injections`). Code at HEAD (includes
the `713fbd1` D_g fix). Grid = 7-pt [0.60…0.86], `normalization_mode="volume_deconv"`.
Artifacts: `docs/gates/G7row9_eddington_m_impact_postDgfix.json` (this run) vs
`docs/gates/G7row9_eddington_m_impact.json` (pre-fix, superseded — see caveat 1).

## VERDICT: the 2D subsample no longer shows the pathological inflation/railing. Under current code the 494-event 2D subsample is well-behaved (edge_mass 0.216 → 0.003, mean 0.790 → 0.768) and consistent with the full-venue 2D (0.7546) up to a subsample-selection offset (+0.0135). No 2D subsample-dependent code defect remains. The remaining venue-level +0.025 2D residual is campaign-gated (D4), unchanged by this probe.

## Numbers

| channel | quantity | PRE-fix artifact | POST-fix (current code) | full-venue 17-pt (current) |
|---|---|---|---|---|
| 1D | mean | 0.73029 | **0.74501** | 0.74320 (+0.013) |
| 1D | edge_mass | 0.0000 | 0.0001 | — |
| 2D | mean | 0.78967 | **0.76813** | **0.75455** |
| 2D | edge_mass | **0.2159** | **0.0028** | — |
| — | Eddington-in-M Δmean_2d (edd − base) | −0.01998 | **−0.00218** | — |

## Reading

1. **The pre-fix artifact is NOT a clean "current-minus-D_g" baseline.** Its 1D mean (0.730) is
   unbiased, whereas current-code 1D (0.745) reproduces the known seed600 +0.013 residual — so the
   pre-fix artifact predates several 1D-affecting changes (#29 fallback, z≥0 clamp, etc.), not only
   the D_g fix. The clean D_g attribution already lives in the L-B **full-venue** A/B
   (`results/seed600_ab_20260710/ANALYSIS.md`: 0.787 → 0.7546 on identical inputs). N-5 therefore
   does NOT re-attribute; it checks the subsample's current behaviour.
2. **2D subsample is now well-behaved.** edge_mass collapsed 0.216 → 0.003: the pre-fix 2D
   "railing toward 0.86" (the source of the 0.79/0.787 subsample inflation) is gone under current
   code. The subsample 2D mean (0.768) sits +0.0135 above the full-venue 2D (0.7546); this is a
   selection effect of the 494-event non-random "local derail" subsample, not a defect — the
   authoritative venue number is the full-venue 0.7546.
3. **1D subsample reproduces the venue.** 0.745 ≈ full-venue 0.7432 (+0.013) — the [L8] shallow
   σ_z/z Eddington residual, consistent across the subsample.
4. **Bonus (stale comment):** the post-fix Eddington-in-M impact on the 2D mean is **−0.0022**,
   an order of magnitude below the **−0.020** cited in `bayesian_statistics.py:2400-2401` (which
   references the pre-fix artifact). The Eddington-in-M correction is even MORE negligible than
   documented. That comment (and the value it quotes) should be refreshed to the post-D_g-fix
   number in a future doc/comment pass. FLAGGED, not edited here (physics-trigger file).

## Decision mapping

- **D4 (2D residual):** unchanged — 57% of the original +0.057 was the D_g defect (L-B); the
  remaining venue-level +0.025 (full-venue 2D 0.7546 vs truth 0.73) is real and **campaign-gated**.
  N-5 confirms no *additional* subsample/grid pathology hides in the 2D channel under current code.
- **Local 2D work is now exhausted**; cross-seed 2D systematic-vs-scatter needs the multi-seed
  campaign (do NOT force locally).

## Caveats

1. Pre/post is NOT a clean single-variable A/B (caveat 1 above); a code-revert A/B isolating
   `713fbd1` alone on the subsample was NOT run (the full-venue L-B A/B already did this cleanly —
   marginal value low). 
2. 494-event subsample is a non-random "local derail" selection; its absolute offset from the full
   venue is a selection effect, not a prediction.
3. Shallow archived venue (pool z_max 0.5); `allow_shallow_pool=True` used deliberately (events at
   z < 0.12 are fully covered).
