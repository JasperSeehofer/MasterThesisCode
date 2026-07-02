# G6 — Starvation post-mortem: reconciling a8cbab0 with the de-rail verdict (2026-07-02)

## Question

The a8cbab0 verdict ("in-catalogue photo-z dark sirens are information-starved; NO normalization
fix recovers a peaked H₀") and the commission's de-rail result (peaked 0.73 via volume_deconv on
identical real data) appear to contradict. Which is right, and why did the earlier tests conclude
starvation?

## Evidence assembled

1. **rung G (bare conv, historical):** rails at σ_z×1.0, MAP 0.857/0.87 — reproduced this session
   on the full 3355-event set (`rungG2_results.json`, "bare photo-z x1.0": MAP 0.87, railed).
2. **rung G2 NEW (volume prior, numerator-only):** still rails — MAP 0.87 at σ_z×{0.5, 1.0},
   0.60 at ×0.25; N-scaling flat (width pinned at the railed grid floor). The harness's
   `regularise_photoz` weights ONLY the numerator host-z kernel; its L_cat denominator is the
   bare global selection sum — so this variant is a numerator-tilted / denominator-bare estimator.
3. **D_sm (a8cbab0's candidate):** global smeared denominator — falsified (flat/multimodal,
   width did not shrink with N).
4. **Production ablation cube (real 494-event seed600, G3):** bare+global rails 0.60;
   volume kernel in BOTH N_g and D_g with global denominator peaks at 0.76;
   with local denominator peaks at 0.73 (calibrated).
5. **d2 synthetic coverage (120 realizations × 250 events):** the CONSISTENT volume-weighted
   estimator is calibrated (≈nominal coverage, bias −0.002) — width behaves statistically.

## Verdict

**"Information starvation" was a property of prior-INCONSISTENT estimators, not of the data.**
Every configuration that ever "starved" or railed shares one defect: the host-z (or host-M) prior
enters the numerator and the selection denominator differently — bare/bare-global (rail up),
4π-fixed bare (rail down), numerator-only volume tilt (rail, rung G2), global smeared denominator
D_sm (flat/multimodal). The single configuration family that recovers and calibrates is the one
where the SAME population prior multiplies the same measurement likelihood on BOTH sides
("counted exactly once"): volume_deconv (and its volume_global diagnostic sibling).

rung G2's failure is therefore a FEATURE of the record: it is the negative control showing a
partial (numerator-only) fix does not cure the rail — consistency, not any single factor, is the
cure. This subsumes a8cbab0: its conclusion was correct FOR THE ESTIMATORS IT TESTED and does not
extend to the consistent estimator, which demonstrably extracts a peaked, calibrated H₀ from the
same photo-z-dominated data.

Paper A framing: "partial-fix matrix" — the ablation cube (G3) plus rung G2 as the
numerator-only column; the σ_z² Eddington law (G2b) explains the mechanism.

Artifacts: `scripts/bridge_closure/outputs/rungG2_results.json` + `rungG2_volume_prior.pdf`;
`.planning/gate/G3_ablation_cube.json`; `results/commission_20260701/scratch/d2/`.
