<!-- A20 clean-context adversarial review of the [P3-IMP] SHAPE-ONLY arm (Opus, banked verbatim
2026-08-22 ~12:00). Orchestrator note: the reviewer's reproduction was exact to every printed
digit via an independent construction (pivots, not the instrument's merge); the four amendments
are adopted verbatim into PREREGISTRATION_P3_TWIN_20260822.md. AMENDMENT 8's process finding
(silent gate substitution) is owned by the orchestrator. -->

# A20 ADVERSARIAL REVIEW — [P3-IMP] SHAPE-ONLY arm
**Recommendation: BANK-WITH-AMENDMENTS. Zero FATAL.** The label (SHAPE-NULL) survives every attack, including a stress-test far outside the registered sensitivity range. The interpretation does not survive unqualified.

## Recomputed numbers (independent construction; only compute_seed_statistics reused)
Δ̄_shape(12) = +0.000569556 ± 0.000099230 (sd 0.000343741), 12/12 positive — exact reproduction. Baseline fleet bias −0.108302267 (GATE B-S anchor). Δ̄_phi re-referenced independently confirmed: +0.015524133 ± 0.003657306. Level implied +0.014954577. h_ref 0.70/0.76: +0.000500612/+0.000637689. Band: SHAPE-NULL, two-sided as registered; falsifier does not fire.

## PASS items (verified)
A21/A22 ordering clean (registration+bands 0afc4e4e precede the instrument c372678 precede the output; A22 stamp truthful — HEAD never moved, tree clean). Same committed scoring path both arms (compute_seed_statistics consumes only event_idx/h/combined_no_bh — the patched-CSV route is exact). Cross-cell h-only columns bit-identical on all 12 seeds. Zero-ref edge case genuinely benign (every h_ref-zero event is zero at all 46 nodes in both cells). New evidence: the reassembly identity round-trips the φ column to its directly-scored mean_h at ≤3.8e-8/seed.

## MAJOR-1 — GATE B-S as implemented ≠ as registered [MEASURED]
Registered per-seed (≤1e-9) comparand never existed as an artifact (amendment 4 banked fleet numbers only); the instrument silently substituted a fleet-mean anchor check (tol 1e-5; blind to per-seed errors below ~1.2e-4 = 21% of Δ̄_shape) without an A21 STOP-and-amend — the third unavailable-evidence-channel instance, first silent substitution. Discharged in substance by this review (all 12 per-seed baselines re-derived, max diff 0.0). New A17 rule: comparands must exist as banked artifacts, else amend, never substitute.

## MAJOR-2 — "96% is level / shape contributes essentially nothing" is smuggled [MEASURED]
The anchor factor L_cat_phi(0.73)/L_cat_off(0.73): median 0.359, IQR 0.231–0.492, range 0.019–0.958 — the "level" is the per-event, host-z-dependent S̄_φ(z_host) suppression, a z-shape effect acting through the population, anchored away by design. SHAPE-NULL licenses exactly: the residual h-tilt after per-event anchoring at 0.73 is null at the 0.001 level. Not: "the z-shape is physically irrelevant."

## MINOR-1 — additivity measured: direct LEVEL-ONLY arm +0.014929 vs residual +0.014955 (gap −2.5e-5; subtraction legitimate; bank both).
## MINOR-2 — DELTA_PHI_REREFERENCED hardcoded literal (0.015524) vs re-derived +0.015524133 — contrary to A17(e)'s own principle; immaterial.
## MINOR-3 — h_ref falsifier range too narrow; stress-test 0.62–0.86 (REPORTED-ONLY, verdict-inert): +0.000312→+0.000863 monotone, 12/12 positive, SHAPE-NULL everywhere; level share anchor-conditioned ~94–98%.
## MINOR-4 — GATE I-S residual 5.1–5.5e-7 vs 2e-6: approximate identity, harmless here; recorded so no future arm inherits the tolerance blindly.

## VERDICT: BANK-WITH-AMENDMENTS — amendments 8–11 as adopted verbatim into the prereg.
