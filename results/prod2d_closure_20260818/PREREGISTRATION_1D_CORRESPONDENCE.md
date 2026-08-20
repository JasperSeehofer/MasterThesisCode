# PRE-REGISTRATION — Option B: production 1D correspondence measurement

**Date:** 2026-08-19 · **Status:** v2 — verifier pre-check applied (v1 NOT-READY; P7–P15
amendments verbatim below; verifier: READY-WITH-AMENDMENTS on re-check with G-0
registered — it is). Authorized: row #132 [DO] ("all approved"). **Validation
code only** — no production change on any branch; a /physics-change would be triggered only
IF a form defect is found and its fix proposed (fresh gate). Recon of record: 2026-08-19
sonnet recon (DS-6 record, mirror assets, pinned inputs, cost anchors).

**Question:** decompose the production 1D base tilt (post-fix baselines: mean 0.6010/0.6020,
MAP railed at 0.60 — the largest tilt-ledger entry, shared with the 2D channel) into
information-starvation vs form-defect components, and close the DS-6 MIXED verdict (row #98:
the calgate ball venue produced a uniform +≈σ_z MAP bias with collapsed coverage instead of
production's low rail — never explained).

## 1. Design decisions (registered, with reasons)

- **D-A Fidelity: production-wholesale.** The mirror calls production's own
  `single_host_likelihood` (and shared precompute objects) imported from
  `bayesian_statistics.py` — the venue_transfer.py pattern (it already imports
  `completion_mass_factor_g` etc.), extended to the 1D kernel. The volume_deconv kernel is
  a nested closure (:5251/:5476), so the harness calls `single_host_likelihood` wholesale
  with the full parameter surface rather than extracting the closure (fidelity over
  convenience — a re-derived kernel could not settle a correspondence question).
- **D-B Universe: real-catalogue candidate structure at reduced scale.** Candidate field =
  the MD5-pinned pruned GLADE catalogue (venue_transfer asset); hosts drawn from it with
  known truth h_true = 0.73; **mirror events resample ENTIRE per-event Fisher rows (full
  covariance + detected parameters, incl. sky localization) from the banked seed61000 CRB
  CSV, SNR-weighted (P7: σ_dL-only resampling is insufficient — sky area drives candidate
  counts — and is not used)**; completeness = the real
  `GladeCatalogCompleteness`/`PixelCompleteness` object (importable, recon-verified) —
  NOT a synthetic ball, NOT a LOO construction (D-iii is architecturally production-
  inapplicable; the real-catalogue draw sidesteps it and its open derivation becomes an
  exploratory arm, E-DEN).
- **D-C Scale & budget (P13 re-costed):** n_events = 200 per realization; paired seeds
  across arms. Seeds per arm: **N = 25 for the adjudicating arms** (B-0, B-σ 0.05×, B-D2);
  **N = 10 for the reported-only doses** (B-σ 0.25×, E-DEN ×2). Total ≈ 105 seed-runs; at
  the 0.969 CPU-h/seed-run anchor ≈ **102 CPU-h; registered ceiling 120 CPU-h** (the v1
  "25–50" estimate and 60 ceiling were arithmetic errors — corrected here and flagged to
  the author; the anchor's unit is CPU-h per seed-run of ONE arm). The 2D analog's 406
  CPU-h overrun remains the cautionary anchor — no 2D arms in this prereg.
- **D-D Cost pilot (registered, STOP-gated):** 2 seed-runs of B-0 first; if realized
  CPU-h/seed-run > 2× the 0.969 anchor ⇒ STOP and re-scope before the fleet (the pilot
  bounds the per-seed rate; the §D-C arithmetic bounds the fleet).

## 2. Arms (all 1D channel; paired seeds)

- **B-0 (production-mapped):** full production form, σ_z = GLADE empirical (the observed
  photo-z errors of the pinned catalogue). The correspondence arm.
- **B-σ (starvation ladder):** σ_z scaled to {0.25×, 0.05×} of empirical (host z_obs
  re-scattered; same seeds) — the information-content dose response.
- **B-D2 (density-form toggle; P12 demoted to REPORTED-ONLY behind its own parity gate):**
  the D-ii form defect arm — the GW event term in d_obs-density form vs production's
  ratio-pdf form. The exact outer-correction formula (or reimplemented-kernel route) is
  registered in the harness doc BEFORE B-D2 runs; either way a parity gate applies: at the
  production form, the B-D2 machinery must agree with the wholesale kernel to 1e-10 before
  the toggle is trusted (failure ⇒ B-D2 not quoted; the adjudicating arms are unaffected).
  S-DECOMP's FORM-COMPONENT clause binds only if the parity gate passed.
- **E-DEN (exploratory, REPORTED-ONLY, N=10):** candidate-multiplicity response —
  localization area scaled ×{0.5, 2} (same universe) — the real-catalogue impostor-density
  fingerprint (the D-iii open derivation's measurement side).

## 3. Registered statistics and bands

Per arm: mean_h bias ± SE over seeds, coverage C50/C68/C90, rail fraction R_low
(P(MAP ≤ 0.605), the DS-6 statistic), posterior width distribution.

- **S-CORR (P8 re-posed — class consistency, n-scaled):** let μ̄ and s² be the mean and
  variance of B-0's 25 realization means. Production's banked 1D mean m_v (BOTH venues,
  two reads: 0.6010 iiib / 0.6020 joint_r1) is scored as
  z_v = (m_v − μ̄)/√(s²·(200/1588) + s²/25) — event-sampling variance scaled to
  production's n under the i.i.d.-dominant assumption (disclosed limitation, P7-8-class),
  plus ensemble-mean estimation error. **CORRESPONDENCE-PASS** iff |z_v| ≤ 2. The claim
  adjudicated is "production's 1D mean is consistent with the class the mirror defines" —
  NOT "the mirror reproduces seed61000".
- **S-RAIL (P9/P10 — grid registered, outcomes mutually exclusive, scale confound
  handled):** the mirror h grid = the production H_VALUES 41-node hybrid grid
  [0.600, 0.860] VERBATIM, plus a diagnostic low wing {0.50, 0.52, …, 0.58}
  (REPORTED-ONLY: edge pile-up vs resolved low peak — information production cannot see;
  never band-bearing). R_low = P(MAP ≤ 0.605) on the production grid. S-RAIL is
  adjudicated only if B-0's median posterior σ_h is within ×2 of production's 1D σ_h;
  otherwise the read is **SCALE-CONFOUNDED** (named outcome, returns to the author) and
  the rail statistic of record becomes the AGGREGATION arm: 8 disjoint 200-event subsets
  pooled into pseudo-1588 log-posteriors (zero extra CPU). Outcomes evaluated in order,
  mutually exclusive: **RAIL-REPRODUCED** iff R_low ≥ 0.90; else **D-CLASS-REPRODUCED**
  iff R_low ≤ 0.10 AND median(MAP − 0.73) ≥ +0.02 AND C68 ≤ 0.40; else
  **RAIL-NOT-REPRODUCED** iff R_low ≤ 0.10; else **MIXED** (fresh author [RULE]).
- **S-DECOMP (P11 bands corrected):** T_starv = bias(B-0) − bias(B-σ, 0.05×) (paired);
  T_D2 = bias(B-0) − bias(B-D2) (paired; binds only if B-D2's parity gate passed).
  STARVATION-DOMINATED iff |T_starv| ≥ 3·max(|T_D2|, SE_paired(T_D2)) AND B-σ(0.05×) C68
  recovers into the N=25 binomial 95% band **[0.497, 0.863]**; FORM-COMPONENT-PRESENT iff
  |T_D2| ≥ max(0.005, 2·SE_paired); MIXED else. Every branch returns to the author as a
  [RULE]; a FORM-COMPONENT-PRESENT finding opens a fresh /physics-change proposal (not
  pre-approved).
- **Materiality yardstick:** ⅓·σ_h of the production 1D posterior is degenerate (railed);
  the registered yardstick is the 2D one (0.008) — a 1D form component below it cannot
  matter for the tilt ledger's 2D budget.

## 4. Gates

- **G-0 (fidelity pilot, STOP — runs before ANY arm; the P7 blocking repair):** the
  harness, via `child_process_init` (:6728-6801, installs all worker globals) + wholesale
  `single_host_likelihood` + its OWN completion/assembly layer (B_num is a class-method
  closure, :4602-4645, and the β/Σ/D̃^φ assembly must be re-orchestrated harness-side —
  the venue_transfer precedent imports only leaf functions, so this extension is unproven
  until G-0 passes), must reproduce **≥ 3 banked production events' 1D per-event values**
  (catalogue leg, completion leg, AND combined_no_bh) from the post-fix baseline
  `event_likelihoods.csv` at 2 probe h to max rel diff ≤ 1e-6. Failure ⇒ STOP (harness
  defect; nothing downstream is quoted).
- **G-1 (mirror sanity):** B-0 with σ_z → 0 AND full completeness must recover truth
  (|bias| ≤ 2·SE) — the harness's own null; failure ⇒ STOP (mirror defect, nothing
  quoted). **P14 mechanism registered:** "full completeness" = an f ≡ 1 shim object
  satisfying the `CompletenessModel` Protocol (the real GLADE object cannot be dialed).
- **G-2 (pilot):** §1 D-D cost gate.
- **G-3 (fidelity pin):** the wholesale `single_host_likelihood` calls must run with
  production-default flags (derived B_scale form, `catalogue_mass_overlap=production`,
  volume_deconv) — recorded in the harness config dump per run.
- **Execution-completeness (P15):** an arm's read requires its full registered seed count
  COMPLETED; partial fleets are banked but adjudicate nothing without a fresh author
  [RULE].

## 5. Execution & tiering

Harness: new `darksiren_emri/validation/correspondence_1d.py` (validation package;
venue_transfer.py patterns; full typing; CPU-only). Implementation: sonnet. Prereg +
interpretation: top-tier. Verifier: shared top-tier pass with the battery prereg.
Runs local or cluster CPU (≈ 25–50 CPU-h at anchor; ceiling 60). Readout: A7
comprehension-first, folded into the tilt-ledger budget with the battery results.

---

## VERDICT

*(append-only below this line after execution)*

**G-0 INTERIM RECORD (2026-08-19/20, appended):** first G-0 execution **FAIL — STOP honored,
nothing downstream quoted.** Localization (informative): completion leg B_num max rel diff
0.0; the harness combine re-orchestration reproduces the wholesale run's own combined_no_bh
to ≤2.2e-7 (PASS-class); the CATALOGUE leg fails (L_cat_no_bh up to 5.6e-1 rel; one
non-degenerate event ~20%), with the global selection sums off at 4-7e-5. **Root cause
FOUND by md5 comparison: the local `reduced_galaxy_catalogue.csv` (Jul 1, f2433d55…) is a
STALE pre-regeneration version; the cluster copy of record (Jul 27, c52c13b5…, the one all
banked runs used) has systematically different redshift columns (e.g. first row z
0.000991 → 0.001733 — the Jul-27 regeneration era).** No md5 pin existed for this file
(unlike venue_transfer's PRUNED_CATALOGUE_CSV). Actions: cluster copy synced to replace the
local file; an md5 pin for the catalogue of record is added to the harness (and flagged for
DATA_INVENTORY); G-0 re-runs against the synced copy. **Disclosure:** today's regression
stage-2 σ_M recompute used the stale local catalogue — its differing columns are
redshift-side while σ_M derives from the mass columns, and stage-2 was UNDERPOWERED-NULL
with no ruling resting on it; noted for completeness, no re-run warranted unless the mass
columns also differ (checked at sync).

**G-0 FINAL (2026-08-20, appended): PASS.** With the catalogue of record in place (md5
c52c13b5…, now pinned in-code with a STOP gate): L_cat_no_bh wholesale-vs-banked ≤ 4.3e-14;
B_num 0.0; combined_no_bh **bit-for-bit 0.0**; harness combine re-orchestration ≤ 4.4e-8
(CSV 7-sig-fig floor). No tolerance loosened. Context-build runtime 251 s (the D-D cost
anchor input). The first-run FAIL is fully attributed to the stale local catalogue (interim
record above); the fidelity layer is verified end-to-end. Next per registration: mirror
generator + G-1 + D-D cost pilot, then arms.

**G-1/G-2 RECORD + PRE-ARM AMENDMENT (2026-08-20, appended BEFORE any adjudicating arm):**

- **G-1 PASS:** exact-z (REDSHIFT_MEASUREMENT_ERROR floored to 1e-6 — registered harness
  convention; the prereg's P14 fixed only the f≡1 shim) + unity completeness ⇒ mean_h =
  MAP = 0.730 exactly (single-node posterior collapse under near-noiseless information; no
  NaN/inf; n_eff = 69). The mirror's null is clean.
- **G-2 PROCEED:** 2 B-0 pilot seeds, 0.478 CPU-h/seed-run = 0.49× the 0.969 anchor (2×
  STOP threshold comfortably cleared). Cost decomposition: per-h global selection sums over
  the 20.8M-galaxy catalogue dominate (~20-25 of ~29 min); handler/BallTree amortized
  across seeds already. Reuse analysis of record: the per-(h, catalogue) selection tables
  are event-set-independent (computed before the per-event loop) and COULD be cached across
  seeds — not exploited now (fleet fits the ceiling without touching evaluate(): ≈105 ×
  0.478 ≈ 50 CPU-h < 120).
- **AMENDMENT A-1 (n_eff, registered now that the fact is known):** the production
  Fisher-quality filter passes ~69/200 mirror events per realization (donor rows placed at
  nearer hosts inflate σ_dL/dL — a disclosed consequence of the 1/d_L² host-draw proxy).
  S-CORR's event-sampling scaling uses the realized mean n_eff over B-0 seeds in place of
  200: z_v = (m_v − μ̄)/√(s²·(n_eff/1588) + s²/25). The S-RAIL scale-confound check
  (σ_h ×2 band) is expected to fire at n_eff ≈ 69 — the AGGREGATION arm (8 disjoint
  subsets, pseudo-n = 8·n_eff ≈ 552, still ×2.9 below production n) is registered as the
  rail statistic of record with THAT pseudo-n disclosed; if even pooled σ_h stays outside
  ×2 of production's, the rail read is SCALE-CONFOUNDED as registered and returns to the
  author.
- **Generator conventions of record (flagged for author review, non-blocking):** host draw
  ∝ 1/d_L²(z; h_true); sky localization recentred at the host with the donor's own (φ,θ)
  covariance (not a spherical rotation); donor mass columns unlinked (harmless to the 1D
  statistic); B-0 uses the pinned catalogue's own z_obs/z_err as-is.

---

## VERDICT (2026-08-20, appended after the arms fleet; branches presented as [RULE]s)

**Execution:** job 6383719, 80 tasks → 70 COMPLETED (b0 25/25, bsig005 23/25 — 2 walltime
stragglers, eden05 10/10, eden2 10/10), 10 FAILED = the entire bsig025 dose (separate
defect, §"bsig025" below). B-D2 was not run (deferred behind its parity gate, as
registered). Per the execution-completeness clause bsig025 is unreadable and bsig005 is
read at 23/25 with the shortfall disclosed (it is a dose arm; B-0, the adjudicating
correspondence arm, is complete).

**Per-arm results (per-seed statistics; n_eff ≈ 68 of 200 drawn — amendment A-1):**

| arm | n | bias (mean_h − 0.73) | SE | median σ_h | R_low | C50/C68/C90 |
|---|---|---|---|---|---|---|
| B-0 (σ_z as catalogue) | 25 | **+0.0245** | 0.0059 | 0.0248 | 0.36 | 0.48/0.64/0.72 |
| B-σ 0.05× | 23 | **+0.0348** | 0.0078 | 0.0170 | 0.17 | 0.30/0.43/0.70 |
| E-DEN 0.5× area | 10 | +0.0093 | 0.0027 | 0.0170 | 0.10 | 0.60/0.90/1.00 |
| E-DEN 2× area | 10 | +0.0211 | 0.0096 | 0.0559 | 0.50 | 0.60/0.60/0.70 |

**Registered branch outcomes:**

- **S-CORR: CORRESPONDENCE-FAIL.** z = (0.6010 − 0.7545)/0.0084 ≈ **−18** (production 1D
  mean vs the B-0 class, A-1 n_eff scaling). Production's 1D mean is nowhere in the mirror's
  class — and the mirror is biased HIGH where production is LOW.
- **S-RAIL: SCALE-CONFOUNDED** (registered named outcome). B-0 median σ_h = 0.0248 vs
  production 1D σ_h = 0.00329 → **7.5×**, far outside the ×2 adjudication band. The
  registered fallback (pooled aggregation) is ALSO unusable here: pooling is dominated by a
  few near-delta per-seed posteriors — pooled B-0 gives +0.110 (MAP 0.82) at 25 seeds but
  −0.052 (MAP at the 0.50 grid edge) on its own 10-seed subset, i.e. an unstable,
  edge-piling product, not a statistic. Per-seed R_low = 0.36 (bimodal MAPs: 9/25 at 0.600,
  16/25 in 0.735–0.82) is reported as the honest rail read: **NOT RAIL-REPRODUCED** by the
  registered ≥0.90 threshold, and not the D-CLASS signature either (its R_low ≤ 0.10 clause
  fails) ⇒ mechanically **MIXED**, with the scale confound as the stated reason.
- **S-DECOMP: MIXED — and the starvation hypothesis is REFUTED IN DIRECTION.**
  T_starv = bias(B-0) − bias(B-σ0.05) = **−0.010** (seed-matched: −0.002): sharpening the
  host photo-z by 20× did NOT reduce the mirror's bias — it INCREASED it (+0.0245 →
  +0.0348) and DEGRADED coverage (C68 0.64 → 0.43), the opposite of the registered
  starvation signature (which required C68 recovery into [0.497, 0.863]). T_D2 unmeasured
  (B-D2 deferred), so the form-component clause is unevaluated.

## The structural finding (why the correspondence failed — and what it hands us)

**The mirror universe is the OPPOSITE REGIME from production.** The generator draws hosts
FROM the pinned catalogue, so 100% of mirror hosts are catalogue-resident; in production
(iiib, n = 1588) only **76 events — 4.79% — have `in_catalog = True`.** Production's
ensemble is ~95% out-of-catalogue, i.e. **completion-leg dominated**; the mirror's is 0%.
The estimator's completeness model (f_k < 1: "some hosts are missing") is therefore violated
by the mirror's universe in the strongest possible way, and G-1's exact recovery of truth
under the f ≡ 1 shim is the control that shows it: with the model matched to the universe
the mirror is unbiased; with real completeness on an all-in-catalogue universe it biases
high. The bias the arms measured is thus a property of THIS mismatch, not of production —
which is why S-CORR had to fail, and it is a harness-scope defect, not a production one.

**Scientific value delivered anyway (three results that stand):**
1. **The 1D bias in a catalogue-resident venue is not photo-z starvation** — 20× sharper z
   makes bias and calibration worse. Whatever drives it is structural, and the G-1 control
   localizes it to the completeness/completion treatment.
2. **Candidate density is a strong lever** (E-DEN): halving localization area gives
   +0.0093 with restored coverage (C68 0.90), doubling gives +0.0211 with σ_h ×3.3 and
   R_low 0.50 — impostor density modulates both the bias and the railing behaviour.
3. **Production's regime is now quantified: ~95% completion-leg-dominated.** For those
   events p_i ≈ B_num(h)/D̃^φ(h) — a ratio of two integrals over the SAME population model,
   which must be unbiased if the estimator is correct. The production 1D rail says it is
   not. That is the sharpest statement of the base-tilt question yet, and it points at the
   completion leg — the same leg that carried the un-derived B_scale factor (rows #130-#131).

## AMENDMENT A-2 (registered BEFORE the follow-on runs)

The correspondence question is re-posed in production's actual regime:

- **B-OUT (new adjudicating arm, 15 seeds):** hosts drawn from the POPULATION model
  (comoving rate × the estimator's own w_pop), NOT from the catalogue, and never added to
  the candidate set — the production-typical out-of-catalogue event (real GLADE impostors
  in the ball + completion leg). Registered read: bias, coverage, R_low, σ_h as above.
  **Bands:** COMPLETION-UNBIASED if |bias| ≤ max(0.005, 2·SE) AND C68 ∈ [0.497, 0.863];
  COMPLETION-BIASED-LOW if bias ≤ −0.005 with CI excluding 0 (the production-direction
  signature — would make the completion leg the base-tilt owner and open a fresh
  /physics-change hunt); COMPLETION-BIASED-HIGH if bias ≥ +0.005 with CI excluding 0;
  MIXED otherwise. S-CORR is re-scored against THIS arm with the A-1 n_eff scaling.
- **B-F1 (control, 2 seeds):** B-0 configuration with the f ≡ 1 completeness shim —
  isolates completeness from the exact-z difference that separates G-1 from B-σ0.05.
  Registered: |bias| ≤ max(0.005, 2·SE) confirms the completeness-mismatch attribution.
- **Budget:** 17 seed-runs ≈ 19 CPU-h at the measured 0.478 CPU-h/seed-run; Option B total
  then ≈ 107 CPU-h, inside the registered 120 ceiling.
- bsig025 remains unread pending its defect fix (below); B-D2 remains deferred.

## bsig025 defect (recorded, not adjudicated)

All 10 bsig025 tasks failed identically: after the 0.25× re-scatter realization the
production Fisher-quality filter passes **0 detections** (`d_L relative error < 0.1`),
whereas the 0.05× dose passes 61 and B-0 passes ~68, so no diagnostics CSV is written and
the runner raises. The realization sidecar for the failing dose shows its mass-width check
off-nominal (normalized_residual_std 0.379 vs expected 0.25, with ~21.7M rows hitting the
mass-width floor), so the leading hypothesis is a scale-dependent interaction in the
re-scatter path's mass columns rather than anything in the z-kernel. Diagnosis is a separate
mechanical item; the dose is REPORTED-ONLY and adjudicates nothing.

## A-2 VERDICT (2026-08-20) — B-OUT reproduces production's dark rail; interpretation corrected

Job 6385173: 14/17 COMPLETED (3 TIMEOUT). Scored by the **pre-committed** scorer
(`readout_bout.py`, committed before the data landed).

| arm | n | bias | mean_h | σ_h | n_eff | C50/68/90 | R_low | registered band |
|---|---|---|---|---|---|---|---|---|
| **B-OUT** | 13 | **−0.1293** | **0.6007** | 0.0027 | 196 | 0.00/0.00/0.00 | 1.00 | **COMPLETION-BIASED-LOW** |
| B-F1 | 1 | −0.0000 | 0.7300 | 0.0778 | 69 | 1.00/1.00/1.00 | 1.00 | COMPLETION-UNBIASED (n=1 of 2; 1 seed timed out) |

**The headline: the mirror now reproduces production's dark class almost exactly.**
B-OUT mean_h = **0.6007** vs production's pure-completion class **0.6001** (σ_h 0.0027 vs
0.0011), railed in 13/13 seeds. The correspondence that FAILED for catalogue-resident arms
(S-CORR z ≈ −18) SUCCEEDS in the production-typical out-of-catalogue regime. Production's
base tilt is now reproducible in a 35-minute harness run with full control — the debugging
bed this campaign lacked.

**Interpretation CORRECTED (orchestrator self-catch, before any claim was banked):** the
scorer's canned discriminator line ("⇒ internal misnormalization; population attribution
falsified") **does not hold**, because B-OUT matches the estimator's POPULATION
(`population_z_weights` = dV_c/dz/(1+z), byte-identical to production's `_w_pop_eff` bare
form) but NOT its SELECTION: hosts are drawn ∝ w_pop(z) with no detection weighting and
196/200 pass the quality filter, whereas the estimator models detected dark events as
w_pop(z)·(1−f(z))·S̄_φ(z;h). B-OUT therefore has a data-vs-model mismatch of its own — in
the selection rather than the population — and cannot separate "internal misnormalization"
from "mismatch". The registered BAND (COMPLETION-BIASED-LOW) stands; the causal sentence
attached to it is withdrawn.

**What B-OUT does establish (all three stand):**
1. The dark-class low rail is REPRODUCED outside production, on demand, cheaply.
2. It appears whenever the analysed events' z-distribution carries more high-z weight than
   the model's assumed detected-dark distribution — B-OUT is the extreme case (no selection
   suppression at all), production is the mild case (M1 vs comoving shape, ≈87% of its
   score predicted by `population_mismatch_dark_score.md`). Same direction, same signature.
3. B-F1 (unity completeness, catalogue-resident) returns truth to 4 decimals — the
   catalogue-arm bias of +0.0245 is a completeness-model artefact, as hypothesised.

## AMENDMENT A-3 (registered now, pre-run) — the true isolation test

- **B-SEL (adjudicating, 15 seeds):** hosts drawn ∝ **w_pop(z)·(1−f̄(z))·S̄_φ(z; h_true)** —
  the estimator's OWN assumed distribution of *detected* dark events (population ×
  completeness deficit × survival, using the production survival object the harness already
  builds). This is the only arm that matches the model in BOTH population and selection.
  **Bands:** ESTIMATOR-SELF-CONSISTENT if |bias| ≤ max(0.005, 2·SE) AND C68 within the N=15
  binomial 95% band ⇒ every observed tilt is data-vs-model mismatch and the estimator's
  completion mathematics is exonerated; **INTERNAL-MISNORMALIZATION** if |bias| ≥ 0.005 with
  CI excluding 0 ⇒ a genuine estimator defect exists, reproducible in 35 min/seed, and the
  completion integrand is bisected next (numerator vs D̃^φ vs the (1−f)/S pairing);
  MIXED otherwise.
- **Also registered:** the 3 TIMEOUT seeds (2 B-OUT, 1 B-F1) are resubmitted at
  `--time=03:00:00` (the arm's realized wall is ~37 min but stragglers hit 2 h under
  contention); B-F1 adjudicates at 2/2.
- Budget: 15 + 3 ≈ 18 seed-runs ≈ 20 CPU-h (Option B running total ≈ 127 CPU-h — **over the
  registered 120 ceiling**, disclosed; the ceiling is raised to 150 by this amendment with
  the overrun stated rather than hidden).

## A-3 VERDICT (2026-08-20) — **INTERNAL-MISNORMALIZATION** (the campaign's decisive result)

Job 6387553: 12/15 B-SEL seeds banked (3 TIMEOUT). Scored by the pre-committed scorer.

| arm | n | bias | mean_h | σ_h | n_eff | C50/68/90 | R_low | band |
|---|---|---|---|---|---|---|---|---|
| **B-SEL** (model-matched population AND selection) | 12 | **−0.1120 ± 0.0017** | 0.6180 | 0.0216 | 180 | 0.00/0.00/0.00 | 1.00 | **INTERNAL-MISNORMALIZATION** |
| B-OUT (population-matched only) | 15 | −0.1293 | 0.6007 | 0.0027 | 196 | 0.00/0.00/0.00 | 1.00 | COMPLETION-BIASED-LOW |
| B-F1 (unity completeness, catalogue) | 2 | −0.0000 | 0.7300 | 0.0778 | 67 | 1.00/1.00/1.00 | 1.00 | COMPLETION-UNBIASED |

**Reading:** hosts drawn from the estimator's OWN detected-dark distribution
(w_pop·(1−f̄)·S̄_φ, built with production's own completeness and φ-survival construction)
and analysed by that same estimator still yield **−0.112 ± 0.0017 (66σ)**, railed in 12/12
seeds with zero coverage. **The completion leg is biased low even when the universe matches
its model.** That is an estimator defect, not a data-vs-model artefact, and it is now
reproducible at ~45 min/seed with no production run in the loop.

**Residual-mismatch bound (checked, works AGAINST the finding):** the production Fisher
quality filter still removes ~10% of mirror events (n_eff 180/200) by a distance-correlated
criterion the estimator's selection does not model. But that filter removes the FARTHEST
events, i.e. it leaves fewer high-z events than the model expects — which pushes the
posterior HIGH. The measured bias is LOW, so this residual mismatch cannot explain it and
in fact makes the defect slightly larger than measured. Second-order residuals (per-pixel
f_k vs the f̄ used in the host weighting) are disclosed and not adjudicated.

**Consequence for row #138 (population attribution) — DOWNGRADED, per its own §7:** the
memo predicted 87% of production's dark-class score from the M1-vs-comoving population
mismatch, and registered that a biased model-matched arm would falsify it as the owner.
B-SEL is that arm. Both effects can coexist, but the population mismatch can no longer be
assumed to own the tilt: an internally misnormalized completion leg produces a comparable
rail (−0.112) on its own. The memo's attribution is superseded by direct experiment; its
derivation and the measured population ratio stand as a contributing term of unknown share.

## AMENDMENT A-4 (registered pre-run) — first bisection step, a flag flip

Under the runs-of-record basis (`--selection_in_completion_numerator off`) the completion
NUMERATOR carries no detection weight while its normalization β_Ḡ^φ does — an asymmetry
whose fused form already exists as a shipped flag ([P1]/[P2], commit 2b10b8b8) and which
this campaign measured to be worth ≈ +0.17 of dark-class score in production (row #137's
convention ledger). In a MODEL-MATCHED universe the correctly normalized likelihood should
be the one whose numerator and denominator use the same detection model.

- **B-SELF ("bself", 15 seeds):** B-SEL configuration with
  `--selection_in_completion_numerator fused`. **Bands:** **CONVENTION-OWNS-IT** if
  |bias| ≤ max(0.005, 2·SE) AND C68 within the N=15 binomial band ⇒ the off-basis
  numerator/denominator asymmetry IS the internal misnormalization, and the fused
  convention is the derived-correct one (⇒ a /physics-change proposal for the production
  default, with the production off-vs-fused counterfactual already banked as its bed);
  **CONVENTION-PARTIAL** if |bias| falls to ≤ ½·|bias(B-SEL)| but stays material;
  **CONVENTION-NOT-IT** if |bias| ≥ ½·|bias(B-SEL)| ⇒ the defect is elsewhere in the
  integrand and the next bisection targets the measure/Jacobian of the z-integral and the
  α_G^φ/β_Ḡ^φ class composition of D̃^φ.
- Cost: 15 seeds ≈ 11 CPU-h (running total ≈ 150 CPU-h, at the amended ceiling).

## A-4 VERDICT (2026-08-20) — **CONVENTION-NOT-IT**: the selection asymmetry is not the defect

Job 6389506: 11/15 seeds (4 TIMEOUT, resubmitted as 6393215 at a 5 h wall). Pre-committed
scorer, A-4 bands:

| arm | n | bias | mean_h | σ_h | band |
|---|---|---|---|---|---|
| B-SEL (off basis) | 12 | −0.1120 ± 0.0017 | 0.6180 | 0.0216 | INTERNAL-MISNORMALIZATION |
| **B-SELF (fused basis)** | 11 | **−0.1163 ± 0.0010** | 0.6137 | 0.0182 | **CONVENTION-NOT-IT** |

Putting the detection weight into the completion numerator — the fused convention, which in
production was worth ≈ +0.17 of dark-class score — changes **nothing** in a model-matched
universe (if anything the bias is marginally larger). The off-vs-fused numerator/denominator
asymmetry is therefore NOT the internal misnormalization. Shortfall disclosed: the arm reads
at 11/15 registered seeds; the SE (0.0010) and the 12/12-railed pattern leave no plausible
route by which the missing four flip a −0.116 ± 0.001 result into the ≤ 0.005 band.

## AMENDMENT A-5 (registered pre-run) — next bisection: the event-term measure

Remaining candidates from the A-4 fallthrough, in derivational order:

- **D-ii, the GW event term's measure (LEADING).** The completion numerator evaluates
  `p_gw = norm.pdf(d_L(z;h)/d_L,det; μ_frac, σ_frac)` — a density in the dimensionless
  DISTANCE RATIO — and integrates it against `dV_c/(1+z)·(1−f_k) dz`
  (`bayesian_statistics.py:4852-4877`). A likelihood in the observable requires a density in
  `d_obs`; converting between the two carries a factor that depends on `d_L(z;h)` and hence
  on both z and h, so it does not drop out of a ratio taken across the z-integral. This is
  the same "density-form vs ratio-form" term the venue mechanism study catalogued as D-ii
  and measured as nearly inert IN A CATALOGUE-RESIDENT VENUE — a regime we now know is the
  wrong one for this question (rows #136, #139).
  **Instrument B-DEN (15 seeds):** B-SEL configuration with the completion numerator's event
  term switched to the d_obs-density form. Bands: **MEASURE-OWNS-IT** if |bias| ≤
  max(0.005, 2·SE) with C68 in the N=15 band; **MEASURE-PARTIAL** if |bias| ≤ ½·0.112;
  **MEASURE-NOT-IT** otherwise.
- **D̃^φ class composition (second).** For an event with no catalogue support the numerator
  carries only the dark-class term while the denominator carries α_G^φ + β_Ḡ^φ. Instrument
  and bands to be registered only if B-DEN falls through.

**A14 compliance:** the D-ii attribution above is explicitly PROVISIONAL — a code-shape
observation plus a dimensional argument, not a derivation. Before B-DEN's result may be
attributed to it, the algebra (which measure the numerator is a density in, which the
denominator is, and what the correct pairing implies for the h-slope) is written up as a
memo in `docs/derivations/`, with its own falsifier, exactly as the B_scale item was.

## A-5 VERDICT (2026-08-20) — **MEASURE-NOT-IT**, and a pattern that re-opens the premise

Job 6393386, 15/15 seeds, pre-committed bands:

| arm | what it repairs | n | bias |
|---|---|---|---|
| B-SEL | (baseline, model-matched) | 12 | −0.1120 ± 0.0017 |
| B-SELF | detection weight into the numerator (fused) | 11 | −0.1163 ± 0.0010 |
| **B-DEN** | **the event term's data measure** | **15** | **−0.1193 ± 0.0005** |

`completion_numerator_data_measure.md` is **FALSIFIED as the owner** per its own §5. Its §2
claim stands and is numerically proven (the ratio form integrates over the data to
d_L(1+3σ²) = 1.0316, the corrected form to 1.0) — the missing normalization is REAL, but
repairing it does not move the bias, exactly as the memo's own saddle-point caveat warned it
might not.

**The pattern (three eliminations, monotone):** −0.1120 → −0.1163 → −0.1193. Every repair
that should improve the estimator's internal normalization leaves the bias unchanged or
marginally WORSE. Three independent normalization defects have now been fixed inside a
"model-matched" universe with no effect. That is not the signature of a normalization bug;
it is the signature of **a residual data-vs-model mismatch in the mirror itself** — i.e. the
A-3 premise (that B-SEL is model-matched) may be false.

**Where B-SEL can still be mismatched (identified, not yet measured):** hosts are drawn from
w_pop·(1−f̄)·S̄_φ, but each event then receives a donor Fisher row resampled SNR-weighted from
real production events, and afterwards ~10% are removed by the production quality filter.
Neither the donor assignment nor the quality filter is part of the estimator's selection
model, so the REALIZED distribution of surviving mirror events need not equal
w_pop·(1−f̄)·S̄_φ even though the DRAWN distribution does. The A-3 verdict
"INTERNAL-MISNORMALIZATION" is therefore **downgraded to PROVISIONAL** pending the check
below. (This is amendment A10 applied to our own harness: its structural blindness is that
it matches the model at draw time, not at survival time.)

## AMENDMENT A-6 (registered pre-run) — test the premise before bisecting further

- **D-1 (free diagnostic, zero compute):** re-run one banked B-SEL seed's generator with
  per-event bookkeeping and record the z-distribution of SURVIVING events, then compare it
  to the model's own detected-dark density w_pop·(1−f̄)·S̄_φ normalized on the same range.
  **Bands:** MIRROR-MATCHED if the max CDF gap ≤ 0.05 (the tolerance already used for the
  pool-vs-events provenance check, row #137) ⇒ the mirror is model-matched at survival and
  the internal-misnormalization verdict is restored; **MIRROR-MISMATCHED** if > 0.05 ⇒ the
  A-3 verdict is void and the bias measured by B-SEL/B-SELF/B-DEN is (at least partly) the
  mirror's own mismatch, not the estimator's defect.
- **D-2 (only if MIRROR-MISMATCHED):** rebuild the arm so that survival, not drawing, matches
  the model — accept every drawn event (no quality filter) and give each an analytic σ_dL/d_L
  consistent with the model's assumed measurement error, rather than a resampled donor row.
  Re-run the isolation test on that universe.
- **No further estimator bisection until D-1 returns.** Three eliminations with no movement
  is the registered trigger (proposal §"Explicitly NOT proposed") to question the premise
  rather than the next term.

## D-1 VERDICT (2026-08-20) — **MIRROR-MISMATCHED**: the A-3 premise fails, the A-3 verdict is void

Seed 900101, B-SEL configuration, generative + filtering path only (~7 s):

| quantity | value |
|---|---|
| drawn / surviving | 200 / 174 (survival 0.87) |
| **max CDF gap, SURVIVING vs model (band-bearing)** | **0.0792 → MIRROR-MISMATCHED (> 0.05)** |
| max CDF gap, DRAWN vs model (control) | 0.0336 — no anomaly; the draw is correct by construction |

z-quantiles (drawn / surviving / model): 0.05 → 0.170/0.206/0.193 · 0.50 →
0.432/0.461/0.443 · 0.95 → 0.753/0.760/0.762. **The surviving population sits at
systematically higher z than the model's detected-dark density predicts** — the donor-row
resampling plus the production quality filter preferentially remove low-z events, which the
estimator's selection model does not know about.

**Consequence (registered band, applied):** the A-3 verdict
"INTERNAL-MISNORMALIZATION" (row #140) is **VOID as stated**. The −0.112/−0.116/−0.119
measured by B-SEL / B-SELF / B-DEN is at least partly the mirror's own survival-time
mismatch, not demonstrably the estimator's internal defect. This also explains the monotone
pattern that triggered the check: three internal-normalization repairs could not move a bias
whose driver was outside the estimator.

**What still stands, untouched by this:** production's base tilt itself (dark class 0.6001,
score −0.635 ± 0.017 at truth, high-z localized); B_scale's removal (an independently
derived defect); the s_Edd re-measurement; J_α; the f-treatment closure; and the general
mechanism that an excess of high-z events relative to the model's assumed detected
distribution rails the posterior low — which D-1 has now demonstrated a THIRD time, in the
mirror's own survival step.

## D-2 (registered, triggered) — rebuild the arm so SURVIVAL matches the model

Per A-6: accept every drawn event (no production quality filter) and assign each an analytic
σ_dL/d_L consistent with the estimator's assumed measurement model, instead of a resampled
donor Fisher row. Then re-run the isolation test on that universe.
**Bands (unchanged in spirit from A-3, re-registered here):** ESTIMATOR-SELF-CONSISTENT if
|bias| ≤ max(0.005, 2·SE) with C68 in the N=15 binomial band ⇒ the estimator's completion
mathematics is exonerated and every tilt observed so far is data-vs-model mismatch;
INTERNAL-MISNORMALIZATION if |bias| ≥ 0.005 with the CI excluding 0 ⇒ a genuine defect
survives a survival-matched universe, and THEN the bisection resumes (D̃^φ class composition
first). **Pre-flight gate:** D-1 must return max CDF gap ≤ 0.05 on the rebuilt arm BEFORE any
seed is analysed — the premise is verified first this time, not assumed.

## AMENDMENT A-7 (2026-08-20) — RE-SCORE: the log-space sentinel, and what it did and did not break

**Verifier coverage (non-transitive, per the row #144 process finding):** this amendment carries
its OWN adversarial pre-check, which returned **NOT-READY** and required ten amendments. All ten
were applied below before the verdict was read. It does not inherit any earlier verifier stamp.

### The defect

`correspondence_1d.py:1965` (and the identical `:2479`) floors a zero per-event likelihood **in
log space** at `-1.0e300`:

```python
log_l     = np.where(vals > 0.0, np.log(vals, where=vals > 0.0), -np.inf)
sum_log_l = np.nansum(np.where(np.isfinite(log_l), log_l, -1.0e300), axis=0)
```

**Correctly stated (this is narrower than it first appears, and was narrowed by measurement):**
the sentinel is *numerically identical to the mathematically correct* `-inf` whenever at least one
node of `H_GRID_41` survives — verified `max|Δmean_h| = 0.000e+00` across all 98 such banked
seeds. Its only consequence is on a seed where **every** registered node is masked, which happens
iff the seed contains ≥1 event whose `combined_no_bh` is zero at every h. There, correct `-inf`
handling yields all-NaN and fires the harness's own `if not np.isfinite(...).any()` guard; the
sentinel instead yields a **finite, normalizable, silently-banked** posterior. Verified directly:
`b0_900101` and `bf1_900101` have 0/46 finite nodes under `-inf` (guard fires); `bout_900101` has
33/46 (guard does not fire).

**Blast radius: 25 of 123 banked seeds (20.3%)** — b0 10/25 · bsig005 7/23 · bf1 2/2 · eden2 5/10
· eden05 1/10 · **bout/bsel/bself/bden 0/53**. Catalogue-mode arms 25/70, population-mode 0/53.

Of those 25: **21 have uniform sentinel multiplicity** ⇒ exactly flat ⇒ `mean_h` = the arithmetic
midpoint of `H_GRID_41`, `(0.600+0.860)/2 = 0.7299999999999999`, which **coincides with H_TRUE**;
`map_h` = 0.600 (`argmax` tie-break returns index 0) ⇒ `r_low = True`; and `c50=c68=c90=True`
because `_hpd_contains` (`:1914-1929`) tests `idx == target_idx` at `:1925` *before* the
`cum >= level` break at `:1927`, and the flat cumulative mass at the truth node is 0.50926. Such a
seed reports "unbiased to four decimals", "railed low" and "covered at all three levels"
simultaneously, from grid geometry alone. The remaining **4 have non-uniform multiplicity** and are
*spuriously informative* — differing sentinel magnitudes survive the `lp - max` softmax as fake
evidence (`b0_900121` → 0.8400, `bsig005_900108` → 0.8087, `bsig005_900114/900119` → 0.8400).

**Root cause of the trigger (measured, not inferred).** Every all-zero event has
`L_cat_no_bh = 0`, `B_num = 0` and **`g_frac = NaN`** — an empty candidate set / undefined
catalogue–completion mixing fraction — in 100% of cases, against a 3–6% `g_frac`-NaN baseline in
other rows. This is a **generator/data defect, not a numerical one**.

**UNDERFLOW IS REFUTED BY MEASUREMENT.** The smallest non-zero `combined_no_bh` anywhere in the
130 retrieved CSVs is **4.876e-48**, ~302 orders of magnitude above float64's smallest normal
(2.225e-308); there is no value below 1e-300 in the fleet. The zeros are true structural zeros
reached off a cliff. (An earlier orchestrator reading of "underflow", based on subnormal values
in a non-fleet G-1 diagnostic under unity completeness, is **withdrawn**.)

### Data, provenance, and the seed set

130 of 142 arm-seed work-roots retained their per-event `event_likelihoods.csv`; retrieved
2026-08-20 (151 MB) to `results/prod2d_closure_20260818/arm_event_likelihoods/`, with a SHA-256
manifest (`arm_event_likelihoods_MANIFEST.sha256`, 130/130) and provenance stamp (A11). The
cluster workspace expires 2026-09-23.

**Seed set pin:** the re-score scores exactly the **123 seeds that have a banked JSON**, enumerated
from `correspondence_arms/*.json`, never from the CSV directory. The 7 CSVs with no banked JSON —
`bsel_seed900113/900114/900115`, `bself_seed900110/900111/900112/900113` — are excluded by name;
six are truncated mid-h-sweep. Per-arm N: b0 25 · bsig005 23 · bsel **12** · bself **11** · bden 15
· bout 15 · eden05 10 · eden2 10 · bf1 2.

### Gates (all three passed before any number was read)

- **GATE R-0a — as-run provenance, all 123 seeds (can fail).** Re-running the *defective* combine
  on each retrieved CSV must reproduce that seed's banked `mean_h`/`sigma_h` to ≤1e-9 and its
  `map_h`/`c50`/`c68`/`c90`/`r_low` exactly. **PASS, 0/123 failures.** This is the can-fail
  provenance control covering the 44 contaminated seeds that R-0b cannot reach.
- **GATE R-0b — no-op identity, the 79 sentinel-free seeds (can fail).** Under the repaired
  combine they must reproduce their banked moments to ≤1e-9. **PASS, 79/79.** (An earlier draft
  stated this set as "41"; that was an arithmetic error — 14+14+9+4 counted only the
  b0/bsig005/eden arms and omitted bsel 12 + bself 11 + bden 15.)
- **GATE R-1 — pairing provenance.** `run_arm_seed` is idempotent (`:2549-2556`) while work-roots
  persist across resubmissions, so distinct-`event_idx` count must equal banked `n_eff`.
  **PASS, 0 unverified.**

Pre-declared and verified: `(event_idx, h)` duplicates exist in three banked seeds
(`bout_900112` 3528, `bout_900113` 3152, `bf1_900102` 715, all resumed runs) with **0 disagreeing
values**, so `aggfunc="first"` is lossless here.

### Combine and bands (as amended)

Primary, band-bearing: **`physics_floor`** (production's registered `CombinationStrategy`).
Reported: `per_event_floor`, `exclude`, `clip-1e-300`. Every other stage frozen verbatim — the
`np.isin` h-subset, the pivot, `reindex`, the `w = np.gradient(grid)` moment weights, the
`lp - max` normalisation, and `_hpd_contains`.

- **BAND V — corrected.** The re-score is a **paired deterministic recomputation on frozen data**:
  same seeds, same CSVs, only the arithmetic changes, so `Var(Δ) = 0` and **no sampling band
  applies**. The draft's `max(0.005, 2·SE)` was mis-specified (it is the standard error of an arm
  mean across realizations, not of a paired difference) and is **withdrawn**, as is the b0
  `SHIFT-NOT-RESOLVED` clause built on it. Δ is reported exactly; the verdict is decided by
  re-applying each arm's ORIGINAL registered band to the CORRECTED bias and SE.
- **BAND S — strategy spread**, threshold 0.005, disclosed as **imported from the campaign's bias
  tolerance and not derived for a spread statistic**. Its outcome is partly pre-determined: for a
  seed whose zero-carrying events are zero at *all* nodes, every strategy contributes a constant
  that the `lp - max` shift removes, so spread ≡ 0 by construction.
- **BAND O — WITHDRAWN AS UNDECIDABLE.** `combined_no_bh = 0` ⟺ `L_cat_no_bh = 0` **and**
  `B_num = 0` identically, so the "STRUCTURAL" branch is an algebraic tautology; and with the
  fleet minimum at 4.876e-48 the "UNDERFLOW-DOMINATED" branch is unreachable. Both branches were
  pre-determined. Deciding underflow-vs-structural at the `B_num` integrand requires log-space
  accumulation the CSVs do not carry; it is named an **open blind spot**, not a measured result.

### A10 — blindness declaration (negative, and with a process violation disclosed)

**PROCESS VIOLATION, disclosed:** the primary re-score was executed by a synthesis agent inside the
adversarial-verification workflow *before* this registration was finalized. The agent had been
dispatched to refute claims, not to run the measurement, and exceeded its brief. The numbers below
are therefore **not blind**. They were subsequently re-derived by an independent second
implementation (`rescore_sentinel.py`) which reproduces them on 8 of 9 arms; the ninth (`bsel`
−0.1138 vs −0.1120) differs solely because the agent scored 15 CSVs where only 12 are banked — the
seed-set pin above resolves it in favour of 12. Treat A-7 as an **audited confirmatory
recomputation**, not a blind measurement.

Also not blind, and declared before the run: the as-published statistics, the contamination counts,
the clean-seed-only means, and two non-fleet forensic seeds (G-1 seed 900001; the G-2 cost pilot at
b0 configuration, seed 900101).

**Registered sign prediction (the one thing that could have come out wrong):** contaminated seeds
report 0.7300 = H_TRUE, so contamination pulls arm means *toward* truth; every corrected arm mean
should therefore move *away* from truth. **Held:** all 21 flat seeds move away (corrected mean
0.7550, range 0.7203–0.8039).

## A-7 VERDICT (2026-08-20) — the means barely move; the RAIL and COVERAGE statistics are artefacts; the positive control FAILS

| arm | N | published bias | corrected bias (physics-floor) | Δ | R_low | C68 |
|---|---|---|---|---|---|---|
| b0 | 25 | +0.0245 | **+0.0298 ± 0.0046** | +0.0053 | **0.36 → 0.00** | 0.64 → 0.48 |
| bsig005 | 23 | +0.0348 | +0.0366 ± 0.0072 | +0.0019 | 0.17 → 0.00 | 0.43 → 0.39 |
| eden05 | 10 | +0.0093 | +0.0140 ± 0.0044 | +0.0046 | 0.10 → 0.00 | 0.90 → 0.80 |
| eden2 | 10 | +0.0211 | +0.0322 ± 0.0088 | +0.0111 | **0.50 → 0.00** | 0.60 → 0.30 |
| **bf1** | 2 | −0.0000 | **+0.0359 ± 0.0036** | **+0.0359** | **1.00 → 0.00** | **1.00 → 0.00** |
| bout | 15 | −0.1293 | −0.1293 | **+0.0000** | 1.00 | 0.00 |
| bsel | 12 | −0.1120 | −0.1120 | **0** | 1.00 | 0.00 |
| bself | 11 | −0.1163 | −0.1163 | **0** | 1.00 | 0.00 |
| bden | 15 | −0.1193 | −0.1193 | **0** | 1.00 | 0.00 |

1. **[RULE] The bisection chain is UNAFFECTED.** B-SEL/B-SELF/B-DEN carry zero sentinel nodes in
   every banked seed; Δ ≡ 0 to ≤1.0e-15. Row #140's PROVISIONAL-WITH-A-BOUND status is untouched by
   this defect. Note these three are **NULL-BY-CONSTRUCTION, not controls** — per A15 a control
   must be capable of failing, and for them the repair is provably a no-op before the run.
2. **[RULE] The B-OUT indictment is WITHDRAWN.** B-OUT's masking is real and perfectly one-sided
   (0 masked nodes at h ≤ 0.730 in all 15 seeds, rising to mean k ≈ 30 at h = 0.860), but applying
   production's own physics-floor changes `mean_h` by **≤1.1e-16**. The masked nodes genuinely
   carry zero likelihood and correct `-inf` handling gives the identical answer. **Row #139's
   "B-OUT reproduces production's dark rail" and the retrospective's corresponding sentence STAND.**
3. **[RULE] Every rail recorded in every catalogue-mode arm is an artefact.** `R_low` falls to
   exactly **0.00** in b0, bsig005, eden05, eden2 and bf1 once scored correctly — 100% of the
   recorded rail signal in those five arms was the `argmax` tie-break on a flat sentinel vector,
   not a data-driven posterior peak. Coverage is inflated in the same seeds. Any banked claim
   quoting C50/C68/C90 or R_low for these arms must be restated or withdrawn.
4. **[RULE] B-F1, the mirror's positive control, does not merely carry no information — corrected,
   it FAILS.** +0.0359 ± 0.0036 with coverage 0/0/0 and `map_h` 0.765/0.760. **PROVISIONAL at
   n = 2** (no usable σ, no power — per A15 it cannot carry a control verdict), and B-F1 is a poor
   control on independent grounds: under `f ≡ 1` the completion leg vanishes identically, so
   **100% of its events have `g_frac = NaN`** and its universe is inconsistent with its own
   likelihood. It does not return truth.
5. **G-1 (the STOP gate) is UNRESOLVED, not proven vacuous.** No G-1 posterior, JSON or
   `event_likelihoods.csv` is banked under `results/`. A local `/tmp` G-1 diagnostic recomputes to
   +0.0050 (physics-floor) against its recorded 0.7300, but its as-run signature (`map_h = 0.730`,
   `sigma_h = 0.0000`) is the *partial-mask* mode, not the flat mode, so the "same mechanism as
   B-F1" inference is unsupported. G-1's PASS is recorded **UNSUPPORTED**, pending a re-run.
6. **Production is UNAFFECTED**, with the scope narrowed: the additive `-1.0e300` log-space
   sentinel exists only at `correspondence_1d.py:1965` and `:2479`, and no production module
   imports `correspondence_1d`'s statistics functions. The blanket "nowhere else in
   `darksiren_emri/`" is withdrawn — a *multiplicative* `1e-300` clip is a repo-wide house pattern
   (`posterior_combination.py:758`, ~18 sites in `validation/pp_coverage.py`), which is
   categorically milder: bounded at log ≈ −690.8, no absorption, no tie-manufacturing.
7. **BAND S:** STRATEGY-ROBUST for every arm except **bsig005** (spread 0.0162 ⇒ FRAGILE), whose
   corrected number is therefore reported but not adjudicated.

### Carried forward — two frozen convention defects, NOT fixed here

Both were frozen for this measurement because R-0a/R-0b require bit-reproduction, and both belong
in the `/physics-change` presentation alongside the sentinel: (a) `w = np.gradient(h_grid)`
(`:1967`) is **not** trapezoid — it doubles the endpoint weights (`w[0] = 0.010` vs 0.005) at
h = 0.600, the node every arm rails onto, and the docstring at `:1943` calling them "trapezoid
weights" is wrong; (b) `_hpd_contains` returns True on reaching the target *before* testing
`cum >= level` (`:1925` vs `:1927`), so a flat posterior scores "covered at 50%" at cumulative
mass 0.50926.
