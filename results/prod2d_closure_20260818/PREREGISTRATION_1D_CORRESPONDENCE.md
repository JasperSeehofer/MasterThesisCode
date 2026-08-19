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
