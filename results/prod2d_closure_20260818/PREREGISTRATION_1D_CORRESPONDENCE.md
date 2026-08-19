# PRE-REGISTRATION — Option B: production 1D correspondence measurement

**Date:** 2026-08-19 · **Status:** DRAFT v1 — awaiting verifier pre-check (shared pass with
`PREREGISTRATION_TILT_BATTERY.md`). Authorized: row #132 [DO] ("all approved"). **Validation
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
  known truth h_true = 0.73; per-event GW σ_dL/dL resampled from the seed61000 CRB
  empirical distribution (SNR-weighted); completeness = the real
  `GladeCatalogCompleteness`/`PixelCompleteness` object (importable, recon-verified) —
  NOT a synthetic ball, NOT a LOO construction (D-iii is architecturally production-
  inapplicable; the real-catalogue draw sidesteps it and its open derivation becomes an
  exploratory arm, E-DEN).
- **D-C Scale:** n_events = 200 per realization (cost control; the tilt class is a broad
  ensemble effect per T0 jackknife), N = 25 seeds per arm, paired seeds across arms.
- **D-D Cost pilot (registered, STOP-gated):** 2 seeds of B-0 first; if realized CPU-h/seed
  > 2× the 0.969 anchor ⇒ STOP and re-scope (ceiling 60 CPU-h total; the 2D analog's 406
  CPU-h overrun is the cautionary anchor — no 2D arms in this prereg).

## 2. Arms (all 1D channel; paired seeds)

- **B-0 (production-mapped):** full production form, σ_z = GLADE empirical (the observed
  photo-z errors of the pinned catalogue). The correspondence arm.
- **B-σ (starvation ladder):** σ_z scaled to {0.25×, 0.05×} of empirical (host z_obs
  re-scattered; same seeds) — the information-content dose response.
- **B-D2 (density-form toggle):** the D-ii form defect arm — the GW event term evaluated in
  d_obs-density form vs production's ratio-pdf form (the one venue-measured-inert term
  whose production magnitude is unknown; implemented harness-side around the wholesale
  call, production code untouched).
- **E-DEN (exploratory, REPORTED-ONLY):** candidate-multiplicity response — localization
  area scaled ×{0.5, 2} (same universe) — the real-catalogue impostor-density fingerprint
  (the D-iii open derivation's measurement side).

## 3. Registered statistics and bands

Per arm: mean_h bias ± SE over seeds, coverage C50/C68/C90, rail fraction R_low
(P(MAP ≤ 0.605), the DS-6 statistic), posterior width distribution.

- **S-CORR (the correspondence read, fixing the P7-8 one-draw question):** production's
  banked 1D means (0.6010/0.6020) are placed in B-0's realization distribution of mean_h.
  **CORRESPONDENCE-PASS** if within [q05, q95]; **FAIL** outside. This replaces
  single-number matching with ensemble placement — production is one draw; the mirror
  provides the ensemble it should be a draw from.
- **S-RAIL (closing DS-6, bands fixed NOW):** RAIL-REPRODUCED if R_low(B-0) ≥ 0.90;
  RAIL-NOT-REPRODUCED if ≤ 0.10; the calgate alternative signature (uniform positive MAP
  bias + coverage collapse) is a REGISTERED NAMED OUTCOME (D-CLASS-REPRODUCED) rather than
  a MIXED bucket — if B-0 shows it, the verdict is that the defect-class phenomenon, not
  the rail, is the transferable object (DS-6's ambiguity resolved either way).
- **S-DECOMP:** T_starv = bias(B-0) − bias(B-σ, 0.05×) (paired); T_D2 = bias(B-0) −
  bias(B-D2) (paired). Registered reading: STARVATION-DOMINATED if |T_starv| ≥ 3·|T_D2|
  AND B-σ(0.05×) coverage recovers to [0.594, 0.766] at C68; FORM-COMPONENT-PRESENT if
  |T_D2| ≥ max(0.005, 2·SE_paired); MIXED else. Every branch returns to the author as a
  [RULE]; a FORM-COMPONENT-PRESENT finding opens a fresh /physics-change proposal (not
  pre-approved).
- **Materiality yardstick:** ⅓·σ_h of the production 1D posterior is degenerate (railed);
  the registered yardstick is the 2D one (0.008) — a 1D form component below it cannot
  matter for the tilt ledger's 2D budget.

## 4. Gates

- **G-1 (mirror sanity):** B-0 with σ_z → 0 AND full completeness must recover truth
  (|bias| ≤ 2·SE) — the harness's own null; failure ⇒ STOP (mirror defect, nothing quoted).
- **G-2 (pilot):** §1 D-D cost gate.
- **G-3 (fidelity pin):** the wholesale `single_host_likelihood` calls must run with
  production-default flags (derived B_scale form, `catalogue_mass_overlap=production`,
  volume_deconv) — recorded in the harness config dump per run.

## 5. Execution & tiering

Harness: new `darksiren_emri/validation/correspondence_1d.py` (validation package;
venue_transfer.py patterns; full typing; CPU-only). Implementation: sonnet. Prereg +
interpretation: top-tier. Verifier: shared top-tier pass with the battery prereg.
Runs local or cluster CPU (≈ 25–50 CPU-h at anchor; ceiling 60). Readout: A7
comprehension-first, folded into the tilt-ledger budget with the battery results.

---

## VERDICT

*(append-only below this line after execution)*
