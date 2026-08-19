# PRE-REGISTRATION — Tilt-ledger battery: s_Edd re-measurement, J_α (F withdrawn by derivation)

**Date:** 2026-08-19 · **Status:** v2 — verifier pre-check applied (v1 NOT-READY; P1–P5
amendments verbatim below; instrument F WITHDRAWN per P1 — its ledger entry closes by
derivation at zero cost). Instrument code is physics-change-gated (§6 presentations) and
awaits the author's gate approval. Authorized as
a [DO] in rows #130–#132 ("all approved"). **Reads against the post-fix baselines of
record** (`PREREG_POSTFIX_BASELINE.md` VERDICT): 2D mean_h **0.6771 (iiib) / 0.6788
(joint_r1)**, σ_h 0.0239/0.0225, offsets −0.0529/−0.0512. No production change on any
branch: every instrument is a default-production CLI flag (rows #128/#131 pattern), one
flag active per cell, never combined.

**Purpose:** measure three documented tilt-ledger entries production-natively, replacing
one stale budget leg (s_Edd) and bounding two documented inconsistencies (f-treatment,
point-vs-kernel Σ⁴ᴰ), so the post-fix residual budget is built from measured, derivation-
statused legs. Seam recon of record: 2026-08-19 sonnet recon (line refs below at current
HEAD `6ec1c78d`).

## 1. Instruments (each: default = production, both venues, 41-h)

- **E — `--eddington_m {on,off}`** (default `on` = production). Under `off`, the 2D
  catalogue leg's `mu_gal` uses raw `host_M` instead of `eddington_shifted_host_mass`
  (`bayesian_statistics.py:585-619`); consumers pinned by recon: scalar :5629-5633 (gate
  `_use_volume_deconv and not _use_mass_trunc`) with numerator/denominator uses at
  :5685/:5721/:5748/:5788; batch :6182-6191 with :6222/:6257/:6270/:6290. The
  `_mass_trunc` branch never consumes the shift (untouched). Both kernels switched;
  parity test required.
  **Read R-E:** s_Edd,new per venue = mean_h(2D, baseline) − mean_h(2D, E-off) — the shift
  the treatment CAUSES at the current operating point (derived form, exact-quadrature
  path, seed61000, 41-pt). Replaces the stale −0.020 (audit finding 5; the stale comment
  at :5623-5625 is refreshed in the same commit, citing this measurement). Expectation
  (REPORTED-ONLY anchor): −0.002-class (G7row9_N5 post-D_g-fix). No adjudicating band —
  this is a measurement; its registered USE is the budget leg.
  **P3 clarification (registered):** the toggle is the single `_host_M_eff` assignment, so
  the numerator mass prior AND the per-host denominator D_g erf-sum switch TOGETHER (G2d
  counted-once preserved; all four downstream uses read the one variable); Σ⁴ᴰ never
  consumed the shift (grep-verified: :5630/:6185 are the only call sites) and is unchanged
  in both arms. Sign convention verified against audit finding 5 (R-E = mean(treated) −
  mean(untreated), same convention as the stale −0.020).
- **F — WITHDRAWN (P1, verifier derivation).** With the isotropic S̄_φ, a banded β_Ḡ^φ
  built from the same `CompletenessModel` equals the f̄ form IDENTICALLY (`f_bar` is the
  equal-area pixel mean of f_k, pixel_completeness.py:270-290; linearity) — v1's N-2 gate
  could not engage, and v1's consistency claim was wrong as stated: per-EVENT pixel f_k in
  the numerator vs the population sky-average f̄ in the denominator IS the A2-consistent
  pairing for an isotropically modelled population. The genuine residual (event-ensemble ×
  completeness sky covariance) is bounded by gate (ii-e) at ≲2e-4 — 40× below materiality.
  **Ledger disposition: the bscale-memo §4 f-treatment entry closes as
  BOUNDED-IMMATERIAL by derivation, zero cost.** A measured version would need a per-band
  S̄_φ,b table (new physics object, separate gate) and is not warranted at the bound.
- **J — `--sigma4d_mass_kernel {point,kernel}`** (default `point` = production). Under
  `kernel`, the `with_bh_mass=True` branch of `precompute_global_catalog_selection`
  (:2707-2731) replaces the point evaluation with the **registered kernel (P2):**
  p_det,g = E_{M ~ N(M_eff_g, σ_g²)}[S_4D(d_L_g, M(1+z_g))] via the existing erf-sum
  inner-M machinery (`_bh_mass_denominator_inner_m_integral` pattern), with M_eff_g the
  SAME Eddington-shifted mean production's D_g uses and σ_g = `host_M_error` (linear,
  M_sun, the catalogue `BH_MASS_ERROR` column). **NO `_mass_trunc` lognormal, NO R_eff
  inside the kernel** — w_g stays the point rate weight outside, so R_eff is counted once
  and cancels in r_Malm. **Σᶲ is untouched by derivation** (it contains no per-galaxy mass
  evaluation; fixb_pathA D3: r_Malm → r_Malm·J_α only; w_g stays point-evaluated —
  registered). This is the A2-consistent smearing matched to production's own D_g Gaussian
  kernel. Single precompute seam.
  **Read R-J:** ΔJ = mean_h(variant) − baseline (2D; 1D reported), plus the r_Malm(h)
  ratio table as a diagnostic. REPORTED-ONLY + materiality flag; documented adverse
  direction carried as context, not a band.

**Independence (recon-verified, P2-extended):** the two remaining instruments touch
disjoint functions and are disjoint from the `--catalogue_mass_overlap` and
`--completion_b_scale` seams. Soft entanglements disclosed: (a)
`catalogue_mass_overlap=inflated` freezes μ_eff computed WITH the Eddington shift; (b) J's
kernel mean M_eff_g also uses the Eddington shift that E toggles. Battery cells never
combine flags, so no arm is affected — noted for future combined use only.

## 2. Gates

- **N-0 (continuity, scored, STOP):** V0 cells (all defaults) at probe h ∈ {0.675, 0.700}
  per venue must reproduce the banked post-fix baseline `event_likelihoods.csv` rows to
  max rel diff ≤ 1e-10 (both channels).
- **N-1 (1D-channel discipline):** instrument E must leave `combined_no_bh` BIT-IDENTICAL
  (the shift lives in the with-BH path only). Instruments F and J legitimately move BOTH
  channels (β_Ḡ^φ and Σ⁴ᴰ feed both) — their 1D deltas are REPORTED, not gated.
- **N-2 (engagement, scored, STOP — never silently null):** E: ≥ 10% of
  catalogue-supported events change `combined_with_bh` by ≥ 1e-6 rel at the probes
  (computable from the banked per-event CSVs). J: max_h |r_Malm,kernel/r_Malm,point − 1|
  > 1e-4 — **banked source registered (P4):** the variant arms dump the per-h
  {β_G^φ, β_Ḡ^φ, Σᶲ, Σ⁴ᴰ, r_Malm} table to a JSON in the run dir (small addition to the
  instrument commit); the gate is scored from those files, not from log-line scraping.
  Failure ⇒ STOP (instrument not engaged).
- **Execution-completeness (P5, the 6364821 lesson):** a venue × instrument read requires
  41/41 tasks COMPLETED; partial fleets are banked but adjudicate nothing without a fresh
  author [RULE].
- **Metadata:** run_metadata cli_args diff vs the post-fix baseline referent shows only
  whitelisted keys ({git_commit, timestamp, working_directory, seed/random_seed, SLURM_*,
  eddington_m, sigma4d_mass_kernel}).
- Scalar↔batch parity pytest for E; table-level unit tests for F/J (banded reduces to
  isotropic under uniform completeness; kernel reduces to point as σ_lnM → 0 — limiting
  cases, pinned).

## 3. Execution

Cluster, evaluate.sbatch pattern, config of record (absolute_marginal, volume_deconv,
pdet_z_resolved, `--selection_in_completion_numerator off`, EVAL_SEED 777000): 2
instruments × 2 venues × 41 h + 4 V0 probes = **168 tasks ≈ 8.5 CPU-h** (P1: F withdrawn).
Precondition:
instrument [PHYSICS] commit pushed + cluster ff + preflight READY.

## 4. Registered use of the results (budget re-assembly, presented to the author)

The post-fix tilt budget per venue: Δ_v(new) = −0.0529/−0.0512. Ledger entries:
s_Edd,new (measured, R-E), ΔF and ΔJ (measured, with derivational status: F = consistency
repair candidate, J = D3-documented open term), plus the Option B base-tilt decomposition
(separate prereg). Materiality yardstick: ⅓·σ_h(new) ≈ 0.008. Each entry returns to the
author with its measured size and a derivation-status recommendation; any entry that is
BOTH material AND a derivable inconsistency becomes a /physics-change candidate — none is
pre-decided. P7-8 single-realization disclosure carried.

## 5. Tiering (stated)

Recon: sonnet (done). Prereg: top-tier (this doc). Verifier: ONE top-tier xhigh (shared
pass with the Option B prereg). Implementation: sonnet. Readout: sonnet compute + top-tier
interpretation. Top-tier this front: 2 (chair + verifier) ≤ cap.

## 6. Physics-change gate presentations (author approval before implementation)

**E:** Old: `_host_M_eff = eddington_shifted_host_mass(host_M, host_M_error)` under the
:5629 gate (Bishop/G2d shift, docs/derivations/G2d_host_mass_rate_prior.md). New: under
`off`, `_host_M_eff = host_M` (raw). Reference: this is a counterfactual toggle of an
EXISTING derived treatment, not new physics; G2d doc + audit finding 5. Dimensional: mass
in, mass out. Limiting: `on` ⇒ bit-identical (N-0); σ_M → 0 ⇒ shift → 0 ⇒ modes converge
(pinned). Regression: parity + N-0/N-2.
**F:** withdrawn — no instrument, no gate; the ledger entry closes by the P1 derivation
(recorded in §1), which is presented to the author WITH this prereg as the zero-cost
resolution of the bscale-memo §4 item.
**J:** Old: Σ⁴ᴰ point-evaluates S_4D at (d_L_g, M_g(1+z_g)) (:2713-2730). New (variant):
p_det,g = E_{M~N(M_eff_g, σ_g²)}[S_4D(d_L_g, M(1+z_g))] per galaxy — the erf-sum Gaussian
kernel at the Eddington-shifted mean, matched to production's own D_g kernel (P2); σ_g =
BH_MASS_ERROR (linear M_sun); no R_eff inside the kernel; Σᶲ and w_g untouched. Reference:
fixb_pathA §D3 (documented open term; MFG A2 numerator/denominator population
consistency); σ_g enters as a declared physics input. Dimensional: probability in [0,1]
either way. Limiting: σ_g → 0 ⇒ kernel ≡ point (pinned test). Regression: N-0/N-2 +
limiting test + the per-h selection-table JSON dump (P4).

---

## VERDICT

*(append-only below this line after execution)*

**VERDICT (2026-08-20, appended after execution; branches presented, not adjudicated):**

- Fleet: jobs 6373194–6373199, 168/168 COMPLETED. **N-0 PASS** (v0 bit-identical to
  baseline, 0.0 both channels/venues; metadata diffs whitelisted-only; v0 run_metadata
  absent — flagged, pre-dates the dump, not gate-bearing). **N-1(E) PASS** (1D bit-identical
  over the full grid). **N-2(E) PASS** (87.1%/78.9% engagement). **N-2(J) PASS** (max
  |r_Malm ratio − 1| = 0.0775/0.0557; eoff-vs-v0 table identity verified first).
- **R-E: s_Edd,new = +0.0012 (iiib) / +0.0019 (joint_r1)** — POSITIVE and immaterial
  (≪ 0.008). The stale −0.020 was off by an order of magnitude AND sign at the current
  operating point (the audit's staleness caution fully vindicated; the G7row9_N5 −0.0022
  anchor was closer but also sign-differed — measured at the derived-form baseline, this
  is the number of record). Budget leg: s_Edd ≈ +0.001–0.002, immaterial.
- **R-J: ΔJ = −0.0025 (iiib) / −0.0061 (joint_r1)**; r_Malm rises 4.0–7.8% under the
  kernel-consistent Σ⁴ᴰ. Sub-material in iiib, 76% of materiality in joint_r1. Per the
  registered use (§4): J is a DOCUMENTED derivable inconsistency (D3/F10, MFG A2) with a
  measured near-material size — it returns to the author as a /physics-change CANDIDATE
  (adopting the kernel form would move the post-fix offsets to ≈ −0.055/−0.057), not
  pre-decided.
- **Tilt-ledger state after this battery** (baseline offsets −0.0529/−0.0512): s_Edd
  measured immaterial (+0.001/+0.002); f-treatment BOUNDED-IMMATERIAL by derivation
  (≤ 2e-4); J_α measured −0.0025/−0.0061 (fork pending); the BASE TILT remains the open
  dominant entry — exactly what the running Option B correspondence measurement decomposes.
