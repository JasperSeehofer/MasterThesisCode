# PRE-REGISTRATION — Tilt-ledger battery: s_Edd re-measurement, f_k/f̄ consistency, J_α

**Date:** 2026-08-19 · **Status:** DRAFT v1 — awaiting verifier pre-check; instrument code is
physics-change-gated (§6 presentations) and awaits the author's gate approval. Authorized as
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
- **F — `--beta_gbar_completeness {isotropic,banded}`** (default `isotropic` = production).
  Under `banded`, `precompute_phi_selection_integrals` (:1964-2019) computes β_Ḡ^φ with
  the per-pixel/banded f_k treatment (pattern of the legacy precompute :1342-1363; same
  `CompletenessModel` instance, recon-verified) instead of the isotropic `f_bar` (:2008) —
  making the completion numerator (banded f_k at :4637-4644) and its β leg CONSISTENT
  (`bscale_completion_normalization.md` §4's documented residual inconsistency). Single
  precompute seam, no scalar/batch split; downstream tables re-derive automatically.
  **Read R-F:** ΔF = mean_h(variant) − baseline, both channels. REPORTED-ONLY + materiality
  flag (≥ ⅓σ_h). Direction not registered (unknown sign; that is the point of measuring).
- **J — `--sigma4d_mass_kernel {point,kernel}`** (default `point` = production). Under
  `kernel`, the `with_bh_mass=True` branch of `precompute_global_catalog_selection`
  (:2707-2731) integrates the per-galaxy detection probability against the galaxy's lnM
  kernel (σ_lnM from the catalogue's `BH_MASS_ERROR` column, available in the iterated
  DataFrame per recon; kernel machinery template: `_mass_trunc_*` :622-674) instead of
  point-evaluating at (d_L_g, M_g(1+z_g)) — the D3/F10 J_α term (fixb_pathA §"Mass
  evaluation": changes r_Malm → r_Malm·J_α; σ_lnM enters as a declared physics input).
  Single precompute seam.
  **Read R-J:** ΔJ = mean_h(variant) − baseline (2D; 1D reported), plus the r_Malm(h)
  ratio table as a diagnostic. REPORTED-ONLY + materiality flag; documented adverse
  direction carried as context, not a band.

**Independence (recon-verified):** the three instruments touch disjoint functions and are
disjoint from the `--catalogue_mass_overlap` and `--completion_b_scale` seams. One soft
entanglement disclosed: `catalogue_mass_overlap=inflated` freezes μ_eff computed WITH the
Eddington shift; battery cells never combine flags, so no arm is affected — noted for
future combined use only.

## 2. Gates

- **N-0 (continuity, scored, STOP):** V0 cells (all defaults) at probe h ∈ {0.675, 0.700}
  per venue must reproduce the banked post-fix baseline `event_likelihoods.csv` rows to
  max rel diff ≤ 1e-10 (both channels).
- **N-1 (1D-channel discipline):** instrument E must leave `combined_no_bh` BIT-IDENTICAL
  (the shift lives in the with-BH path only). Instruments F and J legitimately move BOTH
  channels (β_Ḡ^φ and Σ⁴ᴰ feed both) — their 1D deltas are REPORTED, not gated.
- **N-2 (engagement, scored, STOP — never silently null):** E: ≥ 10% of
  catalogue-supported events change `combined_with_bh` by ≥ 1e-6 rel at the probes.
  F: max_h |β_Ḡ^φ,banded/β_Ḡ^φ,iso − 1| > 1e-4 (table-level engagement). J: max_h
  |r_Malm,kernel/r_Malm,point − 1| > 1e-4. Failure ⇒ STOP (instrument not engaged).
- **Metadata:** run_metadata cli_args diff vs the post-fix baseline referent shows only
  whitelisted keys ({git_commit, timestamp, working_directory, seed/random_seed, SLURM_*,
  eddington_m, beta_gbar_completeness, sigma4d_mass_kernel}).
- Scalar↔batch parity pytest for E; table-level unit tests for F/J (banded reduces to
  isotropic under uniform completeness; kernel reduces to point as σ_lnM → 0 — limiting
  cases, pinned).

## 3. Execution

Cluster, evaluate.sbatch pattern, config of record (absolute_marginal, volume_deconv,
pdet_z_resolved, `--selection_in_completion_numerator off`, EVAL_SEED 777000): 3
instruments × 2 venues × 41 h + 4 V0 probes = **250 tasks ≈ 12.5 CPU-h**. Precondition:
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
**F:** Old: β_Ḡ^φ from isotropic `f_bar` (:2008). New (variant): banded f_k sum (legacy
pattern :1342-1363), same completeness object. Reference: Gray-Messenger-Veitch 2022
arXiv:2111.04629 Eq. 5 (the f_k the numerator already uses); consistency argument
bscale memo §4. Dimensional: completeness fraction, dimensionless. Limiting: uniform
completeness ⇒ banded ≡ isotropic (pinned test). Regression: N-0/N-2 + limiting test.
**J:** Old: Σ⁴ᴰ point-evaluates S_4D at (d_L_g, M_g(1+z_g)) (:2713-2730). New (variant):
E_{lnM~N(ln M_g, σ_lnM,g)}[S_4D(d_L_g, M(1+z_g))] per galaxy (truncated-lognormal kernel
per `_mass_trunc_*` machinery; σ_lnM from BH_MASS_ERROR). Reference: fixb_pathA §D3
(documented open term; MFG A2 numerator/denominator population consistency). Dimensional:
probability in [0,1] either way. Limiting: σ_lnM → 0 ⇒ kernel ≡ point (pinned test).
Regression: N-0/N-2 + limiting test.

---

## VERDICT

*(append-only below this line after execution)*
