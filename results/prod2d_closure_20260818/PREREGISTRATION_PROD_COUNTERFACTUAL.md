# PRE-REGISTRATION — Production counterfactual: catalogue-leg mass overlap (2D residual attribution)

**Date:** 2026-08-19 · **Status:** v2 — verifier pre-check applied (v1 verdict NOT-READY;
P1/P2 blocking repairs + P3–P8 registrations below, applied verbatim; verifier ruled the
design converts to READY with these applied). **Awaiting author approval — the instrument
edit is physics-change-gated; §6 is the gate presentation; NO code until approval.**
**Sequenced by the author's 2026-08-19 [RULE]:** landscape/T1 gated behind the final 2D
residual resolution; this measurement is the next step of that resolution. **No production
change on any branch** — the flag defaults to production behavior (row #119 pattern).

**Provenance:** T0 offsets +0.054 (iiib) / +0.067 (joint_r1); regression VERDICT (R-MIXED):
the 2D−1D per-event excess concentrates in catalogue-supported events, impostor-borne — the
CATALOGUE-LEG locus; mechanism doc M-A/F1/F2. This counterfactual converts the per-event
attribution into a production-native MAGNITUDE attribution — the P7-4 budget's missing leg.

## 1. Variants (instrument = one new CLI flag, default production)

New flag `--catalogue_mass_overlap {production,neutralized,inflated}` (+
`--catalogue_mass_error_scale <float>`, default 1.0, argparse-REJECTED unless mode is
`inflated` — P8). The flag must switch BOTH kernels — `single_host_likelihood` AND
`single_host_likelihood_batch` (production dispatches exclusively through the batch kernel
via `_starmap_host_batches:6149`) — with a scalar↔batch parity pytest under every mode (P2).

- **V0 (null / continuity):** default flag + explicit `--selection_in_completion_numerator
  off` (the basis of the +0.054/+0.067 legs of record; `auto` now resolves to `fused` —
  P3). Gate N-0: per-event `combined_no_bh`/`combined_with_bh` reproduce the banked
  `run_20260804_postfix` values to max rel. diff ≤ 1e-10 at probe h ∈ {0.72, 0.78} per
  venue (empirically supported: the verifier measured 0.0 / 2.5e-13 against the 2026-08-17
  off twin across 13 days of code churn; the evaluate path is deterministic — no rng
  consumers, deterministic erf-sum denominator, order-preserving chunking). Failure ⇒ STOP.
- **V1′ (neutralized — the P1 repair, replacing v1's ill-defined ablation):** in the 2D
  catalogue leg, each candidate's `mz_integral` (:5514–5524 scalar / :6000 batch) is
  REPLACED by the SAME population mass factor `completion_mass_factor_g(z, d_L/d_L_det,
  M_z_det, proj, σ_cond)` (:2022) that the completion leg uses — the candidate becomes
  mass-UNINFORMATIVE (population-prior weight instead of its own measured mass) while BOTH
  2D legs remain densities in the same x_M measure, so the Path-A mixture
  `(alpha_G_phi·L_cat + B_num_wbh_phi)/D_tilde_phi` (:4842) stays commensurate.
  Normalization coherence (registered statement, verifier-verified): `Σ^4D`
  (`precompute_global_catalog_selection`, :2500–2523) uses point masses `M_g(1+z_g)` and NO
  `host_M_error`; `alpha_G_phi·L_cat` cancels `Σ^4D` (:2371 + :4392); `B_num_wbh`,
  `D_tilde_phi` are σ_M-blind — none co-vary. The per-candidate denominator D_g
  (:4991/:5069) is DIAGNOSTIC-ONLY in the production `absolute_marginal` assembly (dead on
  this path) and is left untouched.
- **V2 (inflated, k ∈ {0.5, 2.0} — P5's discriminating ladder, registered NOW):** pure
  WIDTH variant (P4): `host_M_error → k·host_M_error` enters the numerator width σ_gal
  (:5515/:6000) with the Eddington-shifted mean `_host_M_eff`
  (`eddington_shifted_host_mass:585–619`, consumed at :5461) FROZEN at its k = 1 value —
  isolating the F1/F2-vs-F3 width response from the −0.020-class Eddington-mean channel
  (G7 row 9). The σ = min(σ_M, 2M) clamp (:609) is disclosed; freezing μ_eff sidesteps its
  k = 2 near-saturation. Consumer list of record (P4): numerator σ_gal, Eddington μ_eff
  (frozen), diagnostic D_g (untouched). k = 0.5 is the discriminating direction at the
  σ_M ≈ 0.9-fractional operating point: tilt (F1/F2) predicts ~4× amplitude drop;
  dilution (F3) predicts catalogue re-anchoring (Δ toward V0 from the other side). k = 2.0
  approaches the neutralized limit by flattening (reported).

**Registered nulls:**
- **N-1:** `combined_no_bh` BIT-IDENTICAL to V0 under V1′ and V2 (σ_M/mz only in the
  with-BH-mass path). Violation ⇒ STOP (seam leak).
- **N-2 (engagement — P2):** at the probe h-values, V1′ must change `combined_with_bh` by
  ≥ 1e-6 relative for ≥ 10% of catalogue-supported events (cat_e of record: 18.5% iiib /
  38.2% joint_r1). Failure ⇒ STOP (instrument not engaged), NEVER C-REFUTED.
- Out-of-scope disclosure (P7): the test-only `single_host_likelihood_integration_testing`
  (:6156–6350) is not modified; its σ_M references are inert in production.

## 2. Execution

Cluster (joint_r1 observed catalogue lives only there), `cluster/evaluate.sbatch` pattern:
{V1′, V2(k=0.5), V2(k=2.0)} × {iiib, joint_r1} × 41 h + 4 V0 probe tasks = 250 tasks ×
~3 min ≈ 12.5 CPU-h. All variants carry explicit `--selection_in_completion_numerator off`
(P3; a fused-basis rerun would confound with row #119's lever). Inputs: banked seed61000
`prepared_cramer_rao_bounds.csv` symlinks, same seeds/config as the legs of record.
**run_metadata byte-diff referent (P3):** `run_20260817_fusion_counterfactual/off_*/
run_metadata_*.json` (bit-equivalence basis proven by the verifier), allowed differing keys
whitelist: {git_commit, timestamp, working_directory, seed/random_seed, SLURM_*,
catalogue_mass_overlap, catalogue_mass_error_scale, selection_in_completion_numerator
(present-and-off vs absent)}. Preconditions: cluster repo re-synced to rewritten main +
re-tagged (author command pending), fast-forwarded to the instrument commit; preflight
READY ✓. Workspace expiry 2026-09-23 restated (seed61000 symlink targets live there — P8).

## 3. Registered reads and bands

Per venue, 2D channel, trapezoid conventions of record (T0 P7-2):

- **R1 (primary):** ΔV1_v = mean_h(V1′) − mean_h(V0-banked). Prediction: negative (toward
  truth), material. Bands: **C-OWNED(cat-leg)** iff ΔV1 ≤ −0.006 in BOTH venues AND
  |Δ_v(V1′)| < |Δ_v(V0)| in both (Δ_v = mean_h − 0.73); **C-REFUTED** iff |ΔV1| < 0.003 in
  BOTH venues (with N-2 passed); **C-MIXED** otherwise — explicitly including (P6): the
  0.003–0.006 gap; an overshoot past truth violating the toward-truth clause (mixed
  evidence: the leg carries more than the offset); venue-discordant outcomes. P7-8
  one-realization disclosure carried.
- **R2 (width ladder, REPORTED-ONLY):** ΔV2_v(k) for k ∈ {0.5, 2.0}. Discrimination read
  (registered wording): tilt-dominated ⇒ |ΔV2(0.5)| ≪ |ΔV1| with V0-ward sign; dilution-
  dominated ⇒ ΔV2(0.5) moves AWAY from V1′ (catalogue re-anchoring); ΔV2(2.0) expected to
  shadow V1′ (flattening limit) under either — its role is consistency, not discrimination.
- **R3 (residual view, presentation-only):** Δ_v(V1′) quoted against the V-prod off-class
  completion-venue bias (+0.008…+0.015, descriptive).
- **Materiality:** ≥ ⅓·σ_h ≈ 0.006.
- **Fork mapping (returns to the author, none pre-decided):** C-OWNED quantifies the
  catalogue-leg share of +0.054/+0.067 → central evidence for fork (a) correct-form kernel /
  (b) marginalization repair / (c) document-as-systematic; C-REFUTED returns the residual to
  claim intake with the regression's per-event structure as an open anomaly.

## 4. Caveats (registered)

1. V1′ is a NEUTRALIZATION (candidate made mass-uninformative), not a repair candidate — it
   measures the leg's contribution.
2. Single realization (seed61000, shared CRB): venue agreement is one universe (P7-8).
3. Venue-differential ΔV1(joint_r1) − ΔV1(iiib) reported descriptively as the photo-z
   coupling read (venues differ only in `observed_catalogue`), non-band-bearing.
4. Instrument commit on main, default bit-exact (N-0 scored); `[PHYSICS]` commit +
   PHYSICS-GATE-LEDGER row.
5. V2's frozen-μ_eff convention is a counterfactual construction (production co-moves μ and
   σ); registered to isolate the width channel, stated whenever V2 is quoted.

## 5. Tiering (stated per the orchestration mandate)

Recon: sonnet (done). Prereg + interpretation: top-tier (this doc). Verifier: ONE top-tier
xhigh (done, applied). Implementation + tests + sbatch driver: sonnet, medium. Readout:
sonnet compute + top-tier interpretation. Top-tier count: 2 — within the ≤3 cap.

## 6. Physics-change gate presentation (author approval required BEFORE implementation)

- **Old formula (file:line):** 2D catalogue-leg per-candidate integrand
  (`bayesian_statistics.py:5514–5530` scalar; `:5995–6004` batch — the production consumer):
  `L_g(z;h) = gw_3d(z;h) · mz_integral(z;h) · p_gal(z)`, with
  `mz_integral = N(mu_cond; mu_gal_frac, √(σ²_cond + σ²_gal_frac))` [1/x_M],
  `σ_gal_frac = host_M_error·(1+z)/M_z,det` (:5515), `mu_gal_frac` via the Eddington-shifted
  `_host_M_eff` (:585–619, :5461). Assembly: `L_cat_with_bh = Σ_ball w_g N_g^2D / Σ^4D`
  (:4392–4399), combined at :4842.
- **New behavior:** default flag ⇒ UNCHANGED (guarded branch; no arithmetic on the default
  float stream — the guard pattern is `if mode != "production":` around the substitution,
  never an always-executed ×1.0). `neutralized` ⇒ `mz_integral` replaced by
  `completion_mass_factor_g(...)` [same 1/x_M density; §1 V1′]. `inflated` ⇒
  `host_M_error → k·host_M_error` in σ_gal only, μ_eff frozen (§1 V2).
- **Reference:** row #119 counterfactual-instrument pattern; the neutralized form uses the
  production completion-leg factor (Mandel, Farr & Gair 2019, arXiv:1809.02063 Eqs. 5–7;
  Gray et al. 2020, arXiv:1908.06050 Eq. A.19) — no new physics formula anywhere.
- **Dimensional analysis:** V1′ substitutes one [1/x_M] density for another inside the same
  mixture — the Path-A combine stays commensurate (the v1 draft's dimensionless ablation is
  withdrawn for exactly this reason). k·σ_M preserves dimensions trivially.
- **Limiting cases (each pinned by a test):** (i) default ⇒ bit-identical to production
  (N-0, two probe h per venue); (ii) `inflated` k = 1 ⇒ identical to default; (iii) V1′/V2
  1D channel ⇒ bit-identical to production 1D (N-1); (iv) scalar↔batch parity under every
  mode (P2); (v) N-2 engagement at probe h.
- **Regression tests:** N-0/N-1/N-2 + parity as pytest against banked probe rows before
  cluster launch.

---

## VERDICT

*(append-only below this line after execution)*

**VERDICT (2026-08-19, appended after execution; branch presented to the author as a [RULE]):**

- Fleet: jobs 6369297–6369304, 250/250 COMPLETED, zero failures; instrument commit range
  `a4dae5a3` ([PHYSICS], gate-ledger rows). Gates: **N-0 PASS** (combined_no_bh rel diff 0.0;
  combined_with_bh ≤ 6.2e-14 vs 1e-10 gate; metadata diffs all whitelisted, both venues);
  **N-1 PASS** (1D bit-identical, all variants); **N-2 PASS** (engagement 93.5% iiib / 84.7%
  joint_r1 vs 10% gate).
- **R1: ΔV1 = +0.0010 (iiib) / +0.0032 (joint_r1)** — POSITIVE (registered prediction was
  negative-material) and both far below the 0.006 materiality yardstick.
- **R2: ΔV2(k=0.5) = −0.0006 / −0.0039; ΔV2(k=2.0) = +0.0089 / +0.0164** — the width
  response is monotone INCREASING in k and does not saturate at the neutralized value
  (k=2 exceeds V1′), i.e. the catalogue leg has real sensitivity; its production
  operating-point contribution is ≈ 0.
- Venue differential ΔV1(joint) − ΔV1(iiib) = +0.0022 (descriptive photo-z read).
- **Branch (registered table): C-MIXED by the letter** — iiib lands in the C-REFUTED range
  (|ΔV1| < 0.003, N-2 passed) while joint_r1 falls in the registered 0.003–0.006 gap
  (0.0032). Substantive orchestrator reading (non-binding): **catalogue-leg mass-overlap
  OWNERSHIP of the +0.054/+0.067 offsets is refuted at materiality in both venues**; the
  regression's per-event slope structure (catalogue-supported concentration) reflects
  variation AROUND the shift, exactly as its P8 caveat anticipated — not the shift's owner.
- Consequence for the mechanism register: M-A (catalogue-kernel shift) is now refuted at
  production magnitude; M-B was refuted in direction by the regression; the remaining
  candidates for the B-UNOWNED residual are the COMPLETION-leg mass factor g_i/g_frac
  geometry (untouched by V1′ by design, M-C's home) and non-mass-overlap 2D/1D structural
  differences (the alpha_G_phi vs beta_G_phi path asymmetry). A completion-leg counterfactual
  is the natural next registered measurement — returns to the author as a fresh [DO].
