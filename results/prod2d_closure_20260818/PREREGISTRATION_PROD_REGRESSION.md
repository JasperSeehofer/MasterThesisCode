# PRE-REGISTRATION — Production-native per-event slope regression (runbook 21 §3 item 1)

**Date:** 2026-08-19 · **Status:** v2 — verifier pre-check applied (P1–P5 verbatim, P6/P9–P12
incorporated; verifier verdict on v1: NOT-READY, resolved by this revision; the verifier ruled
v2 READY without a second full round provided the c_e′ recon appendix (§2b) is committed
before execution — it is). **Scope:** FREE read on banked outputs
(`results/run_20260804_postfix/{iiib,joint_r1}/diagnostics/`) — zero new simulation, no
production change on any branch. Covered by the row #127 autonomy grant (execution through
readout; every branch returns to the author as a [RULE]). **Append-only after this commit;
VERDICT appended below the line.**

**Class under test:** the σ_M × σ_z coupling class (interim T1/T2 harness evidence; mechanism
candidates M-A/M-B/M-C in `MECHANISM_SIGMA_M_SIGMA_Z_DERIVATION.md` §4, registered BEFORE this
regression runs). Purpose: close the P7-4 transfer gap — test whether the PRODUCTION per-event
structure matches the class prediction, production-natively.

## 1. Data and join (recon of record, 2026-08-19 sonnet recon + verifier spot-checks)

- `event_likelihoods.csv`: 1588 events × 41 h-nodes per venue; columns incl. `combined_no_bh`
  (1D), `combined_with_bh` (2D), `L_cat_with_bh`, `B_num_wbh`, `alpha_G_phi`, `D_tilde_phi`,
  `g_frac`.
- Join to `prepared_cramer_rao_bounds.csv` (1590 rows) by ORIGINAL ROW POSITION:
  `crb.iloc[event_idx]` (event_idx = pre-filter RangeIndex, span 0–1589 with {1203, 1356}
  absent = the n_events_empty = 2 of record; verifier-verified on event 889: SNR 1424.7236,
  in_catalog present).
- Conventions inherited VERBATIM from the T0 scorer (P7-2): trapezoid weights
  `np.gradient(h_grid)`, physics-floor zero-handling — which operates on the COMBINED columns
  only (verified no-op: no zeros there; `L_cat_with_bh` zeros are real physics, handled in
  §2's covariate definitions, never floored).
- 1D reads of record (T0): full-sample 1D means 0.6040 (iiib) / 0.6074 (joint_r1), MAP on the
  low grid edge — the banked "1D starves" phenomenon; needed for gate G-b.

## 2. Registered quantities (per venue v ∈ {iiib, joint_r1}, per event e)

- **h\*_v** = the venue's T0 full-sample 2D posterior mean of record (0.7842 iiib /
  0.7966 joint_r1; N-0 gated).
- **Per-event slopes (P4 applied):** two-point SECANT slope of ln L_e on the registered node
  pairs: **iiib (0.780, 0.785); joint_r1 (0.790, 0.800)** — the pairs bracketing each
  venue's h\*; the same pair for all events in a venue; slopes are never compared across
  venues in absolute value. s_e^{2D} from `combined_with_bh`, s_e^{1D} from
  `combined_no_bh` (log of floored L). Registered sensitivity (reported, non-band-bearing):
  S1–S3 recomputed on the one-step-left and one-step-right node pairs; a sign flip of any
  adjudicating ρ under this shift is disclosed with the branch.
- **Mass-channel slope excess (response):** Δs_e = s_e^{2D} − s_e^{1D}. Note (P2/P8): Δs
  attributes the 2D−1D difference (≈ +0.18 in mean_h), NOT the 2D-vs-truth offset
  (+0.054/+0.067); the bands adjudicate the coupling class's per-event structure only.
- **Covariates (P1 applied — c_e as v1 defined it is NOT computable: `B_num_wbh` and
  `L_cat_with_bh` differ by ~7 orders of normalization; the v1 ratio is ≈1 for >80%/>60% of
  events. Registered replacements):**
  - **cat_e** = 1[`L_cat_with_bh` > 0 at both bracketing nodes] (catalogue-support flag;
    verifier-measured 18.5% iiib / 38.2% joint_r1 of events True).
  - **g_e** = `g_frac` at the bracketing nodes, averaged (banked, varies per event).
  - **c_e′** = common-normalization completion share, defined in §2b from the pinned combine
    mixture; evaluated at both bracketing nodes and averaged.
  - in_catalog (bool, CRB column), SNR (log10), σ_dL/dL = sqrt(var_dL)/d_L,
    z_e = dist_to_redshift(d_L, h = TRUE_HUBBLE_CONSTANT of the run, the `main.py:1480`
    pattern; P9: every registered statistic is invariant under this monotone map / its
    standardization, so the fiducial choice is a label, not an inference input),
  - m_e = |ln(M/median(M))| (detector-frame mass extremity from the CRB `M` column).
    P10 (reported-only): alongside S3, ρ(Δs, +ln(M/median(M))) and the two one-sided
    extremity legs are reported; bands stay on S3.
- **Per-event σ_M (R&V15-propagated)** is NOT banked — **STAGE-2 (mechanical recompute)**,
  triggered ONLY by rule (3) of the §3 decision table: recompute σ_M for the true hosts of
  in-catalog events via `galaxy_catalogue/handler.py` (host_galaxy_index → stellar mass →
  R&V15 + 0.24 dex) and add it as a covariate under the SAME registered statistics.
  **P5 disclosure:** the stage-2 subset is the N_cat ≈ 76 in-catalog events (4.78% of 1588),
  the SAME set in both venues (P7-8) — the two-venue requirement is disclosure, not
  independent confirmation, for this leg. A stage-2 null on n ≈ 76 is reported as
  UNDERPOWERED-NULL (with the bootstrap CI width), never as refutation of M-A.

## 2b. c_e′ recon appendix (committed before execution, per the verifier's condition)

The production combine for this run is the phi-convention Path-A mixture
(`bayesian_statistics.py:4842-4843`):

    combined_with_bh = (alpha_G_phi · L_cat_with_bh + B_num_wbh_phi) / D_tilde_phi

with `alpha_G_phi` and `D_tilde_phi` banked per row (columns 6, 8) and `B_num_wbh_phi` =
`B_num_wbh · β̄_Gφ/β̄_G` not banked directly. The completion term is therefore recovered as
the exact residual of the pinned identity:

    comp_e(h) = combined_with_bh · D_tilde_phi − alpha_G_phi · L_cat_with_bh
    c_e′(h)   = comp_e(h) / (combined_with_bh · D_tilde_phi)
              = 1 − alpha_G_phi · L_cat_with_bh / (combined_with_bh · D_tilde_phi)

Registered validity gate: c_e′ must lie in [−0.01, 1.01] for ≥ 99% of (event, node) pairs
used (identity check; the `D_tilde_phi = 0` fallback branch at `bayesian_statistics.py:4851`
would break it — any violating rows are listed and excluded with count disclosed; > 1%
violations ⇒ STOP, recon error). For events with `L_cat_with_bh` = 0 (verifier: 81.5% iiib /
61.8% joint_r1 at h ≈ 0.78–0.80), c_e′ = 1 EXACTLY — a tie mass > 40%, so per the P1
amendment S2's tercile form is REPLACED by the registered two-group read (§3).

**M-B sign convention (fixed now, per P1):** M-B predicts Δs LARGER for completion-dominated
events — i.e. the catalogue-supported group (cat_e = True) has the LOWER mean Δs rank;
ρ(Δs, c_e′) > 0; point-biserial ρ(Δs, cat_e) < 0.

## 3. Registered statistics, gates, and decision table

Statistics per venue (rank-based for the T0-established heavy tail; P7: Spearman reduces
event 889's 85× slope to a rank; 889 is NOT excluded — T0 jackknife shows it pulls toward
truth, and exclusion would be an unregistered choice):

- **S1:** Spearman ρ(Δs_e, c_e′), bootstrap B = 10,000 over events (seed 20280612, numpy
  default_rng), 95% percentile CI. **S1a:** point-biserial ρ(Δs_e, cat_e). **S1b:**
  Spearman ρ(Δs_e, g_e). Same bootstrap.
- **S2 (two-group read, replacing the v1 tercile — tie mass > 40%):** Mann-Whitney U of Δs,
  catalogue-supported (cat_e True) vs not, with the rank-biserial effect size r_rb and a
  bootstrap 95% CI on r_rb. M-B direction: Δs stochastically larger in the NON-supported
  group (r_rb sign registered accordingly).
- **S3:** Spearman ρ(Δs_e, m_e), same bootstrap.
- **S4 (controls, reported, non-adjudicating):** ρ(Δs, log10 SNR), ρ(Δs, σ_dL/dL), ρ(Δs, z_e),
  point-biserial ρ(Δs, in_catalog); OLS of Δs on standardized {c_e′, m_e, log10 SNR, z_e}
  with HC3 errors — collinearity context only.
- **P6:** h\* and the node pairs are FIXED at the full-sample T0 values inside every
  bootstrap resample (per-event influence on h\* is O(1/n); re-estimating per resample could
  shift the bracket to a different node pair, changing the statistic's definition
  mid-bootstrap).

**Aggregate construction gates (P2 applied — scored, STOP on failure, no branch
adjudicated):**
- **G-a:** |Σ_e s_e^{2D}| ≤ 0.05 · Σ_e |s_e^{2D}| per venue — the 2D slopes must sum to ≈ 0
  at the venue's own 2D posterior mean (the construction check on the 2D leg).
- **G-b:** Σ_e s_e^{1D} < 0 per venue, consistent with the banked 1D means (0.6040/0.6074)
  lying below h\* (construction check on the 1D leg).
- **G-c:** consequently Σ_e Δs_e > 0 (implied, reported).

**Verifier blindness disclosure (P11, of record):** to adjudicate v1's gate the verifier
computed from the banked CSVs: Σ s^{2D} = −14.9/+6.4, Σ s^{1D} = −407/−382, Σ Δs =
+392/+388, the Δs sign fraction (~95% positive) and median (+0.175), the 1D full-sample
means, and the c_e/L_cat zero-tie structure. NO S1–S4 correlation, group read, or covariate
joint distribution was computed. The gates' outcomes are therefore known pre-execution (they
pass); the adjudicating bands remain blind.

**Leg definitions and decision table (P3 applied, verbatim):**

Leg **L-B(v)** = S1 CI excludes 0 in the M-B direction AND the S2 two-group read fires
(r_rb CI excludes 0 in the M-B direction), in venue v. Leg **L-C(v)** = S3 CI excludes 0 in
venue v. Precedence:

1. L-B both venues → **R-CLASS-OWNED (M-B form)**; if additionally L-C both venues, M-C is
   reported as a secondary co-firing, branch unchanged.
2. not-(1) AND L-C both venues AND S1 CI includes 0 both venues → **R-MC**.
3. **Stage-2 triggers** iff neither (1) nor (2) fired AND at least one of {L-B, L-C} fired
   in exactly one venue, OR S1 CI excludes 0 in both venues but S2 fails in ≥ 1 venue.
4. After stage-2 (or if stage-2 not triggered and no leg fired anywhere): all-null →
   **R-REFUTED** (the coupling class does NOT show per-event production-native structure;
   the budget residual stays harness-class-supported only, P7-4 gap OPEN, stated — a
   decisive, quotable null); σ_M leg (ρ(Δs, σ_M) CI excludes 0 in the positive direction,
   both venues, on the in-catalog subset, P5 caveat attached) → **R-CLASS-OWNED (M-A
   form)**; anything else → **R-MIXED**, reported with both venues' numbers, no forcing.

**P8 (branch wording, binding):** every R-CLASS-OWNED text reads "per-event structure
consistent with the [M-B/M-A] form of the coupling class" — never "the regression quantifies
the +0.054/+0.067". ~95% of events share a positive Δs; the covariates explain the variation
around a near-universal shift, not the shift itself.

**P7-8 disclosure carried:** the venues share one realization/CRB set — two-venue agreement
is one universe's evidence, REQUIRED (not counted twice) by the legs above.

## 4. Materiality yardstick

Same as the closure prereg §5: material residual class ≥ 0.006 in h. The regression does not
quote magnitudes into the budget (attribution only; P7-4's production-native-magnitudes rule
untouched — the budget's magnitude legs remain T0 + the documented −0.020).

## 5. Execution

`uv run python regression_prod_native.py` (new script in this directory, committed with this
prereg) → `regression_prod_native_output.json` + a per-event covariate CSV for audit.
Runtime: seconds, local. No cluster dependency; independent of job 6364821.

## 6. Caveats (registered)

1. Δs at a single node pair is a local read; the offset is a global posterior property.
   Mitigation: registered secondary column Δmean_e = leave-one-out shift of the 2D−1D mean
   difference (T0 machinery), reported alongside; bands bind on Δs.
2. c_e′/g_frac derive from diagnostic columns; c_e′ additionally rests on the §2b identity —
   its validity gate is scored.
3. z_e uses the run's fiducial H — a covariate label, not an inference input (P9 invariance).
4. Rank statistics + no winsorizing; event 889 retained (see §3 preamble).
5. The secant-slope venue scales differ (0.005 vs 0.010 node spacing); no cross-venue
   absolute comparison anywhere in this design (P4).

---

## VERDICT

*(append-only below this line after execution)*

**VERDICT (2026-08-19, appended after stage-1 + stage-2 execution; branch presented to the
author as a [RULE], not adjudicated):**

- Gates: G-a/G-b PASS both venues (Σs2D ratio 1.8%/0.86%; Σs1D −407/−382); G-c positive
  (+392/+388). c′ validity gate: 0/3176 violations per venue.
- **Stage-1:** S1 = −0.190 [−0.262, −0.115] (iiib) / −0.276 [−0.337, −0.212] (joint_r1);
  S2 r_rb = −0.359 [−0.459, −0.255] / −0.303 [−0.365, −0.240]; S1a = +0.253 / +0.090 (CIs
  exclude 0); S1b = +0.046 (incl. 0) / +0.065 (excl. 0); S3 = +0.014 / +0.019 (incl. 0 both).
  Sensitivity: no sign flips on adjudicating statistics under one-step node-pair shifts.
  **All firing statistics are in the direction OPPOSITE the registered M-B convention:** the
  Δs excess is stochastically LARGER in catalogue-supported events and decreases with
  completion share. L-B = False both venues (direction), L-C = False both venues.
- **Stage-2 (triggered by rule 3):** σ_M recomputed for the n = 76 in-catalog true hosts
  (handler.py:1337-1351 R&V15 + 0.24 dex; σ_M dex median 0.390, range 0.301–1.643; row-join
  cross-verified against the live handler, 0 mismatches). ρ(Δs, σ_M) = −0.033
  [−0.244, +0.188] (iiib) / −0.268 [−0.458, −0.058] (joint_r1) — the positive-direction leg
  does NOT fire; **UNDERPOWERED-NULL** per P5 (CI widths 0.43/0.40; joint_r1's negative
  excursion reported, not interpreted — P5's caveat and the true-host-vs-impostor-candidate
  σ_M mismatch noted below).
- **Branch (rule 4): R-MIXED** — legs fired in both venues but in the anti-registered
  direction; σ_M leg null. Reported with both venues' numbers, no forcing.
- **Orchestrator reading for the author (non-binding):** the registered M-B (completion
  re-balance) form is refuted in direction — decisively and in both venues. The per-event
  structure that DOES fire localizes the 2D−1D excess in the CATALOGUE leg: it concentrates
  where catalogue candidates engage the mass overlap (cat_e), while the true host being in
  the catalogue ANTI-correlates (S4) — i.e. the excess rides on impostor-candidate mass
  overlaps, the M-A (inverse-mass kernel shift) locus of the mechanism doc, whose specific
  true-host σ_M covariate is the wrong probe for impostor-driven structure (registered
  stage-2 tested what it registered; the impostor-side σ_M test would be a NEW registered
  read). M-C shows nothing. Per P8: this is per-event structure consistent with a
  catalogue-leg form of the coupling class — it does not quantify the +0.054/+0.067.
- P7-4 status: the transfer gap is PARTIALLY closed — production-native per-event structure
  exists and localizes to the catalogue-leg locus; the harness class attribution (σ_M × σ_z
  collapse) remains the class evidence. Follow-up candidates (each a fresh prereg, author-
  gated): impostor-candidate-side σ_M read; per-event Δmean_e leave-one-out column (§6.1).
