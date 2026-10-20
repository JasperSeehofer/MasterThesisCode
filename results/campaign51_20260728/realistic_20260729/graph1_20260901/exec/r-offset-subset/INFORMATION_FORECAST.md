# Stage-1 information forecast — r-offset-subset (the Graph 2 seed)

Date: 2026-09-03 (night), amended 2026-09-04 00:45 for the R8 three-label class covariates. Author: the
batch-2 prereg author A (top-tier). Verdict-free: every number is a forecast INPUT with its source; no
registered statistic has been computed. Companion to `REGISTRATION_DRAFT.md`.

## 1. What a perfect analysis of the banked data can say (rule 9)

Everything the question needs is on disk: the two re-baseline CSVs (41 h × 1588 events), the production
CRB (truth class, sky covariance, M, d_L, SNR), the h = 0.73 log (per-event candidate counts, P6) and the
catalogue (host positions). The arm is a pure re-read; the ≤ 2 CPU-h cap is a ceiling on nothing.
What the data CANNOT say: whether the 82 events are influential because of a property not among the
eleven registered axes (structural blindness, draft §6), and whether events matter jointly rather than
singly (LOO limitation).

## 2. Forecast inputs

| input | value | source | tag |
|---|---|---|---|
| S size / offset | k = 82 of 1588 (5.16 %); mean_h − 0.73 = −0.0641 (iiib 2D) | rows #302/#342 | [DOC] |
| pre-flip class/z structure of the score tilt (production) | dark class −0.635 ± 0.017 (37σ); ≈ 0 at z < 0.4, −1.08 by z ≈ 0.9 | row #137 (ledger line 1347), `docs/RETROSPECTIVE_D1_20260820.md` | [DOC], STALE ([A11], pre-flip 1D leg) |
| post-flip harness class structure | catalogue-hosted (truth label) Z 9.76/7.15; dark 1.26/1.76 | row #335 | [DOC], harness venue, N = 200 |
| truth in_catalog share | 76/1588 = 4.8 % | CRB | [LOCAL] count |
| estimator class shares (iiib, h = 0.73) | exact-zero: 606 dark / 982 hosted; relative 1e-6: 1241 / 347 | R8 build | [DOC] |
| in_catalog class heavy tail | plain/MAD SD 8.5; events 889 (+52), 474 (−24) | row #344 | [DOC] |
| OUT events (10) pull toward truth; leave-out −0.0049 | row #344 | [DOC] — pre-read of C8 |
| deep-venue precedent (pre-fusion, seed1000) | top decile carries 25.5 % of the host sum — "carried broadly" | EXP-40 | [DOC], venue-scoped |
| completion residual: 74 % production-only | ρ = 0.257 | row #347 | [DOC] |

## 3. Expected covariate and why

**Expectation (author's own reading, not a measurement):** the separating axis, if any, is the
**estimator-side catalogue-leg weight** — (c) `log10_f_cat` first, then (b) `hosted_rel` — jointly with
**redshift** (C4). Reasoning: (i) the offset is a coherent low-h pull; per-event pull toward low h in a
dark-siren likelihood comes from the completion leg's z-integral shape and from impostor-dominated
catalogue legs at high z, both of which scale with z and with how much catalogue weight the event carries;
(ii) the pre-flip z-resolution of the tilt was strong (≈ 0 → −1.08 across z), and the flip changed the
catalogue leg, not the completion leg's z-shape; (iii) the R8 table shows the exact-zero "hosted" class is
mostly (635/982) negligible-weight — if S is catalogue-driven it should be enriched in the 347 material-
weight events, which is exactly the (b)-vs-(a) discrimination the chair asked for; if S is completion-
driven, (c) should be anti-enriched (S sits at the floor) and C4 high. Against: a truth-class (C1)
separation needs ≥ 11 of the 76 in_catalog events in S; the row #344 IN-class tail (474 with s_e −24) says
a few are, but the OUT events pull the other way — I forecast C1 WEAK, not SEPARATES. C5 (sky area) and C7
(candidate count) are correlated with z and with (c); I expect them to co-move at |AUC − 0.5| ≈ 0.10–0.20,
i.e. WEAK. C6 (mass-window retention) and C10/C10b (M axis): NULL expected — the timeout selection acts on
the injection pool, not on how an accepted event scores. C8: anti-enriched by pre-read. C11 (SNR): NULL.

Probability weights: **INTERMEDIATE 0.40** (a WEAK-to-SEPARATES axis in z / f_cat whose decile stratum
moves mean_h by 0.003–0.008 — below T_mat, because S is 82 events and a decile is 159, so the stratum
dilutes the pull), **DIFFUSE-IN-COVARIATES 0.30** (the deep-venue "carried broadly" precedent, plus the
LOO tail being a likelihood-shape object), **SUBSET-IDENTIFIED 0.30** (most likely via (c)+C4 together;
a single covariate reaching AUC ≥ 0.70 AND Δ_strat ≥ 0.008 requires S to be concentrated in one decile
tail, which the 5 % fraction makes possible but not likely).

## 4. What each disposition changes on the board

- **SUBSET-IDENTIFIED** ⇒ Graph 2 gets its first mechanism node: "the −0.064 is carried by <covariate>
  stratum"; the S3 revision-2 question (docket R2, parked) gets a concrete "what to change" (stratify the
  harness by that covariate); d-residual-attribution's 74 % production-only part gets a candidate index;
  R14 gets a measured answer on which class notion indexes the defect. Nothing is fixed: a stratum owning
  the offset is a localization, not a cause — the falsifier of the attribution (rule 15) is the harness
  replicate on that stratum, to be registered separately.
- **DIFFUSE-IN-COVARIATES** ⇒ q-offset-subset settles bounded: none of class (four notions), z, sky
  area, candidate count, mass channel, cone status, M axis indexes the subset. The heavy-tail reading of
  row #342 becomes "heavy tail in per-event likelihood shape, not in any catalogued property"; the paper's
  honest-bound framing inherits a citable negative; R2 stays parked; Graph 2's next node is a likelihood-
  SHAPE read on S (e.g. per-event ln L(h) curvature / bimodality), which this arm deliberately did not
  register.
- **INTERMEDIATE** ⇒ the flagged axis is named with its non-material Δ_strat and captured fraction; one
  revision (≤ 2) may refine the stratum on that axis only (no new covariates); R14 still receives the
  three-label line regardless.
- **R14 line (all dispositions):** if (b) separates but (a) does not, the 635 negligible-weight "hosted"
  events are bulk and every A12 class-conditioned read (row #335 included) was measured on a mixed class —
  a re-labelling, not a re-measurement, follows. If (a) separates and (b) does not, S lives among the 635 —
  the support-only catalogue leg is the object, a genuinely new lead.
