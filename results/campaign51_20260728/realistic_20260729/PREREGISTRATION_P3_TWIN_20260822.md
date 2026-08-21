# PRE-REGISTRATION — [P3-IMP] "the catalogue-leg twin" (stage 2)

**Date:** 2026-08-22 (overnight, row #160 grant) · **Thread:** `[P3-IMP]`
(claim file `CLAIM_P3_IMPOSTOR_CONVENTION_20260822.md`, stages 0–1 complete).
**Append-only after commit.** A21 governs: any premise correction discovered during
implementation STOPS execution; this file is amended and bands re-derived first.

## 1. The mechanism hypothesis (derivation-grounded, stage-1 refined)

**[INFER from verified code reads + [DOC]]** The production catalogue leg has the SAME broken
numerator/normalizer pairing the fused fix repaired in the completion leg, on its own side of
the mixture:

- Its mixture normalizer **carries the survival factor**: `beta_G_phi(h) = ∫ f̄·S̄_φ·w_pop dz`
  (`bayesian_statistics.py:2065`, verified at source), rescaled into the mixture weight
  `alpha_G_phi = Σ_4D·β_G_φ/Σ_φ` (`:2423-2427`).
- Its per-host numerator **carries none**: `numerator_integrant_without_bh_mass`
  (`:5889-5907`) = GW-Gaussian × volume-deconv host-z kernel, with the `:5896-5901` comment
  codifying the absence as the Gray (2020) Eq. (A.10)/MFG convention.
- The generator is latent-thresholded — the accepted per-event density carries S̄_φ (proven,
  A20/O4 review FATAL-2 table + O6/row #158) — so for THIS venue the generator-matched
  catalogue numerator gains a per-host S̄-type factor, exactly as the completion numerator did
  (L6-DER3 §4: "the catalogue leg is the same fork, per-galaxy"; a per-host S̄-type factor in
  the 1D leg).

**Registered approximation:** the per-host survival is taken as the φ-marginal S̄_φ(z;h)
(population proxy for the host's own p_gal — L6-DER3's "with p_gal in place of φ" refinement is
a registered follow-up, NOT tonight's measurement). The FULL-F LOO weight (1/imp_k) and the
density-form GW factor (V2/D-ii) are NOT included — this registration isolates ONE axis: the
survival-factor pairing. (Ledger §2 item 6 compliance: this is a PAIRED insertion — the
numerator gains the factor its own normalizer already integrates — not the exonerated
"p_det alone" move.)

## 2. The instrument (branch-only; instrumentation-tagged, not a physics change)

Branch **`p3/catalogue-survival-counterfactual`** (never merged tonight; row #160 grant 3).
`BayesianStatistics` gains `catalogue_numerator_survival ∈ {"off","phi"}`:

- `"off"` (default): **byte-identical** to production — the whole-tree test suite plus GATE
  R-P3 prove it.
- `"phi"`: `numerator_integrant_without_bh_mass`'s integrand is multiplied by
  `np.interp(z, *self._phi_survival_table[h])` (endpoint-clamped — the SAME table object the
  normalizer chain integrates; the O6-verified identity `:3812→:3819→:4133` extends to this
  consumer). Applied in **BOTH dispatch paths** — `single_host_likelihood` AND
  `single_host_likelihood_batch` (A13; the batch path is the one production actually uses).
  Log line on activation: `COUNTERFACTUAL: catalogue_numerator_survival='phi' — per-host
  S̄_φ in the catalogue numerator ([P3-IMP] twin cell). Not a production posterior.`
- The **with-BH-mass** catalogue numerator (`mz_integral` leg) is HELD at coded (invariant;
  its S-fork is L6-DER3's 2D analog — follow-up, not tonight).
- Refused unless `normalization_mode="absolute_marginal"` (the S̄_φ table exists only there).

## 3. Arms

| arm | what runs | seeds | cost |
|---|---|---|---|
| **LEV (leverage instrument, pre-pilot)** | zero-`evaluate()`: per banked B-SEL seed, deterministic candidate redraw; compute the share_cat-weighted prediction of the per-event catalogue-term reweight `∂_h ln S̄_φ(z_obs,k)` → predicted per-seed Δscore order of magnitude | 12 | < 15 min local |
| **P (replica gate)** | branch code, flag `"off"`, fresh work root → bit-exact vs banked columns (O6 D6 pattern: ≤1e-12 fallback + wall-time floor > 60 s) | 900101 | ~30 min |
| **PILOT** | flag `"phi"`, 2 seeds (900101, 900102 — first two banked, registry order) → realized paired σ̂ for band freezing + costing line | 2 | ~1 h local |
| **F-φ (primary)** | flag `"phi"`, all 12 banked B-SEL seeds (fresh work roots) | 12 | ≈ 6 CPU-h — local 2-wide ~3 h; cluster fallback (preflight first) within the 50 CPU-h cap |
| **K-flat (kill test, reported-only)** | 1 seed with S̄_φ replaced by its window-average (constant): if the effect is the z-SLOPE mechanism, K-flat ≈ coded | 900101 | ~30 min |

Coded baseline = the banked 12-seed columns and posteriors (no re-run; GATE R-P3 anchors
comparability).

## 4. Gates (scored before any statistic is read; failure ⇒ VOID)

- **GATE R-P3:** arm P reproduces the banked `event_likelihoods.csv` `L_cat_no_bh`, `B_num`,
  `combined_no_bh` columns bit-exact (≤1e-12 multiprocessing fallback), wall > 60 s.
- **GATE E-P3 (A13 engagement + dispatch):** under `"phi"`, ≥10% of catalogue-bearing events'
  `L_cat_no_bh` move ≥1e-6 relative vs banked, AND runtime assertions confirm the flag value
  reached BOTH the scalar and batch host-likelihood paths (counter or log per path).
- **GATE L-P3:** the counterfactual log line present in every `"phi"` run, absent in arm P;
  flag recorded in each run's metadata.
- **GATE N-P3 (A18):** every readout prints, per statistic, the reference it subtracts and the
  band constants, machine-readable.

## 5. Statistics and bands

**Primary:** paired per-seed `Δ_s = mean_h(F-φ, s) − mean_h(coded banked, s)`, fleet mean
`Δ̄(12)` with **paired** SEM (the C-SG BAND R lesson: same seeds, same realizations — the
paired band is the informative one).

**Band FORMULAS (registered now; NUMBERS frozen from the pilot's realized paired σ̂ and
appended below pre-fleet — the O2/pilot precedent):**

- **TWIN-MATERIAL:** `|Δ̄(12)| > max(3·SEM_paired(12), 0.02)` — the catalogue-leg pairing owns
  a material share of the headline bias; the production-facing fix becomes a physics-change
  [RULE] for the author (branch implementation permitted tonight per grant 3; merge never).
- **TWIN-IMMATERIAL:** `|Δ̄(12)| ≤ 0.01` AND `SEM_paired(12) ≤ 0.004` — a precision-qualified
  bound; [P3] returns to the row #110 paper task carrying it.
- **REPORT-BOUND (first-class):** anything else — report `Δ̄ ± SEM` as a bound, no label.

The 0.02/0.01 anchors are ~25%/13% of the impostor drag (−0.079) and ~2σ/1σ of the UNPAIRED
fleet SEM; they are provisional pending the pilot σ̂ re-derivation (A17 realized-scatter
re-check) and may only TIGHTEN, never widen, after data exist.

**Secondary (registered, reported with the verdict):**
1. O2-style impostor decomposition re-run on the F-φ diagnostics (impostor-leg share under the
   twin cell) — the direct answer to the stage-0 claim.
2. Per-event paired Δln L_cat read (rule 10/[A2]) — distribution, not just the mean.
3. Score-at-truth (A12) for the matched and full channels under F-φ.
4. Rail read: r_low count and floor-node mass under F-φ vs banked (expected NULL — the rail is
   photo-z territory; a rail change would be a surprise finding, registered as such).
5. K-flat kill test: |mean_h(K-flat) − mean_h(coded)| ≤ the pilot noise floor if the z-slope
   is the mechanism.

**Axis-leverage statement (A17, pre-data):** GATE-LEV must show the predicted |Δ per seed| is
≥ 5× the band resolution (SEM_paired-scale) BEFORE the pilot runs; if the instrument predicts
sub-resolvable leverage, execution STOPS at LEV and the thread returns with a re-design (the
O4 lesson applied prospectively). The completion-leg twin's measured leverage (+0.125/seed on
the matched channel) is the order-of-magnitude prior; the catalogue leg is active in ~74% of
events and carries −0.079 of the headline.

## 6. A10 — invariants and structural blindness

**Invariants (held fixed in every arm):** completion-leg cell = `off` (the banked baseline of
record; the fused-basis replication is a registered follow-up) · with-BH catalogue numerator =
coded · candidate membership and ball geometry · photo-z kernels/σ · σ_frac columns · the
φ-survival table construction · H grid (41-node + wing) · `PRODUCTION_FLAGS` otherwise
verbatim.

**Structural blindness:** (i) errors COMMON to both conventions — above all an error inside
`precompute_phi_marginal_survival` itself (the O6/O7 disclosed blind spot, now shared by three
instruments); (ii) this design cannot adjudicate data-deterministic vs latent-thresholded
detection for the REAL universe — it measures generator-matched consistency for this venue;
(iii) the p_gal→φ proxy; (iv) the LOO/density-form axes (excluded by design).

## 7. Falsifiers (A19)

- TWIN-MATERIAL is falsified by (i) K-flat NOT nulling (the effect would not be the z-slope
  mechanism claimed); (ii) GATE E-P3 failing on re-audit (engagement vacuity); (iii) a
  zero-compute audit showing the inserted factor is not the same object β_G_φ integrates.
- TWIN-IMMATERIAL is falsified by the paired per-event read (secondary 2) exhibiting opposing
  sub-population shifts that cancel in the mean ([A2] — the read is registered, so the
  cancellation cannot hide).
- The banked verdict, either way, carries its provisional status until the author's stage-5
  ruling (row #160: rulings stay author-gated).

## 8. Costing line (A6/A17)

LEV < 15 min, < 4 GB. P + PILOT + K-flat: 4 × `evaluate()` ≈ 2 h local sequential, ~9 GB each.
F-φ: 12 × ~30 min ≈ 6 CPU-h — local 2-wide (~18 GB peak, 30 GB box) ≈ 3 h wall; cluster
fallback via preflight if the box is contended. Total ≤ ~10 CPU-h of the 50 CPU-h grant.

*(Committed before the branch is created; LEV values and frozen band numbers appended below
pre-fleet; VERDICT appended when the committed scorer reports; A20 review before any banking.)*
