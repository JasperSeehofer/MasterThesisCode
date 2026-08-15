# L0-SB — sandwich/score-balance diagnostic on stored posteriors — result

**Status: PRESENTED, NOT ADJUDICATED.** Recomputed measurements against the item registered in
`PROPOSAL_STAGE3_20260815.md` §2 ("L0-SB"), testing **H-SB** (§1: "the residual displacement is
the misspecification/score-balance mechanism"). No branch call, repair, or register status change
is made here; that is the author's. Ledger row #105.

**Data:** committed per-seed `ln_post_1d`/`ln_post_2d` grid vectors and `map_*`/`post_sd_*` fields
only — never a file's `aggregate` block. All 22 mechanism-study result files under
`results/mechanism_study_20260813/*_h0p730_results_seeds*.json` (16 S-cells + AM2P, ANULL, MEH,
MEI, MN0, MN0X — 510 seeds) **and** the venue-transfer campaign per-seed records at
`results/venue_transfer_20260811/*_h0p730_results_seeds*.json` (T0, Ta, Tb, Tc, merged across
seed-batch files — 1,000 seeds), which **are** present on this machine. 26 cells/arms, 1,510 seeds
total, both channels. Method: `l0_sb_diagnostic.py`; raw output: `L0_SB_output.json`.

**Two independent local estimators**, both grid-based, no smoothing:

* **Score (first derivative) at truth, T** — the verbatim central-difference geometry of
  `m6r_l0_decomposition._slope_at_truth_per_seed` / `venue_transfer._slope_at_truth`, so the T
  values here reproduce the previously-verified numbers exactly and are directly comparable.
* **Curvature (second derivative), A** — an *independent* local quadratic least-squares fit
  `ln_post(h) ≈ c0 + c1(h−h0) + c2(h−h0)²` over up to 5 grid neighbours of a target point h0;
  `A = −2c2`. Applied at each seed's own grid-argmax MAP (§1) and at h_true (§2's Ā). Fewer than 3
  finite points near h0 (or a non-concave fit, A ≤ 0) is reported, not silently dropped.

A cell/arm is flagged **degenerate** (excluded from headline statistics, reported for context) if
it is one of the known f_h = 0 S-cells (S00–S03) or if `post_sd` is exactly 0 for every seed in
either channel — this catches MEI and venue-transfer T0 in addition to the S-cells, both
near-delta posteriors (host resolved exactly). The f_i = 0 edge column (S10, S20, S30) is flagged
separately (near-degenerate, M6R §4's known closure-breaking caveat) and also excluded from
headline statistics. **17 of 26 cells/arms are headline; 9 are degenerate/edge, reported below.**

## 1. Information width: σ_A (curvature at MAP) vs stored post_sd

| cell | post_sd (1D median) | σ_A (1D median) | σ_A / post_sd |
|---|---:|---:|---:|
| AM2P | 0.003953 | 0.003889 | 0.984 |
| ANULL | 0.004265 | 0.004241 | 0.994 |
| MN0 | 0.004265 | 0.004241 | 0.994 |
| MN0X | 0.004386 | 0.004361 | 0.994 |
| S11 | 0.002331 | 0.002115 | 0.907 |
| S12 | 0.002487 | 0.002274 | 0.914 |
| S13 | 0.002615 | 0.002590 | 0.990 |
| S21 | 0.002672 | 0.002558 | 0.957 |
| S22 | 0.002764 | 0.002723 | 0.985 |
| S23 | 0.003352 | 0.003312 | 0.988 |
| S31 | 0.002870 | 0.002826 | 0.985 |
| S32 | 0.003198 | 0.003217 | 1.006 |
| S33 | 0.004384 | 0.004363 | 0.995 |
| Ta | 0.003619 | 0.003618 | 1.000 |
| Tb | 0.003689 | 0.003689 | 1.000 |
| Tc | 0.004376 | 0.004364 | 0.997 |
| MEH | 0.000187 | 0.001077 | 5.747 |
| S10 | 0.000070 | 0.001037 | 14.895 |
| S20 | 0.000228 | 0.001147 | 5.038 |
| S30 | 0.000731 | 0.001117 | 1.528 |

**σ_A tracks post_sd to within 1–10% in every well-conditioned cell** (the 13 interior S-cells and
6 non-degenerate arms — 0.907–1.006). This validates the local-quadratic curvature estimator: it
recovers, from the grid shape alone at the MAP, essentially the same width the pipeline's own
grid-moment `post_sd` reports. The four cells where it departs badly (S10/S20/S30 the f_i = 0 edge
column, MEH) are exactly the pre-flagged near-degenerate group — `post_sd` there is a moment over a
near-delta distribution and collapses toward zero faster than the local curvature does, matching
the M6R §4 caveat that this same group breaks the T·σ²_post closure. No new failure mode surfaces.

## 2. Sandwich width σ_SW and predicted overconfidence

Per headline cell: `B` = Var_seeds[score at truth] (quadratic-fit estimator), `Ā` = mean curvature
at truth, `σ_SW = √B / Ā`, compared against `σ_A` (§1's median). Full per-cell table in
`L0_SB_output.json` §`section2_sandwich_width`. Headline range: `σ_SW` = 0.00097–0.0064 (1D),
uniformly **smaller** than `σ_A` in every headline cell — predicted `σ_SW/σ_A` = 0.37–1.24 (mean
≈ 0.53, both channels), i.e. the between-seed score-variance sandwich predicts the posterior should
be *as narrow or narrower* than the information width, never markedly wider.

## 3. H-SB's three parameter-free predictions

### (a) bias ≈ T/Ā (pseudo-true displacement), vs the σ_post²-route (0.749 ± 0.046)

| cell | T (1D, nats/h) | bias measured | predicted T/Ā | ratio meas/pred (Ā-route) | predicted T·σ²_post | ratio meas/pred (σ_post²-route) |
|---|---:|---:|---:|---:|---:|---:|
| S11 | 3,370.7 | 0.01267 | 0.01200 | **1.055** | 0.01832 | 0.691 |
| S12 | 2,956.2 | 0.01200 | 0.01061 | **1.131** | 0.01828 | 0.656 |
| S13 | 2,719.5 | 0.01400 | 0.01205 | **1.162** | 0.01860 | 0.753 |
| S21 | 3,544.3 | 0.01900 | 0.01691 | **1.123** | 0.02531 | 0.751 |
| S22 | 2,745.0 | 0.01600 | 0.01397 | **1.145** | 0.02098 | 0.763 |
| S23 | 2,642.9 | 0.02365 | 0.01982 | **1.193** | 0.02970 | 0.796 |
| S31 | 3,399.6 | 0.02200 | 0.01695 | **1.298** | 0.02801 | 0.786 |
| S32 | 2,962.8 | 0.02333 | 0.02001 | **1.166** | 0.03030 | 0.770 |
| S33 | 2,667.1 | 0.03967 | 0.03098 | **1.280** | 0.05125 | 0.774 |
| AM2P | 1,492.0 | 0.01920 | 0.01714 | **1.120** | — | 0.823 |
| ANULL | 2,644.0 | 0.03467 | 0.02801 | **1.238** | — | 0.721 |
| MN0 | 2,644.0 | 0.03467 | 0.02801 | **1.238** | — | 0.721 |
| MN0X | 2,624.9 | 0.03725 | 0.02945 | **1.265** | — | 0.738 |
| MEH | 4,093.8 | 0.00400 | 0.00526 | 0.761 | — | 27.803 |
| Ta | 2,672.6 | 0.03490 | 0.03416 | **1.022** | — | 0.997 |
| Tb | 2,672.7 | 0.03588 | 0.03519 | **1.020** | — | 0.987 |
| Tc | 2,600.2 | 0.03724 | 0.02924 | **1.274** | — | 0.748 |

**Headline (17 cells, both channels) ratio-of-ratios statistics:**

| route | 1D mean ± sd | 1D range | 2D mean ± sd | 2D range |
|---|---|---|---|---|
| T/Ā (new) | **1.147 ± 0.132** | [0.761, 1.298] | **1.164 ± 0.134** | [0.798, 1.307] |
| T·σ²_post (M6R, old) | 2.369 ± 6.555 | [0.656, 27.80] | 1.607 ± 3.421 | [0.658, 14.88] |

Using the **actual local curvature Ā at truth** in place of the implied curvature `1/σ²_post`
**closes the previous 0.749 ± 0.046 factor to 1.147 ± 0.132** — roughly halving the mean
proportional miss (0.75 is 25% low; 1.15 is 15% high) and, more importantly, **collapsing the
variance and removing MEH's 27.8× outlier entirely**: the σ_post²-route mean is dominated by MEH
(post_sd ≈ 1.9e-4, tiny, blowing up the ratio); the Ā-route is not sensitive to this because Ā is
computed directly from the log-posterior curvature, not from the possibly-collapsed grid-moment
width. **Zero of the 17 headline cells fail the a-priori ±3× band on the Ā-route (worst case
1.298); the σ_post²-route fails once (MEH, 27.8×).** `Ā/(1/σ²_post)` itself runs 1.02–1.72 across
the headline set (mean ≈ 1.44 ± 0.42, 1D) — i.e. the local curvature at truth is systematically
1.0–1.7× *larger* than the curvature implied by the stored moment-width, which is exactly the
gap the σ_post²-route was missing.

### (b) does σ_SW predict the ~8.5× overconfidence (bias/post_sd), i.e. is post_sd ≈ σ_A ≪ σ_SW?

| cell | overconfidence observed (|bias|/post_sd) | overconfidence predicted (σ_SW/σ_A) | ratio pred/obs |
|---|---:|---:|---:|
| S11 | 5.433 | 0.473 | 0.087 |
| S12 | 4.825 | 0.530 | 0.110 |
| S13 | 5.353 | 0.617 | 0.115 |
| S21 | 7.110 | 0.603 | 0.085 |
| S22 | 5.788 | 0.368 | 0.064 |
| S23 | 7.055 | 0.622 | 0.088 |
| S31 | 7.665 | 0.455 | 0.059 |
| S32 | 7.296 | 0.539 | 0.074 |
| S33 | 9.049 | 0.546 | 0.060 |
| AM2P | 4.857 | 0.441 | 0.091 |
| ANULL | 8.129 | 0.432 | 0.053 |
| MN0 | 8.129 | 0.432 | 0.053 |
| MN0X | 8.494 | 0.468 | 0.055 |
| MEH | 21.337 | 1.242 | 0.058 |
| Ta | 9.645 | 0.620 | 0.064 |
| Tb | 9.726 | 0.602 | 0.062 |
| Tc | 8.510 | 0.484 | 0.057 |

**Headline (17 cells) ratio-of-ratios: 1D mean 0.073 ± 0.020 (range [0.053, 0.115]); 2D mean
0.069 ± 0.018 (range [0.049, 0.105]).** The predicted overconfidence from `σ_SW/σ_A` is
**everywhere less than 1.25 and mostly less than 1** — the between-seed score-variance sandwich
predicts the posterior should be roughly as wide as (or narrower than) `σ_A`, never the observed
5–21× narrower-than-displacement. The reconstructed prediction is off from the observed scale by a
factor **8.7–19×, in every one of the 17 headline cells, with no exceptions and small scatter**
(sd/mean ≈ 0.27) — this is not one bad cell dragging a mean, it is a uniform, one-sided miss.

### (c) counterexample flag (>3× off, parameter-free — no cell used to set a coefficient)

* **(a) route (T/Ā):** 0 of 17 headline cells exceed the ±3× band (worst case 1.298×). **No
  counterexample.**
* **(b) route (σ_SW/σ_A):** **all 17 of 17** headline cells exceed the ±3× band (all in the
  under-predicting direction, ratio pred/obs ∈ [0.05, 0.12] ≪ 1/3). **Uniform counterexample.**
* Degenerate/edge cells (S00–S03, MEI, T0, S10/S20/S30) are reported in §1 and `L0_SB_output.json`
  but excluded from both counts above — their `post_sd`/curvature ratios were already known to
  break the σ_post²-route closure (M6R §4) before this diagnostic ran, and a near-delta posterior
  is not a fair test of either prediction.

## 4. Verdict

**H-SB: PARTIAL.**

* **Prediction (a) — the pseudo-true displacement bias ≈ T/Ā — is quantitatively supported and is
  a genuine improvement over the previously-reported closure route.** It reproduces the measured
  MAP bias to within 15–30% (mean 1.147×, no cell off by more than 1.3×) using **the same measured
  T** the σ_post²-route used, differing only in using the log-posterior's own local curvature
  instead of the grid-moment `post_sd` as the "effective information." It removes the σ_post²-route's
  worst failure (MEH, 27.8×) entirely, because Ā does not collapse the way a moment-based width does
  near a near-degenerate posterior. This is the sharper, more mechanistically grounded restatement
  of the same closure the author's earlier M6R work found at 0.749 ± 0.046 — not a new number pulled
  from nowhere, but the same displacement measured with the curvature the White/KV asymptotics
  actually reference (A, not `1/σ²_post`).
* **Prediction (b) — that the same sandwich apparatus explains the ~8.5× coverage
  overconfidence via `σ_SW/σ_A` — is REFUTED, cleanly and uniformly.** The between-seed
  score-variance `B` is *smaller* than the mean curvature `Ā` in every headline cell (predicted
  `σ_SW/σ_A` < 1.25 always, often < 1), an order of magnitude short of the observed 4.8–9.7×
  (headline)/up to 21× (MEH) narrowness. If seed-to-seed score variance were the source of the
  reported overconfidence, `σ_SW` would need to be several times `σ_A`; instead it is comparable to
  or smaller than `σ_A`. **The classical information-matrix-equality intuition (Var[score] ≈
  curvature under correct specification) is roughly holding here in order of magnitude — B and Ā
  are within a factor of ~2–3 of each other, not the factor of ~50–100 needed to produce an 8.5×
  width ratio.** Whatever mechanism narrows `post_sd` to ~8.5× below the actual bias spread, it is
  not captured by the seed-to-seed variance of the score at truth.

**Net reading:** H-SB's *point* (the MAP sits near a pseudo-true value displaced by T/Ā, not by
chance) is confirmed at the level of a parameter-free, order-15%-accurate prediction across 17
independent cells/arms and both channels — a real, useful result for the mechanism-isolation
thread. H-SB's *width* story (the same sandwich construction also explains why the reported
posterior is ~8.5× too narrow) is not supported by this diagnostic; the observed overconfidence
scale needs an account that is NOT simply "seed-to-seed score dispersion vs local curvature" —
most plausibly the *within-seed* shape of the aggregate log-posterior (e.g. the same
non-renormalized-kernel or distributed-misspecification structure H-REN targets), not a
between-realization variance effect. This is consistent with, and does not contradict, H-REN as
the more likely owner of the coverage-failure scale; H-SB and H-REN remain complementary as stated
in the proposal, with this diagnostic locating H-SB's own explanatory power specifically at the
bias, not at the width.

---

*Method: `l0_sb_diagnostic.py`. Raw per-cell/per-channel numbers (all 26 cells/arms, both
degenerate-flagged and headline): `L0_SB_output.json`. No repair is proposed here; the
`/physics-change` gate is untouched.*

---

## Addendum (2026-08-15) — adversarial-verification amendment: prediction (b) was miscast, and the correct reading is stronger

The verifier reproduced every number (T values exact under independent stencils; T/Ā headline 1.1465 ± 0.1325; zero ±3× counterexamples) with one count correction: ANULL and MN0 are the identical dataset (same seeds, bit-identical MAPs), so "17 independent cells" reads "16 distinct (one deliberate ANULL/MN0 duplication)"; the mean moves to ~1.141, conclusion unchanged. The substantive amendment: Var_seeds[score]/Ā² estimates the frequentist MAP *scatter* (which it under-predicts by 2–3× — sd(MAP) 0.0026–0.0061 vs σ_SW 0.0011–0.0023, a real open detail), while |bias|/post_sd is a *displacement-to-width* ratio driven by the common systematic offset T/Ā that prediction (a) confirms. The sandwich theory never predicts that ratio, so its "refutation" refuted a non-prediction. **Correct restatement: the coverage failure is displacement-dominated; post_sd ≈ σ_A is approximately correctly calibrated to local curvature (§1); the 8.5× is algebraically (T/Ā)/σ_A — fully accounted for by the confirmed displacement law. No separate width mechanism is needed; the single remaining unexplained object is T itself.**
