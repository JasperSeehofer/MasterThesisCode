# B3.1 [POP] — does row #138's population-mismatch prediction survive on the fused HEAD diagnostics?

**Launched under rows #222/#223 — charter node B3.1.**
**Date:** 2026-08-29. **Class:** zero-compute measure-first read (no hypothesis test, no
pre-registration apparatus — per the node spec). **No production code touched.**

**Object:** row #138's claim (`BIAS_HISTORY_LEDGER.md:1367-1392`) that the dark-class score
bias is dominated by a population-model mismatch — the estimator's dark-class prior assumes
constant comoving number density (`bayesian_statistics.py:1192`, `precompute_completion_denominator`)
while events are injected from the Barausse (2012) M1 rate — and its quantitative memo
`docs/derivations/population_mismatch_dark_score.md`, which reported the prediction covering
87% of a −0.635 ± 0.017 (iiib) dark-class ensemble score at truth. That attribution was later
DOWNGRADED (rows #140–#144: an "internal misnormalization" claim was raised then itself
DOWNGRADED/reinstated PROVISIONAL-WITH-A-BOUND after a chain of harness-control defects), but
row #138's population term itself was never re-measured against current data. This node
re-measures it on the fused HEAD diagnostics (`headreadout_20260827/{iiib,joint_r1}`,
commit `d04d9dc9`/tree `7bfff25d`, the same event set scored throughout `MEASUREMENT_HEAD_READOUT_20260827.md`).

**Exoneration-check (hard rule 5), done before opening:** grepped
`EXONERATION_REGISTER_20260827.md` and `BIAS_HISTORY_LEDGER.md` §2 "DO NOT RE-TRY" for the
population-mismatch MECHANISM (not just the [POP]/row-#138 tag). No hit — the exonerations on
record are WGEO (window geometry), the mirror's `g_frac`=NaN/sentinel defects, and the
S̄_φ-pairing defect (rows #149–#157, explicitly disclosed as "does NOT cure the production
rail"). Row #138's population term itself was DOWNGRADED to "contributing term of unknown
share," never exonerated as zero. This node is not a re-try of a dead mechanism.

---

## 1. Method — re-derived, not copied

**Predicted profile.** Score-zero identity: for data drawn from the model,
`E_true[∂_h ln p]|_truth = 0`. The dark-class model is
`p(d|h) ∝ (1/D̃^φ(h)) ∫dz w_model(z;h)(1-f(z))S(z;h) p_gw(d_obs|d_L(z,h))`, and at the
Laplace saddle `z*(h)` (where `d_L(z*,h) ≈ d_obs`),

```
Δscore(z) ≈ [d ln(w_model/w_true)/dz](z) × (dz*/dh)(z),   dz*/dh = a(z*) / (h · da/dz(z*))
```

with `a(z) ≡ h·d_L(z,h)`. This node **re-derives both densities from the two source
definitions themselves, independently of the memo's numbers**:

- `w_model(z) = dV_c/dz(z,h) / (1+z)` — `comoving_volume_element()` (`physical_relations.py:571`),
  matching the exact integrand of `precompute_completion_denominator`
  (`bayesian_statistics.py:1170`, verified by reading the function body: population density
  and rate are z-independent constants that cancel, per its own docstring, leaving `1/(1+z)`).
- `w_true(z) = ∫ d(log₁₀M) · dN/dz|_mass(M,z) · R_EMRI(M)` — `Model1CrossCheck.dN_dz_of_mass`
  and `Model1CrossCheck.R_emri` (`cosmological_model.py`), the exact static functions that
  define the injected (M, z) sampling density via `emri_distribution`/`_log_probability`,
  called directly (not re-implemented).
- `a(z) = dist(z, h=1.0)` (`physical_relations.dist`) — exact, not approximate: since
  `H_0 = h·100 km/s/Mpc` with all else fixed, `d_L(z,h) = d_L(z,1)/h` identically, so
  `a(z) ≡ h·d_L(z,h)` is exactly `h`-independent and can be evaluated directly at `h=1`
  rather than via any saddle approximation.

**Integration-measure check (load-bearing, disclosed).** `dN_dz_of_mass`'s
`merger_distribution_coefficients` are anchored at log₁₀-mass bins (4.5/5.0/5.5/6.0/6.25) and
the emcee sampler in `setup_emri_events_sampler` walks `x=(log₁₀M, z)`, calling
`_log_probability(10**x[0], x[1])` with **no Jacobian applied anywhere in that call chain** —
so the stationary MCMC density lives in `(log₁₀M, z)` coordinates directly, and the correct
z-marginal integrates over `d(log₁₀M)`, not `dM`. Verified as decisive by a disclosed A/B: the
rejected `∫dM` convention (extra factor of `M`) gives a dark-class ensemble prediction of
**−0.325/−0.308** (iiib/joint_r1) that undershoots the memo's own independently-computed
−0.555 by 41%; the `∫d(log₁₀M)` convention gives **−0.533/−0.512**, matching the memo's
−0.555 to within 4% *despite using a completely different data source* (analytic M1 marginal
vs. the memo's empirical "pool stratum-a" histogram). This cross-agreement is the strongest
available check that the re-derivation is right; both numbers are carried in
`b3_pop_prediction.json` for the record.

**Measured profile.** Per-event score at truth = central finite difference of
`ln(combined_no_bh)` on the production h-grid's **immediate neighbours of h=0.73: h=0.72 and
h=0.74 (step = 0.02, the grid's native 0.01 spacing on both sides)**. No coarser step was
tried; a wider step would trade curvature bias for less finite-sample noise and is flagged as
an open robustness item (§5).

**Class split (registered from the readout's own definition, before computing anything):**
"dark" = `L_cat_no_bh == 0` at every one of the 41 h nodes (class C-C,
`PREREG_COMPLETION_CLASS_DECOMPOSITION.md`); "matched" = the complement (≥1 node with
`L_cat_no_bh > 0`, i.e. C-A ∪ C-B combined — this conflates true in-catalogue hosts with
impostor-only catalogue support; row #141 found C-A alone pulls the *opposite* sign, so the
"matched" number here is a coarser read than that finding and is reported only for context,
not compared against a per-class prediction).

**z-bins (registered before looking at any HEAD number):** the memo's own fixed edges
(0.075, 0.392, 0.559, 0.659, 0.753, 1.018), reused verbatim rather than re-quantiled on the
HEAD data, so each bin is a paired comparison against a previously published number rather
than a fresh partition chosen after seeing the result.

**z_true source:** `results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv`,
md5 `9a1f2a14384a9281c97ca3be312ddaab` — **verified identical** to the md5 named in both
venues' `run_metadata_21.json` (A11 dataset pin). `event_idx` in `event_likelihoods.csv` is
the direct 0-based row index into this 1590-row CRB file (1588 scored, 2 filtered, matching
`MEASUREMENT_HEAD_READOUT_20260827.md:387`). `z_true = dist_to_redshift(luminosity_distance, h=0.73)`.

Helper script (builds only, does not gate anything):
`results/campaign51_20260728/realistic_20260729/fanout1_20260829/b3_1_pop_measure.py`.
Full numbers: `results/campaign51_20260728/realistic_20260729/fanout1_20260829/b3_pop_prediction.json`.

---

## 2. Per-bin paired comparison

### iiib

| z bin | n | measured mean ± SEM | predicted mean | ratio (pred/meas) |
|---|---:|---|---:|---:|
| 0.075–0.392 | 121 | **+0.081 ± 0.030** | −0.265 | −3.28 (opposite sign — uninformative, see caveat) |
| 0.392–0.559 | 121 | −0.332 ± 0.0073 | −0.488 | 1.47 |
| 0.559–0.659 | 122 | −0.562 ± 0.0044 | −0.589 | 1.05 |
| 0.659–0.753 | 120 | −0.701 ± 0.0034 | −0.639 | 0.91 |
| 0.753–1.018 | 121 | −0.855 ± 0.0058 | −0.697 | 0.81 |
| **ensemble (all 5 bins)** | **606** | **−0.4668 ± 0.0162** | **−0.5334** | **1.143** |
| **bins 2–5 only (n-weighted)** | **484** | **−0.6123** | **−0.6031** | **0.985** |

### joint_r1

| z bin | n | measured mean ± SEM | predicted mean | ratio (pred/meas) |
|---|---:|---|---:|---:|
| 0.075–0.392 | 119 | **+0.103 ± 0.030** | −0.266 | −2.57 (opposite sign — uninformative) |
| 0.392–0.559 | 118 | −0.309 ± 0.0074 | −0.488 | 1.58 |
| 0.559–0.659 | 77 | −0.539 ± 0.0055 | −0.588 | 1.09 |
| 0.659–0.753 | 78 | −0.673 ± 0.0044 | −0.638 | 0.95 |
| 0.753–1.018 | 99 | −0.840 ± 0.0067 | −0.700 | 0.83 |
| **ensemble (all 5 bins)** | **493** | **−0.3938 ± 0.0207** | **−0.5115** | **1.299** |
| **bins 2–5 only (n-weighted)** | **372** | **−0.5744** | **−0.5965** | **1.039** |

Monotone z-growth of the measured score is reproduced on both venues, matching row #137's
qualitative description ("score ≈ 0 below z ≈ 0.4, monotone to −1.08 at z ≈ 0.9" — HEAD's
top bin is now −0.855/−0.840, smaller in magnitude, see §4).

**Bin 1 caveat (both venues):** the predicted profile is NOT monotone-with-sign at low z — an
independent diagnostic scan (z = 0.05→1.02) finds `Δscore_pred(z)` is **positive** below
z ≈ 0.17–0.18 (where the model/true population-ratio is still rising) and negative above it.
Bin 1 (0.075–0.392) straddles this crossover, and its 121/119 events are not uniformly spread
across that range, so its bin-mean prediction (−0.265/−0.266) is not close to the memo's
original "both ≈ 0" characterization of the same bin even though the *measured* value is
still near zero (+0.08/+0.10). This bin's ratio is reported but excluded from the coverage
verdict (§3) as uninformative — a near-zero, sign-ambiguous denominator makes any ratio
unstable, and the predicted side is genuinely different in kind (sign-changing) there, not
just noisy.

---

## 3. Coverage and the charter rule

Per node instruction: coverage = predicted / measured, summed over bins (i.e. computed on the
n-weighted ensemble, equivalent to the aggregate ratio).

| | iiib | joint_r1 |
|---|---:|---:|
| **Coverage, all 5 bins** | **114.3%** | **129.9%** |
| **Coverage, bins 2–5 only (excl. sign-ambiguous bin 1)** | **98.5%** | **103.9%** |

Both readings clear the ≥ 50% band on both venues, decisively — the bins-2-5 number in
particular lands the prediction within 2–4% of full coverage on both venues, tighter than row
#138's original 87%.

**Rule fired: "3.2 warranted."** The population-mismatch mechanism is not merely a
contributing term at the current HEAD — on the informative (z ≥ 0.39) bins it accounts for
essentially the *entire* measured dark-class tilt on both venues, independently re-derived
from first principles rather than copied from the memo.

**This does not by itself re-open or re-rule on rows #140–#144's "internal misnormalization"
provisional-with-a-bound finding** — that was a *different* isolation test (B-SEL/B-SELF/B-DEN,
model-matched synthetic universes) measuring whether the completion leg's *mathematics* is
unbiased even when the population IS matched; it is not contradicted or confirmed by this
purely-diagnostic z-bin re-measurement on production-typical data, and the two findings are
compatible (a real population mismatch AND a residual completion-leg defect can both be true;
row #144 already registered a residual ≥ 0.073 unaccounted for at the OLD −0.635 baseline).
What HAS changed, and is new information for that residual (§4), is that the total to be
explained is now smaller.

---

## 4. Item 4 — the 1D historical comparison: the baseline itself moved

| venue | row #138/#137 historical (1D, dark class) | HEAD (this node, 1D, dark class) | |Δ| | combined-σ |
|---|---:|---:|---:|---:|
| iiib | −0.635 ± 0.017 | **−0.4668 ± 0.0162** | 0.168 | **7.16σ** |
| joint_r1 | −0.565 ± 0.020 | **−0.3938 ± 0.0207** | 0.171 | **5.95σ** |

**This is itself a finding, not noise.** The dark-class score-at-truth shrank by ~26–30% in
magnitude between row #137/#138's baseline and the fused HEAD, at 6–7σ significance on both
venues. `MEASUREMENT_HEAD_READOUT_20260827.md` §G.1 records that the three intervening
estimator-code changes (Σ^φ-divisor, twin z-kernel, symmetric mass filter) plus the sentinel/
combine fix moved the 2D channel MATERIALLY-GROWN on the *full-sample* channel-of-record
statistic; this node shows the *dark-class 1D score* moved the opposite direction (shrank) over
the same period. The two statistics are different objects (full-sample 2D posterior mean vs.
dark-class-only 1D per-event score slope) so this is not a direct contradiction, but it means
row #137/#138's exact numbers are **STALE** and should not be quoted going forward without this
node's update.

**Why coverage went UP while the total went DOWN:** the predicted profile is a fixed function
of (z_true, fixed cosmology) and is essentially unchanged from row #138 (−0.533/−0.512 here vs.
−0.555 there, a ≤4% move, consistent with the dark-class population being the same up to the
n=605→606 boundary reclassification). The measured total fell because of the intervening fixes.
A fixed numerator over a shrinking denominator mechanically raises the coverage fraction — this
is the arithmetic explanation, not a new physical claim, and is flagged so the reader does not
read "coverage rose" as "the mechanism got stronger."

---

## 5. Caveats (complete list)

1. **w_true's integration measure was ambiguous and is resolved by inference, not by
   documentation.** No docstring states whether `dN_dz_of_mass` is a density per unit mass or
   per unit log-mass; this node infers the latter from the emcee sampler's own coordinate
   convention (§1) and cross-checks it against the memo's independently-computed number (4%
   agreement). This is the single most consequential methodological choice in the node —
   the rejected convention would have given ~70–78% coverage (still ≥50%, same verdict, but a
   materially different number) instead of ~98–130%.
2. **Bin 1 (z < 0.39) is sign-ambiguous** for both measured and predicted values and is
   excluded from the headline coverage read (§2–3); it is disclosed rather than dropped.
3. **"Matched" class is a coarse C-A ∪ C-B union**, known (row #141) to conflate two classes
   that pull in opposite directions; the matched-ensemble number here
   (−0.017 ± 0.064 iiib, −0.057 ± 0.040 joint_r1, both consistent with zero) is context only,
   not evidence for or against the population mechanism, which is a dark-class-specific claim.
4. **Finite-difference resolution:** central difference at the grid's native ±0.01 step
   (h = 0.72/0.74). Not tested against a wider step (e.g. ±0.05) or the full 41-node
   log-likelihood slope at h=0.73; a curvature-vs-noise trade-off is open but not expected to
   change the coverage verdict given the size of the observed effect relative to per-event SEM.
5. **Single realization.** Both venues share one universe/injection draw (P7-8 class caveat,
   carried from `PREREG_COMPLETION_CLASS_DECOMPOSITION.md`); iiib and joint_r1 are not
   independent confirmations of the coverage number, only of its robustness to the two
   different config bases sharing that universe.
6. **No z_true source beyond the single 1590-row CRB pool** was available locally for w_true's
   empirical cross-check (the memo's "pool stratum-a" file could not be located in this repo);
   the corroboration in §1 (4% agreement with the memo's own number) substitutes for it but is
   not a from-scratch validation against a second, larger injected sample.
7. **This read does not re-run, extend, or adjudicate rows #140–#159** (internal
   misnormalization / S̄_φ pairing / G-frac threads); it is scoped exactly to row #138's
   original population-mismatch claim, re-measured on current data, per the node's charge.
8. Every number above is independently reproducible from
   `b3_1_pop_measure.py` + `b3_pop_prediction.json`; no number in this record is copied from
   the memo or the ledger without a fresh recomputation alongside it.

---

## 6. Numbers with provenance (A11)

| value | source file:line | date |
|---|---|---|
| dark-class score identity, `E_true[∂_h ln p]=0` at truth | `docs/derivations/population_mismatch_dark_score.md:14-15` | 2026-08-20 |
| `w_model` integrand `dVc/(1+z)` | `darksiren_emri/bayesian_inference/bayesian_statistics.py:1169-1216` (`precompute_completion_denominator`) | read 2026-08-29 |
| M1 rate functions `dN_dz_of_mass`, `R_emri` | `darksiren_emri/cosmological_model.py:245-273` | read 2026-08-29 |
| `dist`, `dist_to_redshift` | `darksiren_emri/physical_relations.py:132-243, 447-490` | read 2026-08-29 |
| dark-class definition (`L_cat_no_bh==0` at every h) | `results/prod2d_closure_20260818/PREREG_COMPLETION_CLASS_DECOMPOSITION.md:20-22` | 2026-08-20 |
| CRB event set + md5 | `results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv`, md5 `9a1f2a14384a9281c97ca3be312ddaab` | verified 2026-08-29, matches `MEASUREMENT_HEAD_READOUT_20260827.md:42-43` |
| HEAD event likelihoods, both venues | `results/campaign51_20260728/realistic_20260729/headreadout_20260827/{iiib,joint_r1}/event_likelihoods.csv` (65109 rows each, 1588 events × 41 h) | run_metadata timestamp 2026-08-27T19:40:20 |
| historical −0.635 ± 0.017 (iiib) / −0.565 ± 0.020 (joint_r1) | `gate_b_20260730/BIAS_HISTORY_LEDGER.md:1347-1348`; `hier_provenance_stamps_20260826.md:150` | 2026-08-20 (measured), quoted 2026-08-26 |
| this node's ensemble/bin numbers | `results/campaign51_20260728/realistic_20260729/fanout1_20260829/b3_pop_prediction.json` | computed 2026-08-29 |

---

## 7. Disposition

**Verdict: "3.2 warranted"** on both venues, by a wide margin (≥98.5% coverage on the
informative bins, ≥114% including the sign-ambiguous low-z bin). Row #138's population-mismatch
term is re-confirmed as (at minimum) the dominant explanation of the current dark-class tilt,
independently re-derived rather than copied, and its currently-measured share of the
(now-smaller) total tilt is larger than it was at row #138's original measurement — because the
total shrank, not because the predicted term grew. The historical −0.635/−0.565 baseline numbers
are STALE (item 4, 6–7σ) and should be superseded by −0.4668/−0.3938 in any future citation of
"the dark-class score." This does not adjudicate the separate internal-misnormalization/
completion-leg-defect thread (rows #140–#159), which remains open on its own evidence.
