# CLAIM [WGEO] — the mass-eligibility window is linear-symmetric while the catalogue mass error is log-normal, so the window's asymmetry is z-structured and drives the high-z dark-class tilt (stage 0) — **CLOSED NULL**

**[OPUS-ORCH 2026-08-27]** — produced under **overnight autonomous orchestration**, without the
author in the loop, by an Opus synthesis chair over three parallel stage-0 reads and one adversarial
verification pass. **Weigh it accordingly:** no author granted this thread's intake, no author ruled
on any number in it, and the verdict below is a chair verdict on chair-and-verifier evidence, not a
ratified finding. Repo HEAD at authorship: `597360fa`. Compute discipline: **local CPU only** — no
cluster, no SLURM, no SSH, no GPU, no pipeline execution, no simulation. No file under
`darksiren_emri/` was read for anything but source citation, and none was modified.

**Thread tag `[WGEO]`** (window geometry). **Stage:** research-cycle stage 0 (claim intake) with the
stage-1 information forecast appended (§7). **Verdict: the lead is DEAD as a bias-attribution
hypothesis, at stage 0, on a decisive measurement (§5).** This card is written as a *closed null
result*, which per the project's standing discipline is a full contribution and is banked with the
same care as a positive.

**Provenance-gating (what upstream gate made this test necessary):** [P3-MKER] R2.7(ii)
(`CLAIM_P3_MKER_20260826.md:773-792`) registered — as an *observation, not a proposal* — that the
window's functional form is linear while its error model is log-normal, exhibiting one candidate with
σ_ln = 1.3032, a **negative** linear lower edge (−213 766 M☉) and an upper edge reaching only 2.955×
the central mass against 7.06× for a log-space window at the same nominal 1.5σ. R2.10 item 4 banked
it explicitly "so that it cannot be re-discovered and presented later as a new finding." This card is
that re-discovery, run deliberately and closed.

---

## 1. The claim, in falsifiable form

> **[WGEO]-H1 (the bias-attribution claim).** A linear-symmetric cut applied to a log-normally
> distributed quantity is asymmetric in the true variable, and the asymmetry grows with the
> fractional error σ_ln ≡ CV. If CV varies systematically with redshift across the catalogue, the
> mass-eligibility window imposes a **z-structured selection** on which hosts are eligible. That
> selection is a candidate mechanism for the campaign's standing unexplained signature — the base
> tilt localized to the dark class at high z (per-event score at truth **−0.635 ± 0.017**, 37σ,
> `BIAS_HISTORY_LEDGER.md:1345-1346` [DOC]).
>
> **Falsifiable content, made numeric:** for H1 to be viable the window's asymmetry must *grow with z
> across the z-band where the tilt lives*, tracking a statistic that runs −0.465 → −0.743 → −0.902 →
> −1.081 over the four banked dark-class z-bins (`docs/derivations/population_mismatch_dark_score.md:41-46`
> [DOC]) — i.e. a factor ≈ 2.3 of growth from the first tilted bin to the last.

> **[WGEO]-H2 (the correctness claim, separable from H1).** Independently of any bias effect, a
> linear-symmetric window is the wrong *shape* for a log-normal error model, and its use is a
> correctness defect rather than a design choice.

H1 and H2 are deliberately split. **H1 is the claim this card kills. H2 is not this card's to open**
(§4.3) — it is already owned elsewhere in the record.

## 2. Refute by (the claim's own falsifier, registered before the reads were adjudicated)

> **Refute-by(H1):** measure the window-asymmetry statistic in the *exact z-bins of the banked
> dark-class score*. If its median is **flat** across those bins — spread small compared with the
> factor-2.3 growth the score exhibits — then there is no z-structure in the window for the tilt to
> be attributed to, and H1 is refuted regardless of the asymmetry's absolute size or sign.
>
> Secondary falsifier: if the marginal z-trend in CV runs the *wrong direction* (asymmetry shrinking
> with z), H1 is refuted in sign as well as in shape.

Both falsifiers fired. §5.

## 3. Evidence at intake — stage-0 reads, adjudicated

Three parallel reads (analytic / census / coupling) and one adversarial verifier ran. **Where a read
and the verifier conflict, the verifier governs and the read's number is recorded as refuted, not
reported.** Every number below is tagged: **✓CHAIR** = independently re-derived by this chair on its
own code path this session (script:
`/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/cc9f013f-039a-498e-8252-edc212df148c/scratchpad/wgeo_chair_census.py`,
read-only); **✓VER** = confirmed by the adversarial verifier but *not* chair-re-derived.

### 3.0 Dataset pin — DISCHARGED [LOCAL] ✓CHAIR

| item | value | source |
|---|---|---|
| file | `darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv` | — |
| size | **1 681 954 844 bytes** | `ls -l`, this session |
| md5 | **`c52c13b5cab61f6b3f04bbe202550969`** | `md5sum`, this session |
| cluster copy of record | byte-identical (verified 2026-08-26) | carried pin, CLAUDE.md dataset-pinning rule |

No mismatch; the STOP gate did not trigger.

### 3.1 Index semantics and the prune chain — REVALIDATED [LOCAL] ✓CHAIR

The chair rebuilt the `GalaxyCatalogueHandler` chain independently — R&V15 map
(`handler.py:1368-1382`, constants `:37-44`) → NaN-drop (`:1131-1134`) → `_mass_redshift_prune_mask`
(`:215-251`, called at `:352` with `M_min = 1e4`/`M_max = 1e7` from `constants.py:125-126` and
`z_max = 1.5` from `handler.py:30`) → the single `reset_index()` at `:555`:

- `n_raw = 22 641 048`; after NaN-drop `21 753 847`; **`N_pruned = 20 834 171`**.
- All three pinned positional indices reproduce **exactly** (`==`, not `isclose`, on `BH_MASS`):
  `6791138` → 709 540.708756878 / 894 866.2758100418; `6791158` → 709 540.708756878 /
  1 570 331.1654161075; `6791151` → 223 872.11385683485 / 291 758.99489010876.
- The [P3-MKER] exhibit's `CV = BH_MASS_ERROR/BH_MASS` at `6791151` = **1.3032395587986776**,
  matching R2.7(ii)'s banked σ_ln = 1.3032 to quoted precision.

**Consequence for the claim's own premise:** `CV ≡ σ_ln` *identically, by construction* — the code
builds `BH_mass_error = BH_mass · √(Var[ln M_BH])` with every constant in ln-space
(`handler.py:1376-1381`, `:37-44`) ✓VER. This is the first-order **linearization** σ_lin = M·σ_ln,
**not** the exact log-normal moment relation CV_exact = √(exp(σ_ln²)−1); at the exhibit's σ_ln the
exact relation would give 2.113, ~1.6× larger ✓VER.

### 3.2 The window's geometry, closed-form [INFER, verifier-re-derived from scratch] ✓VER

With `M ≡ BH_MASS`, `k = 1.5` (production `sigma_multiplier`, single call site
`bayesian_statistics.py:4691`; `_bh_mass_error_multiplier = k` in the adopted `"symmetric"` mode,
`handler.py:654-661` — both chair-read at source), the candidate window is
`W = [M(1−k·CV), M(1+k·CV)]`, linear-symmetric.

Read in ln M: upper half-width `w_up = ln(1+x)`, lower half-width `w_lo = −ln(1−x)`, `x := k·CV`, so
`A(x) = [ln(1−x)+ln(1+x)] / [ln(1+x)−ln(1−x)] = ln(1−x²)/ln[(1+x)/(1−x)]`, strictly negative and
monotone decreasing on (0,1), verified against the direct half-width form to max abs diff
**1.776e-15** and monotone over 200 000 points ✓VER. Spot values: A = −0.050084, −0.152350,
−0.261860, −0.388184, −0.564023, −0.635421 at x = 0.1, 0.3, 0.5, 0.7, 0.9, 0.95 ✓VER.
**Negative-lower-edge threshold: CV ≥ 1/k = 0.6666666666666666** ✓VER ✓CHAIR — above it the linear
lower edge is non-positive and the "too heavy" leg of the mask (`handler.py:668-672`) is vacuous.

### 3.3 The catalogue census — the CV distribution [LOCAL] ✓CHAIR ✓VER (agree digit for digit)

Over all `N = 20 834 171` pruned galaxies:

| statistic | min | p10 | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|---|
| **CV ( = σ_ln)** | 0.5930 | 0.7846 | 0.7906 | **0.8614** | 0.9401 | 1.2137 | 9.5220 |
| **asym** ≡ `k·CV − ln(1+k·CV)` | — | 0.3990 | 0.4039 | **0.4626** | 0.5304 | 0.7837 | 11.5562 (mean 0.5276) |

**Negative-lower-edge fraction: 0.996112** ✓CHAIR ✓VER. The window is one-sided in the linear
variable for **99.61 %** of the pruned catalogue.

*Definition note, stated because it is the chair's, not the record's:* `asym` is the verifier-adopted
scalar `k·CV − ln(1+k·CV)`. It is a **strictly monotone transform of CV**, which is why
spearman(z, asym) and spearman(z, CV) are numerically identical (−0.6521 both) — the asymmetry scalar
carries **no information beyond CV** ✓VER. It is retained only because it is on the same scale as the
banked R2.7(ii) edge factors, which it reproduces exactly at the exhibit (2.954859338198016 vs the
card's 2.955; 7.062925470001435 vs 7.06) ✓VER.

### 3.4 The z-structure — marginal [LOCAL] ✓CHAIR

- spearman(z, CV) over the full catalogue = **−0.6521** (n = 20 834 171) ✓CHAIR ✓VER — **shrinking**
  with z, i.e. the **opposite sign** from what H1 requires.
- spearman(M_BH, CV) = **−0.6558**; spearman(M_BH, z) = **+0.7398** (the flux-limit confound) ✓CHAIR.
- The trend survives conditioning on mass but weakens: population-weighted mean rho(z,CV)|M =
  **−0.4478** at 0.05 dex (35 bins, 20 788 060 galaxies) and **−0.4477** at 0.01 dex (44 bins,
  20 720 823 galaxies = 99.5 % of the catalogue) ✓VER. It does not vanish and does not flip.

### 3.5 THE DECISIVE MEASUREMENT — the asymmetry in the tilt's own z-bins [LOCAL] ✓CHAIR ✓VER

Bin edges taken verbatim from the banked dark-class score table
(`docs/derivations/population_mismatch_dark_score.md:41-46`), scores from that table and
`BIAS_HISTORY_LEDGER.md:1345`:

| z bin | n (pruned catalogue) | median CV | median asym | **banked dark-class score** |
|---|---|---|---|---|
| 0.075–0.392 | 18 983 740 | 0.8372 | 0.4423 | **+0.014** |
| 0.392–0.559 | 259 403 | **0.7846** | **0.3990** | **−0.465** |
| 0.559–0.659 | 1 189 | **0.7846** | **0.3990** | **−0.743** |
| 0.659–0.753 | 800 | **0.7842** | **0.3987** | **−0.902** |
| 0.753–1.018 | 1 251 | **0.7842** | **0.3987** | **−1.081** |

**Across the four tilted bins the window asymmetry is FLAT: spread 0.0003 on a median of 0.399, i.e.
0.08 %** — while the score it would have to explain grows by a factor **2.3**. Median CV over the
same four bins is likewise flat (0.7846 / 0.7846 / 0.7842 / 0.7842).

Supporting decay profile ✓CHAIR — the entire marginal −0.6521 is a **low-z** phenomenon:

| floor | z ≥ 0.00 | ≥ 0.05 | ≥ 0.10 | ≥ 0.15 | ≥ 0.20 | ≥ 0.25 | ≥ 0.30 | 0.4 ≤ z < 1.0 |
|---|---|---|---|---|---|---|---|---|
| spearman(z, CV) | −0.6521 | −0.6457 | −0.5943 | −0.4806 | −0.3671 | −0.3187 | −0.1655 | **−0.1703** (n = 240 520) |

### 3.6 Two independent structural kills, both at source [LOCAL/DOC] ✓CHAIR

1. **The window cannot touch the class the tilt is measured on.** `handler.py:646` builds
   `candidate_hosts_without_bh_mass` with the **redshift filter only**; `mass_filter_mask`
   (`:663-673`) is applied *afterwards* at `:674`. The dark class C-C is defined as
   `L_cat_no_bh == 0` (`PREREG_COMPLETION_CLASS_DECOMPOSITION.md:20-22` [DOC]) — zero candidates in
   the cone *before* the mass window runs. **A filter that only subsets an already-nonzero candidate
   set cannot be the mechanism for a statistic computed on events where that set is empty.**
   Chair-verified at source, lines read this session. *The 2026-07-30 record already knew the
   adjacent fact and used it against HB:* "the 'why is 1D spared?' screen has no discriminating
   power for mass hypotheses (**1D never sees the mass window** — `handler.py:592` vs `:605`)"
   (`HANDOFF_20260730.md:88-90` [DOC]; those line numbers have since drifted to `:646` vs `:663`).
2. **The tilt is channel-identical, and the 1D leg never sees the mass window.** Row #137
   (`BIAS_HISTORY_LEDGER.md:1338-1346` [DOC]): the pure-completion class sits at "1D mean **0.6001**"
   and "2D identical (**C-C 0.6004**) ⇒ the base tilt is **NOT** mass-channel structure." A mechanism
   absent from the 1D leg cannot produce a tilt numerically identical in the 1D leg ✓VER.

### 3.7 The catalogue is nearly empty where the tilt lives [LOCAL] ✓CHAIR

Of 20 834 171 pruned galaxies, **240 906 (1.16 %)** sit at z ≥ 0.4 and **5 069 (0.024 %)** at z ≥ 0.5;
of the latter only **463** have M_BH in [1e4, 1e7]. High-z events are overwhelmingly dark — and the
mass window never runs on a dark event (§3.6.1).

### 3.8 The fleet cannot speak to the tilt regime at all [LOCAL] ✓CHAIR

Across all 48 arms under `p3_2d_fleet_20260825/` (`prepared_cramer_rao_bounds.csv`, column `z_true`,
9 600 rows): **z_true ∈ [0.004890, 0.340081]**, mean 0.118790, median 0.113166, and **0 events at
z ≥ 0.4** ✓CHAIR. The posterior-joined subset the census read used is 2 261 rows over 24 `b*` arms
with the same maximum, 0.340081 ✓VER.

The census read labelled its fleet result "independent event-level confirmation" — fleet pass
fraction 0.9577 (2 154 066 passed, 95 165 excluded), spearman(z_true, frac_passed) = **+0.2271**,
p = 1.47e-27, n = 2237 ✓VER. **That confirmation is real but out of band:** every event in it sits
*below* the z ≈ 0.4 threshold at which the banked score first departs from zero (+0.014 in the
0.075–0.392 bin). The card records it as a low-z fact, not as evidence about the tilt.

### 3.9 The direction of the geometry switch is the OPPOSITE of the reads' framing [LOCAL] ✓CHAIR ✓VER

Reads 1 and 2 framed the linear window as compressed on the high-mass side and stretched on the
low-mass side, hence *narrower* than a log window. **Operationally that is backwards.**

- **✓VER (cone-exact, whole fleet, 4 800 event rows):** n_lin/n_all = 0.9490, n_log/n_all = 0.4210,
  **n_log/n_lin = 0.4437**. Linear failures split **112 416 623 too-LIGHT** vs **3 868 708 too-HEAVY**
  — **29 : 1**.
- **✓CHAIR (independent, different scope — one banked GW interval, whole pruned catalogue):** using
  the chair-banked GW floor/ceiling for seed 900121 event 20
  (1 237 046.5023702232 / 1 265 461.692070722, `CLAIM_P3_MKER_20260826.md` R2.3), linear passes
  20 415 268 (0.9799), log passes 6 174 863 (0.2964), **n_log/n_lin = 0.3025**; linear failures split
  388 839 too-light vs 30 064 too-heavy = **12.93 : 1**; log failures are overwhelmingly **too-heavy**
  (14 623 829 vs 35 479).

The two scopes differ (cone-restricted fleet vs single-interval whole-catalogue), so the *ratios*
are not the same number and are not presented as one — but **the sign and the mechanism agree
independently**: because the linear lower edge is non-positive for 99.61 % of the catalogue, the
too-heavy leg is vacuous, and a log-symmetric window **re-introduces a heavy-end cut** against a
pruned catalogue in which **46.38 %** of rows exceed M_BH = 1e7 (median 7.961e6, p75 1.648e7,
p90 2.523e7, max 3.564e10) ✓CHAIR while GW masses sit near 1.3e6.

Both measurements independently reproduce, in new venues, HB's banked 2026-07-30 census of
**"193 low-side rejections vs 1 high-side"** (`HANDOFF_20260730.md:105-107` [DOC]).

### 3.10 What the z-trend actually is [INFER] ✓VER (not chair-re-derived)

Exact decomposition, verifier-checked to 1.776e-15 over all 20 834 171 rows:
`CV² = C0² + [d_beta·(ln M_BH − alpha)/beta]² + (beta·r)²` with
`C0 = √(sigma_int² + d_alpha²) = 0.5825130720169441` and `r ≡ STELLAR_MASS_ABSOULTE_ERROR/STELLAR_MASS`
— the middle term is an exact function of M_BH because `handler.py:1371` makes
`ln(M*/10) = (ln M_BH − alpha)/beta`. Therefore **at fixed M_BH the only free degree of freedom is
r**, and rho(z,CV)|M ≡ rho(z,r)|M identically ✓VER.

`r` is a heavily quantized catalogue-reported quantity: r = 1/2 for 32.37 % of rows, 2/3 for 14.41 %,
1 for 9.79 %, 3/5 for 8.88 %; the top 12 values cover 84.2 %. Its z-trend: median r = 1.2500 (z<0.05)
→ 1.0000 → 0.6923 → 0.6000 → 0.5000 (z>0.2, flat thereafter) ✓VER.

> **So "the z-structure of the window asymmetry" is, mechanically, the z-structure of GLADE's quoted
> stellar-mass-error significant figures, mediated by flux limiting.** It is real and
> selection-caused, not a measurement artefact — but it lives **below z ≈ 0.2** and flatlines above.

A corollary worth stating: `CVdet` (the CV a galaxy would have with zero stellar-mass error) has
median 0.5974, p90 0.6596, max 1.1687 — nearly pinned at the floor C0 = 0.5825 ✓VER. **The small-CV
regime in which a linear window approximates a log window is never reached anywhere in this
catalogue** (minimum CV = 0.5930 ✓CHAIR); the linearization error is O(1) fleet-wide, not
perturbative.

### 3.11 Figures REFUTED at stage 0 — recorded so they cannot re-enter

- ✗ *"the window is compressed high / stretched low, therefore narrower"* (reads 1 and 2). Refuted
  operationally, §3.9, in two independent scopes.
- ✗ *"negative (shrinking) in every mass bin with n > 50 000"* (census read). Refuted at finer
  resolution: at 0.05 dex, bin log M_BH 6.60–6.65 has n = 136 026 and rho(z,CV) = **+0.3836**; bin
  6.65–6.70 has n = 37 398 and rho = +0.1846 ✓VER. An artefact of 0.25-dex binning. *The overall sign
  conclusion is unaffected (§3.4).*
- ✗ The census read's `binned_median_CV_by_mass_Msun` table, presented as the catalogue's mass
  structure: its 12 bins span [1e4, 1e7] and cover only **53.6 %** of the pruned catalogue, because
  the prune (`handler.py:248-250`) is an **interval-overlap** test, giving an effective upper cut of
  ~M_max/(1−CV) ≈ 5·M_max at typical CV, and none at all for CV ≥ 1 ✓VER ✓CHAIR (46.38 % above 1e7).
- ✗ *spearman(z_true, n_excluded) = −0.0603, n = 2237* (census read). Does not reproduce; the
  verifier gets **−0.0533, p = 0.0113, n = 2261** ✓VER. Direction unchanged, immaterial to any
  verdict.
- ✗ *"the window's H₀ contribution is UNCONSTRAINED in the current record"* (coupling read). Refuted
  — see §4.1. A bound exists and the read missed it.
- ✗ *"Rule-1 check PASSED"* (coupling read). Refuted for the bias framing — see §4.1.
- ✗ *"165 galaxies all-sky at z ≥ 0.5"* as the production figure (coupling read, citing
  `docs/H0_BIAS_RESOLUTION.md:1383-1388`). Accurate to its source, but that source uses the narrower
  M_BH ∈ [10^4.5, 10^6] band; under the production prune it is **5 069 at z ≥ 0.5, of which 463 in
  [1e4, 1e7]** ✓CHAIR. Amend the number, not the conclusion.
- ✗ *"treat CV ≥ 1/k as A ≡ −1 in effect"* (analytic read). Overstated near the threshold:
  A(x = 0.999) = −0.8177, A(1−1e−9) = −0.9353 — the approach is logarithmically slow ✓VER.
  Immaterial here, since 99.61 % of the catalogue is already past the threshold.

---

## 4. Rule-1 delimitation against the standing exonerations

Hard rule 1 requires checking **both** layers: the local `## Exonerated` list in
`CLAIM_2D_BIAS_20260730.md` **and** §2 of `gate_b_20260730/BIAS_HISTORY_LEDGER.md`. The binding set
is the union. This section is the reason the card's verdict is written the way it is.

### 4.1 [WGEO]-H1 **DOES** collide with a standing exoneration: **HB**. This is decisive.

`CLAIM_2D_BIAS_20260730.md:732-734` [DOC], chair-read this session, verbatim:

> **HB** hard mass window as support truncation (tilt −0.317 nats = 0.063% of the target,
> sign-inverted, 40–50× too small).

`BIAS_HISTORY_LEDGER.md:135` carries **HB** by name in its own §2 list. The measurement is at
`HANDOFF_20260730.md:85-88` [DOC]: "the truncation's h-tilt is −0.317 nats over 0.73→0.81 =
**0.063 %** of the 504.8-nat target, ~50× too small at its ceiling, and **sign-inverted** (removing
the window moves the MAP *up* by ~+0.0015)." Strengthened at `ADJUDICATION_20260730.md:275`: "HB (its
hard-zeros are worth 1.5 % of the target)".

**And HB's banked rationale is [WGEO]-H1 almost verbatim** (`HANDOFF_20260730.md:102-109` [DOC],
chair-read):

> "the window … is a HARD cut where the mass information should enter as a soft kernel weight. The GW
> mass is near-exact … while the catalogue mass error is huge (median σ_lnM = 1.28 ≈ 0.56 dex), so
> the upper leg is vacuous (M − σ_M < 0) and only a hard LOWER floor bites: pooled 193 low-side
> rejections vs 1 high-side. The window bounds depend on z_min/z_max, which depend on h ⇒ the
> truncation is h-dependent by construction, a candidate source of the −504.8-nat dark tilt."

The negative lower edge, the one-sidedness, the low-side/high-side census, and the h-dependence
argument were **all banked on 2026-07-30, measured, and self-refuted by their own investigator.**
The coupling read's rule-1 check passed only because it checked "candidate-window membership" and
"mass-kernel family" and stopped two lines short of HB.

> **BINDING CONSEQUENCE: [WGEO] must not be banked as a bias-attribution claim. Doing so would
> re-litigate HB — this project's most expensive documented failure mode.** The card is therefore
> banked as a null that **corroborates** HB in two new venues (§3.9's 29:1 and 12.93:1 both reproduce
> HB's 193:1), never as a challenge to it.

### 4.2 [WGEO] **CAN** be delimited from the freshly-adopted symmetric window (rows #198–#202). NOT BLOCKED.

The task asked the chair to mark the card **BLOCKED-PENDING-AUTHOR** if this lead cannot be
delimited from the two-day-old author decision. **It can be, and the card is not blocked.** The
delimitation, verified at source this session:

- Rows #198–#202 (`BIAS_HISTORY_LEDGER.md:2925-2935`, adopted `[PHYSICS]` cf4f8a2a) adopted symmetry
  on the axis of **which side's uncertainty gets the multiplier**: "symmetric = `sigma_multiplier`
  applied to BOTH the GW mass uncertainty and the galaxy `BH_MASS_ERROR`". The code comment at
  `handler.py:648-661` states the scope in its own words: *"'symmetric' (default) scales BH_MASS_ERROR
  by the SAME sigma_multiplier as the GW side on both window sides … **Scope: the MASS filter only.**"*
- [WGEO]'s axis is **linear-vs-log geometry within one side's error model** — the *shape* of `W`, not
  which quantity is multiplied by k. These are orthogonal: switching the shape changes eligible
  counts by a factor 2–3 (§3.9) while the symmetric/asymmetric flag changes only the galaxy-side
  half-width.
- Row #202 itself flags this exact gap as an **un-opened thread** ("the filter-vs-kernel
  model-consistency question, proposal §6 caveat 2") — i.e. the record already records that #198–#202
  did **not** decide it.

> **Nothing in this card disturbs the rows #198–#202 ruling, and nothing in it may be presented as
> disturbing it.** The symmetric window remains adopted production.

### 4.3 Also delimited, and also not this card's to overturn

- **"candidate-window membership"** (exact removal moves MAP 0.81 → 0.82, wrong sign) —
  `CLAIM_2D_BIAS_20260730.md:726-727`. A coarse binary lever (window OFF vs ON), not the window's
  internal geometry. Not overturned here.
- **"mass-kernel family" (bounded +0.002)** — `CLAIM_2D_BIAS_20260730.md:727`, rows #72 (`:92`) and
  #89 (`:109`). A different object: the functional form of the mass-marginalization *kernel* inside
  `bayesian_statistics.py`, applied to candidates that already survived eligibility. Not overturned
  here.
- **[WGEO]-H2 (correctness) is already owned and is not a new thread.** Row #206 opened [P3-MKER]
  item (b) ("the eligibility window's k = 1.5 should be an ε-derived truncation bound on the
  corrected kernel, not a physics choice"); `CLAIM_P3_MKER_20260826.md` R2.7(ii) banked the
  linear-vs-log observation; **D-MKER-2** (R2.11) is a pending author authorization to pre-register
  exactly that window-**geometry** measurement, including the negative-lower-edge symptom.
  **[WGEO] adds no new thread there and should not become one.**

### 4.4 Two different perturbations, not an inconsistency

> **[OPUS-ORCH 2026-08-27, corrected]** The original text of this section described "exact removal
> moves MAP 0.81→0.82" (+0.010) and "removing the window moves the MAP up by ~+0.0015" as both being
> "the window-removal counterfactual" and asserted an unexplained "factor ≈ 7" between them. That was
> a conflation introduced by this card, not a genuine record inconsistency — §4.3, two paragraphs
> above, already describes them correctly as separate objects. It has also mis-cited the membership
> figure's location; corrected below. Numbers and verdicts are unchanged; only this section's own
> framing is amended.

The two figures are **different perturbations of the same code object**, not two measurements of one
counterfactual, so there is no discrepancy to resolve:
- **+0.010** — "exact removal moves MAP 0.81→0.82" — is removal of *realization-added candidates*
  (`CLAIM_2D_BIAS_20260730.md:726-727`; `HANDOFF_20260730.md:63-64`). This is the
  [WINDOW-MEMBERSHIP] object: a coarse binary lever (window OFF vs ON).
- **+0.0015** — "removing the window moves the MAP up by ~+0.0015" — is removal of *the window itself*
  (`HANDOFF_20260730.md:87-88`). This is the HB object.

Both are sign-inverted relative to the direction a bias-driver fix would need; neither needs
reconciling against the other, since they are not the same measurement.

### 4.5 Citation drift, for the record

`BIAS_HISTORY_LEDGER.md:129` cites the exoneration list as `CLAIM_2D_BIAS_20260730.md:191-204`. That
citation has drifted: the list now begins at **line 721** of that file, with HB at **:732-734**
✓CHAIR. Content unchanged; only the pointer is stale.

`[OPUS-ORCH 2026-08-27, corrected]` This section originally gave the pointer's own location as
`:130`; it is at `:129`. The correction is itself an instance of the drift the section documents —
an off-by-one repeated across several documents that all copied it from one another rather than
from the source.

---

## 5. VERDICT — **LEAD-DEAD**. H1 is refuted at stage 0.

> **THE DECISIVE NUMBER.** Across the four banked dark-class tilt z-bins, the median window asymmetry
> `k·CV − ln(1+k·CV)` (k = 1.5) is **0.3990 / 0.3990 / 0.3987 / 0.3987** — a spread of **0.0003
> (0.08 %)** — and median CV is **0.7846 / 0.7846 / 0.7842 / 0.7842**, while the banked dark-class
> score over those same bins runs **−0.465 / −0.743 / −0.902 / −1.081** (a factor **2.3**).
> Re-derived by this chair over the full 20 834 171-row pruned catalogue rebuilt from
> `reduced_galaxy_catalogue.csv` (md5 `c52c13b5cab61f6b3f04bbe202550969`, pin verified; three pinned
> `catalog_index` rows reproduced to full float precision). Bin edges:
> `docs/derivations/population_mismatch_dark_score.md:41-46`. Score values: that table and
> `BIAS_HISTORY_LEDGER.md:1345`.

The Refute-by(H1) of §2 fired on its primary clause. Four independent reasons, any one of which is
sufficient:

1. **No shape.** In the tilt's own z-band the asymmetry is flat to 0.08 % against a factor-2.3
   signal. *There is not merely a wrong-signed structure — there is no structure to have a sign.*
2. **Wrong sign where structure exists.** The marginal trend (−0.6521) is *shrinking* asymmetry with
   z, and it decays monotonically to −0.1655 once z < 0.3 is dropped; restricted to 0.4 ≤ z < 1.0 it
   is −0.1703. What structure exists is a **low-z** phenomenon and is mechanically GLADE's quoted
   stellar-mass-error quantization (§3.10), not window geometry.
3. **Structurally impossible.** The dark class C-C is `L_cat_no_bh == 0`, fixed by the sky/z cone at
   `handler.py:646`, strictly upstream of `mass_filter_mask` at `:663-674`. The window never runs on
   a dark event (§3.6.1).
4. **Channel-identical tilt.** Row #137: 1D mean 0.6001 vs 2D C-C 0.6004; the 1D leg never sees the
   mass window; the row's own conclusion is "the base tilt is NOT mass-channel structure" (§3.6.2).

**Confidence: high.** The verdict rests on a chair-re-derived measurement over the complete
catalogue, not on a sample or a model.

**Stated symmetrically, as the discipline requires:** this project has repeatedly been burned by a
reconciling quantity that *agreed* in shape and then failed to predict both arms (row #145's −1e300
sentinel; the D-1 mismatch withdrawn as underpowered, row #144). The same caution applies to a null.
Here the shapes **actively disagree** — which is the more decisive kind of null — and the null is
reinforced by two independent *structural* arguments that do not depend on any statistic at all
(§5.3, §5.4). That is why the verdict is high-confidence rather than provisional.

**What is NOT concluded:** that the window is correct; that a linear-symmetric window is the right
shape for a log-normal error model; that the window has no H₀ effect of any size. See §6.

---

## 6. What is NOT measured (honest statement)

1. **No H₀ effect of anything in this card is measured.** Not of the linear→log geometry switch, not
   of the window's z-structure, not of any counterfactual. **The only banked H₀ bound on this object
   is HB's** (−0.317 nats = 0.063 % of the target, sign-inverted, ~50× too small at its ceiling), and
   HB measured **window removal**, which is a different perturbation from a **shape change**.
2. **The geometry switch is NOT small, and this is verifier-only.** ✓VER (not chair-re-derived): the
   linear→log-symmetric counterfactual shifts the eligible set's mean redshift by a median
   **−2.259e-02 (−14.5 % relative)**, mean −3.596e-02, p5 −1.202e-01, max|·| 1.856e-01, with |shift|
   growing with event z_true (−8.967e-03 at z < 0.05 to −4.331e-02 at z 0.3–0.4) — **12.2×** the
   window-*removal* perturbation on the same metric (median −1.248e-03). **This number is tagged
   ✓VER and must be chair-re-derived before it is used quantitatively anywhere.** It is reported
   because suppressing it would be dishonest, not because it is bankable.
3. **The stage-L R0 sweep mandated at every new stage-0 thread was NOT run fresh.** The thread is
   closing as a null at stage 0, and the R0 ring for exactly this object was swept two days ago by
   [P3-MKER] §6, whose **LIT-4** is a reportable absence squarely on [WGEO]'s subject: *"No cited
   selection-cut/truncation-bias warning has ever been checked against the k = 1.5 hard pre-filter …
   the cut-on-observed-vs-cut-on-true question is untouched."* That absence is **carried forward
   unresolved**, not discharged by this card.
4. **The window's effect on the C-A/C-B (catalogue-supported) classes is not quantified.** §5.3 rules
   the window out for the C-C class specifically. It says nothing about the other classes, nor about
   the separately-material [P3-WBHZERO] channel (rows #200–#202).
5. **The fleet cannot test the tilt band.** Zero of 9 600 banked CRB rows sit at z ≥ 0.4 (§3.8). Any
   future event-level test of a high-z window effect needs a fleet that does not yet exist.
6. **The `get_redshift_outer_bounds` dead-`sigma_multiplier` defect** (`physical_relations.py:546-567`;
   body hardcodes `3 *`) is known, filed as D-MKER-3, respected here, and **not re-litigated**. Every
   window bound in this card that depends on it is either chair-taken from a banked production call
   or explicitly scoped as a single-interval illustration.
7. **`asym` is a chair/verifier-adopted scalar, not a record-banked definition.** It is a monotone
   transform of CV and carries no independent information (§3.3).

---

## 7. STAGE-1 INFORMATION FORECAST

*What each candidate next measurement would tell us, what it costs, what outcome would change a
decision, and what outcome would not — stated for the author BEFORE any compute is spent.*

### F-1 — The log-space-window counterfactual. **NOT WARRANTED as a [WGEO] action. Belongs to D-MKER-2 if the author authorizes it there.**

| | |
|---|---|
| **What it would tell us** | The H₀ effect of switching the eligibility window from linear-symmetric to log-symmetric at fixed k — the *only* construction found so far that flips the [P3-MKER] exhibit (R2.7(ii)). |
| **What it needs** | **Its own instrument.** No `window_geometry ∈ {linear, log}` flag exists; `handler.py:663-673` hard-codes the linear form. Building it is a counterfactual-flag change in a physics-trigger file ⇒ `/physics-change` hard gate, plus rule [A13]: a registered engagement threshold and an assertion that the flag reaches **every** dispatch path production uses. |
| **Cost** | Instrument + gate package: ~1 session, zero compute. Validation: a **48-arm fleet re-run** (the [P3-2D] fleet's cost) — and that fleet, by §3.8, contains **zero events above z = 0.34**, so it cannot test the regime [WGEO] was about. |
| **What outcome would change a decision** | A geometry switch moving the joint MAP by more than HB's banked ceiling (0.063 % of target) **with a demonstrated causal route to a class the window actually acts on** ⇒ a genuine `[PHYSICS]` question. |
| **What outcome would NOT change a decision** | Any eligible-count or mean-redshift shift, however large. §6 item 2 already shows those are large (−14.5 %) and they are **not H₀ statements**. Confusing an eligibility-set moment with a posterior effect is precisely the error the record's exoneration list guards against. |
| **Verdict** | **Do not fund under [WGEO].** If it runs at all it runs as [P3-MKER]'s D-MKER-2, whose scope already covers it, and it must be sequenced behind F-2. |

### F-2 — **BOUND-FIRST: can the window's H₀ contribution be bounded cheaply, before any mechanism work?** **PARTIALLY ALREADY DONE — and the cheap remaining piece is a records question, not a measurement.**

| | |
|---|---|
| **What already exists** | HB's bound: **−0.317 nats = 0.063 % of the 504.8-nat target, ~50× too small at its ceiling, sign-inverted** (`HANDOFF_20260730.md:85-88`), plus "hard-zeros worth 1.5 % of the target" (`ADJUDICATION_20260730.md:275`). This is a banked bound on **this object**, which the coupling read missed. |
| **What is missing** | (a) the bound covers **window removal**, not **window shape**; (b) the record quotes the removal effect at two magnitudes differing by ≈7× (§4.4), so the bound is not currently quotable as a point value. |
| **Cheapest decisive action** | **Resolve §4.4 from banked artifacts.** Both figures (+0.010 vs +0.0015) trace to 2026-07-30 artifacts already on disk. Reconciling them is a **zero-compute records read** (rule [A1]) of ≤ 1 h. |
| **What outcome would change a decision** | If the true removal effect is the larger (+0.010) and if a shape switch is plausibly ~12× a removal (§6 item 2), the implied ceiling stops being negligible ⇒ the correctness thread gains a *quantitative* motivation it currently lacks. |
| **What outcome would NOT change a decision** | Either resolution leaves H1 dead — the bound is about magnitude, and H1 died on **shape** (§5), which no bound can revive. |
| **Verdict** | **Warranted, cheap, and the only [WGEO]-adjacent action that is worth funding.** It is a bound-first action in the exact sense the stage-1 forecast asks for. |

### F-3 — Measurements the stage-0 reads showed are **NOT worth running**

1. **A z-resolved sweep of window pass-rate on the existing fleet, to test the tilt band.**
   **NULL BY CONSTRUCTION** — the fleet has 0 events at z ≥ 0.4 (§3.8). *Cost avoided: an
   uninterpretable answer that would have looked like a measurement.*
2. **A finer census of CV vs z or CV vs M_BH.** **SUPERSEDED BY AN IDENTITY.** §3.10's exact
   decomposition shows that at fixed M_BH the only free d.o.f. is the catalogue's quoted `r`, which
   is quantized to 12 values covering 84.2 % of rows. Further binning measures GLADE's significant
   figures, not physics. *Cost avoided: repeated 30 s full-catalogue passes with no decision value.*
3. **Re-deriving the asymmetry scalar in another functional form.** **ZERO INFORMATION.** Any such
   scalar is a monotone transform of CV and produces the identical rank statistic (−0.6521 both,
   §3.3).
4. **Building a C-C-class window counterfactual.** **STRUCTURALLY IMPOSSIBLE** — C-C is defined by
   `L_cat_no_bh == 0`, upstream of the window (§3.6.1). There is no arm to build.
5. **A new stage-2 pre-registration under the tag [WGEO].** **DUPLICATIVE** — D-MKER-2 already covers
   the surviving correctness question and awaits the author (§4.3).

### F-4 — RECOMMENDED NEXT ACTION (one)

> **[OPUS-ORCH 2026-08-27, corrected] RESOLVED.** §4.4 has been corrected: +0.010 and +0.0015 are
> different perturbations (membership removal vs window removal), not two measurements of one
> counterfactual, so there was no "factor ≈ 7" to resolve. HB's bound is quotable as a point value
> per entry already; see `EXONERATION_REGISTER_20260827.md`'s HB entry, which now also names each
> denominator inline.

> Resolve the §4.4 record inconsistency in the window-removal counterfactual (+0.010 vs +0.0015,
> factor ≈ 7, both sign-inverted) from banked 2026-07-30 artifacts — a zero-compute records read —
> so that HB's bound becomes quotable as a point value.

Rationale: it is the cheapest action in the forecast, it is pure re-reading of artifacts already on
disk (rule [A1] — exhaust free re-reads before requesting compute), it repairs the **only** banked
H₀ bound on the eligibility window, and it is a prerequisite for the author's decision on D-MKER-2.
It does **not** revive [WGEO]-H1 under any outcome, and this recommendation does not ask for it to.

---

## 8. RECOMMENDATION — **KILL**

**KILL.** The lead does not earn a stage-2 pre-registration. It is banked here as a closed null.

- **[WGEO]-H1 is refuted** on a chair-re-derived, full-catalogue, decisive measurement, reinforced by
  two independent structural impossibility arguments (§5).
- **Banking it as a bias claim would re-open HB**, a standing exoneration, which hard rule 1 forbids
  (§4.1). The null **corroborates** HB in two new venues and is therefore rule-1-safe as written.
- **[WGEO]-H2 (correctness) survives but is not [WGEO]'s** — it is already owned by [P3-MKER]
  R2.7(ii) / item (b) / **D-MKER-2**, and by row #202's un-opened filter-vs-kernel thread (§4.3).
  Opening a parallel tag would fragment an owned question.
- The thread's one genuinely new contribution beyond the null — that the linear→log geometry switch
  is **not small** (§6 item 2), which no banked bound covers — is routed to the **author** as a
  ranked-priority input to D-MKER-2, tagged ✓VER and explicitly *not* bankable until re-derived.

**This card closes `[WGEO]`.** It is append-only from here; the null is the result.

---

## 9. AUTHOR DECISIONS REQUIRED

All four are **fresh** — they rest on evidence created overnight that did not exist when any prior
approval was given, and are therefore **not** covered by any standing grant (binding default,
CLAUDE.md "Approval scope").

- **[RULE] R-WGEO-1 — ratify the KILL and the banking of this card as a closed null.** The window
  asymmetry is flat to 0.08 % across the four banked dark-class tilt z-bins against a factor-2.3
  signal; the marginal trend runs the opposite way and is a low-z, GLADE-quantization phenomenon; and
  the window is structurally downstream of the class the tilt is measured on.
- **[RULE] R-WGEO-2 — ratify the rule-1 finding that HB is the governing exoneration for this
  object,** and the corresponding constraint that no window-as-bias-driver claim may be banked
  without new evidence that engages HB's −0.317-nat/0.063 %/sign-inverted measurement directly. The
  coupling read's "UNCONSTRAINED" and "rule-1 PASSED" are recorded as refuted (§3.11, §4.1).
- **[DO] D-WGEO-1 — authorize the F-4 records read** (reconcile +0.010 vs +0.0015, §4.4). Zero
  compute, ≤ 1 h, read-only, no source change. Its output is a repaired bound, not a new thread.
- **[RULE] R-WGEO-3 — rule on whether §6 item 2 (the linear→log switch moves the eligible set's mean
  redshift by −14.5 %, 12.2× the removal perturbation; ✓VER-only) raises the priority of the pending
  **D-MKER-2**.** The chair takes no position on scope. It notes only that this quantity is an
  eligibility-set moment, **not** an H₀ effect, and that treating it as one would be the exact error
  §7 F-1 warns against.

**Uncertainty flagged, not smoothed.** (i) §6 item 2 and §3.10's decomposition are ✓VER-only and were
not chair-re-derived; every number the *verdict* rests on was. (ii) The chair's §3.9 direction check
uses a single banked GW interval against the whole pruned catalogue and is **not** the same
measurement as the verifier's cone-exact fleet reconstruction — the ratios differ (0.3025 vs 0.4437)
and are reported separately, with only the sign and mechanism claimed as jointly established.
(iii) No stage-L R0 sweep ran fresh for this thread; [P3-MKER] §6's **LIT-4** absence is carried
forward unresolved. (iv) This card was produced without author supervision.

*— end [WGEO] stage 0, CLOSED NULL [OPUS-ORCH 2026-08-27]*

---

## AUTHOR RULINGS [appended 2026-08-28; blanket "all ratified also the thirteen earlier ones", itemization orchestrator-derived]

- **R-WGEO-1 RATIFIED** — the [WGEO] KILL and closed-null banking stand.
- **R-WGEO-2 RATIFIED** — exoneration HB governs this object; no window-as-bias claim may be
  banked without new evidence engaging HB's −0.317-nat / 0.063% / sign-inverted measurement
  directly.
- **D-WGEO-1 APPROVED** — the ≤1h zero-compute records read reconciling the two-magnitude
  window-removal counterfactual (+0.010 vs +0.0015) launched 2026-08-28; result to be appended
  here when it returns.
- **R-WGEO-3 — MOOT** as a priority question: D-MKER-2 (the window-geometry pre-registration)
  is itself approved, so the −14.5% mean-redshift moment no longer gates anything; it remains
  ✓VER-only input to the D-MKER-2 prereg.

---

## D-WGEO-1 RESULT [2026-08-28] — RECONCILED; HB's bound IS quotable

Records read executed (agent) + decisive passages independently re-verified by the orchestrator
(`CLAIM_2D_BIAS_20260730.md:724-734`, `HANDOFF_20260730.md:60-66,83-91,758-762`).

**The factor-7 discrepancy is not a discrepancy: the two numbers are two different exonerated
mechanisms that both involve "removing a window".**

| number | mechanism | space |
|---|---|---|
| **+0.010** (MAP 0.81→0.82) | candidate-window **membership** — removal of realization-added 2D candidates from the search list | search-space (candidate list) |
| **+0.0015** | **HB** — hard mass window as likelihood support truncation | likelihood-space (mass filter) |

HB's two recorded forms agree internally: −0.317 nats × the file's own corrected conversion
4.9e-3 h/nat = **+0.00155 ≈ +0.0015** (the directly quoted ΔMAP). **HB's quotable bound:
ΔMAP ≈ +0.0015 (−0.317 nats = 0.063% of the −504.8-nat dark-class target), WRONG-SIGNED
(removal moves MAP up), 40–50× too small at ceiling.** The +0.010 belongs to
candidate-window membership and must never be substituted for HB's bound.

**Citation drift corrected:** the exoneration list cited as `CLAIM_2D_BIAS_20260730.md:191-204`
(ledger row area, `BIAS_HISTORY_LEDGER.md:129`) now lives at **:721-734** (adjudication addendum
:736-744, venue-scoping caveat :745-751); the ledger's bullet CONTENTS match — only the line
numbers reflowed. Correction noted in the ledger append; per runbook 35 §4.4, anchor text ("HB")
is the primary key, line numbers are conveniences.

**Consequence for D-MKER-2:** un-gated. The window-geometry prereg must engage HB's bound in
this reconciled form (per R-WGEO-2): any claimed geometry effect larger than ~+0.0015 in MAP on
the 2026-07-30 venue contradicts HB's measurement and needs to say why.
