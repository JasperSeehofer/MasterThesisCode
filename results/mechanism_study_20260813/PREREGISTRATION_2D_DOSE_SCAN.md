# Pre-registration (stage 2) — 2D DOSE SCAN: map the interaction surface bias(f_host, f_imp)

**REGISTERED 2026-08-13**, BEFORE any cell is run. Stage-2 successor to
`PREREGISTRATION_MECHANISM_ISOLATION.md` (the parent) and companion to
`AMENDMENT_A1_VM1_NULL_AT_N100.md`. Append-only from this commit: verdicts append below the final
rule, and no line above it may be edited.

**RATIFICATION REQUIRED BEFORE ANY CELL RUNS — two registered ceilings are exceeded.** *(Both were
put to the author and both are **GRANTED 2026-08-13** — the ratification record is §A below. This
block is preserved verbatim as the consult that was made; it is not a live blocker.)*

1. **Budget.** The parent §1 registers *"L0 unlimited (toys), L1 ≤ 5 arms, L2 ≤ 1 arm. Exceeding it
   is a STOP-and-consult, not a silent extension."* This scan is **16 cells**. It is presented as
   the consult, not as an extension taken.
2. **Seed plan.** The parent §1 registers the decade **+50000…+50999**. This scan requires
   **+51000…+52514**, a new adjacent block (§3.2). It is disjoint from every consumed and every
   reserved block, and it is registered here rather than assumed.

**This scan proposes no repair and adopts no candidate.** It measures a surface. Any production
change remains a separate `/physics-change` package downstream.

## A. AUTHOR RATIFICATION — 2026-08-13 — BOTH CEILINGS GRANTED

**Ratified 2026-08-13 on the author's verbatim words:**

> **"all approved as you recommend"**

**What was put to the author.** Three items were on the page when those words were written, and the
author was shown them together. The itemisation below is **orchestrator-derived** from the state of
the two registered documents at that moment — it reconstructs what "as you recommend" referred to,
and it is **not** author dictation. The author's ruling is the seven words quoted above and nothing
else; each item below is ratified because it was on the page, not because the author enumerated it.

1. **Budget ceiling — GRANTED.** The 16-cell scan against the parent §1 registered *"L0 unlimited
   (toys), L1 ≤ 5 arms, L2 ≤ 1 arm. Exceeding it is a STOP-and-consult, not a silent extension."*
   The consult was made (opening block item 1) and is now answered: **the 16 cells are ratified.**
   This is a granted extension of the parent's L1 ceiling for this scan only; it sets no precedent
   and does not raise the parent's ceiling for any other successor.
2. **Seed decade — GRANTED.** The new adjacent block **+51000…+52514** (§3.2), outside the parent §1
   registered decade **+50000…+50999**. **The block is ratified as a registered extension of the
   thread's seed plan**, on the disjointness argument shown in §3.2, which is unit-tested before any
   cell runs and is unchanged by this ratification.
3. **S23 at N = 100 — GRANTED.** The recommendation to register the sole shape discriminator at
   N = 100 rather than N = 15. Implemented throughout this file and recorded, with its full
   arithmetic and its consequences, in §B below.

**Scope of the ratification.** It grants the two ceilings and the S23 design change. It does **not**
adjudicate the parent's branch call, which remains pending and belongs to a separate session; it does
not alter Amendment A1's ±0.002 window or any of its rules; and it does not relax §4.7's anti-tuning
clause, which continues to bind every threshold in this file from this commit onward.

## B. S23 REGISTERED AT N = 100 — a PRE-DATA design change, not a post-hoc adjustment

**Status, stated first because it is the whole legitimacy of the change: NO S23 DATA EXISTS.** No
cell of this scan has been run. This change is made **before any readout**, at registration time,
and is therefore a **design improvement to a pre-registration** — the category the anti-tuning clause
exists to permit. It is explicitly **NOT** a post-hoc band adjustment: nothing here is being moved to
accommodate a number that has been seen. (Contrast Amendment A1 §3, where a derivation written
*after* a readout is expressly barred from rescuing the result it explains. That bar does not apply
here because there is no result.)

**Why the change is worth paying for.** §4.3 registers a structural limitation: H-INT and H-THRESH
are **degenerate on the f_h = 0 row and on the f_h = 1 row**, so all discrimination lives in rows
h = 1 and h = 2, and **S23 is the only cell that clears the 3σ dead-band.** At N = 15 its decision
boundary was midpoint **0.0096667 ± 0.004737**, i.e. the two hypotheses' predictions (0.017333 and
0.002000) sit only

```
(0.017333 - 0.002000)/2  =  0.0076665      half-separation
0.0076665 / 0.001579     =  4.855          in units of the N = 15 cell SE
4.855 - 3                =  1.855 sigma    outside the 3-sigma dead-band
```

**a bare 1.86σ outside the dead-band.** A truth anywhere near the midpoint — the exact case a shape
test is built to resolve — returns **SHAPE-UNDECIDED**, which then triggers the escape that was
*already registered* in §4.3: *S23 alone at N = 100*. The scan would pay for S23 twice and reach the
same place one round-trip later.

**Paying it upfront converts a likely-UNDECIDED first pass into a decisive one.** At N = 100 the
same two predictions sit 12.54 SE from the midpoint and **9.54σ outside** the new dead-band (§4.3
arithmetic). The cost is **85 extra seeds = ≈82 CPU-h** (§3.1), which is exactly the cost the escape
would have charged anyway, minus a second submit-and-wait cycle.

**Everything else in the scan is unchanged.** The fraction grid, the 15 other cells at N = 15, the
code form of §2.1, the corner cross-checks, the DS-D1/2/4/5/6 rules, the branches, and every band not
enumerated in §4.3 stand exactly as first registered.

---

## 0. Why this scan exists

The parent study ran three arms at N = 15 on the campaign decision cell and extracted (parent
*Operational completion record*, mechanical extraction, no scoring):

| arm | dose | 1D bias | SE | 2D bias | SE |
|---|---|---|---|---|---|
| **MN0** | everything | **+0.034667** | 0.001579 | +0.037000 | 0.001604 |
| **MEH** | host only | **+0.004000** | 0.000535 | +0.004333 | 0.000454 |
| **MEI** | impostors only | **+0.000000** | 0.000000 | +0.000000 | 0.000000 |

Two structural facts follow, and they are the reason this file exists.

**(i) The split is strongly non-additive.** MEH + MEI = +0.004000 against MN0's +0.034667. The
registered DS-M5 prediction — *"M5′-CONFIRMED requires **both**: E1-imp bias ≥ 0.030 **and** E1-host
bias ≤ 0.012"* (parent §2) — is **inverted on its decisive half**: the impostor-only arm carries
nothing at all. **M5′ as registered is refuted**, and the toy's split-dose evidence at K = 50
(+0.0247 impostors-only vs +0.0062 host-only, `M5_smeared_candidate_prior.md` §4.3) does **not**
transfer to the instrument at production K.

**(ii) MEI's posterior collapses onto a single grid point.** Not merely unbiased — degenerate: one
exact host redshift, at weight 1/K with K̄ ≈ 1216, overwhelms ~1216 smeared impostors outright. The
toy predicted the opposite (*"at K = 50 it cannot [pin]"*, `M5_smeared_candidate_prior.md` §4.3);
on the instrument the pin is total. **The toy's K-saturation account is falsified at production K**,
and every L0 closure that leaned on the toy's K-behaviour inherits that caveat.

**Therefore the bias is an INTERACTION**: it requires a de-pinned host **and** a smeared impostor
sea *simultaneously*. Neither ingredient alone produces more than 12 % of it. A two-arm split cannot
characterise an interaction; a surface can. This scan measures that surface.

**Registered independence.** This design does **not** depend on the parent's branch call, which is
pending and belongs to a separate session, nor on the outcome of Amendment A1. If A1 returns
A1-FAIL (the null arm genuinely does not reproduce the campaign), this scan is **void along with
every other measurement in the thread** — that dependency is registered in §7 branch 1 and is the
only coupling between the two documents.

## 1. Question of record

> **What is the shape of bias(f_host, f_imp) on the campaign decision cell — and does it identify
> the interaction as multiplicative in the two dose fractions, or as a threshold in the host dose?**

Not asked here, and registered as such in §6: whether any of this transfers to production
`BayesianStatistics`; what the repair is; whether reweighting helps (already answered NO, §5.1).

## 2. Design — the 16-cell grid

**Dose fractions:** `f ∈ (0.0, 0.25, 0.5, 1.0)`, indexed `0,1,2,3`, applied as a fraction of **each
candidate's own GLADE σ_z** — not of a flat dose. The heterogeneous GLADE sampler
(`draw_member_sigma_z`) is unchanged; only a per-candidate scalar multiplies its output.

**Cells:** `S{h}{i}` for `h` = host-dose index, `i` = impostor-dose index. **N = 15 seeds per cell
for 15 of the 16 cells, and N = 100 at S23** (the sole shape discriminator, §B and §4.3), i.e.
15 × 15 + 100 = **325 seeds total** across 16 cells.

| | f_imp = 0.0 | 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|
| **f_host = 0.0** | S00 | S01 | S02 | S03 |
| **0.25** | S10 | S11 | S12 | S13 |
| **0.5** | S20 | S21 | S22 | **S23 — N = 100** |
| **1.0** | S30 | S31 | S32 | S33 |

**All cells not marked otherwise are N = 15.** S23 is the only cell whose N differs, and the reason
is registered in §B: it is the only cell whose two competing predictions separate by enough to decide
the shape question, and at N = 15 it would most likely have returned SHAPE-UNDECIDED and forced the
already-registered N = 100 follow-up anyway.

**Base configuration for every cell:** the campaign decision cell verbatim — pinned 982 events,
`balls="real_k"`, real K_i, GLADE-empirical σ_z sampler, `h_true = 0.730`, canonical 41-point grid
at spacing 0.005, `n_events_cap=None`, `chunk_pairs=16384`, the four parent §1 pins.

### 2.1 The single instrument change — exact code form, fixed at registration

Two new `VenueConfig` fields, replacing nothing:

```python
dose_frac_host: float = 1.0     # fraction of each host's own sigma_z
dose_frac_imp:  float = 1.0     # fraction of each impostor's own sigma_z
```

Both default to `1.0`, which **is** `dose_target="all"`, so the registered campaign path, the
committed MN0 path, and the V-M5 golden are untouched. In `_draw_seed_realization`, after all four
draws, in the registered RNG order (noise → ball → σ_z vector → standard-normal scatter vector):

```python
frac = np.where(host_mask, vcfg.dose_frac_host, vcfg.dose_frac_imp)
ball.z_obs = ball.z_obs + frac * sigma_pairs * noise
sigma_pairs = frac * sigma_pairs
```

**Both lines are scaled, and that is load-bearing** — the identical argument `ARMS.md` makes for
masking both lines. A candidate dosed at fraction `f` has scatter `f·σ_k`, so the estimator **must**
be told its kernel width is `f·σ_k`. Scaling only the scatter would hand the estimator a kernel
wider than the truth, i.e. deliberate misspecification, and would confound the read with the very
thing the campaign's matched-model principle (parent constraint (c)) exists to exclude. At `f = 0`
this reduces to the registered `dose_target` mask exactly, including the point-evaluation branch.

**No estimator code changes.** `_channel_terms_at_h`,
`log_channel_posteriors_ball_sigma_vector` and `_g_ball_capped` are byte-identical across all 16
cells — verifiable by `git diff`. **No production module is modified.** No new randomness is
consumed: `frac` is a pure relabelling of the existing `host_mask`.

This code form is fixed at this commit and may not be adjusted after any cell is read.

### 2.2 Registered null checks (the AR-family, extended)

- **AD-1** — at `(dose_frac_host, dose_frac_imp) = (1,1) / (1,0) / (0,1) / (0,0)` and a fixed seed,
  the realisation (`z_obs`, `sigma_pairs`, `K_sum`) is **bit-identical** to `dose_target =
  "all" / "host" / "impostors"` / a σ = 0 realisation respectively. Unit-tested before any cell.
- **AD-2** — `host_mask.sum() == 982` and `frac` takes exactly two distinct values, for every seed
  and every cell.
- **AD-3** — at a fixed seed, across **all 16** `(f_host, f_imp)` pairs, `K_sum`, `event_idx`, the
  **pre-dose** `z_obs`, the σ vector and the scatter vector are **bit-identical**; only `frac`
  differs. This is the parent's V-M2 for this scan. Verified in a unit test at fixed seed — **not**
  across the campaign, because the cells use disjoint seeds by design (§3.2, §6 item 2).

## 3. Seeds, cost, venue

### 3.1 Cost anchor

From the parent's operational record (array 6303086, 15 cores/task): fully-dosed **0.969
CPU-h/seed** (MN0: 00:58:08 × 15 cores / 15 seeds), impostors-dosed 0.942 (MEI), **host-only 0.039**
(MEH: 00:02:20 — undosed impostors take the point-evaluation branch and the cell is ~25× cheaper).
This is **≈3.9× faster than the campaign's registered 3.79 CPU-h/seed**, plausibly the
author-ratified Route 1 adaptive Gauss–Hermite contraction reaching the validation stack (see
Amendment A1 §7). **0.97 CPU-h/seed is the anchor this scan budgets against; 3.79 is stale.**

| block | cells | seeds | CPU-h/seed | CPU-h |
|---|---|---|---|---|
| `f_imp = 0` column (S00, S10, S20, S30), N = 15 each | 4 | 60 | 0.039 | 2.3 |
| `f_imp > 0` cells at N = 15 (S01–S03, S11–S13, S21, S22, S31–S33) | 11 | 165 | 0.969 | 159.9 |
| **S23 at N = 100** (§B) | 1 | 100 | 0.969 | 96.9 |
| **total** | **16** | **325** | — | **≈259** |

Arithmetic: 60·0.039 = 2.34; 165·0.969 = 159.885; 100·0.969 = 96.900; sum **259.1 ≈ 259 CPU-h**.
Cross-check against the pre-§B design (240 seeds, **≈177 CPU-h**): S23 gains 85 seeds, and
85 · 0.969 = **82.4 CPU-h**, so 177 + 82 = **259**. **The S23 upgrade costs ≈82 CPU-h and the scan
total moves from ≈177 to ≈259 CPU-h.**

Wall clock in the arms' measured shape (15 seeds/task, 15 cores/task): 15 single-task cells plus S23
split over 7 tasks (6 × 15 + 1 × 10 = 100) = **22 tasks**, still **≈1 h** wall clock because the
tasks run concurrently and no task exceeds 15 seeds. Subject to backfill. **`sbatch --test-only` is
registered as non-predictive for this shape** — wrong by four days on array 6303086, the second
recorded instance (EXP-61 discipline).

### 3.2 Seed plan (VT-D7 discipline carried)

`seed(S{h}{i}, j) = 20260808 + 51000 + 100·(4h + i) + j`, with **`j = 0…14` for every cell except
S23, where `j = 0…99`** (§B).

| cell | offset | absolute seeds | | cell | offset | absolute seeds |
|---|---|---|---|---|---|---|
| S00 | +51000 | 20311808–20311822 | | S20 | +51800 | 20312608–20312622 |
| S01 | +51100 | 20311908–20311922 | | S21 | +51900 | 20312708–20312722 |
| S02 | +51200 | 20312008–20312022 | | S22 | +52000 | 20312808–20312822 |
| S03 | +51300 | 20312108–20312122 | | **S23** | **+52100** | **20312908–20313007 (N = 100)** |
| S10 | +51400 | 20312208–20312222 | | S30 | +52200 | 20313008–20313022 |
| S11 | +51500 | 20312308–20312322 | | S31 | +52300 | 20313108–20313122 |
| S12 | +51600 | 20312408–20312422 | | S32 | +52400 | 20313208–20313222 |
| S13 | +51700 | 20312508–20312522 | | S33 | +52500 | 20313308–20313322 |

**S23 seed-block integrity — verified, and stated explicitly so it is checked rather than assumed.**
The map allocates a **100-wide slot per cell** (`100·(4h+i)`), of which the N = 15 cells consume only
the first 15. S23 at N = 100 therefore **exactly fills its own already-allocated slot** and
**cannot** reach into any other cell's:

```
S23 first seed = 20260808 + 52100 +  0  =  20312908
S23 last  seed = 20260808 + 52100 + 99  =  20313007
S30 first seed = 20260808 + 52200 +  0  =  20313008
```

**S23 terminates at 20313007, exactly one seed below S30's first (20313008).** It is a tight
abutment and no seed is shared. Against every other cell the separation is larger still, since each
occupies a distinct 100-wide slot; S23's occupied offsets are **+52100…+52199** and every other
cell's slot is disjoint from that interval by construction. **No collision exists anywhere in the
grid.**

Range **+51000…+52514** (unchanged — the range is set by S33's last seed at +52514, and S23's
extension stays strictly inside the interval). Disjoint from v1 (+0…9049), v2 (+20000…29049),
v3 (+40000…45199), the reserved-and-unconsumed W1 (+46000…46399) and O2 (+47000…47399), the parent's
arm decade (+50000…+50999) and Amendment A1's MN0X block (+50000…50099). **Unit-tested before any
cell runs, S23's 100-seed extent included.**

**Registered consequence of disjoint per-cell seeds, stated as a cost:** cell-to-cell differences
carry the *full independent* noise of both cells; paired variance reduction is **forfeited**, and
the resolution floor in §4 follows from that choice. It is taken deliberately so that the four
corner cross-checks (§5.2) are genuinely independent replications of the parent's arms rather than
re-readings of the same realisations. It is registered as NOT-EVALUABLE item 2.

## 4. Decision statistics — bands locked at this commit, every band DERIVED

**Provenance of every input:** the parent's committed MN0/MEH/MEI extraction; the venue-transfer
committed decision cell (`d45fbf15`); the M5 toy note; and the arithmetic shown below. **Nothing
in this section uses any number this scan will produce.**

### 4.0 The resolution floor — derived first, because every band descends from it

Measured per-cell SE at N = 15, worst case (MN0, fully dosed): **σ_cell = 0.001579**, i.e. a
per-seed spread of 0.001579·√15 = **0.0061154**. Lightly dosed cells are tighter (MEH: 0.000535);
the f_host = 0 row is expected degenerate (MEI: 0.000000).

**S23 is the one exception and it is tighter, not looser.** At N = 100 its SE is
0.0061154/√100 = 0.0061154/10 = **0.00061154** (§4.3, §B). The worst-case figures below are stated
at the N = 15 spread and therefore remain **conservative bounds** for every pair involving S23; the
registered rule is in any case to compute each class with the two cells' **realized** SEs, so S23's
smaller SE flows through automatically wherever it appears (DS-D2's SE_D at (0.5, 1.0) included).

Between two cells on **disjoint** seeds:

```
SE_diff  =  sqrt(sigma_A^2 + sigma_B^2)
         <= sqrt(2) * 0.001579  =  0.0022330      (worst case, both cells at MN0's spread)
```

**Registered resolution classes for any cell-pair difference**, computed with the two cells'
**realized** SEs, with the worst-case values pre-stated:

- **RESOLVED** = |Δb| ≥ 3·SE_diff — worst case **0.0066990**
- **MARGINAL** = 2·SE_diff ≤ |Δb| < 3·SE_diff — worst case **0.0044659** to 0.0066990
- **UNRESOLVED** = |Δb| < 2·SE_diff — worst case below **0.0044659**

**What the grid can and cannot resolve, stated before the data.** The full dynamic range is MN0's
+0.034667. At the worst-case 3σ floor of 0.0067 the surface supports **0.034667/0.0066990 ≈ 5.2
distinguishable levels**; at 2σ, ≈7.8. **The grid is a coarse map, not a fit.** Any claim about a
functional form finer than ~5 levels is out of scope of this N and is barred from the verdict.

### 4.1 DS-D1 — the surface

The 16-cell table of 1D MAP bias, with per-cell SE, per-seed sd, `post_sd` median, realized
`sigma_z_mean_pairs`, and HPD 50/68/90 coverage. **2D reported alongside in every cell** (parent §6
convention: a 1D/2D split is itself a finding). Headline channel is **1D**, carried from the parent.

### 4.2 DS-D2 — additivity, the primary test

Define the **interaction residual**

```
D(f_h, f_i)  =  b(f_h, f_i)  -  b(f_h, 0)  -  b(0, f_i)  +  b(0, 0)
```

which is identically zero for any separable-additive surface, for every `(f_h, f_i)`, exactly — no
approximation. Its standard error uses the four realized cell SEs:

```
SE_D = sqrt( s(f_h,f_i)^2 + s(f_h,0)^2 + s(0,f_i)^2 + s(0,0)^2 )
```

Two pre-stated brackets:
- **conservative** (all four cells at MN0's spread): SE_D = 2·0.001579 = **0.0031580**, 3σ = **0.0094740**
- **expected** (the f_host = 0 row degenerate at 0.000000 and the f_imp = 0 column at MEH's 0.000535):
  SE_D = sqrt(0.001579² + 0.000535²) = sqrt(2.49324e-6 + 2.8622e-7) = **0.0016672**, 3σ = **0.0050017**

**Registered rule:** a cell is **NON-ADDITIVE** iff |D| ≥ 3·SE_D computed with its own realized SEs;
**ADDITIVE-CONSISTENT** iff |D| < 2·SE_D; **AMBIGUOUS** between.

**Pre-computed from the parent's already-run corners** (independent seeds, so this is a prediction
for S33 and not an input to it):
`D(1,1) = 0.034667 − 0.004000 − 0.000000 + 0 = **0.030667**`, which is **9.7σ** against the
conservative SE_D and 18.4σ against the expected one. **H-ADD is expected to be refuted at S33 by a
wide margin**; the scan re-measures it on fresh seeds anyway.

### 4.3 DS-D3 — shape discrimination, and its honest reach

Three hypotheses, each with a falsifiable signature **on this grid**. All are anchored on the four
already-measured corners: `a = b(1,0) = 0.004000`, `c = b(0,1) = 0.000000`, `b(0,0) = 0`,
`I = D(1,1) = 0.030667`.

**H-ADD — additive/separable.** `b = a·f_h + c·f_i`. Signature: `D ≡ 0` everywhere; the surface is a
plane; every row-difference is independent of the other index. Predicts `b(1,1) = 0.004000`.
**Stated for completeness; refuted by the corners at 9.7σ (§4.2).**

**H-INT — multiplicative interaction**, `bias ~ f_host · g(f_imp)` in its bilinear form:
`b = a·f_h + c·f_i + I·f_h·f_i`. Signature: `D(f_h,f_i) = I·f_h·f_i`, i.e. **strictly bilinear** and
**positive in both interior directions**; zero along both f = 0 edges up to the additive edges.

**H-THRESH — threshold in the host dose.** The bias switches on only once the host kernel width
becomes comparable to the impostor sea / GW window scale in z. From `M5_smeared_candidate_prior.md`
§1: the GW width in z is `s = σ_d·τ ≈ 0.022`, the window half-width `4s ≈ 0.088`, and the realized
GLADE dose is `σ̄_z = 0.041813` (MN0 measured). The registered threshold is therefore

```
f*  =  s / sigma_bar_z  =  0.022 / 0.041813  =  0.5262
```

Signature: `b(f_h, f_i) ≈ a·f_h` (edge only) for `f_h < f*`, and switches to the full
impostor-driven behaviour for `f_h ≥ f*` — a **step in f_host between rows h = 2 and h = 3**, with
no ramp through row h = 1. Above threshold, the M5′ flank argument gives `Δζ ∝ σ`, so
`b(1, f_i) = a + I·f_i` — **identical to H-INT on that row.**

**Predicted surfaces (1D bias), registered before the data:**

| | f_i = 0 | 0.25 | 0.5 | 1.0 | |
|---|---|---|---|---|---|
| **f_h = 0** | 0 | 0 | 0 | 0 | all three hypotheses agree (degenerate row) |
| **0.25** H-INT | 0.001000 | 0.002917 | 0.004833 | **0.008667** | |
| **0.25** H-THRESH | 0.001000 | 0.001000 | 0.001000 | **0.001000** | |
| **0.5** H-INT | 0.002000 | 0.005833 | 0.009667 | **0.017333** | |
| **0.5** H-THRESH | 0.002000 | 0.002000 | 0.002000 | **0.002000** | |
| **1.0** both | 0.004000 | 0.011667 | 0.019333 | 0.034667 | **degenerate — no discrimination** |

**Registered structural limitation, stated before the data.** H-INT and H-THRESH are **degenerate on
the f_h = 0 row and on the f_h = 1 row**. All discrimination lives in rows h = 1 and h = 2, and of
those cells only one clears the 3σ dead-band. **S23 is the primary discriminator; S13 is a weak
secondary; the other fourteen cells constrain the surface but do not separate these two accounts.**
**This is exactly why S23 — and only S23 — is registered at N = 100 (§B):** the entire shape question
rests on one cell, so that one cell is bought at a precision that can answer it on the first pass.

**DS-D3 rule — decided at S23 (f_h = 0.5, f_i = 1.0), at N = 100.** Boundary at the **same midpoint
of the same two predictions** as first registered — the midpoint is *not* moved by the N change —
with the dead-band recomputed at S23's N = 100 standard error:

```
per-seed sd (carried, MN0-measured)  =  0.0061154
SE(S23, N = 100)                     =  0.0061154 / sqrt(100)  =  0.0061154 / 10  =  0.00061154
3-sigma dead-band                    =  3 * 0.00061154         =  0.00183462

midpoint = (0.017333 + 0.002000)/2 = 0.0096667                 (UNCHANGED)
SHAPE-INTERACTION  iff  b(S23) >= 0.0096667 + 0.00183462 = 0.01150132
SHAPE-THRESHOLD    iff  b(S23) <= 0.0096667 - 0.00183462 = 0.00783208
SHAPE-UNDECIDED    otherwise
```

**Superseded, and recorded so the change is auditable:** the N = 15 form of this rule was
`3·0.001579 = 0.004737` giving boundaries **0.0144037 / 0.0049297**. Those two numbers are
**retired** by the pre-data design change of §B and **must not be applied to S23**. The
**±0.004737 single-cell dead-band remains in force, unchanged, for every OTHER cell of this scan**,
all of which stay at N = 15.

**Reach of the rule at N = 100, pre-stated.** Each hypothesis' prediction sits
(0.017333 − 0.002000)/2 = 0.0076665 from the midpoint, i.e. **12.54 SE** (0.0076665/0.00061154), and
therefore 0.017333 − 0.01150132 = **0.00583168 = 9.54 SE outside** the decision boundary it must
clear (symmetrically, 0.00783208 − 0.002000 = 0.00583208 below the other). **Either hypothesis, if
true, is called at ≈9.5σ.** At N = 15 the same predictions stood only 1.86σ outside the dead-band
(§B), which is what made SHAPE-UNDECIDED the likely first-pass outcome.

**Secondary at S13** (f_h = 0.25, f_i = 1.0), **N = 15, unchanged**: midpoint
(0.008667+0.001000)/2 = 0.0048333, single-cell dead-band ±0.004737 as before, so
SHAPE-INTERACTION iff `b(S13) ≥ 0.0095703`; SHAPE-THRESHOLD iff `b(S13) ≤ 0.0000963`. **The
threshold arm of this test is essentially "exactly zero" — S13 is registered as a weak
discriminator and carries no branch weight on its own.** The S23 upgrade **does not change S13's
numbers and does not promote it**; if anything it further demotes it, since the branch call is now
expected to be settled outright at S23 (≈9.5σ) rather than needing corroboration. **S13's registered
role is therefore purely corroborative: it is reported alongside S23, and a disagreement between them
is a first-class finding routing to branch 5 (UNDECIDED), never a tie-break in S13's favour.**

**What SHAPE-UNDECIDED now means, and what escape remains.** S23-at-N=100 **is** the primary read;
the old escape — *"S23 alone at N = 100"* — has been **paid upfront and is therefore consumed, not
available.** Consequently:

- **SHAPE-UNDECIDED at N = 100 no longer means "under-powered".** With the predictions 9.54σ outside
  the boundaries, an UNDECIDED read means the measured surface sits in a region **neither registered
  hypothesis predicts** — i.e. both H-INT and H-THRESH are quantitatively wrong at S23, not merely
  unresolved. That is a substantive negative result about the two accounts on the register.
- **No further-N escape is registered for S23.** There is no N = 200 and no third attempt; the
  precision argument was made once, in §B, before the data. Raising N again after an UNDECIDED read
  would be precisely the post-hoc move §B is careful not to be.
- **The remaining registered escapes are unchanged and are elsewhere:** the §6 item 1 low-corner
  cells (S11, S12, S21) at N = 100, **buildable only by author order**. An UNDECIDED read routes to
  branch 5, where **neither H-INT nor H-THRESH may be quoted** and **no repair may be proposed.**

### 4.4 DS-D4 — the pin test (the f_host = 0 row)

MEI's posterior collapsed onto a single grid point at f_h = 0, f_i = 1. The pin is a property of the
**host's exactness**, not of the impostor dose, so:

**Registered prediction: all four cells S00, S01, S02, S03 return bias exactly +0.000000 with
per-seed sd exactly 0.000000, every posterior on one grid point.**

- **PIN-BINARY** = all four collapse ⇒ the pin is a property of host exactness alone, confirmed.
- **PIN-GRADED** = any of S01/S02 returns a nonzero bias or nonzero spread ⇒ **first-class finding**:
  the pin is not binary and depends on the impostor sea. Reported, no branch weight against the
  §4.3 shape hypotheses, which are silent on this row.
- **S00 is the σ = 0 anchor** (the T-0 analogue; the campaign put all 200 T-0 seeds exactly on
  truth). **b(S00) ≠ 0.000000 ⇒ SCAN-CONFOUNDED** — this is an apparatus check, not a measurement.

### 4.5 DS-D5 — linearity in the impostor dose along f_host = 1

M5′ predicts `Δζ ∝ σ`, i.e. R_dose ≈ constant, i.e. the h = 3 row is a straight line from
`b(1,0) = 0.004000` to `b(1,1) = 0.034667`. Registered predictions: `b(S31) = 0.011667`,
`b(S32) = 0.019333`. **SUB-LINEAR / SUPER-LINEAR** iff a cell departs from that line by ≥ 3·σ_cell =
**0.004737**; **LINEAR-CONSISTENT** otherwise. This is the instrument's version of the toy's
K-saturation question, which the MEI collapse already falsified at production K (§0(ii)).

### 4.6 DS-D6 — R_dose, per cell

`R_dose(f_h, f_i) = bias / (f_i · 0.041813)`, evaluated only for `f_i > 0` (0.041813 = MN0's
realized `sigma_z_mean_pairs`; the host contributes 982/1,193,703 = 8.2265e-4 of the candidates and
is negligible in the denominator).

**Band carried from the venue prereg DS-VT3 — `R_dose ∈ [0.75, 1.25]` — applies to S33 ONLY**, the
one cell with a committed anchor (MN0's own R_dose = 0.034667/0.041813 = **0.8291**, in band). **All
other cells have no committed anchor and are REPORTED UNBANDED.** Registering a band for them would
be asserting a number, which this file does not do.

### 4.7 Anti-tuning clause

Every threshold in this file is fixed at this commit and derived from committed artifacts or the
arithmetic shown above; **none may be adjusted after any cell is read**:

the fraction grid (0.0, 0.25, 0.5, 1.0); **16 cells, at N = 15/cell for the fifteen cells other than
S23 and N = 100 at S23** (fixed pre-data in §B, on the author ratification of §A), for **325 seeds**;
the seed block +51000…+52514 and the map `51000 + 100·(4h+i)`, with `j = 0…14` everywhere except
`j = 0…99` at S23 (block +52100…+52199, seeds 20312908–20313007); σ_cell = 0.001579 and per-seed sd
0.0061154; **SE(S23) = 0.00061154 at N = 100**; SE_diff worst case 0.0022330 and the resolution
classes 0.0044659 / 0.0066990; SE_D brackets 0.0031580 / 0.0016672 and the 2σ/3σ rules on |D|; the
anchors a = 0.004000, c = 0.000000, I = 0.030667; the threshold f* = 0.5262 and its inputs s = 0.022
and σ̄_z = 0.041813; **the DS-D3 midpoint 0.0096667 and the DS-D3 boundaries 0.01150132 / 0.00783208
at S23** and 0.0095703 / 0.0000963 at S13; **the S23 dead-band ±0.00183462 (= 3 × 0.00061154), which
applies to S23 and to S23 only**; **the single-cell dead-band ±0.004737, which applies to every cell
OTHER than S23** (S13's secondary test included) and which is **retired at S23**; the DS-D4
exact-zero requirement and the S00 confound trigger; the DS-D5 line through
(0, 0.004000)–(1, 0.034667) with its ±0.004737 departure edge (an N = 15 row, unaffected);
R_dose [0.75, 1.25] at S33 only; the corner tolerances of §5.2; the dosing-verification tolerances
2 % / 10 % of §5.3; the cost anchor 0.969 CPU-h/seed and the ≈259 CPU-h budget; **the absence of any
further-N escape at S23** (§4.3) and the §6 item 1 low-corner escape at author order only.

**On the one change made to this clause since it was first drafted.** The S23 row (N, seed extent,
SE, dead-band, boundaries) was changed **before any cell was run and before any number this scan
produces existed**, as the pre-data design improvement recorded in §B and ratified in §A. Every
value in the list above is fixed **from this commit** and **none may be adjusted after any cell is
read** — the S23 numbers now included.

## 5. Cross-checks (pre-stated, and — where marked — carrying NO branch weight)

### 5.1 Reweighting is not a lever, and no cell of this scan may be read as suggesting it is

`M5_smeared_candidate_prior.md` §4.2 measured, in the validated toy at σ_z = 0.035 against a
baseline of +0.02518:

| variant | bias | vs baseline |
|---|---|---|
| W1 rate weights `v(z_obs)/(1+z_obs)` | +0.02565 | **+2 %** |
| oracle weights at **true** z | +0.02538 | **+1 %** |
| `w_pop(z)` prior **inside** the integral | +0.03217 | **+28 %** |
| flat prior renormalised on the window | +0.03074 | **+22 %** |
| both of the above | +0.03282 | +30 % |
| Jacobian `τ` divided out | +0.01718 | −32 % |

**Every reweighting scheme made the bias WORSE or left it unmoved; none attenuated.** This is not an
empirical coincidence — M5 §2 proves it exactly: a displacement shared by all candidate kernels
factors out of *any* h-independent convex combination, so the stationary point moves by exactly the
same amount for any K and any weights.

**Registered consequence:** **no weighting variant is run in this scan**, and the surface is
predicted **weight-invariant**. A cell that appears "repairable by reweighting" would be a
misreading of this scan, and is pre-emptively barred from the verdict.

### 5.2 Corner consistency — cross-checks, NOT inputs to the scan's own conclusion

Four cells replicate parent arms **at different seeds**. They are registered as independent
replications of the parent's extraction, and the scan's conclusions rest on its **own** 16 cells.

| cell | replicates | parent 1D value | tolerance | rule |
|---|---|---|---|---|
| **S33** | MN0 | +0.034667 ± 0.001579 | 3·sqrt(s_S33² + 0.001579²); worst case **±0.0066990** | outside ⇒ CROSS-CHECK-FAILED |
| **S30** | MEH | +0.004000 ± 0.000535 | 3·sqrt(s_S30² + 0.000535²); expected **±0.0022697** | outside ⇒ CROSS-CHECK-FAILED |
| **S03** | MEI | +0.000000, zero spread | **exact equality required** | nonzero ⇒ CROSS-CHECK-DISCREPANT |
| **S00** | σ = 0 anchor (T-0 analogue) | +0.000000 exactly | **exact** | nonzero ⇒ **SCAN-CONFOUNDED** (§7 branch 1) |

**CROSS-CHECK-FAILED** at S33 or S30 ⇒ reported, author consulted, **no self-adjudication**; it does
not automatically void the scan, because a 15-seed replication at different seeds is entitled to
scatter — but a failure at the registered 3σ tolerance means the arm-level values in the parent's
record are less reproducible than their SEs claim, which is itself a finding about the instrument.
**CROSS-CHECK-DISCREPANT** at S03 is the PIN-GRADED reading of §4.4 and is a first-class finding.
**Only S00 is a hard confound trigger**, because it is an apparatus check.

### 5.3 Dosing verification, per cell

Predicted realized mean per-candidate σ:

```
sigma_bar(f_h, f_i)  =  0.041813 * [ f_h * 8.2265e-4  +  f_i * (1 - 8.2265e-4) ]
```

(8.2265e-4 = 982/1,193,703, the host share of the candidate pool). Registered check against the
already-measured arms: predicted S30 = 0.041813·8.2265e-4 = **0.0000344** vs MEH's measured
**0.000035** (1.7 %); predicted S03 = **0.041779** vs MEI's measured **0.041786** (0.02 %); S33 =
**0.041813** = MN0 exactly. **Registered tolerance:** ≤ **2 %** for every cell with `f_i > 0`;
≤ **10 %** for the `f_i = 0` column, where the quantity is ~1e-5 and sampler noise dominates.
Outside tolerance ⇒ the cell was not dosed as registered ⇒ SCAN-CONFOUNDED.

### 5.4 The `pp_coverage` sign flip — carried from the parent §7, NO branch weight

`validation/pp_coverage.py` carries the structurally identical bare kernel
(`pp_coverage.py:868` vs `venue_transfer.py:1136`) but is a **single-host estimator with no impostor
sea**. Its committed comparisons (`results/commission_20260701/scratch/d2/NOTE_calibration_findings.md`,
`results/pp_coverage_sigmaz_scan_20260703/`) show `bare` biasing H₀ **LOW** by −0.02 to −0.046
across σ_z = 0.005–0.05, with coverage collapsing to 0–3 %. Our venue, same kernel, biases **HIGH**
by +0.037.

**The interaction account may now explain this.** pp_coverage is structurally the
**(dosed host, absent impostor sea)** limit — the instrument's nearest analogue is the **f_imp = 0
column** (S00, S10, S20, S30). Under the interaction account the bias requires *both* ingredients,
so removing the sea should remove the positive term and leave whatever negative term is underneath.
MEH already measured that column's endpoint at **+0.004000 — small, but POSITIVE, not negative.**

**Pre-stated sub-prediction:** if the f_imp = 0 column is small and positive across all four host
doses, the interaction account explains the venue's +0.037 but **does not by itself deliver
pp_coverage's −0.02…−0.046**, and the pre-named carrier of the sign flip remains the parent §7 M1
negative term — the missing `w_pop` volume prior, `Δz = σ_z²·λ` with λ ≈ 2.3–2.9 on this
population, fitted as `bias(σ) = aσ + bσ²` with a ≈ +1.15, b ≈ −5.29 — which should **dominate
exactly when the positive interaction term is switched off by removing the sea.** If instead the
f_imp = 0 column turns **negative** as f_host grows, that is a direct instrument-side reproduction
of pp_coverage's sign and a first-class finding.

**This carries NO branch weight.** It is a directional sub-prediction reported alongside the
verdict. pp_coverage differs from this venue in **K (1 vs K̄ ≈ 1216)** and in the **α(h) selection
normalisation**, so the f_imp = 0 column is an *analogue*, not a replication — registered as
NOT-EVALUABLE item 6.

## 6. NOT-EVALUABLE registry (registered exclusions, with reserved escapes)

1. **Bilinear interaction below f_h·f_i ≈ 0.16.** With the expected SE_D = 0.0016672, |D| clears 3σ
   only for `f_h·f_i ≥ 0.0050017/0.030667 = 0.1631`. Cells **S11 (product 0.0625)** and
   **S12 / S21 (0.125)** are **below the grid's resolving power at N = 15** and cannot test H-INT's
   bilinearity in the low corner. *Escape:* N = 100 on those cells, author order only.
2. **Paired variance reduction.** Disjoint per-cell seeds (§3.2) forfeit it. All cell-to-cell
   comparisons carry √2 noise inflation. Deliberate, for corner independence. *No escape registered.*
3. **K-dependence.** K is pinned at `real_k` in every cell. The scan says nothing about how the
   interaction scales with multiplicity — the toy's K-ladder was falsified at production K (§0(ii))
   and is not re-opened here. *Escape:* a K-ladder arm, author order only.
4. **Transfer to production `BayesianStatistics`.** This is the certified mirror, not the production
   code path (venue prereg §9 item 1, carried). Any estimator fix routes `/physics-change`.
5. **f_incl < 1 / empty-ball events / completeness / window-interior n(z) / sky-cone geometry.**
   Carried verbatim from the venue prereg §9 items 4, 5, 6. The read is conditional on host-in-ball
   over the 982 nonempty-ball events.
6. **The pp_coverage sign flip.** §5.4 — analogue, not replication; K and α both differ.
7. **Any repair.** This scan proposes none and adopts no candidate. A surface is not a mechanism;
   naming the mechanism is the next stage's job.
8. **Functional forms finer than ~5 levels of contrast.** §4.0 — barred by the resolution floor.

## 7. Branches (presented to the author, never self-adjudicated)

Checked in order. Headline channel **1D**; the 2D classification reported alongside in every branch.

1. **SCAN-CONFOUNDED** — `b(S00) ≠ 0.000000`; or any validity check in §8 fails; or any cell's
   dosing verification (§5.3) is out of tolerance; or **Amendment A1 returns A1-FAIL** (the null arm
   genuinely does not reproduce the campaign, in which case every measurement in this thread is void
   by the parent's branch 1). Every measurement below is void; author call on repair-and-rerun.
2. **INTERACTION-BILINEAR** — DS-D2 NON-ADDITIVE at S33 **and** DS-D3 SHAPE-INTERACTION at S23.
   Meaning, pre-stated: the bias is a genuine product-form interaction between host de-pinning and
   impostor smearing; neither ingredient is the mechanism, and **any candidate term proposed
   downstream must be one that vanishes when either ingredient is removed.** M5′-as-registered stays
   refuted; the surviving M5′ *structure* (over-broad effective measure × Jacobian τ against a box
   support) must be re-derived with the host pin included, which it was not.
3. **INTERACTION-THRESHOLD** — DS-D2 NON-ADDITIVE at S33 **and** DS-D3 SHAPE-THRESHOLD at S23.
   Meaning, pre-stated: the bias switches on when the host kernel width reaches the GW scale
   `s ≈ 0.022`, i.e. the mechanism is a **competition between the host's kernel and the GW
   likelihood width**, not a smooth product. The pin is destroyed discontinuously, and the natural
   next object is the crossover itself — the parent's registered E3 dose ladder, at the crossover
   rather than at the registered doses.
4. **ADDITIVE (predicted refuted)** — |D(1,1)| below the DS-D2 non-additivity edge on this scan's
   own fresh seeds. This would contradict the parent's corners at 9.7σ and is therefore
   simultaneously a **CROSS-CHECK-FAILED** condition; treat as branch 1 pending author call.
5. **UNDECIDED (first-class, non-forcing)** — anything else: DS-D3 SHAPE-UNDECIDED, a 1D/2D split,
   PIN-GRADED, a resolved but non-bilinear and non-threshold surface, or an f_imp = 0 column that
   turns negative. Handling, pre-stated: the 16-cell surface is reported raw with its resolution
   classes; **neither H-INT nor H-THRESH may be quoted from an UNDECIDED read**. **Note the changed
   meaning of a SHAPE-UNDECIDED here:** S23 is registered at N = 100 (§B), so the former escape
   (*S23 alone at N = 100*) is **already spent** and UNDECIDED no longer means "under-powered" — at
   9.54σ of headroom (§4.3) it means **both registered shape accounts are quantitatively wrong at
   S23**, which is itself the finding. The only pre-named follow-up that survives is the §6 item 1
   low-corner cells (S11, S12, S21) at N = 100, buildable only by author order; **no higher N at S23
   is registered.** **No repair may be proposed from an UNDECIDED read.**

## 8. Validity checks and STOP criteria

Carried from the parent §5, adapted to this scan:

- **V-D1 — pin integrity, before any cell.** CRB CSV md5 `9a1f2a14384a9281c97ca3be312ddaab`;
  frozeng emit md5 `34c50e91028b6a6458a2b145db545705`; K census 1588/606/982/ΣK 1,193,703/max
  245,364; pruned-frame σ_z stats n = 20,834,171, median 0.0393412950539589, min
  0.0005317263419419, n<5e-3 231,098. Any mismatch ⇒ STOP.
- **V-D2 — generator invariance across cells.** AD-1/AD-2/AD-3 (§2.2), unit-tested at fixed seed
  before any cell. Any cell that perturbs a **draw** rather than a **fraction** is invalid by
  construction — the whole design rests on this.
- **V-D3 — K_sum pin.** `K_sum = 1,193,703` and `n_events = n_events_run = 982` on **every seed of
  every cell**. Any mismatch ⇒ STOP.
- **V-D4 — clean rule.** Parent V-M4 verbatim: import path = `darksiren_emri/` +
  `darksiren_emri_test/`; a registered cell **refuses to start** on any uncommitted change, modified
  or untracked, inside it; `--allow-dirty` accepted only with `--smoke`/`--validate`; the full
  `git status --porcelain` inventory embedded in every output JSON.
- **V-D5 — values golden.** Parent V-M5 as re-registered: every shared field agrees with the
  committed record to **rtol ≤ 1e-12** and both channels' MAP values are exactly equal.
- **V-D6 — dosing verification.** §5.3, every cell.
- **V-D7 — corner cross-checks.** §5.2. S00 is a hard trigger; S33/S30/S03 are reported findings.
- **Abort criteria:** (a) non-finite `ln_post` in > 1 % of any cell's seeds ⇒ STOP; (b) horizon-drop
  > 5 % ⇒ STOP; (c) any V-D failure ⇒ STOP; (d) any rail in any cell or channel ⇒ STOP and report
  (all three parent arms measured zero rails); (e) measured per-seed cost > 2× the 0.969 CPU-h
  anchor ⇒ STOP and re-derive the budget before continuing.
- **Retrieval discipline (carried, and load-bearing).** `rsync -az --exclude='*.md'` — the exclude
  makes it impossible for a cluster copy to overwrite this registered file, `ARMS.md`, the parent,
  or Amendment A1. This guard exists because a plain `rsync` silently reverted the venue-transfer
  prereg on 2026-08-13. Verify post-transfer that no `.md` shows modification in `git status`.

## 9. Expected NULL results (pre-registered)

Stated in advance so that a null is a *reading*, not an absence:

- **The f_host = 0 row is expected to be entirely null and degenerate** (§4.4). Four cells and 60
  seeds buy one confirmation — that the pin is a property of host exactness alone. That is a
  deliberate purchase, registered as such, and its cost is low (S00/S10/S20/S30 aside, S01–S03 are
  full-cost at ~0.94 CPU-h/seed, ≈42 CPU-h for the row).
- **H-ADD is expected to be refuted** at S33 (§4.2). A confirmation would contradict the parent's
  corners and routes to branch 1.
- **DS-D5 is expected LINEAR-CONSISTENT** along f_host = 1, from M5′'s `Δζ ∝ σ`. Sub-linearity would
  reopen the saturation question the MEI collapse closed.
- **The f_imp = 0 column is expected small and positive** (§5.4), i.e. **not** to reproduce
  pp_coverage's negative sign. A negative column is a first-class finding.
- **2D is expected to track 1D in every cell**, as it did in every campaign cell and in all three
  parent arms. A 1D/2D split in any cell is itself a finding and routes to branch 5.

## 10. Model/effort policy for the readout

Carried verbatim from the parent and the venue prereg: mechanical extraction at low effort;
interpretation and the adversarial pass at high effort; **the branch call is presented to the
author, never self-adjudicated.**

---

*Verdict to be appended below by the session that reads out this scan — after this file is
committed, no edits above this line.*
