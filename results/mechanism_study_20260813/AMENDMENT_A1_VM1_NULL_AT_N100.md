# Amendment A1 to `PREREGISTRATION_MECHANISM_ISOLATION.md` — settle V-M1 on data, not on a band

**REGISTERED 2026-08-13**, as an append-only amendment to the parent pre-registration
`results/mechanism_study_20260813/PREREGISTRATION_MECHANISM_ISOLATION.md`. Registered **before**
arm MN0X is run and **after** the MN0/MEH/MEI extraction recorded in the parent's *Operational
completion record*. Append-only from this commit: the verdict appends below the final rule, and no
line above it may be edited. **The parent file is not edited by this amendment** — an amendment is
a new registered document, not a revision of a registered one.

Author instruction of record, 2026-08-13: the ±0.002 window is **not** to be widened
("don't get carried away"). This amendment implements that instruction.

## 0. AUTHOR RATIFICATION — 2026-08-13

**Ratified 2026-08-13 on the author's verbatim words:**

> **"all approved as you recommend"**

**What was put to the author.** Three items were on the page when those words were written, and the
author was shown them together. The itemisation below is **orchestrator-derived** from the state of
the two registered documents at that moment — it reconstructs what "as you recommend" referred to,
and it is **not** author dictation. The author's ruling is the seven words quoted above and nothing
else; each item below is ratified because it was on the page, not because the author enumerated it.

1. The companion scan prereg's **budget ceiling breach** — 16 cells against the parent §1 registered
   *"L1 ≤ 5 arms"* STOP-and-consult ceiling (`PREREGISTRATION_2D_DOSE_SCAN.md`, opening block item 1).
2. The companion scan prereg's **seed-decade extension** — the new block **+51000…+52514**, lying
   outside the parent §1 registered decade **+50000…+50999** (same file, opening block item 2).
3. The recommendation to register scan cell **S23 at N = 100 rather than N = 15**, on the ground
   that S23 is the scan's sole shape discriminator (same file §4.3).

**What this ratification does for THIS amendment.** Items 1–3 are all changes to the *companion*
document; **no band, arm, seed block, prediction or rule in this amendment is altered by the ruling,
and none was proposed for alteration.** Its effect here is exactly one thing: §9 of this file
registers a mutual hold — *"Neither document may be run before the other is ratified"* — and the
author's approval, given across both documents presented together, **releases that hold.** Arm MN0X
is therefore ratified as registered, in the form fixed above and below this block, and may be run.

**What it does not do.** It does not touch the ±0.002 window (§1), the A1-FAIL pre-committed reading
(§4.4), the registered ceiling *"MN0X only"* (§7), or the parent's pending branch call. The
anti-tuning clause of §8 stands unamended and unrelaxed by this ratification.

---

## 1. What this amendment does and does not do

**It does NOT adjust any band.** V-M1's window stays at **±0.002** in exactly the form the parent
registered it:

> **V-M1 — null-arm reproduction.** Arm N-0 must reproduce the campaign's decision-cell bias
> within ±0.002. This is the study's own anchor; failure ⇒ STUDY-CONFOUNDED.
> — parent §5

and the parent's anti-tuning clause is quoted here in full because it is the rule this amendment
is written to obey, not to route around:

> **Anti-tuning.** Every threshold in this section (0.010 / 0.030 / 0.004 / 0.60 / 0.25 / 1e-3 at
> L0; the N = 25 and N = 200 rows; the +50000 seed decade) is fixed at this commit and derived from
> committed campaign artifacts or standard binomial arithmetic. None may be adjusted after any arm
> is read.
> — parent §3

The ±0.002 window is registered in the parent's §4 (branch 1) and §5 (V-M1) rather than enumerated
in §3's list. It is nonetheless covered by the same rule, and independently by the venue-transfer
prereg's abort clause carried into this thread — *"No band may be adjusted after any readout"*
(`PREREGISTRATION_VENUE_TRANSFER.md` §10). **This amendment treats ±0.002 as untouchable.**

**It DOES register a new arm.** The remedy is to buy the precision the band always presupposed:
re-run the null at **N = 100**, where the arm's own standard error falls to ≈0.00061 and the
**original, unchanged** ±0.002 window is satisfied at ≈3σ rather than ≈1.3σ. The band is settled on
data. That is the only legitimate move available once a readout has happened.

**It does NOT adjudicate the parent study.** The parent's scoring readout, its DS-M5 call, and its
branch call belong to a separate session and are untouched here. This amendment concerns V-M1 and
nothing else.

## 2. The observed failure, stated exactly

| quantity | value | source |
|---|---|---|
| campaign decision-cell 1D MAP bias (N = 400) | **+0.037237 ± 0.000230** | venue-transfer readout `d45fbf15`, parent §0 |
| arm MN0, 1D, N = 15 | **+0.034667 ± 0.001579** | parent *Operational completion record* |
| arm MN0, 2D, N = 15 | **+0.037000 ± 0.001604** | same |
| \|Δ\| against the campaign, 1D | **0.002570** | 0.037237 − 0.034667 |
| registered V-M1 window | **±0.002** | parent §5 |

So **STUDY-CONFOUNDED fires mechanically on the 1D channel**, while the 2D channel lands at
+0.037000 — inside the window at |Δ| = 0.000237, on the campaign value.

## 3. Why the window was under-specified — derivation

> **THIS DERIVATION IS BEING WRITTEN AFTER THE READOUT.** It is recorded so the design fault is
> auditable, and it is **barred from use as grounds to pass the N = 15 result.** Nothing below
> reclassifies MN0. MN0's V-M1 status is exactly what the registered rule says it is at N = 15:
> **FAILED**. The only permitted remedy is new data at higher N against the same window, which is
> what §4 registers.

The ±0.002 window is an **asserted** number. It has no derivation anywhere in the parent file, and
it is tighter than every comparable edge in the thread it inherits from (the venue prereg's in-band
edge is |b| ≤ 0.010, its DEFECT edge |b| ≥ 0.030). It gates a **difference of two estimates**, and
its registration accounted for the sampling error of **neither**, nor for the discreteness of the
statistic. Three error sources, all pre-computable at registration time, were omitted:

**(i) The arm's own sampling error.** MN0 measures a mean over 15 seeds. Its measured SE is
**0.001579**. The parent itself estimated this in §2 — *"at N = 15 (SE ≈ 0.0013)"* — and then
registered a ±0.002 window on top of it anyway. A ±0.002 window around a statistic whose SE is
0.001579 is a **±1.27σ** acceptance region.

**(ii) The reference's own sampling error.** +0.037237 is not a constant of nature; it is a
400-seed mean with SE **0.000230**. The correct null distribution of the difference has

```
SE_diff = sqrt(0.001579^2 + 0.000230^2)
        = sqrt(2.493241e-6 + 5.29e-8)
        = sqrt(2.546141e-6)
        = 0.00159566
```

**(iii) Grid quantisation.** The MAP is a grid-argmax on the canonical 41-point grid at spacing
0.005, so a per-seed bias is an integer multiple of 0.005 and an N = 15 mean is an integer multiple
of 0.005/15 = **0.000333**. This is verifiable directly against the four extracted means, and it
holds exactly:

| arm/channel | mean bias | × 15 | ticks of 0.005 |
|---|---|---|---|
| MN0 1D | +0.034667 | 0.520005 | 104 |
| MN0 2D | +0.037000 | 0.555000 | 111 |
| MEH 1D | +0.004000 | 0.060000 | 12 |
| MEH 2D | +0.004333 | 0.064995 | 13 |

All four are integers. The reference +0.037237 is **not attainable at N = 15**: it sits at
111.71 ticks. The nearest attainable values are 0.037000 (111 ticks) and 0.0373333 (112 ticks), so
comparing an N = 15 mean to this reference carries a deterministic quantisation offset of up to
0.005/(2×15) = **0.000167** before any noise at all — 8.4 % of the entire window.

**Consequence — the false-fail rate the window carried.** Under the exact null "MN0 reproduces the
campaign", the deviation is Gaussian with SE 0.00159566, so

```
P(|Δ| > 0.002)  =  2 * Φ̄(0.002 / 0.00159566)  =  2 * Φ̄(1.2534)  ≈  0.210
```

i.e. the registered V-M1 had a **≈21 % probability of declaring STUDY-CONFOUNDED even for a perfect
reproduction.** Using the parent's own registration-time SE estimate of 0.0013, the figure it could
have computed before running anything was ≈13 %. **The band was tighter than the statistic it
gated.** That is the design fault, and it is a fault in the *precision purchased*, not in the
*tolerance demanded*.

**Where the observed failure actually sits.** 0.002570 / 0.00159566 = **1.611 σ** (1.628 σ against
the arm SE alone, which is the 1.63 σ figure the parent's operational record quotes). Concretely,
the entire failure is 0.03855 in summed bias, i.e. **7.71 grid steps spread over 15 seeds** — about
eight seeds each landing one single 0.005 grid point below where an exact reproduction would put
them. This is a plausible fluctuation and it is *also* consistent with a genuine small offset. **At
N = 15 the two are not separable, which is the whole problem, and is why the answer is more seeds
rather than a different band.**

## 4. THE REMEDY — arm MN0X, registered here before it is run

### 4.1 Arm definition

| field | value |
|---|---|
| **arm id** | **MN0X** |
| `sigma_mode` | `glade` |
| `dose_target` | `"all"` |
| `h_true` | 0.730 |
| **N** | **100** |
| **seeds** | base 20260808 + **50000…50099** → 20310808–20310907 |

Everything else is the parent's base configuration verbatim: the campaign decision cell, pinned 982
events, `balls="real_k"`, real K_i, GLADE-empirical σ_z sampler, canonical 41-point grid,
`n_events_cap=None`, `chunk_pairs=16384`, the four §1 pins.

**No code change of any kind.** MN0X is MN0 with a longer seed list. `ARMS.md` therefore requires
no amendment and is not touched; MN0X inherits MN0's registered code form (`dose_target="all"`,
which `ARMS.md` fixes as the default and as byte-identical to the committed campaign path) and its
registered null checks AR-1/AR-2/AR-3 unchanged.

### 4.2 Seed-block integrity

- MN0's registered seeds are base+50000…50014 (`ARMS.md`). MN0X's block **starts at the same
  seed** and extends to +50099.
- MEH occupies base+50100…50114 and MEI base+50200…50214. The MN0X block terminates at **+50099,
  exactly one seed below MEH's first**, so **no seed is shared with MEH or MEI.** This is a tight
  abutment and is stated explicitly so it is checked rather than assumed.
- The whole block lies inside the parent §1 registered decade +50000…+50999, and is disjoint from
  v1 (+0…9049), v2 (+20000…29049), v3 (+40000…45199), and the reserved-and-unconsumed W1
  (+46000…46399) and O2 (+47000…47399) blocks. **Unit-tested before the arm runs**, per the parent's
  VT-D7 discipline.

### 4.3 The already-run 15 seeds are INCLUDED, not discarded

MN0X is a deliberate **superset** of MN0. The 15 seeds already run
(20310808–20310822) are carried into the N = 100 mean.

**Why, stated before the result exists:** those 15 seeds were run under the registered protocol and
produced a mean *below* the reference. **Dropping them and running 100 fresh seeds would be
selection** — it would remove, from the estimate, exactly the observations that caused the
inconvenient reading, and the resulting mean would be biased upward relative to the protocol by
construction. Their inclusion is what makes MN0X an honest re-measurement rather than a second
attempt. It is also why the pre-registered expectation in §5 is *not* the campaign value: the
already-observed downward fluctuation is carried and quantified.

**Reuse rather than re-run.** The arm is deterministic in the seed (parent V-M2 / `ARMS.md`
AR-1/AR-3), so the 15 stored per-seed records in `MN0_h0p730_results_seeds0_15.json` are reused
verbatim, subject to one registered check:

- **A1-DET** — re-run **two** of the 15 stored seeds and require their per-seed records to match the
  stored ones to rtol ≤ 1e-12 with MAP values exactly equal (the V-M5 values-golden convention,
  ratified in the parent §5). **Failure ⇒ do not reuse; run all 100 seeds fresh and report the
  determinism failure as a first-class finding.**

### 4.4 Decision rule — registered BEFORE the data

**A1-PASS.** MN0X satisfies V-M1 iff

```
| mean_bias_1D(MN0X, N=100)  -  0.037237 |  <=  0.002
```

— the **original window, unchanged, on the original 1D channel**, against the same reference. The
2D channel is reported alongside under the same rule, per the parent's §6 convention that a 1D/2D
split is itself a finding.

**A1-FAIL — pre-committed reading.** If MN0X **also** misses the window at N = 100, then:

> **The null arm genuinely does not reproduce the campaign decision cell.** STUDY-CONFOUNDED stands
> as a **real finding about the instrument, not as an artefact of an under-powered check.** The
> parent's branch 1 fires on its merits: every mechanism measurement in the study is void, and the
> question moves to *why* the arm and the campaign differ (candidates to be opened then, not now:
> the Route 1 Gauss–Hermite contraction change certified at rel err 1.3e-15 and reaching the
> validation stack via `bayesian_statistics.py`; a seed-block population difference; an unnoticed
> configuration drift between the campaign path and the arm path). **This reading is pre-committed
> here so that a failure cannot later be re-narrated as noise.** No further widening, no third N,
> no fourth arm: on A1-FAIL the study is confounded and the author is consulted.

**No intermediate escape is registered.** There is no "marginal" class and no third attempt.
At N = 100 the window is a ≈3σ acceptance region (§5), which is the point of the amendment.

### 4.5 What is NOT amended

- Every other band, edge and threshold in the parent (0.010 / 0.030 / 0.004 / 0.60 / 0.25 / 1e-3;
  the DS-M5 conjunction 0.030 / 0.012; the N = 25 and N = 200 rows; the branch definitions).
- `ARMS.md` in any respect.
- The parent's DS-M5 verdict, its branch call, and its §7 closures.
- The parent's registered budget ceiling (L0 unlimited, L1 ≤ 5 arms, L2 ≤ 1 arm). MN0X is a
  **re-run of the existing null arm at higher N**, not a sixth L1 arm; it consumes no new arm slot.

## 5. Pre-registered expectation, with uncertainty — so the result is falsifiable

Registered before the arm runs. All inputs are committed values or arithmetic shown here.

**Per-seed spread**, inferred from the measured arm SE:

```
sd_per_seed = 0.001579 * sqrt(15) = 0.001579 * 3.872983 = 0.0061154
```

**SE at N = 100:** 0.0061154 / 10 = **0.00061154**.
**SE of the difference against the reference:** sqrt(0.00061154² + 0.000230²) = sqrt(3.7399e-7 +
5.29e-8) = **0.00065337**.

**The window in σ, unchanged at ±0.002:** 0.002 / 0.00061154 = **3.27 σ** (arm SE alone);
0.002 / 0.00065337 = **3.06 σ** (difference SE). This is the whole content of the remedy: the same
window, bought at ≈3σ instead of ≈1.3σ. Under the exact null the false-fail rate falls from ≈21 %
to 2·Φ̄(3.06) ≈ **0.22 %**.

**Registered point prediction.** Under the null "the arm reproduces the campaign, and the observed
N = 15 shortfall is sampling noise", the 15 included seeds are *known* and the 85 fresh seeds have
expectation 0.037237:

```
E[ mean_bias(MN0X) ]  =  0.15 * 0.034667  +  0.85 * 0.037237
                      =  0.00520005 + 0.03165145
                      =  0.0368515
```

with conditional SE = 0.0061154 · sqrt(85)/100 = **0.000564**.

> **REGISTERED PREDICTION: MN0X 1D mean bias = +0.03685 ± 0.00056 (1σ), 68 % interval
> [0.03629, 0.03741], and |mean − 0.037237| = 0.000386, comfortably inside the ±0.002 window.**

**What would falsify it.** A1-FAIL requires mean_bias ≤ 0.035237 (the upper miss, ≥ 0.039237, is not
a live direction here but is registered symmetrically). Given the 15 included seeds at +0.034667,
that requires the 85 fresh seeds to mean

```
85 * m  <=  100 * 0.035237  -  15 * 0.034667  =  3.5237 - 0.520005 = 3.003695
m       <=  0.0353376
```

i.e. a fresh-seed mean **0.0018994 below** the reference, which against the fresh-seed SE
0.0061154/sqrt(85) = 0.000663 is a **−2.86 σ** excursion. **So MN0X can only fail if the fresh
seeds independently confirm a downward offset at ≈3σ — which is exactly the evidence that would
make STUDY-CONFOUNDED a real finding.** The test is therefore genuinely two-sided in its meaning:
it cannot rescue the study by construction, and it cannot condemn it by noise.

## 6. Validity checks carried

Carried verbatim from the parent §5, unchanged, and all must pass for MN0X to be scored:

- **V-M2** — generator invariance (AR-1/AR-3): MN0X uses `dose_target="all"`, the registered
  default, so its realisation must be bit-identical to the current registered path per seed.
- **V-M3** — pin integrity before the arm: CRB CSV md5 `9a1f2a14384a9281c97ca3be312ddaab`; frozeng
  emit md5 `34c50e91028b6a6458a2b145db545705`; K census 1588/606/982/ΣK 1,193,703/max 245,364;
  pruned-frame σ_z stats n = 20,834,171, median 0.0393412950539589, min 0.0005317263419419,
  n<5e-3 231,098. Any mismatch ⇒ STOP.
- **V-M4** — clean rule (import path `darksiren_emri/` + `darksiren_emri_test/`).
- **V-M5** — values golden at rtol ≤ 1e-12 with both channels' MAPs exactly equal.
- **A1-DET** — §4.3, new to this amendment.
- **Abort criteria** — parent §5 (a)–(d) unchanged: non-finite `ln_post` in > 1 % of seeds ⇒ STOP;
  horizon-drop > 5 % ⇒ STOP; any V-M failure ⇒ STOP.
- **Per-seed pins to re-verify on all 100 seeds:** `K_sum = 1,193,703`;
  `n_events = n_events_run = 982`; zero rails in either channel; zero non-finite `ln_post`;
  `sigma_z_mean_pairs` ≈ 0.0418 (MN0 measured 0.041813) confirming full dosing.

## 7. Cost anchor

The arms measured, from the parent's operational record (array 6303086, 15 cores/task, 15 seeds):

| arm | elapsed | core-hours | CPU-h/seed |
|---|---|---|---|
| MN0 (dose=all) | 00:58:08 | 14.53 | **0.969** |
| MEI (dose=impostors) | 00:56:30 | 14.13 | 0.942 |
| MEH (dose=host) | 00:02:20 | 0.58 | 0.039 |

**≈0.97 CPU-h/seed for a fully dosed arm — about 3.9× faster than the campaign's registered
3.79 CPU-h/seed anchor.** The plausible cause, stated as a hypothesis and not as a claim, is the
author-ratified **Route 1 adaptive Gauss–Hermite** contraction of the g_i integral
(`bayesian_statistics.py`, ledger 2026-08-12, certified max rel err 1.3e-15) reaching the validation
stack through `venue_transfer.py`'s import — the same change that forced the V-M5 re-registration in
the parent §5. **Registered consequence: the campaign's 3.79 anchor is stale for cost planning in
this thread, and 0.97 CPU-h/seed is the anchor MN0X and any successor is budgeted against.**

**MN0X budget:**

- 85 fresh seeds × 0.969 = **≈82 CPU-h** incremental, reusing the 15 stored records (§4.3).
- 100 seeds × 0.969 = **≈97 CPU-h** if A1-DET fails and everything is re-run.
- Wall clock in the arms' measured shape (15 seeds/task, 15 cores/task): **≈1 h** for six tasks.
- The `sbatch --test-only` start estimate is registered as **non-predictive** for this shape — it
  was wrong by four days on array 6303086, the second recorded instance (parent operational record,
  EXP-61 discipline). Budget against backfill reality, not the probe.

**Registered ceiling:** MN0X only. If MN0X fails, the answer is A1-FAIL (§4.4), **not** N = 200.

## 8. Anti-tuning clause for this amendment

Fixed at this commit, derived from committed artifacts or the arithmetic shown above, and not
adjustable after MN0X is read: the **±0.002 window (unchanged from the parent)**; the reference
value 0.037237 and its SE 0.000230; N = 100; the seed block +50000…50099; the inclusion of the 15
already-run seeds; the per-seed sd 0.0061154 and the derived SEs 0.00061154 / 0.00065337; the
registered point prediction +0.03685 ± 0.00056; the A1-FAIL pre-committed reading; the A1-DET
rtol 1e-12 check on two stored seeds; the 0.97 CPU-h/seed cost anchor. **The derivation in §3 is
explanatory only and confers no pass on any result already read.**

## 9. Scope note

Consequence (A) of the MN0/MEH/MEI extraction — that the registered DS-M5 split prediction is
inverted on its decisive half and the effect is a **host × impostor interaction** rather than a sum
of two terms — is **not adjudicated here.** It is the subject of the companion stage-2
pre-registration `PREREGISTRATION_2D_DOSE_SCAN.md`, registered alongside this amendment, which also
carries the M5 reweighting-ablation register and the `pp_coverage` sign-flip cross-check.
**Neither document may be run before the other is ratified, and neither depends on the parent's
branch call, which remains pending and belongs to a separate session.**

---

*Verdict to be appended below by the session that reads out MN0X — after this file is committed, no
edits above this line.*

---

## VERDICT (appended by the read-out session, 2026-08-14)

Scored against `results/mechanism_study_20260813/A1_READOUT.md` /
`A1_READOUT.json` (`score_a1.py`), instrument commit `3aedbe55`, data commit `5b0bd17a`.

- **A1-PASS.** MN0X 1D mean bias +0.037250 ± 0.000494 vs reference 0.037237; |Δ| = 0.000013
  against the UNCHANGED ±0.002 window, 153.8× inside, 0.024σ of the difference SE. Registered
  pre-data prediction +0.03685 ± 0.00056 met at +0.71σ.
- The measurement is as close as the statistic allows: at N = 100 a mean is an integer multiple of
  5e-5; the reference sits at 744.74 ticks (unattainable), the measurement at 745 exactly.
- The 85 FRESH seeds alone mean +0.037706 ± 0.000499, i.e. +3.57σ above the registered fail
  threshold; the original N = 15 block was a −1.90σ fluctuation of its own seeds.
- 2D secondary +0.039750 vs 0.039713, also inside. No 1D/2D split. A1-FAIL does not fire.
- **A1-DET PASS, exceeding its registered minimum**: 15/15 shared seeds compared (registered
  minimum 2) across 44 value fields each including both full 41-point `ln_post` vectors; max
  relative deviation 0.0, bit-identical, MAPs exactly equal. A genuine CROSS-COMMIT test (MN0 at
  `e83ed0b9`, MN0X at `3aedbe55`, which refactors the dose application), so it empirically confirms
  `ARMS.md` AR-1 on real campaign data. The run log confirms seeds=0:100 were recomputed fresh, no
  stored record reused — more conservative than registered.
- **Adversarially verified: CONFIRMED.** Independent reimplementation rebuilt every
  posterior-derived statistic from raw `ln_post` vectors for all 425 seeds with max deviation
  exactly 0.0.
- **What A1-PASS does NOT license**: it does not retroactively pass the N = 15 MN0 result (A1 §3
  bars it; MN0's V-M1 status at N = 15 remains FAILED). The correct reading is that the null is now
  measured at a precision where the registered window is meaningful, and at that precision it
  reproduces. No band changed.
- **Open disclosure D-A1-2: V-M5 was NOT re-executed at `3aedbe55`.** A1-DET's bit-identity plus
  the AR unit tests are strong evidence but are not the registered V-M5 artifact. Carried forward
  for the parent readout session. Also D-A1-3: MEH/MEI sit at the pre-refactor commit.
- Verification's two cosmetic notes on A1: one line mixes ddof=0 and ddof=1 SE conventions
  (4.07/3.68 vs 4.05/3.67 — verdict unchanged); and the quantisation argument says "the 0.005 grid
  spacing" when the canonical grid is non-uniform {0.005, 0.01} — the argument holds because all
  100 MAPs land in the fine region, but the condition was unstated.
