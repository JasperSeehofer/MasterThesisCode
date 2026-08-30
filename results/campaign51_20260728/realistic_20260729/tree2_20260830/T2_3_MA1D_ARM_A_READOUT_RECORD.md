# T2.3 arm (a) independent readout — the mass-aware 1D catalogue-leg mirror FT-fleet paired counterfactual

Independent reader, 2026-08-30. Foreground only, read-only, no git, no ssh, wall time this reader
about 1 second (pure CSV arithmetic over an already-completed run). Launched under row #255 — tree
2 node T2.3 arm (a). Registration: `PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md` section 6.1 (the
paired 4-seed prediction), section 8 (falsifiers), section 9 (arms); implementation record
`T2_3_MA1D_IMPLEMENTATION_RECORD.md` section 2. Run of record:
`ma1d_ft_counterfactual_run/s0a_seed900101..900104/{node_truth_ft, node_truth_ft_ma1d}` (off/on),
log `ma1d_ft_counterfactual_run/logs/runner6_tree2_20260830.log` (off arm 10:41:36→13:03:44, on arm
13:03:44→15:26:52, 14-core budget, `--jobs 1`). Companion machine-readable file:
`t2_3_arm_a_readout.json`.

## 1. Bottom line

The mirror confirms the flag has a real, positive, all-4-seeds effect on `mean_h`, but the
**measured size is about 2.3x the registered point prediction and sits above the registered
two-sided band's upper edge in 2 of 4 seeds** (and in the 4-seed mean). Per the registered
verdict-bucket definition (section 6.1: "MASS-AWARE-MATERIAL iff Delta >= +0.03"), the bucket read
is **MASS-AWARE-MATERIAL**, but this is not a clean "landed inside the band" result — it is a
genuine size surprise the author should see before any [RULE] on the production flip.

| | value |
|---|---|
| Registered point prediction | +0.05 |
| Registered band | [+0.03, +0.10] |
| Registered NULL threshold | <= +0.008 |
| **Measured Delta mean_h (4-seed mean)** | **+0.1158** |
| SEM (4 seeds) | 0.0136 |
| Per-seed SD | 0.0272 |
| Per-seed vector | +0.0970, +0.1294, +0.0895, +0.1474 (seeds 900101-900104) |
| Verdict (registered bucket) | **MASS-AWARE-MATERIAL** (>= +0.03; not NULL, not REFUTING) |
| Caveat | measured mean exceeds the band's upper edge (+0.10) by 0.016 (1.2 SEM); 2/4 seeds individually exceed +0.10 |

## 2. Gates

### GATE T-ID (off arm vs KW-Q1 truth node, both h nodes, all 4 seeds)

**PASS, bit-identical.** Compared `combined_no_bh` and `L_cat_no_bh` at h in {0.725, 0.735} between
this run's "off" arm and `fanout1_20260829/kwq1_registered_run/s0a_seed<seed>/node_truth_ft_sites2.2_nosmear`.
Every event/h row matched (no unmatched rows), `max_abs_diff = 0.0` on both columns, all 4 seeds:

| seed | rows compared | max_abs combined_no_bh | max_abs L_cat_no_bh |
|---|---|---|---|
| 900101 | 348 | 0.0 | 0.0 |
| 900102 | 368 | 0.0 | 0.0 |
| 900103 | 348 | 0.0 | 0.0 |
| 900104 | 364 | 0.0 | 0.0 |

The "off" arm's fresh run reproduces the banked KW-Q1 truth node exactly — the pre-check named in
section 6.1 is satisfied, so the "on" arm's numbers can be trusted as a clean counterfactual against
this baseline.

### GATE ENG (fraction of active events whose L_cat_no_bh / combined_no_bh change, on vs off, h=0.73)

Denominator note (disclosed): `event_likelihoods.csv` carries no `n_cand_no_bh` column. "Active" is
read from the separate T2.2 candidate-dump `per_event_h_0_73.csv` (`n_cand_no_bh > 0` at h=0.73),
which is upstream of the flag (candidate search precedes the survival evaluation the flag touches —
section 7's structural-blindness list) and is therefore a valid flag-independent denominator.

| seed | n active (n_cand>0) | frac L_cat_no_bh changed | frac combined_no_bh changed |
|---|---|---|---|
| 900101 | 128 | 100.0% | 92.97% |
| 900102 | 130 | 100.0% | 96.15% |
| 900103 | 117 | 100.0% | 85.47% |
| 900104 | 111 | 100.0% | 96.40% |

**L_cat_no_bh: PASS all 4 seeds** against the registered >= 99% bar (section 6.1) — 100% exactly.
**combined_no_bh vs the R13 regression bar (>= 90%): PASS 3/4 seeds, FAILS on seed 900103 (85.47%)** —
flagged, not adjudicated; R13 is a regression-suite bar on a live smoke cell, not section 6.1's
primary A13 gate, so this does not by itself invalidate the arm, but it is disclosed because the
registered bar is not met on every seed.

**R7 cross-check (dark events, n_cand_no_bh == 0, must be bit-identical on/off):** PASS all 4 seeds —
`combined_no_bh` is exactly unchanged (`np.allclose(..., atol=0.0)` True) for every dark (no-candidate)
event, across the full H_GRID_41, not just h=0.73. This is a stronger confirmation than the unit
test alone: it holds on the production-shaped mirror data across all 41 h-nodes.

### Z = 1 check (on arm)

Not independently re-derivable foreground-only without invoking `evaluate()` on the R2 synthetic
fixture (a code-execution gate, not a read of existing output) — that check is inherited from the
builder's own test-suite run (presentation section 20.5: R2 reported PASS there). As a necessary
(not sufficient) condition checked directly from the "on" arm's own diagnostics: `r_Malm`,
`D_tilde_phi`, and `alpha_G_phi` are each **constant across all events at every h-node** (exactly one
unique value per h, all 4 seeds) — consistent with Z being a single per-h normalisation constant
rather than a per-event artefact of a broken (unpaired) numerator/divisor build. This does not
substitute for the registered R2 unit test; it is a plausibility check only.

## 3. The registered statistic

Corrected-combine (row #146 form: `PHYSICS_FLOOR` zero-handling + composite-trapezoid moment
weights), reproduced here via `darksiren_emri.validation.correspondence_1d.compute_seed_statistics`
on `combined_no_bh` pivoted over `H_GRID_41` (41 nodes, the run's own `--h-nodes` list — identical
to the registered grid). No `H_GRID_FULL`/low-wing companion is available: this run's `--h-nodes`
covered only H_GRID_41, not the 0.50-0.58 wing, so amendment 20's un-truncated companion cannot be
reported (not run, not merely unread).

### Full population (all events)

| seed | off mean_h | on mean_h | Delta mean_h | off MAP | on MAP | Delta MAP |
|---|---|---|---|---|---|---|
| 900101 | 0.63444 | 0.73139 | **+0.09695** | 0.60 | 0.725 | +0.125 |
| 900102 | 0.64640 | 0.77576 | **+0.12936** | 0.60 | 0.82 | +0.220 |
| 900103 | 0.62998 | 0.71949 | **+0.08951** | 0.60 | 0.70 | +0.100 |
| 900104 | 0.62352 | 0.77090 | **+0.14738** | 0.60 | 0.82 | +0.220 |
| **mean +/- SEM (n=4)** | | | **+0.1158 +/- 0.0136** | | | **+0.1663 +/- 0.0314** |

Every one of the 4 seeds is positive (4/4), consistent with the registered sign prediction. The
off-arm MAP rails at the H_GRID_41 floor node (0.60) in all 4 seeds, matching the banked production
floor-rail pattern (row #213) — the paired Delta is therefore, per the registration's own censoring
disclosure (amendment 20), a **lower bound** on the un-truncated effect.

### Dark-class-only split (L_cat_no_bh == 0 at h=0.73 in the off arm)

| seed | n dark | off mean_h | on mean_h | Delta mean_h |
|---|---|---|---|---|
| 900101 | 46 | 0.64895 | 0.64895 | 0.0 |
| 900102 | 54 | 0.65473 | 0.65473 | 0.0 |
| 900103 | 57 | 0.63703 | 0.63703 | 0.0 |
| 900104 | 71 | 0.62652 | 0.62652 | 0.0 |

**Exactly zero movement on every dark event, every seed, over the full H_GRID_41** — a clean,
stronger-than-unit-test confirmation of the R7/structural-blindness invariant: the flag reaches
`combined_no_bh` only through galaxies that are actually present as catalogue candidates.

### Matched-class-only split (L_cat_no_bh != 0 at h=0.73 in the off arm)

| seed | n matched | off mean_h | on mean_h | Delta mean_h | off MAP | on MAP |
|---|---|---|---|---|---|---|
| 900101 | 128 | 0.68294 | 0.79191 | +0.10897 | 0.60 | **0.86** |
| 900102 | 130 | 0.70024 | 0.81247 | +0.11223 | 0.63 | **0.86** |
| 900103 | 117 | 0.68075 | 0.80334 | +0.12259 | 0.61 | 0.70 |
| 900104 | 111 | 0.70503 | 0.82959 | +0.12456 | 0.60 | **0.86** |

**Whole effect lives in the matched class** (as expected — the dark class is exactly invariant).
**New caveat, not previously registered:** under "on", the matched-class-only posterior's MAP
**rails at the H_GRID_41 ceiling node (0.86) in 3 of 4 seeds.** This is the mirror image of the
off-arm's floor rail (0.60) and implies the *matched-class* Delta (+0.109 to +0.125) is itself a
**lower bound** — the grid's top edge (0.86) may be truncating a larger shift for this subset. The
full-population Delta reported above is diluted by the (grid-interior) dark class and does not
itself hit the ceiling, but this ceiling-rail in the class split that drives 100% of the effect is
worth the author's attention before treating +0.116 as the true asymptotic size.

## 4. A15 — per-seed scatter vs the registered SEM

Measured per-seed SD = 0.0272, closest to the registered **"drag" (upper, conservative) anchor of
0.0268** used in section 6.1's power calculation (twin anchor 0.0176 and b2 anchor 0.0150 are both
noticeably tighter than what was actually observed). At N=4 with the drag anchor the registration
predicted a 3.1 sigma read of the point prediction (+0.05); the observed effect (+0.1158) is
(0.1158-0.008)/0.0136 = **7.9 sigma above the NULL threshold** and (0.1158-0.03)/0.0136 = **6.3
sigma above the lower band edge** — decisively MATERIAL by any anchor, but the point estimate itself
(not just its significance) is what has moved, which the sigma-count does not capture.

## 5. Cost

Off arm: 10:41:36 -> 13:03:44 = 8528 s (2.369 h). On arm: 13:03:44 -> 15:26:52 = 8588 s (2.386 h).
Total wall = 17116 s = 4.754 h. At the run's 14-core budget: **wall x 14 = 66.56 CPU-h.** This is
well above the presentation's own registered anchor for the 4-seed form ("about 4 CPU-h", section 9
item 2, and "about 8-9 CPU-h" in the section-9 total) — roughly **8-16x the registered cost anchor**.
Disclosed as a cost-anchor miss, not adjudicated (the anchor assumed lighter per-seed wall time than
this HEAD-basis full-41-node truth-only run in fact took).

## 6. Verdict per the registered map (section 6.1) and caps

- **Bucket: MASS-AWARE-MATERIAL** (Delta mean_h = +0.1158 >= +0.03; not <= +0.008 NULL; not
  negative/REFUTING).
- **Caveat on the bucket:** the two-sided band [+0.03, +0.10] was a *point-prediction* band, not the
  bucket boundary; the measured value is materially above it (+0.1158, and 2/4 seeds individually
  above +0.10). This is reported as a size surprise, not re-classified into a bucket the
  registration does not define — the registration's own bucket rule has no "above-band" category,
  only NULL / MIXED / MASS-AWARE-MATERIAL / REFUTING.
- **F-1 (mirror attribution, section 8):** NOT triggered (Delta is far from <= +0.008 and is not
  negative) — the Z(h)/class-share attribution of the impostor-leg remainder is **not refuted** by
  this arm.
- **Caps, as instructed:** this reading is **instrument-only and REPORTED**. The production-default
  flip of `catalogue_leg_1d_mass_aware` is explicitly **not** authorized by row #255's standing grant
  (presentation section 11) and **returns to the author as a fresh [RULE]**, now carrying these
  numbers: Delta mean_h = +0.1158 +/- 0.0136 (4 seeds, all positive), ABOVE the registered band's
  upper edge, all-effect-in-the-matched-class, with a newly observed matched-class MAP ceiling-rail
  caveat (section 3 above) — the row #169-pairing precedent (a ratified fused paired design) for how
  such a flip decision has previously been framed.

## 7. Caveats summary (for the author)

1. Measured Delta mean_h (+0.1158) is ~2.3x the registered point prediction (+0.05) and above the
   registered band's upper edge (+0.10) in the 4-seed mean and in 2/4 individual seeds.
2. The matched-class MAP rails at the H_GRID_41 ceiling (0.86) in 3/4 seeds — a newly observed,
   previously-unregistered truncation caveat symmetric to the already-registered off-arm floor rail;
   the true matched-class (and hence full) effect size may be understated by this run's grid.
3. GATE ENG's combined_no_bh sub-bar (R13's >= 90%) is met on 3/4 seeds and missed on seed 900103
   (85.47%); the primary section 6.1 gate (L_cat_no_bh >= 99%) is met cleanly (100%) on all 4 seeds.
4. Cost overran the registered 4-seed anchor by roughly 8-16x (66.56 measured CPU-h vs ~4-9 CPU-h
   registered).
5. No H_GRID_FULL/low-wing companion was run; the un-truncated companion (amendment 20, row #173)
   is not available from this run.
6. Z=1 was checked only as a necessary-condition plausibility read (per-h constancy of r_Malm /
   D_tilde_phi / alpha_G_phi); the decisive R2 synthetic-fixture unit test was not re-executed by
   this reader (foreground read-only; no code execution) and is inherited from the builder's own
   test-suite run.
7. GATE T-ID and the R7 dark-class invariant both PASS cleanly and are the strongest results here:
   the off-arm baseline is proven correct, and the flag's effect is proven to live entirely in the
   matched (catalogue-candidate-bearing) class, exactly as designed.
