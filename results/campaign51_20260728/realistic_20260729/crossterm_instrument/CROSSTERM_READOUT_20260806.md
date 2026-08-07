# CROSSTERM READOUT — band scoring of the Eq. (31) cross-term production run

**Document ID:** CROSSTERM_READOUT_20260806 (orchestrator-prescribed date; mechanical scoring
executed 2026-08-07 immediately after the last chunk completed at 05:30 local).
**Scored against:** `PREREGISTRATION_CROSSTERM_INSTRUMENT.md.DRAFT` (this directory) — band
**LOCKED** at author ratification 2026-08-06 ("all approved"): X = 2.78, Y = 7.96 class-summed
chord nats, applied to the **mixture-composed** correction (flag (c) RATIFIED).
**This document is a new readout. No prereg, claim, or ledger file was edited.**
**Machine-readable companion:** `readout_20260806.json` (same directory; every number below is
reproduced there at full precision with the same structure).

Scoring inputs (all paths relative to
`results/campaign51_20260728/realistic_20260729/crossterm_instrument/`):
`outputs/run_joint_r1_1d.json` (1396 rows), `outputs/run_joint_r1_2d.json` (416 rows),
`outputs/run_iiib_1d.json` (1120 rows), `outputs/run_iiib_2d.json` (84 rows);
ratified pair lists `target_pairs_joint_r1.json`, `target_pairs_iiib.json`;
W denominators from prereg §7.4 (1D) and the certification-record §7.4-completion table (2D,
from `outside_c4_2d_wpop.json`).

---

## 0. Headline (mechanical result; branch call presented, not self-adjudicated — §12)

**All four (venue × channel) score NEGLECT-WITH-NUMBER.** The scored (mixture-composed) decision
statistic T is 5.4e-06 – 5.6e-05 class-summed chord nats — between 4.9e+05× and 1.9e+05× **below**
the NEGLECT threshold X = 2.78. The anti-dilution clause does not trigger in any stratum of any
stratification (largest one-signed sub-sum anywhere: 5.649e-05 nats, vs the Y = 7.96 blocking
scale). No venue split. No −inf or NaN row exists. The M-2 revived-H-2 signature (REGARD-scale,
2D-concentrated, low-h-preferring) is **not matched** on the scored statistic — the 2D composed
chord has the **opposite sign** (high-h-preferring) at 5–6 orders below Y (§6 below).

Honesty note carried with the verdict (prereg §5: raw is "reported alongside as a diagnostic,
never scored"; M-3 aggregate-honesty rule): the **raw catalogue-leg** class sums are large
(1D full-grid range 181.2 / 77.5 nats; joint_r1 2D 5.80 nats — numerically inside [X, Y) had the
band applied to raw, which flag (c) RATIFIED forecloses), and the raw 2D chord **is**
low-h-preferring (+2.507 nats joint_r1). The mixture composition — the object the posterior
actually consumes — suppresses this by the catalogue leg's weight (median per-pair composition
factor 1.5e-07 joint_r1 / 2.5e-06 iiib at h = 0.73), leaving the microscopic scored values.

---

## 1. Integrity, guard contract, and escalation audit

Verified at readout time (independent of the runner's own integrity report):

| check | result | source |
|---|---|---|
| Instrument hash | `crossterm_instrument.py` sha256 = `340b66d2f970e48cf5152676e8b6bed6b171f9538efa33bab6e5ef04abd87692` — **matches the ratified pin** (prereg Author Ratification section) | sha256 recomputed at readout |
| Row coverage | Each file's (event_i, event_j, h_requested) set **equals** its ratified target-pair list × h-grid {0.60, 0.73, 0.81, 0.86}, zero duplicates/extras: joint_r1 349×4 = 1396, 104×4 = 416; iiib 280×4 = 1120, 21×4 = 84 | run_*.json rows vs `target_pairs_*.json` |
| NaN rows (guard: S ≤ 0 voids pair) | **0 in all four files** → zero voided pairs | run_*.json, `delta_joint_lnL_nats`/`S_*` scan |
| −inf rows (guard: never summed; auto-escalates to author adjudication) | **0 in all four files** → auto-escalate NOT triggered; all class sums are over all rows (excluded-row census: empty) | run_*.json scan |
| h-grid match | `h_grid_matched == h_requested` on every row | run_*.json |
| Raw class sums vs runner report | All 16 (venue, channel, h) raw sums match the runner-reported values to ≤ 1e-9 relative | recompute vs orchestrator-supplied class_sums_per_h |
| R-2 escalation audit | Adaptive path fired **only** in joint_r1/1d, single pair (114, 1035) (n_shared = 8861), 1 escalated shared galaxy per h; `quad_n_max_shared` = 339/398/434/455 at h = 0.60/0.73/0.81/0.86 (455 = ceil(50·(68.17/30)·4) for the known worst-R galaxy R = 68.17 — matches prereg R-2 expectation, certified error ~6e-14 nats). All other rows in all files: quad_n_max_shared = 50, n_escalated_shared = 0 | run_joint_r1_1d.json rows |
| S own-quad vs frozeng cross-check (R-4-adjacent free read, prereg §8) | max relative deviation \|S_raw − S_frozeng\|/S_frozeng: joint_r1/1d 5.50e-09, joint_r1/2d 1.12e-14, iiib/1d 1.20e-07, iiib/2d 1.26e-14 — all within the R-4 1e-6 criterion | run_*.json `S_*_raw` vs `S_*_frozeng` |
| w_share h-invariance | max spread of `w_share_ball_i/j` across the h-grid per pair = 0.0 exactly (h-independent, as the blind-read convention requires) | run_*.json |
| git commit | `e94ff8abc9d93df0e653064620e10ff36e8ece5d` identical in all four run files | run_*.json `meta.git_commit` |

Runner-reported execution provenance (recorded, not re-verified beyond the coverage/merge checks
above): 1d channels ran as orchestrator-authorized order-preserving sequential chunks of the
ratified pair lists (joint_r1 4 chunks, 7190 s wall; iiib 8×35-pair chunks, 10908 s wall, after
OOM exit-137 aborts at coarser granularities on the 30 GB host; ~25–29 GB RSS ceilings; no partial
writes — end-of-run atomic emit); 2d channels single invocations (894 s / 418 s). `meta.chunked_run`
present in both merged 1d files (n_chunks 4 / 8); chunk artifacts retained in `outputs/chunks/`.
All four venue × channel dry-runs preceded production with plan counts matching certified
expectations. Instrument hash verified before first and after last invocation (runner) and again
at readout (this document).

---

## 2. The decision statistic T (prereg §5, exactly as defined)

Definition applied — per row, the **mixture-composed** correction (identity from prereg §5,
composable from emitted row fields alone; rr1 N8 verified to 3.9e-16):

```
Δ̃_ij(h) = ln[ 1 + w_G,i·w_G,j·L_cat,i·L_cat,j·(e^{Δ_ij(h)} − 1) / (combined_i·combined_j) ]
```

Class sum D(h) = Σ_{(i,j)∈P} Δ̃_ij(h) over the ratified target set P;
**T = max_{h,h'∈grid} |D(h) − D(h')|** (full-grid range). The raw catalogue-leg Δ is the
diagnostic, never scored. All values below computed from `outputs/run_{venue}_{channel}.json`
rows; full precision in `readout_20260806.json` → `venues.<venue>.<channel>`.

### 2.1 Mixture-composed class sums D(h) [scored object]

| venue/channel | D(0.60) | D(0.73) | D(0.81) | D(0.86) | **T** (range; endpoints 0.60↔0.86 in all four) | chord D(0.60)−D(0.73) |
|---|---|---|---|---|---|---|
| joint_r1 / 1d | −1.661349e-05 | −6.069000e-06 | −3.038002e-06 | −1.931892e-06 | **1.468160e-05** | −1.054449e-05 |
| joint_r1 / 2d | −5.887665e-05 | −1.483637e-05 | −4.786495e-06 | −2.389123e-06 | **5.648753e-05** | −4.404028e-05 |
| iiib / 1d | −1.159605e-05 | −4.434363e-06 | −2.246039e-06 | −1.349903e-06 | **1.024615e-05** | −7.161692e-06 |
| iiib / 2d | −5.510299e-06 | −1.136030e-06 | −3.157384e-07 | −1.307574e-07 | **5.379542e-06** | −4.374269e-06 |

Reported, never scored (M-3 discipline, prereg §5): the level D(0.73) is the third column above;
the M-2-comparable endpoint chord D(0.60) − D(0.73) is the last column. All four composed chords
are **negative**: the correction removes more joint ln L at low h than at high h, i.e. it is
(microscopically) **high-h-preferring** everywhere.

### 2.2 Raw catalogue-leg diagnostics [reported alongside, never scored — prereg §5]

| venue/channel | D_raw(0.60) | D_raw(0.73) | D_raw(0.81) | D_raw(0.86) | T_raw (range) | chord_raw(0.60)−(0.73) |
|---|---|---|---|---|---|---|
| joint_r1 / 1d | +4.913479 | +50.14817 | +118.8707 | +186.1212 | 181.2078 | −45.23469 |
| joint_r1 / 2d | −2.264333 | −4.771684 | −6.742511 | −8.063511 | 5.799179 | **+2.507351** |
| iiib / 1d | +1.779282 | +24.83578 | +52.55062 | +79.29283 | 77.51355 | −23.05650 |
| iiib / 2d | −0.01534250 | −0.02695368 | −0.05781517 | −0.11131023 | 0.09596773 | **+0.01161117** |

(Sums verified against the runner-reported values to ≤ 1e-9 relative.) The raw 2D chord is
positive = low-h-preferring — the H-2 sign — at both venues; the mixture composition reverses the
sign of the class-level chord and suppresses the magnitude by ~5 orders (§6). Median per-pair
composition factor w_G²·L_cat,i·L_cat,j/(combined_i·combined_j) at h = 0.73: 1.018e-07 (joint_r1
1d), 1.524e-07 (joint_r1 2d), 1.553e-07 (iiib 1d), 2.542e-06 (iiib 2d); maxima 9.46e-02 /
1.29e-01 / 7.16e-02 / 3.74e-02 (run_*.json row fields). The single largest raw per-pair value,
Δ = +15.482 nats (joint_r1/1d pair (212, 854), h = 0.86, n_shared = 1), composes to
Δ̃ = +4.2e-18 (its factor is 8.0e-25).

---

## 3. Band comparison — X/Y and the per-unit x/y tables

Band of record (LOCKED): **X = 2.78, Y = 7.96** class-summed chord nats, both channels
(prereg §7.2/§7.3, ratified flag (d)). Per-unit currency (prereg §7.4 formula x = X/W, y = Y/W;
1D W from `band_derivation.json`/m4; 2D W from the certification-record completion table):

| venue/channel | n pairs | W (min-side w_pop share sum) | x = X/W | y = Y/W | **T** | T/W | T vs X | verdict input |
|---|---|---|---|---|---|---|---|---|
| joint_r1 / 1d | 349 | 19.803870 | 0.140376 | 0.401942 | 1.468160e-05 | 7.413498e-07 | T = 5.28e-06 · X (T < X) | NEGLECT branch |
| iiib / 1d | 280 | 14.728652 | 0.188748 | 0.540443 | 1.024615e-05 | 6.956612e-07 | T = 3.69e-06 · X (T < X) | NEGLECT branch |
| joint_r1 / 2d | 104 | 4.094919851 | 0.678890 | 1.943872 | 5.648753e-05 | 1.379454e-05 | T = 2.03e-05 · X (T < X) | NEGLECT branch |
| iiib / 2d | 21 | 0.664499854 | 4.183598 | 11.978934 | 5.379542e-06 | 8.095625e-06 | T = 1.94e-06 · X (T < X) | NEGLECT branch |

T < X in all four; T ≥ Y in none; gap [X, Y) in none. T/W < x everywhere (by ≥ 4.9e+04×) — the
per-unit read supports NEGLECT uniformly (no stratum-level per-unit escalation arises; see §4).

---

## 4. FULL [A2] stratified report (prereg §6 — class sum alone is inadmissible)

Stratified object: the per-pair composed chord c_ij = Δ̃_ij(0.60) − Δ̃_ij(0.86) at the class
argmax endpoints (0.60, 0.86 — identical in all four venue × channel), whose total equals the
signed class chord (−T in all four). Per stratum: n pairs, net sub-sum, positive-only sub-sum,
negative-only sub-sum. **Anti-dilution clause (§9a): NEGLECT is blocked if any single stratum
carries a one-signed sub-sum ≥ Y = 7.96.** Source: computed from run_*.json rows; full tables in
`readout_20260806.json` → `venues.<venue>.<channel>.a2_stratified`.

### 4.1 joint_r1 / 1d (349 pairs; T = 1.468160e-05)

| stratification | stratum | n | net | Σ pos | Σ neg | max one-signed |
|---|---|---|---|---|---|---|
| n_shared bands | 1 | 36 | −4.13e-10 | +3.65e-10 | −7.78e-10 | 7.78e-10 |
| | 2–10 | 95 | −4.57e-07 | +2.02e-09 | −4.59e-07 | 4.59e-07 |
| | 11–100 | 140 | −5.23e-07 | +8.95e-07 | −1.42e-06 | 1.42e-06 |
| | 101–1000 | 66 | −9.02e-06 | +1.14e-06 | −1.02e-05 | 1.02e-05 |
| | >1000 | 12 | −4.68e-06 | +7.61e-08 | −4.76e-06 | 4.76e-06 |
| overlap degree (max side) | 0/1/2/3/4/5/6 | 70/133/68/42/21/10/5 | −1.04e-05 / −4.67e-06 / +9.02e-07 / −5.23e-07 / −9.08e-10 / +1.89e-12 / +2.86e-11 | — | — | 1.04e-05 (deg 0) |
| w_G (degenerate — proven: single unique value per h: 0.095686608947 / 0.070802251082 / 0.060668219574 / 0.055607912665 at h = 0.60/0.73/0.81/0.86; matches prereg §6 expected 0.07080 at 0.73) | all pairs | 349 | −1.468160e-05 | +2.117213e-06 | −1.679881e-05 | 1.68e-05 |
| min-side w_share quartiles (cuts 0.004488 / 0.018078 / 0.069082) | Q1/Q2/Q3/Q4 | 88/87/87/87 | −1.95e-07 / −1.20e-06 / −7.96e-06 / −5.33e-06 | — | — | 7.99e-06 (Q3 neg) |
| in-C-4 vs outside | in_c4 | 80 | **+1.812e-06** | +2.117e-06 | −3.05e-07 | 2.12e-06 |
| | outside_c4 | 269 | **−1.649e-05** | +1.8e-18 | −1.649e-05 | 1.65e-05 |
| sign decomposition | pos/neg/zero | 56/290/3 | net −1.468e-05 | **+2.117213e-06** | **−1.679881e-05** | 1.68e-05 |

### 4.2 joint_r1 / 2d (104 pairs; T = 5.648753e-05) — **every nonzero chord is negative (one-signed class)**

| stratification | stratum | n | net = Σ neg (Σ pos = 0 in every stratum) | max one-signed |
|---|---|---|---|---|
| n_shared bands | 1 / 2–10 / 11–100 / 101–1000 / >1000 | 31/40/21/11/1 | −3.65e-07 / −4.79e-06 / −3.27e-06 / −4.69e-05 / −1.13e-06 | 4.69e-05 |
| overlap degree (max side) | 0/1/2/3/4/5 | 24/48/11/16/4/1 | −1.67e-05 / −3.94e-05 / −3.43e-07 / −8.16e-08 / −1.80e-10 / −5.1e-29 | 3.94e-05 |
| w_G (degenerate — same four values as 4.1) | all pairs | 104 | −5.648753e-05 | 5.65e-05 |
| min-side w_share quartiles (cuts 0.002349 / 0.009313 / 0.047187) | Q1/Q2/Q3/Q4 | 26/26/26/26 | −1.39e-07 / −3.72e-06 / −3.75e-05 / −1.52e-05 | 3.75e-05 |
| in-C-4 vs outside | in_c4 27 / outside 77 | | −1.800e-06 / −5.469e-05 | 5.47e-05 |
| sign decomposition | pos/neg/zero = 0/103/1 | | Σ pos = 0; Σ neg = −5.648753e-05 | 5.65e-05 |

### 4.3 iiib / 1d (280 pairs; T = 1.024615e-05)

| stratification | stratum | n | net | Σ pos | Σ neg | max one-signed |
|---|---|---|---|---|---|---|
| n_shared bands | 1 / 2–10 / 11–100 / 101–1000 / >1000 | 36/64/106/60/14 | −1.41e-09 / −1.29e-07 / −6.38e-07 / −6.28e-06 / −3.19e-06 | — | — | 7.07e-06 (101–1000 neg) |
| overlap degree (max side) | 0/1/2/3/4/5 | 63/107/56/34/15/5 | −7.48e-06 / −3.12e-06 / +6.25e-07 / −2.70e-07 / −6.48e-10 / −1.12e-12 | — | — | 7.48e-06 (deg 0) |
| w_G (degenerate — proven: 0.083202481114 / 0.061966841111 / 0.053323567925 / 0.049000197345 at h = 0.60/0.73/0.81/0.86; matches prereg §6 expected 0.06197 at 0.73) | all pairs | 280 | −1.024615e-05 | +1.390640e-06 | −1.163679e-05 | 1.16e-05 |
| min-side w_share quartiles (cuts 0.003039 / 0.013903 / 0.068685) | Q1/Q2/Q3/Q4 | 70/70/70/70 | −1.52e-07 / −1.29e-06 / −5.46e-06 / −3.34e-06 | — | — | 5.94e-06 (Q3 neg) |
| in-C-4 vs outside | in_c4 63 / outside 217 | | **+1.234e-06** / **−1.148e-05** | +1.391e-06 / 0 | −1.57e-07 / −1.148e-05 | 1.15e-05 |
| sign decomposition | pos/neg/zero = 39/239/2 | | net −1.025e-05 | **+1.390640e-06** | **−1.163679e-05** | 1.16e-05 |

### 4.4 iiib / 2d (21 pairs; T = 5.379542e-06) — **every chord is negative (one-signed class)**

| stratification | stratum | n | net = Σ neg (Σ pos = 0) | max one-signed |
|---|---|---|---|---|
| n_shared bands | 1 / 2–10 / 11–100 / 101–1000 / >1000 | 5/9/2/3/2 | −4.08e-09 / −1.68e-07 / −2.91e-10 / −8.11e-07 / −4.40e-06 | 4.40e-06 |
| overlap degree (max side) | 0/1/2 | 5/15/1 | −4.25e-06 / −1.13e-06 / −1.55e-10 | 4.25e-06 |
| w_G (degenerate — same four values as 4.3) | all pairs | 21 | −5.379542e-06 | 5.38e-06 |
| min-side w_share quartiles (cuts 0.001531 / 0.006758 / 0.017717) | Q1/Q2/Q3/Q4 | 6/5/5/5 | −3.26e-08 / −3.21e-08 / −9.13e-07 / −4.40e-06 | 4.40e-06 |
| in-C-4 vs outside | in_c4 5 / outside 16 | | −2.710e-07 / −5.109e-06 | 5.11e-06 |
| sign decomposition | pos/neg/zero = 0/21/0 | | Σ pos = 0; Σ neg = −5.379542e-06 | 5.38e-06 |

### 4.5 Anti-dilution clause — evaluated explicitly

**Largest one-signed sub-sum in ANY stratum of ANY stratification (including the class-wide sign
decomposition itself), per venue × channel:**

| venue/channel | max one-signed sub-sum (nats) | ≥ Y = 7.96? | blocks NEGLECT? |
|---|---|---|---|
| joint_r1 / 1d | 1.679881e-05 (class-wide Σ neg) | no — 4.7e+05× below | **no** |
| joint_r1 / 2d | 5.648753e-05 (class-wide Σ neg; class is one-signed) | no — 1.4e+05× below | **no** |
| iiib / 1d | 1.163679e-05 (class-wide Σ neg) | no — 6.8e+05× below | **no** |
| iiib / 2d | 5.379542e-06 (class-wide Σ neg; class is one-signed) | no — 1.5e+06× below | **no** |

No concentrated effect hides inside the diluted class sum: the *entire one-signed mass* of every
channel is 5–6 orders of magnitude below the blocking scale. Concentration structure (reported for
completeness, per-pair top-10 in `readout_20260806.json` → `per_pair_distributions.top10_abs_chord`):
the top pair carries 27% of T in joint_r1/1d ((573, 1521), c = −3.965e-06), 58% in joint_r1/2d
((927, 1035), c = −3.284e-05), 27% in iiib/1d ((573, 1521), c = −2.807e-06), 77% in iiib/2d
((98, 294), c = −4.126e-06) — real concentration, microscopic absolute scale.

Findings inside the stratified table (reported, none verdict-changing):

- **in-C-4 vs outside (stratum 4) is sign-split in 1d:** the in-C-4 sub-sum is net *positive*
  (low-h) at both venues (+1.81e-06 joint_r1, +1.23e-06 iiib) while the outside-C-4 sub-sum is
  strictly negative and ~9× larger — the M-4 supersession (flag (a)) was load-bearing: a C-4-only
  run would have measured the smaller, opposite-signed component.
- **The 2D channel is perfectly one-signed** (0 positive chords out of 103/21 nonzero) — coherent
  cancellation-free structure, exactly the coherence H-2 posits, but at 1e-05-nat class scale.
- **w_G stratification is degenerate as predicted** (prereg §6 item 2): single unique value per h
  per venue, equal on both pair sides, matching the quoted 0.07080/0.06197 at h = 0.73.

---

## 5. Verdicts (prereg §9 decision tree, mechanical)

Preconditions: no −inf rows (auto-escalate path not taken); no NaN-voided pairs; coverage exact
(§1). Tree: T ≥ Y → REGARD; T < X and no ≥Y one-signed stratum sub-sum → NEGLECT-WITH-NUMBER;
X ≤ T < Y → GAP; venue split → GAP-with-finding.

| venue | channel | T | vs X = 2.78 | vs Y = 7.96 | anti-dilution | **VERDICT** |
|---|---|---|---|---|---|---|
| joint_r1 | 1d | 1.468160e-05 | T < X | T < Y | not triggered | **NEGLECT-WITH-NUMBER** |
| joint_r1 | 2d | 5.648753e-05 | T < X | T < Y | not triggered | **NEGLECT-WITH-NUMBER** |
| iiib | 1d | 1.024615e-05 | T < X | T < Y | not triggered | **NEGLECT-WITH-NUMBER** |
| iiib | 2d | 5.379542e-06 | T < X | T < Y | not triggered | **NEGLECT-WITH-NUMBER** |

**Venue split check (§9 branch d):** none — both venues return the identical verdict in both
channels. GAP-with-finding is not invoked.

**GAP protocol (§10):** not applicable — no (venue, channel) scored into [X, Y). For the record,
the prereg's ordered MEASURE-MORE steps (which would apply, in order, and are **not executed**)
are: (1) densify the h-grid from the 4-point floor to the canonical 41-point grid (M-3
interior-excursion check applies); (2) full stratified drill-down with per-pair Δ̃ published for
the top decile by |Δ̃|; (3) per-unit re-expression T/W vs x and y — stratum with T/W ≥ y escalates
as REGARD-scale-on-a-stratum, T/W < x everywhere supports NEGLECT on the extended read; (4) only
if 1–3 leave the call between bands, return to the author with the full table — the author rules.

**The permanent NEGLECT-WITH-NUMBER record (prereg §9a — a value, not "small"):** measured
T = 1.468160e-05 / 5.648753e-05 / 1.024615e-05 / 5.379542e-06 class-summed mixture-composed chord
nats (joint_r1 1d/2d, iiib 1d/2d); levels D(0.73) = −6.069000e-06 / −1.483637e-05 /
−4.434363e-06 / −1.136030e-06 nats; per-unit T/W = 7.413498e-07 / 1.379454e-05 / 6.956612e-07 /
8.095625e-06 vs x = 0.140376 / 0.678890 / 0.188748 / 4.183598; full stratified table in §4 and
`readout_20260806.json`.

Per prereg §12 (model/effort policy), this mechanical readout presents the branch call; the
NEGLECT closure of the claim leg is the author's to enter in the claim/ledger files (none edited
here).

---

## 6. H-hypothesis implications (mandated statement)

**H-1 (claim file: the 1D low rail at h = 0.600, both venues, owned by the neglected pairwise
Eq. (31) correlations; refutation-by-argument collapsed after M-1's KERNEL-FINITE finding;
prereg §1 item 3: exit from LIVE only by measurement against this band).**
ADDRESSED AND CLOSED by measurement: the scored 1D cross-term chord is T = 1.468160e-05
(joint_r1) / 1.024615e-05 (iiib) class-summed nats — ~5 orders below X and ~5 orders below the
−1.1…−1.6 nats-to-next-grid-point depth of the rail itself (ledger #93). The Eq. (31) cross-term
**cannot own the 1D rail**. The cross-term leg of H-1 — and of H-3, which prereg §1 item 3 names
jointly — exits LIVE as NEGLECT-WITH-NUMBER, by measurement, not argument. (H-3's separate f_k
h-slope leg was bounded by M-3, context-only, and is not adjudicated here.)

**H-2 (claim file: the +0.05–0.07 2D displacement via shared-host coherence; REVIVED by M-2's
non-null 2D matched read — overlap stratum pulls low-h at +0.02070/+0.02225 nats/event,
cluster-robust p 0.0042/0.0050).**
The prereg's novel-result clause (§9b) defines the confirmation signature: a REGARD-scale term
**concentrated in the 2D channel** with **net sign preferring low h** (positive D(0.60) − D(0.73)).
Measured, on the scored statistic:

- **Channel concentration: present in relative terms** — 2D T exceeds 1D T at joint_r1
  (5.65e-05 vs 1.47e-05) and the 2D class is perfectly one-signed — but at absolute scale
  1.4e+05× below Y.
- **Sign: OPPOSITE.** The composed 2D chord D(0.60) − D(0.73) = −4.404028e-05 (joint_r1) /
  −4.374269e-06 (iiib): **high-h-preferring**, not low-h.
- **Scale: 5–6 orders below Y.**

**The M-2 revived-H-2 signature is NOT matched.** The Eq. (31) pairwise cross-term is thereby
**excluded as the mechanism behind M-2's 2D overlap-stratum residual**: a class-scale ~8-nat
(0.0207 × 385) low-h effect cannot be produced by a coupling whose total one-signed 2D budget is
5.6e-05 nats with the opposite sign. The residual's origin lies elsewhere — it is not
likelihood-factorization coupling; per prereg §1 consequence 1, "the bias explanation lies
elsewhere and saying so is equally a discharge of the mandate."

**Diagnostic footnote (reported, never scored — §5):** the RAW catalogue-leg 2D chord *does*
carry H-2's sign — chord_raw(0.60)−(0.73) = +2.507351 nats (joint_r1) / +0.01161117 nats (iiib),
low-h-preferring, with the raw joint_r1 2D full-grid range at 5.799 nats. The coherence H-2
posits is physically present *inside the catalogue leg*, but the mixture weight of that leg
(w_G·L_cat/combined per event; median pair-level composition factor 1.5e-07 joint_r1 / 2.5e-06
iiib at h = 0.73) annihilates it in the combined likelihood the posterior consumes, and the
composition's h-dependence (w_G falls from 0.0957 to 0.0556 across the grid) reverses the
class-level chord sign. This is recorded per M-3's aggregate-honesty rule; it does not enter the
verdict.

---

## 7. File inventory (this readout)

| file | role |
|---|---|
| `CROSSTERM_READOUT_20260806.md` | this document |
| `readout_20260806.json` | full-precision machine-readable readout: integrity block, per-(venue, channel) class sums (raw + mixture), T, band comparison, complete [A2] strata, anti-dilution evaluation, per-pair distributions and top-10, verdicts, venue-split check, gap-protocol listing, H-hypothesis statements, provenance |
| inputs (read-only) | `outputs/run_{joint_r1,iiib}_{1d,2d}.json`, `target_pairs_{joint_r1,iiib}.json`, `PREREGISTRATION_CROSSTERM_INSTRUMENT.md.DRAFT`, `outputs/chunks/*` (untouched) |

No file outside `crossterm_instrument/` was written; no prereg/claim/ledger text was edited;
`master_thesis_code/` untouched.

---

## CORRECTION (2026-08-07, appended post-adjudication — presentation only, verdicts unaffected)

The §0 headline factor range "between 4.9e+05× and 1.9e+05× below X" is wrong. The correct
X/T factors, from `readout_adjudication_20260807.json` (independent recompute, verdict
CONFIRMED, T values bit-identical): joint_r1/1d **1.89e+05×**, joint_r1/2d **4.92e+04×**,
iiib/1d **2.71e+05×**, iiib/2d **5.17e+05×** — i.e. the correct range is **~4.9e+04× to
~5.2e+05× below X = 2.78**. ("4.9e+05" was an exponent error for 4.9e+04; "1.9e+05" is an
interior value, not a range end.) The per-channel values in §3 and the anti-dilution table
(§ "below Y" factors) were verified correct as printed; no scored quantity or verdict changes.
Adjudication also noted two diagnostic-convention nits (median/quartile interpolation methods
on even-count sets, disclosed cuts verified) and a truncated-not-rounded x = 0.140376 print
inherited from prereg §7.4 — all non-binding; see `readout_adjudication_20260807.json`
`discrepancies` for the full list.

---

## STAGE-5 CLOSURE (2026-08-07)

The author (Jasper Seehofer) ruled on 2026-08-07: **NEGLECT-WITH-NUMBER accepted in all four
venue × channel cells**, CONDITIONED on an explicit re-evaluation trigger register. The closure
artifacts are:

- **`NEGLECT_TRIGGER_REGISTER.md`** (this directory) — the register of record: the quantitative
  anatomy of the 4.92e+04× minimum margin (T = F_eff × T_raw; in 3 of 4 cells the raw
  catalogue-leg range exceeds X, so the verdict rests entirely on the mixture composition), six
  named triggers (a)–(f) with thresholds and the cheapest re-evaluation instrument each (census
  re-run / composition arithmetic / full instrument), and the conditional-closure sentence: the
  NEGLECT stands unless at least one named trigger fires.
- **`../gate_b_20260730/BIAS_HISTORY_LEDGER.md` row 96** + §4 open thread 16 (owner of M-2's
  unowned 2D overlap residual, opened with the composition-annihilation finding) + the §5
  AUTHOR RULING (2026-08-07) block recording the conditional acceptance.

This section is a pointer only; no verdict, number, or table above is modified.
