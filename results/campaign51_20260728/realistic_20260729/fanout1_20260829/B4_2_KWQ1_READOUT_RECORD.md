# B4.2 "KW-Q1" independent-reader readout — RESULT RECORD

**Stamp:** read out 2026-08-29 by the independent reader; run by the orchestrator; launched
under rows #222/#223 — charter node B4.2. Foreground only, no ssh, no git, append-only.

**Registration:** `CLAIM_IMPOSTOR_DRAG_20260829.md` §1.3 (statistic, bands, gates, invariants,
blindness, falsifier, cost) + §1.4 (merge declaration); run form fixed by
`B4_2_KWQ1_RUN_FORM_NOTE.md` (`theta_sites="2.2"`/unsmeared is the registered run form; `s_imp`
is form-invariant by the cancellation algebra there); docket `SYNTHESIS_DOCKET_1_20260829.md`
§2 "B4 [IMP]" conditions (a)–(d).

## 0. Comprehension-first summary

KW-Q1 asks whether the impostor-drag remainder localised to the lowest-z quartile (C2) is caused
by a **kernel-width** misstatement in the photo-z error law, by widening/narrowing the photo-z
kernel by a factor of √2 either side of its registered width and watching how much the q1
impostor-leg score at truth moves. It moves very little: widening the kernel by √2 changes the
q1 score by **+8.5 %** of the score's own size — an order of magnitude below the 50 % bar that
would implicate kernel width, and well inside the ≤ 20 % bar for calling it inert. All three
gates pass (bit-identical parity, exact assembly identity, the catalogue leg does respond to the
width change so the gate is not vacuous), and the q1 localisation itself is *reconfirmed* on this
run (92.2 % of the impostor-leg score sits in the lowest quartile, against a 50 % falsifier
floor). The verdict is **KERNEL-WIDTH-INERT**, robust seed-by-seed (every one of the 4 individual
seeds independently lands inside the INERT band, none within a factor of 3 of the OWNS
boundary) — but it is REPORTED-ONLY, not adopted, because the θ-hook instrument that KW-Q1 rides
was separately certified INSTRUMENT-DEFECT on a different design (the b0i mirror score-at-truth
null test, S0-A). KW-Q1's own design differs (a within-run paired comparison across kernel-width
nodes on the same events, not a score-at-truth null), so the defect does not automatically
transfer, but the instrument itself is not yet clean, so this verdict is carried with that
disclosure rather than banked outright.

**Bottom line for the charter:** B4 does **not** merge into B1. The remainder's mechanism moves
to B4.3 — the mixture-weight/catalogue-depth h-slope derivation (C6 (b)/(c)) plus, if a
non-physics hook ruling authorises it, the 3.4 CPU-h per-candidate instrumented run.

## 1. T-ID / PARITY (seed 900101, node_truth_ft_sites2.2_nosmear)

Compared the primary run's truth-node CSV against an independent same-commit re-evaluation
(`kwq1_parity_run`) at both h-nodes.

| column | h | n events | max\|Δ\| |
|---|---|---|---|
| `combined_no_bh` | 0.725 | 174 | 0.0 |
| `combined_no_bh` | 0.735 | 174 | 0.0 |
| `L_cat_no_bh` | 0.725 | 174 | 0.0 |
| `L_cat_no_bh` | 0.735 | 174 | 0.0 |

**PASS — bit-identical** on both columns, both h-nodes, all 174 events. {this readout;
`kwq1_registered_run/s0a_seed900101/node_truth_ft_sites2.2_nosmear/simulations/diagnostics/event_likelihoods.csv`
vs `kwq1_parity_run/s0a_seed900101/node_truth_ft_sites2.2_nosmear/simulations/diagnostics/event_likelihoods.csv`;
2026-08-29}

Also confirmed: the earlier scorer invocation without the `_ft_sites2.2_nosmear` suffix flags
found 0 rows on disk (`hier_s0_registered_run/logs/runner3_wave2pre_20260829.log`,
"12 node CSV(s) missing on disk", `2026-08-29T23:15:32+02:00 START KW-Q1 score`, exit rc=1) — a
runner-side path/invocation error (looked for `node_*_ft/` instead of
`node_*_ft_sites2.2_nosmear/`), not a measurement. The re-invocation with the correct
`--theta-sites 2.2 --smear off` suffix produced the registered output below.

## 2. Independent re-derivation of S(s) and R

Re-derived from scratch (not importing `kwq1_score.py` or `hier_s0_driver.py`) directly from the
per-node `event_likelihoods.csv` files, using the identical subtraction to C2 /
`b4_imp_stage1_forecast.py`: `cat_term = (alpha_G_phi/r_Malm)*L_cat_no_bh/D_tilde_phi`,
`pure = clip(combined_no_bh - cat_term, 0, None)`, secant of `ln` over h = 0.725/0.735,
`s_imp = s_full - s_pure`. Frozen q1 membership loaded from `b4_imp_stage1_events.csv`
(`arm=="ft"`) quartiled by `b4_imp_stage1_forecast.json`'s `covariates.ft.z_true.edges[0]` =
0.35750209116114345 (the "z_true < 0.358" cutoff).

### Pooled (4 seeds, 191 q1 event-pairs)

| node (s) | S(s) | sem | n |
|---|---|---|---|
| s_minus (1/√2) | −1.0456670 | 0.076542 | 191 |
| truth (1) | −1.0205308 | 0.069300 | 191 |
| s_plus (√2) | −0.9591134 | 0.062842 | 191 |

**R = [S(√2) − S(1/√2)] / \|S(1)\| = +0.0848123** — matches the scorer's `kwq1_score_output.json`
to full float precision (independently re-derived, not copied).

**Ordering:** S(1/√2) = −1.0457 < S(1) = −1.0205 < S(√2) = −0.9591 — monotonically increasing
with s (widening the kernel makes the q1 impostor-leg score *less* negative), but the whole
excursion across a factor-2 width range is 8.5 % of \|S(1)\|.

### Per-seed (robustness check, not part of the registered pooled statistic)

| seed | S(1/√2) | S(1) | S(√2) | R (per seed) |
|---|---|---|---|---|
| 900101 | −1.02397 | −0.95125 | −0.87528 | +0.1563 |
| 900102 | −0.89926 | −0.90388 | −0.86440 | +0.0386 |
| 900103 | −1.17319 | −1.12050 | −1.04939 | +0.1105 |
| 900104 | −1.08718 | −1.09299 | −1.03083 | +0.0516 |

Across-seed: mean R = 0.0892, SD = 0.0546, SEM (N=4) = 0.0273. **All four individual seeds land
inside the INERT band** (max \|R_seed\| = 0.156, still 22 % below the 0.2 INERT ceiling and far
below the 0.5 OWNS floor) — the pooled INERT verdict is not a borderline call that a different
seed draw would flip.

### q2–q4 share (falsifier input, at truth/s=1)

| quartile | Σ(mean·n) | share of total |
|---|---|---|
| q1 (z<0.358) | −194.921 | 92.25 % |
| q2 | −15.384 | 7.28 % |
| q3 | −0.994 | 0.47 % |
| q4 | −0.005 | 0.002 % |

**Falsifier (A14) NOT withdrawn:** q1 share = 92.25 % ≥ the 50 % floor — the C2 low-z
localisation is reconfirmed on this run, at an even higher concentration than the 12-seed
forecast (91.7 % ft / 86.2 % fc).

## 3. Gates

| gate | value | threshold | verdict |
|---|---|---|---|
| GATE I (assembly identity, cat+comp≡full) | max_rel 7.613×10⁻⁸ | ≤ 2×10⁻⁶ | **PASS** |
| GATE ENG (catalogue leg differs across s, active rows) | 486/486 active rows differ (fraction 1.0) | ≥ 0.99 | **PASS**, non-vacuous |
| GATE T-ID / PARITY | max\|Δ\| = 0.0 on `combined_no_bh`, `L_cat_no_bh`, both h, seed 900101 | bit-identical | **PASS** |

All three gates independently re-derived (§1–§3 above) and match the scorer's own printed values
exactly.

## 4. Band read (§1.3 bands)

|R| = 0.0848 ≤ 0.2 ⇒ **KERNEL-WIDTH-INERT**.

## 5. A15 — seed-generalisation check

The registered instrument's A15 note forecast a seed-generalisation SEM of S ≈ 0.073 (9 % of
\|S(1)\|≈0.80), obtained by scaling the 12-seed pooled-per-event SEM (0.042 on −0.798) up to
N = 4 by √(12/4). This readout computes the actual quantity directly from the 4 registered
seeds' own per-seed means (§2 per-seed table):

- Per-seed S(1) (truth node, q1): −0.95125 (900101), −0.90388 (900102), −1.12050 (900103),
  −1.09299 (900104).
- **Across-seed SD of S(1) = 0.10584** (10.4 % relative to \|S(1)\| = 1.0205); **across-seed SEM
  (N=4) = 0.05292**.

This is somewhat tighter than the extrapolated forecast (0.073) but of the same order — expected
sampling noise in a 4-of-12-seed SD estimate, not a discrepancy that bears on the verdict. More
directly relevant: **R itself is a within-seed, within-event paired ratio** (all three s-nodes
share identical events per seed), so the large seed-to-seed level shift visible in S(1) alone
(−0.90 to −1.12) mostly cancels in the ratio — the across-seed SD of R is only 0.0546 (§2), and
every individual seed's R stays inside INERT. The INERT verdict generalises comfortably; it is
not a borderline read that 4-seed noise could have pushed into MIXED or OWNS.

Note: the pooled mean S(1) at these 4 seeds (−1.0205) differs from the 12-seed forecast's q1
mean (−0.798, C2 table) by more than the naive per-event SEM would suggest — consistent with the
0.106 across-seed SD found here (a 4-seed subsample of a 12-seed fleet with SD ≈ 0.11 per seed
easily lands 0.2 away from the full-fleet mean). This affects the absolute level of S, not R or
the band, since R is paired within these same 4 seeds throughout.

## 6. Instrument disclosure (carried forward, not resolved here)

The same θ-hook driver family (`hier_s0_driver.py`, S0-A) was separately certified
**B0-A′ INSTRUMENT-DEFECT** on the b0i mirror score-at-truth null test (`hier_s0_registered_run/s0a_score.md`:
Z_b = −3.676, Z_s = −7.079, "INSTRUMENT-DEFECT -- STOP (prereg §4.5)"; forensic in progress). KW-Q1
is a different design — a within-run paired comparison across kernel-width (s) nodes on the FT
config, not a score-at-truth null test against zero — so the defect is **not automatically
inherited**, but the instrument as a whole is not yet certified clean. **This verdict is
REPORTED-ONLY, carried with this disclosure**, per the launch instruction; it is not banked as an
adopted result until the S0-A forensic resolves.

## 7. Verdict of record and next step

**KERNEL-WIDTH-INERT (REPORTED-ONLY, with the §6 instrument disclosure).** Per §1.4 of
`CLAIM_IMPOSTOR_DRAG_20260829.md`: **B4 does NOT merge into B1.** The docket §2 "B4 [IMP]"
condition (d) resolves to its INERT branch: **B4.3 = the mixture-weight/catalogue-depth h-slope
derivation** (C6 (b)/(c): `s_β = −3.2891/h` at first order, 63 % of the impostor-leg score;
`s_L = −27.08/h` on active events, the catalogue-depth skew) **plus, contingent on a
non-physics-hook ruling, the per-candidate instrumented run** (part 1 §7, 3.4 CPU-h) to attribute
the q1 impostor z-offsets directly. Neither of these is authorised by this readout; they are the
named next items for the orchestrator's path decision.

## 8. Cost (measured)

| item | wall_s | cpu_per_job | CPU-h |
|---|---|---|---|
| KW-Q1 main run (4 seeds × 3 nodes × 2 h, `--jobs 1`) | 1417.786 | 14 | 5.514 |
| Parity re-evaluation (1 seed, truth node) | 164.070 | 14 | 0.638 |
| **Total measured** | | | **6.152** |

Against the registered 8.4 CPU-h primary estimate (`"2.2"`/unsmeared form): **≈27 % below
estimate** (0.73×). Source: `hier_s0_registered_run/logs/runner3_wave2pre_20260829.log`
(`wall_s`/`cpu_per_job` JSON blocks, `2026-08-29T22:49:07+02:00 START KW-Q1 ft 4 seeds...` and
`2026-08-29T23:12:46+02:00 START KW-Q1 parity...`).

## 9. Files

- Independent re-derivation script (this reader, not committed to the run directory):
  scratchpad `kwq1_independent_rederive.py`.
- Raw scorer output: `kwq1_registered_run/kwq1_score_output.json`.
- This record's own numeric dump: `b4_2_readout.json` (this directory).
