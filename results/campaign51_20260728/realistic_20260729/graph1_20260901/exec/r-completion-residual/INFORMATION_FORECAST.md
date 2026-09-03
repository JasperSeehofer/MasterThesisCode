# Stage-1 information forecast — Branch G (r-completion-residual) and Branch H (r-cone-loss)

Date: 2026-09-03. Author: the wave-3 prereg author (top-tier). Verdict-free; every number is a
forecast input with its source, never a result. Companion to the two REGISTRATION_DRAFT.md files.

## 1. What a perfect analysis of the banked data can say (rule 9 — free re-reads first)

Both arms are re-reads: the production post-flip re-baseline CSVs (rows #298/#299/#302) and the 67
post-flip S3 cell-S universes (rd-s3-readout) already contain every per-event likelihood the two
questions need. Neither arm requires a new evaluation to reach a disposition. The cluster caps
(≤ 80 / ≤ 20 CPU-h) are therefore ceilings on OPTIONAL replication, not on the decisive reads.

## 2. Branch G — the completion-leg residual

| forecast input | value | source |
|---|---|---|
| object, pre-flip, inferred | ≈ −0.147/event (dark matched-channel shortfall) | B4_3 §4.4 (ARITH on C5 0.7134/σ 0.0277 and tilt +0.1326) — STALE, [INFER] |
| harness dark full score, post-flip | +0.0082 ± 0.0063 (67 universes, 11,525 events) | `b8_cal_harness_work_s4_postflip` checkpoints, `score_at_truth.no_bh.dark` |
| harness catalogue-hosted full score | +0.587 ± 0.064 (843 events) | same; the S3 DEFECT-SIGNATURE locus |
| production dark N, SE | 1512 events; SE ≈ 0.68/√1512 = 0.0175 per event | per-event SD 0.68 (harness), production pool census |
| I_1D post-flip | 3256 (σ_h 0.017526) | re-baseline iiib 1D |
| F band in h (context) | 3·F·σ_floor = 0.0600 vs offset −0.0630 | F 11.44 (DEFECT-context), floor 0.001747058397810697 |

**Expected outcome:** INTERMEDIATE (a) — harness-clean, production-displaced. Reasoning: the harness
dark class is already clean on the FULL score at ±0.006; the matched-channel score removes only a global
term, so |T_harn| ≲ 0.02 is the likely bound (3.2σ detectability). Production's dark matched-channel
score is the unknown: if the §4.4 inference survives the flip at even a third of its size (−0.05), Z_prod
≈ −2.9 to −8 — displaced. Probability weights (author's own reading, not a measurement): INTERMEDIATE (a)
0.6; FLOOR-CONSISTENT 0.2 (the flip absorbed most of the §4.4 shortfall — its prediction (d) in B4_3 §7
did include "+ a closed dark-class residual ⇒ 0.73"); ILLEGITIMATE 0.1; INTERMEDIATE (b)/(c) 0.1.

**What each outcome changes on the board:**
- ILLEGITIMATE ⇒ c-residual-illegitimate promoted; a `/physics-change` intake on the completion-leg
  normalisation; the "largest unexplained item" becomes a FIX candidate; d-paper-1d2d-verdict waits.
- FLOOR-CONSISTENT ⇒ c-residual-floor-consistent promoted; the 1D rail −0.063 is booked as
  floor-consistent given F (with the rail disclosure); d-residual-attribution's first bucket closes at
  a bound; the paper's "honest-bound framing" gets its number.
- INTERMEDIATE (a) ⇒ neither claim; the residual moves to the THIRD bucket (generator–estimator
  population mismatch) with T_prod ± SE — a sharper object than "−0.14/event" because it is a measured
  per-event score with a class label and a closure identity; Graph 2's question layer inherits it.
- Effect sizes vs bands: the arm resolves illegitimate components ≥ 0.02/event (14 % of −0.14) and
  production displacements ≥ 0.05/event; both are well inside the plausible range of the object.

## 3. Branch H — the cone loss

| forecast input | value | source |
|---|---|---|
| mirror-fleet cone loss | 16.8 % (380/2261); 16.3 % (380/2336, different fleet) | R-MKER-6 census; CMEM-A1 |
| production cone loss (disclosed-seen) | 13.2 % (10/76); P6 66/76 | stage-0 census 2026-09-03; production log line 8622 |
| in-catalogue impostor score, pre-flip | −1.707/event | C5 (ASSUMPTION-JOIN, secondary) |
| I_1D, I_2D | 3256, 2930 | re-baseline |
| SE(Δh_cone,1D) | ≈ 0.0007 | 0.68·√11.5/3256 |

**Expected outcome:** IMMATERIAL-FLOOR-SHARE. Reasoning: 10 events × O(−1.7) nats/h ⇒ Δh ≈ −0.005, φ ≈ 8 %
of the −0.063 offset; even at 3× that pull φ ≈ 25 % (INTERMEDIATE). CONE-OWNS-FLOOR needs a per-event
excess pull of ≈ −20 nats/h on each OUT event — 30× the in-catalogue class mean; implausible but exactly
what the registered statistic would show if the CMEM rerouting hypothesis were strongly true.
Probability weights: IMMATERIAL 0.7; INTERMEDIATE 0.25; CONE-OWNS-FLOOR 0.05.

**What each outcome changes:** IMMATERIAL ⇒ q-cone-loss SETTLED (charter kill criterion), the board's
"leading candidate for the absolute floor" is demoted to a bounded ≤ 20 % share, and d-residual-attribution's
"irreducible venue physics" bucket loses its largest named member — the floor must then be carried by
depth skew (73.0 % ± 1.4 %, 16σ) and photo-z starvation (S0-B). CONE-OWNS-FLOOR ⇒ the floor is geometric
and the paper reports it as such. INTERMEDIATE ⇒ the share is banked; the z-window component question
(h-dependent membership) returns as the one thing a revision could resolve.
Secondary information (free): whether the S3 catalogue-hosted DEFECT-SIGNATURE (+0.587/event) sits on
the OUT events — this is the first paired test of [CMEM]'s mechanism at n ≈ 140 vs the A1 read's 68 % power.

## 4. Information per CPU-h — honest ranking

Both arms cost ≈ 0.1–0.2 CPU-h for their decisive reads; the ratio is dominated by the numerator.
- **Higher information per CPU-h: Branch H (cone loss).** Its most likely outcome SETTLES a charter
  question outright (kill criterion reached), demotes a named "leading candidate", and delivers a paired
  test of the [CMEM] mechanism as a by-product — all at zero compute, with an 11-SE margin to materiality.
- **Higher absolute information: Branch G.** It re-measures the largest unexplained item on the board
  post-flip with a class label and a closure identity, and bounds the illegitimate share at ±0.02/event —
  but its most likely disposition is a bound routed to a third bucket, not a settled question.
- Sequencing recommendation to the chair: run H's reads first (they also validate the production JOIN and
  the sky-scatter law G-4 that G's class labels share), then G. Neither needs the optional cell R
  tonight; cell R is spent only on ILLEGITIMATE / INTERMEDIATE (b), and only under docket 2.2.
