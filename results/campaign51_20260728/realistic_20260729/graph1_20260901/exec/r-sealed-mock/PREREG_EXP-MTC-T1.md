# PRE-REGISTRATION EXP-MTC-T1: blind sealed-h anti-tuning mock (redteam T-1, stage (m2))

Written 2026-10-19 by mission M162, **before any draw, pool or run**. Author of record: Jasper
Seehofer.
**Status: planned, awaiting ratification.** Append-only: Result, Verdict and dated amendments are
appended below; nothing above the "Results" heading is edited after ratification. Vault mirror:
`wiki/meta/agentic-experiments-research-physics.md` § EXP-MTC-T1.

The registration of record is these four documents together:
- `REGISTRATION_DRAFT.md` (2026-09-03), as amended by
- `ERRATUM_m2_R27_R28.md` (rule blocks E1–E8 and the T1-SEAL-PARAMS block), with
- `POOL_OPTION_MEMO_m2.md` (the reasoning behind E-5) and
- `UNSEAL_CEREMONY_m2.md` (the procedure).

Where they differ, the erratum binds. Rule-block hashes: `.planning/research/THREAD.md`
§ "T-1 package hashes".

## Hypothesis

H-MTC-genmarg, kill criterion 1. The production estimator's response to the true H₀ is not tuned
to the production value h = 0.73. If the stack is run unchanged (frozen tree `71acb98b…`) on a mock
universe at a sealed h_inj ≠ 0.73, its posterior follows h_inj, and does not stay anchored at
the 0.73-production value 0.666.

## Prediction (one primary statistic, registered cells; erratum E-2/E-3)

Channels: 2D `combined_with_bh` (primary) and 1D `combined_no_bh` (must agree). Statistic: the
frozen T0 gradient-weighted scorer on the 55-node grid 0.600–1.000. It gives mean_h, MAP, σ_h and
the edge masses.

| cell | criterion (2D; 1D agrees) | expected if the hypothesis holds |
|---|---|---|
| TRANSFER | mean and MAP within 3σ of h_inj − 0.0641 (1D −0.0630) | **yes**: the production −0.064 offset is h-independent (R24 term) |
| UNBIASED-AT-h_inj | mean within 2σ of h_inj | possible; it would make the 0.73 offset 0.73-specific |
| TUNED | mean within 3σ of 0.6659 and MAP within 3σ of 0.665 | **no**: this cell refutes the hypothesis |
| AT-RAIL-LOW/HIGH | MAP on an edge node, or an edge-node mass > 1e-3 (the rail wins every tie) | no |
| INTERMEDIATE | several / none / channel disagreement / σ-sensitive | no prediction; banked |
| NO-READ | any gate red | nothing banked |

Mapping to kill criterion 1: the proposed mapping is in erratum E-8; the TRANSFER mapping is a
[RULE] on the ratification card.

## Setup

- Prior: P-A U[0.79, 0.82], drawn once from `secrets.SystemRandom` by
  `scripts/sealed_mock/draw_sealed_h.py`, committed as sha256(h_inj_6dp | salt).
- Code: H_code chosen at launch with tree `71acb98b0fd8a0f9d3aad9595eb9fc084cae0b6d`. Strict
  HEAD equality, dirty count 0 per task (R24: no code change before T-1).
- Data: GLADE+ reduced catalogue md5 `c52c13b5…`; Barausse M1 population; the completeness cache;
  `REGISTERED_RESOLVED_FLAGS` verbatim.
- Pool: P2 first (reuse the 707-file pool of record, list-md5 `75f4030d…`, behind a paired
  20-task 1e-3 invariance gate at h = 0.67). P1 (fresh sealed pool ≤ 250 GPU-h) if P2 fails.
- Simulate: at h_inj, N_tasks per cluster read R-1. Evaluate: 55 nodes, venue iiib, cap 94
  core-h. Pool: ≤ 10 GPU-h (P2) or ≤ 250 GPU-h (P1).
- Blindness: erratum E-6 whitelist; monitoring by `squeue`/`sacct` only; agents never read sealed
  run logs.

## Metric and instrument check

- Metric: the cell, by erratum E-2 then E-3, assigned at the unseal ceremony U-4.
- Instrument check (run before any metric is read):
  - the pre-unseal reader reproduces the 0.73 re-baseline anchors (2D mean 0.665854 / σ
    0.018475; 1D 0.666987 / σ 0.017526) from the banked `run_20260827_headreadout_iiib` likelihood
    table to ≤ 1e-6;
  - its N_used equals the distinct `event_idx` count;
  - VOID if either fails. VOID ⇒ the read is NO-READ.

## Dated default

The deadline is 2026-12-15, end of day Europe/Berlin. No banked cell by then ⇒ kill criterion 3
fires: methods paper, closure reported unblinded (erratum E-7). No third extension without a new
author ruling.

## Structural blindness (registered limitations)

- An h-invariant tuning, meaning a defect biasing every truth by the same −0.064, lands in
  TRANSFER and is classified, not exonerated. The R24 B_num term is exactly such a candidate.
- Common-mode population-model defects are invisible.
- One seed: seed scatter at h_inj is unmeasured beyond the 3σ bands.
- Under P2, T-1's literal "fresh pool at that cosmology" is replaced by a tested invariance
  property.
- The windows overlap for σ ≳ 0.010–0.015 (erratum E-2 disclosure), so INTERMEDIATE (several)
  is a live outcome.

## Linked

Hypothesis H-MTC-genmarg; kill criteria 1 and 3; docket R24–R28; redteam T-1 and T-4
(`results/redteam_20260726/PHYSICS_METHODOLOGY_REVIEW.md:401-440`); GitHub #39; missions M162 (this
package), M38 (cluster window), M65 (landing).

## Results

*(appended after the unseal row; empty at registration)*
