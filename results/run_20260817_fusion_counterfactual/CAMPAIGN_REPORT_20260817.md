# Campaign Readout Report — the fusion counterfactual

*Production counterfactual measurement · selection-fusion thread (rows #115–#118) · 2 cells
× 2 venues × 41 h-points × 1588 events (164 array tasks) · 2026-08-17*

**The question:** now that the fused-survival estimator is the production default
(`[PHYSICS]` `2b10b8b8`), *how much did production actually move, channel by channel — and
does the catalogue leg's exemption from the fusion skew the mixture enough to force the
deferred [P3] fork back open?*

> **Verdict strip:** Measurement completed in full, all NULLs held. 1D-dominated action
> confirmed exactly as row #118 MAJOR-1 predicted; no MAP moved anywhere; the mixture skew
> exists, is small in the mean, and is concentrated in the ~10% of events that have a
> catalogue leg at all. **Presented, not adjudicated** — no bands existed (A3: measurement
> seeded from nothing), so there is no pass/fail; there are two author decisions (§10).

## 2. The goal

**The prior finding.** The venue arm (rows #115–#116) proved the fused correct-form
estimator repairs the 2D selection bias *in the venue* (DS-G1 −11.8±0.61 in band, coverage
restored). The production derivation (L6-DER3) said the same fusion belongs in production's
completion legs, and the author ratified shipping it. But A3 forbids carrying any venue
magnitude across, and the verifier (MAJOR-1) showed production sits in the opposite regime
the proposal assumed: σ_cond ≈ 8.8e-8, the sharp-likelihood limit, where [P1] should be
nearly inert and the pair's action should ride on [P2].

**The objection this measures.** Three unknowns survived the gate: (i) the actual
production-side magnitude of each leg separately (a lumped number could hide a
cancellation); (ii) whether the landed change moves the posterior of record at all
(campaign-re-run scope); (iii) whether leaving the catalogue leg unfused (ratified row 2
deferral) skews the G/Ḡ mixture materially — the "unless material" condition attached to
that deferral.

**Design in one line:** two cells (`off` = pre-#118 estimator, `fused` = production), same
seeds/CRBs/pools/catalogues/commit, both venues; the `1d`/`2d` single-leg cells recovered
channel-wise from the pair (the 1D mixture never reads `B_num_wbh`, the 2D never reads
`B_num` — proven bit-exact by `test_fused_pairing_identity`), halving the fleet.

## 3. The design — cells and control

| cell | role | what breaking looked like |
|---|---|---|
| `off` (both venues) | fresh pre-#118 twin at the gate commit | NULL-2: any drift vs the run of record beyond the two ratified 08-12 divergence classes would VOID the comparison |
| `fused` (both venues) | the production estimator | selection-side leak (w̃_G, D̃^φ, …) ≠ 0 would mean the fusion touched the normalisation it must not touch |

**The control is NULL-2 and it passed at a stronger level than required:** the off twin's
1D channel is **bit-identical** (max rel drift 0.0) to `run_20260804_postfix`, and the 2D
drift is ≤2.5e-13 — five orders inside the ratified 1e-8 class. This simultaneously proves
the apparatus and explains why M-2 reproduces the N-2 measurement to three decimals.

## 4. The result

**Per-h Σ Δln (fused − off), the tilt that drives any MAP motion** (source:
`readout.json`, `sum_delta_ln_by_h`):

| | iiib | joint_r1 |
|---|---|---|
| **[P2] 1D channel** — chord / central@0.73 | **+24.6 / +30.9 nats/h** | **+22.7 / +32.3 nats/h** |
| **[P1] 2D channel** — chord / central@0.73 | **+1.2 / +1.2 nats/h** | **−3.3 / −2.9 nats/h** |

Headline cards (expected vs observed):

- **[P1] production magnitude** — expected: near-inert, |Σ| ≤ 20 nats/h prior bracket
  (N-2 §3.1, restated by MAJOR-1). Observed: |1.2| and |3.3| nats/h, sign venue-dependent.
  **The regime call was right.**
- **[P2] production magnitude** — expected: the N-2 measurement of record as context
  (+24.6/+22.7 chord). Observed: **identical to 3 decimals** — the 1D leg is bit-stable
  across the code drift, so N-2's numbers ARE this measurement's numbers.
- **Posterior motion (M-3)** — expected: unknown (the campaign-re-run input). Observed:
  **zero MAP motion in every channel × venue** at the 41-point grid resolution; 1D width
  tightens (σ 0.0068→0.0053 iiib, 0.0086→0.0065 joint), 2D width unchanged (0.0177→0.0178,
  0.0216→0.0217).

**Read this before anything else:** the shape is *suppression-without-motion*. Both S
factors are < 1, so every per-event likelihood LEVEL drops (median Δln at 0.73: −1.32 (1D),
−0.41 (2D)); what matters for H₀ is only the h-slope of that suppression, and it is
carried almost entirely by the 1D channel — where the MAP was already hard-railed at 0.600
with a −180…−250 nats/h down-tilt of record. +25 nats/h of counter-tilt narrows the 1D
posterior but cannot un-rail it, and the 2D channel's ±3 nats/h does not graze its
σ_h ≈ 0.018–0.022 posterior. This is the predicted outcome, stated before the run in the
prereg's regime paragraph — not a disappointment discovered after it.

## 5. The mechanism check

No dose axis was registered (measurement, not arm). The mechanism-consistency reads that
WERE locked: (a) channel-wise decomposition — the fused pair's 1D leg must equal the
promoted N-2 branch: it does, bit-consistently (M-2 ≡ N-2 of record); (b) the M-1
magnitude must sit inside the only prior bracket that exists (|Σ| ≤ 20): it does, in both
venues; (c) the venue-independence of the completion machinery — iiib and joint_r1 move
together on M-2 (Δchord ≈ 1.9 nats/h) exactly as the N-2 run found, while M-1's small
venue split (+1.2 vs −3.3) sits in the catalogue-bearing subset where venues genuinely
differ.

## 6. The scorecard

| read | registered as | iiib | joint_r1 | status |
|---|---|---|---|---|
| **M-1** [P1] 2D tilt chord / central | measured, no band | +1.245 / +1.158 | −3.268 / −2.893 | REPORTED |
| **M-2** [P2] 1D tilt chord / central | measured, no band (N-2 context: +24.6/+22.7 chord) | +24.588 / +30.901 | +22.736 / +32.315 | REPORTED (≡ N-2) |
| **M-3** 1D MAP off→fused | measured | 0.600→0.600 (railed, σ 0.0068→0.0053) | 0.600→0.600 (σ 0.0086→0.0065) | REPORTED |
| **M-3** 2D MAP off→fused | measured | 0.780→0.780 (σ ≈ unchanged) | 0.800→0.800 (σ ≈ unchanged) | REPORTED |
| **M-4** skew @0.73 | measured → author [RULE] | mean Δshare +6.1e-3; 161/1588 movers, med +0.034, max +0.204 | mean +5.7e-3; 159/1588, med +0.022, max +0.203 | **RETURNS TO AUTHOR** |
| NULL-1 metadata (cells, freeze, commit) | must hold | PASS 41+41 | PASS 41+41 | PASS |
| NULL-2 off-twin drift | ≤ ratified 1e-8 class | 1D 0.0; 2D 2.5e-13 | 1D 0.0; 2D 1.6e-13 | PASS |
| selection-side leak | must be 0 | 0/65108 cells | 0/65108 | PASS |

No bands were locked (A3); nothing above was re-centred after readout.

## 7. Vocabulary

- **`off` / `fused`** — the pre-#118 estimator vs the landed production form
  ([P2] S̄_φ in the 1D completion numerator + [P1] S_4D inside the 2D mass quadrature).
  Decision cells: the only two run.
- **Σ Δln tilt (nats/h)** — the h-slope of the summed per-event log-likelihood difference
  fused−off; the quantity that can move a MAP. Values: +23–32 (1D), −3…+1 (2D).
- **S̄_φ(z;h)** — φ-marginal detection survival; < 1 everywhere, → 0 at the horizon. The
  [P2] weight. Suppresses the completion numerator's beyond-horizon overweight.
- **g_sel,prod** — the fused 2D mass kernel: population φ × observed-mass Gaussian ×
  S_4D in one integral. At production σ_cond ≈ 1e-7 it ≈ g_i·S at the measured mass —
  why [P1] is near-inert here (and why the venue, with its different regime, saw −11.8).
- **share_cat** — per-event catalogue fraction of the 1D mixture, A_cat/(A_cat+B_num).
  Median 0 (most events are completion-dominated); the ~160 movers have median 0.62.
- **Δshare_cat (M-4)** — how much the unfused catalogue leg gains weight when the fused
  completion leg shrinks: the concrete size of the row-2 deferral's mixture skew.
  Decision value: median +0.02–0.03, max +0.20, confined to catalogue-bearing events.
- **Rail at 0.600** — the 1D MAP sits on the grid edge (photo-z root cause of record,
  ledger #36). Still owned by photo-z: the fusion narrows but does not move it.

## 8. Why the numbers stand

- **Validity:** one commit across all 164 tasks (`ac24b632`, docs-only descendant of the
  gate commit); cells verified in every `run_metadata.json`; freeze flag null; same
  CRB/pool symlinks as every run of record (seed61000); seed-identical across cells
  (777000+task). The control (NULL-2) is bit-exact on the decisive channel.
- **Independent recompute:** the channel-decomposition identity was proven in the test
  suite before launch (`test_fused_pairing_identity`, bit-exact); M-2's three-decimal
  agreement with the independently-computed N-2 readout (different session, different
  code commit, same venue) is an end-to-end cross-check of the entire 1D pipeline.

## 9. Flags

1. **Sidecar path repair (compliance deviation, enters the ratification bundle).** Both
   joint_r1 arrays initially failed on a stale absolute `parent_csv` in the realization
   sidecars (pre-rename repo path). Repaired path-only after verifying the current file
   hashes exactly to the recorded `parent_csv_sha256`; backups kept; ~2 CPU-h burned;
   arrays resubmitted clean. Does not change any number. Gotcha added to the cluster
   skill.
2. **Budget:** ~170.4 of 270 CPU-h — in budget (the pessimistic-rate discipline of row
   #116 held).
3. **Grid resolution (interpretive).** M-3's "no MAP motion" is at the canonical 41-point
   grid (peak step 0.005). The 2D tilts (±3 nats/h) are far too small to move a σ≈0.02
   posterior, so a zoom grid would not change the reading; stated for completeness.
4. **Calibration ≠ magnitude (interpretive, carried from MINOR-4/#66-#67).** This run
   measures magnitude. Whether the fused 1D channel *calibrates* without the
   noise-model companion remains unmeasured in production (pp_coverage mass-channel
   harness is TO-BUILD) and remains the likeliest way the correction disappoints.
5. **M-1's venue-dependent sign (interpretive).** +1.2 (iiib) vs −3.3 (joint_r1) nats/h —
   both inside the prior bracket; the split lives in the catalogue-bearing subset where
   the venues genuinely differ. Not evidence of a defect; sized for the record.

## 10. The decisions

| # | Decision | What it authorizes / forecloses | Tag |
|---|---|---|---|
| 1 | **[P3]/row-2 materiality ruling on M-4:** is a median +0.02–0.03 (max +0.20) catalogue-share over-weighting, confined to ~10% of events, *material*? Ruling "not material" keeps the catalogue-leg fork deferred to the Gray-convention paper task (row #110) as ratified; ruling "material" returns [P3] to a fresh physics-change proposal now | binds the row #117 item 2 condition | **[RULE]** |
| 2 | **Campaign-re-run scope with M-3 in hand:** zero MAP/width motion in the 2D channel of record ⇒ a full campaign re-run under the fused estimator would reproduce the campaign posteriors within their quoted widths. Options: (a) no re-run — record this counterfactual as the bridge between pre- and post-fusion results; (b) targeted re-run of a named subset; (c) full re-run | commits or saves the campaign CPU budget | **[RULE]** |
| 3 | Bank this measurement: ledger row, claim status for the fusion-magnitude numbers | record-keeping | [DO] |

## 11. Provenance

Arrays 6343495/6343496/6343678/6343679 (+smoke 6343451), 164/164 COMPLETED ·
pre-registration `a6a98d2a` + budget append `ac24b632` (before sbatch) · instrument
(gate) commit `2b10b8b8` · readout `readout.{py,json}` this directory · branch presented,
not adjudicated; no bands existed to re-centre (A3 measurement).
