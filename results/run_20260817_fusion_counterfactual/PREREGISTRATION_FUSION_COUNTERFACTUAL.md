# Mini-prereg — [P5-3] production selection-fusion counterfactual (item 4, rows #117–#118)

Registered 2026-08-17, **BEFORE the run**. Authorization: ledger row #117 item 4 ([DO],
ratified) with the row #118 MAJOR-1 amendment (1D/2D decomposition mandatory). Template
lineage: `results/run_20260804_postfix/gate_vii/PREREGISTRATION_N2_SEL1D.md`. Discipline:
A8-v2 (per-branch referents, execution completeness, no toy-calibrated numbers);
research-cycle rule 6 (measurement, never a formula ruling). **This file authorises a
MEASUREMENT of the production-side magnitude of the landed fusion (`2b10b8b8`); it neither
re-opens the gate nor authorises a campaign re-run** (row #117: campaign scope returns to
the author WITH this result).

## What is being measured

The `[PHYSICS]` selection fusion (commit `2b10b8b8`, gate rows in
`docs/gates/PHYSICS-GATE-LEDGER.md`) made `fused` the `absolute_marginal` default:
S̄_φ(z;h) in the 1D completion numerator ([P2]) and `g_sel,prod` (S_4D inside the mass
quadrature) in the 2D leg ([P1]). The venue arm (rows #115–#116) proved the CHANNEL and the
FORM; **A3 forbids transferring its magnitude**. This run measures the production-side
consequence on the campaign venue, old form vs fused form, same seeds/realizations/inputs.

**Regime expectation of record (row #118 MAJOR-1, stated before the run so a small 2D read
is the PREDICTED outcome, not a disappointment):** production's completion leg sits in the
sharp-likelihood limit (measured d_L-conditional σ_cond p5/p50/p95 = 2.47e-8 / 8.80e-8 /
2.99e-7 on the production CRB reference). The pair's action is expected 1D-dominated
([P2]); [P1] is correct-form and possibly near-inert (prior bracket: N-2 §3.1's unmeasured
M2 band |Σ| ≤ 20 nats/h — context, not a band). **No band is registered for any quantity:
this is a measurement seeded from nothing (A3).**

## Design — two cells, four channels, full decomposition

Cells per venue (both at commit `2b10b8b8` or a descendant with no `[PHYSICS]` change in
between — the pre-#118 runs of record are NOT byte-comparable across the ratified 08-12
quadrature/φ changes, so the off twin is re-run fresh):

| cell | flag | role |
|---|---|---|
| `off` | `--selection_in_completion_numerator off` (EXPLICIT — `auto` now resolves to `fused`) | pre-#118 estimator, fresh twin |
| `fused` | `--selection_in_completion_numerator fused` (or `auto`; record which) | production form |

**The single-leg cells are recovered channel-wise, not re-run** (MAJOR-1 decomposition at
zero extra cost). By construction — proven bit-exact at the completion-numerator level by
`test_selection_fusion.py::test_fused_pairing_identity` and by the estimator's structure
(the 1D mixture never reads `B_num_wbh`; the 2D mixture never reads `B_num`):

- the fused run's **1D channel** ≡ a `1d`-cell run's 1D channel → Δ(1D) = the [P2] effect;
- the fused run's **2D channel** ≡ a `2d`-cell run's 2D channel → Δ(2D) = the [P1] effect.

If either identity is doubted at readout, a single spot-check task (one h, one venue, cells
`1d` and `2d`) may be run to confirm — pre-authorised here, bounded to ≤ 2 tasks.

## The run

Mirrors the N-2 counterfactual pair exactly, at the new commit:

| | value |
|---|---|
| RUN_DIRs | `$WS/run_20260817_fusioncf_{off,fused}_iiib/`, `$WS/run_20260817_fusioncf_{off,fused}_joint_r1/` (4 arrays) |
| CRB input | the `prepared_cramer_rao_bounds.csv` symlink target `run_20260729_seed61000/` — **no re-simulation** |
| Injection pool | `injection_pool_mix200k_20260728` symlink, unchanged |
| Catalogues | unchanged per venue: iiib = idealized (parent/exact-z); joint_r1 = realization r1 (delivered/observed). **No new realization** |
| Estimator | `NORMALIZATION_MODE=absolute_marginal`, `HOST_Z_KERNEL=volume_deconv`, `HOST_MASS_KERNEL=auto` — post-fix path-(A) pairing |
| h grid | canonical 41-point grid (0.01 on [0.60,0.65]∪[0.79,0.86], 0.005 on [0.655,0.79]) |
| Code commit | `2b10b8b8` or descendant (record actual; `run_metadata.json` `git_commit` checked before any read) |
| Workspace note | **workspace expires 2026-09-23, 0 extensions** — outputs retrieved and committed promptly |

**Budget (row #116 item 2 discipline — pessimistic premeasure rate, filled at submission):**
before submitting, read the realized per-array cost of the N-2 pair (jobs 6152554/6152556)
from `sacct`; ceiling = 4 arrays × that per-array cost × 1.15 (fused-cell S-queries; G1
keeps the n=8 fast path for smooth-S rows) × 1.3 (pessimism factor). The filled number is
appended below AT SUBMISSION, before `sbatch`. Overrun beyond the filled ceiling pauses the
run and returns to the author.

## Measured quantities (readout contract — all reported, none banded)

- **M-1 ([P1] magnitude):** 2D-channel Σ Δln tilt, fused vs off — full-grid chord
  [0.60, 0.86] AND central difference at h=0.73, both venues; plus the 2D MAP pair.
- **M-2 ([P2] magnitude):** 1D-channel Σ Δln tilt, same two statistics, both venues; 1D MAP
  pair. *Context (not a band): the N-2 run of record measured +24.6/+22.7 chord,
  +30.9/+32.3 central at commit `0167df53`; drift vs that value is reported and attributed
  (code moved: φ affine swap, Route-1 adaptive, fusion plumbing).*
- **M-3 (pair):** joint-posterior (2D channel of record) MAP and width, fused vs off — the
  number the campaign-re-run decision needs.
- **M-4 ([P3] forcing function, G3 corrected direction):** the mixture skew — per-event
  catalogue-vs-completion share (1D: `A_cat` vs `B_num`; weights `w̃_G`) fused vs off, and
  its h-dependence. Direction of record (MAJOR-3): the S̄-free catalogue leg is
  OVER-weighted wherever S̄_φ < 1. **No materiality threshold is pre-committed: the measured
  skew returns to the author as the fresh [RULE] input for decision-table row 2's
  "unless material" condition (binding default).**
- **NULL-1:** `run_metadata.json` carries the intended cell in every task; `git_commit`
  matches; `freeze_g_frac_ref_h: null`.
- **NULL-2 (off twin sanity):** the off twin's diagnostics reproduce the run-of-record
  columns up to the two ratified 08-12 divergence classes (rel ~1e-8 2D tolerance of
  `87c6670b`; Route-1 1.26e-14 smoke of `dfedf19c`) — any larger unexplained drift VOIDS
  the comparison and is investigated before any M-read.

## Carried caveats (stated before the run)

1. **#66/#67 (MINOR-4, the single most likely disappointment):** in the G4b harness the
   selection-inside factor only *calibrated* when paired with the σ(d_L^obs)-vs-σ(d_L^true)
   noise-model companion. This run measures MAGNITUDE, not calibration; DS-G3's restoration
   is in-venue evidence only. A calibration read on production requires the pp_coverage
   mass-channel harness (TO-BUILD) and is NOT claimed here.
2. **Pool-ψ prior-weighted survival (MINOR-5):** the fused numerator extends the existing
   prior-marginal S approximation into the numerator; part of the pool-vs-model residual
   class (open residual of record, r = 0.847 correlation thread).
3. **No production posterior is superseded by this run's cells**: `off` is a counterfactual
   of the NEW default; `fused` twins the production path but is quoted only as a contrast
   until the author rules on campaign scope with M-3 in hand.

## Scope guard

No re-simulation; no campaign re-run; no formula ruling; the [P3]/row #110 fork is not
settled here (M-4 only supplies its input); A3 governs — venue magnitudes were never
predictions for these cells.

**Append-only.** Verdict appended below by the readout session; no edits above this line
after the registering commit.
