# Runbook — next session (written 2026-08-20, supersedes RUNBOOK_NEXT_SESSION_22)

**Read first:** ledger rows **#127–#143** (they are the campaign; #140/#142/#143 are the
crux), then `results/prod2d_closure_20260818/PREREGISTRATION_1D_CORRESPONDENCE.md` (all
VERDICT blocks), then this file. The 2026-08-19/20 arc is fully banked — do NOT redo any of
it.

## 0. State of the science (what is TRUE as of this writing)

**Production, post-fix baselines of record** (derived-form B_scale, off basis, seed61000):
2D mean_h **0.6771 (iiib) / 0.6788 (joint_r1)** → offsets **−0.0529 / −0.0512**;
1D **0.6010 / 0.6020**, MAP railed at the 0.600 grid edge. σ_h(2D) 0.0239 / 0.0225.

**The base tilt is carried by the pure-completion (dark) class** — events with no catalogue
support (605/1588 iiib, 491/1588 joint_r1): 1D mean **0.6001**, σ_h **0.0011**, and ~195% of
the full-sample slope; identical in 2D (0.6004) ⇒ NOT mass-channel structure. Its per-event
score at truth is **−0.635 ± 0.017 (37σ)**, and it is a DEEP-z phenomenon: ≈0 below z ≈ 0.4,
−1.08 by z ≈ 0.9. Catalogue-supported events pull the OTHER way (in-catalogue class 0.828,
score +1.507) — production's posterior is a balance point.

**Tilt-ledger entries, closed:** B_scale (un-derived, +0.12-class, REMOVED as a defect —
`docs/derivations/bscale_completion_normalization.md`); s_Edd (re-measured **+0.0012/+0.0019**,
immaterial; the old −0.020 was stale by an order of magnitude AND sign); f_k-vs-f̄ (closed by
derivation, ≤2e-4); catalogue-leg mass overlap (refuted as owner, +0.001/+0.003); selection
fusion (inert in production). **J_α** measured −0.0025/−0.0061, derived correct
(`docs/derivations/jalpha_selection_mass_kernel.md`), default flip **batched** with the base
tilt by author ruling (row #135) — when it flips, s_Edd needs a cheap re-measure (E×J
interaction, memo §5).

**The open item is the base tilt, and its status is UNRESOLVED-BY-DESIGN.** Row #140 banked
an "internal misnormalization" verdict from the mirror; row #143 VOIDED it when D-1 found the
mirror mismatched at survival time (surviving-vs-model CDF gap 0.0792 > 0.05 band; drawn-vs-
model 0.0336 clean). Two bisection arms (B-SELF fused convention, B-DEN data measure) moved
nothing (−0.116, −0.119 vs −0.112) — consistent with the driver being outside the estimator.
`docs/derivations/completion_numerator_data_measure.md` is FALSIFIED as owner (its §2 defect
is real and unit-proven — production's event term integrates over the data to 1.0316 vs 1.0 —
but repairing it does not move the bias).

## 1. The next step is registered and ready: **D-2**

Per AMENDMENT A-6 (end of the correspondence prereg): rebuild the mirror arm so **survival**
matches the model — accept every drawn event (NO production quality filter) and assign each an
analytic σ_dL/d_L consistent with the estimator's assumed measurement model instead of a
resampled donor Fisher row. **Pre-flight gate (scored, this time not assumed): D-1 must return
max CDF gap ≤ 0.05 on the rebuilt arm BEFORE any seed is analysed.** Then re-run the isolation
test with the A-3 bands (ESTIMATOR-SELF-CONSISTENT vs INTERNAL-MISNORMALIZATION). Only if a
defect survives THAT does estimator bisection resume, starting with D̃^φ's α_G^φ/β_Ḡ^φ class
composition.

Cost anchor: ~0.5 CPU-h/seed at 2 cpus; 15 seeds ≈ 8 CPU-h. Fleet pattern:
`results/prod2d_closure_20260818/correspondence_fleet.sbatch` (currently 142 tasks; arms
b0/bsig005/bsig025/eden05/eden2/bout/bf1/bsel/bself/bden). Scorer with pre-committed bands:
`readout_bout.py` (add the new arm's band BEFORE its data lands — that discipline has held
all campaign).

## 2. Instruments now live (all default-off, bit-identical to production when unset)

`--catalogue_mass_overlap {production,neutralized,inflated}` · `--completion_b_scale
{derived,legacy}` (**derived is the production default now**) · `--eddington_m {on,off}` ·
`--sigma4d_mass_kernel {point,kernel}` · `--completion_event_measure {ratio,data}`. Each has
N-0 bit-identity + N-2 engagement gates and a plumbing test asserting the runtime value per
arm. Harness: `darksiren_emri/validation/correspondence_1d.py` (stages g0/g1/g2/arm/d1).

## 3. Standing constraints (unchanged, plus what was added)

Prereg-first for EVERY measurement including free reads; adversarial verifier pre-check on
every registration; scorers committed BEFORE their data; physics-change gate for instrument
code; P7-4 venue scoping; P7-8 one-realization disclosure; append-only ledgers/preregs;
tiering table + ≤3 top-tier per workflow. **New (adopted 2026-08-20, `docs/RESEARCH_CYCLE.md`):
A10** invariance & blindness declaration in every prereg · **A11** provenance stamps on budget
inputs (+ checksum pins on unversioned datasets, now in CLAUDE.md) · **A12** score-zero test as
the standing FIRST diagnostic · **A13** engagement gate + dispatch-path check · **A14**
attributions ship with their falsifier. A15 (from the D-1 retrospective) may be pending — check
`docs/RETROSPECTIVE_D1_20260820.md`.

## 4. Operational gotchas of record (cost real time this campaign)

Cluster stragglers: a few tasks per fleet land on contended nodes and run 3–4× slow — submit
with generous walltime (5 h) and expect to resubmit a tail. **Use `--cpus-per-task=2`** for
correspondence arms (single-core-bound; 16 would have blown the budget ~8×). bwUniCluster
certificates expire — a dead SSH is an auth problem, not a job problem; jobs keep running.
Subagents must be told to block in the foreground (they park waiting for notifications that
never come — 5 incidents). The GitHub push rejection is resolved (4 GB CSVs excised, hash map
in `docs/HISTORY_REWRITE_20260819.md`); CI is flaky on byte-exact float pins (issue #56,
deliberately deferred). Workspace expires **2026-09-23**, 0 extensions.

## 5. Author decisions still open

1. Systematics-budget **row 16** re-grade — its "affects rates/shape, not estimator
   calibration" clause is contradicted by measurement (row #138); proposed re-grade to a
   measured, calibration-affecting systematic.
2. The **fix fork** for the base tilt — opens only when D-2 (and any surviving bisection)
   localizes it; options will be a corrected form, a marginalization repair, or documentation
   as a systematic, presented with the derivation.
3. Whether to **un-gate the landscape/T1 round** (13 fused cells) once the tilt resolves —
   currently gated by the author's row #128 ruling.

## 6. Resume recipe (one line)

Read rows #127–#143 → build D-2 (survival-matched arm) → pass the D-1 pre-flight gate (≤0.05)
→ run 15 seeds → score with pre-committed A-3 bands → if a defect survives, resume bisection at
D̃^φ's class composition; if not, the tilt is data-vs-model and the fork goes to the author.
