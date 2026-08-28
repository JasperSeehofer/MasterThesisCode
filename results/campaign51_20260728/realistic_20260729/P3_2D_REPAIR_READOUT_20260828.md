# [P3-2D] class-G venue repair — RUN READOUT (job 6723958)

`[FABLE-ORCH 2026-08-28]` · registered design: `PREREGISTRATION_P3_2D_REPAIR_20260827.md`
REGISTERED DESIGN v2 (frozen commit `3694233d`, submitted at `d04d9dc9`, PA-2DR-13).

**Class:** post-data readout of a pre-registered stage-2 measurement.
**Status:** CHAIR verdict on chair-and-verifier evidence — **NOT author-ratified.** Per the
approval-scope rule, every band call below returns to the author as a fresh **[RULE]**.

---

## 0. Independent verification of the registered instrument (pre-readout)

Before any number was read, the 2026-08-27 Opus session's work was adversarially verified by a
6-agent workflow (all sonnet, independent re-derivation mandate). Verdicts:

| target | verdict | decisive findings |
|---|---|---|
| prereg v2.2–v2.4 arithmetic | DEFECTS-FOUND | **1 MATERIAL — see §0.1** |
| gates G1–G3 at HEAD | CLEAN | 3+2 tests pass; non-vacuity of G3 confirmed (pre-repair ref fails at D=0.1103) |
| commit `3694233d` code audit | CLEAN | exactly "reject M≤0 + delete floor" on the class-G 2D path; no other host_mode/1D/RNG side effects; production `--evaluate` untouched (`main..HEAD -- darksiren_emri/` = `correspondence_1d.py` only); gate-ledger row present |
| submission record PA-2DR-12/13 | CLEAN | five-file manifest exact; `4af1baec→d04d9dc9` = sbatch OUT_ROOT override only (+6/−1); tree `7bfff25d` identical to HEAD; fast suite reproduces **exactly** 1831 passed / 15 skipped |
| residual-accounting doc (`a662e684`) | 1 MINOR | ladder arithmetic all reproduced; §1a "verbatim" quote has an inserted clarifying clause `RHS2/LHS2 =` not in the source (numerically identical) — the "copied verbatim" claim is textually false |
| HEAD-readout registration §2.2/§5.1 | 1 MINOR | all 32 table cells reproduce to ≤5e-5; `R_LOW_THRESHOLD` is at `correspondence_1d.py:360`, not :359 (content quoted correctly) |

### 0.1 MATERIAL finding — the v2.2 relative-SEM chain (candidate amendment PA-2DR-14)

v2.2's rung chain `2.3370% → 2.5606% → 2.6681% → 3.3719%` was back-derived from the source
table's ALREADY-ROUNDED `X ± σ_X` figures instead of read directly as σ/value from
`p32d_residual_accounting_20260827.md` §1's raw ladder. The true chain is
`2.3194% → 2.5449% → 2.6560% → 3.3719%`. Corrected predictions' σ:
σ_pred(P1) = 0.00020474 (vs 0.00020476, negligible), σ_pred(P2) = **0.00014599** (vs 0.00014679,
0.55% low), σ_pred(P4) = 0.00018232 (vs 0.00018166). The same document independently carries the
TRUE value (2.32%) in its own G4 section — an internal inconsistency. **Every readout below was
evaluated under BOTH chains; no disposition differs.** The verifier also confirmed no band edge,
σ-distance, or verdict boundary flips. Registered here so the correction cannot be silently
dropped; ratification of the corrected chain is author item **R-2DR-1** (§4).

---

## 1. Run and instrument provenance

- SLURM array **6723958**, tasks 0–23, partition `cpu_il`, **24/24 COMPLETED** (~32.5 min/task),
  2026-08-27. Out-root `$WS/p3_2d_fleet_repair_20260827` (fresh; 48 arm-seed pairs, bc+bt).
- Per-task provenance stamps present (`provenance_*.json`): commit
  `d04d9dc9bfe39e6c5a72e768a26f2dcc38355bf5`, branch `fix/p32d-classg-venue-repair` — the first
  fleet in this campaign to self-stamp (commit `67b18592`).
- Readout executed **by the orchestrator, not a subagent** (standing rule: subagents never run
  the registered measurement), on the cluster login node from `~/darksiren-emri` at the frozen
  checkout: `p3_2d_fleet.py --stage lhs2d --arm {bt,bc} --out-root $WS/p3_2d_fleet_repair_20260827`.

## 2. Gates (§v2.6) — ALL PASS; no INSTRUMENT-DEFECT

| gate | result |
|---|---|
| G1 byte-identity (dev) | PASS at HEAD (verifier-run) |
| G2 RNG draw-count (dev) | PASS at HEAD (verifier-run) |
| G3 truncated-normal KS (dev) | PASS at HEAD; non-vacuity confirmed against pre-repair reference |
| G4 arm-coherence | **PASS** — fresh ratio P4/P1 = 0.005581208/0.006448604 = **0.865491**, inside the registered interval [0.8613, 0.8675] with no tolerance needed |
| G5 PA-2DR-7 mass-window count | **PASS** — `pa2dr7_fraction = 0.0` (0 accepted latents in (0,1) M_sun across all seeds, both arms) → the ×1.1944 factor transfers exactly |
| G6 exact dead-row identity (P5) | **PASS** — `dead_row_identity_all_ok = true`, both arms, all 24 seeds, to float round-off |

## 3. Registered reads (raw instrument output, verbatim)

| read | realized (mean ± SEM, 24 seeds) | registered prediction | deviation | disposition |
|---|---|---|---|---|
| **P1** `LHS2_D1+D2` (bt) | **0.00644860 ± 0.00013657** | 0.00638792, band ±0.00073837 | **+0.247 σ_comb** | **INSIDE** |
| **P2** `LHS2_D1only` (bt) | 0.00600203 ± 0.00017134 | 0.00598120, planning band ±0.00062278 | +0.100 σ_comb | central value inside, but realized SEM **16.7% above planning** → **UNDERPOWERED** (freeze rule, §v2.3) |
| **P3** paired ratio | 1.08118 ± 0.014080 | 1.0680, planning band [1.02753, 1.10847] | +0.977 σ_plan | inside planning band; realized paired SEM **4.4% above planning** (0.014080 > 0.013492) → **UNDERPOWERED-ON-STEP-3** (freeze rule, §v2.4) |
| **P4** `LHS2_D1+D2` (bc) | **0.00558121 ± 0.00013325** | 0.00550223, band ±0.00067587 | **+0.351 σ_comb** | **INSIDE** |
| **P5** exact identity | exact on 24/24 seeds, both arms | equality to round-off | — | **PASS** (= G6) |

All dispositions identical under the corrected σ chain of §0.1 (P2's excess is then 17.4%; P1/P4
distances change in the 3rd decimal of σ only).

**Substantive companion reads (REPORTED-ONLY, not band-bearing):**

- P3 excludes `R = 1` ("Defect 2 entirely spurious") at **5.77 σ_realized** — the dead-row
  mechanism is real on the fresh fleet.
- The **pre-registered non-discrimination stands exactly as registered**: 1.08118 sits between
  the two candidate predictions (+0.98 σ from 1.0680, −1.54 σ from 1.1019). This run does not
  adjudicate which sample the 1.0680 was measured on, and no verdict claims it does.
- bc arm mirrors bt: paired ratio 1.09604 ± 0.017040; both arms' rung structure is common-mode,
  as G4's in-interval pass independently confirms.

## 4. CHAIR VERDICT (returns to author; nothing banks)

**Per the registered verdict map (§v2.5), CONFIRMED cannot bank**: it requires P1 ∧ P2 ∧ P3
inside their bands, and P2/P3 carry UNDERPOWERED dispositions, which the map defines as "not a
pass and not a refutation". **Not REFUTED** (P1 is inside at +0.25 σ). The honest summary:

> Both level reads (P1 bt, P4 bc) land inside their registered 3σ bands within 0.36 σ; every
> gate passes; the exact identity holds on all 48 arm-seed pairs; the Defect-2 discriminator's
> central value lands 0.98 σ from prediction and excludes the null at 5.8 σ — but P2 and P3
> bank UNDERPOWERED because their realized SEMs exceeded the frozen planning values by 16.7%
> and 4.4%. The two-rung reweighting model is *supported* by every central value and refuted by
> nothing, yet the registered map's letter forbids the CONFIRMED label.

**Author decisions arising ([RULE]/[DO], all fresh — none covered by prior approvals):**

- **[RULE] R-2DR-1** — ratify the §0.1 corrected relative-SEM chain (candidate amendment
  PA-2DR-14 to the prereg; no disposition changes under it).
- **[RULE] R-2DR-2** — rule on the disposition of the run: bank as
  "SUPPORTED-BUT-UNDERPOWERED (P2/P3)" per the map's letter, or rule that the freeze-rule
  breach at 4.4%/16.7% with central values at 0.10 σ/0.98 σ still refutes nothing and the
  registered UNDERPOWERED label is the verdict of record. (The chair recommends the map's
  letter: UNDERPOWERED is the verdict of record; the central values are reported companions.)
- **[DO] D-2DR-1** — (optional, ~2.2 CPU-h/seed × N) authorize a seed-extension arm
  (900125…900140, same instrument, same out-root) to recover the SEM deficit: +9 seeds
  (24→33) shrinks SEM by ~17% at fixed scatter, converting P2/P3's excess into headroom.
  Zero new design decisions; the prereg's bands stay frozen.
- **[RULE] R-2DR-3** — the PARKED verdict (row #211) remains PARKED: this run tested the free
  residual-ladder prediction only. Confirm no re-opening.

## 5. Blindness and caps (unchanged, restated)

The §v2.7 missing-anchor cap stands: whatever the author rules, the epistemic status is capped
at `supported`, never `verified`. §v2.8's eight blindness items stand; nothing here attributes
the production 2D offset (see the HEAD readout, a separate measurement).

---

*Every number above is either the verbatim JSON output of the registered instrument
(`stage_lhs2d`, run 2026-08-28 by the orchestrator at the frozen checkout) or an arithmetic
evaluation of the registered bands reproduced in this session. Raw JSON archived in the out-root
and reproduced in this document's tables.*

---

## 6. AUTHOR RATIFICATION [appended 2026-08-28, post-presentation]

Author reply (verbatim, to the Runbook 36 Docket artifact presenting §4's items with chair
recommendations inline): **"all ratified also the thirteen earlier ones"**. Itemization below is
orchestrator-derived from that blanket, per the attribution-precise recording convention:

- **R-2DR-1 RATIFIED** — the corrected σ-chain (PA-2DR-14) is the record. No disposition changes.
- **R-2DR-2 RATIFIED** in the chair-recommended form (option (a)): **UNDERPOWERED is the verdict
  of record**; the central values (P1 +0.247σ, P2 +0.100σ, P3 +0.977σ, P4 +0.351σ) are reported
  companions. Not REFUTED; CONFIRMED did not bank.
- **D-2DR-1 APPROVED** — the seed-extension arm. Registered as PA-2DR-15 (prereg appendix)
  before submission.
- **R-2DR-3 RATIFIED** — row #211 stays PARKED; the exoneration record stays closed.

---

## 7. 33-SEED RE-READOUT [2026-08-28, PA-2DR-15 extension] — CHAIR VERDICT: **CONFIRMED**

Extension job **6730213** (seeds 900125–900133, 9/9 COMPLETED, ~32.5 min/task, same frozen
instrument `d04d9dc9`, same out-root). Registered instrument re-run by the orchestrator over
all 33 seeds. Side-by-side per PA-2DR-15:

| read | 24-seed | 33-seed | dev (33) | SEM vs planning | disposition (33) |
|---|---|---|---|---|---|
| P1 (bt) | 0.00644860 ± 0.00013657 | **0.00644266 ± 0.00012212** | +0.230 σ | below ✓ | **INSIDE** |
| P2 (bt D1only) | 0.00600203 ± 0.00017134 | **0.00601580 ± 0.00013437** | +0.174 σ | **below ✓** (was +16.7%) | **INSIDE** |
| P3 (ratio) | 1.08118 ± 0.014080 | **1.07529 ± 0.011037** | +0.661 σ | **below ✓** (was +4.4%) | **INSIDE** (band tightens to ±0.033110 per the freeze rule) |
| P4 (bc) | 0.00558121 ± 0.00013325 | **0.00558246 ± 0.00012014** | +0.368 σ | below ✓ | **INSIDE** |

Gates at 33 seeds: G4 = 0.866484 ∈ [0.8613, 0.8675] ✓ · G5 `pa2dr7_fraction` = 0.0 ✓ ·
G6/P5 exact on **66/66** arm-seed pairs ✓. All dispositions identical under the ratified
corrected σ-chain (PA-2DR-14).

**Per the registered verdict map §v2.5, all four CONFIRMED legs are now satisfied**
(P1 ∧ P2 ∧ P3 inside, P5 exact, every gate passing): **CONFIRMED — "the two-rung reweighting
model's predictions landed inside their own bands." Capped at epistemic status `supported`,
never `verified` (§v2.7), and it licenses NO claim about rung 1 (untested).**

Companions (REPORTED-ONLY): R = 1 ("Defect 2 spurious") excluded at **6.82 σ**; the
pre-registered 1.0680-vs-1.1019 non-discrimination **narrows to 2.41 σ but formally stands**
(< 3 σ — no verdict claims to have resolved which sample the 1.0680 was measured on).
Sequential-analysis disclosure per PA-2DR-15: this is the single pre-committed extension,
decided post-data; both readouts are reported above; no further extension may run.

**Returns to the author as a fresh [RULE]: ratify CONFIRMED-at-33-seeds as the verdict of
record (superseding the 24-seed UNDERPOWERED disposition of §6, which stays on the record).**
