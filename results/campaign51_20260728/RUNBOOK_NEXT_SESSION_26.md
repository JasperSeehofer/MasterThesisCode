# Runbook — next session (written 2026-08-21 ~07:45, supersedes RUNBOOK_NEXT_SESSION_25)

> **⚠ SUPERSEDING ADDENDUM (2026-08-21 ~09:30, ledger row #152):** an author-requested adversarial
> review CONFIRMED two FATAL defects in the fleet readout's presentation (δ-arm bias reference;
> the "full channel reproduces −0.108" rail coincidence) plus decisive MAJORs (realized scatter
> 1.56× pilot σ̂ → defect-edge margin only 1.07σ; GATE S void-candidate; BAND R actually paired).
> **Read `ADVERSARIAL_REVIEW_CSG_20260821.md` + the prereg's CORRECTION & REVIEW ADDENDUM before
> §1 below.** The decision queue is RESTATED by row #152: BAND C ratification becomes "bank the
> 6.05σ non-zero score (B_num/β_Ḡ h-derivative mismatch ~10%); INTERNAL-DEFECT label PROVISIONAL
> pending pre-check O4 (common-domain/quadrature pairing test — supersedes the S̄_φ audit)";
> GATE S ruling becomes "void, not attenuated"; plus the A17-extension and two new proposed
> amendments in retrospective entry 3. Everything else below stands.

**Read first:** `results/prod2d_closure_20260818/CAMPAIGN_READOUT_REPORT_CSG_20260821.md` — the
whole overnight campaign in one comprehension-first artifact — then ledger rows **#149 → #151**.
The prereg (`PREREGISTRATION_SELFGEN_CONTROL.md`) now carries, below its freeze line, in order:
O2 band registration → O2 gate amendment → O2 verdict → O3 registration → O3 verdict → v3 design
change → pilot band formulas → PILOT GATE V AMENDMENT → **FLEET VERDICT**. Do not redo any of it.

## 0. What the overnight session established (2026-08-21, autonomous under author grant)

1. **B-SEL's −0.1083 is now decomposed, and the decomposition transfers.** Three channels, all on
   banked data first (rows #149/#150), then reproduced by a clean generator (row #151):
   **matched-channel violation −0.085 ⊕ dark-fraction tilt +0.055 ⊕ impostor drag −0.079 = −0.108.**
2. **C-SG v3 ran 46/46 and fired BAND C = INTERNAL-DEFECT on both frozen statistics**
   (S̄₁₅ = −0.1173 vs edge −0.0966; bias₁₅ = −0.0665 vs edge −0.0423). The violation is
   h_gen-independent (−0.113…−0.133 at h_gen 0.68/0.73/0.78, both σ modes) and the full channel
   lands on −0.104…−0.110 in every arm. B-SEL's generator caveats are eliminated as owners.
3. **GATE S fired CONTROL-INERT by its letter** (ŝ = 0.368 ± 0.186) against strictly ordered arm
   means — an attenuated, not absent, h-response. Unresolved; queued as a [RULE].
4. **Governance:** A16 adopted (retrospective ledger + amendment impact tracker, both live and
   already fed); A17 + a slope-gate amendment PROPOSED; Stage-L found the venue sits below
   Gray 2020's own validated completeness floor and that the per-sector decomposition is
   literature-novel (paper-relevant).

## 1. OPEN AUTHOR DECISIONS — the morning queue (all [RULE], none pre-empted)

From the readout report §10, verbatim order:

1. Ratify **BAND C = INTERNAL-DEFECT** → row #140 promoted to a banked estimator-defect claim
   (conditional on the six named invariants).
2. Rule on **GATE S**'s INERT-vs-attenuated reading (score primary unaffected either way);
   register the sub-unit slope (ŝ ≈ 0.37) as its own follow-up or fold into the fix fork.
3. Re-grade **rows #137/#140/#144** per O2/O3 (the "pure completion carries it" language; the
   ≥0.073 bound whose premise is refuted by measurement).
4. **Open the fix fork** (carried since row #128): designated first step under either branch =
   independent audit of `S̄_φ` (`bayesian_statistics.py:1932-1975`) — never audited, and it is
   the exact normalization the matched channel divides by.
5. **A17** (gate-porting re-derivation) and the slope-gate three-outcome amendment — adopt/reject.
6. **Landscape/T1 un-gate** (carried): the chain C-SG → B-SEL verdict → fix fork → landscape is
   at its second link.

Also carried, unchanged: systematics-budget row 16 re-grade; workspace `emri` expires
**2026-09-23** (extend is the author's call; `ws_extend emri 60` documented).

## 2. Next technical step (free until the author rules)

Nothing is registered to run. The `S̄_φ` audit (decision 4) is the designated next front but its
pre-registration should follow the author's BAND C ruling. Zero-compute options that need no
ruling: the GATE S slope diagnosis on banked C-SG diagnostics (why ŝ ≈ 0.37 — posterior-width vs
h_gen structure, edge effects at 0.68 ruled *against*), and the b0 catalogue-sector conditional
(+0.040, exploratory in O3) formalized as its own registered read.

## 3. Where everything is

- Fleet data: `results/prod2d_closure_20260818/csg_pilot_20260821/` (46 JSONs + 46 per-event
  diagnostics CSVs + `MANIFEST.sha256`; every channel recomputable at zero compute).
- Scorers (all pre-data): `decompose_impostor_leg.py`, `decompose_matched_channel.py`,
  `csg_pilot_bands.py` (+ output = frozen bands), `csg_fleet_readout.py` (+ output).
- Module: `darksiren_emri/validation/selfgen_control.py` (+ 25 tests). Not a physics-trigger file.
- Governance: `docs/RETROSPECTIVE_LEDGER.md` (entries 1–2), `docs/AMENDMENT_IMPACT_TRACKER.md`
  (A1:2, A5:1, A10:1, A12:1, A15:2), `docs/RESEARCH_CYCLE.md` (A16 adopted).
- Literature: `docs/LITERATURE_WARNINGS.md` (ABC25/B25/VW25 sections, G20-d, field-gap note).
- Cluster: jobs 6415588 + 6420343 both COMPLETED; queue empty; ONE repo synced to main.

## 4. Standing constraints that mattered tonight (keep them)

- Prereg-first including free reads; scorers committed before data — held 4/4 times.
- Re-derive every decisive number (O2 via firewalled agent; fleet via orchestrator raw-recompute).
- A15 both ways: bands need operating characteristics AND gates get re-derived on reference data
  when they move channels (the GATE V lesson; A17 proposed).
- Subagent briefs forbid executing the registered measurement — held (the workflow's hard
  constraint line).

## 5. Resume recipe (one line)

Read the readout report → collect the author's six rulings → pre-register the `S̄_φ` audit (or
whatever the fix-fork ruling directs) → the GATE S slope diagnosis is free work in the meantime.
