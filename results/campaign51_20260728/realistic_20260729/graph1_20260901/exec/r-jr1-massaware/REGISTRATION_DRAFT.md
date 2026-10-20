# r-jr1-massaware — REGISTRATION DRAFT: the joint_r1 mass-aware readout (m-joint-r1-mass-aware)

Date: 2026-09-02. Node: r-jr1-massaware (Research Graph 1, Branch C, wave 1) — **DRAFT**.
Status: **EVERYTHING BAND-SHAPED HERE IS PROPOSED, NOT FROZEN.** Per the ratified
decisions-table row 5 (row #290), the registered band itself is NOT covered by the wave-1
grant: the band, the grid scope, and every disposition rule below return to the author as
the fresh [RULE] **d-jr1-band** before m-joint-r1-mass-aware may launch. This draft
proposes; it never freezes. max_revisions 2 (ORCHESTRATOR-DERIVED, charter-ratified with
row 0 — same derivation sentence as r-b82-s4, graph proposal §1.1).

Derivation input: `../dv-jr1-transform/DERIVATION.md` (this wave, same author-node) — all
PROPOSED numbers below trace to its §5/§7 or to the cited ledger rows.

## 1. Object and statistic

The full 1588-event **joint_r1** 1D posterior under the post-flip production default
(`catalogue_leg_1d_mass_aware = "auto"`, which resolves "on" on the joint_r1 φ-stack;
commit lineage row #286), on the elected h-grid (§3), paired against the **post-flip
comparand of record from m-head-rebaseline** (Branch B; C0-prime-then-blind-HEAD pattern,
rows #279/#281/#283). Reported: map_h, mean_h, floor-node mass, per-class impostor scores
(secant 0.725/0.735), and the true-host transform read (§4). Scorer: the frozen T0
gradient-weighted convention (the A18 arm (c) convention, row #286;
`WAVE3_A14_DELTA_READ_20260831.md` correction note).

Discriminates: c-auto-default-venue-general (graph proposal §1.0). Promotion of that claim
is NOT part of this registration — it is decided at d-calibration.

## 2. PROPOSED registered band and criterion (MAP-AND-mean, analogous to A18 §6.3)

- **PROPOSED band: map_h ∈ [0.64, 0.70] AND mean_h ∈ [0.64, 0.70].**
  Derivation: scan hull MAP [0.64, 0.68] / mean [0.645, 0.691] over ρ_jr1 ∈ [0.26, 0.50] ×
  {linear, quadratic} response extension on the banked off grid, rounded outward to grid
  nodes (DERIVATION.md §5). Central prediction: MAP 0.655-0.670, mean 0.657-0.674
  (ρ_jr1 = 0.365 = iiib's measured 0.2604 [row #282] × derived R_K 1.40).
- **Z-CONFIRMED** iff map_h AND mean_h are both in-band (the A18 map-AND-mean rule form,
  `PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md` §6.3 / rows #282/#286).
- **REFUTED** iff map_h ≤ 0.605 (floor-node read; the row #213 rail statistic transferred
  to joint_r1's identical 0.60-floor grid) WITH the C-C pin intact (§5) — i.e. the flip
  fails to lift joint_r1 off the floor while the pure arm is untouched. Off comparand for
  scale: banked off MAP 0.600, floor mass 0.2208; predicted on floor mass ≤ 5e-3 at every
  scanned ρ, so this refuter is decisive at N = 1 and cannot be confounded by grid
  resolution (the predicted departure is ≥ 5 nodes).
- **Neither band (INTERMEDIATE)**: 0.605 < map_h < 0.64, or map_h > 0.70, or the
  MAP/mean pair splits across the band edge → the read is booked INTERMEDIATE and
  **returns to the author as a fresh [RULE]** (charter disposition, graph proposal §1.3) —
  no verdict is self-assigned, no re-run is launched without that ruling. An INTERMEDIATE
  consumes no revision by itself; a re-registration after the ruling consumes one
  (max_revisions 2).
- Rail-flagged reads (any MAP at a grid rail) demote the read to a bound (graph §1.3).
- NOT predicted: truth. The residual mean_h − 0.73 is the dark-class completion-leg
  object (B8 [CAL], q-completion-residual) and no in-band result claims it closed — the
  same non-claim A18 carried (row #286).

## 3. Grid-scope election (couples to Branch I, b-hprior-fix)

- **Elected: H_GRID_41** (0.600-0.860; 0.01 coarse, 0.005 in 0.65-0.75 — the banked
  joint_r1 grid, verified 41 nodes locally), **extended to the 55-node G-EXT grid
  (∪ {0.870...1.000}) CONDITIONAL on b-hprior-fix's byte-identity gate going green**
  (0 mismatches below the old bound; graph proposal §1.9/row 11). Rationale for wanting
  G-EXT: the A18 matched-class sub-read railed at the 0.86 ceiling in 4/4 mirror seeds
  (G-EXT amendment, `PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md`), and the 14 extension
  tasks failed on the h-prior upper bound in the A18 arm (row #286) — the exact defect
  b-hprior-fix repairs.
- If b-hprior-fix is red or not yet green at launch: run H_GRID_41 alone; the
  matched-class sub-read is then reported as a censored bound, disclosed, and does NOT
  block the primary MAP/mean verdict (posterior tail above 0.85 was 5e-13 on the A18 arm,
  row #286 — the primary read does not need the wing).
- The primary band of §2 reads identically on either grid (all edges ≤ 0.70).

## 4. Secondary registered reads (PROPOSED, verdict-free diagnostics with bands)

Run with the T2.2b-schema candidate dump enabled at the 3 secant nodes h = 0.725/0.730/0.735
(the `T2_2B_ARM_B_RUNSHEET.md` §6.2 column schema):

1. **True-host transform read** (the venue's T2.2b-equivalent object): median over
   recovered true in-catalogue hosts of s_4d_zg_mg/s_bar_phi_zg. PROPOSED band:
   **[1.021, 1.036]** (the 95% MC predictive band of the derived realized-median;
   central 1.031; DERIVATION.md §2). h-stability required at the T2.2b order (spread
   ≤ 5e-3 across the 3 nodes). A read outside the band does not flip the §2 verdict; it
   is routed to d-jr1-band's successor ruling as mechanism evidence.
2. **Dark-class effective ρ**: on-vs-off class-mean s_imp ratio. PROPOSED band
   **[0.26, 0.50]**, central 0.365 (= 0.2604 × R_K 1.40; the C2 proportional-transfer
   assumption is what this read tests).
3. **F-3-analog refuter**: q_i > 1 on > 10% of active dark events REFUTES the remedy
   mechanism on this venue (the A18 F-3 bar, row #282; iiib measured 0.77%).
4. **Median-q**: REPORTED-ONLY, no band — the iiib registration's median-q band was
   REFUTED-IN-DETAIL as a mechanism narrative (row #282); we do not re-register a known
   mean-vs-median confusion. Expected tail-carried concentration; report deciles.

## 5. Pins and gates

- **C-C pin (hard)**: B_num, D̃_φ, and all with-BH/companion columns exact-zero vs the
  comparand (the row #282/#286 invariance; re-verified on local T2.2b data at max_rel 0.0,
  DERIVATION.md §1.4). Any nonzero → INSTRUMENT-DEFECT, nothing downstream is read.
  Same-machine comparison required (the GATE BI cross-machine ulp lesson, row #282);
  cross-machine deltas ≤ 1e-13 relative are booked as an amendment note, not a pass.
- Dark-only pure-arm sum: off value −59.87 (DERIVED-HERE on the banked grid) must be
  unchanged on the on arm to float noise.
- Panel (graph §1.3 row): **g-znorm** (flipped-leg Z = 1 identity — the §2.4 property;
  numerically: the implied global divisor uniform across events to ≤ 1e-12, the
  DERIVATION §1.4 check run on the fresh dump), **g-score-null**, **g-censoring** (floor/
  ceiling flags on every read; the off floor mass 0.2208 makes the paired Δmean_h a LOWER
  bound on the un-truncated effect — amendment-20 disclosure carried over), **g-precision**
  (full-precision columns for any score arithmetic; the +123.11 storage-artifact lesson,
  row #282 — no 7-s.f. CSV reconstruction may feed a verdict).
- Class split of record: joint_r1's OWN host map (73 in-catalogue events, row #270 §1.5),
  not the iiib 76-event map used in the derivation (disclosed C1). The dump provides it.
- Existence contract on every remote read: three-valued (present/absent/unreachable;
  row #288 lesson).

## 6. Disposition table (all verdicts return to the author; nothing self-ratifies)

| outcome | booking | next |
|---|---|---|
| Z-CONFIRMED (both in [0.64, 0.70]) | c-auto-default-venue-general SUPPORTED on joint_r1 | feeds d-calibration; promotion is the author's there |
| REFUTED (map ≤ 0.605, pins intact) | flip NOT venue-general; auto default on joint_r1 impeached | fresh [RULE]: scope of the default (venue-scoped flag?) — author |
| INTERMEDIATE (neither) | bounded read, banked verdict-free | fresh [RULE] to the author; possible revision (≤ 2) |
| any pin/gate red | INSTRUMENT-DEFECT, no science read | repair node; revision counter NOT consumed by the repair itself |

## 7. Cost and launch preconditions

- Cost: ~90-100 CPU-h for an A18-equivalent grid; less on H_GRID_41 alone (state
  candidate 10; A18 41-node registered 69.7 CPU-h, row #282). Cluster launch behind
  /cluster preflight VERDICT: READY and the Lustre OST 5 clearance (charter §0).
- Launch blockers, in order: (1) d-jr1-band ratified (band + grid scope + dispositions);
  (2) m-head-rebaseline comparand banked (its own banking RULE is inside d-calibration —
  if not yet banked, the author may elect the banked pre-flip off grid as interim
  comparand at d-jr1-band, disclosed); (3) grid election resolved per §3.
- Revision budget: max_revisions 2 (ORCHESTRATOR-DERIVED, charter-ratified).

## 8. Open questions put to d-jr1-band (the fresh [RULE])

1. Ratify or amend the PROPOSED band [0.64, 0.70] (MAP-AND-mean). The central prediction
   depends on the C2 proportional-transfer assumption; the author may prefer the narrower
   central bracket [0.65, 0.68]/[0.65, 0.69] at the price of booking more outcomes
   INTERMEDIATE, or the scan hull as proposed.
2. Ratify the conditional grid election (§3), including the interim-comparand question in
   §7 item (2).
3. Ratify the secondary-read bands (§4 items 1-2) as verdict-free diagnostics.
4. Rule whether an out-of-band-LOW transform read (< 1.021) alone should escalate to a
   registered mechanism arm (candidate-set composition, caveat C4) or remain evidence.
