# Must-fix list for revision 1 of BOTH drafts (chair-compiled from the 6 design-gate records, 2026-09-03 night)

**Blindness notice.** The statistics-lens gate reviewers computed the per-event scores while checking
SE claims and thereby unblinded both arms' primary statistics (disclosed in DESIGN_GATE_stats.md of
each arm). The revising author MUST NOT open either DESIGN_GATE_stats.md. This list carries every
must-fix from those records with all decisive numbers removed. **Band THRESHOLDS (|Z| ≤ 3, ρ ≥ 0.5 /
≤ 0.2, φ ≥ 0.5 / < 0.2, T_mat 0.008) are FROZEN as drafted before the gate and must not change.**
Only SE sourcing, power/false-fail statements, citations and wording change.

## r-completion-residual
1. (stats) SE_prod forecast input: re-source the per-event SD from the matched-channel score on the
   PRODUCTION venue dark class, not the harness full score (harness full-score SD ≈ 0.68 undershoots
   the production matched-channel SD by ~11 %). Give the SE formula and let the registered read
   compute the number; drop the numeric placeholder or label it "harness-borrowed proxy".
2. (stats) Separate the harness full-score SE (informational, 0.0063 over 67 universes — reproduces
   exactly) from the matched-channel SE_harn (the registered statistic's own SE, larger) throughout
   §2.3/§3. Restate the "detects ≥ 0.02/event" power claim with the matched-channel SE.
3. (design) g-znorm: add an explicit tolerance (exact equality is natural: den_log_term is one
   math.log per h-node) and add "g-znorm red" to the S4 NO-READ trigger list.
4. (design) Quote the parent kill criterion verbatim (RESEARCH_GRAPH_1_PROPOSAL_20260901.md:45):
   "registered arm fails to discriminate at its registered band after revision 2 -> park
   bounded-undetermined with the measured bound".
5. (provenance) The artifact a8824799 board-card quote is not a git-tracked source; re-cite to
   B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md §4.4 + ledger row #261.
6. (byte-id) The T0 anchor mean_h = 0.666987 is a 6-dp display value; a literal 1e-9 tolerance is
   unsatisfiable. Re-anchor to the full-precision value in the m-head-rebaseline JSON/CSV (name the
   file) or state the tolerance as 1e-6 on the 6-dp display. (BYTEID otherwise GREEN: 67/67 exact.)

## r-cone-loss
1. (stats) SD_IN(s_e,1D) and SD_IN(s_e,2D): re-source from the PRODUCTION in-catalogue IN class
   (the population n_OUT/n_IN are drawn from), not the harness dark class; redo the SE(Δh_cone)
   formula and the false-fail statement with that sourcing. State the materiality margin in SE
   units ONLY as a formula to be filled by the registered read; disclose that the margin may be of
   order 1 SE, i.e. the arm may be UNDER-POWERED to distinguish "owns" from "immaterial" — and add
   the honest consequence: if SE(Δh_cone) > T_mat/3 the disposition is INTERMEDIATE-UNPOWERED
   (a bound), returning as fresh RULE. Disclose the 2-outlier sensitivity and adopt a stated robust-SD
   convention (e.g. MAD-scaled) BEFORE any number is read.
2. (design) Disposition table: rewrite the CONE-OWNS-FLOOR action cell — it contributes evidence
   toward d-residual-attribution (which stays open pending d-calibration + d-photoz-leverage,
   charter line 189) and returns as a fresh RULE deferred to the morning (docket 2.3), matching the
   INTERMEDIATE row's phrasing.
3. (design) G-3: add "mismatch => INSTRUMENT-DEFECT" on its own line, like G-1/G-2/G-4.
4. (design) Kill criterion: literal verbatim quote of charter line 46, in quotation marks.
5. (provenance) Re-cite "R2c NOT-DISTINGUISHED (p=0.0358, power ~68 %)" to row #226 (row #220
   supports only the C-STRUCTURAL-ONLY clause).
6. (provenance) Re-cite "+0.587 ± 0.064 per event" to the raw checkpoints
   (b8_cal_harness_work_s4_postflip/*_S.json score_at_truth.no_bh.catalogue_hosted) or to
   INFORMATION_FORECAST.md:19, not to rd-s3-readout (which carries only Z = 9.76 for that class).
7. (build dry-run) The scorer's dry-run gate suite reported INSTRUMENT-DEFECT on one gate
   (cone_loss_work/cone_loss_gates.json) — the author must read that JSON (it holds gate
   pass/fail only, no statistic), identify the failing gate (G-1 pins, G-2 anchors passed; check
   G-3/G-4), and either fix the registration's expectation or declare the STOP.

## Both
- Add a "Blindness status" line: "primary statistic point estimates exist in a gate record dated
  2026-09-03 (unblinded by a design-gate side effect); band thresholds were frozen before that
  record; the registered read is executed by an agent that has not opened that record."
