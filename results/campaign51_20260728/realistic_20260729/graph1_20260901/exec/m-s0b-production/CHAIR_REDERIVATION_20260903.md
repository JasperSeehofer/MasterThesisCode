# m-s0b-production — chair re-derivation + booking (2026-09-03, Fable 5.1 chair)

Reader record: READOUT_RECORD.md (fresh sonnet reader). Evidence, not authority: every decisive
number below was recomputed by the chair directly from the five per-node `event_likelihoods.csv`
files (ln of `combined_no_bh` / `combined_with_bh`, N = 1588, secants per PA-HIER-31(d)/(4)).

## 1. Re-derived (chair script; MATCH the reader and the driver's --score-only to the digit)
| channel | statistic | mean | SEM | Z |
|---|---|---|---|---|
| no_bh (primary) | score_b_re = [lnL(+0.033,1) − lnL(−0.033,1)]/0.066 | −0.6822 | 0.1293 | **−5.274** |
| no_bh | score_lns = [lnL(0,√2) − lnL(0,1/√2)]/ln 2 | −0.0327 | 0.0045 | **−7.188** |
| with_bh | score_b_re | −0.7412 | 0.1195 | −6.204 |
| with_bh | score_lns | −0.0368 | 0.0050 | −7.333 |
Reader's PA-HIER-33 Bartlett-corrected s-axis Z = −7.101 (not re-derived by the chair; within
0.09 of the raw secant, direction identical — immaterial to any band).
Class split at truth: L_cat_no_bh > 0: 1139 · == 0 (dark): 449 (chair count, both channels).

## 2. Registered dispositions (PA-HIER-31(e)), chair-confirmed from the reader's curvature leg
- B0-B: |Z_b| = 5.27 > 3 and |Z_lns| = 7.1 > 3 → **LEVER-LIVE** (not LEVER-DEAD-AT-N).
- B0-M: |b̂| = 0.0114 < 0.0165 (b-axis small) ; |ln ŝ| = 1.165 > 0.173 (s-axis material) → **MIXED**.
- B0-P: σ_b = 0.0032 < 0.0661 ; σ_ln s = 0.150 < ln 2 → **POWERED both axes**.
- Cap: REPORTED-ONLY, unconditional (PA-HIER-28 item 9). No CALIBRATED claim.
- GATE ENG on the registered `_re` pair: 54 % / 49 % of events move ≥1e-6 → pass (the driver's
  own gate_eng looked for the as-built ±0.02 names and reported an absence, not a failure).
- C-C identity check: 449 events, 0 deviation across all 5 nodes → instrument pass.

## 3. Two things the chair FLAGS for the author (docket 2.3: this returns as a dossier, no ruling)
(a) **Charter clause conflict.** Graph §2 panel: "g-score-null … abs Z ≤ 3 … a red STOPs
    d-photoz-leverage; reopens the instrument question as a fresh RULE, never auto-recertifies".
    On the production venue there is no separate control: the score at the assumed-truth node IS
    the registered measurement (PA-HIER-31(h)), and its non-nullness is precisely the LEVER-LIVE
    outcome the prereg's own disposition table anticipates. Read literally, the panel clause would
    forbid ever reading a positive S0-B. Chair-derived reading (flagged, veto-able): the instrument
    certification of record is row #287 (both axes |Z| ≤ 3 on the mirror, where truth is known);
    the production red is a MEASUREMENT under the registered table, not an instrument red. The
    conflict itself is put to the author as part of d-photoz-leverage; the chair does not
    "reopen the instrument question" on its own.
(b) **Class-count anchor mismatch (OPEN, instrument-level).** Registered anchor (b3_pop_prediction,
    2026-08-29, venue iiib, same 1588 events): dark C-C = 606, C-A∪C-B = 982. This run: 449 / 1139.
    157 events changed class. Forensics launched (CLASS_COUNT_FORENSICS.md); until it lands, the
    dispositions above are booked PROVISIONAL-ON-POPULATION. Chair check: all 5 nodes resolved
    `catalogue_leg_1d_mass_aware: "off"` (grep of the node JSONs), so the row #287 certified config
    WAS honoured and the A18 flip is EXCLUDED as the cause; remaining candidates are the 2026-08-25
    symmetric mass window, the h-grid decoupling, or a different class criterion / catalogue state
    behind the 2026-08-29 anchor.
(c) Cost: 5 tasks × ~7.5 min wall; the reader's 9.98 CPU-h "allocated" is the sbatch reservation,
    the row #332 ≈2 CPU-h is elapsed×cores actually used — both are true, different quantities.

## 4. Chair booking (under the row #325 grant; returns inside d-photoz-leverage)
m-s0b-production: DONE. Disposition **LEVER-LIVE / MIXED / POWERED, REPORTED-ONLY,
PROVISIONAL-ON-POPULATION** pending (b). c-theta-pull-venue-physics: the registered discriminating
measurement did NOT null it; promotion/interpretation is the author's (kill criterion of
q-theta-pull "production null at |Z|≤3 → not venue physics" is NOT met). d-photoz-leverage dossier
assembled for the morning once (b) resolves.
