# r-cone-loss — DESIGN-VALIDITY GATE RECORD

Node: `r-cone-loss` design gate. Research Graph 1, Branch H, wave 3.
Author of record for all scientific decisions: Jasper Seehofer.
Lens: **DESIGN VALIDITY, blind to results** — no `cone_loss_result.json` exists yet and none
was read to produce this record; only `REGISTRATION_DRAFT.md`, its cited source files
(`cmem_a1.py`, `cmem_reads.py`, `b4_imp_stage1_forecast.py`), on-disk data (CSVs, checkpoints,
CRBs), and the governing charter/docket text were consulted. Six checks adapted from the
`r-b82-s4` precedent (`../r-b82-s4/DESIGN_GATE_RECORD.md`) to this arm's own six named items.

Every number below was re-derived or read from the file cited, per the standing rule
(verifier output is evidence, not authority).

## Check 1 — Object + population pins unambiguous and disjoint from the sibling seed blocks: **GREEN**

The three seed blocks named in the task (`901000-901099`, `902000-902024`, `901100+`) belong to
the *sibling* arm `r-b82-s4` / `m-s3-postflip-coverage` (Cell S, Cell T, falsifier reservation
respectively — confirmed in `r-b82-s4/DESIGN_GATE_RECORD.md` Check 3). `r-cone-loss` touches two
populations:

- **Production pool** (`seed61000/prepared_cramer_rao_bounds.csv`, md5
  `9a1f2a14384a9281c97ca3be312ddaab`, byte-identical to the retrieved run's copy at
  `graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib/simulations/`, confirmed by
  `md5sum`) — a single pool, unrelated to the `90xxxx` seed-block numbering. Unambiguous.
- **Harness replicate** — reads `b8_cal_harness_work_s4_postflip/seed9010NN_S/...`. Verified by
  directory listing: exactly 67 `_S` universes on disk, seeds `901000`–`901066` (contiguous),
  **zero** seeds ≥ `901100`. This is a **subset of the Cell-S block currently 33 % populated**,
  not the falsifier block — confirmed disjoint from `901100+` by direct enumeration
  (`awk '$1>=901100'` on the seed list returns empty) and disjoint from `902000-902024`
  (different cell letter, never globbed). The registration's own text frames this as a read of
  already-banked checkpoints ("free re-reads first", `INFORMATION_FORECAST.md` §1), consistent
  with zero new compute in that seed range.

No collision with the reserved falsifier block; no ambiguity in either population definition.

## Check 2 — Statistic fully specified, zero fresh choices in the launch block: **GREEN**

- Columns `combined_no_bh` / `combined_with_bh` confirmed present in
  `.../simulations/diagnostics/event_likelihoods.csv` header (checked on the retrieved
  `headrebaseline_iiib` run).
- The stencil (0.725, 0.735) and the score formula `(ln(hi) - ln(lo)) / (0.735-0.725)` are not a
  fresh choice — they reproduce `per_event_scores()` in the cited
  `fanout1_20260829/b4_imp_stage1_forecast.py:136-143` line-for-line (`i_lo`/`i_hi` via
  `np.isclose(grid, 0.725/0.735)`, same division).
- `cone_radius()` (cited as `cmem_a1.py:85`) exists at that line and matches the K·√λ_max form
  used in the definition table; the G-2 double anchor (`cmem_a1.py:67` →
  `ANCHOR = ("bc", 900101, 0, 0.0116656941007181, 0.0359121946154451)`; `cmem_reads.py:32` →
  `ANCHOR = ("bc_900121_work", 20, 1.674660e-03, 1.4956979545757095e-03)`) reproduces byte-exact
  against the registration's quoted digits.
- SE/φ/materiality arithmetic re-derived independently and matches the draft to the last digit:
  `SE(Δh_1D) = 0.68·√(10+100/66)/3256 = 0.000708 ≈ 0.0007`; `φ=0.2 → Δh=0.0126`; `T_mat/SE ≈ 11.3`;
  the forecast's `−17 nats/h → Δh≈−0.00522 → φ≈0.083` also reproduces.
- Row counts cited in §6 (65,108 rows; 7,216 rows) match `event_likelihoods.csv` line counts
  (65,109 and 7,217 including header) to the header-line rounding.
- Minor non-blocking note: `event_likelihoods.csv` is long-format (one row per event×h), so the
  builder must pivot to an event×h matrix before calling `per_event_scores()`; the pivot key
  (`event_idx`, `h`) is unambiguous from the header and is not itself a fresh scientific choice.

## Check 3 — Disposition table three-valued, every outcome returns as a fresh RULE: **RED**

Structurally the table is a 3-valued science trichotomy (IMMATERIAL-FLOOR-SHARE /
CONE-OWNS-FLOOR / INTERMEDIATE) plus a terminal INSTRUMENT/NO-READ row that pre-empts the
trichotomy — the same shape Check 6 of `r-b82-s4`'s own gate found GREEN. That part is fine.

The section-4 header claims **"every row returns as a fresh RULE."** Tracing this against the
governing docket (`DECISION_DOCKET_WAVE3_20260903.md` item 2.3, RULE/procedural, Ratified)
finds a **concrete conflict on the CONE-OWNS-FLOOR row**:

- The row's action cell reads: `d-residual-attribution: the floor is geometric; no consistency
  fix pursued **(charter)**` — citing the charter itself as authorizing this outcome directly,
  with no "returns as fresh RULE" language (unlike the INTERMEDIATE row, which explicitly says
  "fresh RULE: bank the share").
- `RESEARCH_GRAPH_1_PROPOSAL_20260901.md:189` defines `d-residual-attribution` as a **RULE node**
  gated on **four** prerequisites: `d-calibration ruled; d-photoz-leverage ruled;
  m-completion-residual done with g-closure green; m-cone-loss done` — not on `m-cone-loss`'s
  CONE-OWNS-FLOOR outcome alone.
- `DECISION_DOCKET_WAVE3_20260903.md` item 2.3 (RULE, Ratified, same session as this launch)
  states explicitly: **"d-photoz-leverage, d-calibration, d-residual-attribution and the paper
  rulings return as dossiers for the MORNING, not as chair rulings."**
- The charter's own decision-table item 9 (line 276) draws the same line for the sibling arm
  `r-completion-residual`: registration **authoring** is Approved, but "the attribution split
  (d-residual-attribution)" is listed under **explicitly NOT covered**.

So a CONE-OWNS-FLOOR read (registered at 5 % probability per `INFORMATION_FORECAST.md` §3, but
a live registered branch, not a hypothetical) would, if its action cell is executed as literally
written, have the chair settle a piece of `d-residual-attribution` under the "(charter)" citation
— the exact node this session's own ratified docket item just reserved for the morning and for
the author, and whose charter-defined prerequisite set (`d-calibration`, `d-photoz-leverage`)
is not met tonight regardless. This is not a wording nitpick: it is a live path, inside tonight's
STANDING grant (docket 2.2), for an agent to over-execute past the approval-scope boundary that
`CLAUDE.md`'s binding default and this session's own docket item 2.3 draw. `IMMATERIAL-FLOOR-SHARE`'s
"q-cone-loss SETTLED" action is **not** the same problem — `q-cone-loss`'s kill criterion
(`RESEARCH_GRAPH_1_PROPOSAL_20260901.md:46`) was pre-ratified at charter time (row 0), so a
registered-band confirmation closes it mechanically, matching the IMMATERIAL row's own bound
language. `d-residual-attribution` has no such pre-ratified single-arm closure rule.

**Must-fix:** rewrite the CONE-OWNS-FLOOR action cell to (a) drop the "(charter)" citation as
sole authority, (b) state that it *contributes* CONE-OWNS-FLOOR evidence toward
`d-residual-attribution`, which remains open pending `d-calibration` + `d-photoz-leverage` per
charter line 189, and (c) route the disposition itself as a fresh RULE — explicitly deferred to
the morning per docket item 2.3, exactly like the INTERMEDIATE row's phrasing. Until this is
fixed, an agent following the table as written could rule a reserved node tonight.

## Check 4 — Gates named with bands and a STOP consequence: **AMBER**

G-1 (`STOP on mismatch`) and G-4 (`⇒ INSTRUMENT-DEFECT, STOP, fresh RULE`, with the explicit
band `[13.4 %, 32.5 %]` and `α = 0.05`) both name an explicit band and an explicit STOP. G-2
names the consequence ("a miss = INSTRUMENT-DEFECT") but not the literal word STOP — functionally
equivalent, minor vocabulary drift from G-4. **G-3** ("event_idx = CRB row index... the P6
numerator 66 must equal n_IN") states an equality condition but **no consequence at all** on
mismatch — it relies entirely on §4's catch-all last row ("G-1…G-4 red ⇒ nothing banked") to be
covered, unlike every other named gate in the section. Functionally reachable but not
self-contained at the point where a reader checks G-3 alone.

**Must-fix (non-blocking, low severity):** add "mismatch ⇒ INSTRUMENT-DEFECT" to G-3's own line,
for the same self-containment G-1/G-2/G-4 already have. `g-population`/`g-censoring`/`g-precision`
correctly carry no STOP language — per the charter's own taxonomy (line 245) these are disclosure
invariants (promote bound→estimate at `d-calibration`), not launch gates, so their omission is
consistent with project convention, not a gap.

## Check 5 — Cost derived and under the docket cap (≤ 20 CPU-h): **GREEN**

Docket item 2.2 pins `m-cone-loss (≤20 CPU-h)` verbatim — confirmed by direct read of
`DECISION_DOCKET_WAVE3_20260903.md`. The registration's cost line items are traceable: the
catalogue load path and the production/harness CSVs are all present on disk (paths verified
above); the quoted row counts (65,108; 7,216) match the actual files to the header-line rounding.
`≈ 0.1 CPU-h` vs. the `20 CPU-h` cap is a 200× margin — even a 10-20× underestimate of the local
compute would not approach the cap. No cluster component (§6 states "Zero cluster" — consistent
with the task's own constraint that this session never launches cluster jobs).

## Check 6 — `max_revisions = 2` and the parent kill criterion quoted: **AMBER**

`max_revisions 2 (ORCHESTRATOR-DERIVED, charter §1.8/§1.13)` is correct: `RESEARCH_GRAPH_1_
PROPOSAL_20260901.md` §1.8 (line 157) and §1.13 (lines 206-211) both carry `max_revisions 2`
for `r-cone-loss`, ORCHESTRATOR-DERIVED, matching the draft exactly.

The kill criterion, however, is **paraphrased, not quoted verbatim**. Charter line 46 (the
`q-cone-loss` row, quote-verbatim convention per runbook 42 §5 / the `BIAS_HISTORY_LEDGER`
attribution-precise convention CLAUDE.md invokes for exactly this kind of citation):

> "measurement confirms the floor within its registered uncertainty band -> settled as
> irreducible geometry; no fix pursued"

Registration draft §4 (line 95):

> "confirms the floor within its band ⇒ irreducible geometry, no fix"

Substance matches; the words "measurement", "registered uncertainty", "settled as", and
"pursued" are dropped, and it is not set off as a quotation. Given this project's explicit
verbatim-quoting discipline for exactly this class of citation (attribution-precise recording,
CLAUDE.md "Proposing decisions"), this should be a literal quote in quotation marks, not a
compressed paraphrase — low severity (no substantive drift), but worth a mechanical fix before
the record that resolves `q-cone-loss` cites it as its authority.

## Overall verdict: **RED — do not launch `m-cone-loss` until Check 3 is fixed**

One check is RED: the CONE-OWNS-FLOOR disposition row, as currently drafted, lets an
in-scope-tonight agent settle a piece of `d-residual-attribution` — a RULE node this same
session's docket (item 2.3, ratified) explicitly reserved for the morning and whose charter-
defined prerequisites are not met tonight — under a bare "(charter)" citation rather than a
fresh-RULE routing. This is a low-probability branch (5 % per the stage-1 forecast) but a live
one inside tonight's STANDING launch grant, and the fix is small: rewrite that one action cell
per the must-fix above. Checks 4 and 6 carry non-blocking must-fix notes (G-3's missing explicit
STOP line; the kill-criterion paraphrase) that should be folded into the same revision pass but
do not independently justify a red verdict. Checks 1, 2, and 5 are clean: the pins are
unambiguous and correctly disjoint from the sibling arm's reserved seed block, the statistic is
fully specified and its arithmetic reproduces independently, and the cost is honestly derived
with large headroom under the cap.

**Recommendation:** the fix touches only the REGISTRATION_DRAFT.md disposition table (Check 3)
and, optionally in the same pass, G-3's consequence line (Check 4) and the kill-criterion
quotation (Check 6) — none of it touches the launch parameter block itself (§7), which is
otherwise ready. Re-run this gate against the revised draft; it does not need to consume one of
`r-cone-loss`'s two `max_revisions` (this is a design-gate correction on the still-PROPOSED
draft per docket 2.1/2.2, not a post-launch revision of a frozen registration).
