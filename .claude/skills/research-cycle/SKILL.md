---
name: research-cycle
description: >
  Use at the start of every new investigation, mechanism hunt, or claim
  assessment in this project — instead of reinventing a runbook each time.
  Chains the seven battle-tested stages (claim intake → information forecast →
  pre-registration → measure/refute → calibration gate → decision → chronicle)
  onto the assets that already exist in this repo. Invoke before opening any
  new bias/mechanism hypothesis, before pre-registering a run, and at every
  keep-digging vs stop-and-report-bound decision point.
argument-hint: <investigation name or claim to assess> [--stage N]
---

## The Research Cycle

Standing cycle of record, mandated by
`results/campaign51_20260728/RUNBOOK_NEXT_SESSION_7.md` §2 (2026-08-04).
Fuller stage-by-stage wiring: **`docs/RESEARCH_CYCLE.md`** — read it when
running a stage; this file is the index and the hard rules.

**These are guard-rails, not a decision procedure.** The author
(Jasper Seehofer) owns every scientific decision. Stage 5 decisions and every
`/physics-change` gate are author-gated: present, then STOP.

**D1 — the p0-window mass band-pass** (RUNBOOK-7 §1 item 2) is the designated
first investigation to run through the full cycle — constraint of record: the
3135-event catalogue stays band-passed and must never be re-scored against
band-blind objects (RUNBOOK-7 §1.2b); the p0-bounds retirement is simulation-side,
for future campaigns only.

### Stage map

| # | stage | question | primary assets |
|---|---|---|---|
| 0 | Claim intake | what exactly is claimed, with what provenance? | `results/campaign51_20260728/realistic_20260729/CLAIM_2D_BIAS_20260730.md`; both exoneration layers |
| 1 | Information forecast | what would a perfect analysis of this data say? | `docs/SIGMA_Z_SIGMA_M_FORECAST.md`, `scripts/bridge_closure/sigma_z_sigma_M_forecast.py` |
| 2 | Pre-registration | hypotheses, decisive reads, branches, bands — BEFORE running | `results/campaign51_20260728/realistic_20260729/PREREGISTRATION_2x2_cellB.md` |
| 3 | Measure / refute | Gates A→B→C, in order, no skipping forward | `results/campaign51_20260728/RUNBOOK_NEXT_SESSION_6.md` §1–§5; `/commission --research` |
| 4 | Calibration gate | is the estimator calibrated at the production venue? | three legs: `darksiren_emri/validation/pp_coverage.py` + the (ii-d) absolute detected-count audit + forecast-consistent width |
| 5 | Decision | measure / report bound / fix / one more measurement | `/physics-change`, `docs/gates/PHYSICS-GATE-LEDGER.md`, author gate |
| 6 | Chronicle | ledger rows, claim writebacks, next runbook | `docs/gates/PHYSICS-GATE-LEDGER.md`, `results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`, `results/campaign51_20260728/RUNBOOK_NEXT_SESSION_<N>.md` lineage |

`--stage N` enters mid-cycle; without it, start at 0. Stages are ordered — do
not skip forward, and say so explicitly when you deliberately re-enter earlier.

### Stage L — external consult (cross-cutting)

Invocable from stage 0 and stage 5; the 0–6 pipeline is **not** renumbered.
Full wiring: `docs/RESEARCH_CYCLE.md` § "Stage L"; register:
`docs/LITERATURE_WARNINGS.md`.

**Procedure.** (1) **Symptom card** — the problem as observed signatures (numbers,
venues, what survived which controls), then restated up an abstraction ladder:
field-specific → method class → math class → generic inference pathology, each rung
with its own search vocabulary. (2) **Rings, in order, R0 first always** — R0 =
papers **already cited in this repo**, re-read for warnings/caveats/validity
conditions (*already-cited ≠ already-heeded*); R1 = citation neighborhood, esp.
**forward** citations filtered on bias/inconsistent/caveat/erratum/corrigendum, plus
mock-data-challenge and code-comparison papers; R2 = the field; R3 = the math class
(astrostatistics, selection-effect primers, SBC); R4 = code/numerics. (3)
**Independence** — the searcher gets the symptom card and **never** the current
suspect list; timeboxed. (4) **Intake** — each candidate enters stage 0 as a `[LIT]`
claim with section/eq numbers; **quote-verification before mapping is MANDATORY**;
then the two-layer exoneration check; then the cheapest decisive measurement. Exit:
candidates ranked by signature-match × cost-to-test → stage-2 preregs, **or** a
documented "the field has no answer" (itself a reportable result), plus updated
`docs/LITERATURE_WARNINGS.md` rows.

**Triggers.** (a) MANDATORY lightweight R0 sweep at every new mechanism thread's
stage 0; (b) **STUCK SIGNAL** — two consecutive MIXED/UNDETERMINED verdicts on one
thread auto-triggers a full Stage L; (c) before any `/physics-change` **adoption**
(does the literature document this fix's failure modes?); (d) lightweight pass at
each runbook lineage rollover.

**Assumption register.** The `/physics-change` gate package is now **6 items**: the
five plus **the source equations' stated validity conditions, checked against our
venue/regime** (approved 2026-08-05, pending application to
`.claude/skills/physics-change/SKILL.md`). Registered rows live in
`docs/LITERATURE_WARNINGS.md` with status CHECKED / VIOLATED / UNDER MEASUREMENT /
OPEN / N-A / UNCHECKED — every row cites evidence or says UNCHECKED.

### Hard rules

1. **Check both exoneration layers before opening any mechanism.** The local
   `## Exonerated — do NOT re-open without new evidence` list in the relevant
   `CLAIM_*.md`, **plus** §2 of
   `results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`
   — the standing rule of RUNBOOK-7 §1. The binding set is the union.
   Re-litigating an exonerated suspect is this project's most expensive failure
   mode. Exonerations are venue-scoped, not universal.
2. **Never build on an un-adjudicated `[AGENT]` claim.** Tag vocabulary:
   `[LOCAL]` re-measured here · `[AGENT]` measured by a subagent from a source
   not re-measured here; NOT independently reproduced · `[DOC]` read from a
   committed artifact · `[INFER]` inference from `[LOCAL]`/`[DOC]` ·
   `[LIT]` quoted from external literature (Stage L), full citation with
   section/eq numbers, quote-verified before it is mapped onto our pipeline.
   Reproduce before you use.
3. **Every claim carries a `Refute by:` clause** naming the cheapest decisive
   falsification test. A claim without one is not intake-complete.
4. **Pre-register before running.** Numeric falsifiable bands, a first-class
   `Mixed` branch, secondary reads including expected *nulls*, provenance-gating
   ("what upstream gate made this test necessary"), and append-only discipline
   ("after this file is committed, no edits above this line").
5. **Gates A→B→C, strictly ordered.** A = provenance (regenerate any headline
   number whose source artifact is gone, or the claim stays unfalsifiable).
   B = adversarial refutation (survivors get promoted CLAIM→FINDING and written
   back). C = alternative causes, before accepting any surviving mechanism.
   "Refute before you build" — start from "is that true, and is it even
   attributable?", never from "the mechanism is X."
6. **Measurement-before-gate.** A cheap mechanical measurement that could
   collapse the need for an expensive test (or for a `/physics-change` gate at
   all) runs first.
7. **Pass paths, not paraphrases.** Give every subagent the claim-file path and
   the exoneration list verbatim. Paraphrase drift caused prior dead ends. Cap a
   refutation workflow at 6 agents.
8. **Physics routes to `/physics-change`.** Any formula/constant change in a
   trigger file is a hard gate: 5-item package → author approval → ledger rows.
   Tag fixes `instrumentation` (plain GSD) vs `formula` (`/physics-change`).
9. **Exhaust the free re-reads before requesting compute.** [A1] Every
   diagnostics artifact already on disk (`event_likelihoods.csv`, readouts,
   run bookkeeping) is re-readable at zero marginal cost. Enumerate what the
   existing artifacts can already decide, and read them, before asking for a run.
10. **Never compare a class-summed statistic without a paired per-event read.**
    [A2] Any Σ-over-events quantity compared across venues, configurations, or
    eras requires a paired/stratified per-event read **alongside** the aggregate.
    Opposing sub-population effects cancel in the mean and manufacture spurious
    agreement. The paired read is free; the retraction is not.
11. **Declare invariants and structural blindness at pre-registration (stage 2).**
    [A10] Every prereg lists (a) **Invariants** — everything held fixed across
    every arm, one line each with a last-audited date or `NEVER` — and (b)
    **Structural blindness** — one sentence naming the defect class this design
    cannot detect by construction. A `NEVER`-audited, load-bearing invariant
    either gets audited this cycle, or the VERDICT states its conclusions are
    conditional on it, by name. Evidence: `B_scale` (+0.12 posterior mean, twice
    the bias under study) survived four campaigns because every arm held it
    fixed — no amount of fan-out would have found it.
12. **Stamp provenance freshness on every number entering a budget (stage 4/5).**
    [A11] Any quantity quoted into a budget, band, or verdict carries `{value,
    source (commit/artifact), date, configuration-of-record}`. A stamp whose
    configuration no longer matches the current one is STALE and may not be
    quoted as a point value — re-measure it, or carry it as an explicit band
    with the staleness disclosed. Evidence: `s_Edd = -0.020` sat in a code
    comment, was frozen into a prereg and a ratified decision, and was wrong by
    an order of magnitude and sign (re-measured: `+0.0012`) six weeks after the
    repo's own gate record had flagged it. Attached harness rule (`CLAUDE.md`):
    every multi-GB input not in version control carries a checksum pin at each
    consumer with a STOP gate.
13. **Run the score-zero test first, before building instruments (stage 3).**
    [A12] For data drawn from the estimator's own model, `E[∂_θ ln L]` at truth
    is zero. Compute the per-event score at truth over the relevant event class
    before building any other instrument; a non-zero mean at high significance
    localizes a misnormalization, and reading it further by class and by
    covariate localizes it further — all on banked data, zero compute.
    Evidence: this is what cracked the campaign — `−0.635 ± 0.017` (37σ) on the
    dark class, then class-resolved (~195% of the slope on one class) and
    z-resolved (≈0 below z≈0.4, −1.08 by z≈0.9).
14. **Gate every counterfactual instrument on engagement and dispatch path
    (stage 3).** [A13] No null from an instrument is interpretable until it is
    shown to move the output: register an engagement threshold (e.g. "≥10% of
    relevant events move by ≥1e-6 relative", or a table ratio ≠ 1), score it,
    STOP-gate on failure, and confirm the switch reaches every dispatch path
    production actually uses — assert the switch's runtime value per arm.
    Evidence: production dispatches through a batch kernel; an instrument
    patched only into the scalar kernel would have passed bit-identity and
    continuity gates and returned a confident false "no effect" — caught
    pre-execution by a verifier.
15. **An attribution ships with its own falsifier (stage 5/6).** [A14] Any memo
    or verdict that attributes an effect to a cause registers, in the same
    document and before the attribution is banked, the experiment that would
    falsify it, with bands. An unrun falsifier means the attribution stays
    explicitly provisional. Evidence, both directions: the population-
    misspecification memo registered its falsifier in §7 and was cleanly
    downgraded when it ran; the earlier impostor-overlap reading had none and
    had to be undercut by a later measurement instead.

Full rationale for A10–A14: `.claude/skills/research-cycle/PROPOSED_AMENDMENTS_20260820.md`
(author-ratified 2026-08-20).

### Model & effort policy (RUNBOOK-6 §2)

| task shape | model | effort |
|---|---|---|
| Mechanical extraction: grep logs, checksum, pull columns, recompute a documented formula | haiku | low |
| Bounded code tracing with a named file/function target | sonnet | medium |
| Literature check against a cited paper's equations | sonnet | medium |
| Independent re-derivation; adversarial refutation of a claim | opus | high |
| Final adjudication across conflicting agents; the physics decision | fable | xhigh |

Never spawn opus/fable for what haiku can verify. Tell the Gate-B adjudicator
explicitly that "refuted" and "undetermined" are acceptable, valued outputs.

### Stop/continue rule of record (author-endorsed 2026-08-04)

The per-event ln-posterior min/max range is a **screen, never a stopping gate** —
N coherent sub-threshold tilts dominate the ensemble (measured: per-event
0.3–0.5σ rails vs +3.4–6.1σ class-summed).

**"Stop digging" requires all three:**
1. coverage pass (stage 4), **and**
2. width on the F5 forecast (stage 1), **and**
3. no unmodeled selection between generator and estimator.

SBC/coverage **alone cannot catch** a filter both sides silently share (the D1
class of defect) — it cancels out of the coverage statistic. The absolute-count
audit is the complement that caught it; both legs of stage 4 are mandatory.

### Stage-5 decision mapping

| verdict | action |
|---|---|
| CALIBRATED + narrow | measure — report the H₀ measurement |
| CALIBRATED + wide (≈ forecast) | **stop digging, report a bound** |
| DEFECT (≥3σ coherent class displacement, or coverage failure) | fix via `/physics-change` |
| UNDETERMINED | identify and run the *one* measurement that decides |

Author-gated. Present the verdict with its evidence and STOP.

### Known gaps — do not pretend these exist

- **Fisher forecasts** (named in RUNBOOK-7 §2 stage 1) are **not** a repo asset.
  Only the F5 σ_z/σ_M closure engine exists. TO-BUILD if a stage-1 forecast
  needs a Fisher leg.
- `pp_coverage.py` is **single-channel and synthetic-catalogue**: no mass/BH-mass
  channel, no realistic host-observation model. The two-channel extension and the
  GLADE-like host model that stage 4 requires are **TO-BUILD**, and the build is
  not accepted unless it is [A3]: (i) genuinely 2-channel with the completion-leg
  mass factor **g recomputed per h**, never frozen or elided; (ii) run at
  **production N** — the mechanism is N-coherent (per-event sub-threshold, ~0.05 nats grid-wide in ḡ) and invisible
  at small N; (iii) **multi-candidate host balls** — a one-candidate-per-event
  harness structurally cannot exercise it (ledger row 86).
- `/commission --research` is a **user-level** skill (`~/.claude/skills/commission/`),
  not committed with this repo, and RESEARCH mode needs a `.commission-research.yaml`
  config that this repo does not have. **TO-SCAFFOLD** before first use.

### Cross-session counterpart

`/commission --research` is the standing, memory-bearing falsification pass over
a thread's accumulated claims (delta → falsify → claim-history regression diff →
typed feedback). Propose-only. It is stage 3's periodic counterpart to the
in-session Gates A–C; run it at go/kill/pivot points, i.e. alongside stage 5.

### Amendments

The cycle is itself append-only-governed: every change to it adds a row to the
**Amendment ledger** at the end of `docs/RESEARCH_CYCLE.md`. No silent edits.
Rules tagged `[A1]`/`[A2]`/`[A3]`/`[A10]`/`[A11]`/`[A12]`/`[A13]`/`[A14]` above are
session-earned amendments — the ledger carries the evidence that earned them.
