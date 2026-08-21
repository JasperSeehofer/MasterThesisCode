# The Research Cycle

The standing, standardized investigation procedure for this project. Mandated by
`results/campaign51_20260728/RUNBOOK_NEXT_SESSION_7.md` §2 (author mandate,
2026-08-04): every future investigation follows this one cycle instead of
reinventing its runbook. The pieces below already existed and are battle-tested;
the cycle chains them.

Entry point: **`/research-cycle`** (`.claude/skills/research-cycle/SKILL.md`) —
that file is the index and the hard rules; this file is the stage-by-stage
wiring.

**Authorship discipline.** The cycle is guard-rails, not a decision procedure.
The author (Jasper Seehofer) owns every scientific decision. Stage-5 decisions
and every `/physics-change` gate are author-gated: present, then STOP.

**First full run:** D1 — the p0-window mass band-pass (RUNBOOK-7 §1 item 2) — is
designated the first investigation to go through all seven stages — constraint of
record: the 3135-event catalogue stays band-passed and must never be re-scored
against band-blind objects (RUNBOOK-7 §1.2b); the p0-bounds retirement is
simulation-side, for future campaigns only.

Paths below are repo-relative.

---

## Stage 0 — Claim intake

**Question:** What exactly is claimed, with what provenance, and has it already
been ruled out?

**Entry criteria:** A candidate mechanism, suspect, or headline number exists.

**Assets**
- Claim file, canonical example:
  `results/campaign51_20260728/realistic_20260729/CLAIM_2D_BIAS_20260730.md`
- Exoneration layer 1 (local): that file's
  `## Exonerated — do NOT re-open without new evidence` section.
- Exoneration layer 2 (project-wide, authoritative):
  `results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`
  §2 "DO NOT RE-TRY", plus §1 chronology, §3 history-vs-current-claims, §4 open
  threads.

**Procedure**
1. **Two-layer exoneration check — standing rule, before opening anything.**
   Grep both layers. The binding set is their union; ledger items flagged ⚠ are
   absent from the local list and are the live re-litigation risk. Exonerations
   are **venue-scoped**, not universal (two exonerations measured on the same
   subsample would both be fooled by a shared venue idiosyncrasy).
2. Write or extend the claim file. Header states
   `Status: **CLAIM, NOT ESTABLISHED.** Written to be attacked.`
3. Tag every statement: `[LOCAL]` re-measured here, reproducible offline ·
   `[AGENT]` measured by a subagent from a source not re-measured here; **NOT**
   independently reproduced · `[DOC]` read from a committed artifact · `[INFER]` inference from
   `[LOCAL]`/`[DOC]` with no new measurement. Never build on an un-adjudicated
   `[AGENT]` claim without reproducing it.
4. Each numbered claim `## CN — <title> [TAGS]` carries: the measured statement,
   the exact command/method, and a **`Refute by:`** clause naming the cheapest
   decisive falsification test.
5. Close with `## What is explicitly NOT claimed`, the Exonerated list, and
   `## Errors made this session — do not inherit them`.

**Exit artifact:** A committed `CLAIM_<topic>_<date>.md` whose every claim is
tagged and has a `Refute by:` route, plus a recorded exoneration check.

---

## Stage 1 — Information forecast

**Question:** What would a *perfect* analysis of this data say? Pre-register the
expected σ(H₀) before measuring it.

**Entry criteria:** Stage 0 complete; the claim implicates a precision or a bias
that a forecast can bound.

**Assets**
- `docs/SIGMA_Z_SIGMA_M_FORECAST.md` (the F5 feasibility figure, §6 result
  table, §4.3 floor caveat).
- `scripts/bridge_closure/sigma_z_sigma_M_forecast.py` — self-consistent-closure
  engine, unbiased by construction; metric is σ_eff(H₀)/H₀ = RMSE-to-truth (raw
  width misleads on railed posteriors). Two channels: 1-D and 2-D-with-BH-mass.
  ```bash
  uv run python scripts/bridge_closure/sigma_z_sigma_M_forecast.py --smoke
  OMP_NUM_THREADS=1 uv run python scripts/bridge_closure/sigma_z_sigma_M_forecast.py \
      --sweep --workers 14 --seeds 16 --population real_nz \
      --out scripts/bridge_closure/outputs/sigma_z_sigma_M_forecast_realnz.json
  ```
  `--out` defaults to `outputs/sigma_z_sigma_M_forecast.json` — the paper's
  headline synthetic result. Always redirect the `real_nz` pass, or it is
  silently overwritten.

**Procedure**
1. Pick the channel; read σ_eff/H₀ at the target (σ_z, σ_M) from §6 or run the
   engine.
2. Rescale from the N=400 baseline as σ ∝ N^(−1/2). Only *structural*
   conclusions (railing thresholds, the σ_M·(1+z) ≲ σ_z frontier) are
   N-independent.
3. Use `--population real_nz` as the load-bearing realistic check — the
   synthetic-population 2-D advantage is a pure-numerator gain under
   idealizations that flatter it.
4. Do not quote sub-≈1.4% forecasts (floor caveat, §4.3).

**GAP — TO BUILD.** RUNBOOK-7 §2 names "Fisher forecasts" for this stage. **No
Fisher-forecast asset exists in the repo**; only the F5 closure engine does. If
a stage-1 forecast needs a Fisher leg, that leg must be built first and said so.

**Exit artifact:** A pre-registered expected σ(H₀) (number + channel + N +
population mode + engine commit), carried into stage 2.

---

## Stage 2 — Pre-registration

**Question:** What are the hypotheses, the decisive reads, the outcome branches,
the STOP signals, and the calibration bands — written **before** running?

**Entry criteria:** Stages 0–1 done and an upstream gate has established that
the test is necessary (pre-registration is downstream of a provenance check).

**Assets / precedents**
- `results/campaign51_20260728/realistic_20260729/PREREGISTRATION_2x2_cellB.md`
  — the design-matrix pattern.
- `results/lcat_h_dependence_20260725/INFORMATION_FLOOR_PREREGISTRATION.md` —
  threshold registration with explicit anti-tuning provisions.
- `results/lcat_h_dependence_20260725/SEED600_GATE_REGISTRATION.md` — per-criterion
  PASS/FAIL table and the append-only verdict rule.

**Skeleton (required properties in bold)**
```
# Pre-registration — <test name>
Registered <date>, BEFORE <the action>. Per <parent runbook> §<n>.
<why this test is necessary, with concrete provenance: job ID, file, metadata values>

## The run
RUN_DIR / inputs / catalogue / estimator config / code commit — each with a
concrete identifier (sha256, symlink target, commit hash) so the run is
byte-reproducible from this file alone. Explicit "no code change" note where true.

## The design matrix
<table; which cell is the new one, what each margin means physically>

## Pre-registered readings
- <branch 1>: <quantitative numeric band> ⇒ <meaning>
- <branch 2>: <band> ⇒ <implication>
- **Mixed**: read the split directly, do not force a branch.

Secondary pre-registered reads:
- directional sub-predictions conditional on the leading mechanism
- expected NULL results ("expected bit-identical; if it differs, that is itself a finding")

<verdict appended below by the reading session — no edits above this line>
```
- **Quantitative falsifiable bands**, preferably a conjunction of two
  independent reads.
- **A `Mixed`/undetermined branch is always first-class** — the non-forcing
  fallback, and this project's STOP-equivalent clause.
- **Append-only:** commit before running; verdicts append below; no edits above
  the line.
- **Anti-tuning:** criteria fixed before submission, never adjusted post-hoc;
  report both floored and unfloored readouts; compute criteria mechanically.

**Exit artifact:** A committed, dated pre-registration file, in git *before* the
run starts.

---

## Stage 3 — Measure / refute

**Question:** Is the claim true, and is it even attributable?

**Entry criteria:** Stage 2 committed.

**Assets:** `results/campaign51_20260728/RUNBOOK_NEXT_SESSION_6.md` §1–§5 (the
Gates A–C pattern, model/effort policy, measurement-before-gate);
`/commission --research`.

**Procedure — Prime directive: "Refute before you build."** Do not start from
"the mechanism is X." Start from "is that true, and is it even attributable?"

| gate | content | rule |
|---|---|---|
| **A — provenance** | Regenerate any headline number whose source artifact no longer exists | Otherwise the claim stays unfalsifiable |
| **B — refutation** | Attack claims adversarially; survivors promoted CLAIM → FINDING and written back into the claim file | Adjudicate with one fable/xhigh agent, told that "refuted" and "undetermined" are acceptable, valued outputs |
| **C — alternatives** | Alternative-cause sweep before accepting any surviving mechanism | Prior failure mode: committing early to a plausible mechanism and burning ~1M tokens on it |

Strictly ordered. Do not skip forward.

**Measurement-before-gate:** a cheap haiku/low mechanical measurement that could
collapse the need for the expensive test — or decide a formula's shape and so
remove the need for a `/physics-change` gate at all — runs first.

**[A1] Free re-reads before compute.** Before requesting **any** new compute,
enumerate the diagnostics artifacts already on disk — `event_likelihoods.csv`,
gate readouts, run bookkeeping — and exhaust what they can already decide. They
are re-readable at zero marginal cost. Precedent: the gate-(vii) g_frac(h)
finding — the whole 2D MAP displacement collapses under frozen-g (0.780/0.800 →
0.66/0.64, per-event freeze, single-instrument adjudicated;
`adjudicate_g_frac.py`) — came out of an existing `event_likelihoods.csv` at
zero marginal compute, and was findable *before* the post-fix runs that were
commissioned to look for it.

**[A2] Paired per-event read alongside every class-summed comparison.** Whenever
a Σ-over-events statistic is compared across venues, configurations, or eras, a
paired/stratified **per-event** read is MANDATORY alongside the aggregate — not
optional, not deferred. Aggregates hide opposing sub-population effects that
cancel in the mean and manufacture spurious agreement. Precedent: gate (vii)'s
aggregate per-event tilts agreed across venues to 2.6% *by pure coincidence* —
scatter diluting shared-event tilts (×0.469) against 316 resurrected
dead-2D-leg events tilting 3.01× steeper and carrying 81% of the headline. A
D1-demotion conclusion was built on that agreement and had to be withdrawn when
the paired check — free, zero compute — refuted it.
Evidence: `results/run_20260804_postfix/gate_vii/paired_check.json`.

**Model & effort policy** (RUNBOOK-6 §2): haiku/low for mechanical extraction ·
sonnet/medium for bounded code tracing and literature checks against a cited
paper · opus/high for independent re-derivation and adversarial refutation ·
fable/xhigh for final adjudication and the physics decision. Cap the refutation
workflow at 6 agents. Never spawn opus/fable for what haiku can verify. Give
every agent the claim-file **path** and the exoneration list **verbatim** —
paraphrase drift caused two prior dead ends.

**Standing cross-session counterpart:** `/commission --research` — a
memory-bearing, falsification-first pass over the thread's accumulated claims:
report-delta → audit → reproduce → tournament → red-team → synthesis, with a
regression diff against `commission_history.jsonl` and typed feedback
(`request-change | suggest-change | suggest-direction | raise-consideration`).
Independence-enforced, propose-only. Run it at go/kill/pivot points.
**Prerequisite — TO SCAFFOLD:** it is a *user-level* skill
(`~/.claude/skills/commission/`), not committed with this repo, and RESEARCH mode
needs a `.commission-research.yaml` config, which this repo does **not** have yet.

### Ritual A6 — periodic assumption & performance audit

Distinct from Stage L (external-literature pull) and from `/commission
--research` (claim-history falsification): A6 is an **internal** re-check that
today's approximations and today's performance choices still hold under
today's premises — nothing here requires new literature or a live claim
thread. Cadence-driven, not decision-blocking.

**Cadence.** Every 2 completed campaigns, or 6 weeks of wall time, whichever
comes first. Owner: whoever holds the orchestration seat (Fable-tier session)
at the time the cadence fires; may delegate the checklist mechanics (§below)
to a `sonnet`/medium subagent, keep verdict authorship at orchestration tier.

**Trigger events (out-of-cycle).** See `PERF_ROADMAP`-adjacent proposal doc
`AUDIT_RITUAL_PROPOSAL.md` §4 for the canonical list; summarized: new
campaign kickoff, new venue/instrument, a `/physics-change` merge that
touches an approximation this ritual tracks, cluster config change (node
type, partition catalogue, packing rules).

**Procedure.** Run the shared checklist (below). Each item gets one of:
CURRENT (re-verified, no drift) / DRIFTED (re-verified, now stale — states
the delta) / UNCHECKED (not yet re-verified this cycle — never silently
skipped). File findings as an `AUDIT_<date>.md` next to the campaign it was
run against. DRIFTED items on approximation error budgets route to
`/physics-change`; DRIFTED items on perf choices route to a `PERF_ROADMAP.md`
update; DRIFTED items on the assumption register route to a
`docs/LITERATURE_WARNINGS.md` status update.

**Checklist.** Shared with Option B (routine prompt) verbatim — see
`AUDIT_RITUAL_PROPOSAL.md` §3, not duplicated here to avoid drift between two
copies of the same list.

**Exit artifact:** An `ADJUDICATION_<date>.md` with per-claim verdicts from the
fixed vocabulary — `FINDING` · `REFUTED` · `AMENDED` · `UNDETERMINED` — plus the
verdict appended below the pre-registration's line.

---

## Stage 4 — Calibration gate

**Question:** Is the full estimator calibrated at the production venue?

**Entry criteria:** Stage 3 produced a surviving mechanism, or a stop/continue
decision is pending. Per RUNBOOK-7 §1 item 5, this gate runs before any further
mechanism hunt beyond D1, and it is the explicit gatekeeper for "trusted run".

**Three legs, all required**

1. **SBC / P–P coverage** of the FULL two-channel estimator on truth-known
   synthetic universes at the production venue.
   Asset: `darksiren_emri/validation/pp_coverage.py` (`PPCoverageConfig`,
   `run_coverage`, CLI `uv run python -m darksiren_emri.validation.pp_coverage
   --kernel volume --mixture-mode ... --output pp_coverage_results.json`).
   Written from scratch by the 2026-07-01 commission and deliberately *not*
   importing production inference code — that independence is its scientific
   value.
   **GAP — TO BUILD:** the harness is currently single-channel (no σ_M / BH-mass
   dimension) and uses a synthetic smooth-density catalogue, not GLADE+. The
   **2-channel extension** and the **realistic host-observation model** (real
   n(z), genuine multi-galaxy impostor balls, mass observables, completeness-edge
   truncation) that this stage requires **do not exist yet**.

   **[A3] Harness acceptance criteria** — the extension is not accepted, and its
   coverage verdict does not count, unless all three hold:
   (i) **genuinely 2-channel**, with the completion-leg mass factor **g recomputed
   per h** — never frozen across the h grid, never elided;
   (ii) **run at production N** — the mechanism is N-coherent (per-event sub-threshold; event-summed ḡ(h) Δln ≈ 0.048 grid-wide)
   and is invisible at small N;
   (iii) **multi-candidate host balls** — a one-candidate-per-event harness
   structurally cannot exercise the mechanism (cf. `BIAS_HISTORY_LEDGER.md` §1
   row 86).
   Rationale: the residual 2D displacement is carried by an event-independent
   per-h scalar — exactly the class small-N SBC cannot see.
2. **Generator-closure absolute-count audit** — the (ii-d)-style check.
   Defined `.planning/derivation-2dbias-fix-20260803/GATE_PACKAGE_FINAL.md` §2.6;
   executed and closed in
   `.planning/derivation-2dbias-fix-20260803/FIXB_PATHA_PACKAGE.md` §0–§1.
   (ii-d) is the **absolute detected-count check** (D̃/D = 0.926 ⇒ −7.4% N_det):
   it compares the detected count the estimator's selection/completion objects
   predict against the realized total, and its diagnostic job is to **separate
   α_G-too-big from β_Ḡ-too-small** — which the gate-(ii) ratio test (w̃_G vs
   164/3135) cannot do on its own. It was closed by the FIXB §1 attribution
   decomposition into named factors (waveform validity REFUTED; p0-window
   retention CONFIRMED, dominant ×1.342; host-mass noise mechanism confirmed,
   remedy refuted).
   **Why it is not optional:** a filter applied by the generator but unmodeled by
   the estimator is silently shared by both legs of an SBC test, so it cancels
   out of the coverage statistic. SBC alone cannot catch the D1 class of defect;
   this audit checks an external invariant that is sensitive to exactly those
   filters. It is what caught the p0-window rejecting 69.3% of SNR-passers.
3. **Forecast-consistent width** — the measured width must sit on the stage-1 F5
   forecast for the venue's (σ_z, σ_M, N).

**Exit artifact:** A calibration readout stating, per leg, PASS/FAIL with
numbers: coverage (50/68/90% HPD) + MAP bias; count-audit residual ratio ± its
uncertainty; measured-vs-forecast width.

---

## Stage 5 — Decision

**Question:** Measure, bound, fix, or one more measurement?

**Entry criteria:** Stage 4 readout complete on all three legs.

**Stop/continue rule of record (author-endorsed 2026-08-04).** The per-event
ln-posterior min/max range is a **screen, never a stopping gate** — N coherent
sub-threshold tilts dominate the ensemble (measured: per-event 0.3–0.5σ rails vs
+3.4–6.1σ class-summed). **"Stop digging" requires all of:** coverage pass **and**
width on the F5 forecast **and** no unmodeled selection between generator and
estimator. SBC alone cannot catch a filter both sides silently share (the D1
class); the absolute-count audit is the complement.

| verdict | action |
|---|---|
| **CALIBRATED + narrow** | measure — report the H₀ measurement |
| **CALIBRATED + wide (≈ forecast)** | **stop digging, report the bound** |
| **DEFECT** — ≥3σ coherent class displacement, or coverage failure | fix via `/physics-change` |
| **UNDETERMINED** | identify and run the *one* measurement that decides; return to stage 2 |

**Routing.** Any fix touching a formula, constant, waveform parameter, frequency
limit, PSD coefficient, or cosmological/galaxy model in a trigger file is a hard
`/physics-change` gate: the 5-item package (old formula with file:line, new
formula, reference, dimensional analysis, limiting case) → **STOP for explicit
author approval** → implement → verify, with a ledger row per step. Fixes tagged
`instrumentation` stay in plain GSD. Each formula fix carries its own
pre-registered acceptance/limiting-case criterion (e.g. "σ_z → 0 must reproduce
the point kernel exactly").

**Exit artifact:** A dated decision paragraph in the claim file plus, where
applicable, the `presented` row in `docs/gates/PHYSICS-GATE-LEDGER.md`.

---

## Stage 6 — Chronicle

**Question:** What must the next session be able to find without re-deriving it?

**Entry criteria:** Stage 5 decided.

**Duties**
1. **Gate ledger.** `docs/gates/PHYSICS-GATE-LEDGER.md`, append-only, never
   back-filled. Row: `| YYYY-MM-DD | <commit-ref> | <step> | <verdict> |
   <target> | <note> |`; step ∈ {`presented`, `implemented`, `verified`};
   verdict ∈ {`APPROVED`, `REJECTED`, `PASS`, `FAIL`, `WAIVED`}. A complete gate
   run is three rows sharing a target; a REJECTED run is one row. A `[PHYSICS]`
   commit with no ledger row is a gate that cannot be shown to have run.
2. **Claim-file writeback** — the two-file pattern: the separate
   `ADJUDICATION_<date>.md` first, then hand-applied edits into the living claim
   file. Strike superseded text with `~~...~~`, append a bold dated verdict
   (`**[date, §N adjudication] → VERDICT**`), and retag provenance in place
   (`~~[AGENT, NOT REPRODUCED]~~ **[LOCAL, VERIFIED]**`). Never fork a new claim
   file for a verdict on an old claim.
3. **Ledger propagation.** New verdict → a §1 chronological row with file:line
   citation. Newly refuted mechanism → also a §2 ⚠ item, and the local claim
   file's Exonerated list. Bears on an existing claim → update §3. New unresolved
   thread → §4.
4. **Next runbook.** Write `RUNBOOK_NEXT_SESSION_<N+1>.md` in the campaign dir
   with the lineage conventions: `Supersedes <prior runbook>` naming what became
   DONE; `## 0. State of the physics`; `## 1. Task queue` (ordered, actionable,
   with model/effort hints like `[haiku/low compute, opus/high interpret]`);
   `## Gotchas` marking which prior gotchas they supersede; `## Author decisions
   open`.
5. **GitHub sync** per `CLAUDE.md`: close resolved issues with a commit/phase
   reference, open issues for newly discovered bugs with the right labels.

**Exit artifact:** ledger rows + updated claim file + updated
`BIAS_HISTORY_LEDGER.md` + the next runbook.

---

## Stage L — external consult (cross-cutting)

**Not a numbered stage.** Stage L is invocable *from* stage 0 (intake) and *from*
stage 5 (decision); the 0–6 pipeline is unchanged and is **not** renumbered.

**Question:** Has the field already documented this failure mode — and if so, where
exactly, with what validity conditions, and did we heed it?

**Why it exists.** On 2026-08-05 the decisive external input for the Hitchhiker
thread (arXiv:2212.08694 §2.3, the paragraph after Eq. 30 — the selection
denominator's perfect-z condition) came from **author memory**, not from the cycle,
and the paper was **already cited in this repo**: the adjacent passage sits quoted in
`docs/BIAS_RESOLUTION_ATTEMPTS_REPORT.md:174-179`. *Already-cited ≠ already-heeded.*
Stage L makes the consult a procedure instead of a recollection.

### Entry triggers (all four are binding)

| # | trigger | weight |
|---|---|---|
| L-a | **Every new mechanism thread's stage 0** | MANDATORY lightweight **R0 sweep** (see rings) — no full Stage L required, but the sweep is part of intake completeness |
| L-b | **STUCK SIGNAL** — two consecutive `MIXED`/`UNDETERMINED` verdicts on the same thread | auto-triggers a **full** Stage L before the third measurement is designed |
| L-c | **Before any `/physics-change` ADOPTION** | does the literature document *this fix's* failure modes? A fix with an undocumented-in-our-notes failure mode is not gate-ready |
| L-d | **Each runbook lineage rollover** (`RUNBOOK_NEXT_SESSION_<N+1>`) | lightweight pass — refresh `docs/LITERATURE_WARNINGS.md` statuses against the session's verdicts |

### Procedure

**1. Symptom card.** State the problem *as observed signatures* — the numbers, the
venues, what survived which controls — and then restate it **up an abstraction
ladder**, each rung with its own search vocabulary:

| rung | example (the 2026-08-05 case) |
|---|---|
| field-specific | "dark-siren H₀ rails high with photo-z hosts" |
| method class | "selection effects in hierarchical Bayesian population inference" |
| math class | "measure / normalization mismatch under data reduction; latent-variable marginalisation coupling" |
| generic inference pathology | "per-item independence assumed where a shared latent breaks separability" |

A card that only carries the field-specific rung will only find papers we already know.

**2. Rings, searched in order. R0 first, always.**

| ring | contents | the specific reading instruction |
|---|---|---|
| **R0** | papers **already cited in this repo** | re-read them **for warnings**: caveat/limitation sections, validity conditions attached to the equations we imported, "inconsistency" catalogues, footnotes. Not for the equations — we already took those |
| **R1** | citation neighborhood of R0 | especially **forward** citations, filtered on *bias · inconsistent · caveat · erratum · corrigendum*; plus mock-data-challenge and code-comparison papers, which exist precisely to catalogue these failures |
| **R2** | the field | the broader dark-siren / GW-cosmology literature on the method class |
| **R3** | the math class | astrostatistics, selection-effect primers, simulation-based calibration (SBC) |
| **R4** | code / numerics | implementations, issue trackers, release notes of the codes that solve our problem |

**3. Independence.** The searcher receives the **symptom card only** — **never** the
current suspect list, and never "we think it's X". This mirrors the commission's
independence rule: a searcher told the suspect finds confirmation for the suspect.
Timebox the search and say what the box was.

**4. Intake.** Each candidate enters **stage 0** as a claim with provenance tag
**`[LIT]`**, carrying the full citation *with section and equation numbers*. Then, in
order:

1. **Quote-verification before mapping — MANDATORY.** Pull the passage verbatim
   before mapping it onto our pipeline. The 2026-08-05 lesson: without it, S1 and S2
   were treated as two statements when they are two ends of *one*, and a genuinely
   distinct third statement (P1) was missed entirely.
2. The standard **two-layer exoneration check** (stage 0 procedure step 1).
3. The **cheapest decisive measurement** (`Refute by:`), per stage-0 rule 4.

**Exit artifacts**
- Candidates **ranked by signature-match × cost-to-test**, filed as stage-0 claims and
  feeding stage-2 pre-registrations; **or**
- a documented **"the field has no answer"** — which is itself a reportable result, and
  is stated as such rather than left as a silent absence; **and**
- updated rows in **`docs/LITERATURE_WARNINGS.md`** (below).

### The prevention half — assumption register

Stage L is reactive; these two provisions make it preventive.

**(i) The `/physics-change` gate package gains a SIXTH item.** In addition to the five
(old formula with file:line · new formula · reference · dimensional analysis ·
limiting case), a gate package must state **the source equations' stated validity
conditions, each checked against our venue/regime.** Motivating case: Eq. 15's
perfect-z condition was never registered when the per-event form was imported; M-1
later showed it violated on *every* venue (ledger row 95). Approved 2026-08-05,
**pending application** to `.claude/skills/physics-change/SKILL.md` — that edit belongs
to a physics-change-owned commit and is tracked as a TODO in amendment row A5.

**(ii) `docs/LITERATURE_WARNINGS.md`** — the register itself: the field's documented
pitfalls mapped onto our status (`CHECKED` / `VIOLATED` / `UNDER MEASUREMENT` / `OPEN` /
`N-A` / `UNCHECKED`), every row citing evidence or honestly saying `UNCHECKED`. Rows are
written at Stage L intake (ring R0) and at gate item 6. Seeded 2026-08-05 with the
arXiv:2212.08694 section.

---

## Amendment ledger

The cycle governs itself by its own discipline: **every future change to the
Research Cycle — this file or `.claude/skills/research-cycle/SKILL.md` — adds a
row here. Append-only, never back-filled, no silent edits.** An amendment
without a row is a change that cannot be shown to have been earned.

| date | amendment | stage | what changed | why (one clause) | evidence |
|---|---|---|---|---|---|
| 2026-08-04 | A0 — establishment | all | Cycle established: 7 stages wired to existing assets; `/research-cycle` made the entry point for every investigation | author mandate to stop reinventing a runbook per investigation | `results/campaign51_20260728/RUNBOOK_NEXT_SESSION_7.md` §2; commit `f8a01b04` |
| 2026-08-04 | A1 — free re-reads before compute | 3 | New hard rule: exhaust re-reads of on-disk diagnostics artifacts before requesting ANY new compute | the g_frac(h) finding (2D MAP displacement collapses 0.780/0.800 → 0.66/0.64 under per-event frozen-g) came at zero marginal compute from an existing `event_likelihoods.csv`, and was findable before the post-fix runs | `results/run_20260804_postfix/gate_vii/gate_vii_readout.json`, `.../adjudicate_g_frac.py` |
| 2026-08-04 | A2 — paired read with every class-summed comparison | 3 | New hard rule: any Σ-over-events statistic compared across venues/configs/eras requires a paired/stratified per-event read alongside the aggregate | gate (vii)'s aggregate tilts agreed across venues to 2.6% by pure coincidence — scatter diluting shared-event tilts ×0.469 vs 316 resurrected dead-2D-leg events tilting 3.01× steeper (81% of the headline) cancelled in the mean, and a D1-demotion conclusion built on the agreement had to be withdrawn | `results/run_20260804_postfix/gate_vii/paired_check.json`, `.../paired_check.py` |
| 2026-08-04 | A3 — harness acceptance criteria | 4 | TO-BUILD `pp_coverage.py` extension now carries three acceptance criteria: 2-channel with g recomputed per h; production N; multi-candidate host balls | the residual 2D displacement rides the completion-leg mass factor's h-slope (event-summed ḡ(h) Δln ≈ 0.048 grid-wide; per-event sub-threshold, ensemble-dominant) — exactly the N-coherent class small-N, single-candidate SBC structurally cannot see | `results/run_20260804_postfix/gate_vii/gate_vii_readout.json`; `BIAS_HISTORY_LEDGER.md` §1 row 86 |
| 2026-08-04 | A4 — evidence correction on A1/A3 | 3, 4 | A1/A3 precedent numbers corrected after adjudication: g_frac is NOT a per-h near-scalar (1587 distinct per-event values at h=0.73, range 0.076–0.242); frozen-g 2D MAPs are 0.66/0.64 (not 0.700); event-summed ḡ(h) 0.1348→0.1413, bit-identical both venues | two subagents disagreed on the numbers; a deciding single-instrument run refuted the interpreter's near-scalar claim while STRENGTHENING the qualitative finding (larger collapse, overshoots below injected 0.73) | `results/run_20260804_postfix/gate_vii/adjudicate_g_frac.py`, `.../viz_data.json` |
| 2026-08-05 | A5 — Stage L: external consult | 0, 5, cross-cutting + `/physics-change` gate | New cross-cutting **Stage L** (symptom card up an abstraction ladder → rings R0…R4, R0 = already-cited papers re-read for warnings → independence-preserving timeboxed search → `[LIT]`-tagged stage-0 intake with mandatory quote-verification-before-mapping), four entry triggers (mandatory R0 sweep at every stage 0; auto-trigger on two consecutive MIXED/UNDETERMINED; before any `/physics-change` adoption; lightweight at runbook rollover); `[LIT]` added to the tag vocabulary; the `/physics-change` gate package extended to a **6th item** (source equations' stated validity conditions, checked per venue) — approved, pending application; new register `docs/LITERATURE_WARNINGS.md`. Pipeline **not** renumbered. **TODO (physics-change-owned commit): add gate item 6 to `.claude/skills/physics-change/SKILL.md`.** | the decisive external input on the Hitchhiker thread sat in an **already-cited** paper (arXiv:2212.08694 §2.3, after Eq. 30) quoted in this repo for weeks and was surfaced by author memory, not by the cycle — already-cited ≠ already-heeded | `results/campaign51_20260728/realistic_20260729/CLAIM_HITCHHIKER_INDEPENDENCE_20260805.md.DRAFT`; `gate_b_20260730/BIAS_HISTORY_LEDGER.md` §1 row 95; `results/campaign51_20260728/RUNBOOK_NEXT_SESSION_8.md` §5; `docs/BIAS_RESOLUTION_ATTEMPTS_REPORT.md:174-179` |
| 2026-08-12 | A6 — periodic assumption & performance audit | 1, 3, 6, cross-cutting | New recurring ritual, cadence and trigger-gated (not stage-blocking): a standing **Assumption & Performance Audit** re-validates (a) approximation error budgets (interpolation tolerances, surrogate forms e.g. kappa_cap/p0 surrogates in `emri_rate.py`), (b) perf choices vs current unit economics (CPU-h/seed anchors, contention factors, packing rules), (c) `docs/LITERATURE_WARNINGS.md` register entries for staleness. Runs on a cadence (recommend: every 2 campaigns or 6 weeks, whichever is sooner) OR on a trigger event (§4). Produces an AUDIT_<date>.md under the active campaign's results dir; does not gate stage 3/5 decisions on its own — findings route to `/physics-change` (formula drift) or a perf-roadmap update (perf drift) as appropriate. | author mandate 2026-08-12: approximations and perf choices earn re-validation on a schedule, not only when a symptom forces the question — [[realistic-venue-performance-goal]] names realistic-venue infra as reusable, so its assumptions must stay current for follow-on projects too | `results/campaign51_20260728/RUNBOOK_NEXT_SESSION_9.md` §2 item 4; `results/venue_transfer_20260811/perf/PERF_ROADMAP.md` (the class of finding this ritual re-validates) |
| 2026-08-13 | A7 — campaign readout report | 4→5 boundary, 6 | Every campaign that reaches a readout now also produces a **Campaign Readout Report** (`docs/templates/CAMPAIGN_READOUT_REPORT.md`) — a comprehension-first artifact written after the mechanical readout and the adversarial adjudication, and *before* the stage-5 decision, carrying: the question asked, the arms and the control, the per-trial distribution against truth, the mechanism/dose check, the scorecard vs locked bands, a mandatory vocabulary section, the validity + independent-recompute basis, the adjudicator's non-branch-impacting flags, and the numbered decision table. Markdown in the campaign results dir is the record; a rendered page is a presentation layer, never a substitute. Binding rule carried in: the report presents the fired branch and never adjudicates. | author signal 2026-08-13 after the venue-transfer readout — "these kind of reports is exactly what I need after a research cycle has run, because a lot of things happened and this clearly shows me the whole picture"; the existing readout/adjudication artifacts are built for re-derivation and the ledger for the record, so nothing in the cycle was built for *understanding* the campaign, which is what the stage-5 decision actually requires | `docs/templates/CAMPAIGN_READOUT_REPORT.md`; worked example = venue-transfer readout `results/venue_transfer_20260811/` + ledger row #99; the CLAUDE.md "Proposing decisions" rule (decision tables live in reviewable artifacts, not chat) |
| 2026-08-14 | **A8 — branch-referent, two-sidedness & band-derivation checks — PROPOSED, PENDING AUTHOR APPROVAL** | 2 | *(NOT ADOPTED — drafted for author ruling; no registered document under `results/` is edited by it.)* Three registration-time checks added to the stage-2 skeleton, each a line the pre-registration must carry before it is committed. **(1) Branch-referent check [PROPOSED BLOCKING]:** for every branch, name the arm(s) whose data can satisfy its condition and state, per arm, that it can support the branch's stated *meaning*. A branch whose meaning asserts something no arm in the design can establish is a **drafting error to be fixed at registration, not a finding to be disclosed at readout**. **(2) Two-sidedness check [PROPOSED BLOCKING, satisfiable by one stated sentence]:** where the hypothesis a rule names makes a POINT prediction, the rule must be two-sided — an upper edge as well as a lower — or the file must state why a one-sided rule is adequate for that hypothesis. A one-sided threshold cannot distinguish *consistent with H* from *far beyond H*, and will happily fire the branch that names H on data that refutes H. **(3) Band-derivation disclosure [PROPOSED NON-BLOCKING, in A6's spirit]:** every validity/decision threshold states its derivation and its implied false-fail probability under the exact null **at that arm's own N**. Registration is not blocked on the *value* — the author may knowingly accept an under-powered check — only on its *absence*; an asserted round number with no stated false-fail rate is a latent false-fail generator. Anti-tuning is untouched: all three are pre-data drafting duties, never grounds to move a band after a readout. **Why (1) and (2) are proposed blocking and (3) is not:** (1) and (2) are pure logic and arithmetic against the file's own text, cost minutes, and their failure mode is a *whole campaign* whose headline conclusion cannot be quoted — the mechanism-isolation branch was unsatisfiable-in-meaning from the moment it was written, contradicted by its own §2 two sections above it; (3) needs a number the author may legitimately trade against budget, so it is a mandatory disclosure rather than a veto. | two independently drafted stage-2 trees in one thread each fired a branch whose condition was met and whose registered *meaning* the data did not support — the cycle validates a pre-registration's arithmetic and validates its interpretive clauses not at all | `docs/gates/BRANCH_REFERENT_FAULT_20260814.md` (rationale note, also PROPOSED); scan: `results/mechanism_study_20260813/PREREGISTRATION_2D_DOSE_SCAN.md` §4.3 + verdict (DS-D3 one-sided, `b(S23) = +0.023650` fires SHAPE-INTERACTION at +28.2 realized SE above the boundary while sitting **+10.33σ above H-INT's own point prediction 0.017333** on the registered SE; branch 2 fired, meaning BARRED from being quoted), `SCAN_READOUT.md` §3.2/§5; isolation: `PREREGISTRATION_MECHANISM_ISOLATION.md` §2/§4 + `ARMS.md` (branch 2 SINGLE-OWNER fires on **MEI (E1-imp)**, registered as a *zero-estimator-change, generator-side* arm — *"the estimator is byte-identical across N-0, E1-host and E1-imp"* — so its "that term is the identified mechanism" clause **has no referent**), `MECHANISM_ISOLATION_READOUT.md` D-M-3 + §9 items 1 and 5; band: `AMENDMENT_A1_VM1_NULL_AT_N100.md` §3 (±0.002 asserted, not derived; **~21 % false-fail under the exact null at N = 15**; false-failed on a 1.611σ fluctuation, remedied by N = 100 at the same unchanged window) |

**A8 — ADOPTED AS REVISED, 2026-08-14 (author ruling, ledger row #102).** Revisions relative to the 2026-08-14 draft, per the commission review (`results/commission_research_20260814/REPORT.md`): (a) Instance 2's predicate is corrected — the parent's branch 2 was NOT unsatisfiable-in-meaning from registration (A-M2′ was a registered estimator-side, referent-bearing arm); the actual fault was readout-time. (b) The branch-referent check is kept BLOCKING: every branch names, at registration, the registered arm(s) that can satisfy it and what each ablates. (c) The two-sidedness check is kept BLOCKING: any rule that names a point prediction must bound both sides. (d) NEW, BLOCKING — **execution-completeness**: no count-based branch may be adjudicated while a registered arm capable of changing the count is unrun; the adjudication is deferred, or the arm is first withdrawn by an author [RULE]. (e) Band-derivation disclosure stays NON-BLOCKING: each band states its derivation and its false-fail rate at the arm's own N. Scope: all future pre-registrations in this thread and its successors; lapses only on author revocation.

**A9 — PROPOSED, NOT ADOPTED (2026-08-14).** Evidenced by Amendment A1's buy-more-seeds-after-a-fired-fail pattern (legitimate in that instance: window unchanged, FAIL reading pre-committed, unfavourable seeds included — but exploitable in general): any design permitting sample-size escalation after a fired validity rule must pre-register its sequential rule (escalation trigger, maximum N, and the decision statistic's correction, e.g. alpha-spending or a pre-committed two-stage read). Awaits author scope-setting.

**A10 — ADOPTED, 2026-08-20 (author ruling: "I agree with the amendments").** Invariance & blindness declaration. Every pre-registration lists (a) the invariants held FIXED across all arms, each with the date it was last derivation-audited or `NEVER`, and (b) one sentence naming the defect class the design cannot detect by construction. A campaign carrying a `NEVER`-audited load-bearing invariant either audits one in the same cycle or states in its VERDICT that its conclusions are conditional on the named invariants. Evidence: the un-derived `B_scale` factor (worth +0.12 in the posterior mean, 2× the bias under study) survived four counterfactual campaigns because every arm held it fixed — differential designs are blind to their own invariants. Full rationale: `.claude/skills/research-cycle/PROPOSED_AMENDMENTS_20260820.md`.

**A11 — ADOPTED, 2026-08-20 (same ruling).** Provenance freshness. Any quantity quoted into a budget, band, or verdict carries `{value, source, date, configuration-of-record}`; a stamp whose configuration no longer matches is STALE and may not be quoted as a point value — re-measure it or carry it as a disclosed band. Attached harness rule (now in `CLAUDE.md`): every multi-GB input outside version control carries a checksum pin at each consumer, STOP-gated. Evidence: `s_Edd = −0.020` was frozen from a code comment into a prereg and a ratified decision and proved wrong by an order of magnitude AND sign (`+0.0012` re-measured), six weeks after the repo's own gate record flagged it; separately, a stale local galaxy catalogue silently fed every local analysis until a fidelity gate caught it.

**A12 — ADOPTED, 2026-08-20 (same ruling).** The score-zero test is the standing FIRST diagnostic on any estimator-bias question, before instruments are built: for data from the estimator's own model, `E[∂_θ ln L]` at truth is zero, so a non-zero per-event mean score at truth localizes a misnormalization — and the same read resolved by event class and by covariate localizes it further, on banked data at zero compute. Evidence: this is what cracked the base-tilt front (`−0.635 ± 0.017`, 37σ, on the dark class; class- and z-resolved for free) after three fronts of instrument-building had not.

**A13 — ADOPTED, 2026-08-20 (same ruling).** Engagement gate. No null from a counterfactual instrument is interpretable until the instrument is shown to change the output: a registered engagement threshold, scored and STOP-gated, plus an assertion that the switch reaches every dispatch path production actually uses and that each labelled arm carries its intended value at runtime. Evidence: production dispatches through a batch kernel — an instrument patched only into the scalar kernel would have passed bit-identity and continuity gates and returned a confident, wrong "no effect".

**A14 — ADOPTED, 2026-08-20 (same ruling).** An attribution ships with its own falsifier: any memo or verdict attributing an effect to a cause registers, in the same document and before the attribution is banked, the experiment that would falsify it and its bands; if unrun, the attribution is explicitly provisional. Evidence, both directions: the population-misspecification memo registered its falsifier in §7, the falsifier ran, and the attribution was downgraded cleanly; the earlier impostor-overlap reading shipped without one and had to be undercut by a later measurement.

**A15 — ADOPTED, 2026-08-20 (author ruling: "please continue, approved", on the decision table of `docs/derivations/GATE_PRESENTATION_SENTINEL_COMBINE_20260820.md` §6 item 8).** Power-calibrated gates, demonstrably-sensitive controls. No scored threshold may be registered without its operating characteristics at the ACTUAL N it will run at: state the statistic's null distribution, the false-fail rate of the chosen band, and the effect size the band can detect at reasonable power. A band lying inside the null's expected fluctuation is void on its face, and a threshold imported from another measurement is invalid unless its N and statistic match. Symmetrically, **no null/positive control counts as a control until it is shown to be informative** — a control must be demonstrated capable of failing; a flat likelihood is not a pass, and neither is a comparison whose two sides are provably identical before the run. *Evidence:* D-1's 0.05 max-CDF-gap band false-fails 58% of the time at n = 174 (E[D_null] = 0.0659) and was imported from an n = 1588 check where the same number was too loose; the mirror's B-F1/G-1 unity-completeness control returned a perfectly flat log-posterior whose "0.7300, truth to four decimals" was the arithmetic midpoint of the h-grid, and it had gated every arm since G-0; and, in the same cycle, the orchestrator's own A-7 draft twice reproduced the failure this amendment forbids — a band of `max(0.005, 2·SE)` applied to a PAIRED deterministic recomputation whose sampling variance is exactly zero, and a "BAND O" whose two branches were both pre-determined by algebra and by the data range, withdrawn as undecidable. *Relation to existing rules:* A13 requires an INSTRUMENT to be shown to move the output; A15 is the same demand one level up, applied to the GATES and CONTROLS that adjudicate instruments. *Corollary adopted with it:* an arm whose repair is provably a no-op before the run is labelled NULL-BY-CONSTRUCTION and carries no verdict. *Cost:* minutes per gate — one null-distribution calculation and one control-sensitivity check.

**A16 — ADOPTED, 2026-08-21 (author mandate, verbatim: "Once a research cylce ends or something
fails I want you to put this in a retrospective ledger or a comparable ledger that tracks and
suggests amendements. I also want you to add a tracker to the existing amendements that get +1 each
time they contribute meaningfully. you can also implement new amendements.").** Two standing
instruments, both append-only: **(a) `docs/RETROSPECTIVE_LEDGER.md`** — one entry per research-cycle
end AND per failure (fired gate, vacuous control, NOT-READY registration, withdrawn claim, wasted
compute, operational loss), carrying what-ran / what-worked / what-failed / suggested-amendments /
disposition; suggested amendments are born there and bind here. **(b)
`docs/AMENDMENT_IMPACT_TRACKER.md`** — a per-amendment counter incremented +1 for each *meaningful*
contribution (changed a design pre-mistake, caught a defect, forced a material disclosure, saved
compute — routine compliance does not count), every +1 backed by a dated evidence line in the same
commit. The mandate's final sentence is read as a **[STANDING]** grant to implement new amendments;
under the binding default (CLAUDE.md, approval-scope rule), any amendment adopted under this
standing is tagged `ADOPTED-UNDER-STANDING` and listed for author review at the next session — the
author may revoke, at which point it reverts to PROPOSED. *Itemisation (a)/(b) and the standing
reading are orchestrator-derived from the quoted words.*

**A17 — ADOPTED, 2026-08-21 (author ruling: "The updated onces are approved", on row #152's
restated card 5; ledger row #153).** Gate/band portability. No scored threshold may be moved to a
new statistic, channel, or venue without re-deriving its operating characteristics against
known-informative reference data **in the same commit**; and every gate/band re-states its
operating characteristics on the **realized** scatter at readout, not only on pilot/launch
estimates. *Evidence:* GATE V's v2 numbers ported to the matched channel false-failed 5/12
reference seeds and STOPped 3/4 pilot seeds (retrospective entry 1); the C-SG N-adequacy gate
passed at 7.76σ on pilot σ̂ but realizes 4.98σ (< its registered 5) on fleet scatter (review
MAJOR-2). Extends A15 one level: A15 demands operating characteristics exist; A17 demands they
survive transport and be re-checked against reality.

**A18 — ADOPTED, 2026-08-21 (same ruling).** Explicit bias reference. Every readout scorer prints,
per arm and per statistic, the reference value each "bias"/"error" subtracts, as a machine-readable
field — a wrong implicit reference is a silent FATAL. *Evidence:* review FATAL-1 — the C-SG fleet
scorer subtracted the global 0.73 for the δ arms; the corrupted numbers reached ledger row #151 and
the readout report before an independent review caught them.

**A19 — ADOPTED, 2026-08-21 (same ruling).** Symmetric falsifiers. A pre-registration arms a
registered falsifier for **every** branch its bands can fire, not only the branch the designer
expects; a branch without a falsifier may fire but its claim stays PROVISIONAL until one is
registered and run. *Evidence:* review MAJOR-9 — C-SG's §8 falsifier covered only
ESTIMATOR-SELF-CONSISTENT; INTERNAL-DEFECT fired with none, and pre-check O4 is its retrofit.
