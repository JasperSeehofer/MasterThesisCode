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
finding — ~83% of the dark 2D-vs-1D up-tilt, and the whole 2D MAP displacement
(0.78/0.80 → 0.700 under frozen-g) — came out of an existing
`event_likelihoods.csv` at zero marginal compute, and was findable *before* the
post-fix runs that were commissioned to look for it.

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
   Asset: `master_thesis_code/validation/pp_coverage.py` (`PPCoverageConfig`,
   `run_coverage`, CLI `uv run python -m master_thesis_code.validation.pp_coverage
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
   (ii) **run at production N** — the mechanism is N-coherent at 0.019 nats/event
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

## Amendment ledger

The cycle governs itself by its own discipline: **every future change to the
Research Cycle — this file or `.claude/skills/research-cycle/SKILL.md` — adds a
row here. Append-only, never back-filled, no silent edits.** An amendment
without a row is a change that cannot be shown to have been earned.

| date | amendment | stage | what changed | why (one clause) | evidence |
|---|---|---|---|---|---|
| 2026-08-04 | A0 — establishment | all | Cycle established: 7 stages wired to existing assets; `/research-cycle` made the entry point for every investigation | author mandate to stop reinventing a runbook per investigation | `results/campaign51_20260728/RUNBOOK_NEXT_SESSION_7.md` §2; commit `f8a01b04` |
| 2026-08-04 | A1 — free re-reads before compute | 3 | New hard rule: exhaust re-reads of on-disk diagnostics artifacts before requesting ANY new compute | the g_frac(h) finding (~83% of the dark 2D-vs-1D up-tilt; whole 2D MAP displacement 0.78/0.80 → 0.700 under frozen-g) came at zero marginal compute from an existing `event_likelihoods.csv`, and was findable before the post-fix runs | `results/run_20260804_postfix/gate_vii/gate_vii_readout.json`, `.../compute_gate_vii.py` |
| 2026-08-04 | A2 — paired read with every class-summed comparison | 3 | New hard rule: any Σ-over-events statistic compared across venues/configs/eras requires a paired/stratified per-event read alongside the aggregate | gate (vii)'s aggregate tilts agreed across venues to 2.6% by pure coincidence — scatter diluting shared-event tilts ×0.469 vs 316 resurrected dead-2D-leg events tilting 3.01× steeper (81% of the headline) cancelled in the mean, and a D1-demotion conclusion built on the agreement had to be withdrawn | `results/run_20260804_postfix/gate_vii/paired_check.json`, `.../paired_check.py` |
| 2026-08-04 | A3 — harness acceptance criteria | 4 | TO-BUILD `pp_coverage.py` extension now carries three acceptance criteria: 2-channel with g recomputed per h; production N; multi-candidate host balls | the residual 2D displacement is an event-independent per-h scalar (N-coherent, 0.019 nats/event) — exactly the class small-N, single-candidate SBC structurally cannot see | `results/run_20260804_postfix/gate_vii/gate_vii_readout.json`; `BIAS_HISTORY_LEDGER.md` §1 row 86 |
