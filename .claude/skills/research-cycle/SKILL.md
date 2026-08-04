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
disable-model-invocation: true
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
| 4 | Calibration gate | is the estimator calibrated at the production venue? | three legs: `master_thesis_code/validation/pp_coverage.py` + the (ii-d) absolute detected-count audit + forecast-consistent width |
| 5 | Decision | measure / report bound / fix / one more measurement | `/physics-change`, `docs/gates/PHYSICS-GATE-LEDGER.md`, author gate |
| 6 | Chronicle | ledger rows, claim writebacks, next runbook | `docs/gates/PHYSICS-GATE-LEDGER.md`, `results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`, `results/campaign51_20260728/RUNBOOK_NEXT_SESSION_<N>.md` lineage |

`--stage N` enters mid-cycle; without it, start at 0. Stages are ordered — do
not skip forward, and say so explicitly when you deliberately re-enter earlier.

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
   committed artifact · `[INFER]` inference from `[LOCAL]`/`[DOC]`. Reproduce
   before you use.
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
  GLADE-like host model that stage 4 requires are **TO-BUILD**.
- `/commission --research` is a **user-level** skill (`~/.claude/skills/commission/`),
  not committed with this repo, and RESEARCH mode needs a `.commission-research.yaml`
  config that this repo does not have. **TO-SCAFFOLD** before first use.

### Cross-session counterpart

`/commission --research` is the standing, memory-bearing falsification pass over
a thread's accumulated claims (delta → falsify → claim-history regression diff →
typed feedback). It is stage 3's periodic counterpart to the in-session Gates
A–C, and it should be run at go/kill/pivot points, i.e. alongside stage 5.
Propose-only; it never lands fixes.
