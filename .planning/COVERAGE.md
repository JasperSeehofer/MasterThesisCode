---
title: "Coverage Map — MasterThesisCode"
schema: coverage-map/v1
seeded: 2026-07-12 (advisory tracer run — hand-seeded, pre-Phase-2; NOT via /project-init or /gardener --coverage)
last_full_audit: 2026-07-12 (partial — sim/eval area only; full 6-step audit pending gardener --coverage retrofit)
blind_spot: "Covers named task-areas only. A class absent from the inventory is
  invisible; the inventory is re-tested at every full audit (commit- and
  incident-classification), and any unclassifiable incident files a WATCH row.
  WATCH latency is bounded by full-audit cadence, not drift-check cadence. This
  seed populated ONE finding (the sim/eval divergence class); the other 6
  task-areas are inventory-only and their owners are not yet audit-verified."
---

## Task-areas  (5–9 rows, MECE; changes to THIS table are propose-class)

Inventory transcribed from [[orbiter-upgrade-design]] C.6 Step-1 output (7 areas).
Owners are the C.6-documented state; only Area 2's UNCOVERED status is (re)confirmed
by this pass. The other rows are carried forward, not independently re-audited.

| # | Task-area | Owner (artifact) | Rung | Trigger set (globs/events/keywords) | Constitution pointer | Coverage evidence | Review-by | Status |
|---|-----------|------------------|------|-------------------------------------|----------------------|-------------------|-----------|--------|
| 1 | Repo+cluster interaction (bwUniCluster preflight/submit/retrieve; don't-cancel) | `cluster` skill (`disable-model-invocation`) | skill | `cluster/**`, `*.sbatch`, submit/scancel/sacct events | `cluster` SKILL runbook | COVERED | 2026-10 | COVERED |
| 2 | **Sim/eval convention consistency** (frames, units, redshift/mass conventions, likelihood structure across sim↔eval) | **manifest floor (`CONVENTIONS-MANIFEST.md`) + campaign-gated advisory tracer** — *proposed this pass; not yet ratified* | tool + sub-agent (proposed) | `bayesian_statistics.py`, `simulation_detection_probability.py`, `main.injection_campaign`, `handler.py`, `physical_relations.py`; event: pre-campaign | `.planning/CONVENTIONS-MANIFEST.md` (skeleton) | **was UNCOVERED — 4 incidents;** proposed owner pending Jasper ratification (open decision 5) | 2026-08-01 (Phase-2 submit) | **UNCOVERED → owner PROPOSED (advisory)** |
| 3 | Physics theory & formula change control | `physics-change` (advisory, description-triggered, no hard gate) | skill | 7 trigger files | `physics-change` SKILL | COVERED — scoped to single-file changes, not boundary consistency | 2026-10 | COVERED (WATCH: advisory-only triggering) |
| 4 | HPC implementation & provenance | `run-pipeline`, `gpu-audit` | skill + tool | pipeline/GPU globs | run-pipeline SKILL | COVERED | 2026-10 | COVERED |
| 5 | Quality & regression | `check`, `integration-test-eval` | skill + tool | `tests/**`, CI events | check SKILL | COVERED — declared blind spot: self-consistent wrong conventions pass it (568/568) | 2026-10 | COVERED |
| 6 | Known-bug triage | `known-bugs` | skill | issue/known-bug keywords | known-bugs SKILL | COVERED | 2026-10 | COVERED |
| 7 | State/docs upkeep | `pre-commit-docs` + Layer-1 STATE.md | skill + convention | `*.md`, `STATE.md`, docs CI | `.planning/STATE.md` | COVERED | 2026-10 | COVERED (CONSTITUTION finding — see C-MTC-20260712-002) |

## Findings  (append-only; each row carries a unique id: C-<host>-YYYYMMDD-<seq>)

| id | date | kind | area | evidence rows (dated, cited) | cost of last incident | proposed owner + rung | status |
|----|------|------|------|------------------------------|-----------------------|-----------------------|--------|
| C-MTC-20260712-001 | 2026-07-12 | UNCOVERED | 2 (Sim/eval convention consistency) | **≥2 dated (4 total):** (a) coordinate-frame 2026-04-21 — [[scientific-computing-validation]]:36, 0.0% apparent bias / 6 shipped milestones; (b) mass-redshift 2026-06-20 — SCV:38, W-CONF-13, passed all 568 tests; (c) `L_cat` mean-of-ratios — commit `816f904`, SCV:203; (d) `p_det` denominator / incomplete-fix — commit `341ca62`, W-PRE-12 | **one full retired data inventory** (`simulations/_RETIRED_20260620_pre_massfix_lcat/`) **+ full re-simulation campaigns** (weeks of cluster GPU time + days of human orchestration) | manifest + invariant tests (tool/convention) as standing floor **+** standing pre-campaign advisory tracer (sub-agent, event-gated at campaign boundaries), human-ratified pre-PASS (anchor-1) | **OPEN — this is the organism's FIRST missing-coverage finding; owner proposed, pending Jasper ratification (open decision 5). Tracer run 2026-07-12: no live divergence found on the 4 seeded classes + HOST_DRAW_Z_MAX; verdict ADVISORY (see tracer-verdict-2026-07-12.md).** |
| C-MTC-20260712-002 | 2026-07-12 | CONSTITUTION | 7 (State/docs upkeep) | [[master-thesis-code]] entity page ~3 months stale (C.6 Step-5); live state migrated into registry Key-Conventions cell; [[scientific-computing-validation]] frontmatter `updated: 2026-05-04` lags its own body (entries through 2026-07-11) | none (documentation drift, no wrong result) | refresh entity page + SCV frontmatter at next `/wiki-caretaker` | WATCH |
| C-MTC-20260712-003 | 2026-07-12 | WATCH | 2 (Sim/eval convention consistency) | `pp_coverage.Z_MAX_POP = 0.95` (validation harness) vs production `HOST_DRAW_Z_MAX = 1.5`; SCV 2026-07-11 shows estimator bias is depth- and σ_z/z-dependent | none yet (calibration-depth mismatch, not a shipped result) | confirm per-seed `pp_coverage` runs at campaign depth, not the hardcoded 0.95 | **WATCH — under active exploration (Jasper 2026-07-12): both depth scenarios (0.95 vs 1.5) are being run and evidence collected before the final setup is decided. Not a submission blocker; revisit when the setup is finalized.** |
| C-MTC-20260712-004 | 2026-07-12 | WATCH | 2 (Sim/eval convention consistency) | tracer refuter 2026-07-12: the injection catalog `"M" = M_z` write (`main.py:899/983`) has **no paired invariant test** asserting the column is redshifted — only the CRB path is guarded; consistency is held by a runtime comment, not CI, so a future revert of the M_z lift fails silently (the W-PRE-12 "every-output invariant" gap) | none yet (latent) | **paired invariant test in CI** asserting the injection catalog `"M"` column == `M_source·(1+z)` (tool/convention rung) — the standing-floor half of finding C-001's owner | **APPROVED FOR IMPLEMENTATION (Jasper 2026-07-12).** A future MTC session should add this test; it operationalises the C-001 "manifest + invariant tests" floor for the M_z boundary. |

## Archive  (drained/superseded findings move here under a dated '### YYYY-MM' heading with a supersession pointer appended — never edited in place, never deleted, never left only to git history)

_(empty)_

## Routing notes

Trigger-set semantics are AUDIT-TIME in v1: the map is documentation agents read
and auditors join against — no hook or router evaluates triggers during a live
session. Escalation convention: work matching no owner → session default; the
session files a WATCH row at the next classification pass. Determinism rule:
trigger sets pairwise disjoint; overlap is an OVERLAP finding.

**Seed provenance / honesty note.** This map was hand-seeded during the 2026-07-12
advisory tracer run to emit the organism's first missing-coverage finding
(C-MTC-20260712-001) before Phase-2 submission — it is NOT the output of the
`/gardener --coverage` full 6-step retrofit (which is propose-class and estimated
30–60 min in a fresh context). Areas 1,3–7 are transcribed from the C.6 worked
example, not independently re-audited here. The proper retrofit (commit-classification
+ incident-classification MECE tests, RACI-lite owner assignment, rung + governance
pressure test) remains outstanding and should supersede this seed.
