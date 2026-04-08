---
phase: 30-baseline-evaluation-infrastructure
verified: 2026-04-08T10:15:00Z
status: gaps_found
score: 4/5 must-haves verified
overrides_applied: 0
gaps:
  - truth: "Baseline snapshot is committed to .planning/debug/ so it can be referenced by all subsequent phases"
    status: failed
    reason: ".planning/debug/baseline.json exists on disk but is untracked (not committed to git). The file also contains synthetic test data (5 h-values [0.69-0.77], n_events=5, bias=0%) rather than a real production baseline from the existing evaluation posteriors (e.g. run_v12_validation with 11 real h-values)."
    artifacts:
      - path: ".planning/debug/baseline.json"
        issue: "File exists but is untracked by git (shown in git status as '?? .planning/debug/baseline.json'). Contains synthetic data, not production posteriors."
    missing:
      - "Run --save_baseline against a real production posteriors directory (e.g. evaluation/run_v12_validation/simulations/posteriors/) to capture the actual H0 bias state"
      - "Commit the real baseline.json (and optionally comparison reports) to .planning/debug/ so subsequent phases can reference it"
---

# Phase 30: Baseline & Evaluation Infrastructure Verification Report

**Phase Goal:** Capture a reproducible baseline posterior snapshot and establish the comparison framework that all subsequent phases use to measure their effect
**Verified:** 2026-04-08T10:15:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Running --save_baseline on a directory with h_*.json posteriors produces a baseline.json with MAP h, 68% CI, bias %, event count, and per-event summaries | VERIFIED | extract_baseline() in evaluation_report.py:211-273 computes all these. Integration test test_save_baseline_cli_integration passes (20/20 tests). |
| 2 | Running --compare_baseline <path> on a directory with posteriors produces a markdown + JSON comparison report | VERIFIED | generate_comparison_report() writes comparison_{label}.md and comparison_{label}.json. Tests test_compare_baseline_cli_integration and test_generate_comparison_report_* all pass. |
| 3 | --compare_baseline works standalone (without --evaluate) comparing existing result sets | VERIFIED | test_compare_baseline_standalone confirms this. _compare_baseline() in main.py reads existing posteriors from disk without requiring --evaluate. |
| 4 | Baseline JSON schema is self-documenting and machine-parseable | VERIFIED | BaselineSnapshot.to_json() returns all 11 documented fields: map_h, ci_lower, ci_upper, ci_width, bias_percent, n_events, h_values, log_posteriors, per_event_summaries, created_at, git_commit. test_baseline_snapshot_json_roundtrip passes. |
| 5 | Baseline snapshot is committed to .planning/debug/ so it can be referenced by all subsequent phases | FAILED | .planning/debug/baseline.json is untracked by git (confirmed via git status). The file also contains synthetic test data (5 h-values, bias=0%, n_events=5) rather than a real production baseline. |

**Score:** 4/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `master_thesis_code/bayesian_inference/evaluation_report.py` | Baseline extraction, comparison report generation, CI computation | VERIFIED | 481 lines. Exports: BaselineSnapshot, load_posteriors, compute_credible_interval, extract_baseline, generate_comparison_report. All substantive. |
| `master_thesis_code/arguments.py` | CLI flags --save_baseline, --compare_baseline | VERIFIED | Both flags present at lines 235-250, properties at lines 110-117. `--help` confirms flags visible. |
| `master_thesis_code/main.py` | Dispatch logic for --save_baseline and --compare_baseline | VERIFIED | Dispatch at lines 72-76, _save_baseline at line 130, _compare_baseline at line 159. Uses lazy imports per plan. |
| `master_thesis_code_test/bayesian_inference/test_evaluation_report.py` | Tests for baseline extraction, CI computation, comparison report | VERIFIED | 450 lines (min_lines: 80). 20 tests, all passing. Includes unit, integration, and standalone tests. |
| `.planning/debug/baseline.json` | Real production baseline committed to git | FAILED | File exists but untracked (git status shows ??). Contains synthetic data from test/manual run, not production posteriors. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `master_thesis_code/main.py` | `master_thesis_code/bayesian_inference/evaluation_report.py` | import and call extract_baseline / generate_comparison_report | WIRED | Lines 134 and 164-166 in main.py. Lazy imports inside _save_baseline and _compare_baseline functions. |
| `master_thesis_code/arguments.py` | `master_thesis_code/main.py` | arguments.save_baseline / arguments.compare_baseline properties | WIRED | arguments.save_baseline used at main.py:72, arguments.compare_baseline at main.py:75. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| `evaluation_report.py:extract_baseline` | posteriors (list[dict]) | load_posteriors reads h_*.json files from disk | Yes — reads real JSON files, computes log_posterior from actual likelihoods | FLOWING |
| `evaluation_report.py:generate_comparison_report` | baseline, current (BaselineSnapshot) | Caller-provided via extract_baseline | Yes — all fields derived from real file reads | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| CLI flags appear in --help | `uv run python -m master_thesis_code --help \| grep -E "save_baseline\|compare_baseline"` | Both flags shown with descriptions | PASS |
| All 20 unit + integration tests pass | `uv run pytest test_evaluation_report.py -v --no-cov` | 20 passed in 1.06s | PASS |
| Commits claimed in SUMMARY exist | git show 7eab0aa, e2d45cb, a9cbc3d | All three valid, correct files | PASS |
| baseline.json committed to git | `git status .planning/debug/` | Shows as untracked (??) | FAIL |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| DIAG-03 | 30-01-PLAN.md | Baseline posterior snapshot (current MAP h, 68% CI, bias %) saved before any fixes | PARTIAL | Infrastructure to save baseline is built and functional. However the actual baseline capture (running against real production posteriors) has not been committed. The .planning/debug/baseline.json is synthetic test data, untracked. |
| EVAL-01 | 30-01-PLAN.md | Before/after comparison report generated automatically: MAP h, 68% CI width, bias %, number of events used | SATISFIED | generate_comparison_report() produces both markdown table and JSON sidecar with all required metrics. Verified by tests and code inspection. |
| EVAL-02 | 30-01-PLAN.md | Each fix produces a comparison against the baseline, stored in a structured format for cumulative tracking | SATISFIED | --compare_baseline flag reads a baseline.json and produces structured JSON output. The infrastructure supports per-phase comparison tracking. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `evaluation_report.py` | 254 | `n_events = int(posteriors[0]["n_detections"])` — always first file, not MAP-h file | Warning | Could silently miscount if different h-files have different event sets (flagged as WR-01 in code review). Not a stub — real data flows, but potentially wrong row. |
| `evaluation_report.py` | 96-98 | `from typing import Any` inside method body to cast `dict[str, object]` | Info | Bypasses proper mypy type checking (flagged as WR-03 in code review). |
| `evaluation_report.py` | 271 | `created_at=...` explicitly passed despite `default_factory` | Info | Redundant dual timestamp generation (flagged as WR-04 in code review). |
| `main.py` | 184 | `generate_comparison_report(baseline, current, output_dir)` — no `label` argument | Info | Every --compare_baseline call overwrites `comparison_current.md` (flagged as IN-03 in code review). |

None of these are stubs. The warnings are correctness edge cases and style issues from the existing code review report.

### Gaps Summary

One gap blocks full goal achievement:

**The baseline.json is not committed to git.** The phase goal is "capture a baseline posterior snapshot" and ROADMAP SC #4 explicitly requires the snapshot to be "committed to `.planning/debug/`." The file currently in `.planning/debug/baseline.json` is:
1. Untracked (git shows it as `??`)
2. Contains synthetic data (5 h-values, perfect Gaussian, bias=0%) rather than a real production baseline

The production posteriors exist at `evaluation/run_v12_validation/simulations/posteriors/` (11 h-values). The `--save_baseline` tooling works correctly — it just needs to be run against real data and the result committed.

This is a one-command fix: run `--save_baseline` against the appropriate production working directory, then `git add .planning/debug/baseline.json && git commit`.

---

_Verified: 2026-04-08T10:15:00Z_
_Verifier: Claude (gsd-verifier)_
