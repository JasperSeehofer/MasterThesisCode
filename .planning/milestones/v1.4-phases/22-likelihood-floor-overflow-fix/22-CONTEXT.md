# Phase 22: Likelihood Floor & Overflow Fix - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Make the evaluate pipeline produce physically grounded likelihoods for all events (no zeros from catalog gaps) and remove obsolete overflow detection. The floor is a pragmatic stopgap until catalog incompleteness modeling is implemented; it must not introduce bias.

</domain>

<decisions>
## Implementation Decisions

### Floor Derivation (NFIX-02)
- **D-01:** The floor uses the **minimum nonzero likelihood** approach — for each event, the floor = the smallest nonzero likelihood value across all h-bins for that event. This is a quick, thought-through solution that preserves relative event weights without introducing h-dependent bias.
- **D-02:** This is explicitly a **stopgap** — catalog incompleteness modeling will be added to the evaluation pipeline in a future phase and may eliminate the zero-likelihood problem entirely. The floor must be easy to remove/replace.
- **D-03:** The floor is scoped **per-event** (not global, not per-h-bin) to avoid coupling events together or distorting relative event contributions.

### Floor Placement
- **D-04:** The floor is applied **in the combination step** (post-hoc in `combine_posteriors`), NOT inside `single_host_likelihood`. The evaluation pipeline continues to output honest zeros; the floor is a combination-time decision. This keeps the physics code clean and makes the floor trivial to swap out later.

### Underflow Detection (NFIX-03)
- **D-05:** `check_overflow` is **removed entirely** rather than fixed. Log-space accumulation (Phase 21) and the per-event-min floor handle the numerical problems at the combination layer, making `check_overflow` dead code. Remove the function and all call sites.

### Validation
- **D-06:** Validation uses **both MAP comparison and visual posterior overlay** against known baselines (naive MAP=0.72/0.86, exclude MAP=0.68/0.66).
- **D-07:** Acceptable MAP shift: physics-floor MAP must be within **+/-0.05 of exclude** (Option 1) MAP for both with/without BH mass variants.
- **D-08:** Success criterion: all h-bins have nonzero likelihood for every event after floor application.

### Claude's Discretion
- Implementation details of the per-event-min floor computation inside `combine_posteriors`
- How to log/report which events had zeros filled by the floor
- Test structure for floor behavior (unit tests for edge cases like all-zero events, single-nonzero events)
- Whether to add the floor as a new strategy or modify the existing `physics-floor` strategy stub from Phase 21

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Evaluate Pipeline
- `master_thesis_code/bayesian_inference/bayesian_statistics.py` — `single_host_likelihood()` (line 500): the core integration that produces zeros when no catalog galaxy overlaps the GW error volume. `check_overflow()` (line 1023): the function to remove.
- `master_thesis_code/bayesian_inference/bayesian_statistics.py` — `check_overflow` call sites must be found and removed

### Combination Pipeline (Phase 21 output)
- `master_thesis_code/bayesian_inference/combine_posteriors.py` — (or wherever Phase 21 placed the combination logic) — the `physics-floor` strategy stub that needs the real floor implementation
- `master_thesis_code/__main__.py` — CLI entry point with `--combine --strategy` flags

### Existing Campaign Data
- `results/h_sweep_20260401/posteriors/` — Per-h-value JSON files for "without BH mass" variant
- `results/h_sweep_20260401/posteriors_with_bh_mass/` — Per-h-value JSON files for "with BH mass" variant (111 zero-events, 21%)

### Requirements
- `.planning/REQUIREMENTS.md` — NFIX-02 (physics floor), NFIX-03 (underflow detection)

### Prior Phase Context
- `.planning/phases/21-analysis-post-processing/21-CONTEXT.md` — D-05/D-06/D-07 define the strategy enum and fallback behavior that Phase 22 completes

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 21's `CombinationStrategy` enum already includes `PHYSICS_FLOOR` with a fallback to `EXCLUDE` + logged warning — Phase 22 replaces the fallback with the real implementation
- Phase 21's log-space accumulation (`log_shift_exp`) is already in place — the floor just needs to ensure no zeros enter the log computation

### Established Patterns
- Per-event JSON format: `{"0": [val], "1": [val], ..., "h": 0.73}` — detection index as string key, likelihood as single-element list
- CLI subcommands via argparse in `__main__.py` with `--combine --strategy` flags

### Integration Points
- `combine_posteriors` module — the `physics-floor` strategy code path is the insertion point for the real floor logic
- `check_overflow` (line 1023) and its call sites — removal targets

</code_context>

<specifics>
## Specific Ideas

- The per-event-min floor is intentionally simple and bias-neutral — it's a placeholder until the real fix (catalog incompleteness modeling) arrives
- Floor application must be logged: which events had zeros, how many bins were floored, what the floor value was
- The validation plots (physics-floor overlay on naive/exclude) should be saved to the working directory alongside the combination output

</specifics>

<deferred>
## Deferred Ideas

- **Catalog incompleteness model** — the proper fix for zero-likelihood events; will replace the per-event-min floor. Tracked in REQUIREMENTS.md as a future requirement.
- **Full log-space accumulation inside the evaluate pipeline itself** (not just post-processing) — also tracked as deferred in REQUIREMENTS.md

</deferred>

---

*Phase: 22-likelihood-floor-overflow-fix*
*Context gathered: 2026-04-02*
