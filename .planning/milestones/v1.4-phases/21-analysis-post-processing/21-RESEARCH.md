# Phase 21: Analysis & Post-Processing - Research

**Researched:** 2026-04-02
**Domain:** Bayesian posterior combination, numerical stability, diagnostic analysis
**Confidence:** HIGH

## Summary

Phase 21 formalizes the zero-likelihood diagnostic analysis already begun in conversation, builds a standalone posterior combination script using log-space accumulation, and produces a comparison table of all four combination methods. The technical domain is straightforward: NumPy log-sum-exp arithmetic, JSON I/O for existing per-event posterior files, argparse CLI extension, and markdown report generation.

The existing campaign data in `results/h_sweep_20260401/posteriors/` contains 15 h-value JSON files, each with 538 detection keys. Analysis confirms 17 events produce zero likelihoods at one or more h-bins, with 3 events (163, 223, 507) zero at ALL 15 h-bins. The naive product posterior is zero everywhere due to these all-zeros events. The 4 "empty" events (missing from JSONs) need distinct handling from the 0.0-valued events.

**Primary recommendation:** Build a new `bayesian_inference/posterior_combination.py` module with pure functions for loading, diagnosing, and combining posteriors, wired to `--combine` CLI flag in `__main__.py`. Use `np.sum(np.log(...))` with log-shift-exp trick. All four strategies as an enum.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** The combination script is a new `--combine` CLI subcommand on `__main__.py`, following the same pattern as `--evaluate` and `--snr_analysis`
- **D-02:** The combination logic lives in a new module under `bayesian_inference/` (importable) and is wired to the CLI via `__main__.py`
- **D-03:** The zero-likelihood diagnostic report is a generated markdown file (`diagnostic_report.md`) written to the working directory
- **D-04:** The report covers: which events produce zeros at which h-bins, root causes (no hosts, catalog gaps, redshift mismatch), and summary statistics
- **D-05:** Default strategy is Option 3 (physics floor) when available, with automatic fallback to Option 1 (exclude zeros) when the physics floor is not yet implemented
- **D-06:** All four strategies are selectable via CLI flag: `naive`, `exclude` (Option 1), `per-event-floor` (Option 2), `physics-floor` (Option 3)
- **D-07:** Option 3 depends on Phase 22 (`single_host_likelihood` floor) -- until then, selecting it explicitly warns and falls back to Option 1
- **D-08:** The combination script outputs JSON only (`combined_posterior.json` with joint H0 posterior array + metadata)
- **D-09:** Plotting is handled separately by existing/future plotting infrastructure -- the combination script is a pure data processing step

### Claude's Discretion
- Internal module structure and function decomposition
- JSON schema for combined_posterior.json (must include h-values, posterior array, method used, event count, metadata)
- Comparison table format for ANAL-02 (markdown file in working directory)
- Log-shift-exp implementation details (standard numerical technique)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ANAL-01 | Diagnostic report documenting zero-likelihood origins per h-bin | Data analysis confirms 17 zero-events, 3 all-zeros events, 4 empty events. JSON structure well understood. Root cause categories identified. |
| ANAL-02 | Comparison table of all combination methods with MAP estimates | All four methods implementable from existing data. Known baselines: naive MAP=0.72/0.86, Option 1 MAP=0.68/0.66. |
| POST-01 | Standalone combination script with log-space accumulation + configurable zero-handling | NumPy log-sum-exp is standard. JSON I/O pattern established. CLI pattern from `--evaluate` reusable. |
| NFIX-01 | Log-space posterior accumulation replacing `np.prod` with `np.sum(np.log(...))` | Standard numerical technique, well understood. Log-shift-exp trick documented below. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | (already installed) | Log-space arithmetic, array operations | Project dependency |
| json | (stdlib) | Load per-event posterior JSONs, write combined output | Existing format |
| argparse | (stdlib) | CLI `--combine` flag + `--strategy` option | Existing pattern in `arguments.py` |
| logging | (stdlib) | Diagnostic output and fallback warnings | Existing pattern |
| pathlib/os | (stdlib) | Directory traversal for posterior JSON discovery | Existing pattern |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| enum | (stdlib) | `CombinationStrategy` enum for the 4 methods | Type-safe strategy selection |

No new dependencies required. Everything uses NumPy and stdlib.

## Architecture Patterns

### Recommended Project Structure
```
master_thesis_code/
  bayesian_inference/
    posterior_combination.py   # NEW: all combination logic
  __main__.py                  # ADD: --combine flag
  arguments.py                 # ADD: combine + strategy args
```

### Pattern 1: CLI Subcommand Integration
**What:** Add `--combine` as a new action flag alongside `--evaluate`, `--snr_analysis`, `--injection_campaign`
**When to use:** This phase
**Example:**
```python
# In arguments.py _parse_arguments():
parser.add_argument("--combine", action="store_true", default=False,
    help="Combine per-event posteriors into joint H0 posterior.")
parser.add_argument("--strategy", type=str, default="physics-floor",
    choices=["naive", "exclude", "per-event-floor", "physics-floor"],
    help="Zero-handling strategy for posterior combination.")

# In main.py main():
if arguments.combine:
    combine_posteriors(arguments.working_directory, arguments.strategy)
```

### Pattern 2: Per-Event Posterior JSON Loading
**What:** Load all `h_*.json` files from a posteriors directory, build a 2D array (events x h-bins)
**When to use:** Both combination and diagnostic generation
**Example:**
```python
def load_posteriors(posteriors_dir: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[int]]:
    """Load per-h-value JSONs into (h_values, likelihoods[n_events, n_h]) arrays.

    Returns:
        h_values: 1D array of h grid points
        likelihoods: 2D array [n_events, n_h_values], 0.0 for zeros, NaN for missing
        detection_indices: list of detection index integers
    """
```

The existing JSON format is: `{"0": [10.83], "1": [0.025], ..., "h": 0.73}` where:
- Keys are string detection indices
- Values are single-element lists `[likelihood_value]`
- Zero likelihoods appear as `[0.0]`
- Missing events appear as `[]` (empty list)
- The `"h"` key stores the h-value for this file

### Pattern 3: Log-Shift-Exp Trick
**What:** Numerically stable log-space combination avoiding underflow
**When to use:** Core of NFIX-01

```python
def combine_log_space(
    log_likelihoods: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Combine per-event log-likelihoods into joint posterior.

    log_likelihoods shape: [n_events, n_h_values]
    Returns: normalized posterior [n_h_values]

    Uses log-sum-exp: log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
    But here we SUM log-likelihoods (multiply in linear space), not logsumexp.
    """
    # Sum log-likelihoods across events for each h-bin
    joint_log_posterior = np.sum(log_likelihoods, axis=0)  # shape [n_h_values]

    # Shift before exp to avoid underflow/overflow
    max_log = np.max(joint_log_posterior)
    posterior = np.exp(joint_log_posterior - max_log)

    # Normalize
    posterior /= np.sum(posterior)
    return posterior
```

**Key insight:** The "log-shift-exp trick" here is simpler than logsumexp. We sum logs (= log of product), then shift by the max before exponentiating to get a normalized posterior. The shift cancels in normalization anyway but prevents exp() from returning 0.0 or inf.

### Pattern 4: Zero-Handling Strategies as Enum
**What:** Type-safe strategy selection with clear semantics
```python
from enum import Enum

class CombinationStrategy(str, Enum):
    NAIVE = "naive"
    EXCLUDE = "exclude"           # Option 1: skip events with any zeros
    PER_EVENT_FLOOR = "per-event-floor"  # Option 2: replace 0 with min nonzero / 100
    PHYSICS_FLOOR = "physics-floor"      # Option 3: Phase 22 floor (not yet implemented)
```

### Pattern 5: Diagnostic Report Generation
**What:** Structured markdown report written to working directory
**When to use:** ANAL-01

The report needs to identify:
1. **Which events produce zeros** at which h-bins (17 events, pattern: more zeros at low h)
2. **All-zeros events** (3 events: 163, 223, 507 -- zero at all 15 h-bins)
3. **Empty events** (4 events: missing from JSONs entirely, different from zero)
4. **Root causes**: no hosts in error volume, catalog coverage gaps at high z, redshift mismatch between GW measurement and galaxy catalog
5. **Impact**: how zeros propagate through naive multiplication to kill the entire posterior

### Anti-Patterns to Avoid
- **Using `np.prod()` for combination:** The entire point of NFIX-01 is to replace this with log-space arithmetic
- **Silent fallback without logging:** D-07 requires logged warnings when physics-floor falls back to exclude
- **Hardcoding posterior directory paths:** Must use working_directory convention from CLI
- **Reading "with BH mass" posteriors that don't exist:** The `posteriors_with_bh_mass/` directory does not exist in the current campaign data. The comparison table (ANAL-02) should handle this gracefully.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Log-sum-exp | Custom overflow protection | `np.max` shift + `np.exp` | Standard technique, but here we just need shift-before-exp since we sum logs not logsumexp |
| JSON schema validation | Custom parser | Simple dict key checks | The format is trivial and well-known |
| CLI argument parsing | Custom flag system | argparse (existing pattern) | Already established in arguments.py |
| Markdown generation | Template engine | f-string formatted markdown | Reports are simple tables, no need for jinja2 |

## Common Pitfalls

### Pitfall 1: Confusing Empty Events with Zero-Likelihood Events
**What goes wrong:** Treating `[]` (event not in JSON, missing data) the same as `[0.0]` (event evaluated, got zero likelihood)
**Why it happens:** Both are "no contribution" but have different meanings
**How to avoid:** Track separately. Empty = never evaluated for this h-value (4 events). Zero = evaluated but no hosts found (17 events at various h-bins). Empty events should be excluded from ALL strategies. Zero events are what the strategies handle.
**Warning signs:** Event count mismatch between strategies

### Pitfall 2: Log of Zero
**What goes wrong:** `np.log(0.0)` produces `-inf`, which propagates through sum to `-inf` for that h-bin
**Why it happens:** Zero-likelihood events exist in the data
**How to avoid:** Zero-handling strategy MUST be applied BEFORE taking the log. For "exclude" strategy, filter out events with any zeros before log. For "per-event-floor", replace zeros with floor value before log.
**Warning signs:** `-inf` values in log-likelihood arrays

### Pitfall 3: Per-Event Floor Overcorrection (Option 2)
**What goes wrong:** MAP estimate collapses to h=0.60 (lowest grid point) for both variants
**Why it happens:** Replacing per-event zeros with a floor value adds artificial probability mass at h-bins where the event truly has no support, but the floor is constant while real likelihoods decrease, so the floor dominates at extreme h values
**How to avoid:** This is a known issue (STATE.md documents it). Option 2 is included for completeness in the comparison table but should not be recommended as the default.
**Warning signs:** MAP at grid boundary

### Pitfall 4: Missing "With BH Mass" Data
**What goes wrong:** Code assumes both `posteriors/` and `posteriors_with_bh_mass/` directories exist
**Why it happens:** The current campaign data only has `posteriors/` -- the BH mass variant directory does not exist
**How to avoid:** Check directory existence before loading. The comparison table should note "data not available" for the BH mass variant, or the user may need to point to a different campaign directory.
**Warning signs:** FileNotFoundError on startup

### Pitfall 5: Detection Index Gaps
**What goes wrong:** Assuming detection indices are contiguous 0..N-1
**Why it happens:** Some indices are missing (e.g., index 5 is absent in the data). The JSONs have gaps in their key sequences.
**How to avoid:** Use the actual key set from the JSON, not `range(max_key)`. The 4 "empty" events are indices that appear in some h-files but not others.
**Warning signs:** KeyError when building the 2D array

### Pitfall 6: Normalization Before vs After Combination
**What goes wrong:** Normalizing per-event likelihoods before multiplying changes the posterior shape
**Why it happens:** Tempting to normalize to avoid large values
**How to avoid:** Combine raw likelihoods in log-space first, then normalize the final posterior only. The log-shift-exp trick handles the magnitude issue without altering relative weights.
**Warning signs:** Different MAP from the naive baseline

## Code Examples

### Loading Existing JSON Format
```python
import json
import numpy as np
from pathlib import Path

def load_posterior_jsons(posteriors_dir: Path) -> tuple[list[float], dict[int, dict[float, float]]]:
    """Load all h_*.json files from a posteriors directory.

    Returns:
        h_values: sorted list of h grid points
        event_likelihoods: {detection_index: {h_value: likelihood}}
            - likelihood is float (including 0.0 for zeros)
            - missing events for a given h are absent from inner dict
    """
    h_values: list[float] = []
    event_likelihoods: dict[int, dict[float, float]] = {}

    for json_path in sorted(posteriors_dir.glob("h_*.json")):
        with open(json_path) as f:
            data = json.load(f)
        h = data.pop("h")
        h_values.append(h)

        for det_key, val in data.items():
            det_idx = int(det_key)
            if det_idx not in event_likelihoods:
                event_likelihoods[det_idx] = {}
            if isinstance(val, list) and len(val) > 0:
                event_likelihoods[det_idx][h] = val[0]
            # Empty list [] means event not evaluated -- skip

    return sorted(h_values), event_likelihoods
```

### Applying Zero-Handling Strategy
```python
def apply_strategy(
    h_values: npt.NDArray[np.float64],
    likelihoods: npt.NDArray[np.float64],  # [n_events, n_h]
    strategy: CombinationStrategy,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
    """Apply zero-handling strategy, return (filtered_likelihoods, log_likelihoods, excluded_count)."""
    if strategy == CombinationStrategy.NAIVE:
        # Replace zeros with tiny value to avoid log(0), but result will be ~0
        safe = np.where(likelihoods > 0, likelihoods, np.finfo(float).tiny)
        return safe, np.log(safe), 0

    elif strategy == CombinationStrategy.EXCLUDE:
        # Exclude any event that has zero at ANY h-bin
        has_zero = np.any(likelihoods == 0.0, axis=1)
        excluded = int(np.sum(has_zero))
        kept = likelihoods[~has_zero]
        return kept, np.log(kept), excluded

    elif strategy == CombinationStrategy.PER_EVENT_FLOOR:
        # Replace zeros with min(nonzero values for that event) / 100
        result = likelihoods.copy()
        for i in range(result.shape[0]):
            row = result[i]
            nonzero_mask = row > 0
            if np.any(nonzero_mask):
                floor = np.min(row[nonzero_mask]) / 100.0
                result[i] = np.where(row > 0, row, floor)
            # else: all-zero event, leave as zeros (will be -inf in log)
        safe = np.where(result > 0, result, np.finfo(float).tiny)
        return result, np.log(safe), 0

    elif strategy == CombinationStrategy.PHYSICS_FLOOR:
        # Phase 22 -- not yet implemented, fall back to EXCLUDE
        import logging
        logging.getLogger().warning(
            "Physics floor (Option 3) not yet implemented (Phase 22). "
            "Falling back to 'exclude' strategy."
        )
        return apply_strategy(h_values, likelihoods, CombinationStrategy.EXCLUDE)
```

### Sanity Check Against Known Baselines
```python
def sanity_check(
    h_values: npt.NDArray[np.float64],
    posterior: npt.NDArray[np.float64],
    strategy: CombinationStrategy,
    variant: str,  # "without_bh_mass" or "with_bh_mass"
) -> bool:
    """Check MAP against known baselines from prior analysis."""
    map_idx = np.argmax(posterior)
    map_h = h_values[map_idx]

    known_maps = {
        ("without_bh_mass", CombinationStrategy.NAIVE): 0.86,
        ("with_bh_mass", CombinationStrategy.NAIVE): 0.72,
        ("without_bh_mass", CombinationStrategy.EXCLUDE): 0.66,
        ("with_bh_mass", CombinationStrategy.EXCLUDE): 0.68,
    }

    expected = known_maps.get((variant, strategy))
    if expected is not None:
        # Grid spacing is 0.02, so allow one grid step tolerance
        if abs(map_h - expected) <= 0.02:
            return True
        logging.warning(f"MAP mismatch: got {map_h}, expected {expected}")
        return False
    return True  # No baseline to check against
```

## Existing Campaign Data Analysis

Verified by direct inspection of `results/h_sweep_20260401/posteriors/`:

| Property | Value |
|----------|-------|
| H-value grid | 15 points: 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.73, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86 |
| Total detection keys per file | 538 |
| Empty events (all files) | 4 (missing data, `[]`) |
| Events with at least one zero | 17 |
| Events zero at ALL h-bins | 3 (indices 163, 223, 507) |
| Zero pattern | More zeros at low h (17 at h=0.6, 3 at h=0.86) |
| `posteriors_with_bh_mass/` | Does NOT exist in current campaign |
| Naive product | 0.0 at all h-bins (killed by all-zeros events) |

**Zero-event detail:**

| Detection | Zero h-bins | Pattern |
|-----------|-------------|---------|
| 80 | 0.6 | Low-h only |
| 107 | 0.6-0.64 | Low-h cluster |
| 160 | 0.6-0.70 | Wide low-h band |
| 163 | ALL (0.6-0.86) | Complete zero -- no hosts at any h |
| 194 | 0.6-0.70 | Wide low-h band |
| 213 | 0.6 | Low-h only |
| 223 | ALL (0.6-0.86) | Complete zero -- no hosts at any h |
| 228 | 0.6-0.64 | Low-h cluster |
| 244 | 0.6-0.64 | Low-h cluster |
| 249, 362, 363, 423, 440, 455 | 0.6 only | Low-h only |
| 506 | 0.6-0.84 | Nearly all zeros (only h=0.86 nonzero) |
| 507 | ALL (0.6-0.86) | Complete zero -- no hosts at any h |

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `np.prod(likelihoods, axis=0)` | `np.sum(np.log(likelihoods), axis=0)` + shift-exp | This phase | Prevents underflow to zero for large event counts |
| No zero handling | Strategy-based zero handling (4 options) | This phase | Allows meaningful posterior even with problematic events |

## Open Questions

1. **"With BH mass" comparison data**
   - What we know: `posteriors_with_bh_mass/` does not exist in current campaign results
   - What's unclear: Whether it will be generated by pending cluster jobs, or if it existed in a previous campaign
   - Recommendation: Make comparison table handle missing variant gracefully. If the user has prior data elsewhere, accept a `--posteriors_dir` argument to point to it. The known MAP values (0.72 with BH mass, 0.86 without) suggest the "without BH mass" data IS the existing posteriors/ directory (MAP would be 0.86 naive if zeros were not present). Need to clarify: does the current `posteriors/` directory represent "without BH mass" or "with BH mass"?

2. **Root cause attribution for zero events**
   - What we know: Zeros occur when `single_host_likelihood` returns 0 because no galaxy hosts are found in the error volume (`possible_hosts is None` at line 317 of bayesian_statistics.py)
   - What's unclear: Whether we can distinguish "no hosts in catalog" from "catalog coverage gap" vs "redshift mismatch" without re-running the evaluation. The JSON files only store the final likelihood value, not the intermediate diagnostic.
   - Recommendation: The diagnostic report can categorize by zero-pattern (all-zeros = likely no hosts for any h, low-h-only zeros = redshift-dependent coverage gap). Exact root cause attribution would require cross-referencing the detection's sky position and distance against the galaxy catalog, which is possible but adds complexity. Start with pattern-based diagnosis.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `uv run pytest -m "not gpu and not slow" -x` |
| Full suite command | `uv run pytest -m "not gpu"` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| NFIX-01 | Log-space accumulation produces same MAP as naive (when no zeros) | unit | `uv run pytest master_thesis_code_test/bayesian_inference/test_posterior_combination.py::test_log_space_matches_naive -x` | Wave 0 |
| NFIX-01 | Log-shift-exp avoids underflow for 500+ events | unit | `uv run pytest master_thesis_code_test/bayesian_inference/test_posterior_combination.py::test_no_underflow_large_event_count -x` | Wave 0 |
| POST-01 | JSON loading parses all 15 h-value files correctly | unit | `uv run pytest master_thesis_code_test/bayesian_inference/test_posterior_combination.py::test_load_posteriors -x` | Wave 0 |
| POST-01 | Exclude strategy removes correct events | unit | `uv run pytest master_thesis_code_test/bayesian_inference/test_posterior_combination.py::test_exclude_strategy -x` | Wave 0 |
| POST-01 | Per-event-floor replaces zeros correctly | unit | `uv run pytest master_thesis_code_test/bayesian_inference/test_posterior_combination.py::test_per_event_floor_strategy -x` | Wave 0 |
| POST-01 | Physics-floor falls back to exclude with warning | unit | `uv run pytest master_thesis_code_test/bayesian_inference/test_posterior_combination.py::test_physics_floor_fallback -x` | Wave 0 |
| POST-01 | CLI --combine flag accepted by argument parser | unit | `uv run pytest master_thesis_code_test/test_arguments.py::test_combine_flag -x` | Wave 0 |
| ANAL-01 | Diagnostic report generation produces valid markdown | unit | `uv run pytest master_thesis_code_test/bayesian_inference/test_posterior_combination.py::test_diagnostic_report -x` | Wave 0 |
| ANAL-02 | Comparison table includes all 4 strategies | unit | `uv run pytest master_thesis_code_test/bayesian_inference/test_posterior_combination.py::test_comparison_table -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest master_thesis_code_test/bayesian_inference/test_posterior_combination.py -x`
- **Per wave merge:** `uv run pytest -m "not gpu and not slow" -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `master_thesis_code_test/bayesian_inference/test_posterior_combination.py` -- covers NFIX-01, POST-01, ANAL-01, ANAL-02
- [ ] Test fixtures: synthetic posterior JSON files with known zero patterns

## Project Constraints (from CLAUDE.md)

- **Package manager:** uv (never manually edit pyproject.toml dependencies)
- **Type annotations:** All functions must have complete type annotations (npt.NDArray[np.float64], not bare np.ndarray)
- **Linting:** ruff check + ruff format + mypy before commit
- **Pre-commit hooks:** Run automatically on git commit (ruff + mypy)
- **Testing:** `uv run pytest -m "not gpu and not slow"` must pass
- **Naming:** snake_case for files/functions, PascalCase for classes, SCREAMING_SNAKE for constants
- **Docstrings:** NumPy-style for new code
- **No mutable defaults:** Use `field(default_factory=...)` for dataclass fields
- **CombinationStrategy enum:** Use `str, Enum` pattern for CLI compatibility
- **Physics change protocol:** NOT triggered by this phase (no formulas/constants modified -- this is pure post-processing/analysis code)
- **GSD workflow:** This stays in GSD (not GPD) because it is software/analysis work, not physics formula changes

## Sources

### Primary (HIGH confidence)
- Direct inspection of `results/h_sweep_20260401/posteriors/*.json` -- 15 files analyzed programmatically
- `master_thesis_code/bayesian_inference/bayesian_statistics.py` -- JSON output format, `single_host_likelihood`, `p_D`, `evaluate` methods
- `master_thesis_code/arguments.py` -- existing CLI pattern for `--evaluate`, `--injection_campaign`
- `master_thesis_code/main.py` -- existing dispatch pattern for CLI subcommands
- `.planning/STATE.md` -- known MAP values and option assessments

### Secondary (MEDIUM confidence)
- NumPy log-shift-exp technique -- standard numerical analysis, well-documented in textbooks

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new dependencies, all NumPy/stdlib
- Architecture: HIGH - follows established patterns in existing codebase
- Pitfalls: HIGH - verified by direct data analysis of campaign results
- Zero-event analysis: HIGH - programmatically confirmed from JSON files

**Research date:** 2026-04-02
**Valid until:** 2026-05-02 (stable domain, no external dependency changes expected)
