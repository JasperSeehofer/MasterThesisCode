# Phase 34: Fisher Matrix Quality - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-09
**Phase:** 34-fisher-matrix-quality
**Areas discussed:** Detection strategy, Handling policy, Threshold calibration, Diagnostic output

---

## Detection Strategy

### Where should degeneracy detection happen?

| Option | Description | Selected |
|--------|-------------|----------|
| Evaluation time | Check condition number when building covariance matrices in bayesian_statistics.py (~line 427). No re-simulation needed. | ✓ |
| Simulation time | Flag during Fisher matrix inversion in parameter_estimation.py (~line 391). Would need quality flag in CRB CSV. | |
| Both stages | Log at simulation time AND enforce at evaluation time. Double coverage but more complexity. | |

**User's choice:** Evaluation time
**Notes:** None

### What metric should identify a degenerate covariance matrix?

| Option | Description | Selected |
|--------|-------------|----------|
| Condition number | np.linalg.cond(cov). Standard measure — ratio of largest to smallest eigenvalue. Easy to threshold. | ✓ |
| Smallest eigenvalue | Flag if min eigenvalue < epsilon. More direct but scale-dependent. | |
| slogdet sign check | Already called at line 428. Catches true singularity but misses near-singular. | |

**User's choice:** Condition number
**Notes:** None

### Should both 3D and 4D covariance be checked?

| Option | Description | Selected |
|--------|-------------|----------|
| Both independently | Check cond(cov_3d) and cond(cov_4d) separately. An event could have well-conditioned 3D but degenerate 4D. | ✓ |
| 4D only | Only check the 4D matrix. If 4D is healthy, 3D is likely fine too. | |
| 3D only | Only check the 3D matrix. Ignores BH mass quality entirely. | |

**User's choice:** Both independently
**Notes:** None

---

## Handling Policy

### What should happen to events flagged as degenerate?

| Option | Description | Selected |
|--------|-------------|----------|
| Exclude from posterior | Skip flagged events in the likelihood product. Clean and safe. | ✓ |
| Regularize (ridge) | Add small diagonal to covariance before inversion. Keeps all events. | |
| Keep but downweight | Include with reduced weight based on condition number. | |
| Configurable strategy | CLI flag --fisher_strategy exclude\|regularize. | |

**User's choice:** Exclude + debug plot
**Notes:** User emphasized that physically and mathematically it doesn't make sense for the Fisher matrix to become singular. The primary goal is to understand WHY it happens. A debugging plot showing all singular events should help identify the root cause. Exclusion is the safe handling while investigating.

### What should the debugging plot show?

| Option | Description | Selected |
|--------|-------------|----------|
| Eigenvalue spectrum | Bar chart of eigenvalues per flagged event. Shows which direction is degenerate. | |
| Parameter scatter | Scatter of flagged events in (d_L, SNR, M) space. Shows physical correlation. | |
| Both | Two-panel figure with eigenvalue spectrum + parameter scatter. | ✓ |

**User's choice:** Both panels
**Notes:** None

### Should the plot be generated automatically every run?

| Option | Description | Selected |
|--------|-------------|----------|
| Always | Generate every run. Cheap (only flagged events). Never miss a regression. | ✓ |
| Opt-in flag | Only with --fisher_diagnostics. Keeps output clean. | |
| Only if events excluded | Auto-generate only when events are flagged. | |

**User's choice:** Always
**Notes:** None

---

## Threshold Calibration

### How should the condition number threshold be set?

| Option | Description | Selected |
|--------|-------------|----------|
| Empirical from data | Run evaluation, collect condition numbers, find the gap. Document the choice. | ✓ |
| Fixed conservative default | Use 1e12 (standard numerical threshold for double precision). | |
| CLI-configurable | Add --fisher_cond_threshold with sensible default. | |

**User's choice:** Empirical from data
**Notes:** None

### Should the threshold be CLI-configurable after empirical default is chosen?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, with empirical default | Add --fisher_cond_threshold with empirically determined default. Follows Phase 33 pattern. | ✓ |
| Hardcode the empirical value | Set as module constant. Simpler, fewer CLI flags. | |

**User's choice:** Yes, configurable with empirical default
**Notes:** None

### Same threshold for 3D and 4D?

| Option | Description | Selected |
|--------|-------------|----------|
| Same threshold | One threshold applies to both. If either exceeds, event is excluded from both branches. | ✓ |
| Separate thresholds | Different thresholds for 3D and 4D. More nuanced. | |

**User's choice:** Same threshold (start unified)
**Notes:** User expects they'll be the same, but if empirical data tells a different story, deviate to separate thresholds. Start unified, let data guide.

---

## Diagnostic Output

### What summary should be logged per evaluation run?

| Option | Description | Selected |
|--------|-------------|----------|
| Count + worst cases | Total events, flagged count, excluded count, worst condition numbers (top 5). | ✓ |
| Full per-event table | Every event's condition number at INFO level. Complete but verbose. | |
| Count only | Just "N of M events excluded". Minimal. | |

**User's choice:** Count + worst cases
**Notes:** None

### Should per-event condition numbers be written to a file?

| Option | Description | Selected |
|--------|-------------|----------|
| CSV alongside posteriors | fisher_quality.csv with detection_index, cond_3d, cond_4d, excluded columns. | ✓ |
| Add to baseline JSON | Include stats in Phase 30 baseline.json. | |
| No file export | Only log to console. | |

**User's choice:** CSV alongside posteriors
**Notes:** None

### Should the Phase 30 comparison report include Fisher quality metrics?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, add section | Fisher Quality section: events excluded before vs after, condition number distribution shift. | ✓ |
| Separate report | Own standalone report. | |
| No integration | Fisher quality stays in its own CSV and logs. | |

**User's choice:** Yes, add section to comparison report
**Notes:** None

---

## Claude's Discretion

- Exact eigenvalue visualization style (grouped bars, stacked, or per-event subplots)
- Whether to use apply_style() theming on the debug plot
- Module placement for Fisher quality utilities
- How to compute empirical threshold (gap detection, percentile, or manual inspection)

## Deferred Ideas

None — discussion stayed within phase scope.
