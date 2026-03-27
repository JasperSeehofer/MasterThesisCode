# Phase 8: Simulation Campaign - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-03-28
**Phase:** 08-simulation-campaign
**Areas discussed:** Campaign parameters, Validation criteria, Monitoring & failure response, Result handling

---

## Campaign Parameters

### Steps per task

| Option | Description | Selected |
|--------|-------------|----------|
| 50 steps (Recommended) | Lower bound of roadmap range. Faster turnaround. | |
| 100 steps | Upper bound. Better statistics but longer wall time. | |
| 25 steps (smoke test) | Minimal run to confirm pipeline chain works. | ✓ |

**User's choice:** 25 steps (smoke test)
**Notes:** Deliberately minimal to validate infrastructure before investing GPU hours.

### Number of tasks

| Option | Description | Selected |
|--------|-------------|----------|
| 5 tasks (Recommended) | Matches roadmap spec. | |
| 3 tasks | Minimal GPU usage, still tests array + merge. | ✓ |
| 10 tasks | More data points, more GPU allocation. | |

**User's choice:** 3 tasks
**Notes:** Smallest viable array to test the full pipeline chain.

### Base seed

| Option | Description | Selected |
|--------|-------------|----------|
| 42 (Recommended) | Classic reproducible seed, matches cluster/README.md example. | ✓ |
| 1000 | Round number, leaves room for future campaigns. | |
| You decide | Claude picks a sensible default. | |

**User's choice:** 42

### Scale-up after smoke test

| Option | Description | Selected |
|--------|-------------|----------|
| Smoke test only | Phase 8 = validate pipeline. Production deferred. | ✓ |
| Scale up if smoke passes | Submit larger run in same phase. | |
| Decide after results | Run smoke first, then discuss. | |

**User's choice:** Smoke test only
**Notes:** Production-scale campaign explicitly deferred to future milestone (PROD-01).

---

## Validation Criteria

### Validation level

| Option | Description | Selected |
|--------|-------------|----------|
| Pipeline completion only (Recommended) | Just confirm chain runs without errors. | |
| Basic sanity checks | Pipeline completes + positive SNR, >0 detections, valid H0 distribution. | |
| Full validation | Quantitative thresholds on all outputs. | ✓ |

**User's choice:** Full validation
**Notes:** Full quantitative checks even with small sample size.

### SNR validation

| Option | Description | Selected |
|--------|-------------|----------|
| All SNR values positive + at least 1 detection | Minimal check. | |
| Detection rate in expected range | Check 1-30% detection fraction. | ✓ |
| You decide | Claude determines based on sample size. | |

**User's choice:** Detection rate in expected range (1-30%)

### H0 posterior validation

| Option | Description | Selected |
|--------|-------------|----------|
| Posterior peaks in [0.5, 1.0] | Broad check, physically reasonable. | |
| Posterior peaks in [0.6, 0.9] | Tighter check, neighborhood of 0.73. | ✓ |
| You decide | Claude determines based on detections and uncertainty. | |

**User's choice:** Posterior peaks in [0.6, 0.9]

---

## Monitoring & Failure Response

### Monitoring approach

| Option | Description | Selected |
|--------|-------------|----------|
| Poll sacct periodically (Recommended) | Simple, uses existing tools. | |
| Tail log files | Real-time output, more detailed but noisier. | |
| Both | sacct for overview, tail logs for diagnostics. | ✓ |

**User's choice:** Both

### Failure response

| Option | Description | Selected |
|--------|-------------|----------|
| Investigate first (Recommended) | Read .err log, diagnose, report before resubmitting. | ✓ |
| Auto-resubmit once | Retry immediately, investigate on second failure. | |
| Abort and debug | Any failure = stop and debug. | |

**User's choice:** Investigate first

---

## Result Handling

### Copy results locally

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, rsync back (Recommended) | Copy key files after validation. Workspace expires in 60 days. | ✓ |
| Leave on cluster | Results stay in $WORKSPACE only. | |
| You decide | Claude judges based on smoke test outcome. | |

**User's choice:** Yes, rsync back

### Local storage path

| Option | Description | Selected |
|--------|-------------|----------|
| evaluation/ directory (Recommended) | Already in .gitignore. Subdirectory per run. | ✓ |
| A new results/ directory | Separate from evaluation/. Needs .gitignore update. | |
| You decide | Claude picks based on conventions. | |

**User's choice:** evaluation/ directory

---

## Claude's Discretion

- Polling interval for sacct monitoring
- Exact rsync flags and file selection
- Validation result presentation format
- Whether to generate local plots from rsynced results

## Deferred Ideas

- Production-scale campaign (PROD-01) -- future milestone
- Performance profiling (PERF-01, PERF-02) -- future milestone
- Local plot generation from results -- not in scope
