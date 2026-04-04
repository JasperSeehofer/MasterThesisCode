# Phase 12: Production Campaign - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-01
**Phase:** 12-production-campaign
**Areas discussed:** Campaign Scale, d_L Error Threshold, Failure Handling & Monitoring, Data Transfer & Persistence

---

## Campaign Scale

| Option | Description | Selected |
|--------|-------------|----------|
| 100 tasks x 50 steps (5,000 events) | Conservative, ~1,000-2,000 detections | ✓ |
| 200 tasks x 50 steps (10,000 events) | Larger catalog, more GPU-hours | |
| 100 tasks x 100 steps (10,000 events) | Same count, fewer SLURM jobs | |

**User's choice:** 100 tasks x 50 steps — "5k -> 1k detections is more than fine for the beginning"
**Notes:** Seed 200 chosen to avoid overlap with Phase 11 (seed 100) and Phase 11.1 injections (seed 12345).

---

## d_L Error Threshold

| Option | Description | Selected |
|--------|-------------|----------|
| Apply Phase 11 recommendation before running | Update threshold in code first | |
| Keep 10%, save all CRB data, re-filter later | Preserve raw data alongside filtered | |
| Remove filter entirely for campaign | Save everything, filter at evaluation time | ✓ |

**User's choice:** Remove filter entirely — "we don't want to filter in the simulation part"
**Notes:** Rationale: "only the best detections define the shape of the posterior" — threshold has no strong physical argument, so tuning it per-evaluation is preferred.

---

## Failure Handling & Monitoring

| Option | Description | Selected |
|--------|-------------|----------|
| Resubmit once, accept losses | Single resubmission pass | |
| Resubmit until complete | Keep resubmitting until all 100 succeed | |
| Set a yield target | Monitor detection count, stop at ~1,000+ | ✓ |

**User's choice:** Yield-driven approach
**Notes:** None.

---

## Data Transfer & Persistence

| Option | Description | Selected |
|--------|-------------|----------|
| rsync into repo | Pull CSV + metadata into project directory | |
| rsync to local directory outside repo | Keep large data separate | ✓ |
| Copy to persistent cluster storage | Backup on cluster, then rsync | |

**User's choice:** rsync locally outside the repo for now
**Notes:** Git LFS mentioned as future consideration for integrating production data into the repo.

---

## Claude's Discretion

- rsync flags and target directory structure
- Detection yield monitoring approach
- Whether to skip the evaluate step in submit_pipeline.sh
- Task batching strategy

## Deferred Ideas

- Git LFS integration for production data
- Larger campaign (10,000+ events) if initial yield is insufficient
