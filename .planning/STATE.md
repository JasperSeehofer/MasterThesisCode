---
gsd_state_version: 1.0
milestone: v1.5
milestone_name: Galaxy Catalog Completeness Correction
status: roadmap_complete
stopped_at: Roadmap written — Phase 24 ready to plan
last_updated: "2026-04-04"
last_activity: 2026-04-04
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-04)

**Core value:** The simulation pipeline runs reliably on the GPU cluster, producing enough Cramér-Rao bounds for statistically meaningful Hubble constant posteriors.
**Current focus:** v1.5 — Galaxy catalog completeness correction to eliminate H0 posterior bias (MAP=0.66 -> target h=0.73)

## Current Position

Phase: 24 (Completeness Estimation) — not started
Plan: —
Status: Roadmap complete, ready to plan Phase 24
Last activity: 2026-04-04 — v1.5 roadmap written (Phases 24-27)

Progress: [░░░░░░░░░░] 0% (0/4 phases)

## Performance Metrics

**Velocity:**

- Total plans completed: 41 (v1.0: 9, v1.1: 4, v1.2: 12, v1.3: 11, v1.4: 5)
- Total phases: 27 (23 complete across v1.0-v1.4, 4 planned for v1.5)

## Accumulated Context

### Pending Todos

- Fix galaxy catalog unconditional init blocking generate-figures (`main.py:49`)

### Blockers/Concerns

None.

### Key Context for v1.5

- **Root cause confirmed:** GLADE+ catalog incompleteness at z > 0.08 causes systematic H0 bias (MAP=0.66 vs true h=0.73). See `scripts/bias_investigation/FINDINGS.md`.
- **Research specification:** `.gpd/quick/3-literature-research-galaxy-catalog-in/galaxy-catalog-completeness-research.md` (Gray et al. 2020 framework, full implementation spec)
- **Physics changes required:** LIKE-01/02/03/04 all modify computed values in `bayesian_statistics.py` and `physical_relations.py` — Physics Change Protocol mandatory before any code changes
- **Injection data available:** `simulations/injections/` contains 7 h-value injection campaigns (fetched from cluster)
- **Baseline to beat:** Pre-correction MAP = 0.66 on 534-detection dataset. Target: MAP visibly closer to h=0.73.
- **Limiting case checkpoints:** f=1 must recover current code exactly; f=0 must produce broad posterior near h=0.73

### Phase Notes

**Phase 24 (Completeness Estimation):**
- Use B-band luminosity comparison (Approach A from research doc): sum GLADE+ B-band luminosities in z-shells vs expected from j_B * dV_c
- j_B = (1.98 +/- 0.16) x 10^{-2} L_10 Mpc^{-3} (Dalya et al. 2022)
- Expected shape: ~90% at z~0.029, drops to <<50% at z>0.11
- Output: interpolating function f(z) for range z=0.0 to z=0.25

**Phase 25 (Likelihood Correction — PHYSICS):**
- GPD routing required for LIKE-01/02/03/04 (physics changes)
- Add dV_c/dz/dOmega to physical_relations.py (comoving volume element)
- Add L_comp^i(h) computation in p_Di() (completion term integral)
- Modify p_Di() to return f_i * L_cat^i + (1 - f_i) * L_comp^i
- Thread f(z) through evaluate() -> p_D() -> p_Di()
- All changes: Physics Change Protocol + arXiv:1908.06050 reference comments

**Phase 26 (Verification):**
- Four quantitative checks: f=1 recovery, f=0 statistical siren, 534-detection bias shift, +/-20% sensitivity
- Use existing 534-detection dataset (seed 200 campaign)

**Phase 27 (Cluster Deployment):**
- SSH to bwUniCluster, git pull, run evaluate job
- Record commit hash and MAP result in this STATE.md

## Session Continuity

Last session: 2026-04-04
Stopped at: v1.5 roadmap created — Phase 24 ready to plan
Resume file: None
