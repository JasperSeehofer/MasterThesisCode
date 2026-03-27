# Roadmap: EMRI Parameter Estimation

## Milestones

- ✅ **v1.0 EMRI HPC Integration** — Phases 1-5 (shipped 2026-03-27)
- 🚧 **v1.1 Clean Simulation Campaign** — Phases 6-8 (in progress)

## Phases

<details>
<summary>v1.0 EMRI HPC Integration (Phases 1-5) -- SHIPPED 2026-03-27</summary>

- [x] Phase 1: Code Hardening (2/2 plans) — CPU-safe imports, --use_gpu/--num_workers CLI flags
- [x] Phase 2: Batch Compatibility (1/1 plan) — Non-interactive merge/prepare scripts with entry points
- [x] Phase 3: Cluster Environment (1/1 plan) — modules.sh + setup.sh for bwUniCluster 3.0
- [x] Phase 4: SLURM Job Infrastructure (3/3 plans) — simulate/merge/evaluate pipeline with dependency chaining
- [x] Phase 5: Documentation (2/2 plans) — cluster/README.md quickstart, CLAUDE.md/README.md sections

Full details: `.planning/milestones/v1.0-ROADMAP.md`

</details>

### v1.1 Clean Simulation Campaign

- [ ] **Phase 6: Data Cleanup** - Delete stale outputs and prevent future tracking of generated files
- [ ] **Phase 7: Cluster Access** - Establish SSH connectivity and verify cluster environment readiness
- [ ] **Phase 8: Simulation Campaign** - Run test simulation, evaluate H0 posterior, and validate results

## Phase Details

### Phase 6: Data Cleanup
**Goal**: Repository is free of stale simulation artifacts and configured to keep generated outputs out of version control
**Depends on**: Nothing (first phase of v1.1)
**Requirements**: DATA-01, DATA-02
**Plans:** 1 plan

Plans:
- [x] 06-01-PLAN.md — Remove stale evaluation/mean_bounds.xlsx from git tracking and verify .gitignore coverage

**Success Criteria** (what must be TRUE):
  1. No stale simulation outputs exist in the repository (`evaluation/mean_bounds.xlsx`, `run_metadata.json` removed from tracking)
  2. `git status` shows a clean working tree after deletion (no untracked evaluation artifacts)
  3. The `evaluation/` directory is listed in `.gitignore` so future simulation outputs are never accidentally committed

### Phase 7: Cluster Access
**Goal**: User can reach bwUniCluster from their local machine and Claude can execute cluster commands, with the environment verified ready for simulation
**Depends on**: Phase 6
**Requirements**: ACCESS-01, ACCESS-02, ACCESS-03, ACCESS-04
**Plans:** 1 plan

Plans:
- [ ] 07-01-PLAN.md — SSH config setup, key registration (human), and cluster environment preflight verification

**Success Criteria** (what must be TRUE):
  1. `ssh bwunicluster` connects without password prompt (SSH key registered and config entry working) -- human-verify: ACCESS-01, ACCESS-02 are manual user actions via bwUniCluster portal and local SSH config
  2. On the cluster: required modules load, the Python venv exists and activates, and `sinfo -p gpu_h100` confirms GPU partition is accessible
  3. Claude can execute commands on the cluster (via MCP server, SSH-based tool, or Bash) and receive output

### Phase 8: Simulation Campaign
**Goal**: A complete test simulation campaign produces validated Cramer-Rao bounds and a physically reasonable H0 posterior
**Depends on**: Phase 7
**Requirements**: SIM-01, SIM-02, SIM-03
**Success Criteria** (what must be TRUE):
  1. A test simulation run (5 tasks, 50-100 steps) completes successfully with timing data recorded in `run_metadata.json`
  2. The evaluation pipeline runs on the fresh Cramer-Rao bounds and produces an H0 posterior distribution
  3. Results pass sanity checks: SNR distribution is physical (peaked above threshold), detection rate is reasonable, and the H0 posterior peaks near the true value (0.73)
**Plans**: TBD

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Code Hardening | v1.0 | 2/2 | Complete | 2026-03-26 |
| 2. Batch Compatibility | v1.0 | 1/1 | Complete | 2026-03-26 |
| 3. Cluster Environment | v1.0 | 1/1 | Complete | 2026-03-27 |
| 4. SLURM Job Infrastructure | v1.0 | 3/3 | Complete | 2026-03-27 |
| 5. Documentation | v1.0 | 2/2 | Complete | 2026-03-27 |
| 6. Data Cleanup | v1.1 | 0/1 | Not started | - |
| 7. Cluster Access | v1.1 | 0/1 | Not started | - |
| 8. Simulation Campaign | v1.1 | 0/? | Not started | - |
