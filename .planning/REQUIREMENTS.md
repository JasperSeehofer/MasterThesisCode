# Requirements: EMRI Parameter Estimation — Clean Simulation Campaign

**Defined:** 2026-03-27
**Core Value:** The simulation pipeline runs reliably on the GPU cluster as SLURM array jobs, producing enough Cramér-Rao bounds for statistically meaningful Hubble constant posteriors.

## v1.1 Requirements

Requirements for the Clean Simulation Campaign milestone. Each maps to roadmap phases.

### Data Cleanup

- [ ] **DATA-01**: Stale simulation outputs deleted (`evaluation/mean_bounds.xlsx`, `run_metadata.json`) and repo verified clean
- [ ] **DATA-02**: `evaluation/` directory added to `.gitignore` to prevent future tracking of generated outputs

### Cluster Access

- [ ] **ACCESS-01**: SSH key registered via bwUniCluster user portal for passwordless authentication
- [ ] **ACCESS-02**: `~/.ssh/config` entry configured for direct `ssh bwunicluster` access
- [ ] **ACCESS-03**: Environment preflight verified (modules load, venv exists, GPU partition accessible)
- [ ] **ACCESS-04**: Claude SSH integration configured (MCP server or Bash-based) for direct cluster command execution

### Simulation Campaign

- [ ] **SIM-01**: Test simulation run completed (5 tasks, 50-100 steps) with timing data recorded
- [ ] **SIM-02**: Evaluation pipeline run on fresh Cramér-Rao bounds produces H₀ posterior
- [ ] **SIM-03**: Results validated (SNR distributions physical, detection rates reasonable, posterior sanity-checked)

## Future Requirements

Deferred to subsequent milestones.

### Production Campaign

- **PROD-01**: Production simulation run scaled to available GPU hours
- **PROD-02**: Clean cluster workspace between campaigns

### Performance

- **PERF-01**: Profiling infrastructure for GPU simulation pipeline
- **PERF-02**: Performance optimization based on profiling results

### Physics Corrections

- **PHYS-01**: Fisher matrix upgraded to 5-point stencil derivative
- **PHYS-02**: Galactic confusion noise added to PSD
- **PHYS-03**: wCDM parameters properly integrated into distance function

## Out of Scope

| Feature | Reason |
|---------|--------|
| Production-scale campaign (100+ tasks) | Requires validated test run first; deferred to next milestone |
| Physics bug fixes | Tracked in CLAUDE.md; separate milestone for correctness improvements |
| Profiling infrastructure | First test run serves as natural baseline; only invest if bottleneck found |
| Multi-node MPI distribution | Array jobs provide sufficient parallelism |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | — | Pending |
| DATA-02 | — | Pending |
| ACCESS-01 | — | Pending |
| ACCESS-02 | — | Pending |
| ACCESS-03 | — | Pending |
| ACCESS-04 | — | Pending |
| SIM-01 | — | Pending |
| SIM-02 | — | Pending |
| SIM-03 | — | Pending |

**Coverage:**
- v1.1 requirements: 9 total
- Mapped to phases: 0
- Unmapped: 9 (will be mapped during roadmap creation)

---
*Requirements defined: 2026-03-27*
*Last updated: 2026-03-27 after initial definition*
