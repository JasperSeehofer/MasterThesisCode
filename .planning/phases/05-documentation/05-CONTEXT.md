# Phase 5: Documentation - Context

**Gathered:** 2026-03-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Document the cluster workflow so a new user (or future-you) can go from cluster login to running a full simulation campaign using only in-repo documentation. Three deliverables: `cluster/README.md` (quickstart + reference), a "Cluster Deployment" section in `CLAUDE.md`, and a "Running on HPC" section in `README.md`.

</domain>

<decisions>
## Implementation Decisions

### Audience & Tone
- **D-01:** Primary audience is a collaborator or supervisor — someone unfamiliar with the codebase but who knows SLURM basics (sbatch, squeue, sacct). No SLURM tutorial needed.
- **D-02:** Language is English, consistent with all existing documentation.
- **D-03:** Use visual elements: ASCII diagrams for the pipeline flow, directory structure trees, and callout boxes for warnings. More visual than the current README style.

### cluster/README.md Structure (DOCS-01)
- **D-04:** Structure: quickstart at the top (the 5 commands to run a campaign), then detailed reference sections below for each topic.
- **D-05:** Include a full worked example with real parameter values (e.g., "run 100 tasks x 50 steps with seed 42") — from login to results retrieval.
- **D-06:** Include an ASCII pipeline flow diagram showing: `submit_pipeline.sh` -> simulate (array) -> merge -> evaluate, with dependency arrows.
- **D-07:** Include a dev partition tip/callout: "Test with `dev_gpu_h100` first" showing how to run 1-2 tasks before submitting a large array.
- **D-08:** Sections follow the user journey: Prerequisites -> First-time Setup -> Running a Campaign -> Monitoring -> Results Retrieval -> Troubleshooting -> Script Reference.

### CLAUDE.md Cluster Section (DOCS-02)
- **D-09:** Add a "Cluster Deployment" section with: summary of what the `cluster/` directory contains, the key CLI flags (`--use_gpu`, `--num_workers`), and pointer to `cluster/README.md` for full guide. No command duplication.
- **D-10:** Include a one-liner-per-script inventory table/list: `modules.sh` — loads env modules, `setup.sh` — first-time setup, `simulate.sbatch` — GPU array job, etc. Gives Claude Code quick context.

### README.md HPC Section (DOCS-03)
- **D-11:** Add a "Running on HPC" section that points to `cluster/README.md`. Brief — just enough to let the reader know cluster support exists and where to find the guide.

### Troubleshooting & Failure Guidance
- **D-12:** Cover common failures: OOM kills, timeout failures, `resubmit_failed.sh` usage, CUDA errors. Practical and focused on known risks.
- **D-13:** Workspace expiration gets a prominent warning box (admonition-style) at the top of relevant sections. Include `ws_extend` command so the reader can act immediately.
- **D-14:** Include a brief log inspection guide: where SLURM output/error files land, key patterns to grep for (OOM, CUDA error, Python traceback).

### Claude's Discretion
- Exact ASCII diagram style and layout
- Heading levels and markdown formatting details
- Whether to use blockquote admonitions (`> **Warning:**`) or other callout styles
- Ordering within the script reference section
- How much detail in the monitoring section (squeue, sacct examples)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Cluster Scripts (document these)
- `cluster/modules.sh` — Loads environment modules, exports `$WORKSPACE`, `$PROJECT_ROOT`, `$VENV_PATH`
- `cluster/setup.sh` — First-time cluster setup (uv, workspace, venv)
- `cluster/simulate.sbatch` — GPU array job for EMRI simulation
- `cluster/merge.sbatch` — CPU job for merging per-task CSVs
- `cluster/evaluate.sbatch` — CPU job for Bayesian inference
- `cluster/submit_pipeline.sh` — Pipeline orchestrator chaining all three jobs
- `cluster/resubmit_failed.sh` — Failure recovery helper
- `cluster/vpn.sh` — VPN helper script

### Existing Documentation (update these)
- `CLAUDE.md` — Add "Cluster Deployment" section (DOCS-02)
- `README.md` — Add "Running on HPC" section (DOCS-03)

### Prior Phase Context
- `.planning/phases/03-cluster-environment/03-CONTEXT.md` — modules.sh and setup.sh decisions
- `.planning/phases/04-slurm-job-infrastructure/04-CONTEXT.md` — sbatch scripts, pipeline orchestrator, output directory structure, seed strategy

### Requirements
- `.planning/REQUIREMENTS.md` — DOCS-01, DOCS-02, DOCS-03

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `cluster/` directory already contains all 8 scripts to document — no new scripts needed
- Existing `README.md` has installation and usage sections that set the style precedent
- `CLAUDE.md` has extensive architecture and convention docs to follow for section style

### Established Patterns
- README.md uses markdown with code blocks, heading hierarchy, and bullet lists
- CLAUDE.md uses tables for structured info (e.g., Known Bugs, Trigger rules)
- No existing ASCII diagrams in the repo — this will be a new visual element

### Integration Points
- `README.md` "Running on HPC" section should sit after "Running the Code" and before "Running Tests"
- `CLAUDE.md` "Cluster Deployment" section should sit after "Running the Code" section
- `cluster/README.md` is a new standalone file

</code_context>

<specifics>
## Specific Ideas

- Full worked example with concrete values (100 tasks, 50 steps, seed 42) — not just placeholders
- Dev partition tip as a callout box for quick sanity checks before large submissions
- Workspace expiration warning as a prominent admonition, not buried in prose
- `ws_extend` command included alongside the warning so reader can act immediately
- Log file inspection guide with grep patterns for common failure signatures

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-documentation*
*Context gathered: 2026-03-27*
