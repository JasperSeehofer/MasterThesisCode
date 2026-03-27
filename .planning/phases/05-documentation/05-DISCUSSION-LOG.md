# Phase 5: Documentation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-27
**Phase:** 05-documentation
**Areas discussed:** Audience & tone, cluster/README.md scope & structure, CLAUDE.md cluster section, Troubleshooting & failure guidance

---

## Audience & Tone

### Primary audience

| Option | Description | Selected |
|--------|-------------|----------|
| Future-me only | Terse, reference-style. Assumes SLURM knowledge and codebase familiarity. | |
| Collaborator / supervisor | More context: explains WHY the pipeline is structured this way. | ✓ |
| Full onboarding guide | Tutorial-style with background on EMRI simulations. | |

**User's choice:** Collaborator / supervisor
**Notes:** None

### SLURM knowledge level

| Option | Description | Selected |
|--------|-------------|----------|
| Assume SLURM basics | Reader knows sbatch/squeue/sacct. Docs show project-specific commands. | ✓ |
| Brief SLURM primer | Include a short refresher box with the 5-6 commands used. | |

**User's choice:** Assume SLURM basics
**Notes:** None

### Language

| Option | Description | Selected |
|--------|-------------|----------|
| English | Consistent with all existing code and documentation. | ✓ |
| German | If supervisor/collaborators prefer German. | |

**User's choice:** English
**Notes:** None

### Formatting

| Option | Description | Selected |
|--------|-------------|----------|
| Match existing README style | Markdown with code blocks, same heading structure. | |
| More visual | Add ASCII diagrams for pipeline flow, directory structure trees. | ✓ |

**User's choice:** More visual
**Notes:** None

---

## cluster/README.md Scope & Structure

### Structure approach

| Option | Description | Selected |
|--------|-------------|----------|
| Workflow-ordered | Sections follow user journey: Prerequisites -> Setup -> Running -> Monitoring -> Results -> Troubleshooting. | |
| Script-by-script reference | One section per script with usage, flags, examples. | |
| Both: quickstart + reference | Short quickstart at top, then detailed reference sections below. | ✓ |

**User's choice:** Both: quickstart + reference
**Notes:** None

### Worked example

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, full worked example | Exact commands from login to results retrieval with real parameter values. | ✓ |
| Just command templates | Commands with placeholders like <N_TASKS>. | |

**User's choice:** Full worked example
**Notes:** None

### Dev partition

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, as a tip | Short callout box: "Test with dev partition first". | ✓ |
| Skip it | Only document production workflow. | |

**User's choice:** Yes, as a tip
**Notes:** None

### Pipeline diagram

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, ASCII diagram | ASCII art showing pipeline stages with dependency arrows. | ✓ |
| Just describe in text | Explain pipeline stages in prose. | |

**User's choice:** Yes, ASCII diagram
**Notes:** None

---

## CLAUDE.md Cluster Section

### Detail level

| Option | Description | Selected |
|--------|-------------|----------|
| Summary + pointer | "Cluster Deployment" section with key info + pointer to cluster/README.md. No duplication. | ✓ |
| Duplicate key commands | Include quickstart commands in CLAUDE.md too. Some duplication but convenient. | |
| Minimal pointer only | Just "see cluster/README.md" under a new heading. | |

**User's choice:** Summary + pointer
**Notes:** None

### Script inventory

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, one-liner per script | Table/list with one-liner descriptions per script. Quick context for Claude Code. | ✓ |
| No, keep in cluster/README.md only | CLAUDE.md just says "cluster/ contains SLURM job scripts". | |

**User's choice:** One-liner per script
**Notes:** None

---

## Troubleshooting & Failure Guidance

### Coverage level

| Option | Description | Selected |
|--------|-------------|----------|
| Common failures + workspace | OOM kills, timeouts, resubmit_failed.sh, workspace expiration, job output inspection. | ✓ |
| Minimal — just resubmit | Only document resubmit_failed.sh usage. | |
| Comprehensive FAQ | Full troubleshooting FAQ covering all possible errors. | |

**User's choice:** Common failures + workspace
**Notes:** None

### Workspace expiration emphasis

| Option | Description | Selected |
|--------|-------------|----------|
| Prominent warning box | Callout/admonition at top of relevant sections. | ✓ |
| Just mention in text | Note it without special emphasis. | |

**User's choice:** Prominent warning box
**Notes:** STATE.md flagged workspace expiration as an operational risk

### Log file inspection

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, brief guide | Show where SLURM logs land and key grep patterns. | ✓ |
| Skip — reader knows SLURM logs | Assume reader can find slurm-*.out files. | |

**User's choice:** Yes, brief guide
**Notes:** None

### ws_extend command

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, include the command | Show ws_extend usage alongside expiration warning. Actionable. | ✓ |
| Just warn, no fix | Warn about expiration, let reader look up ws_extend. | |

**User's choice:** Yes, include the command
**Notes:** None

---

## Claude's Discretion

- ASCII diagram style and layout
- Heading levels and markdown formatting details
- Callout/admonition style (blockquote vs other)
- Script reference section ordering
- Monitoring section depth (squeue/sacct examples)

## Deferred Ideas

None — discussion stayed within phase scope
