# Phase 3: Cluster Environment - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-26
**Phase:** 03-cluster-environment
**Areas discussed:** Module versions, uv installation, Workspace integration, Python version strategy

---

## Module Versions

| Option | Description | Selected |
|--------|-------------|----------|
| I know the modules | User provides exact module names; scripts hardcode them | ~partial |
| Best-guess with verification | Scripts use likely names and run module avail checks | |
| Parameterized | modules.sh reads module names from a config file | |

**User's choice:** "I know the modules" initially, then delegated to web research. Best-guess names from bwHPC wiki accepted: `compiler/gnu/14.2`, `devel/cuda/12.8`, `numlib/gsl`, `devel/python/3.13.3-gnu-14.2`.
**Notes:** GSL version string not publicly documented. User confirmed these look good for now, with on-cluster verification needed later.

---

## uv Installation

| Option | Description | Selected |
|--------|-------------|----------|
| Official installer | curl pipe-to-shell from astral.sh | |
| Pre-built binary download | Download specific release tarball | |
| Check-and-skip | Check if uv exists, only install if missing | ✓ (combo) |

**User's choice:** Check-and-skip first, fall back to official installer if missing.
**Notes:** User explicitly said "it should be 3 and if not installed 1" — combining idempotency with the simplest install method.

---

## Workspace Integration

### Allocation Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Allocate in setup.sh | setup.sh calls ws_allocate automatically | |
| Expect pre-existing | User must manually allocate before running setup.sh | |
| Check-and-allocate | Try ws_find first; allocate if not found | ✓ |

**User's choice:** Check-and-allocate
**Notes:** Idempotent — safe to re-run setup.sh.

### Path Resolution

| Option | Description | Selected |
|--------|-------------|----------|
| modules.sh exports $WORKSPACE | Single source of truth, all scripts use $WORKSPACE | ✓ |
| Each script calls ws_find | Independent resolution per script | |
| Hardcoded in config file | setup.sh writes path to a config file | |

**User's choice:** modules.sh exports $WORKSPACE

### Workspace Parameters

| Option | Description | Selected |
|--------|-------------|----------|
| emri, 60 days | ws_allocate emri 60 | ✓ |
| master_thesis, 60 days | Broader name for other thesis work | |
| Custom | User specifies | |

**User's choice:** emri, 60 days

---

## Python Version Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Cluster module | module load devel/python/3.13.3-gnu-14.2 | ✓ |
| uv-managed Python | uv python install 3.13 | |
| Either with fallback | Try module first, fall back to uv | |

**User's choice:** Cluster module
**Notes:** Uses cluster-optimized build, avoids extra downloads.

---

## Claude's Discretion

- Error handling and messaging in setup.sh
- Convenience variables in modules.sh ($PROJECT_ROOT, $VENV_PATH)
- Verification smoke test at end of setup.sh
- Script structure and logging style

## Deferred Ideas

- Apptainer container (v2 requirement CONT-01/CONT-02)
- Workspace expiration warnings (Phase 5 documentation)
