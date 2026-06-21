---
phase: 40-verification-gate
plan: "00"
status: COMPLETE
date: 2026-04-23
requirements: []
tags: [wave-0, archive, preflight]
---

# Phase 40 Plan 00: v2.1 Baseline Archive — Summary

**One-liner:** Archived pre-v2.2 posteriors to `simulations/_archive_v2_1_baseline/` before any Phase 40 re-evaluation.

## What was done

- Created `cluster/scripts/archive_v2_1_baseline.sh` — idempotent bash script that
  copies `simulations/{combined_posterior.json, combined_posterior_with_bh_mass.json,
  posteriors/, posteriors_with_bh_mass/}` to `simulations/_archive_v2_1_baseline/`
  and writes `ARCHIVE_MANIFEST.md` with git-commit + sha256 provenance.
- Script fails loudly (`exit 1`) if the archive already exists (D-09).
- Executed the script; archive verified byte-exact vs source (sha256 + diff -q per-h loop).
- Re-run idempotency guard verified: second invocation exits 1 with `Archive already exists`.
- Confirmed archive path is git-ignored (`simulations/` in .gitignore); only the script committed.

## Artifacts

| Path | Detail |
|------|--------|
| `cluster/scripts/archive_v2_1_baseline.sh` | 72-line idempotent bash script (committed) |
| `simulations/_archive_v2_1_baseline/combined_posterior.json` | byte-exact copy (git-ignored) |
| `simulations/_archive_v2_1_baseline/combined_posterior_with_bh_mass.json` | byte-exact copy (git-ignored) |
| `simulations/_archive_v2_1_baseline/posteriors/` | 38 h-value files, all byte-exact (git-ignored) |
| `simulations/_archive_v2_1_baseline/posteriors_with_bh_mass/` | 38 h-value files, all byte-exact (git-ignored) |
| `simulations/_archive_v2_1_baseline/ARCHIVE_MANIFEST.md` | git-commit + sha256 provenance (git-ignored) |

## Provenance

- git_commit at archive time: `1df6e8cc9dc770e605050597ed3b1f7aaf78858f`
- Archive timestamp (UTC): `2026-04-23T17:21:33Z`
- sha256(combined_posterior.json): `7375dfc846447853ae4f26fceb0cc5248826495aed450dc9e50e75611b28f66b`
- sha256(combined_posterior_with_bh_mass.json): `59ba1353413a0bd42f9506e0e3765f3e0ad429a4db5673d45ee9831b9482edd0`

## Self-Check: PASSED

- ✓ Archive exists with all required paths
- ✓ All 38 per-h files byte-exact (diff -q loop)
- ✓ Manifest contains `git_commit:` line and sha256 entries
- ✓ Idempotency guard fires correctly on second run
- ✓ Archive git-ignored; only script committed

## Commits

| Hash | Message |
|------|---------|
| `1df6e8c` | feat(40-00): add v2.1 baseline archive script (Task 1) |

## Next

Wave 1: Plan 40-01 (VERIFY-01 — full CPU test-suite gate).
