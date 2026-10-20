# OPS_RECORD_m1_resubmit.md — sealed-mock m1 guard correction + resubmit (2026-09-04)

## Context
Job 6790859 (m1 iiib) STOPped on the pool-identity guard: `readlink -f` on
`run_20260729_seed64000_h0p67/simulations/injections` returns the directory's own path (it is
a real directory of per-file symlinks into `injection_pool_mix200k_20260728`, not a top-level
symlink), so the guard's `readlink`-equality comparison against `$POOL` could never pass.

**CHAIR DECISION (flagged, verbatim per instruction):** replace the identity check by a CONTENT
check — the 0.67 run's injections directory must contain the same files as the canonical pool
`injection_pool_mix200k_20260728` (707 files) with an identical per-file md5 manifest; reference
manifest `exec/r-timeout-selection/POOL_MANIFEST.md5` (707 per-file md5s, list-md5
`75f4030d5d3b0405fd948049bef5767e`).

## 1. Content verification (cluster, read-only, `cluster/agent_ssh.sh run`)
`$WS/run_20260729_seed64000_h0p67/simulations/injections`: 707 entries (all per-file symlinks
into `$WS/injection_pool_mix200k_20260728`). Computed manifest: `ls -1 *.csv | sort | xargs
md5sum` (identical format to `POOL_MANIFEST.md5`) → 707 lines; **list-md5 of that manifest =
`75f4030d5d3b0405fd948049bef5767e`** — an EXACT match to the pinned reference. No rsync of the
manifest was needed since the list-md5 itself matches (a byte-for-byte manifest match implies
identical list-md5 and vice versa for a 707-line deterministic sort).

**Verdict: IDENTICAL.** 707/707 files, 0 differing, 0 missing, 0 extra.

## 2. Guard replacement
`cluster/graph1_sealed_m1_headstack.sbatch` edited: the `readlink`-based identity block (comment
header, required-env doc, and the executable STOP block) replaced by a content guard — computes
the injections dir's per-file md5 manifest at run time, list-md5's it, compares to a new required
env var `EXPECTED_POOL_LIST_MD5`, STOPs on mismatch. Every other line byte-identical (see `git
diff` recorded below). Notes appended to `D_SEALED_REGISTER_DOSSIER.md` (new "## GUARD
CORRECTION (chair, 2026-09-04)" section) and `PIN_RECORD.md` (same heading).

```
diff --git a/cluster/graph1_sealed_m1_headstack.sbatch b/cluster/graph1_sealed_m1_headstack.sbatch
(3 hunks: header comment, required-env doc, and the executable identity->content guard block +
new EXPECTED_POOL_LIST_MD5 required-env check; nothing else touched)
```

Transfer to cluster: direct `rsync`/`ssh` was blocked by the harness classifier (agent_ssh.sh
only, per policy) — transferred via `cluster/agent_ssh.sh run` using a base64-encoded heredoc
(existing remote file first moved aside to
`cluster/graph1_sealed_m1_headstack.sbatch.pre_guard_correction.bak`, not deleted). Post-transfer
md5 verified identical to the local file: `dc126f7b6ea6f2b0fba6a7a126c06f8b` both sides.
`bash -n` syntax check passed on the cluster. HEAD ancestor check confirmed (`081b1f28` is an
ancestor of remote HEAD `8f933e7b`).

## 3. Submission
```
sbatch --export=ALL,RUN_DIR=$WS/graph1_sealed_m1_iiib_20260904,\
EXPECTED_CRB67_MD5=8e9253fef42f574c569a04a3e19299ab,\
EXPECTED_POOL_LIST_MD5=75f4030d5d3b0405fd948049bef5767e,VENUE=iiib \
cluster/graph1_sealed_m1_headstack.sbatch
```
→ **Submitted batch job 6794421** (array 0-40, cpu_il).

Task-0 log confirms the new guard passed live:
```
pool content-manifest pin OK: 75f4030d5d3b0405fd948049bef5767e
dataset pins OK: CRB67=8e9253fef42f574c569a04a3e19299ab rows=1345 (expected 1343; g-population
read-time disclosure) catalogue=c52c13b5cab61f6b3f04bbe202550969
pool=.../injection_pool_mix200k_20260728 (707 files)
```

## 4. State at handoff (one poll issued, 120s; moved to background per harness — not re-polled,
per the 3-call cap and "do not wait beyond that" instruction)
- 6794421: tasks 0-1 RUNNING (~2 min elapsed), tasks 2-40 PENDING (Priority). No STOP exits seen.
- **GATE-ACC 6790465**: sacct State = PENDING (00:00:00 elapsed, ExitCode 0:0, never started).
  `squeue` reason: **`(Priority)`** — not blocked by partition/QOS limits, just queue priority
  ahead of it; no dependency or resource-limit reason shown.

## Deviations
- Rsync of `POOL_MANIFEST.md5` itself was not needed (list-md5 comparison sufficed per the
  identical-manifest-format guarantee); flagging in case the chair wants the literal file diffed
  byte-for-byte as a belt-and-suspenders check.
- The background-polling task (`b2rlunsio`) from the single `poll` call was not awaited further,
  per the 3-call/9-minute cap and the instruction not to wait for completion.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
