# Lens 3 — Infrastructure + Data Lifecycle Health Scan

**Date:** 2026-09-03 · **Scope:** read-only, local records only (no SSH — cluster access expired
tonight). Sources: `cluster/README.md`, `.claude/skills/cluster/SKILL.md`, `cluster/preflight.sh`,
`cluster/datasets.yaml`, `DATA_INVENTORY.md`, `results/_archive/*`, `results/campaign51_20260728/
RUNBOOK_NEXT_SESSION_41.md`, `cluster/WORKSPACE_ARCHIVAL_TRIAGE_20260827.md`, `du`/`df` on the dev
box, `cluster/*.sbatch`.

---

## 1. Workspace expiry — evidence trail and what MUST move

- **Deadline: 2026-09-23, 0 extensions left** (confirmed by an actual `ws_extend emri 60` probe on
  2026-08-27, which returned `Error: no more extensions.`, exit 255 — this is verified, not
  inferred). `DATA_INVENTORY.md:27` and `:399` both carry this. **20 days out from today
  (2026-09-03).**
- The **2026-08-27 triage** (`cluster/WORKSPACE_ARCHIVAL_TRIAGE_20260827.md`) found ~130 workspace
  dirs, ~644 GB total, with the canonical registries (`datasets.yaml`, `DATA_INVENTORY.md`) stale
  by a month — the single biggest finding of that pass. It sequenced a MUST-ARCHIVE priority list
  (seed600 4-arm evidence locker 161 GB → prodstack/densecore ~65 GB → ~53 GB of iiib/joint_r1
  pairs → ~130 GB of UNKNOWN post-campaign51 dirs → retired dirs last → `work/` 104 GB deprioritized
  as reproducible scratch cache).
- **Progress since:** the `archive_run_wave2.sh` re-run on 2026-09-01 evening pulled 7/8 registered
  wave-2/wave-3 items successfully (the 8th, `c1`, is a legitimate SKIP — never launched, not a
  defect hit). `DATA_INVENTORY.md` was itself updated 2026-09-02 with a full storage-decision
  section, so archival triage is an active, tracked thread, not stalled.
- **`archive_run_wave2.sh`'s three-valued-existence fix is NOT landed.** Row #288 named the defect
  (SSH failure read as "not found on cluster" — a conflation bug in the `test -d` existence check);
  the runbook explicitly says "fix owed" and instructs treating any current SKIP as suspect if the
  SSH session could have expired. The script as it stands today (`results/_archive/
  archive_run_wave2.sh:34`) still does a bare `ssh ... test -d ...` with no distinction between
  "confirmed absent" and "unreachable." **Verified from the 2026-09-01T03:56 log**: all 8 items
  SKIPPED in one run where every item had previously succeeded or later succeeded 14 hours later —
  a textbook case of the exact defect the fix would catch (that run was almost certainly an
  auth/SSH-expiry false-negative, not real absence).
- **The single most acute infra risk found this pass is *not* the `emri` workspace — it's
  `~/emri-archive/` (159 GB, on this dev box, NOT in `results/`, NOT in git).** `DATA_INVENTORY.md`
  itself flags this as "sole copy — highest-priority evacuation target": the seed600 4-arm evidence
  locker (`run_20260628_seed600` 56G + three `_ab_*` arms 35G each = 161G, matches `du -sh
  ~/emri-archive` = 159G measured just now) has **no second copy anywhere** — not on the cluster
  workspace (already evacuated FROM there), not backed up. If this machine's disk fails before an
  external/institutional copy is made, this data is gone permanently and is NOT cheaply
  regenerable (GPU re-sim at a non-default, now-historical Ω_m era). This is more urgent than the
  Sep-23 cluster deadline because it has *zero* redundancy today, cluster-side or otherwise.
- **Open action, unresolved and time-critical per `DATA_INVENTORY.md`'s own 2026-09-02 note**: no
  institutional archive (KIT LSDF / bwDataArchive-class) has been identified or added to the Device
  Registry, despite `cluster/README.md:185` instructing "copy to persistent storage before
  expiration" without ever naming what that storage is. The recommended path (buy a ≥2TB external
  SSD/HDD + identify the institutional archive) is proposed but not executed.

**Severity: CRITICAL.** **Effort: S** (evacuate `~/emri-archive/` to any second medium — external
disk, second machine, cloud object storage — is a few hours of `rsync`/`cp` once a destination
exists) **but NEEDS-AUTHOR-WORD** (buying hardware / picking institutional vs. object storage is a
decision only the author can make; the mechanical copy itself is SAFE-HYGIENE once a destination is
named). **DEADLINE-DRIVEN: two clocks, not one** — `~/emri-archive/` has no deadline but zero
redundancy (act first); the `emri` workspace has a hard Sep-23 deadline with partial redundancy
already banked locally.

---

## 2. Local disk pressure

- **Dev box: 749G used / 931G total, 136G free (85% used)** — measured just now via `df -h /`.
  This is a marked jump from the 2026-08-27 triage's own snapshot (277G used, 32%, 607G free) and
  is now inside the danger zone the 2026-09-02 `DATA_INVENTORY.md` pass called "82% full."
- `du -sh results` → **383 GB**. Largest subdirs under `realistic_20260729/`:
  - `tree2_20260830/` — **114 GB**
  - `graph1_20260901/` — **42 GB**
  - `headreadout_20260827/` — 20 GB
  - `p3_work/` — 16 GB, `p3_b0_work/` — 14 GB
  - `wave3_20260830/` — 11 GB
  - `ca_rhs_work/` — 5.3 GB (the dir called out in the task; not currently a large fraction)
- `results/_archive/` — **119 GB** (the local mirror of MUST-ARCHIVE cluster items; per
  `DATA_INVENTORY.md` this is already "warm" — a second copy exists on the cluster until Sep 23, so
  this is the safest prune candidate of the big local dirs, but ONLY after the cluster copy is
  independently confirmed and only after `~/emri-archive/` is off this machine).
- **`~/emri-archive/` — 159 GB — is outside `results/` and easy to miss in a `du results` sweep;
  it is not redundant with anything (see §1) and must NOT be treated as prunable.**
- **What is safe to prune once cluster archiving is confirmed:** per `DATA_INVENTORY.md`'s own
  table, two named cull candidates total ~15 GB: `results/run_20260620_seed500_phase50/` (2.3 GB,
  explicitly RETIRED era, "cold — safe to delete") and `results/run_20260817_fusion_
  counterfactual/` (13 GB, thread CLOSED at row #119, "cull candidate"). Both still present per
  this scan. **`results/` is fully gitignored**, so any deletion there is unrecoverable — the
  DATA_INVENTORY note is explicit that no agent should delete unilaterally; author sign-off
  required even for these two.
- **Severity: HIGH** (136G free is tight given campaign51 "grows with every wave" per the author's
  own DATA_INVENTORY note, and the graph-1 charter about to launch a fresh wave of retrievals).
  **Effort: S** for the two named ~15GB cull items (author approval + `rm -rf`); **M** for a
  systematic `results/_archive/` vs. cluster-copy reconciliation before any bulk prune there.
  **SAFE-HYGIENE** only after explicit author sign-off (repo convention: `results/` deletions are
  never agent-unilateral). **DEADLINE-DRIVEN**: indirectly — if disk fills before Sep 23, the
  workspace evacuation itself has nowhere to land locally.

---

## 3. Unregistered-dataset preflight WARN backlog

- `cluster/preflight.sh`'s `[DATASETS]` block (lines ~190-209) does a mechanical cross-check: every
  top-level dir in the live workspace gets grepped against both `cluster/datasets.yaml` and
  `DATA_INVENTORY.md`; anything present in neither is counted UNREGISTERED and raises a
  `note_warning`. This check was itself added *because* a prior "remember to update the inventory"
  convention failed silently — ~30 run dirs / ~250 GB went uninventoried for a month (documented in
  its own comment, citing `WORKSPACE_ARCHIVAL_TRIAGE_20260827.md`).
- **No live preflight run exists locally tonight** (SSH is expired, no cached preflight output with
  a current unregistered count was found — the only saved preflight-shaped log,
  `results/prod2d_closure_20260818/preflight_run.log`, predates this backlog). The task brief's
  "75 unregistered dirs" figure could not be independently re-verified from local records this
  session; treat it as the author's own most recent number, not something this scan reproduced.
- **What registering the backlog would take, characterized from the registry files' own structure**:
  `cluster/datasets.yaml` is a hand-maintained YAML with `injections:`/`crbs:`/(presumably
  posteriors) sections, each entry needing `id`, `path`, `status`, and a `note` with real
  provenance (job IDs, commits, retirement reasoning) — NOT mechanically derivable from a
  directory name alone; `DATA_INVENTORY.md`'s Dataset Registry rows are even more narrative
  (git commit, pipeline-tier staleness, evidence-locker role). The 2026-08-27 triage did exactly
  this exercise for ~30 dirs and it took a dedicated multi-hour pass with per-dir classification
  (MUST-ARCHIVE / REPRODUCIBLE / UNKNOWN) — this is NOT a script-writeable task; it requires
  reconstructing provenance from git log + session memory for each dir, which is exactly the kind
  of narrative-provenance work the graph-1 charter's own INFRA proposal flags as a structural
  failure mode ("registry not updated in a month while workspace kept growing").
- **Severity: MEDIUM** (doesn't block current work, but is the same failure mode that already cost
  a month once and both registries are stale relative to the ~30+ post-campaign51 fleet dirs named
  in the 2026-08-27 triage: `csg_pilot_20260821`, `o4_shards_20260821`, `p3_b0_identity_
  fleet_20260823`, `realizations_20260729`, `massab_*`, the `iiib`/`joint_r1` pair families, etc.).
  **Effort: L** (narrative provenance reconstruction per dir, not mechanical). **SAFE-HYGIENE**
  for the mechanical parts (adding rows once provenance is known) but the provenance-reconstruction
  itself needs no author *decision*, just author or agent *time* — a good sonnet-tier
  mechanical-backfill task once someone (ideally the graph-1 infra proposal's own claim-layer
  model) supplies the per-dir provenance narrative.

---

## 4. `cluster/` script sprawl

- **39 `.sbatch` files** in `cluster/` total; **8 are `graph1_*.sbatch`** (the Research Graph 1
  branch nodes: `headrebaseline_{iiib,joint_r1}`, `c0prime_headrebaseline_{gate,iiib,joint_r1}
  ` [3 gate variants], `m_s0b_{byteid_precheck,production}`, `t5_armR{,_c0prime}`,
  `t5_armS_iiib`). **49 total submit-related scripts** (`.sbatch` + `submit_*.sh`).
- **Direct diff of `graph1_headrebaseline_iiib.sbatch` vs. its `joint_r1` sibling**: the two files
  are near-identical — same SBATCH headers, same dataset-pin md5 checks, same H_GRID_41 array
  logic, same CLI flag block — differing only in (a) the venue string in comments/job-name, (b) one
  `--observed_catalogue` flag for `joint_r1`, and (c) the source CRB dataset. This is a textbook
  **copy-paste-and-edit-one-flag** pattern, repeated across at least 4 more files in the `graph1_*`
  family (the `t5_armR`/`t5_armR_c0prime`/`t5_armS_iiib` set looks like the same pattern by naming
  convention).
- **A template/param-driven generator is worth it**: a single parametrized sbatch (or a Python/Jinja
  generator emitting the sbatch text) taking `{venue, extra_cli_flags, crb_dataset_id}` would
  collapse the 8-file graph1 family (and likely the earlier `wave2_*`/`wave3_*` families, same
  pattern by the `WAVE3_SUBMISSION_NOTE_20260830.md` references seen in this scan) to one template
  + a small per-node config table, cutting the maintenance surface and the chance of a
  copy-paste drift bug (e.g. a dataset-pin md5 hand-edited in one sibling but not the other).
- **Severity: LOW** (no evidence of an actual drift bug from this — the md5 pins matched across the
  two files diffed). **Effort: M** (writing + validating a generator against the existing 8+ files
  without changing any already-run job's exact CLI is real work, and graph-1 execution is
  IN-FLIGHT right now — this is a between-waves cleanup, not urgent). **SAFE-HYGIENE** (pure
  tooling refactor, no science content change) but should wait until the current graph-1 wave's
  active branches finish, to avoid touching a script mid-use.

---

## 5. Monitor/runbook hygiene

- **37 `RUNBOOK_NEXT_SESSION_*.md` files** in `results/campaign51_20260728/`, numbered 5 through 41
  (runbook 41 is the current entry point per its own header; **runbook 42 will be owed at this
  session's close**, consistent with the convention). The convention itself (each runbook
  supersedes the last, states "read first," links back to `BIAS_HISTORY_LEDGER.md` rows) is being
  followed consistently — no gaps or duplicate numbers spotted in the 37 files present.
  **Effort to write runbook 42: S** (mechanical, sonnet-tier — append-only synthesis of this
  session's rows, matching the existing runbook 40→41 pattern).
- **Session-memory dir** (`~/.claude/projects/.../memory/`): **300 KB, 40 files** — small, not a
  pressure point. The `MEMORY.md` index itself (auto-loaded every session) lists ~30 dated session
  summaries; no size or count concern here, though it is worth noting the memory index is
  growing linearly with sessions and will eventually want the `/gardener` consolidation pass this
  project already has available as a skill.
- **Severity: LOW** (process is healthy, no backlog). **Effort: S.** **SAFE-HYGIENE.**

---

## 6. Reproducibility posture

- **Dataset pinning is being followed consistently in the newest work**: every `graph1_*.sbatch`
  reviewed carries an explicit md5 STOP-gate on both the CRB CSV and the reduced galaxy catalogue
  before running (`EXPECTED_CRB_MD5`/`EXPECTED_CATALOGUE_MD5`, hard `exit 1` on mismatch) — this
  is exactly the CLAUDE.md "Dataset pinning (2026-08-20)" convention working as designed, and
  matches the incident that motivated it (a stale local galaxy catalogue silently feeding
  analyses).
- **`GIT_COMMIT_AT_RUN.txt`** is written per run-dir before execution (`git rev-parse HEAD >
  "$RUN_DIR/GIT_COMMIT_AT_RUN.txt"`), giving a durable per-run commit pin independent of
  `run_metadata.json`, matching the CLAUDE.md reproducible-runs contract (git_commit + timestamp +
  CLI args recorded).
- **Coverage spot check**: `results/campaign51_20260728/realistic_20260729/graph1_20260901/` alone
  has **107 `run_metadata*.json` files** across its `retrieved/` and `exec/` subtrees — provenance
  capture looks systematic for the in-flight graph-1 wave, not spotty.
- **`write_provenance.sh`** is sourced by every graph1 sbatch reviewed and stamps a human-readable
  description of what's blind/not-blind for the run into the logs dir — this is a good practice
  beyond the minimum CLAUDE.md contract (it captures *intent*, not just mechanical metadata).
- **Known reproducibility gap** (carried, not new): the wave-2 `c4` item in
  `DATA_INVENTORY.md:292` is flagged "PARTIALLY RECOVERABLE" — `run_metadata_*.json` and `logs/`
  for that arm were never retrieved (SSH outage mid-transfer) though the gate/reading numbers don't
  depend on them. This is the kind of gap the three-valued-existence fix (§1) would have caught
  faster.
- **Severity: LOW** (posture is strong for current work; the one gap is already flagged and
  bounded). **Effort: N/A** (no action needed beyond the existing tracking). **SAFE-HYGIENE.**

---

## Summary table

| # | Finding | Severity | Effort | Author-gate | Deadline |
|---|---|---|---|---|---|
| 1a | `~/emri-archive/` 159GB, sole copy, zero redundancy | **CRITICAL** | S (once destination named) | NEEDS-AUTHOR-WORD (pick destination) | No hard date but zero redundancy TODAY — act first |
| 1b | `emri` workspace expires 2026-09-23, 0 extensions, ~130 dirs / ~250GB (the post-campaign51 fleet) still UNKNOWN/unarchived per the 2026-08-27 triage | **CRITICAL** | M-L (sequenced multi-hundred-GB transfer) | NEEDS-AUTHOR-WORD (storage medium decision pending) | **20 days out** |
| 1c | `archive_run_wave2.sh` three-valued-existence fix still NOT landed (row #288, "fix owed") | HIGH | S (one script fix + test) | SAFE-HYGIENE (pure bugfix, already designed into graph-1 infra model) | Compounds risk against the Sep-23 deadline — fix before the next big archive push |
| 2 | Local disk 85% full, 136G free, growing every wave | HIGH | S (2 named ~15GB cull items) / M (systematic reconciliation) | NEEDS-AUTHOR-WORD (results/ deletions are never agent-unilateral) | Indirect — blocks landing the workspace evacuation locally |
| 3 | Unregistered-dataset preflight WARN backlog (registries stale ~1 month behind workspace) | MEDIUM | L (narrative provenance reconstruction, not mechanical) | SAFE-HYGIENE once provenance known | None hard, but same failure mode already cost a month once |
| 4 | 8+ near-duplicate `graph1_*.sbatch` files, template-worthy | LOW | M | SAFE-HYGIENE | None — do between waves |
| 5 | Runbook/memory hygiene | LOW | S | SAFE-HYGIENE | Runbook 42 owed at session close |
| 6 | Reproducibility posture (pinning, provenance, commit-stamping) | LOW | — | SAFE-HYGIENE | None — already strong |
