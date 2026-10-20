# REPORT-R2 — M65 Round 2: evidence hygiene and landing on main

*Executor report, 2026-10-19. Contract: garden `wiki/meta/missions/M65-darksiren-emri-into-manyfold.md`
§ Round 2 plus the 2026-10-19 amendment. Branches: darksiren-emri
`mission/M65-darksiren-emri-into-manyfold` (pushed; tip in § Branches), garden
`mission/M65-darksiren-emri-into-manyfold` (pushed). Nothing was merged, nothing touched `main`
in either repo, and no cluster access of any kind was used.*

## Outcome in one paragraph

The thread contract, the m1 docket it cites and the m1 read inputs are all on the branch. The
cluster-ops files are on the branch too, except the `.claude/` guard hook (see the first finding).
All 85 `MANIFEST.md5` entries are now tracked and pass `md5sum -c` (85/85 OK), each with a sha256
line in a tracked manifest. **The scorer `t0_read.py` is unrecoverable from anything this host
can read**, so the four m1 numbers **cannot be reproduced from tracked artifacts in this round**.
Done criterion 3 is met only in its fallback form: the missing input is named, and so is who
recovers it. The brain-host registry row works: a synthetic hook run briefs both host paths. P-1
and P-2 are filed as proposals. THREAD.md and the two atoms carry the 2026-12-15 date and the
dated rulings.

## Least-confident decisions (read these first)

### D1. The size line: everything on the 85-entry manifest is committed; the 2D posteriors never are

- **Committed as plain files: 17 MB in total.** That is `event_likelihoods.csv` (15,573,458 B),
  `fisher_quality.csv` (62,927 B) and the 41 `posteriors/h_*.json` (41,147–41,250 B each, about
  1.7 MB together).
- **The repo's own rule pointed the other way.** `.gitignore:16` ignores every `simulations/`
  directory, under the heading "Data files (regenerable, live on $WORKSPACE on cluster)". I
  force-added the files past it, for three reasons:
  - The premise of that rule is false for these files. They are not regenerable without the
    cluster: a 41-task CPU array against the CRBs and the pool.
  - The repo already tracks evidence of exactly this shape. Four `event_likelihoods.csv` of
    14.4 MB each sit under `results/prod2d_closure_20260818/`, and
    `results/seed600_ab_20260710/*/simulations/posteriors*/combined_posterior.json` are tracked
    under a `simulations/` path. The largest tracked file is 19.2 MB
    (`results/venue_transfer_20260811/perf/route1_study/route1_harvest.npz`), and 15.6 MB stays
    inside that.
  - Jasper's R3 ruling (2026-10-19) names "the m1 scorer and posterior JSONs" as what lands on
    `main`.
- **Where I drew the line: about 20 MB per file**, the repo's own largest precedent. No repo
  document states a threshold.
- **Never pulled or committed: the 2D `posteriors_with_bh_mass/`.** It holds 41 files totalling
  3,540,577,702 B by `du -sb` on the box; `du -s` printed 2,496,551 there, so the apparent and
  allocated sizes disagree and I did not resolve which is right. It is outside the 85-entry
  manifest. It gets a directory-level line in the checksum manifest (bytes and locations), with
  no per-file sha256.
- **If Jasper prefers a checksum-only policy** for the 15.6 MB CSV, the fallback is one commit:
  `git rm --cached` the CSV. Its sha256 line is already in `M1_EVIDENCE_SHA256.txt`. The cost is
  that the m1 numbers become testimony again until the file is fetched.

### D2. The allowlist: a path-limited checkout from `origin/fix/p32d-classg-venue-repair` (`387cf33f`), with `.claude/` left out

- **How it was built.** Every entry marked `=p32d` below is byte-identical to the work branch.
  Nothing under `darksiren_emri/` or `darksiren_emri_test/` is on the branch:
  `git rev-parse HEAD:darksiren_emri` = `91446066…`, the same as `main`.
- **The landing order works both ways.** The 2026-10-19 amendment fast-forwards `main` to the
  work branch first and lands this branch on top. I ran that merge in a throwaway worktree
  (`--no-commit`, then aborted): **it is clean**. On top of `387cf33f` the branch changes only
  four things:
  - `.gitignore`
  - `cluster/datasets.yaml`
  - `.planning/**`
  - the 43 evidence files plus `M1_EVIDENCE_SHA256.txt`

  So the allowlist is correct against today's `main`, and it is a no-op for the files it shares
  with the work branch after the fast-forward.

Allowlist, least-confident entries first:

| path | state vs work branch | why it is here / doubt |
|---|---|---|
| `.claude/hooks/ssh-guard.py`, `.claude/settings.json` (registration), `.claude/skills/cluster/SKILL.md` | **NOT on the branch** | Scope 2(b) asked for them. The harness's auto-mode classifier refused to write `.claude/` hook config into this session's own project dir ("Self-Modification"); I did not work around that. **They reach `main` anyway through step 0 of the ruled landing**, the fast-forward to `387cf33f`, which carries all three. See Finding F1. |
| `CLAUDE.md` | =p32d | The only difference from `main` is the "Cluster access discipline" paragraph documenting `agent_ssh.sh` + `ssh-guard.py`. Until step 0 lands, it names a hook file this branch does not carry. |
| `cluster/datasets.yaml` | p32d + **one new entry** | `mix200k_20260728` pool pin: 707 files, list-md5 `75f4030d…`, the R28 ratification, per the amendment. This is the only allowlisted file I edited. |
| `.gitignore` | p32d + R1's hunk | The work branch's version, with R1's `.planning/*` / `!.planning/research/` exception re-applied. `.cartographer-notes.md` is present at line 70. |
| `cluster/graph1_sealed_m1_headstack.sbatch`, `cluster/graph1_headrebaseline_iiib.sbatch` | =p32d | These are the m1 job script and the anchor job script; the record and THREAD.md cite them. They are inert unless submitted, and they change no numerical behaviour of the package. |
| `graph1_20260901/exec/END_VERIFICATION_20260903.md`, `exec/m-head-rebaseline/READOUT_RECORD.md`, `exec/r-timeout-selection/POOL_MANIFEST.md5` | =p32d | These are one hop from the docket: the anchors 0.665854 / 0.666987, and the pool manifest whose md5 **is** the R28 list-md5. I stopped the transitive closure at one hop; deeper references stay on the work branch and arrive with step 0. |
| `graph1_20260901/exec/batch2_cluster_ops/{OPS_RECORD_m1_retrieval,OPS_RECORD_m1_resubmit,SUBMIT_RECORD_m1,SUBMIT_RECORD_s0c_m1}.md` | =p32d | The m1 ops records the read record and dossier cite. |
| `graph1_20260901/MORNING_DOCKET_20260904.md`, `GRAPH1_ADDENDUM_PROPOSAL_20260903.md` | =p32d | Cited by THREAD.md and VERDICT-m1.md. |
| `graph1_20260901/exec/r-sealed-mock/**` (5 docs + `m1_iiib_retrieved/` 43 files) | =p32d | The docket tree itself. `DESIGN_GATE_20260904.md`, which the dossier cites, does not exist on the work branch either: it is a dangling reference in the record, not one this landing creates. |
| `r-sealed-mock/m1_iiib_retrieved/simulations/**` (43 files), `r-sealed-mock/M1_EVIDENCE_SHA256.txt` | new | The evidence (D1) and its manifest. |
| `cluster/cluster.env` (names `emri2`), `cluster/preflight.sh`, `cluster/workspace_check.sh`, `cluster/agent_ssh.sh` | =p32d | Cluster-ops set, scope 2(b). |
| `DATA_INVENTORY.md` | =p32d | Carries the data-register commits `4239bc50` and `ecd01f14`, scope 2(c). The other files those commits touch (`CHANGELOG.md`, `cluster/README.md`, the sbatch `emri→emri2` renames) are left to step 0. |
| `.planning/research/{THREAD,CONVENTIONS}.md`, `.planning/missions/M65-…/**` | new (R1 + R2) | The thread contract and the mission record. `CONVENTIONS.md` is unchanged: no path it cites moved. |

### D3. The registry fix: a second `master-thesis-code` row, brain-host path, placed above the laptop row

- **What changed (garden, `wiki/meta/registry.md`).** The `/home/jasper/...` row was copied with
  only the `Path` column changed to `/home/ops/Repositories/darksiren-emri`. A contract bullet
  was added that documents one row per host path.
- **Why above the laptop row.** Every hook reader is first-prefix-match per row, so order does
  not matter to them. `wiki/assets/pi-extensions/compile-briefing.sh` is different: it writes
  `briefings/<slug>.full.md` once per row, the last row wins, and its default vault is the
  laptop's. With this order the laptop row compiles last on the laptop.
- **Tested** with `wiki-session-start.sh` run unmodified from the garden worktree, `HOME` set to
  a temp dir and `GARDEN_DIR` set to the worktree:

  | cwd | result |
  |---|---|
  | `/home/ops/Repositories/darksiren-emri` | `=== PROJECT OFFICER BRIEFING: master-thesis-code ===`, `/home/ops` row |
  | `/home/jasper/Repositories/darksiren-emri` | briefing, laptop row |
  | `/home/ops/Repositories/darksiren-emri-wt/M65-…` | briefing, via the prefix match |
  | `/home/ops/Repositories/not-registered` | silent, exit 0 |
  | control: `/home/ops/...` against `origin/main`'s registry | **0 briefings**, which confirms the bug |

  All writes went to the temp HOME (`.orbit/usage.jsonl`, `.manyfold/hook-beats*`, canary
  files). The live `~/.orbit` and `~/.attine` were not touched.
- **Other readers** (read-only survey, then spot tests):
  - The statusline wrapper shows `📘 master-thesis-code` for the `/home/ops` cwd.
  - `/gardener --all`'s `test -d` set on this host contains the project exactly once.
  - `/chronicler` step 1 matches. Its `Last Updated` bump should touch both rows; the new
    contract bullet says so.
  - `/pulse` and `mission-brief` read slug lists and would show the slug twice unless they
    dedupe. This is cosmetic, but it is a small double-count risk; the bullet says to dedupe.
  - Not verified: `lb-weathervane.sh` iterates every row, and I did not read its scoring body.
    It is probably benign, because an absent path scores nothing.
- No protected path was edited. `registry.md` is not in `protected-paths.list`;
  `wiki/assets/claude-hooks/**` is, and I did not touch it.

### D4. The atom write path

- **What I did.** I edited `x-t1-blind-sealed-h-mock` and `q-what-ships-by-2026-09-23` in place
  on the garden mission branch. The changes were the `updated:` field, the title, the
  `kill_criterion` / tags fields, and an appended dated-rulings section. `status` and
  `confidence` were left alone.
- **Why this counts as the write path.** Per `wiki/research/README.md` rule 8, the single write
  path files content "after a human accepts". Here the human acceptance is the 2026-10-18/19
  rulings, and the filing lands through the overseer's garden landing. New atoms would go to
  `.planning/research/darksiren-emri/proposed/`; none were needed.
- **The atom id `q-what-ships-by-2026-09-23` keeps its date** so its edges stay stable. Only the
  title and the criterion carry the new date.
- **Further atoms that still say 2026-09-23** are a judgement call, left for `/cartographer`
  (step R-4):
  - `a-ships-as-methods-paper`, `a-ships-as-h0-measurement` and `q-h0-from-emri-dark-sirens`
    mention it in body text.
  - `PROPOSAL-2026-09-13.md` is a historical proposal and should stay as written.

## Evidence map — the 85 `MANIFEST.md5` entries

The base is `graph1_20260901/exec/r-sealed-mock/m1_iiib_retrieved/`. There are three locations:

- **git**: tracked on this branch
- **WS2**: `/pfs/work9/workspace/scratch/st_ac147838-emri2/graph1_sealed_m1_iiib_20260904` on
  the cluster, unreachable from here
- **BOX**: `hbox:archive/darksiren-emri/cluster/graph1_sealed_m1_iiib_20260904`

| entries | count | before R2 | now | check |
|---|---:|---|---|---|
| `GIT_COMMIT_AT_RUN.txt` | 1 | git (work branch only) | git + WS2 + BOX | md5 OK |
| `run_metadata_{0..40}.json` | 41 | git (work branch only) | git + WS2 + BOX | md5 OK; box md5s match |
| `simulations/diagnostics/event_likelihoods.csv` | 1 | WS2 + BOX | **git** + WS2 + BOX | md5 `ed3d5479…` OK; sha256 `bef5da05…` equal on box and local |
| `simulations/fisher_quality.csv` | 1 | WS2 + BOX | **git** + WS2 + BOX | md5 OK; sha256 `dfe32a9c…` equal |
| `simulations/posteriors/h_*.json` | 41 | WS2 + BOX | **git** + WS2 + BOX | md5 OK (41/41); sha256 equal on box and local (spot-checked first and last; full list in the manifest) |
| **total** | **85** | 42 tracked anywhere | **85 tracked** | `md5sum -c MANIFEST.md5`: **85/85 OK** |

The per-file sha256, bytes and both locations are in the tracked
`graph1_20260901/exec/r-sealed-mock/M1_EVIDENCE_SHA256.txt`. Two off-manifest items are listed
there too:

- `posteriors_with_bh_mass/`: cluster and box only, by design.
- `t0_read.py`: **UNRECOVERED**.

The pull itself:

- Disk before the pull: `df -h /` showed 21 GB free; `df -i` showed 13 % of inodes used.
- Method: one `sftp -b` batch of `get` lines. The Storage Box was only listed, checksummed and
  read. `find` is not available there; nothing was written to the box.
- `graph1_sealed_m1_joint_r1_20260904` does not exist on the box. It was never run, which is
  consistent with the dossier.

## Reproduction of the four m1 numbers — NOT POSSIBLE this round

| number | record | reproduced from tracked artifacts? |
|---|---|---|
| 2D \|mean − anchor\|/σ | 2.73 | **no**: scorer unrecoverable |
| 1D \|mean − anchor\|/σ | 2.58 | **no** |
| 2D mass at h = 0.600 | 4.39 % | **no** |
| 1D mass at h = 0.600 | 2.59 % | **no** |

**The inputs are now tracked and byte-verified against the run's own manifest. The scorer is
not.** Where I looked for `t0_read.py`:

- every git ref of darksiren-emri, including `mission/M162-t1-governance-package` (`git log --all
  -- '*t0_read*'` is empty);
- the brain host filesystem (`find /` by name);
- the Storage Box: listings of the m1 run dir, `scripts/`, `_loose-scripts/`, the
  `darksiren-emri-wave3/` checkout (dated 2026-08-30, before the 2026-09-04 run) and the cluster
  root.

It is in none of them. The box listing was not exhaustive: `find` is unavailable there and
`ls -R` was not run over all 166 directories. Even so, the garden audit
(`.planning/darksiren-data-inventory-2026-09-23.md` A10) already recorded it as "exists nowhere
else".

**Per the contract, I did not reconstruct the scorer from the prose.** The record says the
2026-09-04 decisive verifier wrote an independent scorer from scratch, so a re-implementation is
possible in principle. It would be new code making a scoring-convention choice, and the contract
reserves that.

**The human act that recovers it.** Jasper checks his laptop for:

1. `~/t0_scorer_rescue_20260923.tar.gz`, the rescue command in the garden audit (line 199), if it
   was ever run;
2. an untracked `t0_read.py` in his laptop checkout of `darksiren-emri` (the chair session of
   2026-09-04 "validated this session" locally).

Failing both, the [[M38-cluster-session-unlock]] Round 3 cluster-ops agent searches
`$(ws_find emri2)` with `find . -maxdepth 4 -name t0_read.py`. Once the file is committed beside
`READ_RECORD_m1.md`, the four numbers are a one-command re-run on the tracked CSV.

## Findings for the tap

- **F1 — the `.claude/settings.json` guard-hook registration (and `ssh-guard.py`, the cluster
  skill) are not on this branch.** The harness refused writing them into this session's own
  project dir. They land with step 0 of the ruled landing, the fast-forward of `main` to
  `387cf33f`. **Tapping merge therefore registers `ssh-guard.py` as a `PreToolUse` Bash hook for
  every agent session in darksiren-emri.** It blocks:
  - `ssh -O exit|stop`;
  - deleting `~/.ssh/cm-*`;
  - `pkill ssh`;
  - `sleep ≥ 60` inside a remote command;
  - parallel ssh fan-out.

  This is what [[M38-cluster-session-unlock]] Round 3 needs on `main`. **If step 0 is skipped**
  and only this branch is merged, `CLAUDE.md` would name a hook file that does not exist, and M38
  would still lack the hook.
- **F2 — the m1 numbers remain testimony until the scorer is found.** The inputs are no longer
  testimony (D1, evidence map).
- **F3 — the hazard named in the reference check still holds.** Pre-fast-forward `main` carries
  the `darksiren_emri/` tree `91446066…`, not the frozen `71acb98b…`. (m2) must never be cut
  before step 0. THREAD.md records this.
- **F4 — `trellis check wiki` fails on garden `origin/main` already.** There are 6 errors: 5 from
  `wiki/research/darksiren-emri/PROPOSAL-2026-09-13.md` (the `sources` edge kind, the `pillar`
  key, a dangling target), counted by trellis as the 28th "atom"; and 1 dangling `sources` link
  from `packet-cluster-policy-2026-10-19` to that page. The count is identical before and after
  my changes, and none is in a touched atom. Carried to `/cartographer` R-4.
- **F5 — `wiki/meta/proposal-queue.md` is 1,148 lines against `max_lines: 900`.** It was 1,080
  before this round; the two new blocks add 68.
- **F6 — no route to the cluster was found or used.** `~/authorize-cluster-key.sh` exists in the
  brain host's home. I did not open or run it; it is M38's territory.

## Scope items — status

| # | item | status |
|---|---|---|
| 1 | locate the evidence, read-only | **done** (branch tips below; evidence map above) |
| 2 | land contract + docs by allowlist | **done except the `.claude/` trio** (F1; it lands with step 0) |
| 3 | evidence under version control | **done**: 85/85 tracked plus the sha256 manifest |
| 4 | reproduce the four numbers | **not possible**: scorer unrecoverable; the recovery act is named |
| 5 | registry path | **done**, tested (D3) |
| 6 | P-1 / P-2 proposals | **done**: `PROP-20261019-65`, `PROP-20261019-66` in `wiki/meta/proposal-queue.md` |
| 7 | THREAD.md | **done**: criterion 3 → 2026-12-15 with history; dated rulings 2026-10-18/19 (Continue; R1–R4; R25/R27; R28/R24); gate 1 = M38 (~2026-11-20); gate 2 unchanged. Hard core, criteria 1–2, the claim table and the verdict text are byte-identical to R1 (checked by section compare). |
| 8 | atoms + graph | **done** (D4; graph regenerated by `trellis research darksiren-emri wiki`) |
| 9 | push both branches | **done** |

Branch tips at item 1:

- `main` = `origin/main` = `6e1bc488`
- `origin/fix/p32d-classg-venue-repair` = `387cf33f`
- this branch at R1 = `456e42b4`
- `merge-base --is-ancestor main 387cf33f`: **yes**
- `merge-base --is-ancestor 456e42b4 main`: **no**
- Law 10 (`git merge main`): "Already up to date"

Re-run before this report: still up to date, and the trial merge onto `387cf33f` is clean.

## Done criterion — self-check (the verifier decides)

1. The branch diff vs `main` equals the allowlist (D2): nothing under `darksiren_emri*/`;
   `THREAD.md`, `CONVENTIONS.md` and the `.gitignore` line are present; `cluster.env` names
   `emri2`; the guard-hook registration is listed as finding F1. **Deviation:** the `.claude/`
   trio is absent from the diff.
2. Met: 85/85 entries are tracked, and each has a sha256 line.
3. Met only in its fallback form: the unrecoverable input (`t0_read.py`) is named, with who
   recovers it.
4. Met (item 7).
5. Garden: the hook test passes on both paths; the proposals are filed; the touched atoms add no
   `check` error; `graph.md` was regenerated.
6. This report, the staged `publish-round-r2/`, and the card id in the page's Status log.
   VERIFY-R2 is the verifier's.

## Budget

These are approximate: the session has no meter readout, so the figures are this session's own
context counter plus the subagents' reported usage.

| phase | tokens (approx.) |
|---|---|
| orientation + item 1 (git survey; Sonnet box survey 80 k) | ~110 k |
| registry-reader survey (Sonnet) | ~62 k |
| items 2–3 (allowlist, pull + verify via Sonnet 47 k, commits) | ~90 k |
| item 7 THREAD.md | ~15 k |
| items 5, 6, 8 garden | ~45 k |
| report + round dir | ~40 k |
| **total** | **~0.37 M of the 2.0 M declared** |
