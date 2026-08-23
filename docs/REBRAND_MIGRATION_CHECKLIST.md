# Rebrand Migration Checklist — deferred, coordinated-window items

**Status:** tracking doc. Phases 0–2 of the `master_thesis_code` → `darksiren_emri` rebrand
(package rename, branding collateral) landed on branch `rebrand/darksiren-emri`; Phase 3
(GitHub repo rename) follows. This file is the punch list for everything intentionally **not**
done in those phases because it needs a coordinated window, a scripted migration, or the author's
separate sign-off. Source plan: `docs/REBRAND_PROPOSAL.md` §5 (operational rename plan) and
`results/campaign51_20260728/RUNBOOK_NEXT_SESSION_9.md` §3.1.

---

## 1. Local directory rename (dev box)

**Done on this dev box 2026-08-15** — `/home/jasper/Repositories/MasterThesisCode` ->
`/home/jasper/Repositories/darksiren-emri`, Claude project state re-keyed, registry Path
confirmed, §3 references fixed. Executed as `scripts/migrate_local_rename.sh` steps 1–3 plus
by-hand steps 4–7 (the script aborted at step 4; see below). **Per-machine, not once-and-for-all:**
every dev box needs its own run. Five things the first execution surfaced, all now fixed in the
script (which is now a 9-step, fully idempotent run):

- **Step 4 could never pass on a second box.** The garden registry is a *shared, synced* file, and
  its Path column was migrated by a garden-side commit (`6caf391`, 2026-08-13) before any box ran
  this script. Step 4's "the old path must be present" precondition therefore fails as an ABORT on
  every box, including the first. It now treats "already migrated" as success.
- **The venv does not survive the `mv`.** Every `.venv/bin/*` console-script shebang and the
  `activate*` scripts hardcode the absolute venv path, so `uv run mypy`, `uv run pytest` and
  `source .venv/bin/activate` all break after the rename with an opaque
  `Failed to spawn: <tool>` / `ModuleNotFoundError`. The interpreter and site-packages are fine —
  only the entry points are stale. Fix: `uv sync --reinstall --extra cpu --extra dev` (rewrites the
  shebangs) plus a path rewrite of `.venv/bin/activate*`. This is the local mirror of §2's
  "rebuild the venv" step, and it was missing from §1.
- **`-book` is a prefix collision.** `/home/jasper/Repositories/MasterThesisCode` is a strict prefix
  of the sibling worktree `.../MasterThesisCode-book`, which is **not** being renamed. Any
  find-replace over the old path must be guarded (`(?!-book)`), or it silently repoints live
  references at a directory that does not exist.
- **`origin` was still riding GitHub's rename redirect.** The dev-box remote pointed at
  `github.com:JasperSeehofer/MasterThesisCode.git` long after Phase 3's `gh repo rename`; push and
  fetch kept working and merely printed a `This repository moved` notice, which is exactly the
  RUNBOOK-9 §3.1(c) grace-period trap (§2 did `git remote set-url` on the cluster; the dev box never
  did). Repointed 2026-08-15 to `…/darksiren-emri.git`, and the script now does it.
- **The rename broke the linked `-book` worktree.** A linked worktree stores an *absolute* path back
  to the main repo in its `.git` file, so after the `mv` every git command inside
  `~/Repositories/MasterThesisCode-book` died with `fatal: not a git repository: (null)`. Fixed with
  `git worktree repair` from the main checkout (the main repo's own back-pointer was still fine —
  only the worktree→repo direction breaks). The worktree directory itself keeps its old name on
  purpose; renaming it is a separate migration.

The original reason for deferring — renaming first silently orphans two dependent systems:

- **Claude memory / session state**, keyed by
  `~/.claude/projects/-home-jasper-Repositories-MasterThesisCode`. A local `mv` of the repo
  directory does not move this — the project's memory index and session history become
  unreachable under the new path until re-keyed.
- **Garden registry Path column** (`~/Repositories/garden`, `registry.md`) — the vault's
  cross-repo tracking keys this project by its filesystem path. Needs an explicit registry update
  in the same step as the directory rename, not after.

Both were re-keyed correctly on 2026-08-15 and verified: the renamed project directory carries all
44 memory files and the session transcripts back to 2026-07-17, and the vault's session-start hook
fired in a fresh session — which it only does when a registry Path column matches `$CWD`.

## 2. Cluster repo rename (bwUniCluster)

**Not done.** All `cluster/` scripts and `.claude/skills/cluster/SKILL.md` still reference the
cluster filesystem repo path `~/MasterThesisCode` / `$HOME/MasterThesisCode` / `$CLUSTER_REPO`
deliberately — this describes the cluster's current reality, not a rename target. The **ONE-repo
rule** (`.claude/rules/hpc-gpu.md:26`, `.claude/skills/cluster/SKILL.md:30`) requires the cluster
rename to happen in the *same window* as the GitHub repo rename (Phase 3), not staggered — a
stale `~/MasterThesisCode` pointing at a now-redirected GitHub URL is exactly the trap RUNBOOK-9
§3.1(c) flags.

**Sequence for that window:**
1. `gh repo rename` (Phase 3, this session) — confirm redirect is live.
2. On the cluster: `mv ~/MasterThesisCode ~/darksiren-emri && cd ~/darksiren-emri && git remote set-url origin <new-url> && git pull`.
3. **Rebuild the venv** — the cluster venv is built against the old package name; a stale venv
   after the Phase 1 package rename (already landed) will import-error opaquely. Use
   `cluster/modules.sh` per the `/cluster` skill.
4. Update every literal `~/MasterThesisCode` reference below to `~/darksiren-emri` in the same
   commit as the cluster-side rename, so the docs match the new cluster reality:

   - `cluster/JOB_TEMPLATE.sbatch:31`
   - `cluster/evaluate_production_h0p73_superdense.sbatch:47`
   - `cluster/calibration_gate_v2.sbatch:11,13,14,23`
   - `cluster/combine.sbatch:31`
   - `cluster/evaluate_densecore.sbatch:64,65`
   - `cluster/evaluate_closure_h_true_finegrid.sbatch:88`
   - `cluster/merge.sbatch:34,35`
   - `cluster/evaluate_closure_h065_finegrid.sbatch:47`
   - `cluster/gpu_smoke.sbatch:21`
   - `cluster/datasets.yaml:25` (`dev_box_repo:` — also needs the local-dir-rename from §1 done first)
   - `cluster/README.md:19`
   - `cluster/evaluate_closure_h065.sbatch:41`
   - `cluster/campaign_orchestrator.sh:17`
   - `cluster/venue_transfer.sbatch:17,20,21,56`
   - `cluster/evaluate.sbatch:79,80`
   - `cluster/preflight.sh:14,19,21`
   - `cluster/inject.sbatch:30,31`
   - `cluster/simulate.sbatch:49,50`
   - `cluster/evaluate_production_h0p73_dense.sbatch:46`
   - `cluster/LAUNCHING_JOBS.md:15,33,76,169,229`
   - `cluster/cluster.env:9`
   - `.claude/rules/hpc-gpu.md:26`
   - `.claude/skills/cluster/SKILL.md:30,46,108`

   (`.claude/skills/cluster/SKILL.md:46`'s rsync target `bwunicluster:MasterThesisCode/darksiren_emri/...`
   needs the `MasterThesisCode/` segment updated to `darksiren-emri/` in this same pass — the
   `darksiren_emri/` package-subdirectory segment was already renamed in Phase 1.)

5. **Workspace paths / `DATA_INVENTORY` references** — bwHPC workspace symlinks and any inventory
   manifest entries that hardcode `MasterThesisCode` as a path component live on the cluster
   filesystem, not in git. Audit these by hand during the cluster-rename window; a blind
   find-replace across git-tracked files won't reach them.

## 3. Local-path references — fixed 2026-08-15 (commit `acd1528`)

**Done.** These git-tracked files read `/home/jasper/Repositories/MasterThesisCode` until the §1
pass; all of them now read `darksiren-emri`. Two additions to the list below were folded into the
same commit: `book/README.md:39` (same `.venv/bin/python` invocation class) and, from the last
bullet, `test_30`/`test_31` — those two are *local* dev-box paths in a functional
`REPO = Path(...)`, so only `run_multi_truth_sweep.sh`'s three lines were cluster-path references
deferred to §2. Deliberately **not** rewritten: the `-book` worktree paths (see §1), the
`github.io/MasterThesisCode/` Pages URLs and the line-wrapped path at `BOOK_TECH_DESIGN.md:252`
(dated snapshots, per §4), `docs/H0_BIAS_RESOLUTION.md` (a dated record — its `cd` lines at
1724/2000/2029 are stale by design), and `.claude/settings.local.json`, which is fixed in the
working tree on each box but is globally gitignored and so cannot be committed.

The list as it stood:

- `.claude/skills/known-bugs/SKILL.md:13,16`
- `.claude/skills/physics-change/SKILL.md:68`
- `.claude/settings.local.json:9,12,13` (line 9 is a sibling `-book` worktree path, same pattern)
- `darksiren_emri_test/bayesian_inference/test_posterior_combination.py:418`
- `book/design/BOOK_DESIGN.md:15,16,848`
- `book/design/BOOK_SOURCES_MAP.md:9,621`
- `book/design/BOOK_TECH_DESIGN.md:4,5,11,131,204`
- `book/design/reviews/expert_A_ch00-06_museum.md:12`, `expert_B_ch07-11_cellB.md:4,24`
- `book/generators/*.py` shebang comments and `REPO_ROOT.parent / "MasterThesisCode"` sibling-checkout
  fallback paths (13 files — `gen_ch00.py` through `gen_ch11.py`, `gen_museum.py`, `make_all.py`,
  `qa_gates.py`)
- `scripts/bias_investigation/test_30_f4_estimator_smoothness.py:7`,
  `test_31_completion_term_characterization.py:33`,
  `run_multi_truth_sweep.sh:84,129,133` (the last three are cluster-path references, folded into §2
  instead once the cluster rename lands)

## 4. GitHub Pages / book URL migration

**Done, verified live 2026-08-12.** `https://jasperseehofer.github.io/darksiren-emri/` and
`.../book/` both serve 200. The old path
(`https://jasperseehofer.github.io/MasterThesisCode/`) 404s — GitHub Pages does **not** redirect
a renamed-repo's project-pages URL, confirming the RUNBOOK-9 "redirects are a grace period, not a
permanent alias" warning applied here too (no redirect at all, in fact). `README.md` (Docs +
Interactive Figures badges, hero book link, and the limitations-doc link),
`ROADMAP.md:5`, and `pyproject.toml` `[project.urls] Documentation` are updated to the new URL.
`docs/REBRAND_PROPOSAL.md`, `book/BUILD_REPORT.md`, and `book/design/BOOK_TECH_DESIGN.md` still
reference the old URL deliberately — they are dated reports/design snapshots describing history,
left as-is.

Two things change together once Phase 3's `gh repo rename` lands, but Pages
propagation and any external links need a deliberate check, not an assumption:

- Docs Pages: `https://jasperseehofer.github.io/MasterThesisCode/` →
  `https://jasperseehofer.github.io/darksiren-emri/`. GitHub Pages served from a renamed repo
  updates automatically for a `github.io/<reponame>/` project-pages URL, but **verify** the new
  URL serves before treating old links as safe to leave. Old-name Pages URLs are **not**
  guaranteed to redirect the way `git remote` URLs are — the RUNBOOK-9 warning that "redirects
  are a grace period, not a permanent alias" applies doubly here.
- Book path: `https://jasperseehofer.github.io/MasterThesisCode/book/` →
  `.../darksiren-emri/book/`.
- Update, once verified live:
  - `README.md` — the `Docs` and `Interactive Figures` badge URLs (currently still pointing at
    the old Pages path deliberately, per this checklist)
  - `README.md` book link in the new hero paragraph
  - `ROADMAP.md:5` documentation link
  - `pyproject.toml` `[project.urls] Documentation`
  - Any external links from the papers, once submitted (per RUNBOOK-9 §3.1(c) — a redirect page
    at the old Pages path is worth leaving up if papers already cite the old URL before this
    migration completes).

## 5. PyPI distribution name

**Not done, low priority.** This project is not currently published to PyPI
(`docs/REBRAND_PROPOSAL.md` §5(c) confirmed no `pypi.org/project/master-thesis-code` exists at
survey time). The `pyproject.toml` dist name is now `darksiren-emri` (Phase 1). If/when this
project is ever published, reserve `darksiren-emri` on PyPI early and cheaply (an empty
placeholder release) to prevent squatting — PyPI names, unlike GitHub repos, cannot be reclaimed
or redirected once taken.

## 6. Verification gate for calling this checklist done

- [~] §1: **verified on this dev box 2026-08-15** — a fresh Claude Code session in
      `/home/jasper/Repositories/darksiren-emri` retrieves prior project memory (44 memory files,
      transcripts back to 2026-07-17) and the vault session-start hook briefs the project, so the
      registry Path resolves. §3 references fixed in `acd1528`; venv relocated
      (`uv sync --reinstall` + `activate*` path rewrite); `origin` repointed off the rename
      redirect; `-book` worktree linkage repaired. **Box stays `[~]` until every dev box has
      run `scripts/migrate_local_rename.sh`** — the rename is per-machine, and a box still sitting
      at the old path has a live, broken checkout, not merely a stale one.
      **Second dev box (`jasper-ThinkPad-T490s`) migrated 2026-08-23** - script run from `$HOME`,
      steps 1-3 applied (dir renamed, Claude project state re-keyed), step 4 a no-op (the registry
      already carried the new Path from the first box), step 5 all-skips (references already fixed
      upstream; only the gitignored `.claude/settings.local.json` was rewritten), `origin` repointed
      off the redirect, venv relocated via `uv sync --reinstall` (mypy 1.19.1 / pytest 9.0.2 answer),
      and the two `.claude/worktrees/agent-*` linked worktrees repaired at the new path (the stale
      `/tmp` scratchpad worktree pruned). Flip §1 to `[x]` if no further box remains.
- [~] §2: cluster migration EXECUTED 2026-08-13 — `~/MasterThesisCode` -> `~/darksiren-emri`,
      remote repointed, pulled to `e83ed0b9`, venv rebuilt from scratch (`uv sync --extra gpu`),
      stale `master_thesis_code/` removed. Preflight reads **VERDICT: READY ✓** and V-T3
      `pin_integrity.pass = True` on the renamed checkout. Job array **6303086**
      (`cluster/mechanism_isolation.sbatch`) submitted from `~/darksiren-emri`; **box closes when
      it completes.** Migration notes worth keeping:
      (a) the pull aborted three times on untracked campaign outputs that had since been committed
      from the dev box — 147 files were moved to `~/rename_backup_20260813` only after all 49
      chunk md5s were verified byte-identical to the committed copies, and all 147 re-verified
      against the checkout afterwards (identical=147, differs=0, only-in-backup=0);
      (b) git truncates the "would be overwritten" list, so the move had to loop (67, then 20);
      (c) the gitignored 1.6 GB `reduced_galaxy_catalogue.csv` had to be moved by hand into
      `darksiren_emri/galaxy_catalogue/` — `git pull` cannot relocate an ignored file — and the
      pinned symlink repointed at it (verified against the registered md5 pin, not just existence);
      (d) 4 dead dev-box probe symlinks deleted; 0 symlinks now embed the old name.
- [x] §4: `https://jasperseehofer.github.io/darksiren-emri/` and `.../book/` both resolve and
      serve current content (verified 2026-08-12; old URL 404s, no redirect)
- [ ] §5: PyPI name reserved (only if/when publication becomes real)

Until all boxes are checked, treat the GitHub-repo-level rename (Phase 3) as complete but the
*operational* rebrand as still in progress.
