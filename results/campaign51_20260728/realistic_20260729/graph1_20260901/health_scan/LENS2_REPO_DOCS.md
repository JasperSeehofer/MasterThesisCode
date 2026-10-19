# Lens 2 — Repo + Docs Hygiene

Scan date: 2026-09-02/03 (overnight health scan). Read-only. All commands run against
`/home/jasper/Repositories/darksiren-emri` on branch `fix/p32d-classg-venue-repair`.

---

## 1. Untracked-file sprawl

`git status --porcelain` shows **1,113 untracked entries**, one modified tracked file
(`DATA_INVENTORY.md`), zero staged.

Breakdown by top-level bucket:

| Bucket | Count | Classification |
|---|---|---|
| `results/campaign51_20260728/realistic_20260729/...` (mostly `graph1_20260901/exec/**/ca_rhs_work/`, `score_chunk*_{coded,twin}_work/`) | 1,082 | **SHOULD-BE-GITIGNORED DATA.** These are per-chunk working directories from the graph-1 campaign's in-flight execution (`ca_rhs_acceptance_output.json`, `fidelity_*_work/`, `score_chunk{0,100-108,...}_{coded,twin}_work/`). They match the existing `.gitignore` *intent* (campaign51 bulk mirrors are already excluded by several narrow rules) but these specific `graph1_20260901/exec/**/*_work/` directories are **not yet covered** — only 8 hand-picked sub-paths under `graph1_20260901/` are ignored (see §6). This is why `git status` is showing them at all. |
| `results/prod2d_closure_20260818/`, `results/run_20260817_fusion_counterfactual/`, `results/run_20260805_d1/`, `results/run_20260805_n2sel1d/`, `results/run_20260804_postfix/`, `results/run_20260804_frozeng/`, `results/run_20260628_seed600_figures/`, `results/run_20260620_seed500_phase50/` | 24 | **GENUINELY STALE.** These are old, closed-thread run directories (row #119 thread CLOSED for `fusion_counterfactual`; `run_20260620_seed500_phase50` is explicitly flagged in `DATA_INVENTORY.md`'s new Local Storage Register as "pre-2026-06-20 RETIRED era — cold, safe to delete"). They are untracked *and* the author's own new inventory section marks two of them as delete candidates. Effort: S (author sign-off, then `rm -rf` + no gitignore change needed since results/ is already broadly ignored — see §6 anomaly). |
| `selection_tables_h_0_73.json/`, `selection_tables_h_0_735.json/`, `selection_tables_h_0_725.json/` (repo root) | 3 | **STRAY / SHOULD BE UNDER `results/`.** These sit at repo root, not under `results/`, so they are NOT covered by any `results/...` gitignore rule. Named like per-h selection-table dumps from the `write_selection_table_json` instrumentation (row #133 battery). Root-level placement looks like an accidental `cwd`-relative write (script run from repo root instead of a run dir). Should be deleted/relocated, and if this write pattern recurs, a root-level `.gitignore` line is needed. Effort: S. |
| `scripts/bridge_closure` | 1 | Untracked script/dir at `scripts/` — not data, worth a look before the next commit touches `scripts/` (could be a forgotten deliverable or scratch). Effort: S to triage. |
| `docs/CLAUDE_SCIENCE_ABSTRACT.md`, `docs/CLAUDE_SCIENCE_BRIEF.md` | 2 | **RECORDS-THAT-SHOULD-BE-COMMITTED (needs author word).** These are docs-tree Markdown, not data — the kind of artifact CLAUDE.md's "Proposing decisions" convention says belongs in a reviewable, persistent artifact. Untracked for what looks like several days at minimum (present in git status at scan time, no matching recent commit touches them). If these are live/current science communication, they should be committed; if superseded drafts, they should be deleted. NEEDS-AUTHOR-WORD — can't tell content currency without reading them, and lens scope is read-only/hygiene not content review. |

**Overall verdict:** the sprawl is ~97% legitimately-should-be-gitignored working data from the
active graph-1 campaign, not accumulated cruft requiring triage file-by-file — but the
`.gitignore` pattern that would suppress it cleanly does not exist yet (see §6). SAFE-HYGIENE to
add the pattern; the 24 stale run-dirs and the docs/CLAUDE_SCIENCE_*.md pair are
NEEDS-AUTHOR-WORD (deletion / commit decisions).

## 2. Modified tracked file: `DATA_INVENTORY.md`

**Diff is NOT an unfinished registration — it is a completed, self-contained addition.** The diff
adds:
1. Device tags (`thinkpad`/`bwuni`) retrofitted onto ~6 pre-existing provenance rows, three of
   which are corrected from "durable copy exists" to "⚠️ VERIFIED ABSENT 2026-09-02" (the
   `~/data-backups/` tree and two `simulations/` CSVs are gone with no prior ledger row recording
   the loss).
2. A new, complete "Local Storage Register" section (~130 lines): device registry, a
   single-filesystem finding (`/` and `/home` share one partition — **no real local redundancy
   exists today**), a dated 161 GB dedup action log, a "what's on disk now" table, and an explicit
   storage decision put to the author (buy ≥2 TB external SSD; evacuate `~/emri-archive/` first
   since it's the sole copy of 159 GB; institutional archive still unidentified).

This reads as complete prose with an explicit "Open action" callout at the end, not a half-written
draft — it just hasn't been committed yet. **NEEDS-AUTHOR-WORD**, but not for hygiene reasons: the
content itself asks the author three things (buy storage, evacuate `~/emri-archive/`, identify the
institutional archive) and should be committed once the author has seen it, not sat on as a diff.
Severity: MEDIUM (the underlying disk situation it describes — 82% full, cluster workspace expiring
2026-09-23, 159 GB with zero backup — is itself a real operational risk this scan should flag up,
even though the file describing it is fine). Effort: S (git add + commit once author acknowledges).

## 3. Branch state — `fix/p32d-classg-venue-repair`

- `git rev-list --left-right --count main...fix/p32d-classg-venue-repair` → **0 behind, 112 ahead**.
  Not diverged in the conflicting sense (no rebase needed against main), but 112 unmerged commits
  is a large, long-lived branch.
- Root commit on the branch: `3694233d fix(harness): P3-2D class-G venue — reject M<=0 latents...`
  — a genuine, scoped venue-repair fix, consistent with the branch name.
- But the branch's later ~90 commits carry an entirely different, much larger scope: the full
  fan-out-1 campaign (rows #212–#254, charter B1–B8), tree-2 (rows #255–#276), the wave-3/A18 flip
  work (rows #278–#287), and the just-ratified Research Graph 1 charter (row #290) — six `[PHYSICS]`
  production-default flips are on this branch (θ-hook, mass window, catalogue-leg twin, 2D
  `mz_sel`, `theta_phi_divisor`, the 1D mass-aware flip `5e7fda16`).
- **The branch name is now materially misleading** for anyone reading `git log --oneline main..HEAD`
  cold — "venue repair" describes commit 1 of 112.
- **Merge-to-main is overdue by any normal cadence**: it has been carrying live production physics
  flips for weeks without landing on `main`, meaning `main` is stale relative to the actual
  production configuration the campaign is running against. This is a real risk if anyone (human or
  agent) checks out `main` expecting it to reflect current production defaults.

**Verdict:** SEVERITY HIGH / NEEDS-AUTHOR-WORD. Two decisions bundled: (a) rename/re-scope the
branch (or cut a new branch for future work and let this one merge as-is), and (b) schedule the
actual merge — this is a science-authorship decision (which flips are "ready for main"), not a
hygiene one, so it must go back to the author, not be auto-merged. Effort: L (not for the git
mechanics, which are cheap, but for the review/verification gate a 112-commit six-physics-flip
merge should get before landing on main).

## 4. GitHub state

`gh issue list` (17 open, all `enhancement`/`bug`/`physics` tagged, oldest from 2026-07-04):
- Several open issues pre-date and are **not clearly reconciled** with CLAUDE.md's "Known Bugs"
  section: issue #57 (`sigma_multiplier` dead code) and #25 (`get_redshift_outer_bounds` silently
  ignores `sigma_multiplier`/`Omega_m` bounds) look like duplicates of each other (same function,
  same defect, filed 2026-07-04 and 2026-08-28) — worth a dedup pass.
- CLAUDE.md's Known Bugs section (items 1–9) is **almost entirely struck-through/resolved** and
  does not mention any of the currently-open GitHub issues (#39–#58, the redteam/paper-blocker
  physics items, the flaky-CI issue #56, or the smeared-quadrature perf issue #58 from
  2026-08-29). The two trackers have drifted apart: CLAUDE.md's list is a stale historical log of
  now-fixed bugs, while the live backlog lives entirely in GitHub issues. Not necessarily wrong
  (CLAUDE.md may intend to only document *fixed* items as an audit trail — the `/known-bugs` skill
  trigger presumably reads GitHub directly) but worth confirming that's the intended split.
- Milestone "Paper Submission": 11 open / 13 closed issues, `updated_at` 2026-08-14, `due_on: null`.
  No due date set despite the paper-blocker-tagged issues (#40, #52) still open — can't assess
  "accuracy" against a schedule that doesn't exist. At minimum three open issues are tagged
  `paper-blocker` (#52, #40, #23) and should be the visible milestone-blocking set; worth checking
  they're the ones actually gating submission.
- `gh pr list`: 2 open PRs — #47 (`paper: fill production-run numbers`, opened 2026-07-26, still
  has "22/24 pending markers" per its title — over five weeks stale) and #14 (dependabot
  actions/checkout bump, opened 2026-06-25, over two months stale, zero-risk to merge).

**Verdict:** SEVERITY MEDIUM. PR #14 (dependabot) is SAFE-HYGIENE to merge (mechanical, no physics
content) — effort S. PR #47 and the milestone accuracy are NEEDS-AUTHOR-WORD (content decisions).
Issue dedup (#25/#57) is SAFE-HYGIENE to flag/link but closing needs author confirmation — effort S.

## 5. Docs staleness

| File | Last touched | Gap vs current state |
|---|---|---|
| `CHANGELOG.md` | 2026-08-31 (`5e7fda16`, the 1D mass-aware flip) | Reasonably current — captures the most recent `[PHYSICS]` flip. Does **not** mention the 2026-09-01 Research Graph 1 charter ratification (row #290) or the branches-A-I trigger, since those are docs-only/process events rather than code changes — consistent with the file's apparent scope (code changes only). |
| `TODO.md` | 2026-08-12 | **3 weeks stale.** Predates essentially the entire fan-out-1 campaign, tree-2, the wave-3/A18 flip, and Research Graph 1. If TODO.md is meant to reflect a live backlog it is now describing a different project-state than reality; if it's only meant for author-dictated free-time items (its last entry: "five author-dictated future-free-time items") it may be intentionally slow-moving — worth a one-line author confirmation of scope. |
| `README.md` | 2026-08-12 | Also 3 weeks stale, but README content (branding/Pages URLs, architecture overview) is less time-sensitive than TODO — lower urgency. |
| `CLAUDE.md` | 2026-08-20 (research-cycle A10–A14 adoption) | 22,581 bytes. The file itself references "keep this file within its byte budget" (line 349) but never states the actual numeric budget anywhere I could find in CLAUDE.md or `.claude/rules/*.md` — the constraint is referenced, not documented, so there's no way to check current compliance against a number. Also 12 days stale relative to: the A18 production flip (`5e7fda16`, 08-31), Research Graph 1 charter ratification (row #290, 09-01), and five more `[PHYSICS]` commits landed since 08-20. The **Known Bugs section is accurate but inert** (see §4) — every live item has moved to GitHub issues and CLAUDE.md's list is now purely a historical/resolved-bugs record, which may be fine but isn't stated as the design intent anywhere in the file. |

**Verdict:** SEVERITY LOW-MEDIUM (none of these gaps look like they're blocking work — the
operative process docs, `docs/gates/PHYSICS-GATE-LEDGER.md` and the runbooks referenced in
MEMORY.md, are current). NEEDS-AUTHOR-WORD on scope confirmation for TODO.md and on stating the
CLAUDE.md byte budget number explicitly; effort S for both once scope is confirmed.

## 6. `docs/gates/PHYSICS-GATE-LEDGER.md` integrity vs `[PHYSICS]` commits

Checked all `[PHYSICS]`-tagged commits since the ledger's 2026-07-30 start (30 commits) against
ledger rows. **Naive hash-grep initially flagged 6 commits as gate-less**
(`5e7fda16`, `7e1ed96f`, `62f7d61e`, `6c6f2a63`, `d4765539`, `1f003da6`) — but this is a **false
positive from the ledger's own convention**, not a real gap: gate rows are written *before* the
commit lands (the "presented"/"before code" rows are dated and cited against a proposal doc), and
the "implemented"/"verified" rows that follow record the commit ref as the **literal string
`pre-commit`**, which is never back-filled with the real hash once the commit is made. Content
cross-check (by date + flag name) confirms all 6 of the flagged commits have full presented →
implemented → verified row triads (e.g. `5e7fda16` ↔ ledger line 122; `d4765539` ↔ lines 100–102;
`6c6f2a63`/`1f003da6` also confirmed present).

**This is nonetheless a real hygiene finding**: the `pre-commit` placeholder means the ledger is
**not hash-traceable** — CLAUDE.md's own stated purpose ("a `[PHYSICS]` commit with no ledger row
is a gate that cannot be shown to have run") is undermined for every row using the placeholder,
because `git log --grep` / hash-based auditing (exactly what this lens attempted) silently
misses them. A future `/check` or audit pass using hash-matching will keep producing false
positives, and one day a *real* gate-less commit could hide inside that noise. Fix: either (a)
back-fill the real hash into ledger rows via a small script run right after each `[PHYSICS]` commit
lands, or (b) document the `pre-commit`-placeholder convention explicitly in
`docs/gates/README.md` so auditors know to content-match, not hash-match. SEVERITY MEDIUM
(auditability gap, not a compliance gap — no evidence any gate was actually skipped).
Effort: S (doc the convention) to M (retroactive hash back-fill script).

## 7. `.gitignore` coherence

Current file (129 lines) is broadly well-organized into commented sections (data files, figures,
run metadata, worktrees, references, LaTeX, credentials, internal agent state, campaign #51/#53
mirrors, D1 pools, commission notes, cluster-staging, wave-2/3 retrieved data). But the
**campaign51/graph1 section has drifted into an anti-pattern**: 8 separate, hand-typed, fully
literal paths under `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/...`
(e.g. `.../m-t5-armR-c0prime/eval/retrieved_run/`, `.../b-hprior-fix/byteid_eval/run_20260902_...`,
`.../m-s0b-byteid/local_discharge_run/`), each added as the campaign produced a new working
directory — this is exactly the sprawl seen in §1: 1,082 untracked entries under
`graph1_20260901/` that these 8 literal lines do NOT cover, because new node/branch subdirectories
get created faster than lines get added.

**Cleaner pattern rule available**: every other campaign-data section in the file already uses a
glob at the right level (`results/h_sweep_*/`, `results/campaign51_20260728/pool_mix200k/`,
`results/campaign51_20260728/run_seed*/prepared_*.csv`). The graph1 section should collapse to
something like:
```
results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/**/*_work/
results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/**/retrieved_run/
results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/
```
(retaining the existing narrow first line) rather than growing one literal line per node directory
every session. This is the direct fix for the 1,082-entry sprawl in §1.

**Verdict:** SAFE-HYGIENE — this is a mechanical pattern consolidation with no science content risk
(it only ever narrows what's *tracked* by making more things ignored, and `results/` data is
already established as cluster-is-copy-of-record / regenerable-from-seed throughout the file).
Severity MEDIUM (it's the direct cause of the git-status noise that could hide a real accidental
`git add` of multi-GB data), Effort S.

---

## Summary table

| # | Finding | Severity | Effort | Category |
|---|---|---|---|---|
| 1 | 1,082 untracked graph1 working-dirs — `.gitignore` pattern doesn't cover new node dirs | MEDIUM | S | SAFE-HYGIENE (fix in §7) |
| 1b | 3 stray `selection_tables_h_0_*.json/` dirs at repo root | LOW | S | SAFE-HYGIENE |
| 1c | 24 untracked stale run-dirs, 2 explicitly flagged deletable by DATA_INVENTORY.md itself | LOW | S | NEEDS-AUTHOR-WORD (deletion) |
| 1d | `docs/CLAUDE_SCIENCE_ABSTRACT.md` / `_BRIEF.md` untracked | LOW-MED | S | NEEDS-AUTHOR-WORD (commit or discard) |
| 2 | `DATA_INVENTORY.md` diff = completed storage-crisis writeup, not half-finished; describes a real 82%-full-disk / expiring-workspace / single-copy risk | MEDIUM | S | NEEDS-AUTHOR-WORD |
| 3 | Branch 112 commits ahead of main, name ("venue repair") no longer matches scope (whole graph-1 campaign + 6 production `[PHYSICS]` flips); merge overdue | HIGH | L | NEEDS-AUTHOR-WORD |
| 4 | GitHub: PR #14 (dependabot) safe to merge; PR #47 5+ weeks stale; issues #25/#57 likely duplicates; milestone has no due date despite paper-blockers open | MEDIUM | S | mixed (PR#14 safe; rest needs author) |
| 5 | TODO.md/README.md 3 weeks stale; CLAUDE.md 12 days stale and references an undocumented "byte budget" number; Known Bugs section is a historical log, not live (live backlog fully moved to GitHub, undocumented as intentional) | LOW-MED | S | NEEDS-AUTHOR-WORD (confirm scope) |
| 6 | PHYSICS-GATE-LEDGER rows use literal `pre-commit` placeholder, never back-filled with real hash → hash-based gate auditing produces false positives (all 6 initially-flagged commits actually have full gate triads on content check) | MEDIUM | S–M | SAFE-HYGIENE (document convention) |
| 7 | `.gitignore` graph1 section: 8 hand-typed literal paths, one per campaign node dir, instead of a `**/*_work/`-style glob — direct cause of finding #1 | MEDIUM | S | SAFE-HYGIENE |
