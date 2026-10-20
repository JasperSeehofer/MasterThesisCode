# PROPOSALS — M65 vault wiring (NOT applied)

Per the `project-init` governance contract: *"CLAUDE.md edits are always proposals. New file
creation (entity page, THREAD.md, CONVENTIONS.md) is auto-apply. **Existing file updates are
proposals.**"* Everything in this file touches an existing file, so nothing here has been applied.

**→ confirm and I'll apply** is written against each.

---

## P-1 — `wiki/entities/master-thesis-code.md`: add `## Vault Wiring`

**File:** `/home/ops/Repositories/garden/wiki/entities/master-thesis-code.md`
**Placement:** insert between the existing `## GitHub` section and `## Next Steps`, matching the
`project-init` entity-page template's section order.
**Note:** no filled exemplar of this section exists anywhere in the vault — this is the first, so
it is built strictly from the skill's template (`project-init/SKILL.md` lines 152-161) with the
research-conditional block expanded and `EXP_PREFIX` = `MTC`.

```markdown
## Vault Wiring

- `/consult research "<question>"` — vault-grounded lookups via [[librarian]]
- `/chronicler` — auto-files session lessons, patterns, incidents to the vault
- `/gardener` — .claude/ health audit; run every 3 days or after major skill/agent changes
- **Thread state**: `.planning/research/THREAD.md` + `CONVENTIONS.md` in the repo — read by [[cartographer]]
- **Experiment registry**: `EXP-MTC-*` rows in [[agentic-experiments-research]]
- **Hypothesis rows**: `H-MTC-*` in [[hypothesis-ledger]]; thread row in [[research-portfolio]]
- **Cartographer channel**: `.cartographer-notes.md` (gitignored)
- **Independent falsification**: `/commission --research "<claim>"`

*Wired 2026-09-21 by [[M65-darksiren-emri-into-manyfold]] — the first research project migrated
into the manyfold universe. Note the repo directory is `darksiren-emri` while this page's slug is
`master-thesis-code`: a literal `/project-init` run derives the slug from the directory name and
would create a duplicate `darksiren-emri.md`. Override the slug to `master-thesis-code` when
re-running any vault-wiring tooling against this repo.*
```

**→ confirm and I'll apply**

---

## P-2 — `wiki/entities/master-thesis-code.md`: three stale facts

These are corrections to existing lines, not additions, so they are proposed separately — a
reviewer may want P-1 without them.

**P-2a — the local checkout path is wrong.** Current text in `## GitHub`:

```markdown
- Local checkout: still `~/Repositories/MasterThesisCode` — rename pending, see
  [[darksiren-emri-rename-plan]]
```

The page's own `## Next Steps` already records the 2026-09-13 clone to
`/home/ops/Repositories/darksiren-emri`, so the page contradicts itself. Proposed replacement:

```markdown
- Local checkout: `~/Repositories/darksiren-emri` on the brain host (cloned 2026-09-13). The
  vault slug stays `master-thesis-code`, see [[darksiren-emri-rename-plan]]
```

**P-2b — `## Related` has no link to the hypothesis ledger**, despite the page being the subject
of five `H-MTC-*` rows there. Proposed addition to `## Related`:

```markdown
- [[hypothesis-ledger]] — five `H-MTC-*` hypothesis rows track this thread; two are live (`genmarg`, `realdata`)
```

**P-2d — the deploy-key claim is now stale (verified).** `## Next Steps` states: *"pushes work
only once Jasper adds the public key as a deploy key with write access on GitHub."* M65 pushed
`mission/M65-darksiren-emri-into-manyfold` successfully over `git@github-darksiren` on 2026-09-22
(exit 0, new branch created). **Write access is live.** Proposed: strike that caveat, since it may
be discouraging sessions from pushing work that would otherwise survive.

**P-2c — frontmatter `status: mature`** reads oddly for a thread with two live hypotheses, an
unrun decisive experiment and a kill gate firing in two days. Suggest `status: active`. *Flagged,
not insisted on — "mature" may be deliberate.*

**→ confirm and I'll apply** (each independently)

---

## P-3 — `wiki/index.md`: a stale duplicate entry

**File:** `/home/ops/Repositories/garden/wiki/index.md`

Line 67 is current and correct — **no change proposed**:

```markdown
- [[master-thesis-code]] — Bayesian EMRI→H₀ inference pipeline (Python; renamed `darksiren-emri` 2026-08, vault slug unchanged)
```

Line 622 carries a second, stale entry in a different section:

```markdown
- [[master-thesis-code]] — EMRI Bayesian inference (completed)
```

**"(completed)" is wrong** — the thread has two live hypotheses and an unrun decisive experiment.
Proposed replacement:

```markdown
- [[master-thesis-code]] — EMRI dark-siren H₀ inference; closure claim pending the T-1 blind mock
```

*This is an edit to an existing line, so it is a proposal even though index additions are
normally auto-apply.*

**→ confirm and I'll apply**

---

## P-4 — the project's own `CLAUDE.md`: a Vault Wiring section

**File:** `/home/ops/Repositories/darksiren-emri/CLAUDE.md` (and this worktree's copy)
**Always a proposal** — CLAUDE.md edits are never auto-applied, in either direction.
**Placement:** suggest after the `## Skill-Driven Workflows` table, before `## Python Conventions`,
so the vault skills sit alongside the repo's own skill roster.

Built from `project-init/SKILL.md` Step 8's research/both variant, with `<slug>` →
`master-thesis-code` and `<EXP_PREFIX>` → `MTC`.

```markdown
## Vault wiring (`garden`)

Research infrastructure is managed at `~/Repositories/garden`. See [[handbook]] for the full skill roster.

- **Vault entity page**: `wiki/entities/master-thesis-code.md` — note the slug is decoupled from
  this repo's directory name (`darksiren-emri`); do not let tooling derive it from `basename $PWD`
- **Lookups**: `/consult research "<question>"` — vault-grounded answers via [[librarian]]
- **Session debrief**: `/chronicler` at session end — auto-files lessons, patterns, incidents
- **Portfolio audit**: `/gardener` — .claude/ health check; run every 3 days or after major changes

### Research thread wiring

- **Thread state** (read by `/cartographer`): `.planning/research/THREAD.md` (hard core, live
  hypotheses, kill criteria, claims) + `.planning/research/CONVENTIONS.md` (append-only
  convention lock — honor it; append, never edit). Update THREAD.md same-day after every
  experiment run.
- **Experiment registry**: `EXP-MTC-*` rows in `wiki/meta/agentic-experiments-research.md`.
  Predictions are pre-registered **before** running — never adjusted post-hoc.
- **Independent falsification**: `/commission --research "<claim>"` — adversarial subagent
  before accepting any key result.
- **Cartographer channel**: `.cartographer-notes.md` (gitignored) — proposals from vault research
  reviews land there; go/kill/pivot decisions are the human's, recorded in the vault
  hypothesis-ledger and decision-timeline.

**Thread-level kill criteria live in the vault** (`wiki/meta/hypothesis-ledger.md`, written by the
author 2026-08-11), mirrored into `.planning/research/THREAD.md`. The repo is not their source of
truth; do not restate them here where they can drift.
```

**→ confirm and I'll apply**

---

## P-5 — a note on what M65 did NOT propose

- **No `wiki/entities/darksiren-emri.md`.** A literal `/project-init` run would have created one
  (Step 3 derives the slug from the directory basename, finds no match, and routes to
  auto-create). That would have forked away from the populated `master-thesis-code.md`. Suppressed
  deliberately.
- **No edits to `wiki/meta/hypothesis-ledger.md` or `research-portfolio.md`.** The m1 adjudication
  returns UNRESOLVED and the kill-gate memo asks a question; neither produces a ledger writeback
  until the author rules. Writing a verdict into the ledger now would bank a call the author has
  not made — the same error the chair correctly avoided on 2026-09-04.
- **No status change on `H-MTC-vtransfer`.** Its prune is recorded in the vault as *"PROPOSED, not
  applied — a prune is a kill and the human confirms kills."* That stands.
