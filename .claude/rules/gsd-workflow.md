---
paths: ["**/*"]
description: Workflow entry points (GSD) and the GitHub issue/label/milestone sync contract — lazy-loaded out of CLAUDE.md 2026-08-12
---

<!-- Relocated from CLAUDE.md 2026-08-12 (gardener proposal mtc-02). The GSD marker comments were
     dropped deliberately so a GSD regeneration cannot re-inject the block into CLAUDE.md.
     GSD itself is slated for retirement — if it goes unused through the next iterations, delete
     this file and the pointer in CLAUDE.md. GPD was confirmed retired on 2026-08-12 (mtc-03c);
     its routing table and `.gpd/` references are gone. -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside the `/research-cycle` protocol unless the user explicitly asks to bypass it.

### GitHub Integration

GitHub issues, labels, and milestones are the **external-facing** record of project state. Workflows must keep them in sync as work progresses. This is not optional — stale issues erode trust in the tracker.

**When to update GitHub (mandatory):**

| Event | GitHub action | Command pattern |
|---|---|---|
| A bug or issue is **resolved** by a phase or quick task | Close the issue with a comment referencing the fix (commit, phase, file:line) | `gh issue close N --comment "..."` |
| A new bug is **discovered** during work | Open a new issue with appropriate labels (`bug`, `physics`, `paper-blocker`, etc.) | `gh issue create --title "..." --label "..." --milestone "..."` |
| A phase or milestone is **planned** that maps to open issues | Assign those issues to the relevant GitHub milestone | `gh issue edit N --milestone "..."` |
| A phase **completes** and resolves multiple issues | Close all resolved issues in one pass with per-issue comments | Batch `gh issue close` |
| Work priority changes (e.g., issue becomes paper-blocking) | Update labels accordingly | `gh issue edit N --add-label "paper-blocker"` |
| A new milestone cycle starts | Create a GitHub milestone if one doesn't exist for it | `gh api repos/.../milestones --method POST` |

**Labels to use:**
- `bug` — something is broken
- `physics` — physics formula or scientific correctness
- `paper-blocker` — must fix before paper submission
- `design-choice` — deliberate simplification, documented
- `enhancement` — new feature or improvement
- `documentation` — docs improvement

**Milestone:** The "Paper Submission" milestone tracks all issues that must be resolved before the paper is submitted. All open physics/design issues should be assigned to it.

**What NOT to do:**
- Do not create GitHub issues for internal planning (phase plans, verification checklists) — those belong in `.planning/`
- Do not duplicate TODO.md items as issues unless they represent distinct, actionable bugs or features
- Do not update issues for purely internal refactoring that has no user-facing effect
