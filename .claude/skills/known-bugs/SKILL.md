---
name: known-bugs
description: >
  Show the status of all known physics and code health bugs. Use when deciding
  what to work on next or when the user asks about remaining issues.
disable-model-invocation: true
allowed-tools: Read, Grep, Bash(git log *)
---

## Known Bugs Status

### Current bug list from CLAUDE.md:
!`sed -n '/Known Bugs/,/^---$/p' /home/jasper/Repositories/MasterThesisCode/CLAUDE.md`

### Check which bugs have been fixed (look for [PHYSICS] commits):
!`git -C /home/jasper/Repositories/MasterThesisCode log --oneline --all --grep="PHYSICS" | head -20`

### For each bug, report:
1. **Bug ID + description** (from CLAUDE.md)
2. **Priority** (CRITICAL / HIGH / MEDIUM / LOW)
3. **Status**: check if the file:line still contains the buggy code
4. **Effort estimate**: single-line fix vs multi-file refactor
5. **Dependencies**: does fixing this bug require fixing another first?

### Suggest next action:
- Recommend the highest-priority unfixed bug
- If user specifies a bug, invoke `/physics-change` to begin the fix workflow
