---
phase: quick
plan: 260330-otu
type: execute
wave: 1
depends_on: []
files_modified: [CLAUDE.md]
autonomous: true
requirements: [CONDENSE-01]

must_haves:
  truths:
    - "CLAUDE.md is ~400 lines (down from ~846)"
    - "All GSD markers (project, stack, conventions, architecture, workflow, profile) preserved"
    - "All essential rules and protocols retained — nothing lost, only verbosity removed"
    - "No new content added"
  artifacts:
    - path: "CLAUDE.md"
      provides: "Condensed project instructions"
      contains: "GSD:stack-start"
  key_links: []
---

<objective>
Condense CLAUDE.md from ~846 lines to ~400 lines by removing redundancy, verbose code examples, resolved bugs, and GSD-managed sections that duplicate hand-written content.

Purpose: Reduce context window cost of loading CLAUDE.md while preserving all essential rules.
Output: Rewritten CLAUDE.md at ~400 lines.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Rewrite CLAUDE.md with condensed content</name>
  <files>CLAUDE.md</files>
  <action>
Use the Write tool to completely rewrite CLAUDE.md. Apply these specific condensations:

**1. Known Bugs (lines 203-234):** Remove all resolved/strikethrough items (bugs 2, 3). Keep only open bugs (1, 4-9). Remove verbose resolution notes. Keep the section header and the two sub-headers (Code health, Physics/mathematics).

**2. Technology Stack (lines 571-653):** Replace the entire ~80-line body between GSD markers with a 5-line summary: "Python 3.13, uv, NumPy/SciPy/Pandas/Matplotlib, CuPy+CUDA 12 (GPU), few/fastlisaresponse (waveforms). See `pyproject.toml` for full dependency list and tool configuration." Keep `<!-- GSD:stack-start -->` and `<!-- GSD:stack-end -->` markers.

**3. Architecture GSD section (lines 704-793):** Replace the entire body between GSD markers with: "See Architecture section above for pipeline descriptions and module responsibilities." Keep `<!-- GSD:architecture-start -->` and `<!-- GSD:architecture-end -->` markers.

**4. Conventions GSD section (lines 655-702):** Replace the entire body between GSD markers with: "See Typing Conventions, Dataclass Conventions, and HPC/GPU sections above. Naming: snake_case functions, PascalCase classes, SCREAMING_SNAKE physics constants. Physics symbols (M, H, d_L) preserved in names (ruff N802/N803/N806/N815/N816 ignored). Docstrings: NumPy-style for new code." Keep `<!-- GSD:conventions-start -->` and `<!-- GSD:conventions-end -->` markers.

**5. Typing Conventions (lines 295-358):** Condense to essentials:
- Keep the opening rule about complete type annotations.
- Python 3.10 syntax: 2 lines of rules (use `list[float]` not `List[float]`, use `X | None` not `Optional[X]`, no `from __future__ import annotations`). Remove the 10-line code example.
- NumPy arrays: keep the `npt.NDArray[np.float64]` rule, remove the code block. One line: "Use `npt.NDArray[np.float64]` for typed arrays. Never use bare `np.ndarray`."
- CuPy: one line saying annotate with numpy types + comment.
- Callable: one line saying use `Callable` from `typing`, never lowercase `callable`. Remove the decorator TypeVar example entirely.
- mypy: keep the 2-line summary about config and flags.

**6. HPC/GPU Best Practices (lines 360-443):** Condense to:
- Keep the intro sentence.
- Array namespace pattern: keep the `_get_xp` code example (the one at lines 370-382) as it's the key pattern. Remove the second usage example (lines 386-392).
- GPU imports guarded: 1 sentence rule.
- Vectorize: 1 sentence rule (no code example).
- Avoid GPU-to-CPU transfers: 1 sentence rule (no code example).
- GPU memory management: 2 sentences (free after step, not in inner loops). No code.
- USE_GPU flag: 1 sentence.

**7. Testing Strategy (lines 446-507):** Condense to:
- Core principle: 1 sentence.
- Running tests: keep the 3-line bash block.
- GPU marker: 1 sentence rule + `@pytest.mark.gpu` inline. Remove code block.
- Math/physical tests must NOT require GPU: 1 sentence.
- xp fixture: 1 sentence description. Remove code block.
- Test priority: keep the 3-item numbered list as-is.
- Guarding cupy imports: 2 sentences. Remove code blocks.

**8. Cluster Deployment (lines 114-151):** Remove Script Inventory table (lines 127-139) and Quick Reference (lines 140-151). Keep intro line pointing to cluster/README.md and the CLI Flags table.

**9. Dataclass Conventions (lines 269-291):** Replace the 2-block example with a condensed version:
```python
# Wrong: bar: MyMutable = MyMutable()  -- crashes Python 3.13
# Correct: bar: MyMutable = field(default_factory=MyMutable)
```

**10. Running Tests (lines 153-168):** Remove entirely — already covered in Environment Setup "Running code" and Testing Strategy sections.

**Sections to keep as-is (copy verbatim):**
- Environment Setup (lines 1-65)
- Dev Workflow (lines 67-99)
- Running the Code (lines 101-112)
- Architecture hand-written (lines 170-201)
- Skill-Driven Workflows (lines 237-266)
- Math/Physics Validation (lines 510-551)
- GSD Project section (lines 553-569, between GSD markers)
- GSD Workflow Enforcement (lines 795-838, between GSD markers)
- Developer Profile (lines 840-845, between GSD markers)

**Formatting:** Maintain section separators (`---`) between major sections. Keep markdown headers at same levels.
  </action>
  <verify>
    <automated>wc -l CLAUDE.md | awk '{if ($1 >= 350 && $1 <= 450) print "PASS: "$1" lines"; else print "FAIL: "$1" lines (target 350-450)"}'</automated>
  </verify>
  <done>CLAUDE.md is 350-450 lines. All GSD markers intact. All essential rules preserved. No resolved bugs. No duplicate GSD sections.</done>
</task>

</tasks>

<verification>
1. `wc -l CLAUDE.md` — should be 350-450 lines
2. `grep -c 'GSD:.*-start' CLAUDE.md` — should be 6 (project, stack, conventions, architecture, workflow, profile)
3. `grep -c 'GSD:.*-end' CLAUDE.md` — should be 6
4. Spot-check: "Physics Change Protocol" section present
5. Spot-check: "_get_xp" pattern example present
6. Spot-check: No "RESOLVED" or strikethrough text remaining
</verification>

<success_criteria>
CLAUDE.md reduced to ~400 lines while preserving all essential rules, protocols, and GSD markers.
</success_criteria>

<output>
After completion, create `.planning/quick/260330-otu-condense-claude-md-remove-redundancy-tri/260330-otu-SUMMARY.md`
</output>
