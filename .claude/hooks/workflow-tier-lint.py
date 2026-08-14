#!/usr/bin/env python3
"""PreToolUse hook: hard-block Workflow launches that violate the project's
model-tiering cap (CLAUDE.md, "Orchestration: model & effort tiering").

Rule: at most ~3 top-tier (inherit-model) agent() calls per workflow script;
every other agent() call must carry a cheap model override
(model:'sonnet' or model:'haiku').

Contract: PreToolUse hooks block a tool call by exiting with code 2 and
writing a message to stderr. Any other exit code (notably 0) allows the
tool call to proceed. This script must never crash in a way that blocks
legitimate work by accident -- all error paths exit 0 with a stderr warning.
"""

from __future__ import annotations

import json
import re
import sys

MAX_TOP_TIER_AGENTS = 3

AGENT_CALL_RE = re.compile(r"\bagent\s*\(")
MODEL_OVERRIDE_RE = re.compile(
    r"""model\s*:\s*['"](sonnet|haiku)['"]""",
    re.IGNORECASE,
)

# How far past the call site to look when we can't find a balanced close
# paren (e.g. malformed/truncated script text).
MAX_LOOKAHEAD = 500


def warn(msg: str) -> None:
    print(f"workflow-tier-lint: {msg}", file=sys.stderr)


def find_call_span(script: str, open_paren_idx: int) -> str:
    """Return the text of an agent(...) call starting at the index of its
    opening paren, up to the matching close paren, or MAX_LOOKAHEAD chars
    if no balanced close is found in that window.
    """
    depth = 0
    end = min(len(script), open_paren_idx + MAX_LOOKAHEAD)
    for i in range(open_paren_idx, end):
        ch = script[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return script[open_paren_idx : i + 1]
    # No balanced close found within the lookahead window.
    return script[open_paren_idx:end]


def lint_script(script: str) -> tuple[int, list[int]]:
    """Return (top_tier_count, offending_line_numbers)."""
    offending_lines: list[int] = []
    for m in AGENT_CALL_RE.finditer(script):
        open_paren_idx = m.end() - 1
        assert script[open_paren_idx] == "("
        call_text = find_call_span(script, open_paren_idx)
        if not MODEL_OVERRIDE_RE.search(call_text):
            line_no = script.count("\n", 0, m.start()) + 1
            offending_lines.append(line_no)
    return len(offending_lines), offending_lines


def main() -> int:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        warn("could not parse stdin as JSON; allowing tool call")
        return 0

    if not isinstance(payload, dict):
        warn("stdin JSON was not an object; allowing tool call")
        return 0

    tool_name = payload.get("tool_name")
    if tool_name != "Workflow":
        return 0

    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        tool_input = {}

    script = tool_input.get("script")
    if not isinstance(script, str) or not script:
        script_path = tool_input.get("scriptPath")
        if isinstance(script_path, str) and script_path:
            try:
                with open(script_path, encoding="utf-8") as f:
                    script = f.read()
            except OSError as exc:
                warn(
                    f"could not read scriptPath {script_path!r} ({exc}); "
                    "allowing tool call without linting"
                )
                return 0
        else:
            warn(
                "Workflow tool_input had no inline script or scriptPath "
                "(e.g. a named workflow); could not lint, allowing tool call"
            )
            return 0

    top_tier_count, offending_lines = lint_script(script)

    if top_tier_count > MAX_TOP_TIER_AGENTS:
        lines_str = ", ".join(str(n) for n in offending_lines)
        print(
            f"BLOCKED: workflow has {top_tier_count} top-tier (inherit-model) "
            f"agent() calls (cap is {MAX_TOP_TIER_AGENTS}). Offending call "
            f"site line numbers: {lines_str}. "
            "CLAUDE.md tiering cap: at most ~3 inherit-model agents per "
            "workflow; add model:'sonnet' to fanned-out stages or split "
            "the workflow.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - never crash-block by accident
        warn(f"unexpected error ({exc!r}); allowing tool call")
        sys.exit(0)
