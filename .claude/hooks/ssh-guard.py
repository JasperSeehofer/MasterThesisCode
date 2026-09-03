#!/usr/bin/env python3
"""PreToolUse hook (Bash): protect the bwunicluster SSH ControlMaster.

Incident 2026-09-03 (ledger row #357): agents polling concurrently filled the
multiplexed connection's server-side session cap (sshd MaxSessions); another
agent read "Session open refused by peer" as a dead master and ran
`ssh -O exit`, destroying the only OTP-authenticated socket for the night.

Blocked (exit 2 + stderr message):
  * any `ssh -O exit|stop|forward|cancel` / `ssh -O check` is allowed
  * deleting the control socket (rm/unlink of ~/.ssh/cm-*)
  * killing ssh processes (pkill/killall ssh, kill <master pid>)
  * a `sleep N` with N >= 60 anywhere inside a remote ssh command string
    (long sleeps must run LOCALLY between short ssh calls)
  * more than one `ssh <host>` invocation chained in a single Bash command
    with `&` (parallel local fan-out onto the same master)
All error paths exit 0 (never block legitimate work by accident).
"""

from __future__ import annotations

import json
import re
import sys

HOSTS = r"(bwunicluster|uc3\.scc\.kit\.edu)"

RULES = [
    (
        re.compile(r"\bssh\b[^\n|;&]*\s-O\s*(exit|stop|forward|cancel)\b"),
        "ssh -O exit/stop on the ControlMaster is forbidden: a refused mux session means the "
        "server-side session cap is full, not a dead master. Wait 60 s and retry (see "
        "cluster/agent_ssh.sh). Only the author re-authenticates (OTP).",
    ),
    (
        re.compile(r"\b(rm|unlink)\b[^\n;&|]*\.ssh/cm-"),
        "deleting the SSH control socket is forbidden.",
    ),
    (
        re.compile(r"\b(pkill|killall)\b[^\n;&|]*\bssh\b"),
        "killing ssh processes is forbidden (the ControlMaster is one of them).",
    ),
]

REMOTE_SLEEP = re.compile(
    r"\bssh\b[^\n]*?" + HOSTS + r"[^\n]*?['\"][^'\"]*?\bsleep\s+(\d+)", re.DOTALL
)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    if payload.get("tool_name") != "Bash":
        return 0
    cmd = (payload.get("tool_input") or {}).get("command") or ""
    if not cmd:
        return 0
    for rx, msg in RULES:
        if rx.search(cmd):
            sys.stderr.write(f"BLOCKED by ssh-guard: {msg}\n")
            return 2
    m = REMOTE_SLEEP.search(cmd)
    if m and int(m.group(2)) >= 60:
        sys.stderr.write(
            "BLOCKED by ssh-guard: a `sleep >= 60` inside a remote ssh command holds a mux "
            "session open and exhausts MaxSessions for every other agent. Poll with SHORT "
            "remote commands and sleep LOCALLY between them (cluster/agent_ssh.sh poll ...).\n"
        )
        return 2
    # parallel fan-out onto the master from one Bash call
    if len(re.findall(r"\bssh\b[^\n&]*" + HOSTS, cmd)) >= 2 and re.search(
        r"\)\s*&|&\s*$|&\s*\n", cmd
    ):
        sys.stderr.write(
            "BLOCKED by ssh-guard: parallel ssh calls onto the shared ControlMaster from one "
            "command; run them sequentially.\n"
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
