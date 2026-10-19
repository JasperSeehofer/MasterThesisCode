"""T-1 sealed-truth draw for the (m2) anti-tuning mock (EXP-MTC-T1).

Implements the commitment mechanism of
``results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-sealed-mock/
REGISTRATION_DRAFT.md`` section 4 item 2, as amended by ``ERRATUM_m2_R27_R28.md`` (M162).
Standard library only. Nothing in this file is physics: it draws a number, seals it, and
proves later when it was fixed relative to the frozen code.

Three subcommands:

``draw``
    Run ONCE, by a human or by a script on the cluster login node, never by an agent.
    Reads the registered parameter block (``T1-SEAL-PARAMS``) from the ratified erratum,
    checks its sha256 against the value recorded at ratification, checks the code freeze
    (HEAD == H_code, ``darksiren_emri/`` tree hash == the registered tree, dirty count 0),
    draws ``h_inj ~ U[prior_lo, prior_hi]`` from ``secrets.SystemRandom`` (never a seeded
    RNG: a public seed is a leak), writes ``SEALED_H_INJ.json`` (mode 0400, directory 0700,
    exclusive create: a second draw is refused, revision cap 1) and prints ONLY the
    commitment, the params hash, H_code and the UTC stamp.

``unseal``
    The unseal ceremony's first step (``UNSEAL_CEREMONY_m2.md``). Human-run. Reads the sealed
    file, recomputes the commitment, and prints ``{h_inj, salt, commitment, match}``.

``verify``
    Pure hash check for the G-3 gate: recomputes ``sha256(f"{h_inj}|{salt}")`` from values
    quoted in the unseal ledger row and compares it with the launch-row commitment. Any
    agent may run it; it reads no sealed file.

Usage::

    python scripts/sealed_mock/draw_sealed_h.py draw \\
        --erratum <path to ERRATUM_m2_R27_R28.md> --params-sha256 <hex from THREAD.md> \\
        --h-code <40-hex commit> [--repo .] [--sealed-dir ~/.sealed/t1-m2]
    python scripts/sealed_mock/draw_sealed_h.py unseal --commitment <hex> [--sealed-dir ...]
    python scripts/sealed_mock/draw_sealed_h.py verify --h-inj 0.801234 --salt <hex> \\
        --commitment <hex>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PARAMS_BEGIN = "<!-- T1-SEAL-PARAMS BEGIN -->"
PARAMS_END = "<!-- T1-SEAL-PARAMS END -->"
SEALED_FILENAME = "SEALED_H_INJ.json"
DEFAULT_SEALED_DIR = Path.home() / ".sealed" / "t1-m2"

# Second lock on the registered prior and grid bounds (REGISTRATION_DRAFT.md section 5, P-A;
# docket R10). The params block must agree with these or the draw is refused.
REGISTERED_PRIOR = (0.79, 0.82)
GRID_FLOOR, HOST_WINDOW_CEILING = 0.60, 0.86

# Environment variables set inside Claude Code sessions. Agents never draw or unseal.
AGENT_ENV_MARKERS = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")


class SealError(RuntimeError):
    """A registered precondition of the draw or the unseal does not hold."""


def extract_params_block(text: str) -> str:
    """Return the text strictly between the params markers (exactly one block required)."""
    if text.count(PARAMS_BEGIN) != 1 or text.count(PARAMS_END) != 1:
        raise SealError("erratum must contain exactly one T1-SEAL-PARAMS block")
    start = text.index(PARAMS_BEGIN) + len(PARAMS_BEGIN)
    end = text.index(PARAMS_END)
    if end <= start:
        raise SealError("T1-SEAL-PARAMS END precedes BEGIN")
    return text[start:end]


def params_sha256(block: str) -> str:
    return hashlib.sha256(block.encode("utf-8")).hexdigest()


def parse_params(block: str) -> dict[str, str]:
    """Parse ``key = value`` lines; lines starting with ``#`` or a code fence are skipped."""
    params: dict[str, str] = {}
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            continue
        m = re.fullmatch(r"([a-z_][a-z0-9_]*)\s*=\s*(\S+)", line)
        if m is None:
            raise SealError(f"unparseable params line: {raw!r}")
        params[m.group(1)] = m.group(2)
    for key in ("prior_lo", "prior_hi", "code_tree", "commitment_format", "revision_cap"):
        if key not in params:
            raise SealError(f"params block lacks {key}")
    return params


def check_params(params: dict[str, str]) -> tuple[float, float]:
    lo, hi = float(params["prior_lo"]), float(params["prior_hi"])
    if (lo, hi) != REGISTERED_PRIOR:
        raise SealError(f"params prior [{lo}, {hi}] != registered P-A {REGISTERED_PRIOR}")
    if not GRID_FLOOR < lo < hi < HOST_WINDOW_CEILING:
        raise SealError("prior outside the evaluable window (0.60, 0.86)")
    if params["commitment_format"] != "sha256(h_inj_6dp|salt_hex)":
        raise SealError("unregistered commitment format")
    if params["revision_cap"] != "1":
        raise SealError("sealed-draw revision cap must be 1")
    if not re.fullmatch(r"[0-9a-f]{40}", params["code_tree"]):
        raise SealError("code_tree must be a full 40-hex git tree hash")
    return lo, hi


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


def check_code_freeze(repo: Path, h_code: str, code_tree: str) -> None:
    """Strict equality (draft section 4 item 1): no ancestor form in a sealed run."""
    if not re.fullmatch(r"[0-9a-f]{40}", h_code):
        raise SealError("--h-code must be a full 40-hex commit hash")
    head = _git(repo, "rev-parse", "HEAD")
    if head != h_code:
        raise SealError(f"HEAD {head} != H_code {h_code} (strict equality required)")
    tree = _git(repo, "rev-parse", "HEAD:darksiren_emri")
    if tree != code_tree:
        raise SealError(f"darksiren_emri/ tree {tree} != registered {code_tree}")
    dirty = _git(repo, "status", "--porcelain", "--", "darksiren_emri")
    if dirty:
        n = len(dirty.splitlines())
        raise SealError(f"darksiren_emri/ dirty count {n} != 0")


def refuse_agent_session() -> None:
    hits = [k for k in AGENT_ENV_MARKERS if os.environ.get(k)]
    if hits:
        raise SealError(f"refusing inside an agent session ({', '.join(hits)} set)")


def commitment_of(h_inj_6dp: str, salt_hex: str) -> str:
    return hashlib.sha256(f"{h_inj_6dp}|{salt_hex}".encode()).hexdigest()


def draw(
    erratum: Path, expected_params_sha: str, h_code: str, repo: Path, sealed_dir: Path
) -> dict[str, str]:
    refuse_agent_session()
    block = extract_params_block(erratum.read_text(encoding="utf-8"))
    got_sha = params_sha256(block)
    if got_sha != expected_params_sha:
        raise SealError(f"params block sha256 {got_sha} != ratified {expected_params_sha}")
    params = parse_params(block)
    lo, hi = check_params(params)
    check_code_freeze(repo, h_code, params["code_tree"])

    sealed_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(sealed_dir, 0o700)
    target = sealed_dir / SEALED_FILENAME
    if target.exists():
        raise SealError(f"{target} exists: the seal is spent (revision cap 1)")

    h_inj = lo + (hi - lo) * secrets.SystemRandom().random()
    h_inj_6dp = f"{h_inj:.6f}"
    salt = secrets.token_hex(32)
    utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = {
        "h_inj": h_inj_6dp,
        "salt": salt,
        "commitment": commitment_of(h_inj_6dp, salt),
        "prior": [lo, hi],
        "h_code": h_code,
        "code_tree": params["code_tree"],
        "params_sha256": got_sha,
        "utc": utc,
    }
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")
    os.chmod(target, 0o400)
    # Public receipt: everything except h_inj and salt.
    return {k: str(record[k]) for k in ("commitment", "params_sha256", "h_code", "utc")}


def unseal(sealed_dir: Path, expected_commitment: str) -> dict[str, object]:
    refuse_agent_session()
    record = json.loads((sealed_dir / SEALED_FILENAME).read_text(encoding="utf-8"))
    recomputed = commitment_of(record["h_inj"], record["salt"])
    return {
        "h_inj": record["h_inj"],
        "salt": record["salt"],
        "commitment_launch_row": expected_commitment,
        "commitment_recomputed": recomputed,
        "match": recomputed == expected_commitment == record["commitment"],
    }


def verify(h_inj_6dp: str, salt_hex: str, expected_commitment: str) -> dict[str, object]:
    if not re.fullmatch(r"\d\.\d{6}", h_inj_6dp):
        raise SealError("h_inj must be quoted to exactly 6 decimals, e.g. 0.801234")
    recomputed = commitment_of(h_inj_6dp, salt_hex)
    return {"commitment_recomputed": recomputed, "match": recomputed == expected_commitment}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_draw = sub.add_parser("draw", help="draw and seal h_inj (human or cluster script only)")
    p_draw.add_argument("--erratum", type=Path, required=True)
    p_draw.add_argument("--params-sha256", required=True)
    p_draw.add_argument("--h-code", required=True)
    p_draw.add_argument("--repo", type=Path, default=Path.cwd())
    p_draw.add_argument("--sealed-dir", type=Path, default=DEFAULT_SEALED_DIR)
    p_unseal = sub.add_parser("unseal", help="unseal ceremony step 1 (human only)")
    p_unseal.add_argument("--commitment", required=True)
    p_unseal.add_argument("--sealed-dir", type=Path, default=DEFAULT_SEALED_DIR)
    p_verify = sub.add_parser("verify", help="G-3 hash check from quoted values")
    p_verify.add_argument("--h-inj", required=True)
    p_verify.add_argument("--salt", required=True)
    p_verify.add_argument("--commitment", required=True)
    args = parser.parse_args(argv)

    try:
        if args.cmd == "draw":
            out: dict[str, object] = dict(
                draw(args.erratum, args.params_sha256, args.h_code, args.repo, args.sealed_dir)
            )
        elif args.cmd == "unseal":
            out = unseal(args.sealed_dir, args.commitment)
        else:
            out = verify(args.h_inj, args.salt, args.commitment)
    except (SealError, subprocess.CalledProcessError, OSError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2))
    return 0 if out.get("match", True) else 1


if __name__ == "__main__":
    sys.exit(main())
