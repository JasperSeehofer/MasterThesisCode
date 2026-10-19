"""Print the sha256 of every RULE-BLOCK and of the T1-SEAL-PARAMS block in a registration file.

The hashes recorded in ``.planning/research/THREAD.md`` (section "T-1 package hashes") are the
output of this script at ratification; any later edit inside a block changes its hash.
Hashed text is the exact bytes strictly between the BEGIN and END marker lines' markers.

Usage: python scripts/sealed_mock/rule_block_hashes.py <file.md> [<file.md> ...]
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

BLOCK_RE = re.compile(
    r"<!-- RULE-BLOCK:([A-Za-z0-9_-]+) BEGIN -->(.*?)<!-- RULE-BLOCK:\1 END -->", re.S
)

sys.path.insert(0, str(Path(__file__).parent))
import draw_sealed_h as dsh  # noqa: E402


def block_hashes(text: str) -> list[tuple[str, str]]:
    out = [
        (name, hashlib.sha256(body.encode()).hexdigest()) for name, body in BLOCK_RE.findall(text)
    ]
    if dsh.PARAMS_BEGIN in text:
        out.append(("T1-SEAL-PARAMS", dsh.params_sha256(dsh.extract_params_block(text))))
    return out


def main(argv: list[str]) -> int:
    for arg in argv:
        path = Path(arg)
        text = path.read_text(encoding="utf-8")
        print(f"{path}  file-sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")
        for name, digest in block_hashes(text):
            print(f"  {name:<24} {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
