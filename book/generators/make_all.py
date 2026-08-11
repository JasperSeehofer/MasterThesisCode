"""Driver for all book data generators.

Auto-discovers every ``gen_ch*.py`` / ``gen_museum*.py`` module in this
directory and runs each one's ``main() -> None`` in sorted (chapter) order.
Chapter agents therefore NEVER edit this file: dropping a correctly named
generator into ``book/generators/`` is registration.

Each generator is deterministic and independently re-runnable; this script
exists only so a fresh clone (or CI) can rebuild ``book/site/data/`` with a
single command:

    /home/jasper/Repositories/MasterThesisCode/.venv/bin/python \\
        book/generators/make_all.py
or, once this repo has its own synced `.venv`:
    uv run python book/generators/make_all.py

Generators are plain modules with a ``main() -> None`` entry point that
write their own file(s) under ``book/site/data/`` -- read-only with respect
to the main package and ``results/`` (only ``book/`` is written).

After every generator has run, the content gates in ``qa_gates.py``
(REVISION_WORKLIST.md §D item 12) are executed against the built site and
data. They fail the build loudly: a gate hit means a page still asserts
something the project has since measured otherwise. Run them on their own
with ``python book/generators/qa_gates.py``.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import qa_gates

GENERATORS_DIR = Path(__file__).resolve().parent


def discover() -> list[str]:
    """Return sorted module names of all chapter/museum generators."""
    names = set()
    for pattern in ("gen_ch*.py", "gen_museum*.py"):
        for path in GENERATORS_DIR.glob(pattern):
            names.add(path.stem)
    return sorted(names)


def main() -> None:
    """Run every generator in its own subprocess.

    Isolation is deliberate (integrator fix, 2026-07-31): generators resolve
    their own source checkout (this worktree vs a sibling ``MasterThesisCode``)
    and import ``darksiren_emri`` from it.  In a single shared process the
    first import wins for every later generator via ``sys.modules``, which
    broke ``gen_ch03`` (it needs the sibling checkout's newer package).  A
    subprocess per generator restores the documented contract that each one is
    independently re-runnable.
    """
    failures: list[str] = []
    for name in discover():
        print(f"--- running {name} ---", flush=True)
        t0 = time.monotonic()
        result = subprocess.run(
            [sys.executable, str(GENERATORS_DIR / f"{name}.py")],
            cwd=GENERATORS_DIR.parent.parent,  # repo root, matching the documented invocation
            check=False,
        )
        if result.returncode != 0:
            failures.append(name)
            print(f"!!! {name} FAILED with exit code {result.returncode}")
        dt = time.monotonic() - t0
        print(f"--- {name} done in {dt:.2f}s ---\n", flush=True)

    # Content gates run even when a generator failed: their report is the
    # most useful thing on the screen when the build is red.
    violations = qa_gates.run()

    if failures:
        print(f"FAILED generators: {', '.join(failures)}")
    if failures or violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
