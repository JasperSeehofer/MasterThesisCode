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
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

GENERATORS_DIR = Path(__file__).resolve().parent


def discover() -> list[str]:
    """Return sorted module names of all chapter/museum generators."""
    names = set()
    for pattern in ("gen_ch*.py", "gen_museum*.py"):
        for path in GENERATORS_DIR.glob(pattern):
            names.add(path.stem)
    return sorted(names)


def main() -> None:
    sys.path.insert(0, str(GENERATORS_DIR))
    failures: list[str] = []
    for name in discover():
        print(f"--- running {name} ---")
        t0 = time.monotonic()
        try:
            module = importlib.import_module(name)
            module.main()
        except Exception as exc:  # noqa: BLE001 -- one broken generator must not block the rest
            failures.append(name)
            print(f"!!! {name} FAILED: {exc!r}")
        dt = time.monotonic() - t0
        print(f"--- {name} done in {dt:.2f}s ---\n")
    if failures:
        print(f"FAILED generators: {', '.join(failures)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
