"""Thin CLI shim for the calibration-gate instrument.

The instrument itself lives at ``master_thesis_code/validation/calibration_gate.py``
— the exact location the committed prereg (``b50ccc65``,
``PREREGISTRATION_CALIBRATION_GATE.md`` §0) names for the extension module.
This shim only makes the results directory self-driving:

    cd <repo-root> && uv run python results/calibration_gate_20260808/calibration_gate.py --smoke --cell B2 --allow-dirty
    uv run python results/calibration_gate_20260808/calibration_gate.py --validate
    uv run python results/calibration_gate_20260808/calibration_gate.py --cell B2 --truth 0.730 --seed-range 0:400 --workers 14

Equivalent module form:

    uv run python -m master_thesis_code.validation.calibration_gate <same args>

No logic lives here on purpose: forking the instrument into results/ would
drift from the prereg's registered code identity.
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from master_thesis_code.validation.calibration_gate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
