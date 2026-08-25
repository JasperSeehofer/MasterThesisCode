from pathlib import Path

import pytest


def requires_artifact(*paths: str) -> pytest.MarkDecorator:
    """Skip a test unless every machine-of-record data artifact in ``paths`` exists.

    Several validation tests read a pinned production CSV (e.g. the CRB CSV
    at :data:`darksiren_emri.validation.correspondence_1d.CRB_CSV_PATH`)
    directly off disk rather than through a fixture/monkeypatch seam --
    genuinely exercising the wiring against real data, not test-double
    machinery. Those files are large, machine-local ``results/`` artifacts
    that are intentionally NOT committed to version control, so they are
    absent on CI runners (and any other checkout without the data-bearing
    machine's local state). Apply this guard so those tests stay fully
    enforced wherever the artifact is present (the data-bearing machine)
    while degrading to a reported skip -- not a failure -- everywhere else.

    Args:
        *paths: One or more artifact paths (absolute or relative to the
            repo root) that must all exist for the guarded test to run.

    Returns:
        A ``pytest.mark.skipif`` decorator, condition True (skip) when any
        path is missing.
    """
    missing = [p for p in paths if not Path(p).is_file()]
    reason = "; ".join(f"machine-of-record artifact not in VCS: {p}" for p in missing)
    return pytest.mark.skipif(bool(missing), reason=reason)
