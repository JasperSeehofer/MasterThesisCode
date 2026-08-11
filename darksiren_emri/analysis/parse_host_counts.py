"""Parse per-event host-galaxy counts from inference log files.

The ``BayesianStatistics.evaluate`` pipeline logs one line per detected EMRI
event of the form::

    [handler.py:399 - get_possible_hosts_from_ball_tree()] Found 58 possible
    hosts without BH mass and 34 possible hosts with BH mass.

This module regex-scans those lines (which are emitted in event-index order
within each log file) and returns a tidy ``pandas.DataFrame`` with one row
per event.

Why log parsing instead of instrumentation? — instrumenting
``BayesianStatistics.evaluate`` to persist a structured CSV would require
re-running the inference (~4 h on the 1 473-event Phase 48 production set).
The host counts are h-independent (they're determined by sky position and
galaxy mass cuts), so a single h-value log captures all events; parsing is
a one-shot read of ~1 MB of text and is fully reproducible. Persistent
instrumentation is a backlog follow-up.
"""

import logging
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# Anchored on the literal substring emitted by
# galaxy_catalogue/handler.py:399 (get_possible_hosts_from_ball_tree).
_HOST_COUNT_RE = re.compile(
    r"Found\s+(?P<no>\d+)\s+possible hosts without BH mass"
    r"\s+and\s+(?P<wm>\d+)\s+possible hosts with BH mass"
)


def parse_host_count_lines(log_path: Path) -> list[tuple[int, int]]:
    """Yield ``(n_without_mass, n_with_mass)`` tuples from *log_path* in order.

    Parameters
    ----------
    log_path:
        Path to a single inference log file.

    Returns
    -------
    list[tuple[int, int]]
        One pair per host-count log line, in the order they appear in the
        log (which is also the order in which events were evaluated).
    """
    out: list[tuple[int, int]] = []
    with open(log_path) as fh:
        for line in fh:
            m = _HOST_COUNT_RE.search(line)
            if m is None:
                continue
            out.append((int(m.group("no")), int(m.group("wm"))))
    return out


def parse_host_counts(
    log_paths: Iterable[Path],
    *,
    expected_n_events: int | None = None,
) -> pd.DataFrame:
    """Build a per-event ``DataFrame`` of host counts from one or more logs.

    The host counts are h-independent (sky-position + mass-cut driven), so
    in principle a single log file at any h is sufficient. When multiple
    logs are supplied, the counts are cross-checked: if two logs disagree
    at the same event index a warning is logged and the first value is
    kept (consistency check rather than averaging — host counts are
    deterministic given the catalog).

    Parameters
    ----------
    log_paths:
        Iterable of paths to inference log files.
    expected_n_events:
        Optional sanity check; if given and the parsed row count does not
        match, a warning is logged but the DataFrame is still returned.

    Returns
    -------
    pandas.DataFrame
        Columns: ``event_idx`` (int), ``n_without_mass`` (int),
        ``n_with_mass`` (int), ``reduction_frac`` (float in [0, 1]).
        Empty DataFrame with the correct columns when no log matches.
    """
    columns = ["event_idx", "n_without_mass", "n_with_mass", "reduction_frac"]

    pairs_per_log: list[list[tuple[int, int]]] = []
    for p in log_paths:
        if not p.is_file():
            _logger.warning("Host-count log not found: %s", p)
            continue
        pairs_per_log.append(parse_host_count_lines(p))

    if not pairs_per_log or all(not p for p in pairs_per_log):
        _logger.warning("No host-count lines parsed from any log.")
        return pd.DataFrame(columns=columns).astype(
            {
                "event_idx": "int64",
                "n_without_mass": "int64",
                "n_with_mass": "int64",
                "reduction_frac": "float64",
            }
        )

    # Use the longest log as primary; cross-check shorter ones.
    pairs_per_log.sort(key=len, reverse=True)
    primary = pairs_per_log[0]
    for other in pairs_per_log[1:]:
        for i, (no_p, wm_p) in enumerate(other):
            if i >= len(primary):
                break
            if (no_p, wm_p) != primary[i]:
                _logger.warning(
                    "Host-count disagreement at event_idx=%d: primary=(%d,%d) other=(%d,%d)",
                    i,
                    primary[i][0],
                    primary[i][1],
                    no_p,
                    wm_p,
                )

    event_idx = np.arange(len(primary), dtype=np.int64)
    n_without = np.array([p[0] for p in primary], dtype=np.int64)
    n_with = np.array([p[1] for p in primary], dtype=np.int64)
    reduction = np.where(n_without > 0, (n_without - n_with) / n_without, 0.0)
    df = pd.DataFrame(
        {
            "event_idx": event_idx,
            "n_without_mass": n_without,
            "n_with_mass": n_with,
            "reduction_frac": reduction.astype(np.float64),
        }
    )

    if expected_n_events is not None and len(df) != expected_n_events:
        _logger.warning("Parsed %d host-count rows, expected %d.", len(df), expected_n_events)
    return df


def build_host_count_csv(
    data_dir: Path,
    *,
    output_csv: Path | None = None,
    h_value_filter: str | None = "h_0_73",
) -> pd.DataFrame:
    """One-shot helper: scan inference logs in *data_dir* and write a CSV.

    Parameters
    ----------
    data_dir:
        Directory containing inference log files
        (``darksiren_emri_*_h_*.log``).
    output_csv:
        Output CSV path. Defaults to
        ``<data_dir>/diagnostics/host_counts.csv``.
    h_value_filter:
        Substring used to filter logs by h-value (e.g. ``"h_0_73"``).
        Since host counts are h-independent, restricting to one h-value
        is faster and yields identical results. Pass ``None`` to scan
        all log files (uses cross-check).

    Returns
    -------
    pandas.DataFrame
        The host-count DataFrame written to disk.
    """
    pattern = "darksiren_emri_*.log"
    # Search both data_dir and an optional logs/ subdirectory. Inference logs
    # (which contain the host-count lines) and figure-generation logs share
    # the same naming pattern; the host-count regex naturally filters which
    # ones actually carry the data.
    candidates: list[Path] = sorted(data_dir.glob(pattern))
    logs_subdir = data_dir / "logs"
    if logs_subdir.is_dir():
        candidates += sorted(logs_subdir.glob(pattern))
    if h_value_filter:
        candidates = [p for p in candidates if h_value_filter in p.name]

    if not candidates:
        raise FileNotFoundError(
            f"No inference logs found under {data_dir} matching '{pattern}'"
            + (f" with filter '{h_value_filter}'" if h_value_filter else "")
        )

    df = parse_host_counts(candidates)
    if df.empty:
        # No host-count lines found — likely only figure-generation logs.
        raise FileNotFoundError(
            f"Found {len(candidates)} log files under {data_dir} but none "
            "contain 'Found N possible hosts ...' lines. Need an inference "
            "log (rsync from the cluster's run directory)."
        )
    if output_csv is None:
        output_csv = data_dir / "diagnostics" / "host_counts.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    _logger.info("Wrote host counts for %d events to %s", len(df), output_csv)
    return df
