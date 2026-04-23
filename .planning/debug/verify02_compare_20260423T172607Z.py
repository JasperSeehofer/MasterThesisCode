"""VERIFY-02 abort-gate comparison driver.

Reads v2.1 archived posteriors and v2.2 re-evaluated posteriors, extracts
BaselineSnapshots from both, runs generate_comparison_report, computes the
KS statistic on log_posterior curves (D-03 #4), and writes the VERIFY-02
verdict markdown.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from master_thesis_code.bayesian_inference.evaluation_report import (  # noqa: E402
    BaselineSnapshot,
    extract_baseline,
    generate_comparison_report,
)

TRUE_H = 0.73
ABORT_THRESHOLD = 0.05  # |ΔMAP| / 0.73

# Paths — this script lives in a git worktree at .claude/worktrees/agent-acde7378/.planning/debug/
# The simulations/ directory is gitignored and lives only in the main repo root.
# parents[2] from this file = agent-acde7378 worktree root (no simulations/).
# We resolve the main repo root by following the git common directory.
import subprocess

_git_common = (
    subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=str(Path(__file__).resolve().parent),
        stderr=subprocess.DEVNULL,
    )
    .decode()
    .strip()
)
# --git-common-dir returns the .git directory of the main worktree (e.g., /path/to/repo/.git)
REPO = Path(_git_common).resolve().parent

TS_FILE = Path(__file__).resolve().parent / "verify_gate_timestamp.txt"
TS = TS_FILE.read_text().strip()

ARCHIVE_DIR = REPO / "simulations" / "_archive_v2_1_baseline"
CURRENT_DIR = REPO / "simulations"
DEBUG_DIR = Path(__file__).resolve().parent
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

report_md = DEBUG_DIR / f"verify02_abort_check_{TS}.md"
report_json = DEBUG_DIR / f"verify02_comparison_{TS}.json"

# --- Extract baselines ---
baseline = extract_baseline(
    posteriors_dir=ARCHIVE_DIR / "posteriors",
    crb_csv_path=None,  # CRB CSV unchanged since v2.1; not needed for MAP/CI
    true_h=TRUE_H,
)
current = extract_baseline(
    posteriors_dir=CURRENT_DIR / "posteriors",
    crb_csv_path=None,
    true_h=TRUE_H,
)

# --- D-03 #4: Two-sample KS on log_posterior curves ---
# Align on h-values present in both sweeps
b_map = dict(zip(baseline.h_values, baseline.log_posteriors))
c_map = dict(zip(current.h_values, current.log_posteriors))
common_h = sorted(set(b_map) & set(c_map))
b_lp = np.array([b_map[h] for h in common_h])
c_lp = np.array([c_map[h] for h in common_h])
ks_stat, ks_pvalue = ks_2samp(b_lp, c_lp)

# --- Generate the standard Markdown + JSON via existing helper ---
generate_comparison_report(baseline, current, DEBUG_DIR, label=f"verify02_{TS}")

# --- D-03 #1: Abort-gate verdict ---
delta_map = abs(current.map_h - baseline.map_h)
delta_map_relative = delta_map / TRUE_H
abort = delta_map_relative >= ABORT_THRESHOLD

# --- Compose the VERIFY-02 abort-check Markdown ---
verdict = "ABORT" if abort else "PASS"
lines = [
    "# VERIFY-02: Abort-Gate Check",
    "",
    f"**Timestamp:** {TS}",
    "**Phase:** 40 Wave 2",
    "**Requirement:** VERIFY-02",
    "**Gate rule (D-03 #1):** `|MAP_v2.2 - MAP_v2.1| / 0.73 >= 0.05` → ABORT",
    "",
    "## Summary",
    "",
    "| Metric                            | v2.1 (baseline) | v2.2 (current) | Delta         | Role           |",
    "|-----------------------------------|-----------------|----------------|---------------|----------------|",
    f"| MAP h                             | {baseline.map_h:.4f}          | {current.map_h:.4f}         | {current.map_h - baseline.map_h:+.4f}      | ABORT gate     |",
    f"| CI lower (68%)                    | {baseline.ci_lower:.4f}          | {current.ci_lower:.4f}         | {current.ci_lower - baseline.ci_lower:+.4f}      | report only    |",
    f"| CI upper (68%)                    | {baseline.ci_upper:.4f}          | {current.ci_upper:.4f}         | {current.ci_upper - baseline.ci_upper:+.4f}      | report only    |",
    f"| CI width                          | {baseline.ci_width:.4f}          | {current.ci_width:.4f}         | {current.ci_width - baseline.ci_width:+.4f}      | report only    |",
    f"| bias_percent                      | {baseline.bias_percent:+.2f}%        | {current.bias_percent:+.2f}%       | {current.bias_percent - baseline.bias_percent:+.2f}pp   | SC-2 (< 1%)    |",
    f"| KS statistic (log P curves)       | -               | {ks_stat:.4f}         | p = {ks_pvalue:.3g} | report only    |",
    f"| N events                          | {baseline.n_events}             | {current.n_events}            | {current.n_events - baseline.n_events:+d}            | info           |",
    "",
    "## Abort-Gate Computation",
    "",
    f"- |ΔMAP| = |{current.map_h:.4f} - {baseline.map_h:.4f}| = {delta_map:.4f}",
    f"- |ΔMAP| / 0.73 = {delta_map_relative:.4%}",
    "- Threshold (D-03 #1): 5.0000%",
    f"- **Verdict: {verdict}**",
    "",
    "## SC-2 (bias < 1% at h=0.73)",
    "",
    f"- v2.2 bias_percent = {current.bias_percent:+.2f}%",
    "- SC-2 threshold: |bias_percent| < 1.00%",
    f"- SC-2 status: {'PASS' if abs(current.bias_percent) < 1.0 else 'FAIL (reported, not abort)'}",
    "",
    "## Provenance",
    "",
    f"- v2.1 baseline: `simulations/_archive_v2_1_baseline/posteriors/` (see `ARCHIVE_MANIFEST.md` for git_commit + shas)",
    f"- v2.2 current:  `simulations/posteriors/` (re-evaluated in Task 1 of this plan)",
    "- Comparison helper: `master_thesis_code.bayesian_inference.evaluation_report.generate_comparison_report`",
    f"- KS test: `scipy.stats.ks_2samp` on {len(common_h)} aligned h-values",
    "",
    "## Related artifacts",
    "",
    f"- Raw re-eval log: `.planning/debug/verify02_reeval_{TS}.log`",
    f"- Quadrature warnings capture: `.planning/debug/verify02_quadrature_warnings_{TS}.log`",
    f"- JSON sidecar (standard generate_comparison_report output): `.planning/debug/comparison_verify02_{TS}.json`",
    "",
]

if abort:
    lines.extend([
        "## ABORT",
        "",
        "The abort gate fired. Wave 3 does NOT run. See the companion diagnostic:",
        f"  → `.planning/debug/abort_verify_gate_{TS}.md`",
        "",
    ])

report_md.write_text("\n".join(lines))

# --- Machine-readable sidecar for downstream plans ---
report_json.write_text(json.dumps({
    "ts": TS,
    "baseline_map_h": baseline.map_h,
    "current_map_h": current.map_h,
    "delta_map_abs": float(delta_map),
    "delta_map_relative": float(delta_map_relative),
    "abort_threshold": ABORT_THRESHOLD,
    "abort": abort,
    "bias_percent_v2_2": current.bias_percent,
    "ci_width_v2_1": baseline.ci_width,
    "ci_width_v2_2": current.ci_width,
    "ks_statistic": float(ks_stat),
    "ks_pvalue": float(ks_pvalue),
    "n_common_h": len(common_h),
    "baseline_git_commit": baseline.git_commit,
    "current_git_commit": current.git_commit,
}, indent=2))

print(f"VERIFY-02 verdict: {verdict}")
print(f"  MAP shift relative to true_h: {delta_map_relative:.4%}")
print(f"  Report: {report_md}")
print(f"  JSON:   {report_json}")

# --- Exit code drives downstream plan blocking ---
# D-08/D-10: ABORT → non-zero exit → Wave 3 does not start
sys.exit(1 if abort else 0)
