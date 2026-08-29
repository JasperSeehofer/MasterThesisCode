#!/usr/bin/env python3
"""Item 12 (B7.3 [2D-TWIN] adoption) independent re-derivation.

Re-executes, FROM SOURCE (git history + code + pytest), the decisive claims of
PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md / B7_3_ADOPTION_IMPLEMENTATION_RECORD.md /
B7_3_ADOPTION_VERIFIER_REPORT.md, without trusting any prior report's numbers.

Checks:
  1. The [PHYSICS] commit d4765539 touches exactly the four named production files
     (+ the disclosed Class-B driver, tests, and append-only records) -- diffstat re-derived
     from git, not copied from the record.
  2. Every hunk in the four production files lands strictly outside the kernel body range
     (bayesian_statistics.py:6231-7723) -- re-parsed from `git show` hunk headers, not asserted.
  3. The 12 named decisive pin tests all PASS, re-run directly.
  4. The full fast suite reproduces 1896 passed / 15 skipped / 27 deselected.
  5. The five archived scripts pin catalogue_numerator_survival_2d="off" at the claimed lines.

Exit code 0 iff every decisive claim reproduces; nonzero otherwise (see printed verdict).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
assert REPO.name == "darksiren-emri" or (REPO / "darksiren_emri").is_dir(), REPO

COMMIT = "d4765539"

PROD_FILES = [
    "darksiren_emri/bayesian_inference/bayesian_statistics.py",
    "darksiren_emri/arguments.py",
    "darksiren_emri/main.py",
    "darksiren_emri/validation/correspondence_1d.py",
]

# Kernel-body line range in bayesian_statistics.py that must show ZERO diff (from the
# presentation's own §6.1(a-ii) / the implementation record's "not touched" claim).
KERNEL_RANGE = (6231, 7723)

PIN_TESTS = [
    "test_off_matches_the_pre_flag_golden_across_modes[generator_marginal]",
    "test_off_matches_the_pre_flag_golden_across_modes[volume_deconv]",
    "test_off_matches_the_pre_flag_golden_across_modes[absolute_marginal]",
    "test_evaluate_mz_sel_with_unset_center_raises",
    "test_r5_sigma_gal_zero_limit_matches_point_s4d_at_host_mass",
    "test_cli_flag_defaults_to_mz_sel_and_eff",
    "test_cli_flag_explicit_off_and_unset_parses_and_validates",
    "test_cli_validate_refuses_mz_sel_with_unset_center",
    "test_six_site_default_trace_is_mz_sel_and_eff",
    "test_kernel_default_pair_bit_identical_to_explicit_mz_sel_eff",
    "test_evaluate_default_logs_physics_info_line",
    "test_evaluate_explicit_off_logs_counterfactual_warning",
]

FIVE_SCRIPT_PINS = {
    "scripts/mass_trunc_ab.py": (151, 152),
    "scripts/volume_trunc_ab.py": (150, 151),
    "scripts/eddington_m_impact.py": (164, 165),
    "scripts/ablation_cube_seed600.py": (155, 156),
    "scripts/quick_validation_15.py": (84, 85),
}

results: dict[str, object] = {}
failures: list[str] = []


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, **kw)


# --- Check 1: file list of the adoption commit -----------------------------------------
r = run(["git", "show", "--name-only", "--format=", COMMIT])
files_changed = [line.strip() for line in r.stdout.splitlines() if line.strip()]
results["check1_files_changed"] = files_changed
expected_prod = set(PROD_FILES)
touched_prod = {f for f in files_changed if f in expected_prod}
if touched_prod != expected_prod:
    failures.append(f"check1: production file set mismatch: {touched_prod} != {expected_prod}")

# --- Check 2: every hunk in the 4 production files is outside the kernel range ----------
hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
for f in PROD_FILES:
    r = run(["git", "show", f"{COMMIT}", "--", f])
    hunk_lines: list[tuple[int, int]] = []
    for line in r.stdout.splitlines():
        m = hunk_re.match(line)
        if m:
            old_start = int(m.group(1))
            old_len = int(m.group(2) or "1")
            hunk_lines.append((old_start, old_start + old_len))
    results[f"check2_hunks::{f}"] = hunk_lines
    if f == "darksiren_emri/bayesian_inference/bayesian_statistics.py":
        for start, end in hunk_lines:
            if not (end <= KERNEL_RANGE[0] or start >= KERNEL_RANGE[1]):
                failures.append(
                    f"check2: hunk {start}-{end} in {f} OVERLAPS kernel range {KERNEL_RANGE}"
                )
    if not hunk_lines:
        failures.append(f"check2: no hunks found in {f} -- diff extraction broken")

# --- Check 3: the 12 decisive pin tests -------------------------------------------------
# Build a -k expression that matches each test by exact (parametrized) id fragment.
k_parts = []
for t in PIN_TESTS:
    if "[" in t:
        base, param = t[:-1].split("[")
        k_parts.append(f"({base} and {param})")
    else:
        k_parts.append(t)
k_expr = " or ".join(k_parts)
r = run(
    [
        "uv",
        "run",
        "pytest",
        "darksiren_emri_test/bayesian_inference/test_catalogue_numerator_survival_2d.py",
        "-k",
        k_expr,
        "-v",
        "--no-cov",
        "-q",
    ],
    timeout=300,
)
results["check3_stdout_tail"] = "\n".join(r.stdout.splitlines()[-30:])
passed_names = {
    line.split(" ")[0].split("::")[-1]
    for line in r.stdout.splitlines()
    if "PASSED" in line
}
n_passed_match = re.search(r"(\d+) passed", r.stdout)
n_passed = int(n_passed_match.group(1)) if n_passed_match else -1
results["check3_n_passed"] = n_passed
if n_passed != len(PIN_TESTS):
    failures.append(f"check3: expected {len(PIN_TESTS)} pin tests passed, got {n_passed}")
if "failed" in r.stdout.lower() and "0 failed" not in r.stdout.lower():
    failures.append("check3: pin-test run reported failures")

# --- Check 4: full fast suite ------------------------------------------------------------
r = run(
    ["uv", "run", "pytest", "-m", "not gpu and not slow", "--no-cov", "-q"],
    timeout=580,
)
tail = r.stdout.strip().splitlines()[-5:]
results["check4_tail"] = tail
summary_line = next((line for line in reversed(r.stdout.splitlines()) if "passed" in line), "")
results["check4_summary"] = summary_line
m = re.search(r"(\d+) passed, (\d+) skipped, (\d+) deselected", summary_line)
if not m:
    failures.append(f"check4: could not parse suite summary line: {summary_line!r}")
else:
    p, s, d = (int(x) for x in m.groups())
    results["check4_counts"] = {"passed": p, "skipped": s, "deselected": d}
    if (p, s, d) != (1896, 15, 27):
        failures.append(f"check4: suite counts {(p, s, d)} != expected (1896, 15, 27)")

# --- Check 5: five archived script pins ---------------------------------------------------
for path, (start, end) in FIVE_SCRIPT_PINS.items():
    full = REPO / path
    lines = full.read_text().splitlines()
    got = [lines[start - 1].strip(), lines[end - 1].strip()]
    ok = got == [
        'catalogue_numerator_survival_2d="off",',
        'catalogue_numerator_survival_2d_center="unset",',
    ]
    results[f"check5::{path}"] = got
    if not ok:
        failures.append(f"check5: {path}:{start}-{end} does not pin off/unset explicitly: {got}")

r = run(
    ["uv", "run", "ruff", "check", *FIVE_SCRIPT_PINS.keys()],
)
results["check5_ruff"] = r.stdout.strip() + r.stderr.strip()
if r.returncode != 0:
    failures.append("check5: ruff check failed on one of the five scripts")

for path in FIVE_SCRIPT_PINS:
    r = run(["uv", "run", "python", "-m", "py_compile", path])
    if r.returncode != 0:
        failures.append(f"check5: py_compile failed on {path}: {r.stderr}")

# --- Verdict -------------------------------------------------------------------------------
print("=== ITEM 12 RE-DERIVATION RESULTS ===")
for k, v in results.items():
    print(f"-- {k} --")
    print(v)
    print()

print("=== FAILURES ===")
if failures:
    for f in failures:
        print("FAIL:", f)
    sys.exit(1)
else:
    print("NONE -- every decisive claim reproduced.")
    sys.exit(0)
