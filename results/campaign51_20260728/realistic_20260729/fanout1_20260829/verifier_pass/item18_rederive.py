#!/usr/bin/env python3
"""Item 18 (end-of-fan-out verifier pass) — governance incidents re-derivation.

Re-derives, from raw source (logs, filesystem, git plumbing), the three claims
made about item 18:

(a) runner-1 -> runner-2 -> runner-3 crash chain diagnoses
(b) SSH outage genuinely interrupted C4 provenance-extras retrieval (files
    really absent locally, not silently dropped data that was retrieved)
(c) commit-hygiene: hier_s0_registered_run/ + hier_s0_work/ simulations/
    intermediates were actually excluded from every commit that touched
    these trees (COMMIT_PLAN_3.md §4-5 claim)

No git state is mutated. Read-only against the working tree and git plumbing.
"""
import json
import re
import subprocess
from pathlib import Path

REPO = Path("/home/jasper/Repositories/darksiren-emri")
FAN = REPO / "results/campaign51_20260728/realistic_20260729/fanout1_20260829"
LOGDIR = FAN / "hier_s0_registered_run" / "logs"


def run(cmd, cwd=REPO):
    p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    return p.stdout.strip(), p.stderr.strip(), p.returncode


out = {}

# ---------------------------------------------------------------- (a) runner chain
r1 = (LOGDIR / "runner_wave2pre_20260829.log").read_text()
r2 = (LOGDIR / "runner2_wave2pre_20260829.log").read_text()
r3 = (LOGDIR / "runner3_wave2pre_20260829.log").read_text()

a = {}
# runner-1: pd.concat ValueError, empty b_plus list
m1 = re.search(r"raise ValueError\(\"No objects to concatenate\"\)\nValueError: No objects to concatenate", r1)
m1_ctx = re.search(r"all_nodes\[\"b_plus\"\]", r1)
a["runner1_pd_concat_valueerror_found"] = bool(m1)
a["runner1_root_is_empty_b_plus_list"] = bool(m1_ctx)
a["runner1_rc1_lines"] = re.findall(r"END rc=1", r1)

# runner-2: daemonic AssertionError, nested Pool inside evaluate() at bayesian_statistics.py:4562
m2 = re.search(r"AssertionError: daemonic processes are not allowed to have children", r2)
m2_site = re.search(r'bayesian_statistics\.py", line (\d+), in evaluate', r2)
a["runner2_daemonic_assertionerror_found"] = bool(m2)
a["runner2_nested_pool_site_evaluate_lineno"] = m2_site.group(1) if m2_site else None
a["runner2_num_worker_errors"] = len(re.findall(r"WORKER ERROR: AssertionError", r2))
a["runner2_rc1_lines"] = re.findall(r"END rc=1", r2)

# runner-3: run of record, jobs=1 despite cosmetic "jobs2" label, rc=0
m3_label = re.search(r"START P0 S0-A 4 seeds 5 nodes sites2\.2 nosmear (jobs\d)", r3)
m3_json_jobs = re.findall(r'"jobs":\s*(\d+)', r3)
m3_rc0 = re.findall(r"END rc=0", r3)
a["runner3_label_string"] = m3_label.group(1) if m3_label else None
a["runner3_json_jobs_values_seen"] = sorted(set(m3_json_jobs))
a["runner3_rc0_count"] = len(m3_rc0)

# cross-check against s0a_full_output.json directly (independent of the log)
s0a_json = json.loads((FAN / "hier_s0_registered_run" / "s0a_full_output.json").read_text())
a["s0a_full_output_json_jobs_field"] = s0a_json.get("jobs")

out["a_runner_chain"] = a

# ---------------------------------------------------------------- (b) SSH outage / C4 retrieval
c4 = FAN.parent / "wave2_20260829" / "c4"
b = {}
diag_csv = c4 / "simulations" / "diagnostics" / "event_likelihoods.csv"
b["diag_csv_exists"] = diag_csv.exists()
if diag_csv.exists():
    b["diag_csv_lines"] = sum(1 for _ in open(diag_csv))
    b["diag_csv_bytes"] = diag_csv.stat().st_size

post_dir = c4 / "simulations" / "posteriors"
b["posteriors_present"] = sorted(p.name for p in post_dir.glob("*.json")) if post_dir.exists() else []

postbh_dir = c4 / "simulations" / "posteriors_with_bh_mass"
b["posteriors_with_bh_mass_present"] = sorted(p.name for p in postbh_dir.glob("*.json")) if postbh_dir.exists() else []
b["posteriors_with_bh_mass_67_73_missing"] = (
    "h_0_67.json" not in b["posteriors_with_bh_mass_present"]
    and "h_0_73.json" not in b["posteriors_with_bh_mass_present"]
)

b["run_metadata_files_present"] = sorted(p.name for p in c4.glob("run_metadata_*.json"))
b["logs_dir_present"] = (c4 / "logs").exists()
b["git_commit_at_run_file_present"] = (c4 / "GIT_COMMIT_AT_RUN.txt").exists()

out["b_ssh_outage_c4"] = b

# ---------------------------------------------------------------- (c) commit hygiene
c = {}

# current on-disk footprint of simulations/ subtrees under the two run-artifact dirs
sim_dirs = []
total_sim_bytes = 0
for base in ("hier_s0_registered_run", "hier_s0_work"):
    for simdir in (FAN / base).rglob("simulations"):
        if simdir.is_dir():
            sz_out, _, _ = run(f"du -sb {simdir}")
            sz = int(sz_out.split()[0]) if sz_out else 0
            sim_dirs.append((str(simdir.relative_to(FAN)), sz))
            total_sim_bytes += sz
c["current_total_simulations_subtree_bytes"] = total_sim_bytes
c["current_total_simulations_subtree_MB"] = round(total_sim_bytes / 1e6, 1)
c["n_simulations_dirs_found"] = len(sim_dirs)
c["commit_plan_3_claimed_MB"] = 93.5

# max individual event_likelihoods.csv size (COMMIT_PLAN_3.md claims all < 4MB)
csvs = list((FAN / "hier_s0_registered_run").rglob("event_likelihoods.csv")) + list(
    (FAN / "hier_s0_work").rglob("event_likelihoods.csv")
)
sizes = [(str(p.relative_to(FAN)), p.stat().st_size) for p in csvs]
c["event_likelihoods_csv_count"] = len(sizes)
c["event_likelihoods_csv_max_bytes"] = max((s for _, s in sizes), default=0)
c["event_likelihoods_csv_all_under_4MB"] = all(s < 4 * 1024 * 1024 for _, s in sizes)

# what is actually tracked by git under the two dirs, right now
tracked_reg, _, _ = run(f"git ls-files results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run")
tracked_work, _, _ = run(f"git ls-files results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_work")
c["tracked_files_registered_run"] = [l for l in tracked_reg.splitlines() if l]
c["tracked_files_work"] = [l for l in tracked_work.splitlines() if l]
c["any_tracked_simulations_path"] = any(
    "/simulations/" in l for l in c["tracked_files_registered_run"] + c["tracked_files_work"]
)

# gitignore mechanism check: does a blanket `git add hier_s0_registered_run/` actually
# sweep in simulations/ subtrees, or does .gitignore already block it?
sample_sim = next((FAN / "hier_s0_registered_run").rglob("simulations"), None)
if sample_sim is not None:
    probe = sample_sim / "diagnostics" / "event_likelihoods.csv"
    if not probe.exists():
        probe = sample_sim
    ig_out, _, ig_rc = run(f"git check-ignore -v {probe}")
    c["gitignore_blocks_simulations_subtree"] = ig_rc == 0
    c["gitignore_rule_matched"] = ig_out

# does *.log block the raw crash-diagnosis logs from being tracked?
log_probe = LOGDIR / "runner3_wave2pre_20260829.log"
ig2_out, _, ig2_rc = run(f"git check-ignore -v {log_probe}")
c["gitignore_blocks_raw_wave2pre_logs"] = ig2_rc == 0
c["gitignore_rule_matched_for_logs"] = ig2_out
c["raw_wave2pre_logs_tracked"] = log_probe.name in "\n".join(c["tracked_files_registered_run"])

# scan ALL commits that ever touched these two paths, and the largest blob size
# ever committed anywhere under them (would reveal any historical sweep-in)
hist, _, _ = run(
    "git log --oneline -- "
    "results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run "
    "results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_work"
)
c["commits_touching_these_paths"] = hist.splitlines()

blobscan, _, _ = run(
    "git rev-list --objects --all -- "
    "results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run "
    "results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_work "
    "| git cat-file --batch-check='%(objecttype) %(objectname) %(rest) %(objectsize)'"
)
blob_sizes = []
for line in blobscan.splitlines():
    parts = line.split()
    if parts and parts[0] == "blob":
        try:
            blob_sizes.append(int(parts[-1]))
        except ValueError:
            pass
c["largest_blob_ever_committed_bytes"] = max(blob_sizes, default=0)
c["largest_blob_ever_committed_MB"] = round(max(blob_sizes, default=0) / 1e6, 3)
c["n_blobs_ever_committed_under_these_paths"] = len(blob_sizes)

out["c_commit_hygiene"] = c

print(json.dumps(out, indent=2, default=str))
