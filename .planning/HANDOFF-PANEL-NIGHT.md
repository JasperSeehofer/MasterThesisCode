# Handoff — Multi-Truth Panel Night Run (2026-05-04)

**Audience:** a fresh Claude Code session picking up after the daytime work in this branch. **Author:** the daytime session (~13:00–17:30 local), commits `b110ba7` and `a8626de` on `main`.

**Goal of the night run:** fire the full 7-truth bias-vs-h_true panel on `cpu_il` post-20:00, produce the panel verdict, interpret it, file results.

---

## TL;DR — turnkey night protocol

```bash
# 0. Re-auth SSH (ControlMaster socket may have died; expect to retype OTP).
ssh bwunicluster exit                  # creates the persistent socket

# 1. Check the GPU extension status and decide whether to merge.
ssh bwunicluster '
  WS=/pfs/work9/workspace/scratch/st_ac147838-emri/run_20260504_seed300_extension
  echo "Per-task CSVs:"
  ls $WS/simulations/cramer_rao_bounds_*.csv 2>/dev/null | wc -l
  echo "Total CRB rows so far:"
  total=0; for f in $WS/simulations/cramer_rao_bounds_*.csv; do
    [[ -f "$f" ]] || continue
    n=$(($(wc -l < "$f") - 1)); total=$((total + n))
  done; echo $total
  echo "Job state:"
  sacct -j 4216323 -X --format=State -n | sort | uniq -c | sort -rn
'

# 2. Decide: merge or skip?  (See "Merge decision" below.)
#    If merging: run cluster/merge.sbatch.  If skipping: proceed straight to panel.

# 3. Launch the full panel on cpu_il (defaults, no env overrides).
nohup bash scripts/bias_investigation/run_multi_truth_sweep.sh \
    > .claude/runlogs/full_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > .claude/runlogs/full.pid

# 4. Watch.  Arm a Monitor with the WIDER filter alternation (W-TOOL-06 lesson):
#    'ssh_askpass|Permission denied|QOSMax|sbatch.*reject|^\[[0-9]+\] h_truth=|^\[[1-5]/5\]|Job ID:|Job [0-9]+ done|Posteriors rsynced|All truths complete|Sweep complete|ERROR|Traceback|FAILED|invalid number|^  \[(1D|2D)\]|verdict_'

# 5. Verdict ships at: scripts/bias_investigation/outputs/phase45/multi_truth_sweep.{json,png}
```

---

## State at handoff

### What landed today (commits)

| SHA | Subject | What it does |
|-----|---------|-------------|
| `b110ba7` | verification: widen multi-truth grid + parameterize sbatch + per-event diagnostic | Smoke-validated (c) edits |
| `a8626de` | data: register Phase 45 + seed=300 extension; archive 2-truth smoke output | DATA_INVENTORY.md current; smoke JSON + plot archived |
| (vault) `db8cf7a` | wiki-debrief master-thesis-code | 9 tentative wiki filings |

### Smoke verdict (canonical reference)

The 2-truth smoke (`TRUTHS="0.65 0.73"`, 2026-05-04) on dev_cpu_il PASSED **all four panel verdicts** — weighted-mean-z ≤ 2, sign concordance random, no boundary-rail, per-event pos_frac dispersion not suspicious.

Most striking finding: the bias **sign flipped** between truths. h=0.65 → +1.6σ to +1.9σ; h=0.73 → −1.1σ (1D) to −0.3σ (2D). Strong directional evidence consistent with H0 (residual is statistical, not a structural always-positive pull).

Full smoke JSON: `scripts/bias_investigation/outputs/phase45/multi_truth_sweep.json`.

### What's running on the cluster

**Job `4216323`** (gpu_h100, 50 tasks, BASE_SEED=300) — extension toward ~1000 SNR≥20 events.

- Submitted 2026-05-04 ~14:30 local. First successful submit (the prior `4216105` failed all 16 dispatched tasks due to a `PROJECT_ROOT` env-var-quoting bug; cancelled and replaced — see "Known traps" below).
- Concurrency observed: **5 simultaneous tasks** on gpu_h100 (better than the 2-3 estimated from seed=200 history).
- Per-task walltime cap: 1h. Per-task yield: ~45 events at SNR≥15, ~9.4% pass SNR≥20.
- ETA: full 50 tasks complete around **00:30 tonight**. By 20:00 expect ~25 tasks done → roughly **100–140 new SNR≥20 events**.
- Per-task CSVs flush only at task completion. As of 17:30 local: 0 written (first task should finish ~15:35; check status when you start).

### Cluster repo state

- `~/MasterThesisCode` is on `main`, HEAD `020341d` (synced from origin earlier today). The (c) edits land via `git pull` if you re-sync — they're already on origin/main.
- `~/.ssh/config` has `Host bwunicluster` with `ControlMaster auto`, `ControlPath ~/.ssh/cm-%r@%h:%p`, `ControlPersist 10h`. Socket is at `~/.ssh/cm-st_ac147838@uc3.scc.kit.edu:22`. Apparent flakiness — may need a fresh OTP at session start despite ControlPersist.

---

## Merge decision

If the seed=300 extension has produced enough events by the time you start, merging into the Phase 45 CRB before the panel run gives tighter σ_boot and better D(h). If not, run the panel on the existing 424-event CRB and re-run later when the extension lands.

**Rule of thumb:**

| New SNR≥20 events available | Action |
|----------------------------|--------|
| ≥ 50 (≥ 12% bump) | Merge before panel. Total ≈ 475+ events; meaningful tightening. |
| 20–50 | Borderline. Probably skip the merge — the marginal σ_boot improvement is < 5% and merge-adds-complexity. |
| < 20 | Skip merge. Run panel on 424. Re-run later. |

**How to merge** (if going that path):

```bash
# On cluster — assumes job 4216323 has produced per-task CSVs.
ssh bwunicluster '
  cd ~/MasterThesisCode
  WS=/pfs/work9/workspace/scratch/st_ac147838-emri/run_20260504_seed300_extension
  sbatch --export=ALL,RUN_DIR=$WS,PROJECT_ROOT=$HOME/MasterThesisCode \
    --output=$WS/logs/merge_%j.out --error=$WS/logs/merge_%j.err \
    cluster/merge.sbatch
  squeue -u $USER -h | tail -3
'
# Wait for merge to finish (~few min on cpu_il), then concatenate the
# merged CRB into the Phase 45 path.  Important: the rescaler
# (test_23_rescale_crb_to_h_true.py) reads from
# simulations/cluster_run_phase45_20260501/cramer_rao_bounds.csv — that
# is what the orchestrator uses.  To make the panel see the new events,
# either:
#   (a) APPEND the merged extension CSV to the Phase 45 CSV in place
#       (preserve original as .bak), or
#   (b) Point the rescaler at a new merged path via --input-crb.
# Option (a) is simplest if you trust the merge.  Option (b) is safer
# but requires editing the orchestrator.

# Option (a) one-liner (after rsync of merged CRB locally):
# uv run python -c "
# import pandas as pd
# old = pd.read_csv('simulations/cluster_run_phase45_20260501/cramer_rao_bounds.csv')
# new = pd.read_csv('<rsync-target>/cramer_rao_bounds.csv')
# import shutil; shutil.copy(
#     'simulations/cluster_run_phase45_20260501/cramer_rao_bounds.csv',
#     'simulations/cluster_run_phase45_20260501/cramer_rao_bounds.csv.bak_pre_merge_20260504'
# )
# pd.concat([old, new], ignore_index=True).to_csv(
#     'simulations/cluster_run_phase45_20260501/cramer_rao_bounds.csv', index=False
# )
# print(f'merged: {len(old)} + {len(new)} = {len(old)+len(new)} rows')
# "
```

**Document the merge in DATA_INVENTORY.md** — add a row to the Evaluation Log and a new "phase46-merged" entry once the merge is final.

---

## Panel run mechanics

### Defaults the orchestrator picks up

```bash
TRUTHS="0.60 0.65 0.70 0.73 0.75 0.80 0.85"   # 7 truths
SBATCH_PARTITION=cpu_il                        # 128 cpus, post-20:00
SBATCH_CPUS=128
SBATCH_ARRAY=0-6                               # stride 7 across 21-pt grids
SEED_BASE=200                                  # per-truth seed = 200+idx
```

### Per-truth grid sizes (clamped to LamCDM prior [0.60, 0.86])

| h_true | Grid points | Range |
|--------|-------------|-------|
| 0.60 | 11 | 0.6000–0.6500 |
| 0.65 | 21 | 0.6000–0.7000 |
| 0.70 | 21 | 0.6500–0.7500 |
| 0.73 | 21 | 0.6800–0.7800 |
| 0.75 | 21 | 0.7000–0.8000 |
| 0.80 | 21 | 0.7500–0.8500 |
| 0.85 | 13 | 0.8000–0.8600 |

Total: 129 h-evaluations × 2 channels (1D + 2D) = 258 inferences.

### Wallclock estimate

cpu_il post-20:00 with 7 array tasks running in parallel per truth, ~1.5 min/h-value: ~5 min wallclock per truth + ~30s rsync + ~1 min prepare_detections + ~30s rescale = **~8 min per truth × 7 truths = ~55 min serialized + queue waits ≈ 1.5–2h total**.

### Watch with WIDER Monitor filter

The smoke session bit on this — `Monitor` armed with `ERROR|FAILED|Traceback|invalid number` missed `ssh_askpass:` SSH-auth failures. Use this alternation when re-arming for the night run:

```
ssh_askpass|Permission denied|QOSMax|sbatch.*reject|slurm_load_node|^\[[0-9]+\] h_truth=|^\[[1-5]/5\]|Job ID:|Job [0-9]+ done|Posteriors rsynced|All truths complete|Sweep complete|ERROR|Traceback|FAILED|invalid number|^  \[(1D|2D)\]|verdict_|=== Panel
```

This is filed as `EXP-18` in [[agentic-experiments]] for tracking.

---

## Verdict interpretation

Output: `scripts/bias_investigation/outputs/phase45/multi_truth_sweep.{json,png}`.

The analyzer prints 4 panel verdicts per channel:

1. **`verdict_mean`** — PASS (`|z_panel| ≤ 2`), MARGINAL (2–3σ), FAIL (>3σ). Weighted mean of per-truth biases vs zero. **Most important verdict.**
2. **`verdict_sign_concordance`** — PASS if signs scatter, FLAG if all same sign or binomial p < 0.10. The smoke saw a clean sign flip between 0.65 and 0.73; if the full panel keeps mixed signs we're in good shape.
3. **`verdict_boundary_rail`** — FLAG if any truth's discrete MAP is at a grid edge. Should PASS unless the panel extremes (0.60, 0.85) have surprisingly large biases.
4. **`verdict_shared_injection_pull`** — FLAG if per-event pos_frac is tightly clustered far from 0.5 across truths (mean far from 0.5 + std < 0.05). This is the diagnostic for the shared-injection-set caveat.

### Outcome scenarios

| Scenario | Numbers | Action |
|----------|---------|--------|
| **All four PASS** | weighted z ≤ 2, χ²_red ≈ 1, mixed signs | Pipeline closure-validated. Update `project_bias_status.md`, write up for paper. |
| **`verdict_mean` PASS, χ²_red > 3** | per-truth biases scatter wider than σ_boot | σ_boot underestimates uncertainty. **Shared-injection-set caveat is live** — see below. |
| **`verdict_mean` FAIL** | weighted z > 3 from zero | Structural residual. Investigate. Don't ship paper without resolution. |
| **`verdict_shared_injection_pull` FLAG** | pos_frac stable ≠ 0.5 across truths | Same injections pulling MAP same direction at every truth. Flag in paper as "underestimated systematic"; consider injection-set bootstrap as follow-up. |
| **`verdict_boundary_rail` FLAG** at extremes | MAP at 0.60 or 0.85 grid edge | Re-run that truth with a wider grid (e.g. ±0.08 instead of ±0.05). |

### The shared-injection caveat (MUST mention if verdict surprises)

`σ_boot` resamples *events* at fixed truth, but all 7 truths reuse the same Phase 45 injection campaign — only `prepare_detections` seed varies (which decorrelates observed-d_L noise but not underlying sky positions, masses, true redshifts). If the panel χ²_red > 3, the inflated scatter could be:
- **(a)** structural pipeline residual (bad), OR
- **(b)** shared-injection-set pull through correlated injection draws (less bad).

The current test cannot distinguish these. A proper injection-set bootstrap (resample CRB rows before rescaling, repeat the panel) is the rigorous follow-up, ~10× compute cost — out of scope for tonight, file as next phase.

### Seed-dependent MAP shift (paper-relevant)

`finding_seed_dependent_map.md` in memory: at h=0.73, the same Phase 45 injection set yields different MAPs under different `prepare_detections` seeds (0.7400 → 0.7233 with seed=202, ΔMAP ≈ 0.017 ≈ 2.7× σ_boot). **Don't quote a single MAP in the paper without acknowledging seed dependence.** A seed-bootstrap study (K=5–10 fresh seeds at each truth, single h-value evaluation per seed) is a useful follow-up but was deferred today.

---

## Known traps from today

These are wiki-filed (commit `db8cf7a` in vault) but worth surfacing here:

1. **Bash `printf "%.2f"` requires `LC_ALL=C` under non-`C` locales.** The orchestrator already exports it; don't strip.
2. **slurm `(Priority)` pending reason can hide reservation blocking.** `cpu_il` is reserved 08:00–20:00 weekdays by `juypter_weekday_cpuonly`. Check `scontrol show reservation | grep -B1 -A3 cpu_il` if a job sits in `(Priority)` longer than expected.
3. **OTP-protected SSH expires; ControlMaster is required for unattended polling.** Already configured but the socket flakes occasionally. Re-auth at session start to be safe.
4. **`SLURM_ARRAY_TASK_COUNT` for portable stride** — the sbatch already does this; same file works on cpu_il (`--array=0-6`) and dev_cpu_il (`--array=0-3`) without edits.
5. **Cluster repo HEAD drift** — verify `ssh bwunicluster 'cd ~/MasterThesisCode && git log -1 --oneline'` matches origin/main before any submit.
6. **`PROJECT_ROOT=\$HOME/...` through single-quoted ssh heredoc** preserves the literal `$HOME` and SLURM stores it unexpanded (today's bug, cost: 15 wasted GPU tasks). When using ssh + sbatch, use double-quoted heredoc OR pass `PROJECT_ROOT=$HOME/MasterThesisCode` (no escape) so the remote shell expands at sbatch-invocation time.
7. **Monitor filter must include `ssh_askpass`** — already covered above. Don't forget.

---

## After the panel finishes

1. **Read the JSON.** Print the per-truth + panel verdicts (the orchestrator does this; just re-grep the log).
2. **Update DATA_INVENTORY.md Evaluation Log** with the panel row.
3. **Update memory:**
   - `project_bias_status.md` — replace "Multi-truth verification underway" → final verdict.
   - If verdict surprises (FAIL or FLAG), write a new finding-* memory file.
4. **Commit results** (`git add scripts/bias_investigation/outputs/phase45/multi_truth_sweep.json && git commit -m "verification: full 7-truth panel landed"`).
5. **Run `/wiki-debrief`** at session end. Reusable lessons from this session likely include: panel-night protocol observations, any new failure modes, the eventual χ²_red interpretation.

---

## Open questions for the user (not blocking; surface if relevant)

- After tonight's verdict, do we want to commission a proper injection-set bootstrap (next phase)? Required to resolve the shared-injection caveat if χ²_red > 3.
- Continue the GPU extension toward ~1000 events (next 100-task batch with BASE_SEED=350)? Useful for the seed-bootstrap study and tighter σ_boot.
- Begin paper writing on the assumption tonight passes? Or hold for the injection-bootstrap result?

---

## Files this handoff references

- `scripts/bias_investigation/run_multi_truth_sweep.sh` — orchestrator (parameterized)
- `scripts/bias_investigation/test_23_rescale_crb_to_h_true.py` — local rescaler
- `scripts/bias_investigation/test_24_multi_truth_bias_sweep.py` — analyzer (panel verdicts + per-event diagnostic)
- `cluster/evaluate_closure_h_true_finegrid.sbatch` — single-truth fine-grid evaluation
- `cluster/simulate.sbatch` — GPU EMRI simulation (the seed=300 job uses this)
- `cluster/merge.sbatch` — combines per-task CSVs into a single CRB
- `simulations/cluster_run_phase45_20260501/cramer_rao_bounds.csv` — current canonical CRB (424 SNR≥20)
- `DATA_INVENTORY.md` — dataset registry, Evaluation Log
- `~/.claude/projects/-home-jasper-Repositories-MasterThesisCode/memory/` — auto-memory (read on session start)
