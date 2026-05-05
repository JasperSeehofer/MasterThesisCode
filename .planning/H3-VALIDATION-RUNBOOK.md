# H3 fix — cluster validation runbook (post-commit `f01595c`)

Commit `f01595c` ([PHYSICS] H3 fix) is on origin/main. Cluster needs
interactive SSH (publickey-only auth refused our keys this session).
Once you SSH in, run the following.

---

## R1 — primary gate: h=0.73 phase46-merged 1473 events

**Cost:** ~15 min on cpu_il, 1 sbatch (or up to 7 parallel tasks).
**Gate G_H3b PASS:** 2D z ≤ 2σ AND 2D bias ≤ 1D bias.

```bash
ssh bwunicluster
cd $HOME/MasterThesisCode
git pull origin main   # picks up f01595c

# Re-use the prepared CRB from yesterday's post-bridge closure run.
# (The CRB itself is unchanged; only the p_det grid/integrand changed.)
PREVIOUS_RUN="$WORKSPACE/run_closure_h0p73_postfix_20260505"
RUN_DIR="$WORKSPACE/run_closure_h0p73_h3_$(date +%Y%m%d)"

mkdir -p "$RUN_DIR/simulations"
cp "$PREVIOUS_RUN/simulations/prepared_cramer_rao_bounds.csv" "$RUN_DIR/simulations/"
cp "$PREVIOUS_RUN/simulations/cramer_rao_bounds.csv" "$RUN_DIR/simulations/"
ln -sfn "$PREVIOUS_RUN/simulations/injections" "$RUN_DIR/simulations/injections"

sbatch \
    --array=0-6 \
    --output="$RUN_DIR/logs/eval_%A_%a.out" \
    --error="$RUN_DIR/logs/eval_%A_%a.err" \
    --export=ALL,RUN_DIR="$RUN_DIR",H_TRUE=0.73 \
    cluster/evaluate_closure_h_true_finegrid.sbatch

# Monitor: sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode
```

When the array completes (~15 min on cpu_il, less on dev_cpu_il):

```bash
# rsync results back to dev machine
rsync -avz "bwunicluster:$RUN_DIR/simulations/posteriors/" \
    simulations/cluster_run_closure_h0p73_h3_$(date +%Y%m%d)/posteriors/
rsync -avz "bwunicluster:$RUN_DIR/simulations/posteriors_with_bh_mass/" \
    simulations/cluster_run_closure_h0p73_h3_$(date +%Y%m%d)/posteriors_with_bh_mass/
```

Then on dev machine, run the analyzer:

```bash
uv run python scripts/bias_investigation/test_24_multi_truth_bias_sweep.py \
    --crb simulations/cluster_run_phase46_merged_20260504/cramer_rao_bounds.csv \
    --posteriors-dir simulations/cluster_run_closure_h0p73_h3_$(date +%Y%m%d)/posteriors_with_bh_mass \
    --output scripts/bias_investigation/outputs/phase46_merged/h3_postfix_verdict.json
```

---

## R2 — Phase 45 412-event regression

**Goal:** verify the previously-PASSING Phase 45 412-event closure
(`z=+1.97` post-Tier-3 pre-bridge) is still passing under the combined
bridge+H3 fixes. Also clears the bridge fix's pending Phase 45
re-validation.

```bash
PREVIOUS_RUN="$WORKSPACE/run_phase45_20260501"   # confirm path on cluster
RUN_DIR="$WORKSPACE/run_phase45_h3_$(date +%Y%m%d)"

mkdir -p "$RUN_DIR/simulations"
cp "$PREVIOUS_RUN/simulations/prepared_cramer_rao_bounds.csv" "$RUN_DIR/simulations/"
cp "$PREVIOUS_RUN/simulations/cramer_rao_bounds.csv" "$RUN_DIR/simulations/"
ln -sfn "$PREVIOUS_RUN/simulations/injections" "$RUN_DIR/simulations/injections"

sbatch \
    --array=0-3 \
    --partition=dev_cpu_il \
    --output="$RUN_DIR/logs/eval_%A_%a.out" \
    --error="$RUN_DIR/logs/eval_%A_%a.err" \
    --export=ALL,RUN_DIR="$RUN_DIR",H_TRUE=0.73 \
    cluster/evaluate_closure_h_true_finegrid.sbatch
```

R2 gate: 2D z ≤ 2σ at h=0.73 on Phase 45 412 events.

---

## Post-results

After both R1 and R2 land:

1. Run the analyzer (test_24_multi_truth_bias_sweep.py) on each
   posteriors_with_bh_mass dir.
2. Post the headline numbers (1D MAP, 2D MAP, σ_boot, bias, z) for both
   runs back into this Claude Code session.
3. The session will then promote §4.7 → §3.15 in
   `docs/H0_BIAS_RESOLUTION.md` with the post-fix numbers and update §1
   Executive Summary, §4.0 Continuation guide, and DATA_INVENTORY.md.

---

## Decision tree

| R1 outcome | R2 outcome | Action |
|---|---|---|
| 2D z ≤ 2σ AND 2D bias ≤ 1D bias | 2D z ≤ 2σ | **PASS G_H3b** — promote §4.7 to §3.15 with shipping numbers; queue follow-up multi-truth panel (h=0.60/0.65/0.70) as separate phase. |
| Partial close (2D z 2–4σ) | any | Investigate residual: check if 1D bias is still ~0 (i.e., the fix is necessary but not sufficient) or whether something destabilized. |
| 2D z > 4σ or wrong direction | any | Pivot: handoff's H3b (L_cat/L_comp entropy) or H1 (realization-bootstrap). The diagnostic test_27 predicted Δp_det in the right physical direction; if cluster shows otherwise, the integrand/posterior coupling is more complex than modeled. |
| any | R2 FAIL | Combined fixes regressed Phase 45's prior PASS. Investigate; do not declare paper-readiness. |

Plan: `~/.claude/plans/please-look-at-the-velvety-quail.md`.
Memory: `project_pdet_hypothesis_convention.md` (the underlying physics
principle is durable across projects).
