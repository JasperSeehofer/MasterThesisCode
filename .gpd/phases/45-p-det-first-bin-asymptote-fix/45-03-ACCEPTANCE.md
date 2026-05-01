# Phase 45-03 Acceptance Gate — Cluster Re-Evaluation

**Purpose:** This document is the deterministic flowchart for the Phase 45 cluster re-evaluation. The acceptance criteria are locked from `45-RESEARCH.md §5` and `HANDOFF-phase45-diagnosis.md`; the escalation tree is binding.

**Locked anchor:** `_P_MAX_EMPIRICAL_ANCHOR = 0.7931` (pooled Wilson 95% lower bound; LR homogeneity p=0.199; Plan 45-01).
**Load-bearing commit:** `09ee262 [PHYSICS] Phase 45: anchor p_det interpolator at empirical asymptote (d_L=0)`.

---

## 1. Pre-flight checklist (must pass before submitting)

All gates must be green before invoking `cluster/submit_phase45_eval.sh`.

### Local
- [ ] `git log -1 --format="%H %s" 09ee262` returns `09ee262995b2ca95a997bba7d19f645d7ecff1de [PHYSICS] Phase 45: anchor p_det interpolator at empirical asymptote (d_L=0)` — anchor patch committed.
- [ ] `uv run pytest -m "not gpu and not slow" --no-cov` — green (564 passed, 6 skipped, 16 deselected per Plan 45-02 SUMMARY).
- [ ] `cluster/submit_phase45_eval.sh` is committed and executable (Plan 45-03 Task 1).

### Cluster login node
- [ ] `ssh bwunicluster && cd <repo_path> && git pull` runs cleanly.
- [ ] `git rev-parse HEAD` matches `09ee262995b2ca95a997bba7d19f645d7ecff1de` (or any descendant that includes the anchor patch).
- [ ] `uv run pytest -m "not gpu and not slow" --no-cov master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py` — green. Specifically:
  - `TestPhase45EmpiricalAnchor` — 7 tests pass.
  - `TestZeroFillBoundaryConvention` — 4 tests pass (Phase 44 invariants preserved).
- [ ] `echo $WORKSPACE` — non-empty (bwHPC workspace path).
- [ ] `[[ -f /pfs/work9/workspace/scratch/st_ac147838-emri/run_20260401_seed200/simulations/cramer_rao_bounds.csv ]]` — source CRB present (542 ecliptic-migrated events from Phase 43).
- [ ] `[[ -f /pfs/work9/workspace/scratch/st_ac147838-emri/run_20260401_seed200/simulations/prepared_cramer_rao_bounds.csv ]]` — prepared CSV present.

If **any** gate fails: stop. Reconcile environment first; do not submit a stale checkout (forbidden proxy in Plan 45-03 contract).

---

## 2. Submission

```bash
ssh bwunicluster
cd <repo_path>
bash cluster/submit_phase45_eval.sh
```

Capture both job IDs from stdout:

```
EVAL_JOB=<id>     # 38-element array, 0–37 (h ∈ {0.60, …, 0.86} step 0.005, with two extras for grid)
COMBINE_JOB=<id>  # afterok dependency on EVAL_JOB
```

Post these IDs back to the agent so monitoring can resume.

---

## 3. Monitoring

```bash
sacct -j <EVAL_JOB>,<COMBINE_JOB> --format=JobID,State,Elapsed,ExitCode
```

**Expected wallclock** (matches Phase 44 baseline):
- `evaluate.sbatch`: ≤ 4 h per array task on `gpu_h100`.
- `combine.sbatch`: ≤ 30 min after the array completes.
- **Total:** ≤ 6 h end-to-end.

If walltime exceeds 8 h: investigate cluster contention or accidental wider grid; do not silently retry.

---

## 4. Rsync

```bash
rsync -avz bwunicluster:$NEW_RUN/simulations/posteriors/ ./results/phase45_posteriors/
rsync -avz bwunicluster:$NEW_RUN/logs/ ./results/phase45_posteriors/logs/
```

Required artifacts after rsync:
- `results/phase45_posteriors/combined_posterior.json` — contains `map_h`, `interval_68`, `h_values`, `posterior`, `n_events_used`, `D_h_per_h`, `git_commit`.
- `results/phase45_posteriors/logs/combine_*.out` — strategy-comparison stdout (4 zero-handling strategies).
- Per-h posterior files (`posterior_h<value>.json` × 38).

**Sanity gate:** `combined_posterior.json["git_commit"]` must equal `09ee262995b2ca95a997bba7d19f645d7ecff1de`. If not, the cluster ran a stale checkout — abort, investigate, do not interpret MAP.

---

## 5. Acceptance criteria (locked from RESEARCH.md §5)

All four criteria must PASS for Phase 45 to be marked complete.

### 5.1 Primary — MAP in range
**Test ID:** `test-map-in-range`
**Procedure:** Read `results/phase45_posteriors/combined_posterior.json`, inspect `map_h`.
**Pass condition:** `0.72 ≤ map_h ≤ 0.74`.

### 5.2 Coverage — 68% interval contains 0.73
**Test ID:** `test-bootstrap-coverage`
**Procedure:** Run the bootstrap MAP estimator on the post-fix posterior. Either:
- (a) Edit `scripts/bias_investigation/test_08_bootstrap_map.py`: change `POSTERIORS_DIR` from `phase44_posteriors` → `phase45_posteriors`, run `uv run python scripts/bias_investigation/test_08_bootstrap_map.py`, then revert the edit; OR
- (b) Copy to `scripts/bias_investigation/test_08_bootstrap_map_phase45.py` with the path edit and run it.

Inspect the output JSON's `interval_68`.
**Pass condition:** `interval_68[0] ≤ 0.73 ≤ interval_68[1]`.

### 5.3 Strategy invariance — Phase 44 invariant preserved
**Test ID:** `test-zero-strategies-equal`
**Procedure:** Inspect `results/phase45_posteriors/logs/combine_*.out` for the four strategies' MAPs (naive, exclude, per-event-floor, physics-floor).
**Pass condition:** All 4 MAPs are exactly equal (Phase 44 STAT-03 invariant; no zero events post-anchor).

### 5.4 σ_MAP stability
**Test ID:** `test-bootstrap-sigma-stable`
**Procedure:** Compare the bootstrap output's `std` field to the Phase 45 Step 0 baseline `σ_MAP = 0.0114`.
**Pass condition:** `std ≤ 0.025` (≤ 2× baseline; allows moderate posterior re-shaping).

---

## 6. Outcome → next step (escalation tree)

| Outcome | Branch | Action |
|---|---|---|
| **All 4 acceptance tests PASS** | A — SUCCESS | Phase 45 SUCCESS. Mark phase complete in `.gpd/STATE.md` (`Current Phase: 46` or `none`). Update `.gpd/ROADMAP.md` Phase 45 entry to `Status: Complete`. Sync GSD: update `.planning/STATE.md` if the active milestone (v2.2 Pipeline Correctness) is fully realized. Close any open GitHub issues tagged `paper-blocker` related to MAP=0.86 / MAP=0.7650 with a comment referencing commit `09ee262` and `45-03-SUMMARY.md`. |
| **MAP ∈ [0.74, 0.76]** (under-corrected) | B — UNDER | Conservative anchor 0.7931 was too low. Escalate to RESEARCH.md §4c — hybrid: keep `(0, 0.7931)` anchor + insert `(c_0/2, 1.0)` intermediate point (16/16 detected at d_L < 0.10 Gpc → p̂_split = 1.0). Open Plan 45-04 with the hybrid fix family. Do **not** mark Phase 45 complete. |
| **MAP ∈ [0.70, 0.72]** (over-corrected) | C — OVER | Conservative anchor 0.7931 was too high (unlikely with the Wilson lower bound; investigate). Open Plan 45-04: try anchor = pooled point estimate `p̂ = 0.8873` instead of CI lower bound, OR re-derive p_max from a tighter d_L bin (< 0.05 Gpc instead of < 0.10 Gpc). |
| **MAP unchanged at 0.7650** | D — NO-EFFECT | Patch did not take effect on cluster. **CHECKPOINT** — investigate: (1) commit hash mismatch on cluster, (2) Python cache / `.venv` not refreshed, (3) wrong `injection_data_dir` on cluster, (4) `evaluate.sbatch` running a different branch. Do not proceed; ask researcher. |
| **4 strategies disagree** | E — REGRESSION | Unrelated regression introduced in Plan 45-02. **CHECKPOINT** — diff `bayesian_statistics.py` (must be empty since Plan 45-02 did not touch it; if non-empty, something else changed); revert and reapply with the minimal anchor edit isolated. |
| **Interval_68 excludes 0.73 but MAP ∈ [0.72, 0.74]** | F — INFO | Bias is fixed; posterior is just narrower than the residual. Document and accept (RESEARCH.md §5 footnote: posterior shape is acceptable when MAP is in range). Mark Phase 45 complete with INFO flag in SUMMARY for Phase 46 follow-up. |

**Forbidden proxies (rejection conditions, per Plan 45-03 contract):**
- Skipping the cluster re-eval and claiming success on local-only proxy MAP from `simulations/prepared_cramer_rao_bounds.csv` (60 events) — the 412-event seed200 cluster posterior is required.
- Re-using `results/phase44_posteriors/` and just re-bootstrapping — those posteriors were generated with the pre-Plan-45 interpolator and are stale.
- Submitting before the [PHYSICS] commit hash matches on the cluster login node — pre-flight gate (§1) must pass first.
- Accepting `MAP > 0.74` or `MAP < 0.72` as "close enough" — the acceptance window is locked.
- Treating MAP slightly above 0.74 as a conservative-bound side-effect of using 0.7931 instead of 1.0 — this is real overshoot and triggers escalation to hybrid 4c (Branch B).

---

## 7. Reproducibility

- `cluster/submit_phase45_eval.sh` is committed (Plan 45-03 Task 1; differs from `submit_phase44_eval.sh` only in 4 cosmetic surgical edits — header, NEW_RUN, console messages, rsync hint).
- `cluster/evaluate.sbatch` and `cluster/combine.sbatch` are unchanged from Phase 44.
- The post-fix master commit hash `09ee262995b2ca95a997bba7d19f645d7ecff1de` is recorded:
  - In `cluster/submit_phase45_eval.sh` header comment.
  - In `45-03-ACCEPTANCE.md` (this document).
  - In `combined_posterior.json["git_commit"]` (auto-recorded per CLAUDE.md `--seed` policy).
  - In `45-03-SUMMARY.md` (written after the cluster re-eval).
- `_P_MAX_EMPIRICAL_ANCHOR = 0.7931` is a module-level constant in `master_thesis_code/bayesian_inference/simulation_detection_probability.py` with full Wilson-CI / LR-test provenance comment block above its definition.

---

## 8. Open questions (deferred to SUMMARY after run)

- Magnitude of MAP shift in production: Plan 45-02 predicted `−0.01 to −0.03` toward truth 0.73. The 412-event posterior is the binding test.
- Will the conservative anchor (0.7931) be enough to reach `[0.72, 0.74]`, or will hybrid 4c be required?
- Stability of the linear-interp slope discontinuity at c_0: acceptable per RESEARCH.md §9 risk register; revisit only if cluster re-eval shows posterior shape artifacts near c_0.
