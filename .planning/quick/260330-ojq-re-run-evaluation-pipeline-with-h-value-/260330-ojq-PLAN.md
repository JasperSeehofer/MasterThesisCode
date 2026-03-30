---
phase: quick
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/compare_posterior_bias.py
autonomous: true
requirements: [diagnostic-bias-rerun]
must_haves:
  truths:
    - "Diagnostic posteriors exist for all 11 h-values in evaluation/run_v12_diagnostic/simulations/posteriors/"
    - "Biased posteriors are preserved unchanged in evaluation/run_v12_validation/simulations/posteriors/"
    - "Comparison report shows where the log-posterior peaks for old vs new runs"
  artifacts:
    - path: "evaluation/run_v12_diagnostic/simulations/posteriors/"
      provides: "11 h-value posterior JSON files from diagnostic code"
    - path: "scripts/compare_posterior_bias.py"
      provides: "Comparison script: old biased vs new diagnostic posteriors"
    - path: "evaluation/run_v12_diagnostic/comparison_report.md"
      provides: "Markdown summary of bias shift"
  key_links:
    - from: "evaluation/run_v12_validation/simulations/prepared_cramer_rao_bounds.csv"
      to: "evaluation/run_v12_diagnostic/simulations/prepared_cramer_rao_bounds.csv"
      via: "symlink or copy"
      pattern: "same CRB data for both runs"
---

<objective>
Re-run Pipeline B evaluation with the diagnostic bias fix (quick task 260330-oaf) for all 11
h-values and compare against the original biased results.

Purpose: Confirm whether removing the spurious /d_L factor and disabling P_det shifts the
posterior peak from h=0.60 (biased) to h=0.73 (true value).
Output: 11 new posterior JSONs + a comparison report showing old vs new peak location.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/quick/260330-oaf-diagnostic-bias-fix-remove-d-l-factor-an/260330-oaf-SUMMARY.md
@master_thesis_code/bayesian_inference/bayesian_statistics.py
@evaluation/run_v12_validation/simulations/posteriors/h_0_73.json
</context>

<tasks>

<task type="auto">
  <name>Task 1: Set up diagnostic run directory and execute 11 h-value evaluations</name>
  <files>evaluation/run_v12_diagnostic/</files>
  <action>
1. Create the diagnostic run directory structure:
   ```
   mkdir -p evaluation/run_v12_diagnostic/simulations/posteriors
   mkdir -p evaluation/run_v12_diagnostic/simulations/posteriors_with_bh_mass
   ```

2. Symlink the CRB data so the evaluation code can find it (it reads from
   `simulations/prepared_cramer_rao_bounds.csv` relative to CWD):
   ```
   ln -sfn "$(pwd)/evaluation/run_v12_validation/simulations/prepared_cramer_rao_bounds.csv" \
           evaluation/run_v12_diagnostic/simulations/prepared_cramer_rao_bounds.csv
   ```
   Also symlink the full `cramer_rao_bounds.csv` if it exists (some code paths reference it):
   ```
   ln -sfn "$(pwd)/evaluation/run_v12_validation/simulations/cramer_rao_bounds.csv" \
           evaluation/run_v12_diagnostic/simulations/cramer_rao_bounds.csv
   ```

3. Verify the diagnostic flag is active by importing the module:
   ```bash
   cd evaluation/run_v12_diagnostic
   uv run python -c "from master_thesis_code.bayesian_inference.bayesian_statistics import _DEBUG_DISABLE_DETECTION_PROBABILITY; assert _DEBUG_DISABLE_DETECTION_PROBABILITY, 'Flag must be True'"
   ```
   This must succeed (the flag was set to True in quick task 260330-oaf).

4. Run all 11 h-value evaluations sequentially from the diagnostic directory.
   The evaluation writes posteriors to `simulations/posteriors/` relative to CWD:
   ```bash
   cd evaluation/run_v12_diagnostic
   for H in 0.600 0.626 0.652 0.678 0.704 0.730 0.756 0.782 0.808 0.834 0.860; do
     echo "=== Evaluating h=$H ==="
     uv run python -m master_thesis_code . --evaluate --h_value "$H" 2>&1 | tail -5
   done
   ```
   Note: Each evaluation takes ~1-5 minutes on CPU with 22 detections. Total: ~15-55 min.
   The working directory argument is `.` (current dir = run_v12_diagnostic), so the code
   will find `simulations/prepared_cramer_rao_bounds.csv` via the symlink.

5. Verify all 11 posterior files were created:
   ```bash
   ls -la evaluation/run_v12_diagnostic/simulations/posteriors/
   ```
   Expect 11 JSON files matching the pattern `h_*.json`.
  </action>
  <verify>
    <automated>test $(ls evaluation/run_v12_diagnostic/simulations/posteriors/h_*.json 2>/dev/null | wc -l) -eq 11 && echo "PASS: 11 posterior files created"</automated>
  </verify>
  <done>All 11 h-value posterior JSONs exist in evaluation/run_v12_diagnostic/simulations/posteriors/</done>
</task>

<task type="auto">
  <name>Task 2: Create comparison script and generate bias report</name>
  <files>scripts/compare_posterior_bias.py, evaluation/run_v12_diagnostic/comparison_report.md</files>
  <action>
Create `scripts/compare_posterior_bias.py` that:

1. Accepts `--biased` and `--diagnostic` directory arguments (paths to run directories,
   e.g., `evaluation/run_v12_validation` and `evaluation/run_v12_diagnostic`).
   Also accepts `--output` for the report path.

2. For each of the 11 h-value JSON files in both directories:
   - Load the JSON (keys are detection indices 0..21 plus "h"; values are single-element lists)
   - Compute `log_posterior = sum(log(likelihood_i))` for each detection i (this is log of the
     product, which is the actual posterior)
   - Also compute `sum_likelihood = sum(likelihood_i)` for reference
   - Store h_value, log_posterior, sum_likelihood

3. Find the peak h-value for each run (h where log_posterior is maximum).

4. Generate a markdown report with:
   - Table: h_value | biased_log_posterior | diagnostic_log_posterior | delta
   - Summary: "Biased peak: h={X}, Diagnostic peak: h={Y}, True value: h=0.73"
   - Verdict: Does the diagnostic fix shift the posterior peak toward h=0.73?

5. Also generate a simple ASCII bar chart or visual indicator showing the posterior shape
   for both runs (normalized log-posterior relative to peak).

6. Write the report to `--output` path.

Then run the comparison:
```bash
uv run python scripts/compare_posterior_bias.py \
  --biased evaluation/run_v12_validation \
  --diagnostic evaluation/run_v12_diagnostic \
  --output evaluation/run_v12_diagnostic/comparison_report.md
```

Print the report to stdout as well so results are visible in the execution log.
  </action>
  <verify>
    <automated>test -f evaluation/run_v12_diagnostic/comparison_report.md && echo "PASS: report exists" && head -30 evaluation/run_v12_diagnostic/comparison_report.md</automated>
  </verify>
  <done>Comparison report exists showing old peak (expected h~0.60) vs new peak (expected h~0.73) with a clear verdict on whether the diagnostic fix resolved the bias</done>
</task>

</tasks>

<verification>
1. `ls evaluation/run_v12_diagnostic/simulations/posteriors/h_*.json | wc -l` returns 11
2. `ls evaluation/run_v12_validation/simulations/posteriors/h_*.json | wc -l` still returns 11 (unchanged)
3. `cat evaluation/run_v12_diagnostic/comparison_report.md` shows both peaks and a verdict
4. The biased posteriors in `evaluation/run_v12_validation/` were NOT modified
</verification>

<success_criteria>
- All 11 diagnostic evaluations completed successfully
- Comparison report clearly shows whether posterior peak shifted from h=0.60 toward h=0.73
- Original biased results preserved in run_v12_validation (no files modified)
- Comparison script is reusable for future runs
</success_criteria>

<output>
After completion, create `.planning/quick/260330-ojq-re-run-evaluation-pipeline-with-h-value-/260330-ojq-SUMMARY.md`
</output>
