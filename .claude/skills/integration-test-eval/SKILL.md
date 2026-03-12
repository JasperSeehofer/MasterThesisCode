---
name: integration-test-eval
description: >
  Run the evaluation pipeline integration test with synthetic fixture data.
  Verifies BayesianStatistics.evaluate() produces valid posteriors and a plot.
disable-model-invocation: true
argument-hint: [--verbose|-v] [--capture=no|-s]
allowed-tools: Bash(uv run *), Read, Glob
---

## Integration Test: Evaluation Pipeline (Pipeline B)

### Run
```bash
uv run pytest master_thesis_code_test/integration/test_evaluation_pipeline.py -v -m slow --tb=long -s
```

### After completion
1. Report pass/fail
2. If failed, show full traceback
3. If passed, report detections processed and whether posterior JSON + plot were produced
4. Read the posterior JSON and report: number of detections, non-zero likelihoods count
