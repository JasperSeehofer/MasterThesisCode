# VERIFY-02: Abort-Gate Check

**Timestamp:** 20260423T172607Z
**Phase:** 40 Wave 2
**Requirement:** VERIFY-02
**Gate rule (D-03 #1):** `|MAP_v2.2 - MAP_v2.1| / 0.73 >= 0.05` → ABORT

## Summary

| Metric                            | v2.1 (baseline) | v2.2 (current) | Delta         | Role           |
|-----------------------------------|-----------------|----------------|---------------|----------------|
| MAP h                             | 0.7350          | 0.7350         | +0.0000      | ABORT gate     |
| CI lower (68%)                    | 0.7262          | 0.7262         | +0.0000      | report only    |
| CI upper (68%)                    | 0.7401          | 0.7401         | +0.0000      | report only    |
| CI width                          | 0.0139          | 0.0139         | +0.0000      | report only    |
| bias_percent                      | +0.68%        | +0.68%       | +0.00pp   | SC-2 (< 1%)    |
| KS statistic (log P curves)       | -               | 0.0263         | p = 1 | report only    |
| N events                          | 417             | 417            | +0            | info           |

## Abort-Gate Computation

- |ΔMAP| = |0.7350 - 0.7350| = 0.0000
- |ΔMAP| / 0.73 = 0.0000%
- Threshold (D-03 #1): 5.0000%
- **Verdict: PASS**

## SC-2 (bias < 1% at h=0.73)

- v2.2 bias_percent = +0.68%
- SC-2 threshold: |bias_percent| < 1.00%
- SC-2 status: PASS

## Provenance

- v2.1 baseline: `simulations/_archive_v2_1_baseline/posteriors/` (see `ARCHIVE_MANIFEST.md` for git_commit + shas)
- v2.2 current:  `simulations/posteriors/` (re-evaluated in Task 1 of this plan)
- Comparison helper: `master_thesis_code.bayesian_inference.evaluation_report.generate_comparison_report`
- KS test: `scipy.stats.ks_2samp` on 38 aligned h-values

## Related artifacts

- Raw re-eval log: `.planning/debug/verify02_reeval_20260423T172607Z.log`
- Quadrature warnings capture: `.planning/debug/verify02_quadrature_warnings_20260423T172607Z.log`
- JSON sidecar (standard generate_comparison_report output): `.planning/debug/comparison_verify02_20260423T172607Z.json`
