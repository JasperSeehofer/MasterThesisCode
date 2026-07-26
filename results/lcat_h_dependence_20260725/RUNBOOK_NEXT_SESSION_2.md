# Runbook — post-adoption open threads (written 2026-07-26, session end)

Supersedes `RUNBOOK_NEXT_SESSION.md` (fully executed + closed out; kept for the
record). State ledger: `MULTISEED_READOUT_20260726.md` (campaign + ratification
+ redteam rescopes + fixpool + dense-core, all in one file). Redteam:
`../redteam_20260726/CONSOLIDATED_VERDICT.md`.

## State at handoff (all verified, nothing in flight)

- **Production stack ADOPTED and on `main` (9ebf6d9)**: PRs #37 + #43 merged;
  defaults are `generator_marginal` + `pdet_z_resolved=True` (`ce6338e`);
  pipeline-parity golden pins the defaults (`6b9dd98`) — any future default
  change must regenerate it per the test docstring.
- **Verification**: 4-seed bias PASS; dense-core (1e-4) grids on out-of-sample
  seeds 2000/3000 MEASURED the peak: MAP 0.7298–0.7299, σ 2.1–2.5e-4, offset
  −0.0002 ± 0.0002 → no detectable bias (#38 closed). Redteam: no truth-tuning;
  claims are mock-internal (rescope R3).
- Seed900: uninformative venue (zero golden events), excluded on measured
  information content; #44 tracks the information-floor criterion.
- P–P impostor harness on `feat/pp-impostor-harness` (786dc8d + 44ee912,
  pushed, UNMERGED); smoke (n=200) favorable; needs a full-power run + merge.
- Cluster queue empty. Workspace expires **2026-09-23**. Local /home ~99%
  full — clean `results/campaign_phase2_runs/run_20260726_seed{1000,2000,3000}_prodstack/logs/`
  (0.8–1.3 GB each, only warning spam) before any new retrieval.

## Open threads (pick per author priority)

1. **#40 [paper-blocker]** δ-kernel decomposition flag + photo-z/PV kernel
   derivation for real-data mode; paper methods must scope precision as
   mock-internal. Physics derivation → /gpd routing + /physics-change.
2. **#23 [paper-blocker]** completion-term realism at depth 1.5 (L_cat
   double-count, luminosity- vs rate-weighted completeness, K-correction).
3. **#39** blind alternative-truth mock (sealed h_inj) — decisive dynamic
   anti-tuning test; needs a new mock generation on the cluster.
4. **Paper fill**: 21 `\pending{}` markers (18 results.tex + conclusions +
   abstract). MAP/σ/N_det now measurement-backed from the prodstack +
   dense-core runs; subset-scatter figure needs a bootstrap analysis job.
   Use gpd-paper-writer; widths = measured σ ≈ 2.2e-4-class numbers with the
   mock-internal caveat sentence (rescope R3), per-venue table from the ledger.
5. **P–P harness**: full-power coverage run (n ≥ 2000/cell) on
   `feat/pp-impostor-harness`, then PR + merge; B_num residual-bias
   characterization (the smoke isolated B_num as sole carrier).
6. **#41/#42/#44** smaller: dgen 4d_exact rationale; medium bundle
   (ball-radius convergence, d_L cut in α(h), floor reporting); pre-register
   the information floor BEFORE any next campaign.
7. Log-spam fix (quadrature warning → per-event counter) before any large
   campaign — logs were 0.8–1.3 GB/run.

## Conventions worth re-reading next session

- Dense grids: `cluster/evaluate_densecore.sbatch` (manual `--array=0-40`;
  `submit_pipeline.sh` TC-14 parses only `evaluate.sbatch`'s H_VALUES line).
- Combine `.out` "Baseline/Current MAP" lines are channel summaries — read
  the JSONs, not the log lines.
- Per-event diagnostics: `simulations/diagnostics/event_likelihoods.csv`
  (event_idx × h × combined_no_bh) — first tool for any posterior anomaly.
