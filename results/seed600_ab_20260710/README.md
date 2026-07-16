# seed600 frozen-venue A/B re-evaluations (2026-07-10) — DEV RUNS, NOT CAMPAIGN DATA

Handoff item L-B (.planning/HANDOFF-LOCAL-NO-CLUSTER-20260710.md); plan of record
.planning/BIAS-INVESTIGATION-20260710.md [L2]. Venue: seed600 frozen shallow venue,
A/B-only (Omega_m era mismatch: CRBs simulated at 0.25, eval at 0.2726 — never quote
absolute bias numbers from this venue without the era term).

Inputs (identical to the 562918ef PV-test run_live):
- CRB + prepared CSV: symlinked from results/pv_correction_test_20260703/run_live/simulations/
- Injection pool: 80 CSVs -> simulations/injections_RETIRED_predt2_zcut0p5_20260703 (shallow z<=0.5; --allow_low_pdet_coverage escape)
- Catalogue: live z_cmb reduced_galaxy_catalogue.csv (mtime 2026-07-02, unchanged since run_live)
- 17-pt grid 0.725..0.805 (fused --h_values), --seed 600999, 8 workers

Arms:
- run_A_fc45d1f: perf branch tip (pre-fallback). Purpose: code-drift gate vs the
  562918ef run_live artifacts (expected: 1D byte/1e-9-comparable).
- run_B_f29a5e7: physics/zero-host-completion-fallback (#29 + #30 caps). Purpose:
  first real-data fallback footprint (run_live had 13/3355 = 0.4% zero-host drops;
  expected MAP shift <= grid step) + per-h host-lookup yield metric in logs (INFO).
