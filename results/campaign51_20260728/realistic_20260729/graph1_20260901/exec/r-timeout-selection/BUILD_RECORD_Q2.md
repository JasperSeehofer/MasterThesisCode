# BUILD_RECORD_Q2.md — r-timeout-selection, `q-timeout-population-mismatch` (Q2) ONLY

Builder: sonnet, mechanical build task. Script: `timeout_q2_reads.py`. Built per
`DESIGN_GATE_Q2_computability_rev2.md` (GREEN, F5/AMBER/F6 all closed) against
`REGISTRATION_DRAFT.md` REVISION 2 (Q2) + the `CHAIR ERRATUM`. Q1
(`q-timeout-selection-pdet`) is OUT OF SCOPE and untouched by this script. The
p0 axis is OUT OF SCOPE by construction (D1 record, ratified bound).

**Real mode was NOT run.** Per the build mandate, this record contains only
(a) a `--dry-run` invocation of the draft's Q2 launch block on the REAL pinned
inputs (loads, pins, schema, per-bin counts; no aggregate; must exit 0), and
(b) a synthetic (≤10-row) check exercising every disposition row and the
INSTRUMENT-DEFECT path.

## 1. `--dry-run` on the real, pinned §1 inputs — exit 0, no PIN CORRECTION needed

Every file the draft's Q2 launch block names exists on disk at the pinned
path, and every md5 verified exactly against `REGISTRATION_DRAFT.md` §1 /
`DESIGN_GATE_Q2_computability_rev2.md` — no `PIN CORRECTION (Q2 build)` was
required.

```
$ uv run python timeout_q2_reads.py \
    --crb-csv results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv \
    --crb-csv-md5 9a1f2a14384a9281c97ca3be312ddaab \
    --bin-edges-json .../graph1_20260901/exec/rd-timeout-bin-seed61000/design_gate_bin_edges.json \
    --bin-edges-md5 e24b07fe3948559b02d8dd4dbe8df8b3 \
    --rate-table-m-csv .../rd-timeout-bin-seed61000/rate_table_M.csv \
    --rate-table-m-md5 b0d6284c06eb2f185158819d47123de5 \
    --pool-dir results/campaign51_20260728/realistic_20260729/gate_b_20260730/injection_pool_mix200k_20260728 \
    --pool-manifest .../r-timeout-selection/POOL_MANIFEST.md5 \
    --pool-manifest-md5 75f4030d5d3b0405fd948049bef5767e \
    --log-dir results/campaign51_20260728/realistic_20260729/seed61000/cluster_logs_fetch_20260904 \
    --log-manifest results/campaign51_20260728/realistic_20260729/seed61000/cluster_logs_fetch_20260904_MANIFEST.md5 \
    --log-manifest-md5 ebf09fc4ab66b55e4eb592731ee46ae6 \
    --event-likelihoods-iiib .../retrieved/run_20260902_graph1_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv \
    --event-likelihoods-iiib-md5 8e6a2c18dc5838dd1d52641589243672 \
    --event-likelihoods-jr1 .../retrieved/run_20260902_graph1_headrebaseline_joint_r1/simulations/diagnostics/event_likelihoods.csv \
    --event-likelihoods-jr1-md5 745954a0fdee5f10878fb5e622a06144 \
    --influence-iiib .../r-offset-subset/influence_iiib.csv \
    --influence-iiib-md5 d20a01734cc825625f14ba7ec82c67ae \
    --influence-jr1 .../r-offset-subset/influence_joint_r1.csv \
    --influence-jr1-md5 38f3f1813a3d460093763dd89019ca8a \
    --out /tmp/.../q2_dryrun_final.json \
    --dry-run

CRB CSV: results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv (1590 rows; scored subset 1588), md5 OK
bin edges: .../rd-timeout-bin-seed61000/design_gate_bin_edges.json, md5 OK (6 edges)
rate_table_M.csv: .../rd-timeout-bin-seed61000/rate_table_M.csv, md5 OK (reported-only)
pool: .../injection_pool_mix200k_20260728 (200100 rows), manifest md5 OK
logs: .../seed61000/cluster_logs_fetch_20260904 (100/100 tasks), manifest md5 OK; Sigma Y = 89456; D1-gate lines = 4071
influence_iiib.csv: .../influence_iiib.csv (1588 rows), md5 OK
influence_joint_r1.csv: .../influence_joint_r1.csv (1588 rows), md5 OK
n_kept per bin (g-byteid): [0, 9, 1276, 303, 0]
n_timeout SNR-stage per bin (g-byteid): [206, 302, 216, 81, 15]
CRB-stage timeouts (reported only): n=2
g-closure(i) residual (Q1-scoped, disclosed only): 384
dry-run OK (no aggregate computed)
$ echo $?
0
```

`n_kept` and `n_timeout` (SNR-stage) reproduce the `g-byteid` anchors
`[0, 9, 1276, 303, 0]` / `[206, 302, 216, 81, 15]` bit-identically; the
`g-closure(i)` residual (384, Q1-scoped, disclosed only per
`REGISTRATION_DRAFT.md` §6) matches `MECHANISM_NOTE.md` §3's own reconciliation
exactly. `--dry-run` loads every §1 pin, verifies every md5 (file-level and,
for the pool/log manifests, per-member-file `md5sum -c`), reports per-bin
counts, and exits 0 **without computing any S2.x aggregate** — no Spearman, no
`w_b`, no re-weighted posterior, no `Δmean_h^Q2`.

## 2. Synthetic (≤10-row) check — `synth_check_q2.py`

Two parts, neither touching the registered 1588-event population:

- **Part 1 (INSTRUMENT-DEFECT, real CLI mechanism, tiny fixtures):** (a) a
  wrong `--crb-csv-md5` against a 3-row synthetic CRB CSV → exit 1,
  `INSTRUMENT-DEFECT` printed, JSON `disposition.value ==
  "INSTRUMENT-DEFECT"`; (b) a correct md5 but a 3-row table (the
  `N_SCORED == 1588` anchor cannot match) → exit 1, `INSTRUMENT-DEFECT`; (c)
  the same broken fixture under `--dry-run` → exit 1, `--out` never written
  (dry-run never writes a file on any path).
- **Part 2 (disposition rows + gates, direct function calls — the population
  anchors are hard-coded to 1588 and cannot be exercised end-to-end at
  ≤10 rows, so the registered LOGIC functions are called directly with
  fabricated arrays instead):**
  - `gbyteid_gate`: matching anchors → no raise; `n_kept` mismatch → raises
    `InstrumentDefectError` naming `n_kept`; `n_timeout` mismatch → raises
    naming `n_timeout`.
  - `s2_3_weights`: a 6-event fixture (1 unsupported bin-1 event at `w_e=1`,
    3 bin-2 + 2 bin-3 events reweighted) → supported bins `{2,3}`,
    `n_events_unit_weight=1`, `n_events_reweighted=5`, `Σ w_e` renormalises to
    exactly `len(events)` (a genuine off-by-population-size bug this check
    caught and the builder fixed — see §3).
  - `_weighted_moments`: all-zero weights reproduce the flat-posterior grid
    mean over `H_GRID_41`-style grids (the same `mean_h_all_removed`
    cross-check `build_influence_vector.py` reports); a larger uniform weight
    sharpens the posterior (`sigma_h` non-increasing).
  - `disposition_s2_2`: MATERIAL / IMMATERIAL / INTERMEDIATE all hit,
    including the "p_perm<0.01 alone, no Holm" non-MATERIAL edge case; the
    mandatory p0-scope line is present on every row.
  - `disposition_s2_3`: MATERIAL via `|Δ|≥T_mat` alone, MATERIAL via
    `sigma_ratio` outside `[0.80,1.25]` alone (with `Δ=0`), IMMATERIAL,
    INTERMEDIATE; mandatory p0-scope line present on every row.
  - `_holm`: monotone, dominates the raw p-values, and the α=0.05 boundary
    behaves as expected on a 5-value fixture.

```
$ uv run python synth_check_q2.py
[... 9 Part-1 PASS lines ...]
[... 4 Part-2a PASS lines ...]
[... 5 Part-2b PASS lines ...]
[... 4 Part-2c PASS lines ...]
[... 7 Part-2d PASS lines ...]
[... 7 Part-2e PASS lines ...]
[... 4 Part-2f PASS lines ...]

============================================================
SYNTH CHECK: all checks PASSED
$ echo $?
0
```

40/40 checks PASS, exit 0.

## 3. Bug the synthetic check caught and the fix

`s2_3_weights`'s final renormalisation originally read
`events["w_e"] = events["w_e_raw"] * (N_SCORED / total)`, hard-coding the
module-level `N_SCORED = 1588` constant instead of the actual row count of
the `crb_scored` argument passed in. On the real 1588-event population this
is a no-op (harmless), but the synthetic 6-event fixture caught it
immediately (`Σ w_e` renormalised to 1588, not 6). Fixed to
`* (len(events) / total)` — `g-closure(ii)` now holds for any input size, not
only the registered one. Re-run of the synthetic check after the fix: all 5
`s2_3_weights` checks PASS (§2 above).

## 4. Checklist table — draft item → function/line

| draft item (REGISTRATION_DRAFT.md / DESIGN_GATE_Q2_computability_rev2.md) | function / location in `timeout_q2_reads.py` |
|---|---|
| G-1 pins, every §1 Q2 input, STOP on mismatch | `_check_pin`, `_check_manifest`, `_verify_pins`-equivalent calls at the top of `main()` |
| `g-byteid` n_kept `[0,9,1276,303,0]` (REVISION 2 F5) | `compute_n_kept` + `gbyteid_gate` |
| `g-byteid` n_timeout (SNR-stage) `[206,302,216,81,15]` (CHAIR ERRATUM) | `compute_n_timeout_snr_stage` + `gbyteid_gate`; log parse via `_SNR_STAGE_MSG`/`_M_RE` in `parse_logs` |
| CRB-stage 2 timeouts, reported alongside, never in n_timeout/S2.3 decomp (CHAIR ERRATUM) | `_CRB_STAGE_MSG` branch in `parse_logs`; `N_TIMEOUT_CRB_STAGE_TOTAL`; `s2_3_decomposition`'s `crb_stage_timeouts_reported_only` |
| `g-population`: 100/100 tasks, Σ Y = 89,456, 4,071 D1-gate lines, pool 200,100/99,014 | `parse_logs` (`_Y_RE`, `"in dervative"` count), `load_pool` |
| `g-closure(i)` residual, Q1-scoped, disclosed only | `gclosure_i_gate` (never gates Q2, disclosed-only per §meta) |
| `g-closure(ii)` Σ w_e = 1588 | `s2_3_weights` renormalisation + explicit re-check in `main()` before `S2_3_reweighted_posterior` runs |
| Scored-event set = `{0..1589} − {1203,1356}` | `scored_crb` |
| S2.1 info map (REPORTED-ONLY): n/median/IQR of σ_lnDL, Ω, SNR, generation_time per M/p0/e0 bin; Spearman(log10 M, ln σ_lnDL) + 10k-perm p | `s2_1_information_map`, `_sky_area` |
| S2.2 ρ_S(log10 M, d_e) [gates] + ρ_S(log10 M, \|d_e\|) [REPORTED-ONLY] (REVISION 1 F4); 10k-perm p seed 20260904; top-k Fisher/Holm k=82/94/72 | `s2_2_influence_vs_M`, `_holm` |
| S2.3 PRIMARY: `w_b` over supported bins {2,3} only (REVISION 1 F2/F3, REVISION 2 F5), unit weight elsewhere, ONE renormalisation | `s2_3_weights` |
| Re-weighted T0 posterior (frozen convention, cited from `build_influence_vector.py`) | `_weighted_moments` (uses `biv._moments`/`biv._md5`/`biv._load_matrix`), `s2_3_reweighted_posterior`, `_load_logl` |
| `Δmean_h^Q2`, `σ'_h/σ_h` (iiib 2D primary; anchor `0.6658540600`/`0.018474739`) | `s2_3_reweighted_posterior` (`anchor_mean_h`, `anchor_sigma_h`, `sigma_ratio` set only for `PRIMARY_FAMILY`) |
| Same-size null: 1000 draws, `w_e` permuted over events, seed 20260904 | `s2_3_reweighted_posterior` (`rng.permutation(w_e)` loop) |
| S2.3 decomposition REPORTED-ONLY: `share_to(b)` of the 820 SNR-stage timeouts (+2 CRB-stage, reported); D1-gate share NOT-EVALUABLE | `s2_3_decomposition` |
| S2.4 REPORTED-ONLY: timeouts' (log10 M, p0, mu/M) vs kept | `s2_4_scatter_summary` |
| Q2-S2.2 disposition (3-valued, fresh RULE) | `disposition_s2_2` |
| Q2-S2.3 disposition (3-valued, fresh RULE, band [0.80,1.25]) | `disposition_s2_3` |
| Mandatory p0-scope line on every disposition row | `MANDATORY_P0_LINE`, embedded in both `disposition_s2_*` returns |
| `--dry-run`: loads, pins, schema, per-bin counts; no aggregate; exit 0 | `main()`'s `if args.dry_run:` branch (before any `s2_*` call) |
| `--out` JSON with every intermediate | `main()`'s real-mode branch: `report` dict assembles `meta`, `S2_1_*`, `S2_2_*`, `S2_3_weights`, `S2_3_reweighted_posterior`, `S2_3_decomposition_reported_only`, `S2_4_reported_only`, both dispositions |
| Q1 out of scope — no S1.x statistic, no `p_det`/pool-timeout-tally touch | script never imports `SimulationDetectionProbability`; `load_pool` reads only `M`/`SNR`/`stratum` |
| p0 axis out of scope — REPORTED-ONLY only | `s2_1_information_map`'s `by_p0_bin_reported_only`; `s2_4_scatter_summary`'s `p0_median`; no p0 input to any gated statistic |

## 5. ruff / mypy

```
$ uv run ruff check timeout_q2_reads.py synth_check_q2.py
All checks passed!
$ uv run mypy timeout_q2_reads.py synth_check_q2.py
Success: no issues found in 2 source files
```

## 6. Real mode

**Not run**, per the build mandate. `timeout_q2_reads.py` is ready for the
author/orchestrator to invoke in real mode on the pinned §1 inputs once Q2 is
launched (`CHAIR ERRATUM`: "LAUNCH of Q2 is deferred to the morning docket as
[DO] R18").
