# READ_RECORD_Q2.md — r-timeout-selection, `q-timeout-population-mismatch` (Q2) ONLY

Disjoint reader for `m-timeout-q2`. Q1 (`q-timeout-selection-pdet`) is out of scope and was not touched.
The p0 axis is out of scope by construction (D1 record, ratified bound). `INFORMATION_FORECAST.md` was
**not** opened (FORBIDDEN, honored). `MECHANISM_NOTE.md` was read (permitted). Data files were not
inspected by hand; every number below comes from the script's `--out` JSON or from stdout at run time.
**This record is VERDICT-FREE**: the three-valued outcomes below are the script's own computed values,
reported verbatim, with no adjudication added.

## 0. Gates consulted before launch

- `DESIGN_GATE_Q2_computability_rev2.md` — **Verdict: GREEN** (F5/AMBER/F6 all independently re-verified
  closed; "Q2 is launchable as registered (REVISION 2 + the `CHAIR ERRATUM`)").
- `DESIGN_GATE_Q2_formula.md` — **Verdict: GREEN**, with one open, non-blocking, REPORTED-ONLY finding
  (F1, an S2.1 p0/e0-quintile edge-source question) that "does not touch S2.2, S2.3, or any
  `g-byteid`/`g-closure` gate" and "does not block Q2 launch on its own terms."

Both gates GREEN. Launch proceeded.

## 1. Exact command

Run from repo root (`/home/jasper/Repositories/darksiren-emri`), `uv run python`, once, real mode (no
`--dry-run`). Flags and paths reproduced from `BUILD_RECORD_Q2.md` §1's dry-run invocation (which itself
cites `REGISTRATION_DRAFT.md` REVISION 2 + the `CHAIR ERRATUM`), with `.../` elisions expanded to full
repo-relative paths, `--dry-run` dropped, and `--out` set to this task's target path. `--null-draws`,
`--null-seed`, `--perm-draws`, `--perm-seed` were left at their script defaults (`1000`/`20260904`/
`10000`/`20260904`), which are exactly the registered values (§4 of `REGISTRATION_DRAFT.md`: "10,000-
permutation p ... seed `20260904`"; "1000 draws ... seed `20260904`"), so no explicit override was needed:

```
uv run python results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-timeout-selection/timeout_q2_reads.py \
    --crb-csv results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv \
    --crb-csv-md5 9a1f2a14384a9281c97ca3be312ddaab \
    --bin-edges-json results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/rd-timeout-bin-seed61000/design_gate_bin_edges.json \
    --bin-edges-md5 e24b07fe3948559b02d8dd4dbe8df8b3 \
    --rate-table-m-csv results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/rd-timeout-bin-seed61000/rate_table_M.csv \
    --rate-table-m-md5 b0d6284c06eb2f185158819d47123de5 \
    --pool-dir results/campaign51_20260728/realistic_20260729/gate_b_20260730/injection_pool_mix200k_20260728 \
    --pool-manifest results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-timeout-selection/POOL_MANIFEST.md5 \
    --pool-manifest-md5 75f4030d5d3b0405fd948049bef5767e \
    --log-dir results/campaign51_20260728/realistic_20260729/seed61000/cluster_logs_fetch_20260904 \
    --log-manifest results/campaign51_20260728/realistic_20260729/seed61000/cluster_logs_fetch_20260904_MANIFEST.md5 \
    --log-manifest-md5 ebf09fc4ab66b55e4eb592731ee46ae6 \
    --event-likelihoods-iiib results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv \
    --event-likelihoods-iiib-md5 8e6a2c18dc5838dd1d52641589243672 \
    --event-likelihoods-jr1 results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_joint_r1/simulations/diagnostics/event_likelihoods.csv \
    --event-likelihoods-jr1-md5 745954a0fdee5f10878fb5e622a06144 \
    --influence-iiib results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/influence_iiib.csv \
    --influence-iiib-md5 d20a01734cc825625f14ba7ec82c67ae \
    --influence-jr1 results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/influence_joint_r1.csv \
    --influence-jr1-md5 38f3f1813a3d460093763dd89019ca8a \
    --out results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-timeout-selection/timeout_q2_result_read.json
```

**stdout:**
```
wrote results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-timeout-selection/timeout_q2_result_read.json: Q2-S2.2 = M-STRUCTURED, Q2-S2.3 = POPULATION-MISMATCH-MATERIAL
```

**Exit code: 0.** Script did not crash; no traceback; no `InstrumentDefectError` was raised (real-mode
`main()` raises to a JSON `disposition.value == "INSTRUMENT-DEFECT"` + exit 1 on any pin/gate failure —
none fired here). Output written to `timeout_q2_result_read.json` (source of every number below unless
otherwise cited).

## 2. Pins and gates

All checks below run unconditionally inside `main()` **before** the `--dry-run` branch (`timeout_q2_reads.py:884-919`),
so their passing is entailed by exit 0 even though real mode prints only the one summary line above (the
per-pin echo lines are `--dry-run`-only). Cross-referenced against `BUILD_RECORD_Q2.md`'s dry-run of the
identical pinned inputs (2026-09-04, same script), which does print them.

| gate | check | result |
|---|---|---|
| **G-1 pins** | `_check_pin` on CRB CSV, bin-edges JSON, `rate_table_M.csv`; `load_pool`/`parse_logs` manifest md5 (per-file); `load_influence` md5 ×2 | all OK — no `InstrumentDefectError`; md5s match §1 of `REGISTRATION_DRAFT.md` exactly (CRB `9a1f2a14384a9281c97ca3be312ddaab`, edges `e24b07fe3948559b02d8dd4dbe8df8b3`, rate table `b0d6284c06eb2f185158819d47123de5`, pool manifest `75f4030d5d3b0405fd948049bef5767e`, log manifest `ebf09fc4ab66b55e4eb592731ee46ae6`, influence_iiib `d20a01734cc825625f14ba7ec82c67ae`, influence_jr1 `38f3f1813a3d460093763dd89019ca8a`) |
| scored-population size check | `len(crb_scored) == N_SCORED (1588)` | held (else `InstrumentDefectError`) — JSON `meta.n_scored = 1588` |
| **g-byteid** | `n_kept == (0,9,1276,303,0)` AND `n_timeout_snr == (206,302,216,81,15)` | **both matched, no raise.** JSON: `n_kept_per_bin = [0, 9, 1276, 303, 0]`; `n_timeout_snr_stage_per_bin = [206, 302, 216, 81, 15]` |
| CRB-stage timeouts (reported, not a gate target) | `logs["crb_stage_timeout_M"]` | n=2, bins `[3, 2]` (M=1950892.90→bin3, M=576074.30→bin2) |
| **g-closure(i)** (Q1-scoped, disclosed only — never gates Q2) | `sum_Y − 3488 − (84,762+822)` | **residual = 384** — matches `BUILD_RECORD_Q2.md`'s dry-run value exactly; not a Q2 gate, disclosed per §6 |
| **g-closure(ii)** (`Σ_e w_e = 1588`) | `abs(sum_w_e_post_renorm − 1588) ≤ 1e-6` re-check in `main()` before `S2_3_reweighted_posterior` | held — `sum_w_e_pre_renorm = 1588.0000000000005`, `sum_w_e_post_renorm = 1587.999999999999` (\|Δ\| ≈ 1.0e-12 ≪ 1e-6) |
| physics-floor exclusions (`g-precision`) | `_load_logl` raises if `n_excluded != 0` for any of iiib/jr1 × {with_bh, no_bh} | held (0 exclusions in all channels used — no raise) |
| `g-population` (raw counts, not re-printed in real mode; cross-referenced from `BUILD_RECORD_Q2.md`'s dry-run of the same pinned inputs, same script) | 100/100 tasks; Σ Y; pool rows; D1-gate lines | Σ Y = 89,456; pool = 200,100 rows (99,014 a-stratum); D1-gate lines = 4,071 — all as pinned in §1/§6 of `REGISTRATION_DRAFT.md`; **not independently re-echoed by this real-mode run's stdout**, flagged for the record rather than asserted as freshly observed |
| `g-scope` | script never imports `SimulationDetectionProbability`; `s2_1_information_map`'s `by_p0_bin_reported_only` and `s2_4_scatter_summary`'s `p0_median` are the only p0 touches, both REPORTED-ONLY | held by code structure (per `BUILD_RECORD_Q2.md` §4 checklist); no p0 input feeds either disposition |

## 3. Per-M-bin information table (S2.1, REPORTED-ONLY)

Bins 0 and 4 have `n_kept = 0` and are absent from the JSON's `by_M_bin` map (no rows to summarize).

| M bin | n | σ_lnDL median | σ_lnDL IQR | Ω median (sr) | Ω IQR (sr) | SNR median | SNR IQR | gen_time median (s) | gen_time IQR (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 9 | 0.042937 | 0.008932 | 1.9620e-4 | 2.9704e-4 | 24.781 | 2.148 | 0.278 | 0.061 |
| 2 | 1276 | 0.037063 | 0.018859 | 2.6157e-4 | 4.8086e-4 | 28.506 | 16.016 | 0.233 | 0.037 |
| 3 | 303 | 0.037286 | 0.018715 | 5.6205e-4 | 1.0390e-3 | 27.679 | 16.618 | 0.208 | 0.0305 |

**Spearman ρ_S(log10 M, ln σ_lnDL) [REPORTED-ONLY, S2.1]:** ρ = −0.056522, p_perm = 0.024198 (n_perm =
10,000, seed 20260904).

(p0/e0-bin tables also present in the JSON, REPORTED-ONLY, unused by any gated statistic — omitted here
for length; available at `S2_1_information_map_reported_only.by_p0_bin_reported_only` /
`.by_e0_bin_reported_only` in the `--out` JSON.)

## 4. S2.2 — ρ_S(log10 M, d_e) and top-k composition, three families

| family | k | n | ρ_S(log10 M, d_e) [gates] | p_perm(d_e) | ρ_S(log10 M, \|d_e\|) [REPORTED-ONLY] | min Holm-p | any bin Holm-p < 0.05 |
|---|---|---|---|---|---|---|---|
| iiib_2d (PRIMARY) | 82 | 1588 | 0.226246 | 9.999e-05 | 0.071129 | 1.7497e-09 | true |
| iiib_1d (replicate) | 94 | 1588 | 0.184643 | 9.999e-05 | 0.073154 | 2.4883e-08 | true |
| jr1_2d (replicate) | 72 | 1588 | 0.229843 | 9.999e-05 | 0.051339 | 2.8400e-08 | true |

10,000-perm p, seed 20260904 for all three. `p_perm(d_e)` is bounded below by `1/(n_perm+1) =
9.999e-05` in every family (no permutation draw reached or exceeded the observed statistic).

**Top-k Fisher-exact per M bin, Holm-corrected over 5 bins** (`top_k_in_bin` / `top_k_not_in_bin` vs
`bulk_in_bin` / `bulk_not_in_bin`, `bulk` = all 1588 minus the top-k):

| family | bin | top-k in bin | top-k not in bin | bulk in bin | bulk not in bin | Fisher p | Holm p |
|---|---|---|---|---|---|---|---|
| iiib_2d (k=82) | 0 | 0 | 82 | 0 | 1506 | 1.0 | 1.0 |
| iiib_2d | 1 | 0 | 82 | 9 | 1497 | 1.0 | 1.0 |
| iiib_2d | 2 | 42 | 40 | 1234 | 272 | 9.169e-10 | 3.668e-09 |
| iiib_2d | 3 | 40 | 42 | 263 | 1243 | 3.499e-10 | 1.750e-09 |
| iiib_2d | 4 | 0 | 82 | 0 | 1506 | 1.0 | 1.0 |
| iiib_1d (k=94) | 0 | 0 | 94 | 0 | 1494 | 1.0 | 1.0 |
| iiib_1d | 1 | 0 | 94 | 9 | 1485 | 1.0 | 1.0 |
| iiib_1d | 2 | 52 | 42 | 1224 | 270 | 1.057e-08 | 4.229e-08 |
| iiib_1d | 3 | 42 | 52 | 261 | 1233 | 4.977e-09 | 2.488e-08 |
| iiib_1d | 4 | 0 | 94 | 0 | 1494 | 1.0 | 1.0 |
| jr1_2d (k=72) | 0 | 0 | 72 | 0 | 1516 | 1.0 | 1.0 |
| jr1_2d | 1 | 0 | 72 | 9 | 1507 | 1.0 | 1.0 |
| jr1_2d | 2 | 37 | 35 | 1239 | 277 | 1.307e-08 | 5.227e-08 |
| jr1_2d | 3 | 35 | 37 | 268 | 1248 | 5.680e-09 | 2.840e-08 |
| jr1_2d | 4 | 0 | 72 | 0 | 1516 | 1.0 | 1.0 |

## 5. S2.3 PRIMARY — w_b, re-weighted posterior, Δmean_h, σ'_h/σ_h, null band

**Supported bins:** {2, 3} (`n_kept ≥ 10`), per REVISION 1 F2/F3 + REVISION 2 F5. `n_events_reweighted =
1579` (bins 2+3), `n_events_unit_weight = 9` (bin 1, `w_e = 1`, disclosed BOUND — bins 0/4 have 0 kept
events, nothing to unit-weight there).

| bin | n_kept | n_pool_det (SNR≥20) | share_kept (support-normalized) | share_pool,det (support-normalized) | w_b |
|---|---|---|---|---|---|
| 2 | 1276 | 4387 | 0.808106 | 0.703158 | 0.870130 |
| 3 | 303 | 1852 | 0.191894 | 0.296842 | 1.546912 |

`n_pool_det_per_bin` (all 5 bins, for reference): [76, 1217, 4387, 1852, 16].
`sum_w_e_pre_renorm = 1588.0000000000005`, `sum_w_e_post_renorm = 1587.999999999999` (renormalised once
over all 1588 events, per REVISION 1).

**Re-weighted posterior, three families:**

| family | mean_h (anchor) | mean_h (re-weighted) | Δmean_h^Q2 | σ_h (anchor) | σ_h (re-weighted) | σ'_h/σ_h | null draws / seed | SD(Δ_null) | T_null = max(0.002, 2·SD) |
|---|---|---|---|---|---|---|---|---|---|
| iiib_2d (PRIMARY) | 0.6658540600 | 0.655075149 | **−0.010778911** | 0.018474739 | 0.018576749 | **1.005522** | 1000 / 20260904 | 0.002989539 | 0.005979078 |
| iiib_1d (replicate) | 0.6669870586 | 0.661962597 | −0.005024461 | (not gated — no anchor σ_h stored for 1D in report) | 0.017839656 | — | 1000 / 20260904 | 0.002649684 | 0.005299368 |
| jr1_2d (replicate) | 0.6671274830 | 0.654387834 | −0.012739649 | (not gated) | 0.018370227 | — | 1000 / 20260904 | 0.003023299 | 0.006046598 |

`sigma_ratio` (`σ'_h/σ_h`) is computed and reported only for the PRIMARY family (`iiib_2d`), per
`s2_3_reweighted_posterior`'s field population; the replicates' `delta_mean_h` are reported alongside per
§4's "Read `Δmean_h^Q2`, `σ'_h/σ_h` (iiib 2D primary; 1D + joint_r1 replicates)."

**Band context (§5 of `REGISTRATION_DRAFT.md`, quoted, not applied as a verdict here):** `T_mat = 0.008`;
MATERIAL if `|Δmean_h^Q2| ≥ T_mat` OR `σ'_h/σ_h ∉ [0.80, 1.25]`; IMMATERIAL if `|Δ| ≤ T_null` AND ratio ∈
`[0.95, 1.05]`; else INTERMEDIATE.

## 6. S2.3 decomposition (REPORTED-ONLY) and S2.4 (REPORTED-ONLY)

`share_to(b)` of the 820 SNR-stage timeouts (+2 CRB-stage, reported separately, never folded in):

| bin | share_to (of 820) |
|---|---|
| 0 | 0.251220 |
| 1 | 0.368293 |
| 2 | 0.263415 |
| 3 | 0.098780 |
| 4 | 0.018293 |

CRB-stage timeouts (n=2): bins [3, 2], reported only, never in `n_timeout` or the decomposition denominator.
D1-gate share per bin: **NOT-EVALUABLE** ("no params logged at the D1 catch site; disclosed").

S2.4 (timeout vs. kept scatter summary, REPORTED-ONLY): all 822 timeouts — n=822, log10(M) median =
5.129245, IQR = 0.863720. Kept 1588 — log10(M) median = 5.758561, p0 median = 12.828789. `mu/M` note:
"mu constant 10 in every sampled record (`MECHANISM_NOTE.md` §5, spot-checked); mu/M reported via M only."

## 7. Three-valued disposition outcomes (verbatim from the JSON — VERDICT-FREE)

**Q2-S2.2** (`disposition_s2_2`, driven by the PRIMARY family `iiib_2d`'s `p_perm_d_e` and
`any_bin_holm_p_lt_0p05`; band from §5 of the draft: `p_perm < 0.01` AND top-k Fisher/Holm-p < 0.05 in ≥1
bin → MATERIAL tag `M-STRUCTURED`; `p_perm ≥ 0.10` → `M-FLAT`; else INTERMEDIATE):

```json
{
  "value": "M-STRUCTURED",
  "p_perm": 9.999000099990002e-05,
  "any_bin_holm_p_lt_0p05": true,
  "mandatory_note": "p0 axis not evaluated (D1 record, ratified bound; read's p0 row corrected per MECHANISM_NOTE.md §3)"
}
```

**Q2-S2.3** (`disposition_s2_3`, driven by the PRIMARY family `iiib_2d`'s `|Δmean_h|`, `sigma_ratio`,
`t_null`; band from §5 of the draft: `|Δ| ≥ T_mat` OR ratio ∉ `[0.80,1.25]` → MATERIAL tag
`POPULATION-MISMATCH-MATERIAL`; `|Δ| ≤ T_null` AND ratio ∈ `[0.95,1.05]` → IMMATERIAL; else
INTERMEDIATE):

```json
{
  "value": "POPULATION-MISMATCH-MATERIAL",
  "abs_delta_mean_h": 0.010778910960530852,
  "t_mat": 0.008,
  "t_null": 0.005979077616678977,
  "sigma_ratio": 1.0055215875664847,
  "mandatory_note": "p0 axis not evaluated (D1 record, ratified bound; read's p0 row corrected per MECHANISM_NOTE.md §3)"
}
```

Both rows carry the mandatory p0-scope line verbatim, as required by §5 of the draft. No fresh RULE is
adjudicated in this record — the values above are the script's own three-valued outputs, reported for the
author/orchestrator's RULE at `d-timeout-selection-register`.

## 8. Output artifact

Full JSON: `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-timeout-selection/timeout_q2_result_read.json`
(this run's sole `--out` write; 504 lines pretty-printed). Every number in §§2–7 above is read from that
file (or, where marked, cross-referenced to `BUILD_RECORD_Q2.md`'s dry-run of the identical pinned inputs).
