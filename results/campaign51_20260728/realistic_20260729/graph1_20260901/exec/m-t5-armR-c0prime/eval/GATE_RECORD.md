# m-t5-armR-c0prime — GATE RECORD

Research Graph 1, Branch F. Ingredient check for Arm R's baseline (PROPOSAL_MASS_LAW_KEYED_WINDOW_20260830.md
§6.2). Evaluated 2026-09-02 against `exec/m-t5-armR-c0prime/LAUNCH_RECORD.md`.

## Verdict

**GREEN — g-c0-baseline-equivalent (Arm R) identity holds.**

## sacct

```
6767465_0        graph1-t5-armR-c0prime  COMPLETED  0:0
6767465_0.batch  batch                   COMPLETED  0:0
6767465_0.extern extern                  COMPLETED  0:0
```
Runtime 6:37, single task (`--array=0-0`), h=0.730, seed 777021. Zero non-`COMPLETED 0:0` records.

## Retrieval

Source: `$WS/run_20260902_graph1_t5_armR_c0prime_joint_r1` (1.8G, under the 10G pre-transfer gate).
Transferred with `rsync -aL` to
`exec/m-t5-armR-c0prime/eval/retrieved_run/` (1549 files). Remote md5 manifest built cluster-side
(`find -L . -type f -print0 | sort -z | xargs -0 md5sum`, same convention as `wave1-retrieval/RECORD.md`)
and saved to `eval/remote_manifest.md5`; local `md5sum -c --quiet` against that manifest post-transfer:
**0 mismatches, 0 missing, 1549/1549 files verified.**

Existence checks (three-valued, all EXISTS): remote run dir before retrieval, both posterior JSONs,
the diagnostics CSV, both `run_metadata_21.json` files (new run + comparand).

## Comparand selection (evidence)

LAUNCH_RECORD names the comparand explicitly: the banked `run_20260902_graph1_headrebaseline_joint_r1`
task-21 row (already retrieved wave-1, row #300; local copy at
`graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_joint_r1/`), production-baseline flags
(`mass_filter_geometry=linear`, `mass_filter_k=1.5`, BLIND to `catalogue_numerator_survival_2d`/`_center`
— i.e. left at CLI default, not the C0-prime `off` variant used by the unrelated `g-c0-baseline` gate).
Only one banked joint_r1 variant matches this description in the retrieved set (`run_20260902_graph1_c0prime_headrebaseline_joint_r1`
is a different node — the wave-1 C0-prime-`off` gate — and was NOT used).

**Flag-match verification (the row #298/#299 lesson, applied here):** diffed the full `cli_args` dict from
both runs' `run_metadata_21.json`. **0 differences across all 61 keys**, excluding `working_directory`
(expected — different run roots) and `seed`/`simulation_index`/`h_value` (both 777021/21/0.73, identical
by construction). In particular `catalogue_numerator_survival_2d="mz_sel"` and `catalogue_numerator_survival_2d_center="eff"`
are identical in both — confirming both runs resolve the same CLI default, not a `c0prime_off`-style
override on either side. `mass_filter_geometry=linear`, `mass_filter_k=1.5` identical in both. The
comparand selected is flag-matched; no phantom-delta risk from a mismatched comparand.

Git commits differ by design: new run at `dcc75352` (this session's HEAD at launch), comparand at
`1ec9514d` (the wave-1 head-rebaseline commit) — this is exactly the code-state delta the gate exists
to certify away.

## Bit-identity table (shared columns, max_abs / md5)

| Artifact | Check | Result |
|---|---|---|
| `simulations/posteriors/h_0_73.json` | md5 | NEW `8ac1f2a4b461d681353da252652457f3` = COMP `8ac1f2a4b461d681353da252652457f3` — **MATCH** |
| `simulations/posteriors_with_bh_mass/h_0_73.json` | md5 | NEW `ae1e361cbed715fd0e362b3affdc596d` = COMP `ae1e361cbed715fd0e362b3affdc596d` — **MATCH** |
| `simulations/diagnostics/event_likelihoods.csv`, h=0.73 rows (1588/1588, columns equal) | max_abs per column | all 17 numeric columns: **max_abs=0.000000e+00, nonzero=0/1588** |

Diagnostics CSV columns checked: `w_G`, `w_G_legacy`, `w_tilde_G`, `alpha_G_phi`, `r_Malm`, `D_tilde_phi`,
`L_cat_no_bh`, `L_cat_with_bh`, `B_num`, `B_num_wbh`, `g_frac`, `L_comp`, `combined_no_bh`,
`combined_with_bh`, `den_log_term`, `num_log_term_no_bh`, `num_log_term_with_bh` — zero deltas on every
column, every row. 0 missing / 0 extra event indices between the two h=0.73 slices.

## Verdict statement

Bit-identity holds on every shared column of every named comparand artifact (both posterior JSONs by
md5, the diagnostics CSV by max_abs=0 across 17 columns × 1588 rows). **GREEN.** No numbers are banked
as a delta because there is no delta to bank.

If GREEN: Arm R's launch precondition (row #290 decisions row 8 / row #284(4a): "Arm R launch strictly
behind its own C0-prime-equivalent gate") is **satisfied** by this record. The launch of Arm R's own
measurement (`--mass_filter_geometry log --mass_filter_k 3.0`, H4 grid) is the chair's to dispatch, not
this record's — this record performs no launch and does not itself authorize one.

## Provenance

- sacct verified via `ssh bwunicluster 'sacct -j 6767465 ...'`, one poll.
- Retrieval: `rsync -aL` (foreground, backgrounded by the harness at 120s, completed; re-verified via
  local md5 manifest check post-transfer rather than trusting rsync's own exit code alone).
- Comparison script: ad hoc Python (md5 for JSON, per-column max_abs for the CSV), not committed.
- No commits made by this record. No interpretation beyond the identity/no-identity stamp.
