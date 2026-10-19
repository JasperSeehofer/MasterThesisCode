# m-s0b-byteid — comparison record — 2026-09-02

## Stamp: RED

0 mismatches was the gate. Mismatches were found (non-zero on 5 of 8 compared
files). Numbers are banked below, not interpreted — per the node's own
registered gate ("Any mismatch -> STOP `m-s0b-production` launch and return
to the author"), `m-s0b-production` launch is **STOPped**; the build question
reopens as a fresh [RULE] for the author.

## (1) sacct verify

```
JobID|JobName|State|ExitCode|Elapsed|Start|End
6769608|graph1-m-s0b-byteid|COMPLETED|0:0|00:38:24|2026-09-02T13:17:02|2026-09-02T13:55:26
6769608.batch|batch|COMPLETED|0:0|00:38:24|2026-09-02T13:17:02|2026-09-02T13:55:26
6769608.extern|extern|COMPLETED|0:0|00:38:24|2026-09-02T13:17:02|2026-09-02T13:55:26
```
COMPLETED, 0:0, all three steps. Verified.

## (2) Retrieval

`rsync -aLv --exclude='injections'` from the cluster out-root
(`.../graph1_20260901/exec/m-s0b-byteid/byteid_cell_run/`) to local
`exec/m-s0b-byteid/comparison/byteid_cell_run/`. Confirmed before transfer
that `simulations/injections` is a symlink to the shared pool
(`.../gate_b_20260730/injection_pool_mix200k_20260728`) on the fresh cell too
(row #311 gotcha reproduced) — excluded, not dereferenced. 31,043,312 bytes
received, 12 files. md5 manifests written:
`exec/m-s0b-byteid/comparison/fresh_manifest.md5`,
`exec/m-s0b-byteid/comparison/banked_manifest.md5`.

Banked reference resolved locally (no ssh needed for this side):
`results/campaign51_20260728/realistic_20260729/tree2_20260830/
hier_s0_zwin_bnodes_run/s0a_seed900103/
node_b_minus_sites2.2_nosmear_divisor_zwin_zk4/`. Its own
`simulations/injections` is also a symlink to the same shared-pool target —
excluded from the banked manifest the same way.

## (3) CONFIG-MATCH — PASS

Fresh (`s0a_full_output.json`) vs banked (`s0a_score_output.json`, the
config-of-record cited in the task):

| field | banked | fresh | match |
|---|---|---|---|
| arm | S0-A | S0-A | yes |
| config | b0i | b0i | yes |
| theta_sites | 2.2 | 2.2 | yes |
| smear | off | off | yes |
| theta_phi_divisor | on | on | yes |
| sky_cone_k | 1.5 | 1.5 | yes |
| theta_zwindow | on | on | yes |
| z_window_k | 4.0 | 4.0 | yes |
| catalogue_leg_1d_mass_aware | off | off | yes |
| h_values | [0.73] | [0.73] | yes |
| node_dir_suffix | (n/a) | _sites2.2_nosmear_divisor_zwin_zk4 | matches dir name |
| n_events (seed 900103, b_minus) | 105 | 105 | yes |

Differences found and dispositioned as expected/non-substantive:
- `score_h`: banked `0.73` vs fresh `None` — banked is a `score_only` run
  (aggregates a scoring pass over the raw cell); fresh is the `s0a_full_output`
  of the raw re-run itself, which does not populate `score_h` at all. Field is
  produced by a different code path, not a config divergence of the executed
  cell.
- `registration` path: `/pfs/data6/home/st/.../PREREGISTRATION_HIER_HTHETA_20260826.md`
  (fresh, cluster Lustre mount) vs `/home/jasper/.../PREREGISTRATION_HIER_HTHETA_20260826.md`
  (banked, local mount) — same file, mount-prefix difference only.
- `provenance_6769608_none.json`: `git_commit=c83e391d8994da46033abdfe02529b7572b892a1`
  (matches the pinned HEAD), `tree_dirty_file_count=597` (untracked results
  artefacts, not source dirt), timestamps/hostname/job-id as expected for a
  fresh job.

0 config differences beyond commit/timestamp/cwd/mount-prefix. Proceeded to
byte-compare per the gate's own rule.

## (4) Byte/numeric comparison

md5 first, all 8 comparable files (excluding `injections`, `logs/`,
`provenance_*.json`, `s0a_full_output.json` — none of which have banked
counterparts at the per-node level and are not "shared output files"):

| file | banked md5 | fresh md5 | md5 match | max\|Δ\| | leaf/value count |
|---|---|---|---|---|---|
| `es_null_det.csv` | b8be167d... | b8be167d... | **yes** | — | 400 |
| `selection_tables_h_0_73.json` | 11e2150a... | 11e2150a... | **yes** | — | 15 |
| `fisher_quality.csv` | 623416a9... | 623416a9... | **yes** | — | 420 |
| `cramer_rao_bounds.csv` | 31f3329b... | 722019e5... | no | 4.440892e-16 | 26,200 |
| `prepared_cramer_rao_bounds.csv` | 31f3329b... | 722019e5... | no | 4.440892e-16 | 26,200 |
| `diagnostics/event_likelihoods.csv` | 60971f35... | ca35d91f... | no | 3.725290e-09 (`B_num`) | 1,995 |
| `posteriors/h_0_73.json` | 420ea47d... | e969bae9... | no | 1.040834e-16 | 106 |
| `posteriors_with_bh_mass/h_0_73.json` | 7bed212e... | 9018ceec... | no | 3.148671e-09 (`.galaxy_likelihoods.165[70][1][0]`) | 1,807,408 |
| `fisher_quality_diagnostic.pdf` | ce26a717... | fc7b6a44... | no | n/a (binary plot) | — |

Notes on the numeric-mismatch rows (banked, not interpreted as "safe" —
flagged for the author's own read):
- `cramer_rao_bounds.csv` / `prepared_cramer_rao_bounds.csv`: identical
  content to each other in both banked and fresh (each pair shares one md5),
  200x131, all 131 columns numeric+categorical checked; only column `qS`
  carries any nonzero diff, at `4.440892098500626e-16` (2 ULP at double
  precision) — every other column's max abs diff is exactly `0.0`.
- `event_likelihoods.csv`: 105x19, `B_num` max diff `3.725290298461914e-09`,
  `B_num_wbh` max diff `9.313225746154785e-10`; the remaining 5 non-integer
  columns with any diff (`num_log_term_with_bh`, `num_log_term_no_bh`,
  `L_cat_no_bh`, `combined_no_bh`, `L_cat_with_bh`) are all ≤5.33e-15; `w_G`,
  `h`, `event_idx` and the rest are exact.
- `posteriors/h_0_73.json`: 106 scalar leaves, all paths structurally
  identical, single max diff `1.0408340855860843e-16`.
- `posteriors_with_bh_mass/h_0_73.json`: recursive walk found 1,807,408
  scalar leaves (vs the launch record's independent estimate of ~1,862,936
  from the banked file — same file family, ~2.9% lower on this seed's actual
  event/candidate count; still >>1e5). Structure identical (0 key-set/length
  mismatches at any level), 0 non-numeric-value mismatches, single max diff
  `3.1486706575378776e-09` at one leaf.
- `fisher_quality_diagnostic.pdf`: identical byte length (19,022 bytes both
  sides); `cmp -l` found exactly 8 differing bytes, all inside the
  `/CreationDate` PDF metadata field (`D:20260902135525+02'00'` fresh vs
  `D:20260831143718+02'00'` banked) — a plot-render timestamp, not a data
  field.

## Total compared value count

400 + 15 + 420 + 26,200 + 26,200 + 1,995 + 106 + 1,807,408 = **1,862,744**
scalar/cell values compared (N>=1e5 satisfied by ~18.6x, consistent with the
launch record's sizing argument).

## Summary

- md5-identical: 3 of 8 files (`es_null_det.csv`, `selection_tables_h_0_73.json`,
  `fisher_quality.csv`).
- md5-mismatched with numeric max\|Δ\| in the 1e-16 to 1e-9 range: 5 files
  (`cramer_rao_bounds.csv`, `prepared_cramer_rao_bounds.csv`,
  `event_likelihoods.csv`, `posteriors/h_0_73.json`,
  `posteriors_with_bh_mass/h_0_73.json`).
- md5-mismatched, non-numeric (timestamp-only): 1 file
  (`fisher_quality_diagnostic.pdf`).
- 0 structural mismatches (shapes, columns, key sets, list lengths) anywhere.
- 0 config differences.

Per the node's own gate ("0 mismatches -> green... Any mismatch -> STOP"):
**not** 0 mismatches. **RED.** `m-s0b-production` launch STOPped; return to
the author as a fresh [RULE] — specifically whether sub-1e-8 floating-point
non-associativity (plausible cause: BLAS/thread-order/hardware differences
between the original 2026-08-31 `runner11` node and this 2026-09-02 re-run
node, both `--jobs 1` / single-process) is within the tolerance the author
intends for "byte-identity," or whether the gate as registered requires
exact reproduction and this counts as a genuine RED.

*Stamp: m-s0b-byteid comparison, 2026-09-02. No `git commit` made.*
