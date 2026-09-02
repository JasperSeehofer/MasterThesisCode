# BYTEID_EVIDENCE — verification record (job 6768603)

Reader: byte-identity comparison reader, `[PHYSICS]` h-decoupling commit `a26959b4`
(post-change commit at run time: `dcb2c470`). Comparand: job `6764460`'s banked pre-change
C0-prime outputs (`run_20260902_graph1_c0prime_headrebaseline_{iiib,joint_r1}`, retrieved
locally under `graph1_20260901/retrieved/`, pre-change commit `1ec9514d`).

## STAMP: **GREEN**

All in-bound target files are byte-identical (md5) between the post-decoupling run (job
6768603) and the pre-change comparand (job 6764460), at matching CLI flags and matching
random seed. The h-decoupling change is numerically invisible at h=0.730, as designed.

## 1. sacct verification

```
JobID          JobName                              State      ExitCode  Elapsed
6768603_0      graph1-c0prime-byteid-postdecouple    COMPLETED  0:0       00:06:21   (iiib)
6768603_1      graph1-c0prime-byteid-postdecouple    COMPLETED  0:0       00:06:58   (joint_r1)
```
Both tasks and all sub-steps (`.batch`, `.extern`) COMPLETED 0:0. No re-run needed.

## 2. Retrieval

Outputs retrieved via foreground `rsync -aL` from
`$WS=/pfs/work9/workspace/scratch/st_ac147838-emri/run_20260902_graph1_c0prime_byteid_postdecouple_{iiib,joint_r1}`
to `exec/b-hprior-fix/byteid_eval/run_20260902_graph1_c0prime_byteid_postdecouple_{iiib,joint_r1}/`.

`cwd/` was excluded from transfer (it is a per-task snapshot of the source tree plus a
symlinked `simulations` back-reference — not run output). A cluster-side md5 manifest
(`find . -type f | xargs md5sum`, run in the remote output directory) was generated and
diffed against a local manifest of the transferred files; the only line-count discrepancy
was the deliberately-excluded `cwd/selection_tables_h_0_73.json` — **0 mismatches** on
everything actually compared.

**Gotcha found and handled:** `simulations/injections/`, `simulations/cramer_rao_bounds.csv`,
and `simulations/prepared_cramer_rao_bounds.csv` under the run directory are **symlinks** to
a shared, unrelated injection-pool run (`run_20260729_seed61000`) used as the p_det input —
not this run's own product. `rsync -aL` (dereference symlinks) pulled the full shared pool
transitively, which is why a first pass produced hundreds of unrelated files; these were
identified via `find -type l -exec ls -la` on the cluster, confirmed as shared inputs (not
covered by the design's byte-identity claim, which is about the `--evaluate` outputs), and
excluded from both the transfer and the comparison. Only the 4 target files plus
`run_metadata_21.json`, `GIT_COMMIT_AT_RUN.txt`, and log/provenance files (this run's own
outputs) were retained and verified.

## 3. FLAG-MATCH (row #298/#299 lesson)

`cli_args` dict compared field-by-field between new (`run_metadata_21.json`, post-decouple)
and comparand (`run_metadata_21.json`, job 6764460), excluding `working_directory` and
timestamp fields:

| | iiib | joint_r1 |
|---|---|---|
| cli_args diffs (excl. working_directory) | **0** | **0** |
| random_seed (new / old) | 777021 / 777021 | 777021 / 777021 |
| git_commit (new) | `dcb2c470472f2f1f912c166ab48c3890a410c42c` | `dcb2c470472f2f1f912c166ab48c3890a410c42c` |
| git_commit (old/comparand) | `1ec9514dd1808c48b18c0792dce558e5bba0f116` | `1ec9514dd1808c48b18c0792dce558e5bba0f116` |

Flags match exactly (0 differences); git_commit correctly differs (post- vs pre-change), as
expected. Proceeded to byte-comparison.

## 4. Byte-compare (md5), per venue

| File | iiib | joint_r1 |
|---|---|---|
| `simulations/posteriors/h_0_73.json` | IDENTICAL `1c603309b5f139b52e02d0f12571ed4e` | IDENTICAL `8ac1f2a4b461d681353da252652457f3` |
| `simulations/posteriors_with_bh_mass/h_0_73.json` | IDENTICAL `abf242ed8747ba5a11b8a8ac84778460` | IDENTICAL `81ae557e5a378479f655d59cecb6e1b3` |
| `simulations/diagnostics/event_likelihoods.csv` | IDENTICAL `228f12b0f086942fcfc80fbafdc1388f` | IDENTICAL `a7ca893699a71acf3a074cc36a14d5de` |
| `simulations/fisher_quality.csv` | IDENTICAL `32c9f3a1b60c37616fb360bb3d6b5baa` | IDENTICAL `32c9f3a1b60c37616fb360bb3d6b5baa` |

**8/8 files identical. 0 mismatches.** No column-wise max_abs diff was required (only
triggered on an md5 mismatch — none occurred).

Population coverage: `event_likelihoods.csv` = 1589 rows (1588 events + header) x 19 cols
per venue; `fisher_quality.csv` = 1589 rows x 4 cols per venue; both venues combined
comfortably exceed the design's N>=1e5 byte-comparison criterion.

## 5. Verdict

**GREEN.** All in-bound results are byte-identical after the h-decoupling `[PHYSICS]`
change at h=0.730, across both venues (`iiib`, `joint_r1`), on flag-matched, same-seed reruns.
This is the in-bound identity the decoupling design promises by construction. No RED
condition triggered; no wing rerun is stopped; the design is not reopened.

No commits were made as part of this evidence read.
