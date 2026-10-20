# m-t5-armR — verdict-free readout (joint_r1)

Research Graph 1, Branch F. Design of record: `tree2_20260830/
PROPOSAL_MASS_LAW_KEYED_WINDOW_20260830.md` §6.2 (Arm R). Launch: `LAUNCH_RECORD.md` (job
6768608, 4-task array, H4 grid {0.660, 0.665, 0.670, 0.730}). **No verdicts beyond the design's
own pre-registered, mechanical band assignment (IMMATERIAL-CONSISTENT-WITH-HB /
INTERMEDIATE / MATERIAL) — assigned per the design's own stated rule (§6.2 "read on the same
three-way map", i.e. the T_mat=0.008 / 0.003 bands of §6.1/MEASUREMENT_HEAD_READOUT §5.1), not
chair interpretation. Window adoption itself stays with `d-t5-window`.**

## 1. sacct verify

```
JobID          State      ExitCode  Elapsed   NCPUS
6768608_0      COMPLETED  0:0       00:05:13   16
6768608_1      COMPLETED  0:0       00:05:12   16
6768608_2      COMPLETED  0:0       00:05:17   16
6768608_3      COMPLETED  0:0       00:05:04   16
```
4/4 tasks `COMPLETED 0:0`. (`.batch`/`.extern` substeps also `0:0`, omitted above.)

## 2. Retrieval

`rsync -aL` of `$WS/run_20260902_graph1_t5_armR_joint_r1/{simulations,run_metadata_{7,8,9,21}.json,
GIT_COMMIT_AT_RUN.txt}` to `retrieved/run_20260902_graph1_t5_armR_joint_r1/` (727 files, 398.1 MB,
0 deleted/skipped). md5 manifest of the 19 load-bearing files (event_likelihoods.csv,
cramer_rao_bounds.csv, prepared_cramer_rao_bounds.csv, fisher_quality.csv, all posteriors +
posteriors_with_bh_mass JSONs, all 4 run_metadata JSONs, GIT_COMMIT_AT_RUN.txt) —
**0/19 mismatches**, remote vs. local, see `retrieved/run_20260902_graph1_t5_armR_joint_r1/
MD5_MANIFEST.txt`. `prepared_cramer_rao_bounds.csv` md5 `9a1f2a14384a9281c97ca3be312ddaab`
matches the LAUNCH_RECORD's pinned CRB checksum — confirms the correct pinned dataset was read.

## 3. Flag-match against the banked headrebaseline joint_r1 baseline (tasks 7/8/9/21)

Comparand: `retrieved/run_20260902_graph1_headrebaseline_joint_r1/run_metadata_{7,8,9,21}.json`
(the banked joint_r1 HEAD readout the design elects as zero-compute baseline, §6.2). `cli_args`
diffed key-by-key against `run_metadata_{7,8,9,21}.json` of this arm's own retrieval, all 4 tasks:

| task (H4 idx) | h | diff found |
|---|---|---|
| 7 | 0.660 | `mass_filter_geometry`: linear→log; `mass_filter_k`: 1.5→3.0; `working_directory` (path only) |
| 8 | 0.665 | same two flags; `working_directory` (path only) |
| 9 | 0.670 | same two flags; `working_directory` (path only) |
| 21 | 0.730 | same two flags; `working_directory` (path only) |

**Result: PASS.** The only substantive differences at every one of the 4 nodes are exactly the
two registered arm variables (`mass_filter_geometry`, `mass_filter_k`); `working_directory` is an
expected path artifact, not a physics flag. No STOP triggered.

**Code-state note (disclosed, already gated):** baseline `git_commit` = `1ec9514d...`; this arm's
`git_commit` = `dcb2c470...` (both `run_metadata_*.json` and `GIT_COMMIT_AT_RUN.txt` agree). This
commit delta is the same one `LAUNCH_RECORD.md` already addresses: `dcb2c470` sits behind the
`g-c0-baseline-equivalent (Arm R)` gate (`exec/m-t5-armR-c0prime/eval/GATE_RECORD.md`, **GREEN**),
which reproduced the banked joint_r1 with-BH channel bit-for-bit at the current commit (17 columns
× 1588 rows, max_abs 0) — i.e. the intervening commits (including the 1D-only `a26959b4`
`[PHYSICS]` change) are independently certified immaterial to this comparand. Not re-litigated
here; cited as evidence the flag-match's baseline choice is code-state-safe.

Seeding: task 21 `seed=777021`, `h_value=0.73` matches in both baseline and new run (spot-checked;
seed convention `EVAL_SEED(777000) + H41 index` holds).

## 4. Stencil convention — which one applies to joint_r1 (task item 3 check)

§6.2's own text does not restate the `Δmean_h,pred = Δℓ'/I_HEAD` formula or an `I_HEAD` value; it
says only "(ii) Delta mean_h,pred on joint_r1 read **on the same three-way map**" — i.e. reuse of
the §6.1/`MEASUREMENT_HEAD_READOUT_20260827.md` §5.1 **band thresholds** (`T_mat=0.008`,
`IMMATERIAL ≤ 0.003`), which that source states are common to both 2D venues by deliberate
conservative choice ("A single `T_mat` = 0.008 is used for both 2D venues... conservative on
joint_r1").

The stencil **conversion constant** `I_HEAD = 1/σ_h²`, however, is venue-specific in the design's
own cited source, `MEASUREMENT_HEAD_READOUT_20260827.md` §C.1:

| venue | σ_h | I_HEAD = 1/σ_h² |
|---|---:|---:|
| iiib (Arm S's convention, `B7_2`/`B5_2` records) | 0.018366 | 2964.63 |
| **joint_r1 (this arm)** | **0.018637** | **2879.04** |

**This record uses the joint_r1-specific `I_HEAD = 2879.04`** (from `MEASUREMENT_HEAD_READOUT_
20260827.md` §C.1's joint_r1 row), not Arm S's iiib value of 2965 — the design's own source
registers them as distinct per-venue quantities, and §6.2 gives no instruction to reuse Arm S's
number. Both values are reported below for transparency; the band call is unaffected by the choice
(see §5).

## 5. Registered delta read (§6.2 item ii)

`Δℓ(h) = Σ_events ln(combined_with_bh^armR(h) / combined_with_bh^baseline(h))` over events with
both > 0 (all 1588/1588 events qualify at every node — no filtering needed); central-difference
stencil over `{0.660, 0.665, 0.670}` at spacing 0.005; `Δmean_h,pred = Δℓ'(0.665) / I_HEAD`.

| h | Δℓ(h) | n_used / n_common |
|---|---:|---:|
| 0.660 | +7.209958 | 1588 / 1588 |
| 0.665 | +7.245853 | 1588 / 1588 |
| 0.670 | +7.284247 | 1588 / 1588 |
| 0.730 (reported only, off-stencil) | +9.723085 | 1588 / 1588 |

- `Δℓ'(0.665) = +7.428881` nats/h
- `Δℓ''(0.665) = +100.002`

| I_HEAD used | Δmean_h,pred | band (design's own mechanical rule, T_mat=0.008 / 0.003) |
|---|---:|---|
| **2879.04 (joint_r1, this record's convention)** | **+0.0025803** | **IMMATERIAL-CONSISTENT-WITH-HB** (\|Δ\| ≤ 0.003) |
| 2964.63 (iiib, Arm S's convention — for comparison only, not applied) | +0.0025058 | IMMATERIAL-CONSISTENT-WITH-HB (\|Δ\| ≤ 0.003) |

**Band call is robust to the venue-constant choice**: both give IMMATERIAL-CONSISTENT-WITH-HB
(the design registered no MATERIAL-AT-SOME-k-style scan rule for Arm R — it is a single-point
arm, not a k-scan, so no scan-level verdict applies).

## 6. Gates

**R6 — 1D channel bit-identity** (design §6.2 item iii: "the 1D channel bit-identical"). `combined_
no_bh`, armR (log k=3.0) vs. baseline (linear k=1.5), all 4 H4 nodes, 1588 matched events each:

| h | max_abs diff |
|---|---:|
| 0.660 | 0.0 |
| 0.665 | 0.0 |
| 0.670 | 1.006e-16 |
| 0.730 | 0.0 |

**PASS** — floating-point noise only, matches the design's own registered prediction exactly.

**R5 — stencil validity** (`|Δℓ''(0.665)| ≪ I_HEAD`, same disclosed 10%-of-I_HEAD operationalization
used in the Arm S / B7.2 records): `100.002 / 2879.04 = 3.47%` — well inside "≪". **PASS, not
ambiguous**; no G27 escalation.

**R2 — engagement** (not explicitly registered in §6.2's text, reported for completeness in the
same convention Arm S used). At h=0.730: baseline non-empty `L_cat_with_bh` = 1094/1588; changed
among non-empty = 1068 → fraction **0.9762** (≥0.90 threshold if applied).

**g-znorm** — not evaluated. Same reasoning as `m-head-rebaseline`/Arm S records: the identity
check operates on `global_denom_no_bh`/`global_denom_with_bh`, which are not columns in
`event_likelihoods.csv` for this venue either; no fresh evaluation offered.

## 7. Item (i) of §6.2 — true-host recovery gain (NOT computed here)

§6.2's registered prediction (i) — true-host recovery among the 73 in-catalogue events rising by
16-22 points (expected +12 to +16 hosts) — requires per-event true-host identification data that
is not present in `event_likelihoods.csv`/`cramer_rao_bounds.csv` (no host-truth column retrieved
for either the armR run or the baseline). Computing it needs the host-recovery machinery §6.3 item
3 references (recompute of the in-catalogue count from the HEAD-readout diagnostics). **Left
unread here** — disclosed as an explicit gap, not silently dropped; a later record with the
host-truth join is needed before item (i)'s falsifier band (`[+8, +20]` hosts) can be checked.

## What this record is not

- Not a ruling on window adoption — reserved for `d-t5-window` (§6.2's own falsifier language:
  "STOP and return" is a trigger for author return, not a verdict this record makes).
- Not a read of §6.2 item (i) (true-host recovery) — see §7 above, explicitly deferred.
- No code edited, no commits, no cluster jobs, nothing awaited.
