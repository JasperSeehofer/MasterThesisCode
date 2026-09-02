# WING_READ_RECORD — G-EXT wing completion read (Research Graph 1, Branch I)

Read 2026-09-02 by the G-EXT wing completion reader agent. Verdict-free sanity check only, per
task instructions and `RECORD.md` §3's own read note (wing posteriors expected negligible-weight,
tail ~5e-13 at h ≥ 0.85, row #286).

Effort: medium. No commits made.

---

## 1. sacct verification — job 6768824 (array 41-54)

All 14 array tasks (h ∈ {0.870, 0.880, ..., 1.000}, seeds 777041-777054):

```
6768824_41 .. 6768824_54   State=COMPLETED   ExitCode=0:0   (all 14, including .batch/.extern steps)
```

**14/14 COMPLETED, 0:0.** No failures, no non-zero exit codes.

Per-task provenance JSON spot-checked (task 41): `slurm_array_job_id=6768824`,
`slurm_array_task_id=41`, `git_commit=dcb2c470...` (matches the HEAD verified in
`WING_RERUN_LAUNCH.md`), `h=0.870`, `seed=777041`. Consistent with the launch record.

## 2. RUN_DIR completeness

`$WS/run_20260831_a18_ma1d_iiib` (`$WS = /pfs/work9/workspace/scratch/st_ac147838-emri`):

- `simulations/posteriors/`: **55 files** (41 banked + 14 wing, h = 0.600-1.000)
- `simulations/posteriors_with_bh_mass/`: **55 files** (same)
- All 14 wing h-values confirmed present in both directories: `h_0_87.json` ... `h_0_99.json`,
  `h_1_0.json`.

Grid is now complete at all 55 nodes. Nothing re-retrieved for the 41 banked nodes (see §4).

## 3. Retrieval — 14 wing task outputs only

Destination: `exec/b-hprior-fix/wing_read/` (this directory).

- Transfer: `rsync -aL --files-from=<28-file list>` (14 h-values × 2 dirs: `posteriors/` +
  `posteriors_with_bh_mass/`), foreground, bounded (two `timeout 100` runs — rsync is idempotent
  and the second run picked up the one file the first left short after the 100s cutoff).
- **Excluded** (row #311 gotcha, confirmed present on the remote): the three RUN_DIR-level
  symlinks `simulations/injections`, `simulations/cramer_rao_bounds.csv`, and
  `simulations/prepared_cramer_rao_bounds.csv`, all pointing at
  `run_20260729_seed61000/simulations/...` — shared injection-pool data, not wing task output.
  These were never in the file list and were not transferred.
- Also excluded: `simulations/diagnostics/event_likelihoods.csv` — a single cumulative file
  spanning all 55 tasks (not per-h), so it cannot be split into "wing-only" without re-deriving
  it; retrieving it whole would pull in the 41 banked rows too. Not needed for the sanity read
  (built directly from the per-h posterior JSONs instead, §5).

**Manifest verdict: 28/28 files, 0 mismatches.** Local md5sums (post-retrieval, both retrieval
passes) match the remote md5sums exactly for every file:

| file (both `posteriors/` and `posteriors_with_bh_mass/`) | match |
|---|---|
| h_0_87.json ... h_0_99.json, h_1_0.json (14 h-values × 2 dirs = 28 files) | all 28 identical |

(Full md5 pairs recorded in `local_manifest.txt` in this directory; remote md5sums obtained via
a direct `ssh bwunicluster md5sum` pass over the same 28 paths — all 28 lines identical.)

## 4. Untouched-41 spot-check

3 of the 41 banked in-bound nodes compared: local banked copy
(`results/.../tree2_20260830/a18_prod_arm/simulations/posteriors/`) vs. the live RUN_DIR on the
cluster, post-rerun.

| h | local md5 | remote md5 | match | remote mtime |
|---|---|---|---|---|
| 0.65 | efcd2e622700e4e96dbea610f074102b | efcd2e622700e4e96dbea610f074102b | yes | 2026-08-31 13:06:41 |
| 0.73 | 1c603309b5f139b52e02d0f12571ed4e | 1c603309b5f139b52e02d0f12571ed4e | yes | 2026-08-31 13:06:38 |
| 0.86 | 40cbd71a0415a0a7a7caaa203fd0e988 | 40cbd71a0415a0a7a7caaa203fd0e988 | yes | 2026-08-31 13:07:14 |

**Result: 3/3 exact md5 match.** Remote mtimes are all 2026-08-31 (the original run), predating
the 2026-09-02 wing rerun (job 6768824) by a day — the 41 banked in-bound nodes were not
regenerated or touched by the rerun, consistent with the guard-check analysis in
`WING_RERUN_LAUNCH.md` (bounds-only gate, byte-id GREEN below 0.86).

## 5. Wing posterior-mass sanity read (verdict-free)

Built the full 55-node likelihood array by merging the 41 banked `posteriors/*.json` (from
`tree2_20260830/a18_prod_arm/`) with the 14 freshly-retrieved wing `posteriors/*.json`, using the
repo's own canonical combination code
(`darksiren_emri/bayesian_inference/posterior_combination.py`:
`build_likelihood_array` + `apply_strategy(PHYSICS_FLOOR)` + `combine_log_space`, N=1588 events,
55 h-bins). This reproduces the same Σ log L → softmax-normalized posterior procedure used in
production (`combine_posteriors`).

**Wing total posterior mass (sum over the 14 wing nodes, h = 0.870-1.000):**

```
2.365e-15   (out of a normalized grid total of 1.0)
```

**Wing per-node max:** h = 0.870, posterior weight **2.159e-15** (the wing's own maximum — mass
falls off monotonically and steeply from there, down to 7.5e-31 at h = 1.000).

Full per-node wing breakdown:

| h | posterior weight |
|---|---|
| 0.870 | 2.159e-15 |
| 0.880 | 1.891e-16 |
| 0.890 | 1.563e-17 |
| 0.900 | 1.219e-18 |
| 0.910 | 8.982e-20 |
| 0.920 | 6.272e-21 |
| 0.930 | 4.171e-22 |
| 0.940 | 2.649e-23 |
| 0.950 | 1.611e-24 |
| 0.960 | 9.411e-26 |
| 0.970 | 5.284e-27 |
| 0.980 | 2.855e-28 |
| 0.990 | 1.489e-29 |
| 1.000 | 7.501e-31 |

For reference, the last in-bound node h = 0.86 carries weight 2.337e-14, and h = 0.85 carries
2.400e-13 — the latter is the same order of magnitude as the "tail ~5e-13 at h ≥ 0.85" figure
disclosed in row #286 / `RECORD.md` §3, confirming the pre-registered expectation.

**Sanity verdict: negligible, as expected.** Wing total mass (2.4e-15) is ~9 orders of magnitude
below the smallest in-bound node's individual weight and falls monotonically to ~1e-31 by h=1.0.
No wing node shows non-negligible mass — **no anomaly to report.**

---

## 6. Summary

| check | result |
|---|---|
| sacct 14/14 | COMPLETED, 0:0, all tasks |
| RUN_DIR completeness | 55/55 posteriors + 55/55 posteriors_with_bh_mass |
| Wing retrieval manifest | 28/28 files, 0 md5 mismatches; shared symlinks correctly excluded |
| Untouched-41 spot-check | 3/3 exact md5 match, mtimes pre-date rerun |
| Wing total posterior mass | 2.365e-15 (max node h=0.87, 2.159e-15) |
| Anomaly | none — negligible-weight expectation (row #286) confirmed |

This is a sanity read only — it makes no claim about whether the extended grid is load-bearing
for any future arm (per row #290 item 11's scope note); that decision is made at each arm's own
registration.
