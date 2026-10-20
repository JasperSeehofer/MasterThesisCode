# BYTEID_RECORD.md -- Independent byte-id verifier for BUILD_RECORD_B2.md (Phase B, b-offset-subset-scorer / influence vector)

**Verifier role:** independent byte-id check only. Wrote `byteid_check.py` from scratch
(does not import `build_influence_vector.py` or any other builder code). Did not run the
production pipeline, cluster, or any registered aggregate over the registered population.
Compared literal values only.

## Verdict

**GREEN** -- 30/30 checks passed. Every anchor listed in `REGISTRATION_DRAFT.md` §"G-2
byte-id anchors" (lines ~163-168) matches `BUILD_RECORD_B2.md` and the `influence_*.csv`
files within its stated tolerance, cross-checked against the reference JSON
`exec/rd-2d-bootstrap-jackknife/rd_2d_bootstrap_jackknife_output.json`.

## What was checked

Script: `byteid_check.py` (this directory). Sources used, all read directly (no import of
B2's own script):

- `REGISTRATION_DRAFT.md` -- literal anchor values (quoted, not re-derived)
- `BUILD_RECORD_B2.md` -- parsed via regex from its own markdown tables (full-sample table +
  the four "(B) top-10 by decreasing directional influence d_e" tables)
- `influence_iiib.csv`, `influence_joint_r1.csv` -- read and independently re-sorted by each
  of `influence_2D` / `influence_1D` (the builder's own `rank` column, keyed to
  `influence_2D` only, was not trusted for the 1D re-sort)
- `exec/rd-2d-bootstrap-jackknife/rd_2d_bootstrap_jackknife_output.json` -- treated as the
  reference/ground truth per the draft's own citation (row #342)

### (i) Full-sample mean_h, 10 s.f.

| family | JSON `full_sample.mean_h` | registered anchor | \|delta\| | tol | verdict |
|---|---|---|---|---|---|
| iiib 2D | 0.6658540599535224 | 0.6658540600 | 4.648e-11 | 1e-9 | PASS |
| iiib 1D | 0.6669869414473403 | 0.6669869414 | 4.734e-11 | 1e-9 | PASS |

`BUILD_RECORD_B2.md`'s reported `mean_h_full` for all four families (iiib 2D/1D, joint_r1
2D/1D) matches the JSON `full_sample.mean_h` to <5e-11 (well inside the 1e-9 tolerance),
including the two joint_r1 families that carry no literal registered anchor in the draft
(reported here as an internal-consistency check only, not a registered comparison).

### (ii) Minimal directional-influence subset k

| family | JSON `minimal_k_events_removed` | registered k | BUILD_RECORD `minimal_k_recomputed` | BUILD_RECORD `banked_k` | verdict |
|---|---|---|---|---|---|
| iiib 2D (PRIMARY) | 82 | 82 | 82 | 82 | PASS |
| iiib 1D | 94 | 94 | 94 | 94 | PASS |
| joint_r1 2D | 72 | 72 | 72 | 72 | PASS |
| joint_r1 1D | 46 | 46 | 46 | 46 | PASS |

Exact integer match, all four families, both the JSON, the draft's literal anchor, and both
of BUILD_RECORD_B2.md's k columns.

### (iii) Top-10 `top10_events_by_abs_influence` -- event_idx + value, 1e-12 relative

**Sign-convention note (not a defect):** the JSON field stores the raw signed
`infl_e = mean_h(full) - mean_h(full-e)` (negative for these events, since removing a
truth-ward-pulling event moves mean_h away from 0.73). `BUILD_RECORD_B2.md`'s list (B) and
the `influence_*.csv` columns store the *directional* `d_e = sign(0.73 - mean_h(full)) *
(-infl_e)` per `REGISTRATION_DRAFT.md` line 68 (`0.73 - mean_h(full) > 0` for every one of
these four families, so `d_e = -infl_e` throughout). A literal value comparison without this
transform shows a spurious exact sign flip (relative "error" = 2.0) on all 40 entries; this
script applies the documented transform before comparing, and the underlying event_idx
ordering and magnitudes are identical to 1e-12 relative in every case.

| family | events checked | JSON vs BUILD_RECORD (B) | JSON vs `influence_*.csv` (re-sorted) |
|---|---|---|---|
| iiib 2D | 10 | PASS (all 1e-12) | PASS (all 1e-12) |
| iiib 1D | 10 | PASS (all 1e-12) | PASS (all 1e-12) |
| joint_r1 2D | 10 | PASS (all 1e-12) | PASS (all 1e-12) |
| joint_r1 1D | 10 | PASS (all 1e-12) | PASS (all 1e-12) |

Event-index ordering matches exactly (no permutation, no substitution) in all 4x10 = 40
entries, for both the BUILD_RECORD.md table and the raw CSV (re-derived by this script's own
independent sort, not read from the builder's `rank` column).

### (iv) k=1588 endpoint == 0.73

| family | curve_sample k=1588 mean_h | \|delta\| | tol | BUILD_RECORD `mean_h(all removed)` | verdict |
|---|---|---|---|---|---|
| iiib 2D | 0.730000000000078 | 7.805e-14 | 1e-12 | 0.73 | PASS |
| iiib 1D | 0.7299999999997618 | 2.381e-13 | 1e-12 | 0.73 | PASS |
| joint_r1 2D | 0.7300000000001493 | 1.493e-13 | 1e-12 | 0.73 | PASS |
| joint_r1 1D | 0.7300000000001783 | 1.783e-13 | 1e-12 | 0.73 | PASS |

### (v) Zero physics-floor exclusions

All four families: JSON `n_excluded_physics_floor == 0` and `BUILD_RECORD_B2.md`'s
`n_excluded` column `== 0`. PASS, all four.

### md5 pins (spot re-check, informational -- not one of the numbered G-2 anchors but claimed as "verified" in BUILD_RECORD_B2.md)

| input | claimed md5 | recomputed md5 | verdict |
|---|---|---|---|
| iiib `event_likelihoods.csv` | `8e6a2c18dc5838dd1d52641589243672` | `8e6a2c18dc5838dd1d52641589243672` | MATCH |
| joint_r1 `event_likelihoods.csv` | `745954a0fdee5f10878fb5e622a06144` | `745954a0fdee5f10878fb5e622a06144` | MATCH |

## Full check log

```
30/30 checks passed
VERDICT: GREEN
```

Full itemized pass/fail output of `byteid_check.py` reproduced above by section; the script
is deterministic and re-runnable (`python3 byteid_check.py` from this directory).

## Scope note

This is a byte-id comparison against the values BUILD_RECORD_B2.md and the reference JSON
already report -- it is not a from-scratch re-derivation of the jackknife influence vector
from the raw `event_likelihoods.csv` files (that full LOO recompute is Phase B's own job,
already run once by `build_influence_vector.py` and cross-referenced once more by the row
#342 JSON). No registered aggregate (AUC, OR, p-value, Delta_strat) was computed by this
verifier, consistent with the draft's blindness/no-compute constraint on builders and
reviewers.
