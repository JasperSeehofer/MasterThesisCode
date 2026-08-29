# Node B5.1 [WIN] part (B) — IMPLEMENT record

*launched under rows #222/#223 — charter node B5.1*

Status: **IMPLEMENTED, not committed.** The orchestrator commits. No `git commit`/`add`/
`reset`/`checkout`/`stash` was run by this node.

## 1. What this node did

Implemented the mass-window GEOMETRY instrument flag presented in
`PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md` §2 (row #221 F-ii pre-code half; ledger
rows #220-#223): two new independent flags on
`GalaxyCatalogueHandler.get_possible_hosts_from_ball_tree` —
`mass_filter_geometry: Literal["linear", "log"] = "linear"` and
`mass_filter_k: float = 1.5` — threaded end-to-end through `BayesianStatistics`,
`correspondence_1d.run_mirror_seed_inprocess`, `arguments.py`, and `main.py`. Both default to
the byte-identical pre-flag path. Per STANDING RULE 2 (verifier independence), this node BUILT
the instrument and ran its own regression suite (§4 below), consistent with a builder role;
node B5.2 (wave 2) is the separate, later gate that runs the registered fleet-level
counterfactual (§6).

This node started from B6's uncommitted working-tree state (θ-hook s-placement alignment in
`bayesian_statistics.py`/`correspondence_1d.py`/`test_theta_hook.py`) and did not touch any of
B6's lines — confirmed by `git diff` before and after (B6's hunks are the θ-hook sites
2.1/2.2/2.3 plus the s-placement docstrings; this node's hunks are the mass-filter class
defaults/`__init__`/`evaluate()` signature/read+call sites, disjoint line ranges, verified by
inspection of the post-edit diff).

## 2. Code-vs-document match (verified before editing)

Confirmed by direct read at HEAD `a794404c` that the OLD-formula code blocks quoted in the
presentation's §1 were, modulo the already-disclosed line-number drift (Note R3: docstring at
598-609, not 648-660; mask at 654-673, confirmed correct), byte-identical to the code actually
edited. The presentation's §2 code sketch ("illustrative; not yet written") was implemented
literally for the "linear"/"log" branch bodies themselves.

## 3. The one design gap in §2, and how it was resolved (disclosed)

§2's illustrative code shows the log-geometry test using a bare `M_err` (`BH_MASS_ERROR`)
directly, with no reference to the pre-existing `mass_filter_sigma` flag or its
`_bh_mass_error_multiplier` split ("symmetric" = scale the candidate error by the same k as
the GW side; "asymmetric" = pin it at its bare ×1 value). This is a genuine gap, not a typo:
§2 calls the two flags "independent," but invariant 1 (default byte-identity, §5/§6 item 1/R7
invariant 1) REQUIRES `mass_filter_sigma`'s existing behaviour to keep working exactly as
today under the new default pairing (`geometry="linear"`, `k=1.5`) — which is impossible if
the candidate-side multiplier logic is simply deleted, as §2's sketch would imply if taken
literally for the "linear" branch too.

**Resolution taken (disclosed):** the candidate-side multiplier that `mass_filter_sigma`
selects is now `mass_filter_k` (was `sigma_multiplier`) for `"symmetric"`, and `1.0` for
`"asymmetric"` — under EITHER geometry. The two flags are read independently, each at its own
single site (`mass_filter_geometry`/`mass_filter_k` validated first; `mass_filter_sigma`
validated immediately after, unchanged in its own three-way structure). This:

- Preserves default byte-identity exactly, since `mass_filter_k` literal-defaults to `1.5`,
  the same value the call site already hardcodes for `sigma_multiplier` — proven both by unit
  test and by an independent 100,000-pair script (§5).
- Gives `"log"` a well-defined, non-arbitrary `mass_filter_sigma` semantics (bare `σ_lnM` vs.
  `k·σ_lnM` in the exponent) instead of silently ignoring the flag under that geometry, which
  would have been a worse and less-disclosed gap than the one being resolved.
- Is covered by a dedicated discriminator test
  (`TestMassFilterGeometryLogHandComputedEdges.test_asymmetric_uses_bare_candidate_multiplier_in_log_geometry`)
  proving the two multiplier conventions are actually distinguishable in production code, not
  just asserted.

This mirrors B6's own precedent on this branch (disclose a presentation-vs-implementation gap,
resolve conservatively toward the invariant the presentation itself demands, and record the
reasoning) rather than silently picking an interpretation.

## 4. Diff summary (line numbers as of this node's edit, HEAD `a794404c` + B6's uncommitted work)

| File | What changed | Lines (post-edit) |
|---|---|---|
| `darksiren_emri/galaxy_catalogue/handler.py` | `get_possible_hosts_from_ball_tree` signature (+2 params), docstring (+2 Args blocks), mask branch (geometry validation + linear/log dispatch) | signature 558-573; docstring additions ~613-641; mask logic 679-745 |
| `darksiren_emri/bayesian_inference/bayesian_statistics.py` | class-level defaults, `__init__` defaults, `evaluate()` signature (+2 params), read/assign site, call site (+2 kwargs) | class defaults 3309-3310; `__init__` 3377-3378; `evaluate()` signature 3541,3548; read site 3767-3768; call site 4843-4844 |
| `darksiren_emri/validation/correspondence_1d.py` | `run_mirror_seed_inprocess` signature (+2 params), passthrough call (+2 kwargs) | signature 2773-2774; passthrough 2951-2952 |
| `darksiren_emri/arguments.py` | 2 new `@property` accessors, 2 new `argparse.add_argument` blocks | properties 394-412; argparse 1079-1105 |
| `darksiren_emri/main.py` | `main()` call site (+2 kwargs), module `evaluate()` signature (+2 params) and its internal call (+2 kwargs) | call site 214-215; signature 1428-1429; internal call 1464-1465 |
| `darksiren_emri_test/test_mass_filter_geometry.py` (new) | 19-test regression module | 450 lines |
| `darksiren_emri_test/validation/test_correspondence_1d.py` | 1 new signature-passthrough test | +14 lines |

`git diff --stat` for the 5 touched source files + 1 touched test file (the new test module
is untracked, not counted here): **277 insertions(+), 28 deletions(-)** across 6 files. Note
the `bayesian_statistics.py` and `correspondence_1d.py` diffs also carry B6's concurrent,
disjoint, uncommitted θ-hook edits — the line ranges in the table above are this node's own
hunks only.

**Exact file list to commit for this node** (orchestrator's commit, not this node's):
- `darksiren_emri/galaxy_catalogue/handler.py`
- `darksiren_emri/bayesian_inference/bayesian_statistics.py` (shared with B6 — same file,
  disjoint hunks; commit once, contains both nodes' work)
- `darksiren_emri/validation/correspondence_1d.py` (shared with B6 — same file, disjoint
  hunks)
- `darksiren_emri/arguments.py`
- `darksiren_emri/main.py`
- `darksiren_emri_test/test_mass_filter_geometry.py` (new file)
- `darksiren_emri_test/validation/test_correspondence_1d.py` (shared with B6's own test edits
  earlier in the file — this node adds one function, disjoint)
- `docs/gates/PHYSICS-GATE-LEDGER.md` (shared with B6 — append-only, disjoint rows)
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md`
  (this node's own "Implementation record" section appended)
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/B5_1_WIN_RECORD.md`
  (this file, new)

## 5. Byte-identity evidence

1. **Unit tests** (`TestMassFilterGeometryDefaultByteIdentity`, 2 cases): the new-flag call
   with explicit defaults (`mass_filter_geometry="linear"`, `mass_filter_k=1.5`) reproduces
   the exact candidate set of the pre-flag-aware call, for both `mass_filter_sigma` cells, on
   a 3-galaxy synthetic catalogue (inside/boundary/outside). A second test confirms
   `sigma_multiplier` no longer moves the mass-window membership at all (1.5 vs 5.0 give
   identical results once `mass_filter_k` is held fixed) — directly exercising the "decouple
   from `sigma_multiplier`" half of the design.
2. **Independent 100,000-pair script** (`b51_byte_identity_check.py`, run in scratchpad, not
   committed): 50 random GW events (`M_z`, `M_z_sigma` drawn `U(1e5,1e7)` /
   `U(0.02,0.5)·M_z`) × 2,000 random synthetic candidates each, comparing the NEW production
   function called with `mass_filter_geometry`/`mass_filter_k` **omitted** (pure defaults)
   against an independently re-typed copy of the pre-B5.1 formula (not calling into
   `handler.py` at all — a literal transcription of the old `654-673` block). Result:
   **0 mismatches across all 50 events / 100,000 candidate-event pairs.**
3. **Full regression suite**: 1871 passed / 15 skipped / 27 deselected (baseline before this
   node's tests, per B6's own reported run: 1851 passed; +20 new tests = 1871, exact match).
   `ruff check`, `ruff format --check`, and `mypy` all clean on the full
   `darksiren_emri/` + `darksiren_emri_test/` tree, not just the touched files.

Falsifier R4 item 2 (re-deriving the fleet-level true-host retention drop by calling the
NOW-existing production flags directly on the real 24-arm fleet, to check whether
`b5_window_count.py`'s REIMPLEMENTED mass-window logic diverges from production by more than
±2 points) is **explicitly not attempted here** — it is a fleet-scale run and is charter node
B5.2 / wave 2's registered counterfactual, not this implementation task's smoke-test scope.

## 6. Test counts

| Suite | Result |
|---|---|
| `test_mass_filter_geometry.py` (new) | 19 passed |
| `test_mass_filter_sigma.py` (pre-existing, unaffected) | 5 passed |
| `test_handler_catalog_io.py` | 5 passed |
| `test_catalogue_global_selection.py` | 21 passed |
| `test_coordinate_roundtrip.py` | 9 passed |
| `test_theta_hook.py` (B6's, unaffected by this node) | 20 passed |
| `test_correspondence_1d.py` (incl. this node's 1 new signature test) | 69 passed |
| **Targeted subtotal above** | **148 passed** |
| **Full suite** `pytest -m "not gpu and not slow" -q -x -p no:cacheprovider` | **1871 passed, 15 skipped, 27 deselected** (152.4s) |

`ruff check --fix`, `ruff format`, and `mypy` all clean (both on the touched-files subset and
on the full `darksiren_emri/ darksiren_emri_test/` tree — 212 source files, 0 mypy issues, 0
ruff findings).

## 7. Count-factor table (attributed — computed by the presentation's zero-compute-read
instrument, part (A) of this node, NOT re-derived here)

Reproduced verbatim from `PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md` §7 (post-fix,
Revision Note R2) for the orchestrator's convenience; source = `b5_window_count.json`,
`bc_9001XX_work` × 24 arms, `n_all = 2,249,231` candidate rows over 2,261 events, regenerated
2026-08-29 by a different agent from this node:

| config | geometry | k | pass fraction (n_pass/n_all) | true-host retention |
|---|---|---|---|---|
| (i) | linear | 1.5 | 0.95768 (gate target 0.9577, PASSED 4dp) | 0.9567 |
| (ii) | log | 1.5 | 0.40613 | 0.7001 |
| (iii) | log | 3.0 | 0.69509 | **0.7890** |
| (iv) | log | 2.5 | 0.61489 | 0.7682 |

Per-event candidate growth, (iii) log k=3 vs (i) linear k=1.5 (2,221 events with >=1
linear-passing candidate): **mean 0.814, median 0.949, p95 1.498, max 10.0** (16 events gain
candidates linear admitted zero of; 24 empty under both). Aggregate ratio 0.695/0.958 = 0.726
— markedly below the median per-event ratio (right-skew: most events see a small net loss or
wash; a minority see up to 10x growth). Arm-to-arm (24-arm jackknife) spread: pass fraction
(iii) 0.6971 ± 0.1127 (SE 0.0230); retention (iii) 0.7898 ± 0.0455 (SE 0.0093) — both headline
drops from (i) clear >=3 arm-SEs, i.e. distinguishable from this fleet's own seed-to-seed
noise floor (source: `b5_window_count_arm_jackknife.json`).

## 8. Wave-2 counterfactual arm shape (proposed, NOT launched by this node)

One venue at HEAD (this implementation), engaging `mass_filter_geometry="log"`,
`mass_filter_k=3.0` (the ratified candidate design, row #221 F-ii) against the current
production default (`"linear"`, `k=1.5`) as the control arm — the registered counterfactual
§7's own text flags as still open (net sign of the ΔH0/ΔMAP effect from the 17-point true-host
retention loss vs. the candidate-set composition change is UNDETERMINED by the zero-compute
read alone).

- **Primary readouts**: candidate-count growth per event (already measured, §7 above, to be
  reproduced end-to-end through the NOW-existing production flags rather than the
  `b5_window_count.py` replica — closing R4 falsifier item 2 as a side effect); ΔMAP vs. the
  banked HB `+0.0015` ceiling (`CLAIM_WGEO_20260827.md` §4.1) — the comparison §7 itself
  states must be made explicitly against this specific host-loss mode, not assumed bounded by
  it, since a 17-point true-host loss rate is a qualitatively different quantity from a
  candidate-count change.
- **Cost class**: 50-130 CPU-h, **scaled by the measured candidate factor** (§7): the
  *aggregate* pass-fraction ratio (0.726) and *median* per-event growth ratio (0.949) both sit
  near unity or below, suggesting the scaled range stays close to baseline (~35-125 CPU-h) for
  most of the fleet; but the *max* per-event growth (10.0x) and the *p95* (1.498x) mean a
  nontrivial tail of events (the 16 that gain candidates from zero) will individually cost up
  to an order of magnitude more per-event mass-quadrature evaluation than their linear-k1.5
  counterpart, which is a wall-clock/task-imbalance risk for a fixed-array-size cluster
  submission even though the AGGREGATE compute is not expected to exceed the stated 50-130
  CPU-h class by more than the ~1.5x p95 factor. Recommend sizing the SLURM array's per-task
  time limit off the p95 (not mean) growth factor, and flagging the 16 zero-to-nonzero events
  for a first (small, k=3, log) smoke run before committing the full 24-arm counterfactual.
