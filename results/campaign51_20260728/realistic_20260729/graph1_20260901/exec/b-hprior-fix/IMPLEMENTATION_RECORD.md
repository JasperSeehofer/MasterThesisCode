# b-hprior-fix — IMPLEMENTATION RECORD (Research Graph 1 item 6)

Date: 2026-09-02. Authorization: author's "both approved" ruling (row #308) against
`graph1_20260901/exec/b-hprior-fix/DECOUPLING_DESIGN.md`. Implementer node: this session,
effort medium, zero fresh design choices — the design's exact minimal diff, applied verbatim.
Pre-implementation check: the working tree's `cosmological_model.py:386-393` and
`bayesian_statistics.py:4656-4660` matched the design's quoted "OLD code" byte-for-byte
(only line numbers had drifted since the design was authored: guard now at 4656-4660,
`get_redshift_outer_bounds` call at 5716 pre-edit / 5733 post-edit due to the added comment
lines above it in the same function) — no conflict, proceeded.

## 1. Diff applied

```diff
diff --git a/darksiren_emri/cosmological_model.py b/darksiren_emri/cosmological_model.py
index 35602485..d5156340 100644
--- a/darksiren_emri/cosmological_model.py
+++ b/darksiren_emri/cosmological_model.py
@@ -379,18 +379,32 @@ class LamCDMScenario:

     h: CosmologicalParameter
     Omega_m: CosmologicalParameter
+    # Grid-admissibility ceiling for evaluate()'s entry guard ONLY (G-EXT wing,
+    # AMENDMENT G-EXT row #284; decoupling ratified row #301 item 4(a)).  This is
+    # NOT the host-window bound: get_redshift_outer_bounds(h_max=...) reads
+    # h.upper_limit below, which stays 0.86 so every detection's candidate-host
+    # z-window is unchanged (byte-identity below 0.86 by construction; see
+    # graph1_20260901/exec/b-hprior-fix/DECOUPLING_DESIGN.md).
+    h_grid_admissibility_max: float
     w_0: float = -1.0
     w_a: float = 0.0

     def __init__(self) -> None:
         self.h = CosmologicalParameter(
             symbol="h",
+            # Prior-support/admissibility decoupling — DECOUPLING_DESIGN.md (graph1), rows #293/#301/#304/#308; G-EXT wing rows #284/#286.
+            # HOST-WINDOW / prior-support bound — deliberately NOT raised for the
+            # G-EXT wing (row #293: raising it widens every candidate-host z-window
+            # and breaks byte-identity below 0.86). The wing is admitted via
+            # h_grid_admissibility_max instead.
             upper_limit=0.86,
             lower_limit=0.6,
             unit="s*Mpc/km",
             randomize_by_distribution=uniform,
             fiducial_value=0.73,
         )
+        # Prior-support/admissibility decoupling — DECOUPLING_DESIGN.md (graph1), rows #293/#301/#304/#308; G-EXT wing rows #284/#286.
+        self.h_grid_admissibility_max = 1.00
         self.Omega_m = CosmologicalParameter(
             symbol="Omega_m",
             upper_limit=0.5,
diff --git a/darksiren_emri/bayesian_inference/bayesian_statistics.py b/darksiren_emri/bayesian_inference/bayesian_statistics.py
index ca442ec6..30412727 100644
--- a/darksiren_emri/bayesian_inference/bayesian_statistics.py
+++ b/darksiren_emri/bayesian_inference/bayesian_statistics.py
@@ -4653,9 +4653,26 @@ class BayesianStatistics:
         _LOGGER.info(
             f"Computing posteriors for h = {_h_list[0] if len(_h_list) == 1 else _h_list}..."
         )
+        # Admissibility guard ONLY (row #301 item 4(a) decoupling): the ceiling is
+        # max(host-window bound, grid-admissibility ceiling) so (i) the ratified
+        # G-EXT wing (h <= 1.00) is admissible, (ii) the mirror harness's runtime
+        # widening of h.upper_limit (correspondence_1d.py:3398-3399, [P3-HGRID])
+        # keeps working, and (iii) setting h_grid_admissibility_max ==
+        # h.upper_limit reproduces the old guard exactly. The host-window call
+        # site (get_redshift_outer_bounds(h_max=h.upper_limit), :5716) is
+        # deliberately NOT changed — see DECOUPLING_DESIGN.md.
+        # Prior-support/admissibility decoupling — DECOUPLING_DESIGN.md (graph1), rows #293/#301/#304/#308; G-EXT wing rows #284/#286.
+        _h_admissible_max = max(
+            self.cosmological_model.h.upper_limit,
+            getattr(
+                self.cosmological_model,
+                "h_grid_admissibility_max",
+                self.cosmological_model.h.upper_limit,
+            ),
+        )
         for _h_check in _h_list:
             if (_h_check < self.cosmological_model.h.lower_limit) or (
-                _h_check > self.cosmological_model.h.upper_limit
+                _h_check > _h_admissible_max
             ):
                 raise ValueError("Hubble constant out of bounds.")
```

Confirmed byte-identical, unchanged: `bayesian_statistics.py` host-window call
`h_max=self.cosmological_model.h.upper_limit` (post-edit line 5733; pre-edit 5716 —
line drift only, from the guard comment block added above it in the same function) and
`upper_limit=0.86` (post-edit `cosmological_model.py:400`). `constants.py` and
`physical_relations.py:get_redshift_outer_bounds`'s own `h_max=0.86` default untouched.
No other lines in either file changed.

## 2. Regression tests

New file `darksiren_emri_test/test_h_bound_decoupling.py` (10 tests). Written and run
**before** the diff was applied, against the unmodified tree, to make the diff visible:

Pre-diff run (`uv run pytest darksiren_emri_test/test_h_bound_decoupling.py -v`):
```
test_lamcdm_h_upper_limit_is_0_86                          PASSED
test_lamcdm_h_lower_limit_is_0_6                            PASSED
test_get_redshift_outer_bounds_default_h_max_is_0_86        PASSED
test_lamcdm_h_grid_admissibility_max_is_1_00                FAILED (AttributeError — attribute doesn't exist yet, as expected)
test_guard_admits_wing_up_to_1_00                            FAILED (assert not True — h=0.87 rejected under old guard, as expected)
test_guard_still_rejects_above_1_01                         PASSED
test_guard_still_rejects_below_lower_limit                  PASSED
test_guard_degenerate_ceiling_equals_old_behavior           PASSED
test_guard_absent_attribute_falls_back_to_upper_limit       PASSED
test_guard_mirror_widening_of_upper_limit_still_admits      PASSED
=========== 2 failed, 8 passed in 3.86s ===========
```
This confirms: (a) `LamCDMScenario().h.upper_limit == 0.86` and `.lower_limit == 0.6`
(old behavior, task step 1a) — PASS pre-diff; (b) the guard rejects h=0.90-class wing
values today (mirrored directly via `_guard_rejects`, since a full `evaluate()` call is
too heavy to construct at unit level; the `getattr` fallback in the mirror already
reproduces the exact old two-line comparison when the new attribute is absent) — PASS
pre-diff (`test_guard_still_rejects_above_1_01`, and `test_guard_admits_wing_up_to_1_00`
failing pre-diff is itself the same evidence in reverse); (c)
`get_redshift_outer_bounds`'s own `h_max` signature default is 0.86 — PASS pre-diff and
untouched by this design.

Post-diff run — same file, all 10 pass (see §3).

## 3. Full verification run (post-diff)

`uv run pytest darksiren_emri_test/test_h_bound_decoupling.py -v --no-cov`:
```
test_lamcdm_h_upper_limit_is_0_86                          PASSED
test_lamcdm_h_lower_limit_is_0_6                            PASSED
test_get_redshift_outer_bounds_default_h_max_is_0_86        PASSED
test_lamcdm_h_grid_admissibility_max_is_1_00                PASSED
test_guard_admits_wing_up_to_1_00                            PASSED
test_guard_still_rejects_above_1_01                         PASSED
test_guard_still_rejects_below_lower_limit                  PASSED
test_guard_degenerate_ceiling_equals_old_behavior           PASSED
test_guard_absent_attribute_falls_back_to_upper_limit       PASSED
test_guard_mirror_widening_of_upper_limit_still_admits      PASSED
======================= 10 passed in 2.06s =======================
```

`uv run pytest -m "not gpu and not slow"` (full fast suite, run twice — see note):
```
=== 2026 passed, 15 skipped, 30 deselected, 12 warnings in 209.95s (0:03:29) ===
[exited with code 0]
Required test coverage of 25.0% reached. Total coverage: 73.40%
```
(A second identical invocation was launched by mistake in parallel with the first while
waiting on the background task notification; only the first completed run's output — shown
above — is reported. Both used the same post-diff working tree.)

`uv run ruff check darksiren_emri/ darksiren_emri_test/test_h_bound_decoupling.py`:
```
All checks passed!
```

`uv run mypy darksiren_emri/ darksiren_emri_test/`:
```
Success: no issues found in 220 source files
```
(One intermediate mypy failure was fixed before this final clean run: the test file's
`_guard_admissible_max`/`_guard_rejects` helpers were originally typed to accept
`LamCDMScenario`, but the duck-typed-fallback test (`_BareScenario`, limiting case 3 in
the design §6) doesn't satisfy that type. Re-typed both helpers to `Any` — this is test-only
scaffolding mirroring the guard's `getattr`-based duck-typing tolerance, not a production
change.)

## 4. Ledger rows appended

`docs/gates/PHYSICS-GATE-LEDGER.md` (two new rows, newest last, matching the existing
`| YYYY-MM-DD | <commit-ref> | <step> | <verdict> | <target> | <note> |` column contract):

```
| 2026-09-02 | pre-commit | presented | APPROVED | cosmological_model.py:388 | h admissibility decoupled from host-window bound: h_grid_admissibility_max=1.00, guard max(); design DECOUPLING_DESIGN.md, author word row #308 |
| 2026-09-02 | pre-commit | implemented | PASS | cosmological_model.py:388+bayesian_statistics.py:4655 | ref comment + regression tests (old-behavior tests written first) |
```

## 5. What was NOT done (per instructions)

No commit was created ([PHYSICS] commit is the chair's). No cluster submission. No
byte-identity evidence run against the banked H_GRID_41 nodes (job 6747032) — that rerun
is separately orchestrated per DECOUPLING_DESIGN.md §8, blocked on the 4b CPU-h cap word.
