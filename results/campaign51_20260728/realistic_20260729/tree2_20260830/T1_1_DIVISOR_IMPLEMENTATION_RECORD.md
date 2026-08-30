# T1.1 implementation record -- theta-consistent no-BH divisor (site 2.3phi)

Launched under row #255 -- tree 2 node T1.1. Builder pass (a different agent from the T1.1
presenter, per the presentation's section 0 requirement). No git operations performed by this
node; the orchestrator commits. Branch fix/p32d-classg-venue-repair. No backtick characters in
this record.

Implements exactly the presentation at
results/campaign51_20260728/realistic_20260729/tree2_20260830/PHYSICS_CHANGE_THETA_DIVISOR_20260830.md
(panel-clean after 0 rounds). Full narrative of what was built is appended as section 12 of that
same file (12.1-12.5). This record exists so the orchestrator has, in one place: the exact file
list to commit, and the exact T1.2 command together with the decisive driver-gap finding.

---

## 1. Exact file list to commit

Modified (source):
- darksiren_emri/bayesian_inference/bayesian_statistics.py
- darksiren_emri/arguments.py
- darksiren_emri/main.py
- darksiren_emri/validation/correspondence_1d.py

New (test):
- darksiren_emri_test/bayesian_inference/test_theta_phi_divisor.py

Modified (docs, append-only):
- docs/gates/PHYSICS-GATE-LEDGER.md (two new rows: implemented, verified; both dated
  2026-08-30, commit ref pre-commit)
- results/campaign51_20260728/realistic_20260729/tree2_20260830/PHYSICS_CHANGE_THETA_DIVISOR_20260830.md
  (new section 12, implementation record narrative)

New (this file):
- results/campaign51_20260728/realistic_20260729/tree2_20260830/T1_1_DIVISOR_IMPLEMENTATION_RECORD.md

NOT touched (confirmed by design and by grep, disclosed per the presentation's own text):
- darksiren_emri/galaxy_catalogue/handler.py (get_possible_hosts_from_ball_tree already accepts
  sigma_multiplier as a parameter since commit 0b308828; no signature change needed)
- results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py (owned by the
  T1.2 builder per regression item R13; see section 3 below)
- Every other physics-trigger file (constants.py, LISA_configuration.py,
  parameter_estimation.py, cosmological_model.py, simulation_detection_probability.py,
  physical_relations.py)

Suggested commit message prefix: [PHYSICS] (per the physics-validation convention and the
presentation's section 7 reference-comment instruction).

---

## 2. Quality gate results (evidence for the ledger rows)

- ruff check --fix darksiren_emri/: all checks passed (no findings to fix)
- ruff format darksiren_emri/: clean (70 files unchanged after the earlier pass; the touched
  files were reformatted once during development and are clean now)
- mypy darksiren_emri/: Success, no issues found in 70 source files
- darksiren_emri_test/bayesian_inference/test_theta_phi_divisor.py: 19 passed
- darksiren_emri_test/bayesian_inference/test_theta_hook.py +
  darksiren_emri_test/test_smear_global_selection.py +
  darksiren_emri_test/test_catalogue_global_selection.py +
  darksiren_emri_test/test_mass_filter_geometry.py + the new file: 85 passed, 0 failed
- Full suite, split in two halves for the 600 s foreground limit:
  - darksiren_emri_test/ minus darksiren_emri_test/validation: 1514 passed, 15 skipped,
    25 deselected
  - darksiren_emri_test/validation: 401 passed, 2 deselected
  - Combined: 1915 passed / 15 skipped / 27 deselected (baseline of record, per the row-#223
    ledger entry, was 1896 passed / 15 skipped / 27 deselected -- the delta is exactly the 19
    new tests, zero regressions)

One regression was caught and fixed during this pass (disclosed in section 12.4 of the
presentation file): the global_denom_no_bh consumer needed a getattr fallback
(getattr(self, "_global_cat_selection_phi_theta", {})) because several pre-existing tests in
test_catalogue_global_selection.py construct BayesianStatistics via object.__new__ and call p_Di
directly, bypassing __init__ entirely.

Deferred, disclosed (not attempted by this node): regression items R3, R5, R11 (bit-for-bit pins
against the banked S0-A CSVs at
results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run/, and the
correspondence_1d.kernel_smeared_survival harness-parity check) require a full evaluate() run
against the real GLADE catalogue -- integration-level cost, not a fast unit test. R3/R11 in
particular ARE substantially what the T1.2 re-certification itself measures, so deferring them
there rather than duplicating the cost here is a disclosed scope decision, not an oversight.

---

## 3. THE decisive finding: does --theta-sites 2.2 engage the divisor?

**No.** By registered design (presentation section 2.2), theta_phi_divisor is an INDEPENDENT flag
from theta_sites. theta_sites governs which of the PRE-EXISTING sites (2.1 the scalar numerator
twin, 2.2 the batched numerator, 2.3 the smeared global-selection denominator) receive the theta
reparametrization; the new site 2.3phi (this node's divisor) is a wholly separate code path with
its own switch, and engages only when theta_phi_divisor="on" AND theta is not the identity. It
composes with theta_sites="2.2" (both then read the same self._theta_b/self._theta_s), which is
exactly the registered CoR-P/CoR-M-faithful form of record -- but theta_sites="2.2" ALONE, with
theta_phi_divisor left at its default "off", changes nothing at the divisor.

**No new theta_sites site label is needed, and none was added.** theta_sites stays exactly
{"all", "2.1", "2.2", "2.3"}, unchanged. What is actually missing is a driver-level CLI surface
for the two NEW, independent flags (theta_phi_divisor, sky_cone_k) -- confirmed absent by reading
results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py (grep only, per
this node's authorization): no --theta_phi_divisor or --sky_cone_k argument exists, and none of
the three run_mirror_seed_inprocess call sites forward either name.

### Consequence for the orchestrator's proposed T1.2 command

The literal command as specified in this node's task text:

results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py --arm S0-A
--theta-sites 2.2 --smear off --jobs 1
--out-root results/campaign51_20260728/realistic_20260729/tree2_20260830/hier_s0_recert_run

(4 seeds and 5 nodes are the driver's OWN defaults at --arm S0-A, so they need not be spelled out
explicitly -- but doing so explicitly is harmless and recommended for the record: --seeds
900101,900102,900103,900104 --nodes truth,b_plus,b_minus,s_plus,s_minus)

**will run with theta_phi_divisor defaulting to off throughout.** It will therefore reproduce the
ORIGINAL S0-A INSTRUMENT-DEFECT result byte-for-byte (score_b approx -1.616 +/- 0.440, Z_b approx
-3.68, per the 2026-08-29 record cited in section 0 of the presentation), not the registered
prediction (score_b = -0.27 +/- 0.43, Z_b = -0.62). Running this command as-is and reading its
|Z_b| > 3 outcome as a falsification of mechanism (i) (per A14 falsifier F1) would be WRONG: the
fix was never armed, so F1's REFUTES branch is not actually triggered -- the run would be
uninformative, not decisive.

### What must happen before T1.2 can run as intended

hier_s0_driver.py needs a --theta_phi_divisor {off,on} argument (default off, byte-identical)
threaded to its three run_mirror_seed_inprocess call sites, following the exact same pattern
already used for --theta-sites and --smear (argparse choices tuple, forwarded verbatim, default
reproduces the pre-existing dispatch exactly). This is squarely regression item R13's own text:
"hier_s0_driver.py pass-through is the T1.2 builder's job (driver owned outside this gate;
non-physics file)" -- so it is correctly NOT done by this T1.1 node, but it is flagged here loudly
so the orchestrator does not discover it only after a wasted local run. sky_cone_k does not need a
driver flag for F1 (it is left at its byte-identical default 1.5 throughout the S0-A
re-certification, matching section 9's F1 specification exactly); a --sky_cone_k passthrough would
only be needed for a future F2 enlarged-ball arm, which is out of scope for T1.2 as specified.

### The corrected command, once the driver flag exists

results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py --arm S0-A
--theta-sites 2.2 --smear off --theta_phi_divisor on --jobs 1
--out-root results/campaign51_20260728/realistic_20260729/tree2_20260830/hier_s0_recert_run

with --seeds 900101,900102,900103,900104 and --nodes truth,b_plus,b_minus,s_plus,s_minus left at
their S0-A defaults (spelling them out explicitly in the invocation is fine and recommended). The
truth node stays at the literal-skip identity regardless of --theta_phi_divisor (GATE T-ID), so
--theta_phi_divisor on is safe to pass unconditionally across all 5 nodes and all 4 seeds -- it is
a no-op exactly at the truth node and engages the fix at the four off-truth (b_plus, b_minus,
s_plus, s_minus) nodes, matching A14 falsifier F1's own scope (the b-axis nodes) and disclosing
that the s-axis nodes are PREDICTED to remain outside the band by the separate, un-addressed
truncation mechanism (ii) (section 9, F2) -- not a falsifier of this change.

---

## 4. What this record does NOT license

No git operations (the orchestrator commits). No S0-A re-certification run (T1.2 is a different
agent's job; builder != runner, row #255 charter). No hier_s0_driver.py edit (T1.2 builder's job,
flagged in section 3 above). No z_window_k implementation (a path choice the presentation left
open to the orchestrator, beyond this node's authorized scope). No change to Sigma_4D, path-A
objects, the with-BH channel, or the generator. Cluster inactive this node; every check above ran
local and foreground, each well under the 600 s limit.
