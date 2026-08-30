# T1.1 independent verifier report -- theta-consistent no-BH divisor (site 2.3phi)

Launched under row #255 -- tree 2 node T1.1 (verifier pass; verifier is a different agent from
both the presenter and the builder). Branch fix/p32d-classg-venue-repair, HEAD ecd33336 at read
time. Cluster inactive; every check below ran local and foreground. No git operations performed
by this node (no add/commit); the orchestrator commits. No backtick characters in this record.

Reviewed against: PHYSICS_CHANGE_THETA_DIVISOR_20260830.md (sections 0-12, full text read) and
T1_1_DIVISOR_IMPLEMENTATION_RECORD.md (full text read); git diff of darksiren_emri/,
darksiren_emri_test/, and fanout1_20260829/hier_s0_driver.py.

## Verdict table

| # | item | verdict | evidence |
|---|---|---|---|
| 1 | Every hunk inside the presented scope | PASS | Full git diff read hunk-by-hunk (628 diff lines, 15 @@ hunks in bayesian_statistics.py + arguments.py + main.py + correspondence_1d.py). Every hunk maps onto section 2 of the presentation and section 12 of the implementation record: the site_2_3_phi hook-counter entry; nine new optional kwargs on write_selection_table_json (all None-default); the two new module functions _phi_divisor_kernel_pass and precompute_phi_divisor_theta_ratio inserted after precompute_global_catalog_selection; class-level defaults _theta_phi_divisor="off"/_sky_cone_k=1.5 plus their __init__ copies; evaluate() signature gains sky_cone_k and theta_phi_divisor; the validation/guard block (sky_cone_k finite/>0; theta_phi_divisor token + "on" requires catalogue_global_selection=="phi" + normalization_mode=="absolute_marginal"); the site-2.3phi ratio-table build immediately after the existing Sigma_phi call site; the stored self._global_cat_selection_phi_theta attribute; the sky-cone literal 1.5 -> self._sky_cone_k at the single ball-tree call site; and the global_denom_no_bh consumer's getattr fallback. arguments.py/main.py/correspondence_1d.py carry only byte-identical-default plumbing (the mass_filter_k precedent pattern). No hunk touches path_a_mixture_objects, Sigma_4D, the with-BH channel, the generator, or any other physics-trigger file (constants.py, LISA_configuration.py, parameter_estimation.py, cosmological_model.py, simulation_detection_probability.py, physical_relations.py) -- confirmed by the diff's own file list (git status: exactly arguments.py, bayesian_statistics.py, main.py, validation/correspondence_1d.py, docs/gates/PHYSICS-GATE-LEDGER.md, plus the new test file). handler.py and hier_s0_driver.py confirmed UNMODIFIED (git diff --stat empty for both) -- matches the record's own claim that both were read-only for this node. galaxy_catalogue/handler.py's get_possible_hosts_from_ball_tree already accepted sigma_multiplier as a parameter pre-change (commit 0b308828), so no handler.py edit was needed, as claimed. |
| 2 | Byte-identity at the default (pin tests re-run; smoke cell reproduced) | PASS | ruff check --fix darksiren_emri/: all checks passed (matches claim). ruff format --check: 71 files already formatted (matches claim). mypy darksiren_emri/: Success, no issues in 70 source files (matches claim). New file darksiren_emri_test/bayesian_inference/test_theta_phi_divisor.py: 19 passed (matches claim). Combined with test_theta_hook.py + test_smear_global_selection.py + test_catalogue_global_selection.py + test_mass_filter_geometry.py: 85 passed (matches claim). Full pytest -m "not gpu and not slow", split exactly as the record split it: non-validation half 1514 passed / 15 skipped / 25 deselected; validation half 401 passed / 2 deselected; combined 1915 passed / 15 skipped / 27 deselected -- EXACT reproduction of the record's cited totals (baseline 1896 + 19 net-new, zero regressions). Direct call to write_selection_table_json with every new kwarg omitted reproduces byte-for-byte the original 6-key JSON payload (h, beta_G_phi, beta_Gbar_phi, sigma_phi, sigma_4d, r_Malm), confirming the "None omits the key" claim. Live smoke cell reproduced against the REAL GLADE+ catalogue (not a unit fixture): `hier_s0_driver.py --arm S0-A --smoke --seeds 900101 --nodes truth --theta-sites 2.2 --smear off --jobs 1` at defaults (theta_phi_divisor absent from the driver, so "off" applies) -- completed with exit code 0, n_seeds_error=0, n_events=9, wall_s=459.2s (well under the 600s foreground cap once background-waited). The GATE_PARITY diagnostic in that run's own output (ln_L_no_bh max_abs_diff 1.37e-4, ln_L_with_bh max_abs_diff 0.332, both "exact": false) is NOT a regression from this change: theta=(0,1) at the truth node means theta_phi_divisor's literal skip applies regardless of the flag's value (the new dict is never populated), and this residual matches the pre-existing, already-ratified E19 finding (row #251/#255 A2(c): "the forensic's E19 diagnosis of the residual RATIFIED as its disposition") -- it is the driver's own theta-sites-2.2-vs-reference GATE PARITY check, unrelated to the divisor. No exact byte-for-bit bank pin was available to diff against (R3/R5/R11 need a full production-scale run against the banked S0-A CSVs at the SAME event count and node set, which the record correctly scopes as integration-level and defers to T1.2 -- the smoke cell used here (event-cap 12, single node) is not a like-for-like comparand to the banked 106-130-event full-pool S0-A cells, so no diff was attempted against them; this is disclosed, not a gap in this verifier's coverage of T1.1's own presented scope). |
| 3 | The divisor actually changes under theta != (0,1) (engagement test) | PASS | Independent script (fresh RNG seed 12345, an independently-hand-typed decaying phi table exp(-z/0.6) different in form from the builder's linear-decay fixture, and a catalogue/completeness setup written from scratch, not reusing the builder's test helpers) called precompute_phi_divisor_theta_ratio directly: rho((0,1)) == 1.0 exactly (bit-for-bit, GATE T-ID). rho(b=+0.03) = 0.940375 and rho(b=-0.03) = 1.061845 -- both != 1.0 and with the sign the presentation's section 5.3 predicts (S_bar_phi decreasing in z => rho(b>0)<1<rho(b<0)). A from-scratch scipy.integrate.quad reimplementation of the single-host C7-core kernel (independent of _phi_divisor_kernel_pass's own code, using the actual SIGMA_V_PEC_KM_S=0.0 constant read from constants.py, not assumed) matched the module's internal per-row contribution to 5.9e-8 relative error at theta=(0.03,1.0), confirming both that the kernel form is correctly implemented AND that theta engagement is real, not a no-op. |
| 4 | Tests/ruff/mypy counts reproduced | PASS | See item 2: every count cited in the implementation record and in the two new PHYSICS-GATE-LEDGER.md rows (ruff clean, ruff format clean, mypy 70-files clean, 19 new tests, 85-test regression group, 1915/15/27 full-suite total) was independently reproduced exactly, not merely re-quoted. |
| 5 | Gate-ledger rows match the diff | PASS (one cosmetic citation note) | Three new docs/gates/PHYSICS-GATE-LEDGER.md rows (2026-08-30: presented / implemented / verified) correctly cite the touched-file list, the registered formula, the cost anchors, the test counts, and the driver-gap finding, all consistent with the diff and with T1_1_DIVISOR_IMPLEMENTATION_RECORD.md's own section 1/2/3. The "implemented" row's line-citation "bayesian_statistics.py:1625-1631" for the site_2_3_phi hook-counter dict is off by roughly 1-2 lines against the actual current span (the dict literal plus its new entry sits at 1625-1633; the new key itself is at line 1632) -- a trivial citation imprecision, not a scope or content error (must_fix: none, cosmetic only). BIAS_HISTORY_LEDGER.md's new row #256 (tree-2-charter-opening) is correctly NOT claimed by the T1.1 implementation record's file list -- it is a separate, earlier node (tree 2 node 0), not part of this diff's scope. |

## must_fix

None.

## Minor notes (not must_fix)

1. PHYSICS-GATE-LEDGER.md's "implemented" row cites bayesian_statistics.py:1625-1631 for the
   site_2_3_phi hook-counter entry; the actual dict (with the new entry) spans roughly 1625-1633.
   Cosmetic citation drift only.
2. Regression items R3, R5, R11 (byte-for-bit pins against the banked S0-A CSVs and the
   correspondence_1d harness-parity check) remain genuinely unattempted by both the builder and
   this verifier pass -- correctly and explicitly deferred to the T1.2 re-certification per the
   builder != runner separation (row #255 charter). This verifier's smoke cell (event-cap 12,
   truth node only) is not a substitute for those pins and was not represented as one.
3. The presentation's and implementation record's "driver gap" finding was independently
   confirmed by direct inspection: hier_s0_driver.py has exactly three run_mirror_seed_inprocess
   call sites (lines 389, 402, 629 as read), all invoked with keyword arguments only (via
   `**common_kwargs`/explicit kwargs, never positional overflow), so the two new trailing
   keyword parameters with byte-identical defaults are safely inert and the file needed no edit
   at this node -- but this also means the orchestrator's literal proposed T1.2 command in this
   task's own text will NOT engage the fix (theta_phi_divisor stays "off" throughout) unless
   hier_s0_driver.py is first given a --theta_phi_divisor passthrough, exactly as both source
   documents state.

## Independent instruments used by this verifier (not present before this pass)

- /tmp scratch script (fresh RNG, independent phi table and catalogue fixture, from-scratch
  scipy.integrate.quad cross-check) for item 3 -- not committed, not part of the repo.
- Live smoke cell at
  results/campaign51_20260728/realistic_20260729/tree2_20260830/t1_1_verifier_work/smoke_run/
  (S0-A, seed 900101, node truth, theta-sites 2.2, smear off, event-cap 12) for item 2's
  production-pipeline sanity check.

Launched under row #255 -- tree 2 node T1.1 (independent verifier pass).
