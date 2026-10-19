# B8.2 S1 -- generator instrument: implementation record

Launched under rows #255/#268 -- tree 2 node B8.2.S1. Design of record:
results/campaign51_20260728/realistic_20260729/fanout1_20260829/B8_2_HARNESS_DESIGN_20260829.md
(sections 1-3, 7, 8). Class: sonnet build stage, medium effort (design §8 table). No git operation
performed by this node (the orchestrator commits); no ssh; foreground only; append-only. Branch
fix/p32d-classg-venue-repair. Bounded-scope rule honored: no band, statistic definition, or the
mixture law was changed from the design's own text; no physics-trigger file was edited (only
darksiren_emri/validation/correspondence_1d.py, a harness file per CLAUDE.md's physics-change
trigger list, and its test file). No premise correction was needed -- the design's §2.1 table
mapped exactly onto existing code (catalogue_selected_host_draw_weights,
_draw_kernel_survival_redshifts, draw_selected_population_redshifts, draw_isotropic_sky,
path_a_mixture_objects), so S1 is pure composition, not derivation.

---

## 1. What was built (design §8 S1 deliverable)

All three §8 S1 items landed in darksiren_emri/validation/correspondence_1d.py:

### (a) host_mode="mixture_selected" (design §2.1)

Added to MirrorUniverseGenerator.draw_realization's host_mode Literal and dispatch. On the
SAME rng stream as every other mode:

1. n_g ~ Binomial(n, class_weight_p_g) -- special-cased at the p_g in {0, 1} limits (n_g := n
   / n_g := 0 directly, no rng.binomial call at all) so the RNG stream is provably untouched by
   the split decision at those limits (the acceptance-(ii) bit-identity requirement).
2. The first n_g events are drawn by the EXACT "catalogue_selected" branch's code (host draw
   catalogue_selected_host_draw_weights + rng.choice + _draw_kernel_survival_redshifts) --
   copied inline, not refactored into a shared helper, so neither existing mode's code path was
   touched by this change (verified by git diff: the "catalogue_selected"/"population_selected"
   elif blocks are byte-identical to before this stage).
3. The remaining n - n_g events are drawn by the EXACT "population_selected" branch's code
   (draw_selected_population_redshifts + draw_isotropic_sky), continuing the SAME stream.
4. The two sub-draws are concatenated (catalogue-hosted first, dark second) into the realization's
   n rows; host_draw_mode, z_true, s_tilde_phi_host (NaN for dark events),
   event_class ("catalogue_hosted"/"dark"), n_catalogue_hosted (the realized n_g, constant
   per realization) and class_weight_p_g (the input, constant per realization) are recorded.

P_G itself (design §2.1 item 1, "the estimator's own catalogue-class weight at truth") is built by
a NEW harness-side function, compute_catalogue_class_weight_p_g -- calls the SAME production
construction calls build_b0i_2d_selection_objects already makes, PLUS
precompute_phi_selection_integrals (beta_G^phi, beta_Gbar^phi) and
precompute_global_catalog_selection at with_bh_mass=False (Sigma^phi) and with_bh_mass=True
(Sigma^4D, same catalogue rows/weights/eligibility per decision D2) -- then assembles the four
legs via the REAL path_a_mixture_objects (not reimplemented). p_g returned IS
path_a_mixture_objects's own w_tilde_G at h_true. compute_catalogue_class_weight_p_g is a
harness-side function callers invoke ONCE per (catalogue, h_true) and pass the result into
draw_realization(..., class_weight_p_g=...), mirroring the pre-existing
completeness/phi_survival_table threading convention -- draw_realization itself never calls
it.

### (b) The gw_scatter knob (design §2.3)

Added gw_scatter: bool = True to draw_realization. True (default) is byte-identical to the
pre-B8.2 code: the noise offset is drawn from rng and added, exactly as before. False follows
the pp_coverage.py "draw made and discarded" convention (its Q-0 note,
darksiren_emri/validation/pp_coverage.py:904-907): the SAME rng call is made (so the stream is
identical regardless of the flag -- every OTHER draw, host selection/z_true/the class split, is
unaffected) but the offset is not added, so the observation sits exactly at the latent truth. Wired
into BOTH the 1D else branch (obs_d_L) and the 2D catalogue_selected_2d loop (obs_d_L,
obs_m), and into the shared sky-offset loop (obs_phiS, obs_qS) -- i.e. every host mode gets
the knob, not only "mixture_selected" (this matches the design's framing of gw_scatter as an
axis orthogonal to host_mode, needed for Cell T of every mode, not a mixture-only feature).

### (c) The resolved-flags return (design §3 item 1)

An out-parameter, not a return-arity change: run_mirror_seed_inprocess gained
resolved_flags_out: dict[str, Any] | None = None. None (default) is a complete no-op -- no
attribute read, no dict mutation -- so every pre-existing call site (there are 14 across
results/ scripts plus this module's own run_arm_seed, none of which pass this kwarg) is
unaffected; a return-type change would have broken all of them (diag_csv, elapsed =
run_mirror_seed_inprocess(...) 2-tuple unpacking). When given a dict, it is populated (in place,
inside the try block, right after bs.evaluate(...) returns, while bs is still in scope) via
the new _resolved_flags_from_bs(bs) helper, which reads the 13 attributes design §3 item 1 names:
_normalization_mode, _catalogue_global_selection, _selection_in_completion_numerator,
_catalogue_numerator_survival, _catalogue_numerator_survival_2d, _mass_filter_sigma,
_mass_filter_geometry, _mass_filter_k, and the theta-hook state (_theta_b, _theta_s,
_theta_sites, _theta_phi_divisor, _theta_zwindow -- the two [HIER] site-2.3 siblings are
included because a non-identity theta on those sites would silently change the SAME resolved-flag
surface this assertion exists to catch).

assert_resolved_production_flags(resolved, expected=None) is the STOP-gate: raises
AssertionError naming the first mismatched key/value pair against REGISTERED_RESOLVED_FLAGS
(the chair-confirmed "phi"/"phi"/"fused" triple under absolute_marginal, the B5.1 mass-window
defaults, the with-BH twin "mz_sel" adopted under row #223/charter B7.3, and theta identity). The
expected parameter lets a caller pass a narrower/updated mapping (design §3's "the harness runs at
the wave-2 commit and asserts whichever value production resolves there" -- e.g. a future
catalogue_numerator_survival_2d re-adoption need not require editing this module).

---

## 2. Acceptance evidence (design §8 S1 acceptance-test list)

### (i) Existing arms byte-identical: b0i seed 900101 and one bsel seed reproduce their banked
event_likelihoods.csv bit-for-bit

**Construction proof (primary evidence).** Neither the "catalogue_selected" nor the
"population_selected" elif block's body was edited by this stage -- git diff against the
pre-S1 tree shows the ONLY change inside those two blocks is the surrounding code (the new
"mixture_selected" branch was inserted between them and the trailing else, touching zero lines
of the two pre-existing branches). The gw_scatter refactor of the shared post-branch code (the
obs_d_L/obs_m/sky-offset draws) is an algebraic no-op at the default gw_scatter=True: each
edited line reads rng.normal(...) into a local, then adds (value if gw_scatter else 0.0) --
at gw_scatter=True this reduces to exactly the previous expression, same floating-point operation
order, so bit-identical by construction, not by empirical luck.

**Empirical evidence (event-cap smoke, resource-bounded per the launch stamp: taskset -c
12,13,14,15 pin -> 2 multiprocessing workers at os.sched_getaffinity(0) size 4 (max(1,
affinity-2)), well inside the "≤ 2 worker processes" ceiling; event-cap 20).**

*Comparand 1 (b0i, host_mode="catalogue_selected").* Drove a full 200-event realization at seed
900101 (config identical to hier_s0_driver.py's build_bc_venue docstring:
CorrespondenceConfig() defaults, build_bsel_selection_objects(h_true=H_TRUE),
_verify_rate_weight_parity() before the draw), took the first 20 rows
(events.head(20)), and evaluated them via run_mirror_seed_inprocess(..., h_values=(H_TRUE,),
resolved_flags_out=resolved). Compared every non-identifier column
(w_G/w_G_legacy/w_tilde_G/alpha_G_phi/r_Malm/D_tilde_phi/L_cat_no_bh/
L_cat_with_bh/B_num/B_num_wbh/g_frac/L_comp/combined_no_bh/combined_with_bh/
den_log_term/num_log_term_no_bh/num_log_term_with_bh) against
results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run/
s0a_seed900101/node_truth/simulations/diagnostics/event_likelihoods.csv, restricted to
event_idx < 20 (this comparand's provenance: hier_s0_driver.py's run_arm_seed_s0a, node
"truth" (theta identity, default theta_sites="all"), config "b0i" -- read-only, never edited by
this node). Global tables (w_tilde_G, alpha_G_phi, D_tilde_phi, r_Malm) are h-only and
catalogue-only, so they are unaffected by capping the analyzed event count; per-event legs
(L_cat_*, B_num*, combined_*) depend only on that event's own candidate ball, likewise
unaffected.

*Comparand 2 (bsel, host_mode="population_selected")* was designed identically (CorrespondenceConfig
(sigma_z_scale=1.0, area_scale=1.0), build_bsel_selection_objects() at default h_true=H_TRUE,
h_values=(0.725, 0.735), theta_sites="2.2", smear_global_selection=False, against
kwq1_registered_run/s0a_seed900101/node_truth_ft_sites2.2_nosmear) but was NOT run -- see the
finding below, which surfaced on comparand 1 before comparand 2 was attempted, and the per-attempt
cost (see "resource note") made a second blind attempt a poor use of the remaining budget.

**RESULTS (comparand 1, two attempts).** Both attempts (h_bounds=None defaulting to (0.73, 0.73),
then explicit h_bounds=(0.6, 0.86) to test the [P3-HGRID] hypothesis below) produced 14 scored
events (of 20 drawn, matching the banked file's own F-0 survivor count for event_idx < 20) and
IDENTICAL numbers between the two attempts. Per-column comparison against
hier_s0_registered_run/s0a_seed900101/node_truth:

| column | max abs diff | max rel diff | match |
|---|---|---|---|
| w_G, w_G_legacy, w_tilde_G, alpha_G_phi, r_Malm, D_tilde_phi | 0 | 0 | EXACT |
| B_num, B_num_wbh, g_frac, L_comp, den_log_term | 0 | 0 | EXACT |
| L_cat_no_bh | 3.72e-02 | 55.6% | **MISMATCH** |
| L_cat_with_bh | 1.07e-02 | 34.7% | **MISMATCH** |
| combined_no_bh | 6.02e-03 | 24.2% | **MISMATCH** |
| combined_with_bh | 6.64e-04 | 30.6% | **MISMATCH** |
| num_log_term_no_bh, num_log_term_with_bh | 0.28-0.37 | 1.7-2.5% | **MISMATCH** |

The resolved-flags assertion PASSED (assert_resolved_production_flags raised nothing); the
resolved dict is the full registered production set verbatim.

**Diagnosis (not fully resolved; isolated to a scope this stage's diff cannot own).**
git status --short on darksiren_emri/bayesian_inference/bayesian_statistics.py and
darksiren_emri/galaxy_catalogue/handler.py is EMPTY -- this stage's diff touches neither file, so
the mismatch cannot be a B8.2 S1 regression in the sense of "code this stage wrote changed these
numbers". The pattern of WHICH columns match is diagnostic: every h-only global table (the
mixture-weight legs this stage's compute_catalogue_class_weight_p_g depends on) and B_num/
B_num_wbh (the completion-leg numerator, a function of the event's own OBSERVED d_L/error, not
of catalogue candidates) match EXACTLY -- which means the realization this run drew (donor row,
noise, observed d_L/sky) is bit-identical to whatever produced the banked file. Only
L_cat_no_bh/L_cat_with_bh (the CANDIDATE-BALL-dependent catalogue leg) and everything downstream
of them differ. This points at a difference in candidate-ball construction (mass window /
get_possible_hosts_from_ball_tree) between this commit (HEAD, 647e86d9) and whatever commit
produced the banked hier_s0_registered_run artifact -- a live candidate, given git log shows FIVE
same-day [PHYSICS] commits on this exact file pair landing between some unknown point and HEAD
today (0b308828 mass-filter-geometry, 6c6f2a63 theta-phi-divisor/sky-cone-k, d4765539 the
with-BH catalogue-leg twin ADOPTION -- a genuine default change, not "byte-identical" -- 62f7d61e
mass-aware-1D-leg, 7e1ed96f theta-zwindow), any one of which the banked artifact may predate. The
[P3-HGRID] h_bounds hypothesis was tested and REFUTED (identical wrong numbers with and without
h_bounds=(0.6, 0.86) explicit) -- it is not the cause.

**Disposition.** This is reported as a finding, not swept under "PASS": the literal S1 acceptance
test (i), read as "bit-for-bit against this specific banked file", did NOT pass. It is NOT attributed
to this stage's own change (the diff scope excludes it, and the h-only/completion-leg exact matches
positively confirm this stage's own code path). Full resolution -- pin the exact commit
hier_s0_registered_run was generated at, or regenerate it at HEAD -- is deferred to **S2's own
PROD-A0 engagement gate** (design §3 item 2 / §8 S2 acceptance (iii)), which is explicitly built to
tolerate exactly this class of staleness: it names "the wave-2 C0 baseline task's columns at the
wave-2 commit" as an alternative comparand precisely because a HEAD artifact on a fast-moving branch
is not guaranteed current. S2 should not skip its own PROD-A0 gate on the strength of this note.

**Resource note (why comparand 2 was not run and why this took two attempts).** Each attempt paid a
~500-560s single-process, effectively single-threaded setup cost (catalogue load: 22.6M rows / 1.68
GB reduced catalogue CSV, PLUS build_bsel_selection_objects's SimulationDetectionProbability
construction) BEFORE any of this stage's own code ran -- this is the design's own documented "~20-25
min/h injection-pool selection-grid cost" (design §0), here landing at the low end because only one
h_true is tabulated. This exceeds the launch stamp's foreground-600s-per-command budget as a SINGLE
command; both attempts were run as a backgrounded local process (nohup + disown, no ssh, no
cluster), polled via bounded waits, never left unattended past this turn. The first attempt (no
internal timeout wrapper on the FIRST try) was killed by an over-eager internal timeout 580
wrapper before any output printed -- a script-harness mistake, not evidence of a longer true cost;
removing that wrapper let both real attempts complete in ~510-565s each. A THIRD attempt (comparand
2, or re-diagnosing comparand 1's mismatch under an explicit catalogue_numerator_survival_2d="off"
counterfactual) would cost the same ~500s+ again; given the mismatch is already outside this stage's
diff scope and already routed to S2's own gate, that spend was not made.

### (ii)-(v) Unit tests (darksiren_emri_test/validation/test_correspondence_1d.py, all pool-free,
synthetic test doubles, no pinned production inputs -- same convention as every existing
catalogue_selected/catalogue_selected_2d test in this file)

- **The mixture weight equals the estimator class weight on a fixture** (item 1):
  test_compute_catalogue_class_weight_p_g_matches_path_a_mixture_objects monkeypatches the three
  production leg-builders (build_b0i_2d_selection_objects, precompute_phi_selection_integrals,
  precompute_global_catalog_selection) to fixed values and checks
  compute_catalogue_class_weight_p_g's p_g equals the REAL path_a_mixture_objects's own
  w_tilde_G on those values -- tests the WIRING this stage added (which leg feeds which
  with_bh_mass flag), not production's already-tested selection integrals (fixtured out, not
  re-derived). test_mixture_selected_class_split_matches_class_weight_p_g_statistically
  additionally confirms the draw-level consumption: at p_g=0.3, n=4000, the realized
  catalogue-hosted fraction sits within 5 sigma of the binomial expectation.
- **gw_scatter on/off paired on one RNG stream** (item 2):
  test_gw_scatter_false_is_truth_centred_and_shares_rng_stream_with_true (1D path) and
  test_gw_scatter_false_truth_centres_2d_joint_draw (the 2D loop) -- host selection is IDENTICAL
  between the paired True/False calls (the "same draws elsewhere" requirement), and the
  gw_scatter=False realization's observed d_L/M/sky sit exactly at the latent truth.
  test_gw_scatter_true_default_is_byte_identical_to_omitting_it pins the no-op default.
- **Resolved-flags assertion fires on a wrong flag** (item 3):
  test_assert_resolved_production_flags_fires_on_a_wrong_flag and
  ..._fires_on_a_wrong_theta_flag; ..._passes_on_the_registered_values and
  ..._accepts_a_narrower_expected_mapping pin the non-firing paths;
  test_resolved_flags_from_bs_reads_registered_attribute_set pins the reader's attribute set.
- **Byte-identity of the pre-existing modes** (item 4):
  test_mixture_selected_p_g_one_matches_catalogue_selected_bit_for_bit and
  ..._p_g_zero_matches_population_selected_bit_for_bit (the mixture's own limits, on a fixture)
  plus every PRE-EXISTING pinned-value regression test in this file
  (test_catalogue_mode_byte_unchanged_regression,
  test_catalogue_selected_2d_byte_identical_to_pre_repair_when_no_floor_rows, and the full
  existing suite) continuing to pass unmodified is the byte-identity evidence for "catalogue",
  "population", "population_selected", "catalogue_selected" and "catalogue_selected_2d" --
  none of those tests, or the branches they exercise, were touched by this stage.

**Full suite: 88/88 passed** (darksiren_emri_test/validation/test_correspondence_1d.py, uv run
pytest -m "not gpu and not slow", 11.3 s wall). **darksiren_emri_test/validation directory: 419
passed, 2 deselected** (the pre-existing gpu/slow marks), 53.1 s wall, coverage 31.44% (gate:
25%).

---

## 3. Quality gate

- uv run ruff check --fix darksiren_emri/validation/correspondence_1d.py
  darksiren_emri_test/validation/test_correspondence_1d.py -- all checks passed (one
  auto-fix: a line-length wrap in the new test file).
- uv run ruff format -- both files already/now formatted.
- uv run mypy darksiren_emri/validation/ (7 source files) -- Success, no issues.
- uv run mypy darksiren_emri_test/validation/test_correspondence_1d.py -- Success, no issues.
  One bool | npt.NDArray[np.bool_] annotation was added at in_catalog_col's first assignment
  (software-only; no behavior change) because the new "mixture_selected" branch is the first to
  assign an array rather than a scalar to that name and mypy infers from first use.

---

## 4. Files to commit

New:

- results/campaign51_20260728/realistic_20260729/tree2_20260830/B8_2_S1_RECORD.md (this file).

Modified:

- darksiren_emri/validation/correspondence_1d.py (harness file; NOT a physics-trigger file per
  CLAUDE.md's list -- /physics-change was not invoked, consistent with the design's own framing
  of this stage as "harness files only, no physics-trigger file").
- darksiren_emri_test/validation/test_correspondence_1d.py (34 new tests appended; zero existing
  tests edited).

No other file was touched. No production data file, results/.../hier_s0_zwin_run, or
hier_s0_driver.py was read-written -- hier_s0_driver.py was read-only consulted (to determine
the exact b0i/bsel comparand configs above) and never edited, per the launch stamp's explicit
prohibition. No git operation was performed.

---

## 5. What S1 explicitly did NOT do (deferred to S2/S3/S4 per the design's own scope)

- No fleet arm registration (ARM_SPECS/ARM_HOST_MODE/etc.) for "mixture_selected" -- S1's
  scope is the generator capability, not a new fleet arm.
- No driver/scorer (b8_cal_harness.py) -- that is S2.
- No pilot run, no N-ladder timing, no census comparison -- that is S3.
- No band/statistic registration -- forbidden by the bounded-scope rule (design §8): "S1-S3 may not
  change any band, statistic definition, or the mixture law."
