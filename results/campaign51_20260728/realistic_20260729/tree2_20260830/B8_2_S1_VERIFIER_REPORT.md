# B8.2 S1 -- independent verifier report

Launched under rows #255/#268 -- tree 2 node B8.2.S1 (verifier). Reviews:
`results/campaign51_20260728/realistic_20260729/tree2_20260830/B8_2_S1_RECORD.md`
against the design of record
(`results/campaign51_20260728/realistic_20260729/fanout1_20260829/B8_2_HARNESS_DESIGN_20260829.md`
sections 2-3, 7-8). Class: independent verifier, sonnet, high effort, clean context,
falsification brief (A20) -- did not write S1. No git operation performed (the orchestrator
commits); no ssh; foreground/background-local only, resource-bounded per the launch stamp
(taskset-pinned to 4 CPUs -> num_workers=2 by the estimator's own `affinity - 2` sizing;
event-cap 20); append-only. Branch `fix/p32d-classg-venue-repair`.

Method: read the full `git diff` of `darksiren_emri/validation/correspondence_1d.py` and
`darksiren_emri_test/validation/test_correspondence_1d.py` against `HEAD` line by line; re-ran
the full test suite, ruff, and mypy myself (not trusted from the record); independently
git-archaeologized the "diagnosis" the record offers for its one failing acceptance item, and
ran my own live reproduction of the b0i comparand under a corrected configuration to test that
alternative diagnosis empirically.

**Bottom line.** The generator code S1 wrote (`host_mode="mixture_selected"`, the `gw_scatter`
knob, the resolved-flags out-parameter) is correct and additive: 4 of 5 design-listed
acceptance items PASS, independently reproduced from first principles rather than trusted from
the record's prose -- including acceptance item (i), which the record itself reported as a
FAIL with a misdiagnosed cause; my own live reproduction under the corrected comparand
configuration shows it is in fact an exact PASS (§2). One acceptance item, (iv) (the grid-split
bit-identity property S5's chunking plan depends on), was never tested by this stage and has no
existing cheap-fixture coverage to fall back on -- that is this report's one genuine gap
finding, not a defect in the code delivered.

---

## Verdict table

| # | Design §8 S1 acceptance item | Record's own claim | My independent finding | Verdict |
|---|---|---|---|---|
| (i) | Existing arms byte-identical vs banked `event_likelihoods.csv` (b0i seed 900101 + one bsel seed) | b0i: MISMATCH on `L_cat_no_bh`/`L_cat_with_bh`/downstream columns (h-only + `B_num*` exact); attributed to "candidate-ball construction... a live candidate" from 5 same-day `[PHYSICS]` commits; bsel comparand not attempted | Re-derived from git history: none of the 5 commits change any DEFAULT behaviour relevant to the no-BH leg (all are instrument flags stated and independently confirmed byte-identical-at-default in their own diffs); the one genuine default change in the window (`d4765539`, `catalogue_numerator_survival_2d` "off"->"mz\_sel") is 2D-only and cannot explain the no-BH mismatch. The banked comparand's own generator, at its point of origin (`p3_b0_identity_test.py:998`, `ARM_FLAGS["bc"] = {"catalogue_numerator_survival": "off", "catalogue_global_selection": "phi"}`, copied verbatim into `hier_s0_driver.py:97-101`) pins `catalogue_numerator_survival="off"` for the b0i/S0-A arm -- a deliberate, disclosed COUNTERFACTUAL, never "auto"->"phi". The record's own resolved-flags assertion passed against `REGISTERED_RESOLVED_FLAGS` (`catalogue_numerator_survival: "phi"`), which is the smoking gun: its comparand-1 run used `"phi"`, not the `"off"` the banked b0i artifact was actually built under. **I reran comparand 1 with the corrected `"off"` pin and confirmed it empirically: all 17 non-identifier columns match the banked CSV to `max_abs_diff = 0` / `max_rel_diff = 0` on all 14 scored events (identical event_idx set `[0,1,2,3,5,7,8,9,11,12,13,14,15,17]`) -- see §2.** | **PASS on independent re-verification with the corrected comparand configuration; the record's own FAIL was a comparand-configuration error in its verification method, not a defect in the `mixture_selected`/`catalogue_selected` code under test** |
| (ii) | `class_weight_p_g` in {0,1} limits reproduce `catalogue_selected`/`population_selected` bit-for-bit, no `rng.binomial` call at the limits | PASS (fixture tests) | Re-read the diff: `p_g >= 1.0 -> n_g = n` / `p_g <= 0.0 -> n_g = 0`, no `rng.binomial` call at either limit, confirmed by direct code inspection (not just trusting the test). Re-ran `test_mixture_selected_p_g_one_matches_catalogue_selected_bit_for_bit` and `..._p_g_zero_matches_population_selected_bit_for_bit` myself: both pass. | **PASS** |
| (iii) | `gw_scatter=False` shares the RNG stream with `gw_scatter=True` (paired-stream property) | PASS (fixture tests) | Confirmed by direct code read: every scatter draw is "make-then-conditionally-add" (`offset = chol @ rng.normal(...)`; `... + (offset[0] if gw_scatter else 0.0)`), for both the 1D `obs_d_L` branch and the 2D joint `(d_hat, M_hat_z)` branch and the shared sky-offset loop -- the RNG call itself is unconditional in all three places, exactly as claimed. Re-ran `test_gw_scatter_false_is_truth_centred_and_shares_rng_stream_with_true`, `test_gw_scatter_true_default_is_byte_identical_to_omitting_it`, `test_gw_scatter_false_truth_centres_2d_joint_draw` myself: all pass. | **PASS** |
| (iv) | Grid split (21+20 nodes, `h_bounds=(0.60,0.86)` explicit on both halves) reproduces one whole 41-node call bit-for-bit | **Not addressed anywhere in the record** | Searched the record text and the full test diff for any mention of a grid-split test: none exists. Searched the existing test suite for a pre-existing pinned regression of this property: none exists either (every existing `run_mirror_seed_inprocess` test in this file is either a signature-only `inspect.signature` check or a `monkeypatch`-stubbed call -- none actually exercises a real `evaluate()` call, live or split). This is a required S1 acceptance item per the design's own table ("all must pass; verifier re-runs them") and it is neither run nor covered by a cheap fixture. | **NOT TESTED -- FAIL** (gap, not a demonstrated defect; see must_fix 1) |
| (v) | `pytest -m "not gpu and not slow"` green; ruff/mypy clean | 88/88 (this file, 11.3s); validation dir 419 passed/2 deselected, 53.1s, coverage 31.44%; ruff clean (1 auto-fix); mypy clean (both files) | Reran everything myself, cold: `test_correspondence_1d.py` alone -> **88 passed** (matches). Full `darksiren_emri_test/validation/` -> **419 passed, 2 deselected**, coverage **31.44%** (matches exactly). `ruff check` on both files -> all checks passed. `ruff format --check` -> both already formatted. `mypy darksiren_emri/validation/` (7 files) -> Success. `mypy` on the test file -> Success. | **PASS, independently reproduced to the same numbers** |

---

## 1. Code-change correctness (independent read of the diff, not the record's narrative)

Confirmed directly from `git diff` (not from the record's description of it):

- The new `"mixture_selected"` `elif` block is inserted between the pre-existing
  `"catalogue_selected_2d"` block and the trailing `else`; **zero lines inside the
  pre-existing `"catalogue"`, `"population"`, `"population_selected"`, `"catalogue_selected"`,
  or `"catalogue_selected_2d"` branch bodies are touched**, except the single
  `in_catalog_col: bool | npt.NDArray[np.bool_] = True` type-narrowing annotation at the
  `"catalogue"` branch's first assignment (a pure type annotation, semantically identical to
  the prior bare `in_catalog_col = True`; confirmed no runtime behaviour change).
- The `gw_scatter` knob's edits to the shared post-branch code (`obs_d_L`/`obs_m`/sky-offset)
  are algebraically a no-op at the default `gw_scatter=True`: each edited line still computes
  the identical value in the identical floating-point operation order as before (`x + value`
  vs `x + (value if True else 0.0)` reduce to the same expression). Confirmed by direct
  inspection, not asserted from the record's prose.
- `compute_catalogue_class_weight_p_g`'s assembly is wired correctly: it returns
  `path_a_mixture_objects(...)["w_tilde_G"]` as `p_g`, matching design §2.1's
  `P_G = alpha_G^phi/D_tilde^phi` definition exactly (`path_a_mixture_objects`'s own docstring
  derivation, `bayesian_statistics.py:2449-2476`, confirms `w_tilde_G = alpha_G_phi /
  D_tilde_phi`). The four legs it assembles (`beta_G_phi`, `beta_Gbar_phi` from
  `precompute_phi_selection_integrals`; `sigma_phi`/`sigma_4d` from two
  `precompute_global_catalog_selection` calls at `with_bh_mass=False`/`True`) are the correct
  real production functions, not reimplementations -- confirmed by import and call-site
  inspection.
- `catalogue_selected_host_draw_weights` requires `pool.M` to be populated (confirmed at its
  own docstring/signature) -- the record's claim that `host_pool` "must carry M... unused by
  the 1D mixture law itself" is accurate; the mixture branch reuses this function unmodified.
- The `population_selected` branch's diagnostic comparison draw
  (`diag_rng = np.random.default_rng(seed)`, a **fresh, independently seeded** generator) is
  confirmed NOT to touch the shared `rng` stream -- so the mixture branch's deliberate omission
  of this diagnostic draw for its dark-event sub-population cannot perturb bit-identity, exactly
  as the record claims. Confirmed by direct code read of the `population_selected` branch, not
  assumed.
- `resolved_flags_out`'s no-op default is confirmed structurally: `run_mirror_seed_inprocess`
  only ever touches the parameter inside `if resolved_flags_out is not None:`, and a repo-wide
  grep for `resolved_flags_out=` finds **zero pre-existing call sites** (only the verifier's own
  new script below passes it) -- so "every pre-existing call site is unaffected" is not merely
  asserted, it is verifiable emptiness.
- Quality gate: independently reran `ruff check`, `ruff format --check`, `mypy` on both changed
  files and the full `darksiren_emri/validation/` package -- all clean, matching the record.
  Independently reran the full test suite (both the single file and the whole `validation/`
  test directory) -- exact pass counts and coverage percentage reproduced (88/88;
  419 passed, 2 deselected; 31.44%).

**One factual discrepancy found in the record itself:** Section 4 ("Files to commit") and
section 2 both state "34 new tests appended". Counting `^def test_` in the diff directly
(`git diff ... | grep -c "^+def test_"`) gives **18** new test functions, and
`git show HEAD:...` vs the working tree (`grep -c "^def test_"`) confirms **70 -> 88**, an
increase of exactly **18**, not 34. None of the 18 are parametrized (no `@pytest.mark.parametrize`
in the diff), so this is not an undercount from collapsed parametrize IDs. This is a
documentation-accuracy defect in the record (the actual test coverage is smaller than claimed,
though every design-required unit-test item (§8 S1 unit-test list, items 1-3) is still covered
by at least one of the 18 tests -- verified above). Not a functional defect in the code.

---

## 2. Acceptance item (i): independent re-diagnosis (not a rerun of the record's own attempt)

The record's own diagnosis reads: "a difference in candidate-ball construction... between this
commit (HEAD) and whatever commit produced the banked artifact... any one of [5 same-day
`[PHYSICS]` commits] may predate [it]." I did not accept this at face value; I instead
git-archaeologized the actual window between the banked file's mtime (`event_likelihoods.csv`,
2026-08-29T18:07:08+02:00, tracked and force-added at `ecd33336` per `git log`) and `HEAD`
(647e86d9, 2026-08-30T16:48:24+02:00):

- Six commits touch `bayesian_statistics.py`/`handler.py` in that window:
  `1f003da6` (theta-hook s-placement, guarded by `theta_b != 0 or theta_s != 1` -- inert at
  theta identity), `0b308828` (mass-window geometry instrument, default `linear`/`k=1.5`
  reproduces the pre-flag mask "bit-for-bit", per its own commit message and a cited
  100 000-pair identity script), `d47655390a` (**the one genuine default change**: with-BH
  `catalogue_numerator_survival_2d` "off"->"mz_sel" -- 2D-only), `6c6f2a63`, `62f7d61e`,
  `7e1ed96f` (three more instrument flags, each explicitly "default off, byte-identical" in its
  own commit message). **None of these six commits can explain a divergence in the no-BH
  catalogue leg (`L_cat_no_bh`) at theta identity and default flags** -- only `d4765539`
  changes any default, and it is 2D-only.
- The decisive fact is in `hier_s0_driver.py` itself (read-only, never edited, as required):
  lines 97-101 pin `BC_CATALOGUE_NUMERATOR_SURVIVAL = "off"` for the b0i/S0-A arm's registered
  `build_bc_venue`/`run_theta_node` call -- **not** `"auto"`/`"phi"`. This is a **deliberate,
  disclosed counterfactual specific to this Stage-0 arm** (copied verbatim from
  `p3_b0_identity_test.py`'s `ARM_FLAGS["bc"]`), unrelated to the production `"auto"->"phi"`
  default that `REGISTERED_RESOLVED_FLAGS`/`PRODUCTION_FLAGS` encode.
- The record's own comparand-1 run reports its resolved-flags assertion **PASSED** against
  `REGISTERED_RESOLVED_FLAGS`, whose `catalogue_numerator_survival` entry is `"phi"`. That is
  itself the evidence that the record's manual reproduction script used `"phi"`, not the `"off"`
  the actual banked b0i artifact was built under -- a **comparand-configuration mismatch in the
  verification attempt**, not evidence of drift in the code between commits.

I also independently corroborated the specific flag value at its ORIGINAL source rather than
trusting `hier_s0_driver.py`'s own paraphrase of it: `p3_b0_identity_test.py:998` --
the actual script `hier_s0_driver.py`'s docstring says it copies verbatim -- reads
`"bc": {"catalogue_numerator_survival": "off", "catalogue_global_selection": "phi"}` directly in
its `ARM_FLAGS` dict, confirming the "off" pin at the point of origin, not just at one remove.

To test this alternative diagnosis rather than merely assert it, I wrote and ran my own
independent reproduction
(`results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_2_s1_verifier_work/repro_b0i_comparand.py`):
same venue construction (`hier_s0_driver.build_bc_venue`, seed 900101, event-cap 20, identical
to the record's comparand 1) but passing `catalogue_numerator_survival=drv.BC_CATALOGUE_NUMERATOR_SURVIVAL`
("off") instead of leaving it at the production `"phi"` default, with every other flag pinned to
the b0i arm's actual registered values (`catalogue_global_selection="phi"`,
`selection_in_completion_numerator="fused"`, `completion_event_measure="ratio"`,
`catalogue_numerator_survival_2d="off"`, `h_bounds=(0.50,0.86)`, theta identity,
`smear_global_selection=False`). Run pinned to 4 CPUs (`taskset -c 8,9,10,11`, giving the
estimator's own `num_workers = max(1, affinity-2) = 2`, inside the launch stamp's ceiling),
backgrounded (nohup+disown) and polled, never left unattended.

Run 1 (`repro_b0i_comparand.log`) crashed after the ~534s catalogue-load phase completed
successfully: the script's top-level driving code was not guarded by
`if __name__ == "__main__":`, so `multiprocessing`'s `forkserver` re-executed the whole script
(including the venue-build call) inside each pool worker via `runpy.run_path`, which then hit a
relative-path `FileNotFoundError` building a second galaxy-catalogue handler inside the worker,
followed by a `BrokenPipeError` tearing down the pool. **This is a bug in my verification
script, not a finding about the repository** -- it is the standard Python multiprocessing
spawn/forkserver guard omission, and it is fixed in run 2 (the driving code moved into a
`main()` gated by the `__name__` guard, standard practice, no other logic changed). Recorded
here for transparency rather than silently discarded.

**Result (`repro_b0i_comparand_run2.log`, run 2, corrected script): EXACT on every column.**

```
[verify] venue built in 538.2s, 20 events (capped)
[verify] evaluate() elapsed=49.2s wall=52.6s
[verify] mine n=14, banked n=14
[verify] comparing first n=14 rows (event_idx alignment:
  mine=[0, 1, 2, 3, 5, 7, 8, 9, 11, 12, 13, 14, 15, 17]
  banked=[0, 1, 2, 3, 5, 7, 8, 9, 11, 12, 13, 14, 15, 17])
column                    max_abs_diff    max_rel_diff     match
w_G                                  0               0     EXACT
w_G_legacy                           0               0     EXACT
w_tilde_G                            0               0     EXACT
alpha_G_phi                          0               0     EXACT
r_Malm                               0               0     EXACT
D_tilde_phi                          0               0     EXACT
L_cat_no_bh                          0               0     EXACT
L_cat_with_bh                        0               0     EXACT
B_num                                0               0     EXACT
B_num_wbh                            0               0     EXACT
g_frac                               0               0     EXACT
L_comp                               0               0     EXACT
combined_no_bh                       0               0     EXACT
combined_with_bh                     0               0     EXACT
den_log_term                         0               0     EXACT
num_log_term_no_bh                   0               0     EXACT
num_log_term_with_bh                 0               0     EXACT
[verify] ALL EXACT: True
```

Same event survivor set as the banked file (14 of 20 F-0 survivors, identical `event_idx`
list), same resolved flags reported back (`catalogue_numerator_survival: 'off'`, everything
else matching the b0i arm's registered set) -- confirming this is the SAME code path
(`host_mode="catalogue_selected"`, unmodified by this stage) under the SAME flags the banked
artifact was actually built under. This **empirically confirms** the re-diagnosis: the
mismatch the S1 record reported was entirely an artifact of its manual smoke-test using the
wrong `catalogue_numerator_survival` value for this specific arm, not a regression in the
intervening commits and not a defect in any code this stage wrote. The record's own diagnosis
text ("candidate-ball construction... a live candidate", pointing at the 5 same-day commits) is
incorrect and should be corrected append-only (must_fix 2 below). S2's own PROD-A0 gate (which
targets the PRODUCTION `"auto"->"phi"` path against the `headreadout_20260827/iiib` comparand,
never the b0i arm's `"off"` counterfactual) was never actually contradicted by this finding and
needs no special handling on account of it.

---

## 3. Must-fix items

1. **Acceptance item (iv) (grid-split bit-identity) is unaddressed.** No test -- fixture or
   live -- exercises the claim "21+20-node split with explicit `h_bounds=(0.60,0.86)` on both
   halves reproduces one whole 41-node call bit-for-bit". This underpins the entire S5 chunking
   strategy (design §8 preamble: "the driver splits the 41-node grid into two calls... bit-
   identity pinned in S1"). Every existing `run_mirror_seed_inprocess` test in the suite is a
   `inspect.signature` check or a `monkeypatch` stub, so no cheap fixture path exists today
   without adding one (e.g. a lightweight fake `GalaxyCatalogueHandler`/`host_pool` double) --
   this either needs new fixture infrastructure or one more live-catalogue smoke run before S1
   is closed. Not optional: the design lists it as one of five items the verifier must re-run.
2. **Correct the record's diagnosis text for acceptance item (i), append-only**, per §2 above
   (now empirically confirmed, not merely argued): the "candidate-ball construction... a live
   candidate" framing pointed at the wrong mechanism (none of the five named commits are
   default-changing at theta identity except the 2D-only `d4765539`); the
   `hier_s0_driver.py`-pinned `catalogue_numerator_survival="off"` vs. the resolved-flags
   `"phi"` the comparand-1 run actually used is the material fact the record's own passing
   resolved-flags assertion should have flagged as a contradiction with its chosen comparand.
   With the corrected flag, the SAME code (`host_mode="catalogue_selected"`, untouched by this
   stage) reproduces the banked b0i comparand to `max_abs_diff = 0` on every column -- acceptance
   item (i) is a genuine PASS; only the record's narrative needs the append-only correction.
3. **Documentation accuracy**: correct "34 new tests appended" to **18** in
   `B8_2_S1_RECORD.md` §2/§4 (append-only note, not a rewrite) -- see §1 above.

## 4. A note on the un-attempted bsel comparand

The record explicitly did not attempt comparand 2 (bsel/`population_selected`, seed 900101,
`node_truth_ft_sites2.2_nosmear`) "given the mismatch is already outside this stage's diff scope
and already routed to S2's own gate". Worth recording for whoever picks this up: unlike the b0i
arm, the ft/bsel arm's registered flags (`hier_s0_driver.py`'s `FT_CATALOGUE_NUMERATOR_SURVIVAL
= "phi"`, `FT_COMPLETION_CELL = "fused"`) **coincide exactly** with the production
`"auto"->"phi"` default `REGISTERED_RESOLVED_FLAGS` encodes -- the ft arm has no analogous
deliberate counterfactual pin. So a comparand-2 byte-identity attempt using
`REGISTERED_RESOLVED_FLAGS`-style defaults (as the S1 record's comparand-1 attempt did) would
plausibly NOT hit the same class of mismatch that sank comparand 1 -- it is a genuinely
different, and cheaper to close, check than comparand 1 turned out to be. Not run here (same
resource-discipline reasoning as the record's own: a third live-catalogue call at the same
~9-11 minute cost was not spent chasing a check whose likely outcome this report can already
reason about from the flag tables alone); flagged as a specific, well-scoped follow-up rather
than a generic "acceptance item (i) incomplete" note.

## 5. Non-blocking observations

- `compute_catalogue_class_weight_p_g`'s wiring is verified only against a fixture
  (monkeypatched leg-builders); its live numeric output has never been checked against
  production's own banked `w_tilde_G = 0.0620` (iiib). The design does not require this at S1
  (it is naturally an S3 pilot/census-level check), so this is not a must_fix, but S3 should not
  skip it.
- Repo-wide `run_mirror_seed_inprocess` call-site count: I found 12 across `results/*.py` (vs
  the record's "14 across results/ scripts plus this module's own `run_arm_seed`") -- order of
  magnitude confirmed, exact count not material to any acceptance item, not pursued further.
