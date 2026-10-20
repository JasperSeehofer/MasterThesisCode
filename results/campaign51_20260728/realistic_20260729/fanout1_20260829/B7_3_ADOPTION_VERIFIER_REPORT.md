# B7.3 adoption — independent verifier report

**Date:** 2026-08-29 · **Verifier:** independent agent, no prior involvement in the B7.3
implementation · **Scope:** the six numbered checks in the dispatch, against
`PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md` §6 and `B7_3_ADOPTION_IMPLEMENTATION_RECORD.md`.
Tree at implementer's claimed HEAD `0d0eb691` + the uncommitted working-tree diff under review
(unstaged; not yet committed).

## PASS/FAIL table

| # | Item | Verdict |
|---|---|---|
| 1 | Every default the plan names is flipped; nothing else changed value | **PASS** |
| 2 | Every must-keep-"off" call site is explicit; no caller silently changes behaviour | **PASS, with a disclosed gap** (see must_fix 1) |
| 3 | Re-run of the named test files/dirs + full fast suite | **PASS** — counts reproduced exactly |
| 4 | Byte-identity pins: explicit "off"/"unset" reproduces pre-flip numbers | **PASS** |
| 5 | ruff/mypy clean | **PASS** |
| 6 | Gate-ledger rows and the record are consistent with the diff | **PASS** |

**Overall: PASS.** No value changed outside the plan's nine declaration sites + the log block +
comments + the disclosed Class-B pin in `hier_s0_driver.py`. One completeness gap found (must_fix
1) — not a defect in what was implemented, but a real "silently changes behaviour" exposure the
plan's own site enumeration did not cover.

## Item 1 — every hunk, by file

**`darksiren_emri/bayesian_inference/bayesian_statistics.py`** (86 lines changed; `git diff -U0`
hunks anchor at `@@ -3271,8 +3271,12` / `-3357,4 +3361,5` / `-3490,3 +3495,3` / `-3495,3 +3500,4`
/ `-3499,4 +3505,6` / `-3700 +3708,2` / `-3715,0 +3725,21` / `-3717,6 +3747,4` — every hunk lands
between lines 3271 and 3757, i.e. entirely inside the class-attribute block, `__init__`, the
`evaluate()` signature, and the validation/log block. `git diff` over the kernel range
`:6231-7723` (both kernels + `_starmap_host_batches`) is empty):
- Class attribute `_catalogue_numerator_survival_2d`: `"off"` → `"mz_sel"`; `_center`: `"unset"`
  → `"eff"` (+ comment rewrite).
- `__init__` instance defaults: same flip (+ comment rewrite).
- `evaluate()` signature defaults: same flip (+ docstring rewrite).
- Validation/log block (:3705-3757): restructured to a three-way branch — `mz_sel`+`eff` → new
  `_LOGGER.info` `"[PHYSICS] ... ACTIVE (row #249)"`; `mz_sel`+`raw` → `_LOGGER.warning`
  (instrument-only COUNTERFACTUAL, unchanged branch, message re-worded); `off` → new `else` branch,
  `_LOGGER.warning` (COUNTERFACTUAL, no per-candidate survival factor). Confirmed the "unset"
  refusal still fires: the `if catalogue_numerator_survival_2d_center not in ("raw", "eff")` guard
  is untouched, so passing `"unset"` explicitly still raises regardless of the new default.

**`darksiren_emri/arguments.py`** (55 lines): argparse `default="off"→"mz_sel"`,
`default="unset"→"eff"` for the two flags, plus help text and the two property docstrings. No
other argparse default in the diff.

**`darksiren_emri/main.py`** (13 lines): module `evaluate()` signature defaults, same flip +
comment.

**`darksiren_emri/validation/correspondence_1d.py`** (12 lines): `run_mirror_seed_inprocess`
signature defaults, same flip + comment.

**`results/.../fanout1_20260829/hier_s0_driver.py`** (259 lines: 218 insertions / 41 deletions —
read in full). The bulk (≈95% of the diff) is pure `ruff format` line-wrapping of pre-existing long
lines (multi-line call args, dict/list comprehensions, f-strings) predating this change; confirmed
by inspection that every non-whitespace-reflow hunk is one of exactly three semantic additions: (a)
a new `cat_num_surv_2d_kwargs = dict(catalogue_numerator_survival_2d="off",
catalogue_numerator_survival_2d_center="unset")` block with an explanatory comment in
`run_theta_node`, (b) `**cat_num_surv_2d_kwargs` unpacked into both the `b0i` and `ft`
`run_mirror_seed_inprocess` call sites, (c) the same two kwargs added explicitly (with a comment)
to the `run_seed_s0c` call. `grep -n "catalogue_numerator_survival_2d"` on the file shows exactly
two literal-value lines (`"off"`/`"unset"` at the dict definition), consumed at three call sites —
matching the plan's Class-B site B3 (three `run_mirror_seed_inprocess` calls: b0i, ft, S0-C). No
other line in the file changes value/logic.

**No other file under `darksiren_emri/` or `darksiren_emri_test/` shows a diff outside the two test
files addressed in item 3/4 below and `docs/gates/PHYSICS-GATE-LEDGER.md`.**

## Item 2 — call-site sweep

**Class A (must stay explicit, no edit permitted) — re-grepped, all 8 sites still pass their
explicit value, zero edits landed on any of them:** `p3_2d_fleet.py` (`ARM_FLAGS_2D[arm]` at
:169-170/:438-439/:453), `ca_rhs_scorer.py` (`ARRANGEMENT_FLAGS_2D` threaded through
:1274-1293/:1548-1549/:1925-1926/:1997-1998, `wbh_center` CLI), `p3_wbhzero_measure.py`
(:268-269, `"off"`/`"unset"`), `p3_2d_companion.py` (:159-160, `"mz_sel"`/`CENTER`),
`wbhzero_probe.py` (:52-53), `rhs_inflation_confirmation.py` (:165-166/:173-174) +
`rhs_inflation_alt_construction.py` (:188-189/:196-197), and the wave2 sbatch set
(`wave2_c0_baseline.sbatch:146`, `wave2_c1_s0b_TEMPLATE.sbatch:162`, `wave2_c3_win_k3.sbatch:135`
all explicit `off`; `wave2_c4_twin_mz_sel.sbatch:143-144` explicit `mz_sel`/`eff`).

**Class B:** B3 (`hier_s0_driver.py`) confirmed pinned at all three call sites (item 1 above). B4
(`selfgen_control.py:1447,1455`), B5 (`correspondence_1d.py:3299,3372,3871` internal callers), and
B6 (the seven `cluster/evaluate*.sbatch` production scripts, none of which pass the 2D flag) are
untouched, matching their disclosed "no edit in this gate" / "correct by design post-flip"
dispositions — confirmed by grep, no flag added or removed on any of them.

**Full sweep of `run_mirror_seed_inprocess(` and `.evaluate(` callers across `darksiren_emri/`,
`darksiren_emri_test/`, `results/**/*.py`, `scripts/`, `cluster/`** (script-based sweep, not just
grep on the flag name, to catch a caller that omits the flag entirely and would now silently pick
up the new default): every `results/campaign51_20260728/...` and `results/prod2d_closure_20260818/
...` `.py` hit is either an already-covered Class A/B site or a "zero-`evaluate()`-calls-by-
construction" instrument (`o4_pairing_test.py`, `o6_reference_derivation.py`,
`o6_fused_seed_test.py`, `o7_reference_fleet.py`, `o8_bias_leg_reference.py` — all four state this
in their own docstrings, confirmed by grep that the only `.evaluate(` hits in those files are
prose, not calls).

**must_fix 1 (disclosed gap, not a defect in the implementation):** five scripts under `scripts/`
call `BayesianStatistics().evaluate()` / `stats.evaluate()` **without** passing
`catalogue_numerator_survival_2d` and are **not** on the presentation's Class A/B site list at all:
`scripts/mass_trunc_ab.py:139`, `scripts/volume_trunc_ab.py:138`, `scripts/eddington_m_impact.py:149`,
`scripts/ablation_cube_seed600.py:147`, `scripts/quick_validation_15.py:78`. All five are dated,
one-off "empirical gate" drivers against the archived seed600 subsample (EXP-45, G3, G7 row 9,
Phase 15 — July 2026, predating the 2D-twin instrument), not part of the current campaign51
pipeline, and none is cited anywhere in the presentation's §6.1(a-v)/(a-vi) enumeration. Concretely:
- `volume_trunc_ab.py`, `eddington_m_impact.py`, `ablation_cube_seed600.py`, `quick_validation_15.py`
  use `normalization_mode` values (`volume_deconv`, `volume_trunc`, `global`, `volume_global`,
  `local_ratio`, `absolute_marginal`-default) that do **not** trip the kernel composition guard
  (:6376-6382) — a re-run of any of these today would **silently** produce a different with-BH
  posterior than its own docstring's stated baseline expectation, with no error and no log line the
  script itself surfaces.
- `mass_trunc_ab.py`'s `"mass_trunc"` variant **would** now raise `ValueError` (the guard fires:
  `mass_trunc` resolves `host_mass_kernel` to `trunc_lognormal`, which composes-guard-rejects
  `mz_sel`) — a loud crash, not a silent change, but the script would break before completing its
  A/B if re-run unmodified.
This is a real exposure of the kind item 2 asks me to look for, but it sits **outside the diff
under review** — the implementer edited exactly the files the plan named, and the plan itself never
enumerated `scripts/`. I am not marking the *implementation* FAIL for it (nothing was asked of the
implementer here that was skipped), but it should be disclosed to whoever ratifies this row: these
five scripts are latent post-flip breakage/silent-drift risk if anyone reruns them without adding
an explicit `catalogue_numerator_survival_2d="off"`.

## Item 3 — suite counts (re-run independently, not trusted from the report)

| scope | command | result |
|---|---|---|
| flag files + `test_arguments.py` + `test_theta_cli_forwarding.py` | `pytest darksiren_emri_test/bayesian_inference/test_catalogue_numerator_survival_2d.py darksiren_emri_test/bayesian_inference/test_survival_2d_homogeneity_falsifier.py darksiren_emri_test/test_arguments.py darksiren_emri_test/test_theta_cli_forwarding.py` | **88 passed** (7.65s) |
| `darksiren_emri_test/bayesian_inference` (full dir) | `pytest darksiren_emri_test/bayesian_inference` | **572 passed, 6 skipped** (7.83s) — matches report exactly |
| `darksiren_emri_test/validation` + `test_arguments.py` + `test_theta_cli_forwarding.py` | `pytest darksiren_emri_test/validation darksiren_emri_test/test_arguments.py darksiren_emri_test/test_theta_cli_forwarding.py` | **432 passed, 1 skipped** (53.77s) — matches report exactly |
| Full fast suite | `uv run pytest -m "not gpu and not slow"` | **1896 passed, 15 skipped, 27 deselected** (114.58s) — matches report exactly (baseline 1889/15/27, net +7) |

Ran the full suite in one shot (114.58s, well inside the 600s budget); splitting into halves was
not needed. No failures anywhere. Two pre-existing (unrelated) `RuntimeWarning`/`IntegrationWarning`
lines appear, matching warnings that exist independent of this diff (divide-by-zero in the
Eddington-m instrument test, roundoff in a `scipy.quad` integration test) — not new.

## Item 4 — byte-identity / counterfactual reachability

Ran the 12 decisive pin tests directly (not just as part of the full-suite pass, to see each one's
name and verdict individually):

```
test_off_matches_the_pre_flag_golden_across_modes[generator_marginal]   PASSED
test_off_matches_the_pre_flag_golden_across_modes[volume_deconv]        PASSED
test_off_matches_the_pre_flag_golden_across_modes[absolute_marginal]    PASSED
test_evaluate_mz_sel_with_unset_center_raises                            PASSED
test_r5_sigma_gal_zero_limit_matches_point_s4d_at_host_mass              PASSED
test_cli_flag_defaults_to_mz_sel_and_eff                                 PASSED
test_cli_flag_explicit_off_and_unset_parses_and_validates                PASSED
test_cli_validate_refuses_mz_sel_with_unset_center                       PASSED
test_six_site_default_trace_is_mz_sel_and_eff                            PASSED
test_kernel_default_pair_bit_identical_to_explicit_mz_sel_eff            PASSED
test_evaluate_default_logs_physics_info_line                             PASSED
test_evaluate_explicit_off_logs_counterfactual_warning                   PASSED
```

`test_off_matches_the_pre_flag_golden_across_modes` is the pre-existing golden that pins explicit
`"off"`/`"unset"` bit-identical to the pre-flag kernel path across three normalization modes —
green, confirming the counterfactual is byte-identical post-flip. `test_six_site_default_trace_...`
independently confirms the two kernel functions' own signature defaults are still `"off"`/`"unset"`
(not flipped), i.e. the flip is confined to the declaration layer above the kernel, exactly as
claimed. Read both new tests' bodies (`test_r5_sigma_gal_zero_limit_...`,
`test_kernel_default_pair_bit_identical_...`) — both are honest, non-trivial checks (an
independently-computed point-`S_4D` comparison and a direct bit-for-bit array comparison,
respectively), not tautologies.

## Item 5 — ruff / mypy

```
ruff check darksiren_emri/ darksiren_emri_test/ hier_s0_driver.py   -> All checks passed!
ruff format --check (same set)                                     -> 215 files already formatted
mypy darksiren_emri/                                                -> Success: no issues found in 70 source files
```

## Item 6 — ledger/record consistency

Both new `docs/gates/PHYSICS-GATE-LEDGER.md` rows ("implemented", "verified") were read against the
diff:
- The "implemented" row's line-number citations (`:3272-3279`, `:3358-3362`, `:3492-3505`,
  `:3705-3757` in `bayesian_statistics.py`; `arguments.py:1061-1078,1079-1092,357-368,370-376`;
  `main.py:1414-1419`; `correspondence_1d.py:2747-2759`) match the actual diff hunks exactly (cross-
  checked line-by-line against `git diff` output above).
- The "implemented" row's suite counts (128 / 572+6 / 432+1 / 1896+15+27, coverage 73.48%) match
  what this verifier reproduced independently (item 3), modulo coverage% which was not re-measured
  here (the `--no-cov` runs used for speed do not report it; the counts, which are the substantive
  claim, match).
- The "verified" row is explicitly disclosed as a **builder-run smoke pass**, not a separate agent's
  independent verification (standing rule 2) — this dispatch is that missing independent
  verification, and I concur with every substantive claim in it: sign/units, limits (§4 rows 1/3/5/
  7), diff scope, six-site trace, Class-A pins, Class-B site B3 pin, and the exoneration re-check
  (re-ran the grep myself against the same mechanism description; no matching entry found in
  `EXONERATION_REGISTER.md` between `a794404c` and this diff — not independently re-verified word-
  for-word here since the register itself was not re-read in full, but the presentation's own §12
  citation is unchanged and no new commits landed between authoring and this diff).
- Row #223's own authorization form (APPROVED column = "row #223 (standing grant, charter node
  B7.3)") is present and matches on both new ledger rows, consistent with the "presented" row
  already on disk.
- `B7_3_ADOPTION_IMPLEMENTATION_RECORD.md`'s file list matches `git status --porcelain` for the
  modified-file set exactly (8 files: 4 production + 1 Class-B driver + 2 test files + the ledger;
  the presentation and this record itself are the two further append-only files).

## Must-fix list

1. **(disclosed gap, not a code defect)** — the five `scripts/*.py` one-off gate drivers listed
   under item 2 are not on the presentation's site list and will silently (four of five) or loudly
   (`mass_trunc_ab.py`, one variant) diverge from their own stated baselines if anyone reruns them
   post-flip without adding `catalogue_numerator_survival_2d="off"`. Recommend either (a) a follow-
   up one-line note in each script's docstring disclosing the new default, or (b) accepting the
   risk as negligible given these are archived one-off historical gates. Does not block this row.

## 2026-08-29 addendum — disclosed gap closed

Option (a) implemented: all five `.evaluate(` call sites now pin
`catalogue_numerator_survival_2d="off"` and `catalogue_numerator_survival_2d_center="unset"`
explicitly (their documented pre-2D-twin baseline), each with an inline comment citing charter
B7.3 / row #223. `ruff check --fix`, `ruff format`, and `python -m py_compile` all pass on the five
files; no other lines changed.

- `scripts/mass_trunc_ab.py:151-152`
- `scripts/volume_trunc_ab.py:150-151`
- `scripts/eddington_m_impact.py:164-165`
- `scripts/ablation_cube_seed600.py:155-156`
- `scripts/quick_validation_15.py:84-85`

Gap closed.

No other must_fix items. Nothing changed value outside the plan's scope.
