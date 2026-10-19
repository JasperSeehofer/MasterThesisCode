# Node B6.1 [ALIGN] — IMPLEMENT record

*launched under rows #222/#223 — charter node B6.1*

Status: **IMPLEMENTED, not committed.** The orchestrator commits. No `git commit`/`add`/
`reset`/`checkout`/`stash` was run by this node.

## 1. What this node did

Implemented the s-placement alignment gate presented in the appended note (2026-08-29) of
`results/campaign51_20260728/realistic_20260729/PHYSICS_CHANGE_THETA_HOOK_20260828.md`
(row #221 item 4, [RULE] decision-table item 1 of that note): `theta_s` now scales the RAW
catalogue redshift error BEFORE the peculiar-velocity (PV) quadrature fold, at all three
θ-parameterized sites, superseding the 2026-08-28 "s scales the folded width" placement. `b`'s
placement is unchanged (still shifts the kernel centre AFTER the fold).

Per STANDING RULE 2 (verifier independence), this node BUILT the instrument and did its own
smoke-testing (the full test run below), consistent with a builder role; there is no separate
runner role specified for this node in the charter, so the test execution reported here is
the builder's own regression run, not an independently-run registered measurement.

## 2. Code-vs-document match (verified before editing)

Confirmed by direct read at HEAD `a794404c` that the OLD-formula code blocks quoted in the
appended note's §1 were byte-identical to the code at the cited lines:
- `bayesian_statistics.py:6370-6381` (site 2.1) — matched.
- `bayesian_statistics.py:7041-7050` (site 2.2) — matched.
- `bayesian_statistics.py:1692-1704` (site 2.3) — matched (actual pre-edit lines were
  1691-1701 in the file at HEAD; a ~1-3 line drift from the note's citation, immaterial to
  content — same code, same order).
- `constants.py:95` — confirmed `SIGMA_V_PEC_KM_S = 0.0` still holds.

## 3. PRIMARY FINDING carried into this node, and how it was resolved

The task handed to this node included a pre-existing review finding (from an earlier review
pass on the appended note) identifying a genuine internal inconsistency in that note:

- The note's own §2 **formula literal** computes `sigma_z_pv` from `z̃` (the POST-b-shift
  redshift): `sigma_z_pv = (1 + z̃) · SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S`.
- The same section's **prose** states the opposite: *"the `b` placement is UNCHANGED by this
  note — `b` still shifts the centre; only the fold ORDER for `s` moves."*

These disagree whenever `b ≠ 0` and `SIGMA_V_PEC_KM_S ≠ 0`: the formula literal silently
reverses the 2026-08-28 pin (`sigma_z_pv` from the RAW, unshifted host redshift), while the
prose claims that pin stands. The note's own limiting cases (§5) and regression plan (§6(b))
never exercise `theta_b ≠ 0` together with `SIGMA_V_PEC_KM_S ≠ 0` — the one regime where the
two readings diverge — so this would have shipped undetected by every test the note names.
This is numerically inert today only because `SIGMA_V_PEC_KM_S = 0.0` makes `sigma_z_pv ≡ 0`
under either reading (the note's own limiting-case-1 argument).

**Resolution taken (this node's judgment call, disclosed for the orchestrator's review):**
implemented the PROSE, not the formula literal — `sigma_z_pv` is computed from the host
redshift local as it stood BEFORE the `theta_b` shift, exactly as in the 2026-08-28 code, at
all three sites. Only the `host_z_error_eff`/`sigma_eff` computation changes (RAW catalogue
error scaled by `s`, combined in quadrature with that unshifted-z `sigma_z_pv`). Rationale:
(a) matches the note's own stated scope ("b's placement is unchanged by this note") and its
Reference §3 ("`s` scales the catalogue's quoted error... never the peculiar-velocity term");
(b) is the conservative choice — introduces no `b`-behavior change beyond what was already
approved 2026-08-28, so it does not require a fresh [RULE] gate to land within this note's
stated scope; (c) is bit-identical to the literal-formula reading at every θ while
`SIGMA_V_PEC_KM_S = 0.0`, so nothing observable today distinguishes the two readings — the
choice only matters if that constant is ever set nonzero, at which point this resolution keeps
faith with "b unchanged."

A new regression test was added specifically to pin this resolution against silent regression:
`test_theta_b_order_unchanged_uses_raw_host_z_for_pv` (patches `SIGMA_V_PEC_KM_S` nonzero,
engages `theta_b` alone, asserts the kernel matches the RAW-host_z closed form for
`sigma_z_pv`). If a future edit reintroduces the z̃-based literal, this test fails instead of
shipping silently.

**This resolution is a judgment call, not an author ruling** — it is flagged here, and in the
appended implementation-record note in `PHYSICS_CHANGE_THETA_HOOK_20260828.md`, for the
orchestrator/author to confirm or override. If overridden (i.e., the z̃-based literal is what
was actually intended), the fix is a one-line change at each site (move the `host_z = host_z +
theta_b * (1.0 + host_z)` line to run BEFORE the `sigma_z_pv` computation instead of after),
and the new b-order test above would need to be inverted.

## 4. Exact diff (line numbers post-edit)

`darksiren_emri/bayesian_inference/bayesian_statistics.py`:
- Site 2.3, `_smeared_global_pdet_expectation`, lines ~1696-1710 (post-edit numbering; was
  1692-1704 in the note's citation): `sigma_eff` inside the θ-branch now
  `np.maximum(np.sqrt((theta_s * z_err_g) ** 2 + sigma_z_pv**2), 1e-10)`, computed BEFORE the
  `z_g = z_g + theta_b * (1.0 + z_g)` line (order swapped from the original so the shift can't
  leak into `sigma_z_pv`, which is a local already fixed above from the pre-shift `z_g`).
- Site 2.1, `single_host_likelihood`, lines ~6374-6386 (was 6370-6381): `host_z_error_eff`
  inside the θ-branch now `float(np.sqrt((theta_s * host_z_error) ** 2 + sigma_z_pv**2))`,
  computed before the `host_z` shift line (same reordering as site 2.3).
- Site 2.2, `single_host_likelihood_batch`, lines ~7047-7060 (was 7041-7050): same pattern,
  vectorized (`host_z_error_eff = np.sqrt((theta_s * host_z_error) ** 2 + sigma_z_pv**2)`).
- Reference comments at all three sites updated to cite the row #221 item 4 note and state
  the b-unchanged/s-reordered semantics explicitly.
- `single_host_likelihood_integration_testing` (site 2.7 twin) — untouched, per PA-HIER-11
  (twin deliberately not θ-parameterized).

`darksiren_emri_test/bayesian_inference/test_theta_hook.py`:
- Module docstring updated to describe the new formula and add gate item "7" for the
  s-placement discriminators.
- 6 new test functions appended after `test_theta_validation_errors`:
  1. `test_theta_s_placement_old_new_forms_diverge` — sanity check that OLD/NEW closed forms
     actually differ once `SIGMA_V_PEC_KM_S != 0` and `theta_s != 1` (discriminator is not
     vacuous).
  2. `test_theta_s_placement_prefold_scalar` — site 2.1 discriminator (note §6(b)).
  3. `test_theta_s_placement_prefold_batch` — site 2.2 discriminator (note §6(b)).
  4. `test_theta_s_placement_prefold_smeared_site23` — site 2.3 discriminator (note §6(b)).
  5. `test_theta_b_order_unchanged_uses_raw_host_z_for_pv` — the extra b-order regression pin
     for the PRIMARY FINDING resolution (§3 above), not required by the note's own plan but
     added to close the gap the note's regression plan leaves open.

All existing tests in the file (θ=(0,1) identity pins, closed-form substitution equivalence,
counters, twin-parity, defaults guard, validation-error guards) were left unmodified — per
limiting case 1 of the note, they hold bit-identically under both the OLD and NEW s-placement
while `SIGMA_V_PEC_KM_S == 0.0`.

## 5. Bit-identity evidence

- `test_theta_identity_bit_equality_scalar`/`_batch`, `test_smeared_site23_identity_and_substitution`
  (pre-existing, unmodified): θ = (0,1) still bit-equal to the no-θ default at all 3 sites —
  27/27 targeted tests green (see §6).
- `test_theta_engaged_equals_substituted_inputs_scalar`, `test_smeared_site23_identity_and_substitution`'s
  substitution half (pre-existing): θ-engaged with `SIGMA_V_PEC_KM_S == 0.0` still matches the
  substituted-inputs closed form at `rtol=1e-12` — confirms the OLD/NEW s-placement rewrite is
  a genuine no-op today, not just "close."
- New discriminator tests (this node): with `SIGMA_V_PEC_KM_S` monkeypatched to 200.0 km/s and
  `theta_s = 1.4142`, the hooked call matches the NEW (pre-fold) closed form at `rtol=1e-9`
  (site 2.1, 2.2, 2.3) and the b-order test confirms the RAW-host_z form specifically.

## 6. Test run counts

- `uv run pytest darksiren_emri_test/bayesian_inference/test_theta_hook.py darksiren_emri_test/bayesian_inference/test_catalog_only_diagnostic.py -q -x -p no:cacheprovider`:
  **27 passed** (the subset's own coverage-threshold failure line is a global `fail-under=25%`
  artifact of running a narrow file subset, not a test failure — 10.42% coverage on a 2-file
  run is expected and unrelated to correctness).
- `uv run pytest -m "not gpu and not slow" -q -p no:cacheprovider` (full fast suite):
  **1851 passed, 15 skipped, 27 deselected**, 134.77s. No pre-existing pin moved.
- `uv run ruff check --fix` + `uv run ruff format` on both changed files: clean, no changes
  beyond what was already written (0 additional fixes applied).
- `uv run mypy darksiren_emri/bayesian_inference/bayesian_statistics.py`: `Success: no issues
  found in 1 source file`.

## 7. Ledger and provenance records

- `docs/gates/PHYSICS-GATE-LEDGER.md`: appended `implemented` (PASS) and `verified` (PASS)
  rows following the pre-existing `presented` row for this node (row's commit-ref column:
  `pre-commit`, per protocol until the orchestrator's commit lands). One formatting defect
  from the edit (a duplicated trailing `| |` at the end of the `verified` row, caused by the
  append landing before a pre-existing trailing pipe on the last line of the table at the time
  of editing) was found and corrected in the same pass.
- `results/campaign51_20260728/realistic_20260729/PHYSICS_CHANGE_THETA_HOOK_20260828.md`:
  appended an "Implementation record 2026-08-29" section (append-only; no existing text
  edited) documenting the same PRIMARY FINDING resolution as §3 above.

**Caveat — shared working tree.** This repo checkout is being edited concurrently by sibling
fan-out nodes (observed: `docs/RESEARCH_CYCLE.md`, `results/.../gate_b_20260730/BIAS_HISTORY_LEDGER.md`,
and a `PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md` + ledger row for node B5.1 all appeared
in `git status`/`git diff` during this node's run, none touched by this node). The `git diff
--stat` and ledger tail used to write this record were taken at the time of writing and may
have moved on further since; the files-to-commit list in §8 is scoped strictly to this node's
own edits.

## 8. Files to commit (this node's scope only)

- `darksiren_emri/bayesian_inference/bayesian_statistics.py`
- `darksiren_emri_test/bayesian_inference/test_theta_hook.py`
- `docs/gates/PHYSICS-GATE-LEDGER.md` (this node added 2 rows; the file also carries other
  nodes' concurrent rows — the orchestrator should diff-review before committing whole-file)
- `results/campaign51_20260728/realistic_20260729/PHYSICS_CHANGE_THETA_HOOK_20260828.md`
  (this node appended one "Implementation record" section; append-only, nothing existing
  edited)
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/B6_1_ALIGN_RECORD.md`
  (this file)
