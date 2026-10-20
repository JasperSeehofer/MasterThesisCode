# B7.3 adoption — implementation record

**Date:** 2026-08-29 · **Charter node:** B7.3 (row #223 standing grant) · **Authorization:** ledger
rows #222/#223; gate presentation `PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md` (PANEL-CLEAN, 0
rounds), its "presented" gate-ledger row cites APPROVED = "row #223 (standing grant, charter node
B7.3)".

This record exists to give the orchestrator/committer the exact file list for the `[PHYSICS]`
commit (§9.3/§13.4 of the presentation) in one place, separate from the narrative implementation
record appended to the presentation file itself (§13 there has the diff summary, re-pinned tests,
suite counts, and the filled-in commit message).

## Files to commit

Production code (the literal default flip + comment sweep):
- `darksiren_emri/bayesian_inference/bayesian_statistics.py`
- `darksiren_emri/arguments.py`
- `darksiren_emri/main.py`
- `darksiren_emri/validation/correspondence_1d.py`

Class-B call-site pin (keeps banked Stage-0/KW-Q1 comparands byte-identical after the flip):
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py`

Tests (re-pinned + new, §6.1(a-vi)/§6.2):
- `darksiren_emri_test/bayesian_inference/test_catalogue_numerator_survival_2d.py`
- `darksiren_emri_test/validation/test_correspondence_1d.py`

Ledger + presentation records (append-only):
- `docs/gates/PHYSICS-GATE-LEDGER.md` (implemented + verified rows appended)
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md`
  (§13 implementation record appended; §§1-12/stamp unedited)
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/B7_3_ADOPTION_IMPLEMENTATION_RECORD.md`
  (this file, new)

`BIAS_HISTORY_LEDGER.md` row #249 is **not** filed by this row — per the presentation §9.2, "the
orchestrator files it."

## Not touched by this implementation (verified, not merely asserted)

- Kernel bodies `bayesian_statistics.py:6231-7723` (both kernels + `_starmap_host_batches`):
  `git diff` over that range is empty.
- Worker signature defaults (`single_host_likelihood`, `single_host_likelihood_batch`) and the
  composition guards (:6376-6382, :7316-7322): unedited, confirmed still `"off"`/`"unset"` by
  `test_six_site_default_trace_is_mz_sel_and_eff`.
- All 8 Class-A call sites (§6.1(a-v)): re-grepped post-implementation, all still explicit.
- `p3_wbhzero_measure.py:268-269` comment ("also the flag's own default") — now stale; left
  unedited per the presentation's own disposition for this site ("AMEND on next use, never re-run
  banked"), and it was not in this task's authorized edit-file list.
- `darksiren_emri/validation/selfgen_control.py` (Class-B site B4) and the harness-internal callers
  in `correspondence_1d.py` (Class-B site B5) — the presentation's own disposition for these is "no
  edit in this gate" / "pin if a banked arm is ever regenerated"; not touched.
- `cluster/evaluate*.sbatch` (Class-B site B6) — production scripts; post-flip they correctly
  produce the ADOPTED estimator by design, no pin needed.

## What the plan asked for that was not fully done (disclosed)

1. **(a-iv) early evaluate()-layer composition guard** — not added. The presentation explicitly
   permits this ("if the implementer judges the early guard out of the minimal diff, the kernel
   guards alone still realize G-1"). The existing kernel guards (:6376-6382 scalar, :7316-7322
   batch) already raise on `mz_sel` composed with `trunc_lognormal` or non-`"production"`
   `catalogue_mass_overlap`, so G-1 is realized without a duplicate check.
2. **§6.2 item 3 (production-scale cluster pin)** — not attempted. It requires the production CRB
   set, reduced catalogue, and injection pool (`cluster/datasets.yaml:246`), none of which are
   available to a local implementation task; the presentation itself says this "rides the wave-3
   per-change arm" rather than running as a local pytest.
3. **Independent "verified" row** — the gate-ledger "verified" row filed by this task is a
   builder-run smoke pass, not a separate agent's independent verification (standing rule 2 asks
   for builder != runner on registered measurements; no second agent was dispatched to this node).
   Disclosed in the ledger row itself, following the row-#(B5.1) precedent.

## Suite counts (reproduced from the ledger rows)

- Flag files only (`test_catalogue_numerator_survival_2d.py` + `test_survival_2d_homogeneity_falsifier.py`): 56 passed.
- + `test_correspondence_1d.py`: 128 passed.
- `darksiren_emri_test/bayesian_inference` (directory): 572 passed, 6 skipped.
- `test_arguments.py` + `test_theta_cli_forwarding.py` + `darksiren_emri_test/validation`: 432 passed, 1 skipped.
- Full `pytest -m "not gpu and not slow"`: **1896 passed, 15 skipped, 27 deselected** (baseline
  1889; net +7 new tests), 119.75s, coverage 73.48%.
- `ruff check --fix` + `ruff format`: clean on all touched files. `mypy darksiren_emri/`: clean (70
  source files).
