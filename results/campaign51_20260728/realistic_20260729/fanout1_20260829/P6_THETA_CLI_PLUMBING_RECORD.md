# Node P6 — theta-hook CLI plumbing (production dispatch path)

**launched under rows #222/#223 — charter node B1.2 (P6)**

Status: **IMPLEMENTED, not committed.** The orchestrator commits. No `git commit`/`add`/
`reset`/`checkout`/`stash` was run by this node.

## 1. What this node closes

`WAVE2_REGISTRATION_CHECK_20260829.md` §0 F-C ("theta is not on the production dispatch path"):
`BayesianStatistics.evaluate()` already accepts `theta_b`/`theta_s`/`theta_sites`
(`bayesian_statistics.py:3555-3556,3561`), but neither `darksiren_emri/arguments.py` nor
`darksiren_emri/main.py` exposed a CLI surface for them — production `evaluate.sbatch` runs
(`EXTRA_EVAL_ARGS`) could not reach the theta-hook at all. This node is pure plumbing: it adds
`--theta_b`/`--theta_s`/`--theta_sites` to the CLI and forwards them unmodified to
`BayesianStatistics.evaluate()`, following the exact pattern `mass_filter_geometry`/
`mass_filter_k` used in commit `0b308828` (`arguments.py` property + argparse block →
`main.py` module-level `evaluate()` kwarg → `main.py`'s `--evaluate` CLI dispatch call site →
`BayesianStatistics.evaluate()` kwarg). Defaults are byte-identical to the pre-flag path
(`theta_b=0.0`, `theta_s=1.0`, `theta_sites="all"` — the literal-skip identity, GATE T-ID,
already enforced inside `BayesianStatistics.evaluate()`).

`--smear_global_selection` was **already** on the CLI (`arguments.py:182-184,798-810`;
`main.py:198,1379,1448`) — not touched by this node, per the task brief's "if so only add
theta" instruction.

No physics-trigger file was edited. `bayesian_statistics.py` (a physics-trigger file) was
**not** touched by this node — its theta_b/theta_s/theta_sites parameters, defaults, and
validation (`_validate_theta`, the `theta_sites in ("all","2.1","2.2","2.3")` check, the
site-2.3-requires-smear guard at `:3587-3591`) are all pre-existing (landed by the [HIER]
θ-hook build, ledger row #216, commit `d40fe5c8`). This node only gives the CLI a way to reach
those already-implemented, already-validated kwargs.

## 2. Diff summary

### `darksiren_emri/arguments.py` (+80 lines; `git diff --stat`)

- **Properties** (after the existing `mass_filter_k` property, `:417-451` post-edit):
  - `theta_b -> float` (`:417-429`) — reads `self._parsed_arguments.theta_b`.
  - `theta_s -> float` (`:430-435`) — reads `self._parsed_arguments.theta_s`.
  - `theta_sites -> str` (`:436-451`, ends just before `def to_dict`'s prior neighbour) — reads
    `self._parsed_arguments.theta_sites`.
- **Argparse** (after the existing `--mass_filter_k` block, `:1108-1187` post-edit):
  - `--theta_b` (`:1142-1156`): `type=float`, `default=0.0`.
  - `--theta_s` (`:1158-1168`): `type=float`, `default=1.0`.
  - `--theta_sites` (`:1170-1186`): `type=str`, `choices=["all","2.1","2.2","2.3"]`,
    `default="all"`. The `choices=` set is copied verbatim from
    `BayesianStatistics.evaluate()`'s own validation
    (`if theta_sites not in ("all","2.1","2.2","2.3")`, `bayesian_statistics.py:~3587`, read
    2026-08-29) — an invalid value now fails at the argparse layer (`SystemExit`, exit code 2)
    with argparse's own "invalid choice" message rather than reaching `evaluate()`'s
    `ValueError`. The site-2.3-requires-`--smear_global_selection` cross-flag guard is
    deliberately **not** duplicated at the CLI layer — it stays a single source of truth inside
    `evaluate()` (`:3587-3591`), consistent with how `mass_filter_sigma`/`mass_filter_geometry`
    cross-validation is not duplicated at the CLI layer either.

`to_dict()` (`arguments.py:123-133`) serializes `dict(vars(self._parsed_arguments))` —
the full parsed namespace — so `theta_b`/`theta_s`/`theta_sites` are captured in
`run_metadata.json`'s `cli_args` automatically; no `_write_run_metadata` change was needed
(verified by reading `main.py:356-386`, unchanged).

### `darksiren_emri/main.py` (+13 lines; `git diff --stat`)

- `main()`'s `--evaluate` CLI dispatch call (inside `if arguments.evaluate:`), after the
  existing `mass_filter_k=arguments.mass_filter_k,` line (`:216-218` post-edit):
  ```python
  theta_b=arguments.theta_b,
  theta_s=arguments.theta_s,
  theta_sites=arguments.theta_sites,
  ```
- Module-level `evaluate()` signature, after the existing `mass_filter_k: float = 1.5,`
  parameter (`:1433-1439` post-edit): adds `theta_b: float = 0.0`, `theta_s: float = 1.0`,
  `theta_sites: str = "all"` with a comment citing this node and GATE T-ID.
- That same `evaluate()`'s body, forwarding into `hubble_constant_evaluation.evaluate(...)`
  (`BayesianStatistics.evaluate()`), after the existing `mass_filter_k=mass_filter_k,` line
  (`:1476-1478` post-edit): `theta_b=theta_b, theta_s=theta_s, theta_sites=theta_sites,`.

  Note: `main.py` also has an unrelated local variable named `theta_s` inside
  `_generate_sky_localization_figure`-type plotting code (sky angle, `:1754` post-edit,
  `:1741` pre-edit per the task brief) — different function scope, no collision; confirmed by
  reading both sites.

Both files: `ruff check --fix` clean, `ruff format` no changes, `mypy` clean (see §4).

## 3. Tests added

`darksiren_emri_test/test_arguments.py` (+62 lines, appended after
`test_fisher_cond_threshold_custom`):

- `test_theta_b_default_is_identity`, `test_theta_s_default_is_identity`,
  `test_theta_sites_default_is_all` — parser defaults.
- `test_theta_b_custom`, `test_theta_s_custom` — custom float values parse.
- `test_theta_sites_valid_choices` (parametrized over `all`/`2.1`/`2.2`/`2.3`) — every value
  `evaluate()` accepts parses at the CLI layer.
- `test_theta_sites_invalid_rejected` — an invalid `--theta_sites` value raises `SystemExit`
  (argparse `choices=`).
- `test_help_shows_theta_flags` — `--help` lists all three flags (mirrors the existing
  `test_help_shows_flags` pattern).

`darksiren_emri_test/test_theta_cli_forwarding.py` (new file):

- `test_evaluate_forwards_theta_defaults` — mocks
  `darksiren_emri.bayesian_inference.bayesian_statistics.BayesianStatistics`, calls
  `darksiren_emri.main.evaluate()` with no theta args, asserts the mocked `.evaluate()` call's
  kwargs carry `theta_b=0.0`, `theta_s=1.0`, `theta_sites="all"`.
- `test_evaluate_forwards_custom_theta` — same mock, non-default
  `theta_b=0.01, theta_s=1.2, theta_sites="2.2"`, asserts unmodified forwarding.
- `test_arguments_theta_values_parse_for_cli_dispatch` — parses
  `--evaluate --theta_b 0.02 --theta_s 1.1 --theta_sites 2.1` and asserts the values `main()`'s
  CLI dispatch reads and forwards. (Driving `main()` itself end-to-end was judged out of scope
  for a plumbing node — it requires constructing a full `Model1CrossCheck` +
  `GalaxyCatalogueHandler`; the dispatch call site itself is read-verified in §2 above, and its
  three new lines follow the identical, already-tested `mass_filter_geometry`/`mass_filter_k`
  pattern at the same call site.)

`BayesianStatistics.evaluate()` itself is never called by these tests (mocked out); its theta
validation and site dispatch are unchanged and out of scope for this node.

## 4. Verification run (foreground, 2026-08-29)

```
uv run pytest darksiren_emri_test/test_arguments.py darksiren_emri_test/test_theta_cli_forwarding.py -q
30 passed in 7.96s
```

(The suite's global `--cov-fail-under=25` gate fails on this narrow subset, as expected — it is
a whole-tree coverage threshold, not a test failure; 0 test failures, 0 errors.)

```
uv run ruff check --fix darksiren_emri/arguments.py darksiren_emri/main.py \
  darksiren_emri_test/test_arguments.py darksiren_emri_test/test_theta_cli_forwarding.py
All checks passed!

uv run ruff format darksiren_emri/arguments.py darksiren_emri/main.py \
  darksiren_emri_test/test_arguments.py darksiren_emri_test/test_theta_cli_forwarding.py
4 files left unchanged

uv run mypy darksiren_emri/arguments.py darksiren_emri/main.py \
  darksiren_emri_test/test_arguments.py darksiren_emri_test/test_theta_cli_forwarding.py
Success: no issues found in 4 source files
```

Full-suite `uv run pytest -m "not gpu and not slow"` was launched foreground with a 550s
shell-level timeout to confirm no regression outside the touched files:

```
=== 1889 passed, 15 skipped, 27 deselected, 12 warnings in 172.94s (0:02:52) ===
Required test coverage of 25.0% reached. Total coverage: 73.21%
[exited with code 0]
```

Zero failures, zero errors. Consistent with an additive-only, byte-identical-default plumbing
change touching no file outside `arguments.py`/`main.py`/the two test files.

## 5. Exact production CLI for one S0-B node

Per the orchestrator's B1 path decision (row #222, "orchestrator decision 2026-08-29"): S0-B
(C1) runs in the CoR-P-faithful form `theta_sites="2.2"` + `smear_global_selection=False`
(site 2.3 out of scope for the no-BH read), with the b-node at `b = ±0.033` (re-derived from
`b_max = 0.0661`, PA-HIER-29) and `theta_s = 1.0` (s not varied at this node; S0-A keeps the
as-built s-grid separately). For the venue-iiib, h=0.73, seed-900101 CoR-P baseline
(`headreadout_20260827/iiib/run_metadata_21.json:cli_args`), the exact CLI a wave-2 sbatch
`EXTRA_EVAL_ARGS` line for the **+0.033** S0-B node now reads (new flags only; the rest of
`cli_args` unchanged from the banked CoR-P run):

```
--evaluate --h_value 0.73 \
  --normalization_mode generator_marginal \
  --mass_filter_geometry linear --mass_filter_k 1.5 \
  --theta_b 0.033 --theta_s 1.0 --theta_sites 2.2
  # (smear_global_selection NOT passed -> defaults False, matching CoR-P)
```

and the mirror **−0.033** node is identical except `--theta_b -0.033`. Neither passes
`--smear_global_selection`, so `theta_sites="2.2"` needs no smear guard (the
`theta_sites in ("all","2.3")` branch of the smear-required check at
`bayesian_statistics.py:3587-3591` is not entered for `"2.2"`).

## 6. Scope discipline

- Did not touch `bayesian_statistics.py` (physics-trigger, this node's brief forbids it and no
  edit was needed — theta already lives there from row #216).
- Did not touch `hier_s0_driver.py` or `kwq1_score.py` (owned by another wave-2 agent).
- Did not `git add`/`commit`/`reset`/`stash` anything.
- `smear_global_selection` was left untouched (already fully plumbed pre-existing).

---

**Append-only note (2026-08-29, post-verification):** full-suite `uv run pytest -m "not gpu and
not slow"` = 1889 passed / 15 skipped / 27 deselected, 0 failed, exit 0, 172.94s. §4 above
holds the final numbers; this node is complete.
