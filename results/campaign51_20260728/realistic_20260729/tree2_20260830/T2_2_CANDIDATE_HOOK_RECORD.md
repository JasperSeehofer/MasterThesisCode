# T2.2 -- per-candidate p_Di instrumentation hook: IMPLEMENTATION RECORD

launched under row #255 -- tree 2 node T2.2

Date: 2026-08-30. HEAD at implementation: ecd33336 (branch fix/p32d-classg-venue-repair).
A10 (row #255, author-ruled): "the per-candidate p_Di instrumentation hook is an
instrumentation change (no computed value changes; byte-identity guard + gate-ledger row),
not a full /physics-change gate." No physics-trigger-file formula, constant, or waveform
parameter changed; the physics-change protocol was accordingly NOT invoked for this node
(the instrumentation-guard route was, per A10). Cluster: bwUniCluster Lustre /pfs/data6 OST 5
inactive this node -- everything below is local-only; no ssh, no cluster job.

Inputs read before writing code:
`results/campaign51_20260728/realistic_20260729/tree2_20260830/B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md`
section 4 (what to serialise, where) and section 6 (the full T2.2 design: placement, the
exact column list, the byte-identity/reconstruction/engagement gates, cost);
`results/campaign51_20260728/realistic_20260729/fanout1_20260829/B4_1_IMP_DECOMPOSITION.md`
section 7 (the missing-data finding this hook closes, and its own cost anchor).

## 1. What was built

An OPT-IN diagnostic writer on `BayesianStatistics.evaluate()`: a new
`candidate_dump_dir: str | None = None` keyword. Default `None` is byte-identical --
no directory is created, no attribute is read differently by any existing branch, nothing is
collected, nothing is written. When set to a directory path, a read-only serialiser --
placed strictly AFTER `p_Di` returns for each event, inside `p_D`'s event loop, per the
design's section 6.1 ("no change inside p_Di") -- builds one row per (event, candidate) and
one row per (event, h) from state `p_Di` already computed for a normal run, and writes them
to `per_candidate_h_<label>.csv` / `per_event_h_<label>.csv` under that directory (one
overwrite write per h, the same `write_selection_table_json` naming convention). The hook
never writes into any object the likelihood consumes; the whole collection path is wrapped in
try/except so a diagnostic failure can never fail the run (logged once, not per event).

Per-candidate columns (17, section 6.2): `event_idx`, `h`, `catalog_index`, `batch`
(`with_bh` / `no_bh_only`), `z_g`, `z_err_g`, `M_g`, `M_err_g`, `phiS_g`, `qS_g`, `w_g`,
`N_g_used` (the no-BH numerator as consumed, `result_row[0]`), `D_g` (`result_row[1]`),
`s_bar_phi_zg` (`S_bar_phi(z_g;h)`, plain endpoint-clamped `np.interp` against
`self._phi_survival_table[h]`, the exact convention `p_Di` itself uses), `s_4d_zg_mg`
(`S_4D(z_g,M_g;h)` via `detection_probability_with_bh_mass_interpolated(dist(z_g,h),
M_g*(1+z_g), 0.0, 0.0, h=h)`), `u_g` (`(dist(z_g,h) - d_hat)/sigma_dL`), `sky_mahalanobis`
(optional per the design -- kept as an always-NaN placeholder column for schema stability;
not computed by this hook), `is_true_host` (`catalog_index ==
galaxy_catalog.resolve_host_recovery_position(host_galaxy_index)`).

Per-event columns (13, one row per (event, h)): `event_idx`, `h`, `d_hat`, `sigma_dL`,
`z_true` (read from the CRB row's `z_true` column when present, NaN otherwise -- the
synthetic test fixture lacks it, exercising that fallback), `host_galaxy_index`,
`n_cand_no_bh`, `n_cand_with_bh`, `f_bar_z_true`, `f_k_z_true`, `L_cat_no_bh`, `B_num`,
`D_tilde_phi` (the last three read from `self._diagnostic_rows[-1]`, the same dict `p_Di`
appends for the existing diagnostics CSV -- guaranteed to be this event's row since `p_Di`
appends exactly one row per call before returning).

## 2. File list (exact diff)

- `darksiren_emri/bayesian_inference/bayesian_statistics.py` -- class attribute
  `_candidate_dump_dir` (:3660-3665); `__init__` instance defaults `_candidate_dump_dir`/
  `_candidate_dump_rows`/`_candidate_dump_event_rows`/`_candidate_dump_warned`
  (:3744-3747); `evaluate()` signature `candidate_dump_dir` kwarg (:3962) + store/`os.makedirs`
  (:3980-3982); per-h reset inside the `h_values is not None` grid-mode block (:5134-5136);
  new method `_collect_candidate_dump_rows` (:5305-5471, read-only, never raises); new method
  `_write_candidate_dump_csvs` (:5478-5506); call site inside `p_D`'s event loop, immediately
  after `p_Di` returns (:5670-5682); write-out call site alongside the existing
  `_write_diagnostic_csv` call (:5199-5204).
- `darksiren_emri/arguments.py` -- `--candidate_dump_dir` argparse entry + matching
  `candidate_dump_dir` property (pattern 0b308828, same shape as `sky_cone_k`/
  `freeze_g_frac_ref_h`).
- `darksiren_emri/main.py` -- module-level `evaluate()` signature gains `candidate_dump_dir`
  and forwards it to `BayesianStatistics.evaluate()`; the top-level `--evaluate` dispatch
  passes `arguments.candidate_dump_dir`.
- `darksiren_emri/validation/correspondence_1d.py` -- `run_mirror_seed_inprocess` gains
  `candidate_dump_dir: str | None = None`, forwarded verbatim to `bs.evaluate(...)`, plus a
  docstring note.
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py` --
  `run_theta_node` gains `candidate_dump_dir`, folded into its existing `common_kwargs` dict
  (reaches both the `b0i` and `ft` config branches); `run_arm_seed_s0a`/`run_arm_seed_s0r`
  gain the same kwarg and pass a per-(seed, node) subdirectory
  (`<candidate_dump_dir>/seed<seed>_node<node>`) into `run_theta_node`, so parallel cells
  never overwrite each other's `per_candidate_h_*.csv`; `_run_one_seed_worker`'s args tuple
  extended with one trailing field (byte-identical worker dispatch when it is `None`);
  `run_arm`'s `task_args` extended to match; new CLI flag `--candidate-dump` (dest
  `candidate_dump`, default `None`) forwarded into `run_arm(..., candidate_dump_dir=...)`.
  `run_seed_s0c` (arm S0-C) is deliberately NOT touched -- out of T2.2's registered scope,
  same convention this driver already applies to `theta_sites`/`smear`/`config`/`h_values`/
  `score_h` ("S0-C ignores them, per its own registered costing-probe scope").
- `darksiren_emri_test/integration/test_candidate_dump_instrumentation.py` -- new test file
  (3 tests, `@pytest.mark.slow`, matching this repo's convention for every other
  full-`evaluate()`-pipeline test): `test_candidate_dump_off_is_default`,
  `test_candidate_dump_on_is_byte_identical_to_off` (GATE BI), `test_candidate_dump_schema`
  (GATE SCHEMA).
- `docs/gates/PHYSICS-GATE-LEDGER.md` -- one "instrumentation" row (2026-08-30, verdict PASS,
  APPROVED "row #255 A10 (instrumentation guard)").

## 3. Verification (builder-run; no separate verifier agent dispatched to this node)

- **GATE BI (byte-identity).** `test_candidate_dump_on_is_byte_identical_to_off` runs the
  same deterministic synthetic fixture (`test_pipeline_parity.py`'s seeded galaxy catalogue,
  seeded `Model1CrossCheck`, single h=0.73) twice -- once with `candidate_dump_dir=None`, once
  with a real directory -- and asserts the raw bytes of both posterior JSONs and the
  `event_likelihoods.csv` diagnostics CSV are identical (`==` on `Path.read_bytes()`, i.e.
  `max |Delta| = 0.0` by construction). PASS. The comparison is not vacuous: the same test
  also asserts the dump CSVs were actually produced.
- **GATE SCHEMA.** `test_candidate_dump_schema` asserts both dump files exist, their column
  sets match section 6.2 exactly, there is one event row per event that reached `p_Di`
  (`<= 5`, the fixture's `_N_EVENTS`), `batch`/`is_true_host` take their expected value sets,
  and `z_g`/`N_g_used`/`D_g`/`h` are finite/consistent on every candidate row. PASS.
- **Default-off check.** `test_candidate_dump_off_is_default` confirms omitting the kwarg
  writes zero `per_candidate_h_*.csv`/`per_event_h_*.csv` files anywhere under the run tree.
  PASS.
- **Reconstruction sanity (informal, not GATE R of the design -- that gate needs the real
  candidate ball on a live catalogue, out of scope for a synthetic-fixture unit test):** the
  schema test's finite-value assertions on `N_g_used`/`D_g` confirm the hook reads the SAME
  `result_row` tuples `p_Di` produces (`galaxy_likelihoods`/`additional_likelihoods` from
  `self.posterior_data_with_bh_mass`), not independently recomputed values.
- **Static checks.** `uv run ruff check` and `uv run ruff format --check` clean on all five
  touched Python files under `darksiren_emri/`; `uv run mypy darksiren_emri/
  darksiren_emri_test/` clean (216 source files, no issues). `hier_s0_driver.py` is outside
  mypy's configured path (`pyproject.toml` covers `darksiren_emri/`/`darksiren_emri_test/`
  only); it was syntax-checked (`python3 -m ast.parse`) and is `ruff check`-clean.
- **Full suite.** `uv run pytest -m "not gpu and not slow"`: **1915 passed / 15 skipped / 30
  deselected** (identical pass count to the T1.1 baseline immediately before this node --
  the three new tests are `slow`-marked and sit in the deselected count, matching every other
  full-`evaluate()`-pipeline test in this repo). The three new tests independently PASS under
  `uv run pytest -m slow darksiren_emri_test/integration/test_candidate_dump_instrumentation.py`.

## 4. The registered instrumented run (for the orchestrator; builder != runner, rule 2)

4 seeds (`DEFAULT_BC_SEEDS = (900101, 900102, 900103, 900104)`, the same KW-Q1/B-SEL
realisations B4.3 section 6.4 registers), FT config (`--config ft`: `absolute_marginal`,
`catalogue_numerator_survival="phi"`, fused, HEAD `Sigma_phi` divisor, theta identity), truth
node only (`theta=(0,1)`, the literal-skip identity -- GATE T-ID), h = 0.73:

```
uv run python results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py \
  --arm S0-A \
  --nodes truth \
  --config ft \
  --h-nodes 0.73 \
  --out-root results/campaign51_20260728/realistic_20260729/tree2_20260830/candidate_dump_run \
  --candidate-dump results/campaign51_20260728/realistic_20260729/tree2_20260830/candidate_dump_run/candidate_dump \
  --jobs 4
```

`--seeds`/`--theta-sites`/`--smear` are left at their defaults (`DEFAULT_BC_SEEDS`, `"all"`,
`"auto"`) -- byte-identical to every pre-existing `S0-A`/`truth`/`config=ft` invocation of
this driver except for the new `--candidate-dump` output. `--jobs 4` runs the 4 seeds
concurrently (one worker per seed, `cpu_per_job = 14 // 4 = 3` under the default
`--total-cpu-budget 14`); it changes wall time only, not the registered CPU-h figure below,
and is a convenience, not a correctness requirement (`--jobs 1` produces byte-identical
per-candidate CSVs, just serially).

Output: `<out-root>/s0a_seed<seed>/node_truth/` per seed (the existing driver layout,
`_node_dir_suffix` empty at these defaults), each holding `event_likelihoods.csv` (unchanged)
plus, under `<candidate-dump>/seed<seed>_node_truth/`, `per_candidate_h_0_73.csv` and
`per_event_h_0_73.csv`.

**Cost: 3.4 CPU-h** (registered anchor, B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md section
6.4's lower bound of its 3.4-3.9 CPU-h range for the 4-seed FT fleet; this command evaluates
a single h=0.73 point per the orchestrator's node-T2.2 instruction rather than section 6.4's
own 3-node {0.725, 0.730, 0.735} secant design, so 3.4 CPU-h is carried here as the given
figure of record for the orchestrator's exact command, not independently re-derived from a
single-h-point cost model). Local only; no cluster job, no dataset pin needed (the FT config
reuses the same banked injection pool every other `hier_s0_driver.py` `--config ft` run
consumes).

## 5. Scope notes / what this node did NOT do

- No physics-trigger-file formula, constant, waveform parameter, or model choice changed;
  `/physics-change` was not invoked (A10 already ruled this route).
- The instrumented run above was NOT executed by this node (builder != runner, rule 2; the
  design's section 6.6 zero-compute rescore and the registered Phi_low/`<u>`_W statistic of
  section 6.5 are for whichever agent runs the command above against the T2.2 predictions).
- `run_seed_s0c` (arm S0-C) does not accept `--candidate-dump` -- out of scope, consistent
  with this driver's existing convention for every other node-added flag.
- The enlarged-ball counterfactual and B4.3 section 6.6's mass-aware rescore are separate
  tree-2 nodes, not attempted here.
