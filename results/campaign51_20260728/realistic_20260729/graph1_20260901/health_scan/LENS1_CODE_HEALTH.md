# Lens 1 — Code Health & Refactor Candidates

Scope: `darksiren_emri/` + `darksiren_emri_test/`. Read-only survey (no edits, no heavy
compute). Date: 2026-09-03. All line counts via `wc -l` / `awk` on the working tree at
`fix/p32d-classg-venue-repair`.

## 1. Module size / structure

### Largest modules (production code, excluding `M1_model_extracted_data/` generated tables)

| File | Lines |
|---|---|
| `bayesian_inference/bayesian_statistics.py` | 9,332 |
| `validation/correspondence_1d.py` | 4,626 |
| `validation/venue_transfer.py` | 3,080 |
| `validation/pp_coverage.py` | 2,898 |
| `main.py` | 2,349 |
| `bayesian_inference/simulation_detection_probability.py` | 2,316 |
| `plotting/interactive.py` | 2,250 |
| `validation/calibration_gate.py` | 1,793 |
| `galaxy_catalogue/handler.py` | 1,559 |
| `validation/selfgen_control.py` | 1,531 |
| `arguments.py` | 1,369 |

(`M1_model_extracted_data/emri_distribution.py` at 9,510 lines and `detection_fraction.py`
at 7,066 lines are excluded — these are generated/extracted lookup-table modules, not
hand-written logic; not a refactor target under this lens.)

### `bayesian_statistics.py` — the reputed monolith, confirmed

- **86 top-level `def`/`class` statements**, one class (`BayesianStatistics`, line 3526)
  spanning **~3,290 lines** (3526→6818) with only **8 methods**.
- Longest functions (by body line count):
  - `evaluate()` — **1,621 lines** (3788–5409). **Signature alone has ~74 parameters**
    (see §6 — this is the flag-sprawl smoking gun).
  - `p_Di()` — **886 lines** (5932–…)
  - `_z_prior_pdf_at()` — 404 lines (nested closure inside `single_host_likelihood_batch`)
  - `denominator_integrant_without_bh_mass()` — 397 lines (nested closure)
  - `precompute_global_catalog_selection()` — 347 lines
  - `single_host_likelihood()` — 307 lines, `single_host_likelihood_batch()` — 301 lines
  - `p_D()` — 247 lines
- **`__init__(self) -> None`** (line 3697) takes **zero parameters** — all ~90 lines of it
  are hardcoded default-state assignments (`self._catalogue_leg_1d_mass_aware = "auto"`,
  `self._catalogue_numerator_survival_2d = "mz_sel"`, etc.). All the actual configuration
  surface lives on `evaluate()`'s 74-parameter signature instead of on the object. This is
  backwards from normal object design and is precisely why `evaluate()` is 1,621 lines: it
  has to thread every flag through its own body and into every helper it calls.
- **Natural split seams** (visible from the section banners/comments already in the file):
  1. Mass-aware catalogue-leg block (`catalogue_leg_1d_mass_aware_factor`,
     `_bh_mass_denominator_inner_m_integral*`, `_sigma4d_mass_kernel_expectation`,
     `_mz_sel_2d_expectation*`) — self-contained, ~600 lines, low coupling to `evaluate()`
     body other than reading config flags.
  2. Mixture-assembly / phi-divisor block (`precompute_phi_marginal_survival`,
     `precompute_phi_selection_integrals`, `completion_mass_factor_g[_sel]`,
     `path_a_mixture_objects`, `path_a_completion_numerators`, `_phi_divisor_kernel_pass`,
     `precompute_phi_divisor_theta_ratio`) — ~1,400 lines, module-level already (not
     methods), could move to a `mixture_assembly.py` submodule with minimal churn.
  3. `evaluate()` orchestration itself — candidate to decompose into an internal
     pipeline of already-existing precompute helpers plus a much smaller "wire it
     together" method once config is pulled onto a dataclass (see §6).
  4. Worker functions (`single_host_likelihood`, `single_host_likelihood_batch`,
     `_starmap_host_batches`, `child_process_init`, `_hosts_to_arrays`) are
     multiprocessing-worker code, module-level already — natural candidate for
     `bayesian_inference/likelihood_workers.py`.
  - **Severity: HIGH** (maintainability/onboarding cost, not correctness). **Effort: L**
    (this is a large mechanical extraction touching import graphs across the package,
    but should be behavior-preserving if done as pure move+re-export). **SAFE-HYGIENE**
    in principle (module split with re-exports), but given the file's role as the sole
    production H0 pipeline, any split must be followed by a full regression-test run
    before trusting it — recommend treating as SAFE-HYGIENE-WITH-VERIFICATION, not a
    `/physics-change`-gated change (no formula changes), but not a rubber-stamp either.

### Other outliers

- `parameter_estimation/parameter_estimation.py` — 663 lines, unremarkable size; coverage
  is thin (50.4%, see §5) which is the more pressing issue there, not size.
- `validation/correspondence_1d.py` — 4,626 lines, 3 classes + ~35 module functions,
  including a `MirrorUniverseGenerator` class (line 1974) that alone likely accounts for
  a large fraction of the file (not fully measured — out of scope for a quick pass, but
  flag for lens boundary: this file is a **validation/harness** module, not the
  production pipeline, so a split is lower priority than `bayesian_statistics.py`).
  **Severity: MED. Effort: M.**
- `validation/venue_transfer.py` (3,080) and `validation/pp_coverage.py` (2,898) are
  comparable in size to `correspondence_1d.py` and are all harness/validation code from
  the recent bias-hunting campaigns — as a group they represent a large and growing body
  of one-off campaign infrastructure that has not been curated down. Worth a follow-up
  lens on "which validation modules are still live vs. campaign-specific and archivable"
  (out of scope here). **Severity: LOW (as code health), MED (as project debt). Effort: M.**

## 2. TODO/FIXME/HACK/XXX inventory

Only **3 hits** in production code, **0** in tests:

- `darksiren_emri/waveform_generator.py:66` — `remove_garbage=True,  # TODO: understand why to use this`. **Severity: MED** (an unexplained `few`-waveform-generator flag left on with no rationale is exactly the kind of thing that should either get a comment explaining it or a regression test pinning behavior with it off). **Effort: S** (investigate + document, or file as a physics question — BEHAVIOR-TOUCHING if the flag is ever flipped, but documenting it is SAFE-HYGIENE).
- `darksiren_emri/validation/venue_transfer.py:153` — `TODO(seed-grain-parents): if the gate is ever...` — attributed/tracked TODO, looks intentionally deferred with a named tag. **Severity: LOW. Effort: — (already tracked).**
- `darksiren_emri/bayesian_inference/bayesian_statistics.py:1857` — comment referencing "is deferred to the next campaign — see TODO.md" (not itself a code TODO marker, just prose pointing at `TODO.md`). **Severity: LOW.**

Overall TODO density is very low for a 65K-line codebase — this is a positive signal, not
a debt pile. **Severity: LOW / informational.**

## 3. Dead / vestigial code

- **Pipeline-A remnants**: CLAUDE.md documents Pipeline A (`bayesian_inference.py`,
  `bayesian_inference_mwe.py`) as removed in commit `c1571a2`, and `datamodels/galaxy.py`
  as removed in `90bd40ee`. Confirmed: no such files exist in the tree; grep for
  `bayesian_inference_mwe` / `pipeline_a` / `Pipeline A` returns **zero matches**. The only
  matches for a broader `detection_probability` grep are legitimate references to the
  live `SimulationDetectionProbability` class (Pipeline B) — not Pipeline-A leftovers.
  **Clean. No action needed.**
- **`cosmological_model.py` backward-compat re-export**: CLAUDE.md's architecture section
  says "Backward-compat re-export of `BayesianStatistics`" — this is now **stale
  documentation**, not stale code. The actual file (line ~452) has the re-exports
  **already removed**, with an explanatory comment: *"These symbols were extracted to
  bayesian_inference/ subpackage modules. The re-exports caused a circular import that
  crashed multiprocessing workers in the evaluation pipeline."* So the code is fine; the
  one-line description in CLAUDE.md's Architecture section (`cosmological_model.py —
  ... Backward-compat re-export of BayesianStatistics`) should be updated to match.
  **Severity: LOW (docs only). Effort: S. SAFE-HYGIENE** (single-line CLAUDE.md edit).
- **Ruff F841 (unused local variables), 7 total, F401 (unused imports) = 0**:
  - `physical_relations.py:561-562` — `Omega_de_min`/`Omega_de_max` computed but unused,
    inside a function with the comment `# FOR NOW IGNORE UNCERTAINTIES IN OMEGA_DE AND W`
    — this looks like genuinely vestigial dead code from an abandoned uncertainty
    propagation path. **BEHAVIOR-TOUCHING caution**: this is a physics-trigger file; even
    though removing two obviously-unused locals is mechanically safe, `/physics-change`
    gate territory norms in this repo mean it should go through the standard review even
    for a deletion, since it sits inside a distance-uncertainty bound calculation.
    **Severity: LOW. Effort: S.**
  - `plotting/evaluation_plots.py:91` (`parts = ax.violinplot(...)`), `plotting/interactive.py:1353`
    (`n_metric_traces`), `plotting/interactive.py:1649` (`param_cols`) — all plotting-code
    unused locals, cosmetic, zero behavior risk. **Severity: LOW. Effort: S. SAFE-HYGIENE.**

## 4. Known-bugs status check (CLAUDE.md bug #6)

**Bug #6 — `physical_relations.py` wCDM `w0`/`wa` silently ignored — the guard HAS landed.**
`_reject_unsupported_wcdm()` (line 36) raises `NotImplementedError` whenever `w_0 != -1.0`
or `w_a != 0.0`, and is called from `dist()`, `cached_dist()`, `dist_vectorized()` (and
likely others). Module docstring confirms: *"raises `NotImplementedError` on genuine wCDM
inputs... `hubble_function` does implement the full CPL [form]"*. CLAUDE.md's bug #6 entry
is still marked open/unresolved in the doc — **this is a documentation staleness item**,
not a code problem: the fix shipped (referenced as "the review PR (2026-07-04)" in
CLAUDE.md itself) but the bug-list entry was never struck through like bugs #4, #5, #7-#9.
**Severity: LOW (docs only) — recommend striking bug #6 in CLAUDE.md's Known Bugs list
the same way #4/#5/#7-#9 were struck. Effort: S. SAFE-HYGIENE.**

## 5. Typing / test coverage cold spots

From `coverage.xml` (fresh — generated Sep 2 15:27, same day as latest `bayesian_statistics.py`
commit Sep 2 11:17):

Lowest-coverage production modules:

| Coverage | Module |
|---|---|
| 0.0% | `__main__.py` |
| 0.0% | `callbacks.py` |
| 0.0% | `parameter_estimation/evaluation.py` |
| 9.0% | `plotting/single_event_detail.py` |
| 27.6% | `plotting/convergence_analysis.py` |
| 38.9% | `plotting/interactive.py` |
| 44.5% | `plotting/catalog_plots.py` |
| 46.5% | `plotting/evaluation_plots.py` |
| 50.4% | `parameter_estimation/parameter_estimation.py` |
| 50.8% | `main.py` |
| 74.1% | `bayesian_inference/bayesian_statistics.py` |

Pattern: **plotting/ modules and CLI entry points are the coverage cold spots** — expected
for visualization code (matplotlib figure generation is inherently hard to unit-test
meaningfully) and CLI glue, so this is lower-severity than it looks. The one item worth
flagging: **`parameter_estimation/parameter_estimation.py` at 50.4%** is the module CLAUDE.md
calls out as "the computational bottleneck" (Fisher matrix / SNR / `scalar_product_of_functions`)
— i.e. a physics-trigger file with only half its lines exercised by tests. Per the Testing
Strategy in CLAUDE.md ("Physical correctness" is test-priority #1), this module deserves the
most attention of the cold spots found. **Severity: MED. Effort: M** (needs physics-literate
test authoring, likely BEHAVIOR-TOUCHING-adjacent since new tests may surface existing
formula issues — treat any resulting fix under `/physics-change`).

`bayesian_statistics.py` itself is at a respectable 74.1% for a 9,332-line file, but given
`evaluate()` is 1,621 lines with 74 parameters, a coverage percentage doesn't tell you much
about branch/flag-combination coverage — most of that 74.1% is likely one or two blessed
flag combinations (the production defaults) exercised repeatedly, not the combinatorial
flag space. Not measured directly in this pass (would need branch coverage broken out by
flag path); flagging as a coverage-quality caveat rather than a number.

Did not run a full mypy pass (heavy compute prohibition for this lens); `__init__(self) -> None`
signatures are properly annotated at least at the entry points sampled.

## 6. Flag-sprawl in `BayesianStatistics`

**Confirmed and worse than "reputedly enormous" suggested.** `BayesianStatistics.__init__`
takes **zero arguments** — all configuration lives instead on `evaluate()`'s signature,
which has **~74 parameters**, including (non-exhaustive, matching the lens's own list):
`catalogue_leg_1d_mass_aware`-equivalent (`catalogue_numerator_survival`,
`catalogue_mass_overlap`, `catalogue_mass_error_scale`), `smear_global_selection`,
`pdet_z_resolved`, `pdet_wbh_z_resolved`, `host_z_kernel`, `host_mass_kernel`,
`freeze_g_frac_ref_h`, `selection_in_completion_numerator`, `completion_b_scale`,
`eddington_m`, `completion_event_measure`, `normalization_mode`, `dgen_catalog_selection`,
`fisher_cond_threshold`, `allow_low_pdet_coverage`, `base_seed`, plus many string-valued
mode switches (`"auto"` / `"on"` / `"off"` / named variants) that then get resolved
internally (e.g. `_catalogue_leg_1d_mass_aware: str = "auto"` resolved later based on the
"absolute_marginal phi stack" production flip).

**This is a textbook config-object refactor case**: a `BayesianEvaluationConfig` dataclass
(or similar) holding all ~74 flags with their defaults, constructed once and passed to
`evaluate()`, would (a) let `__init__` actually take config instead of being a no-op shell,
(b) let `evaluate()` shrink from 1,621 lines toward an orchestration-only body, (c) make the
flag space introspectable/loggable in one place (useful for the run-provenance logging
CLAUDE.md already cares about — `run_metadata.json`), and (d) make it possible to unit-test
flag-resolution logic (`"auto"` → resolved value) independent of running the full pipeline.
**Severity: HIGH** (this is the single biggest structural debt item found in this lens —
it is both a maintainability risk and, because `evaluate()` mixes flag-resolution logic
with numerical orchestration in one enormous function body, an increasing risk of a
silent flag-interaction bug going unnoticed). **Effort: L.** **BEHAVIOR-TOUCHING at the
seams** — a pure signature refactor (dataclass wrapping the same defaults, same call
sites) can be done as SAFE-HYGIENE, but any refactor of this scale on the sole production
H0 pipeline should not proceed without author sign-off and a full before/after regression
comparison run, given `/physics-change` gate philosophy on this repo even though no
formula changes — recommend treating as a `/physics-change`-adjacent structural proposal
requiring explicit author [DO] before starting, not a routine cleanup PR.

---

## Top 5 findings (one line each)

1. **[HIGH, L, SAFE-HYGIENE-w/-verification]** `bayesian_statistics.py` is 9,332 lines with one 3,290-line class holding an `evaluate()` method that is 1,621 lines long with ~74 parameters — clear split seams exist (mass-aware catalogue leg, mixture/phi-divisor assembly, worker functions) but any extraction needs a full regression run before trust.
2. **[HIGH, L, BEHAVIOR-TOUCHING-adjacent, needs author DO]** `BayesianStatistics.__init__` takes zero arguments while all ~74 configuration flags live on `evaluate()`'s signature instead — a config-dataclass refactor is the clear fix but is large enough to warrant explicit author sign-off before starting.
3. **[MED, M]** `parameter_estimation/parameter_estimation.py` — the Fisher/SNR "computational bottleneck" module CLAUDE.md flags by name — sits at only 50.4% test coverage, the weakest coverage among physics-trigger files.
4. **[LOW, S, SAFE-HYGIENE]** Two known-bugs items in CLAUDE.md are stale documentation, not live bugs: bug #6's wCDM guard (`_reject_unsupported_wcdm`) has already landed and should be struck through, and the "cosmological_model.py backward-compat re-export" architecture note describes code that was already removed (with an explanatory comment in-file).
5. **[LOW, S, SAFE-HYGIENE]** Dead code is otherwise minimal: zero Pipeline-A remnants, zero unused imports (ruff F401), only 7 unused-local warnings (F841) — the one worth a second look is two unused `Omega_de_min`/`Omega_de_max` locals in `physical_relations.py` (a physics-trigger file) left over from an admittedly-incomplete uncertainty-propagation path.

Overall the codebase is in decent hygiene shape (near-zero TODOs, clean dead-code grep,
guard already landed) — the real debt is concentrated almost entirely in one file's
structure and its flag-passing design, not scattered mess.
