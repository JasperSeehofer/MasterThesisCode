---
phase: quick-260710-pp-coverage-deepvenue-mode
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - master_thesis_code/validation/pp_coverage.py
  - master_thesis_code_test/validation/test_pp_coverage.py
  - results/pp_coverage_deepvenue_20260710/RUNBOOK.md
autonomous: true
requirements: [L-A]   # handoff item L-A, .planning/HANDOFF-LOCAL-NO-CLUSTER-20260710.md

must_haves:
  truths:
    - "With z_support=None (default), the harness produces bit-identical results to current HEAD (golden pin passes)."
    - "z_support >= Z_MAX_POP is identical to z_support=None (limiting case; completion_fraction == 0)."
    - "Setting z_support < Z_MAX_POP routes true hosts with z_host >= z_support into the B_num/D pure-completion branch."
    - "completion_fraction is reported per truth: 0 when disabled, strictly in (0,1) for moderate z_support, and increases as z_support decreases."
    - "At small z_support (~0.05) the posterior stays finite/normalizable (no NaN) and completion_fraction ~= 1."
    - "The CLI exposes --z-support (float, default None)."
    - "A RUNBOOK exists specifying the exact 8-cell + anchor-rerun sweep commands and the SUMMARY.md verdict format for the orchestrator."
  artifacts:
    - path: "master_thesis_code/validation/pp_coverage.py"
      provides: "z_support config field + CLI flag, membership split, B_num/D completion branch, completion_fraction output"
      contains: "z_support"
    - path: "master_thesis_code_test/validation/test_pp_coverage.py"
      provides: "golden pin (z_support=None) + limiting-case + small-z_support + monotonicity tests"
      contains: "z_support"
    - path: "results/pp_coverage_deepvenue_20260710/RUNBOOK.md"
      provides: "orchestrator sweep commands + SUMMARY verdict format"
      contains: "pp_zs"
  key_links:
    - from: "PPCoverageConfig.z_support"
      to: "_run_realization membership split"
      via: "z_host < z_support routes catalogue vs zero-host"
      pattern: "z_host\\s*<\\s*.*z_support|z_support"
    - from: "_run_realization zero-host count"
      to: "run_coverage results[...].completion_fraction"
      via: "returned per-realization completion count aggregated over realizations"
      pattern: "completion_fraction"
    - from: "CLI --z-support"
      to: "PPCoverageConfig(z_support=...)"
      via: "argparse float default None threaded into config"
      pattern: "z_support"
---

<objective>
Extend the independent P–P / coverage harness (`master_thesis_code/validation/pp_coverage.py`)
with a **catalogue-support-truncated mode** (`z_support`) so it can validate the issue-#29
zero-host pure-completion fallback estimator (`p_i = B_num/D`) at deep catalogue
incompleteness — the synthetic closure for handoff item L-A
(`.planning/HANDOFF-LOCAL-NO-CLUSTER-20260710.md`, lines 23–35).

The new mode splits the detected population by the true host redshift: hosts with
`z_host < z_support` are "in the catalogue" and follow the EXISTING single-host kernel branch
(bit-unchanged — mirrors production "hosts-present events undisturbed"); hosts with
`z_host >= z_support` become **zero-host events** whose likelihood is the pure-completion
term `B_num(h)/D(h)` — the exact `L_cat → 0` limit of the Gray mixture that production commit
`8db6c6e` (#29) installed in `bayesian_statistics.py`.

Purpose: measure P–P coverage + MAP bias of the fallback estimator at 60–95% incompleteness
NOW, in a from-scratch synthetic universe, so the eventual cluster re-eval (EXP-40) is a
confirmation rather than a first look. If the fallback is well-calibrated in the closure, the
campaign de-risks a week early; if it is biased, we learn it cheaply.

Output:
- `pp_coverage.py` with a `z_support` knob (None ⇒ bit-identical to current code), CLI flag,
  completion branch, and per-truth `completion_fraction`.
- Extended `test_pp_coverage.py` (golden pin-first + limiting-case + small-z + monotonicity).
- `results/pp_coverage_deepvenue_20260710/RUNBOOK.md` — the orchestrator-run sweep spec.

**NOT a /physics-change.** The harness is deliberately independent of production code (see its
module docstring "Scientific independence"); it re-derives the estimator from the written
formulas. The repo's **pin-test-first** convention still applies (cf. `ed46390 → 8db6c6e`).
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/HANDOFF-LOCAL-NO-CLUSTER-20260710.md
@.planning/BIAS-INVESTIGATION-20260710.md
@master_thesis_code/validation/pp_coverage.py
@master_thesis_code_test/validation/test_pp_coverage.py
@results/pp_coverage_sigmaz_scan_20260703/SUMMARY.md
@docs/derivations/G2a_completion_sky_marginal_4pi.md

<interfaces>
<!-- Key contracts the executor needs. Extracted from pp_coverage.py @ HEAD. Use directly. -->

Module constants (do NOT change):
  Z_MIN = 1e-4
  Z_MAX_POP = 0.95              # population / catalogue redshift ceiling
  OMEGA_M = 0.30, OMEGA_L = 0.70

Helpers already available (reuse; do NOT reimplement):
  comoving_amplitude_of_z(z) -> A(z) [Gpc], with d_L(z,h) = A(z)/h
  z_of_comoving_amplitude(a) -> z at which d_L*h == a
  population_weight_of_z(z)  -> UNNORMALIZED w_pop(z) ∝ dV_c/dz / (1+z)
  detection_probability(d_L) -> p_det in [0,1]
  _norm_pdf(x, mu, sig)      -> Gaussian pdf

Config (dataclass) @ HEAD — ADD z_support here:
  class PPCoverageConfig:
      n_realizations:int=120; n_events:int=250; sigma_z:float=0.035
      sigma_z_pv:float=0.0; sigma_dl_frac:float=0.05
      injected_truths:list[float]=[0.62,0.72,0.84]; seed:int=20260701
      kernel:Literal["bare","volume"]="volume"
      h_min=0.600; h_max=0.860; h_step=0.004; n_z_quad=160
      def h_grid(self) -> npt.NDArray[np.float64]

Inner loop @ HEAD — the single-host branch to preserve, per event i:
  z_lo = max(Z_MIN, z_of_comoving_amplitude((dL_obs[i]-5*sig_dl[i])*h_grid.min()) - 4*sigma_z)
  z_hi = min(_Z_GRID[-1], z_of_comoving_amplitude((dL_obs[i]+5*sig_dl[i])*h_grid.max()) + 4*sigma_z)
  zq   = linspace(z_lo, z_hi, n_z_quad); wq = gradient(zq)
  pGW  = _norm_pdf(A(zq)/h_grid, dL_obs[i], sig_dl[i])          # (nz, nh)
  kernel_z = _norm_pdf(zq, z_gal[i], sigma_z)                    # (nz,)
  if kernel=="volume": kernel_z *= w_pop(zq); kernel_z /= trapz(kernel_z, zq)
  num  = (wq * kernel_z) @ pGW                                  # (nh,)
  logL += log(clip(num, 1e-300, None)) - log_Dh

Shared denominator (already computed once in run_coverage, do NOT change):
  D(h) = trapz( p_det(A(z)/h) * w_pop(z), z )  over z in [Z_MIN, Z_MAX_POP]; log_Dh = log(Dh)
</interfaces>

<production_analog>
<!-- The #29 fallback this harness mode mirrors (commit 8db6c6e, bayesian_statistics.py). -->
Production replaced the silent `if possible_hosts is None: continue` skip with the pure-completion
likelihood p_i = (β_G·0 + B_num)/D = B_num/D — the exact L_cat→0 limit of the Gray mixture.
Refs to cite in docstrings: Gray et al. (2020) arXiv:1908.06050 Eqs. 29+32; Gray, Messenger &
Veitch (2022) arXiv:2111.04629 Eq. 5; docs/derivations/G2a_completion_sky_marginal_4pi.md
limiting case 2; issue #29. The #30 z-cap parallel: the completion integral MUST cap at Z_MAX_POP
(matches the shared D(h) domain).
</production_analog>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Pin the z_support=None behaviour (pin-first commit)</name>
  <files>master_thesis_code_test/validation/test_pp_coverage.py</files>
  <action>
    FIRST commit of the pin-test-first workflow (mirrors ed46390 before 8db6c6e). Add ONE new
    golden-pin test to the existing test module that freezes the CURRENT harness output for a
    tiny config, so the later z_support change proves the default (z_support=None) path is
    bit-untouched.

    Config for the pin (call `PPCoverageConfig(...)` WITHOUT any z_support arg — the field does
    not exist yet at this commit):
      n_realizations=2, n_events=25, injected_truths=[0.72], seed=20260710, kernel="volume".

    Steps:
    1. Run `run_coverage(<that config>)` once via `uv run python -c ...` (do NOT write a
       throwaway script file — inline `-c` only, per the no-ad-hoc-scripts rule) to MEASURE the
       exact `results["0.7200"]` values: `map_mean`, `map_std`, `map_bias`, and
       `coverage["50"/"68"/"90"]`, `rail_fraction`.
    2. Add `test_z_support_none_golden_pin()` that constructs the same config, runs
       `run_coverage`, and asserts the measured values with `pytest.approx(rel=1e-12)` for the
       float stats and exact `==` for the fractional coverages/rail (they are rationals like k/2).
       Docstring: "Golden pin measured at HEAD; the z_support=None path MUST stay bit-identical
       after the truncated-mode change (issue #29 harness validation, pin-first per ed46390)."
    3. Do NOT modify pp_coverage.py in this task. Do NOT touch the existing
       `test_tiny_config_exact_value_pins` — it stays as an additional guard.

    Typing/style: full annotations (`-> None`), NumPy docstring, no `from __future__ import
    annotations`. Pre-commit runs whole-tree mypy — an untyped test blocks ALL commits.
  </action>
  <verify>
    <automated>uv run pytest master_thesis_code_test/validation/test_pp_coverage.py -k "golden_pin" -x -q</automated>
  </verify>
  <done>New golden-pin test passes at HEAD with hard-coded expected values; pp_coverage.py unchanged; ruff+mypy clean on the test file.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Add the z_support truncated mode (B_num/D completion branch) + tests</name>
  <files>master_thesis_code/validation/pp_coverage.py, master_thesis_code_test/validation/test_pp_coverage.py</files>
  <behavior>
    - Golden pin from Task 1 STILL passes byte-for-byte (z_support=None path unchanged; NO new
      RNG draw is consumed in either branch — membership is a comparison on the already-sampled
      z_host, and B_num reuses dL_obs/sig_dl).
    - Test (b) limiting case: z_support = Z_MAX_POP (0.95) gives results == z_support=None, and
      completion_fraction == 0.0 (z_host is sampled in [Z_MIN, Z_MAX_POP], so all hosts are
      catalogue hosts).
    - Test (c) small z_support (0.05): completion_fraction > 0.9; map_mean/map_std/coverage all
      finite (no NaN/inf); posterior normalizable (map_mean within [h_min, h_max]).
    - Test (d) monotonic membership split: with moderate z_support the completion_fraction is
      strictly in (0,1) and increases as z_support decreases — assert
      0 < cf(z_support=0.5) < cf(z_support=0.2) < 1 (same seed/truth, tiny config).
  </behavior>
  <action>
    Implement the locked estimator design EXACTLY (do NOT redesign):

    (1) Config knob — add to `PPCoverageConfig`:
        `z_support: float | None = None`  with a NumPy-docstring line: "Catalogue support
        ceiling: true hosts with z_host < z_support are in the catalogue (existing single-host
        kernel branch); z_host >= z_support are zero-host events using the pure-completion
        likelihood B_num/D (issue #29 analog). None (default) ⇒ no truncation, bit-identical to
        the pre-2026-07-10 harness."
        `asdict(config)` will then serialize `z_support` (expected; see Task 3 anchor note).

    (2) CLI flag in `main()`:
        `parser.add_argument("--z-support", type=float, default=None)` and thread
        `z_support=args.z_support` into the `PPCoverageConfig(...)` construction.

    (3) Membership split + completion branch in `_run_realization` — change its return type to
        `tuple[npt.NDArray[np.float64], int]` (logL, n_zero_host). After z_host/z_gal/dL_obs/
        sig_dl are drawn (UNCHANGED sampling), for each event i:
          - `is_zero_host = (config.z_support is not None) and (z_host[i] >= config.z_support)`
          - Catalogue host (not is_zero_host): the EXISTING single-host kernel block, verbatim.
          - Zero-host event: pure-completion B_num(h)/D(h). Build the integration domain WITHOUT
            the kernel's ±4σ_z padding (no kernel here) and cap at Z_MAX_POP (the #30 parallel):
              z_lo_b = max(Z_MIN, config.z_support,
                           float(z_of_comoving_amplitude(np.asarray((dL_obs[i]-5*sig_dl[i])*h_grid.min()))))
              z_hi_b = min(Z_MAX_POP,
                           float(z_of_comoving_amplitude(np.asarray((dL_obs[i]+5*sig_dl[i])*h_grid.max()))))
            If z_hi_b <= z_lo_b (empty domain), set `num_b = np.full(h_grid.size, 1e-300)`
            (skip quadrature — avoids a degenerate linspace; the existing 1e-300 clip semantics).
            Else:
              zq_b = np.linspace(z_lo_b, z_hi_b, config.n_z_quad); wq_b = np.gradient(zq_b)
              dLg_b = comoving_amplitude_of_z(zq_b)[:, None] / h_grid[None, :]
              pGW_b = _norm_pdf(dLg_b, float(dL_obs[i]), float(sig_dl[i]))      # (nz, nh)
              wpop_b = population_weight_of_z(zq_b)                              # UNNORMALIZED
              num_b = (wq_b * wpop_b) @ pGW_b                                    # (nh,)
            Then `logL += np.log(np.clip(num_b, 1e-300, None)) - log_Dh` (SAME log_Dh denominator
            as the single-host branch — B_num and D share the exact same unnormalized measure;
            do NOT insert any h-dependent normalization). Increment a local `n_zero_host` counter.
          - `_run_realization` returns `(logL, n_zero_host)`.

    (4) Aggregate in `run_coverage`: unpack `logL, n_zero_host = _run_realization(...)`; collect
        per-realization `n_zero_host / config.n_events` and store the mean as a new per-truth
        result key `"completion_fraction"` (float). ALL existing metrics (coverage 50/68/90,
        rail_fraction, map_mean/std/median/bias) stay computed over ALL events exactly as now.

    (5) Docstrings: update `_run_realization` and the module header to describe the completion
        branch, citing Gray et al. (2020) arXiv:1908.06050 Eqs. 29+32; Gray, Messenger & Veitch
        (2022) arXiv:2111.04629 Eq. 5; docs/derivations/G2a_completion_sky_marginal_4pi.md
        limiting case 2; issue #29. Optionally add `completion_fraction` to the CLI print line.

    (6) Tests — add (b), (c), (d) from <behavior> to test_pp_coverage.py (tiny configs, fast, NOT
        @slow). For (b) prefer an exact-equality assertion on the two `results` dicts. Keep
        determinism-friendly small sizes (e.g. n_realizations 4–8, n_events 25–40). Do NOT weaken
        Task 1's golden pin or the existing pins.

    Style: `float | None`, `list[float]`, `npt.NDArray[np.float64]`, no
    `from __future__ import annotations`; NumPy docstrings; ruff + whole-tree mypy clean.
  </action>
  <verify>
    <automated>uv run ruff check master_thesis_code/validation/pp_coverage.py master_thesis_code_test/validation/test_pp_coverage.py && uv run ruff format --check master_thesis_code/validation/pp_coverage.py master_thesis_code_test/validation/test_pp_coverage.py && uv run mypy master_thesis_code/validation/pp_coverage.py && uv run pytest master_thesis_code_test/validation/test_pp_coverage.py -m "not gpu and not slow" -q</automated>
  </verify>
  <done>z_support field + --z-support CLI flag exist; z_host >= z_support routes into the B_num/D branch capped at Z_MAX_POP; completion_fraction reported per truth; golden pin + limiting-case + small-z + monotonicity tests pass; z_support=None bit-identical to HEAD; ruff+mypy clean.</done>
</task>

<task type="auto">
  <name>Task 3: Write the orchestrator sweep RUNBOOK + SUMMARY verdict format</name>
  <files>results/pp_coverage_deepvenue_20260710/RUNBOOK.md</files>
  <action>
    Create the deliverable directory and write `RUNBOOK.md` specifying the sweep the ORCHESTRATOR
    runs AFTER this plan merges (the executor does NOT run the sweep — cells are ~120×250 and
    minutes each). The runbook is the single source the orchestrator follows; it also fixes the
    SUMMARY.md format the orchestrator fills in post-sweep.

    Contents:

    A. Sweep grid — 8 cells = z_support ∈ {0.2, 0.3, 0.5, 1.0} × σ_z ∈ {0.015, 0.035},
       kernel=volume, defaults otherwise (n_realizations=120, n_events=250,
       truths [0.62, 0.72, 0.84], seed 20260701). z_support=1.0 (> Z_MAX_POP=0.95) is the
       untruncated CONTROL at each σ_z (completion_fraction ≡ 0). Per-cell command template:

         uv run python -m master_thesis_code.validation.pp_coverage \
           --n-realizations 120 --n-events 250 --sigma-z {SZ} --z-support {ZS} \
           --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
           --output results/pp_coverage_deepvenue_20260710/pp_zs{ZS}_sz{SZ}_volume.json \
           2>&1 | tee results/pp_coverage_deepvenue_20260710/pp_zs{ZS}_sz{SZ}_volume.log

       Enumerate all 8 concrete commands (zs∈{0.2,0.3,0.5,1.0} × sz∈{0.015,0.035}); outputs
       named pp_zs{ZS}_sz{SZ}_volume.json + .log.

    B. Anchor bit-identity re-run — reproduce the committed anchor config
       (n_realizations=250, n_events=250, sigma_z=0.10, kernel=volume, seed=20260701, NO
       z_support) to `pp_sigmaz0.10_volume_rerun.json`, then diff its `results` object against
       `results/pp_coverage_sigmaz_scan_20260703/pp_sigmaz0.10_volume.json`:

         uv run python -m master_thesis_code.validation.pp_coverage \
           --n-realizations 250 --n-events 250 --sigma-z 0.10 \
           --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
           --output results/pp_coverage_deepvenue_20260710/pp_sigmaz0.10_volume_rerun.json
         diff <(jq -S .results results/pp_coverage_deepvenue_20260710/pp_sigmaz0.10_volume_rerun.json) \
              <(jq -S .results results/pp_coverage_sigmaz_scan_20260703/pp_sigmaz0.10_volume.json)

       Note in the runbook: the `.results` block MUST be byte-identical (proves z_support=None is
       a no-op); the `.config` block legitimately gains the `sigma_z_pv` and `z_support` keys
       (added since the anchor was generated) — that difference is EXPECTED, not a regression, so
       diff `.results` only.

    C. SUMMARY.md verdict format (template the orchestrator fills post-sweep):
       - Per-cell × truth table columns: z_support, σ_z, h_true, cov50, cov68, cov90,
         rail_fraction, MAP mean, MAP bias, completion_fraction.
       - For each truncated cell (zs ∈ {0.2,0.3,0.5}) a comparison against its z_support=1.0
         control AT THE SAME σ_z.
       - Verdict criteria:
         * coverage collapse ⇒ cov68 falls outside ±2·SE ≈ ±0.086 of the control (n=120,
           2·sqrt(0.68·0.32/120) ≈ 0.085).
         * bias flag ⇒ |Δ map_mean vs control| > 2·SEM (SEM = map_std/√120).
       - Carried caveats (state verbatim in the SUMMARY):
         1. 1D-channel only — the 2D (+0.057) question is NOT covered by this harness.
         2. Single-host clean limit — production host-found events ALSO carry a B_num admixture
            in the mixture; this harness omits that, so ONLY the zero-host branch is the exact
            production analog.
         3. Hard truncation (z_support step) vs production's soft M_BH-prune truncation of the
            effective catalogue.

    Do NOT run any sweep command in this task — only author the runbook. Keep it markdown, cite
    issue #29 and the handoff L-A item as provenance.
  </action>
  <verify>
    <automated>test -f results/pp_coverage_deepvenue_20260710/RUNBOOK.md && grep -q "pp_zs" results/pp_coverage_deepvenue_20260710/RUNBOOK.md && grep -q "z_support=1.0" results/pp_coverage_deepvenue_20260710/RUNBOOK.md && grep -Eq "0.08[56]" results/pp_coverage_deepvenue_20260710/RUNBOOK.md</automated>
  </verify>
  <done>RUNBOOK.md exists with all 8 sweep commands, the anchor bit-identity re-run + .results-only diff note, and the SUMMARY verdict format (table columns, control comparison, ±2·SE / 2·SEM criteria, 3 caveats). No sweep executed.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| CLI args → harness | `--z-support` is a single float coerced by `argparse type=float`; no code path, filesystem, or network input crosses. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-ppcov-01 | Tampering | `--z-support` CLI float | accept | Pure synthetic numerics, local dev-only harness, no untrusted input, no persisted secrets; `type=float` rejects non-numeric. |
| T-ppcov-02 | Information disclosure | JSON/log outputs under `results/` | accept | Outputs are synthetic coverage stats only — no PII, no credentials. |
</threat_model>

<verification>
- `uv run pytest master_thesis_code_test/validation/test_pp_coverage.py -m "not gpu and not slow" -q` passes (golden pin + limiting-case + small-z + monotonicity + all pre-existing tests).
- `uv run ruff check master_thesis_code/validation/ master_thesis_code_test/validation/` and `uv run ruff format --check ...` clean.
- `uv run mypy master_thesis_code/validation/pp_coverage.py` clean (whole-tree mypy runs on pre-commit).
- `--z-support` present in `--help`; z_support=None serializes into the config dict.
- `results/pp_coverage_deepvenue_20260710/RUNBOOK.md` exists with the 8 sweep commands, anchor re-run, and SUMMARY format.
- Run `/check` (ruff + mypy + pytest quality gate) before committing.
</verification>

<success_criteria>
- z_support=None is bit-identical to current HEAD (golden pin passes; anchor `.results` re-run
  diff empty when the orchestrator runs it).
- z_support < Z_MAX_POP routes z_host >= z_support events into `B_num(h)/D(h)`, integral capped
  at Z_MAX_POP, sharing the exact unnormalized measure of D(h) (no h-dependent normalization).
- completion_fraction reported per truth: 0 at z_support≥Z_MAX_POP, strictly in (0,1) for
  moderate z_support and monotonically increasing as z_support decreases, ~1 at z_support≈0.05.
- Posterior stays finite/normalizable at deep truncation (no NaN).
- RUNBOOK.md gives the orchestrator an unambiguous, ready-to-run sweep + verdict format.
- ruff + mypy + pytest (not gpu, not slow) all green.
</success_criteria>

<output>
After completion, create `.planning/quick/260710-sjm-pp-coverage-deepvenue-mode/260710-sjm-SUMMARY.md`.
</output>
