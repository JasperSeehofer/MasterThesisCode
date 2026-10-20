# T2.3 mass-aware 1D catalogue leg instrument — INDEPENDENT VERIFIER REPORT

Launched under row #255 — tree 2 node T2.3 (verifier pass, row #255 A17 / charter section 4
mandatory scope). Verifier is a different agent from the presenter and the builder. No git, no
ssh, foreground-only commands, no edits to any file under `results/.../tree2_20260830/` other than
this report. Evidence below is reproduced today (2026-08-30) from the working tree at HEAD
`5e1e66aa` with the uncommitted T2.3 diff on top (`darksiren_emri/bayesian_inference/bayesian_statistics.py`,
`arguments.py`, `main.py`, `validation/correspondence_1d.py`,
`results/.../fanout1_20260829/hier_s0_driver.py`, new test file, two `docs/gates/PHYSICS-GATE-LEDGER.md`
rows).

## Verdict table

| # | Item | Verdict | Notes |
|---|---|---|---|
| 1 | Every hunk inside presented scope; "on" path implements the presented objects (S_4D per candidate, Sigma_4D divisor, alpha_G_phi weight) formula-by-formula | **PASS** | Site D1 (`global_denom_no_bh` ternary, now at working-tree lines 6019/6027): "on" reads `global_denom_with_bh` (Sigma_4D, already computed) verbatim, "off" falls through to the pre-existing ternary unchanged — matches §2.2. Site W1 (`_cat_num_weight_no_bh`, lines 6621/6626-6627): "on" selects `alpha_G_phi` (the identical float the 2D assembly at the next lines consumes), "off" selects `beta_G_phi` unchanged — matches §2.3. Site N1 batch (point path :8361, quadrature path :8383) and scalar twin (quadrature :7547, point :7608) both replace the `np.interp(..., catalogue_survival_table)` factor by `catalogue_leg_1d_mass_aware_factor(...)` under "on" only — matches §2.2. New helper `catalogue_leg_1d_mass_aware_factor` (line 6961): "point" sub-form computes `d_L = dist_vectorized(z,h)` and `M_z = M_g*(1+z)` then calls `detection_probability_with_bh_mass_interpolated(d_L, M_z, 0, 0, h=h, **_wbh_z_kwargs(...))` — byte-for-byte the same accessor, isotropic-sky convention (`phi_iso=theta_iso=zeros_like`) and z-rider Sigma_4D's own with-BH point branch uses (verified by direct grep at the with-BH `precompute_global_catalog_selection` site: `d_L_g = dist_vectorized(z_g,h=h)`, `phi_iso=np.zeros_like(z_g)`, `theta_iso=np.zeros_like(z_g)`, same `_wbh_z_kwargs` call). "kernel" sub-form mirrors `_sigma4d_mass_kernel_expectation` and `_eddington_shifted_host_mass_batch` with the same signature/argument order Sigma_4D's own kernel branch uses. Guards (evaluate()-level, 4 total: `catalogue_numerator_survival=="phi"`, `catalogue_global_selection=="phi"`, `theta_phi_divisor=="off"`, token check) and the worker-level defence-in-depth guard (`catalogue_leg_1d_mass_aware=="on"` requires `_cat_surv_on`) are present exactly as registered in §2.1/§2.2. Minor informational note (not a defect): §2.2 states the point-path factor is "the T2.2 column `s_4d_zg_mg` exactly" — the built helper instead matches **Sigma_4D's own point branch** exactly (it includes `**_wbh_z_kwargs(...)`, which the T2.2 diagnostic column at line ~5462 omits). This is the *correct* choice — matching Sigma_4D, not the T2.2 diagnostic, is what the §2.3 exact-identity chain (`alpha_G_phi/Sigma_4D == beta_G_phi/Sigma_phi`) actually requires — and is inert for the registered arms today (`wbh_z_resolved` is not set by the FT venue in `correspondence_1d.py`, confirmed by grep — no hits). Flag as a presentation-wording imprecision only, no code fix needed. |
| 2 | Byte-identity at "off" | **PASS** | R1 unit tests (kwarg-omitted vs. explicit `"off"`, both scalar and batch) pass. Independently re-ran the full pre-existing suite (1941 tests, see item 4) with zero regressions — the strongest evidence available that every pre-flag numeric baseline is untouched. Live smoke run (`--config ft --smoke --nodes truth --event-cap 12`, `catalogue_leg_1d_mass_aware=off`) executes cleanly end-to-end. |
| 3 | Z = 1 under "on" | **PASS** | R2 unit test (self-contained 200-galaxy synthetic fixture, r_Malm = 0.850, an informative can-fail-control value, using the REAL `path_a_mixture_objects`): confirms ∫p_i dd = 1.0 to atol 1e-10 under "on" and equals D_phi/D_tilde_phi (≠1, verified) under "off" — reproduced, passes. Additionally ran a **live event-cap-12 driver smoke** at h=0.73, `--config ft`, seed 900101, both flag values (see command/output below): `combined_with_bh` is bit-identical between "off" and "on" on every one of 11 events (`max_abs_diff = 0.0`), and `combined_no_bh`/`L_cat_no_bh` change on 9 of 11 events (`max_abs_diff` 0.0089 / 0.045 respectively) — the 2 unchanged rows are the expected empty-ball (dark, no in-catalogue candidate) case, consistent with the registered L4 limit. This is a genuine, non-synthetic confirmation of the with-BH/without-BH isolation and of engagement in the expected direction, on top of the algebraic R2 pin. |
| 4 | Test/ruff/mypy counts reproduced | **PASS** | New file alone: 26 passed (reproduced). `darksiren_emri_test/bayesian_inference`: 617 passed / 6 skipped (reproduced exactly). `darksiren_emri_test/validation` + `test_arguments.py` combined: 429 passed / 1 skipped = 402+27 (report's split reproduces exactly when summed). Full suite in two halves: half A (analysis, bayesian_inference, datamodels, fixtures, integration, parameter_estimation, plotting) = 845 passed / 6 skipped / 15 deselected (reproduced exactly); half B (validation, scripts, top-level `*.py`) = 1096 passed / 9 skipped / 15 deselected (reproduced exactly). Combined 1941 passed / 15 skipped / 30 deselected — matches the report bit-for-bit. `ruff check darksiren_emri/ darksiren_emri_test/`: all checks passed. `ruff format --check`: 215 files already formatted (no diff). `mypy darksiren_emri/`: Success, no issues found in 70 source files. |
| 5 | Gate-ledger rows match the diff | **FAIL (must_fix)** | The "implemented" row (descriptive, no line citations) matches the diff correctly. The **"verified" row cites STALE, pre-implementation line numbers** that do not correspond to the actual code locations in the built file: it states `bayesian_statistics.py:8079-8089` (site N1 batch), `:7319-7325`/`:7366-7372` (site N1 scalar), `:5936-5943`/`:6019-6021` (site D1), `:6520-6525` (site W1) — but a direct `sed`/`grep` of the current working tree shows these exact ranges are now unrelated pre-existing code (global-declaration boilerplate, an old "volume_trunc" comment, an old "In-catalogue weighted sums" comment, an old "Single ratio p_i" comment respectively), while the actual T2.3 code sits at `:6961` (helper), `:6019`/`:6027` (site D1, reordered), `:6621`/`:6626-6627` (site W1), `:7547`/`:7608` (site N1 scalar, quadrature/point), `:8361`/`:8383` (site N1 batch, point/quadrature) — the SAME numbers the builder's own implementation record (gate doc §20.1) correctly derived and stated. The presentation's own §10 instruction was explicit that ledger citations must be "working-tree numbers **at build time**," and §17.2's own re-verification precedent in this same document shows the convention of re-grepping before filing was known and applied elsewhere — it was simply not applied when the "verified" ledger row text was composed, which instead copied the presentation's original (pre-code) citation string verbatim. This does not indicate any defect in the physics or the code (items 1-4 above pass independently) but it breaks the audit-trail property the physics-gate ledger exists for (CLAUDE.md: "a `[PHYSICS]` commit with no ledger row is a gate that cannot be shown to have run" — a ledger row whose evidence pointers do not point at the described code is equally not showable). **must_fix**: correct the "verified" row's line citations to `:6961` / `:6019,:6027` / `:6621,:6626-6627` / `:7547,:7608` / `:8361,:8383` (or append a dated correction note, append-only, per this document's own established revision-note convention). |

## Evidence commands (reproduced today, foreground, read-only except the two scratch smoke runs)

```
uv run pytest darksiren_emri_test/bayesian_inference/test_catalogue_leg_1d_mass_aware.py -q --no-cov
  -> 26 passed

uv run pytest darksiren_emri_test/bayesian_inference -q --no-cov
  -> 617 passed, 6 skipped

uv run pytest darksiren_emri_test/validation darksiren_emri_test/test_arguments.py -q --no-cov
  -> 429 passed, 1 skipped   (= 402 + 27, matches the report's separate counts summed)

uv run pytest darksiren_emri_test/analysis darksiren_emri_test/datamodels darksiren_emri_test/fixtures \
  darksiren_emri_test/integration darksiren_emri_test/parameter_estimation darksiren_emri_test/plotting \
  -q --no-cov -m "not gpu and not slow"
  -> 228 passed, 15 deselected   (+ bayesian_inference's 617/6skip = half A total 845/6skip/15deselect)

uv run pytest darksiren_emri_test/validation darksiren_emri_test/scripts darksiren_emri_test/*.py \
  -q --no-cov -m "not gpu and not slow"
  -> 1096 passed, 9 skipped, 15 deselected   (half B total)

uv run ruff check darksiren_emri/ darksiren_emri_test/          -> All checks passed!
uv run ruff format --check darksiren_emri/ darksiren_emri_test/ -> 215 files already formatted
uv run mypy darksiren_emri/                                     -> Success: no issues found in 70 source files

# Live smoke, "off" then "on" (event-cap 12, --config ft, seed 900101, --smoke --nodes truth):
uv run python results/.../fanout1_20260829/hier_s0_driver.py --arm S0-A --config ft --smoke \
  --nodes truth --event-cap 12 --catalogue-leg-1d-mass-aware off --out-root <scratch>/off
uv run python results/.../fanout1_20260829/hier_s0_driver.py --arm S0-A --config ft --smoke \
  --nodes truth --event-cap 12 --catalogue-leg-1d-mass-aware on  --out-root <scratch>/on

# Comparison (pandas), off vs on, s0a_seed900101/node_truth[_ft_ma1d]/simulations/diagnostics/event_likelihoods.csv:
  combined_no_bh:   max_abs_diff = 0.008851411030674602, 9 of 11 rows differ
  combined_with_bh: max_abs_diff = 0.0,                  0 of 11 rows differ   <- with-BH channel untouched
  L_cat_no_bh:      max_abs_diff = 0.0449908243097468,   9 of 11 rows differ
  L_cat_with_bh:    max_abs_diff = 0.0,                  0 of 11 rows differ
```

The `gate_parity` "COMPARED / pass_exact=false" block both runs print is the driver's own truth-node-vs-banked-`b0i`-CSV
comparison (`hier_s0_driver.py` `gate_parity()`), which is expected to mismatch here because both runs use `--config ft`,
not the banked `b0i` config it compares against — identical in both arms, unrelated to this flag, not a regression.

## must_fix summary

1. **(must_fix, documentation-only)** Correct `docs/gates/PHYSICS-GATE-LEDGER.md`'s T2.3 "verified" row line
   citations — they currently reproduce the presentation's pre-implementation numbers rather than the
   as-built ones the same builder's own §20.1 implementation record already derived correctly. No code change
   is implicated; the physics, guards, byte-identity, Z=1 identity, and all reported test/lint/type counts are
   independently confirmed correct (items 1-4, all PASS).

No other must_fix items. Items 1-4 all PASS on independent re-derivation/re-execution.
