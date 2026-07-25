# `max_redshift` CLI semantics audit (issue #30 prep)

Read-only audit, branch `feat/max-redshift-cli`, 2026-07-25. Traces what setting
`Model1CrossCheck.max_redshift = 0.3` (analysis-depth truncation) would do in the
`--evaluate` pipeline, end to end.

## 1. Where `max_redshift` currently enters

- `cosmological_model.py:185` — `Model1CrossCheck._apply_model_assumptions()`
  hardcodes `self.max_redshift = 1.5`. **Not a constructor parameter today** —
  set unconditionally during `__init__`.
- `cosmological_model.py:189-194` — construction-time guard:
  `if HOST_DRAW_Z_MAX > self.max_redshift: raise ValueError(...)`.
  `HOST_DRAW_Z_MAX = 1.5` (`constants.py:99`). This guard runs *inside*
  `_apply_model_assumptions`, i.e. **before** any post-construction override
  could take effect.
- `cosmological_model.py:201-203` — immediately after, the parameter-space
  `luminosity_distance.upper_limit` is derived from `self.max_redshift`:
  `dist(redshift=self.max_redshift, h=H_MIN/100.0)`. Also computed before any
  post-construction override.
- `main.py:105-110` — `cosmological_model = Model1CrossCheck(rng=rng)` is
  constructed once, then `GalaxyCatalogueHandler(..., z_max=cosmological_model.max_redshift)`
  is built immediately after, for **both** the simulation and evaluate paths.
- `bayesian_statistics.py:1314` — `REDSHIFT_UPPER_LIMIT = cosmological_model.max_redshift`,
  the single source of truth threaded into the evaluate pipeline's selection
  integrals and numerator z-window (see below).

**Implication for CLI threading**: overriding `cosmological_model.max_redshift`
*after* `Model1CrossCheck(rng=rng)` returns (the naive plumbing approach) would
skip both the `HOST_DRAW_Z_MAX` consistency guard and the `luminosity_distance.upper_limit`
recompute — those already ran against the hardcoded 1.5. A correct implementation
must either (a) pass the override into the constructor before
`_apply_model_assumptions` runs, or (b) re-run both derived steps after the
override. This is code structure, not a formula change (see decision below).

## 2. Numerator side (per-event candidate host window)

- `bayesian_statistics.py:1849-1859` (`p_D`): per event, `z_min, z_max` come from
  `get_redshift_outer_bounds(...)` (±2σ in `h`/`Ω_m` around the d_L posterior),
  then **`z_max = min(z_max, redshift_upper_limit)`** (line 1859) — this is the
  cap the task context refers to. `z_min` is **not** capped (there is no floor
  analogue needed since `max_redshift` is an upper cut).
- `galaxy_catalogue/handler.py:106-110` (via `main.py`) — the galaxy catalog
  itself is pruned to `z < cosmological_model.max_redshift` at
  `GalaxyCatalogueHandler.__init__` → `_get_pruned_galaxy_catalog` (line 254,
  260-275). So with `max_redshift=0.3`, the BallTree the per-event query hits
  physically cannot return a galaxy at z ≥ 0.3, independent of the line-1859
  cap — the two are redundant but consistent.
- `galaxy_catalogue/handler.py:463-497` (`get_possible_hosts_from_ball_tree`):
  redshift filter mask uses `z_min`/`z_max` band-overlap logic (lines 467-475).
  **Fate of an event whose entire candidate window lies above 0.3**: `z_min`
  (uncapped) can exceed `z_max` (capped at 0.3), producing an inverted
  `[z_min, z_max]` range. The mask logic (independent threshold conditions, not
  a true interval check) then almost always empties out for realistic catalog
  redshift-error bars, and `get_possible_hosts_from_ball_tree` returns `None`
  (line 495-497) — **no crash, no NaN**.
- `bayesian_statistics.py:1874-1899` (`p_D`): `possible_hosts is None` (and not
  `catalog_only` mode) routes to the **issue #29 zero-host pure-completion
  fallback**: `candidate_hosts = []`, and downstream `p_Di` produces
  `p_i = B_num(h) / D(h)` (the `L_cat → 0` limit of the mixture). This is the
  documented, exercised path — confirmed no crash/exception for
  `max_redshift < z_max(h)`.

## 3. Selection side — D(h), beta_Gbar(h), Sigma_global(h)

All three take `z_max_cap` and apply it identically to the numerator's line
1859 pattern:

- `precompute_completion_denominator` (D(h)): `bayesian_statistics.py:688-698,735`
  — `z_max = dist_to_redshift(dl_max, h=h); z_max = min(z_max, z_max_cap)`.
- `precompute_missing_completion_denominator` (beta_Gbar(h)):
  `bayesian_statistics.py:831-835,883` — same pattern.
- `precompute_global_catalog_selection` (Sigma_global(h)):
  `bayesian_statistics.py:986-994` — same pattern, plus explicit
  `eligible = (z_all < z_max) & ...` filter over all catalog galaxies.
- All three are invoked with `z_max_cap=REDSHIFT_UPPER_LIMIT` at
  `bayesian_statistics.py:1374,1388,1396,1403`.

**These are consistent with the numerator's candidate-host window**: same cap
value, same `min(z_max(h), max_redshift)` semantics.

## 4. CONSISTENCY GAP FOUND — completion-numerator integration window (B_num) is NOT capped

`bayesian_statistics.py:2144-2155` (inside `p_Di`), the per-event completion
numerator `B_num(h) = ∫ (1-f(z)) p_GW(z) dVc/(1+z) dz` integrates over

```python
z_upper = dist_to_redshift(self.detection.d_L + 4·self.detection.d_L_uncertainty, h=self.h)
z_lower = dist_to_redshift(self.detection.d_L - 4·self.detection.d_L_uncertainty, h=self.h)
z_lower = max(z_lower, 1e-6)
```

(then `fixed_quad(completion_numerator_integrand, z_lower, z_upper, n=FIXED_QUAD_N)` at
line 2223-2225). **`z_upper` is never clamped to `redshift_upper_limit` /
`max_redshift`.** Every other integral in the pipeline that shares this same
functional form — D(h), beta_Gbar(h), Sigma_global(h), and the candidate-host
window at line 1859 — applies `min(z_max, max_redshift)`. B_num does not.

**Why this matters at `max_redshift=1.5` (current default, a no-op today)**:
z_max(h) ≤ ~1.33 for h ∈ [0.60, 0.86] (comment at `bayesian_statistics.py:691-694`),
so the 4σ `z_upper` for any real event essentially never reaches 1.5 anyway —
the gap is invisible under current production settings.

**Why this matters at `max_redshift=0.3`**: any event whose 4σ d_L window
extends past z=0.3 (the majority of the 3454-event seed1000 venue, given the
EMRI rate model's detectable range) still has `B_num` integrate its *entire*
uncapped ±4σ window — including the population density beyond z=0.3 — while
the denominator `D(h)` it is divided by (`p_i = (beta_G·L_cat + B_num)/D(h)`,
line 2242-2245) has been truncated at z=0.3. For the pure-completion fallback
events (§2, `B_num/D(h)` with `L_cat=0`), this means the completion term is
**not depth-truncated at all** — it still integrates the full localization
volume. This directly undermines the issue #30 experiment: the intended
"analysis-depth truncation" would only bite the in-catalog channel (via the
pruned catalog + capped candidate window) and the selection normalization
(D/beta_Gbar/Sigma_global), but NOT the per-event completion numerator that
increasingly dominates as `max_redshift` shrinks and zero-host fallbacks
proliferate. The measured H0 posterior under `max_redshift=0.3` would not
represent a clean "z<0.3 only" analysis — it would be a hybrid where the
in-catalog/selection side is truncated at 0.3 but the completion numerator
still draws on the full injected population out to z(4σ), silently mismatched
against its own denominator.

**This requires a computed-value change** (capping `z_upper` in `p_Di`'s B_num
integration, analogous to `z_max = min(z_max, redshift_upper_limit)` at line
1859) to make the z-cut self-consistent. Per the physics-change protocol, that
change needs the full Old/New formula + reference + dimensional analysis +
limiting case presentation and explicit user approval before any code is
written — it is out of scope for this audit.

## 5. Other places checked, no gap found

- `bayesian_statistics.py:1319-1330` — `SimulationDetectionProbability(...,
  expected_z_max=HOST_DRAW_Z_MAX, ...)`: this is a **pool-sufficiency gate**
  (issue #20 stale-pool guard), checking the *injection pool* was generated
  deep enough — it validates against the fixed `HOST_DRAW_Z_MAX=1.5` constant,
  not against the analysis-depth cap, and should NOT change with
  `max_redshift`; capping it would falsely fail the coverage gate for a
  legitimately shallower *analysis*.
- p_det grid domain (`simulation_detection_probability.py:769-770,1104`):
  independent of `max_redshift` — built from the injection pool's own
  `dl_max`, unrelated to the analysis truncation. No gap: integrals that use
  `z_max_cap` simply stop short of the grid's full extent; p_det is never
  evaluated beyond the (now-smaller) integration bound.
- `galaxy_catalogue/handler.py:588,652` (`draw_uniform_hosts`,
  `draw_rate_weighted_hosts`, default `z_max=HOST_DRAW_Z_MAX`): used only by
  the **injection/simulation** pipeline (event generation), not by
  `--evaluate` on an existing CRB CSV. Not exercised by an evaluate-only
  cluster run. Flagged for awareness only if `max_redshift` is ever also
  threaded into fresh simulation runs.
- `physical_relations.py:282-319` (`luminosity_distance_prescreen_gpc`): also
  simulation-pipeline only (pre-screens waveform generation), not exercised by
  `--evaluate`.
- Completeness model `f_k(z, pixel, h)` (`pixel_completeness.py`): evaluated
  pointwise on whatever quadrature grid it's given; not itself bounded by
  `max_redshift`, so it doesn't need special-casing — the gap is purely about
  the *integration bounds* passed to it (§4).

## 6. Expected `n_events_used` change (qualitative)

Setting `max_redshift=0.3` does **not** remove events from
`self.cramer_rao_bounds` / `posterior_data` — the `--evaluate` loop
(`p_D`, `bayesian_statistics.py:1833-1957`) still iterates and produces a
likelihood for every event that survives SNR + Fisher-quality filtering
(unchanged, upstream of `max_redshift`). What changes:

- The **galaxy catalog** available for host lookup shrinks to `z<0.3`
  (§2), so most events with `z_min > 0.3` (candidate window entirely beyond
  cut) get **zero catalog hosts** → route through the issue #29
  pure-completion fallback (`L_cat=0`, `p_i = B_num/D(h)`). Given the current
  zero-host rate is already 58-60% at the (no-op) depth-1.5 setting per the
  in-code comment (`bayesian_statistics.py:1947-1957`), expect the zero-host
  fraction at `max_redshift=0.3` to rise sharply — likely a large majority of
  the 3454-event seed1000 venue, since detectable EMRI redshifts commonly
  extend well past 0.3.
- `D(h)`, `beta_Gbar(h)`, `beta_G(h)`, `Sigma_global(h)` shrink (smaller
  integration volume), which reweights every event's likelihood, not just the
  newly-zero-host ones.
- No event is silently dropped/excluded and no crash/NaN path was found for
  `max_redshift=0.3` specifically. But per §4, the resulting posterior would
  not be a clean, self-consistent z<0.3 analysis because `B_num` still draws
  from the untruncated population — so any de-railing effect measured under
  this knob today would be partially an artifact of that inconsistency, not a
  clean test of "truncate the analysis depth."

## Decision

A **fully self-consistent** `max_redshift` z-cut requires changing the
computed value of `B_num` (capping its integration upper bound), which is a
physics-formula change under CLAUDE.md's Math/Physics Validation Workflow.

**Per the task instructions, this halts Task 2.** No CLI/plumbing code has
been written. §4 documents the exact fix location and shape
(`z_upper = min(z_upper, redshift_upper_limit)` analogous to
`bayesian_statistics.py:1859`) for a follow-up `/physics-change`-gated change,
but implementing it — even as a one-line `min()` — changes B_num's computed
value whenever `max_redshift < z_upper(4σ)`, so it needs the Old/New
formula + reference + dimensional analysis + limiting-case presentation and
explicit approval first.
