# Option A′ implementation record — class-G S̄_φ de-double-weight

**Date:** 2026-09-02
**Branch:** `fix/p32d-classg-venue-repair`
**Spec of record:** `results/campaign51_20260728/realistic_20260729/PHYSICS_CHANGE_SBARPHI_20260827.md`
(the presentation, including its own adversarial-review addendum §AR-0..AR-10 / defects D1–D7)
**Authorization:** author's "both items as recommended please" — ledger row #314 (being written by
the chair; not re-litigated here) — ratifying Option A′ (§2.2) as the implementable form.
**Scope discipline:** implemented §2.2 verbatim, with the adversarial review's corrections (D2, D3,
D5, D6) folded into the test design as instructed; zero fresh design choices. **No fleet run, no
measurement, no commit** — chair commits.

---

## 1. What was implemented (§2.2, three changes, `"catalogue_selected_2d"`/b0i2d branch only)

**(i) Host draw uses the plain rate weight.** `MirrorUniverseGenerator.draw_realization`'s
`"catalogue_selected_2d"` branch now normalizes `catalogue_selected_host_draw_weights`'s *second*
return value (`w_g`) itself and passes that as `host_w` into `_draw_2d_accepted_latents`, instead
of passing that function's *first* return value (`w_g * S̃_φ,g`, normalized). No edit to
`catalogue_selected_host_draw_weights` itself — its signature, body, and first return value are
untouched, so the 1D `"catalogue_selected"`/b0i branch (a separate call a few lines above, and the
`"mixture_selected"` branch further down) still consumes exactly what it did before.

**(ii) z-draw drops the survival factor.** `_draw_kernel_survival_redshifts` gained a new keyword
parameter `apply_survival: bool = True`. The density computation is now:

```python
if apply_survival:
    s_i = np.interp(z_i_grid, z_grid, s_phi)
    density_i = kernel_i * w_pop_eff_i * s_i
else:
    density_i = kernel_i * w_pop_eff_i
```

Every existing caller (the 1D `"catalogue_selected"`/b0i branch, the `"mixture_selected"` branch)
does not pass the new keyword, so it defaults to `True` and computes the *exact same expression*
it always did — this is a bit-identical no-op for those callers. Only `_draw_2d_accepted_latents`'s
internal call to `_draw_kernel_survival_redshifts` (the "2D call site") passes
`apply_survival=False`.

**(iii) Mass draw and Bernoulli gate: unchanged**, as §2.2(iii) specifies. Verified by inspection —
no line in `_draw_2d_accepted_latents` between the mass draw and the `accept_mask` computation was
touched.

### Mechanism note on §2.2(ii)'s two disjunctive implementations

The presentation offered two ways to implement item (ii): "a keyword flag on the 2D call site" or
"passing a flat `S̄_φ ≡ 1` table." Its own adversarial review (AR-8, **D6**) strikes the flat-table
option: a flat table handed to `draw_realization` also reaches `catalogue_selected_host_draw_weights`
(used for the diagnostic `s_tilde_phi_host` column and the R4-style venue-drift audit), silently
zeroing that column. I implemented the **keyword-flag** form only, per D6's mandate.

---

## 2. Actual file:line locations vs. the presentation's citations

The presentation's own Appendix C flagged that its line numbers may have drifted by up to ±2, with
"~107-113 line drifts" elsewhere. Located by code content; actual sites (this repo, this commit,
before ruff-format's final pass — see §5 for the post-format numbers) drifted further than that,
consistent with intervening commits between 2026-08-27 (presentation date) and 2026-09-02
(implementation date):

| presentation cites | actual location (post-format) | what's there |
|---|---|---|
| `:1380-1385` (`catalogue_selected_host_draw_weights` first/second return value) | `:1464-1508` (function def), returns at `:1508` | **UNCHANGED** — confirms binding constraint honored |
| `:1490-1498` (z-density body) | `:1563-1665` (function def + body) | new `apply_survival` kwarg added at `:1573`; body branches at `:1635-1640` |
| `:1682` (host `rng.choice`) | `:1879` (`host_idx_batch = rng.choice(...)`) | unchanged — consumes whatever `host_w` its caller now passes |
| `:1687-1696` (2D z-draw call site) | `:1884-1897` | now passes `apply_survival=False` |
| `:2107` (2D branch host-weight call site) | `:2382-2400` (`draw_realization`'s `"catalogue_selected_2d"` elif) | `catalogue_selected_host_draw_weights` call + new plain-`w_g` normalization block |

The drift is consistent with the presentation's own disclosed uncertainty (§Appendix C) plus
~6 days of intervening commits on this branch/repo; content-matching (function names, docstring
text quoted in the presentation, the `w_g * S̃_φ,g` cancellation comment) confirmed each site
unambiguously before editing.

---

## 3. Diff (source)

```diff
--- a/darksiren_emri/validation/correspondence_1d.py
+++ b/darksiren_emri/validation/correspondence_1d.py
@@ _draw_kernel_survival_redshifts signature @@
     h: float = H_TRUE,
     n_grid: int = _B0I_ZTRUE_GRID_N,
+    apply_survival: bool = True,
 ) -> npt.NDArray[np.float64]:

@@ docstring Args @@
+        apply_survival: If ``True`` (default -- the "catalogue_selected"/b0i
+            and "mixture_selected" callers, UNCHANGED), the density includes
+            the ``S_bar_phi(z;h)`` factor as always. If ``False`` (the
+            "catalogue_selected_2d"/b0i2d 2D call site ONLY, ...), the
+            density drops the ``S_bar_phi`` factor: density_i = kernel_i *
+            w_pop_eff_i. This flag changes NOTHING for any caller that does
+            not pass it explicitly ...

@@ body @@
-        s_i = np.interp(z_i_grid, z_grid, s_phi)
-        density_i = kernel_i * w_pop_eff_i * s_i
+        if apply_survival:
+            s_i = np.interp(z_i_grid, z_grid, s_phi)
+            density_i = kernel_i * w_pop_eff_i * s_i
+        else:
+            density_i = kernel_i * w_pop_eff_i

@@ _draw_2d_accepted_latents internal call @@
+        # apply_survival=False -- 2D-only call site; 1D/mixture call sites
+        # do not pass this flag and are therefore bit-identical (L8).
         z_true_batch = _draw_kernel_survival_redshifts(
             rng, host_z_listed, host_z_error_listed, phi_survival_table,
             completeness, host_phiS_batch, host_qS_batch, h=h,
+            apply_survival=False,
         )

@@ draw_realization, "catalogue_selected_2d" branch @@
             pool = host_pool if host_pool is not None else _load_host_pool(REDUCED_CATALOGUE_PATH)
-            host_w, _b0i2d_w_g, b0i2d_s_tilde_phi = catalogue_selected_host_draw_weights(
-                pool, phi_survival_table, completeness, h=H_TRUE
+            _b0i2d_host_w_swphi, _b0i2d_w_g, b0i2d_s_tilde_phi = (
+                catalogue_selected_host_draw_weights(
+                    pool, phi_survival_table, completeness, h=H_TRUE
+                )
             )
+            # Option A'(i): plain rate-weight host draw, normalized here --
+            # do NOT use `_b0i2d_host_w_swphi` (w_g * S̃_φ,g normalized).
+            _b0i2d_w_g_total = float(_b0i2d_w_g.sum())
+            if not (_b0i2d_w_g_total > 0.0):
+                raise ValueError(
+                    f"catalogue_selected_2d plain rate-weight draw weights sum to <= 0 "
+                    f"({_b0i2d_w_g_total})"
+                )
+            host_w = _b0i2d_w_g / _b0i2d_w_g_total
             latents = _draw_2d_accepted_latents(
                 rng, pool, host_w, b0i2d_s_tilde_phi, phi_survival_table,
                 completeness, detection_probability, n, h=H_TRUE,
             )
```

Docstrings for `_B0i2DLatents.z_true`/`.s_tilde_phi_host`, `_draw_2d_accepted_latents`'s own
docstring, `_draw_2d_accepted_latents`'s `host_w`/`s_tilde_phi` Args, and the `draw_realization`
2D-branch comment were all rewritten to state the corrected law (they previously asserted
"UNCHANGED from the 1D mode" / "SAME `w_g*S̃_φ,g` law", which became false). Full diff:
`git diff -- darksiren_emri/validation/correspondence_1d.py` on this branch (128 lines changed,
one file).

**Binding MUST-NOT-CHANGE list (§9.1/L8), verified honored:**
- `_draw_kernel_survival_redshifts`'s density body for the default (`apply_survival=True`) path —
  bit-identical arithmetic, confirmed by `test_draw_kernel_survival_redshifts_matches_model_density`
  (pre-existing, unedited, still passing) and the new
  `test_draw_kernel_survival_redshifts_apply_survival_false_drops_the_factor`'s own default-path
  comparison.
- `catalogue_selected_host_draw_weights`'s first return value — function untouched.
- Everything outside the 2D branch — 1D `"catalogue_selected"`/b0i and `"mixture_selected"` branches
  call `_draw_kernel_survival_redshifts` without the new kwarg (unaffected) and still consume
  `catalogue_selected_host_draw_weights`'s first return value as their own `host_w` (unaffected).
- Every physics-trigger file (`constants.py`, `LISA_configuration.py`,
  `parameter_estimation/parameter_estimation.py`, `bayesian_inference/*.py`,
  `cosmological_model.py`, `physical_relations.py`) — none touched; only
  `darksiren_emri/validation/correspondence_1d.py` (not on the trigger list; gated voluntarily per
  §6.3, per the ledger row).

---

## 4. Tests

File: `darksiren_emri_test/validation/test_correspondence_1d.py`. §7's R1–R8 implemented with the
presentation's own adversarial-review corrections (AR-8 D1–D7) applied as instructed:

| §7 item | disposition |
|---|---|
| R1 (old-value pin) | **Not implemented as a separate two-phase pin.** Implementation was done in one pass per the task's direct-implementation instruction (not the two-commit "pin old, then change" workflow); D4 notes the existing suite pinned no draw law anyway, so no true "old value" regression risk existed to protect against mid-implementation. Routed, not improvised — see §6 below. |
| R2 (decisive law-identity discriminator) | **Implemented**: `test_catalogue_selected_2d_z_draw_matches_q_new_not_q_old` — single host (host weight cancels), M-independent but genuinely z-dependent `S_4D` double; CDF-gap check against `q_new` (PASS, gap<0.02) and against `q_old` (REJECT, gap>0.05). |
| R3 (host-weight coupling guard) | **Implemented, D5-corrected**: `test_catalogue_selected_2d_host_draw_uses_plain_rate_weight_not_sw_tilde_phi` — host-independent mass-selection fraction (identical `M`/`M_error` across hosts) + host-independent constant `S_4D`, wide `z_g` spread so `S̃_φ,g` varies >3x; χ² accepts `w_g/Σw_g` (<60) and rejects the Option-A-literal residual `w_g·S̃_φ,g/Σ` (>200). Plus an explicit call-site wiring guard, `test_catalogue_selected_2d_call_site_passes_plain_w_g_as_host_w` (monkeypatch-captures the actual `host_w` argument `draw_realization` passes and asserts it equals the plain-`w_g` renormalization, not `catalogue_selected_host_draw_weights`'s first return value) — this is the R8-addition the presentation names at §9.1 ("extend it with an explicit '2D branch must not use this value' assertion"). |
| R4 (quantify Option-A-literal residual) | **Not run** — out of scope for this task (measurement/fleet work is explicitly excluded; R4 was already executed by the presentation's own adversarial review, AR-2, at ~69–70% residual). |
| R5 (L1 limit) | **Implemented, D2-corrected**: `test_catalogue_selected_2d_l1_complete_detection_limit` — asserts `n_drawn_total == int(clip(4n,64,4000))` (not `n`, per D2), and a bit-identical `z_true` replay that draws the host batch FIRST (per D2's second correction) before calling `_draw_kernel_survival_redshifts` directly. |
| R6 (L2 limit) | **Folded into R5**, per D3 ("R6 must pass a flat survival table too ... at which point it collapses into R5 and should be merged with it"). R5's flat table (`S̄_φ≡1`) covers both L1 and L2 simultaneously. |
| R7 (mass selection retained, guards L5) | **Implemented**: `test_catalogue_selected_2d_mass_selection_retained_against_drop_bernoulli` — `S_4D` depending on `M_z` only (flat in `d_L`); KS-rejects the bare `p_gal(M|g)` (p<1e-6) and KS-accepts the quadrature target `p_gal·S_4D`/normalization (p>1e-3). |
| R8 (1D non-regression) | **Implemented via the pre-existing suite (unedited, still passing) + one new wiring guard.** `test_catalogue_selected_host_draw_weights_matches_independent_computation` (unedited) still confirms `catalogue_selected_host_draw_weights`'s first return value is `w_g·S̃_φ,g` normalized; `test_catalogue_selected_mode_does_not_enter_catalogue_selected_2d_code_path` (unedited) confirms the 1D branch never enters 2D machinery; the new `test_catalogue_selected_2d_call_site_passes_plain_w_g_as_host_w` is the explicit "2D branch must not use this value" extension the presentation calls for. |
| new: `apply_survival` unit test | `test_draw_kernel_survival_redshifts_apply_survival_false_drops_the_factor` — direct CDF-gap check on the new kwarg in isolation (drops the factor when False; does not when True). |

**Existing tests requiring updates** (§9.1's list, all handled):
- `_draw_2d_accepted_latents_pre_repair` (test-file reference helper for the *separate*, earlier
  R-2D-1 mass-floor defect) now also passes `apply_survival=False` to its internal
  `_draw_kernel_survival_redshifts` call, so
  `test_catalogue_selected_2d_byte_identical_to_pre_repair_when_no_floor_rows` continues to isolate
  *only* the mass-floor mechanism (its actual purpose) rather than becoming a stale comparison
  against a z-density the real function no longer computes.
- `:1616`/`:1647`/`:1678`/`:1734`/`:1758`-area tests (determinism, seed-sensitivity, columns,
  GATE-ACC stop, non-entry guard) — **unedited**, confirmed still passing; per D4 these never pinned
  the draw law's shape, only structural properties, so they are correctly unaffected by this change.

### Test run verdicts (verbatim)

```
$ uv run ruff check darksiren_emri/ darksiren_emri_test/
All checks passed!

$ uv run ruff format --check darksiren_emri/validation/correspondence_1d.py darksiren_emri_test/validation/test_correspondence_1d.py
(reformatted once during implementation; clean on recheck)

$ uv run mypy darksiren_emri/ darksiren_emri_test/
Success: no issues found in 220 source files

$ uv run pytest darksiren_emri_test/validation/test_correspondence_1d.py -q -m "not gpu and not slow"
94 passed, 1 warning in 65.40s (0:01:05)

$ uv run pytest -m "not gpu and not slow" -q   (full repo)
2032 passed, 15 skipped, 30 deselected, 12 warnings in 281.96s (0:04:41)
Coverage: 73.40% (>= 25% gate)
```

All green. Zero failures, zero new skips.

---

## 5. Gate ledger

Three rows appended to `docs/gates/PHYSICS-GATE-LEDGER.md` (`presented` → `implemented` →
`verified`, the ledger's own three-row convention), dated 2026-09-02, commit column `pre-commit`,
approval source "author word row #314", target `validation/correspondence_1d.py` (voluntary gate —
not on the `/physics-change` trigger list per §6.3/§9.1, noted explicitly in the `presented` row).

---

## 6. Ambiguity routed, not improvised

1. **R1's "old-value pin BEFORE the change."** The task instructed direct single-pass
   implementation of an already-ratified spec ("implementation from a complete, ratified gate
   presentation, ZERO fresh design choices"), which is incompatible with a literal two-commit
   "land the old-value pin first" workflow — there was no separate "before" commit to pin against.
   I did not fabricate one. This is flagged for the chair: if a genuine pre-fix numeric pin is
   wanted for the historical record, it must be reconstructed from a clean checkout of the
   pre-this-branch state, not invented here.
2. **R4 (residual quantification).** Already executed by the presentation's own adversarial review
   (AR-2: ~69–70% of the 13.5–16% drift would survive Option-A-literal). Not re-run — no new
   compute was authorized or needed; flagged rather than silently skipped.
3. **Fleet re-run / GATE-ACC re-check (§5 L9, §9.3, item [DO] #3 in the presentation's decision
   list).** Explicitly out of scope per the task brief ("Do NOT run the fleet or any measurement").
   PA-2D-9's frozen numbers and the 24-seed b0i2d fleet remain STALE per §9.3 until the chair
   authorizes and runs that re-run. The presentation's own AR-3 L9 finding (acceptance drops only
   ~3%, ~3 orders of magnitude of GATE-ACC headroom) suggests the STOP will not fire, but this was
   not independently re-verified here.
4. **`hier_blocker_a_generator_law_20260827.md`'s §9.4 dependency / AR-8 D7.** The presentation
   flags that this doc's justification for the `"catalogue_selected_2d"` sibling needs a one-line
   amendment regardless of scoping (D7). Not edited here — out of scope (a different doc, not part
   of §9.1's file list for this change) — flagged for the chair.
5. **Trigger-list amendment (§6.3, a standing [RULE] item).** Not decided or acted on here; this
   implementation treated the change as gated voluntarily, per the presentation's own convention.

---

## 7. What was NOT done (by design, per task scope)

- No fleet run, no measurement, no re-derivation of LHS₂/R_pred.
- No commit (chair commits).
- No edits to any physics-trigger file.
- No edits to `catalogue_selected_host_draw_weights`, `kernel_smeared_survival`, or anything under
  `darksiren_emri/bayesian_inference/`.
