# [P3-MKER] R1 — sigma_gal provenance trace: is the R&V15 scatter in the width the kernel actually uses?

**Date:** 2026-08-26 · **Agent:** sonnet (subagent, zero-compute read) · **Scope:** trace GLADE
stellar mass -> `sigma_gal` -> the with-BH mass kernel, file:line, with an explicit unit check
at the `sigma_gal <= K*sigma_cond` crossover (`bayesian_statistics.py:857`).

## Answer up front

The 0.24 dex **intrinsic** component of the R&V15 scatter **is** present in `host_M_error`
(fixed in commit `555f018`, 2026-07-01) and therefore in `sigma_gal_frac`/`sigma_lnM` wherever
those are computed from it. But the 0.50 dex **measurement-error** component is deliberately
excluded (design choice, not a bug — see §1). More importantly, for the **production default**
with-BH kernel (the plain Gaussian-product branch, `bayesian_statistics.py:6601-6616`, the only
branch the `[P3-2D]` twin composes with), the scatter's effect on the kernel's *absolute* width
collapses for mass-mismatched candidates because `sigma_gal_frac` is **linearized around the
candidate's own mean** (`host_M_error*(1+z)/det_M`), not around the GW peak. A candidate whose
catalogue mass is far below the GW-implied mass gets an artificially narrow absolute kernel
width regardless of the 0.24 dex/0.58-CV scatter magnitude — so `sigma_cond` (~1e-8-3.9e-9
fractional, GW-side) numerically dominates `sigma2_sum = sigma_cond**2 + sigma_gal_frac**2`
even though the *fractional* scatter is properly propagated. This exact failure mode is
independently identified, named, and *guarded against* in the code's own comments for the
**other** (ratified but non-default) `mass_trunc` log-normal kernel
(`bayesian_statistics.py:456-465`, quoted below) — but the production default path has no such
guard. **No dex/ln/linear unit mismatch was found** at the `sigma2_sum` sum points checked (see
§4): every place `sigma_cond`/`sigma_gal_frac` are added, both operands are already in the same
dimensionless "M(1+z)/M_det" fraction coordinate.

---

## 1. Where catalogue host BH mass + its uncertainty are computed

`darksiren_emri/galaxy_catalogue/handler.py`:

- **`:33-44`** — R&V15 constants: `alpha = 7.45*ln(10)`, `beta = 1.05`, `d_alpha = 0.08*ln(10)`,
  `d_beta = 0.11`, and **`sigma_int = 0.24 * np.log(10)`** with the comment (`:41-42`):
  > "Intrinsic scatter epsilon_0 = 0.24 dex (Reines & Volonteri 2015, Sec. 4.1): the true rms of
  > log10(M_BH) at fixed M_* once the calibration's virial measurement error (0.50 dex) is
  > removed."
  This is a **deliberate exclusion of the 0.50 dex term**, not an omission of the whole 0.55 dex
  figure — the reasoning (only in the comment, not in a derivation doc I could find) is that the
  0.50 dex is *measurement* error on the calibration sample's own M_BH estimates, not part of the
  predictive spread of the *true* M_BH given M_*. This is a modeling choice the claim card should
  weigh — it is not what `docs/MASS_RELATION_ASSESSMENT.md` §6 recommended (`≈0.55 dex total`).

- **`:1368-1382`** (`_empiric_stellar_mass_to_BH_mass_relation`):
  ```python
  BH_mass = np.exp(alpha + beta * np.log(stellar_mass / 10))
  BH_mass_error = BH_mass * np.sqrt(
      sigma_int**2
      + d_alpha**2
      + (np.log(stellar_mass / 10) * d_beta) ** 2
      + (beta / stellar_mass * stellar_mass_error) ** 2
  )
  ```
  `sigma_int` enters here, **in natural-log units** (`0.24*ln(10) = 0.5526`), added in quadrature
  with the fit-parameter and propagated stellar-mass-error terms. `BH_mass_error` is therefore a
  **linear (solar-mass) 1-sigma**, but it is `BH_mass * (a natural-log sigma)`, i.e. for small
  sigma it approximates `BH_mass * sigma_lnM`.

- **`:1136-1142`** (`_map_stellar_masses_to_BH_masses`) applies this **in place**, overwriting
  the `STELLAR_MASS`/`STELLAR_MASS_ABSOULTE_ERROR` columns (aliased as `BH_MASS`/
  `BH_MASS_ERROR` — see `InternalCatalogColumns` at `:177-190`) — called unconditionally from
  `__init__` (`:348`), so **every** load of the catalogue (default or `observed_catalogue_path`,
  scattered or not) computes `host_M_error` live from the current code, not from a cached CSV
  column. Verified: `reduced_galaxy_catalogue.csv` on disk (`darksiren_emri/galaxy_catalogue/`,
  1.68 GB, mtime 2026-07-27) has no header and its columns match the *pre-mapping* stellar-mass
  schema (`STELLAR_MASS` in units of 1e10 M_sun, e.g. row values 8.8, 10.85, 12.4 ...) — BH mass
  is computed at runtime, not baked into this file.

**Confirmed empirically** (small analytic python one-liner, current checked-out code, <25s, not
a pipeline run): for 5 representative `stellar_mass` values sampled from that CSV (8.8-14.31,
units 1e10 M_sun), `BH_mass_error/BH_mass` (the fractional CV) = **0.5826-0.5838**, i.e.
`sigma_lnM ~ 0.58` — consistent with (slightly above, because of the added fit-parameter terms)
`docs/MASS_RELATION_ASSESSMENT.md`'s own Table §2 entry "0.24 dex (intrinsic floor) -> 0.60
linear sigma_M/M". **So yes: the 0.24 dex intrinsic term IS present and dominant in
`host_M_error` today** — it is NOT silently zero.

## 2. host_M_error -> sigma_gal / sigma_lnM in the kernel (file:line, exact expressions)

Two distinct with-BH-mass kernel families exist in `bayesian_inference/bayesian_statistics.py`,
selected by `_use_mass_trunc = resolve_host_mass_kernel(host_mass_kernel, normalization_mode,
host_z_kernel) == "trunc_lognormal"` (`:6175-6178`, and `:240-300` for the resolver). `"auto"`
(the default `host_mass_kernel`, `:3180`/`:3346`/`:6064` etc.) resolves to `trunc_lognormal`
**only if** `normalization_mode == "mass_trunc"`; the CLI/pipeline default is
`normalization_mode = "generator_marginal"` (`:3174`, `main.py:1373`), so **`_use_mass_trunc` is
False by default.**

**(A) Production default — the plain Gaussian-product branch** (`:6601-6616`, reached when
`_use_mass_trunc` is False and `catalogue_mass_overlap == "production"`, the default):
```python
mu_gal_frac = _host_M_eff * (1 + z) / _det_M
sigma_gal_frac = host_M_error * (1 + z) / _det_M          # :6607
sigma2_sum = _sigma2_cond + sigma_gal_frac**2               # :6613
mz_integral = np.exp(-0.5 * (mu_cond - mu_gal_frac) ** 2 / sigma2_sum) / np.sqrt(2*np.pi*sigma2_sum)
```
This is the branch the `[P3-2D]` twin (`catalogue_numerator_survival_2d="mz_sel"`) is
**guard-restricted to** — `:6183-6188` raises `ValueError` if `_use_mass_trunc` or
`catalogue_mass_overlap != "production"` are combined with `mz_sel`. **The row #205 exhibit ran
this branch, not `mass_trunc`.** (Confirmed from `bt_900121_meta.json`:
`"catalogue_numerator_survival_2d": "mz_sel"`.)

`sigma_gal_frac` here is a **linearization**: it scales with the candidate's own mean
`host_M*(1+z)/det_M`, not with the GW peak `mu_cond`. This is the exact pathology the code's own
comments (below) name for the *other* kernel — but this branch has **no crossover guard at all**;
it is a pure unconditional Gaussian product every time.

**(B) `mass_trunc` (RATIFIED-M, 2026-07-27, `docs/derivations/mass_marginal_2d_kernel.md`) —
NOT the production default:**

- `:726-741` (`_mass_trunc_sigma_lnM`): `sigma_lnM = host_M_error / host_M` — a dimensionless
  natural-log-space width, recovered directly from the linear `host_M_error`.
- `:788-866` (`_mass_trunc_mz_integral`): GW-centred Gauss-Hermite (order 24) quadrature over the
  **true truncated lognormal x R_eff prior**, exact (no linearization) for the general case.
- `:850-861` — the **crossover** the task asked me to check:
  ```python
  sigma_gal = sigma_lnM * mu_gal          # :850-852, LINEARIZED fraction-space width
  narrow = (sigma_gal <= _MASS_TRUNC_GH_CROSSOVER_K * sigma_cond) & (sigma_lnM <= _MASS_TRUNC_GH_CROSSOVER_SIGMA_LNM_MAX)   # :857-860, K=5.0, cap=0.1
  sigma2_sum = sigma_cond**2 + sigma_gal**2     # :861
  mz_gauss = N(mu_cond; mu_gal, sigma2_sum)     # analytic fallback
  mz = np.where(narrow, mz_gauss, mz_gh)        # :865, GH used unless BOTH conditions hold
  ```
  Both requirements (`sigma_gal<=K*sigma_cond` AND `sigma_lnM<=0.1`) must hold before the cheap
  Gaussian analytic form is used; at production `sigma_lnM ~ 0.58` this **always fails the
  second condition** (0.58 > 0.1), so `mass_trunc` **always** uses the exact GH/lognormal
  integral in production-scale scatter regimes — it never silently falls back to a mismatched
  linear-Gaussian approximation. The module-level comment block (`:446-465`) explicitly documents
  **the exact failure mode this task is investigating**, found by the kernel's own golden tests:

  > "IMPLEMENTATION CORRECTION ... the width condition ALONE misfires for mass-mismatched hosts
  > (a_gal << 1 makes the LINEARIZED width sigma_gal tiny even when the prior is broad, sigma_lnM
  > ~ 0.7, and its fat lognormal tail at the GW peak is exactly what GH integrates correctly — the
  > Gaussian fallback would replace that tail with exp(-thousands), e.g. golden
  > near_lowmass_bound_mt_4d 0.061 -> 7e-15)."

  This is a ~1e13 ratio between the correct (GH) and naive-linear-Gaussian value on that golden —
  the same order of magnitude of collapse as the row #205 exhibit's `L_cat_with_bh` penalty
  (below). **`mass_trunc` was built specifically to fix this**, and per its own derivation
  status is `docs/derivations/mass_marginal_2d_kernel.md` line ~4-16: "RATIFIED (2026-07-27) ...
  the 2D channel itself remains OPEN per RATIFY-M6 until the [§3.8] discriminators run" — i.e.
  **ratified at the derivation level, NOT adopted as the production default kernel.** Production
  (`normalization_mode="generator_marginal"`) and the `[P3-2D]` twin both stay on branch (A).

## 3. Unit-consistency check at the `sigma2_sum` combination points

Checked explicitly (task item 2): at every site where `sigma_cond` (or `sigma_cond_M`,
`_sigma2_cond`) is summed in quadrature with a galaxy-side width —
`:6592`/`:6613`/`:6633`/`:6679`/`:6592`(err="inflated")/`:6716-ish`/`:7276`/`:7290`/`:7679`/`:861`
— **both operands are already in the same dimensionless "M(1+z)/M_det" fraction coordinate**:
`_means_4d[slot] = [det.phi, det.theta, 1, 1]` (`:4284`) fixes the mass-fraction observed mean to
1 by construction, and `_sigma_cond_M_arr`/`_sigma2_cond_arr` (`:4317-4322`) are derived from the
same normalized 4D Fisher covariance. `sigma_gal_frac = host_M_error*(1+z)/det_M` is the same
fraction coordinate by construction. **No dex/ln/linear mismatch found at these combination
points** — the mismatch is not a unit bug, it is the **linearization-point** problem described
in §2.

The one place a genuine unit *conversion* happens is `mass_trunc`'s `sigma_gal = sigma_lnM *
mu_gal` (`:850-852`): `sigma_lnM` is natural-log-space, and multiplying by the mean converts it
to an approximate linear-fraction sigma — a first-order lognormal-to-Gaussian linearization. This
conversion is only ever *used* (branch (B), `narrow=True` case) when `sigma_lnM <= 0.1`, where
the approximation is valid by the crossover's own design; it is invalid at production
`sigma_lnM~0.58` and the code correctly avoids applying it there. Branch (A) — production default
— has no `sigma_lnM` step at all; it consumes `host_M_error` directly as a linear sigma, so no
log-space quantity is converted; the issue there is purely the mean-anchoring, not units.

## 4. Is 0.55 dex, 0.24 dex, or neither present in production `sigma_gal`?

- **0.24 dex is present** (§1: measured fractional CV ~0.58, matching the doc's ~0.60 for the
  0.24 dex intrinsic-only case) in `host_M_error`, and therefore in both `sigma_gal_frac` (branch
  A) and `sigma_lnM` (branch B).
- **The full 0.55 dex (0.50 measurement + 0.24 intrinsic, quadrature) is NOT present** — the
  0.50 dex measurement-error term was deliberately dropped (comment at `handler.py:41-42`), so
  the code's own scatter choice differs from `docs/MASS_RELATION_ASSESSMENT.md`'s recommendation
  (§6: "add the intrinsic/total scatter (~0.55 dex)"). This divergence between the shipped fix
  and the assessment doc's recommendation is itself worth flagging to the author as part of
  `[P3-MKER]`'s kernel-derivation package.
- **Whether that scatter actually WIDENS the production kernel depends on the candidate**: for a
  well-matched host (`a_gal ~ mu_cond`) it does (per the module comment at `:462-464`, "a_gal ~
  mu_cond forces sigma_gal ~ sigma_lnM > K*sigma_cond"). For a badly-mismatched host (the row
  #205 class) branch (A)'s linearization collapses the absolute width toward the candidate's own
  (small) mean, so the fractional scatter is present in `host_M_error` but has almost no effect
  on the realized kernel width, and `sigma_cond` numerically dominates — matching claim card §1(a)
  verbatim ("a width dominated by the GW-conditional sigma_cond").

## 5. Representative production magnitudes (sigma_gal vs sigma_cond)

- **sigma_cond**: the claim card's "production p50 fractional sigma_cond ~ 1e-8" is
  **claim-card-sourced, not independently re-derived here** (would require aggregating
  `_sigma_cond_M_arr` across the fleet — outside zero-compute scope). Partial corroboration
  found: `bt_900121_work/.../prepared_cramer_rao_bounds.csv` event_idx=20 (the exhibit event,
  `host_galaxy_index=6791134`, `in_catalog=True`), `M=1.333246e6` M_sun,
  `delta_M_delta_M=2.6917e-5` (Cramer-Rao *marginal* M-variance, M_sun^2) ->
  `sqrt(2.6917e-5)/M = 3.9e-9` fractional — same order of magnitude as the claim card's 1e-8
  (marginal, not the 3-parameter-conditioned `sigma_cond`, so a modest difference is expected;
  conditioning on phi/theta/d_L generally narrows it further).
- **sigma_gal (host mass side)**: derived from code, not from a banked artifact (`host_M_error`
  is not written to any CSV under `p3_2d_fleet_20260825/` I found; `event_likelihoods.csv` has
  only aggregate `L_cat_with_bh` etc., no per-candidate sigma columns) — fractional CV **~0.58**
  (§1). In *absolute* fraction-coordinate terms for branch (A), `sigma_gal_frac = 0.58 *
  mu_gal_frac`, so its magnitude is candidate-dependent (collapses for `mu_gal_frac << 1`
  mismatches, per §2/§4).
- **Ratio check (claim card's crossover test, line 857)**: `K=5.0`
  (`_MASS_TRUNC_GH_CROSSOVER_K`, `:466`). For a well-matched candidate (`mu_gal_frac ~ 1`),
  `sigma_gal ~ 0.58 >> 5*sigma_cond ~ 5e-8` — the "narrow" branch never fires (correctly falls
  through to GH in `mass_trunc`; branch A has no such check and just adds the (large) term). For
  a badly-mismatched candidate (row #205 class, `mu_gal_frac` small), `sigma_gal` can collapse
  toward `~K*sigma_cond` or below **only in the linearized branch (A)'s arithmetic** — this is
  exactly the failure branch (B)'s crossover was built to detect and reject (`sigma_lnM<=0.1`
  gate at `:859`), but branch (A) has nothing analogous.
- **Row #205 exhibit, re-verified from the raw diagnostics CSV** (not re-derived, sourced):
  `results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825/bt_900121_work/seed900121/simulations/diagnostics/event_likelihoods.csv`,
  `event_idx=20`: `L_cat_no_bh=6.837940436563089e-09`, **`L_cat_with_bh=1.392199591828355e-85`**
  — matches the claim card's "1.39e-85" digit-for-digit. Ratio `L_cat_with_bh/L_cat_no_bh ~
  2.04e-77` — a suppression factor of the same extreme order as the mass_trunc golden's
  documented Gaussian-fallback failure (`0.061 -> 7e-15`, a ~1e13 ratio) scaled up, consistent
  with a genuinely large mass mismatch combined with a collapsed absolute kernel width in branch
  (A), not merely a narrow-but-honest kernel.

## 6. Commit `555f018` — what was fixed, what was deferred

`git show --stat 555f018` (2026-07-01, `[PHYSICS] fix host stellar-mass->BH-mass error budget`):
3 files changed (`CHANGELOG.md`, `master_thesis_code/galaxy_catalogue/handler.py` [+26/-4 lines,
pre-rebrand path], `master_thesis_code_test/test_mass_relation.py` [+76]).

**Fixed:**
1. Added `sigma_int = 0.24 dex` (intrinsic scatter) to `BH_mass_error` — previously fully absent
   (fractional CV 0.18 -> 0.59 at the pivot per the commit message).
2. Fixed an operator-precedence bug: `beta/stellar_mass/10` -> `beta/stellar_mass` (the spurious
   `/10` understated the stellar-mass-propagation term 100x in variance).
3. Fixed the (dead-code, unused) inverse relation's `beta` exponent and added scatter there too.

**Deferred** (commit message, final paragraph): *"model the host-mass error as log-normal (the
linear-Gaussian leaks ~5% to M<0 at 0.24 dex, and the 2-D likelihood in bayesian_statistics.py
would need updating in lockstep)."* — **This deferral was subsequently addressed by the
`mass_trunc` kernel**, derived and RATIFIED 2026-07-27 (`docs/derivations/mass_marginal_2d_kernel.md`,
git history `git log 555f018..fb4ac4e -- .../handler.py` shows `9c948ea0`/`cf4f8a2a` as later,
unrelated `[P3-WBHZERO]` mass-filter-window commits, confirming `555f018` is an ancestor of the
exhibit's commit `fb4ac4eea8bb`) — **but `mass_trunc` was never promoted to the production
default** (`normalization_mode="mass_trunc"` is opt-in only; default is `"generator_marginal"`),
and its own derivation doc records it as still "CANDIDATE/OPEN pending §3.8 discriminators."

**Verified**: `git merge-base --is-ancestor 555f018 fb4ac4eea8bb415e38d542f6f458b3dd259060f0` ->
true — the exhibit's run (`bt_900121_meta.json`: `"git_commit":
"fb4ac4eea8bb415e38d542f6f458b3dd259060f0"`) postdates the sigma_int fix, so **the row #205
result is NOT a stale-catalogue artifact** — it reflects current code's genuine behavior under
branch (A).

---

## Bottom line for [P3-MKER] part (a)

The claim card's characterization is correct, but its likely-diagnosed *mechanism* needs a
precise amendment: the intrinsic scatter is not simply "omitted" from the width computation —
it's present in `host_M_error`/`sigma_lnM` (0.24 dex, not the full 0.55 dex) — but it fails to
widen the kernel for the exact candidates that matter (mismatched ones) because the **production
default kernel (branch A) linearizes the galaxy-mass width around the candidate's own mean**
rather than the GW peak, an approximation failure the codebase has *already* identified, named,
and guarded against in the alternative `mass_trunc` kernel (branch B) — which exists, is
RATIFIED at the derivation level, but is not the production default and is explicitly excluded
from composing with the `[P3-2D]` twin machinery that produced the row #205 exhibit. The
succession structure's "kernel first" package (CLAIM_P3_MKER_20260826.md §3 item 1) should
account for `mass_trunc` as a candidate fix already built and derived, not something to design
from scratch, and should resolve the 0.24-vs-0.55 dex discrepancy against
`docs/MASS_RELATION_ASSESSMENT.md`'s recommendation explicitly.

## Files/lines cited (for quote-verification)

- `darksiren_emri/galaxy_catalogue/handler.py:33-44, 1136-1142, 1368-1382`
- `darksiren_emri/bayesian_inference/bayesian_statistics.py:240-300 (resolver), 446-467
  (mass_trunc constants+crossover comment), 726-741, 788-866 (mass_trunc kernel + crossover),
  3174, 4280-4322 (4D conditional construction), 6175-6188 (twin guard), 6519-6523, 6557-6616
  (branch A vs B dispatch), 6601-6616 (production default kernel)`
- `docs/MASS_RELATION_ASSESSMENT.md` §1-6
- `docs/derivations/mass_marginal_2d_kernel.md` header (RATIFIED 2026-07-27, RATIFY-M6 OPEN
  clause)
- `git show --stat 555f018`; `git log 555f018..fb4ac4eea8bb415e38d542f6f458b3dd259060f0 --
  darksiren_emri/galaxy_catalogue/handler.py`; `git merge-base --is-ancestor 555f018
  fb4ac4eea8bb415e38d542f6f458b3dd259060f0`
- `results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825/bt_900121_meta.json`
- `results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825/bt_900121_work/seed900121/simulations/diagnostics/event_likelihoods.csv`
  (event_idx=20)
- `results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825/bt_900121_work/seed900121/simulations/prepared_cramer_rao_bounds.csv`
  (row 20: `M`, `delta_M_delta_M`, `host_galaxy_index`, `in_catalog`)
- `results/campaign51_20260728/realistic_20260729/CLAIM_P3_MKER_20260826.md` §1(a), §2, §6 LIT-1

## Caveat (explicit, not independently closed here)

`sigma_cond` p50 ~1e-8 (claim card) was **not independently re-derived** at fleet scale in this
zero-compute pass — only order-of-magnitude-corroborated on the single exhibit event via the raw
Cramer-Rao `delta_M_delta_M` column (marginal, not the 3-parameter-conditioned quantity the
kernel actually uses). A full sigma_gal-vs-sigma_cond fleet census (claim card §5 item (i)) would
need to actually run `_map_stellar_masses_to_BH_masses` + the 4D-conditional machinery per host —
outside this task's zero-compute mandate.
