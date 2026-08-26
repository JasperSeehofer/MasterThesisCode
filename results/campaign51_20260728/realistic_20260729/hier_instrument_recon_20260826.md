# [HIER] theta instrument reconciliation — pre-launch dispatch-path recon

**Date:** 2026-08-26 · **Author:** subagent recon (orchestrator task, decision D3 + [A13]) ·
**Scope:** trace every host-z-kernel dispatch path a theta = (b, s) flag pair would have to
reach, and resolve whether theta's scatter-scale `s` generalizes, is independent of, or
collides with the existing `smear_sigma_z` flag. Zero-compute: static read only, no runs.

**Bottom line up front:** under the LITERAL CLI/class production default
(`normalization_mode=generator_marginal`, `host_z_kernel=auto`→`point`,
`smear_global_selection` off), theta's `s` is **structurally inert** — it cannot touch the
likelihood at all, by construction, not by omission. `b` is NOT inert under that default. This
is a scope-defining fact D2/D3 must resolve before any pilot node runs, not an implementation
detail to patch afterward. Full argument in §1.3.

---

## 1. Scalar and batch dispatch paths (task item 1)

### 1.1 The two width-computation sites (task's own line anchors, confirmed exact)

Scalar (`single_host_likelihood`), `bayesian_statistics.py:6223-6224`:
```python
sigma_z_pv = (1.0 + host_z) * SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S
host_z_error_eff = float(np.sqrt(host_z_error**2 + sigma_z_pv**2))
```
Batch (`single_host_likelihood_batch`), `bayesian_statistics.py:6878-6879`:
```python
sigma_z_pv = (1.0 + host_z) * SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S
host_z_error_eff = np.sqrt(host_z_error**2 + sigma_z_pv**2)
```
Byte-identical functional form (float vs. array). `s` would multiply `host_z_error` — but
**where** in this formula is itself an open question (§5): scale the raw catalogue
`host_z_error` before combining with the peculiar-velocity term in quadrature, or scale the
combined `host_z_error_eff`? `SIGMA_V_PEC_KM_S = 0.0` (`constants.py:95`), so the two are
numerically identical today — this ambiguity is currently unobservable and must be nailed down
in the prereg text, not left implicit, since it stops being a no-op the moment
`SIGMA_V_PEC_KM_S` is ever set.

The z-clamp, scalar path, `bayesian_statistics.py:6242-6244`:
```python
denominator_integration_lower_redshift_limit = max(
    host_z - integration_limit_sigma_multiplier * host_z_error_eff, _z_lower_floor
)
```
(mirrored batch form `bayesian_statistics.py:6899-6901`, `den_lo = np.maximum(host_z - 4*host_z_error_eff, _z_lower_floor)`). `b` would apply to `host_z` here too (window re-centers on the shifted mean).

`b`'s injection point — the numerator's location parameter, NOT visible in the width lines
above:
- Quadrature branch: `galaxy_redshift_normal_distribution = norm(loc=host_z, scale=host_z_error_eff)`, `bayesian_statistics.py:6247`.
- Point-kernel branch (see §1.3): `_z_point = np.array([host_z], dtype=np.float64)`,
  `bayesian_statistics.py:6401`, re-used at `:6661` for the with-BH-mass point branch.

### 1.2 Kernel-variant dispatch (task's `~6250-6253` anchor)

That line range is the START of the `_use_volume_deconv` documentation comment
(`bayesian_statistics.py:6249-6277`), which enumerates `normalization_mode` values sharing the
volume-deconvolved-weight numerator: `"volume_deconv", "volume_global", "volume_trunc",
"mass_trunc", "absolute_marginal", "generator_marginal"` (tuple at `:6270-6277`). `"global"` and
`"local_ratio"` are the two modes that keep the BARE Gaussian (no `w_pop` weight) — named
explicitly in the comment at `:6255` ("`'global'/'local_ratio'` use the BARE photo-z Gaussian
... (unchanged behaviour)"). This is a DIFFERENT axis from the point-vs-quadrature kernel
switch in §1.3 below — `_use_volume_deconv` only controls whether the population weight
`w_pop(z) = dV_c/dz/(1+z) [x f_k(z)]` multiplies the Gaussian; it does not control whether the
Gaussian is quadrature-integrated or point-evaluated.

### 1.3 THE decisive dispatch fork: point vs. quadrature kernel, and why `s` can be a no-op

`resolve_host_z_kernel` (`bayesian_statistics.py:166-213`) resolves the numerator kernel:
```python
resolved = (
    ("point" if normalization_mode == "generator_marginal" else "volume_deconv")
    if host_z_kernel == "auto"
    else host_z_kernel
)
```
`host_z_kernel` CLI default is `"auto"` (`arguments.py:699-702`); `normalization_mode` CLI
default is `"generator_marginal"` — **"[PHYSICS] production default since 2026-07-26"**
(`arguments.py:745-748`, `bayesian_statistics.py:3174/6055`). So under the literal default,
`resolve_host_z_kernel` returns `"point"`, and `_use_generator_point = True`
(`bayesian_statistics.py:6205`, `:6903` batch mirror).

On the point path, the numerator is evaluated at `_z_point = host_z` EXACTLY —
`bayesian_statistics.py:6396-6416` (without-BH-mass) and `:6652-6692` (with-BH-mass) — **there
is no `host_z_error`/`sigma_z` term anywhere in either point branch.** The governing comment,
`bayesian_statistics.py:6191-6205`: *"the generator-exact in-catalogue numerator is the GW
likelihood POINT-evaluated at z_g ... no sigma_z scatter anywhere on the production path."*

The per-host denominator `D_g` (quadrature branch, `single_host_likelihood` return slot 1/3)
DOES depend on `host_z_error_eff` even on the point-numerator path (its own dispatch is
independent, `_use_generator_point` only gates the numerator). But in `generator_marginal`
mode the FINAL assembly never divides by it: `bayesian_statistics.py:5069-5091` —
`L_cat_without_bh_mass = cat_num_sum_no_bh / n_hat_w`, summing ONLY `r[0]` (the numerator);
`r[1]` (`D_g`) is absent from the formula. The comment at `:6267-6269` confirms: *"generator_marginal
joins this set for the DENOMINATOR/Z_g machinery only ... (diagnostic only in this mode; the
assembly never divides by it)."*

**Consequence:** under the literal production default, `s` touches nothing the likelihood
value depends on — not the numerator (point-evaluated, no `sigma_z` term), not the assembly
(the one per-host quantity that DOES carry `sigma_z` is computed and then discarded). `b`
remains live (it shifts `_z_point`, feeding both the GW-distance term and the with-BH-mass mass
kernel via `_mu_gal_frac_point` at `_z_point`). This is a structural fact about the default
configuration, not a coverage gap in a not-yet-written theta implementation.

For `s` to have any effect, the run must set `host_z_kernel != "point"` (either
`normalization_mode != "generator_marginal"`, or an explicit `host_z_kernel="volume_deconv"`
override on top of `generator_marginal`). The realistic-venue config this campaign directory
(`results/campaign51_20260728/realistic_20260729/`) is named for is exactly such an override —
`main.py:130-137` logs, immediately after writing an observed (scattered) catalogue: *"evaluate
with --observed_catalogue %s --normalization_mode absolute_marginal --host_z_kernel
volume_deconv"* — i.e. the realistic-venue recommended default is NOT `generator_marginal`.
Compounding this, `validate_scatter_guards` (referenced `bayesian_statistics.py:3864-3869`,
guard docstring `:180-188,214-225`) makes this non-optional on a scattered catalogue: **the
point kernel raises `ValueError` outright** when `catalogue_scattered=True` (`galaxy_catalog.scattered`),
demanding `host_z_kernel=volume_deconv` + `normalization_mode=absolute_marginal` explicitly.

**D2 as written names `"global"/"local_ratio"/"volume_deconv"` as the out-of-scope "other
variants," leaving `generator_marginal` and `absolute_marginal` in scope by omission — but
`generator_marginal` is precisely the mode where `s` is dead, and `absolute_marginal` is the
one mode where making `s` live requires ALSO threading it through a structurally separate
code path (§2.3) that is mutually exclusive with `generator_marginal`.** D2 needs a fourth
sentence naming which of `{generator_marginal, absolute_marginal}` (or an explicit
`host_z_kernel="volume_deconv"` override atop `generator_marginal`) is "the configuration of
record" for the pilot — the two are not interchangeable for this instrument, and the choice is
not implied by "the default."

---

## 2. Every dispatch path production touches (task item 2)

Grepped `single_host_likelihood(`, `single_host_likelihood_batch(`, and
`precompute_global_catalog_selection(` callers across `darksiren_emri/`, plus every module
matching `host_z_error`. Five independent sites, not two:

### 2.1 `single_host_likelihood` (scalar) — `bayesian_statistics.py:6044`
Per-candidate-ball numerator/denominator, both channels. §1 above.

### 2.2 `single_host_likelihood_batch` (batch) — `bayesian_statistics.py:6753`
Production path (the scalar function has NO other caller in `darksiren_emri/` outside its own
module — grep for `single_host_likelihood(` other than the def and its own docstring reference
at `:6806` returns nothing; the batch function is what `BayesianStatistics` actually dispatches
per event). Mirrored width/clamp/kernel-fork formulas, confirmed byte-identical in form (§1.1).

### 2.3 `precompute_global_catalog_selection` + `_smeared_global_pdet_expectation` —
`bayesian_statistics.py:2657`, `:1619`. Called at `:4010` (no-BH global denom),
`:4018` (with-BH global denom), `:4062` (phi-marginal companion). This is the table
`{global, volume_global, absolute_marginal, generator_marginal}` normalization modes divide by
(§1.3, §2.4 formula). It reimplements the SAME width formula independently
(`_smeared_global_pdet_expectation`, `bayesian_statistics.py:1669`:
`sigma_z_pv = (1.0 + z_g) * SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S`; `:1672`:
`sigma_eff = np.maximum(np.sqrt(z_err_g**2 + sigma_z_pv**2), 1e-10)`) — mirrored correctly today
(same PV term), but it is a SECOND site that must independently receive `s`. **It has no
analog of `b` at all** — `zc` (the quadrature center, `:1679`) is always the bare catalogue
`z_g`, unshifted. A theta implementation that threads `s` here but not `b` produces an
asymmetric instrument (denominator sees the scatter but not the bias) even when the numerator
sees both.

### 2.4 `darksiren_emri/validation/correspondence_1d.py:1167-1188` — `host_z_error_eff()`
The mirror-venue harness's OWN copy of the width formula, used by every C-A/P3-2D/P3-TWIN
fleet task (the fleet this HIER pilot's Stage-P/F reuses per D4/D1). Docstring explicitly
claims "byte-identical functional form to production's per-host sigma
(`bayesian_statistics.py:5908-5909`)" [stale line numbers — current production site is
`:6223-6224`/`:6878-6879`, confirming this comment predates a renumbering and was never
re-verified against the current file — a live drift risk for any claim of parity]. This
function is called at `:1323` and `:1485`, feeding the mirror's own `_host_kernel_window`
(`:1191-1199`) and its onward quadrature. **A theta `s`/`b` hook added only inside
`bayesian_statistics.py` does not reach the mirror venue at all** — this file has its own copy
and must receive its own hook, independently maintained (a fourth site, and the one the
Stage-P/F pilot fleet actually executes against, per D1).

### 2.5 `darksiren_emri/validation/pp_coverage.py` — fully independent reimplementation
`grep`-confirmed: this module has **zero import of `bayesian_statistics`** (its many
`bayesian_statistics.py:NNNN` references are comment-only cross-checks, not calls). It carries
its OWN `sigma_z: float = 0.035` config field (`PPCoverageConfig`, `:630`) and its own
`kernel: Literal["bare", "volume"]` switch (`:641`), with hand-written `num_bare(h)`/
`num_volume(h)` formulas (`:828-829`) and its own z-window/clamp logic. **This is the G4b
synthetic-universe coverage/P–P calibration harness** (CLAUDE.md: "switchable host-z kernel
('bare' vs calibrated 'volume')"), and it is EXACTLY the tool the HIER proposal's own §2
registers as read (ii): *"coverage/P–P over seeds at the h-marginal (the stage-4 currency)."*
If theta's promised registered read (ii) is to run, `pp_coverage.py` needs its OWN independent
`b`/`s` parameterization — this is a fifth site, structurally disconnected from the other four,
and the proposal as written gives no indication its author budgeted for a fifth reimplementation.

### 2.6 CLI surface — `darksiren_emri/arguments.py:699-786`
`--host_z_kernel` (`:699`), `--normalization_mode` (`:736`), `--smear_global_selection`
(`:773`) are the existing flags; a `--theta_b`/`--theta_s` pair (or equivalent) would need
registering here with an incompatibility guard mirroring the existing
`generator_marginal`+`smear_global_selection` guard (`bayesian_statistics.py:3849-3858`, quoted
§3) — i.e. theta needs its own version of that same "refuse the combination loudly" pattern for
whichever `(normalization_mode, host_z_kernel)` pairs make `s` inert (§1.3) or partially-wired
(§2.3's missing `b`-in-denominator case).

**Enumeration verdict:** at minimum 2.1 + 2.2 + 2.3 + 2.4 + 2.5 = **five** independent
dispatch paths, not the two (scalar/batch) framed by the task prompt. 2.4 and 2.5 are outside
`bayesian_statistics.py` entirely and share no code with it.

---

## 3. D3 — `smear_sigma_z` reconciliation (task item 3)

`smear_sigma_z: bool = False` is `precompute_global_catalog_selection`'s parameter name
(`bayesian_statistics.py:2664`); it is exposed at the CLI/production level as
`--smear_global_selection` (`arguments.py:773-786`), threaded in at
`bayesian_statistics.py:4016` and `:4024` as `smear_sigma_z=smear_global_selection`.

**What it does, quoted exactly** (`bayesian_statistics.py:2865-2881`):
```python
elif smear_sigma_z:
    # [PHYSICS] num/denom sigma_z symmetry (issue #30 estimator redesign, risk R4):
    # E_{z~kernel_g}[P_det] over the numerator's volume-deconvolved host-z
    # kernel replaces the point evaluation P_det(d_L(z_g;h)). Opt-in via
    # --smear_global_selection; sigma_eff -> 0 recovers the point form.
    p_det = _smeared_global_pdet_expectation(
        z_g, M_g, z_err_all[eligible], theta_all[eligible] if _sky_aware else None,
        h, detection_probability_obj, with_bh_mass=with_bh_mass, sky_aware=_sky_aware,
    )
```
i.e. it toggles the GLOBAL selection denominator between a point evaluation
(`P_det(d_L(z_g;h))`) and the expectation of `P_det` over the SAME Gaussian×`w_pop` kernel the
numerator uses, at the catalogue's UNSCALED `z_err` (`s`-implicit-1). It is a pure Boolean gate
on a fixed-width kernel — it has no scale knob.

**Guard** (`bayesian_statistics.py:3849-3858`, quoted verbatim):
```python
if normalization_mode == "generator_marginal" and smear_global_selection:
    raise ValueError(
        "normalization_mode='generator_marginal' uses the point/point "
        "sigma_z pairing (generator-exact); --smear_global_selection is "
        "incompatible with it. Drop the flag (or use 'absolute_marginal' "
        "for the kernel/smeared pairing)."
    )
```
Also documented at the CLI help string (`arguments.py:773-786`): *"Off by default (point
evaluation, byte-identical legacy behavior). Relevant to normalization modes that consume
Sigma_glob ('global', 'absolute_marginal'). Incompatible with 'generator_marginal'."*

**Resolution — GENERALIZATION, with a live collision hazard if wired naively:**
`smear_sigma_z=True` is exactly "engage the width-sensitive kernel at `s=1`" on the global-
denominator side only. Theta's `s` is a strict generalization of this ONE bit IF AND ONLY IF
the theta implementation:
(a) makes `s != 1` (or `s == 1` with the kernel deliberately engaged) FORCE the
`smear_sigma_z=True` branch itself, rather than depending on a separately-set
`--smear_global_selection` flag left at a stale campaign-config value — because if theta
threads `s` into `z_err_g` but the boolean gate stays at its default `False`, the code takes
the sibling branch (`elif with_bh_mass: ... isotropic point ...`,
`bayesian_statistics.py:2882+`) and `s` is silently discarded on the denominator side while
still live on the numerator side — **the exact "mixed-arm result" the task prompt warns
about**, and it happens by DEFAULT, not by misconfiguration, because `smear_global_selection`
defaults to off;
(b) is REFUSED (raises, mirroring the existing guard) under `normalization_mode="generator_marginal"`
for any `s != 1`, since that mode's incompatibility with the smeared companion is unconditional
— theta cannot silently fall back to a point-evaluated, `s`-blind denominator there and still
claim a well-posed instrument (this is consistent with, and reinforces, §1.3's finding that
generator_marginal cannot host a live `s` at all); and
(c) also threads `b` into `_smeared_global_pdet_expectation`'s `zc` (§2.3 — currently no `b`
analog exists there at all), or explicitly documents that `b` is deliberately NOT applied to
the global denominator (a real, separate design choice, not an oversight to silently inherit).

If (a)-(c) are not satisfied, this is not a "collision" in the double-counting sense (the two
mechanisms do not multiplicatively stack — there is only one `sigma_eff` computation site
downstream, `:1672`) but a **silent-omission collision**: theta's numerator and the
production-selected denominator machinery would be evaluated at different effective error
models for the same nominal `s`, which is exactly the failure mode D3 was raised to block.

---

## 4. Other flags already perturbing the host-z kernel (task item 4)

Grepped every flag argument to `single_host_likelihood`/`_batch` and
`precompute_global_catalog_selection`:

| flag | site | touches z-kernel? | interacts with theta? |
|---|---|---|---|
| `host_z_kernel` (`"auto"/"point"/"volume_deconv"`) | `bayesian_statistics.py:6060,:699` (CLI) | YES — selects point vs. quadrature (§1.3) | **Directly gates whether `s` is live at all** — must be pinned in the theta prereg, not left at the campaign default. |
| `normalization_mode` | `:6055,:736` (CLI) | YES — selects which denominator (`D_g` local vs. global table vs. discarded) consumes the width | **Directly gates whether `s` reaches the assembly** — see §1.3/§3. |
| `smear_global_selection`/`smear_sigma_z` | `:2664,:773` (CLI) | YES — the global-denominator boundary case, §3 | The reconciliation target itself. |
| `catalogue_numerator_survival` (`"off"/"phi"`) | `:6078,6112-6121` | Indirect — multiplies the numerator by `np.interp(z, z_s, s_phi)` at the SAME `z` array the kernel already produced | No collision if `b`/`s` are applied upstream of the quadrature-grid construction (the table lookup rides whatever `z` grid results); a real interaction if the survival table `z_s` grid itself was built at truth-theta assumptions and is not regenerated per theta node (open item, not verified here — table provenance not traced in this pass). |
| `catalogue_numerator_survival_2d`/`_center` (`"off"/"mz_sel"`, `"raw"/"eff"`) | `:6087,6093,6124-6189` | Indirect — same pattern, 2D (mass) leg; guarded against composing with `mass_trunc`/non-production `catalogue_mass_overlap` (`:6183-6189`) | Same upstream-ordering caveat as above; additionally composes with `eddington_m` (below) inside the SAME product-Gaussian mean — no z-width interaction, only a mean-centering choice. |
| `eddington_m` (`"on"/"off"`) | `:6075,6108-6109` | NO — governs `_host_M_eff`, the MASS prior shift, not `z` | Orthogonal axis; no interaction. |
| `host_mass_kernel` (`"auto"/"gaussian"/"trunc_lognormal"`) | `:6064,6175-6178,:716` (CLI) | NO directly, but `_use_mass_trunc` combined with a POINT-resolving `host_z_kernel` raises (`resolve_host_mass_kernel`'s "prior-consistency guard", referenced `:6175-6178,:731`) | A `theta` node that forces `host_z_kernel="point"` (§1.3, `s` inert there) while `host_mass_kernel` resolves to `trunc_lognormal` would hit this pre-existing guard — worth a smoke check before the pilot, not blocking the design. |
| `mass_filter_sigma` (`"symmetric"` default) | `:3248,3311,3460,3633,4692` | NO — a mass-space filter width, not `z` | Orthogonal axis; no interaction found in this pass (not exhaustively traced beyond confirming it is mass-only by name/site; flagged as an open item if the reviewer wants it ruled out formally). |
| `catalogue_mass_overlap`, `catalogue_mass_error_scale` | `:6068-6069` | NO — mass-column overlap/error-scale instrument (Prod2d closure counterfactual) | Orthogonal axis. |

**Summary for item 4:** three flags (`host_z_kernel`, `normalization_mode`,
`smear_global_selection`) are not merely "could interact" but **directly determine whether
theta has any effect and through which path** — they are not peers of theta, they are theta's
own dispatch gates and must be pinned as part of the theta instrument's own definition, not
left as free campaign config. `catalogue_numerator_survival[_2d]` are secondary, ordering-
sensitive interactions (open item, table provenance not traced this pass). Everything else is
orthogonal (mass-axis, or catalogue-column instruments unrelated to `z`).

---

## 5. Where truth-theta byte-identity could fail (task item 5)

1. **Branch-on-presence, not value, at the kernel fork.** `resolve_host_z_kernel`'s `"auto"`
   branch (`bayesian_statistics.py:210`) selects `"point"` vs `"volume_deconv"` off
   `normalization_mode`'s STRING VALUE, not off theta. If theta's `s`/`b` kwargs are threaded
   through as new function parameters with defaults `(0, 1)`, byte-identity at `(0,1)` requires
   the injection to be a literal identity transform (`host_z + b*(1+host_z)` with `b=0` →
   exact `host_z`; `host_z_error * s` with `s=1` → exact `host_z_error`, floating-point
   multiplication by `1.0` is exact in IEEE 754, so this specific case is safe) — BUT if the
   implementation instead ADDS `b` as `host_z*(1+b)` or otherwise reorders the arithmetic, or
   computes `s` via `np.sqrt(host_z_error**2 * s**2 + sigma_z_pv**2)` vs.
   `(host_z_error*s)**2 + sigma_z_pv**2`, the two are bit-identical at `s=1` but NOT
   necessarily at intermediate rounding through `np.sqrt`/`**2` — needs the exact chosen
   formula pinned and a regression test asserting bit-identity at `(0,1)` against the
   pre-theta CSV, not just "should be identity."
2. **The point-kernel branch has no `s`-dependent code to become a no-op in** — at `s=1` this
   is trivially safe (nothing new executes), but a common implementation temptation is to
   short-circuit `if theta_s == 1.0 and theta_b == 0.0: skip the theta code path entirely` —
   this IS the correct byte-identity guarantee, but if instead a generic "apply theta" function
   is called unconditionally and happens to reduce mathematically to the identity, floating-
   point associativity is not guaranteed to reproduce the ORIGINAL expression's exact rounding
   (e.g. `host_z_error_eff` computed via a slightly reordered `np.sqrt` call). **The safe
   pattern is a literal early-return/skip at `(0,1)`, not "trust the math."**
3. **Global-table caching keyed by `h` only, not by `theta`.** `_D_h_table`,
   `_beta_G_table`, `_beta_Gbar_table`, `_global_cat_denom_no_bh`, `_global_cat_denom_with_bh`,
   `_V_f_table` (`bayesian_statistics.py:3987-4360`) are `dict[h -> value]`, computed ONCE at
   `BayesianStatistics` construction / `evaluate()` entry (`:3987-4108`), then read per-event
   via `self.h` lookups (e.g. `:5009-5022`). **If a theta grid sweep reuses one
   `BayesianStatistics` instance across theta nodes without re-triggering this precompute
   block, every node after the first reads STALE tables computed at whichever theta was active
   at construction** — a silent, severe, and easy-to-miss bug given the array-indexing pattern
   in D4 (a flattened `(theta-node, seed)` SLURM index almost certainly maps to ONE fresh
   process/instance per array task, which is actually the safe case — but this must be an
   EXPLICIT invariant in the driver, not an assumption: one `BayesianStatistics` construction
   per theta node, never one instance swept across a theta axis in-process).
4. **`catalogue_scattered` guard changes the SET of valid `(normalization_mode, host_z_kernel)`
   pairs depending on which catalogue is loaded** (`validate_scatter_guards`,
   `bayesian_statistics.py:3864-3869`, guard docstring `:180-225`) — a theta pilot run against
   the unscattered baseline catalogue and a theta fleet run (D1 Stage F, presumably against a
   realistic/observed catalogue given this directory) could silently resolve `host_z_kernel`
   differently at the SAME literal CLI flags, because the guard is catalogue-conditional, not
   flag-conditional. The prereg must state which catalogue each stage loads and confirm the
   resolved kernel is the SAME for both, or explicitly register that it differs and why.
5. **The mirror-venue's independent `host_z_error_eff` copy is drift-prone by its own
   docstring's admission** — it cites `bayesian_statistics.py:5908-5909` as the parity anchor
   (`correspondence_1d.py:1173`), but the current production site is `:6223-6224`/`:6878-6879`
   (§2.4) — the comment's line numbers are already stale, meaning the "byte-identical" claim
   has not been re-verified since at least one renumbering. Byte-identity at truth-theta on the
   MIRROR venue (where D1's pilot actually runs) rests on this un-reverified parity claim; it
   should be re-diffed against current production before the pilot, not trusted from the
   comment.

---

## Sources (file:line index)

- `bayesian_statistics.py`: 166-225 (`resolve_host_z_kernel`), 1619-1720
  (`_smeared_global_pdet_expectation`), 2657-2882 (`precompute_global_catalog_selection`),
  3174 (class default), 3849-3858 (generator_marginal/smear guard), 3864-3878 (scatter guard
  call), 3987-4108 (global-table precompute), 4010/4018/4062 (call sites), 5009-5126 (L_cat
  assembly, all normalization-mode branches), 6044-6751 (`single_host_likelihood`),
  6753-6919+ (`single_host_likelihood_batch`).
- `arguments.py`: 699-786 (`--host_z_kernel`, `--normalization_mode`, `--smear_global_selection`
  CLI definitions and defaults).
- `main.py`: 130-137 (realistic-venue recommended eval flags, logged at catalogue-realization
  time).
- `constants.py`: 95 (`SIGMA_V_PEC_KM_S = 0.0`).
- `validation/correspondence_1d.py`: 1156-1199 (`host_z_error_eff`, `_host_kernel_window`).
- `validation/pp_coverage.py`: 630, 641, 733-2389 (independent kernel reimplementation, no
  import of `bayesian_statistics`).
- `docs/derivations/PROPOSAL_HIER_SELFCAL_20260825.md` §2 (the instrument as proposed).
- `results/campaign51_20260728/realistic_20260729/PREREGISTRATION_P3_2D_20260825.md` §2
  (instrument-enumeration exemplar format).
