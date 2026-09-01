# FORENSICS_RECORD — g-c0-baseline RED stamp, m-head-rebaseline C0-prime

Classification of the mismatch documented in `GATE_RECORD.md`. No repair, no interpretation of
science numbers, no code edits — provenance and code-path tracing only, evidence quoted verbatim.

## Verdict: **MIXED**

- The **no-BH channel** (posteriors/h_0_73.json, event_likelihoods.csv's `L_cat_no_bh` /
  `num_log_term_no_bh` / `combined_no_bh`, and the dominant leaves of posteriors_with_bh_mass) is
  **fully H1** — expected-flip-delta, mechanism identified and quoted below.
- A **small residual in the with-BH diagnostic columns** (`event_likelihoods.csv`'s
  `L_cat_with_bh`, `num_log_term_with_bh`, `combined_with_bh`) is **unexplained H2** — it
  contradicts a specific registered invariance claim in the flip's own design document, and this
  forensics pass could not identify its mechanism.
- The gate's comparand selection was also found to be internally mislabeled (§3 below), but this
  is a documentation/provenance defect, not a driver of the RED magnitude — re-deriving against
  the flag-correct comparand reproduces the same numbers.

---

## 1. Comparand provenance (item 1)

`GATE_RECORD.md` §3 cites `wave3_20260830/{iiib,joint_r1}` task 21 as the comparand. Its
`run_metadata_21.json` (both venues):

```
git_commit: 1e092e82a7fea45fd20c23dfdbc2b96e562be322
catalogue_numerator_survival_2d = mz_sel
catalogue_numerator_survival_2d_center = eff
catalogue_leg_1d_mass_aware = off
catalogue_global_selection = phi
theta_phi_divisor = off
normalization_mode = absolute_marginal
```

The paired `provenance_6746354_21.json` (iiib) note field: *"wave3 blind HEAD readout, venue=iiib,
h=0.730, task=21 -- BLIND to the 2D-twin adoption (no --catalogue_numerator_survival_2d flag
passed)"* — i.e. `mz_sel`/`eff` is that run's own **unflagged default** at commit `1e092e82`
(pre-flip; the 2D twin was already production-adopted by row #223, independent of the 1D flip).

`git merge-base --is-ancestor 1e092e82 5e7fda16` → `YES_ANCESTOR` (pre-flip, confirmed). Full
window `1e092e82..1ec9514d` (11 commits) inspected — see §5.

## 2. C0-prime's own resolved flag state (item 2)

Fetched live from the cluster workspace (`run_20260902_graph1_c0prime_headrebaseline_{iiib,joint_r1}/run_metadata_21.json`,
not previously retrieved to the local `c0prime_eval/` copy — a gap in the GATE_RECORD's own
retrieval, noted for the record):

```
git_commit: 1ec9514dd1808c48b18c0792dce558e5bba0f116     (both venues)
catalogue_numerator_survival_2d = off
catalogue_numerator_survival_2d_center = unset
catalogue_leg_1d_mass_aware = auto        <- NOT overridden on the CLI
catalogue_global_selection = phi
theta_phi_divisor = off
normalization_mode = absolute_marginal
```

`git merge-base --is-ancestor 5e7fda16 1ec9514d` → `YES_ANCESTOR` (post-flip, confirmed; matches
LAUNCH_RECORD's own preflight check).

## 3. What `auto` resolves to, and in which venues (item 3)

`bayesian_statistics.py:4160` (evaluate(), inside the `catalogue_leg_1d_mass_aware` setup block):

```python
if _cat_leg_1d_ma == "auto":
    _cat_leg_1d_ma = (
        "on"
        if (
            self._catalogue_numerator_survival == "phi"
            and self._catalogue_global_selection == "phi"
            and self._theta_phi_divisor == "off"
        )
        else "off"
    )
```

This condition is **venue-agnostic and independent of `catalogue_numerator_survival_2d`** — it
checks only the 1D `catalogue_numerator_survival` (default `"auto"` → resolves to `"phi"` under
`normalization_mode="absolute_marginal"`, which both venues use and neither venue's sbatch
overrides), `catalogue_global_selection` (explicit `"phi"` in both venues' CLI), and
`theta_phi_divisor` (default `"off"`, not overridden by either venue). All three hold in **both**
iiib and joint_r1. **`auto` therefore resolves to `"on"` in both venues of the C0-prime gate** —
the flip is not iiib-only; the A18 arm being iiib-first (row #285) does not mean the code-level
`auto` guard is scoped to iiib. C0-prime effectively ran with `catalogue_leg_1d_mass_aware="on"`,
task-wide, both tasks.

The wave-3 comparand, at pre-flip commit `1e092e82`, has no `"auto"` state to resolve to — the
hardcoded pre-flip default was the literal string `"off"` (confirmed in its own
`run_metadata_21.json`, both venues, both the blind headreadout and the c0prime_off gate — see
§4). **This is the core H1 mechanism**: C0-prime = `catalogue_leg_1d_mass_aware="on"` (auto-resolved,
post-flip); wave-3 comparand = `catalogue_leg_1d_mass_aware="off"` (hardcoded, pre-flip). Same
seed (777021), same h (0.730), same dataset pins, same catalogue_numerator_survival_2d value
(`off`/`unset` — confirmed against the flag-correct comparand, §4) — the ONLY code-relevant input
that differs between the two runs.

## 4. Comparand mislabeling — a secondary, non-driving defect

The C0-prime sbatch script's header comment claims: *"reproduce the wave-3 blind HEAD readout's
own h=0.730 row ... bit-for-bit at the CURRENT commit ... by passing the SAME explicit flags that
wave-3's blind resolution used at ITS commit (pre-flip: --catalogue_numerator_survival_2d off
--catalogue_numerator_survival_2d_center unset)."* This is **false as stated**: the wave-3 blind
headreadout (`wave3_20260830/{iiib,joint_r1}`, the comparand GATE_RECORD actually used) resolved
`mz_sel`/`eff`, not `off`/`unset` (§1). The flag-exact match to C0-prime's explicit
`off`/`unset` CLI is a **different** banked directory:
`wave3_20260830/c0prime_off_{iiib,joint_r1}` (job 6746274, row #281: *"WAVE-3 C0′ OFF-GATE: PASS,
BIT-IDENTICAL (both venues) ... all four posterior JSONs md5-identical to
headreadout_20260827/{iiib,joint_r1}"* — the pre-2D-twin-adoption baseline). Row #283 further
confirms `off`/`unset` and `mz_sel`/`eff` are deliberately **not** meant to agree: *"iiib 2D
Δmean_h = +0.002127, joint_r1 2D Δmean_h = +0.003519"* is the registered, non-zero effect of the
2D-twin adoption itself.

**This is checked, not just asserted**: re-running the GATE_RECORD's own max_abs characterization
against the flag-correct comparand (`c0prime_off_{iiib,joint_r1}`) instead of the mislabeled one
reproduces the posteriors_with_bh_mass max_abs to the reported digit — iiib
`216544.26303892955` at `.galaxy_likelihoods.889[0][1][0]`, joint_r1 `987610.0823674798` at
`.additional_galaxies_without_bh_mass.889[0][1][0]` — **identical** to GATE_RECORD's numbers. The
`posteriors/h_0_73.json` (no-BH) md5s are likewise identical between the two wave-3 comparand
choices (`563ef45b...` iiib, `681364526966e835696946c4733456bb` joint_r1 in both). **Conclusion:
the mislabeled comparand is a real provenance-hygiene defect in the sbatch/GATE_RECORD chain
(worth fixing before this gate is trusted again), but it does not drive or inflate the RED
magnitude** — `catalogue_numerator_survival_2d` does not touch the code paths responsible for
either the no-BH or the dominant with-BH deltas (§6).

## 5. The 8/11-commit window, full sweep (item 5)

`git log --oneline 1e092e82..1ec9514d` (11 commits, the wave-3-comparand → C0-prime-commit span):

| commit | touches `darksiren_emri/`? | effect |
|---|---|---|
| `49fa5ca6` docs row #279 | no | — |
| `b3f17674` docs row #281 | no | — |
| `02c1c5cd` docs row #282 | no | — |
| `3a702166` docs row #283 | no | — |
| `f72c24ee` docs (dataset registration) | no | — |
| `4de30e06` docs row #284 | no | — |
| `43752177` docs (grid extension amendment) | no | — |
| `38cc0f58` cluster: A18 sbatch (untracked-script convention) | no (cluster script only, not committed source) | — |
| **`5e7fda16`** **[PHYSICS] flip** | **YES** — `arguments.py`, `bayesian_statistics.py`, `main.py`, `validation/correspondence_1d.py` | the flip itself (§3) |
| `9ece4ace` docs (README) | no | — |
| `6c43f8f9` fix(driver)+docs row #287 | touches `darksiren_emri_test/bayesian_inference/test_theta_zwindow.py` (a **test** file, not production code) + `results/.../fanout1_20260829/hier_s0_driver.py` (a **campaign-local driver script outside the package**, `gate_eng`'s b-only-node KeyError guard) | **neither is on evaluate()'s call path** — confirmed by reading both diffs; no effect on C0-prime's outputs |
| `37de5a65` docs row #288 | no | — |
| `198255b9` docs row #289 (graph charter) | no | — |
| `1ec9514d` docs row #290 (ratification) | no | — |

**`5e7fda16` is the only commit in the entire window that can affect a `python -m darksiren_emri
--evaluate` output.** No other confound exists in the commit history.

## 6. Whether the with-BH mismatch is explainable by the flip (item 4)

### 6a. Dominant with-BH deltas — H1, mechanism identified and quoted

`bayesian_statistics.py:5990-6035` (evaluate(), building `posterior_data_with_bh_mass`):

```python
results_with_bh_mass = _starmap_host_batches(
    pool, possible_host_galaxies_with_bh_mass, ...,
    catalogue_leg_1d_mass_aware=self._catalogue_leg_1d_mass_aware,
    sigma4d_mass_kernel=self._sigma4d_mass_kernel,
)
results_without_blackhole_mass = _starmap_host_batches(
    pool, possible_host_galaxies_reduced, ...,
    catalogue_leg_1d_mass_aware=self._catalogue_leg_1d_mass_aware,
    sigma4d_mass_kernel=self._sigma4d_mass_kernel,
)
...
self.posterior_data_with_bh_mass[GALAXY_LIKELIHOODS][detection_index] = galaxy_likelihoods
...
self.posterior_data_with_bh_mass[ADDITIONAL_GALAXIES_WITHOUT_BH_MASS][detection_index] = additional_likelihoods
```

**Both** the with-BH-mass candidate batch and the without-BH-mass candidate batch are called with
the SAME resolved `catalogue_leg_1d_mass_aware` token. Inside `single_host_likelihood_batch`
(`:8083-8092`, docstring self-documents this), when `evaluate_with_bh_mass=False` the function
returns:

```python
return np.column_stack([
    numerator_without_bh_mass, denominator_without_bh_mass,
    quadrature_weight_outside_grid_numerator, quadrature_weight_outside_grid_denominator,
])
```

and `numerator_without_bh_mass` is exactly the quantity multiplied by
`catalogue_leg_1d_mass_aware_factor(...)` when the flag resolves `"on"` (`:7649-7660`,
`:8464-8501`). The code's **own comment**, `:8407-8410`:

> `# NOTE (A13): applied for BOTH evaluate_with_bh_mass values — the with-BH host batch's r[0] is
> ALSO a no-BH numerator that feeds L_cat_no_bh (the all_results_without_bh concatenation in the
> caller), so gating on the channel flag would silently engage the cell on a host subset only.`

i.e. **this is a documented, by-design coupling**, not a bug: every host-batch call — including
the one over `possible_host_galaxies_with_bh_mass` — returns a no-BH-leg numerator at slot `[0]`
that IS flag-sensitive, and that slot is stored verbatim inside the `posteriors_with_bh_mass` JSON
(`galaxy_likelihoods` is `zip(catalog_index, results_with_bh_mass)`, so the observed max-delta path
`.galaxy_likelihoods.889[0][1][0]` = candidate 0 → returned-tuple → likelihood-array → slot 0 =
exactly `numerator_without_bh_mass`). Likewise `additional_galaxies_without_bh_mass` is literally
the `results_without_blackhole_mass` batch, entirely the no-BH leg, run over reduced-catalogue
candidates. **This fully accounts for the huge raw-leaf deltas (216544 iiib / 987610 joint_r1) —
they are the documented no-BH-leg numerator embedded inside the file that happens to be labeled
"with_bh_mass," not a defect in the with-BH physics itself.**

`PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md:1058-1061` independently documents the same fact at
design time: *"(_starmap_host_batches, and the two call sites in p_Di that invoke it for the
with-BH and [without-BH galaxy sets]) ... the with-BH batch's r[0] no-BH numerator also feeds
L_cat_no_bh."* This is a pre-registered, acknowledged consequence of the flip — H1, cleanly.

### 6b. Residual with-BH diagnostic-column deltas — H2, unexplained

But `event_likelihoods.csv`'s **named** with-BH columns — `L_cat_with_bh`, `num_log_term_with_bh`,
`combined_with_bh` — are, per direct code trace, built from a **different** slot that should be
untouched by the flag. `bayesian_statistics.py:6206-6223` (production `absolute_marginal` branch):

```python
cat_num_sum_with_bh = weighted_sum([r[2] for r in results_with_bh_mass], weights_with_bh)
L_cat_with_bh_mass = (
    cat_num_sum_with_bh / global_denom_with_bh if global_denom_with_bh > 0 else 0.0
)
```

— `r[2]`, not `r[0]`; and `global_denom_with_bh` is the object `bayesian_statistics.py:6109-6110`'s
own comment calls *"deliberately untouched"* by the flag. `combined_with_bh_mass` at `:6727-6729`
is `(alpha_G_phi * L_cat_with_bh_mass + B_num_wbh_phi) / D_tilde_phi` — none of `alpha_G_phi`,
`B_num_wbh_phi`, `D_tilde_phi` reference `_catalogue_leg_1d_mass_aware` anywhere in their
construction (confirmed by direct grep of every use site, §-listing in item 3's search above).

This matches the flip's **own registered invariance claim**,
`PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md:598`: *"D_tilde_phi, alpha_G_phi, r_Malm, w_G,
L_cat_with_bh, combined_with_bh columns max_abs 0.0 across 'on'/'off' on EVERY event"* (registered
result R7/R11, validated by the flip commit's own unit tests — "R14 tests amended + 3
auto-resolution tests," row #286).

**The actual C0-prime gate measures this claim FALSE at production scale**:

| venue | column | max_abs (measured) | registered claim |
|---|---|---|---|
| iiib | L_cat_with_bh | 0.0047554377123987 | 0.0 (R7/R11) |
| iiib | num_log_term_with_bh | 0.17719762362510494 | (same invariance implied) |
| iiib | combined_with_bh | 0.00029467945313679995 | 0.0 (R7/R11) |
| joint_r1 | L_cat_with_bh | 0.0035793250588652004 | 0.0 (R7/R11) |
| joint_r1 | num_log_term_with_bh | 0.17342658274017353 | (same invariance implied) |
| joint_r1 | combined_with_bh | 0.0002534242715218 | 0.0 (R7/R11) |

For scale: `num_log_term_with_bh` ranges ~6.7–18.9 across events (iiib), so a 0.177 max delta is a
genuine ~1–2% shift on that column's own scale, not floating-point noise (~1e-16 would be
expected from summation-order nondeterminism alone). `L_cat_with_bh`'s mean-abs is ~0.0032 against
a 0.0048 max delta — for low-`L_cat_with_bh` events the flip-associated delta can exceed the
column's own typical value.

**This forensics pass could not identify the mechanism for 6b.** Candidates considered and their
status:
- `catalogue_numerator_survival_2d` (the mislabeled-comparand flag, §4) — **ruled out**: identical
  magnitude reproduces against the flag-matched `c0prime_off` comparand (§4), and `r[2]`'s own
  computation does not reference this flag either.
  do not reference `catalogue_leg_1d_mass_aware` in this or any other production-mode branch.
- A separate, undocumented coupling not caught by grep of `_catalogue_leg_1d_mass_aware` use
  sites, OR floating-point-level nondeterminism from a different source entirely (worker-pool
  chunking order, library version drift outside `darksiren_emri/` — not checked in this pass, e.g.
  `numpy`/`scipy` pin drift between the two run environments — not ruled out, not confirmed) — both
  remain open.
- The registered R7/R11 invariance (design doc §598, row #286's "2006 passed" suite) was validated
  by **unit-scale fixtures**, not the full 1588-event, seed-777021, h=0.730 production
  configuration this gate runs — a scope gap between "tested" and "the actual gate's own inputs"
  that this record flags but does not close.

## Summary for the author

**MIXED.** The RED stamp's no-BH channel and its dominant with-BH deltas are fully explained by
the row #286 flip (`catalogue_leg_1d_mass_aware`: `off`→`auto`→`on`, both venues) via a
documented, pre-registered code coupling (§6a) — this is H1, gate mis-specification (wrong-epoch
comparand), not a pipeline defect. A separate, smaller signal in the with-BH diagnostic columns
(`L_cat_with_bh`, `num_log_term_with_bh`, `combined_with_bh`) directly contradicts the flip's own
registered exact-zero invariance claim and its mechanism is not identified here — this is H2,
genuinely unexplained, and should not be waved through under the H1 umbrella.

**Options for what the gate should have compared (not a decision — a fresh [RULE] for the
author):**
1. A **pre-flip-pinned C0-prime** re-run at commit `1e092e82` (or any pre-flip commit) against the
   existing wave-3 banked row, isolating the code/environment axis from the flip axis entirely.
2. An **explicit-`off`-pinned** C0-prime arm (`--catalogue_leg_1d_mass_aware off` at the current,
   post-flip commit) against the wave-3 banked row — this reintroduces the pre-flip
   counterfactual at the current commit and should reproduce H1's no-BH/dominant-with-BH deltas as
   exact zero if H1 is the complete story for those columns, while leaving 6b's residual isolated
   for direct inspection.
3. A **fresh post-flip comparand** banked once (the blind HEAD arrays this same launch already
   produced, jobs 6764461/6764462) so future g-c0-baseline gates compare like-epoch to like-epoch
   going forward, with the flag-matching defect from §4 fixed in the sbatch header/comment.
4. Independently of (1)-(3): re-run the R7/R11 registered check at the actual production
   configuration (seed 777021, h=0.730, both venues) rather than at unit-fixture scale, to
   determine whether 6b's residual is real or an artifact of this forensics pass's own comparison
   method.

---

## ADDENDUM (mechanism trace)

Chair verification found decisive structure in the §6b residual and requested one more forensic
hop. Findings below; no code edits, no commits, no jobs.

### Chair-measured input (CHAIR-PROVIDED, not independently re-derived by this pass)

> iiib: `L_cat_with_bh` ndiff=982/1588 (max_abs 4.755e-3), `num_log_term_with_bh` ndiff=972
> (max_abs 0.1772), `combined_with_bh` ndiff=972 (max_abs 2.947e-4); `L_cat_no_bh` ndiff=982,
> `num_log_term_no_bh` ndiff=975. with-BH-diff rows are a STRICT SUBSET of no-BH-diff rows
> (intersection 972 of 972). joint_r1: same pattern, `L_cat_with_bh` ndiff=1094,
> `num_log_term_with_bh` 1079 ⊂ no_bh 1083. NOT storage precision — deltas survive rounding to 6
> decimals. 982 = exactly the candidate-bearing event count in iiib (dv-jr1-transform record §1:
> "982 candidate-bearing events").

### Step 1 — elimination of (a), (b), (d) by direct formula trace

For the taken branch (`_use_g_inside and self.h in self._beta_G_phi_table` — confirmed the active
branch for `normalization_mode=absolute_marginal` + `catalogue_global_selection=phi`, both runs),
`bayesian_statistics.py:6727-6729`:

```python
combined_with_bh_mass = float(
    (alpha_G_phi * L_cat_with_bh_mass + B_num_wbh_phi) / D_tilde_phi
)
```

`alpha_G_phi`, `D_tilde_phi`, `r_Malm`, `w_G`, `B_num`, `B_num_wbh` are ALL in GATE_RECORD's own
13-exact-zero-column list — **empirically confirmed max_abs = 0.0 already, both venues**. This
rules out (a) (shared mixture/normalization recomputed differently under "on") directly: if
`alpha_G_phi`/`D_tilde_phi` moved, `combined_with_bh` would move on every event with `D_tilde_phi >
0` uniformly, not on a 972/982-event subset, and the mixture-weight columns themselves would show
nonzero max_abs — they do not. `path_a_mixture_objects(beta_G_phi, beta_Gbar_phi, sigma_phi,
global_denom_with_bh)` at `:6690-6692` has no reference to `catalogue_leg_1d_mass_aware` anywhere
in its own body or its inputs' construction (`:4784-4792`, `precompute_global_catalog_selection`
called with `with_bh_mass=True` and NO `catalogue_leg_1d_mass_aware` kwarg at all).

(b) — the with-BH numerator accessor switching — is ruled out by exhaustive grep: every call site
of `catalogue_leg_1d_mass_aware_factor` (`:7059`) is gated by `_cat_leg_1d_ma_on`, and every one of
those four call sites (`:7650`, `:7710`, `:8466`, `:8488`) sits inside the
`numerator_without_bh_mass`/`_num_integrand` construction only — never inside the with-BH-mass
block (`mz_integral`, `numerator_with_bh_mass`, read in full from `single_host_likelihood_batch`
`:8656-8710`). The ONE flag that does touch `mz_integral` is `catalogue_numerator_survival_2d`
(`_cat_surv_2d_on`, `:8681-8698`) — a **different** flag, confirmed `off` identically in both the
C0-prime run and the flag-matched wave-3 comparand (§4), so this branch is provably inactive in
both runs being diffed.

(d) — divisor leak — is ruled out the same way as (a): `global_denom_with_bh` is never reassigned
under any value of `catalogue_leg_1d_mass_aware` (only `global_denom_no_bh` is, at `:6117-6135`,
row #292's finding), and its own build call (`:4784`) never receives the flag.

### Step 2 — narrowing to `numerator_with_bh_mass` (r[2]), and the strict-subset mechanism

By elimination, since `combined_with_bh_mass`'s only non-invariant input is `L_cat_with_bh_mass`
(`:6218-6222`, `absolute_marginal` branch: `L_cat_with_bh_mass = cat_num_sum_with_bh /
global_denom_with_bh`), and `global_denom_with_bh` and `weights_with_bh` (`:6053`, pure function of
host mass/redshift, no flag reference) are both confirmed flag-independent, the **entire residual
must live in `cat_num_sum_with_bh = weighted_sum([r[2] for r in results_with_bh_mass],
weights_with_bh)`** (`:6215-6216`) — i.e. in the per-candidate `numerator_with_bh_mass` values
themselves (`single_host_likelihood_batch`'s return column index 2, `:8761`).

The **strict-subset structure is explained exactly** by `:6214/6224`:

```python
if len(results_with_bh_mass) > 0:
    cat_num_sum_with_bh = weighted_sum([r[2] for r in results_with_bh_mass], weights_with_bh)
    L_cat_with_bh_mass = (cat_num_sum_with_bh / global_denom_with_bh
                           if global_denom_with_bh > 0 else 0.0)
else:
    L_cat_with_bh_mass = 0.0
```

An event that is candidate-bearing for the no-BH leg (contributes to the 982) but whose candidate
ball contains **zero** galaxies with a known BH-mass measurement falls into the `else: 0.0` branch
identically under both flag values — `L_cat_with_bh_mass` is `0.0` on/off, zero diff, structurally
guaranteed regardless of what `numerator_with_bh_mass`'s formula does. This predicts the observed
gap (982 − 972 = 10 iiib; 1094 − 1079 = 15 joint_r1) as exactly the count of candidate-bearing
events whose candidate ball has no BH-mass-known galaxy — a testable, structurally-forced
prediction, consistent with the measured subset relation (972 ⊂ 982, both directions of the guard
match the same underlying partition of events by "has ≥1 BH-mass-known candidate").

### Step 3 — the residual in r[2] itself: NOT located by static trace

Read `single_host_likelihood_batch` end-to-end for every dependency of `numerator_with_bh_mass`
(`:8656-8710`, `mz_integral`, `gw_3d`, `prior_num`, `host_M_eff`, the Eddington-shift branch
`:8598-8613`): **zero literal references to `catalogue_leg_1d_mass_aware` or `_cat_leg_1d_ma_on`
anywhere in this code.** Checked specifically for in-place aliasing between the no-BH block
(computed first in file order, `:8460-8511`, which DOES read the flag) and the with-BH block
(computed second, `:8656-8710`, sharing `gw_3d`/`prior_num`/`y_num_nodes`): every operation uses
`*` (new-array binary multiply), never `*=`/`+=` — `numerator_without_bh_mass = numerator_without_
bh_mass * _surv_factor` and `_num_integrand = _num_integrand * _surv_factor` both rebind to new
arrays, leaving `gw_3d`/`prior_num` untouched for the later with-BH block to read pristine. No
mutation bug found.

Also checked and ruled out: candidate-set membership (`get_possible_hosts_from_ball_tree`,
`:5724-5744`, takes no `catalogue_leg_1d_mass_aware` argument — purely geometric/mass-window
parameters, all confirmed identical between the two runs); row-dropping/NaN-masking between the
numerator and the later diagnostic weight computation (none found — `n` stays fixed, no filtering
after `:8511`).

**This forensics pass could not identify the file:line mechanism causing r[2] to differ.** The
formula, as written, is provably flag-invariant given identical inputs. The residual must therefore
originate either (i) upstream of `single_host_likelihood_batch`, in an input this pass did not
trace to its full construction (candidates flagged but not confirmed: any state on the shared
`SimulationDetectionProbability` object — `detection_probability._get_or_build_grid(h)`,
`simulation_detection_probability.py:1689`, is a single per-h cache read by BOTH the no-BH leg's
`detection_probability_with_bh_mass_interpolated` call, engaged only under "on" via
`catalogue_leg_1d_mass_aware_factor`, and the with-BH leg's own `detection_probability_with_bh_
mass_interpolated` call at `simulation_detection_probability.py:2090` — a shared cache touched by
one leg under "on" but not "off" is a plausible, UNVERIFIED next hop), or (ii) a numerical effect
this static-code-only pass cannot see without an actual controlled repro (e.g. a two-run,
single-event, flag-toggled invocation instrumented at `r[2]` directly). **Recommendation: this is
the next forensic hop, and it requires execution, not more reading.**

### Row #286 invariance-claim scope — the claim survives ONLY where it was actually tested

`PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md` §10 registers R7 as: *"Limit (L4), the C-C
identity: dark events with n_cand_no_bh = 0 have combined_no_bh bit-identical across 'on'/'off';
B_num, D_tilde_phi, alpha_G_phi, r_Malm, w_G, L_cat_with_bh, combined_with_bh columns max_abs 0.0
across 'on'/'off' on EVERY event"* (line 598) — this is the exact claim the task cited.

Its own retrospective (§20.5-adjacent status section) grades R7:

> **R7 (limit L4, the C-C identity): PARTIAL** -- implemented as "the with-BH channel columns are
> bit-identical between on/off" (both scalar and batch, both host-z-kernel modes), which is **the
> WORKER-LEVEL half of R7**; **the full p_Di-level empty-ball check (n_cand_no_bh = 0 on a live
> catalogue cell) was NOT run by this node** (out of unit-test scope; R13's live-catalogue
> engagement gate is the natural home for it and is also descoped here, below).

And R13 (the ≥99% live-catalogue engagement gate — the check that would exercise the flag against
real candidate data end-to-end):

> **R13 ... NOT RUN** -- requires a real GalaxyCatalogueHandler and BallTree, i.e. a live-catalogue
> cell, out of this node's unit-test scope ... left for the tree-2 verifier pass or for arm (a)/(b)
> below, both of which will exercise it as a side effect.

**Scope reconstruction:** R7/R11's "exact-zero on EVERY event" was measured by calling
`single_host_likelihood`/`single_host_likelihood_batch` directly with synthetic/unit-test fixture
arguments (isolated-function invariance: given identical inputs, does the with-BH formula branch on
the flag?) — answered **correctly, YES it doesn't**, and this pass's own independent code trace
(Steps 1-3) confirms the same formula-level fact. What R7 explicitly never tested is whether the
**full production pipeline** (`p_Di`/`evaluate()`, live GLADE+ catalogue, real candidate resolution,
real detection-probability object lifecycle) delivers bit-identical *inputs* to that formula across
"on"/"off" runs. Row #286's own validation basis (arm (c): a 41-node h-posterior MAP/mean check,
plus "Suite: 2006 passed / 6+1 skipped") never re-ran R7/R13 at p_Di/live-catalogue scale either —
arm (c) measured the 1D channel's posterior, not a with-BH column diff.

**Conclusion: the claim is TRUE in its tested scope (isolated worker-function formula invariance,
confirmed independently by this pass) and was never validated — by the flip's own disclosure, twice
(R7 "PARTIAL", R13 "NOT RUN") — at the scope the C0-prime gate now exercises (full production
p_Di, live catalogue, seed 777021, h=0.730, both venues).** The gate is not contradicting a
confirmed result; it is running the FIRST check of a claim whose production-scale half was always
an open, disclosed gap. This reclassifies §6b from "H2, mechanism unknown, contradicts a confirmed
invariant" to "H2, mechanism unknown, exercises an invariant that was disclosed as UNTESTED at this
scope" — still a genuine open anomaly (the formula-level proof does not by itself explain where the
production-scale delta comes from), but not a regression against anything actually verified before.

---

## ADDENDUM 2 (repro)

Coordinator-authorized minimal local repro to confirm/kill the `_get_or_build_grid` shared-cache
lead flagged in Addendum 1. Scratch script only, under `c0prime_eval/repro/`, never touching
`darksiren_emri/`. CPU-only, local, not a registered measurement. No code edits, no commits, no
cluster jobs.

### Setup

`c0prime_eval/repro/run_repro.py` instantiates `BayesianStatistics` directly (bypassing the CLI),
truncates `self.cramer_rao_bounds` to 5 iiib events with known BH-mass candidates (46, 231, 744,
1061, 1317 — the top with-BH-delta events from the *mismatched* wave-3 blind comparand, per
Addendum 1), and calls `.evaluate()` twice at `h=0.730`, `seed=777021`, with every CLI flag copied
verbatim from the C0-prime sbatch (`catalogue_numerator_survival_2d="off"`,
`catalogue_numerator_survival_2d_center="unset"`, `catalogue_global_selection="phi"`,
`normalization_mode="absolute_marginal"`, `host_z_kernel="volume_deconv"`, etc.), varying only
`catalogue_leg_1d_mass_aware` ("off" then "on"). Inputs are the locally-banked, md5-pinned copies
of the C0-prime gate's own datasets (`prepared_cramer_rao_bounds.csv` md5
`9a1f2a14384a9281c97ca3be312ddaab`, `reduced_galaxy_catalogue.csv` md5
`c52c13b5cab61f6b3f04bbe202550969` — both verified against the pins in `GATE_RECORD.md` before
running). `num_workers=2` (forkserver context, confirmed live via `ps` during the run — NOT
`fork`, so worker processes do not inherit any parent-process monkeypatch; each worker is started
fresh from `simulation_detection_probability.py`'s own top-level import, so the shared-cache
hypothesis is tested as it actually behaves in production, not artificially disabled by the harness).

### Step 1 — does the delta reproduce in isolation?

```
=== DELTAS (on - off) ===
L_cat_no_bh: max_abs=6.810871e-02  nonzero_events=5/5
L_cat_with_bh: max_abs=0.000000e+00  nonzero_events=0/5
combined_no_bh: max_abs=4.414396e-03  nonzero_events=5/5
combined_with_bh: max_abs=0.000000e+00  nonzero_events=0/5
num_log_term_no_bh: max_abs=5.692734e-01  nonzero_events=5/5
num_log_term_with_bh: max_abs=0.000000e+00  nonzero_events=0/5
```

The no-BH channel moves on every event (confirming the run actually exercises the flip — H1
reproduces locally, sanity check passed). **The with-BH channel is exact-zero on every one of the
5 events**, with `catalogue_numerator_survival_2d` held fixed at `"off"` (matching C0-prime's own
CLI) for both legs.

This directly reproduces, from a fresh local run rather than a re-read of banked files, the
correction already found by re-checking GATE_RECORD's own comparand choice (below): **the §6b
residual reported in the original FORENSICS_RECORD/Addendum 1 does not exist when the comparison
holds `catalogue_numerator_survival_2d` fixed.** Re-diffing the ALREADY-BANKED files with the
correct (flag-matched) comparand confirms the same thing at full production scale, not just on
these 5 events:

```
                              vs c0prime_off (flag-matched, off/unset)   vs wave3_20260830/iiib (mismatched, mz_sel/eff)
L_cat_with_bh        max_abs   0.000000e+00                              4.755438e-03
num_log_term_with_bh max_abs   0.000000e+00                              1.771976e-01
combined_with_bh     max_abs   0.000000e+00                              2.946795e-04
L_cat_no_bh          max_abs   4.845906e+00                              4.845906e+00   (identical either way)
combined_no_bh       max_abs   1.198737e-02                              1.198737e-02   (identical either way)
```

(`c0prime_off_iiib`/`c0prime_off_joint_r1` = `results/.../wave3_20260830/c0prime_off_{iiib,joint_r1}/simulations/diagnostics/event_likelihoods.csv`,
the banked wave-3 arm that actually used `catalogue_numerator_survival_2d=off`/`unset`, row #281:
*"WAVE-3 C0′ OFF-GATE: PASS, BIT-IDENTICAL (both venues)... all four posterior JSONs
md5-identical to headreadout_20260827/{iiib,joint_r1}."*)

**This corrects the original FORENSICS_RECORD §4/§6b.** §4 had checked `posteriors_with_bh_mass`
raw JSON leaves against both wave-3 comparands and found identical magnitudes either way (true, and
still stands — that residual is the documented r[0]-embedding coupling, H1, unaffected by
`catalogue_numerator_survival_2d`), but did NOT re-check `event_likelihoods.csv`'s named with-BH
diagnostic columns against the flag-matched comparand before concluding (in Addendum 1) that they
were a genuine, unexplained H2 residual. They are not: **the §6b residual is entirely an artifact
of GATE_RECORD's mis-selected comparand (§4's "mislabeling" finding) — the SAME defect already
identified, just not chased into this specific file.** Once the comparand's `catalogue_numerator_
survival_2d` is matched to C0-prime's own (`off`/`unset`), `L_cat_with_bh`/`combined_with_bh`/
`num_log_term_with_bh` are exact-zero, exactly as R7/R11 registered.

### Step 2 — bisect (a): cache-clear / rebuild between legs

Ran a third pass, `"on"`, with `SimulationDetectionProbability.__init__` monkeypatched (parent
process only — confirmed inert on workers per the forkserver note above, but exercised anyway as
the harness's own no-op control) to force `self._shared_grid = None` / `self._grid_cache = {}`
immediately after construction, guaranteeing the per-h grid is rebuilt from scratch rather than
reused from any earlier state in this process:

```
=== DELTAS (on_cachefresh - on) -- should be exactly 0 if cache state is inert ===
L_cat_no_bh: max_abs=0.000000e+00
L_cat_with_bh: max_abs=0.000000e+00
combined_no_bh: max_abs=0.000000e+00
combined_with_bh: max_abs=0.000000e+00
num_log_term_no_bh: max_abs=0.000000e+00
num_log_term_with_bh: max_abs=0.000000e+00
```

Exact zero across every column, including the no-BH channel that DOES move under the flag —
confirming the grid rebuild changes nothing, consistent with `_get_or_build_grid`
(`simulation_detection_probability.py:1689`) building a fixed, deterministic, query-independent
grid from `self._log_M_z`/`self._dl_bins`/`self._mass_bins` (`_grid_support`, `:1721-1737`) with no
call-order or history dependence, exactly as read in Addendum 1.

### Step 3 — call-order sensitivity

Not separately run: Step 1's own RUN 1/RUN 2 pair already varies call order end-to-end (RUN 1 =
`"off"`, no extra grid-touching call, then RUN 2 = `"on"`, which DOES make the extra
`detection_probability_with_bh_mass_interpolated` call from the no-BH leg before the with-BH leg
runs, per Addendum 1's Step 1 trace) and shows exact-zero with-BH delta regardless. No further
order permutation is informative given Step 2's direct rebuild-vs-reuse null result.

### Verdict: **EXONERATED-cache**

The `_get_or_build_grid` shared-cache lead is **exonerated**, both by architecture
(`forkserver`/`spawn` — confirmed live via `ps` during the run, not `fork` — plus a deterministic,
query-independent grid build) and by direct empirical bisect (cache-fresh vs cache-reused: exact
zero difference, Step 2). Survival record: cache-clear bisect (Step 2) tried, delta unaffected
(0.0 either way) — no further cache-adjacent lead visible from the repro's own printed intermediate
columns (`L_cat_no_bh`/`L_cat_with_bh`/`combined_*`/`num_log_term_*`, the finest grain this
repro instruments).

**More importantly, this repro (together with the banked-file re-check in Step 1) retires the §6b
finding itself, not just the cache hypothesis for it.** There is no longer an unexplained with-BH
residual to attribute to *any* mechanism: `L_cat_with_bh`/`combined_with_bh`/`num_log_term_with_bh`
are exact-zero under the 1D flip when `catalogue_numerator_survival_2d` is held fixed, both in a
fresh local run (5 events, this repro) and in the full banked production data (982/1094
candidate-bearing events, re-diffed against the correct comparand, Step 1). **§6b's original
"H2, unexplained" status is WITHDRAWN**; row #286's invariance claim (§10, R7's tested scope) is
confirmed to hold, without qualification, at C0-prime's own production configuration — the earlier
Addendum 1 finding that R7/R13 were untested at p_Di/production scale is still true as a
documentation-discipline point (nothing in this repo's test suite would have caught a REAL
production-scale violation had one existed), but no such violation exists at C0-prime's actual
scale once the comparand is corrected.

### Order-of-magnitude quantification (item 4)

Not applicable in the form originally anticipated: the repro's per-event with-BH delta (0.0) does
not have a nonzero "chair-measured production delta" counterpart to compare against once the
comparand is corrected — the chair-measured deltas (`L_cat_with_bh` max_abs 4.755e-3 iiib /
num_log_term_with_bh 0.1772, etc.) are reproduced EXACTLY by the mismatched-comparand banked
re-diff in Step 1 (`4.755438e-03` / `1.771976e-01` — matches chair-provided figures to the digit)
and equally exactly reproduced as **zero** by the flag-matched comparand and this repro. The two
"deltas" are answers to two different questions (flag-only vs flag+`catalogue_numerator_survival_2d`
confounded); both are now fully accounted for, at the correct order of magnitude, by known,
documented mechanisms (H1 for no-BH via the flip; the row #223-adopted 2D-twin effect,
`catalogue_numerator_survival_2d`, for the with-BH deltas the chair measured) — no unexplained
residual remains at either scale.

### Updated overall classification (supersedes Addendum 1's MIXED verdict)

**H1, fully explained, no residual.** The RED stamp is entirely accounted for by (i) the row #286
flip (`catalogue_leg_1d_mass_aware`: hardcoded `off` pre-flip → `auto`→`on` post-flip, both venues,
via the documented r[0]-embedding coupling, Addendum 1 §6a) acting against a comparand from a
different code epoch, PLUS (ii) GATE_RECORD's own comparand mis-selection (original §4) additionally
carrying the row #223-adopted 2D-twin effect (`catalogue_numerator_survival_2d`: `mz_sel`/`eff` in
the mis-selected wave-3 blind comparand vs `off`/`unset` in both C0-prime and the correct
flag-matched comparand) into the with-BH diagnostic columns specifically. Both are gate
mis-specification (wrong-epoch / wrong-flag-config comparand), not pipeline defects. The options
listed at the end of the original FORENSICS_RECORD for what the gate should compare going forward
stand unchanged and are now more clearly motivated: any of them (pre-flip-pinned rerun,
explicit-off-pinned arm at the current commit, or a fresh post-flip comparand) would additionally
need to hold `catalogue_numerator_survival_2d` fixed to the C0-prime venue's own value, which the
existing `c0prime_off_{iiib,joint_r1}` banked arm already does correctly — it is the comparand this
gate should have used throughout.
