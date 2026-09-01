# rd-rphi-note — closure note: g-znorm, first standing panel evaluation, flipped 1D catalogue leg

**Node:** rd-rphi-note (type: read, closure chain, §1.10 of
`RESEARCH_GRAPH_1_PROPOSAL_20260901.md`)
**Authorization (verbatim, ledger row #290):** ledger row #290 ratified decisions-table row 12:
"the closure note as the first standing g-znorm evaluation on the flipped leg; STANDING: the
section-2 panel evaluates before every science read in this graph — scope: this batch only;
lapses at graph close." Tag: DO + STANDING, disposition Approved / Granted.
**Node spec (§1.10):** "rd-rphi-note | read | the written confirmation: the FIRST standing
g-znorm panel evaluation on the flipped production leg, tied into the A18 closure record | feeds
from commit 5e7fda16 (row #286; hash quoted at row #288) + row #282 | g-znorm (this node IS its
first panel evaluation) | abs dev <= 1e-6 green, > 1e-3 anomalous (infra 2.5); anomalous -> STOP
d-rphi-retire and reopen as fresh RULE | cheap | sonnet / low".

Read/verification node only. No code edits, no commits, no cluster submission, no registered
production measurement run.

---

## 1. Stamp: **GREEN**

| quantity | value |
|---|---|
| g-znorm band | abs dev ≤ 1e-6 green; > 1e-3 anomalous (infra 2.5) |
| measured deviation, production divisor identity | **0.0 (exact)** — `global_denom_no_bh` is a
literal Python assignment `= global_denom_with_bh` under `catalogue_leg_1d_mass_aware == "on"`
(`bayesian_statistics.py:6125-6126`); there is no floating-point division or subtraction between
the two, so the deviation is not merely small, it is not computed at all — numerator and divisor
share one float value by construction. |
| measured deviation, closed-form Z(h)=1 identity (local numeric check, §3 below) | `\|Z_on − 1\| = 0.000e+00` (`1.0 == 1.0` in float64) for a non-degenerate synthetic fixture (r_Malm = 0.850, informative per the R2 can-fail control) |
| control (pre-flip "off" path, same fixture) | `\|Z_off − 1\| = 1.6908e-2` — confirms the fixture is discriminating (the "off" leg genuinely fails the identity, as expected; this is the mismatch c-rphi-mismatch names) |

Both the code-path demonstration (§2) and the numerical check (§3) agree: **GREEN**, deviation
0.0, four orders of magnitude inside the 1e-6 green band. `d-rphi-retire` is UNBLOCKED.

---

## 2. Code-path demonstration (post-flip HEAD, commit `5e7fda16` onward)

### 2.1 The default is "auto", and "auto" resolves to "on" on the production stack

`darksiren_emri/bayesian_inference/bayesian_statistics.py:4410-4430`:

```python
_cat_leg_1d_ma = str(catalogue_leg_1d_mass_aware)
if _cat_leg_1d_ma not in ("auto", "off", "on"):
    raise ValueError(...)
if _cat_leg_1d_ma == "auto":
    # PRODUCTION DEFAULT since 2026-08-31 (rows #284-#286, gate doc
    # §6.3 Z-CONFIRMED: 1D map_h 0.665 / mean_h 0.66699 in band):
    # engage the mass-aware leg exactly where its guard stack holds
    # (the row #197/#253 auto->engaged pattern); resolve "off"
    # silently elsewhere (legacy/other normalization modes).
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

The class default (`bayesian_statistics.py:3695`) and the `evaluate()` kwarg default
(`bayesian_statistics.py:4036`) are both `"auto"`. On the `absolute_marginal` production stack
(`catalogue_numerator_survival="phi"`, `catalogue_global_selection="phi"`,
`theta_phi_divisor="off"`) this resolves to `"on"` — the flip commit's own message states the
Z-CONFIRMED arm (c) result that authorized it (1D MAP 0.600→0.665, mean 0.6053→0.667).

### 2.2 Divisor: literal identity, not a computed ratio

`bayesian_statistics.py:6117-6135`:

```python
global_denom_with_bh: float = self._global_cat_denom_with_bh.get(self.h, 0.0)
# [HIER T2.3] mass-aware 1D catalogue leg instrument
# ... "on" (guarded at setup) replaces
# the no-BH catalogue divisor by Sigma_4D (global_denom_with_bh,
# ALREADY IN HAND, no new computation) -- the SAME divisor
# Sigma_4D's own with-BH branch already computes. "off"
# (default): byte-identical to the pre-flag ternary below.
global_denom_no_bh: float = (
    global_denom_with_bh
    if getattr(self, "_catalogue_leg_1d_mass_aware", "off") == "on"
    else (
        getattr(self, "_global_cat_selection_phi_theta", {}).get(
            self.h, self._global_cat_selection_phi.get(self.h, 0.0)
        )
        if getattr(self, "_catalogue_global_selection", "s3d") == "phi"
        else self._global_cat_denom_no_bh.get(self.h, 0.0)
    )
)
```

Under `"on"`, `global_denom_no_bh` **is** `global_denom_with_bh` — the same Python float, no
arithmetic performed between them. This is the "divisor" half of the matched-content claim: the
no-BH (catalogue) leg's divisor and the with-BH leg's divisor (Σ_4D) are the same object.

### 2.3 Numerator: the same per-galaxy accessor as the with-BH leg

`bayesian_statistics.py:7059-7118`, `catalogue_leg_1d_mass_aware_factor()`:

> "replaces the population-average catalogue-numerator survival `S_bar_phi(z;h)` by the SAME
> per-galaxy with-BH survival Sigma_4D already evaluates for that galaxy —
> `S_4D(d_L(z;h), M_g(1+z))` ... reusing the SAME accessor and SAME isotropic-sky convention as
> Sigma_4D's own with-BH branch (`precompute_global_catalog_selection`, point query :3022-3038 /
> kernel query :2996-3020) ... the numerator's mass measure can never differ from the divisor's
> (Sigma_4D), so an unpaired point/kernel combination is exactly the [NUMERATOR-ONLY-CLEAN]
> defect class."

Site N1, where this factor multiplies the no-BH survival term, is at
`bayesian_statistics.py:7653` (`single_host_likelihood`) and `:7715`/`:8470,8492`
(`single_host_likelihood_batch`).

### 2.4 Mixture weight: `alpha_G_phi` on both legs

`bayesian_statistics.py:6715-6728`:

```python
# ... "on" (guarded at setup) replaces beta_G_phi by alpha_G_phi -- the
# IDENTICAL float the 2D assembly below already consumes (:6501) -- the
# no-mass-likelihood image of the 2D mixture. "off": byte-identical
# (beta_G_phi, unchanged).
_cat_num_weight_no_bh = (
    alpha_G_phi
    if getattr(self, "_catalogue_leg_1d_mass_aware", "off") == "on"
    else beta_G_phi
)
combined_without_bh_mass = float(
    (_cat_num_weight_no_bh * L_cat_without_bh_mass + B_num_phi) / D_tilde_phi
)
combined_with_bh_mass = float(
    (alpha_G_phi * L_cat_with_bh_mass + B_num_wbh_phi) / D_tilde_phi
)
```

Under `"on"` the no-BH numerator's weight (`alpha_G_phi`) is literally the same variable used for
the with-BH numerator two lines below — not a separately-derived, potentially-mismatched
`beta_G_phi`.

**Conclusion of the code-path reading:** under the flipped `"auto"→"on"` branch, numerator
(§2.3), divisor (§2.2), and mixture weight (§2.4) of the no-BH catalogue leg are each the SAME
object/accessor the with-BH leg already uses — not independently-derived quantities that happen
to agree. Z(h)=1 is therefore an identity of construction, not a numerically-verified
approximation, on this branch. This is the guard-railed sibling of the `"off"` branch, which is
exactly the pre-flip mismatch (§2.2's `else` clause routes through `_global_cat_selection_phi`,
the Σ^φ table that IS the mass-blind object c-rphi-mismatch names — see §4).

---

## 3. Numerical check (local, CPU-only, non-cluster, non-production-measurement)

Feasible and executed. The existing regression test
`darksiren_emri_test/bayesian_inference/test_catalogue_leg_1d_mass_aware.py::test_r2_z_equals_one_identity_under_on_and_not_under_off`
already implements exactly this check against the real production function
`path_a_mixture_objects` and the real `catalogue_leg_1d_mass_aware_factor`, using a synthetic,
deliberately non-degenerate catalogue (r_Malm = 0.850 ≤ the R2 can-fail-control bound of 0.9, so
the fixture is not accidentally close to the identity for a trivial reason).

Ran it directly (`uv run pytest ... -k test_r2_z_equals_one`): **1 passed**. Re-derived the
underlying floats outside the test's `assert_allclose` wrapper (same fixture, same seed
`20260830`, same functions imported from the module) to get the actual numbers rather than a
pass/fail bit:

```
r_Malm = 0.8503612574114741
Z_on   = 1.0
|Z_on - 1| = 0.000e+00
Z_off  = 1.0169076423251329
|Z_off - 1| = 0.016908
```

`Z_on` is `1.0` to the bit (float64 equality, not `atol`-bounded closeness) — deviation is exactly
`0.0`, four orders of magnitude inside the `≤1e-6` green band. The `"off"` control on the same
fixture deviates by `1.7e-2`, ≫ the `1e-3` anomalous threshold — confirming the fixture is
genuinely discriminating and the `"on"` result is not a degenerate/trivial pass.

This is a local, CPU-only script + the pre-existing repo test suite — no cluster job, no new
production evaluate() run, and not the registered `m-head-rebaseline` / `m-joint-r1-mass-aware`
production measurements (those remain separately gated, per §1.2/§1.3 of the graph spec).

---

## 4. Closure narrative: what c-rphi-mismatch claimed and why the flip moots it

**c-rphi-mismatch** (`CLAIM_P3_RPHI_20260822.md`, ledger rows referenced #168 in
`docs/PRIMER_BIAS_CHANNELS_20260822.md`): pre-flip, the catalogue-leg divisor used
`self._global_cat_selection_phi` — a population-averaged, mass-blind phi-marginal survival sum
`Σ^φ(h)` — against a numerator (`L_cat_without_bh_mass`) built from the SAME mass-blind
`S_bar_phi(z;h)` per-candidate factor. Both objects were mass-blind and, on their own, consistent
(`"off"` control gives `L_cat_off = 1.0` exactly in §3 — that half of the ratio was never the
problem). The mismatch flagged by c-rphi-mismatch was against the **with-BH** channel: the
production object's mass-aware divisor `Σ_4D`/`Σ³ᴰ` and the mass-blind `Σ^φ` disagreed by a
measured, un-derived, h-sloped factor `r_φ ≈ 0.886` (realistic venue) / `0.9119` (production,
per the code's gate (ii-b) note) — i.e. "the divisor slot mismatch" between the two channels'
implicit population content (`docs/PRIMER_BIAS_CHANNELS_20260822.md:101,130`;
`docs/derivations/PROPOSAL_SIGMA_PHI_DIVISOR_20260822.md:29`). The claim's own exhibit called out
`"r_phi == 1 by construction/identically"` assertions in the pre-flip code as unqualified and
FALSE for the mass-blind leg (`CLAIM_P3_RPHI_20260822.md:70-71`).

**The flip moots this by construction, not by improved agreement.** Post-flip, under
`"auto"→"on"`, the no-BH catalogue leg no longer uses `Σ^φ` (the mass-blind object `r_φ` was ever
computed against) at all — §2.2 shows `global_denom_no_bh` is reassigned to be `global_denom_with_bh`
(`Σ_4D`) itself, and §2.3/§2.4 show the numerator and mixture weight are likewise the with-BH
leg's own objects. There is no longer a second, independently-derived divisor for `r_φ` to be a
ratio *of* — the comparison c-rphi-mismatch measured (`Σ^φ` vs `Σ_4D`/`Σ³ᴰ`) is structurally
absent from the flipped code path. The pre-flip `"off"` branch (§2.2's `else`, still reachable as
an explicit counterfactual, now logged `COUNTERFACTUAL` at `bayesian_statistics.py:4431-4437`)
still routes through `Σ^φ` and still carries the historical `r_φ ≈ 0.886`/`0.9119` mismatch — it
is simply no longer the production default. This matches the graph spec's own framing at §1.10:
"the flip moots r_phi by construction: under the auto/on mass-aware branch the numerator and
divisor are matched-content, Z=1 identically (rows #269/#286; transform 1.039, row #282)."

Ties to the A18 closure record: A18 (`row #282`/`#286`) is the production-default flip's own
ratification — arm (c) Z-CONFIRMED (1D map_h 0.665/mean_h 0.66699, inside the registered
`[0.64,0.72]` band and the measured band). This node (`rd-rphi-note`) is the first standing
g-znorm evaluation *of that flipped leg's own internal identity* (numerator=divisor by
construction), distinct from and downstream of A18's h-band verdict. Both hold; retiring
c-rphi-mismatch here does not reopen or restate A18.

---

## 5. Disposition

- g-znorm: **GREEN** (deviation 0.0, code-path demonstration + local numeric check both concur).
- `d-rphi-retire`: UNBLOCKED — returns to the author WITH this note, per the graph spec ("never
  pre-granted"). This record does not itself retire c-rphi-mismatch; that RULE is the author's.
- No STOP triggered. No code edited. No commit made. No cluster job submitted.

**Files referenced (absolute paths):**
- `/home/jasper/Repositories/darksiren-emri/darksiren_emri/bayesian_inference/bayesian_statistics.py`
  (lines 3695, 4036, 4410-4470, 6117-6135, 6715-6728, 7059-7118, 7653, 7715, 8470, 8492)
- `/home/jasper/Repositories/darksiren-emri/darksiren_emri_test/bayesian_inference/test_catalogue_leg_1d_mass_aware.py`
  (`test_r2_z_equals_one_identity_under_on_and_not_under_off`, lines 362-452)
- `/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729/CLAIM_P3_RPHI_20260822.md`
- `/home/jasper/Repositories/darksiren-emri/docs/PRIMER_BIAS_CHANNELS_20260822.md` (lines 101, 130)
- `/home/jasper/Repositories/darksiren-emri/docs/derivations/PROPOSAL_SIGMA_PHI_DIVISOR_20260822.md`
- `/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`
  (rows #269, #282, #286)
- `/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729/graph1_20260901/RESEARCH_GRAPH_1_PROPOSAL_20260901.md`
  (§1.10, §2 g-znorm row)
