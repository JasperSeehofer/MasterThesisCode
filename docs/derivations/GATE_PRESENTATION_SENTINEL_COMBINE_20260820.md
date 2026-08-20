# Gate presentation — the mirror harness's log-space sentinel and its moment weights

**Target:** `darksiren_emri/validation/correspondence_1d.py`
**Status:** 5-item `/physics-change` gate, **AWAITING AUTHOR RULING**
**Evidence:** ledger row #145 · `results/prod2d_closure_20260818/PREREGISTRATION_1D_CORRESPONDENCE.md`
AMENDMENT A-7 + verdict · `results/prod2d_closure_20260818/rescore_sentinel.py` (+ its output JSON)

Two changes are proposed. A third, suggested by an adversarial verifier, is **withdrawn by
measurement** and is documented in §3 so it is not re-proposed later.

---

## Change 1 — the log-space `-1.0e300` sentinel

### 1. Old formula

`correspondence_1d.py:1963-1965` (in `compute_seed_statistics`) and the identical
`:2477-2479` (in `compute_full_log_posterior_vector`):

```python
with np.errstate(divide="ignore"):
    log_l = np.where(vals > 0.0, np.log(vals, where=vals > 0.0), -np.inf)
sum_log_l = np.nansum(np.where(np.isfinite(log_l), log_l, -1.0e300), axis=0)
```

i.e. for the per-event likelihood matrix `L[i,h]`,

  `Σ_i ln L[i,h]`  with  `ln L[i,h] → −10³⁰⁰`  wherever  `L[i,h] = 0`.

### 2. New formula

Production's already-registered `CombinationStrategy.PHYSICS_FLOOR`
(`bayesian_inference/posterior_combination.py:219-273`), applied to `L` before the log:

  `L'[i,h] = L[i,h]` if `L[i,h] > 0`, else `min_{h': L[i,h']>0} L[i,h']`
  events with `L[i,h'] = 0 ∀h'` are **excluded**
  `sum_log_l[h] = Σ_i ln L'[i,h]`

### 3. Reference / derivation

Not a physics reference — this is a numerical-hygiene defect, and the authority is IEEE 754
plus the repo's own registered strategy.

The likelihood of an ensemble is `L(h) = Π_i L_i(h)`, so `ln L(h) = Σ_i ln L_i(h)`. If some
`L_i(h) = 0`, the mathematically correct value is `ln L(h) = −∞`. The code substitutes a *finite*
proxy, `−10³⁰⁰`. In float64 (53-bit mantissa) any value `x` with `|x| < 2⁻⁵³ · 10³⁰⁰ ≈ 10²⁸⁴` is
absorbed: `−1e300 + x == −1e300` exactly. Real per-event log-likelihoods here are O(10²), so a
single sentinel annihilates the entire finite signal at that node.

**The consequence is narrower than that sounds, and the narrowing is the important part.** After
the harness's own shift `lp = sum_log_l − max(sum_log_l)` (`:1968`), a node carrying more
sentinels than the maximum node differs by a multiple of `−10³⁰⁰`, whose exponential is exactly
0 — which is precisely what `−∞` would have given. **Verified:** across all 98 banked seeds with
≥1 surviving node, the sentinel and true `−∞` agree to `max|Δ mean_h| = 0.000e+00`.

The defect bites only when **every** node of `H_GRID_41` carries ≥1 sentinel, i.e. when the seed
contains an event with `L_i(h) = 0 ∀h`. Then true `−∞` gives an all-`−∞` vector, so
`lp = sum_log_l − max(sum_log_l)` is `NaN` and `mean_h`/`sigma_h` come out **NaN** — visibly
broken. The sentinel instead yields a finite, normalizable vector that is **silently banked**.
Verified directly: under true `−∞`, `b0_900101` and `bf1_900101` have **0/46 finite nodes**;
`bout_900101` has 33/46 and is unaffected.

> **Correction (2026-08-20, post-approval).** This paragraph originally claimed the all-`−∞` case
> "fires the harness's own `if not np.isfinite(sum_log_l).any()` guard". **No such guard existed** —
> the module's only `isfinite` check is at `:2296`, inside `_normalized_model_cdf`, and is
> unrelated. Correct `−∞` yielded NaN statistics, not a refusal. The claim was inherited from a
> synthesis agent without re-derivation. The substance is unchanged — the sentinel turned a visibly
> broken result into a plausible one — and an explicit guard **was added** as part of the approved
> fix. See the second addendum to ledger row #145.

Two failure modes follow, both reproduced bit-exactly:

- **uniform sentinel count ⇒ exactly flat.** `mean_h` becomes the grid's first moment under a flat
  density, `(0.600 + 0.860)/2 = 0.7299999999999999`, which **coincides with `H_TRUE = 0.73`**;
  `map_h = argmax` of a constant array `= H_GRID_41[0] = 0.600 ≤ R_LOW_THRESHOLD` ⇒ `r_low = True`;
  and `c50 = c68 = c90 = True`. A degenerate seed therefore reports *unbiased*, *railed*, and
  *covered* simultaneously, with no data content whatsoever.
- **non-uniform sentinel count ⇒ spuriously informative.** Differing multiples of `−10³⁰⁰` survive
  the shift as fake evidence, concentrating the posterior on the least-sentinelled nodes
  (`b0_900121 → 0.8400`, `bsig005_900108 → 0.8087`, `bsig005_900114/900119 → 0.8400`).

Why `PHYSICS_FLOOR` rather than `EXCLUDE`, `PER_EVENT_FLOOR`, or a bare `clip(L, 1e-300)`: it is
already registered in production; it does not delete informative events (`EXCLUDE` drops any event
with a single zero — 42/69 events in one G-1 diagnostic); it introduces no undeclared constant
(`PER_EVENT_FLOOR` divides by an arbitrary 100); and unlike a global clip its floor is
per-event-scaled, so it cannot invent support many orders of magnitude above that event's own
likelihood scale.

**Honest caveat.** `PHYSICS_FLOOR` is itself a modelling choice, not neutral infrastructure:
flooring a masked node at that event's minimum non-zero likelihood *invents* support where the
estimator produced none, which flattens a one-sided mask and therefore pushes the posterior
toward the unmasked side. Measured, this is immaterial for every arm except `bsig005`
(strategy spread 0.0162 ⇒ FRAGILE; its corrected number is reported but not adjudicated). All
other arms have spread ≤ 0.0043.

### 4. Dimensional analysis

`L_i(h)` is a probability density in the data, so `ln L_i` is dimensionless (nats) and
`Σ_i ln L_i` is dimensionless. `−10³⁰⁰` is dimensionally a log-likelihood and so passes any units
check — which is exactly why this defect is invisible to dimensional analysis and had to be caught
numerically. The proposed replacement operates on `L` (density units) before the log and preserves
the same dimensionless output. **No units change.**

The moment weights `w` carry units of `h`; `mean_h = Σ p w h / Σ p w` has units of `h`;
`sigma_h` likewise. Unchanged by this proposal.

### 5. Limiting cases

1. **No zeros anywhere ⇒ bit-identical to the current code.** `_physics_floor` hits `continue` on
   every row and returns the input array unmodified. **Verified on 79/79 sentinel-free banked
   seeds: `max|Δ| ≤ 1e-9`** (GATE R-0b).
2. **One event zero at every node.** Correct behaviour: that event is uninformative about `h` and
   must drop out of the product, leaving the other `n−1` events' posterior. `PHYSICS_FLOOR`
   excludes it and returns exactly that. The current code returns the grid midpoint.
3. **Sentinel ≡ −∞ when ≥1 node survives.** Verified `max|Δ mean_h| = 0.000e+00` over 98 seeds —
   the fix is a strict no-op outside its blast radius, which is the strongest possible statement
   that it cannot disturb the campaign's surviving results.

---

## Change 2 — `np.gradient` is not the trapezoid rule

### 1. Old formula

`correspondence_1d.py:1967` (docstring at `:1942-1944` calls these "non-uniform trapezoid
weights"):

```python
weights = np.gradient(grid)
```

### 2. New formula

Composite trapezoid weights on a non-uniform grid:

```python
weights = np.empty_like(grid)
weights[1:-1] = (grid[2:] - grid[:-2]) / 2.0
weights[0]    = (grid[1] - grid[0]) / 2.0
weights[-1]   = (grid[-1] - grid[-2]) / 2.0
```

### 3. Reference / derivation

Composite trapezoid rule: `∫f dh ≈ Σ_i w_i f_i` with `w_i = (h_{i+1} − h_{i−1})/2` in the interior
and `w = Δ/2` at each endpoint. `np.gradient` returns the *central-difference derivative stencil*,
which matches the trapezoid rule in the interior but returns the **full** one-sided spacing at the
boundaries — so both endpoint weights are **doubled**.

**Verified:** `w[0] = w[−1] = 0.010` under `np.gradient` versus `0.005` for trapezoid;
`Σ w = 0.27` versus the true interval length `0.860 − 0.600 = 0.26`. The quadrature over-counts
the interval by exactly one grid step.

### 4. Dimensional analysis

`w` has units of `h`. `Σ p w` is dimensionless (normalisation), `Σ p w h / Σ p w` has units of
`h`. Both old and new weights are dimensionally correct; only their values differ. **No units
change.**

### 5. Limiting cases

1. **Flat posterior.** Trapezoid is exact for a linear integrand, so `mean_h` must equal the
   interval midpoint `(0.600 + 0.860)/2 = 0.730`. **Both** weightings return exactly `0.730`,
   because `H_GRID_41`'s two endpoints sit in equal-spacing (0.01) regions and the doubling is
   symmetric. So the flat-mode `mean_h = 0.7300` artifact of Change 1 is **not** caused by this
   defect. `sigma_h` does differ: `0.07784719124788758` (gradient) versus `0.07512169613879356`
   (trapezoid) — the banked degenerate `sigma_h` is gradient-specific.
2. **Real banked posteriors** (98 seeds with a finite vector). Mean absolute shift in `mean_h`:
   `bsel 3.7e-3`, `bself 3.7e-3`, `bden 3.4e-3`, `bout 6.3e-4`, `b0 2.3e-4`, `bsig005 8.2e-5`,
   `eden2 2.5e-4`, `eden05 3.9e-6`; maximum anywhere `4.2e-3`.

**Materiality, stated plainly.** This shifts the three arms carrying row #140 by ≈3.7e-3 — below
their registered 0.005 band, so no verdict flips, but it is above `bsel`'s own SE of 1.7e-3. Since
it shifts all three by nearly the same amount, the *bisection signal* (the differences
−0.1120 → −0.1163 → −0.1193) is essentially unaffected. Correcting it changes the absolute
numbers of record and therefore requires the author's ruling, not a silent fix.

---

## 3. Withdrawn — `_hpd_contains` is CORRECT

An adversarial verifier proposed a third change: that `_hpd_contains` (`:1914-1929`) returns
`True` on reaching the target *before* testing `cum >= level` (`:1925` versus `:1927`), and that
this is why a flat posterior scores "covered at 50%" at cumulative mass 0.50926.

**Tested against an analytic Gaussian and REJECTED.** The standard HPD set at level α is the
smallest descending-density set whose mass is ≥ α, so the node that *crosses* α **is** a member —
which is exactly what the code implements. Checked at levels 0.50/0.68/0.90 with the target placed
just inside and just outside `z(α)·σ`: the code agrees with the analytic answer in **6/6** cases.

The flat-posterior `coverage = True` is a property of the **degenerate posterior**, not of the HPD
routine: when every node has identical density, any 50% subset is a valid HPD set and "is truth
covered" is ill-posed. **Fixing Change 1 removes those seeds; changing the HPD rule would not, and
would break correct behaviour.** No change proposed.

---

## 4. Blast radius, and what does NOT move

**25 of 123 banked seeds (20.3%)** are affected by Change 1 — catalogue-mode 25/70,
population-mode **0/53**.

Unaffected, verified `Δ ≡ 0` to ≤1.0e-15: **B-SEL (−0.1120, n=12), B-SELF (−0.1163, n=11),
B-DEN (−0.1193, n=15)** — the bisection chain, and hence ledger row #140. Also unaffected:
**B-OUT (−0.1293, Δ ≤ 1.1e-16)**, so row #139's dark-rail correspondence stands.

Production is untouched: the additive log-space sentinel exists nowhere else, and no production
module imports these functions. The *multiplicative* `1e-300` clip used across the repo
(`posterior_combination.py:758`, ~18 sites in `validation/pp_coverage.py`) is a different and
benign pattern — bounded at `ln ≈ −690.8`, no absorption.

---

## 5. The root cause is upstream, and is NOT fixed by either change

Every all-zero event has `L_cat_no_bh = 0`, `B_num = 0` **and `g_frac = NaN`** — an empty
candidate set / undefined catalogue–completion mixing fraction — in **100%** of cases, against a
3–6% `g_frac`-NaN baseline in other rows. It occurs in 25/70 catalogue-mode seeds and 0/60
population-mode seeds.

**That is a generator/data defect, not a numerical one.** Changes 1 and 2 stop the harness from
silently banking a fabricated posterior when it happens; they do not stop it happening. Why the
mirror places a host in the catalogue that the ball-tree lookup then fails to recover is a
separate open thread and is **not** addressed here.

---

## 6. Decision table

| # | Item | Tag | Ask |
|---|---|---|---|
| 1 | **Change 1** — replace the `-1.0e300` log-space sentinel at `:1965` and `:2479` with production's registered `PHYSICS_FLOOR` semantics; add a regression test asserting the old degenerate values first, so the diff is visible | **[DO]** | Approve implementation |
| 2 | **Change 2** — replace `np.gradient(grid)` at `:1967` with true trapezoid weights, correcting the doubled endpoint weights and the wrong docstring at `:1943` | **[DO]** | Approve implementation |
| 3 | Re-scored numbers of record (A-7 verdict table) **supersede** the published ones for b0/bsig005/eden05/eden2/bf1; the published values are retained in the prereg as the as-run record | **[RULE]** | Ratify |
| 4 | **Every rail recorded in every catalogue-mode arm is an artefact** (`R_low → 0.00` in all five); any banked claim quoting `C50/C68/C90` or `R_low` for those arms is withdrawn | **[RULE]** | Ratify |
| 5 | **B-F1's "0.7300, truth to four decimals" is withdrawn**; corrected it is `+0.0359 ± 0.0036` with coverage `0/0/0` — the positive control **fails**, PROVISIONAL at n = 2 | **[RULE]** | Ratify |
| 6 | **G-1's PASS is recorded UNSUPPORTED** (no G-1 output is banked under `results/`), not proven vacuous | **[RULE]** | Ratify |
| 7 | The `g_frac = NaN` empty-candidate-set generator defect (§5) is opened as the next investigation thread, ahead of the row #144 §6 positive control, since that control would inherit it | **[DO]** | Approve or re-order |
| 8 | **Amendment A15** (power-calibrated gates, demonstrably-sensitive controls) — still pending from row #144 §7. A-7's own draft twice reproduced the failure A15 forbids | **[RULE]** | Ratify or reject |

**Nothing in §6 has been implemented.** Per the protocol this presentation stops here and waits.
