# [WGEO] stage 0, read (1) — analytic form of the linear-window / log-normal-error mismatch

Status: MEASUREMENT/DERIVATION ONLY. No production files touched.
Date: 2026-08-27. Repo: darksiren-emri @ commit `6e1bc488` (HEAD at task start; `git status` was
otherwise dirty with unrelated untracked campaign work, not read or touched here).

---

## 1. The window as implemented

Source: `darksiren_emri/galaxy_catalogue/handler.py:663-673`, with the multiplier resolution at
`:654-661`, inside `GalaxyCatalogueHandler.get_possible_hosts_from_ball_tree`.

Symbols (all file:line-cited):

| Symbol | Meaning | Source |
|---|---|---|
| `M_z` | event's GW-inferred mass parameter (`self.detection.M`) | `bayesian_inference/bayesian_statistics.py:4689` |
| `M_z_sigma` | its 1σ uncertainty (`self.detection.M_uncertainty`) | `bayesian_inference/bayesian_statistics.py:4690` |
| `sigma_multiplier` (≡ `k`) | window half-width multiplier; production call passes **`k = 1.5`** | `bayesian_inference/bayesian_statistics.py:4691` |
| `z_min`, `z_max` | event redshift outer bounds from `get_redshift_outer_bounds` | `physical_relations.py:546-567`; consumed at `bayesian_statistics.py:4669-4679` |
| `BH_MASS` | candidate host's catalogue BH mass | `galaxy_catalogue/handler.py:665,669` (`InternalCatalogColumns.BH_MASS`) |
| `BH_MASS_ERROR` | candidate host's catalogue BH mass error | `galaxy_catalogue/handler.py:666,670` |
| `_bh_mass_error_multiplier` | multiplier applied to `BH_MASS_ERROR`; in the adopted **"symmetric"** mode it equals `sigma_multiplier` (= `k`) | `galaxy_catalogue/handler.py:654-661`, specifically `:657` |

The mask (`galaxy_catalogue/handler.py:663-673`), written as two inequalities:

```
(A)  (M_z - k·M_z_sigma) / (1 + z_max)  <=  BH_MASS + k·BH_MASS_ERROR
(B)  BH_MASS - k·BH_MASS_ERROR          <=  (M_z + k·M_z_sigma) / (1 + z_min)
```

Rearranging each to isolate `BH_MASS`:

```
(A')  BH_MASS  >=  (M_z - k·M_z_sigma)/(1+z_max)  -  k·BH_MASS_ERROR
(B')  BH_MASS  <=  (M_z + k·M_z_sigma)/(1+z_min)  +  k·BH_MASS_ERROR
```

This is an **interval-overlap test** between the event's GW-side interval
`G = [(M_z-k·M_z_sigma)/(1+z_max), (M_z+k·M_z_sigma)/(1+z_min)]` and the **candidate-centered
window**

```
W(BH_MASS) = [ BH_MASS - k·BH_MASS_ERROR ,  BH_MASS + k·BH_MASS_ERROR ]
```

`W` is the object this read analyses: it is constructed as a **linear-symmetric** interval around
the catalogue mass `BH_MASS`, with half-width `k·BH_MASS_ERROR` on *both* sides. The (z_min,z_max)
stretch on the GW side (`get_redshift_outer_bounds`, `physical_relations.py:546-567`, including its
known dead `sigma_multiplier` argument) is a separate, event-side mechanism and is out of scope for
this read — it is orthogonal to the candidate-side asymmetry derived below.

---

## 2. Catalogue-side error model: BH_MASS_ERROR is a ln-space σ times M (a linearization), and this is exact by construction, not approximate

Source: `galaxy_catalogue/handler.py:1368-1382` (`_empiric_stellar_mass_to_BH_mass_relation`), constants
at `:37-44`.

```python
BH_mass = np.exp(alpha + beta * np.log(stellar_mass / 10))                      # :1371
BH_mass_error = BH_mass * np.sqrt(
    sigma_int**2
    + d_alpha**2
    + (np.log(stellar_mass / 10) * d_beta) ** 2
    + (beta / stellar_mass * stellar_mass_error) ** 2
)                                                                                 # :1376-1381
```

`alpha, beta, d_alpha, d_beta, sigma_int` are all stored in **natural-log units** (`:37-44`, each is
a dex value × `ln(10)` except `beta`/`d_beta` which are dimensionless slopes and `sigma_int` which is
`0.24·ln(10)`). The four terms summed under the square root are exactly the propagated variance of
`ln(BH_mass)` under standard first-order error propagation of
`ln(BH_mass) = alpha + beta·ln(M*/10) + (calibration + intrinsic-scatter terms)`. That is, the
square-root term **is** `σ_ln(BH_mass)` by construction:

```
σ_ln := sqrt( σ_int² + dα² + (ln(M*/10)·dβ)² + (β/M* · σ_M*)² )
BH_mass_error = BH_mass · σ_ln
```

So:

```
CV := BH_MASS_ERROR / BH_MASS  ≡  σ_ln          (identically, by construction)
```

**This is the linearization `σ_lin = M·σ_ln`, not the exact log-normal moment relation**
`CV² = exp(σ_ln²) − 1` (⟺ `σ_ln_exact = sqrt(ln(1+CV²))`). The code performs delta-method
(first-order) propagation of a log-space Gaussian into a linear-space "error," then hands that linear
number to a **linear**-symmetric window (`W` above). Two approximations compound: (i) a linear
"sigma" manufactured from a log-normal model via linearization, then (ii) a linear-symmetric cut
built from that number — step (ii) is exact for a *linear* Gaussian and increasingly wrong for large
`σ_ln` regardless of step (i).

**Numeric validation** against a HARD-WON-FACTS-pinned catalogue row, computed directly, no lookup:

`catalog_index 6791151`: `BH_MASS = 223872.11385683485`, `BH_MASS_ERROR = 291758.99489010876`
(pinned values, reproduced verbatim from the task prompt — these were not independently re-derived
from the raw CSV in this read, only the ratio below was computed from them).

```
CV = 291758.99489010876 / 223872.11385683485 = 1.3032395587986776
```

This equals, to the quoted precision, the campaign's already-banked `σ_ln = 1.3032` for "the exhibit
candidate" — confirming `CV ≡ σ_ln` by construction for this host, and confirming this catalogue row
is the exhibit candidate referenced in the prompt.

Contrast with the exact log-normal relation at this `σ_ln`: `CV_exact = sqrt(exp(σ_ln²)−1) =
sqrt(exp(1.6984)−1) = sqrt(4.465) = 2.113` — nearly **1.6×** the code's linearized `CV = 1.303`. The
two constructions diverge sharply once `σ_ln ≳ O(1)`, which is exactly the regime this catalogue
exhibits (`σ_ln = 1.30` is not a small-scatter regime — `ln(10)·0.24 = 0.553` from intrinsic scatter
alone already sets a floor well above the small-`σ` regime where linearization is safe).

---

## 3. The window cut expressed in ln M — closed-form induced asymmetry

Write `M ≡ BH_MASS`, `CV ≡ σ_ln` (§2), `k = 1.5` (§1). The linear window is
`W = [M(1 − k·CV), M(1 + k·CV)]` (factor `M` out of `W` in §1).

**Upper ln-half-width** (always defined, `M(1+k·CV) > 0` for `CV>0`):

```
w_up(CV) = ln(1 + k·CV)
```

**Lower ln-half-width** (defined only while `M(1−k·CV) > 0`, i.e. `k·CV < 1` — see §4):

```
w_lo(CV) = ln(M) − ln(M(1−k·CV)) = −ln(1 − k·CV)
```

**Asymmetry measure** (task's suggested form, difference over sum):

```
A(CV) = (w_up − w_lo) / (w_up + w_lo)
      = [ln(1+k·CV) + ln(1−k·CV)] / [ln(1+k·CV) − ln(1−k·CV)]
      = ln(1 − k²CV²) / ln[(1+k·CV)/(1−k·CV)]                      valid for 0 ≤ CV < 1/k
```

Equivalently, in `x := k·CV`: `A(x) = [ln(1−x) + ln(1+x)] / [ln(1+x) − ln(1−x)]`.

**Sign and interpretation:** for `CV>0`, `ln(1+kCV) < kCV < −ln(1−kCV)` (standard bounds
`ln(1+t)<t<−ln(1−t)` for `0<t<1`), so `w_up < w_lo` and **`A(CV) < 0` throughout the domain**: the
linear-symmetric window, re-expressed in log-mass, is *always* narrower on the high-mass side and
wider (eventually unboundedly so) on the low-mass side. This is the opposite asymmetry direction from
what a naive read of "window excluded the high-σ_ln candidate" might suggest in isolation — the
candidate-window itself is generous on the low side and stingy on the high side; the exclusion in the
exhibit case comes from the interaction with the event-side GW interval `G` (§1), not from `W` alone.
That interaction is explicitly out of scope for this analytic read (see §1 and the HARD-WON-FACTS
framing: the exhibit's 1.5σ-ln-space-inside-cut candidate was excluded through the compound test, not
through a bare read of `W`).

**Numeric check at the exhibit candidate** (`CV = 1.303240`, `k=1.5`, `kCV = 1.95486`): already past
the one-sided threshold (§4) — `w_lo` is undefined (window has no lower edge at all in linear space).
The finite-`w_up` value alone: `w_up = ln(2.954859) = 1.08308`; nominally this is what "2.955×"
(§ numeric validation, matches `math.exp` check below) corresponds to on the log scale, versus the
log-symmetric prescription's `k·σ_ln = 1.5 × 1.303240 = 1.954859` on *both* sides (upper ratio
`exp(1.954859) = 7.0629×`, reproducing the banked "7.06×" figure exactly).

---

## 4. Threshold: when the linear lower edge is non-positive

Lower edge: `M(1 − k·CV) ≤ 0  ⟺  1 − k·CV ≤ 0  ⟺  CV ≥ 1/k`.

With `k = 1.5`:

```
CV_threshold = 1/1.5 = 0.6667 (exactly 2/3)
```

For `CV ≥ 2/3`, the window `W` has **no lower exclusion at all** — condition (B') admits any
`BH_MASS ≥ 0` from below; the cut is one-sided in practice (upper-bound-only), regardless of how
large `M` is. The exhibit candidate's `CV = 1.303` is **~2×** past this threshold, so its window is
solidly in the one-sided regime, not marginal.

---

## 5. Limiting checks (mandatory)

**Check 1 — `CV → 0` must recover a symmetric window with vanishing asymmetry.**

Taylor-expand `A(CV)` for small `x = k·CV`:
`ln(1−x²) ≈ −x²`, `ln[(1+x)/(1−x)] ≈ 2x`, so `A(CV) ≈ −x²/(2x) = −x/2 = −k·CV/2 → 0` as `CV→0`.

Also directly: `w_up = ln(1+kCV) ≈ kCV − (kCV)²/2 + …`, `w_lo = −ln(1−kCV) ≈ kCV + (kCV)²/2 + …` — both
`→ kCV` to leading order (both edges recover the log-symmetric width `k·σ_ln` at first order, as they
must — the linear and log-normal window prescriptions agree to `O(CV)` and differ only at `O(CV²)`).

**PASSES.** The symmetric limit is recovered, and the leading-order asymmetry is `O(CV)`, not `O(1)`
— consistent with "a linear approximation to a log-normal error is fine for small fractional error and
breaks down for large fractional error," which is the physically expected behavior.

**Check 2 — large CV must give a strongly one-sided cut.**

As `CV → (1/k)⁻` (`x→1⁻`): `w_up → ln(2)` (finite), `w_lo → +∞`. Setting `ε := 1−x → 0`:
`w_lo = −ln(ε) → ∞` while `w_up → ln2`. Then
`A = (w_up−w_lo)/(w_up+w_lo) → (finite − ∞)/(finite + ∞) → −1`.

**PASSES.** `A(CV) → −1` at the threshold, the maximal-asymmetry value for this normalization, and
beyond the threshold the window is literally one-sided (§4) — `w_lo` is not merely large, it is
undefined (no lower cut exists). Both the approach to `−1` and the hard one-sidedness past `CV=2/3`
confirm the "large CV ⇒ strongly one-sided" requirement.

Numeric spot check of monotone approach to −1 (computed at `k=1.5`, i.e. `x=1.5·CV`), from the closed
form in §3:

| `x = k·CV` | `CV` | `A(CV)` |
|---|---|---|
| 0.1 | 0.0667 | −0.0501 |
| 0.3 | 0.2000 | −0.1524 |
| 0.5 | 0.3333 | −0.2619 |
| 0.7 | 0.4667 | −0.3882 |
| 0.9 | 0.6000 | −0.5640 |
| 0.95 | 0.6333 | −0.6354 |
| →1 | →0.6667 | →−1 |

Both checks pass; no algebra fix was needed.

---

## 6. Monotonicity of A(CV)

From the table in §5 and the closed form `A(x) = [ln(1−x)+ln(1+x)]/[ln(1+x)−ln(1−x)]`
(`x=k·CV ∈ [0,1)`): `A` is **strictly monotonically decreasing** in `CV` over the full two-sided
domain `[0, 2/3)` — it runs from `0` at `CV=0` to `−1` as `CV→2/3⁻`, with no local extrema in the
sampled range (values above are strictly decreasing at every step). Beyond `CV=2/3` the window is
one-sided and `A` is not defined in this closed form (the lower edge no longer excludes anything, so
"asymmetry" ceases to be the right description — the window has degenerated to a single one-sided
bound).

**Consequence for the census read:** because `A(CV)` is monotonic on `[0, 2/3)` and identically `−1`
(fully one-sided, in the sense of §4) for all `CV ≥ 2/3`, **any systematic trend of `CV` (≡ `σ_ln`,
§2) with redshift or host mass maps monotonically onto a trend in the window's induced asymmetry — up
to the point where `CV` crosses `2/3`, after which further increases in `CV` produce no additional
asymmetry by this measure** (the window is already maximally one-sided; the *linear* location of the
now-irrelevant lower edge continues to move but no longer does selection work). This is the shape the
census read (stage-0, read 2) should test for: whether `CV` clusters above vs. below `2/3` as a
function of `z` or `M*`, not merely whether `CV` grows with `z` in the unsaturated region.

---

## Caveats (explicit, per measure-first discipline)

- This read analyses the **candidate-centered window `W` in isolation** (§1). The actual
  `mass_filter_mask` is an **overlap test** between `W` and the event-side GW interval `G`
  (§1, conditions A/B). The asymmetry of `W` alone does not by itself establish the direction or
  magnitude of any *net* selection effect on which hosts pass — that depends on where `G` sits
  relative to `M`, which varies event-by-event and was not measured here.
- The `CV ≡ σ_ln` identity (§2) is a *definitional* consequence of the code's construction, verified
  numerically at one pinned catalogue row; it was not re-derived from raw CSV bytes in this read (per
  HARD-WON-FACTS, the CSV pin and index semantics were established earlier and are not re-litigated
  here).
- This is a **closed-form geometric read**, not a statistical claim: it does not by itself show that
  the asymmetry has z-structure, host-mass structure, or any relationship to the banked dark-class
  high-z base tilt (score −0.635, 37σ). That correlation is the explicit target of the next stage
  (census read) and per the task's own framing, a clean null — no z/M structure in `CV` — is a fully
  successful outcome of this thread, not a failure to find one.
- The event-side `(1+z_max)/(1+z_min)` stretch on `G` (§1, `physical_relations.py:546-567`, including
  its known dead `sigma_multiplier` argument) is a second, orthogonal asymmetry-inducing mechanism not
  analyzed here.
