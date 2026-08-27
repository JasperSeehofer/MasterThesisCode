# [OPUS-ORCH 2026-08-27] `/physics-change` presentation package — class-G S̄_φ de-double-weight

> **NO CODE HAS BEEN WRITTEN.** No file under `darksiren_emri/` has been created, edited, or
> staged by this pass. This document is the **presentation package only**, produced under the
> repo's hard gate (`.claude/rules/physics-validation.md` §"Protocol — before writing any code";
> `.claude/skills/physics-change/SKILL.md` §"STOP and wait for explicit user approval before
> implementing"). **It awaits author approval before any implementation.** No ledger row has been
> appended to `docs/gates/PHYSICS-GATE-LEDGER.md` — the row is written when the author answers
> (proposed row in §10).

**Target:** `darksiren_emri/validation/correspondence_1d.py` — the `"catalogue_selected_2d"`
(b0i2d) mirror-universe generator only.
**Thread:** [P3-2D], runbook 34 §"first action on thread resume".
**Grant status:** asserted as granted at `BIAS_HISTORY_LEDGER.md:2990/2992/2994` (rows #209–#211)
and in runbooks 34/35; **no verbatim author quote exists for this item** — see
`sbarphi_defect_location_20260827.md` §1. Flagged, not re-litigated.

---

## 0. HEADLINE — a limiting check FAILS on the fix as granted

The gate exists to catch exactly this, so it goes first rather than in §5.

**The granted fix is under-specified, and its first registered disjunct, implemented literally,
does not remove the defect.**

`C2_star_review.md:168-170` registers two disjunctive repairs; `sbarphi_defect_location_20260827.md`
§4 transcribes them as Options A and B. Option A is stated as *"z-draw from k̄_g·w_pop without the
S̄_φ factor when the Bernoulli(S_4D) layer is active"* — i.e. drop `s_i` at
`correspondence_1d.py:1497` for the 2D call site and change nothing else.

That is wrong, because **the host-draw weight and the z-conditional are not independent**. The 2D
branch draws hosts with `p = host_w ∝ w_g · S̃_φ,g` (`correspondence_1d.py:1380`, consumed at
`:1682`). Today that `S̃_φ,g` is *exactly* the normalizing constant of the S̄_φ-weighted
z-conditional, so it cancels (this cancellation is the review's own finding,
`C2_star_review.md:44-45`). **Remove `S̄_φ` from the z-density and the cancellation is destroyed:
`S̃_φ,g` survives as an uncancelled host-level survival weight.** The defect is relocated from the
event's drawn `z_ev` to the host's listed `z_g`, not removed (§5, check L6).

Because `S̃_φ,g = ∫ k̄_g(z) S̄_φ(z) dz ≈ S̄_φ(z_g)` to O(σ_z,eff²·S̄_φ″) and σ_z,eff ≈ 0.035 against
a pool z-range of order 0–0.5, the surviving host-level tilt is **highly correlated with the
event-level tilt it replaces**. Option-A-literal is therefore expected to remove only the small
within-kernel part of the measured 13.5–16 %, not the bulk of it. *(The exact residual fraction is
UNMEASURED. It is cheap to measure — §7, test R4.)*

**Two further failures of the record's own disjuncts** (§5, checks L5 and L7):

- `C2_star_review.md:169-170`'s second disjunct *"keep the z-draw and drop the Bernoulli in favor
  of an S̃-reweighting"*, read literally as dropping the mass gate, **destroys the mass selection
  that the entire [P3-2D] venue extension exists to introduce** (prereg §2(ii)/§2.4, GATE M2-LINK).
  It fails the mass-selection limiting check.
- `sbarphi_defect_location_20260827.md` §4's Option B (per-event importance weight
  `S_4D/S̄_φ`) **is algebraically exact** — it is the only one of the record's forms that is
  correct as literally written — but it converts the venue from an unweighted rejection sampler
  into a weighted sampler, which no downstream consumer of the harness supports (§9).

**Recommended corrected form: Option A′** — Option A *plus* the host-weight change it requires.
Stated in §2, derived in §3, and the only form that passes every check in §5.

---

## 1. OLD FORMULA — as implemented

### 1.1 Symbols (all cited)

| symbol | meaning | code site |
|---|---|---|
| `g` | host index in the pinned reduced-catalogue pool | `correspondence_1d.py:1682` |
| `w_g` | per-MBH rate weight `R_eff_per_mbh(M_g)/(1+z_g)` | `correspondence_1d.py:1376` |
| `k_g(z)` | unnormalized host-z kernel `N(z; z_g, σ_eff,g)·w_pop(z)·f_k(z)` | `correspondence_1d.py:1492-1497`; smeared twin `:1330-1331` |
| `Z_g` | window normalizer `∫_{W_g} k_g(z) dz` | `correspondence_1d.py:1335` |
| `k̄_g(z)` | `k_g(z)/Z_g`, normalized on the host window `W_g` | `_host_kernel_window`, `correspondence_1d.py:1486` |
| `p_gal(M\|g)` | Eddington-shifted Gaussian `N(M; M_eff,g, σ_M,g)`, floored | `correspondence_1d.py:1704-1708` |
| `S_4D(d_L, M_z)` | production 2D survival, `d_L` in Gpc, `M_z` in M_⊙ | `correspondence_1d.py:1712-1717` |
| `S̄_φ(z;h)` | φ-marginal survival `∫ φ(log₁₀M)·S_4D(d_L(z;h), M(1+z)) dlog₁₀M` | `bayesian_statistics.py:1947` (`precompute_phi_marginal_survival`); definition `docs/derivations/fixb_pathA_phi_marginal_selection.md:60` |
| `S̃_φ,g` | kernel-smeared survival `∫ k̄_g(z) S̄_φ(z;h) dz` | `correspondence_1d.py:1334-1337` |
| `h` | fixed at `H_TRUE = 0.73` throughout the venue | `correspondence_1d.py:359`, `:1614`, `:2119` |

### 1.2 The three steps as coded

**(i) Host draw** — `correspondence_1d.py:1380-1384` builds, and `:1682` consumes:

```
P(g)  =  w_g · S̃_φ,g  /  Σ_{g'} w_{g'} · S̃_φ,g'
```
```python
# correspondence_1d.py:1380-1384
unnormalized = w_g * s_tilde_phi
...
normalized = unnormalized / total
# correspondence_1d.py:1682
host_idx_batch = rng.choice(pool.n, size=batch, replace=True, p=host_w)
```

**(ii) z draw** — `correspondence_1d.py:1687-1696` calls `_draw_kernel_survival_redshifts`, whose
density is built at `:1492-1498`:

```
p(z | g)  =  k_g(z) · S̄_φ(z;h)  /  ∫_{W_g} k_g · S̄_φ  =  k̄_g(z) · S̄_φ(z;h) / S̃_φ,g
```
```python
# correspondence_1d.py:1496-1498
s_i = np.interp(z_i_grid, z_grid, s_phi)
density_i = kernel_i * w_pop_eff_i * s_i
z_true[i] = _inverse_cdf_draw(rng, 1, z_i_grid, density_i)[0]
```

**(iii) Mass draw + Bernoulli survival gate** — `correspondence_1d.py:1704-1719`:

```
M | g  ~  p_gal(M | g) ,      M_z = M·(1+z)
accept  with probability  S_4D( d_L(z;h), M_z )
```
```python
# correspondence_1d.py:1704-1709
m_eff = _eddington_shifted_host_mass_batch(host_m, host_m_error)
m_true_batch = m_eff + sigma * rng.normal(size=batch)
m_z_true_batch = m_true_batch * (1.0 + z_true_batch)
# correspondence_1d.py:1712-1719
s4d_batch = detection_probability.detection_probability_with_bh_mass_interpolated(
    d_l_true_batch, m_z_true_batch, host_phiS_batch, host_qS_batch, h=h)
u_batch = rng.uniform(size=batch)
accept_mask = u_batch < s4d_batch
```

### 1.3 The realized accepted-event law (OLD)

Composing (i)×(ii)×(iii) and cancelling `S̃_φ,g` between the host weight and the z-conditional's
normalizer:

```
                w_g · S̃_φ,g        k̄_g(z)·S̄_φ(z;h)
q_old(g,z,M)  =  ────────────  ·  ──────────────────  ·  p_gal(M|g) · S_4D(d_L(z;h), M(1+z))  /  N_old
                    Σ_{g'}…             S̃_φ,g
```
```
  ⟹   q_old(g, z, M)  ∝  w_g · k̄_g(z) · p_gal(M|g) · S̄_φ(z;h) · S_4D(d_L(z;h), M(1+z))
                                                     └───── survival, 1st ─────┘ └──── survival, 2nd ────┘
```
with `N_old = Σ̃^{φ4D} ≡ Σ_g w_g ∫∫ k̄_g(z) p_gal(M|g) S̄_φ(z;h) S_4D dM dz`.

This reproduces the reviewer's independently derived statement verbatim —
*"the accepted-event law is exactly (model class-G law) × S̄_φ(z_ev), renormalized"*
(`C2_star_review.md:44-45`), with the renormalizer `Σ̃^{φ4D}/Σ̃^4D = ⟨S̄_φ⟩_{model,1}`
(`C2_star_review.md:70-71`).

**One correction to the campaign's own characterization.** The record and the task brief describe
this as a factor `S²` (rejection-sampling from a density already ∝ `p·S`). That is the correct
*principle* but not the correct *algebra here*: the two factors are **different objects** —
`S̄_φ(z)` is the survival **marginalized over the population mass function φ**, while the Bernoulli
uses `S_4D` **pointwise at the host-conditional drawn mass**. They coincide only if
`p_gal(·|g) ≡ φ` for every host, which is false by a measured +0.28 dex pool-vs-φ offset
(KS D = 0.215, `docs/derivations/fixb_pathA_phi_marginal_selection.md:161`). The defect is
`S̄_φ(z)·S_4D(z,M)`, **not** `S_4D²`. This matters for the limiting cases (§5, L4).

---

## 2. NEW FORMULA — the corrected form

### 2.1 The target, read off the model side (not invented here)

The model-side normalizer that `C₂*` is built on is `Σ̃^4D`, adjudicated at ≤6.5e-10 as
*"the same kernel × Eddington-shifted-Gaussian × S_4D contraction"* (`C2_star_review.md:36-37`):

```
Σ̃^4D  =  Σ_g  w_g  ∫∫  k̄_g(z) · p_gal(M|g) · S_4D(d_L(z;h), M(1+z))  dM dz
```

A venue is faithful iff its accepted-event law is that integrand, normalized. Hence the target:

```
q_new(g, z, M)  =  w_g · k̄_g(z) · p_gal(M|g) · S_4D(d_L(z;h), M(1+z))  /  Σ̃^4D
```

**Provenance:** this is *transcribed*, not supplied — it is the integrand of the already-arbitered
`Σ̃^4D` object. What **is supplied by this package** (and is absent from the record) is the
host-weight consequence in §2.2, item (i).

### 2.2 Option A′ — RECOMMENDED implementable form

Three changes, all confined to the `"catalogue_selected_2d"` branch:

**(i) [SUPPLIED — not in the record] Host draw uses the plain rate weight:**
```
P(g)  =  w_g / Σ_{g'} w_{g'}          (NOT  w_g · S̃_φ,g / Σ …)
```
`catalogue_selected_host_draw_weights` already returns `w_g` as its **second** value
(`correspondence_1d.py:1385`), so the 2D branch at `:2107` can normalize `_b0i2d_w_g` itself.
**No edit to `catalogue_selected_host_draw_weights` is permitted** — the 1D `b0i` branch at `:2062`
consumes its first return value and must keep it (§5, L8).

**(ii) z draw drops the survival factor:**
```
p(z | g)  =  k̄_g(z)          i.e.  density_i = kernel_i * w_pop_eff_i    (no `* s_i`)
```
Implemented **without editing `_draw_kernel_survival_redshifts`'s body** — by a keyword flag on the
2D call site (`:1687-1696`) or by passing a flat `S̄_φ ≡ 1` table. The function is shared with the
1D `b0i` arm at `:2078` and with the live `[HIER]` blocker-A conclusion (§9).

**(iii) Mass draw and Bernoulli gate unchanged** (`:1704-1719`).

Realized law: `∝ w_g · k̄_g(z) · p_gal(M|g) · S_4D` — **exactly `q_new`.** ∎

### 2.3 Option B′ — exact but rejected on blast radius

`sbarphi_defect_location_20260827.md` §4 Option B, per-event importance weight `ω_i = S_4D/S̄_φ`
with the draw law untouched, is **also exactly correct**:
```
[w_g S̃_φ,g] · [k̄_g S̄_φ / S̃_φ,g] · p_gal · [S_4D/S̄_φ]  =  w_g · k̄_g · p_gal · S_4D   ✓
```
It is rejected because it makes the venue's output a **weighted** sample. Every downstream
consumer — the CSV rows fed to `evaluate()`, the F-0 filter, the equal-weight `LHS₂ = (C₂*/200)·Σ_acc(1−w₂)`
sum, the GATE-ACC `n_drawn_total`/`n_rounds` accounting (`:1730-1731`), the `s4d_at_truth` record
(`:1752`) — assumes unweighted accepted events. Option B′ is the honest fallback **if** the author
wants zero change to the z/host draw laws, at the cost of reworking all of the above.

---

## 3. THE DERIVATION — why the term is double-counted

### 3.1 Generative model: where survival legitimately enters, once

The class-G ("in-catalogue") generative model for one **detected** event is a three-stage
hierarchy. Write the *undetected* (astrophysical) law first:

```
p_astro(g, z, M)  =  P_rate(g) · k̄_g(z) · p_gal(M | g)
                  =  [w_g / Σ w] · k̄_g(z) · p_gal(M | g)
```

- `P_rate(g) ∝ w_g` — a host is chosen in proportion to its EMRI rate. No selection here.
- `k̄_g(z)` — the estimator's own host-z prior for host `g` (`galaxy_redshift_prior_pdf`,
  `bayesian_statistics.py:5954-6023`; mirrored at `correspondence_1d.py:1330-1337`). No selection.
- `p_gal(M|g)` — the host's Eddington-shifted mass posterior. No selection.

Selection is a **single** map from the astrophysical law to the detected law:

```
p_det(g, z, M)  =  p_astro(g, z, M) · S(g, z, M)  /  ∫ p_astro · S
```

where `S` is *the probability that this very triple is detected*. For a triple `(g, z, M)` the
detector sees a source at luminosity distance `d_L(z;h)` with detector-frame mass `M(1+z)`, so
**`S(g, z, M) = S_4D(d_L(z;h), M(1+z))` and nothing else.** There is no second, independent
detection event to condition on.

This is the structure of Mandel, Farr & Gair (2019) Eqs. (5)–(7) (§6): the survival/detection
probability appears **exactly once**, and the object it defines is the same object that normalizes
the construction (`α(λ) = ∫ p_det(θ) p_pop(θ|λ) dθ`). Here `∫ p_astro·S = Σ̃^4D/Σ_g w_g`.

### 3.2 What the code realizes instead

`S̄_φ(z;h)` is **not an independent physical fact** about the triple — it is the *same* `S_4D`,
averaged over the population mass function:

```
S̄_φ(z;h)  ≡  ∫ φ(log₁₀M) · S_4D(d_L(z;h), M(1+z)) dlog₁₀M
```
(`bayesian_statistics.py:1947`; `fixb_pathA_phi_marginal_selection.md:60`). It is the correct
*mass-marginalized* survival for a leg that does **not** track a per-event mass — which is exactly
what the 1D `catalogue_selected` (b0i) venue is (§5, L8) and exactly what the without-BH twin
numerator is (`bayesian_statistics.py:6362-6368`, one `S̄_φ`, correct).

The 2D venue **does** track a per-event mass. It therefore must use `S_4D` at that mass — and it
does, at `:1719`. But it also inherited the 1D leg's `S̄_φ`-weighted z-density verbatim
(`_B0i2DLatents.z_true` docstring, `:1572-1575`: *"UNCHANGED from the 1D 'catalogue_selected'
mode"*), so the mass-marginalized survival is applied **as well as** the pointwise survival:

```
q_old  ∝  p_astro · S̄_φ(z) · S_4D(z,M)      versus      q_new  ∝  p_astro · S_4D(z,M)
```

The extra factor is a survival probability applied to a triple that has already been survival-tested.
Elementary rejection-sampling arithmetic: drawing from a density already ∝ `p·A` and then accepting
with probability `B` realizes `p·A·B`. Setting `A = S̄_φ(z)` and `B = S_4D(z,M)` — both survival
objects derived from the same `S_4D` grid — is the double application.

**The code's own docstring already discloses it** (`correspondence_1d.py:1624-1628`): the Bernoulli
*"reproduces exactly the target joint law **up to the (unchanged) z-marginal's own existing survival
weighting**"*. The clause after "up to" is the bug. The same docstring's claim at `:1620-1622` that
the construction is *"algorithmically equivalent to an explicit S̃_4D,g-weighted host reweighting"*
is likewise false: the equivalent host reweighting for `q_new` would be by
`S̃_4D,g = ∫∫ k̄_g p_gal S_4D dM dz`, whereas the code uses `S̃_φ,g`.

### 3.3 Why the host weight must move too

`S̃_φ,g` is defined (`:1334-1337`) as precisely `∫ k̄_g S̄_φ dz`, i.e. the normalizing constant of the
S̄_φ-weighted z-conditional. In `q_old` it cancels exactly:

```
[w_g S̃_φ,g] × [k̄_g(z) S̄_φ(z) / S̃_φ,g]  =  w_g k̄_g(z) S̄_φ(z)
```

That cancellation is the *only* reason `q_old` has a single `S̄_φ(z_ev)` rather than a
`S̃_φ,g · S̄_φ(z_ev)` product. Remove `S̄_φ` from the z-density while leaving the host weight alone
and the cancellation has nothing to cancel against:

```
[w_g S̃_φ,g] × [k̄_g(z)]  =  w_g S̃_φ,g k̄_g(z)      ≠   w_g k̄_g(z)
```

Hence §2.2 item (i). The record does not contain this step. ∎

### 3.4 Sign and direction

`S̄_φ(z)` decreases with `z` (harder to detect farther away). The old venue therefore
**over-represents low-z / under-represents high-z** accepted events relative to the model law.
Measured, consistent: completion-weighted mean `z` = 0.183 vs all-accepted 0.119; arithmetic
`⟨S̄_φ⟩` weighted 0.7812 ± 0.0044 vs unweighted 0.8896 ± 0.0012 (`C2_star_review.md:100-102`). The
fix pushes accepted events to **higher** z. Sign convention consistent with the rest of the
selection stack (`S̄_φ`, `S_4D`, `f̄` all monotone-decreasing survival/completeness in z).

---

## 4. DIMENSIONAL ANALYSIS — both forms

| quantity | units | check |
|---|---|---|
| `w_g = R_eff_per_mbh(M_g)/(1+z_g)` | Gyr⁻¹ (`emri_rate.py:245`) ÷ dimensionless = **Gyr⁻¹** | appears identically in both forms; cancels in normalization |
| `z` | dimensionless | — |
| `k̄_g(z)` | **dz⁻¹** (normalized on `W_g`) | `∫ k̄_g dz = 1` by `:1335-1337` / `_inverse_cdf_draw`'s segment-sum, `:848-850` |
| `p_gal(M\|g)` | **M_⊙⁻¹** (Gaussian in M_⊙, `:1707`) | — |
| `d_L(z;h)` | **Gpc** (`physical_relations.py:250` "luminosity distances in Gpc") | passed at `:1711-1714` to the (d_L[Gpc], M_z[M_⊙]) interpolator ✓ |
| `M_z = M(1+z)` | **M_⊙** (`:1709`) | detector frame ✓ |
| `S_4D(d_L, M_z)` | **dimensionless**, ∈ [0,1] | required: it is a Bernoulli probability at `:1718-1719` ✓ |
| `S̄_φ(z;h)` | **dimensionless**, ∈ [0,1] | φ normalized in dlog₁₀M ⟹ a φ-average of a probability (`fixb_pathA…md:144`) ✓ |
| `S̃_φ,g` | **dimensionless**, ∈ [0,1] | `k̄_g`-average of `S̄_φ` ✓ |

```
[q_old]  =  Gyr⁻¹ · dz⁻¹ · M_⊙⁻¹ · 1 · 1   =  Gyr⁻¹ dz⁻¹ M_⊙⁻¹
[q_new]  =  Gyr⁻¹ · dz⁻¹ · M_⊙⁻¹ · 1       =  Gyr⁻¹ dz⁻¹ M_⊙⁻¹
```

**Both forms carry identical units**; the change removes a **dimensionless** factor `S̄_φ(z;h) ∈ [0,1]`.
The `Gyr⁻¹` is a common overall factor in both and cancels under normalization over `(g,z,M)`.
**PASS — no mixed units, no dimensional change.**

**Bernoulli-argument check:** the accept probability at `:1719` must be dimensionless and in `[0,1]`
in both forms — it is `S_4D` in both, untouched. **PASS.**

**h-consistency check:** `h` is fixed at `H_TRUE = 0.73` (`:359`, `:1614`, `:2119`) in the table
build, the `d_L` evaluation, and the interpolator query. The fix introduces no new `h` dependence
and touches no `h`-scan. **PASS.**

---

## 5. LIMITING CASES

Notation: `q_old`, `q_new` as in §1.3 / §2.1, both normalized.

### L1 — Complete detection, `S_4D ≡ 1` everywhere → **PASS**
`S̄_φ(z) = ∫ φ · 1 dlog₁₀M = 1` (φ normalized). Then
`q_old ∝ w_g k̄_g p_gal · 1 · 1 = q_new`. The two forms are **identical**, and the rejection loop
accepts every candidate (`u_batch < 1` a.s.), so `n_drawn_total == n` in one round.
**The complete-catalogue / perfect-detection limit recovers the same answer in both forms.** ✓

### L2 — No selection *gradient*: `S_4D ≡ c`, constant, `0 < c ≤ 1` → **PASS**
`S̄_φ ≡ c`. Then `q_old ∝ c²·(w_g k̄_g p_gal)` and `q_new ∝ c·(w_g k̄_g p_gal)`. After
normalization **both equal `w_g k̄_g p_gal` — the same distribution.** The `c` in the z-density
cancels inside `_inverse_cdf_draw`'s segment-sum normalization (`:848-853`); the acceptance rate is
`c` in both forms. ✓
**Physical reading (important for interpreting §8): the defect is a *gradient* defect, invisible to
any constant survival. All of the 13.5–16 % comes from the z-*dependence* of `S̄_φ`.**

### L3 — Non-vacuity: realistic z-dependent `S_4D` → **PASS (the forms provably differ)**
On the frozen 24-seed b0i2d fleet, with per-event weight `ω = (1−w₂)·1_F0`:
```
R_pred(ω) = E_b[ω]·E_b[1/S̄_φ] / E_b[ω/S̄_φ] = 0.86473 ± 0.00511   (identity)
                                            = 0.84024 ± 0.00626   (BR)
```
(`C2_star_review.md:73-84, 95-96`). `min S̄_φ = 0.481` over all 4800 accepted events, so the
harmonic estimator has no heavy tail (`:102`). **The two forms differ by 13.5 % / 16.0 % on the
registered statistic — the fix is not vacuous.** ✓

### L4 — Mass-marginal consistency: does the defect equal `S²`? → **PASS, with a correction to the record**
Marginalize `q_old` over `M`. *If* `p_gal(·|g) ≡ φ` for every host, then
`∫ p_gal S_4D dM = S̄_φ(z)` and `q_old`'s z-marginal `∝ w_g k̄_g S̄_φ²` — the literal `S²` form. But
`p_gal` is the host's own Eddington-shifted **catalogue-mass** Gaussian (`:1704-1708`), and the
pool-vs-φ offset is measured at **+0.28 dex, KS D = 0.215** (`fixb_pathA_phi_marginal_selection.md:161`).
**So the defect is `S̄_φ(z)·S_4D(z,M)`, not `S_4D²`.** The check passes (the two objects are the same
survival function, so the double application is real), but the record's `S²` phrasing is imprecise
and should not be carried into the ledger. ✓ *(correction supplied by this package)*

### L5 — `C2_star_review.md:169-170` disjunct 2, literal ("drop the Bernoulli") → **FAILS**
Law becomes `∝ w_g k̄_g(z) S̄_φ(z) p_gal(M|g)`. Its **M-conditional is `p_gal(M|g)`, unselected in
mass**, whereas the target's is `p_gal(M|g)·S_4D(z,M)/∫`. These differ whenever `S_4D` varies with
`M` at fixed `z` — which is the entire premise of the [P3-2D] mass-law extension (prereg §2.4's
"monster event" class; GATE M2-LINK, prereg §3 lines 64-68). **This disjunct must not be
implemented as written.** ✗

### L6 — `C2_star_review.md:168-169` disjunct 1, literal (Option A, host weight untouched) → **FAILS**
Law becomes `∝ w_g · S̃_φ,g · k̄_g(z) · p_gal(M|g) · S_4D`. Residual factor vs target: **`S̃_φ,g`**,
an uncancelled host-level survival weight (§3.3). Since `S̃_φ,g = ∫k̄_g S̄_φ ≈ S̄_φ(z_g)` and
`z_ev ≈ z_g ± σ_z,eff` with `σ_z,eff ≈ 0.035` against a pool z-range of order 0–0.5, the residual
tilt is strongly correlated with the one removed. **Option-A-literal is expected to leave the large
majority of the 13.5–16 % in place.** ✗
*Fraction remaining: UNMEASURED. Directly measurable at zero new compute — §7 test R4.*

### L7 — `sbarphi_defect_location_20260827.md` §4 Option B (importance weight `S_4D/S̄_φ`) → **PASSES algebraically, FAILS on interface**
Exact (§2.3 algebra). But it produces a **weighted** sample, and no consumer of the venue accepts
weights (§9). Not recommended; viable only with the downstream rework enumerated in §9. ⚠

### L8 — Non-regression of the 1D `catalogue_selected` (b0i) arm → **PASS, conditional on scoping**
The 1D venue has **no Bernoulli layer**. Its law is `w_g S̃_φ,g × k̄_g S̄_φ/S̃_φ,g = w_g k̄_g S̄_φ` —
survival applied **once**, matching its own model-side normalizer `Σ̃^φ = Σ_g w_g S̃_φ,g`. **The 1D
arm is correct and must not change.** The b0-identity result (rows #198–#202, UNDISCRIMINATING) and
the live `[HIER]` blocker-A conclusion both depend on it (§9).
**PASS if and only if** the fix is scoped to the 2D branch and neither
`_draw_kernel_survival_redshifts`'s density nor `catalogue_selected_host_draw_weights`'s first
return value is edited in place. ✓ *(conditional — this is a binding implementation constraint,
not an observation)*

### L9 — Acceptance-efficiency / GATE-ACC bound → **NOT A CORRECTNESS CHECK; OPERATIONAL RISK**
Under Option A′ the proposal density shifts to higher z, where `S_4D` is smaller, so the Bernoulli
acceptance rate **drops**, by roughly the same order as the tilt being removed. `_M2D_MAX_ROUNDS = 200`,
`_M2D_MAX_BATCH = 4000`, `_M2D_BATCH_MULTIPLIER = 4` (`:1560-1563`) and the STOP at `:1733-1738`
must be re-checked against the new efficiency before the fleet re-run, or the GATE-ACC-style STOP
may fire on a correct configuration. **UNMEASURED.** ⚠

### Summary

| check | verdict |
|---|---|
| L1 complete catalogue / `S_4D≡1` | **PASS** (forms identical) |
| L2 constant survival (no gradient) | **PASS** (forms identical after normalization) |
| L3 realistic z-dependence | **PASS** (forms differ, 13.5 %/16.0 % — not vacuous) |
| L4 mass-marginal / is it `S²`? | **PASS** with a correction: it is `S̄_φ·S_4D`, not `S_4D²` |
| L5 record's disjunct 2, literal | **FAIL** — destroys the 2D mass selection |
| L6 record's disjunct 1, literal (Option A) | **FAIL** — uncancelled `S̃_φ,g` host weight survives |
| L7 defect-doc Option B (importance weight) | algebraically **PASS**, interface **FAIL** |
| L8 1D b0i arm non-regression | **PASS** iff scoped to the 2D branch (binding constraint) |
| L9 acceptance efficiency / GATE-ACC | not a correctness check — **UNMEASURED operational risk** |
| **Option A′ (§2.2) against L1–L4, L8** | **PASS on every check** |

**Not all checks pass for the fix as granted.** L5 and L6 fail. Option A′ — Option A plus the
supplied host-weight change — passes everything.

---

## 6. REFERENCE — with honest scope

### 6.1 What the literature does establish

**Mandel, Farr & Gair (2019), arXiv:1809.02063, Eqs. (5)–(7)** — quote-verified this pass. Canonical
hierarchical selection form:
`p(d|λ') = ∫ dθ p(d|θ) p_pop(θ|λ') / α(λ')`, with `α(λ') = ∫ dθ p_det(θ) p_pop(θ|λ')`.
**Structurally: the survival/detection probability appears exactly once**, inside the normalizing
integral, and is absent as a second factor on the per-event numerator. This is the principle §3.1
instantiates. Applied **by structural analogy only** — Eqs. (5)–(7) do not discuss mock-data
generation. It is the project's own cited authority for this defect class
(`docs/derivations/fixb_pathA_phi_marginal_selection.md:74-76`, "assumption A2").

**Gray et al. 2023, arXiv:2308.02281v2 (JCAP) §2.1.3 (Eqs. 2.17–2.21) / §2.1.4** — quote-verified.
A normalization/truncation choice for `p(z,M|Λ,I)` has *"no bearing on the result"* **provided**
*"the same expression for `p(z,M|Λ,I)` is also be used in Eq. [16]"* — a **same-object-identity**
condition across the numerator and normalization roles.

### 6.2 What the literature does NOT cover — absences, stated plainly

1. **No surveyed dark-siren methodology paper gives a correctness criterion for this exact
   configuration** — a survival factor folded into an importance-weighted MC *sampling density* for
   synthetic "detected" events, combined with an additional independent Bernoulli accept/reject on
   the same drawn triple. Surveyed: Gray 2020, Gray 2023, the Hitchhiker's Guide (arXiv:2212.08694),
   Mandel/Farr/Gair 2019, Alfradique/Bom/Castro 2025, Borghi et al. 2025, VanWyngarden et al. 2025
   (full roster in `docs/LITERATURE_WARNINGS.md`). **None of them are mock-data/injection-generation
   papers; they are inference papers.** This is a **new recorded absence**, adjacent to two existing
   project precedents (no cited paper treats mass-covariate deconvolution; none decomposes bias into
   catalogue/completion-sector conditionals).
2. **The double-application argument in §3.2 is first-principles elementary probability, not a
   citable domain-specific result.** MFG19 supplies the single-appearance *principle*; the
   rejection-sampling arithmetic is not attributable to any paper.
3. **No literature constrains the choice between the repairs.** Picking A′ over B′ is a code-design
   decision on interface grounds (§2.3, §9), not something the surveyed literature adjudicates.
4. **Gray 2023 §2.1.3/§2.1.4 must not be cited as settling this.** It is the project's anchor for a
   *different, already-resolved* question — why the **production** `S̄_φ` numerator/normalization
   pairing in `bayesian_statistics.py` is self-consistent (register row `G23-c-check`, CHECKED,
   commit `2b10b8b8`). Citing it here would overstate what was verified.
5. **`docs/LITERATURE_WARNINGS.md` row `MFG-a`** — MFG19's single-sentence prose statement of
   "assumption A2" was **not literally located** this pass; only Eqs. (5)–(7) were quote-verified.
   The row should move to a **qualified** status ("equations verified; prose paraphrase not
   literally located"), **not** a bare CHECKED. *The register was not edited by this pass.*
6. **Fetch caveat.** Quotes were obtained via WebFetch (an intermediate summarizing model),
   cross-checked across two independent fetches per paper and mutually consistent — but this is
   **not** the raw-PDF cross-check the project's own register convention requires before a quote is
   used directly in the paper.
7. **`PROPOSAL_SIGMA_PHI_DIVISOR_20260822.md` is a different, still-author-gated proposal**
   (production divisor `Σ^3D → Σ^φ` in `bayesian_statistics.py`). It must not be conflated with this
   harness fix.

Full literature package with quotes and source list:
`results/campaign51_20260728/realistic_20260729/sbarphi_literature_20260827.md`.

### 6.3 Trigger-file status — for the author's explicit ruling

`darksiren_emri/validation/correspondence_1d.py` is **NOT** on the `/physics-change` trigger list
(`CLAUDE.md` §"Physics-change trigger files"; `.claude/skills/physics-change/SKILL.md:17-23`), and
**does not match** `.claude/rules/physics-validation.md`'s `paths` frontmatter (which covers
`**/bayesian_inference/**` but not `**/validation/**`).

This package is nevertheless assembled to the full protocol, on two grounds: CLAUDE.md's
*"When in doubt, treat it as a physics change"*, and the fact that this harness's generative law is
precisely what the `C₂*` physics identity is scored against. **Whether the trigger list should be
amended to include `darksiren_emri/validation/` is a [RULE] for the author** — it is not assumed
here, and no rule file was edited.

---

## 7. REGRESSION TEST DESIGN

All CPU-only, no `@pytest.mark.gpu`, per the testing strategy. All reuse existing doubles in
`darksiren_emri_test/validation/test_correspondence_1d.py`: `_fake_phi_survival_table`
(`:815-824`, `S̄_φ(z) = exp(-3z)` — usefully steep), `_FakeIncompleteness`, `_FakeS4D`,
`_make_host_pool_with_mass_and_error`, `_make_donor_csv_2d`. **Descriptions only — no test code has
been written.**

### R1 — OLD-value pin (**must land BEFORE the change**, per CLAUDE.md testing rule 3)
Call `_draw_2d_accepted_latents` with `np.random.default_rng(0)`, the mass-and-error host pool,
`_fake_phi_survival_table(z_max=1.0)`, `_FakeIncompleteness`, an `_FakeS4D` with a genuinely
`(d_L, M_z)`-dependent return (**not** the constant `_FakeS4D(value=0.9)` used at `:1616` — L2 makes
a constant blind to this change), `n=500`. Assert to ~1e-12: `mean(z_true)`, the 0.1/0.5/0.9
quantiles of `z_true`, `mean(s4d_at_truth)`, and `n_drawn_total`. Purpose: make the diff visible.
Expected to **change** at the fix; the new pins are recorded in the same commit.

### R2 — Law-identity discriminator (the decisive test; RNG-stream-robust)
With the same analytic doubles, compute by quadrature the two candidate target densities
`T_old ∝ w_g k̄_g(z) p_gal(M|g) S̄_φ(z) S_4D(z,M)` and `T_new ∝ w_g k̄_g(z) p_gal(M|g) S_4D(z,M)`
on a small hand-built pool (~50 hosts). Draw `N ≈ 2×10⁵` accepted latents; compare empirical
`E_q[f]` against both quadrature targets for a battery `f ∈ {1, z, M, 1{z<z*}, S̄_φ(z)}` with a 3σ
MC band. **Assertion: the sample matches `T_new` and is rejected against `T_old`.** Under the
current code the verdict inverts. This is the test that proves the new form correct, and it does not
depend on the RNG stream ordering.

### R3 — Host-weight coupling guard (**would have caught the L6 failure**)
Assert directly that the 2D branch's `p=` argument at `:1682` is the normalization of `w_g` alone,
by constructing a pool with a strongly host-varying `S̃_φ,g` (wide spread in `z_g`) and checking the
empirical host-index histogram against `w_g/Σw_g` (χ², 3σ). Under Option-A-literal this **fails**
with a clear `S̃_φ,g` tilt. **This is the single most important new test.**

### R4 — Quantify the L6 residual (zero new compute, run BEFORE the fleet re-run)
Not a unit test: re-run
`results/campaign51_20260728/realistic_20260729/p3_2d_forensic_20260826/venue_drift_adjudication.py`
with `S̄_φ(z_ev)` replaced by the banked `S̃_φ,g` of each event's host. The resulting `R_pred` is the
drift that Option-A-literal would leave behind. Turns §0's analytic claim into a number. ~minutes.

### R5 — L1 limit encoded
`_FakeS4D` returning exactly 1.0 and a flat `S̄_φ ≡ 1` table: assert `n_drawn_total == n`,
`n_rounds == 1`, `s4d_at_truth == 1.0` exactly, and that `z_true` is bit-identical to a direct
`_draw_kernel_survival_redshifts` call with the flat table on the same seed. Must pass identically
before and after the fix.

### R6 — L2 limit encoded
`_FakeS4D(value=c)` constant: assert the `z_true` empirical distributions from old and new code
paths are statistically indistinguishable (two-sample KS across ≥8 seeds, p > 0.01 each, or a
pooled Fisher combination). Guards against a fix that changes behaviour where it provably must not.

### R7 — Mass selection retained (guards against the L5 failure)
`_FakeS4D` depending on `M_z` only (flat in `d_L`): assert the accepted `M_true` distribution is
**not** `p_gal` and **does** match `p_gal·S_4D/∫` by quadrature (3σ). Fails loudly if anyone
implements the "drop the Bernoulli" disjunct.

### R8 — 1D non-regression (the L8 binding constraint, encoded)
Assert the `"catalogue_selected"` (b0i) branch at `:2053-2089` produces **bit-identical** output
before and after the change for a fixed seed, and that `catalogue_selected_host_draw_weights`'s
first return value is still `w_g·S̃_φ,g` normalized (the existing test at `:1244` covers the latter
— extend it with an explicit "2D branch must not use this value" assertion).

### Existing tests that will need updating
`test_correspondence_1d.py:1616` (`…_is_deterministic`), `:1647` (`…_seed_sensitivity`), `:1678`
(`…_records_columns`), `:1734` (`…_rejection_loop_stops_on_zero_survival`), `:1758`
(`test_catalogue_selected_mode_does_not_enter_catalogue_selected_2d_code_path`) — all consume the 2D
draw and several use `_FakeS4D(value=0.9)` (constant, hence L2-blind: they will pass unchanged,
which is itself a reason R1–R3 are needed).

---

## 8. EXPECTED EFFECT

### 8.1 What the 13.5–16 % is

`R_pred(ω) = E_b[ω]·E_b[1/S̄_φ] / E_b[ω/S̄_φ]` — the ratio of the **registered LHS₂ statistic as
measured on the current venue** to what it would be on a venue realizing the model class-G law
(`C2_star_review.md:73-84`, `venue_drift_adjudication.py:9-11`):

| weight | `R_pred` (per-seed mean ± SEM) | pooled | deficit |
|---|---|---|---|
| identity, `ω = (1−w₂)·1_F0` | **0.86473 ± 0.00511** | 0.86288 | **13.5 %** |
| BR, `ω = (1−w₂)/(1+(r₂−1)w₂)·1_F0` | **0.84024 ± 0.00626** | 0.83769 | **16.0 %** |

**Venue:** the 24-seed × 2-arm b0i2d fleet, `p3_2d_fleet_20260825/` — the same fleet that produced
the frozen PA-2D-9 numbers. **Corroboration:** matches the A20 F9 line's independent
"review-computed venue-drift reference" of **−13.6 %** (`C2_star_review.md:107-109`). **Replication:**
the reviewer's recomputed LHS₂/LHS₂,BR match the frozen values to <1e-8 (`:202-204`). This is a
re-derivation-confirmed measurement on already-frozen numbers, not a fresh run.

### 8.2 What it applies to

**Only the [P3-2D] harness identity statistic LHS₂ (and its BR transform) on the b0i2d venue.** It
is a shift of a *harness-internal correspondence check*.

### 8.3 What is explicitly NOT known

- **The effect on H₀ is entirely unknown — and there is no reason to expect one.**
  `darksiren_emri/validation/correspondence_1d.py` is a validation/coverage harness. It produces no
  H₀ posterior of record. **No production H₀ number, no banked H₀ posterior, and no paper H₀ figure
  changes as a result of this fix.** Anyone reading "13.5–16 %" as an H₀ shift is reading it wrong.
  *(If the author wants an H₀-side statement, that requires a separate registered measurement; none
  exists.)*
- **The fix does not resolve the [P3-2D] failure.** Venue-corrected LHS still misses RHS by
  **4.6×** (identity) and **2.7×** (BR) the ε₂ = 1.914e-3 band; the residual inflation factors are
  `X_id = 2.506`, `X_BR = 2.297`, still UNATTRIBUTED (`C2_star_review.md:103-105, 127-128`; rows
  #210/#211). Runbook 34's *"necessary regardless of the big residual"* is exactly right — necessary,
  not sufficient.
- **The post-fix LHS₂ is not predictable from `R_pred` alone.** `R_pred` predicts the *drift* under
  the current draws; the re-run venue is a different sample, so the realized post-fix LHS₂ carries
  its own MC error. Do not pre-commit to `LHS₂_new = LHS₂_old / 0.86473`.
- **The residual left by Option-A-literal is UNMEASURED** (§5 L6; test R4 measures it).
- **The post-fix acceptance efficiency is UNMEASURED** (§5 L9); GATE-ACC bounds may need widening.
- **Whether the CONTROL-FAIL → VENUE-MISSPEC re-map holds after the fix is a [RULE], not a
  consequence.** It returns to the author (`C2_star_review.md:150-155`).

---

## 9. SCOPE AND BLAST RADIUS

### 9.1 Files that would change (Option A′)

| file | change | trigger-list status |
|---|---|---|
| `darksiren_emri/validation/correspondence_1d.py` | 2D branch only: host weight `:2107`, z-draw call `:1687-1696`, docstrings `:1572-1575`/`:1620-1628`/`:1946`/`:1954-1971` | **NOT** on the trigger list — see §6.3 |
| `darksiren_emri_test/validation/test_correspondence_1d.py` | R1–R8; update `:1616`, `:1647`, `:1678`, `:1734`, `:1758` | — |
| `docs/gates/PHYSICS-GATE-LEDGER.md` | rows per §10 | — |

**Files that MUST NOT change** (binding, from L8): `_draw_kernel_survival_redshifts`'s density body
(`:1490-1498`), `catalogue_selected_host_draw_weights`'s first return value (`:1380-1385`),
`kernel_smeared_survival` (`:1242-1338`), and **anything under
`darksiren_emri/bayesian_inference/`** — the production estimator is confirmed defect-free
(`C2_star_review.md:19-34` item 3: no `S̄_φ` in the with-BH numerator; `bayesian_statistics.py:6362-6368`
applies `S̄_φ` once, correctly, in the *without*-BH twin only).

### 9.2 Every consumer of the 2D venue draw law

| consumer | file:line | impact |
|---|---|---|
| `draw_realization`, `"catalogue_selected_2d"` branch | `correspondence_1d.py:2090-2132` | direct — the only in-tree consumer |
| GATE-ACC independent replay | `p3_2d_fleet.py:285-320` (`c1d._draw_2d_accepted_latents` at `:313`) | **must be re-derived**; also calls `catalogue_selected_host_draw_weights` at `:299` and would inherit the stale host weight |
| RHS scorer's cached host-weight monkeypatch | `ca_rhs_scorer.py:168-192`, parity check `:846-875` | **parity check must be re-verified** — if the 2D branch stops consuming the first return value, the cache's contract changes |
| RHS-inflation confirmation instrument | `p3_2d_forensic_20260826/rhs_inflation_confirmation.py:101, 136, 260` | replicates the host-weight and mass-draw code inline — **would silently diverge** |
| `wbhzero` gate-B probe | `gate_b_20260730/wbhzero_gate_b_scripts/wbhzero_probe.py` | references `catalogue_selected_2d`; re-check |
| unit tests | `test_correspondence_1d.py:1558-1790` (8 tests) | update per §7 |

### 9.3 Banked results that become STALE

**Stale on the fix (must be re-run or re-labelled):**
- The **24-seed × 2-arm b0i2d fleet** `p3_2d_fleet_20260825/` — every `bt_*`/`bc_*` work dir and
  `prepared_cramer_rao_bounds.csv`. Re-run cost ≈ **2–4 CPU-h** (`C2_star_review.md:170-172`).
- **PA-2D-9's frozen LHS₂ numbers**: 0.0050077 ± 0.00011615 (identity), 0.00332207 ± 9.164e-5 (BR);
  the observed ratios 0.34505 ± 0.01342 / 0.36575 ± 0.01390; the banked coded-arm venue-drift control
  `S₂ = 0.390399 ± 0.010344` (prereg `:47`).
- **PA-2D-9's CONTROL-FAIL verdict** and row #211's PARKED-at-UNATTRIBUTED-bounded status.
- The **F8 coherence residual** (−6.01e-4) and the **GATE-ACC** bands (F12 extended class-G replay).
- The **`STUCK_P3_2D_SYMPTOM_CARD_20260826.md`** figures.

**NOT stale (survive the fix untouched):**
- `C₂* = 0.06124403326364123` — a model-side constant, derived from the estimator, **venue-independent**
  (`C2_star_review.md:47-50, 185`).
- `Σ̃^4D`, `β̄_Ḡ_φ`, `Σ^φ`, `Σ^4D`, `r₂ = 2.6124925`, `α_G_φ` — all model-side, independently arbitered.
- **Everything on the RHS/completion side**, including PA-2D-10's refutation of the completion-mass
  axis (`X_alt = 0.9997 ± 0.0003`, row #210) — the completion venue is `population_selected`, a
  different branch, untouched.
- **Every production result**: all H₀ posteriors, all `bayesian_statistics.py` outputs, the WBHZERO
  adoption (`[PHYSICS] cf4f8a2a`), the b0-identity UNDISCRIMINATING verdict (rows #198–#202, **1D**
  arm — protected by L8).

### 9.4 Cross-thread coupling — **live, and it is a trap**

`results/campaign51_20260728/realistic_20260729/hier_blocker_a_generator_law_20260827.md` (dated
**today**) concludes that the `[HIER]` thread's blocker is fixed by switching `host_mode` to
`"catalogue_selected"` (1D, arm b0i), and its §3 reasoning **explicitly depends on the `S̄_φ` factor
being present in `_draw_kernel_survival_redshifts`'s density** (*"the remaining factors
`w_pop(z)·f_k(z)·S̄_φ(z)` are not extraneous to the estimator's side … truth-theta = (0,1) genuinely
holds for this mode"*, `:53-59`).

**Any implementation that edits `_draw_kernel_survival_redshifts`'s density in place would silently
invalidate the `[HIER]` blocker-A conclusion reached this morning.** This is the operational reason
L8's scoping constraint is binding rather than stylistic, and it is the single highest-risk aspect
of an otherwise small change.

---

## 10. PROPOSED LEDGER ROW (to be appended only when the author answers)

Per `.claude/skills/physics-change/SKILL.md:52-65`. **Not yet written.**

```
| 2026-08-27 | pre-commit | presented | APPROVED|REJECTED | validation/correspondence_1d.py:1380,1497,1682,2107 | class-G S̄_φ de-double-weight (2D venue): drop S̄_φ from the 2D z-draw AND swap the 2D host weight to plain w_g |
```

Note for the ledger: `correspondence_1d.py` is not currently a trigger file; if the author declines
to amend the trigger list (§6.3), the row should carry an explicit "voluntary gate" marker so the
ledger's evidence contract is not misread.

---

## 11. DECISION LIST

| tag | item |
|---|---|
| **[RULE]** | Is the S̄_φ fix genuinely granted? The record asserts it at rows #209–#211 and in runbooks 34/35, but **no verbatim author quote exists** (`sbarphi_defect_location_20260827.md` §1). One line of confirmation closes a ledger-hygiene gap. |
| **[RULE]** | **§0/§5 headline:** the granted fix's disjunct 1 (Option A literal) FAILS check L6 and disjunct 2 FAILS check L5. Ratify **Option A′** (§2.2) — Option A **plus** the supplied host-weight change — as the corrected form. |
| **[RULE]** | Should `darksiren_emri/validation/` be added to the `/physics-change` trigger list (CLAUDE.md + `.claude/rules/physics-validation.md` frontmatter)? This harness law is what the `C₂*` physics identity is scored against. |
| **[DO]** | Run test **R4** first (zero new compute, minutes): quantify what Option-A-literal would have left behind. Converts §0's analytic claim into a number before any code is written. |
| **[DO]** | Implement Option A′ scoped to the 2D branch, with R1–R3 and R8 landing **before** the change (R1 pins the old values; R3/R8 are the guards that would have caught L6/L8). |
| **[DO]** | Re-run the 24-seed × 2-arm b0i2d fleet (~2–4 CPU-h) **after** re-checking the GATE-ACC bounds against the new acceptance efficiency (§5 L9). |
| **[RULE]** | Post-fix: whether the CONTROL-FAIL → VENUE-MISSPEC re-map holds. Not a consequence of the fix — a fresh ruling on fresh numbers. |
| **[RULE]** | `docs/LITERATURE_WARNINGS.md` row `MFG-a` → **qualified** ("equations verified; prose paraphrase not literally located"), not bare CHECKED. Register not edited by this pass. |
| **[FYI]** | The campaign's "`S²`" phrasing is imprecise: the defect is `S̄_φ(z)·S_4D(z,M)` — two different survival objects — not `S_4D²` (§1.3, §5 L4). Worth correcting in the ledger before it propagates. |
| **[FYI]** | `hier_blocker_a_generator_law_20260827.md` §3 depends on `S̄_φ` staying in the shared 1D z-draw. Scope the fix to the 2D branch or that conclusion silently breaks (§9.4). |

---

## ADVERSARIAL REVIEW [OPUS-ORCH 2026-08-27]

**Reviewer posture: refutation.** Every cited site re-opened independently; every limiting case
re-computed rather than read; the blast radius re-grepped from scratch; the record's own
disjuncts re-read at source. **No code was written and no file under `darksiren_emri/` was
touched by this pass** (`git status` on `darksiren_emri/`, `darksiren_emri_test/`, `docs/`,
`.claude/` clean at review time). This section is append-only.

### AR-0. Verdict

**FIX-MISSPECIFIED.** The package's central claim survives adversarial attack and is now
**quantified**: the fix *as granted in the record* (`C2_star_review.md:168-169`, disjunct 1,
"drop `S̄_φ` from the 2D z-draw") leaves **~69–70 % of the measured 13.5 %/16.0 % drift in
place** (measured this pass, §AR-2 — the package flagged it UNMEASURED). Option A′ is the
correct form and is confirmed exact.

The **derivation holds**. The **package does not go to the author as written**: seven defects
below, of which three (D2, D3, D5) are regression tests that **assert things that are false on
correct code**, and one (D1) is an internal contradiction in the headline.

### AR-1. The crux — is the double-count real, or two legitimate applications?

Attacked directly, on the hypothesis that the granted fix is a **regression**. It is not.

**(a) The realized law, re-derived from code alone.** `catalogue_selected_host_draw_weights`
returns `normalized = w_g·S̃_φ,g/Σ` as its *first* value (`correspondence_1d.py:1380-1385`);
`draw_realization`'s 2D branch passes exactly that as `host_w` (`:2107` → `:2113`) and
`_draw_2d_accepted_latents` consumes it at `:1682`. `_draw_kernel_survival_redshifts` builds
`density_i = kernel_i * w_pop_eff_i * s_i` (`:1496-1497`) and `_inverse_cdf_draw` normalizes it
by its own trapezoid segment-sum (`:848-853`). `kernel_smeared_survival` computes
`S̃_φ,g = numerator/z_g_norm` on the **same** `_host_kernel_window` (`:1324` vs `:1486`), i.e.
literally `∫_{W_g} k̄_g S̄_φ dz`. Therefore `S̃_φ,g` cancels and

    q_old(g,z,M) ∝ w_g · k̄_g(z) · p_gal(M|g) · S̄_φ(z;h) · S_4D(d_L(z;h), M(1+z))

**Independently reproduced. §1.3 is correct.**

**(b) Is the target really the `Σ̃^4D` integrand?** Verified *outside* the package's chain, in
the model-side code: `p3_2d_companion.py:904-951` computes
`s_tilde_4d[g] = numerator/z_norm` (`_segmented_integral_batch`, the `p_gal`×`S_4D` contraction
on `_host_kernel_window`) and `sigma_tilde_4d = float(np.sum(w_g * s_tilde_4d))` (`:951`) —
i.e. `Σ̃^4D = Σ_g w_g ∫∫ k̄_g p_gal S_4D dM dz`, **no `S̄_φ` anywhere**. The prereg is explicit
and independent of the review: the 2D numerator carries survival *inside the candidate's own
mass quadrature* — "**NOT point-S_4D, NOT `S̄_φ(z)`, NOT `S̃_φ,g`**"
(`PREREGISTRATION_P3_2D_20260825.md:14-16`).

**(c) Could the two appearances be two legitimate objects?** No. `S̄_φ(z;h) ≡ ∫φ(log₁₀M)·
S_4D(d_L(z;h), M(1+z)) dlog₁₀M` (`docs/derivations/fixb_pathA_phi_marginal_selection.md:60`,
verified verbatim; builder `bayesian_statistics.py:1947`) is a **φ-marginal of the same
`S_4D`**, not an independent physical fact about the triple. For a venue that carries a
per-event latent mass, detection is `S_4D` at *that* mass, once. `S̄_φ` is the right object
exactly where the leg has no per-event mass — the without-BH twin numerator, where it is
applied once and correctly (`bayesian_statistics.py:6362-6368`, re-read this pass). **The
"double-weight" is genuine; the granted fix's *direction* is not a regression.**

**(d) Index-set check the package omitted, and which passes.** Option A′ can only "exactly"
realize `Σ̃^4D`'s integrand if the venue's host index set equals the model's. Checked: the
venue pool is `_host_pool_from_handler` on the handler pruned to `M ∈ [1e4, 1e7]`, `z < 1.5`
(`constants.py:111,125-126`; `correspondence_1d.py:726-731`); the model's eligible set is
`(z < z_grid.max()) & isfinite(M) & (M > 0)` with `z_grid.max() = z_max_cap = HOST_DRAW_Z_MAX
= 1.5` (`p3_2d_companion.py:181-182`; `correspondence_1d.py:1064`). The extra mask is a
no-op. **No venue/model index mismatch. Gap closed in the package's favour.**

### AR-2. R4 EXECUTED — the package's central UNMEASURED claim is now a number

`s_tilde_phi_host` **is** banked in every fleet CSV (column 131 of
`prepared_cramer_rao_bounds.csv`, written at `correspondence_1d.py:2236`), so R4 is genuinely
zero-new-compute. Run this pass over all 24 `bt_*` seeds.

**Pipeline validated first** (same construction as `venue_drift_adjudication.py`; F-0 =
`σ_dL/d̂ < 0.10 ∧ SNR ≥ 20`, 0-based `event_idx` mapping confirmed, 84/200 on seed 900101):

| replicated | mine | frozen |
|---|---|---|
| LHS₂ (identity) | 0.005008 ± 0.000116 | 0.00500770 ± 0.00011615 |
| LHS₂,BR | 0.003322 ± 0.000092 | 0.00332207 ± 0.00009164 |
| weighted ⟨z⟩ / all-accepted ⟨z⟩ | 0.18301 / 0.11879 | 0.183 / 0.119 |

**Result** — the same estimator `R = E_b[ω]·E_b[1/S]/E_b[ω/S]`, with the residual host weight
`S = S̃_φ,g` (the exact object Option-A-literal leaves uncancelled) instead of `S̄_φ(z_ev)`:

| | identity ω | BR ω |
|---|---|---|
| drift today (record, `S̄_φ(z_ev)`) | 0.86473 ± 0.00511 | 0.84024 ± 0.00626 |
| **drift left by Option-A-literal (`S̃_φ,g`)** | **0.90679 ± 0.00494** | **0.88840 ± 0.00621** |
| same, with the q_old→q_A measure correction | 0.9107 ± 0.0045 | 0.8930 ± 0.0057 |
| **fraction of the deficit that SURVIVES** | **≈ 69 %** | **≈ 70 %** |

(Measure correction: `q_A/q_old = S̃_φ,g/S̄_φ(z_ev)`, applied by importance weight with a
`S̄_φ`-proxy curve regressed from the banked `(z_true, S̃_φ,g)` pairs; it moves the answer by
+0.004, confirming the leading term.)

**L6 is CONFIRMED and is the decisive finding of the whole package.** Corroborating structure:
`corr(z_true, S̃_φ,g) = −0.799` over the 4800 accepted events; `S̃_φ,g ∈ [0.4988, 0.9895]`,
mean 0.8797 — a host-level tilt of nearly the same amplitude as the event-level one it would
replace.

**But the package's *wording* overstates in the other direction.** §0 says Option-A-literal
"is expected to remove only the small within-kernel part". It removes ~30 %, not a "small"
part. Replace with the measured statement.

### AR-3. Limiting cases — re-computed, not read

- **L1 (`S_4D ≡ 1`)** — the *limit* holds: `φ` is normalized (`bayesian_statistics.py:1944`,
  `/Z_phi`), so `S̄_φ ≡ 1`, `S̃_φ,g ≡ 1`, and the host weight `w_g·1` normalizes to the same
  thing as plain `w_g`. Old ≡ new. **PASS.** Its *arithmetic sub-claim* is wrong — see **D2**.
- **L2 (`S_4D ≡ c`)** — re-derived including the host weight (which §5 L2 does not mention):
  `S̄_φ ≡ c`, `S̃_φ,g ≡ c`, host weight `w_g c/Σ w c = w_g/Σ w`, z-density `k_g c` normalizes to
  `k̄_g`, acceptance `= c` in both. Old ≡ new exactly. **PASS**, and stronger than stated.
- **L3 (non-vacuity)** — 13.5 %/16.0 % re-verified via my own LHS replication above. **PASS.**
- **L4 (is it `S²`?)** — the `+0.28 dex`, `KS D = 0.215` pool-vs-φ figure is at
  `fixb_pathA_phi_marginal_selection.md` §3 "tower limit" row, verbatim. The correction to the
  campaign's `S²` phrasing is right and worth carrying. **PASS.**
- **L5** — **strawman, see D1.**
- **L6** — **FAILS, confirmed and quantified (AR-2).**
- **L7** — Option B algebra re-derived: `[w_g S̃][k̄ S̄/S̃][p_gal][S_4D/S̄] = w_g k̄ p_gal S_4D` ✓
  exact. The interface objection is real. **Correct as stated.**
- **L8** — 1D law `w_g S̃_φ,g × k̄_g S̄_φ/S̃_φ,g = w_g k̄_g S̄_φ`, survival once. **PASS**, and
  the scoping constraint is correctly binding. See D7 for one over-claim about what scoping buys.
- **L9 — now measured, and the warning is overstated.** Implied Bernoulli acceptance today is
  `1/E_q[1/S_4D] = 0.535` over the 4800 banked `s4d_at_truth` values (tail-robust: the five
  largest `1/S_4D` contribute 3.7 % of the mean; `max = 121`). Under the fix, ≈ 0.518 — a
  **~3 % drop**. With `batch = 4×remaining` and `_M2D_MAX_ROUNDS = 200` (`:1560-1563`) there
  are ~3 orders of magnitude of headroom. **The GATE-ACC STOP will not fire.** Keep the
  re-check as diligence; do not let it gate the fleet re-run.

### AR-4. Dimensional analysis — independently confirmed, no defect

`R_eff_per_mbh` is `[Gyr⁻¹]` (`emri_rate.py` docstring, Eq. 34 block); `dist`/`dist_vectorized`
return **Gpc** ("Array of luminosity distances in Gpc", `physical_relations.py:250`), matching
the `(d_L[Gpc], M_z[M_⊙])` interpolator query at `:1711-1714`; `k̄_g` is `dz⁻¹` by
`_inverse_cdf_draw`'s trapezoid normalization (`:848-853`) and `Z_g` (`:1335-1337`); `p_gal` is
`M_⊙⁻¹`; `S_4D`, `S̄_φ`, `S̃_φ,g` are dimensionless in `[0,1]` (φ normalized in `dlog₁₀M`).
Both forms `Gyr⁻¹ dz⁻¹ M_⊙⁻¹`; the removed factor is dimensionless. **§4 PASSES.**

### AR-5. Does the derivation *derive* the new form? — Yes

§2.1 transcribes the target from the arbitered `Σ̃^4D` **and says so**; §3.1 derives it
independently from `p_astro × S` applied once. The one step the package *supplies* — the
host-weight coupling — is genuinely absent from the record: `sbarphi_defect_location_20260827.md`
§4 writes the venue law **per host**, `p_venue(z,M) ∝ k̄_g w_pop S̄_φ · S_4D`, with no host
weight at all, so the `S̃_φ,g` cancellation never enters the record's field of view. **Credit
where due: §2.2(i)/§3.3 is a real, new, load-bearing derivation step.**

### AR-6. Blast radius — re-grepped independently, complete

`grep -rn` for `catalogue_selected_2d`, `_draw_2d_accepted_latents`,
`catalogue_selected_host_draw_weights`, `_draw_kernel_survival_redshifts` returns **exactly**
§9.2's list: `p3_2d_fleet.py:299,313`; `ca_rhs_scorer.py:168-192, 846, 875`;
`rhs_inflation_confirmation.py:101,136,260`; `wbhzero_probe.py`; `test_correspondence_1d.py`
(8 tests at `:1558`, `:1577`, `:1597`, `:1616`, `:1647`, `:1678`, `:1734`, `:1758` — counted).
One item not listed: `cb_null_pinning.py:41` references `_draw_kernel_survival_redshifts`'s
density in prose and explicitly "does not draw" — **not a consumer**, but worth a line so the
next grepper does not re-discover it. Every `correspondence_1d.py` line cited in §1, §2, §9
(`:359`, `:1330-1337`, `:1341-1385`, `:1440-1499`, `:1560-1563`, `:1605-1756`, `:1682`,
`:1687-1696`, `:1704-1719`, `:1733-1738`, `:2062`, `:2107`, `:2236`) checked and correct.

Two **exactness caveats** the package asserts past: (i) the `S̃_φ,g` cancellation is exact only
in the continuum — `kernel_smeared_survival` uses GL-50 nodes (`:1327-1337`), the z-draw a
uniform grid + trapezoid CDF (`:1491`, `:848-849`); (ii) the venue clips the latent mass at
`_M2D_MASS_FLOOR = 1.0 M_⊙` (`:1708`) while the model's closed form excludes the `P(M≤0)` mass
(`p3_2d_companion.py:941-943`). Neither is introduced by this fix. §2.2's "**exactly** `q_new`"
should read "exact up to the pre-existing quadrature and mass-floor conventions".

### AR-7. Literature — checked, and honest

`arXiv:1809.02063` is the correct identifier for Mandel, Farr & Gair (2019);
`arXiv:2308.02281` for Gray et al. (2023). `docs/LITERATURE_WARNINGS.md:229` row `MFG-a`
exists and is currently `UNCHECKED` — the package's proposal to move it to *qualified* (not
`CHECKED`) is consistent with the file's actual state, and the register was indeed not edited.
§6.2 states the absences plainly, including that Eqs. (5)–(7) are applied *by structural
analogy only*, that the rejection-sampling argument is attributable to no paper, and that the
quotes came through WebFetch rather than a raw-PDF cross-check. **No overclaim found.**
One nit: §6.2 item 5's "not a bare CHECKED" could be misread as implying `MFG-a` is CHECKED
today; it is `UNCHECKED`.

### AR-8. DEFECTS — seven, to be fixed before this goes to the author

**D1 — [HEADLINE] §0 contradicts itself on disjunct 2.** §0 asserts "two disjunctive repairs;
**both FAIL** as literally written", then concedes three bullets later that Option B "is the
one form algebraically exact as written". `C2_star_review.md:169-170`'s disjunct 2 reads
"drop the Bernoulli **in favor of an S̃-reweighting**" — and the record's own companion doc
operationalizes it as exactly Option B (`sbarphi_defect_location_20260827.md` §4). L5 refutes
"drop the Bernoulli, full stop", a reading **no document in the record proposes**. Honest
statement: disjunct 2 is *ambiguous* — a host-level `S̃`-reweighting fails on the M-conditional
(L5's real content), while the per-event `S_4D/S̄_φ` importance weight is exact and fails only
on interface grounds. **FIX-MISSPECIFIED stands on disjunct 1 alone**, which is enough. Rewrite
§0's headline and L5.

**D2 — [ARITHMETIC, propagates into a test] L1 and R5's `n_drawn_total == n` is FALSE.**
`batch = int(np.clip(_M2D_BATCH_MULTIPLIER*remaining, _M2D_MIN_BATCH, _M2D_MAX_BATCH))`
(`:1681`, with `4/64/4000` at `:1560-1562`) and `n_drawn_total += batch` (`:1730`). With
`S_4D ≡ 1`: one round, but `n_drawn_total = clip(4n, 64, 4000)` — 800 for `n = 200`, 2000 for
the `n = 500` R1 uses. **R5 as written fails on correct code, before and after the fix.**
Correct to `n_drawn_total == int(np.clip(4*n, 64, 4000))`. R5's second half — "`z_true`
bit-identical to a direct `_draw_kernel_survival_redshifts` call ... on the same seed" — is
also unimplementable as stated: the stream consumes `rng.choice` first (`:1682`), so a fresh
generator at the same seed is misaligned; the replay must reproduce the host draw first.

**D3 — [TEST ASSERTS SOMETHING FALSE] R6 is mis-specified.** R6 encodes L2 with
"`_FakeS4D(value=c)` constant" **only**. L2's equivalence needs `S̄_φ ≡ c` as well, and the
test doubles decouple the two: `_fake_phi_survival_table` is `exp(−3z)`
(`test_correspondence_1d.py:815-824`). With a constant `_FakeS4D` and that table, the fix
genuinely **does** change `z_true` (it drops `exp(−3z)` from the z-density and `S̃` from the
host weight), so R6's KS assertion fails on a correct implementation. R6 must pass a flat
survival table too — at which point it collapses into R5 and should be merged with it.

**D4 — [REASONING] §7's stated reason for the existing suite's blindness is wrong.** "several
use `_FakeS4D(value=0.9)` (constant, hence L2-blind: they will pass unchanged)". The
conclusion is right; the reason is wrong, for D3's reason — those tests are not in the L2
regime and their draws **do** change. Read this pass: `:1616` asserts frame equality across
two same-seed calls, `:1647` asserts two different seeds differ, `:1678` asserts column
presence and a 15σ `M_z_obs`↔`M_z_true` window, `:1734` asserts the zero-survival STOP.
**None pins a draw law.** The correct, and stronger, argument for R1–R3 is: *the existing
suite pins no draw law at all.*

**D5 — [TEST CONFOUNDED] R3's discriminator is confounded with acceptance.** R3 proposes to
check "the empirical host-index histogram against `w_g/Σw_g` (χ², 3σ)". The **accepted** host
marginal is `∝ w_g·A_g` with `A_g = ∫∫ k̄_g p_gal S_4D dM dz`, not `w_g`. With the "genuinely
`(d_L, M_z)`-dependent `_FakeS4D`" R1/R2 demand and a pool with "wide spread in `z_g`", `A_g`
varies strongly and the χ² fails on correct code. R3 is sound only with a **host-independent**
`S_4D` (constant `_FakeS4D` **plus** the steep `exp(−3z)` table, which still exposes the `S̃`
tilt — this is the right construction), or by comparing against `w_g·A_g`. As written it is
not the "single most important new test"; as corrected, it is.

**D6 — [IMPLEMENTATION HAZARD] §2.2(ii)'s "or by passing a flat `S̄_φ ≡ 1` table".** A flat
table handed to `draw_realization` also reaches `catalogue_selected_host_draw_weights` at
`:2107`, making `s_tilde_phi ≡ 1` and silently zeroing the banked `s_tilde_phi_host`
diagnostic column (`:2236`) — the very column this review used to execute R4, and the input
any future venue-drift audit needs. **Strike the flat-table alternative; mandate the keyword
flag on the `:1687-1696` call site.**

**D7 — [SCOPING OVER-CLAIM] §9.4 buys less than it says.**
`hier_blocker_a_generator_law_20260827.md:53-59` names `"catalogue_selected_2d"` as the
"**byte-identical-on-the-z-axis sibling**" of `"catalogue_selected"` and puts *both* modes in
the truth-θ = (0,1) class on the strength of the shared density. 2D-only scoping protects the
**conclusion** (dropping `S̄_φ` changes neither `b` nor `s`; the Gaussian's `loc`/`scale` are
untouched) but falsifies the stated **justification** for the 2D sibling. The [HIER] doc needs
a one-line amendment **regardless of scoping**. §9.4 presents 2D-scoping as fully protective;
it is not. (The trap itself — never edit `_draw_kernel_survival_redshifts` in place — is real
and correctly identified.)

### AR-9. What survives to the author unchanged

Items 1 (missing verbatim grant), 3 (run R4 first — now **done**, see AR-2), 5
(`validation/` not on the trigger list), 6 (the `S²` phrasing correction), 7 (13.5–16 % is not
an H₀ number; no production result changes) and 8 (GATE-ACC re-check — now measured at ~3 %,
AR-3 L9) are all **confirmed correct** by this pass. Item 2, the Option A′ ratification, is
**confirmed and strengthened**: not "the grant is under-specified" but "**~69–70 % of the
measured drift survives the grant as recorded**".

### AR-10. Reproducibility of AR-2

Estimator, over the 24 `bt_*` seeds of `p3_2d_fleet_20260825/`, `h = 0.73` rows of
`diagnostics/event_likelihoods.csv`, `w₂ = α_G_φ·L_cat_with_bh/(α_G_φ·L_cat_with_bh +
B_num_wbh)` on `L_cat_with_bh > 0`, `ω_id = 1−w₂`, `ω_BR = (1−w₂)/(1+(r₂−1)w₂)`,
`r₂ = 2.6124925`, `n = 200`:

    R(S) = [ Σ_live ω ] · [ Σ_all 1/S ] / ( n · Σ_live ω/S )

with `S = s_tilde_phi_host` (column 131 of `prepared_cramer_rao_bounds.csv`) for the
Option-A-literal residual, and `S = S̄_φ(z_true)` for the record's 0.86473/0.84024. Acceptance:
`1/mean(1/s4d_at_truth)`. No new compute; no cluster; no production code executed.
