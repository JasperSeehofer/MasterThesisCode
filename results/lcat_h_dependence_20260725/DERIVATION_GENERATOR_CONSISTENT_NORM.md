# Generator-consistent selection normalization (E1 FIX-3) — Physics Change Protocol packet

**Date:** 2026-07-26 · **Status:** DERIVATION ONLY — no implementation code written.
**Branch context:** `physics/absolute-mass-marginal` (V1 `absolute_marginal` mode landed, 49b9ade;
opt-in `--smear_global_selection`, f9c58f4).
**Scope:** the selection normalization of the per-event marginal
`p_i = (A_i + B_num)/D` in `master_thesis_code/bayesian_inference/bayesian_statistics.py` —
specifically the calibration constant `n̄_w = Σ_glob/β_G` and the master denominator
`D(h) = β_G + β_Ḡ`, both of which rest on the Option-A constant-comoving-density identity that
E1 measured to be violated by the real catalogue.
**Inputs:** E1 (`completion_bias/E1_COMPLETION_BIAS.md`), `DERIVATION_ESTIMATOR_REDESIGN.md` (V1),
`OVERNIGHT_REPORT_20260726.md`, and the generator code itself
(`dark_siren_injection.py`, `galaxy_catalogue/handler.py`, `main.py`).
**New numeric artifacts produced for this packet (all re-runnable, this directory):**
`generator_norm_Wcat.json`, `generator_norm_Vf_tables.json`, `generator_norm_Dgen_table.json`,
`generator_norm_composition_check.json`.

---

## 0. Executive summary

1. The exact per-event marginal under the true generator is derived in §2. After exact algebraic
   reduction it is a **two-substitution modification of the existing `absolute_marginal` mode**:

   ```
   p_i(h) = [ Σ_{g∈B_i} w_g N_g(h) / n̂_w(h)  +  B_num,i(h) ] / D_gen(h)

   n̂_w(h)  = W_cat / V_f(h)                      (replaces n̄_w = Σ_glob/β_G)
   D_gen(h) = Σ_glob(h)/n̂_w(h) + β_Ḡ(h)          (replaces D = β_G + β_Ḡ)
   ```

   with two new precomputes only: the **h-independent scalar** `W_cat = Σ_{g: z_g<1.5} w_g`
   (total draw-eligible rate weight of the pruned catalogue) and the table
   `V_f(h) = ∫₀^{1.5} f̄(z,h) (dV_c/dz)/(1+z) dz` (the completeness-weighted population volume —
   the same integral the generator's `F` uses). **`n̄_w` does not survive**: no model
   integral (`β_G`) is ever compared against a discrete catalogue sum anywhere in the estimator.
   The Option-A identity is not approximately enforced — it is never invoked.

2. **The generator's F is derived, not posited** (§2.2): the injection is a per-event
   `Bernoulli(F)` channel split (`draw_mixture_hosts`), so the mixture fraction is exactly
   `F(h) = V_f(h)/V_tot(h)` — the code's own `compute_global_catalog_fraction`. Measured on the
   frozen completeness cache: `F = 0.0175370…`, **exactly h-independent** (the pixel `m_th` map
   makes `f̄` h-invariant; `dlog V_f/dh = −3/h` to 4 significant figures — pure `h⁻³`). E1's
   `F_incat_population = 0.0175370676704831` is reproduced to machine precision — the same object.
   In the reduced form F, `W_dark`, `V_tot` all cancel; only `W_cat` and `V_f(h)` remain.

3. **Smeared vs point Σ_glob (§4.3): the generator identity puts the σ_z kernel on NEITHER side.**
   The mock's catalogue redshifts ARE the true redshifts (the draw copies the catalogue row;
   detection thresholds on SNR at `d_L(z_g; h_inj)`); no σ_z enters the generative selection.
   The internally consistent pairings are (point numerator kernel ↔ point `Σ_glob`) or
   (σ_z numerator kernel ↔ smeared `Σ_glob`); the current mixed state is the inconsistency.
   Under this packet the question loses most of its force: `n̂_w` contains **no P_det at all**,
   so the entire n̄_w kernel-asymmetry residual (the f9c58f4 motivation) is gone by construction;
   the kernel choice only touches `Σ_glob`'s role inside `D_gen` (measured size of the smear
   there: +0.067/h on a term with weight ≈ 0.19 ⇒ ≈ 0.013/h on `D_gen` — negligible).

4. **Limiting cases all pass** (§5), including: exact reduction to the current
   `absolute_marginal` when the catalogue does realize Option A; exact recovery of the Gray-A9
   full-catalogue form as f→1; continuous empty-ball fallback `B_num/D_gen`; and **exact
   algebraic identity with the current estimator in the `p_det → 1` limit** (seed600 prediction:
   unchanged, now as an identity rather than an approximation).

5. **⚠ Predicted effect with the measured numbers (§6): this fix does NOT de-rail the deep
   venue — and E1's FIX-3 estimate had the wrong arithmetic.** The derived, dimensionally
   consistent `D_gen` has `dlog D_gen/dh = −1.68` (shared-3D-survival convention) or `−1.49`
   (generator-exact 4D catalogue selection), NOT E1's `−1.02` (whose posited form
   `F·Σ_glob + (1−F)·β_Ḡ` is dimensionally inconsistent — it under-weights the catalogue share
   ~17× and omits the `h⁻³` volume carrier). Consequences at the measured slopes: catalogue-
   dominated events lose the spurious `n̄_w` tilt (−0.39/h, the intended fix — the composition
   term cancels by construction), but B-dominated events (~91% of the 3454) gain +0.16/h
   (3D convention), leaving the full-mixture 0.73→0.86 gap ≈ +92 ln (3D) or ≈ +52 ln (4D)
   versus the current +54.2. **The deep-venue HIGH rail is therefore NOT primarily
   composition-normalization-sourced.** The residual tilt budget points at the pooled-survival
   shape (E1 FIX-2: `dlog D_zres/dh = −0.56` vs pooled `−1.52`, s7 — if even half real, that is
   ~−1/h × 3454 events, an order of magnitude more than needed to explain the rail). This packet
   remains the correct normalization — it removes a proven inconsistency and eliminates a
   calibration channel — but the adoption rationale must be *correctness*, not *de-railing*.

6. **No free constants.** `W_cat` and `V_f(h)` are deterministic functions of the catalogue and
   the frozen completeness. Two discrete, fully specified author decisions remain (§7): the
   p_det convention inside `D_gen`'s catalogue term (shared-3D vs generator-exact-4D), and the
   σ_z pairing (point/point vs kernel/smeared).

---

## 1. Ground truth: what the generator actually does (code-anchored)

The mock universe IS the injection pipeline. Per event (`main.py:data_simulation`, refill at
`main.py:489–500`):

1. **Channel split**: in-catalogue with probability `F`, dark with `1−F`
   (`dark_siren_injection.draw_mixture_hosts:605–634`, `Bernoulli(F)` mask). `F` is precomputed
   ONCE at the injection cosmology (`main.py:426–428`) as

   ```
   F = ∫₀^{z_max} f̄(z,h) p_pop(z) dz / ∫₀^{z_max} p_pop(z) dz ,   p_pop ∝ (dV_c/dz)/(1+z),
   ```

   `z_max = HOST_DRAW_Z_MAX = 1.5` (`constants.py:99`), `f̄` = sky-averaged pixel completeness
   (`compute_global_catalog_fraction`, `dark_siren_injection.py:241–306` — uses `f_bar`, the
   SAME object `β_Ḡ` uses).

2. **In-catalogue channel**: draw galaxy `g` from the pruned catalogue restricted to
   `z_g < 1.5`, with probability `w_g / W_cat`, `w_g = R_eff_per_mbh(M_g)/(1+z_g)`
   (`handler.draw_rate_weighted_hosts:648–754`; weight identical to `_rate_weight`,
   `bayesian_statistics.py:535–558`). The host's true `(z, Ω, M)` are the **catalogue row values
   verbatim** — no snap, no scatter. `W_cat ≡ Σ_{g: z_g<1.5} w_g` is the draw's normalizer.

3. **Dark channel**: draw `(z, Ω)` jointly from `∝ (1−f_k(z)) p_pop(z)` per pixel
   (`_draw_dark_hosts_pixelated:373–426`, weights `W_k = ∫(1−f_k)p_pop dz`), masses from the
   population mass marginal. Normalizer `W_dark ≡ (1/N_pix)Σ_k ∫(1−f_k)p_pop dz
   = ∫(1−f̄)p_pop dz` (linearity in f).

4. **Detection**: deterministic threshold SNR ≥ 20 on the waveform computed at the TRUE drawn
   parameters (`main.py`, `constants.py:55`) — "oracle selection": detection is a function of the
   latents, marginal over the extrinsic nuisances it defines the per-host detection probability
   `P_det(d_L(z;h_inj), Ω, M)`.

5. **Observation**: the detected event's `d_L` (and sky, M_z) are Fisher-scattered into the
   prepared CSV; the catalogue z is NEVER scattered — the inference reads the same catalogue.

Three structural facts follow that the estimator must honor:

- **(G-i)** The catalogue/dark balance is `Bernoulli(F)` — NOT proportional to the channels'
  absolute rate weights. `W_cat` and `W_dark` enter only as per-channel draw normalizers.
- **(G-ii)** The catalogue channel's z-, sky-, and M-composition is the DISCRETE catalogue's own
  (w_g-weighted), not `f̄(z)p_pop(z)`. Any appearance of `β_G = ∫ f̄ p_det p_pop` as a stand-in
  for a catalogue sum is a model substitution the generator does not make (the Option-A step).
- **(G-iii)** The generative selection contains no σ_z kernel: hosts are drawn and detected at
  their point catalogue z.

---

## 2. The exact per-event marginal under the generator

### 2.1 Marginal and normalization

Under hypothesis `h`, the model universe is the generator's recipe evaluated at `h` (same frozen
catalogue, same frozen completeness, cosmology at `h`). For detected data `x_i`, with the MFG
convention (Mandel, Farr & Gair 2019, arXiv:1809.02063: `P(det|x)=1` for detected data, one
selection factor dividing the marginal — the pipeline's existing convention, kept; the oracle-
selection correction was measured second-order in E1 s6, +0.008):

```
p(x_i | h, det) = [  F(h)/W_cat  · Σ_{g∈cat} w_g N_g(h)
                   + (1−F(h))/W_dark(h) · B_num,i(h) ] / α(h)                     (1)

α(h) = F(h) · Σ_glob(h)/W_cat  +  (1−F(h)) · β_Ḡ(h)/W_dark(h)                    (2)
```

where every symbol is an existing pipeline object except `W_cat`, `W_dark`:

- `N_g(h)`: per-host GW-data density (unchanged; `single_host_likelihood`). The catalogue sum
  self-truncates to the candidate ball `B_i` (Gray 2020 A9 numerator behavior — unchanged).
- `B_num,i(h) = ∫ (1−f_k(z)) p_GW,iso(z;h) p_pop(z;h) dz` at the event pixel (unchanged,
  `p_Di`, `bayesian_statistics.py:2361–2413`): this is EXACTLY `∫∫ p(x|z,Ω) × [dark draw
  density] dz dΩ` for the pixelated dark draw (the code's own docstring at `:2386–2392` states
  the generator identity).
- `Σ_glob(h) = Σ_{g: z_g<z_max(h)} w_g P_det(d_L(z_g;h),…)`: the catalogue-channel detection
  expectation — the point-evaluated sum (`precompute_global_catalog_selection:998–1195`) is
  the generator-exact form by (G-iii).
- `β_Ḡ(h) = ∫ (1−f) P_det p_pop` (`precompute_missing_completion_denominator:762–892`): the
  dark-channel detection expectation, sky-aware — matches the pixelated dark draw exactly.

`α(h)` is a genuine probability (dimensionless): the probability that a generator draw at
hypothesis `h` is detected. Nothing in (1)–(2) invokes constant comoving density.

### 2.2 The principled F — derived, and its h-dependence

By (G-i), the mixing fraction is the generator's own
`F(h) = ∫f̄(z,h)p_pop / ∫p_pop = V_f(h)/V_tot(h)`. Its h-dependence enters only through
`f̄(z,h)` (the `h⁻³` of `p_pop` cancels in the ratio). **Measured on the frozen `m_th` cache**
(`generator_norm_Vf_tables.json`): `F = 0.017537` at every h in [0.60, 0.86] —
`f̄` is exactly h-invariant in this implementation (the `m_th` map is h-independent by
construction, `pixel_completeness.py:161` note), hence

```
V_f(h) = V_f(0.73)·(0.73/h)³ exactly,   V_f(0.73) = 2.3237e8 Mpc³ sr⁻¹  (dlogV_f/dh = −3/h)
```

and `F` is a constant. Cross-check: E1 s7's `F_incat_population = 0.0175370676704831` is
reproduced exactly — same integral, same cache. (If a future completeness model makes `f̄`
h-dependent, (1)–(2) already carry `F(h)`, `V_f(h)` correctly; implement `V_f` as a per-h
integral, not as the `h⁻³` shortcut.)

### 2.3 Exact reduction to implementable form

Two identities: `F = V_f/V_tot` and `1−F = W_dark/V_tot` (both immediate from the definitions,
`W_dark = V_tot − V_f`). Multiply numerator and denominator of (1) by `V_tot(h)`:

```
p_i(h) = [ Σ_{g∈B_i} w_g N_g(h) / n̂_w(h)  +  B_num,i(h) ] / D_gen(h)              (3)

n̂_w(h)  ≡ W_cat / V_f(h)          [generator draw-side rate-weight density]       (4)
D_gen(h) ≡ Σ_glob(h)/n̂_w(h) + β_Ḡ(h)                                              (5)
```

`V_tot`, `F`, `W_dark` cancel. Equation (3) is the current `absolute_marginal` assembly with
`n̄_w → n̂_w` and `D → D_gen`. **Where the current form deviates from the generator's α(h):**
exactly and only in the substitution `Σ_glob/n̂_w → β_G` (denominator) and
`n̂_w → n̄_w = Σ_glob/β_G` (numerator conversion) — i.e. in assuming
`Σ_glob(h) = n̂_w(h)·β_G(h)` (the Option-A identity, documented at
`precompute_completion_denominator:648–656` and `precompute_global_catalog_selection:1007–1017`).
Measured violation (§6.1): ×1.334 in value, −0.39/h in log-slope.

**What replaces n̄_w:** the calibration is now *draw-side* — total catalogue rate weight over
completeness-weighted population volume — with **no P_det realization inside the conversion**.
The discrete↔continuous comparison happens exactly once, in `n̂_w`, and it is the comparison the
generator itself makes when it allots the catalogue channel the fraction `F` of hosts across the
completeness volume `V_f`. The catalogue term is NOT `Σ_ball wN / Σ_glob-type sum` with no
conversion: a pure-sum form would make the catalogue term dimensionally incommensurable with
`B_num` (per-galaxy vs per-volume); the generator's own normalizers force (4).

---

## 3. Protocol items 1–4

### 3.1 Item 1 — OLD formula (exact, file:line, `bayesian_statistics.py` @ 49b9ade/f9c58f4)

Production mode `absolute_marginal` (`:1421`, assembly `:2435–2438`):

```
L_cat_no_bh   = Σ_{g∈B_i} w_g N_g / Σ_glob_no_bh(h)          (:2272–2289, weighted_sum branch)
L_cat_with_bh = Σ_{g∈B_i} w_g N_g^wbh / Σ_glob_wbh(h)
p_i = ( β_G(h) · L_cat + B_num,i(h) ) / D(h)                  (:2435–2438)
    ≡ ( Σ_ball w N / n̄_w(h) + B_num ) / D(h),   n̄_w = Σ_glob(h)/β_G(h)
β_G(h) = D(h) − β_Ḡ(h)                                        (:2238–2239 lookups; identity)
D(h)    : precompute_completion_denominator      (:609–759)   [unchanged by this proposal]
β_Ḡ(h)  : precompute_missing_completion_denominator (:762–892) [unchanged]
Σ_glob(h): precompute_global_catalog_selection   (:998–1195)  [unchanged; point form is primary]
```

Zero-host events: `p_i = B_num/D` (empty-sum guard; `p_D` #29 fallback).

### 3.2 Item 2 — NEW formula

Replace the event assembly (only) by (3)–(5):

```
p_i(h) = [ Σ_{g∈B_i} w_g N_g(h) / n̂_w(h) + B_num,i(h) ] / D_gen(h)

n̂_w(h)  = W_cat / V_f(h)
W_cat    = Σ_{g ∈ reduced catalogue, z_g < HOST_DRAW_Z_MAX} w_g          (scalar, h-independent)
V_f(h)   = ∫_{1e-6}^{HOST_DRAW_Z_MAX} f̄(z,h) (dV_c/dz)/(1+z) dz          (per-sr, table over h)
D_gen(h) = Σ_glob(h)/n̂_w(h) + β_Ḡ(h)
```

Applies to both posterior channels with the SAME `n̂_w` (the conversion is population-side,
channel-independent; the current code's per-channel `n̄_w_wbh = Σ_glob_wbh/β_G` is another
Option-A substitution and is likewise replaced — see §4.2). `N_g`, `w_g`, `B_num`, `β_Ḡ`,
`Σ_glob`, the volume_deconv kernel, the candidate-ball construction, the p_det estimator and its
hypothesis-frame query convention: all **unchanged**. `D(h)` and `β_G(h)` become production-dead
(retained for diagnostics/`catalog_only`).

Domain note: `W_cat` and `V_f` use the DRAW domain `z < HOST_DRAW_Z_MAX = 1.5`, NOT the
horizon `z_max(h)` — they normalize the draw, not the detection. (`Σ_glob`'s horizon eligibility
cap is inert as before: `P_det ≈ 0` beyond.) The issue-#30 `z_max_cap` depth-truncation logic,
if used, must cap `W_cat`'s and `V_f`'s domain together with the candidate window (same
principle as f29a5e7: numerator and denominator move together).

### 3.3 Item 3 — References

Repo-internal (the generator IS the reference; this is the anchor of the whole derivation):

- `master_thesis_code/dark_siren_injection.py` — `compute_global_catalog_fraction` (F; :241–306),
  `draw_mixture_hosts` (Bernoulli split; :556–634), `_draw_dark_hosts_pixelated`
  ((1−f_k)p_pop draw; :373–426).
- `master_thesis_code/galaxy_catalogue/handler.py` — `draw_rate_weighted_hosts`
  (`P(g) = w_g/W_cat`, z<z_max, values verbatim from rows; :648–754).
- `master_thesis_code/main.py:417–500` — per-event mixture refill at the injection cosmology.

External framework (equation numbers verified against ar5iv full texts in the 2026-07-25 session,
`DERIVATION_ESTIMATOR_REDESIGN.md` §3.2/§8; not re-fetched here):

- Mandel, Farr & Gair (2019), MNRAS 486:1086, arXiv:1809.02063 — selection-conditioned
  likelihood: one normalizing factor `α(h) = ∫_det p(x|h)dx`; `P(det|x)=1` in numerators.
  Eqs. (6) ff. This packet is the exact evaluation of their α for THIS generator.
- Chen, Fishbach & Holz (2018), arXiv:1712.06531, Eq. (15) — single selection normalization
  `β(H0)`; here generalized to the two-channel generator mixture.
- Fishbach et al. (2019), arXiv:1807.05667, Eqs. (3)–(5) — `f·p_cat + (1−f)·p_miss` normalized-
  prior presentation; our (1) is that structure with the generator's own (F, W_cat, W_dark)
  instead of the idealized (f̄, n̄) — the difference IS the fix.
- Gray et al. (2020), PRD 101:122001, arXiv:1908.06050 — β_G/β_Ḡ selection integrals (Eqs. 29,
  33; A-numbering caveat as documented in D2 §6). Retained for `β_Ḡ` and the framework; the
  catalogue-side β_G substitution is what this packet removes.
- Gray et al. (2023), arXiv:2308.02281, Eq. (2.4) — one selection normalization per event over
  the full LOS population prior (structural precedent for D_gen as the single divisor).

### 3.4 Item 4 — Dimensional analysis

| Quantity | Expression | Units |
|---|---|---|
| `w_g`, `W_cat` | `R_eff/(1+z)`, Σ over catalogue | `yr⁻¹` |
| `V_f(h)`, `V_tot`, `W_dark` | `∫ (·) dV_c/dz/(1+z) dz` (per sr, same measure as D) | `Mpc³ sr⁻¹` |
| `n̂_w = W_cat/V_f` | draw-side rate-weight density | `yr⁻¹ sr Mpc⁻³` — same as old `n̄_w` |
| `F = V_f/V_tot` | | 1 |
| `Σ_glob(h)` | `Σ w_g P_det` | `yr⁻¹` |
| `Σ_glob/n̂_w` | | `Mpc³ sr⁻¹` — commensurable with `β_Ḡ` ✓ |
| `D_gen` | | `Mpc³ sr⁻¹` — same as old `D` ✓ |
| `α = F·Σ_glob/W_cat + (1−F)·β_Ḡ/W_dark` | exact form (2) | 1 (a probability) ✓ |
| `A_i = Σ wN/n̂_w`, `B_num` | | `[X]⁻¹ Mpc³ sr⁻¹` each ✓ |
| `p_i` | | `[X]⁻¹` ✓ |

Contrast: E1's posited `D_gen = F·Σ_glob + (1−F)·β_Ḡ` adds `yr⁻¹` to `Mpc³ sr⁻¹` — dimensionally
inconsistent; its measured slope (−1.02) is an artifact of that form (§6.2).

`h`-scale bookkeeping (the §3.3-style check, now BY CONSTRUCTION): `Σ_ball wN`, `Σ_glob`,
`W_cat` carry no `h±³` (fixed galaxy set); `V_f, β_Ḡ, B_num` each carry exactly `h⁻³`.
In (3): catalogue term = `ΣwN·V_f/W_cat` → `h⁻³`; dark term `B_num` → `h⁻³`; `D_gen` → `h⁻³`.
One overall `h⁻³` cancels in `p_i`. There is **no discrete-vs-continuous `h³` matching left to
"realize"**: `n̂_w ∝ h³` exactly (measured: `dlog n̂_w/dh = +4.1097 = 3/h` at 0.73), versus the
old `n̄_w`'s measured `+3.7173` (point) / `+3.78` (smeared) against the required `+4.11` — that
−0.39/h realization residual (σ_z kernel ~20% + z-composition ~80%, per the overnight
measurements) is the term that now **cancels identically**.

---

## 4. Consistency requirements (protocol-adjacent)

### 4.1 Fallback pairing (issue #29)

Empty ball ⇒ `A_i = 0` (empty sum) ⇒ `p_i = B_num/D_gen`, continuously — same structural limit
as V1. Note this is a physics change for the 1992 fallback events (`D → D_gen`:
value ×1.049 at h=0.73, slope +0.16/h in the 3D convention) — full-mixture readout only, as E1
warns (fallback-only closure is not a theorem).

### 4.2 Channels (no-BH / with-BH)

One generator ⇒ one `n̂_w` and one `D_gen` for both posterior channels. The with-BH catalogue
term becomes `Σ_ball w_g N_g^wbh / n̂_w` — the current per-channel conversion
`n̄_w^wbh = Σ_glob_wbh/β_G` (value 2.025 at 0.73, extracted from the run logs:
`Σ_glob_wbh(0.73) = 2.8474e8`) is also an Option-A substitution and is removed. Which `Σ_glob`
enters `D_gen` is the §7 author decision (3D-shared vs 4D-exact); whichever is chosen, it is the
SAME `D_gen` for both channels (the current code shares `D` the same way). The with-BH
`M_z` point evaluation inside `Σ_glob_wbh` and the M̂-dimension incommensurability of
`(A^wbh + B_num)` are pre-existing and unchanged (issue #24; §8 risk 5).

### 4.3 σ_z smearing — which side carries the kernel

Generator identity (G-iii): the mock draws hosts at their catalogue z and detects at
`d_L(z_g; h)` — the generative selection is point-evaluated; the generative numerator is also
point (`p(x|z_g)` exactly). Therefore:

- **Fully generator-exact:** δ-kernel in `N_g` AND point `Σ_glob`. (The σ_z kernel in the
  numerator is a deliberate real-data-realism conservatism — a model widening relative to the
  mock's truth.)
- **Internally consistent kernel model:** if the σ_z kernel is retained in `N_g` (modeling
  "true z ~ p_g(z|ẑ_g)"), then the same model's selection is `E_g[∫P_det p_g(z)dz]` — the
  smeared `Σ_glob` (f9c58f4). Kernel-in-numerator + point-selection (today's default state with
  the flag off) is the one combination that is neither.

Verdict: the identity `Σ_cat w_g P_det ≈ n̄_w β_G` whose two sides the question referred to **no
longer exists** — `n̂_w` is kernel-free and P_det-free, so the smear question reduces to
`Σ_glob`'s role inside `D_gen`, where its measured effect is ≈ +0.067/h × weight 0.19 ≈
+0.013/h — negligible either way. Recommendation: adopt the point form as primary (generator-
exact, matches (G-iii)); keep `--smear_global_selection` as the documented companion of the
numerator-kernel model. Author decision (discrete, both fully specified — §7).

### 4.4 Surviving weights

No `w_G = β_G/D`-type weight survives in the estimator. The diagnostic membership weight
becomes `P̂(cat|det, h) = (Σ_glob/n̂_w)/D_gen` (replaces the `w_G` diagnostic column);
the per-event posterior membership `λ_i = A_i/(A_i+B_num)` diagnostic is unchanged in form.

---

## 5. Protocol item 5 — Limiting cases

**(a) Catalogue that DOES realize constant comoving density (Option A holds).** If the catalogue
is a realization of a constant rate-weight density `n` over the completeness volume, then
`W_cat = n·V_f(h)·(sr convention)` and `Σ_glob = n·β_G(h)` (the discrete sums are Monte-Carlo
realizations of the model integrals). Then `n̂_w = W_cat/V_f = n = Σ_glob/β_G = n̄_w` and
`D_gen = Σ_glob/n + β_Ḡ = β_G + β_Ḡ = D`: **(3) reduces exactly to the current
`absolute_marginal`.** The current form is the Option-A special case of this packet.

**(b) f → 1 everywhere.** `V_f → V_tot`, `B_num → 0`, `β_Ḡ → 0`:
`p_i → [ΣwN/n̂_w]/[Σ_glob/n̂_w] = Σ_ball wN / Σ_glob(h)` — the Gray 2020 A9 / Gair 2023
full-catalogue ratio-of-sums, identical to V1's limit (b). ✓

**(c) Empty-ball continuity.** `A_i → 0` through the same arithmetic; `p_i → B_num/D_gen` with
no branch. Removing the last impostor from a 1-galaxy ball changes `p_i` by relative `A/B`
(≲10⁻⁴ on the deep venue, now ×1.334 larger than V1's table T1 values — still ≲10⁻³). ✓

**(d) Shallow venue `p_det ≈ 1` (seed600 must-not-change gate).** With `P_det ≡ 1` over the
catalogue and horizon beyond the draw depth: `Σ_glob → W_cat` (every draw-eligible galaxy sums
at weight w_g), hence `Σ_glob/n̂_w → V_f(h) ≡ β_G(h)|_{p_det=1}` and
`D_gen → V_f + β_Ḡ|_{p=1} = D|_{p=1}`; likewise `n̂_w → W_cat/V_f = Σ_glob/β_G = n̄_w`. The NEW
and CURRENT estimators are **algebraically identical** in this limit — not merely
shape-equivalent. Prediction for the seed600 A/B gate: MAP unchanged within tolerance, with
deviations only from the `p_det < 1` tail of the seed600 catalogue support (same gate,
sharper prediction than V1's case (c)).

**(e) h³ / composition-cancellation check (§3.3 style).** Done in §3.4: `dlog n̂_w/dh = 3/h`
exactly (measured `+4.1097` vs `3/0.73 = +4.1096`); no realization residual exists because no
P_det realization enters the conversion. The composition term — the difference between the
discrete catalogue's detected z-composition and `f̄ p_pop p_det` — now appears in numerator
(`Σ_ball wN`, per event) and denominator (`Σ_glob`, globally) as THE SAME kind of object
(point-evaluated rate-weighted catalogue sums), never converted through a model integral. ✓

**(f) Uninformative event (V1 case d).** `Σ wN → c·Σ_ball w` (h-const), `A_i ∝ c·h⁻³` via
`1/n̂_w`; `B_num ∝ c·h⁻³`; `D_gen ∝ h⁻³ × shape`. The `h⁻³` cancels; the only residual shape is
the shared selection factor `1/(D_gen·h³)` — MFG-correct, event-independent. ✓

---

## 6. Predicted effects with the measured numbers (no tuning anywhere)

All numbers: seed1000 EXP-40 deep venue, 41-h grid; tables in
`generator_norm_Dgen_table.json`; production `Σ_glob`, `D`, `β_Ḡ` from the shipped run logs
(`global_sums.json`, `s1_results.json`); `W_cat`, `V_f` computed this session from the same
catalogue file and frozen `m_th` cache the run used.

### 6.1 Replication anchors (validation of the inputs)

- Streamed catalogue replication of the production eligibility: **9,060,008 galaxies at
  z < 0.992 — exact match** to the run log's eligible count; isotropic-survival
  `Σ_glob(0.73) = 5.1224e8` vs production sky-aware `5.1221e8` (agreement 6e-5 — sky banding
  negligible, consistent with E1 §2.4). ⇒ `W_cat = 6.3477e8 yr⁻¹` is the correct companion
  normalizer (same rows, same weights).
- `V_f(0.73) = 2.3237e8`, `F = 0.017537` = E1's value to machine precision.

### 6.2 The measured Option-A violation, restated in this packet's terms

```
n̄_w(0.73) = Σ_glob/β_G = 3.6429        dlog n̄_w/dh = +3.7173  (smeared: +3.78)
n̂_w(0.73) = W_cat/V_f  = 2.7317        dlog n̂_w/dh = +4.1097  (≡ 3/h, exact)
ratio n̄_w/n̂_w = 1.3336                 slope residual removed: −0.392/h
```

The catalogue's detected rate weight exceeds the Option-A prediction by 33% in value (local
overdensity / low-z concentration of GLADE+) and falls short of the pure-h³ slope by 0.39/h
(detected-composition h-shape). Both inconsistencies vanish identically under (3)–(5).

```
D_gen(0.73) = 1.0031e9 = 1.049·D        dlog D_gen/dh = −1.6772   [3D-shared convention]
P̂(cat|det)  = 0.187  (current w_G = 0.147)
4D-exact variant (Σ_glob_wbh in D_gen): D_gen4 slope ≈ −1.49, P̂ = 0.113
```

**E1's FIX-3 estimate is superseded**: `dlog D_gen/dh ≈ −1.02` came from the dimensionally
inconsistent `F·Σ_glob + (1−F)·β_Ḡ` (implicit catalogue share 0.011 instead of 0.187, no `h⁻³`
carrier on the catalogue term, giving the catalogue term slope +0.36 instead of −3.75). The
derived form's direction of effect on the two channels is the OPPOSITE of E1's prediction:
fallback slopes move UP (+0.16/h), catalogue-dominated slopes move DOWN (−0.23/h net).

### 6.3 Per-event and ensemble predictions (3D-shared convention)

Slope changes at truth, from the measured tables:

| Event class | Δ(dln p_i/dh) | mechanism |
|---|---|---|
| A-dominated (catalogue-term-led; ≈22% of 1461 host-found ≈ 320 events) | **−0.233/h** | −0.392 (n̂_w replaces n̄_w) + 0.159 (D→D_gen) |
| B-dominated host-found (≈78% of host-found) and all 1992 fallback | **+0.159/h** | D→D_gen only |

- Host-found mean slope: `+0.74 → ≈ +0.81` (0.78·0.16 + 0.22·(−0.23) = +0.07). The *shared
  spurious A_i tilt* is removed (that population's A-term slope drops by 0.39), but most
  host-found events are completion-dominated and instead inherit the steeper `D_gen`.
- Fallback-only subset statistic: tilt +0.16/h × 1992 events ⇒ the 0.612 peak moves UP
  (magnitude O(+0.01…0.03) given s1's curvature; direction opposite to E1's FIX-3 note).
- **Full mixture (the only valid readout):** per-event ln-shifts 0.73→0.86 from the exact
  tables: B-dominated +0.0157 each, A-dominated −0.0276 each ⇒
  Δ(gap) ≈ 3133·0.0157 − 321·0.0276 ≈ **+40 ln** ⇒ predicted 1D gap ≈ **+94 ln (from +54.2)**
  — the HIGH rail persists and mildly steepens. Under the 4D-exact variant the gap is
  ≈ unchanged (**≈ +52 ln**). *(Linear extrapolation of measured per-event slopes; the probe
  re-evaluation is the gate.)*

### 6.4 What this implies for the rail attribution (honest reading)

If (3) is the exact generator marginal and the events are generator-drawn, the full-mixture
score at truth would be zero-mean under a correctly specified `N_g`/`B_num`/`P_det`. The
measured tilt that REMAINS after this fix (~+2000/h summed, per the slope bookkeeping above) is
therefore **not composition-normalization-sourced**; it must live in the remaining model
substitutions. The budget singles out the pooled-survival shape (E1 FIX-2 / M-D): s7 measured
`dlog D_zres/dh = −0.56` vs pooled `−1.52` at 0.05-z-bin resolution — a −0.96/h *per-event,
all-events* shared shift if confirmed (≈ −3300/h total, more than the entire rail tilt), with
E1's conservative estimate (−0.05 peak on conditioned statistics) already same-signed. The
FIX-2 derivation (z-resolved survival in `Σ_glob`, `β_Ḡ`, and the p_det grids) is the successor
workstream this packet hands off to; its own `D_zres` resolution issues must be fixed first
(s7 flagged the 0.05-z binning as kinky).

### 6.5 Composition absorption check (item-6 "×8.4" question)

`generator_norm_composition_check.json`: the generator-mixture detected-z density
`F·(dΣ_glob/dz)/W_cat + (1−F)·(1−f̄)p_pop·p_det/W_dark` predicts a low-z excess over the
Option-A density `p_pop·p_det` of **×6.4 (z<0.05), ×3.8 (0.05–0.10), ×2.0 (0.10–0.15)** —
i.e. E1's ×8.4 / ×2.3 measurement on the real events is *predicted* (to ~30%) by the catalogue
channel's low-z rate-weight concentration entering at fraction F, and is absorbed into `α(h)`
via `Σ_glob` by construction. The residual (8.4 vs 6.4; and the 0.10–0.25 deficit region, which
overlaps E1's p_surv-vs-p_true mid-z misfit) is survival-shape (FIX-2) and shot-noise
territory, not normalization. Caveat: the injection-CSV pool in `data/injections` is the
p_det-estimation pool drawn from the SMOOTH population model (`main.py:867`,
`sample_emri_events`) — it correctly matches Option A (measured obs/A = 0.7–1.5 mid-z) and
must NOT be used as the mixture-composition reference; the mixture prediction applies to the
`draw_mixture_hosts` event stream (the 3454 evaluated events).

---

## 7. Free constants and author decisions

**Free (tunable) constants: none.** `W_cat` is a deterministic catalogue sum; `V_f(h)` a
deterministic completeness integral; both are generator-determined. `emri_rate.C_NORM` cancels
(appears in `Σ wN`, `W_cat`, `Σ_glob` homogeneously).

Two discrete author decisions (both fully specified, neither a fit):

1. **p_det convention in `D_gen`'s catalogue term.**
   (i) *Shared-3D* (recommended initially): the same pooled 3D survival as `β_Ḡ`/`D` — keeps ONE
   p_det object across all selection integrals (the code's own guardrail at
   `precompute_global_catalog_selection:1157–1165`); P̂ = 0.187, `dlogD_gen/dh = −1.68`.
   (ii) *Generator-exact 4D*: `Σ_glob_wbh` (each galaxy's actual `M_z`) — exact per (G-ii) but
   splits the p_det convention between catalogue (4D) and dark (3D pool-marginal) terms; the
   measured 4D/3D value ratio 0.556 is entangled with the pooled-survival estimator bias (M-D),
   so adopting (ii) before FIX-2 unifies the estimator would import that bias into the channel
   balance. Slopes barely differ (+0.369 vs +0.357); the choice moves P̂ (0.187→0.113) and the
   predicted gap (+94→+52).
2. **σ_z pairing** (§4.3): point/point (generator-exact; recommended) vs kernel/smeared
   (real-data-realism). Measured consequence inside `D_gen`: ≲0.013/h — immaterial; decide for
   documentation coherence, not effect.

---

## 8. Honest risks and open questions (ranked)

1. **[HIGH — expectation management] This packet does not fix the HIGH rail** (§6.3–6.4): the
   predicted full-mixture gap is +94 (3D) / +52 (4D) vs +54.2 now. Adopt it as a correctness
   fix that eliminates the Option-A calibration channel and by doing so *localizes* the residual
   tilt in the survival model (FIX-2). If the author's priority is de-railing, FIX-2 must ride
   along or ahead.
2. **[HIGH] Pooled-survival mis-specification is now the load-bearing residual.** `Σ_glob`,
   `β_Ḡ` both evaluate the same pooled `S(d_L)`; its z-shape error (p_surv vs p_true: 0.49 vs
   0.35 at z=0.31) biases `D_gen` exactly as it biased `D`. The −0.56-vs-−1.52 `D_zres` slope
   measurement, if it survives a resolution study, dwarfs everything in this packet.
3. **[MEDIUM] Mass composition (the second Option-A flavor).** The catalogue's actual `M_z`
   composition detects at 0.556× the pooled-survival prediction (Σ_glob_wbh/Σ_glob at 0.73,
   from the run logs) — a large VALUE effect on the channel balance with negligible slope
   effect. Decision 7.1 defers it; a generator-exact treatment needs FIX-2-style conditioning
   in the M dimension too.
4. **[MEDIUM] No truth-tagged validation of P̂(cat|det) yet.** The injection pool CSVs carry no
   `catalog_index`, so the predicted detected-catalogue share (0.187 or 0.113) is unvalidated
   against the generator's own tally. Cheap gate: add the tag to the injection campaign (or a
   one-off `draw_mixture_hosts` + survival MC) and compare. E1's related observables
   (fallback fraction 0.577 + 27% impostor-membership ≈ 0.85 dark-ish) bracket but do not pin it.
5. **[LOW-MEDIUM] with-BH channel:** `M_z` point evaluation in `Σ_glob_wbh` (no galaxy
   mass-error kernel; issue #24) and the pre-existing M̂-dimension incommensurability of
   `A^wbh + B_num` are unchanged. The `n̂_w` unification (§4.2) changes the with-BH catalogue
   term's VALUE by ×(2.025/2.732) = ×0.74; with-BH posterior must be re-gated separately.
6. **[LOW] Sky structure:** `n̂_w` is a global scalar — exact because the Bernoulli split is
   global; all sky structure stays in `B_num`/`β_Ḡ`/per-galaxy `Σ_glob` terms, matching the
   pixelated generator. No per-pixel `n̂_w` is derivable or needed. The z<0.05 residual
   (×8.4 measured vs ×6.4 predicted) is partially exposed, not hidden: after adoption it becomes
   a pure survival/shot-noise diagnostic.
7. **[LOW] `HOST_DRAW_Z_MAX` coupling:** `W_cat`, `V_f` hard-inherit the draw depth 1.5. Any
   future change to the draw depth or to the catalogue prune (`M ∈ [10^4.5, 10^6]`,
   `Model1CrossCheck` bounds — note: NOT the ParameterSpace defaults [1e4, 1e7]; this session's
   first W_cat attempt failed exactly on that distinction, §6.1 anchors catch it) must
   recompute both. Guard with the eligible-count anchor (9,060,008 at z<0.992-style log line).

## 9. Validation gates (pre-registered, mirroring V1's §6)

1. **seed600 shallow A/B:** predicted algebraically unchanged in the `p_det→1` limit (§5d);
   tolerance |ΔMAP| ≤ 1 grid step and within σ_boot.
2. **Deep-venue full-mixture re-evaluation (seed1000, 7-pt probe first):** pre-registered
   prediction §6.3 (gap ≈ +94 (3D) / +52 (4D)); mechanism metrics: A-dominated per-event slope
   change −0.23±0.05, fallback +0.16±0.03, `P̂(cat|det)` diagnostic in [0.10, 0.19].
   A measured gap far BELOW the prediction falsifies the §6.4 attribution and would re-open
   FIX-3 as a rail candidate.
3. **Truth-tag MC** (risk 4): generator-side `P(cat|det)` vs `a/D_gen` at h_inj.
4. **P–P harness:** `mixture_mode="absolute"` extension with the harness's own generative
   `W_cat`, `V_f` (known exactly there) — isolates the n̂_w-estimation question from the model;
   requires the multi-galaxy-catalogue upgrade noted in the overnight report.

## 10. Artifacts

- `generator_norm_Wcat.json` — `W_cat = 6.3477e8` (prune-replication: 9,060,008 @ z<0.992 ✓)
- `generator_norm_Vf_tables.json` — `V_f`, `V_tot`, `F(h)` (F h-independent, = E1 value)
- `generator_norm_Dgen_table.json` — full 41-h `n̂_w`, `a = Σ_glob/n̂_w`, `D_gen`, `P̂`, slopes
- `generator_norm_composition_check.json` — observed vs Option-A vs mixture detected-z
  composition (pool caveat §6.5)
- with-BH `Σ_glob_wbh(h)` extracted from `run_20260719_seed1000_exp40/logs/evaluate_*.err.gz`
  (2.8474e8 at 0.73; slope +0.369)
