# Fix B, path (A): one detection model — the φ-marginal survival S̄_φ

**Status: APPROVED and IMPLEMENTED (2026-08-04).** Author-approved
`/physics-change` package of record:
`.planning/derivation-2dbias-fix-20260803/FIXB_PATHA_PACKAGE.md` (§3 formulas,
§4 acceptance gates, §5 regression pins, §7 errata ledger, §8 author
decisions D1–D3). That package **supersedes**
`GATE_PACKAGE_FINAL.md` §2 (Fix B) and `FIXB_MEASUREMENT_REPORT.md` §5.1–§5.4;
those documents were deliberately left unedited as the adjudication record, and
every correction to them is carried in the package's §7 errata ledger. This
file is the in-repo entry point: what shipped, where it lives, and what is
pinned.

Gate-ledger rows: `docs/gates/PHYSICS-GATE-LEDGER.md` (2026-08-04,
`presented` → `implemented` → `verified`, target
`bayesian_statistics.py` selection stack + mixture).

Independence: **Fix A** (the C7 host-z kernel, `GATE_PACKAGE_FINAL.md` §1) is a
separate change on a separate branch and is **not** part of this commit; the
binding ship order keeps the two as independent commits.

---

## 1. The defect

The shipped `absolute_marginal` mixture mixed two *different* detection models:

| leg | detection model at HEAD `f53cc991` |
|---|---|
| `D(h)`, `β_Ḡ(h)`, `β_G = D − β_Ḡ` | the separately fitted **mass-blind** survival `S_3D` (`detection_probability_without_bh_mass_interpolated_zero_fill`) |
| `Σ⁴ᴰ(h)` (the with-BH catalogue sum) and the per-event 2D numerator | the **mass-aware** `S_4D(d_L, M_z)` at catalogue masses |

Nothing in the code enforced the *tower identity* that would make those two
mutually consistent,

```
S_3D(z;h)  ==  ∫ φ(log₁₀M) S_4D(d_L(z;h), M(1+z)) dlog₁₀M ,
```

and it fails: measured `r_φ(0.73) = 0.9119 ± 3e-7` on the production with-BH
object (band `[0.886, 0.912]`, i.e. 8.8–11.4% off unity), with **89–133 % of
the h-slope of `r(h)` being that mismatch rather than Malmquist physics**
(gate (ii-b), `fixb_measurements/GATE_IIB_RPHI.md`; production-convention
re-pin in `fixb_x15_attribution/CHORES_REPIN_ERACHECK.md`). The measure/kernel
attribution on the production `(T, F)` object is ≈ **92 / 8** — i.e. almost all
of the deviation is exactly what the fix removes by construction.

Violated principle: Mandel, Farr & Gair (2019) arXiv:1809.02063 Eqs. (5)–(7),
assumption A2 — the selection normalisation must use the *same* population
model and the *same* detection model as every numerator.

## 2. What shipped

**One detection model, by construction.** Define the φ-marginal survival as a
single contraction over the **production** with-BH object (the pooled-2D 40-bin
`S(d_L | M_z)` grid — `pdet_wbh_z_resolved = False` in every run of record,
cluster-verified 41/41; *not* the FIX-3 joint `z × M_z` grid):

```
S̄_φ(z;h) ≡ ∫ φ(log₁₀M) · S_4D(d_L(z;h), M(1+z)) dlog₁₀M
```

with φ the **generator's own** dark-host mass density
(`dark_siren_injection.dark_mass_log10_density_unnormalised`, the density
`_draw_dark_masses` samples — imported, never re-typed).

`S̄_φ` replaces the fitted `S_3D` in exactly three slots, and the 1D channel's
full-volume normalisation is re-derived in the same convention:

```
β_G^φ(h)  = ∫ f̄(z;h)     · S̄_φ(z;h) · p_pop(z;h) dz
β_Ḡ^φ(h) = ∫ (1−f̄(z;h)) · S̄_φ(z;h) · p_pop(z;h) dz
Σᶲ(h)     = Σ_g w_g · S̄_φ(z_g;h)                       (same catalogue as Σ⁴ᴰ)
D^φ(h)    = β_G^φ + β_Ḡ^φ

n̂_w^φ = Σᶲ/β_G^φ        r_Malm = Σ⁴ᴰ/Σᶲ
α_G^φ  = Σ⁴ᴰ/n̂_w^φ = β_G^φ·r_Malm
D̃^φ   = α_G^φ + β_Ḡ^φ      w̃_G = α_G^φ/D̃^φ

1D:  p_i = ( β_G^φ · L_cat,i^1D + B_num,i^φ ) / D̃^φ
2D:  p_i = ( α_G^φ · L_cat,i^2D + B_num,i^φ · g_i ) / D̃^φ
     B_num^φ = β_Ḡ^φ · L_comp  (= B_num · β_Ḡ^φ/β_Ḡ)
```

Then `r_φ ≡ 1` identically, `r = r_Malm` is a **pure Malmquist ratio**, and
F2/F3/F11 close. `n̂_w^φ` is mass-blind by construction, so it cannot inherit
the Malmquist bias.

**(N8), the C8 half (unchanged by path A, ships as written in
`GATE_PACKAGE_FINAL.md` §2.2):** the 2D completion leg gets its own numerator

```
B_num_wbh = ∫ (1−f_k(z)) p_gw(z) dVc/(1+z) · g_i(z;h) dz
g_i(z;h)  = ∫ dx_M N(x_M; μ_cond(z), σ_cond) φ_x(x_M; z)
φ_x(x_M;z) = φ(x_M·M_z,det,i/(1+z)) · M_z,det,i/(1+z)
```

with `g_i` **inside** the quadrature (μ_cond and the `(1+z)` mass lift both
depend on z, so the factor is not separable), `μ_cond`/`σ_cond` from the
`(d_L_frac, M_z_frac)` 2×2 block of `cov_4d` (Bishop 2006 PRML Eqs. 2.81–2.82).
The **1D `B_num` stays unmultiplied** — the 1D observable set is
`cov_obs = cov_4d[:3,:3]`, its M-integral collapses, and inserting `g` there
would be an MFG double count (gate (iv), PROVEN).

**Mass evaluation (author decision D3):** the **point** form is retained.
The smeared alternative is REFUTED as a remedy (the estimator already reads a
scattered catalogue, so smearing applies the observation noise twice,
`K_σ⊛K_σ = K_{√2σ}`; measured ×1.077 further inflation, gate (ii) −4.04 →
−4.99). The genuine internal-consistency counter-argument (the per-event
with-BH numerator and `D_g` both integrate a mass kernel while `Σ⁴ᴰ`
point-evaluates) is real, documented, adverse in direction, and would change
**only** `r_Malm → r_Malm·J_α` — a separate `/physics-change` with `σ_lnM` as a
declared physics input.

### Code map

| object | function | file |
|---|---|---|
| φ (the one definition) | `dark_mass_log10_density_unnormalised` | `dark_siren_injection.py` |
| φ normalised in log₁₀M / in M | `_phi_dark_mass_log10_grid`, `dark_mass_density_per_mass` | `bayesian_statistics.py` |
| `S̄_φ(z;h)` table | `precompute_phi_marginal_survival` | `bayesian_statistics.py` |
| `β_G^φ`, `β_Ḡ^φ` | `precompute_phi_selection_integrals` | `bayesian_statistics.py` |
| `Σᶲ(h)` | `precompute_global_catalog_selection(phi_survival_table=…)` | `bayesian_statistics.py` |
| `n̂_w^φ, r_Malm, α_G^φ, D^φ, D̃^φ, w̃_G` | `path_a_mixture_objects` | `bayesian_statistics.py` |
| `g_i(z;h)` ((N8)) | `completion_mass_factor_g` | `bayesian_statistics.py` |
| assembly (both channels) | `BayesianStatistics.p_Di` | `bayesian_statistics.py` |
| monitored gate (ii) under `S_and` | `rescore_class_share_joint_selection` | `bayesian_statistics.py` |

Quadrature conventions are those of the measured anchors: 600 log₁₀M nodes on
`[1e4, 1e7]` (the generator's own grid convention) and 1500 z nodes on
`(1e-6, z_max(h)]`, trapezoid in both.

**Scope guard (gate (iii-a)):** the φ-convention tables are **new** and are
consumed by `absolute_marginal` only. The legacy `D`/`β_Ḡ`/`β_G` tables and the
`generator_marginal` assembly are byte-identical, protecting the issue-#51
idealized-1D pin (issue #51 gate P5, reproduced to `rtol=1e-12` against a
committed values golden). *The md5 digest formerly cited here (`1e81ba22`) was
retired 2026-08-12: numpy SIMD dispatch made it host-dependent (max relative
deviation 3.73e-16 between AVX-512 and AVX2 runners) while the pinned physics
never moved.*

## 3. Dimensional analysis and limiting cases

`S̄_φ` is dimensionless (φ is a normalised density in log₁₀M and the integral
contracts it away). `β_G^φ, β_Ḡ^φ, D^φ` carry the units of `p_pop dz` —
identical to the old `D`/`β_Ḡ`. `Σᶲ` and `Σ⁴ᴰ` carry the units of the rate
weight `w_g`, so `r_Malm` and `r_φ` are dimensionless; `n̂_w^φ` carries
`[w_g]/[p_pop dz]`; `α_G^φ` and `D̃^φ` carry `D`'s units and `w̃_G ∈ (0,1)`.
**No object acquires or loses an `x_M` density:** `g_i` remains the only `x_M`
density and sits in the 2D completion numerator only, matching `mz_integral` in
the 2D catalogue leg — which is exactly why the 2D measure invariance survives.

| case | expectation | status |
|---|---|---|
| L1′ (f → 1) | 2D → `L_cat^2D` exactly; 1D → `L_cat^1D/r_Malm` — a *pure Malmquist* factor. Tilt over 0.60→0.86 at N=1588: **+19.03 nats (delivered) / −12.79 (truth)**, inside the pre-registered O(±30) band; the sign is a property of the catalogue convention | measured, §6 of the package |
| L2 (f → 0) | 2D → `L_comp·g_i`, dark-only, measure-covariant; the dark leg now uses `β_Ḡ^φ` — the same φ as `g_i` | ✓ |
| L3 (p_det → const c) | `S̄_φ ≡ c`, `Σᶲ = Σ⁴ᴰ = c·W_cat` ⇒ `r_Malm ≡ 1` **and** `r_φ ≡ 1` — an identity chain, no longer a sharp test | ✓ |
| L4 (s = 0) | the `g_i` tilt vanishes exactly (a flat-in-log φ cancels the z-dependent mass scale) | test |
| L5 (σ_Mz → 0) | `g_i` finite, non-zero, → the point evaluation | test |
| L6 (σ_z → 0) | orthogonal to path (A); the #51 P5 gate untouched | ✓ |
| tower limit | if the pool's a-stratum mass measure equalled φ, `S_3D → S̄_φ` and path (A) reduces to the shipped estimator; the measured +0.28 dex pool-vs-φ gap (KS D = 0.215) is why it does not | measured |

## 4. Pins (see `FIXB_PATHA_PACKAGE.md` §5 for the full tables)

Anchors at `h = 0.73`, measured on the staged catalogues of record and the
production pool, **reproduced by the shipped code to ≤ 4e-8**:

| object | value | convention |
|---|---|---|
| `β_G^φ` | 1.533228e8 | catalogue-free quadrature |
| `β_Ḡ^φ` | 8.884038e8 | catalogue-free quadrature |
| `D^φ` | 1.041727e9 | = sum of the two |
| `Σᶲ` | 9.562370e8 / 9.808671e8 | delivered / truth |
| `Σ⁴ᴰ` | 4.221903e8 / 3.754526e8 | delivered / truth |
| `r_Malm` | **0.4415122** / 0.3827762 | delivered (PRIMARY) / truth |
| `w̃_G` | **0.070802** / 0.061967 | delivered (PRIMARY) / truth |

**Convention (author decision D2).** The **delivered-catalogue** pins are
PRIMARY; the truth-convention values are recorded as secondary/informational.
*Documented promotion path:* promote the truth-convention values to primary
**once the truth-convention `Σ⁴ᴰ(h)` is measured at all 41 h on the
D1-remedied rerun** — today its h-curve between the 0.60/0.73/0.86 anchors is a
3-anchor quadratic model, the one modeled ingredient of the counterfactual.
The code always computes `Σᶲ` and `Σ⁴ᴰ` on **the catalogue the run loads**
(enforced by sharing `precompute_global_catalog_selection`), so the retired
mixed-catalogue pairing that produced `r_Malm = 0.4304` / `w̃_G = 0.069143` /
`z = −3.71` cannot recur.

Retired / re-defined by this change: `w_G = β_G/D = 0.1215039` is no longer the
operative weight (it survives as the RENAMED diagnostic `w_G_legacy`); the
`−80.30`-nat 1D tilt pin and "the 1D rails at 0.600 unconditionally" are
withdrawn; the 2D pre-registration `0.775 ± 0.010` is void. From the executed
recombination counterfactual: 1D MAP grid **0.610, 11/11 (delivered)** /
0.600, 9/10 (truth); 2D MAP **0.7909 ± 0.0119 (delivered)** / 0.7815 ± 0.0160
(truth), with the in-catalogue class still high (≈0.834) — **path (A) does not
close the class tension**, consistent with its home being D1 plus the C7 track.

Surviving quotable systematic (**T8**): `r_pool(z-conditional) = 0.9909` on the
production `(T,F)` object — the ~0.9 % kernel-pairing floor of the `S̄_φ`
construction. It does **not** close to 1 by construction and belongs in the
paper's systematics table.

## 5. Instrumentation shipped with the change

* Diagnostics CSV (`event_likelihoods.csv`) gains `w_G_legacy`, `w_tilde_G`,
  `alpha_G_phi`, `r_Malm`, `D_tilde_phi`, `B_num_wbh`, `g_frac`, written at
  7 significant figures. The legacy `w_G` column is kept (it now carries the
  operative weight) and the legacy value is *renamed*, never overwritten.
* Per-h log lines: `S_bar_phi(h=…)`, `phi-convention legs(h=…)`, `path-A(h=…)`
  (all mixture scalars at 7 s.f. plus `w_G_legacy`).
* **T9**: the `Σ⁴ᴰ` mass-band shares (in-band / above / below / clamped) are
  logged per h — the standing refutation of F8 (98.980 / 0.999 / ~0 / 0.020 %
  at 0.73).
* `g_i` support-exit warning (the φ support is the Babak band; 0/6360 cells hit
  it on the measurement of record).

## 6. Gate (ii) is a MONITORED CONSISTENCY NUMBER, not evidence

The fix ships **on correctness grounds**: tower identity by construction,
measure invariance at machine zero, absolute in-catalogue rate closed to
×1.07 ± 0.10 (0.7σ from unity) after the §1 attribution of the former
×1.50–1.69.

Gate (ii) (generator calibration) is **demoted**. Its re-scored value,
`z = −0.48` (−0.59 including candidate (a)'s class-resolved validity slice),
is conditional on **two** things that must always be quoted with it: the
generator-closure (truth) convention, and modelling the p0-window filter whose
very existence is the package's new defect finding (D1). Scored while ignoring
that filter the same statistic sits at −2.2σ (truth) to −4.0σ (delivered).
The paper must **not** cite gate (ii) as evidence for this fix.

Implementation of the monitoring half of author decision **D1 remedy (ii)**:
`rescore_class_share_joint_selection` rescales a predicted in-catalogue share
by the measured class-conditional retention ratio of the joint selection
`S_and = P(SNR ≥ 20 ∧ p0 ∈ W | d_L, M_z)`, `ρ = s_G/s_Ḡ = 0.7305 ± 0.4 %`.
Only the *ratio* enters, so the monitored number is available without
rebuilding the selection objects from `S_and` (which would re-pin R10–R12 and
require re-running the counterfactual engine). The instrument is pinned against
the measurement of record: `0.07280503 → 0.0542477`, `z = −0.48`.

## 7. Known open items (unchanged by this change)

1. **D1 remedy (i) — retire the stale `ParameterSpace.p0` bounds** (deferred to
   next-campaign prep, tracked in `TODO.md`): 69.3 % of SNR-passing campaign-#51
   events were silently rejected by the 5-point-stencil bound guard against the
   snapshot-era `p0 ∈ [10, 16]`, mass-dependently. It changes *simulation*, not
   inference, so it is not part of this commit. Because a mass band-pass is —
   via `M_z = M(1+z)` — a redshift-selection distortion at fixed source mass,
   **the p0 window belongs on the 2D-bias suspect list**.
2. **F10** (`φ_cat` in the catalogue kernel): UNPROVEN — measure or declare
   unbounded. Candidate (c)'s σ_lnM erratum (median 0.8614 = 0.3741 dex, not
   1.28) shrinks its expected size.
3. **F5/F12** generator_marginal attribution gap (delivered `generator_marginal`
   implying `W_cat = 1.4048e9`): OPEN, unaffected by path (A). The two α's are
   distinct estimands and the test suite documents the distinction (T10).
4. **s-sweep** ±0.036 in h: the quoted dark-class systematic; no published
   dark-siren analysis carries a compact-object mass observable in the
   catalogue/completion split, so the 2D channel is this project's extension.
5. The residual ×1.07 ± 0.10 carries candidate (b)'s +4.35 % forward-model
   closure error and a single catalogue realization; a second realization seed
   and the D1-remedied rerun are the two cheap falsifiers.

## References

* Mandel, Farr & Gair (2019), arXiv:1809.02063, Eqs. (5)–(7) — selection α must
  use the same population and detection model as every numerator (A2), applied
  to the hybrid population density of `GATE_PACKAGE_FINAL.md` Appendix A.
* Turski et al. (2023), arXiv:2302.12037, Eq. (8) — completion numerator and
  denominator carry the population mass/luminosity density.
* Gray et al. (2020), arXiv:1908.06050, Eq. (A.19) (and Eqs. 29, 33) —
  catalogue/completion partition structure.
* Babak et al. (2017), arXiv:1703.09722, Eqs. (5), (23), (31)×(34) — φ and
  `R_eff`.
* Bishop (2006), *PRML*, Eqs. 2.81–2.82 — the Gaussian conditional behind
  `μ_cond`/`σ_cond`.

**ERRATUM (2026-08-19):** §2's line `B_num^φ = β_Ḡ^φ·L_comp (= B_num·β_Ḡ^φ/β_Ḡ)`
is retracted — the transfer is un-derived and the factor is a defect (two
detection models in one term, MFG A2). Derived form: `B_num^φ = B_num`. See
docs/derivations/bscale_completion_normalization.md (ledger rows #130-#131);
production default is the derived form as of the [PHYSICS] commit implementing
it; the legacy factor remains available as `--completion_b_scale legacy` for
historical-run reproduction.
