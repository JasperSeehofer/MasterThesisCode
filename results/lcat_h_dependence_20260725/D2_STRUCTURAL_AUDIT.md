# D2 — Structural audit of the in-catalogue likelihood L_cat: spurious h-dependence vs Gray et al. (2020) / gwcosmo / Gair et al. (2023)

**Date:** 2026-07-25
**Scope:** Derivation-level comparison of the production `volume_deconv` L_cat
(`master_thesis_code/bayesian_inference/bayesian_statistics.py`) against the reference
formulations: Gray et al. (2020) arXiv:1908.06050 Appendix A; Gray et al. (2023)
arXiv:2308.02281 (gwcosmo v2); Gair et al. (2023) arXiv:2212.08694 ("Hitchhiker's guide").
Companion to the D1 empirical factor decomposition (not duplicated here).

**Source-verification note.** All Gray-2020 and Gair-2023 equations below were read from
the **arXiv TeX source tarballs** (`arxiv.org/e-print/1908.06050`, `.../2212.08694`), not
from prose or abstracts. Gray 2020's appendix equations carry symbolic labels in the TeX
(`Eq:p(x|G,D,H0)`, `Eq:G_DH0_end`, ...); I number them here by **counting equation
environments in the appendix TeX in order** (the count gives the ratio-of-sums-with-p(z_i)
equation as the **9th** appendix equation, i.e. A9 in the published PRD 101, 122001
numbering, and the start of the p(G|D_GW,H0) derivation as A10). The published journal
numbering could be offset by one; the *content* quoted is verbatim. gwcosmo v2 equations
(2.4, 2.9, 2.10, 2.14–2.15) were checked via the ar5iv full text of 2308.02281; I did not
hand-verify the gwcosmo *code* (git.ligo.org was not fetched), so statements about the
production gwcosmo implementation rest on the v2 paper's equations. Flagged again in §6.

---

## 1. The pipeline's L_cat as implemented (production `volume_deconv` + local ratio)

Notation: event *i* has Fisher-Gaussian data summary
(φ̂, θ̂, d̂_L, σ_dL, [M̂_z, σ_Mz]); candidate host g has (φ_g, θ_g, ẑ_g, σ_z,g, [M_g, σ_M,g]).
`h` is the hypothesis. All code refs are to `bayesian_statistics.py` @ main (post-bd9081e).

### 1.1 Candidate set (data-conditioned, h-independent)

`p_D` (`:1849-1872`): the ball-tree returns hosts inside

- sky: 1.5σ ellipse around (φ̂, θ̂) (COORD-04 2×2 sky Fisher),
- redshift: z ∈ [z(d̂_L − 3σ_dL; h_min), min(z(d̂_L + 3σ_dL; h_max), `redshift_upper_limit`)]
  (`get_redshift_outer_bounds`, spans the **full h prior range**, so the set does not change
  with h during evaluation),
- BH mass window (1.5σ in M_z) for the with-BH-mass subset.

Call this set **B_i** (the "ball"). Note the high-z side is capped at the analysis depth
(`redshift_upper_limit`, deep venue 1.5).

### 1.2 Per-host kernel and integrals (`single_host_likelihood`, `:2463-2869`)

Effective z-uncertainty: σ_eff² = σ_z,g² + [(1+ẑ_g)σ_v/c]² (issue #16, `:2535`).
Host window: W_g = [max(ẑ_g − 4σ_eff, 10⁻⁶), ẑ_g + 4σ_eff].

**Host-z kernel (volume_deconv, `:2583-2606`):**

    p_g(z) = N(z; ẑ_g, σ_eff) · w_pop(z) / Z_g ,   w_pop(z) = (dV_c/dz)(z;h) / (1+z)
    Z_g    = ∫_{W_g} N(z; ẑ_g, σ_eff) w_pop(z) dz    (fixed_quad n=50)

**h-dependence of the kernel: exactly none.** `comoving_volume_element`
(`physical_relations.py:571`) is d_com²·c/H(z) with d_com = (c/H₀)(1+z)⁻¹(1+z)∫dz'/E(z'),
so for flat ΛCDM at fixed Ω_m,

    dV_c/dz (z; h) = h⁻³ · f(z; Ω_m)          (exact separation)

and the h⁻³ cancels between w_pop and Z_g. p_g(z) is **h-independent**; it differs from the
bare Gaussian only by the h-independent reshaping f(z)/(1+z) (mass pushed to higher z for
wide photo-z kernels). This matches Gair et al. (2023) Eq. (12)–(13) [labels
`eq:redpos`/`eq:redlikeli`]: p_red(z|ẑ) ∝ L_red(ẑ|z)·p_bg(z)/Z with p_bg uniform in
comoving volume — Gair explicitly defends this against the "double counting" worry.
(gwcosmo v2 Eq. (2.9) uses the *bare* Gaussian; the two references disagree with each
other here, see §3 row 4.)

**Per-host numerator (no-BH-mass channel, `:2611-2662`):**

    N_g(h) = ∫_{Z_num(h)} 𝒩₃(φ_g, θ_g, d_L(z;h)/d̂_L ; μ₃, Σ₃) · p_g(z) dz

with Z_num(h) = [z(d̂_L − 4σ_dL; h), z(d̂_L + 4σ_dL; h)] (h-dependent window, fixed_quad
n=50). h enters through d_L(z;h) = (c/h)·g(z;Ω_m)·(a pure 1/h scale) and through the window.
p_det is (correctly) absent from the numerator (MFG 2019, arXiv:1809.02063).

**Per-host selection denominator (`:2631-2671`):**

    D_g(h) = ∫_{W_g} P_det(d_L(z;h), φ_g, θ_g) · p_g(z) dz     ∈ [0,1]

h enters only through d_L(z;h) inside P_det (and the F4-v2 local-linear estimator's grid
lookup at that h). W_g is h-independent and **not** capped at z_max(h).

(The with-BH-mass channel multiplies an M_z-marginal Gaussian factor into N_g and an
erf-sum inner M-integral into D_g; same structure in h.)

### 1.3 Event-level assembly (`p_Di`, `:2029-2135`; precomputes `:609-889`)

Rate weights w_g = R_eff(M_g)/(1+ẑ_g) (h-independent, `_rate_weight` `:535`).

    L_cat(h) = Σ_{g∈B_i} w_g N_g(h)  /  Σ_{g∈B_i} w_g D_g(h)        ["local_ratio", :2122-2127]

    p_i(h)   = [ β_G(h) · L_cat(h) + B_num(h) ] / D(h)              [:2051-2059]

with the precomputed, event-independent volume integrals (all fixed_quad n=64, all
domain-capped at z ∈ [z_min, min(z_max^pdet(h), max_redshift)]):

    D(h)      = ∫ ⟨P_det(d_L(z;h),Ω)⟩_sky · (dV_c/dz)/(1+z) dz      [full-volume selection]
    β_Ḡ(h)   = ∫ (1−f(z)) ⟨P_det⟩ (dV_c/dz)/(1+z) dz               [missing part]
    β_G(h)    = D(h) − β_Ḡ(h)
    B_num(h)  = ∫ (1−f_k(z)) p_GW,iso(d_L(z;h)/d̂_L) (dV_c/dz)/(1+z) dz   [completion numerator]

**h-dependence ledger of p_i(h):**

| factor | h enters through | pure-scale part | shape part |
|---|---|---|---|
| N_g | d_L(z;h) in GW Gaussian; window Z_num(h) | none | **physical H0 signal** |
| p_g(z) kernel | dV_c/dz ∝ h⁻³ | cancels exactly in Z_g | none |
| D_g | P_det(d_L(z;h)) | none | **monotone ↑ in h** (larger h → smaller d_L → higher p_det) |
| Σ_local w_g D_g | sum of the above over B_i only | — | **monotone ↑ in h**, steep at depth |
| β_G, β_Ḡ, D, B_num | h⁻³ from dV_c/dz + P_det(d_L(z;h)) shape + z_max(h) cap | h⁻³ cancels in every ratio (w_G, B_num/D) | shared, domain-matched |

So the only *candidate-set-dependent* h-shape in p_i sits in Σ_local w_g D_g — the
denominator of L_cat.

---

## 2. The reference formulations

### 2.1 Gray et al. (2020), arXiv:1908.06050, Appendix (TeX verbatim)

In-catalogue likelihood with galaxy redshift uncertainties (9th appendix equation; TeX
label `Eq:p(x|G,D,H0)`; ≙ **A9**):

    p(x_GW | G, D_GW, s, H0)
      =  Σ_{i=1}^{N_gal} ∫ p(x_GW|z_i,Ω_i,s,H0) p(s|z_i) p(s|M(z_i,m_i,H0)) p(z_i) dz_i
        ───────────────────────────────────────────────────────────────────────────────
         Σ_{i=1}^{N_gal} ∫ p(D_GW|z_i,Ω_i,s,H0) p(s|z_i) p(s|M(z_i,m_i,H0)) p(z_i) dz_i

Three structural facts:

1. **Both sums run over the FULL catalog** (N_gal), not an event-localized subset. The
   numerator self-truncates because p(x_GW|z_i,Ω_i) ≈ 0 away from the event; the
   **denominator does not self-truncate** — every detectable catalog galaxy contributes
   p(D_GW|z_i,Ω_i,H0).
2. **p(z_i) is the galaxy's redshift-uncertainty pdf.** No dV_c/dz and no 1/(1+z) is
   applied to catalogued galaxies. The population prior p(z) ∝ comoving volume-time
   appears only in p(G|D_GW,H0) (10th–12th appendix eqs., `Eq:G_DH0_start`→`Eq:G_DH0_end`)
   and in the out-of-catalogue term (`Eq:p(x|barG,D,H0)`), whose z-integrals run from the
   magnitude-threshold boundary z(M, m_th, H0) to ∞. p(s|z_i) is the *rate-evolution*
   weight (constant for their MDC).
3. The **combined** dark-siren term is p(x|G,D,H0)·p(G|D,H0) + p(x|Ḡ,D,H0)·p(Ḡ|D,H0)
   (main-text Eq. 9/29-33). Writing p(G|D,H0) = β_G(H0)/D(H0) and noting that the
   catalog realizes the population, the catalog denominator Σ_full p(D|z_i,H0) carries
   (up to the galaxy density n̄, an h-constant) **the same H0-shape as β_G(H0)**, so in
   the product the two cancel and one global 1/D(H0) survives — the Mandel–Farr–Gair
   single-normalization structure.

### 2.2 gwcosmo v2 — Gray et al. (2023), arXiv:2308.02281

- Eq. (2.4): the hierarchical likelihood has **one selection normalization per event**,
  [∬ Σ_j p(D_GW|Ω_j,z,θ',Λ) p(θ'|Λ) p(z|Ω_j,Λ) dθ' dz]^(−N_det), integrating the
  **full LOS population prior** (catalog + out-of-catalog, Eq. 2.15). There is **no
  per-event catalogue-restricted p_det sum** anywhere.
- Eq. (2.9): galaxy redshift uncertainty is the **bare Gaussian** p(z|ẑ_k)=𝒢(z−ẑ_k;σ̂_k)
  (no volume prior on catalogued galaxies); dV_c/dz appears only in the out-of-catalogue
  part (Eq. 2.14), where the H0 scale drops on normalization.

### 2.3 Gair et al. (2023), arXiv:2212.08694 (TeX verbatim)

- Per-event likelihood, Eq. label `eq:hierarchica` (published Eq. ~6):

      L(x_i|H0) = ∫ L_GW(x_i|d_L(z,H0)) p_CBC(z) dz / ∫ P_det^GW(z,H0) p_CBC(z) dz

  with p_CBC built from the **whole catalog** (their analysis region), and the delta-z
  limit `Eq:liknouncert` (published Eq. ~15):

      L(x_i|H0) = Σ_{i}^{N_gal} L_GW(x_i|d_L(ẑ_g^i,H0)) / Σ_{i}^{N_gal} P_det^GW(ẑ_g^i,H0)

  The denominator sums **all N_gal galaxies** — event-independent. With z-uncertainties the
  exact treatment (`eq:pdet_truez`, `eq:full_like`) makes p_det = (1/N_gal)Σ_j
  p_det^GW(d_L(z_g^j,H0)) a function of the *true* z's of the whole catalog, and they
  explicitly argue it "is effectively an average of the galaxy redshift distribution over
  the whole volume within which GW sources can be observed", justifying **factoring it out
  as one global term** — the opposite of localizing it to a per-event ball.
- Photo-z: `eq:redpos`, p_red(z|ẑ) ∝ L_red(ẑ|z) p_bg(z)/Z_j with **p_bg uniform in
  comoving volume**, defended against the double-counting objection. (Their toy σ_z model
  is 0.013(1+z)³ — the same expression as our `datamodels/galaxy.py:64`, resolving Known
  Bug 9's "no reference": it is the Gair/TH21 toy model, though it is a *toy*, still
  non-standard for real catalogs.)
- Information budget / expected behavior: complete-catalog + self-consistent framework ⇒
  **unbiased**; no-clustering / completion-dominated ⇒ **uninformative but unbiased**
  (their Fig. 10 discussion). **A monotonic pull is never expected physically**; every
  monotonic-low bias they demonstrate is an *inconsistency*: double counting of the volume
  weight (Inconsistency 1 → biases H0 **low**), Heaviside/mis-modeled p_det
  (Inconsistency 2 → **low**), z_draw truncation not mirrored in the rate model
  (Inconsistency 3), GW-likelihood normalization mis-modeling (Inconsistency 4 → **low**,
  growing with σ_dL).

---

## 3. Difference table: pipeline vs reference

**One-line summary: every h-scale (h⁻³) factor in the pipeline cancels correctly and the
host-z kernel is exactly h-independent; the single structural departure from ALL THREE
references is that L_cat's selection denominator Σ_local w_g D_g is data-conditioned
(restricted to the per-event candidate ball) instead of catalog-global, which breaks the
Gray identity "catalog-denominator h-shape ≡ β_G h-shape" and leaves an uncancelled
monotone factor β_G(h)/Σ_ball(h) per host-found event.**

| # | Term | Pipeline (volume_deconv) | Reference | Documented design choice? | Verdict |
|---|---|---|---|---|---|
| 1 | In-cat numerator | Σ_ball w_g ∫ p_GW p_g(z) dz; p_det excluded | Gray A9 numerator over full catalog (p(x)≈0 outside ball ⇒ equivalent); MFG-clean | §3.17 (816f904): documented | OK (set restriction harmless for numerator, except see #6) |
| 2 | **In-cat selection denominator** | **Σ_ball w_g D_g — per-event, ball-restricted, kernel-averaged** | Gray A9 denominator over **full catalog**; gwcosmo v2 Eq. 2.4 **one global** term; Gair Eq. 15 denominator over **all N_gal** | Partially: "local_ratio" adopted as de-rail fix #2 (2026-07-01, §3.18 ledger), labeled "Gray A.9/A.10 literal local self-normalized ratio" in code comments (`:2097-2102`) — **but the locality is NOT in Gray**; the code's own `weighted_sum` docstring (`:517`) admits the partition-norm form runs the selection sum over the full catalogue | **UNEXAMINED DISCREPANCY — top defect candidate (§4.1)** |
| 3 | Assembly p_i = (β_G L_cat + B_num)/D | β_G multiplies the *locally normalized* L_cat | Gray: β_G/D × [ΣN/Σ_full D]; Σ_full D ∝ n̄·β_G ⇒ β_G cancels, one 1/D survives | Not discussed anywhere in docs | Same root cause as #2, assembly-level view (§4.2) |
| 4 | Host-z kernel | N·(dV_c/dz)/(1+z), per-galaxy renormalized (Z_g) | Gray A9: bare p(z_i); gwcosmo v2 Eq. 2.9: bare Gaussian; Gair eq:redpos: kernel × volume prior (matches us, minus our extra 1/(1+z)) | Yes: §3.18 ledger + commission de-rail fix #1; validated only on shallow seed600 (P-P harness, results/pp_coverage_deepvenue_20260710/) | Structurally h-clean (exact h⁻³ cancellation); deviation from gwcosmo is a defensible Gair-style choice; residual within-kernel z-upshift interacts with #2 at depth (§4.3) |
| 5 | 1/(1+z) placement | In w_pop (inside kernel, renormalized away across galaxies) AND in w_g = R_eff/(1+ẑ_g) (across galaxies, once) | Gray: p(s|z_i) once per galaxy; Gair: p_rate ∝ R(z)/(1+z) once, p_bg volume-only | Rate-weight documented (`:2029-2036`) | **No double count across galaxies** (Z_g kills the within-kernel copy); within-kernel (1+z)⁻¹ tilt is a tiny h-independent reshaping. OK |
| 6 | Candidate window z-cap | Ball capped at min(z(d̂+3σ;h_max), z_depth); B_num/D/β capped at min(z_max^pdet(h), max_z); **D_g windows uncapped** | Gray/Gair: no data-conditioned caps; population edge handled by rate model (Gair Inc-3: unmirrored z_draw ⇒ bias) | Cap system documented (f4/f29a5e7, issue #30) but the L_cat-vs-completion cap asymmetry is not | Minor at seed1000 depth (max detected z≈0.98 < 1.5 cap) — §4.4 |
| 7 | p_det convention | Numerator-free, denominator-only; hypothesis-frame M·(1+z) | MFG 2019; Gray A19-equivalent | Yes (§3.17, memory pdet-hypothesis-convention) | OK |
| 8 | σ_z(z) = 0.013(1+z)³ | Known Bug 9 "no reference" | It IS Gair eq:redlikeli / TH21's toy model | — | Referenced after all, but toy-model provenance worth a docs note |
| 9 | Quadrature | fixed_quad n=50, numerator window width ∝ ~h | References: analytic/dense grids | n=50 documented; volume_trunc postmortem already observed aliasing of narrow kernels | Numerical candidate (§4.5) |

Documentation cross-check: `docs/H0_BIAS_RESOLUTION.md` §3.17 documents the ratio-of-sums
+ denominator-only-p_det alignment (commit 816f904); the §3.18 ledger documents the
photo-z railing saga in which a "local consistent-denominator" variant *failed the σ_z→0
gate* and the *global* smeared denominator D_sm was the sole gate survivor — yet
production later adopted the local ratio (commission de-rail, 2026-07-01/02) validated
only on the shallow venue. **Note:** the task brief cites "§7" for the volume_deconv
validation; the file's actual §7 is a Glossary — the validation record lives in the §3.18
block and `results/pp_coverage_deepvenue_20260710/SUMMARY.md`. `docs/source/limitations.rst`
does not mention the local denominator at all.

---

## 4. Ranked candidate structural defects

### 4.1 [RANK 1 — HIGH] Ball-restricted selection denominator Σ_ball w_g D_g retains a monotone h-factor at depth

**Location:** `bayesian_statistics.py:2122-2127` (`weighted_ratio_of_sums` call, local mode),
fed by D_g from `single_host_likelihood:2631-2671`; set B_i built in `p_D:1861-1872`.

**Math.** Write the pipeline's host-found event term as

    p_i^cat(h) = β_G(h)/D(h) · [ Σ_ball w N(h) / Σ_ball w D(h) ]

and Gray's as

    p_i^Gray(h) = β_G(h)/D(h) · [ Σ_ball w N(h) / Σ_full w D(h) ]      (numerator sums agree since N≈0 outside B_i)

The ratio of the two is the spurious factor

    S_i(h) = Σ_full w_g D_g(h) / Σ_ball w_g D_g(h)  ≥ 1.

Because the catalog realizes the population, Σ_full w D(h) ≈ n̄·β_G(h) → its log-slope in h
is the *population-averaged* selection slope, dominated by the bulk of detectable galaxies.
Σ_ball w D(h) is instead ⟨P_det(d_L(z;h))⟩ averaged over the ball's z-grade (z ≈ z_event).
d ln P_det/d ln h = −(∂ln P_det/∂ln d_L) > 0 (d_L ∝ 1/h), and the local slope grows
steeply as the event approaches the p_det roll-off. Hence for deep events

    d ln p_i^cat/dh − d ln p_i^Gray/dh = −[ d ln Σ_ball D/dh − d ln Σ_full D/dh ] < 0,

a **per-event monotone tilt toward LOW h**, vanishing where p_det ≈ 1 (shallow events).

**Magnitude at z ~ 0.2–0.4.** Across the grid h ∈ [0.60, 0.86], d_L(z_g;h) varies by
×1.43. With a deep-venue P_det(d_L) roll-off between ~1 Gpc and the z≈1 horizon
(~6.7 Gpc at h=0.73), a z≈0.3 ball (d_L 1.36→1.95 Gpc across the grid) sees
Δln⟨p_det⟩ ≈ 0.1–0.4 across the grid; the full-catalog (β_G-shaped) reference slope
offsets only part of it, leaving Δln L per event ~ 0.05–0.3 toward h=0.60. Multiplied over
~1462 host-found events this is ΔΣlogL = O(10²) across the grid — far more than enough to
rail, and consistent with EXP-40's measured "82–83% of the tilt is host-found L_cat" and
Spearman(argmax_h, z) = −0.40 (tilt grows with depth). The shallow z≤0.2 subset closing at
~0.729 is the p_det≈1 limit where S_i(h) is h-flat — exactly this defect's signature.

**Falsifiable predictions (D1 instrumentation):**
- P1a: per event, log-differentiate the stored factors: the measured per-event tilt
  d ln p_i/dh should be ≈ d ln[Σ_ball w N]/dh − d ln[Σ_ball w D]/dh + d ln w_G/dh, with the
  −d ln Σ_ball D/dh term supplying the majority of the *negative* net slope for host-found
  events with z_inj ≳ 0.25, and ≈ 0 for z_inj ≲ 0.15.
- P1b: per-event tilt correlates with the ball's selection steepness
  Δln[Σ_ball w D] ≡ ln Σ_ball D(0.86) − ln Σ_ball D(0.60) (predicted r ≳ 0.8 among
  host-found events), better than with any numerator-side statistic.
- P1c: **surgical swap:** recompute p_i replacing Σ_ball w D(h) by the same-sky-ball,
  full-z-column selection integral (or by the existing global tables
  `_global_cat_denom_*` h-shape, which `precompute_global_catalog_selection` already
  provides). Prediction: deep-venue MAP moves off the 0.60 edge by ≳ +0.05 while the
  shallow seed600 venue result is unchanged within σ_boot.

### 4.2 [RANK 2 — HIGH, same root cause] β_G(h)·L_cat/D(h) assembly double-counts the in-catalogue selection h-shape

**Location:** `p_Di:2051-2059` + `:2084-2086` (β_G table) with L_cat from 4.1.

**Math.** In Gray, the product p(x|G,D)·p(G|D) contains β_G(h) in the numerator of p(G|D)
and Σ_full D (∝ n̄ β_G(h)) in the denominator of p(x|G,D): the in-catalogue selection shape
appears **once up, once down, and cancels**, leaving the single global 1/D(h) (MFG). In the
pipeline, β_G(h) (up) is paired with Σ_ball D(h) (down). These have different h-shapes, so
the event carries the residual factor β_G(h)/Σ_ball(h) — β_G's shape is set by the *whole
detectable volume* (weighted toward the p_det roll-off where dV_c is largest), so its
h-slope is generically *steeper* than a shallow ball's and *flatter* than a deep ball's;
the mismatch is monotone in event depth and does not average out across an ensemble whose
z-distribution is one-sided (volume-weighted). This is the assembly-level statement of 4.1
— fixing 4.1 by restoring a Σ_full-shaped catalog denominator automatically restores the
cancellation.

**Prediction:** in the diagnostics CSV (`event_likelihoods.csv` has w_G and L_cat per h),
d ln w_G/dh + d ln L_cat/dh for host-found events should show the *non-cancelling* residual:
regress per-event d ln L_cat/dh against −d ln β_G/dh; Gray-consistency requires slope ≈ 1
for the selection part (full cancellation); measured slope ≪ 1 with depth-graded residual
confirms the defect.

### 4.3 [RANK 3 — MEDIUM] volume_deconv kernel × steep p_det interaction at depth (kernel z-upshift enters D_g but the validation only covered p_det≈1)

**Location:** `single_host_likelihood:2583-2606` (kernel) feeding `:2642` (D_g).

**Math.** p_g(z) is h-independent (§1.2) — **structurally sound in h** — but relative to
Gray/gwcosmo's bare Gaussian it shifts each wide photo-z kernel's mass toward higher z by
δz ≈ σ_eff²·d ln[f(z)/(1+z)]/dz (≈ 2σ_eff²/z at low z; ~+0.008 for σ_eff=0.035 at z=0.3;
grows for the deep venue's wider σ_eff·(1+z) and flatter f). In D_g this evaluates P_det
at systematically larger d_L where the *slope* ∂ln P_det/∂ln d_L is larger, so the volume
kernel **amplifies the Rank-1 tilt** for photo-z hosts; with the bare kernel D_g sits on a
flatter part of p_det. Sign: same (low-h). This is second-order (it modulates a defect
rather than creating one) but matters because the production default was validated
(P-P/coverage) **only on the shallow seed600 venue where p_det ≈ 1 and the interaction
term is identically absent** — the deep-venue behavior of volume_deconv × steep p_det has
never been separately gated. Gair Inconsistency 1 shows the generic sign of volume-weight
mismatches is LOW-h, matching.

**Prediction:** among host-found events at fixed z_inj, per-event tilt grows with σ_z,g
(photo-z hosts tilt more than spec-z hosts); re-running single events with the bare-Gaussian
kernel (mode "local_ratio") should reduce |tilt| for wide-σ_z hosts by the ratio of
P_det log-slopes at ẑ_g+δz vs ẑ_g, and do nothing for spec-z hosts.

### 4.4 [RANK 4 — LOW at current depth] Candidate-set z-cap vs uncapped D_g windows (support asymmetry for near-edge events)

**Location:** `p_D:1859` (`z_max = min(z_max, redshift_upper_limit)`) vs
`single_host_likelihood:2544-2556` (W_g uncapped); completion side capped per f29a5e7.

**Math.** For events whose numerator window at high h extends past the ball cap
(z(d̂_L+4σ; h_max) > z_depth), galaxies that would have contributed N_g at high h are
excluded from B_i, suppressing Σ_ball N at high h only; the paired D_g of those *excluded*
galaxies is absent from the denominator too, but since the excluded high-z galaxies have
N/D ≫ ballaverage at high h (large N, small D), their exclusion strictly lowers
L_cat(high h) → low-h tilt for near-edge events. Gray/Gair handle the population edge in
the rate model (Gair Inconsistency 3), never by conditioning the set on the data. At
seed1000 the max detected z ≈ 0.98 against a 1.5 cap, so this binds only for the z ≳ 0.7
tail, which is 94–97% fallback — hence LOW rank here, but it becomes a real hazard for any
z_cut-truncated re-analysis (z_cut ∈ {0.2, 0.3, 0.5} puts many events near the cap —
notable because the rail *survived* those truncations).

**Prediction:** per-event tilt vs the truncated-support fraction
max(0, z(d̂+4σ;0.86) − z_cap)/(window width): near-zero correlation at cap 1.5, but a
strong one within the z_cut-truncated reruns; removing the cap for the ball (set only, not
the physics) flattens the z_cut-run rails but not the full-depth rail.

### 4.5 [RANK 5 — LOW/NUMERICAL] fixed_quad n=50 aliasing of narrow spec-z kernels inside an h-scaled numerator window

**Location:** `single_host_likelihood:2654-2662` (numerator quadrature), window `:2538-2543`.

**Math.** The numerator window width in z scales ≈ ∝ h (Δz ≈ 8σ_dL·(∂z/∂d_L) with
∂z/∂d_L ∝ h), so the ~n=50 Gauss–Legendre node spacing changes with h. A spec-z host
kernel has σ_eff ≈ 2×10⁻³ (PV floor); deep-venue windows (σ_dL/d_L ≲ 0.1 at z~0.3) give
node spacings of comparable size, so the sampled fraction of a narrow kernel varies
systematically as the nodes slide with h — an h-*dependent* quadrature error. The
volume_trunc postmortem (`:2508-2510`) already demonstrated exactly this aliasing class.
Not monotone by necessity, so unlikely to be the rail's core, but it contaminates
per-event slopes that D1 will fit.

**Prediction:** recompute N_g(h) for a sample of spec-z hosts at n=50 vs n=400: relative
differences should be ≫ those for photo-z hosts and should change *sign/magnitude across
h non-smoothly*; the ensemble tilt should be nearly unchanged (discriminates 4.5 from 4.1).

---

## 5. Does the literature ever predict a monotonic pull for a catalogue- vs completion-dominated ensemble?

No. Gair et al. (2023): a self-consistent framework is unbiased at any completeness;
removing catalog information (no clustering / completion-dominated) inflates the width but
does not tilt (their Fig. 10 discussion); every monotone-low failure they exhibit is an
implementation inconsistency (double counting, p_det mis-modeling, unmirrored z_draw,
likelihood normalization). gwcosmo v2's single full-population selection term is
scale-free in h up to the physical p_det shape shared by numerator and denominator. The
EXP-40 phenomenology (82% of tilt in host-found L_cat, tilt graded in z, shallow subset
unbiased) is therefore *not expected physics* under any reference formulation — it is the
signature class of a selection-term inconsistency, and the only structural selection-term
deviation this audit found is the ball-restricted denominator (4.1/4.2).

## 6. What this audit could NOT verify

- **Published equation numbers**: A9/A10 identifications rest on counting equation
  environments in the arXiv TeX appendix; the PRD-published numbering was not checked
  against the journal PDF.
- **gwcosmo production code**: conclusions for "the gwcosmo implementation" are from the
  v2 methods paper (2308.02281) equations, not from reading git.ligo.org source.
- **P_det roll-off steepness at z=0.2–0.4** for the seed1000 injection set (needed to turn
  the Rank-1 magnitude estimate from O(0.05–0.3)/event into a number): D1's per-event
  Σ_ball D(h) tables measure exactly this.
- Whether the commission's 2026-07-01 rejection of the global denominator ("pins the mode
  to the grid edge", report bug #2) was tested *after* the G2a completion-sky fix and the
  #29 fallback landed — if the global-denominator rail predated those fixes, its rejection
  may have been confounded, and the Gray-consistent global/partition normalization
  deserves a re-test on the current stack (this is P1c).

## References

- Gray, Hernandez, Qi, Sur et al. (2020), PRD 101, 122001, arXiv:1908.06050 — Appendix A;
  TeX labels `Eq:p(x|G,D,H0)` (≙A9), `Eq:G_DH0_start`→`Eq:G_DH0_end` (≙A10–A12),
  `Eq:p(x|barG,D,H0)`; main-text Eqs. 9, 24–33. (Equations read from arXiv e-print TeX.)
- Gray et al. (2023), arXiv:2308.02281 (gwcosmo v2) — Eqs. 2.4, 2.9, 2.10, 2.14–2.15.
- Gair et al. (2023), AJ 166, 22, arXiv:2212.08694 — TeX labels `eq:hierarchica`,
  `Eq:liknouncert`, `eq:redpos`, `eq:redlikeli`, `eq:pdet_truez`, `eq:full_like`;
  Sec. 4.2 Inconsistencies 1–5. (Equations read from arXiv e-print TeX.)
- Mandel, Farr & Gair (2019), MNRAS 486, 1086, arXiv:1809.02063 — selection normalization.
- Gray, Messenger & Veitch (2022), MNRAS 512, 1127, arXiv:2111.04629 — pixelated variant, Eq. (5).
- Pipeline: `master_thesis_code/bayesian_inference/bayesian_statistics.py` (lines cited inline);
  `docs/H0_BIAS_RESOLUTION.md` §3.17, §3.18 ledger; `results/pp_coverage_deepvenue_20260710/SUMMARY.md`;
  `results/campaign_phase2_runs/run_20260719_seed1000_exp40/FINDINGS_EXP40_20260725.md`.
