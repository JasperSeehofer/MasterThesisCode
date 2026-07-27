# z×M_z-resolved catalogue-selection composition (FIX-3 §7.1 decision)

**Status: RATIFIED (2026-07-27, rev. B) — all seven [RATIFY-Zn] gates
approved by the author with the stated recommendations, with TWO author
amendments recorded at ratification:**

**(Amendment 1 — Z4 primary fix = campaign sizing.)** The author's
directive: "shouldn't we just insist on having a large enough injection
campaign to have a data-driven p_det? we could request a certain ESS per
node ... We should consider this since we will have to redo the
injection/simulation/evaluation anyway." Resolution: the (K5) shrinkage is
ratified as the estimator's **permanent safety net and the interim policy
on the current 50k pool**, but the DESIGNED fix is a **re-injection
campaign sized and sampled to meet a minimum-ESS-per-node requirement**
(data-driven p_det on the full support), folded into the mass-bounds
campaign redesign below. Campaign design doc owns the ESS floor value and
the sampling measure; on the new pool the shrinkage must be measured
inert (w̄ → 1 on the catalogue's query support) — a pre-registered
acceptance criterion of the campaign, not of this packet.

**(Amendment 2 — mass bounds.)** The author's directive: the parameter
space shall be "as large as scientifically correct" — if 10⁷ M☉
(source frame) is the scientifically correct limit (Babak et al. 2017
mass-function valid band [10⁴, 10⁷]), use it; narrowing is permitted only
if VERIFIED (e.g. no detections above the candidate boundary). This
supersedes the unjustified `cosmological_model.py:179–180` override
(M ∈ [10⁴·⁵, 10⁶]) and dissolves the §0-(g1)/§3.3 m-support caveat at
its root for the future campaign; (g1) remains a live limitation of every
result on the CURRENT pool. Tracked in the campaign-redesign issue.

**Sequencing at ratification:** implementation + the pre-registered §4
A/B proceed NOW on the current, self-consistent 10⁶-capped universe
(pool and venue share the cap); the campaign redesign (bounds + ESS
sizing) is the follow-on that removes (g1) and the starved regime.

Original gate summary (recommendations, all now approved):
(Z1) measure-match rule: joint conditional S(d_L | z, M_z) for
catalogue-composition legs only, pool-marginal legs stay z-only by the tower
identity — recommend adopt **with the §3.1-B β_Ḡ/D complement-measure
assumption stated as an assumption, not a theorem**; (Z2) estimator =
product Gaussian kernel in (u = ln(1+z), log₁₀ M_z) with exact
suffix-survival in d_L, Scott d=2 bandwidth N^(−1/6) on both axes, Abramson
adaptivity retained on u — recommend adopt; (Z3) grid/storage: probe-parity
61×31 nodes + dense-d_L stored survival (~45 MB), build-once in `__init__`
— recommend adopt with node-doubling rider **and an explicit m-interpolant
convention choice (§3.3-C)**; (Z4) **key gate** — starved-node policy =
ESS-weighted shrinkage toward S(d_L | M_z) with w = ESS/(ESS + n₀), n₀ = 10
reused from `_MIN_BAND_INJECTIONS`; hard threshold and 2D-Abramson rejected
— recommend adopt, **but the policy is measured NOT inert on the catalogue's
own query support (catalogue-weighted w̄ = 0.83), so it is load-bearing on
the effect size and must be pre-registered at table level**; (Z5) consumer
scope: ALL with-BH p_det queries switch atomically (Σ_glob_wbh + per-host 2D
inner-M integrals), nothing else changes — **recommend adopt with the scope
table corrected (§3.5): Σ_glob_wbh is the load-bearing 2D denominator in
`absolute_marginal` as well as in `generator_marginal`; the per-host 2D
inner-M integrals are load-bearing only in the local ratio-of-sums modes**;
(Z6) position — **REVISED: recommend adopt only in the re-centred form of
§3.8 rev. B. The production-axis increment is −15.6 ln (undershoot vs the
+24–29 ln residual), not the −58 ln of the tabulated axis; branch (d) splits
into (d1) z-composition (this packet) and (d2) selection-side M scatter /
truncation (still open, RATIFY-M5); the A/B's 2D-specific arm is the
A-cell only.** (Z7) flag = `--pdet_wbh_z_resolved`, default OFF
(byte-identical), guard requiring `pdet_z_resolved` — recommend adopt. This
document is the ratifiable form of the author decision that FIX-2's ratified
packet deliberately deferred (`DERIVATION_ZRESOLVED_SURVIVAL.md` §3.4
caveat, §5.2, §8 risk 4: "it remains a FIX-3 §7.1 author decision, NOT part
of FIX-2's minimal coherent switch") — and, per §0-D, a **partial
supersession** of FIX-2 §3.4's main-text decision for the with-BH per-host
integrals, which that caveat does not cover.**

**Scope.** Derive the z×M_z-resolved composition of the with-BH-mass
selection legs: replace the pooled-in-z 2D survival S(d_L | M_z) by the
joint conditional S(d_L | z, M_z) wherever a selection leg's averaging
measure has an M_z|z composition different from the injection pool's —
the catalogue sum Σ_glob_wbh and the per-host 2D selection denominators.
This is the **(d1) z-composition half** of `mass_marginal_2d_kernel.md`
§3.8 **branch (d)** — the branch that survives the measured elimination
(§0 below) as owner of the +25–29 ln 2D HIGH residual. The **(d2) half**
of that branch — selection-side M scatter/truncation, deferred under
RATIFY-M5 — is **not** addressed here and stays open; on the §3.8 rev.-B
arithmetic (d1) accounts for roughly half the residual, so (d1) and (d2)
are jointly, not severally, the expected owner.
The driving records are the FIX-2 packet
(`results/lcat_h_dependence_20260725/DERIVATION_ZRESOLVED_SURVIVAL.md`) and
the four-cell mass-kernel A/B
(`results/lcat_h_dependence_20260725/mass_ab_20260727/MASS_KERNEL_AB_READOUT.md`).
This is a derivation document only: no `.py` file changes, no mode
promotion, no commit until ratified.

---

## 0. Why this is needed (measured, not hypothetical)

**The deferral being closed.** FIX-2 (z-resolved survival, now default-on in
production) explicitly stopped at the 3D boundary: "for the CATALOGUE
selection sum the pool's M_z|z composition is not the catalogue's … it
remains a FIX-3 §7.1 author decision" (FIX-2 §3.4 caveat); "adopting it
moves the predicted gap by −58 ln (§6) and should ride with the
with-BH-channel decision, not silently inside FIX-2" (FIX-2 §5.2); "[MEDIUM]
Mass-composition channel deliberately left out … whichever way FIX-3 §7.1
decides, this packet's estimator supports it" (FIX-2 §8 risk 4). This
document is that decision, in ratifiable form.

*Two errata in the quoted source, to be corrected upstream rather than
propagated:* (i) FIX-2 §3.4/§5.2 describe the probe's ratio as "value
×0.546 vs zres" — 0.546 is joint/**pooled**; joint/zres is **0.642**.
(ii) The "−58 ln" is the (z-only → joint) increment; on the production
(`4d_exact`, M_z-only) axis the increment is **−15.6 ln** (§3.8 rev. B).

**The elimination-attribution (2026-07-27, seed-1000 deep venue, 3454
events, fused 7-point h-grid, stack `e9bec6d`).** From
`MASS_KERNEL_AB_READOUT.md`:

- The 2D channel carries a **+23.8…+29.1 ln HIGH residual at h = 0.80**
  (A″ +23.8, B″ +26.8 post mass-kernel fix; A′ +25.6, B +29.1 before), with
  2D MAP stuck at 0.80 while **1D is at truth in the same cells**.
- Branch (a), the ratified truncated-lognormal mass kernel: measured
  **−1.8…−2.3 ln at 0.80** — real, ~2–3 ln of the ~26 ln excess. P4(ii)
  fired: necessary-but-not-sufficient is *measured*.
- Branch (b), CRB proj/cross-covariance: **NULL** (proj-ablation a no-op at
  4.4×10⁻⁷…3.2×10⁻⁶ ln per event).
- Branch (e), 2D numerator z-quadrature: **NULL** (n = 50 → 200 converged to
  2.9×10⁻¹⁰ ln per event).
- Branch (c), cell-B instrumentation mismatch: **CLEARED** — the same
  residual appears in `absolute_marginal` (A-cells), and the kernel movement
  is normalization-independent.
- Branch (f), B_num residual: bounded at +0.004…+0.006 per-event-sum scale
  by the P–P harness; common to both channels; not 2D-specific.

**By elimination, the residual falls to branch (d): the selection-side
M-treatment of the 2D p_det legs.** Branch (d) as written in
`mass_marginal_2d_kernel.md` §3.8 is *two* things, and this packet closes
only one of them:

- **(d1) z-composition** — the 2D legs are *pooled in z* at fixed M_z.
  This is what the present derivation replaces.
- **(d2) M scatter/truncation on the selection side** — "Σ_glob_wbh is
  **point-evaluated in M with no scatter/truncation**" (`mass_marginal_2d_
  kernel.md` §3.8-(d)), plus the counted-once ledger's sites 2–3
  (w_g = R_eff(M_g) point-evaluated instead of the scatter-averaged Z_M;
  p_det point-evaluated in M), both explicitly **deferred under
  RATIFY-M5**. The A/B that produced the +23.8/+26.8 residual switched the
  *numerator* to the truncated-lognormal kernel while the selection leg
  stayed point-in-M — precisely the ratified "unmatched pair" statement.
  **(d2) is untouched by this packet and remains open.** A NULL A/B
  outcome therefore does *not* imply an unenumerated owner (see P5).

**Correction to the earlier "different assemblies" reading.** It was
previously argued here that A″ and B″ realize the 2D selection through
different assemblies. Code check (bs.py:2884–2900 vs 3060–3078) says
otherwise: **both** cells route the with-BH catalogue channel through the
same global scalar Σ_glob_wbh. What genuinely differs is *how* it enters,
and the difference is the whole design of the A/B:

- `absolute_marginal` (A-cells): `L_cat_with_bh = cat_num_sum_with_bh /
  global_denom_with_bh` while `L_cat_without_bh` divides by
  `global_denom_no_bh`. Σ_glob_wbh divides **the 2D channel only** ⇒ the
  flag is a genuine **2D-specific** discriminator, and the 1D profile is
  bit-identical.
- `generator_marginal` + `dgen_catalog_selection="4d_exact"` (B-cells):
  `D_gen = Σ_glob_wbh/n̂_w + β_Ḡ` divides **both** channels
  (bs.py:3077–3078). The flag therefore moves 1D and 2D by *exactly the
  same* per-event amount, leaving the 2D-minus-1D residual **invariant**.
  B-cells measure the shared-normalization shift, not branch (d1).

The per-host with-BH inner-M integrals (`r[3]`) are load-bearing in
*neither* A″ nor B″ — they enter only the local ratio-of-sums modes and
`catalog_only` (bs.py:2803–2804, 2912–2913).

**The channel's measured size — corrected labels.** From
`z2_results.json:Sigma_glob` (v073 values), the four constructions give
Σ_cat ratios: **joint/pooled = 0.546**, **joint/z-only = 0.642**,
**joint/M_z-only = 0.927**, **M_z-only/pooled = 0.589**. The number
0.546 is joint/**pooled** — it is *not* joint/z-only and it is *not* the
like-for-like twin of production's `Σ_glob_wbh/Σ_glob = 0.556`. The
like-for-like probe object is **M_z-only/pooled = 0.589**, which differs
from production's 0.556 by **6 %**, not 2 %. (For contrast, the probe's
3D-channel parity against production run tables is 4×10⁻⁵ in value —
FIX-2 §2. The 6 % 2D-channel gap is an open parity item, not a
cross-validation.) The h-slope of the catalogue term moves from +0.505
(z-only) → **+0.364** (M_z-only) → **+0.106** (joint).

**The production-axis effect (recomputed from the existing tables).** The
production 4d_exact stack already carries the M_z-only catalogue term. On
that axis the assembled full-mixture 0.73→0.86 gap moves
**−63.3 → −78.9 ln**, i.e. an increment of **−15.6 ln** — not −58 ln. The
−58 ln quoted in the FIX-2 packet and the mass-kernel readout is the
(z-only → joint) increment, an axis production does not sit on. §3.8
rev. B carries the corrected arithmetic and its consequences for P2/P5.

**Two live alternative owners that the elimination never considered:**

- **(g1) Mass-support mismatch.** The injection pool's detector-frame mass
  support is log₁₀M_z ∈ [4.565, **6.000**] (hard 10⁶ cap in every z-bin),
  while the catalogue's rate-weighted composition has mean log₁₀M_z =
  **6.43** and **81.4 %** of its weight at log₁₀M_z ≥ 6.0 (45.8 % ≥ 6.5)
  — measured from `catalog_zw_profile.json:W_z_lm` + the 50,000-row pool.
  For that 81 % of the weight, `detection_probability_with_bh_mass_
  interpolated` returns the A2-EXTRAP **true-nearest clamp** at the pool's
  top mass node (sdp.py:1393–1467), and any joint grid will clamp at
  m = m_max identically. The measure-match rule is *unsatisfiable* there —
  the pool has no data at the catalogue's masses — so ×0.546/0.556/0.642
  partly measure a boundary extrapolation rather than a conditional.
  Adding a z axis at a clamped m cannot repair this. The underlying
  population mismatch (pool M from the Barausse M1 sampler,
  `cosmological_model.py:337`; catalogue M from Reines–Volonteri) is a
  gate item in its own right.
- **(g2) The complement measure of β_Ḡ** — see §3.1-B.

**(D) Partial supersession of a ratified decision.** FIX-2 §3.4 decides in
its *main text*: "**2D/with-BH consumers** (`Σ_glob_wbh`, with-BH per-host
mass-integrals): keep the existing `S(d_L|M_z)` **UNCHANGED** … This is the
measured, pool-decided answer", on ground (ii) "statistics-starved at
exactly the low-z nodes the catalogue weights" — which this packet's own
ESS audit *confirms* (§3.4: the catalogue's weight lands on ESS ≈ 40–60
nodes). The §3.4 *caveat* defers only the catalogue **sum**; the with-BH
**per-host** integrals are not covered by it. Honest ledger: this packet
**partially supersedes** a ratified decision, on the grounds that a
*slope*, not a *value*, is at stake, and that the starvation ground is met
by the §3.4 shrinkage policy rather than by abstention. Z1/Z5 must be
ratified with that supersession explicit.

## 1. Current state of the code (what already exists)

Anchors: `master_thesis_code/bayesian_inference/simulation_detection_probability.py`
(= sdp.py) and `master_thesis_code/bayesian_inference/bayesian_statistics.py`
(= bs.py); probe `results/lcat_h_dependence_20260725/zres_survival/z2_zres_slopes.py`
(= z2.py).

**FIX-2 is implemented and default-on, not merely derived.** The FIX-2
packet's own header still says "DERIVATION ONLY — no implementation code
written" (DERIVATION_ZRESOLVED_SURVIVAL.md:3); the repo has since passed
that state: `_build_zres_survival` (sdp.py:591–765, u-kernel + Abramson
pilot + 121-node suffix tables), `_zres_survival_at` (779–808),
`_zres_survival_at_band` (810–850, per-(band, node) ESS <
`_MIN_BAND_INJECTIONS` = 10 fallback to the band-marginal z-conditional),
`_require_zres_z` (852–865), quality flags `zres_u_nodes`/`zres_ess`
(1355–1356), CLI `--pdet_z_resolved` default True (arguments.py:404–423).
This drift is recorded here so the ledger is honest; the packet remains the
governing derivation record for what was built.

**The 2D (with-BH) p_det object is pooled in z and untouched by FIX-2.**
`_build_grid_2d` (sdp.py:1180–1295) builds
p_det(d_L, M_z) = Σ_k K_M(log₁₀M_z,k − log₁₀M_z)·1[d_hor_k ≥ d_L] / Σ_k K_M
— Gaussian kernel on log₁₀ M_z only (Scott `_SCOTT_EXPONENT_2D` = −1/6,
`_compute_bandwidths`, sdp.py:527–555), exact suffix-survival in d_L, on a
60 (linear d_L) × 40 (geomspace M_z) grid (`_grid_support`, 1115–1135);
query wrapper `detection_probability_with_bh_mass_interpolated`
(1393–1467) with the A2-EXTRAP clamp semantics (d_L below grid → clamp
(≈1), d_L above last center → exact 0, M_z → true-nearest clamp). The
`pdet_z_resolved` flag gates ONLY the 3D accessors; **there is no
`_build_zres_survival_2d` or any joint (u, log₁₀ M_z) grid in production.**

**Consumers of the 2D object** (bs.py):

- `precompute_global_catalog_selection`, with-BH branch (bs.py:1416–1622):
  Σ_glob_wbh = Σ_g w_g·p_det(d_L(z_g;h), M_g(1+z_g)),
  w_g = R_eff(M_g)/(1+z_g), deliberately ISOTROPIC (sky×M_z is
  statistics-starved, comment bs.py:1560–1565). Each galaxy's z_g is in
  hand (it computes d_L and the (1+z_g) lift) but is **not** passed to
  p_det. Load-bearing as D_gen's catalogue term under `generator_marginal`
  with `dgen_catalog_selection = "4d_exact"` (the default; bs.py:1759,
  2112, comments 3052–3067), and as the divisor in global modes.
- `_bh_mass_denominator_inner_m_integral(_batch)` (bs.py:3140–3283):
  exact erf-sum of the piecewise-linear-in-M_z interpolant against the
  Gaussian mass prior — the per-host 2D D_g for the `"gaussian"` mass
  kernel; pulls `m_centers` off the live 2D interpolator.
- `_mass_trunc_denominator_inner_m_integral(_batch)` (bs.py:596–677):
  GL-64-in-lnM against the ratified truncated LN×R_eff kernel — the
  per-host 2D D_g for `"trunc_lognormal"`. Both per-host paths hold z at
  each quadrature node and query p_det(d_L(z), M(1+z)).
- Load-bearing per mode (**corrected 2026-07-27; the earlier reading was
  inverted for `absolute_marginal`**):
  - `generator_marginal` → **Σ_glob_wbh**, inside `D_gen` (bs.py:3065–3078);
    D_gen divides BOTH the 1D and the 2D channel. Per-host D_g
    diagnostic-only.
  - `absolute_marginal` (also `global`, `volume_global`) → **Σ_glob_wbh**
    as `global_denom_with_bh` (bs.py:2884–2900:
    `L_cat_with_bh_mass = cat_num_sum_with_bh / global_denom_with_bh`),
    divisor of the 2D channel ONLY. Per-host D_g (`r[3]`) is **not** used
    here.
  - `local_ratio` / `volume_deconv` / `mass_trunc` / `catalog_only` → the
    per-host inner-M integrals `r[3]` in `weighted_ratio_of_sums`
    (bs.py:2803–2804, 2912–2913); Σ_glob_wbh not used.

**The joint estimator exists only as a probe.** z2.py:`build_surv_ulm`
(118–135): 61 u-nodes × 31 lm-nodes, product Gaussian kernel
(u Abramson-adaptive at σ_u = N^(−1/5)·std(u) = 0.01924; lm fixed at
σ_lm = N^(−1/6)·std(lm) = 0.05174), exact suffix-survival in d_L read on a
3000-point d_L query grid; bilinear query `q_ulm` (305–317); catalogue
assembly against the binned profile `catalog_zw_profile.json`
(z2.py:294–331; profile anchors verified: 9,060,017 rows, 9,060,008 at
z < 0.992, W_cat = 6.34766e8 = the generator_norm value to machine
precision). Its ESS audit is the §3.4 starting point. N_EVENTS = 3454
(z2.py:48, hardcoded) matches the measured four-cell venue (3454 events,
readout) — verified, not just asserted.

**Constraints inherited by any production implementation:**

- `__getstate__` (sdp.py:421–448) drops the raw per-injection arrays —
  the joint tables must be fully built inside `__init__` before workers
  fork; only pre-built tables ship.
- `_get_or_build_grid` (sdp.py:1082–1113): grids are built ONCE
  (h-invariant horizons) and registered per-h only for API parity — the
  joint grid must follow the same pattern.
- `bandwidth_scale` (constructor, default 1.0) is NOT CLI-wired; it is the
  pre-existing sensitivity knob and stays that way.
- Flag-plumbing template: the 6-hop `--pdet_z_resolved` chain
  (arguments.py:404–423 → property 156–159 → main.py:145 →
  main.py:1015–1057 → bs.py:1800 → bs.py:2015 constructor kwarg).

## 2. Verified literature and repo anchors (2026-07-27)

| Source | Content used here | Verified |
|---|---|---|
| Finn & Chernoff 1993 (arXiv:gr-qc/9301003); Finn 1996 (arXiv:gr-qc/9601048) | horizon-survival framework: det ⇔ d_L ≤ d_hor at fixed detector-frame θ | repo-verified via FIX-2 §4.3 (ratified lineage) |
| Mandel, Farr & Gair 2019 (arXiv:1809.02063), Eqs. (6)ff | selection = expectation of P(det\|θ) over the population AT HYPOTHESIS; the hypothesis population specifies (z, M_z) jointly for the catalogue channel | repo-verified (FIX-2 §4.3, hostz doc §2) |
| Hogg 1999 (arXiv:astro-ph/9905116), Eq. (16) | 1/d_L amplitude law exact within fixed detector-frame θ — the h-invariant horizon | repo-verified (FIX-2 §1.1) |
| Scott 1992, *Multivariate Density Estimation*, Ch. 6 | bandwidth rule N^(−1/(d+4)) per axis; d = 2 → N^(−1/6) | standard; repo-cited (FIX-2 §3.3, sdp.py:97–107 exponent note) |
| Abramson 1982, Ann. Statist. 10:1217 | √-law adaptive bandwidth from a pilot density | repo-verified (FIX-2 §3.3; implemented sdp.py:656–675) |
| Kish 1965, *Survey Sampling* | ESS = (Σw)²/Σw² as the **variance**-equivalent sample size of a weighted mean (it says nothing about kernel bias — §3.4 scope note) | standard; the repo's own ESS convention (sdp.py:693–698, z2.py:131–132); original text not re-opened |
| Conjugate beta-binomial posterior mean (standard, e.g. Gelman et al. BDA3 §2.4) | pseudo-count shrinkage (n₀·S_marg + ESS·Ŝ)/(n₀ + ESS) — used in §3.4 as a **motivating heuristic**, not as a posterior-mean identity (weights shared and indicators correlated across d_L) | standard textbook result; adopted for its limits/continuity, explicitly labelled heuristic in §3.4 |
| Law of total expectation (tower property) | the z-only estimator is the pool-conditional M_z-marginal of the joint — the §3.1 measure-match rule | elementary; derived inline |
| Repo precedent: `_build_grid_2d` (sdp.py:1180–1295) | lm-kernel + exact suffix-survival — one factor of the product kernel | read in full this session |
| Repo precedent: `_build_zres_survival` (sdp.py:591–765) | u-kernel + Abramson + per-node ESS + starved-cell fallback pattern | read in full this session |
| Probe: z2.py `build_surv_ulm`/`q_ulm` + `z2_results.json` | the measured joint construction, ESS audit, slope tables, gap arithmetic | read in full this session |

Consistent with the ratified FIX-2 and mass-kernel derivations: (i) the
estimator is h-invariant at build time; (ii) it lives ONLY in selection
legs (numerators carry no p_det — MFG convention); (iii) the grid axis and
the query coordinate share one convention (detector-frame
M_z = M·(1+z) at hypothesis — `project_pdet_hypothesis_convention`).

## 3. The derivation

### 3.1 The exact object and the measure-match rule **[RATIFY-Z1]**

Every selection leg is an average of the exact detection kernel over a
leg-specific measure. From FIX-2 Eq. (2) (Finn & Chernoff), conditioning on
everything the horizon distribution depends on through the detector frame:

    P(det | z, M_z, h) = P( d_hor ≥ d_L(z; h) | z, M_z ) ≡ S(d_L(z;h) | z, M_z) .   (K1)

FIX-2's ratified S(d_L | z) is the M_z-marginal of (K1) **under the pool's
own conditional**: by the tower property,

    S(d_L | z) = E_{M_z ~ p_pool(·|z)} [ S(d_L | z, M_z) ] .                        (K2)

This identity decides, leg by leg, which conditional is exact — the
**measure-match rule**: *a selection leg may use the z-only estimator if
and only if its averaging measure's M_z|z conditional equals the pool's;
otherwise it must query the joint conditional (K1).*

| Leg | averaging measure's M_z\|z | exact object | consequence |
|---|---|---|---|
| D (`precompute_completion_denominator`) | the population model's = the pool's own (the pool is drawn from it) | z-only, by (K2) **exactly** | unchanged (FIX-2) |
| β_Ḡ (`precompute_missing_completion_denominator`) | **ASSUMED** the pool's — but see §3.1-B: β_Ḡ's true measure is the *complement* of the catalogue within the population, which is not the population's own conditional unless the catalogue is mass-unbiased | z-only *conditionally on that assumption* | unchanged (FIX-2) — **assumption, not theorem** |
| Σ_glob no-BH | catalogue z-weights, but no mass conditioning is *defined* for the 3D channel; (K2) collapses the pool conditional | z-only at z_g | unchanged (FIX-2) |
| **Σ_glob_wbh** | the CATALOGUE's joint (z_g, M_z,g) — measured ≠ pool (joint/pooled = 0.546, production Σ_glob_wbh/Σ_glob = 0.556), **subject to the (g1) support caveat** | **joint (K1) at (z_g, M_g(1+z_g))** | **switches** |
| **per-host 2D D_g** (erf-sum + mass_trunc GL) | the HOST's p_g(z)·p_g(M) — the host mass kernel centred at M_g, not the pool conditional | **joint (K1) at (z_node, M(1+z_node))** | **switches** (load-bearing in the local ratio-of-sums modes; diagnostic in A/B cells) |
| B_num, all numerators N_g | no p_det (MFG) | — | untouched |

**§3.1-B — the complement-measure gap (open, must be closed before Z1).**
Under `generator_marginal`, D_gen = Σ_glob_wbh/n̂_w + β_Ḡ sums **two
complementary halves of one population**: the in-catalogue half and the
missing half. This packet conditions the first on the *catalogue's* M_z|z
and leaves the second on the *pool's*. That is not symmetric, and the
asymmetry is not innocuous: the out-of-catalogue galaxies are the
selection-faint complement of the same galaxy population, so their M_z|z is
the completeness-selected complement of the catalogue mass function, which
is emphatically *not* established to equal the pool's. The packet's own
measured catalogue-vs-pool composition mismatch is the evidence against the
assumption, not for it.

Measured sensitivity (re-running the probe's own assembly,
`z2_zres_slopes.py:342–360`, with β_Ḡ carrying the *same* composition
factor as the catalogue term at every h):

| variant (gap = 92 + N·(dlnD_gen_3D − dlnD_gen_new)) | gap [ln] |
|---|---|
| FIX-2 baseline, z-only catalogue term | −21.4 |
| catalogue → joint, β_Ḡ pool-conditioned (this packet, tabulated axis) | −78.9 |
| catalogue → joint **and** β_Ḡ composition-matched (tabulated axis) | **−10.6** |
| production axis, M_z-only baseline, β_Ḡ pool-conditioned | −63.3 |
| production axis, catalogue → joint, β_Ḡ pool-conditioned | −78.9 |
| production axis, catalogue → joint **and** β_Ḡ composition-matched | **−125.4** |

The spread is ±60 ln around the packet's own number — larger than the
effect being gated. Note the sign of the *symmetric-treatment* correction
is **axis-dependent** (it moves the tabulated-axis increment from −57.5 to
+10.8 but the production-axis increment from −15.6 to −62.1), so no
directional claim can be read off the symmetric variant; only the
magnitude of the sensitivity is robust.

**Consequence for the gate.** Z1 may be ratified only with the β_Ḡ/D row
stated as an **assumption**, and only if the symmetric variant is carried
as a **mandatory control cell** (§4 item 10) rather than assumed away. If
p_missing(M_z|z) is derived and found ≠ p_pool(M_z|z), β_Ḡ must switch too
and §3.8's arithmetic is redone.

Two distinct errors are being fixed, and it matters which is which:

1. **The current Σ_glob_wbh / per-host D_g query S(d_L | M_z)** conditions
   on the right first-order variable (measured: partial
   corr(log₁₀d_hor, log M_z | u) = 0.214 vs partial(·, u | log M_z) = 0.035,
   FIX-2 §1.3) but **pools z at fixed M_z** — the residual z-composition
   error at fixed M_z. Small per query in the pool average (partial 0.035),
   but the *h-slope* of the catalogue term is what the posterior feels, and
   the measured slopes differ grossly: catalogue-term slope +0.106 (joint)
   vs +0.505 (z-only) vs +0.357 (pooled) (FIX-2 §6 table).
2. **The z-only FIX-2 object applied to the catalogue sum** would carry
   the *pool's* M_z|z into a sum whose actual composition is the
   catalogue's — a value error of ×0.642 (joint/z-only; **not** ×0.546,
   which is joint/pooled). Production 4d_exact does NOT make this error
   (it conditions on M_z, and joint/M_z-only is ×0.927 in value); the
   −58 ln arithmetic was computed against the z-only variant, which DOES
   — the axis distinction that §3.8 rev. B confronts and resolves.

The joint conditional (K1) is the unique object exact for both, and the
natural completion of FIX-2 (whose estimator is its pool-conditional
M_z-marginal, (K2)). No leg needs anything finer: sky×z×M_z remains
statistics-starved and the with-BH branch stays isotropic (existing
bs.py:1560–1565 decision, unchanged here).

**[RATIFY-Z1: adopt the measure-match rule — joint conditional (K1) for
Σ_glob_wbh and the per-host 2D selection denominators; z-only retained for
D, β_Ḡ, Σ_glob-no-BH. For D and Σ_glob-no-BH the retention is EXACT by
(K2); for β_Ḡ it rests on the §3.1-B complement-measure **assumption**,
which must be ratified as an assumption and carried as a mandatory control
(§4 item 10), not asserted as a theorem. Numerators untouched. The rule's
applicability to Σ_glob_wbh is further limited by the §0-(g1) support
caveat (81.4 % of catalogue weight on the m_max clamp), which §4 item 11
must quantify. Ratification also enacts the §0-D partial supersession of
FIX-2 §3.4's main-text decision for the with-BH per-host integrals.
Recommendation: adopt with those three qualifications explicit.]**

### 3.2 The estimator — product kernel in (u, log₁₀ M_z), exact suffix-survival in d_L **[RATIFY-Z2]**

    S(d_L | z, M_z) =
        Σ_k  K((u_k − u)/σ_k^u) · K((m_k − m)/σ_m) · 1[ d_hor_k ≥ d_L ]
        ─────────────────────────────────────────────────────────────────           (K3)
        Σ_k  K((u_k − u)/σ_k^u) · K((m_k − m)/σ_m)

    u = ln(1+z),   m = log₁₀ M_z,   d_hor_k = SNR_k·d_L_k/ρ_thr  (h-invariant)
    K = standard Gaussian;   σ_k^u = σ_u·λ_k (Abramson, u only);  σ_m fixed.

**Coordinates (derived, not chosen).** u = ln(1+z): the z-dependence of
d_hor enters exclusively through multiplicative (1+z) detector-frame lifts,
so a z-shift is a *translation* in u — the coordinate in which a fixed
bandwidth is correct (FIX-2 §3.2, ratified lineage). m = log₁₀ M_z:
the existing 2D kernel's coordinate (sdp.py `_log_M_z`), detector-frame —
the pool CSV "M" column is already observer-frame M_z (sdp.py:361–366
comment), and catalogue queries lift M_g(1+z) to the same convention.
(K3) is the exact product completion of the two ratified single-axis
kernels: `_build_zres_survival`'s u-factor × `_build_grid_2d`'s m-factor,
with the shared suffix-survival treatment of d_L (no kernel in d_L —
guarantees S(0|·) = 1, S(d_L > max d_hor|·) = 0, monotone in d_L, bounded
[0,1], all by construction).

**Bandwidths (statistics-derived, no free constants).** For a d = 2
product kernel Scott's rule is σ_j = N^(−1/(d+4))·std_j = **N^(−1/6)·std_j
per axis** (Scott 1992 Ch. 6). On the canonical 50,000-row pool:
σ_u = 0.02759, σ_m = 0.05174. Two consequences, both deliberate:

- **The m-axis bandwidth is *exactly* the existing 2D kernel's** —
  σ_m = `sigma_log_M` from `_compute_bandwidths` (the repo's
  `_SCOTT_EXPONENT_2D` = −1/6, which for the joint kernel is now the
  arithmetically correct d = 2 exponent, resolving the documented
  d=1-mislabel discrepancy *for this estimator* — sdp.py:97–107). The
  σ_lm → ∞ / σ_u → ∞ marginal limits (§3.7) therefore recover the
  existing kernels without bandwidth mismatch on the m side.
- **The probe used N^(−1/5) = 0.01924 on u** (its 1D value). The measured
  insensitivity — z-only D-slopes identical to 0.015 across both exponents
  and a ×4 bandwidth range (FIX-2 §2 table) — says the choice is not
  load-bearing; d = 2 is the derivable choice for the joint kernel and is
  also *wider* on u (0.0276 vs 0.0192), which mildly helps the §3.4
  starvation. A pre-registered exponent-insensitivity check on the joint
  assembly is §4 item 7.

**Adaptivity.** Abramson's √-law on the u axis only, exactly as FIX-2
implements it (pilot KDE, λ_k = (ĝ/f̂(u_k))^(1/2), re-measured on this pool
with the probe's own pilot: **λ ∈ 0.755–24.12**, Ĝ = 1.7228 — FIX-2 quotes
0.76–23.0 at its pilot settings): the u-density is strongly
volume-weighted-non-uniform; the
m-density is not comparably starved on its own axis, and the existing
m-kernel is non-adaptive — kept unchanged. Joint-space starvation that the
1D u-pilot cannot see is the province of the §3.4 policy, NOT of an
invented 2D bandwidth law (rejection grounds there).

**[RATIFY-Z2: estimator (K3); coordinates (u, log₁₀ M_z); Scott d = 2
exponent N^(−1/6)·std per axis (m-bandwidth thereby identical to the
existing 2D kernel); Abramson adaptivity on u only, probe-parity;
suffix-survival exact in d_L. Recommendation: adopt, with the §4 item-7
exponent-insensitivity rider.]**

### 3.3 Grid, storage, build-once, and where h enters **[RATIFY-Z3]**

**Nodes.** 61 u-nodes on [0, ln(1+1.5)] × 31 m-nodes on
[m_min, m_max](pool) = [4.565, 6.000] — probe parity (`build_surv_ulm`,
z2.py:120–121): the construction that produced every pre-registered number
in §3.9. (Production z-only uses 121 u-nodes; the joint grid halves the
u-resolution because the ESS budget is shared with the m axis.
Node-doubling convergence is §4 item 7.)

**The m-support caveat (§0-(g1), inherited, unrepaired by this grid).** The
m-node span is the *pool's* mass range; **81.4 %** of the catalogue's
rate weight (mean log₁₀M_z = 6.43) lies above m_max = 6.000 and is served
by the true-nearest clamp at the top node. On that 81 % of the weight the
joint grid conditions on *z* correctly but on *M_z* only through a boundary
extrapolation — the same clamp the current 2D object already applies. No
grid choice inside the pool's support fixes this; it is escalated as its
own gate item (§4 item 11) and as the (g1) branch in P5.

**Storage scheme — the memory decision.** Two candidate layouts:

- *(a) exact per-node suffix tables*, the `_build_zres_survival` pattern:
  (N × 61×31) float64 ≈ **756 MB** (and ≈1.9 GB at a 121×40 production
  resolution) — shipped to every worker via pickle. Rejected on cost: the
  1D z-resolved tables already ship ~100 MB; an order of magnitude more
  per worker is not justified when
- *(b) stored survival on a dense d_L query grid* (probe parity): evaluate
  the exact suffix-survival at 3000 d_L points
  (`DLQ = linspace(1e-4, 1.02·max d_hor, 3000)`) per (u, m) node and store
  only S: (3000 × 61 × 31) float64 ≈ **45.4 MB**, plus the (61×31) ESS
  array and the (3000 × 31) m-only marginal table (§3.4 shrinkage target,
  ≈0.7 MB). Queries: bilinear across the four bracketing (u, m) nodes, and
  in d_L **one of two conventions must be chosen explicitly (§3.3-C)** —
  the probe's `q_ulm` does a `searchsorted` *step* lookup in d_L
  (z2.py:313), NOT linear interpolation, so "linear in d_L" and "exactly
  `q_ulm`" are mutually exclusive. The d_L-grid discretisation is a
  1/3000-support-resolution effect, refinable by a DLQ-doubling rider
  (§4 item 7).

**§3.3-C — interpolation conventions (must be fixed at Z3, not left
implicit).** Two conventions are load-bearing and neither is settled by
"probe parity":

1. **d_L convention.** Probe: step (`searchsorted`). Production 2D grid:
   `RegularGridInterpolator(method="linear")`, i.e. linear in d_L.
   *Recommendation:* adopt **linear in d_L** (matches production and
   preserves C0, which the step lookup does not), and note that P1's
   probe-parity target then carries a step-vs-linear convention delta of
   order one DLQ cell — pre-register it as ≤10⁻³ relative, or re-run the
   probe assembly with linear-in-d_L to remove it.
2. **m convention, and the erf-sum's validity.** The exact erf-sum
   (bs.py:3149–3175) is closed-form **only for an interpolant that is
   piecewise-linear in M_z**, which the current geomspace-knot
   `RegularGridInterpolator` is. A grid bilinear in m = log₁₀M_z is
   piecewise-linear in **log** M_z, which is *not* piecewise-linear in
   M_z. So §3.5's claim that "the exact erf-sum closed form applies with
   node-blended knot values" is **false as stated**. Either
   (a) interpolate **linear in M_z** between the lifted knots
   10^{m_j}/(1+z) — the erf-sum stays exact, but it is then not `q_ulm`
   parity and P1 inherits a convention delta; or (b) keep `q_ulm`'s
   linear-in-m blend — the erf-sum becomes an approximation whose error
   must be bounded (the per-segment linear-in-M vs linear-in-log-M gap on
   a 31-node log grid of width Δm = 0.0478 is ≤ (ln10·Δm)²/8 ≈ 1.5×10⁻³
   relative on a smooth p_det, to be measured, not assumed).
   The scalar/batch twin tests guard bit-identity between paths, **not**
   this. *Recommendation:* pick (a) for the erf-sum path and re-derive the
   probe-parity target; carry (b)'s bound as a §4 convergence rider.

**Clamp semantics (unchanged conventions).** d_L below the first grid
point → 1-side clamp (suffix guarantees S ≈ 1 there); d_L above the last
point → exact 0 (the A2-EXTRAP rule); m outside the pool span →
true-nearest clamp (existing 2D wrapper); u outside [0, u_max] → clamp to
the node span (existing `_zres_node_pos` rule). No new boundary machinery.

**Build-once and pickling.** The build consumes only
(d_hor_k, u_k, m_k) — all fixed at injection time. It must run inside
`__init__` (the raw arrays do not survive `__getstate__`,
sdp.py:421–448); workers receive the finished (45 MB) tables. Build cost:
the probe builds the 61×31 grid inside a ~3-minute script that also does
six 121-node builds and 4×41 selection integrals — seconds, same class as
today's grids.

**Where h enters (and where it cannot).** The entire estimator state —
horizons, kernel weights, suffix sums, ESS, shrinkage weights — is
h-invariant. h enters ONLY through the query:
d_L(z_g; h) for the catalogue sum (z_g, M_z,g = M_g(1+z_g) are catalogue
data, h-free), and d_L(z_node; h) for the per-host integrals (z_node from
the host kernel window, h-free; M(1+z_node) h-free). The joint
conditioning adds **no new h channel**: the conditioning coordinates ride
the same z the caller already holds ("pass the z you are already
holding", FIX-2 §5.1). One build per run, `_get_or_build_grid`-style
per-h registration for API parity only. Under the opt-in σ_z smearing,
the conditioning z AND the (1+z) mass lift ride the SAME smeared z inside
the expectation — one z per query, counted once
(`project_pdet_hypothesis_convention`).

**[RATIFY-Z3: 61×31 probe-parity nodes; storage scheme (b) (stored
survival, ≈45 MB); existing clamp semantics; build-once in `__init__`;
**plus an explicit §3.3-C convention choice — linear (not step) in d_L,
and linear-in-M_z (not linear-in-m) between lifted knots for the erf-sum
path**, each with its probe-parity delta pre-registered. The m-support
caveat (81.4 % of catalogue weight on the m_max clamp) is recorded as a
known limitation of any grid on this pool, not repaired by this gate.
Recommendation: adopt, with node-doubling and DLQ-doubling riders in §4
and the grid-only control cell (§4 item 12).]**

### 3.4 The ESS-starved policy — the key gate **[RATIFY-Z4]**

**ESS definition (identical to the audit's and to production's).** At each
(u, m) node, over the product-kernel weights w_k of all N = 50,000 pool
injections,

    ESS = (Σ_k w_k)² / Σ_k w_k²                                                     (K4)

— Kish's variance-equivalent sample size; computed exactly as
z2.py:131–132 and sdp.py `_suffix_tables` (693–698). ESS is d_L-independent
(the weights carry no d_L), so one number per node.

**The measured problem** (`z2_results.json:ess_ulm_product`, quoted
verbatim, at the probe's σ_u = N^(−1/5)): min ESS = **1.013** over the
61×31 = 1891 nodes; **25.2 %** of nodes below ESS 100; **60.2 %** below
ESS 500. Compare the z-only kernel's node minimum of **211** (which clears
the repo's n ≥ 10 convention by >20×, FIX-2 §7). The joint grid's floor is
~200× worse: at ESS ≈ 1 a node's "survival curve" is a single injection's
step function. A policy is mandatory, and FIX-2 §7 wrote one only for the
z-only kernel — the joint analog was never specified. This gate specifies
it.

**Re-audited at the Z2 bandwidth (the object actually being ratified).**
Every published probe number above is for the *unshrunk* estimator at
σ_u = N^(−1/5) = 0.01924. Z2 ratifies σ_u = N^(−1/6) = 0.02759 **and**
Z4 adds (K5) shrinkage. Rebuilt on the same 50,000-row pool at the Z2
bandwidth (same Abramson λ, same σ_m):

| σ_u | min ESS | ESS<10 | ESS<100 | ESS<500 | **catalogue-weighted w̄ (n₀=10)** | catalogue W-frac on ESS<100 | catalogue-weighted median ESS |
|---|---|---|---|---|---|---|---|
| probe N^(−1/5) = 0.01924 | 1.013 | 3.4 % | 25.2 % | 60.2 % | **0.771** | 99.98 % | 36 |
| **Z2 N^(−1/6) = 0.02759** | 1.042 | 3.2 % | 19.4 % | 50.9 % | **0.834** | 98.3 % | 53 |

Two consequences that change the gate:

1. The wider d = 2 bandwidth helps only mildly — starvation persists, and
   the shrinkage is active over roughly half the grid. **No probe number
   exists for the shrunk object at the Z2 bandwidth**, and the §4 item-7
   rider n₀ ∈ {5, 10, 20} never brackets the probe's effective n₀ = 0.
   P1/P2 must be restated for the object that ships (§3.9 rev. B).
2. **The policy is not inert where it matters.** 98–100 % of the
   catalogue's rate weight sits on nodes with ESS < 100; the
   catalogue-weighted mean shrinkage weight is w̄ ≈ **0.83**, not > 0.95.
   Since Σ_glob_wbh = Σ_g w_g·S̃ is *linear* in S̃, (K5) attenuates the
   joint-vs-M_z-only movement by ≈ 1 − w̄ ≈ **17 %** (−15.6 → ≈ −13 ln).
   The shrinkage is load-bearing on the effect size, not a starved-corner
   patch — so it belongs in P1/P2 as a pre-registered table quantity, and
   P6's "inert where measurable" claim is vacuous as previously worded.

**Design constraints (first principles + repo feedback rules):**

1. No anchoring to truth or to the observed tilt — the policy must be a
   function of the pool only (`feedback_principled_physics_choices`).
2. C0 continuity is the minimum bar — in the query coordinates AND in the
   pool statistic (no estimator cliff at an ESS threshold).
3. The starved limit must degrade to the *next-coarser conditional that
   preserves the first-order conditioning*, not to the fully pooled
   survival — the FIX-2 §7 principle ("falls back to the z-only
   (band-marginal) conditional — NOT to the fully pooled survival"),
   transposed: here the first-order conditioner is **M_z** (partial
   0.214 vs 0.035), so the fallback target is **S(d_L | M_z)** — the
   current production 2D conditioning.
4. No new fitted constants; reuse the repo's n ≥ 10 reliability
   convention (`_MIN_BAND_INJECTIONS`, sdp.py:75) if a constant is needed.

**Candidate (i) — hierarchical hard fallback** (repo sky-band precedent):
node ESS < threshold ⇒ replace by S(d_L | M_z). Correct target and C0 in
the query coordinates (node values are finite; bilinear interpolation is
continuous) — but it fails constraints 2 and 4 *for this grid*: the
starved territory is not a thin corner (25 %/60 % of nodes under 100/500,
vs the sky-band case where only z ≲ 0.05 was affected), so the estimator
over much of its domain would be *shaped by the threshold value*, and any
threshold above 10 is a new fitted constant. The estimator is also
discontinuous in the pool statistic: ESS 9.99 vs 10.01 selects entirely
different objects, so pool re-draws flip nodes — instability exactly where
the h-slope of the catalogue term is read.

**Candidate (ii) — bandwidth widening (Abramson generalization to the
product space).** Measured to be insufficient in its implementable form:
the probe ALREADY runs u-Abramson and still floors at ESS = 1.01 — the
starvation is joint-anisotropic and invisible to the 1D u-pilot. The
repairs both fail first principles: a scalar 2D-pilot λ applied to both
axes widens m too, eroding the first-order conditioning and degrading
toward the *pooled* survival (wrong target, constraint 3); an
anisotropic per-axis √-law (widen u only, from the joint pilot) has the
right limit (σ_u → ∞ at fixed σ_m gives exactly S(d_L | M_z)) but no
derivation — Abramson's exponent is derived for the full kernel's density
bias, and a per-axis variant driven by joint ESS creates a
bandwidth↔ESS feedback loop with no literature anchor: an invented
estimator, contra constraint 1's spirit and FIX-2's "no free constants"
discipline.

**Candidate (iii) — ESS-weighted shrinkage toward the first-order
marginal (RECOMMENDED).** Per node, blend the raw joint estimate with the
m-only conditional:

    S̃(d_L | u_a, m_b) = w_ab · Ŝ_joint(d_L | u_a, m_b) + (1 − w_ab) · Ŝ_m(d_L | m_b),
    w_ab = ESS_ab / (ESS_ab + n₀),      n₀ = _MIN_BAND_INJECTIONS = 10 .            (K5)

*Motivation (conjugate pseudo-count — a heuristic mapping, NOT a theorem).*
The node's kernel-weighted survival at any d_L is a weighted mean of
Bernoulli indicators whose variance-equivalent sample size is (K4) (Kish).
Map that to a Bernoulli trial count and put a
Beta(n₀·Ŝ_m, n₀·(1 − Ŝ_m)) prior on the node's survival — mean = the
marginal Ŝ_m, concentration = n₀ pseudo-observations, i.e. "below the
repo's n ≥ 10 reliability floor, the node's own data do not outweigh the
next-coarser conditional". The posterior mean is
(n₀·Ŝ_m + ESS·Ŝ_joint)/(n₀ + ESS) — precisely (K5). Every input is
pool-derived or a reused repo constant; nothing is fitted (constraints
1, 4). Ŝ_m is built with the SAME machinery (kernel weights with the
u-factor ≡ 1, same σ_m, same suffix logic, same DLQ grid) — it is the
current 2D grid's conditioning realized exactly in the joint build, so
no second convention enters.

**Two explicit limitations of this motivation, stated rather than glossed:**

- *It is not the beta-binomial theorem.* The weights w_k are **shared**
  across all d_L and the indicators 1[d_hor_k ≥ d_L] are **correlated**
  across d_L, so applying one Beta posterior per node simultaneously at
  every d_L is a shrinkage *heuristic* with the right limits, not a
  posterior-mean identity. It is adopted for its limits, continuity, and
  zero free constants — that is the whole claim.
- *Kish's ESS is a **variance**-equivalent count; the dominant error at a
  starved (u, m) node is **bias*** — weight borrowed from distant m and u.
  The beta derivation presumes the node estimate is unbiased for that
  node's survival. Since the h-slope of the catalogue term is exactly a
  bias-sensitive functional, (K5)'s scope must be read as
  variance-only, and a **bias diagnostic** (weighted mean |m_k − m_b| and
  |u_k − u_a| per node) is registered alongside ESS in the quality flags
  (§4 item 3).

*Properties (each a §3.7 limiting case / §4 test):*

- **ESS → ∞**: w → 1, S̃ → the raw product estimate.
- **ESS → 0**: w → 0, S̃ → S(d_L | M_z) — the starved limit of the new
  estimator IS the current production 2D conditioning. Note the honest
  form of the claim: the joint grid reduces **to** the existing object in
  that limit; it is **not** "never worse than production" at finite ESS —
  at ESS ≈ n₀ a 50/50 blend of a biased-but-conditioned estimate and a
  coarser one can have larger MSE than either component. The dominance
  claim holds only in the ESS → 0 and ESS → ∞ limits.
- **Numerically-empty node (ESS → 0 exactly).** The probe's
  `build_surv_ulm` divides by `tot` **unguarded** (z2.py:132–134): at a far
  corner node all Gaussian weights can underflow (beyond ≈38σ) and the
  node becomes NaN. Production's z-only guard
  (`_suffix_tables`, sdp.py:699–706) falls back to **uniform** weights,
  i.e. the **pooled** survival — which, copied here, would violate this
  section's own constraint 3 (the fallback target must be Ŝ_m, not
  pooled). **Specification:** if a node's total weight is ≤ 0 or
  non-finite, set w_ab = 0 and S̃ ≔ Ŝ_m(d_L | m_b) — never pooled, never
  NaN. This is a required implementation clause of Z4, not an optional
  guard.
- **Continuity**: w is C^∞ in ESS — no threshold, no cliff; S̃ is C0 in
  (d_L, u, m) by interpolation (given the §3.3-C linear-in-d_L choice; the
  probe's step lookup is not C0 in d_L). At ESS = n₀ the blend is exactly
  50/50 — the repo's reliability floor acquires a continuous meaning
  instead of a switch. Representative weights: w = 0.09 at ESS 1, 0.50 at
  10, 0.91 at 100, 0.98 at 500, **0.955 at the z-only floor 211**
  (211/221; w = 0.995 would require ESS ≈ 1990).
- **Structure preserved**: a convex combination of two monotone-in-d_L,
  [0,1]-bounded survivals with S(0) = 1 and S(> max d_hor) = 0 preserves
  all four guarantees.
- **Relation to (i)/(ii)**: (K5) reproduces the hard fallback in the
  deep-starved limit and the raw estimator in the data-rich limit, and
  interpolates smoothly between — it dominates candidate (i) on
  continuity and (ii) on derivability, at zero additional constants.

Abramson-on-u stays in place beneath the shrinkage (it remains the
correct continuous first line along u, ratified in FIX-2); (K5) handles
exactly the joint-space starvation the pilot cannot see.

**[RATIFY-Z4: starved-node policy = (K5), shrinkage toward S(d_L | M_z)
with w = ESS/(ESS + n₀), ESS per (K4), n₀ = 10 reused unchanged from
`_MIN_BAND_INJECTIONS`, **plus the mandatory empty/underflowed-node clause
(w_ab = 0 ⇒ S̃ = Ŝ_m — never the pooled-uniform fallback that the z-only
guard uses)**; candidates (i) and (ii) rejected on the stated grounds.
Ratification must record that the policy is **load-bearing on the effect
size, not inert**: catalogue-weighted w̄ = 0.834 at the Z2 bandwidth,
98 % of catalogue weight on ESS < 100 nodes, ≈ 17 % attenuation of the
predicted movement — so the shrunk-object column is part of P1, and the
(K5) motivation is a variance-scoped heuristic with a registered bias
diagnostic (§4 item 3). Recommendation: adopt, with the §4 item-7
n₀ ∈ {0, 5, 10, 20} insensitivity rider (n₀ = 0 required to bracket the
rev.-A numbers) — if results depend materially on n₀ the policy re-opens,
so the rider is the honesty check on "no free constants".]**

### 3.5 Consumer scope and the counted-once composition ledger **[RATIFY-Z5]**

**The atomic-switch rule (the counted-once analog in z-composition).**
Within any single likelihood ratio, every p_det factor must be the SAME
conditional object. A mixed state — e.g. Σ_glob_wbh queried joint while
the per-host 2D D_g stays pooled-in-z within one evaluation — re-creates
the composition mismatch at the interface between legs, which is exactly
the defect being removed. Therefore ONE flag gates ALL with-BH p_det
queries; there is no partial adoption.

| Consumer (bs.py) | current query | FIX-3 §7.1 query | load-bearing where (**corrected**) |
|---|---|---|---|
| `precompute_global_catalog_selection`, with-BH (Σ_glob_wbh) | S(d_L(z_g;h) \| M_z,g), isotropic | **S(d_L(z_g;h) \| z_g, M_z,g)**, isotropic (unchanged sky) | **`generator_marginal` `4d_exact` (inside D_gen — divides BOTH channels) AND `absolute_marginal`/`global`/`volume_global` (as `global_denom_with_bh` — divides the 2D channel only)** |
| `_bh_mass_denominator_inner_m_integral(_batch)` (Gaussian-kernel per-host D_g) | erf-sum vs S(d_L(z) \| M_z) at node z | erf-sum vs **S(d_L(z) \| z, M_z)** — see the §3.3-C interpolant caveat | **`local_ratio`/`volume_deconv`/`catalog_only` ratio-of-sums only (`r[3]`); diagnostic in `absolute_marginal` and `generator_marginal`** |
| `_mass_trunc_denominator_inner_m_integral(_batch)` (LN×R_eff per-host D_g) | GL nodes query S(d_L(z) \| M(1+z)) | GL nodes query **S(d_L(z) \| z, M(1+z))** (pointwise — trivial switch) | **`mass_trunc` ratio-of-sums only; diagnostic elsewhere** |
| — σ_z smear variants (opt-in) | E_z[S(d_L(z;h) \| M_g(1+z))] | E_z[S(d_L(z;h) \| z, M_g(1+z))] — smear z ≡ conditioning z ≡ lift z, one z per query | opt-in |

**Erf-sum on the joint grid — NOT a free ride (corrected).** At a fixed
query (d_L, u) the stored survival is a linear blend of the two bracketing
u-nodes' m-profiles on the SAME m-knots, so the blend is piecewise-linear
**in m = log₁₀M_z**. The erf-sum closed form (bs.py:3149–3175) is exact
only for an interpolant piecewise-linear **in M_z**. Linear-in-log-M is
*not* linear-in-M, so the previous claim ("the exact erf-sum closed form
applies with node-blended knot values") is **false as stated**. §3.3-C
requires an explicit convention choice — recommended: interpolate linear
in M_z between the lifted knots 10^{m_j}/(1+z), keeping the erf-sum exact
and accepting a documented `q_ulm`-parity delta in P1. This is a
**method-level convention change**, not an implementation detail; the
scalar-vs-batch bit-identity twins guard path equality, not interpolant
validity.

**Grid-convention confound (must be controlled, §4 item 12).** Flag-ON also
changes the mass grid: production uses 40 **geomspace-in-M_z** centres over
×0.9…×1.1 of the pool span with 60 linear d_L bins (`_grid_support`,
sdp.py:1115–1135); the proposed joint grid uses 31 **uniform-in-log₁₀M_z**
nodes and 3000 d_L points. Hence flag-ON ≠ flag-OFF at query level even in
the σ_u → ∞ limit (§3.7 case 1 asserts equality only "at the construction
level on matched grids"). A **grid-only control cell** — joint build with
the u-kernel disabled — is required so the A/B separates the grid/
interpolant change from the conditioning change.

**What does NOT change (explicit ledger):**

- **D** and **β_Ḡ** — z-only remains EXACT by (K2) (their measure is the
  population's own); adding the m axis there is a no-op in expectation
  and pure variance.
- **Σ_glob no-BH** and the 3D per-host D_g — z-only (FIX-2), untouched.
- **B_num and every numerator N_g** — no p_det (MFG); untouched. The 2D
  numerator's mass marginal mz (ratified (M1) kernel) carries no p_det
  and is untouched.
- **n̂_w** — p_det-free; untouched.
- **The 1D channel's own legs** — L_cat_without_bh, B_num, β_G, β_Ḡ, D,
  Σ_glob-no-BH: no with-BH query anywhere, all bit-identical under the
  flag. **CORRECTION (was: "the 1D channel in its entirety"):** the 1D
  *profile* is NOT invariant in `generator_marginal`. There, one D_gen =
  Σ_glob_wbh/n̂_w + β_Ḡ divides both channels (bs.py:3077–3078), so
  flipping the flag shifts the 1D profile by exactly the same per-event
  amount as the 2D profile. Predicted B-cell 1D shift at 0.86:
  −N·Δ[ln D_gen(0.86) − ln D_gen(0.73)] = 3454 × 0.004511 ≈ **−15.6 ln**,
  computable from the existing tables. In `absolute_marginal` the 1D
  channel divides by `global_denom_no_bh` and IS bit-identical. This is
  why P4(i) must be restricted to `absolute_marginal` and `3d_shared`.
- **Sky treatment of the with-BH branch** — stays isotropic
  (bs.py:1560–1565); no sky×z×M_z variant is proposed (triply starved).
- **The M_z-only 2D grid** `_build_grid_2d` — retained as-is: it is the
  flag-OFF path, the shrinkage target Ŝ_m's convention twin, and the
  σ_u → ∞ limit.
- **Mode note:** in generator_marginal the per-host D_g is
  diagnostic-only — it switches too (diagnostics must not report a
  different convention than the load-bearing legs), but the load-bearing
  movement there is Σ_glob_wbh's. Under `dgen_catalog_selection =
  "3d_shared"` there is no with-BH catalogue term and the flag changes
  nothing in D_gen.

**[RATIFY-Z5: scope as tabled — all with-BH p_det queries switch
atomically under one flag; the corrected load-bearing column and the
corrected not-changed ledger above are part of the gate. Ratification
requires acknowledging that (i) Σ_glob_wbh is load-bearing in
`absolute_marginal` as well as `generator_marginal` (the earlier table was
inverted), (ii) the per-host 2D inner-M integrals are load-bearing only in
the local ratio-of-sums modes, and (iii) the 1D profile is NOT invariant in
`generator_marginal` + `4d_exact`. Recommendation: adopt as corrected.]**

### 3.6 Dimensional analysis (every defined object)

- u, m, σ_u, σ_m, λ_k, kernel weights w_k, ESS, n₀, w_ab: dimensionless. ✓
- d_hor, d_L, DLQ nodes: [Gpc]. ✓
- S(d_L | z, M_z), Ŝ_m, S̃: ratios of weight sums — dimensionless ∈ [0,1]. ✓
- Σ_glob_wbh = Σ_g w_g·S̃: w_g = R_eff(M_g)/(1+z_g) [Gyr⁻¹] × dimensionless
  → [Gyr⁻¹], identical units to today (only the S factor changes). ✓
- D_gen = Σ_sel/n̂_w + β_Ḡ: commensurable exactly as in FIX-3 §3.4 /
  FIX-2 §4.4 — no unit structure changes anywhere. ✓
- Per-host D_g inner integrals: p_det ∈ [0,1] against normalized mass
  priors → dimensionless, as today. ✓
- h-bookkeeping: S̃ is built from h-free (d_hor_k, u_k, m_k); h lives only
  in the query d_L(z; h) and in dV_c — no new h channel (§3.3). ✓

### 3.7 Limiting cases (minimum set; each becomes a test in §4)

**Scope note (important for writing the tests).** Cases 1–3 are statements
about the **pre-shrinkage** kernel estimate Ŝ_joint (K3). The **shipped**
accessor returns S̃ (K5). Where a limit is claimed for the shipped object,
the exactness condition is stated explicitly, and where exactness is lost
the bound |S̃ − limit| ≤ (1 − w)·|Ŝ_joint − Ŝ_m| applies. Tests written
from the earlier verbatim text would have failed against the shipped
accessor.

1. **σ_u → ∞ (z-kernel width → ∞):** the u-factor → 1 for every
   injection; (K3) → the m-kernel-only survival — the CURRENT pooled-z 2D
   grid's *construction*. Exactness conditions, all required: same σ_m
   (holds after Z2), same suffix logic, **and matched (d_L, m) grids** —
   which do NOT hold against production (60 linear d_L × 40 geomspace-M_z
   vs 3000 d_L × 31 uniform-log-M). So the test asserts equality against a
   **matched-grid rebuild**, not against the live production interpolator;
   the grid-only control cell (§4 item 12) measures the residual query-level
   difference. Here the shrinkage IS exactly inert (Ŝ_joint ≡ Ŝ_m ⇒
   S̃ ≡ Ŝ_m for any w).
2. **σ_m → ∞ (mass-kernel → marginal):** the m-factor → 1; **Ŝ_joint**
   recovers FIX-2's S(d_L | z) with the same u-machinery. Exact equality
   against `_zres_survival_at` requires *all three*: matched σ_u, **matched
   u-nodes** (the joint 61-node grid is a subset of production's 121-node
   grid only at shared nodes — assert there, or interpolate and bound),
   and matched d_L handling (§3.3-C). **S̃ does NOT satisfy this limit:**
   as σ_m → ∞, Ŝ_m → S_pooled, so
   S̃ → w·S(d_L|z) + (1 − w)·S_pooled, which equals FIX-2's object only at
   w = 1 (worst node w ≈ 0.51 at the Z2 bandwidth; catalogue-weighted
   w̄ ≈ 0.83). Test Ŝ_joint for equality and S̃ against the (1 − w) bound.
3. **Both widths → ∞:** all weights → 1 → **Ŝ_joint** = the pooled
   survival `_survival_at`, exactly; S̃ ≡ pooled too, since Ŝ_m → pooled
   as well (both blend components coincide).
4. **ESS → ∞ at a node** (dense synthetic pool): w → 1, S̃ → the raw
   product estimate — the estimator is unshrunk exactly where it is
   measurable.
5. **ESS → 0 at a node** (synthetic pool with an empty (u, m) corner):
   S̃ → Ŝ_m(d_L | m_b) — the current 2D conditioning; never worse than
   production.
6. **Policy continuity:** w = ESS/(ESS + n₀) is smooth in ESS — no
   threshold exists, so the "C0 at threshold" requirement is met by
   construction; test monotone continuity of S̃ under pool subsampling
   (ESS varied continuously through n₀), and |S̃ − Ŝ_joint| ≤
   (1 − w)·|Ŝ_joint − Ŝ_m| at every d_L.
7. **d_L clamp semantics:** below the grid → 1-clamp; above → exact 0;
   m and u → nearest/span clamps — bit-parity with the existing wrapper
   conventions (A2-EXTRAP, `_zres_node_pos`).
8. **h-invariance of the build:** tables built once are byte-identical
   regardless of which h is queried first or at all (hash test); only
   queries move with h.
9. **Degenerate pool** (n < 2, std(u) ≤ 0, or std(m) ≤ 0): collapse to
   the corresponding marginal (m-only, u-only, or pooled) — mirror of
   `_zres_degenerate`; the flag must not crash a degenerate pool.
10. **Saturated support (seed600 shallow venue):** with *Gaussian* kernels
    every injection carries non-zero weight at every node, so the
    "window" is the whole pool and exact identity
    Ŝ_joint = Ŝ_m = S_pooled = 1 requires
    **d_L ≤ min over ALL injections' d_hor** — the global minimum, not
    "every local horizon" (an inherited looseness from FIX-2 §4.5-iv).
    State the case in that global-min form; outside it, the claim is
    "unchanged within σ_boot", which is what the shallow A/B actually
    tests.

### 3.8 Scale reconciliation — the crux, rev. B **[RATIFY-Z6]**

The honest quantitative position, stated before any A/B. **This section was
rewritten 2026-07-27 after adversarial review: the previous version was
centred on the wrong axis and quoted a mislabelled ratio.**

**What is actually tabulated.** The probe's gap arithmetic
(`z2_results.json:gap_predictions`) is a linear per-event extrapolation,
gap = baseline + N_EVENTS·(dlnD_gen^3D − dlnD_gen^new), N_EVENTS = 3454
(z2.py:48), on the FIX-3 generator-consistent 3D baseline of +92 ln (itself
a linear extrapolation, its §6.3; est. tolerance ±15 ln, FIX-2 §8 risk 2):

| catalogue term in D_gen | Σ_cat(0.73) | slope | dlnD_gen(0.73→0.86) | full-mixture gap |
|---|---|---|---|---|
| pooled | 5.122e8 | +0.358 | −0.18274 | +19.1 |
| z-only (`zres`) | 4.359e8 | +0.505 | −0.17102 | **−21.4** |
| **M_z-only (`mz`) = TODAY'S PRODUCTION (`4d_exact`)** | 3.018e8 | **+0.364** | −0.15888 | **−63.3** |
| joint (`z_mz`) = THIS PACKET | 2.798e8 | +0.106 | −0.15437 | **−78.9** |

**The crux, corrected.** The previous text claimed the M_z-only
catalogue-term slope "was never tabulated in isolation". That is **false**:
`z2_results.json:Sigma_glob.mz` *is* the M_z-only construction (built via
`q_lm`, z2.py:330), with v073 = 3.0182e8, slope = **+0.36426**,
dln_073_086 = +0.042395. Assembling D_gen_mz = Σ_mz/n̂_w + β̄_Ḡ,zres at the
0.73/0.86 endpoints of the same tables gives, with no new compute:

- **production baseline (M_z-only catalogue term): −63.3 ln** — a number
  the packet never previously stated;
- production + joint conditioning: **−78.9 ln**;
- **production-axis increment: −15.6 ln.**

The −57.5 ≈ −58 ln quoted throughout FIX-2 and the mass-kernel readout is
the **(z-only → joint)** increment, on an axis production does not
occupy.

**Consequences, all of which reverse the previous framing:**

1. Against a **+23.8…+29.1 ln** 2D residual, the honest central
   expectation is an **undershoot** — the increment covers roughly 60 % of
   the residual — landing on the CLOSURE-SCALE / MATERIAL-BUT-INSUFFICIENT
   boundary. There is no ~2× overshoot; overshoot lives only on the
   never-to-be-run z-only axis. P5's OVERSHOOT-first ordering was
   calibrated on the wrong axis and is re-centred in rev. B.
2. The dilemma "both cannot be fully right" **resolves quantitatively**:
   the fixed-M_z z-residual is worth ≈ −15.6 ln, i.e. about half the
   residual. The partial-corr-0.035 reading and the elimination reading are
   *both partly right* — which is exactly the branch-(d1)/(d2) split of
   §0. There is no paradox to resolve empirically before the A/B.
3. After (K5) shrinkage the expected movement attenuates by ≈ 1 − w̄ ≈ 17 %
   (§3.4), to ≈ **−13 ln**. That is the number the A/B will actually see.

**Calibration of the extrapolator (new gate condition).** The same linear
machinery predicts **−63.3 ln** for exactly the stack cell B″ ran
(`generator_marginal`, FIX-2 default-on, `4d_exact`, main @ `e9bec6d`),
while B″'s **measured** 1D gap is **−86.1 ln**
(`MASS_KERNEL_AB_READOUT.md`) — a **22.8 ln** prediction error, larger
than the previously stated ±20 tolerance and larger than the −15.6 ln
increment P2 is supposed to gate. Two observations make this tractable:

- The error is in the **baseline**, not the mechanism: back-solving from
  B″'s measured gap gives an implied generator-stack baseline of ≈ **+69.2
  ln**, not +92. (On the `absolute_marginal` axis the same extrapolator is
  accurate: it predicts −69.4 for A′/A″ and −69.4 was measured.)
- Because the baseline **cancels in a difference**, the *increment* is
  robust: −15.6 ln is baseline-independent. Recalibrating on B″'s measured
  gap gives a production-axis prediction of −86.1 → **≈ −101.7 ln**
  (unshrunk) or ≈ −99 ln (shrunk).

**Therefore: P2 gates on the INCREMENT, not on an absolute gap.** The
absolute-gap form is demoted to a direction-only diagnostic.

**Recommended position (rev. B).** The joint conditional (K1) is the
*derived* completion of FIX-2 — correct independent of the A/B outcome (the
measure-match rule stands on (K2) and the measured composition mismatch
alone, subject to the §3.1-B assumption and the §0-(g1) support caveat).
Its *sufficiency* to close the 2D channel is **not** claimed: it addresses
branch (d1) only; branch (d2) (selection-side M scatter/truncation,
RATIFY-M5) is untouched and, on the arithmetic above, the two together are
the natural joint owner of the +25 ln residual. Adoption is therefore:
implement behind the default-OFF flag; run the **A-cell** A/B (the only
2D-specific arm) plus the B-cell as a shared-shift control; score against
the §3.9 rev. B partition; no production default flip except through the
FIX-2/FIX-3 joint ship gate (FIX-2 §6: "must ship and be gated TOGETHER").

**[RATIFY-Z6 (rev. B): adopt the position "derived completion of FIX-2 for
branch (d1) only; production-axis increment −15.6 ln unshrunk / ≈ −13 ln
shrunk; UNDERSHOOT pre-registered as the central expectation; branch (d2)
explicitly still open; A-cell is the 2D-specific arm, B-cell is a
shared-normalization control; P2 gates on the increment because the +92
baseline is measured 22.8 ln off; default flip only at the joint ship
gate". Recommendation: adopt in this rev.-B form; the rev.-A form
(overshoot-first, −79 gate value, z3 as a mandatory unknown) is
**withdrawn**.]**

### 3.9 Pre-registered predictions, rev. B (BEFORE any A/B)

**Rev.-B note.** Rev. A's P1 gate value was a mislabelled ratio, P2 was
stated on the wrong axis with a demonstrably 22.8-ln-off baseline, P3's
sub-prediction rested on an inverted scope table, P4(i) was false by
construction in `generator_marginal`, P5 was centred on overshoot, and all
six were written for the **unshrunk, N^(−1/5)** probe object rather than
the **shrunk, N^(−1/6)** object that Z2+Z4 ratify. Rev. B restates them for
the object that ships.

- **P1 (table level, no posterior run).** The production joint build,
  assembled over the catalogue, reproduces the probe. Ratios must be
  reported with their denominators named:
  - Σ_cat^joint/Σ_cat^**z-only**(0.73) = **0.642 ± 0.02**
  - Σ_cat^joint/Σ_cat^**M_z-only**(0.73) = **0.927 ± 0.02**  ← the
    production-axis value ratio
  - Σ_cat^joint/Σ_cat^**pooled**(0.73) = **0.546 ± 0.02**  ← the number
    previously mislabelled as "joint/z-only"
  - Σ_cat^**M_z-only**/Σ_cat^**pooled**(0.73) = **0.589**; this is the
    like-for-like twin of production's `Σ_glob_wbh/Σ_glob = 0.556`, and
    the **6 % gap between them is an open parity item**, pre-registered
    as such (contrast the 3D channel's 4×10⁻⁵ parity). If the production
    build reproduces 0.589 rather than 0.556, the 2D-channel probe/
    production parity is confirmed and the discrepancy is a production-grid
    artefact; if it reproduces 0.556, the probe's binned profile is the
    outlier. Either way the 6 % must be explained before P2 is scored.
  - catalogue-term slope d ln Σ/dh(0.73) = **+0.106 ± 0.05** unshrunk
    (vs +0.505 z-only, **+0.364 M_z-only**, +0.357 pooled)
  - assembled d ln D_gen/dh(0.73) = −1.345 ± 0.03 (vs −1.454 z-only-stack)
  - P̂(cat|det)(0.73) ≈ 0.135 ± 0.01
  - **Shrunk-object column (REQUIRED, no probe number exists yet):** the
    same four quantities recomputed with (K5) at n₀ = 10 and
    σ_u = N^(−1/6), plus n₀ = 0 for continuity with the rev.-A numbers,
    plus the **catalogue-weighted mean shrinkage weight w̄** (predicted
    **0.834 ± 0.02**) and the catalogue W-fraction on ESS < 100
    (predicted **≈ 0.98**). Expected attenuation of the slope movement:
    ≈ 17 %.
  - Binned-profile-vs-row-by-row parity at the ~10⁻⁴ level applies to the
    3D channel (FIX-2 §2: 4×10⁻⁵ value, 6×10⁻⁴ slope); **no such parity
    is claimed for the 2D channel** pending the 0.589-vs-0.556 item.
- **P2 (generator-stack gap — INCREMENT form; the absolute gap is
  direction-only).** The gate quantity is the **production-axis
  increment**, which is baseline-independent.
  **[z3 REFINEMENT, 2026-07-27 — the mandatory §4 item-1 tabulation, run
  BEFORE any A/B (artifacts `zres_survival/z3_production_axis.py` /
  `z3_results.json`; all twelve reviewer verification targets reproduced
  exactly).] The rev.-B "≈ −13 ln shrunk" sanity figure was
  apples-to-oranges: it applied the Z2-bandwidth attenuation (w̄ = 0.834)
  to the PROBE-bandwidth unshrunk increment (−15.6). The bandwidth switch
  is itself value-side load-bearing: at the ratified Z2 bandwidth the
  unshrunk increment is −8.34 ln, and the SHIPPED object (Z2 bandwidth +
  (K5) shrinkage, per-cell w) gives**
  Δgap(M_z-only → shipped) = **−6.47 ln** (decomposition: −15.58 probe-bw
  unshrunk → −8.34 Z2-bw unshrunk → −6.47 shipped; per-cell-w blend vs
  the uniform-w̄ estimate −6.96, 7 % apart). **The P2 gate value is
  −6.5 ± 4 ln** (table noise + linearization; the rev.-B −15.6/−13
  figures are retained above as the unshrunk/mislabeled history, not the
  gate).
  For orientation only, not as a gate: the absolute assembled gaps are
  −63.3 (production baseline) → −78.9 (joint) on the +92-ln FIX-3
  baseline; recalibrated on B″'s **measured** −86.1 the same increment
  gives ≈ −101.7 (unshrunk). The +92 baseline is measured **22.8 ln off**
  for the B-cell stack, which is why the absolute form cannot gate.
- **P3 (four-cell venue), split by normalization — the two cells measure
  DIFFERENT things.** Cells A‴/B‴ ≔ A″/B″ + `--pdet_wbh_z_resolved`
  (trunc_lognormal mass kernel held fixed).
  - **A-cell (`absolute_marginal`) — the 2D-specific arm.** Σ_glob_wbh
    divides the 2D channel only; the 1D channel divides by
    `global_denom_no_bh` and is bit-identical. Direction: 2D profile moves
    **DOWN (toward truth) at every h > 0.73**. Central expectation
    **[z3-refined]**: Δ(2D ln @0.80) ∈ **[−9, −3]** (the shipped −6.5
    increment re-expressed at 0.80 rather than as a 0.73→0.86 gap;
    rev.-B's [−20, −7] band was built on the mislabeled −13); post-fix
    2D residual @0.80 ∈ [+15, +21], i.e. **branch (d1) closes only a
    quarter-to-third of the A″ residual** — branch (d2)/(g1) carry the
    majority expectation.
  - **B-cell (`generator_marginal`) — a shared-normalization CONTROL, not
    a discriminator.** D_gen divides both channels, so the flag moves 1D
    and 2D by *identical* per-event amounts and the **2D-minus-1D residual
    is invariant**. Prediction: Δ(2D ln @0.80) = Δ(1D ln @0.80) to
    within floating-point; the B-cell's 2D residual **does not improve**.
    A B-cell 2D-minus-1D change is a hard falsifier of the §3.5 scope
    ledger.
  - **Withdrawn:** rev. A's "A-movement ≈ B-movement within ~5 ln, else
    Z5 atomicity re-opens". Both cells route through the same
    Σ_glob_wbh scalar, so agreement is near-tautological in magnitude
    while the *mechanisms* differ in kind; a single magnitude band for
    both cells is unjustified.
- **P4 (no-ops, each a hard falsifier if violated).**
  (i) **1D profiles bit-identical in `absolute_marginal` cells and under
  `dgen_catalog_selection="3d_shared"` only.** In `generator_marginal` +
  `4d_exact` the 1D profile MUST move, by the pre-registered
  −N·Δ[ln D_gen(h) − ln D_gen(0.73)]; at h = 0.86 that is
  **−15.6 ± 5 ln** (unshrunk). A bit-identical B-cell 1D profile is
  itself a falsifier — it would mean the flag never reached D_gen.
  (ii) flag OFF is byte-identical to the current stack (golden
  discipline); (iii) seed600 shallow venue unchanged within σ_boot,
  |ΔMAP| ≤ 1 grid step (case 10, global-min form); (iv) build tables
  h-invariant (case 8).
- **P5 (outcome partition on Δ ≔ post − pre **A-cell** 2D ln at 0.80,
  trunc_lognormal; contiguous, every region assigned). Re-centred on
  undershoot.**
  - **Δ ≥ +3 — UPWARD:** falsifier; §3.1–§3.4 re-open.
  - **−3 < Δ < +3 — NULL:** the production-axis increment is genuinely
    small. This does **not** imply an unenumerated owner: branch (d2)
    (selection-side M scatter/truncation, RATIFY-M5) is enumerated,
    ratified as deferred, and untouched by this packet; and (g1) (81 % of
    catalogue weight on the mass-clamp boundary) is a live alternative.
    The estimator derivation stands regardless; only the (d1)-sufficiency
    reading dies. Consistency check against P1: if P1's table movement was
    large and the A/B is null, probe/production parity is broken — an
    implementation bug, not physics.
  - **−9 < Δ ≤ −3 — CENTRAL / AS-PREDICTED [z3-refined]:** (d1)
    confirmed at the shipped-object scale (−6.5 ± 4), covering roughly a
    quarter-to-third of the +23.8 A″ residual. 2D stays OPEN pending
    (d2)/(g1); the FIX-2/FIX-3 joint ship gate may proceed for the (d1)
    arm only. This is the **expected** outcome. (Rev.-B's band labels
    "MATERIAL BUT SMALL"/"CENTRAL" at [−7,−3]/[−20,−7] are superseded by
    this z3-refined split; the region algebra below is unchanged.)
  - **−20 < Δ ≤ −9 — ABOVE-PREDICTED:** larger than the shipped-object
    arithmetic; check the P1 table parity and the per-cell shrinkage
    weights before interpreting as (d1)+part-of-(d2).
  - **−30 < Δ ≤ −20 — OVER-PREDICTED CLOSURE:** larger than the
    increment arithmetic supports; if the post-fix residual lands within
    ±5 of 0 and 2D MAP = 0.73, record full closure but flag the
    arithmetic gap for audit (linearization or the §3.1-B β_Ḡ asymmetry).
  - **Δ ≤ −30 or post-fix 2D MAP < 0.73 — OVERSHOOT:** ≈ 2× the
    increment arithmetic; halt adoption and audit the Z5 ledger, the
    §3.3-C grid/interpolant confound (via the grid-only control cell),
    and the §3.1-B symmetric-β̄ control before any further claim.
- **P6 (structural).** **Withdrawn as worded** ("shrinkage inert where
  measurable, w > 0.95 wherever ESS > 190"): the ESS > 190 implication is
  arithmetically correct but vacuous, because 98 % of the catalogue's rate
  weight sits on nodes with ESS < 100 and the catalogue-weighted w̄ is
  **0.834**, not > 0.95. Restated: **the shrinkage is load-bearing on the
  effect size (≈ 17 % attenuation, pre-registered in P1/P2), and the P5
  verdict class must nevertheless be stable** under the §4 item-7 riders
  (bandwidth_scale ×0.5/×2, n₀ ∈ {0, 5, 10, 20}, node and DLQ doubling,
  u-exponent swap). If the verdict class changes under any rider, the
  estimator is starvation-dominated and RATIFY-Z4 re-opens — a gate
  condition, not a tolerance footnote. (Measured at the Z2 bandwidth the
  node-ESS floor improves only from 1.013 to 1.042; no gate rides on it.)

## 4. Validation plan (after ratification)

1. **z3 probe extension (still required, but its former headline is
   already answered).** The M_z-only catalogue-term column **exists**:
   `z2_results.json:Sigma_glob.mz` (built via `q_lm`, z2.py:330),
   v073 = 3.0182e8, slope = +0.36426, dln_073_086 = +0.042395; assembling
   D_gen_mz = Σ_mz/n̂_w + β̄_Ḡ,zres at the 0.73/0.86 endpoints gives the
   production baseline **−63.3 ln** and the production-axis increment
   **−15.6 ln** (§3.8 rev. B). No run is needed for that. What z3 *must*
   still deliver, because no number exists for the shipped object:
   (a) the **shrunk** ((K5), n₀ ∈ {0, 5, 10, 20}) joint column at the
   **Z2 σ_u = N^(−1/6)** bandwidth — value, slope, catalogue-weighted w̄,
   and the attenuated increment; (b) the 31-node joint grid vs the 41-node
   `LM_NODES` M_z-only grid at matched nodes, so the value/slope
   comparison is like-for-like; (c) the 0.589-vs-0.556 parity item of P1;
   (d) the §3.3-C convention deltas (step vs linear in d_L; linear-in-m vs
   linear-in-M_z). Append the dated numbers to §3.9-P1/P2 BEFORE any A/B
   cluster job is submitted.
2. **Flag design [RATIFY-Z7].** `--pdet_wbh_z_resolved`
   (`argparse.BooleanOptionalAction`, **default False**), mirrored
   through the full 6-hop `--pdet_z_resolved` chain (arguments.py
   property → main.py `main()` → `evaluate()` → `BayesianStatistics.
   evaluate()` → `SimulationDetectionProbability(...)` kwarg).
   **Guard:** `pdet_wbh_z_resolved=True` with `pdet_z_resolved=False`
   raises `ValueError` — a joint-conditioned 2D channel over pooled 3D
   legs mixes conventions inside D_gen (the Z5 rule at mode level, and
   the FIX-2 ship-together rule in code). Accessor: the 2D query gains
   an optional `z` argument, `_require_zres_z`-style mandatory when the
   flag is on. Default flip to True is a SEPARATE, later decision at the
   FIX-2/FIX-3 joint ship gate — not part of this packet.
   **[RATIFY-Z7: recommendation — adopt as specified.]**
3. **Golden discipline + quality flags.** Flag OFF byte-identical
   (pipeline-parity golden); flag-ON goldens introduced as new files,
   never regenerated over old ones; quality flags `wbh_zres_nodes`,
   `wbh_zres_ess_min`, `wbh_zres_shrunk_frac` (fraction of nodes with
   w < 0.5), **`wbh_zres_wbar_cat`** (catalogue-weighted mean shrinkage
   weight — the P1/P6 quantity), and the **bias diagnostic**
   `wbh_zres_bias_m` / `wbh_zres_bias_u` (per-node weighted mean
   |m_k − m_b| and |u_k − u_a|, §3.4: ESS is variance-only) registered
   per-h for API parity (values h-invariant, as today). Also register
   **`wbh_mclamp_wfrac`** — the catalogue rate-weight fraction served by
   the m_max clamp (expected ≈ 0.814, §0-(g1)).
4. **Limiting-case tests** per §3.7 (10 cases), plus scalar-vs-batch
   bit-identity on the switched erf-sum/GL paths (existing twin-test
   pattern), plus the degenerate-pool no-crash test.
5. **Four-cell rerun (A‴/B‴) + seed600 no-op**, scored against the
   §3.9-P3/P4/P5 partition BEFORE any production claim (the
   volume_trunc regression-gate lesson). **The A-cell is the scoring arm
   (2D-specific); the B-cell is scored as a shared-shift control** —
   report Δ(1D) and Δ(2D) separately in both cells and verify
   Δ(1D) ≡ Δ(2D) in B. Per-leg ln attribution as in the mass_ab readout.
6. **Generator-stack gate.** The FIX-2+FIX-3 joint stack with the joint
   catalogue term, gated on the **increment** (P2: −15.6 unshrunk /
   ≈ −13 shrunk, ±5), **not** on an absolute gap — the +92 baseline is
   measured 22.8 ln off for this stack (§3.8). Mechanism metrics:
   catalogue-term slope ≈ +0.11, D_gen slope ≈ −1.35, P̂(cat|det) ≈ 0.135
   (P1). This is the ship-together gate FIX-2 §6/§8-risk-2 requires; this
   packet supplies its with-BH composition arm.
7. **ESS-policy and estimator stress riders** (FIX-2 §9.4 pattern):
   bandwidth_scale 0.5/2.0 (both axes, via the existing constructor
   knob); **n₀ ∈ {0, 5, 10, 20}** — n₀ = 0 is required because every
   rev.-A number was computed at effective n₀ = 0 and the rider must
   bracket it; u-exponent N^(−1/5) vs N^(−1/6); node doubling
   61×31 → 121×61; DLQ 3000 → 6000. Gate condition: the P5 verdict class
   must be rider-stable (P6).
8. **Memory/worker benchmark.** Measured table sizes (~45 MB expected)
   and per-worker pickle cost vs the existing ~100 MB z-resolved tables;
   build wall-clock (seconds expected).
9. **Coordination.** Branch d is **split**: this packet closes (d1)
   z-composition only; **(d2) selection-side M scatter/truncation
   (w_g → Z_M, p_det point-in-M) remains OPEN under RATIFY-M5** and must
   be recorded as such in `mass_marginal_2d_kernel.md` (RATIFY-M6
   discriminators, §3.8 branch d, §4 item 8) rather than marked closed.
   The FIX-2 packet's §5.2/risk-4 deferral is discharged for the
   catalogue sum; §0-D records that FIX-2 §3.4's main-text decision for
   the with-BH **per-host** integrals is **partially superseded** here,
   not merely deferred. The "z-resolved survival" quality-flag docs gain
   the with-BH variant. GitHub: the A/B outcome decides whether
   #40-adjacent 2D-channel issues close, or issues open for (d2) and for
   the (g1) mass-support mismatch.
10. **Symmetric-β̄ control cell (MANDATORY, §3.1-B).** Re-run the
    generator-stack assembly with β_Ḡ carrying the same composition
    conditioning as the catalogue term, and report the gap alongside the
    asymmetric variant. The measured sensitivity is ±60 ln — larger than
    the effect being gated — so no gate may be scored without it. If
    p_missing(M_z|z) can be derived (completeness-selected complement of
    the catalogue mass function), do that instead and redo §3.8.
11. **Mass-support control (MANDATORY, §0-(g1)).** Restrict Σ_glob_wbh to
    the pool-supported mass range log₁₀M_z ≤ 6.0 (or bound the clamp's
    contribution analytically) and report the resulting value ratio and
    slope. Nothing may be attributed to "composition" until the
    81.4 %-of-weight boundary-clamp contribution is separated from it.
    Escalate the pool-vs-catalogue mass-population mismatch (pool M from
    the Barausse M1 sampler, `cosmological_model.py:337`; catalogue M
    from Reines–Volonteri) as its own gate item.
12. **Grid-only control cell (MANDATORY, §3.5).** Build the joint grid
    with the u-kernel disabled (σ_u → ∞) and run the A-cell with it. This
    isolates the grid/interpolant change (31 uniform-log-M nodes +
    3000-point d_L + linear-in-m blend, vs 40 geomspace-M_z centres + 60
    linear d_L bins) from the conditioning change. Without it the A/B
    confounds the two.

## References

- FIX-2 packet: `results/lcat_h_dependence_20260725/DERIVATION_ZRESOLVED_SURVIVAL.md`
  (§3.4, §5.2, §6, §7, §8, §9) + `zres_survival/z1_results.json`,
  `z2_results.json`, `catalog_zw_profile.json`, `z2_zres_slopes.py`.
- Elimination record: `results/lcat_h_dependence_20260725/mass_ab_20260727/MASS_KERNEL_AB_READOUT.md`.
- Mass-marginal kernel (ratified): `docs/derivations/mass_marginal_2d_kernel.md`
  (§3.4 site 3, §3.8 branch d, §4 item 8 — the coordination contract).
- Host-z kernel (ratified): `docs/derivations/hostz_pv_photoz_kernel.md`.
- Production code: `master_thesis_code/bayesian_inference/simulation_detection_probability.py`,
  `master_thesis_code/bayesian_inference/bayesian_statistics.py`.
- Finn & Chernoff 1993, arXiv:gr-qc/9301003; Finn 1996,
  arXiv:gr-qc/9601048; Mandel, Farr & Gair 2019, arXiv:1809.02063,
  Eqs. (6)ff; Hogg 1999, arXiv:astro-ph/9905116, Eq. (16); Scott 1992,
  *Multivariate Density Estimation*, Ch. 6; Abramson 1982, Ann. Statist.
  10:1217; Kish 1965, *Survey Sampling* (ESS convention; original not
  re-opened); Gelman et al., BDA3 §2.4 (conjugate pseudo-count form —
  standard result, derived inline in §3.4).
