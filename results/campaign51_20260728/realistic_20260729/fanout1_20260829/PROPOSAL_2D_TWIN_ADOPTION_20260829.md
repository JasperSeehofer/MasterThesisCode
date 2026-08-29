# PHYSICS-CHANGE PROPOSAL — the with-BH catalogue-leg twin to production: `catalogue_numerator_survival_2d="mz_sel"` (centre = `eff`)

**Launched under rows #222/#223 — charter node B7.1.** `[FABLE-B7.1 2026-08-29]`

**Date:** 2026-08-29 · **Status:** PROPOSAL — reviewable, NOT the `/physics-change` gate; no code
is changed by this document · **Subject:** `darksiren_emri/bayesian_inference/bayesian_statistics.py`
(trigger file; the flag, both kernels and the tests already exist — this proposes the DEFAULT flip)
· **Branch/HEAD at authoring:** `fix/p32d-classg-venue-repair` @ `a794404c` · **Form:** mirrors
`docs/derivations/PROPOSAL_CATALOGUE_TWIN_PRODUCTION_20260825.md` (the 1D twin's adoption,
row #195 → `[PHYSICS]` `bac48696`, row #197) section by section.

**Evidence chain:** rows #189–#212 (`gate_b_20260730/BIAS_HISTORY_LEDGER.md:2830–2996`) and #216
(`:3006`); `P3_2D_REPAIR_READOUT_20260828.md` (CONFIRMED at 33 seeds, §7, ratified row #216 item 1);
`PREREGISTRATION_P3_2D_20260825.md` (+PA-2D-1..10) and `PREREGISTRATION_P3_2D_REPAIR_20260827.md`
(v2, PA-2DR-1..15); `A20_REVIEW_P3_2D_DESIGN_20260825.md` F2 (the centering ruling); the selection
fusion gate (`docs/derivations/GATE_PRESENTATION_SELECTION_FUSION_20260817.md`, rows #117–#118);
the parked residual (row #211; `p32d_residual_accounting_20260827.md`; `STUCK_P3_2D_SYMPTOM_CARD_20260826.md`).

**Authorization form (row #223, verbatim ruling quoted there):** production-default flips inside
the charter tree are covered by the [STANDING] grant; the 5(+1)-item gate presentation is still
authored BEFORE code, the three PHYSICS-GATE-LEDGER rows are still filed, the APPROVED column
cites "row #223", and this gate is in the end-of-fan-out verifier's mandatory scope. This
proposal is the reviewable artifact that gate is built from (CLAUDE.md "Proposing decisions").

**Every number below carries {value, source file:line, date} (A11); §12 is the provenance table. Code line cites are against the COMMITTED blob `a794404c:darksiren_emri/bayesian_inference/bayesian_statistics.py` — the working tree was under concurrent edit by wave-1 siblings (B5.1/B6.1) while this was authored, so working-tree line numbers may differ by a few lines.**

---

## 1. The per-event 2D catalogue-numerator formula — now vs proposed (item 1 + item 2)

### 1.1 Symbols (all at HEAD `a794404c`)

| symbol | meaning | code site |
|---|---|---|
| `x` | detector-frame mass fraction `M_z / M_z,det,i` (dimensionless coordinate of the with-BH mass marginal) | `bayesian_statistics.py:6759-6764` ("Eq. 14.22") |
| `μ_cond(z)`, `σ²_cond` | GW conditional mean/variance of `x` given `(φ, θ, d_L(z;h)/d_L,det)` | `:6712`, `:6685` (Eqs. 14.23–14.28) |
| `μ_gal,g(z) = M_eff,g (1+z)/M_z,det`, `σ_gal,g(z) = σ_M,g (1+z)/M_z,det` | candidate `g`'s mass prior in `x`, centred at the Eddington-shifted effective mass | `:6763-6764`; `_host_M_eff` `:6667-6675`; `eddington_shifted_host_mass` `:602-636` |
| `mz(z)` | analytic Gaussian-product overlap `N(μ_cond; μ_gal, σ²_cond+σ²_gal)` | `:6770-6773` (Eq. 14.31) |
| `S_4D(d_L, M_z)` | the production with-BH survival (pooled-2D 40-bin grid, isotropic sky) | `simulation_detection_probability.py:2018-2048`; `_wbh_z_kwargs` `:1140-1162` |
| `k̄_g(z)` | volume-deconvolved host-z kernel (production `host_z_kernel=volume_deconv`) | integrand `:6807`; `fixed_quad` with `_HOST_QUAD_N=50` (`:409`) |
| `g_sel,prod(z;h)` | the FUSED 2D completion mass density `∫ N(x;μ_cond,σ_cond) φ_x(x;z) S_4D(d_L(z;h), x M_z,det) dx` | `completion_mass_factor_g_sel` `:2265-2380` (rows #117–#118) |
| `Σ^4D(h)` | global with-BH catalogue selection sum, per-row POINT query `S_4D(d_L(z_g;h), M_g(1+z_g))` | `precompute_global_catalog_selection` `:2965-2983` (`sigma4d_mass_kernel="point"`, production) |
| `α_G_φ = Σ^4D·β_G_φ/Σ^φ`, `D̃_φ = α_G_φ + β̄_Ḡ_φ` | Path-A with-BH class weight and normaliser | `:2485-2489` |

### 1.2 OLD (production at HEAD) — with-BH catalogue leg, per event `i`, per candidate `g`

```
N_g^wbh(h) = ∫ dz k̄_g(z) · gw_3D(z;h) · mz_g(z;h)                       [:6807]
mz_g(z;h)  = ∫ dx N(x; μ_cond(z), σ²_cond) · N(x; μ_gal,g(z), σ²_gal,g(z))   [analytic, :6770-6773]
L_cat,wbh,i(h) = Σ_g w_g N_g^wbh / Σ^4D(h)                                  [:5229-5236]
combined_wbh,i(h) = (α_G_φ · L_cat,wbh,i + B_num,wbh,i) / D̃_φ              [:5726-5729]
```

Because `α_G_φ = Σ^4D β_G_φ/Σ^φ`, `Σ^4D` cancels exactly in the catalogue term:
`α_G_φ·L_cat,wbh,i = β_G_φ · (Σ_g w_g N_g^wbh)/Σ^φ` (A20 review F3: max rel. dev. 6.942e-8 on 24
artifacts, 2026-08-25). **The per-candidate mass integrand carries NO survival factor** — the
Gray (2020) Eq. (A.10)/MFG (2019) convention, codified at `:6695-6699` ("p_det is applied
solely in the denominator") — while the completion term `B_num,wbh,i` carries `S_4D` INSIDE its
mass quadrature via `g_sel,prod` (rows #117–#118), and the 1D catalogue leg carries `S̄_φ(z;h)`
per candidate inside its z-quadrature (row #197, `:6520-6526`, `:6567-6573`).

### 1.3 NEW (proposed default) — the survival enters the innermost quadrature

```
mz_sel,g(z;h) = ∫ dx N(x; μ_cond(z), σ²_cond) · N(x; μ_gal,g(z), σ²_gal,g(z)) · S_4D(d_L(z;h), x·M_z,det,i)
              = mz_g(z;h) · E_{x ~ N(μ*_g(z), σ*²_g(z))}[ S_4D(d_L(z;h), x·M_z,det,i) ]
μ*_g = (μ_cond σ²_gal,g + μ_gal,g σ²_cond)/(σ²_cond + σ²_gal,g),   σ*²_g = σ²_cond σ²_gal,g/(σ²_cond + σ²_gal,g)
N_g^wbh,sel(h) = ∫ dz k̄_g(z) · gw_3D(z;h) · mz_sel,g(z;h)
```

(the product-Gaussian identity, `_mz_sel_2d_expectation` `:6058-6140`; batch twin `:6143-6197`;
`E[·]` by Gauss–Hermite of order `_MASS_TRUNC_GH_ORDER = 24` `:444`, the same nodes the
`mass_trunc` kernel uses). **Where it enters:** at the innermost level — per candidate, per
z-node of the 50-node host-z quadrature, per GH node of the mass quadrature — i.e. the survival
is evaluated at `(d_L(z;h), x·M_z,det)` for every `(z, x)` node the numerator already integrates,
exactly as `g_sel,prod` does for the completion leg. Every other object (`w_g`, `k̄_g`, `gw_3D`,
`Σ^4D`, `α_G_φ`, `B_num,wbh`, `D̃_φ`, the without-BH channel) is UNTOUCHED. Use sites: scalar
quadrature branch `:6775-6801`, scalar delta branch `:6824-6848`, batch (production dispatch via
`_starmap_host_batches`) `:7466-7488`.

**Mechanically the proposal is the row-#197 pattern:** promote the ALREADY-IMPLEMENTED, tested
(`darksiren_emri_test/bayesian_inference/test_catalogue_numerator_survival_2d.py`, 27 test
functions / 48 parametrised cases, all passing at HEAD — smoke-run 2026-08-29) and
fleet-exercised (33-seed repair fleet, `p3_2d_fleet.py:27`, centre `eff`) counterfactual cell to
the production default by an `"auto"` resolution: `catalogue_numerator_survival_2d="auto"` →
`"mz_sel"` under `normalization_mode="absolute_marginal"` (else `"off"`), and
`catalogue_numerator_survival_2d_center="auto"` → `"eff"`. `"off"` is retained as the explicit
COUNTERFACTUAL (warning line, as the 1D twin's `"off"` at `:3657-3662`). Every explicit setting
behaves exactly as today.

### 1.4 Reference / derivation (item 3)

Mandel, Farr & Gair (2019) arXiv:1809.02063 Eqs. (5)–(7): the per-event likelihood of a
latent-thresholded detection model carries the selection at the HYPOTHESIS (here the candidate's
own `(z, M)` posterior), not only in the population normaliser; L6-DER2 §2–§3 / L6-DER3 §3–§4
("the catalogue leg is the same fork, per-galaxy"; quoted in the 1D adoption §3); the generator is
latent-thresholded (proven at the A20/O4 review; O6 MECHANISM-CONFIRMED end-to-end, row #158).
Stage 0 of this thread (`CLAIM_P3_2D_20260825.md` §1, row #189) DERIVED the per-candidate object
as exactly §1.3 ("survival inside the candidate's own (Eddington-shifted) mass posterior
quadrature, NOT point-S_4D and NOT S̄_φ(z)"). The 2D completion analogue is the ratified fusion
gate (`GATE_PRESENTATION_SELECTION_FUSION_20260817.md` §1). Departure from Gray (2020) Eq. (A.10)
is deliberate and mirrors the 1D adoption's §3: the coded arrangement is not the self-consistent
scoring of the pipeline's own mixture (`docs/LITERATURE_WARNINGS.md` MFG-a verbatim check remains
the Stage-L obligation before any paper-facing quotation — carried, not blocking).

### 1.5 The structural asymmetry this closes (the proposal's central argument)

Write `S → c·S` for a uniform rescaling of the with-BH survival (the K-flat probe of the 1D twin,
`catalogue_numerator_survival="phi_flat"`). Degrees in `S` at HEAD, from the definitions above:
`β_G_φ` 1, `Σ^φ` 1 (ratio 0); `N_g^wbh` (coded) **0**; `B_num,wbh` (fused `g_sel,prod`) **1**;
`D̃_φ` 1. Hence

```
coded:  combined_wbh = ( T_cat · c⁰  +  T_comp · c¹ ) / (D̃ · c¹)   — NOT homogeneous: the catalogue
                                                                   share of the with-BH mixture is
                                                                   inflated by 1/c relative to completion
twin:   combined_wbh = ( T_cat · c¹  +  T_comp · c¹ ) / (D̃ · c¹)   — homogeneous of degree 0
```

The with-BH mixture at HEAD weights its catalogue leg as if every candidate survived with
probability 1 while its completion leg carries the same event's `S_4D` inside the same `x`
measure. The twin makes the two legs S-degree-matched, exactly as row #197 did for the no-BH
mixture (its §4 "S̄→cS̄ homogeneity" argument). This is a testable invariant (§8, regression test
R3) and the proposal's A14 falsifier (i) (§6.1).

---

## 2. Centering — DECIDED: `eff` (the numerator's own centre)

### 2.1 What each option means for the GW-mass kernel's centre

The centre parameter selects the mean of the galaxy Gaussian FED TO THE SURVIVAL EXPECTATION
ONLY (`:6782-6786`, `:7471-7475`); the overlap prefactor `mz_g` is ALWAYS computed with
`μ_gal = M_eff,g(1+z)/M_z,det` (`:6763`, `:7457`) and `σ_gal` is unchanged either way.

- **`eff`** — `μ*` is built from the SAME `μ_gal,g` the prefactor uses. The code then computes
  exactly the single integral of §1.3: `∫ N_cond · N_gal,eff · S dx` (completing the square is an
  identity only when both factors share `μ_gal`). The product-Gaussian centre is
  `μ* = (μ_cond σ²_gal + μ_gal,eff σ²_cond)/σ²_sum`.
- **`raw`** — `μ*` is built from `M_g` while the prefactor stays at `M_eff,g`. The result is
  `N(μ_cond; μ_gal,eff, σ²_sum) · E_{N(μ*_raw, σ*²)}[S]`, which is the value of NO integral of the
  form `∫ N_cond · p_gal · S dx`: it is a hybrid of two different galaxy priors. Under
  `eddington_m="off"` (or `generator_marginal`, where `_host_M_eff = host_M`, `:6667-6675`) the
  two options coincide.

### 2.2 Limiting-case argument (own)

1. **`σ_gal → 0` (mass-certain host).** `μ* → μ_gal`; the Eddington shift `M_eff − M_g ∝ σ²_rel`
   (`:602-636`) vanishes; both centres give `S_4D(d_L(z;h), M_g(1+z))` — the very point query
   `Σ^4D` makes per catalogue row (`:2965-2983`). Both options agree; the divisor-consistency
   worry (that `Σ^4D` queries the raw point mass) has no leverage in this limit.
2. **`σ_cond → 0` (sharp GW mass).** `μ* → μ_cond`, `σ* → 0`, `E[S] → S_4D(d_L, μ_cond M_z,det)`
   for BOTH centres (`test_mz_sel_sharp_gw_mass_limit_matches_point_s4d`). This IS the production
   operating point: measured d_L-conditional `σ_cond` p50 = 8.8e-8 (row #118/MAJOR-1, quoted at
   `:2314-2317`, 2026-08-17) against `σ_gal,frac = O(0.3–3)` (GLADE σ_M 60–200 % of M_g,
   PA-2D-2), so `σ²_cond/σ²_sum ≈ 8.6e-14 … 8.6e-16` and `μ*_eff − μ*_raw = (σ²_cond/σ²_sum)·
   (μ_gal,eff − μ_gal,raw) ≲ 1e-14` in `x`. **The centering is numerically inert in production to
   double-precision order; it is a DEFINITIONAL choice, and no production arm can or need
   discriminate it** (disclosed so the wave-2 read is not asked to).
3. **Which makes 1D and 2D structurally symmetric.** The 1D twin multiplies the numerator's OWN
   integrand at its OWN z-nodes by `S̄_φ` (`:6520-6526`). The 2D analogue is "the numerator's own
   `(z, x)` integrand times `S_4D`" — `eff`. `raw` has no 1D analogue (it would be evaluating
   `S̄_φ` on a kernel other than the one being integrated). The fused completion leg is likewise
   "its own integrand times `S`" (`g_sel,prod`, `:2280-2300`). `eff` makes all three fused legs
   the same construction.
4. **Which reproduces the 1D twin's convention and the banked evidence.** The pre-execution
   review RULED `eff` for the Gaussian branch (A20 review F2, verbatim: "the latent model wants
   `_host_M_eff`, and kernel identity is what makes W̃₂ ≤ 1 eventwise"; folded as PA-2D-1); the
   repair fleet that produced CONFIRMED-at-33-seeds ran with centre `eff` (`p3_2d_fleet.py:27`,
   `p3_2d_companion.py:46`). Mirroring row #197 — promote the cell AS EXERCISED — means `eff`.
5. **`raw` would additionally break the eventwise bound `W̃₂ = L^twin/L^coded ≤ 1`** that the
   identity's `D_C₂` sign and the D_C-collapse proof consume (F2/F8): with a hybrid centre `E[S]`
   is still ≤ 1, but the object no longer equals "the coded integrand with `S ∈ [0,1]` inserted",
   so the bound is no longer the statement the derivation makes.

**Decision:** `catalogue_numerator_survival_2d_center` resolves to **`eff`** under `"auto"`;
`raw` stays available as an explicit instrument only.

---

## 3. Dimensional analysis (item 4)

- `S_4D ∈ [0, 1]` is dimensionless (`simulation_detection_probability.py:2044`); `E[S_4D]` is a
  weighted mean of dimensionless values → dimensionless; `mz_sel = mz · E[S]` keeps `mz`'s
  measure: a density in the dimensionless `x = M_z/M_z,det` — the SAME measure as `g_sel,prod`
  (`:2297-2299`), so the 2D catalogue/completion addability (fusion gate (i)) is preserved.
- Query arguments: `d_L(z;h)` in Gpc (`dist_vectorized`, `physical_relations.py:226-238`) — the
  accessor's grid axis; `M_z = x·M_z,det` in M_sun, detector frame — the accessor's second axis;
  `z` rider through `_wbh_z_kwargs` (inert while `pdet_wbh_z_resolved=False`, every run of
  record). Identical to the `Σ^4D`/`S̄_φ`/`g_sel,prod` query convention (`:2029-2050`, `:2340`).
- Degree bookkeeping in `S`: §1.5 — the with-BH mixture becomes homogeneous of degree 0.
- No new table, no new constant, no unit conversion: the proposal consumes the existing
  `detection_probability_with_bh_mass_interpolated` accessor only.

---

## 4. Limiting cases (item 5)

| limit | result | status |
|---|---|---|
| `S_4D ≡ 1` | `E[S] = 1` ⇒ `mz_sel = mz` ⇒ the current code exactly | structural; `"off"` byte-identity tests `test_default_off_omitted_kwarg_is_bit_identical_{scalar,batch}` |
| `S_4D → c` (constant) | with-BH posterior INVARIANT under the twin (§1.5); NOT invariant under coded | REGISTERED PREDICTION → regression test R3 (§8); untested at HEAD |
| `σ_cond → 0` (sharp GW mass) | `mz · S_4D(d_L(z;h), μ_cond(z) M_z,det)` — the stage-0 §1 registered limit; the production operating point (§2.2 item 2) | `test_mz_sel_sharp_gw_mass_limit_matches_point_s4d` passes at HEAD |
| `σ_gal → 0` | `S_4D` at the candidate's catalogue mass `M_g(1+z)` — `Σ^4D`'s own per-row point query | derivation §2.2 item 1 |
| single candidate, `σ_z → 0` (delta kernel) | `gw_3D(z_g)·mz(z_g)·E[S](z_g)` — the selected-prior single-host form (A-FULL of the 1D proposal §5) | code path `:6824-6848` |
| mass-information-free (`σ_cond → ∞` AND the host mass prior → the population `φ_x`) | `mz_sel → ∫ φ_x(x;z) S_4D(d_L, x M_z,det) dx = S̄_φ(z;h)` by the tower identity (`:1988-1997`) — the 1D twin's per-candidate factor | derivation; the same limit takes `g_sel,prod → S̄_φ`, so the 1D twin is the mass-blind limit of BOTH fused 2D legs |
| `E[S] ≤ 1` eventwise | `L_cat,wbh^twin ≤ L_cat,wbh^coded` for every (event, h) | `test_mz_sel_moves_with_bh_numerator_by_a_survival_factor_in_0_1`; wave-2 gate R1 (§6.2) |

---

## 5. The ×2.5 residual — disclosed and bounded (the known unknown)

The 2D bounded identity (`PREREGISTRATION_P3_2D_20260825.md` §1, C₂\* frozen row #204) is the
2D analogue of the identity that certified the 1D twin TWIN-CALIBRATED (row #186). **It has NOT
closed for 2D.** Ladder of record (all {value, source, date}):

| stage | X = RHS₂/LHS₂ (bt/twin) | X (bc/coded) | source |
|---|---|---|---|
| σ freeze, PA-2D-9 | 2.898 ± 0.113 | 3.494 ± 0.138 (derived here from the frozen numbers) | `PREREGISTRATION_P3_2D_20260825.md:309-311`; `p32d_residual_accounting_20260827.md` §0 (2026-08-26/27) |
| after rung 1 (S̄_φ double-application, reweight ×1.1585) | **2.502 ± 0.101** — "the ×2.5" of rows #209–#211 | — | `p32d_residual_accounting_20260827.md` §1 (2026-08-27) |
| after rungs 2+3 (venue mass floor ×1.1944; dead-row convention ×1.0680) — MEASURED end-to-end | implied **2.253 ± 0.082** (RHS₂ 0.01451300 ± 0.00045293 / P1 0.00644266 ± 0.00012212, 33 seeds) | implied 2.700 ± 0.101 (RHS₂,coded 0.01507225 ± 0.00046202 / P4 0.00558246 ± 0.00012014) | `P3_2D_REPAIR_READOUT_20260828.md` §7 (2026-08-28), ratified row #216 item 1 |
| conditional on rung 1 (unimplemented) | ×1.961 ± 0.090 (registered v2.9 conditional prediction LHS2(bt) = 0.00740040 ± 0.00024951) | ×2.348 ± 0.113 | `PREREGISTRATION_P3_2D_REPAIR_20260827.md` v2.9; `p32d_residual_accounting_20260827.md` §5 |

What is EXONERATED for the residual (rows #207–#211): C₂\* correct (blind re-derivation);
completion-side mass axis, two constructions (X = 0.047 ± 0.014 and X_alt = 0.9997 ± 0.0003);
machinery at machine precision; the two-rung venue model CONFIRMED (every read inside its band at
33 seeds, R = 1 excluded 6.82σ). What is NOT: the class-G draw-law contraction vs `Σ̃^4D` and the
identity's acceptance-measure step (symptom card rungs 3/4). **The residual is common-mode across
arms** (X_bt/X_bc = 0.834 at 33 seeds, 0.829 pre-repair; G4 arm-coherence 0.866484 ∈
[0.8613, 0.8675], `P3_2D_REPAIR_READOUT` §7), which is the F8 coherence-clause signature of a
venue/identity-frame mechanism rather than of the twin law — and the twin arm sits 17 % closer to
identity closure than the coded arm at the same venue (REPORTED-ONLY; no registered band).

**Consequence for this proposal, stated plainly:** the 1D adoption rested on a four-rung ladder
(derivation → mechanism → leverage → CALIBRATION). The 2D adoption rests on derivation (§1.4),
the structural-asymmetry argument (§1.5), the confirmed venue model, and the exercised
instrument — **without the calibration rung.** The epistemic status of the twin's calibration is
therefore `supported`, capped, until either the identity closes on a repaired venue or the
residual is attributed to the venue side (§6.1 falsifier (ii)). This is the proposal's known
unknown; the row #211 PARK is not reopened by it. Adoption is nevertheless proposed on the
correctness-over-bias-removal ruling (2026-08-05): a structural omission with a derivation, a
confirmed mechanism class and an exact regression invariant is corrected on its merits, and its
H₀ leverage is MEASURED (§6.2), never presumed.

**The S̄_φ double-weight (rung 1).** It is a defect of the VENUE's draw law
(`correspondence_1d.py:1496-1497` z-draw density × Bernoulli(`S_4D`) at `:1712-1719`), NOT of the
estimator (`sbarphi_defect_location_20260827.md` §2 confirms the production with-BH numerator
inserts `S_4D` once). Its fix as granted is **FIX-MISSPECIFIED** (`PHYSICS_CHANGE_SBARPHI_20260827.md`
§AR: disjunct 1 leaves ~69–70 % of the 13.5 %/16.0 % drift in place; Option A′ is the exact form,
§2.2), and the grant has no verbatim author quote (`sbarphi_defect_location_20260827.md` §1). Under
row #222 it is a tree-scoped INSTRUMENT (harness-only) change the orchestrator may now schedule;
it is the falsifier (ii) below, not a precondition of the adoption.

---

## 6. Falsifiers (A14/A19) and the wave-2 counterfactual arm (charter node B7.2)

### 6.1 Registered falsifiers of the attribution "the with-BH catalogue leg's missing per-candidate survival is a structural omission whose closure is `mz_sel`/`eff`"

(i) **Homogeneity (zero compute, unit-test scale).** Under `S_4D → c·S_4D` (`c ∈ (0,1]`, applied
through a wrapped accessor), the twin's per-event `combined_wbh` must be invariant to ≤1e-10
relative while the coded one is not. Failure ⇒ the §1.5 degree bookkeeping is wrong (some
with-BH term has degree ≠ 1) ⇒ proposal RETURNS. Two-sided by construction (an invariance test).

(ii) **Identity residual is venue-side (the S̄_φ double-weight as the registered falsifier).** On
the class-G venue with rung 1 repaired in the Option A′ form (harness-only gate; fleet re-run
~8.67 CPU-h/task × 24–33 tasks ≈ 208–286 CPU-h, from the readout's ~32.5 min/task at 16 cpus —
the runbook-34 "~2–4 CPU-h" figure is superseded by measurement), the registered v2.9 conditional
prediction must land: LHS2(bt) = 0.00740040 ± 0.00024951, band ±3σ_comb (two-sided), AND the G4
arm-coherence ratio must stay inside its registered interval [0.8613, 0.8675]. Outcomes:
**inside both** ⇒ the ladder model is complete to rung 3 and the remaining ×1.96 is venue-side
until shown otherwise (attribution stays provisional); **LHS2 outside** ⇒ the ladder is wrong
somewhere — the residual bound is not load-bearing, the twin's calibration status drops from
`supported` to `derivation-only`; **G4 outside** ⇒ the venue repair moved the arms DIFFERENTLY,
implicating the twin law itself ⇒ adoption RETURNS to the gate as REFUTED-AS-CALIBRATED.
(Null: paired deterministic fleet re-score; the band's false-fail rate under the exact null is set
by the frozen planning SEMs, already realized below planning at 33 seeds, §7 of the readout.)

(iii) **Production behaviour** — the wave-2 arm below: exact eventwise inequality R1 and 1D
bit-identity R6 are INSTRUMENT-DEFECT falsifiers; the H₀ read is two-sided (MATERIAL-UP /
MATERIAL-DOWN / IMMATERIAL). Per the correctness-over-bias-removal ruling, MATERIAL-DOWN does not
refute the adoption but opens a mandatory stage-0 on the sign before the flip.

### 6.2 Wave-2 arm PROD-CF-2D (one venue, HEAD, `mz_sel`/`eff` vs the fused baseline)

**Precedent and pattern:** the measure-first chain of rows #198 → #201 → #202 (mirror-venue
measurement, then a paired PRODUCTION counterfactual read at h = 0.73 — ΔT, Δw̄, zero rates —
then adoption, then the blind HEAD readout). The mirror-venue leg for this change is the 33-seed
fleet; PROD-CF-2D is the production leg.

- **Venue:** iiib (true reduced catalogue, md5 `c52c13b5…` pin; 1588 events, CRB md5 `9a1f2a14…`;
  `EVAL_SEED = 777000`; `MEASUREMENT_HEAD_READOUT_20260827.md` §7.2). One venue only: T_mat was
  derived as the max over both venues and is conservative on joint_r1; joint_r1's HEAD-config
  cost is ≥ 2.2× iiib's (slowest task ≥ 3.2 h vs 1:25:52, §F of the HEAD readout).
- **Code state:** the wave-2 HEAD (after B6 [ALIGN]'s bit-identical-today `[PHYSICS]` commit and
  B5.1's byte-identical-default flag). Baseline = the banked HEAD readout (`d04d9dc9`, fused/phi/
  symmetric) columns, REUSED only if a same-commit baseline task at h = 0.730 reproduces the
  banked per-event `L_cat_with_bh`/`combined_with_bh` columns to ≤ 1e-12 relative (the row-#201
  PROD-A0 ingredient gate, which passed at ≤ 8.5e-15); otherwise the baseline is re-run at the
  arm's nodes (A22: same commit, dirty-state stamped at run start).
- **Arm T:** `--catalogue_numerator_survival_2d mz_sel --catalogue_numerator_survival_2d_center eff`
  (explicit, pre-flip), everything else at production defaults; `run_metadata_*.json` stamp check
  (STEP-2 pattern) — the `COUNTERFACTUAL: catalogue_numerator_survival_2d='mz_sel'` line is
  EXPECTED in this pre-adoption arm and is the A13 engagement evidence channel.
- **h-grid — argued reduction.** The registered read statistics are the with-BH channel's
  Δmean_h/ΔMAP vs the fused baseline. Three grids, all subsets of `H_GRID_41` so the baseline
  re-scores on the identical nodes at zero compute:
  - **H4 = {0.660, 0.665, 0.670, 0.730}** (RECOMMENDED for B7.2): 0.730 = the row-#201 production
    read (per-event score tilt ΔT at truth, A12; Δw̄₂ = mean with-BH catalogue mixture-weight
    shift; the zero-rate census); {0.660, 0.665, 0.670} bracket the HEAD 2D MAP 0.665 / mean_h
    0.663347 (`MEASUREMENT_HEAD_READOUT` §C.1, 2026-08-28) and give the ensemble log-likelihood
    change Δℓ(h) = Σ_i ln[L_i^T/L_i^B]'s slope and curvature at the peak, hence the first-order
    PREDICTED shift Δmean_h,pred ≈ Δℓ'(0.665)/I_HEAD with I_HEAD = 1/σ_h² = 2965 (σ_h 0.018366):
    a material shift of 0.008 corresponds to Δℓ' = 23.7 nats per unit h. Validity condition,
    registered: |Δℓ''| ≪ I_HEAD; if violated the read is AMBIGUOUS by rule.
  - **G27 = {0.600, 0.610, …, 0.860}** (0.010 step; 27 nodes): the full posterior read at MAP
    resolution T_res = 0.010 (disclosed: coarser than the 41-node 0.005); mean_h is resolved far
    below T_mat — trapezoid aliasing on a σ_h = 0.0184 posterior at Δ = 0.010 is
    exp(−2π²σ²/Δ²) ≈ 1e-29 (at Δ = 0.020 it is 6e-8; at σ = 0.010, Δ = 0.020 it is 7e-3, which is
    why a 14-node grid is NOT proposed).
  - **G41 = `H_GRID_41`**: the unconditional full read — which the wave-3 shared blind HEAD readout
    delivers for this change anyway (F2, §9), so it is NOT proposed for B7.2.
- **Reads and gates (registered form):**
  - **R1 (gate, zero free parameters):** `ln L_cat,wbh^T ≤ ln L_cat,wbh^B` for EVERY (event, h),
    equality only where the candidate set is empty. Any violation ⇒ INSTRUMENT-DEFECT.
  - **R2 (A13 engagement):** fraction of events with a non-empty window-passed with-BH candidate
    set whose `|Δ ln L_cat,wbh| > 1e-6` at h = 0.730; registered threshold ≥ 0.95 (expected ≈ 1.0
    since `S_4D < 1` on any support with `d_L > 0`). Below ⇒ the switch is not reaching the
    production dispatch path ⇒ STOP.
  - **R6 (gate):** the 1D channel's per-event columns (`L_cat_no_bh`, `combined_no_bh`) are
    bit-identical between arms (`test_1d_channel_unaffected_*` at production scale). Violation ⇒
    INSTRUMENT-DEFECT.
  - **R3:** ΔT(0.730) = the with-BH score-at-truth tilt shift (row #201 form); **R4:** Δw̄₂(0.730)
    (expected negative, since `E[S] ≤ 1` lowers the catalogue term relative to completion);
    **R5:** Δmean_h,pred from the {0.660, 0.665, 0.670} stencil. R3–R5 are REPORTED with bands:
  - **Verdict map (two-sided, A8):** MATERIAL-UP-PREDICTED: Δmean_h,pred ≥ +T_mat;
    MATERIAL-DOWN-PREDICTED: ≤ −T_mat; IMMATERIAL-PREDICTED: |Δmean_h,pred| ≤ T_mat/2 = 0.004;
    AMBIGUOUS: 0.004 < |·| < 0.008 or validity condition violated ⇒ conditional escalation to G27
    (below). **T_mat = 0.008** provenance: the HEAD readout's registered threshold
    `max(node spacing, σ_h/3)` = max(0.005, 0.0239/3 = 0.007967) at the row-#132 σ_h, taken as
    the max over both 2D venues and rounded up (`MEASUREMENT_HEAD_READOUT_20260827.md:268-285`),
    ratified with row #213 (§10 item 4). Disclosed: re-deriving at the HEAD σ_h (0.018366 /
    0.018637) would tighten it to 0.0061 / 0.0062; the ratified 0.008 is the band of record and the
    tightened value is carried as a REPORTED-ONLY secondary edge.
  - **Operating characteristics (A15):** all reads are paired deterministic recomputations on the
    identical event set — sampling variance is exactly zero; the "null" (twin ≡ baseline) is
    excluded by construction (R1 strict wherever `S_4D < 1`), so the bands are MATERIALITY
    thresholds, not significance tests: false-fail rate 0 under reproducibility floor ≤ 8.5e-15
    (PROD-A0, row #201); detectable effect: any |Δmean_h| ≥ 0.008 with certainty on the full grid;
    on H4 the limitation is the predictor's model error, bounded by the registered validity
    condition and made harmless by the unconditional wave-3 full read (a misclassification changes
    staging, never the adoption verdict).
- **Expected direction (REPORTED-ONLY, not a band):** `S_4D(d_L(z;h), ·)` increases with h at
  fixed z (d_L ∝ 1/h), so the twin tilts each catalogue-leg likelihood toward higher h while
  lowering its mixture share (R4 < 0); the net sign after the `Σ^φ`/`D̃_φ` chain is not derivable by
  hand. The 1D twin's leverage was upward (+0.029 / +0.063 registered / un-truncated, row #173;
  +0.0566 catalogued-host venue, row #177). An upward 2D move is toward truth (HEAD offset
  −0.066653 iiib). No number is predicted.

**Cost (instructed anchor, A11-flagged).** Production anchor 14.93–20.27 CPU-h per h-point
(`hier_costing_20260826.md:54`; `PREREGISTRATION_HIER_HTHETA_20260826.md:576`; origin
`cluster/LAUNCHING_JOBS.md:47`: 56–76 min @ 16 cpus, jobs 5732036, **3355 events, 2026-07-03**).
Configuration-of-record mismatch: today's production is 1588 events at HEAD (fused + 1D twin +
symmetric filter), so per A11 the anchor is STALE as a point value and is carried as a band,
bracketed by two fresher anchors: (a) row-#132 config `e65d263c` (sel=off, 1588 events): mean
286 s/task ⇒ **1.27 CPU-h per h-point**, max 1306 s (5.8 CPU-h) (`MEASUREMENT_HEAD_READOUT` §9,
sacct 2026-08-27); (b) HEAD-config off arm on iiib: slowest completed task 1:25:52 ⇒ **22.9 CPU-h**,
6/41 tasks exceeded 1.5 h (§F, 2026-08-28) — the fused HEAD run's own per-task times are NOT
recorded in the readout text (disclosed). The instructed band is inside (a)–(b). `mz_sel`
overhead: on the 200-event mirror venue the twin arm was NOT slower than coded (62.944 vs
64.996 s, `p3_2d_rhs2.sbatch:15-16`), but production candidate counts are ~10³× larger and the
batch accessor call scales as n_cand × 50 × 24; an overhead factor **1.0–1.3 is ASSUMED, not
measured** — the h = 0.730 task doubles as the STEP-2 smoke that pins it.

| grid | nodes | twin arm (14.93–20.27/node) | + baseline gate task (1 node) | ×1.3 overhead ceiling |
|---|---:|---:|---:|---:|
| **H4 (recommended)** | 4 | **59.7–81.1** | **74.7–101.4** | 105 (arm) / 132 (total) |
| H4 with full baseline re-run | 4+4 | 59.7–81.1 | 119.4–162.2 | 211 |
| G27 (conditional escalation) | 27 | 403.1–547.3 | 418.0–567.6 | 711 / 738 |
| G41 (not proposed for B7.2; = wave-3 per-change arm) | 41 | 612.1–831.1 | 627.0–851.4 | 1080 / 1107 |

The charter's "~50–130 CPU-h" envelope for B7.2 holds for H4 (nominal 75–101 CPU-h) and fails for
any full-grid design at the instructed anchor by 4–8×; the fresher anchor (a) would give
G27 ≈ 34–52 CPU-h but is a `sel=off` figure the HEAD readout itself warns is a LOWER bound.
Shape: `--array` over the 4 nodes, `cpu_il`, 16 cpus, `--time=03:00:00` (2.1× the HEAD-config
slowest observed task; the joint_r1 6-h sizing does not apply to iiib), backfill-friendly.
Archive: the arm's out-root lands before 2026-09-23 and is MUST-ARCHIVE-tier (Option A).

---

## 7. A10 — invariants and blindness

**Invariants held FIXED across both arms (last derivation-audit date):**
`normalization_mode=absolute_marginal` · `host_z_kernel=volume_deconv` ·
`selection_in_completion_numerator=fused` (rows #117–#118, gate presentation 2026-08-17) ·
`catalogue_numerator_survival=phi` (row #197, 2026-08-25) · `catalogue_global_selection=phi`
(rows #172–#178, 2026-08-23) · `mass_filter_sigma=symmetric` (row #202, `cf4f8a2a`, 2026-08-25) ·
`completion_b_scale=derived` (2026-08-20) · `eddington_m=on` (G2d derivation
`docs/derivations/G2d_host_mass_rate_prior.md`; impact re-measured 2026-08-18 by instrument E) ·
`sigma4d_mass_kernel=point` (Instrument J; measured 2026-08-18, NEVER derivation-audited as a
design choice — disclosed) · `catalogue_mass_overlap=production` · `pdet_wbh_z_resolved=False`
(every run of record) · `H_GRID_41`/subsets · CRB and catalogue md5 pins · `EVAL_SEED=777000` ·
the with-BH survival table (the S_4D object; NEVER independently re-derived — the six-instrument
common mode).

**Blindness sentence:** by construction this design cannot detect (a) a defect in the shared
`S_4D` survival table (common to the numerator, `Σ^4D`, `g_sel,prod` and `S̄_φ` — a rescaling of it
is exactly what the twin makes invisible, §1.5); (b) a mismatch between the point-mass per-row
convention of the `Σ^4D` divisor and the kernel-integrated numerator (Instrument J's axis, out of
scope); (c) anything in the 1D channel (bit-identical by construction); (d) which SIDE (venue or
estimator) carries the ×1.96–2.35 identity residual — no production arm has a truth anchor;
(e) the centering choice (numerically inert to ~1e-14 at the production σ_cond, §2.2).

---

## 8. What the `/physics-change` gate presentation will contain, and the regression plan

**Gate presentation (authored BEFORE code, `.claude/rules/physics-validation.md`; APPROVED
column = "row #223"; ledger rows presented/implemented/verified in `docs/gates/PHYSICS-GATE-LEDGER.md`):**
1. Old formula §1.2 with line cites; 2. New formula §1.3 + the `"auto"` resolution rules
   (§1.3 last paragraph; composition guard decision G-1 below); 3. References §1.4;
4. Dimensional analysis §3; 5. Limiting cases §4; 6. (A5 item 6) source-equation validity
   conditions per venue: MFG (2019) Eqs. (5)–(7) require a latent-thresholded detection model
   (proven for the generator, row #158) and a survival evaluated at the hypothesis — the pooled-2D
   `S(d_L | M_z)` grid is that object only under the isotropic-sky decision (residual bounded
   1.000202, gate (ii-e)) and `pdet_wbh_z_resolved=False`; the FIX-3 joint grid would ride along
   through `_wbh_z_kwargs` unchanged.

**Gate decision items (orchestrator, under #223):**
- **G-1** composition guard under `"auto"`: when `catalogue_mass_overlap != "production"` or
  `host_mass_kernel` resolves to `trunc_lognormal`, `"auto"` must RAISE (require an explicit
  `"off"`) — recommended, mirrors the existing guard `:6327-6333` and A13's explicitness — vs
  resolve to `"off"` with a `COUNTERFACTUAL:` warning (caught by the §8.7 zero-warning gate).
- **G-2** log line: `[PHYSICS] catalogue_numerator_survival_2d="mz_sel" (center="eff") ACTIVE
  (row #<adoption>)` at INFO, replacing the current `COUNTERFACTUAL:` warning for the resolved
  production value (the `:3678-3685` block), exactly as `:3641-3647` does for the 1D twin.
- **G-3** CLI: `--catalogue_numerator_survival_2d {auto,off,mz_sel}` default `auto`;
  `--catalogue_numerator_survival_2d_center {auto,unset,raw,eff}` default `auto`; the
  `arguments.py:459-470` refusal stays for explicit `mz_sel` + `unset` (defense in depth unchanged).

**Regression plan (the row-#178/#195 pattern; builder ≠ runner per standing rule 2):**
- R1 promote the 27 existing functions to default-path tests: `"auto"` resolves to
  (`mz_sel`,`eff`) under `absolute_marginal` and (`off`,`unset`) otherwise; explicit `"off"`
  byte-identical to the pre-flag golden across modes (existing `test_off_matches_the_pre_flag_golden_across_modes`).
- R2 scalar/batch parity under `"auto"` (existing parametrised parity tests, re-pointed).
- R3 **NEW — K-flat 2D homogeneity** (§1.5/§6.1(i)): wrap the accessor with `c·S_4D`, assert the
  twin's `combined_with_bh` invariant to 1e-10 and the coded one NOT invariant (the 1D `phi_flat`
  analogue).
- R4 **NEW — eventwise inequality** `L^twin ≤ L^coded` on the kernel-parity stub detections.
- R5 **NEW — `σ_gal → 0` limit** equals the `Σ^4D` point query at `M_g(1+z)` (§2.2 item 1);
  the existing sharp-GW-mass limit test stays.
- R6 1D channel unaffected under `"auto"` (existing tests, re-pointed) and the without-BH
  closure never receives the flag (architectural, existing).
- R7 CLI plumbing: defaults, parsing, refusal (existing tests updated for `auto`).
- R8 `main.py:211-212`, `:1413-1414`, `:1452` threading unchanged; `run_metadata` stamps the resolved
  values (A22 stamp set gains `catalogue_numerator_survival_2d(_center)` — the PA-2D-6 lesson).
- Independent A20 clean-context verification COMMIT-READY (suite green; the with-BH leg the ONLY
  changed leg; every other leg bit-unchanged) before the `[PHYSICS]` commit; the row-#197
  AMEND pattern for the fleet drivers' explicit pins (`p3_2d_fleet.py`, `ca_rhs_scorer.py`,
  `p3_wbhzero_measure.py:269` — all pass the flags explicitly, so they are unaffected; verify).

---

## 9. F2 note — serialized adoption, batched into the one blind HEAD readout

Per amendment F2 (charter, ratified row #222 with the F5 substitution; append to
`docs/RESEARCH_CYCLE.md` pending) and row #223's last clause: the production-default flip is
serialized with the other tree adoptions (B5 window geometry, B3 population prior, B1/B4 as they
mature) and its H₀ effect is read ONCE, in the wave-3 blind HEAD readout, with a per-change arm
for THIS flag (fused/phi/symmetric baseline ± `mz_sel`) on both venues at `H_GRID_41`. The
HEAD readout's registered blindness (§4.2: no per-change attribution from a composed delta) is
respected precisely because the per-change arm is what licenses attribution for this change.
B7.2 (§6.2) is the measure-first production read that informs the B7.3 gate; it is not the
adoption's H₀ verdict. Sequencing: B6 [ALIGN] lands first (bit-identical today); B7.2 runs at
that HEAD; the flip commit is authored after B7.2 reads out, cites row #223, and rides into the
wave-3 batch.

---

## 10. Exoneration check — MECHANISM grep, both layers (standing rule 5)

Mechanism searched: "survival / p_det factor inside the with-BH (2D) catalogue numerator, per
candidate, inside the mass quadrature; catalogue-leg selection twin; S_4D in the numerator".

**Layer 1 (`EXONERATION_REGISTER_20260827.md` §1, `CLAIM_2D_BIAS_20260730.md:721-734`):**
- [PDET-IO] "p_det inside vs outside the numerator" — adjacent; its own delimitation defers the
  numerator-only family to [NUMERATOR-ONLY-CLEAN] and the unpaired variant to [PDET-NUM-ALONE].
  Not covering: this proposal is the PAIRED construction (the completion leg already carries
  `S_4D` inside the same measure; the twin matches degrees, §1.5) — the same delimitation the
  1D intake applied (`CLAIM_P3_IMPOSTOR_CONVENTION_20260822.md:44-49`, cited by the register as
  the worked example).
- [MASS-KERNEL-FAMILY] — the functional family of the mass kernel; not this axis (the Gaussian
  product is unchanged; only a dimensionless factor enters).
- HA "completion term not mass-marginalised" — refuted framing about the DENOMINATOR; this
  proposal touches the catalogue NUMERATOR and asserts the imbalance is numerator-internal,
  consistent with HA's own §C8 note.
- HB / [WINDOW-MEMBERSHIP] / [WBHZERO-ASYMMETRY] — the eligibility window; untouched (invariant).
**Layer 2 (`BIAS_HISTORY_LEDGER.md:127-155` "DO NOT RE-TRY"):**
- ⚠ item 6 "adding p_det inside the numerator ALONE — refuted (#66); only the joint pair works
  (#67)" — adjacent, NOT covering: `mz_sel` is the joint arrangement (fused completion + twin
  catalogue), and the 1D member of the identical construction is production physics (row #197).
  Any degenerate arm that inserts `S_4D` with the completion cell at `off` is VOID by this item
  (registered: A22 stamp `selection_in_completion_numerator="fused"` in every arm, PA-2D-1 F7).
- ⚠ item 10 "B_num as a defective integral" — completion-leg object, respected, untouched.
- ⚠ item 17 "numerator-only normalization cleans" — a de-rail STRATEGY on the 1D channel with the
  denominator left alone; here the denominator chain is untouched by DESIGN (Σ^4D cancels
  algebraically) and the change is degree-matching, not cleaning. Not covering.
**Conclusion: not exonerated; two adjacent entries delimited (items 6 and 17), no item reopened.**

---

## 11. Decision table (orchestrator under rows #222/#223; every item returns to the end-of-fan-out verifier)

| # | item | tag | recommendation |
|---|---|---|---|
| 1 | Run B7.2 = PROD-CF-2D on iiib, grid H4, reads R1–R6, T_mat 0.008 two-sided, 74.7–101.4 CPU-h nominal (§6.2) | [DO] | launch in the wave-2 batch; baseline reuse gated by the h = 0.730 PROD-A0 task |
| 2 | Conditional escalation to G27 (403–547 CPU-h) only on AMBIGUOUS | [DO, conditional] | pre-authorize the trigger, not the spend |
| 3 | Author the `/physics-change` gate for `"auto"`→(`mz_sel`,`eff`) after B7.2 reads out; APPROVED = "row #223"; batch the flip into the wave-3 blind HEAD readout with its per-change arm | [RULE, tree-scoped] | adopt on correctness grounds; H₀ leverage measured, never presumed |
| 4 | Register falsifier (ii): Option A′ harness gate + 24–33-task fleet re-run (≈208–286 CPU-h) testing v2.9 and G4 | [DO] | schedule after the wave-2 batch (deadline 2026-09-23); the residual bound is not load-bearing until it runs |
| 5 | Centering `eff` | [RULE] | DECIDED here (§2); no arm can discriminate it |
| 6 | MFG-a verbatim check before any paper-facing quotation | [DO] | Stage-L, carried from the 1D adoption §7 item 4 |

**Not proposed:** reopening row #211's PARK; any change to `Σ^4D`'s per-row convention
(Instrument J), the eligibility window (B5's axis), or the 1D channel.

---

## 12. Provenance table (A11)

| quantity | value | source (file:line) | date |
|---|---|---|---|
| repair fleet P1 (bt, 33 seeds) | 0.00644266 ± 0.00012212 | `P3_2D_REPAIR_READOUT_20260828.md` §7 | 2026-08-28 |
| P4 (bc, 33 seeds) | 0.00558246 ± 0.00012014 | same | 2026-08-28 |
| G4 arm-coherence | 0.866484 ∈ [0.8613, 0.8675] | same | 2026-08-28 |
| R = 1 exclusion / non-discrimination | 6.82σ / 2.41σ | same; ledger row #216 (`:3006`) | 2026-08-28 |
| RHS₂(twin), RHS₂(coded) | 0.01451300 ± 0.00045293, 0.01507225 ± 0.00046202 | `PREREGISTRATION_P3_2D_20260825.md:310-311` (PA-2D-9) | 2026-08-26 |
| LHS₂ pre-repair (bt, bc) | 0.00500770 ± 0.00011615, 0.00431338 ± 0.00010642 | same `:309` | 2026-08-26 |
| X after rung 1 ("×2.5") | 2.502 ± 0.101 | `p32d_residual_accounting_20260827.md` §1 | 2026-08-27 |
| residual conditional on rung 1 | ×1.961 ± 0.090 (bt) / ×2.348 ± 0.113 (bc) | same §1/§5; repair prereg v2.9 | 2026-08-27 |
| rung-1 factor, S̄_φ double-weight | ×1.1585; 13.5–16 % drift | same §2; rows #209/#210 (`:2990-2992`) | 2026-08-26/27 |
| rung-1 fix status | FIX-MISSPECIFIED (disjunct 1 leaves ~69–70 %); Option A′ exact | `PHYSICS_CHANGE_SBARPHI_20260827.md:731-736`, §2.2 (`:179`) | 2026-08-27 |
| X_bt(33), X_bc(33), ratio | 2.253 ± 0.082, 2.700 ± 0.101, 0.834 | derived here from the rows above (quadrature SEs) | 2026-08-29 |
| σ_cond p50 (production) | 8.8e-8 | `bayesian_statistics.py:2314-2317` (row #118/MAJOR-1) | 2026-08-17 |
| GLADE σ_M range | 60–200 % of M_g | `PREREGISTRATION_P3_2D_20260825.md` PA-2D-2 | 2026-08-25 |
| GH order / host-z quadrature | 24 / 50 | `bayesian_statistics.py:444`, `:409` | HEAD a794404c |
| fleet centre used | `eff` | `p3_2d_fleet.py:27`; `p3_2d_companion.py:46` | 2026-08-25 |
| centering ruling | F2 (eff for the Gaussian branch) | `A20_REVIEW_P3_2D_DESIGN_20260825.md:17-19` | 2026-08-25 |
| HEAD 2D (iiib): mean_h, offset, σ_h, MAP | 0.663347, −0.066653, 0.018366, 0.665 | `MEASUREMENT_HEAD_READOUT_20260827.md` §C.1 | 2026-08-28 |
| HEAD 2D (joint_r1) | 0.663013, −0.066987, 0.018637, 0.660 | same | 2026-08-28 |
| T_mat / T_res (2D) | 0.008 / 0.005 | same `:268-285`; ratified row #213 §10 item 4 | 2026-08-28 |
| 2-way split LEG 2 | −0.000322 / −0.001858 (H₀-immaterial) | same §G.1; row #216 item 2 | 2026-08-28 |
| 1D twin leverage | +0.029068 ± 0.005088 / +0.063389 ± 0.008897; +0.0566 (12/12) | `PROPOSAL_CATALOGUE_TWIN_PRODUCTION_20260825.md` §6 (rows #173, #177) | 2026-08-25 |
| 1D twin calibration | T_w = −0.0013 ± 0.0012, band 0.005 | same §6 (row #186) | 2026-08-24 |
| production cost anchor | 14.93–20.27 CPU-h per h-point (56–76 min @ 16 cpus, 3355 events) | `hier_costing_20260826.md:54`; `cluster/LAUNCHING_JOBS.md:47` | 2026-07-03 (STALE config) |
| row-#132 config anchor | 1.271 CPU-h per h-point (23 452 s / 82 tasks × 16 cpus); max 1306 s | `MEASUREMENT_HEAD_READOUT_20260827.md:607-609` | 2026-08-27 |
| HEAD-config off arm, iiib | slowest completed 1:25:52 (22.9 CPU-h); 6/41 TIMEOUT at 1.5 h | same §F | 2026-08-28 |
| HEAD-config off arm, joint_r1 | ≥ 3.2 h/task at h = 0.730 | same §F | 2026-08-28 |
| mirror-venue arm cost | 64.996 (bc) / 62.944 (bt) s @ 16 cpus, 200 events, single h | `cluster/p3_2d_rhs2.sbatch:15-16` | 2026-08-25/26 |
| repair fleet task time | ~32.5 min/task × 16 cpus = 8.67 CPU-h | `P3_2D_REPAIR_READOUT_20260828.md` §1/§7; `cluster/p3_2d_fleet.sbatch:53-57` | 2026-08-27/28 |
| PROD-A0 reproducibility floor | ≤ 8.5e-15 over 12 columns | ledger row #201 (`:2957`) | 2026-08-25 |
| catalogue share of the with-BH mixture (context) | median 0.798 / 0.856 | ledger row #219 (`:3012`) | 2026-08-28 |
| tests at HEAD | 27 functions / 48 cases, all pass | `test_catalogue_numerator_survival_2d.py`; smoke run | 2026-08-29 |
| workspace expiry | 2026-09-23 | `COMPUTE_LEDGER.md`; `HANDOFF_20260730.md:179` | 2026-07-30 |

*Builder/runner independence (standing rule 2): this document's author built no instrument and
ran no registered measurement; the smoke run of the existing test file is the only execution.
Nothing here addresses the author; all items return to the orchestrator and, per row #222, to the
end-of-fan-out verifier.*

---

## 13. APPENDED NOTE (2026-08-29) — falsifier (i) executed; wave-2 arm registered form

**Launched under rows #222/#223 — charter node B7.2-pre (P4).** `[FABLE-B7.2-pre 2026-08-29]`
Node scope: implement and run SS6.1 falsifier (i) as a new unit test, then register the wave-2
arm's final form. Builder/runner independence (standing rule 2) unaffected: no registered
measurement was run, only unit tests.

### 13.1 Falsifier (i) result — HOMOGENEITY HOLDS (PASS, not refuted)

New file: `darksiren_emri_test/bayesian_inference/test_survival_2d_homogeneity_falsifier.py` (4
functions). Ruff/mypy clean; run alongside the existing 48-case
`test_catalogue_numerator_survival_2d.py` (all 52 pass, 2026-08-29). Method: a wrapped
with-BH-survival accessor scaling `S_4D -> c*S_4D`; `T_cat` measured via `single_host_likelihood`
(the real kernel, three synthetic hosts, uniform weight); `T_comp` via `completion_mass_factor_g_sel`
(the real fused completion kernel) on a small synthetic z-grid; the `Σ^4D`-style denominator `D̃`
via the literal per-row point-query formula (SS1.1) against the same three hosts — assembled per
SS1.5's boxed form `combined_wbh = (T_cat + T_comp)/D̃` (β_G_φ/Σ^φ elided as 1.0, established
S_4D-invariant in ratio). `T_comp`/`D̃`'s exact linear-in-c scaling verified directly (not assumed).

| check | result | source (file:line) | date |
|---|---|---|---|
| twin `combined_wbh`, rel. dev. at c=0.4 / c=0.15 vs c=1 | 2.60e-16 / 1.30e-16 (≤ 1e-10 gate) | `test_survival_2d_homogeneity_falsifier.py::test_falsifier_i_twin_combined_wbh_invariant_under_s4d_rescaling` | 2026-08-29 |
| coded `combined_wbh`, rel. dev. at c=0.4 / c=0.15 vs c=1 | 1.500 / 5.667 (≫ 1e-3 asymmetry floor) | `::test_falsifier_i_coded_combined_wbh_not_invariant_under_s4d_rescaling` | 2026-08-29 |
| double-applied-survival defect (A15 capable-of-failing probe), rel. dev. at c=0.4 | 0.600 (probe correctly flags it) | `::test_falsifier_i_detects_double_applied_survival_bookkeeping_defect` | 2026-08-29 |
| `T_comp`/`D̃` exact linear-in-c scaling | confirmed to rtol 1e-10 / 1e-12 | `::test_falsifier_i_completion_and_sigma4d_proxies_scale_exactly_with_c` | 2026-08-29 |

**Verdict on SS6.1(i):** PASS — the twin's `combined_wbh` is homogeneous of degree 0 in a uniform
`S_4D` rescaling to 15–16 orders of magnitude better than the 1e-10 gate; the coded arrangement is
NOT invariant (50–470% relative movement over the same c-range), reproducing exactly the SS1.5
degree asymmetry the adoption argument rests on. The A15 discriminating-power probe (a synthetic
double-applied-survival defect, modelled after the real S̄_φ double-weight pattern, SS5) is
correctly flagged as non-homogeneous (60% relative deviation), so the test is not vacuously
passing. Per SS6.1(i)'s own disposition rule, this result does NOT return the proposal — it is a
confirming falsifier outcome, not the refuting one. This closes regression item R3 (SS8) at the
unit-test level; the full-suite promotion (R1/R2/R6 re-pointing) remains a `/physics-change` gate
task, not this prep node's scope.

### 13.2 STEP-2 smoke item (restated for the record, not executed here)

Per SS6.2 ("Cost"): the `h = 0.730` production task in the wave-2 arm doubles as the STEP-2 smoke
that PINS the `mz_sel` overhead factor, currently only ASSUMED at 1.0–1.3× (mirror-venue evidence:
62.944 s (bt) vs 64.996 s (bc) at 200 events, `cluster/p3_2d_rhs2.sbatch:15-16`, 2026-08-25/26 —
NOT slower at that scale). The overhead is expected to scale as `n_cand × 50 × 24` (per-candidate ×
the 50-node host-z quadrature × the 24-node Gauss-Hermite mass quadrature, SS1.3), since production
candidate counts are ~10³× the mirror venue's. No wave-2 task has run yet at this node; the pin
remains an open measurement for whichever node executes the arm.

### 13.3 Wave-2 arm PROD-CF-2D — final registered form (restating SS6.2, confirmed unchanged)

- **Venue:** iiib only (true reduced catalogue, md5 `c52c13b5…`; `EVAL_SEED = 777000`).
- **Arm T:** `--catalogue_numerator_survival_2d mz_sel --catalogue_numerator_survival_2d_center eff`,
  all else at production defaults; baseline = the banked HEAD readout (`d04d9dc9`), reused only if
  the row-#201 PROD-A0 ingredient gate (≤ 1e-12 relative) passes at h = 0.730.
- **h-grid:** **H4 = {0.660, 0.665, 0.670, 0.730}** (the row-#201 production read node + the
  {0.660, 0.665, 0.670} MAP/mean_h bracket, SS6.2).
- **Gates (zero free parameters):** **R1** (`ln L_cat,wbh^T ≤ ln L_cat,wbh^B` every event/h,
  INSTRUMENT-DEFECT on violation); **R2** (A13 engagement, ≥ 0.95 fraction with
  `|Δ ln L_cat,wbh| > 1e-6` at h = 0.730, STOP below); **R6** (1D channel bit-identical between
  arms, INSTRUMENT-DEFECT on violation). R3–R5 (ΔT, Δw̄₂, Δmean_h,pred) are REPORTED with bands.
- **Materiality band (A8, two-sided):** **T_mat = 0.008** (provenance: `max(node spacing, σ_h/3)`
  at the row-#132 σ_h, ratified row #213 §10 item 4); MATERIAL-UP/DOWN-PREDICTED at `|Δmean_h,pred|
  ≥ T_mat`; IMMATERIAL-PREDICTED at `≤ T_mat/2 = 0.004`; AMBIGUOUS in between or on validity-condition
  violation ⇒ conditional escalation to G27.
- **Cost (A11, instructed band):** **59.7–81.1 CPU-h** (arm only) / **74.7–101.4 CPU-h** (+ baseline
  gate task) at the instructed 14.93–20.27 CPU-h/node anchor; **ceiling 105 CPU-h (arm) / 132 CPU-h
  (total)** at the ×1.3 assumed-overhead upper bound (SS6.2 table). This is the form any execution
  node for charter node B7.2 proper must run against; this prep node authorizes no launch.

**Stamp:** launched under rows #222/#223 — charter node B7.2-pre (P4).

---

## 14. Appended note (2026-08-29 — wave-2 GAP-CLOSURE archive/notes worker, launched under rows
#222/#223 — charter node: NODE archive+minor-notes, GAP 8)

Closes `WAVE2_REGISTRATION_CHECK_20260829.md` §1.5 / §5 item 8 (two minor gaps). Standing rule 1
(append-only) applies — nothing above this section is altered.

1. **Attribution provisional on falsifier (ii).** Falsifier (i) has PASSED (§13.1: twin rel. dev.
   2.6e-16/1.3e-16; coded 1.50/5.67; A15 probe 0.60; 52/52 tests passed). Falsifier (ii)
   (208–286 CPU-h) has not run this wave and returns separately (row #220). **This document's
   attribution of the observed 2D-channel effect to the twin's `S_4D`-homogeneity property is
   therefore PROVISIONAL until falsifier (ii) returns** — a PASS on (i) alone is necessary but not
   sufficient for the full attribution claim. {source: `WAVE2_REGISTRATION_CHECK_20260829.md:174`,
   ledger row #220; 2026-08-29}
2. **Walltime resubmit rule for the STEP-2 overhead pin (§13.2).** The `h = 0.730` production
   task pins the currently-ASSUMED 1.0–1.3× `mz_sel` overhead factor. **Rule (registered here):
   if the h = 0.730 task's measured wall time exceeds `--time=03:00:00` (i.e. the realized
   overhead exceeds the 2.1× implied by the 1.3× assumption against the registered SLURM
   `--time`), the H4 arm is resubmitted with `--time` scaled by the measured overhead factor**
   (measured wall / the un-overheaded anchor wall), applied uniformly across the remaining H4
   nodes. This is a walltime/scheduling adjustment only — it is **not** a band change (§13.3's
   `T_mat = 0.008` and the R1/R2/R6 gates are unaffected) and does not require re-registration.
   {source: `WAVE2_REGISTRATION_CHECK_20260829.md:181`, §13.2 above (this document); 2026-08-29}

**Launch-stamp placeholder (A22).** Wave-2 commit: `<hash at launch>` (does not exist yet — the
working tree was dirty at registration-check time, `WAVE2_REGISTRATION_CHECK_20260829.md` §0/§3
item 1; to be filled by whichever node performs the wave-2 commit; note the untracked falsifier
test file `darksiren_emri_test/bayesian_inference/test_survival_2d_homogeneity_falsifier.py` must
be included in that commit for a clean A22 stamp, per `WAVE2_REGISTRATION_CHECK_20260829.md:178`).
Baseline commit: `d04d9dc9bfe39e6c5a72e768a26f2dcc38355bf5` (the banked HEAD readout,
`run_metadata_21.json`, 2026-08-27T19:40:20).

Stamped: launched under rows #222/#223 — charter node NODE archive+minor-notes (GAP 8), 2026-08-29.
