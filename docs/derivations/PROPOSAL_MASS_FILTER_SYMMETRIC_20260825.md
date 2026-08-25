# PROPOSAL — symmetric σ-window for the with-BH candidate mass pre-filter (the 6-item package)

**Date:** 2026-08-25 · **Thread:** `[P3-WBHZERO]` (rows #191 → #201) · **Status:
PRESENT-THEN-STOP — nothing here is adopted; the decision table (§7) returns to the author as
the registered fresh [RULE] (row #198 binding-default).**

The subject is the with-BH candidate **eligibility window** in
`darksiren_emri/galaxy_catalogue/handler.py`, `get_possible_hosts_from_ball_tree`
(single read site of the landed `mass_filter_sigma` flag, commit `9c948ea0`).

## 1. Old formula (current production, `mass_filter_sigma="asymmetric"`)

A galaxy g (catalogue BH mass M_g, catalogue error σ_g) is with-BH-eligible for a detection
with detector-frame mass estimate M_z, GW mass uncertainty σ_GW, and z-window [z_min, z_max]
iff

```
(M_z − k·σ_GW)/(1 + z_max) ≤ M_g + σ_g      and      M_g − σ_g ≤ (M_z + k·σ_GW)/(1 + z_min)
```

with `k = sigma_multiplier = 1.5`. The GW error enters at ±1.5σ; the galaxy's own error at
±1σ — an asymmetric window. Gate-B (row #196): this asymmetry is recorded nowhere as a design
choice (thesis-era commits, no rationale; MATH_REVIEW F5 and IDEALIZATION_LEDGER I4/I7 flag
the window but never ratify the asymmetry).

## 2. New formula (`mass_filter_sigma="symmetric"`)

```
(M_z − k·σ_GW)/(1 + z_max) ≤ M_g + k·σ_g    and    M_g − k·σ_g ≤ (M_z + k·σ_GW)/(1 + z_min)
```

The single multiplier k = 1.5 applies to BOTH error sources. Implemented and live behind the
flag (default unchanged); adoption = flipping the production default, one read site.

## 3. Reference / derivation

The window is an interval-overlap eligibility criterion: candidate retained iff the k-σ mass
interval of the (redshifted) GW estimate and the k-σ interval of the catalogue mass intersect.
For two independent 1-D uncertainties this is `|μ₁ − μ₂| ≤ k·(σ₁ + σ₂)` — a consistent k-σ
cut treats both σ at the same k (standard compatibility-interval practice; cf. the
neighbouring redshift filter, which uses the same interval-overlap form, at k_g = 1 on the
galaxy side as well — shared convention, equally unratified, explicitly OUT OF SCOPE here).
The old form is the k_GW = 1.5, k_g = 1 special case, for which no derivation, reference, or
recorded intent exists (Gate-B verified). There is no literature basis on record for
k_g ≠ k_GW; the new form is the unique single-k member of the family that preserves the
existing GW-side coverage.

## 4. Dimensional analysis

Both inequalities compare source-frame masses (M☉): M_z/(1+z) maps detector→source frame;
σ_GW scales identically; M_g, σ_g are already source-frame. k is dimensionless. The change
multiplies σ_g (M☉) by k (—): dimensions unchanged on both sides. ✓

## 5. Limiting cases

- **σ_g → 0:** old and new coincide (pure GW window) — the change is null exactly where the
  catalogue is certain. ✓
- **k → 1:** old and new coincide identically. ✓
- **σ_GW → 0:** window → `|M_z/(1+z) − M_g| ≤ k·σ_g` (new) vs `≤ σ_g` (old): the new form
  retains a true host whose catalogue mass scatters by up to 1.5σ of its OWN error — the old
  form rejects at an ordinary 13–18% tail (the event-113 walkthrough, row #196: rejected at
  1.12–1.27σ of its own error with an unremarkable kernel value 0.0082).
- **Filter-open limit:** the filter is a pure numerator selection with NO normalization
  counterpart (Σ^4D sums all masses event-independently; B_num_wbh carries no cut — Gate-B).
  Widening therefore moves the estimator MONOTONICALLY toward the modeled limit (the mass
  kernel itself downweights implausible masses); `n_pass(sym) ≥ n_pass(asym)` is a theorem
  for k ≥ 1 and held with zero violations on every measured event (mirror 2400/2400,
  production 1590/1590).

## 6. Validity conditions + measured evidence register

| # | evidence | value | source |
|---|---|---|---|
| E1 | Mirror verdict (12-seed paired b0i fleet, h=0.73) | **EXCLUSION-MATERIAL**: ΔT̄ = +0.6335 ± 0.0379 (band 0.114, M_T 0.5), Δw̄ = +0.00490 ± 0.00024 (band 0.00073, M_w 0.004), 12/12 seeds positive | row #200, `wbhzero_work/readout.json` |
| E2 | Structural exactness | CF-X 2400/2400 mirror events + 689/689 production retention realized EXACTLY as zero-compute-predicted; monotonicity zero violations | rows #200/#201 |
| E3 | Production counterfactual read (fresh-at-HEAD paired arms, iiib CRB, 1588 rows, h=0.73) | ΔT = +0.800030, Δw̄ = +0.000449 (baseline w̄ 0.120), catalogue-leg zero rate **43.32% → 0.00%** | row #201, `wbhzero_work/prod_readout.json` |
| E4 | Flag-default safety | byte-identical default verified at venue scale (WZ-A0 106/106) and production ingredient scale (PROD-A0, 12 columns ≤ 8.5e-15 over 1588 rows) | rows #200/#201 |
| E5 | Regression tests | 4 committed tests incl. bit-identity + event-113-class retention; suite 1826 green | `9c948ea0`, `20fae087` |

**Validity conditions / open caveats (disclosed):**
1. All Δ measured at h = 0.73 only — **h-dependence unmeasured** (a fresh costing line if
   wanted; the structural mechanism is h-independent but the magnitude need not be).
2. The filter remains an **unmodeled numerator selection under EITHER flag value** — the
   symmetric form reduces the exclusion, it does not model it. The
   model-consistency question (filter vs kernel-weighted eligibility) is a separate,
   un-opened thread; this proposal does not close it.
3. The production Δw̄ (+0.00045) is an order below the mirror's (+0.0049) while ΔT is
   comparable — the venue and production differ in completion-weight structure; recorded as
   measured fact, not interpreted here.
4. The redshift filter's shared ±1σ galaxy-side convention is out of the row-#198 grant's
   scope and unchanged.
5. PA-WBZ-3 re-base: production Δ is measured at TODAY'S physics (both arms fresh at HEAD);
   the banked iiib run carries the since-removed 0.665035804 completion multiplier and served
   as configuration provenance only.

## 7. Decision table — returns to the author as [RULE]

| option | action | consequence |
|---|---|---|
| **(a) ADOPT symmetric (recommended)** | flip the production default `"asymmetric"` → `"symmetric"` at the single read site, `[PHYSICS]` commit + gate-ledger row; explicit `"asymmetric"` remains the counterfactual | the 43.3% starvation class is eliminated; the estimator moves toward its modeled limit; caveat 2 stays open and tracked |
| (b) RATIFY asymmetric | document the ±1.5σ/±1σ window as a deliberate design choice (systematics-budget row + idealization ledger) | the measured ΔT/Δw̄ become a tracked systematic; no code change |
| (c) DEFER pending h-dependence | fresh costing line for a multi-h production read first | the [P3-2D] un-HOLD waits on that measurement |

On (a) or (b), [P3-2D] un-HOLDs calibrated against the chosen eligibility model (A21
amendment + the M2-LINK re-attribution, per rows #194/#196/#198).

**Recommendation: (a)** — the asymmetry is an undocumented accident with a Gate-B-verified
material, direction-predicted, structurally-exact measured effect, and the symmetric form is
the unique consistent single-k window preserving current GW-side coverage.
