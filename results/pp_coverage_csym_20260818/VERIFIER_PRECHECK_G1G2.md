# VERIFIER PRE-CHECK — G-1 (catalogue-leg symmetry) & G-2 (spec-z limit)

**Date:** 2026-08-18 · **Role:** adversarial pre-registration verifier, maximum scrutiny ·
**Read-only on everything except this file.** Named failure mode hunted: the V-ctrl-class
structurally-void arm (prodcal VERDICT: z_support 1.5 > Z_MAX_POP 0.95 ⇒ completion_fraction ≡ 0,
discovered only at readout).

**Verdicts:**

| draft | verdict |
|---|---|
| `PREREGISTRATION_G1_CATLEG_SYMMETRY.md` | **GO-WITH-AMENDMENTS** (3 BLOCKING, 4 NON-BLOCKING) |
| `PREREGISTRATION_G2_SPECZ_LIMIT.md` | **GO-WITH-AMENDMENTS** (1 BLOCKING, 6 NON-BLOCKING) |

No further verifier pass required if the BLOCKING amendments are applied verbatim.

---

## Part I — Checks that ran and PASSED (with recomputed numbers)

### 1. Arm-voidness (the named failure mode) — all registered cells non-degenerate

- **Completion windows:** V-deep z_support = 0.40 and V-prod z_support = 0.75 are both <
  `Z_MAX_POP = 0.95` (`pp_coverage.py:286`); the on-disk twins measure completion_fraction
  0.336 (V-deep) / 0.374 (V-prod) at R=120 — no empty window anywhere in either draft. G-2's
  single venue is V-deep at all rungs; completion_fraction is a z_support-truncation property,
  σ_z-independent — non-void at every rung.
- **Catalogue leg populated:** realized catalogue-bearing (host-in-ball) fractions 0.664
  (V-deep) / 0.626 (V-prod), both above the registered 0.5 (N-B) and 0.3 (preflight) floors.
- **S̄_φ amplitude where hypotheses need it:** at V-deep (d50 = 1.85 = default `D50_GPC`) S̄_φ
  falls ~1 → 0.31 across the catalogue range [0, 0.40] and 0.31 → ~0 across the completion
  window — the `cat1d`/`symmetric` levers have amplitude for H-SYM/H-CAT and for G-2's rung-1
  engagement preconditions. At V-prod (d50 ×8) S̄ ≥ 0.95 everywhere — flat, which is exactly
  what H-G1's **null** prediction requires; H-G1 needs no amplitude, so flatness is not voidness.
- **Survival-table support:** `phi_marginal_survival_table` is built whenever
  `mass_channel=True` (`run_coverage`, current tree ~2434) — so `cat1d` (excluded from the
  '1d'/'fused' ValueError guard) cannot hit a None table; and the table's z-grid is
  `np.linspace(Z_MIN, Z_MAX_POP, n_z)` (line 1713) — full-range, so §3b preflight check 4
  (table covers completion window AND catalogue kernel range) is satisfiable by construction.
- **σ_z = 0.002 vs numerical guards:** no lower-bound guard exists on sigma_z; `_norm_pdf`
  requires only σ > 0; the smeared-density grid self-scales (`dz = sigma_z/16`,
  `_smeared_catalogue_density`, lines 1264–1279 → ~8×10³ bins at σ_z = 0.002 — cost, not
  failure). The genuine small-σ_z hazard is `n_z_quad` undersampling of the candidate kernels
  (default 160 nodes over a ~0.3–0.5-wide window ⇒ spacing ~4×10⁻³ ≫ σ_z = 0.002) — which is
  precisely what G-2's registered Q* pretune + N-c gate exist to handle. Correctly designed;
  one disclosure gap on the *fixed* internal grids → A-G2-6.
- **selection_cell is estimator-side only** (validates the deliberate seed-reuse pairing): the
  generative path (`p_draw`, host sampling, eps, cap perturbation, `_build_catalogue` at
  `rng(seed+1)`) contains no `selection_cell` reference; the diff of the working-tree extension
  against the frozen `fe72d52b` is purely additive on pre-existing modes (`common_1d = common`
  when not cat1d/symmetric; `sel_1d/sel_2d` membership extended only by "symmetric"). Pending
  t1/t4 as registered, the design is stream-valid.
- **G-2 σ_z-continuity of the stream** (cross-rung alignment claim): catalogue scatter is
  `z_true + rng.normal(0.0, sigma_z, n)` (line 1323) — the scale parameter does not change RNG
  stream consumption, so latent draws align across rungs. Verified TRUE.
- **Would §3b have caught the historical V-ctrl defect?** Yes, twice over: completion_fraction
  ≡ 0.0000 violates preflight check 1 (∈ [0.05, 0.95]) and the bit-degenerate fused−off pair
  violates check 2 at R=4 — before any scored realization. The gate is real, in both drafts.

### 2. Number fidelity — verified against the artifacts of record

| drafted number | artifact | verdict |
|---|---|---|
| V-prod paired deltas +0.000967±0.000157 / +0.000700±0.000139 / +0.000033±0.000033 | `readout_prodcal_output.json` pair (vprod fused, vprod off) | **exact** |
| V-flat paired SEs ±1.7e-4/±1.4e-4 | same, vflat pair (0.000171/0.000142/0.000033) | **exact** (third truth's ±0.33e-4 omitted, harmless) |
| fused state −0.034, cov68 0.050 (V-deep n=250) | vdeep_250_production_fused, h=0.72: −0.03397, 0.050 | **exact** |
| off −0.0013…−0.0036 | vdeep_250_production_off: −0.00130/−0.00180/−0.00360 | **exact** |
| S-1 1d ≈ −0.033 | vdeep_250_production_1d h=0.72: −0.0328 (absolute); paired (1d−off) = −0.031 | **OK**; note G-2's C_comp is *paired* — anchor off by ~6% (see A-G2-2 note) |
| completion anchors 0.6705/0.3295; 0.384 on 0.371 | `pretuning/CHOSEN.json`, `CHOSEN_VPROD.json` (0.3835), prodcal §7 (0.371) | **exact** — except "0.616" (A-G1-5) |
| cell timings 416 s / 305 s | `prodcal_ladder_6355028.out` (vdeep off 416 s), `vprod.log` (off 305 s) | **exact**; fused 1000 s / vprod fused 640 s ⇒ SYM estimates (1200/770 s) conservative ✓; measured `1d` = 167 s vs G-2's assumed 500 s, and `cat1d` skips the expensive completion `g_sel` loop, so ≈1000 s is a large overestimate — budgets conservative, ceilings unaffected |
| materiality 0.0018 = ⅓ × 0.0053 | 0.001767 → 0.0018 | **exact** |
| audit trio +0.008 / −0.029 / +0.041 | claim card `CLAIM_SYMMETRIC_SELECTION_INSERTION_20260818.md` [LOCAL] tags | **exact** (and both drafts correctly carry them as ad-hoc/UNVERIFIED) |
| line refs 1566, 2006-2010, 620, 2609, 2686-2691 | `git show fe72d52b:...pp_coverage.py` | **all exact against the frozen commit** (the working tree has shifted them — the extension is already drafted in-tree, uncommitted, tests t1–t5 not yet written; consistent with "frozen in the same commit as this file") |

**Misquotes found:** the paired-SE class "1.4–1.7e-4" is real but is the **flat-venue
(V-prod/V-flat) class**; the **V-deep** fused−off paired SE at n=250 is measured
**7.1–8.1e-4** (0.000771/0.000811/0.000705) and V-deep absolute-bias SEs are 0.8–1.6e-3 (not
5.5e-4). Every V-deep power/false-fail figure in both drafts inherits this (A-G1-3/4, A-G2-1/2).

### 3. Seed / pairing logic

- 20270818 = seed of all on-disk `vdeep_250_*` cells; 20271218 = seed of both `vprod_250_*`
  cells (verified in the cell JSON configs). The reuse-for-pairing design is valid given the
  estimator-side-only finding above.
- **20280311 / 20280399 collision check:** extracted every `"seed":` value under `results/`.
  Neither collides. **But** the drafts' stated inventory is wrong:
  `results/calibration_gate_v2_20260810` + `results/venue_transfer_20260811` consumed
  20260805–20261004 **and 20280808–20306007** (4650 seeds ≥ 2028xxxxxxx-class) — the 2028
  block starts 497 above 20280311. Freshness holds; the inventory statement doesn't → A-G2-3.

### 4. Both-fire analysis

- G-1 H-G1: PASS edge ≤ max(0.0010, 2·SE) with the SE ≤ 3.0e-4 guard ⇒ effective edge 0.0010 <
  FAIL edge 0.0018 — disjoint ✓. H-SYM (0.012 vs 0.020/cov 0.50 vs 0.20) disjoint ✓. H-CAT
  (≥ +0.010 vs ≤ +0.005) disjoint ✓.
- G-2 H-Zcomp (⅔ vs 0.10) disjoint ✓. **G-2 H-Zcat can both-fire** (A-G2-1, BLOCKING).

### 5. A8-v2 recomputations (clause e)

- G-1: 0.0018/3.0e-4 = 6.0σ ✓; 0.0010/3.0e-4 = 3.3σ ✓; 0.0010/1.5e-4 = 6.7σ ✓; binomial SE
  √(0.68·0.32/120) = 0.0426 ✓; (0.68−0.50)/0.0426 = 4.2σ ✓; (0.50−0.05)/0.0426 ≈ 10.5σ ✓.
  WRONG at realized SEs: "~76σ apart" → ~30σ; "≥7σ from the fused state" → derivation opaque
  (0.022/5.5e-4 = 40σ; 16σ at realized 1.4e-3 — conclusion survives) → A-G1-4.
- G-2: 0.022 and 0.0033 edges ✓; "~110 paired-SE" = 0.0187/1.7e-4 ✓ *arithmetically* but ~23
  realized V-deep SE; **"≥45 SE" for H-Zcat does not follow from the registered precondition
  minimum on any SE anchor**: (⅓−0.10)·0.010 = 0.00233 → 15.5σ at the draft's own 1.5e-4, 2.9σ
  at the realized 8e-4 class (45+ requires the UNVERIFIED +0.041 baseline) → A-G2-1/2.

### 6. Instrument-spec coherence (G-1 §0 cat1d/symmetric)

Implementable and in fact already drafted (working tree, lines 2339–2352): S̄_φ(z;h) from the
SAME table object multiplies `common` before the z-sum of the 1D catalogue term — algebraically
identical to a per-candidate insertion inside each z-kernel integral since `rho_1d` is the
kernel sum; `inv_norm` (Z_g) untouched ⇒ "no new normalization" is well-defined and honored;
`symmetric` = fused completion legs (sel_1d AND sel_2d) + the cat factor; `cat1d` = off on the
completion leg. Matches the [C-SYM] claim's registered form ("paired against the same α(h)", no
per-leg renormalization — the refuted FULL-C/E route is not reintroduced). One residual
ambiguity worth the build note (already provided for by the CC-3 clause): the constant part of
S̄ does NOT normalize out in `absolute` mixture mode, so at V-prod the symmetric−fused delta
includes a ~5% catalogue-leg level-downweight, not only a gradient effect — this IS the claim's
registered form, so it is the right object; no amendment.

### 7. Design gaps probed and NOT found

- Raised-d50 confound (VERDICT-3): G-1 excludes V-prod absolute legs (N-C, registered
  non-read) ✓; the H-G1 paired read is common-mode-immune at first order ✓. G-2 runs no
  flat-S̄ rung and says why (caveat 2) ✓.
- +0.01-class noise-coupling class: correctly quarantined by G-2 caveat 5 (1D carries the
  read); G-1 fails to quarantine it → A-G1-1 (BLOCKING).
- H-SYM firing [C-SYM]'s Refute-by from V-deep rather than the claim-named V-flat: disclosed
  and argued in §1 ("strictly stronger" — correct for the asymmetry-ownership clause: if the
  asymmetry owns the displacement, the symmetric form must calibrate where amplitude is
  largest); every branch returns as [RULE], so the venue substitution reaches the author. No
  amendment.
- Execution-completeness (A8-v2 d): explicit in both drafts ✓.

---

## Part II — Amendments (exact quotable text)

### G-1 — `PREREGISTRATION_G1_CATLEG_SYMMETRY.md`

**AMENDMENT G1-1 [BLOCKING — A8-v2 (b) branch-referent / check 1 arm-voidness].**
The §4 reads never name a channel. The V-deep 2D channel carries a venue-intrinsic
+0.0077…+0.0101 bias with cov68 0.375–0.483 **in the `off` twin itself** (measured,
`readout_prodcal_output.json`, vdeep_250 cells — the §9-flag-3 noise-coupling class). On the 2D
channel, H-SYM's PASS band (cov68 ≥ 0.50 at every truth) is therefore unsatisfiable even by the
calibrated reference configuration — a structurally void read of exactly the V-ctrl class.
In §4, old text:

> Per truth; paired reads on the shared stream; "delta" = per-realization paired MAP delta mean.

new text:

> Per truth; paired reads on the shared stream; "delta" = per-realization paired MAP delta mean.
> **All registered reads (H-G1, H-SYM, H-CAT, N-A, N-B) are scored on the 1D channel
> (`channel_1d`) only; 2D-channel statistics are reported descriptively and are never
> verdict-bearing** — the V-deep 2D channel carries a venue-intrinsic +0.008…+0.010 bias with
> cov68 0.375–0.483 in the `off` twin itself (the §9-flag-3 noise-coupling class, present with
> the fusion off), so no absolute 2D band is satisfiable at this venue.

And append to §6 as caveat 7:

> 7. **2D-channel absolute statistics are venue-confounded at V-deep** (the +0.01-class
>    noise-coupling bias, present identically in `off`); they are reported descriptively only
>    (mirrors G-2 caveat 5).

**AMENDMENT G1-2 [BLOCKING — A8-v2 (b) / check 4 pairing logic].**
The on-disk `vdeep_250_*` cells were computed on bwUniCluster (job 6355028, DEVIATION-1); the
new cells run on the dev machine. A cross-environment floating-point difference (BLAS kernel
dispatch differs across CPU microarchitectures; `_posterior_normalizers` uses a BLAS matvec)
can satisfy N-A's "any difference" condition **without its registered meaning** ("the extension
broke pre-existing modes or the stream"). REP-OFF-P vs the vprod cells is local-vs-local and
carries no such exposure. In §1 N-A, old text:

> Any diff = the extension broke pre-existing modes or the stream: STOP, audit before
> interpreting anything.

new text:

> Any diff ⇒ STOP before scoring. **Registered disambiguation (the V-deep referent is
> cross-environment — cluster job 6355028 vs the dev machine — while the V-prod referent is
> local-vs-local):** if REP-OFF-D differs while REP-OFF-P is bit-exact, rerun the V-deep `off`
> cell once at the pre-extension freeze (`fe72d52b` chain) on the same dev machine. If that
> rerun ALSO differs from the on-disk cluster cell, the difference is environmental, not an
> instrument break: the local frozen-commit rerun becomes the N-A byte-identity referent, and
> the paired twins for P1 and the H-SYM/H-CAT context must then be locally rerun cells (same
> seeds; ≤ 0.4 CPU-h added, inside the ceiling), not the on-disk cluster cells. Only a diff
> that survives this environment control is scored as "extension broke the instrument": STOP,
> audit.

**AMENDMENT G1-3 [BLOCKING — check 1 (designed-void guard scope) + A8-v2 (b)].**
§2's guard sentence is pair-generic while §4 attaches it only to H-G1/P4. The measured V-deep
paired-SE class at n=250 is 7.1–8.1e-4 (fused−off, `readout_prodcal_output.json`) — if the
3.0e-4 guard were read to cover the V-deep pairs (P1–P3), every V-deep paired read would be
UNDETERMINED-BY-NOISE **by construction**: a designed-void arm. In §2, old text:

> registered guard: the read is scored only if realized paired SE ≤ 3.0e-4, else
> UNDETERMINED-BY-NOISE with the designated next measurement = the same pair at n=800

new text:

> registered guard (**scoped to P4 / H-G1 only** — the V-deep pairs P1–P3 carry the measured
> V-deep paired-SE class 7.1–8.1e-4 at n=250 and are scored against their own 10×-larger
> edges): the P4 read is scored only if realized paired SE ≤ 3.0e-4, else
> UNDETERMINED-BY-NOISE with the designated next measurement = the same pair at n=800

**AMENDMENT G1-4 [NON-BLOCKING — A8-v2 (e)].** Replace the wrong-venue SE anchors in §2/§4:
(i) old "absolute-bias SE at n=250 ≈ 5.5e-4 ⇒ the states are ~76σ apart; the PASS band ±0.012
sits ≥ 7σ from the fused state" → new "absolute-bias SE at n=250, measured at this venue:
0.8–1.6e-3 (`off` 0.00082–0.00133, `fused` 0.00020–0.00158) ⇒ the states are ≥ 26σ apart; the
PASS edge 0.012 sits ≥ 13σ from the fused state and ~2.5–5σ from the audit's +0.008
expectation (false-fail ≲ 2% compounded over 3 truths)". (ii) old "H-CAT: +0.041-class vs
paired SE ≤ 3e-4 — trivially powered" → new "H-CAT: +0.041-class vs the measured V-deep
paired-SE class 7–8e-4 — still ≥ 50σ, trivially powered". (iii) §1 old "(an order beyond the
ad-hoc value)" → new "(≈ 2.4× the ad-hoc value)". (iv) §4 band-derivations line, old "≥ 7σ
from the fused state at SE 5.5e-4" → new "≥ 13σ from the fused state at the measured absolute
SE class ≤ 1.6e-3".

**AMENDMENT G1-5 [NON-BLOCKING — check 3].** §3b item 1, old text:

> (V-deep anchor 0.6705/0.3295; V-prod anchor 0.616/0.384 — CHOSEN files)

new text:

> (V-deep anchor 0.6705/0.3295 — `pretuning/CHOSEN.json`; V-prod: completion 0.384 —
> `CHOSEN_VPROD.json`, which carries no host-in-ball field; realized catalogue-bearing
> fraction 0.626 at R=120 — `vprod_250_production_off` diagnostics)

**AMENDMENT G1-6 [NON-BLOCKING — A8-v2 (b)-adjacent completeness].** Append to the §4 branch
calls:

> - **H-CAT FAIL** (delta ≤ +0.005 at h = 0.72 or negative coherent sign): the audit's
>   mechanism table (catalogue-only +0.041, opposite sign to completion-only) is contradicted
>   at registered fidelity; returns to the author as a [RULE] annotated onto the [C-SYM]
>   intake card, and any H-SYM interpretation that leans on the two-leg sign-cancellation
>   story is barred until re-derived.

**AMENDMENT G1-7 [NON-BLOCKING — execution completeness of the scorer spec].** §3/§7: the P1/P4
reference cells live in `results/pp_coverage_prodcal_20260817/cells/` while the new cells live
in this directory; register that the `readout_g1.py` PAIRS manifest carries explicit
per-cell paths (or that the prodcal reference cells are copied into `<cells_dir>` with a
provenance line), so `--registered <cells_dir>` is a complete invocation.

### G-2 — `PREREGISTRATION_G2_SPECZ_LIMIT.md`

**AMENDMENT G2-1 [BLOCKING — A8-v2 (c) two-sidedness / check 6 both-fire].**
H-Zcat's PASS edge max(0.10·|C_cat(0.035)|, 2·paired-SE) exceeds its FAIL edge
⅓·|C_cat(0.035)| whenever paired SE ≥ |C_cat(0.035)|/6 — i.e. ≥ 1.67e-3 at the registered
precondition minimum 0.010. The measured V-deep paired-SE class at n=250 is 7.1–8.1e-4 (not
the flat-venue 1.4–1.7e-4 the draft anchors), a factor ~2 from overlap: PASS and FAIL could
both fire on one measured number. In §4 H-Zcat row, old text:

> \|C_cat(0.002)\| ≤ max(0.10·\|C_cat(0.035)\|, 2·paired-SE) AND \|C_cat\| monotone
> non-increasing across rungs (2·SE tolerance per step)

new text:

> \|C_cat(0.002)\| ≤ max(0.10·\|C_cat(0.035)\|, 2·paired-SE) AND \|C_cat\| monotone
> non-increasing across rungs (2·SE tolerance per step). **Overlap guard (registered):** if
> 2·paired-SE(rung 0.002) ≥ ⅓·\|C_cat(0.035)\| the PASS and FAIL windows intersect — the
> H-Zcat read is then UNDETERMINED-BY-NOISE (unscored; designated next measurement: the same
> rung-3 pair at n = 800, returns as an amendment), and neither PASS nor FAIL is adjudicated.

**AMENDMENT G2-2 [NON-BLOCKING — A8-v2 (e)].** §2, old text:

> Paired-delta SE class at n=250, R=120, shared stream: 1.4–1.7e-4 (measured, prodcal
> VERDICT-2/3). Dynamic range: C_comp baseline ≈ 0.033 ⇒ the ⅔-persistence edge (0.022) vs the
> 0.10-collapse edge (0.0033) are separated by ~110 paired-SE — the H-Zcomp read is
> effectively noise-free; likewise H-Zcat if its precondition holds (baseline ≥ 0.010 ⇒ edges
> separated by ≥ 45 SE).

new text:

> Paired-delta SE classes at n=250, R=120, shared stream (measured, prodcal record): the
> flat-venue near-null class is 1.4–1.7e-4 (V-prod/V-flat pairs); **at V-deep — this
> registration's venue — the engaged-lever class is 7.1–8.1e-4** (fused−off), and the paired
> SE shrinks with the lever, so collapsed-rung pairs are expected near the near-null class.
> Dynamic range: C_comp baseline ≈ 0.031 (paired 1d−off; the −0.033 figure is the `1d` cell's
> absolute bias) ⇒ the ⅔-persistence edge (0.021) vs the 0.10-collapse edge (0.0031) are
> separated by ~23 realized paired-SE — decisively powered; H-Zcat at the precondition minimum
> (baseline 0.010) has edges separated by 0.0023 ≈ 3 realized SE at the rung-1 class — the §4
> overlap guard owns the residual risk, and the separation is ≥ 14 SE if the rung-3 pair
> lands in the near-null SE class as expected.

**AMENDMENT G2-3 [NON-BLOCKING — check 3/4 number fidelity].** §3, old text:

> Seed freshness: 20280311 and 20280399 lie outside every seed range consumed in `results/`
> (prodcal family 20270818–20271333, coverage family 20260701–20261207, G-1's deliberate
> reuses).

new text:

> Seed freshness (verified against the full extracted `"seed"` inventory of `results/`,
> 2026-08-18): 20280311 and 20280399 collide with nothing. Consumed families: prodcal
> 20270818–20271333; coverage 20260701–20261207; calibration-gate/venue-transfer harnesses
> 20260805–20261004 **and 20280808–20306007** (the nearest consumed seed, 20280808, is 497
> above 20280311; a different instrument in any case); G-1's deliberate reuses.

**AMENDMENT G2-4 [NON-BLOCKING — instrument-spec coherence].** §1 N-c / §3: the pretune
certifies Q* against the "next larger" sweep value (160→240 is ×1.5, not a doubling), while
N-c demands a "doubling Q*" probe — if Q* = 160 lands, the 320 probe is a cell no clause
registers. Old text (§1 N-c): "at the smallest rung, doubling Q* on an R=8 probe changes the
per-truth MAP bias by ≤ 0.0005" → new text: "at the smallest rung, raising Q* to the smallest
registered sweep value ≥ 2·Q* (160→480, 240→480, 480→960) on an R=8 probe (seed 20280399,
probe-flagged, archived under `preflight/`) changes the per-truth MAP bias by ≤ 0.0005".

**AMENDMENT G2-5 [NON-BLOCKING — wording under caveat 1].** §4 first branch call, old text:

> "…the completion-term cost is a population-integral property, present at any σ_z"

new text:

> "…the completion-term cost is a population-integral property, persisting to spectroscopic
> precision (σ_z = 0.002, measured) and stated as a trend extrapolation below that (caveat 1)"

**AMENDMENT G2-6 [NON-BLOCKING — budget/instrument disclosure].** Append to §3 Budget:

> Disclosure: at the plausible Q* ∈ {480, 960} the z-quadrature-dominated cell cost scales
> ~×3–6, projecting the grid at ~4.8–9.6 CPU-h against the 6 CPU-h ceiling — the registered
> κ-STOP is *likely* to fire at the first pretune readout and its firing is the designed
> author-return, not an anomaly. Also registered: the N-c convergence evidence covers
> `n_z_quad` only; the fixed internal grids (`_posterior_normalizers` ngrid = 3000 → spacing
> 3.2e-4 ≈ σ_z/6 at rung 3; D(h)/β_G at 3000 nodes; `n_z_survival` = 1500 on the smooth
> σ_z-independent S̄ table) are adequate by spacing argument, not by the doubling probe.

---

## Part III — Findings that were probed and REFUTED (no defect)

1. "Stale line references": all cited line numbers verified exact against the frozen commit
   `fe72d52b` (the natural referent) — SELECTION_CELLS:1566, dispatch 2006–2010, sigma_z
   config ~620, `--sigma-z` 2609, `--n-z-quad` 2686–2691.
2. "cat1d could crash on a missing survival table": refuted — table built for any
   `mass_channel=True` run, full z-range support.
3. "σ_z = 0.002 hits a numerical guard": refuted — no guard exists to hit; the σ_z/16 grid
   scales; the real hazard (n_z_quad) is owned by the registered pretune + N-c.
4. "Seed collisions": refuted for 20280311/20280399 (direct membership check against the full
   inventory) — though the drafted inventory statement itself was wrong (A-G2-3).
5. "Symmetric lever void at V-prod": refuted — H-G1 registers a null prediction there; the
   constant-S̄ level effect does not normalize out in `absolute` mode, so the read is
   non-degenerate and is the claim's registered object.
6. "G-2 cross-rung stream misalignment": refuted — `rng.normal(0, σ_z, n)` scale does not
   alter stream consumption; alignment claim verified in code.
7. Budget arithmetic in both drafts: verified against measured timings; all estimates
   conservative (cat1d ≈ 1000 s is likely a ~2× overestimate — it skips the completion
   `g_sel` loop that makes `fused` expensive; measured `1d` = 167 s).

---

*Verifier of record: adversarial pre-check session 2026-08-18. Read-only audit; no registered
file was edited. Amendments are quotable verbatim; BLOCKING items G1-1, G1-2, G1-3, G2-1 must
land in the freeze commit before any probe or scored realization runs.*

---

## Part IV — One-item pre-check of PRE-FREEZE AMENDMENT A (both drafts), 2026-08-18

**Verdicts: G-1 AMENDMENT A — GO-WITH-AMENDMENTS (1 BLOCKING, 2 NON-BLOCKING). G-2 AMENDMENT A
— GO-WITH-AMENDMENTS (2 BLOCKING, 1 NON-BLOCKING).** The amendment's core design — wide grid,
all-local same-grid twins, narrow-grid N-A against the environment-control referent, rail gate
— is sound and is the correct response to both triggers.

### (iv) N-A referent migration — FAITHFUL

Verified against AMENDMENT G1-2's registered text: the local pre-extension (clean worktree at
the committed pre-freeze HEAD = the `fe72d52b` chain, since the extension is uncommitted) rerun
becomes the V-deep byte-identity referent; REP-OFF-P stays local-vs-local against the on-disk
vprod cell; and the paired twins migrate to locally-run cells. N-A cells correctly keep the
**original** grid [0.60, 0.86] — mandatory, because the environment-control referent was run on
it, and because the wide grid changes interior logL values (see (i) below), so byte-identity is
only defined same-grid. Item 4 satisfies all of this. The wide-grid twins go beyond G1-2 (same
grid AND same environment as the science cells) — the strongest paired form; stream validity is
preserved since the grid is estimator-side only. **PASS.**

### (ii) Headroom sufficiency — SUFFICIENT, but the disclosed arithmetic uses the wrong σ

The amendment's "3σ" figures (0.885, 0.583) are computed with σ ≈ 1.3e-3 — the SE of the
*mean*. Rail fraction is a **per-realization** statistic; the relevant spread is the per-trial
map_std, measured 0.011–0.015 at V-deep n=250 (map_bias_se × √120). Recomputed per-trial rail
probabilities on the wide grid:

- top edge, cat1d-class center 0.84 + 0.041 = 0.881: P(MAP > 0.92) = 0.02–0.5% per trial —
  far under the 0.10 gate ✓;
- bottom edge, fused-class center 0.62 − 0.034 = 0.586: P(MAP < 0.56) = 0.9–4.2% per trial —
  inside the gate, with real but adequate margin ✓ (the prodcal 99% floor-rail at 0.62 was the
  n=1600 distribution against the old 0.60 floor; at n=250 on the wide grid the expected rail
  fraction is ≲ 5%).

Censoring note (no amendment needed, record it): at h_true = 0.84 the grid can only resolve
positive catalogue-leg deltas up to ≈ +0.08 before railing, so H-CAT's "delta ≤ +0.10
everywhere" leg is decided at 0.84 via the rail gate (UNDETERMINED-BY-RAIL), not by
measurement; H-CAT's PASS/FAIL coherence legs live at 0.62/0.72, which have full headroom
(0.62/0.72 + 0.10 = 0.72/0.82 < 0.92). **PASS with the A-PF-2 disclosure fix.**

### (i) Does the grid extension break any registered read's meaning? — one false claim found

Grid alignment verified: 0.60 − 0.56 = 10 steps, 0.92 − 0.86 = 15 steps; interior **nodes**
align exactly. But interior **values** are NOT identical across grids: the per-event
z-quadrature windows are built from `h_grid.min()`/`h_grid.max()`
(`_completion_numerator_batch` lines 2000–2004; catalogue path 2270–2277), so widening the grid
widens every z-window ~±7% and — with `n_z_quad` fixed — coarsens the z-spacing ~12%. Three
consequences:

1. **G-2's N-b justification "interior MAPs identical on aligned grids" is FALSE** as stated.
   N-b itself survives (it is a 3·combined-SE class comparison; quadrature-level shifts are far
   below that), but the false identity claim must not stand in a registered file → A-PF-3.
2. **G-2's Q\*/N-c convergence evidence must be produced on the wide grid**: a Q\* certified on
   the narrow grid inherits ~12% coarser z-spacing on the wide grid at σ_z = 0.002 — the exact
   hazard the gate exists to bound. AMENDMENT A re-runs the *preflight* on the wide grid but
   does not re-scope the *pretune sweep* or N-c → A-PF-4 (BLOCKING).
3. G-1's §2 anchors correctly demote to cross-checks (item 5); H-SYM/H-CAT band edges are venue
   properties and stand; realized wide-grid SEs at h = 0.62 will grow as the rail-compressed
   distribution de-censors — bands self-adapt (2·SE tolerances) or are absolute with ≥ 13σ
   margins. No A8-v2 clause broken by the extension itself.

### (iii) New both-fire or void arm — one adjudication gap introduced

The rail gate creates an unresolved interaction with "at every truth" PASS legs: if one truth
is UNDETERMINED-BY-RAIL, H-SYM's PASS ("at every truth") becomes unsatisfiable while its FAIL
("≥ 2 truths") remains satisfiable — an asymmetric partial-void where PASS is barred by a
validity flag rather than by data. Precedence must be registered → A-PF-1 (BLOCKING). No new
both-fire found: the gate is a validity state, not a band, everywhere else; N-b's
railed-truth-exclusion clause is coherent (prodcal off cells have rail_fraction ≈ 0 anyway).

Budget: G-1 item 7 arithmetic verified (5331 s × 1.29 + 1137 s ≈ 2.2 CPU-h); but the ×1.29
multiplier's derivation is unstated and the node-count ratio is 91/66 ≈ **1.38** (upper bound;
per-cell cost is sub-linear in nh). Ceilings hold even at 1.38 (G-1: ≈ 2.4 < 3; G-2: ≈ 2.2·κ
with the κ-STOP unchanged) → A-PF-2 note.

### Part-IV amendments (exact quotable text)

**A-PF-1 [BLOCKING — check (iii), A8-v2 (b)] — both drafts.** Append to the rail-gate item
(G-1 item 6; G-2 rail-gate sentence):

> **Precedence (registered):** band legs quantified "at every truth" are evaluated over the
> non-rail-flagged truths only, and any PASS/FAIL adjudication requires ≥ 2 scoreable truths;
> a read with ≥ 2 truths UNDETERMINED-BY-RAIL is itself UNDETERMINED-BY-RAIL (unscored,
> returns to the author with the rail diagnostics). A rail-flagged truth never counts toward a
> "coherent at ≥ 2 truths" FAIL leg.

**A-PF-2 [NON-BLOCKING — A8-v2 (e)] — G-1 item 1 and item 7.** Old text: "(0.84 + 0.041 + 3σ ≈
0.885 < 0.92; 0.62 − 0.034 − 3σ ≈ 0.583 > 0.56)" → new text: "(per-trial map_std class
0.011–0.015, measured V-deep n=250: P(MAP > 0.92 | center 0.881) ≈ 0.02–0.5% and
P(MAP < 0.56 | center 0.586) ≈ 0.9–4.2% per trial — both under the 0.10 rail gate)". Old text:
"Cost +~29% per cell" → new text: "Cost ≤ +38% per cell (node ratio 91/66; sub-linear in
practice — ×1.29 estimated), ceiling holds at either figure". Mirror the cost line in G-2.

**A-PF-3 [BLOCKING — check (i), false claim in a registered file] — G-2.** Old text:

> N-b continuity (off rung-1 vs the prodcal off cell) is unaffected (interior MAPs identical
> on aligned grids; a railed prodcal comparison truth is excluded from N-b, flagged).

new text:

> N-b continuity (off rung-1 vs the prodcal off cell) is retained as registered: interior grid
> NODES align, but interior logL values shift slightly on the wide grid (the per-event
> z-quadrature windows derive from h_grid.min()/max(), so they widen ~±7% and coarsen ~12% at
> fixed n_z_quad) — shifts far below N-b's 3·combined-SE tolerance, which is why the class
> comparison stands; a railed prodcal comparison truth is excluded from N-b, flagged.

**A-PF-4 [BLOCKING — check (i)/(iii) void-arm prevention] — G-2.** Append to the amendment:

> The registered pretuning sweep (Q\* selection, seed 20280399) and the N-c doubling probe run
> **on the wide grid [0.56, 0.92]** — the grid whose z-window coarsening the convergence gate
> must bound; any Q\* evidence produced on the narrow grid is void for scoring.

*Part-IV verifier of record: same session, one-item pre-check under the standing discipline.
BLOCKING: A-PF-1 (both), A-PF-3, A-PF-4 (G-2) — apply verbatim in the freeze commit.*

---

## Part V — One-item pre-check of the N-A comparison-scale clause (G-1), 2026-08-18

**Verdict: GO — zero BLOCKING amendments; 3 NON-BLOCKING wording/order amendments below.**
The clause's central factual claim was **independently reproduced by this verifier**, not taken
from the builder's report.

### Independent verification of the bit-exactness claim

Recomputed directly from the artifacts:
`cells/referent_preext_vdeep_250_production_off.json` (local pre-extension R=120 rerun) vs
`results/pp_coverage_prodcal_20260817/cells/vdeep_250_production_off.json` (on-disk cluster
cell, job 6355028): **configs identical field-for-field (zero diffs); 1D `maps` arrays
bit-identical (`tobytes()` equality, maxabsdiff 0.0) at all three truths; 2D-channel `maps`
bit-identical at all three truths.** The clause's [LOCAL] claim stands on this verifier's own
recompute.

### (ii) Void-at-probe-scale rule — SOUND, with one mechanism precision

Code basis (`run_coverage`, current tree lines 2509–2534): one master
`np.random.default_rng(config.seed)` feeds per-realization child seeds
(`master.integers(1 << 62)`) sequentially through the `for h_true in truths: for _ in
range(R)` loop, and the shared catalogue uses the R-independent `default_rng(seed + 1)`.
Consequences, verified against the loop structure:

- child-seed index = truth_index·R + r, so **truth 1's first min(R, R′) realizations DO
  prefix-match across different R**, while every later truth consumes master draws at
  R-dependent offsets and cannot match;
- an R=4-vs-R=120 byte comparison therefore predicts *agreement at truth 1, divergence at
  truths 2–3* — **exactly the "diverges at 2 of 3 truths" signature reported as trigger 2**.
  The clause's re-diagnosis (comparison-scale artifact, not environment) is mechanically
  corroborated by the code, and the earlier "environmental" reading is correctly withdrawn.

The registered rule (byte-identity defined only between equal-R runs at identical config/seed)
is the right conservative form. The stated *mechanism* ("realization streams do not
prefix-match across different n_realizations") is slightly overbroad → A-PF-6.

### (i) Contradictions with surviving registered text — NONE

- **AMENDMENT G1-2 (Part II):** its disambiguation procedure survives as a dormant escalation
  path and is now *strengthened*: with the pre-extension local referent proven bit-identical
  to the cluster cell, any full-R REP-OFF-D diff can no longer be environmental — it is an
  extension break, which is N-A's original registered meaning restored. No conflict; the
  clause's "superseded to that extent" applies to G1-2's environmental *framing*, not its
  procedure.
- **Part-IV item (iv):** unaffected in its binding parts. Part IV gave two grounds for
  keeping N-A on the original grid; the environment-control-grid ground weakens, but the
  value-nonidentity ground (z-windows derive from `h_grid.min()/max()`) stands alone and is
  sufficient — N-A must remain original-grid, as the clause keeps it.
- The trigger-2 paragraph's "environmental" sentence remains in-file above the clause; the
  clause explicitly supersedes it, which is acceptable in a pre-freeze DRAFT (append-only
  binds only after commit).
- All-local wide-grid twins standing "on the grid-change ground alone": verified sufficient —
  the twins must be same-grid as the science cells regardless of environment (Part IV (i)),
  so the design is unchanged by the re-diagnosis.

### (iii) Residual N-A gaps — two, both cheap

1. **Early-catch property lost.** §3b's probe-scale N-A ("so a broken stream is caught before
   the full replication cells run") is now void, and no surviving text orders the full-R N-A
   cells before the science cells — a stream break would surface only at interpretation time,
   after ~2 CPU-h of science cells. → A-PF-5.
2. **"May score against either" referent ambiguity.** A byte-identity read should pin ONE
   referent of record; "either" is factually safe today (verified identical) but registers an
   ambiguity. → A-PF-7.

### Part-V amendments (exact quotable text, all NON-BLOCKING)

**A-PF-5 [NON-BLOCKING — execution order, restores the early-catch property].** Append to the
comparison-scale clause:

> Execution order (registered): REP-OFF-D and REP-OFF-P run and are scored for N-A
> bit-exactness FIRST — before any wide-grid science cell launches (cost 416 + 305 s); an N-A
> diff therefore stops the campaign at ≤ 0.2 CPU-h spent, preserving the early-catch role the
> probe-scale comparison was registered to provide.

**A-PF-6 [NON-BLOCKING — mechanism precision].** Old text:

> realization streams do not prefix-match across different n_realizations, so probe-scale
> (R=4) N-A comparisons against R=120 cells are VOID

new text:

> per-realization child seeds are drawn from one master stream through the truths × R loop,
> so only the FIRST truth's first min(R, R′) realizations prefix-match across different
> n_realizations and every later truth diverges by construction (the observed
> 2-of-3-truths divergence is exactly this signature); probe-scale (R=4) N-A comparisons
> against R=120 cells are therefore VOID

**A-PF-7 [NON-BLOCKING — referent of record].** Old text:

> item 4's REP-OFF-D check may score against either (they are identical)

new text:

> item 4's REP-OFF-D check scores against the on-disk cluster cell
> `vdeep_250_production_off.json` as the referent of record, with
> `referent_preext_vdeep_250_production_off.json` retained as the pre-extension control
> (verified bit-identical to the referent, both channels, all truths)

*Part-V verifier of record: same session. GO — the clause may enter the freeze commit as
written; the three amendments above are recommended and none gates the freeze.*
