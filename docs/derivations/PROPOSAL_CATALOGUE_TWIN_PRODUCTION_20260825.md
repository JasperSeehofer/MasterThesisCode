# PHYSICS-CHANGE PROPOSAL — the catalogue-leg twin to production: per-candidate S̄_φ in the no-BH catalogue numerator

**Date:** 2026-08-25 · **Status:** PROPOSED — presented, then STOP (author-gated; row #187
item 2 grant to AUTHOR the package; adoption itself is a fresh [RULE]) · **Subject:**
`bayesian_inference/bayesian_statistics.py` (trigger file; nothing changes until ruled) ·
**Evidence chain:** rows #149–#187; `PREREGISTRATION_P3_TWIN_20260822.md` (+21 amendments);
`PROPOSAL_P3_MIXTURE_WEIGHT_20260822.md` Appendix B (ratified row #169);
`PREREGISTRATION_B0_IDENTITY_20260823.md` (UNDISCRIMINATING, ratified row #178);
`GATE_B_ADJUDICATION_FINITE_MOMENT_20260824.md`;
`PREREGISTRATION_CA_BOUNDED_IDENTITY_20260824.md` (**TWIN-CALIBRATED, banked row #186,
ratified row #187**).

## 1. Old formula (item 1)

No-BH catalogue numerator: `numerator_integrant_without_bh_mass` = GW-Gaussian ×
volume-deconv host-z kernel, **survival-free per candidate** (the Gray (2020) Eq. (A.10)/MFG
convention, codified in the code comment at the integrand); the class weight `β_G_φ = ∫
f̄·S̄_φ·w_pop dz` carries the survival globally. Combined as `β_G_φ·L_cat_no_bh/D̃_φ` with
`L_cat = A/Σ^φ` (the row-#178-adopted divisor).

## 2. New formula (item 2)

Each candidate's kernel carries `S̄_φ(z; h)` **inside the z-quadrature** — mechanically: the
ALREADY-IMPLEMENTED, tested, and campaign-exercised `catalogue_numerator_survival="phi"` cell
promoted from counterfactual to production default (an `"auto"` resolution mirroring the two
prior adoptions: `"auto"` → `"phi"` under `absolute_marginal`, `"off"` retained as the
explicit counterfactual). **β_G_φ, D̃_φ, and the Σ-chain are UNTOUCHED** (Appendix B as
ratified: no double count exists; the refuted R-rescale and D̃-completion readings are banked
as the measured costs of the wrong readings, rows #167–#168). The with-BH catalogue numerator
stays coded (its 2D S-fork is L6-DER3's registered follow-up, not this proposal).

## 3. Reference / derivation (item 3)

L6-DER3 §4 ("the catalogue leg is the same fork, per-galaxy"): the generator is
latent-thresholded (proven at the A20/O4 review; O6 MECHANISM-CONFIRMED end-to-end, row #158),
so the accepted per-event density carries S̄_φ per candidate; the coded arrangement replaces a
per-event, h-dependent weight by its ensemble mean. The mixture-consistency derivation is
`PROPOSAL_P3_MIXTURE_WEIGHT_20260822.md` §1 as amended by Appendix B (ratified). Departure
from Gray (2020) Eq. (A.10)/MFG (2019) is deliberate and evidenced: **the coded arrangement is
not the self-consistent scoring of the pipeline's own mixture** (C-A verdict, item 6 below).
`docs/LITERATURE_WARNINGS.md`: MFG-a verbatim check remains OPEN — a Stage-L obligation before
any PAPER-facing quotation of the convention departure (not blocking the code adoption ruling;
listed in §6).

## 4. Dimensional analysis (item 4)

`S̄_φ(z;h)` is dimensionless ∈ [0,1], multiplying a normalized kernel inside the z-integral —
the term's measure and units unchanged. Degree bookkeeping: with the Σ^φ divisor (row #178)
the leg is S̄-degree-matched to its weight chain (the ratified S̄→cS̄ homogeneity argument;
regression-tested).

## 5. Limiting cases (item 5)

- `S̄_φ ≡ 1` (no selection): byte-identical to coded — the change is inert exactly when the
  survival is trivial.
- `S̄_φ = const < 1`: cancels between numerator and the Σ^φ divisor — level-inert (the K-flat
  kill-test structure, measured).
- σ_z → 0, single candidate at truth: reduces to the selected-prior single-host form (A-FULL).

## 6. Validity conditions + measured evidence register (item 6)

- **Leverage (venue-conditional):** TWIN-FUSED-MATERIAL +0.029068 ± 0.005088 registered-grid /
  +0.063389 ± 0.008897 un-truncated (row #173, amendment-20 quotation rules binding);
  catalogued-host venue Δmean_h = +0.0566, 12/12 (row #177 secondary).
- **Calibration (the decisive item):** **TWIN-CALIBRATED** (row #186, ratified #187): the twin
  closes the C-A bounded identity (T_w = −0.0013 ± 0.0012, band 0.005); the coded arrangement
  is displaced −0.0202 ± 0.0012 (−17.4σ) from its own model value, landing at its
  twin-law-derived displacement (−0.86σ); control at predicted value; C-TCI profile in band;
  C-B corroborates (coded-null excluded ~9σ, REPORT-ONLY). **Scope (binding, travels with any
  quotation):** venue- and h-conditional; self-consistency of the pipeline's own mixture;
  blind to the S̄_φ-table common mode and the aligned-generator premise; NOT a real-universe
  correctness claim.
- **Verification plan before/at adoption:** (i) regression: `"off"` counterfactual
  byte-identity + the existing twin-cell test suite promoted to default-path tests (the
  row-#178 pattern); (ii) a production-run counterfactual read (the row #119 M-pattern) on the
  next full `--evaluate`; (iii) the with-BH leg bit-unchanged; (iv) MFG-a verbatim check
  (Stage-L) before paper use; (v) the [P3-HGRID] claim card is independent and stays open.
- **Cost:** the per-candidate `np.interp` over the existing S̄_φ table; measured overhead in
  the campaign fleets: negligible vs the kernel quadrature.

## 7. Decision table

| # | item | tag | recommendation |
|---|---|---|---|
| 1 | Promote `catalogue_numerator_survival` to `"auto"`→`"phi"` under `absolute_marginal` (production default; `"off"` = counterfactual) | [RULE] | adopt — the full ladder (derivation → mechanism → leverage → calibration) is banked |
| 2 | The §6 verification plan (i)–(iii) with the adoption commit | [DO] | approve |
| 3 | The 2D/with-BH S-fork (L6-DER3 analog) | [RULE] | defer — its own future package |
| 4 | MFG-a verbatim check before any paper-facing quotation | [DO] | schedule (Stage-L) |

**STOP.** Presented for the author's ruling.
