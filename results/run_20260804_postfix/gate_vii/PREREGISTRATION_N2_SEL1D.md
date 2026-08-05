# Pre-registration — N-2 selection-in-completion-numerator, `1d` cell, 41-h evaluate

Registered 2026-08-05, **BEFORE the run**. Adapted from the pre-registration skeleton of
`.planning/derivation-gfrac-20260805/N2_SELECTION_NUMERATOR_DERIVATION_20260805.md.DRAFT`
§6.3, with every band **re-centred on measurement M1** (§6.4) — the skeleton's bands were
written against a naive forecast that M1 has since superseded. Per `docs/RESEARCH_CYCLE.md`
stage 2 and the measurement-before-gate rule (research-cycle rule 6): this file authorises a
**measurement**, never a formula change.

## What is being tested

Under this project's **latent-thresholded** detection model — detection is deterministic in
the *true* parameters θ, not in the data (`simulation_detection_probability.py:175-179`) —
the MFG-2019 hierarchical numerator carries `P(det|θ)` *inside* the θ-integral, and it leaves
that integral only for coordinates the analysis **retains**. The 1D channel discards
`M_z^obs`, the coordinate whose `(1+z)` mass lift carries the factor's entire h-dependence.
The derivation's (T3′) is therefore

```
B_num^{1d}(h) = ∫ (1−f_k) · p_gw · dV_c/(1+z) · S̄_φ(z;h) dz
S̄_φ(z;h)     = ∫ φ(log₁₀M) S_4D( d_L(z;h), M(1+z) ) dlog₁₀M
```

`S̄_φ` is **not a new object**: `precompute_phi_marginal_survival`
(`bayesian_statistics.py:1782-1874`) already tabulates it every `absolute_marginal` run for
`β_G^φ`/`β_Ḡ^φ`/`Σ^φ`, and the toggle reads that same table through the same `np.interp`
accessor `precompute_global_catalog_selection` uses for `Σ^φ` (`:2313`).

**The 2D arm is not part of this run.** The draft proposed a `both` cell carrying `S_4D`
inside the 2D Gauss–Hermite mass loop ((T1′)). Measurement M2
(`.planning/derivation-gfrac-20260805/n2_m2_2d_inertness{.py,_results.json}`) evaluated it
offline on the production quadrature nodes across the full 41-h grid, for both `μ_cond`
projections: `max |Σ Δln| = 7.109` nats (proj-0) / `5.791` nats (proj-1) over the **whole**
grid, chord slopes `+0.0899` / `+0.0726` nats/h — **~200× under** the 20 nats/h tolerance
the draft pre-registered as P-4, verdict CONFIRMED (inert) in both venues, bit-identical
between venues. The `both` cell would measure a known zero, so it is **deleted**: the
implemented toggle offers `off` and `1d` only, and argparse rejects `both`.

## Denominator scope — deliberate, and NOT the formula question

Under the `1d` cell the `D̃^φ` denominator **remains the production object**. This run is a
pure **numerator** contrast, exactly as the frozen-g_frac counterfactual was run: one object
is perturbed, every other object in the ratio is held at its run-of-record value, so the
measured tilt is attributable. A "matching" denominator change is **not** invented here —
under MFG the numerator's `P(det|θ)` and the denominator's
`α(h) = ∫ P(det|θ) n(θ|h) dθ` are different contractions of the same indicator and neither
substitutes for the other (derivation §3.4). Whether the production default should carry
either or both is the **author's ruling**, to be made *after* this measurement, through
`/physics-change`.

## The run

Two CPU evaluate arrays (41 h-points, canonical grid) + combine — one per venue, everything
except the new flag identical to the post-fix runs of record and to the frozen-g twins:

| | value |
|---|---|
| RUN_DIR (idealized) | `$WS/run_20260805_n2sel1d_iiib/` |
| RUN_DIR (joint) | `$WS/run_20260805_n2sel1d_joint_r1/` |
| CRB input | the existing `prepared_cramer_rao_bounds.csv` **symlink target `run_20260729_seed61000/`** — the same file the post-fix and frozen-g runs consumed. **No re-simulation.** |
| Injection pool | the same `injections/` pool symlink (`injection_pool_mix200k_20260728`) |
| Catalogues | unchanged per venue: iiib = idealized (parent/exact-z); joint_r1 = realization r1 (delivered/observed). **No new realization is drawn.** |
| Estimator | `NORMALIZATION_MODE=absolute_marginal`, `HOST_Z_KERNEL=volume_deconv`, `HOST_MASS_KERNEL=auto` — the post-fix path-(A) pairing, unchanged |
| **New flag** | `EXTRA_EVAL_ARGS="--selection_in_completion_numerator 1d"` (and **only** that) |
| Code commit | `07904540` (`main`, "instrumentation: --selection_in_completion_numerator — N-2 counterfactual toggle (1d cell)") |
| h grid | the canonical 41-point grid: 0.01 steps on [0.60, 0.65] and [0.79, 0.86], 0.005 steps on [0.655, 0.79] |

`--selection_in_completion_numerator` is recorded automatically in `run_metadata.json` (the
whole argparse namespace is serialised) — **check it before reading any result.** The
estimator also logs a WARNING naming the run a counterfactual.

## M1 — what re-centred the bands

The draft's P-2 band (`+207 ± 40` nats/h) assumed the completion term is the **entire** 1D
mixture for every event (`share_i ≡ 1`). Measurement M1
(`.planning/derivation-gfrac-20260805/n2_m1_completion_share{.py,_results.json}`) measured
the real per-event 1D completion-vs-catalogue share from the diagnostics CSVs and applied the
**exact** (non-linearised) rescaling
`Δln_i = ln(1 − share_i·(1 − S̄_φ,i))`:

| quantity | iiib | joint_r1 |
|---|---|---|
| 1D completion share at h=0.73 — mean / median / q05 / min | 0.9180 / 0.999998 / 0.2164 / 3.3e-7 | 0.9189 / 0.999980 / 0.2274 / 1.2e-7 |
| events with share > 0.99 (of 1588) | 1214 | 1198 |
| **Σ Δln tilt, full-grid chord [0.60, 0.86]** | **+17.503 nats/h** | **+15.656 nats/h** |
| **Σ Δln tilt, central difference at h=0.73** | **+23.983 nats/h** | **+25.397 nats/h** |
| (naive full-share forecast, superseded) | +207.506 chord / +206.668 central | identical (venue-independent) |
| Σ Δln level at h=0.73 | −2365.10 nats | −2364.32 nats |

**The correction is ~10× smaller than the draft's headline.** For scale, the 1D ln-posterior
of record tilts **−247.4** (iiib) / **−184.0** (joint_r1) nats/h over 0.60→0.73 with the MAP
railed at 0.600 in both venues: `+20` nats/h removes **8%** (iiib) / **11%** (joint_r1) of
that down-tilt. This is stated here, before the run, so that a small measured effect is read
as the *predicted* outcome and not retro-fitted as a disappointment.

## Pre-registered bands and branches

**P-2 (primary quantitative read) — the 1D Σ-ln tilt.**
`Σ_i Δln p_i^{1D}` between the `1d` run and its `off` twin:

> **+20 ± 10 nats/h**, i.e. the band **[+10, +30] nats/h**, in **both** venues, for **both**
> statistics (full-grid chord over [0.60, 0.86] **and** central difference at h=0.73).

The band is centred between M1's chord (15.7 / 17.5) and central-difference (24.0 / 25.4)
values and is wide enough to absorb the two approximations M1 could not remove: the point-GW
peak evaluation of `S̄_φ` (the live run integrates it over the z-quadrature) and the
`D̃^φ`-held-fixed scope. **Sign is part of the prediction: positive.**

**P-3 — sign coherence.** Fraction of events with a positive Δ-slope: **≥ 0.90**
(the naive probe measured 0.947 with 94.7% of per-event `dln S̄_φ/dh > 0`).

**P-6 — the 1D MAP, pre-registered honestly given the rescaling.** The never-add-MAP-
displacements rule (`GATE_PACKAGE_FINAL.md:609`, `DERIVATION_C7_HOSTZ_KERNEL.md:545-547`)
forbids converting a ln-tilt into a MAP shift; the draft therefore refused to band P-6 at
all. That refusal is kept — **no MAP number is predicted** — but the *branches* are
registered in advance so the readout cannot choose its interpretation after the fact:

- **(a) N-2 REAL-BUT-SMALL** — the **1D MAP stays railed at 0.600** in both venues **and**
  P-2 lands in band. ⇒ The correction is real, of the derived sign, and **bounded**; the 1D
  rail **has another owner** (the host photo-z root cause of record,
  `[[h0-railing-rootcause-photoz]]`, ledger #36, remains the standing explanation). The
  thread reports a bounded correction, the `/physics-change` package for a default flip is
  drafted with this magnitude attached, and **no re-attribution of the rail is made.**
  *This is the expected branch given M1's 8–11% share of the down-tilt.*
- **(b) ESCALATE** — the **1D MAP moves off the 0.600 rail** in either venue. ⇒ N-2 is
  **larger than M1's exact-rescaling estimate** predicts, which means one of M1's inputs is
  wrong (most likely the point-GW peak approximation for `S̄_φ`, or the completion share
  under the live quadrature). Escalate: re-open M1 against the delivered `S̄_φ` profile over
  the quadrature nodes before any package is drafted. Do **not** treat an off-rail MAP as a
  success.
- **(c) MIXED / read the split** — **P-2 lands outside [+10, +30] on either side**, in either
  venue. Report the split; do **not** force a branch.
  - *Above +30*: the live z-integration of `S̄_φ` is materially different from the peak
    evaluation, or the completion share is larger under the live mixture than the CSV read.
  - *Below +10 (including sign reversal)*: the completion leg is a smaller part of the live
    1D mixture than M1 measured, or a downstream object partially absorbs the change —
    identify it from the diagnostics deltas before interpreting.
  - A **venue split** (one venue in band, one out) is informative on its own: the completion
    machinery was measured venue-independent (`ḡ(h)` bit-identical across venues, gate (vii)),
    so a split must come from the catalogue leg's share, not from the correction itself.

## Expected NULLs (P-9's programmatic form) — a move in any of these VOIDS the run

1. **The 2D posterior must be BIT-IDENTICAL** to the `off` twin: `combined_2d.json`, and the
   `B_num_wbh` / `combined_with_bh` / `L_cat_with_bh` diagnostics columns, byte-for-byte.
   `B_num_wbh` is its own quadrature over the unmultiplied base integrand and the 2D mixture
   never reads `B_num`. This is also the live confirmation of M2.
2. **Both catalogue legs bit-identical**: `L_cat_no_bh`, `L_cat_with_bh`.
3. **Selection-side objects bit-identical**: `w_G`, `w_tilde_G`, `alpha_G_phi`, `r_Malm`,
   `D_tilde_phi` (and `Σ^φ`, `Σ^4D`, `β^φ` in the logs). The toggle must not leak into the
   normalisation.
4. **`run_metadata.json` carries `selection_in_completion_numerator: "1d"`** in every task of
   both arrays, and `freeze_g_frac_ref_h: null`.

**Expected NON-nulls, by construction** (so they are not misread as leaks): `B_num` (the
perturbed object), `L_comp = B_num/β_Ḡ`, `g_frac = B_num_wbh/B_num` (a *reported ratio*, not
an input to any assembly — `B_num_wbh` itself is unchanged), and `combined_no_bh`.

## Scope guard

- **No re-simulation.** CRB set and injection pool consumed through the existing seed61000
  symlinks; no waveform, no Fisher matrix, no injection regenerated.
- **No production posterior.** Both runs are counterfactuals by construction. They must never
  be quoted as a result, only as a diagnostic contrast against their `off` twins.
- **The D̃^φ denominator question is NOT settled by this run** and must not be presented as
  such. See "Denominator scope" above.
- **The D1 constraint of record is untouched.** Nothing here may be read as evidence for or
  against D1's remedy.
- **Any actual fix routes through `/physics-change`** — derivation, dimensional analysis,
  limiting case, literature reference, regression test, ledger row.
- **Deferred, not tested here:** P-10 (the #66/#67 pairing question — whether the corrected 1D
  channel needs the σ(d_L^obs)-vs-σ(d_L^true) noise-model companion) requires a `pp_coverage`
  harness with a mass channel, which is TO-BUILD. Honest gap §8.5 of the derivation stands:
  it is the single most likely way this correction disappoints.

**Append-only.** Verdict to be appended below by the session that reads out the run — after
this file is committed, **no edits above this line.**

---

## VERDICT — to be appended by the readout session

**(c) MIXED, both venues — the bounded direction.** Jobs 6152554 (iiib) / 6152556
(joint_r1), 246/246 tasks COMPLETED each, code commit `0167df53`.

| read | pre-registered | measured (iiib / joint) | status |
|---|---|---|---|
| P-2 full-grid chord, [0.60, 0.86] | band [+10, +30] nats/h | **+24.588 / +22.736** nats/h | IN band |
| P-2 central diff @ h=0.73 | band [+10, +30] nats/h | **+30.901 / +32.315** nats/h | OUT (above ceiling by 0.9 / 2.3) |
| P-3 sign coherence | ≥ 0.90 | **0.7286 / 0.7128** | FAIL |
| P-6 1D MAP | branches (a)/(b)/(c), no number pre-registered | **railed 0.600, both venues** (hard rail: Δln to the next grid point −1.61 / −1.11 nats) | branch (a)-consistent on P-6 alone, but see below |
| Expected NULLs | must hold, all venues | **held bit-exactly**: 2D posterior, catalogue legs, `w_G`/`w_tilde_G`/`alpha_G_phi`/`r_Malm`/`D_tilde_phi` — 0 differing cells of 65108 | PASS |
| Expected non-NULLs | must change | confirmed changed (`B_num`, `L_comp`, `g_frac`, `combined_no_bh`) | PASS |

**Reading, per the prereg's own §"(c) MIXED / read the split" sub-branch.** The chord
statistic lands in band in both venues (the primary, less approximation-sensitive
read), but the central-difference statistic overshoots the ceiling in both venues,
and sign coherence (0.71–0.73) falls well short of the 0.90 bar. Per the pre-registered
"*Above +30*" reading: **the live z-quadrature of S̄_φ differs materially from M1's
point-GW-peak evaluation, and/or the live completion share exceeds the CSV read** —
this is **flagged for follow-up, NOT resolved** by this run. Both venues move together
(no venue split), consistent with the completion machinery's known venue-independence
(gate (vii)).

**What this run does and does not establish.** N-2 is a **real, positive, bounded
correction** — order +23 to +25 nats/h at the chord statistic — that is the derived
sign and order of magnitude M1 predicted (band centred between M1's chord 15.7/17.5
and central-diff 24.0/25.4). It is **not** a phantom or a sign error. But it
**does not un-rail the production 1D channel**: the 1D MAP stays pinned at 0.600 in
both venues, with a hard rail (the next grid point is disfavoured by −1.1 to −1.6
nats). Per P-6's pre-registered discipline (never convert a ln-tilt into a MAP shift,
and the rail-stays-put branch (a) reads as "bounded, another owner"), the standing
explanation for the production 1D rail — the host photo-z root cause of record
(`[[h0-railing-rootcause-photoz]]`, ledger #36) — is **not re-attributed** by this
result. N-2 is real but bounded; it does not carry enough weight, even summed with a
hard rail of only ~1.1–1.6 nats, to move the MAP off 0.600 given the down-tilt of
record (−247.4/−184.0 nats/h) that produced that rail in the first place.

**Formula adoption remains open**, gated by the P-2/P-3 mixed read above: whether the
`1d` numerator selection factor becomes a `/physics-change` default depends on
resolving why the live quadrature diverges from M1's point evaluation (candidates:
the z-integration of `S̄_φ` versus the GW-peak point estimate; the completion share
under the live mixture versus the CSV-measured share) — author ruling R-0..R-5,
per the "Denominator scope" section above, not decided here.

Evidence: `results/run_20260805_n2sel1d/readout.{py,json}`.
