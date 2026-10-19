# [IMP] CLAIM INTAKE — the photo-z impostor-drag REMAINDER: owned by an identifiable covariate/mechanism, and the owner of the 1D rail?

*launched under rows #222/#223 — charter node B4.1 [IMP] part 2*

**Opened:** 2026-08-29, charter branch B4 depth 1 (row #221 item 5: "[DO] register a direct attack on
the impostor-drag remainder (~81 % of the largest contribution unattributed; stage-0 intake first)").
**Stage:** research-cycle stage 0 (intake) + stage 1 (information forecast), per `docs/RESEARCH_CYCLE.md:27-107`.
**Template:** `CLAIM_COMPLETION_MEMBERSHIP_20260828.md` (its §0.2 table is the model).
**Status: CLAIM, NOT ESTABLISHED. Written to be attacked.** Append-only below the committed line.
**Scope:** 1D (no-BH) channel, ABSOLUTE bias, the dark (out-of-catalogue) event class. The 2D
catalogue-leg twin is B7's object; the mass window is B5's; the sky cone is B2's.

Tags per stage 0 rule 3: `[LOCAL]` re-measured here at zero `evaluate()` (scripts in this directory,
reproducible offline) · `[DOC]` read from a committed artifact · `[AGENT]` measured by a subagent (part 1,
`B4_1_IMP_DECOMPOSITION.md`), not re-measured here · `[INFER]` inference with no new measurement.
Every `[LOCAL]` number below is a stage-1 forecast input: **no band is attached, none is a verdict, and
a registered form of any of them must be run by a different agent (standing rule 2).**

---

## 0.1 The claim (intake form)

**Claim as charged (row #221 item 5, orchestrator wording):** *the photo-z impostor drag remainder
(~81 % of the B-SEL −0.1083 after the 1D twin) is owned by an identifiable covariate/mechanism and is
the owner of the 1D rail.*

**Object.** For an event whose true host is NOT in the catalogue (dark class; 95.2 % of production,
`in_catalog` 76/1588 = 0.0479 `[LOCAL]`), the catalogue leg `L_cat_no_bh` is a sum over the real
GLADE+ galaxies inside the sky-cone/z-window ball — all impostors by construction. The mirror assembly
(`bayesian_statistics.py` p_Di, re-derived in `decompose_impostor_leg.py:1-20`, gated here to
≤ 5.5e-7 relative on every arm, GATE I) is

```
p_i(h) = ( beta(h) · L_cat,i(h) + B_num,i(h) ) / D̃_φ(h),   beta = alpha_G_phi / r_Malm .
```

"Impostor drag" = the displacement of the fleet posterior mean caused by that leg, measured by the exact
subtraction `L_cat,i ≡ 0` (the O2 construction, row #149). "Remainder" = what is left after the
catalogue-leg twin (`catalogue_numerator_survival="phi"`, production physics since row #197).

**Provenance chain of the charged numbers `[DOC]`:**
- B-SEL fleet headline **−0.1083** (12 seeds, off cell; row #146/#149; `decompose_impostor_leg_output.json`
  `bias_full = −0.10830226698447765`).
- O2 impostor removal **Δ = +0.0791883246** (12/12 positive, per-seed +0.030…+0.164, seed SD 0.0414, SEM
  0.0119; row #149 item 1, ledger `:1897-1912`; row #154 item 2b for the SEM) ⇒ 73.1 % of the headline.
- Three-contribution split on the off cell: matched −0.0846 ⊕ tilt +0.055 ⊕ impostor −0.079
  (row #150 item 3, ledger `:1946-1947`).
- Twin, off basis: **+0.015524 ± 0.003657**; *"pure−full = +0.0637 ± 0.0090 (80.6 % of the coded −0.079)"*
  (row #162 items 1–2, ledger `:2309-2322`) — this is the "~81 %".
- Twin, fused (production) basis: **+0.029068 ± 0.005088**, FC fleet bias −0.113508, FT −0.084440
  (row #173, ledger `:2509`; `PREREGISTRATION_P3_TWIN_20260822.md:564-567`).
- Current production posteriors (HEAD `d04d9dc9`, fused/phi, row #213): *"2D peaks at 0.665 (σ ≈ 0.017 …);
  1D rails onto the 0.60 grid boundary"* (`RUNBOOK_NEXT_SESSION_37.md:27-30`).

**What "owner of the 1D rail" is taken to mean (fixed here so the claim is falsifiable):** the impostor
leg is a NECESSARY cause of the low rail (removing it from the dark class un-rails the 1D posterior) AND
its contribution is a normalisation violation (E[∂_h ln p] ≠ 0 at truth under the estimator's own model),
i.e. a defect, not a venue-composition artefact.

## 0.2 Exoneration check (both layers, MECHANISM-grepped 2026-08-29)

Layer 1 = `EXONERATION_REGISTER_20260827.md` (930 lines, §1–§9 + adversarial check, read in full);
layer 2 = `gate_b_20260730/BIAS_HISTORY_LEDGER.md` §2 "DO NOT RE-TRY" (`:127-170`), §3 (`:172-207`),
§4 (`:210-243`). Greps run on BOTH files for the mechanism vocabulary, not the tag: `kernel width`,
`sigma_z / σ_z / photo-z`, `impostor`, `candidate / in-ball / in ball`, `cone`, `z-prior / w_pop`,
`Malmquist / magnitude-limit / depth`, `starv`. Every hit read in full including its "WHAT IT DOES NOT
COVER" field. **Verdict: NOT exonerated — with one entry that names the mechanism by name and must be
disposed explicitly (row 1), and four entries that BIND on remedies (rows 5–8).**

| # | hit (verbatim, file:line) | mechanism covered | binds this claim? |
|---|---|---|---|
| 1 | ledger §3 `:195`: *"The impostor channel was measured and EXONERATED as a bias carrier at full power (bias −0.0006 at 93% impostors, +0.0024 at 97.6%); the residual is entirely B_num (#87)"*; register [LCOMP-BNUM-DEFECT] `:446-462`: *"Impostor channel and normalization channel both EXONERATED as residual carriers; the residual is entirely B_num"* (#87, 2026-07-27, `pp_fullpower_20260727/FULLPOWER_READOUT.md:43-58`) | the impostor ball as a bias carrier — **the same mechanism** | **NO — venue-scoped and overtaken.** (i) Standing scoping rule, ledger `:157-159`: *"negative conclusions are venue-scoped"*; #87 is the synthetic pp_fullpower harness (n = 2000, self-consistent generator, one true candidate + impostor ball), not the realistic-catalogue mirror at 4.79 % completeness. (ii) The ledger itself later BANKED the direct measurement that contradicts it on the production-like venue: O2 +0.0792, 12/12 (row #149), ratified in row #153 item 3 (*"the arm-level −0.1083 decomposes as impostor −0.079 ⊕ tilt +0.055 ⊕ matched −0.085"*). A later, direct, banked measurement on the relevant venue supersedes an earlier harness null. The register's own delimitation (`:459-462`) is respected: this claim says the leg CARRIES a violation; it does not say `B_num`'s integral is wrong. |
| 2 | register [VOLUME-DECONV-H-DEP] `:466-473`: *"exactly h-invariant to 1e-15 (Z_g ∝ h⁻³ factors out cleanly)"* (#75) | h-dependence of the `volume_deconv` kernel | NO for the width question — it exonerates a hidden h-dependence, not the kernel's WIDTH scale; the [HIER] prereg delimits the same way (`PREREGISTRATION_HIER_HTHETA_20260826.md:91-120`). Binds only against re-opening "the kernel carries h-dependence". |
| 3 | register [WPOP-TUNING] `:382-388`: *"NEGLIGIBLE — ≤ +0.0004 at a 10% deliberate misspecification. Escape hatch closed."* (#64); ledger §3 C7 `:197`: G2b *"CONFIRMED w_pop = (dV_c/dz)/(1+z) as 'the unique weight consistent with the project's own rate model and with every selection integral', exactly h-independent"* | the host-z population prior (z-prior) as tunable/wrong | **BINDS on remedies:** no B4 arm may tune or replace `w_pop`. Does NOT cover the measurement-error kernel convolved against `w_pop` (HIER §1.6 delimitation) nor the catalogue-leg mixture weight `beta`. |
| 4 | register [DEPTH-TRUNC] `:407-414`: *"empirically dead — rails at every depth tested (0.2 / 0.3 / 0.5)"* (#56); its own field: *"not a mechanism-level exoneration of 'depth' as a concept"* | a `z_max` cut as the rail fix | NO as a mechanism; BINDS against a depth-cut remedy. The low-z localisation found below (§C2) is the opposite end of the z-range anyway. |
| 5 | register [HARD-CLAMP-OBSERVED-Z] `:370-380`: *"REFUTED for production — sign-flipping bias −0.021…+0.015, coverage 0.18–0.46; needs a soft, photo-z-marginalized membership instead"* (#63) | a hard z-membership cut on observed z as the fix | **BINDS on remedies:** any B4.3 proposal touching ball membership must be soft/photo-z-marginalised. Not this claim's mechanism. |
| 6 | register [VOLUME-TRUNC] `:142-154`: *"FALSIFIED as a fix — moved the bias the WRONG way by ~4×"* (#70; seed600 494-event venue) | unifying the numerator z-window | NO (venue-scoped to seed600; a remedy, not this mechanism). |
| 7 | register [NUMERATOR-ONLY-CLEAN] `:541-556`: *"both numerator-only variants rail the estimator UP to 0.870 … DISQUALIFIED"* (#37/#38), and [PDET-NUM-ALONE] `:390-404`: *"Do not cargo-cult it."* (#66) | numerator-only re-weighting; unpaired p_det insertion | **BINDS on remedies:** every catalogue-leg re-weighting must be PAIRED with its normaliser (the twin was; `CLAIM_P3_IMPOSTOR_CONVENTION_20260822.md:44-49` applied this correctly). The §1.3 site-2.2-only diagnostic below is therefore labelled diagnostic-not-fix. |
| 8 | register [WINDOW-MEMBERSHIP] `:186-199`: *"exact removal of realization-added 2D candidates moves MAP 0.81→0.82 — wrong sign"*; *"a coarse binary lever (window OFF vs ON)"* | 2D candidate-window membership | NO — the with-BH ball on the 2D tilt; this claim is the 1D dark-class ball in the absolute scope. |
| 9 | register [Z-LEG] `:156-166`: *"by construction the z leg is bit-identical between channels … cannot produce a 2D-specific residual"* | the z leg as the 2D-ONLY excess source | NO — channel-common is exactly what an absolute 1D claim needs; not an exoneration of the leg in the absolute scope. |
| 10 | register [INFO-STARVATION] `:485-499`: *"OVERTURNED … 'a property of prior-INCONSISTENT estimators, not of the data' … Do NOT resurrect it"* | starvation as the rail explanation | NO — this claim asserts the opposite (a normalisation violation, E[score] ≠ 0); the F5 forecast in §1.0 is quoted only as achievable precision, per the HIER §1.6 vocabulary rule. |
| 11 | register [LCOMP-BNUM-DEFECT] item (i) `:449-452`: E1 *"subset-conditioning artifact — B/β_Ḡ = 0.7366 ± 0.0155 closes at truth on a membership-clean subset"* (#80) | reading a fallback-only subset with the wrong conditional | BINDS on interpretation: class-resolved reads below keep the FULL mixture per event and never re-condition a subset (the E1 trap). |
| 12 | register [CODE-HYGIENE-BUNDLE] `:523-538` (`galaxy.py` (1+z)³ σ_z scaling — file deleted), [SPECZ-RESCUE] `:501-508`, [PV-FRAME] `:512-520` | σ_z scaling in dead code; spec-z subsets; frames | NO. |
| 13 | register HB `:280-330`, [MASS-KERNEL-FAMILY] `:201-225`, §4 [WBHZERO-ASYMMETRY] `:585-620` | mass window / mass kernel (2D) | NO — the 1D leg never sees the mass window (`HANDOFF_20260730.md:88-90` via register §7 item 2 `:703-706`). |
| 14 | ledger §2 item 8 `:146` [ZERO-HOST-FALLBACK]: *"fallback events are h-inert"* (#55, strengthened row #147) | events with no ball | Consistent, not binding: `s_imp,i ≡ 0` for `L_cat = 0` events by construction here (26–39 % of B-SEL rows). |
| 15 | ledger §3 Gate C item 1 `:203`: *"ALREADY MEASURED, and it FAILED … a −17.2% end-to-end residual … §3.21 independently measured that identity violated by 33% in value and 0.39/h in log-slope … the best-evidenced open channel"*; `CLAIM_2D_BIAS_20260730.md:741`: *"w_G is deliberately NOT added to this list: C9 is live"*; [OPTIONA-DRIFT] `:227-240`: *"the surviving +0.017 in h, 1D-only residual remains genuinely open"* | the catalogue-leg normalisation / mixture-weight calibration and its h-slope | **LIVE, not exonerated** — and it is candidate mechanism (b) of §C6. Rule respected: this card does not re-run the refuted `w_G` bookkeeping FIX (`:355-368`, #61); it targets the calibration QUESTION the register keeps open. |
| 16 | `PREREGISTRATION_CMEM_READS_20260828.md:40-63` + RESULT RECORD (rows #219/#220): R2a DISPLACED, R2b 54×, R2c p = 0.0152 ⇒ **C-STRUCTURAL-ONLY**; A1 upgrade granted (row #221 item 3) | the in-catalogue-but-outside-the-cone class (in-ball flag, cone) | Not an exoneration — a LIVE sibling branch (B2). **B4 does not run an in-ball split**; see §1.2 read (iv). |
| 17 | `Malmquist` / `magnitude-limit`: **0 hits in either layer** as a mechanism. `r_Malm` is the adopted Σ^φ divisor slot ([P3-RPHI], rows #171–#179, `e35ea018`) — an adopted object, not an exoneration. | catalogue-depth skew of impostors inside the ball | NOT exonerated; candidate (c) of §C6. |

Venue-scoping rule honoured: no entry above was measured on the dark-class impostor ball of the
realistic-catalogue mirror; the one that names the mechanism (row 1) is the harness null that the
ledger's own later, direct measurement overtook. Register §6/§C near-miss lesson applied: the sweep
read every entry, not only the matching ones.

## 0.3 Stage-L R0 sweep (lightweight, satisfied by reference; no new search this session)

- **G20-d** (`docs/LITERATURE_WARNINGS.md:96`): Gray et al. 2020's MDC validates 25–75 % completeness;
  production is 4.79 % — every result here is venue-scoped by this. `[DOC]`
- **MFG-a** (`:229`): the MFG consistency principle is a paraphrase, UNCHECKED verbatim; a stage-2 prereg
  may cite it only as a supported claim. `[DOC]`
- **B25-a** (`:189`): a Gray-style mixture reported unbiased across completeness configurations
  (UNCHECKED counter-precedent) — a stage-5 input, not a design input. `[DOC]`
- Gray–Messenger–Veitch 2022 Eq. 5 is the coded `B_num` population factor (`bayesian_statistics.py`
  `:5350-5364` region per CMEM Read 1). None of the three sources treats a magnitude-limited impostor
  ball around an out-of-catalogue host explicitly ⇒ no known-failure-mode row; full Stage L only if this
  thread goes MIXED twice. `[DOC]`

## 0.4 Numbered claims (each with `Refute by:`)

### C1 — On the production (fused + twin) basis the remainder is +0.1227 ± 0.0077, i.e. the whole venue headline and more `[LOCAL]`

Method: `b4_imp_stage1_forecast.py` — the O2 construction (exact subtraction, GATE I ≤ 5.5e-7 on all
36 arms, corrected combine `physics_floor` + trapezoid, H_GRID_41) applied to the banked 12-seed
B-SEL realisations on three bases. The O2 number of record is reproduced to 4e-17 (`o2_reproduction`
in `b4_imp_stage1_forecast.json`); FC/FT full-arm biases reproduce the banked −0.113508 / −0.084440
(row #173) to < 1e-6.

| basis (12 seeds) | full mean_h − 0.73 | pure (`L_cat ≡ 0`) − 0.73 | Δ = drag (registered grid) | Δ un-truncated (H_GRID_FULL) | rail full → pure |
|---|---|---|---|---|---|
| off + coded (`bsel`, = O2) | −0.10830 | −0.02911 | **+0.07919 ± 0.01195** (SD 0.0414, 12/12) | +0.14981 ± 0.01437 | 12/12 → 2/12 |
| fused + coded (`fc`) | −0.11351 | **+0.03830** | **+0.15181 ± 0.01071** (SD 0.0371, 12/12) | +0.24989 ± 0.01125 | 12/12 → 0/12 |
| fused + twin (`ft`, = HEAD leg) | −0.08444 | +0.03830 | **+0.12274 ± 0.00774** (SD 0.0268, 12/12) | +0.18650 ± 0.00735 | 12/12 → 0/12 |

Readings: (a) the twin recovers 0.15181 − 0.12274 = +0.02907 (= row #173's +0.029068); the remainder is
**80.8 %** of the fused coded-leg drag — the "~81 %" transfers to the production basis. (b) On the
fused basis the pure-completion arm is biased **HIGH** (+0.038; per-seed 0.707–0.827; c68 5/12; the
+0.032 above O8's matched +0.006 is the all-dark composition tilt of row #150 item 2), so the impostor
leg is not "81 % of −0.108" but the entire negative displacement: it takes the fleet from +0.038 to
−0.084. (c) The drag exceeds its off-basis value because the fused completion numerator no longer
co-rails with it (row #165's +0.0724 correction).
Caveats: FC/FT were run 2026-08-23 at `53b7831e`, BEFORE the Σ^φ divisor adoption (`e35ea018`,
row #179) — the HEAD leg differs by the global factor 1/r_φ(h) (1.1287 at h = 0.73, row #177 item 1);
row #171 measured that slot's venue effect at −0.0043. Grid-censored per amendment 20 (map_h at the 0.60
floor 12/12 for every full arm).
**Refute by:** re-run the exact subtraction on a HEAD-basis (Σ^φ-inclusive) FT fleet; the claim
fails if Δ < 0.10 or if the pure arm ceases to un-rail (≥ 3/12 r_low).

### C2 — The remainder is NOT diffuse: the lowest-z quartile of events carries 87–92 % of the impostor-leg score at truth; SNR carries nothing `[LOCAL]`

Statistic (A12, the standing first diagnostic): per-event secant score at truth over h = 0.725/0.735,
`s_imp,i = ∂_h ln p_i − ∂_h ln pure_i` (identically 0 for the 33.7 % of events with an empty ball).
Fleet means (per-event mean, seed SEM): `fc` −0.3282 ± 0.0218; `ft` −0.2178 ± 0.0158; `bsel`
−0.2167 ± 0.0158. Truth-side covariates joined from `prepared_cramer_rao_bounds.csv` (d_L, SNR;
z_true = `dist_to_redshift(d_L, 0.73)`; the mirror observation is noiseless so d_L is the injected
value — `bsel` shares FT's realisations: `L_cat_no_bh` bit-identical to 1.6e-13 across the two).
Candidate-ball size from the "possible hosts found" log lines (part-1 method; global consistency check
passed 24/24 arms). Pooled 2152 events, 12 seeds:

| covariate (fleet quartiles) | `ft`: q1 / q2 / q3 / q4 mean s_imp (± SEM) | share of Σ s_imp (q1…q4) | η² (`ft`) | η² (`fc`) |
|---|---|---|---|---|
| **z_true** (edges 0.358 / 0.459 / 0.584) | **−0.798 ± 0.042** / −0.067 ± 0.004 / −0.006 ± 0.001 / −0.00003 | **91.7 % / 7.6 % / 0.7 % / 0.0 %** | 0.326 | 0.405 (q1 share 86.2 %) |
| z_true, active events only (66.3 %; edges 0.338 / 0.431 / 0.534) | −1.129 ± 0.054 / −0.169 ± 0.009 / −0.016 ± 0.002 / −0.0006 | — | 0.448 | 0.539 |
| catalogue share c_i(0.73) (top quartile median 0.068) | ≈0 / ≈0 / −0.022 ± 0.001 / **−0.849 ± 0.040** | 0 / 0 / 2.5 % / **97.5 %** | 0.384 (r = −0.77) | 0.525 (r = −0.78) |
| log10 n_cand (no-BH ball; top-quartile median ≈ 1200) | (empty) / −0.043 / −0.294 / −0.491 | — / 9.8 % / 33.8 % / 56.4 % | 0.102 (r = −0.34) | 0.144 (r = −0.40) |
| SNR (`fc`; edges 27.9 / 41.1 / 87.5) | −0.215 / −0.340 / −0.394 / −0.364 | 16 / 26 / 30 / 28 % | — | **0.009** (r = −0.005) |

Reading: the violation lives in nearby events (median z 0.29 in q1) whose ball is dense and whose
catalogue mixture share is high; events beyond z ≈ 0.46 contribute < 2 %. This is the OPPOSITE end of the
z-range from the off-cell completion-leg violation (row #137 item 3: *"score ≈ 0 below z ≈ 0.4, monotone
to −1.08 at z ≈ 0.9"*), which the fused fix removed (the pure score is now +0.07). Same localisation on
production HEAD (§C5): the nearest d_L quartile (median 1.29 Gpc) carries 93.8 % (`iiib`) / 96.7 %
(`off_iiib`) of the impostor-leg score.
**Refute by:** the registered per-event re-run with the part-1 §7 per-candidate hook (≈ 3.4 CPU-h)
finds the q1 impostor-leg score spread across z (q1 share < 50 %), or the s-switch of §1.3 and the
per-candidate z-offsets both come back inert (then the localisation is real but its mechanism is
neither of §C6 (a)/(c)).

### C3 — First-order split: ~63 % of the impostor-leg score rides on the GLOBAL mixture-weight h-slope, ~37 % on the per-event catalogue-vs-completion slope `[LOCAL]`

From the assembly identity, `s_imp,i = c_i · (s_β + s_L,i − s_B,i)` to first order (`b4_imp_stage1_split.py`;
first-order sum −0.21776 vs exact −0.21778). Event-independent (spread 0.0 across events):
`s_β = ∂_h ln beta = −3.2891 /h`, `s_D = ∂_h ln D̃_φ = −1.2373 /h`. On active events: mean c = 0.0627 ±
0.0031 (`ft`), mean `s_L = ∂_h ln L_cat = −27.08 ± 1.06 /h`, mean `s_B = −0.928 ± 0.032 /h`.
Split of the −0.2178 (`ft`): global term `c·s_β` = **−0.1366 ± 0.0068 (62.7 %)**, per-event term
`c·(s_L − s_B)` = **−0.0812 ± 0.0112 (37.3 %)**; `fc`: −0.179 (54.5 %) / −0.149 (45.5 %).
Reading `[INFER]`: the per-event catalogue leg of a dark event falls steeply with h at truth (a factor
e^{−0.27} per 0.01 in h), i.e. the impostors sit BELOW the truth-h-implied redshift; and the catalogue
mixture weight itself falls at −3.29/h. Whether either slope is what the model should produce is a
derivation question (§C6 (b)), not a run.
**Refute by:** a derivation of the model-consistent `∂_h ln beta` for this venue that reproduces
−3.29/h within 10 % removes candidate (b); otherwise it quantifies it.

### C4 — The violation survives at the model's OWN class composition: it is a defect, not an artefact of the all-dark venue `[LOCAL, cross-fleet composite]`

Alternative that had to be closed first: in a correct mixture E_full[score] = w̃_G·E_cat + (1 − w̃_G)·E_dark = 0,
so an all-dark fleet could show a negative impostor-leg score that the catalogue class compensates.
Banked catalogue-class venue b0i (`p3_b0_work/bt_*`, phi, fused, Σ^φ slot on; 12 seeds, 1377 events):
per-event FULL-mixture score at truth **−0.1238 ± 0.0527** (seed SEM; `bc` −0.4128 ± 0.0525); its
completion leg scores +2.292 ± 0.018 and its catalogue leg (true host included) −2.415 ± 0.062.
Model class weight at truth `w̃_G(0.73) = 0.0620` (event-independent, identical in both venues).
Composite: 0.062·(−0.124) + 0.938·(−0.147) = **−0.146** (twin basis; coded basis −0.268) against a
combined seed SEM ≈ 0.016 ⇒ the full mixture's expected score at truth is NOT zero at the model's own
composition; the catalogue class does not compensate (it is itself slightly negative).
Caveats: two separate fleets (different realisations, both model-matched to their own class); b0i
carries the Σ^φ slot, FT does not (a global factor on the catalogue leg, §C1 caveat); `w̃_G` is the
model's global class prior, not a per-event responsibility. None of these is large enough to move
−0.146 to 0 (it would need the catalogue class at ≈ +2.2 per event).
**Refute by:** a registered model-matched MIXED fleet (hosts drawn in-catalogue with probability
`β_G_φ/(β_G_φ+β_Ḡ_φ)` at h_true, else dark) whose full-mixture score at truth is |Z| ≤ 3 from zero
while its dark-class impostor-leg score stays ≈ −0.2 ⇒ composition artefact, C4 dead.

### C5 — Production HEAD: the dark-class impostor leg is NECESSARY for the 1D rail; removing it alone puts 1D at 0.713 ± 0.028 (covers truth) `[LOCAL; ASSUMPTION-JOIN — secondary until validated]`

`b4_imp_stage1_production_o2.py` on `headreadout_20260827/{iiib,joint_r1,off_iiib}/event_likelihoods.csv`
(row #213, `d04d9dc9`, fused/phi/Σ^φ; 1588 events × H_GRID_41; GATE I ≤ 5.5e-7):

| venue | full (mean / MAP / floor-node mass) | pure, ALL `L_cat ≡ 0` (mean / MAP) | pure, DARK events only (mean / MAP / σ / c68) |
|---|---|---|---|
| `iiib` HEAD | 0.6077 / **0.60** / 0.446 | 0.8396 / **0.86** (upper edge) | **0.7134 / 0.70 / 0.0277 / TRUE** |
| `joint_r1` HEAD | 0.6143 / 0.60 / 0.221 | 0.8459 / 0.86 | not computable (no matching CRB on disk: `seed62000` has 1545 rows < 1588) |
| `off_iiib` (off cell) | 0.6032 / 0.60 / 0.718 | 0.6891 / 0.67 | 0.6321 / 0.63 / 0.0160 / False |

Per-event impostor-leg score at truth (`iiib`): −0.265 ± 0.051 (pooled); dark events −0.193, in-catalogue
events −1.707 (their leg contains the true host and is designed to cancel a +2.3 completion score, cf. C4);
dark class carries 69 % of Σ s_imp; d_L-quartile localisation as in C2.
Reading: on production the 1D posterior is a balance between a completion leg that pulls HIGH (pure arm
at the 0.86 edge) and a dark-class impostor leg that pulls LOW past the 0.60 floor; the dark-class leg is a
necessary cause of the rail in the sense fixed in §0.1. "Owner" in the sufficient sense is NOT
established: the pure completion arm is itself +0.11 high on production (the M1-vs-comoving population
question of B3, row #138, and the residual composition tilt live there).
Caveats: the dark/in-catalogue split assumes `event_idx` == row order of
`seed61000/prepared_cramer_rao_bounds.csv` (the joined `in_catalog` fraction reproduces the known
76/1588 = 0.0479 exactly, which is consistent with but does not prove the row alignment); no band; the
pure arm's 0.86 is an upper-edge censoring.
**Refute by:** the registered form (validated join through the harness's own event index, zero
compute) puts the dark-only pure arm at map_h ≤ 0.605 (still railed) or its mean outside [0.66, 0.78].

### C6 — Mechanism: three candidates, UNDETERMINED between them `[INFER]`

(a) **Photo-z kernel width** (the [HIER] `s`-axis, `PREREGISTRATION_HIER_HTHETA_20260826.md:40-62`):
the low-z quartile is where σ_z/z is largest and where ledger #68/#62 found the deconvolution
*"over-corrects at σ_z/z ~ O(1)"*; a width-scale misstatement would tilt every impostor's kernel.
(b) **The catalogue-leg mixture weight's h-slope** (`s_β = −3.29/h`, 63 % of the score at first order):
the C9 / Gate-C-item-1 live object (*"0.39/h in log-slope"*), i.e. `beta = alpha_G_phi/r_Malm` vs the
discrete ball's own content — a normalisation-derivation question.
(c) **Catalogue-depth skew of impostors inside the ball** (`s_L = −27/h` on active events): the
magnitude-limited catalogue places the ball's galaxies preferentially below z_true, so each dark event's
catalogue leg peaks at low h; at ensemble level (b) and (c) are the same object (whether `beta(h)`
compensates the skew the ball actually has).
The banked data cannot separate (a) from (b)/(c): the per-candidate covariates (impostor z, σ_z, mass,
c_k) are structurally absent from every artifact (part 1, `[AGENT]`, corroborated by
`CLAIM_P3_IMPOSTOR_CONVENTION_20260822.md` stage-1 inventory). §1.3 names the discriminator.

---

## 1. Stage 1 — information forecast

### 1.0 F5 leg (`docs/RESEARCH_CYCLE.md:68-107` procedure; `docs/SIGMA_Z_SIGMA_M_FORECAST.md:215-236`) `[DOC]`

1-D channel, N = 400, synthetic n(z): σ_eff(H₀)/H₀ = **26.1 % (σ_z = 0.015), 29.8 % (0.025), 29.3 %
(0.05)** — the *"uninformative/railed saturation band"*; the channel crosses 5 % only for spec-z
(σ_z ≲ 10⁻³). Structural reading (N-independent, per the stage-1 rule): a repaired 1D dark-siren
channel at production photo-z scatter is expected to be un-railed but wide, not tight — the mirror's
pure arms (σ_h 0.03–0.06 at ~175 events, C1) are consistent with that. Not quoted below 1.4 % (floor
caveat §4.3). B8 [CAL] owns the production-venue floor; this card uses F5 only to set expectations for
what "un-railed" can look like. No Fisher leg exists (RESEARCH_CYCLE gap) and none is built here.

### 1.1 Null widths at the available N (A15 inputs, all `[LOCAL]` from C1–C3)

| statistic | N | null / scatter |
|---|---|---|
| paired fleet Δmean_h, same 12 realisations, deterministic switch | 12 seeds | sampling variance of the paired difference is exactly 0 (A15 / O2 precedent); seed-generalisation SD 0.027–0.041 ⇒ SEM 0.008–0.012 |
| per-event impostor-leg score at truth, fleet mean | 2152 events / 12 seeds | seed SEM 0.016 (`ft`), 0.022 (`fc`); pooled per-event SEM 0.013–0.016 |
| q1 (z_true < 0.358) impostor-leg score | 538 events / 12 seeds | SEM 0.042 (`ft`) on −0.798; at 4 seeds (≈ 180 events) SEM ≈ 0.073 (9 % relative) |
| production HEAD dark-only pure arm | 1588 events, 1 realisation | σ_h 0.028 (posterior width); no fleet scatter available |

### 1.2 Candidate decisive reads — statistic · expected effect · null width · CPU-h (mirror cell 0.2843 CPU-h per single-h cell, `PREREGISTRATION_HIER_HTHETA_20260826.md:584`; +0.1333 per cell overhead) · decisive?

| read | statistic | expected effect (from the decomposition / record) | null width at N | CPU-h | decisive for THIS claim? |
|---|---|---|---|---|---|
| (i) impostor-WEIGHT switch (per-host weighting conventions: twin / shape / completed / FULL-F) | paired Δmean_h vs FT | twin measured +0.0291 (fused); SHAPE-NULL +0.0006 (row #164); COMPLETED −0.0028 or +0.0344 depending on the D̃ sub-convention (row #167, author [RULE] pending); the whole re-weighting family is structurally bounded by the FT drag itself, **[0, +0.123]** (C1) | 0 (paired); SEM 0.008 generalisation | 0 (banked rescore) … 12 (fleet re-run at 3 h-nodes) | **NO** — every weight switch is a REMEDY; it cannot attribute the remainder to a mechanism. Exhausted at depth 1 except the D̃ sub-convention, which is an author [RULE] (row #167), not a B4 run. |
| (ii) **kernel-WIDTH switch = [HIER] `s`** on the low-z quartile | `R = [S(√2) − S(1/√2)] / |S(1)|`, `S(s)` = mean q1 impostor-leg score at θ = (0, s) (§1.3) | **unknown a priori** — no banked s-sensitivity of the catalogue leg exists (the only reason it is not already answered); the q1 localisation (C2) and #68/#62 make it the prime candidate; the sign is unpredicted | paired-deterministic null = 0; seed-generalisation ≈ 9 % of |S(1)| at 4 seeds | **8.4** (4 seeds × 3 s-nodes × 2 h-nodes; the s = 1 node is the HEAD-basis T-ID baseline) | **YES — the merge discriminator (charter 4.3).** Named as B4.2 in §1.3. |
| (iii) true-host-only counterfactual | paired Δmean_h with the ball reduced to the true host | for the DARK class "true host only" ≡ `L_cat ≡ 0` ≡ O2 — **already measured**: +0.152 / +0.123 (C1). For the catalogue class (b0i) an oracle-ball arm would measure the catalogue-class impostor contribution — the class carrying 6.2 % of the mixture weight; bounded by its −2.4/event catalogue-leg score, split unknown | 0 (paired); b0i seed SEM 0.06 on the score | 11.8 (12 seeds × 3 h) + a new oracle instrument (builder ≠ runner) | **NO** for the dark-class remainder (identical to O2); informative only for the catalogue class, which C4 shows does not compensate. Deferred. |
| (iv) in-ball split | paired per-event read by true-host-in-ball flag | dark class: trivially "not in ball" (no split exists). Catalogue class: the 18.4 % (1D) not-recovered class (part 1 `:117`) ≈ B2's 16.8 % outside-cone class; B2's own R2c read: −16 % likelihood deficit, p = 0.015 | B2's permutation null | 0 (B2 A1, granted) | **NO — declared B2 [CMEM]'s object** (row #221 item 3); B4 does not duplicate it. |
| (v) ensemble-level free reads (done here as forecast inputs) | composite score at model composition (C4); production dark-only O2 (C5) | −0.146 vs 0; 0.60 → 0.713 | seed SEM 0.016; posterior σ 0.028 | 0 | Registered forms recommended as B4.2's zero-compute SECONDARY (runner ≠ this builder). |

### 1.3 The named 4.2 read — B4.2 "KW-Q1": kernel-width discriminator on the low-z quartile, riding B1's θ-driver (F3: predictions registered here, before B1.1 reports)

**Instrument.** B1.1's S0-A θ-driver (`bayesian_statistics.py:3520-3560` θ = (b, s), `theta_sites`;
`d40fe5c8`) applied to the 4 B-SEL realisations 900101–900104 (the FT arm configuration:
`catalogue_numerator_survival="phi"`, fused, HEAD Σ^φ), at θ = (0, s) for s ∈ {1/√2, 1, √2}
(the HIER grid step), h ∈ {0.725, 0.735}. Primary: `theta_sites="all"` (the paired, registered HIER
form). Optional diagnostic: `theta_sites="2.2"` (the per-host catalogue-leg kernel only) — labelled
DIAGNOSTIC-NOT-FIX under §0.2 row 7 (an unpaired numerator change may never be adopted).
**Statistic.** `S(s)` = mean over the FROZEN q1 event set (z_true < 0.358 by the fleet quartile of
`b4_imp_stage1_events.csv`, ≈ 45 events/seed) of `s_imp,i(s)` (secant over 0.725/0.735, computed from
the driver's diagnostics CSV through the same subtraction as C2); `R = [S(√2) − S(1/√2)] / |S(1)|`.
**Bands (two-sided in |R|, A8; sign and the ordering `S(1/√2), S(1), S(√2)` reported):**
- **KERNEL-WIDTH-OWNS**: |R| ≥ 0.5 (a factor-2 width change moves the q1 impostor-leg score by at least
  half its size) ⇒ the remainder is a kernel-width-class object ⇒ **B4 MERGES INTO B1** (charter 4.3), handing
  B1 the q1 localisation and the absorption prediction (does |S(s)| have a minimum inside the grid?).
- **KERNEL-WIDTH-INERT**: |R| ≤ 0.2 ⇒ not a width object ⇒ B4.3 = the normalisation/mixture-weight
  derivation (C6 (b)/(c)) + the per-candidate instrumented run (part 1 §7, 3.4 CPU-h) for the q1
  impostor z-offsets.
- **MIXED**: 0.2 < |R| < 0.5 ⇒ report both; no branch forced.
**Operating characteristics (A15) at N = 4 seeds:** the paired read on identical events has zero
sampling variance under R = 0 (false-fail from noise: none); the seed-generalisation SEM of `S` is
≈ 0.073 on −0.80 (9 %), so an INERT verdict generalises to the fleet at ≈ 2σ per 0.2 of R and an OWNS
verdict at ≥ 4σ. A single seed would already resolve |R| = 0.5 (per-event SD ≈ 1.0, 45 events ⇒ 15 %
relative). 4 seeds are chosen for the cross-seed ordering check, not for power.
**Gates:** T-ID (the s = 1 node reproduces a HEAD-basis FT re-evaluation bit-identically — GATE
PARITY inherited from B1, since θ engages the smeared kernel path, `bayesian_statistics.py:2799-2806`);
ENG on the CATALOGUE leg (`L_cat_no_bh` must differ across s-nodes on ≥ 99 % of active rows — else
NULL-BY-CONSTRUCTION, A15 corollary); GATE I on every node.
**Invariants (A10):** fused cell · phi twin · Σ^φ slot · symmetric mass window (irrelevant to 1D) ·
H_GRID nodes 0.725/0.735 · the 4 realisations and their q1 membership frozen from this card's CSV ·
`w_pop` form (G2b, last audited 2026-07) · `B_num` domain (O4/O6, 2026-08-21). **Blindness sentence:**
this design cannot detect a kernel SHAPE error (non-Gaussian error law) — only a width scale — nor any
defect common to the catalogue and completion legs (inside S̄_φ / D̃_φ), nor anything in the GW term.
**Falsifier (A14) for the attribution "the remainder is a low-z, catalogue-share-localised
violation":** the KW-Q1 diagnostics CSVs also give S over q2–q4; if q1's share of Σ s_imp at s = 1 on
the HEAD basis falls below 50 %, C2's localisation is withdrawn regardless of R.
**Cost:** 4 × 3 × (2 × 0.2843 + 0.1333) = **8.4 CPU-h** primary; +5.6 CPU-h for the site-2.2 diagnostic
(s ≠ 1 nodes only) ⇒ ≤ 14 CPU-h, local, no cluster exposure. Well inside the charter's ≤ 20 CPU-h.
**Secondary, zero compute (registered with the same prereg):** the production HEAD dark-only O2 read of
C5 with a VALIDATED join (bands: dark-only pure arm map_h ≥ 0.66 and mean ∈ [0.66, 0.78] ⇒
NECESSARY-CAUSE-CONFIRMED; map_h ≤ 0.605 ⇒ REFUTED; else MIXED).
**Rule 2:** this card's author built the forecast instruments; the KW-Q1 runner and the secondary's
runner must be different agents; B1.1's driver author may not be the KW-Q1 runner either.

### 1.4 Merge declaration (charter 4.3)

**NOT merged at depth 1.** The decomposition localises the remainder (C2) but does not establish
kernel width as its mechanism; the merge trigger is KERNEL-WIDTH-OWNS at B4.2. Declared now so the
orchestrator can schedule KW-Q1 behind B1.1's driver (F3) and skip any independent B4 instrument build.

---

## 2. What is explicitly NOT claimed

- Not claimed: that the impostor leg is the SUFFICIENT owner of the production rail — the fused
  completion leg pulls +0.11 high on production (C5) and +0.038 on the mirror (C1); B3 [POP] owns that.
- Not claimed: any per-candidate statement (impostor z, σ_z, mass, c_k) — none exists in banked data.
- Not claimed: a remedy. Every catalogue-leg re-weighting is bounded by C1's [0, +0.123] and must be
  paired (§0.2 rows 3, 5, 7).
- Not claimed: that kernel width is the mechanism (C6 is UNDETERMINED); not claimed that the mixed-fleet
  falsifier of C4 has run.
- Not claimed: transfer of the b0i covariate table (part 1) to the dark class — part 1's η² are
  whole-ball aggregates on the catalogue class; C2's η² are impostor-leg scores on the dark class.
- Not claimed: any number here as banked. All `[LOCAL]` values are stage-1 inputs without bands.

## 3. Exonerated — do NOT re-open without new evidence (carried from §0.2; binding union = register §1 ∪ §2 ∪ ledger §2)

catalogue Jacobian · Fisher frame · p_det estimator choice · p_det inside/outside · h-prior · `volume_trunc`
(seed600-scoped) · the z leg as a 2D-only source · ln-M draw · realization plumbing · 2D candidate-window
membership · mass-kernel family · Option-A drift (+0.017 1D residual stays OPEN) · HA · HC · HB · full Gray
mixture as compensation · `w_G` bookkeeping AS A FIX (calibration LIVE) · hard clamp on observed z ·
`w_pop` tuning · p_det inside the numerator ALONE · depth truncation · zero-host fallback as the rail cause ·
Ω_m era · `L_comp`/`B_num` as a defective integral (B_num "carries but is not shown defective") ·
`volume_deconv` h-dependence · p_det anchor · information starvation · spec-z rescue · PV frame ·
code-hygiene bundle · numerator-only cleans and the local same-kernel denominator · [A2-VOID] must not be
cited as standing · **#87's harness impostor null — venue-scoped and overtaken by O2 (row #149), not to be
cited against the realistic-catalogue mirror.**

## 4. Errors made this session — do not inherit them

1. The first draft of §1.3 named a model-matched MIXED fleet as the 4.2 read; the zero-compute composite
   of C4 (−0.146 at w̃_G = 0.062) made it unnecessary before anything ran. It survives only as C4's
   registered falsifier.
2. The production dark/in-catalogue split (C5) is an ASSUMPTION-JOIN on CSV row order; the joined
   fraction matching 76/1588 is a consistency check, not a validation. The 0.713 number is secondary
   until the join is validated through the harness's own index.
3. The catalogue-share quartile split in `b4_imp_stage1_forecast.json` has degenerate lower quartiles
   (edges 0, 0): only the q3/q4 rows are meaningful; q1/q2 are the inactive events.
4. FC/FT predate the Σ^φ adoption; every fused-basis number in C1–C3 is the 2026-08-23 basis, not
   byte-HEAD. Re-anchor at HEAD before quoting into a prereg band (A11).
5. No number here carries a band; the three scripts are forecast instruments built by this agent and
   may not be run by it as a registered measurement (rule 2).

*(committed line — append only below)*

## 5. B4.2 "KW-Q1" — RESULT RECORD (read out 2026-08-29 by the independent reader; run by the
orchestrator; launched under rows #222/#223 — charter node B4.2)

**Run of record:** `fanout1_20260829/kwq1_registered_run` (4 seeds 900101–900104, nodes
s_minus/truth/s_plus, `_ft_sites2.2_nosmear` suffix per `B4_2_KWQ1_RUN_FORM_NOTE.md`'s registered
run form, h ∈ {0.725, 0.735}, FT config); parity re-evaluation
`fanout1_20260829/kwq1_parity_run/s0a_seed900101/node_truth_ft_sites2.2_nosmear`. Full detail,
independent re-derivation, and the A15 seed-generalisation check:
`fanout1_20260829/B4_2_KWQ1_READOUT_RECORD.md` + `b4_2_readout.json`.

**T-ID/PARITY:** `combined_no_bh` and `L_cat_no_bh` bit-identical (max\|Δ\| = 0.0) between the
primary and parity re-evaluations, seed 900101, both h-nodes, 174 events each — **PASS**.

**S(s), independently re-derived (matches the scorer to full float precision):**
S(1/√2) = −1.0456670 (sem 0.076542, n=191), S(1) = −1.0205308 (sem 0.069300, n=191),
S(√2) = −0.9591134 (sem 0.062842, n=191). Ordering monotonically increasing with s.

**R = [S(√2) − S(1/√2)] / \|S(1)\| = +0.084812.**

**Gates:** GATE I max_rel 7.613×10⁻⁸ (tol 2×10⁻⁶) — PASS. GATE ENG: 486/486 active rows'
`L_cat_no_bh` differ across s_minus/s_plus (fraction 1.0 ≥ 0.99) — PASS, non-vacuous. GATE
T-ID/PARITY — PASS (above).

**Falsifier (A14):** q1 share of Σ s_imp at truth = 92.25 % (q2 7.28 %, q3 0.47 %, q4 0.002 %) ≥
the 50 % floor — **NOT withdrawn**; the C2 low-z localisation is reconfirmed, even more
concentrated than the 12-seed forecast (91.7 % ft).

**Per-seed robustness:** R per seed = +0.1563 (900101), +0.0386 (900102), +0.1105 (900103),
+0.0516 (900104); across-seed mean 0.0892, SD 0.0546, SEM(N=4) 0.0273. Every individual seed
lands inside the INERT band (max \|R_seed\| = 0.156, 22 % below the 0.2 ceiling) — the verdict is
not a borderline call.

**A15:** the registered forecast (SEM of S ≈ 0.073, extrapolated from the 12-seed pooled SEM) is
compared against the actual across-seed SD of S(1) measured directly on these 4 seeds: **SD =
0.10584** (SEM(N=4) = 0.05292), same order as forecast. More directly: R is a within-seed paired
ratio, so its own across-seed scatter (SD 0.0546) is much tighter than S(1)'s (SD 0.106) — level
shifts common to all three s-nodes within a seed cancel in the ratio.

**Band:** \|R\| = 0.0848 ≤ 0.2 ⇒ **KERNEL-WIDTH-INERT.**

**Instrument disclosure (carried, not resolved here):** the same θ-hook driver family
(`hier_s0_driver.py`, S0-A) returned **B0-A′ INSTRUMENT-DEFECT** on the b0i mirror score-at-truth
null test (Z_b = −3.676, Z_s = −7.079; `hier_s0_registered_run/s0a_score.md`; forensic in
progress). KW-Q1's design (a within-run paired comparison across s-nodes) differs from that
score-at-truth null test, so the defect is not automatically inherited, but the instrument as a
whole is not yet certified clean. **Verdict is REPORTED-ONLY** with this disclosure.

**Verdict of record (§1.4 merge declaration):** KERNEL-WIDTH-INERT ⇒ **B4 does NOT merge into
B1.** Per §1.4 and `SYNTHESIS_DOCKET_1_20260829.md` §2 "B4 [IMP]" condition (d): **B4.3 = the
mixture-weight/catalogue-depth h-slope derivation** (C6 (b)/(c): `s_β = −3.2891/h`,
`s_L = −27.08/h` on active events) **+ the per-candidate instrumented run** (part 1 §7,
3.4 CPU-h, contingent on a non-physics-hook ruling), for the q1 impostor z-offsets.

**Cost measured:** 6.152 CPU-h (main run 5.514 CPU-h, wall 1417.79 s at 14 cores; parity
0.638 CPU-h, wall 164.07 s at 14 cores) against the registered 8.4 CPU-h primary estimate —
≈27 % below estimate.
