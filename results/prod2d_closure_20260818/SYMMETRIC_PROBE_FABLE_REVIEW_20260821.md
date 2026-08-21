<!-- SYMMETRIC PROBE (model-tendency protocol section 3): fresh-context FABLE agent, brief byte-identical to the Opus reviewer's up to two documented deviations (worktree path pinned at f59a6f48 = the uncontaminated pre-correction snapshot; venv bootstrap note). The probe could not see the Opus review, rows #152-#153, or any correction. Banked verbatim 2026-08-21. -->

All reading and counter-computation done. Here is the adversarial review.

---

# Adversarial review — C-SG v3 overnight campaign (2026-08-21)

**Scope reviewed:** readout report, full prereg incl. all appended blocks, ledger rows #149–#151, all four scorers, `selfgen_control.py`, retrospective ledger 1–2. All decisive numbers below were **recomputed from the banked 46 JSONs / 92-file manifest**, not taken from the report.

---

## Finding 1 — MAJOR: "The full channel reproduces the campaign's headline −0.108 in every arm" is substantially a grid-rail artifact, and the cross-arm bias table uses the wrong reference for the δ arms

**Claim attacked:** Readout report §4 structural fact 2 ("the full channel lands on the campaign's headline number in every arm (−0.104…−0.110 vs B-SEL's −0.1083): the O2/O3 three-channel structure transfers quantitatively… the '−0.11' is now *reconstructed from first principles*"); same language in ledger row #151 item 3; the §4 arm table.

**Argument (computed):** `csg_fleet_readout.py` computes `bias = mean_h − 0.73` for **every** arm, including δ− (h_gen=0.68) and δ+ (0.78). From the banked JSONs:

| arm | h_gen | full mean_h | report "bias" (vs 0.73) | true bias vs own h_gen |
|---|---|---|---|---|
| csgf | 0.73 | 0.6210 | −0.1090 | −0.1090 |
| csgdm | 0.68 | 0.6201 | −0.1099 | **−0.0599** |
| csgdp | 0.78 | 0.6256 | −0.1044 | **−0.1544** |

The full-channel posterior **rails at the 0.6 grid edge in 31/31 F+δ seeds** (map_h = 0.600, r_low = 100%, all arms — I checked every JSON). A railed posterior's mean is ≈ edge + O(σ_h) ≈ 0.62 *regardless of h_gen* — the arm means move only 0.0055 while h_gen moves 0.10. The "agreement with −0.1083 in every arm" is therefore the signature of grid-floor saturation plus a fixed reference of 0.73, not a quantitative transfer of the three-channel structure. (B-SEL itself was documented as floor-saturated, railed 12/12 — prereg §6 table — so the agreement is two railed posteriors of similar width agreeing about the location of the grid edge.) The matched-channel columns in the same table have the same convention problem: δ-arm matched bias vs own truth is −0.0363 (δ−) and −0.0995 (δ+), not the tabulated −0.0863/−0.0495 — i.e. on the *bias* statistic the violation is strongly h_gen-**dependent** (2.7× spread), directly contradicting the table's visual message. Only the **score** rows are computed at each arm's own h_gen and legitimately support h_gen-independence.

**Resolution:** Restate fact 2 at the score level only (full-channel score −0.225…−0.257 across arms vs B-SEL ≈ −0.28 — a real but looser agreement); relabel the δ-arm bias columns or add own-truth columns; strike "reconstructed from first principles" or downgrade it to the F/E arms at 0.73.

## Finding 2 — MAJOR: all mean_h-based statistics are heavily grid-truncation/prior-shrinkage contaminated; the banked bias₁₅ = −0.0665 is not the defect's size

**Claim attacked:** BAND C confirmatory verdict (bias −0.0665 ≤ −0.0423), the §4 matched-bias comparisons, ledger #151 item 2's "79% of its bias scale".

**Counter-computation:** I rebuilt the matched channel from the banked per-event CSVs and scored it on the extended grid `H_GRID_FULL` (0.50–0.86, identical machinery) vs the registered `H_GRID_41` (0.60–0.86), 8 seeds/arm:

| arm | mean_h on [0.60,0.86] | mean_h on [0.50,0.86] | shift |
|---|---|---|---|
| csgdm | 0.6437 | 0.5828 | **−0.0609** |
| csgf(8) | 0.6543 | 0.6021 | **−0.0522** |
| csgdp | 0.6805 | 0.6444 | **−0.0361** |

Matched σ_h runs 0.021–0.062 (up to 0.83·σ_prior); with the grid centered at exactly 0.73, wide posteriors shrink toward 0.73 and the low edge clips the left tail. The 41-node bias numbers are ~50% shrunk versions of an underlying matched bias ≈ −0.13 *or worse* (the 0.50 edge still truncates). Direction of error is conservative for the defect claim, but the specific number −0.0665 is a property of the scoring grid, not of the estimator, and should not be banked as "the defect magnitude". The per-event **score** statistic is grid-local (nodes 0.725/0.735 etc., verified present with 0.005 spacing, 0 skipped events, 0 physics_floor engagements in the matched matrices) and is clean of all of this — the campaign's decision to make it primary is what saves the night.

**Resolution:** Bank the score, report mean_h-based quantities as grid-conditional diagnostics only; any future bias-scale claim needs an extended grid or a truncation-aware estimator.

## Finding 3 — MAJOR: GATE S's sub-unit slope is quantitatively explained by Finding 2; the report's "grid-edge truncation works *against* explaining this away" is backwards; the independence assumption behind SE(ŝ) is factually wrong

**Claim attacked:** Report §5 ("the matched posterior responds… at a third of the unit slope. Grid-edge truncation at h_gen=0.68 works *against* explaining this away… The sub-unit slope is itself an unexplained diagnostic worth a registered follow-up"); the [RULE] framing in §10 item 2.

**Argument (computed):**
- The three arm means (0.6437/0.6635/0.6805) fit mean ≈ 0.73 + 0.368·(h_gen + b − 0.73): a shrink-toward-grid-center model with factor 1−λ = 0.368 and underlying b ≈ −0.18 — which is essentially S_REF (−0.19). One shrinkage factor simultaneously explains ŝ = 0.368, bias₁₅/underlying ≈ 0.37, and the δ-arm own-truth bias asymmetry (−0.036 low / −0.0995 high: means pulled toward 0.73 from both sides). Nothing "sub-linear in the generator" is needed; the generator has no response function — it draws at h_gen exactly (GATE D checked this at each h_gen).
- Truncation deflates ŝ: on the [0.50,0.86] grid the paired slope rises from 0.368 to (0.6444−0.5828)/0.1 = **0.616**, still truncated at 0.50. So truncation works *for* the mundane explanation, not against it — the report's sentence is at best ambiguous, at worst inverted.
- The OLS SE = 0.186 treats 31 points as independent, but the arms **share seeds with common random numbers** (measured cross-arm correlations of per-seed matched mean_h at equal seed: corr(F,δ−)=0.92, corr(F,δ+)=0.95, corr(δ−,δ+)=0.87). The correct paired per-seed slope estimator gives mean 0.368, **SE 0.130** (t = 2.8 from 0, 4.9σ below 1). INERT still fires by the 3·SE letter — barely (3·0.130 = 0.390 > 0.368) — but the registered sentence "the arm carries no h-information" is factually false, and the registered rule's own operating characteristics were computed under a wrong independence assumption.

**Resolution:** Rule GATE S as miscalibrated-by-design (it measures shrinkage × response, on correlated points); do **not** open a "sub-unit generator response" follow-up — it would chase a scoring artifact. The retrospective ledger's proposed three-outcome slope amendment should add: paired estimation across common-seed arms, and truncation-aware mean statistics.

## Finding 4 — MAJOR: the amended GATE V is nearly vacuous, and it re-admitted exactly the pathology it existed to guard against

**Claim attacked:** "GATE V (amended, all 46): 46/46 PASS" (scorecard §6); the amendment block's claim that the gate "remains can-fail".

**Argument:** The amended thresholds (span ≥ 1 nat over a **200-event** posterior; σ_h ≤ 0.9·σ_prior) were chosen below the minimum of *all* observed data (reference min span 2.01, pilot min 2.15, fleet min 1.6): the gate can now only fail an *exactly flat* posterior (B-F1 mode). But GATE V's registered purpose (prereg §0 item 4) was to protect **mean-based** targets from the grid-center coincidence — and Findings 2–3 demonstrate that posteriors with σ_h up to 0.83·σ_prior passed the amended gate while their means are demonstrably grid-dominated. "46/46 PASS" therefore certifies almost nothing about the mean_h statistics it was designed to protect. Procedurally the amendment was handled well (STOP honored, no fleet on a fired STOP, v2 verdicts preserved in every JSON, reference-derivation on 12 B-SEL posteriors independent of the pilot); the use of B-SEL (the biased arm) as reference is legitimate *for a vacuity gate* — bias doesn't contaminate a span/width calibration. But note the counterfactual freedom: under v2 rules the 3 failing pilot seeds would have been excluded from σ̂ and the F-15 statistic; the amendment decided their admission after their values were visible (mitigated only by the fact that all 46 passed, so no discretion was actually exercised). A15/A17's own logic cuts both ways: a gate re-derived to pass everything observed has an unpublished false-*pass* rate of ~100% for weakly-informative-but-nonflat posteriors.

**Resolution:** For the author's [RULE] on GATE S/mean_h: treat the amended GATE V as an anti-*flat* gate only; any mean_h-bearing conclusion needs a separate informativeness criterion (e.g. σ_h/σ_prior ≤ 0.5, which is what v2 demanded — 5/12 B-SEL and several C-SG seeds genuinely fail it *for mean purposes*).

## Finding 5 — MAJOR (conditional-validity): the matched-channel derivation's load-bearing normalization identity is assumed, never verified — and the generator retains at least one un-listed mismatch

**Claim attacked:** O3 registration ("the model-matched conditional for its draw is `B_num/β_Ḡ_φ`… If the completion leg is internally self-consistent and the draw matches the model's dark-detected density, `E[∂_h ln L_matched] = 0` at truth"); report §7's INTERNAL-DEFECT definition.

**Argument:** The score-zero theorem needs `B_num(x;h)/β_Ḡ_φ(h)` to be the normalized data density, i.e. **∫B_num(x;h)dx = β_Ḡ_φ(h)** under the estimator's own discretizations. This is never tested. Three concrete ways it can fail *h-dependently*, all invisible to the night's gates:
1. **Domain caps:** `B_num` carries an analysis-depth z-cap (`bayesian_statistics.py:~5141-5147`, added to match D(h)); `β_Ḡ_φ` is a separate trapezoid table (`:2066`) over its own z-grid, and `S̄_φ`'s z_max(h) shrinks/grows with h (prereg GATE H note). Any cap mismatch is exactly an h-dependent normalization tilt — i.e. exactly a constant score offset of the observed kind.
2. **Sky structure:** `B_num` uses cone-weighted per-pixel `1−f_k`; `β_Ḡ_φ` uses sky-averaged `1−f̄`. Events land in pixels ∝ their own dark-detected mass Z_Ω, so E_Ω[∂_h ln Z_Ω] ≠ ∂_h ln E_Ω[Z_Ω] (a Jensen-type gap) unless pixel h-slopes are uncorrelated with pixel weights. Probably small, but unbounded by any gate that ran.
3. **The generator observes sky exactly** (`phiS/qS` = drawn truth, no scatter) while the estimator assumes the *donor's* sky covariance — a residual generator–model mismatch that design B did **not** remove and that is not on the six-invariant list; nor is the injection sampler `_draw_dark_hosts_pixelated` itself (a *seventh* shared production object whose correctness the verdict silently assumes; GATE D's KS check is z-marginal-only, n=200, D_crit=0.096 — blind to sky/mass/conditional structure).

**Important caveat to my own attack:** if the ∫B_num = β_Ḡ identity fails h-dependently, the estimator's mixture `(β_G L_cat + B_num)/D̃` with `D̃ = α + β_Ḡ` is *itself* misnormalized in h — so "internal defect" survives in the broad sense either way. What is at stake is the *localization* ("the completion leg's conditional violates score-zero") and whether the fix lives in `S̄_φ`, the β_Ḡ table, the z-caps, or the sampler. The designated `S̄_φ` audit is the right next step; it should **explicitly test the integral identity ∫B_num dd̂ = β_Ḡ_φ per pixel and the z-cap parity**, and include the `clip(S_4D,0,1)` generator-side vs unclipped-integrand estimator-side comparison.

*(On the specific sub-questions: yes, treating α_G_φ/D̃_φ as h-only per-event constants is legitimate — GATE T verified ≤2e-6 spread, and they are global objects by construction (`:2425-2427`); the β_Ḡ = D̃ − α subtraction is exact at storage precision.)*

## Finding 6 — MINOR: the INTERNAL-DEFECT branch fired by only ~1.1σ over an arbitrary edge; "self-consistency rejected" is the robust statement

**Claim attacked:** "BAND C = INTERNAL-DEFECT on both frozen statistics."

**Numbers:** Realized F-arm score sd = 0.0751 (pilot σ̂ = 0.0482, a 1.56× underestimate, within the published 3-dof CI) ⇒ realized SE₁₅ = 0.0194. S̄₁₅ = −0.1173 is **6.0σ from zero** (decisive) but only **1.07σ below** the S_REF/2 = −0.0966 edge. A modest unlucky fleet would have returned MIXED with identical science. The halved-reference edge is registered but conventional — it has no calibration meaning, and S_REF comes from the same banked B-SEL data that motivated the channel choice (mild circularity in band *placement*, not in the statistic). Related: the A15 N-adequacy gate passed at 7.8σ **using the pilot's underestimated σ̂**; at the realized sd the half-reference significance is 0.0966/(0.0751/√15) = **4.98σ — the registered ≥5σ adequacy bar would have failed**. Pilot-seeds-inside-F-15 is clean under normality (sample mean ⊥ sample sd; bands use σ̂ only as a scale), and the 3-dof fragility was published honestly (false-fail up to 42% at the hi edge) — though the same multiplier was never applied to the power line.

**Resolution:** Ratify "matched-channel self-consistency REJECTED at ≥6σ"; treat the INTERNAL-DEFECT-vs-MIXED branch label as low-confidence bookkeeping.

## Finding 7 — MINOR: BAND R's registered premise is false in the implemented code — F and E are event-paired, not independent

**Claim attacked:** Band-formula registration ("F vs E **declared independent** — the σ-draw desynchronizes the shared RNG stream; pairing would need stream surgery"); threshold 3√2·σ̂/√15 = 0.0296.

**Computed:** In `draw_csg_realization` the σ draw happens **after** all candidate/accept/donor draws, so F and E at equal seed share identical accepted events. Measured: corr(F,E per-seed matched mean_h) = **0.998**, mean|F−E| per seed = 0.0018. The observed gap 0.0001 against a 0.0296 band is therefore not a stringent σ-mode invariance result under the registered null — it is a paired comparison scored against an independence-scale band ~20× too loose. The *conclusion* (σ-mode invariance) actually gets **stronger** under the correct pairing, but the registered operating characteristics are wrong, which is precisely the A17 failure mode this same session codified.

## Finding 8 — MINOR: O2/O3 "deterministic read ⇒ no statistical band" conflates data-determinism with inferential uncertainty; the three-channel decomposition is path-dependent and partly rail-saturated

**Claim attacked:** Row #149's "73%" headline; row #150 item 3's decomposition "matched −0.0846 ⊕ tilt ≈+0.055 ⊕ impostor −0.079 ⇒ full −0.1083".

**Argument:** Δ_bias = +0.0792 is exact *for these 12 seeds*, but as an arm-level claim it carries seed scatter: per-seed Δ sd = 0.0414 ⇒ SE = 0.0119, so "73%" is 73 ± 11 (rel. %). The decomposition is a telescoping identity (order matters: "impostor drag" is measured with the tilt present, and all three terms are mean_h-based, so the full-channel term is rail-saturated at the 0.6 edge — without the grid floor the "impostor" share would differ). The subtraction algebra itself is clean: I verified there is **no leak** — the GATE I identity (`combined = (α/r_Malm·L_cat + B_num)/D̃` to 5.5e-7 = 7-sf storage precision, every cell, 12 seeds) simultaneously validates the `B_num`-column/`B_num_phi` equivalence (derived form) and the r_Malm placement; the amended 2e-6 tolerance is correctly derived from the `_seven_sf` quantization (4.9e-7/column) and its ≲1e-5 propagation bound to Δ_bias is right. The post-failure GATE I/P amendments were mechanistically diagnosed and disclosed — acceptable.

## Finding 9 — NOTE: the O2/O3 materiality floor (0.0023) was set from the optimistic edge of the projected σ range

0.0023 = 0.009/√15 used the *best* value of the prereg's own 0.009–0.022 projection; realized C-SG SE₁₅(bias) = 0.0101 — 4.4× coarser. Moot here (fired values 0.079/0.085 ≫ 0.011) but the "below C-SG resolution" band label was miscalibrated at registration.

## Finding 10 — NOTE: shared-machinery conditioning on the significance claims

All 46 seeds share the catalogue, donor pool, completeness maps and p_det interpolator; the quoted σ's cover draw noise only. Appropriate for an internal-consistency test (the shared objects *are* the invariants), but "6σ"/"7.8σ" should always be read as conditional on that machinery — consistent with §7's conditionality, which is stated honestly.

---

# Overall assessment

**Solid (would survive review):**
- The **primary result**: the matched-channel per-event score at h_gen is ≈ −0.117, ~6σ from zero, invariant across h_gen (paired tests: t = −1.38, 0.64) and σ-mode (event-paired, corr 0.998) — the estimator fails its own dark-sector score-zero test on self-generated data. This statistic is grid-local, floor-free (0 zeros, 0 skipped events — verified), and prereg-disciplined.
- **O2's impostor finding** (catalogue leg carries ~73 ± 11% of B-SEL's −0.1083; 12/12 seeds positive) — identity-verified, independently recomputed, no leak found.
- The **procedural spine**: scorers pre-data, STOP honored before fleet spend, superseded verdicts preserved, everything recomputable. Genuinely good practice.

**Conditional:**
- "**INTERNAL-DEFECT**" as a branch: self-consistency rejection is decisive, but the branch label sits 1.1σ over an arbitrary halved-reference edge, and the *localization* to "the completion leg's own conditional" rests on the unverified ∫B_num = β_Ḡ_φ identity plus a seventh shared object (the injection sampler) and a residual sky-observation mismatch. The `S̄_φ` audit ([RULE] 4) is the correct next step and should test that identity and the z-cap parity explicitly.
- **O3's −0.0846** as a magnitude: real in sign and existence, grid-shrunk in size.

**Overclaimed (should not be ratified as written):**
- "The full channel reproduces −0.108 **in every arm** / the −0.11 **reconstructed from first principles**" — rail artifact + wrong bias reference for the δ arms (own-truth full biases: −0.060 / −0.154; all arms rail at map = 0.600, 100% r_low).
- The cross-arm **bias** table's implicit h_gen-independence (only the score is h_gen-independent; own-truth matched bias spans −0.036…−0.0995, the shrinkage signature).
- GATE S's "**attenuated slope is an unexplained diagnostic**" and "truncation works against explaining this away" — a single shrinkage factor (0.37; 0.62 on the extended grid) explains slope, bias ratio, and δ-asymmetry simultaneously; the proposed follow-up thread would chase a scoring artifact; and the INERT SE was computed under a demonstrably false independence assumption (cross-arm seed correlations 0.87–0.95; paired SE 0.130).
- bias₁₅ = −0.0665 as a bankable defect size (underlying ≥ −0.13; grid-dependent).

**On the six [RULE]s:** (1) ratify as "self-consistency rejected", not the branch label per se; (2) GATE S: void the *letter*, keep score primary, no sub-unit-slope thread — register a truncation-aware/extended-grid mean statistic instead; (3) re-grades are supported; (4) endorse, with the ∫B_num = β_Ḡ_φ and z-cap tests added explicitly; (5) A17: endorse — Finding 7 shows a second instance the same night; (6) landscape un-gate: reasonable, conditional on (4).