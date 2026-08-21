<!-- Banked verbatim from the 2026-08-21 Opus adversarial review subagent (author-requested). Findings independently re-derived by the orchestrator before any correction was applied: FATAL-1, FATAL-2, MAJOR-1, MAJOR-8 all CONFIRMED by recomputation (ledger row #152). -->

# Adversarial review — C-SG v3 campaign (2026-08-21 overnight)

I read the readout, the full prereg (above and below the freeze line), rows #149–#151, all four scorers, `selfgen_control.py`, and `RETROSPECTIVE_LEDGER.md` entries 1–2. I then re-derived every decisive number from the banked diagnostics and ran independent counter-computations. Findings ranked most-severe first. Numbers I computed are marked **[computed]**.

---

## FATAL-1 — The δ-arm "bias" numbers are not biases. `csg_fleet_readout.py:98`, readout §4 table + §4 fact 2, ledger #151 item 3

`csg_fleet_readout.py` hardcodes `H_TRUE = 0.73` and computes `"bias": means.mean() - H_TRUE` **for every arm**, including `csgdm` (h_gen = 0.68) and `csgdp` (h_gen = 0.78). The readout §4 table then prints those under the header "matched bias" next to an `h_gen` column, and the report/ledger read them as biases.

**[computed]** True biases (`mean_h − h_gen`) vs reported:

| arm | h_gen | channel | reported "bias" | **true bias** |
|---|---|---|---|---|
| csgdm | 0.68 | matched | −0.0863 | **−0.0363** |
| csgdp | 0.78 | matched | −0.0495 | **−0.0995** |
| csgdm | 0.68 | full | −0.1099 | **−0.0599** |
| csgdp | 0.78 | full | −0.1044 | **−0.1544** |
| csgdm | 0.68 | pure | −0.0129 | **+0.0371** |
| csgdp | 0.78 | pure | +0.0281 | **−0.0219** |

Consequences that propagate into the ruled text:
- Readout §4 fact 2 / ledger #151 item 3 — "**the full channel lands on the campaign's headline number in every arm** (−0.104 … −0.110 vs B-SEL's −0.1083)" — is false as a bias statement. The true full-channel biases are −0.060 / −0.109 / −0.108 / **−0.154**, a 2.6× spread. What is actually constant across arms is `mean_h ≈ 0.62`, i.e. the posterior *location*, not the bias.
- The matched-channel bias is strongly h_gen-dependent (−0.036 → −0.067 → −0.100, a 2.7× ramp), which sits uneasily beside the report's "the violation is h_gen-independent" framing (that claim is true only of the *score*, which is evaluated at each arm's own h_gen — `score_at_h_gen` does this correctly).

**Resolves by:** re-issue the §4/§6 tables with `mean_h − h_gen`, and re-word §4 fact 2 and #151 item 3. The F/E arms are unaffected (h_gen = 0.73).

---

## FATAL-2 — The full channel rails at the grid floor in 46/46 seeds; the "−0.11 reconstructed from first principles" is a rail coincidence. Readout §4 fact 2, §2, ledger #151 item 3

**[computed]** Full-channel `map_h` = **0.600 exactly (the H_GRID_41 floor) in all 46 seeds, all four arms**, with per-seed sd of `mean_h` of only 0.0036–0.0082 — the signature of a posterior pinned at an edge, not of a converged estimate. Extending to `H_GRID_FULL` (floor 0.50) moves the full-channel mean from 0.621 → **0.535** (csgf), i.e. it follows the floor.

B-SEL's full channel is *also* railed — the prereg's own §6 table records `bsel: railed = yes, r_low 12/12`. So the celebrated "quantitative transfer" is two railed posteriors landing on the same grid edge, and the "headline −0.11" is just `0.62 − 0.73`. The full-channel *slope* in h_gen is **[computed] 0.055** on H_GRID_41 (0.110 on the extended grid) — the full channel carries essentially **zero** information about h_gen.

This is the single largest interpretation overreach of the night: "the −0.11 the campaign has chased since row #137 is now *reconstructed from first principles*" (readout §4) is not supported. What was reconstructed is that a railed posterior rails.

**Resolves by:** withdraw the "reconstruction" language; report the full channel as *railed, uninformative, reported-only* (which is what the v3 design already designated it) and drop it from the evidence chain entirely.

---

## MAJOR-1 — INTERNAL-DEFECT vs MIXED is a ~1σ call presented as a clean binary verdict

The band edge is `S_REF/2 = −0.0966`; the observation is `S̄₁₅ = −0.1173`. The frozen band used the pilot's `σ̂_score = 0.0482` (SE15 = 0.01245). **[computed]** the *realized* F-arm scatter is `sd = 0.0751` → **SEM = 0.01939**, 1.56× larger.

- Distance from zero: **6.05σ** — solid.
- Distance past the INTERNAL-DEFECT edge: **1.07σ**.
- Seed bootstrap (20k resamples): **P(S̄₁₅ > −0.0966) = 0.133**.
- Leave-one-seed-out: S̄ ranges −0.1316 … −0.1068, so no single seed flips it — the fragility is scatter, not an outlier.

So "the score is not zero" is bankable; "the score is at least half the B-SEL reference" is a coin-flip-adjacent call. The readout's §6 scorecard and #151 item 1 present it as a categorical branch.

**Resolves by:** state the verdict as `S̄₁₅ = −0.1173 ± 0.0194 (realized)`, "non-zero at 6.0σ; ≥ half-reference at 1.1σ", and rule the branch label on that basis rather than on the point estimate crossing an edge.

---

## MAJOR-2 — The registered N-adequacy STOP would have fired on the realized scatter; the too-narrow band favours the fired conclusion

`csg_pilot_bands.py:123-125`: fleet launches only if `|S_REF/2| / SE15 ≥ 5`. Banked value: **7.761** → PASS.

**[computed]** with the realized F-arm sd: `0.0966 / 0.01939 = **4.98** < 5`. The fleet launched on a gate that the data it then produced would have failed. The pilot's own published caveat covers this direction — the 3-dof false-fail table in `csg_pilot_bands_output.json` states the SELF-CONSISTENT band's false-fail rate could be **42%** if the true sd sits at the high edge of the 95% CI (the realized ratio, 1.56×, is inside that CI). Note the asymmetry: an under-estimated σ̂ narrows the SELF-CONSISTENT band, and narrowing it pushes the verdict *away* from SELF-CONSISTENT — i.e. toward the conclusion that fired.

**Resolves by:** an author ruling on whether a 4.98σ design counts as adequate ex post; and an amendment that the N-adequacy gate is re-evaluated on realized scatter at readout, not only on pilot scatter at launch.

---

## MAJOR-3 — GATE S is not an independent check: its outcome is a mechanical consequence of the hypothesis under test (and the registered rule has overlapping branches)

The v1 pre-check killed the accuracy-form GATE S because "a constant estimator bias displaces all three arms equally, so the gate failed for exactly the cases BAND C would call INTERNAL-DEFECT" (§0 item 3). The regression form reintroduces the identical coupling through **grid truncation**.

**[computed]** Simulation: a plain Gaussian likelihood, no estimator pathology, truncated to H_GRID_41 and calibrated to reproduce the observed truncated mean 0.6635 and σ_h:

| σ_L | truncated σ_h | slope d(mean)/d(centre) |
|---|---|---|
| 0.040 | 0.035 | 0.730 |
| 0.050 | 0.039 | **0.621** |
| 0.060 | 0.043 | 0.519 |
| 0.080 | 0.048 | **0.360** |

The observed σ_h is 0.036–0.060, and the observed ŝ = 0.368. **The sub-unit slope is exactly what truncation of a low-centred, weakly-informative likelihood produces.** No "attenuated h-response of the matched posterior" is needed. So readout §5's "the sub-unit slope is itself an unexplained diagnostic worth a registered follow-up" is not warranted, and the claim that "grid-edge truncation works *against* explaining this away" is at best unquantified and at worst backwards — truncation *is* the explanation.

Two further defects in the same gate:
- **[computed]** On `H_GRID_FULL` the same data gives `ŝ = 0.616 ± 0.290`: `|ŝ−1|/SE = 1.32` (CONTROL-VALID by the letter) **and** `|ŝ|/SE = 2.13` (CONTROL-INERT by the letter). Both registered branches are satisfied simultaneously; `csg_fleet_readout.py:133-137` resolves it by `if`-order to INERT. The registered rule is not a partition.
- The OLS `se_slope` pools residuals across arms whose sd differ 3× (0.0154 csgdm vs 0.0493 csgdp — and csgdm's small sd is itself truncation-induced: **[computed]** it rises to 0.0336 on the extended grid). Heteroscedasticity of ~10× in variance is ignored.

**Answering (e) directly:** the sub-unit slope does *not* indicate the generator responds sub-linearly to h_gen, and it does *not* independently support INTERNAL-DEFECT either. **GATE S carries no information about this verdict in either direction** and should be ruled void rather than ruled on.

---

## MAJOR-4 — Every `mean_h` statistic (incl. B_REF and the confirmatory band) is grid-truncation-dominated

**[computed]** matched channel, H_GRID_41 → H_GRID_FULL:

| arm | mean_h (g41) | mean_h (gfull) | posterior mass at h ≤ 0.62 (g41) |
|---|---|---|---|
| csgf | 0.6635 | **0.6174** | 31.0% |
| csge | 0.6633 | 0.6168 | 31.3% |
| csgdm | 0.6437 | 0.5828 | 41.6% |
| csgdp | 0.6805 | 0.6444 | 24.0% |
| **B-SEL (B_REF)** | 0.6454 (bias −0.0846) | 0.5807 (**bias −0.1493**) | — |

B-SEL's matched `map_h` sits at the 0.600 floor in **10/12 seeds**; C-SG's matched `map_h` reaches the floor in at least one seed of every arm. And `σ_h/σ_prior ≈ 0.65` → the matched "posterior" carries only ~2.3× the flat prior's information. So `B_REF = −0.0846`, the band edge `B_REF/2 = −0.0423`, and `bias₁₅ = −0.0665` are all truncated statistics, and the confirmatory band compares two differently-truncated railed numbers.

Mitigation: **the direction survives** — on the extended grid, C-SG bias −0.1126 vs `B_REF_full/2 = −0.0747` still fires INTERNAL-DEFECT. And the **primary score statistic is grid-invariant** (identical to 5 decimals on both grids) — that is a real design strength and should be said out loud.

**Resolves by:** demote the mean_h band explicitly to "convention-dependent, direction-robust", and state that the primary rests on the score alone.

---

## MAJOR-5 — Question (a): the normalization identity is assumed, never verified, and three different z-domains/quadratures are in play. The verdict is a 10% residual of two near-cancelling large derivatives

`E[∂_h ln L_matched] = 0` requires `∫B_num(x;h) dx = β_Ḡ_φ(h)` **exactly, for all h**. This is required, is nowhere verified, and the two objects are computed by different machinery on different domains:

| object | domain | quadrature | completeness |
|---|---|---|---|
| `β_Ḡ_φ` (`precompute_phi_selection_integrals:2058-2066`) | `[1e-6, min(z_max(h), 1.55)]` — **h-dependent upper limit** | 1500-node trapezoid on the `S̄_φ` grid | sky-marginal `f̄` |
| `B_num` (`bayesian_statistics.py:5129-5170`) | `[z(d̂−4σ;h), min(z(d̂+4σ;h), 1.55)]` — **h-dependent, per-event** | 50-node `fixed_quad` Gauss–Legendre; `S̄_φ` via `np.interp` with **endpoint clamping** outside `[1e-6, z_max(h)]` | per-pixel `f_k` |
| C-SG generator (`selfgen_control.py:564`) | `[POPULATION_Z_MIN, POPULATION_Z_MAX] = [1e-6, 1.5]` — **h-independent** | 4001-node grid | per-pixel `f_k` |

**[computed]** decomposition of the verdict at h = 0.73 (csgf, 15 seeds):

```
E[∂_h ln B_num]      = −1.22203
∂_h ln β_Ḡ_φ         = −1.10471
matched score        = −0.11732   ← the verdict
∂_h ln D̃_φ           = −1.23729   → pure score = +0.01526
```

The verdict is a **~10% residual between two ≈−1.15 numbers** produced by different quadratures of a nominally identical integral. Nothing in the campaign establishes that these two implementations agree to better than 10% in their h-derivative. Additional evidence that the residual tracks the integration domain: **[computed]** the per-event score is strongly ordered in `B_num` magnitude — Q1 +0.982, Q2 +0.015, Q3 −0.508, Q4 −0.958 — i.e. it is a function of where the event sits in z/d̂, not a uniform multiplicative misnormalization.

**On the h-only-ness of α and D̃ (your (a) sub-question):** treating them as per-event h-only constants is *legitimate* and is verified — GATE T measures max relative spread across events at ≤ 2e-6, and I confirmed **[computed]** max 5.5e-7 on B-SEL. The 7-sf storage is also harmless for the score: amplified over Δh = 0.01 it contributes ≲5e-5, ~400× below the effect. That part of the derivation is clean.

**What I checked and cleared, in the campaign's favour:** the `f_k`/`f̄` pairing is exact — `f_bar(z,h) = (1/N_pix) Σ_k f_k(z,h)` by construction (`pixel_completeness.py:275,287`), so `(1/N_pix)Σ_k(1−f_k) ≡ 1−f̄`. Combined with the accept-once `S_4D` design (isotropic, sky-blind), the generator's *accepted* pixel marginal is ∝ `β^{(k)}(h)`, and the pixel-measure part of the identity closes. My initial suspicion that the per-pixel/sky-marginal split was the mechanism is **refuted**. (Worth one confirmation that `_draw_dark_hosts_pixelated` uses the same `npix` and a uniform pixel base measure.)

**Consequence for decision #4:** the designated next step — "independent audit of `S̄_φ`" — likely targets the **wrong object**. `S̄_φ` appears in *both* legs and largely cancels in the residual. The audit that would actually discriminate is the **numerator/normalizer domain-and-quadrature pairing**: re-evaluate `B_num` and `β_Ḡ_φ` on a common z-grid with a common rule, widen the ±4σ window to ±10σ, and lift/align the 1.55 cap and the `np.interp` clamp. If `−0.117` moves materially, the "internal defect" is a numerical pairing artifact, not a defect in the estimator's mathematics. This is a zero-to-cheap-compute test.

---

## MAJOR-6 — A10's invariant list mis-scopes the leading alternative, so the stated conditionality is weaker than it reads (question f)

§7/A10 declares "the z-domain `[1e-6, HOST_DRAW_Z_MAX]`" a **shared** invariant, "INVISIBLE to C-SG". It is not shared: the generator stops at 1.5, `β_Ḡ_φ` integrates to `min(z_max(h), 1.55)` with an **h-dependent** endpoint, and `B_num` uses a per-event ±4σ window. GATE H was *explicitly instructed* to report that `z_max(h)` "shrinks at 0.68, grows at 0.78, asymmetrically" — the campaign observed the h-dependence and never connected it to the score null.

This matters because a generator–model z-domain mismatch lands on the **other branch** of §1's reading (generator–model mismatch, → row #140 refuted), not the INTERNAL-DEFECT branch. So "INTERNAL-DEFECT" is not merely conditional on six invariants; it is conditional on an un-listed seventh (domain/quadrature pairing) that is a live alternative.

**On circularity (your (f)):** C-SG shares `S̄_φ`/`S_4D`, `φ`, `f_k`, `dist()` with the estimator by construction. That is correctly declared, and it means C-SG can *only* detect an inconsistency between two pieces of the estimator — never a wrong physical model. That is not circular; it is a correctly-scoped internal-consistency test. But it does mean the verdict licenses **no statement whatsoever** about production's H₀ bias against the real universe. Row #151 item 3's parenthetical "(production dark rail ~−0.13)" should not be allowed to accrete inferential weight.

---

## MAJOR-7 — "B-SEL's generator caveats are hereby ELIMINATED" overreaches, and §4 fact 1's logic is invalid

Ledger #151 item 2 states the elimination as definite. **[computed]** C-SG matched score −0.1173 ± 0.0194 vs B-SEL's S_REF −0.1932 ± 0.0264 → difference −0.0759 ± 0.0328 = **2.3σ**. Roughly **39% of B-SEL's matched violation remains generator-attributable at 2.3σ**. "Eliminated" should be "not the sole owner; ~40% of B-SEL's excess is still generator-side".

Readout §4 fact 1 — "A generator artifact would move with the generator; this does not" — is a non-sequitur. All four arms share **one** generator design; only `h_gen` and the σ mode vary. Varying h_gen tests h_gen-invariance, not generator-invariance. The one genuine generator change in the record (B-SEL → C-SG) *did* move the score, by 2.3σ.

---

## MAJOR-8 — BAND R is vacuous, and its registered rationale is contradicted by the implementation

The prereg (v2 §6, restated in the band-formula block) declares F and E **independent** because "inserting a σ draw desynchronizes the shared `default_rng(seed)` stream". Reading `draw_csg_realization`, the σ draw happens at **Stage 4, after the accept/reject loop completes** — so `z`, `Ω`, `log10 M` and `donor_idx` are **bit-identical** between F and E at the same seed. Only σ and ε differ.

**[computed]** `corr(mean_h_F, mean_h_E) = 0.9975`; paired diff sd = 0.00288 → correct 3σ paired band **0.00223**. The registered threshold 0.02955 is **13.3× too wide**. "BAND R: CONSISTENT (0.0002 ≪ 0.0296)" is therefore an uninformative pass, and the arms are not a σ-robustness test of anything except the noise realization — they share their entire population draw.

---

## MAJOR-9 — No registered falsifier for the branch that fired

§8/A14 registers a falsifier **only** for ESTIMATOR-SELF-CONSISTENT ("re-run one C-SG-F seed with B-SEL's donor-row + quality-filter machinery"). INTERNAL-DEFECT — the branch that fired — carries none. A pre-registration that arms a falsifier on only one branch is not symmetric, and the branch that fired is the one now being promoted to a banked claim. The MAJOR-5 domain-pairing test is the natural retrofit.

---

## MINOR-1 — GATE V amendment (question d): procedurally clean, but the amended gate has ~zero discriminating power

Verification, in the campaign's favour: **[computed]** I reproduce the v2 thresholds false-failing exactly **5/12** banked B-SEL matched posteriors (seeds 900101–900104, 900110) and the amended thresholds failing **0/12**. `git diff dae957d6 3b43732a -- csg_pilot_bands.py` touches **only** the gate_v evaluation — no band, statistic, or reference moved, exactly as the prereg block asserts. The v2 verdicts remain in every banked JSON. Chronology (03:18 bands → 05:24 amendment → 05:26 fleet scorer → fleet) is consistent.

The problems are substantive rather than procedural:
- **The amended gate is nearly inert.** σ_prior = 0.07506, so the σ prong cuts at 0.0676. Maximum observed ratio: 0.812 (B-SEL 900110), 0.80 (pilot). Minimum observed span: 2.01 nats vs a 1-nat cut. **No real posterior is anywhere near either prong.** Only a literally flat posterior (span 0, ratio 1.0) fails. "GATE V (amended, all 46) → 46/46 PASS" in readout §6 is therefore not evidence of posterior quality and should not be scored as a passed check.
- **The contamination concern is real but not load-bearing.** The reference set is the 12 B-SEL matched posteriors — the same channel whose alleged defect broadens and rails them. Calibrating an *informativeness* gate on defect-broadened posteriors is not independent in the relevant sense. But since the amendment reduces GATE V to a flat-null detector, the contamination does not change any verdict. My reading: the amendment is legitimate; the *presentation* of "46/46 PASS" as a scorecard line is not.

---

## MINOR-2 — Pilot seeds inside the F-15 (question c): technically clean, but it is the wrong worry

`csg_fleet_readout.py:21-23` defends this correctly: under normality, the sample sd of 4 observations is independent of the mean of a 15-sample containing them. I accept the defense. The distribution is heavy-tailed (**[computed]** per-event score skew: median −0.378 vs mean −0.117; csgf seed 910113 an outlier at +0.083, mirroring B-SEL's 900110 at +0.007), so exact independence is not guaranteed — but the real problem is MAJOR-2 (the pilot sd was 1.56× low), not the reuse.

## MINOR-3 — The `S_REF/2` edge is a convention, not a decision rule (question c)

No registered rationale for "half". It has no error-rate interpretation, and `S_REF = −0.1932` is itself derived from B-SEL's railed matched posteriors — the very arm whose bias motivated the exercise. It is defensible as a *materiality* threshold (that is how O2/O3's bands were correctly framed, with an explicit A15 no-statistical-band statement); it is not defensible as the basis for a categorical branch label reported without its 1.07σ margin. The same critique applies to `B_REF/2`.

## MINOR-4 — GATE D is too loose to exclude the alternative, and structurally blind to it

Band = `D_crit(5%, n=200) = 0.0960` — a KS gap of that size at n = 200 tolerates a substantial density mismatch. No operating characteristic was published mapping "CDF gap of X" to "score of Y", which is exactly the A15 discipline the campaign otherwise applied well. And `gate_d` compares the draw against the model density **on `[POPULATION_Z_MIN, POPULATION_Z_MAX] = [1e-6, 1.5]` only** (`selfgen_control.py:1256`) — the generator's own domain — so it cannot detect the MAJOR-5/6 domain mismatch by construction.

## MINOR-5 — GATE H / Q / D outputs are not banked

The readout states "gates H, Q (43% → 0.0%), D all passed pre-run", but I find no artifact in the repo recording their values (only prose in the readout and prereg). Every other decisive number this campaign produced is recomputable from banked JSON/CSV; these are not. That is an audit gap in an otherwise exemplary provenance chain.

---

## NOTES — concerns I tested and cleared (record these; they are load-bearing in the campaign's favour)

- **N-1 (question g, physics_floor).** **[computed]** Zero `B_num == 0` cells and zero `combined_no_bh == 0` cells across **all 46 C-SG seeds and all 12 B-SEL seeds**; zero events skipped by the score's positivity mask (0 of 3000 and 0 of 2152). `physics_floor` zero-handling is **empirically inert** for this measurement, and the open "`B_num = 0` cause" (row #147 item 6) does not touch it. Worth stating explicitly — the matched channel is the one most exposed to it, and it turned out not to bite.
- **N-2 (question g, central difference).** **[computed]** S̄₁₅ = −0.11732 (±0.005 nodes) → −0.11712 (±0.02 nodes). Discretization error ≲ 2e-4, ~100× below the effect. Clean.
- **N-3 (question b, the O2 subtraction).** **No leak.** **[computed]** The assembly identity `combined = (α/r_Malm·L_cat + B_num)/D̃` reproduces the banked column at max rel err **5.5e-7** (consistent with 7-sf storage of three columns), and pure-by-subtraction ≡ `B_num/D̃` to **3.5e-5 relative** with **zero** negative-clipped cells. The `r_Malm` placement is verified against data, not assumed; `B_num_phi ≡ B_num` under the "derived" mode. O2's mechanics are sound.
- **N-4 (provenance).** **[computed]** `S_REF = −0.1932` (seed-sd 0.0914, SEM 0.0264) and `B_REF = −0.0846` both reproduce exactly from raw diagnostics. The 12/12 sign consistency and the 73% impostor share are as reported.
- **N-5.** Sky observation is noiseless (true `phiS`/`qS` written as measured) while the model assumes the donor's sky covariance. Immaterial to the matched channel — `B_num` depends on the sky only through the event's pixel index, which is the true pixel — but it does bias the reported-only full channel's impostor cone. Worth disclosing in §5's inventory.
- **N-6.** Zero analysis-stage attrition: all 46 seeds produced exactly 200 scored events, so the estimator's SNR/σ-ratio/Fisher-PD cuts removed nothing. The accept-once `S_4D` really is the only selection — a genuine design success (the cross-covariance rescale earned its keep).

---

## Overall assessment

**Solid — bankable as stated:**
1. **The matched-channel score at h_gen is non-zero at 6.05σ** (realized SEM), and it is robust to grid choice (bit-identical on H_GRID_41 and H_GRID_FULL), to the difference step (2e-4), to leave-one-seed-out, to zero-handling (inert), and to storage precision (400× margin). Stated precisely: **the implemented `B_num` and the implemented `β_Ḡ_φ` are not a matched numerator/normalizer pair — their h-derivatives differ by ~10% (−1.222 vs −1.105).** That is a real, reproducible, previously-unknown finding.
2. O2's decomposition, its identity gates, and the independent recomputation.
3. The pilot-STOP → diagnose-on-independent-reference → amend-with-published-OCs → launch chronology. This was done well and the retrospective entry is honest about the porting slip.
4. Score-zero as the primary statistic (over `mean_h`) was the right call and is vindicated: it is the only statistic in the campaign immune to the rail.

**Conditional — should not be ruled without the discriminating test:**
5. The **INTERNAL-DEFECT label**: 1.07σ past its edge, bootstrap P = 0.133 of landing in MIXED, and with a named, un-excluded alternative (numerator/normalizer domain + quadrature pairing; generator z-domain 1.5 vs normalizer `min(z_max(h),1.55)`) that lands on the *opposite* branch of §1's reading. The MAJOR-5 test is cheap and decisive; ruling before running it is premature.
6. The A10 conditionality as written (six invariants) — needs a seventh.

**Overclaimed — should be withdrawn or re-worded:**
7. "The full channel lands on the campaign's headline number in every arm" / "the −0.11 reconstructed from first principles" (FATAL-1 + FATAL-2: wrong reference point, and a railed posterior).
8. "A generator artifact would move with the generator; this does not" (MAJOR-7: invalid logic; the one real generator change moved it 2.3σ).
9. "B-SEL's residual generator caveats are hereby ELIMINATED" (MAJOR-7: ~39% still generator-attributable at 2.3σ).
10. "The sub-unit slope is an unexplained diagnostic worth a registered follow-up" (MAJOR-3: fully explained by truncation; my simulation brackets 0.368 exactly).
11. "BAND R: CONSISTENT" and "GATE V 46/46 PASS" as scorecard evidence (MAJOR-8, MINOR-1: both vacuous by construction).

**Recommendations against the six [RULE]s:**
- **#1 (ratify BAND C):** ratify only the 6σ non-zero-score finding, phrased as the numerator/normalizer mismatch. Hold the INTERNAL-DEFECT *label* pending MAJOR-5.
- **#2 (GATE S):** rule it **void**, not INERT-or-attenuated. It cannot discriminate (MAJOR-3), its branches overlap, and its SE is heteroscedasticity-blind. Do not open a follow-up thread on the sub-unit slope.
- **#3 (re-grade #137/#140/#144):** supported by O2/O3, which are the cleanest work of the night. But re-grade against corrected δ-arm biases (FATAL-1).
- **#4 (fix fork / `S̄_φ` audit):** **re-target**. `S̄_φ` cancels between the legs; audit the domain-and-quadrature pairing of `B_num` vs `β_Ḡ_φ` (±4σ window, the 1.55 cap, the `np.interp` clamp at `z_max(h)`, GL-50 vs 1500-trapezoid, and the generator's 1.5 ceiling).
- **#5 (A17):** adopt — this campaign supplied two independent instances (GATE V porting *and* the N-adequacy gate that would have re-fired at readout). Consider extending it: gates re-evaluate their operating characteristics on **realized** scatter at readout, not only on pilot scatter at launch.
- **#6 (landscape/T1 un-gate):** premature until #1 resolves; the chain's second link is not yet closed.