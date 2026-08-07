# STAGE 1 READOUT — m2-residual-owner thread

**Date:** 2026-08-07 (evening synthesis)
**Thread:** Who owns M-2's matched 2D overlap residual (+0.02225 iiib / +0.02070 joint_r1 nats/event, low-h)?
**Stage:** 0–1 of `docs/RESEARCH_CYCLE.md` (intake + first free-read measurement round). FREE READS ONLY — no cluster, no new likelihood evaluations; all numbers from existing `run_20260804_postfix` / `run_20260804_frozeng` emits.
**Approved proposal:** `book/site/ch13-unowned-residual.html` Part A (author approval 2026-08-07: "lets start the research cycle, its approved").
**Governing value ruling:** scientific correctness + novel insight over bias-removal; measure, never refute by convenience.

**Inputs (all in this directory unless noted):**
- Intake claim: `CLAIM_M2_RESIDUAL_OWNER_20260807.md` (hypotheses §4, pre-stated blind expected signatures §5, timestamped 2026-08-07 19:52 CEST, before any D-1/D-2 number was seen).
- D-1 component decomposition: `d1_component_decomposition.py` → `d1_results.json`.
- D-2 extended-covariate confounding check: `d2_confounding_check.py` → `d2_results.json`.
- Adversarial verification: `adjudicate_d1_d2.py` → `adjudication_results.json` — **D-1 CONFIRMED, D-2 CONFIRMED** (independent census/matching/LMDI/covariate implementations, fresh RNG 424242; headline numbers bitwise; six discrepancies filed, all interpretive or minor).
- Established record (committed): `../crossterm_instrument/m2_results.json`, `m2_adjudication.json` (M-2 CONFIRMED), `NEGLECT_TRIGGER_REGISTER.md` (Eq. (31) cross-term = EXCLUDED owner, NEGLECT-WITH-NUMBER ×4, min margin 4.92e4×, ledger row 96).

**Adjudication rule applied throughout:** where the verifier filed a discrepancy, **verifier numbers/readings win** and are flagged inline as [VERIFIER].

---

## 0. One-line verdict

**The residual is localized (completion leg, T_Lcomp) but not mechanism-owned: it is confounding-absorbable.** H-c survives as the leading account in modified form (component-coherent, absorbed by a collinear d_L-geometry + ball-density covariate bundle whose family split matching cannot decide); H-b and H-d are refuted as owners (their named carriers move the *wrong way*); H-a survives only as a carrier statement, not an ownership claim; H-e is disfavored but its pre-declared decisive test was not run (open loophole, free to close). No component supports a shared-galaxy/catalogue mechanism; nothing reopens the NEGLECT-registered Eq. (31) cross-term.

---

## 1. Verified evidence base (numbers used for scoring)

All matched reads use the exact M-2 machinery (census 1620/279/385 asserted; identical deterministic 1-NN pairs; M-2 headline totals reproduced to <1e-12; verifier reproduced the baseline residual **bitwise**: +0.022252643015992925 iiib / +0.020697491999731973 joint_r1).

**D-1 (exact LMDI decomposition of the matched 2D chord, closes to machine precision per verifier):**

| Term (iiib 2D / joint_r1 2D) | matched diff (nats/event) | fraction of total | signflip p | cluster-robust p |
|---|---|---|---|---|
| Total | +0.022253 / +0.020697 | 1.000 | 1.5e-4 / 1.0e-4 | 3.8e-3 / 3.15e-3 |
| **T_Lcomp** (completion-likelihood chord × completion share) | **+0.028425 / +0.027500** | **+1.28 / +1.33** | 5e-5 / 5e-5 | 6.5e-4 / 1.0e-3 |
| T_legB (full completion leg) | +0.026965 / +0.025827 | +1.21 / +1.25 | 5e-5 | 6.5e-4 / 1.0e-3 |
| T_wG (composition-weight channel) | −0.005231 / −0.006295 | −0.24 / −0.30 | 1.65e-3 / 7.5e-4 | 3.05e-2 / 1.62e-2 |
| T_cat (catalogue-leg likelihood movement) | +0.000520 / +0.001165 | +0.02 / +0.06 | 0.68 / 0.51 | NULL |
| T_gfrac | −0.001048 / −0.001092 | −0.05 / −0.05 | 6e-4 | 1.77e-2 / 1.49e-2 |

- [VERIFIER] T_cat NULL qualifier: 283/385 (iiib 2D) and 158/385 (joint_r1 2D) pairs have T_cat exactly 0 on both sides (structural zeros from empty balls); the NULL rides on 102/227 informative pairs. Read as "catalogue leg structurally absent for ~75%/40% of pairs and null on the remainder".
- 1D null-comparison channel: the **same** completion-leg pull exists in 1D (T_Lcomp +0.0217/+0.0220) but is cancelled by a ~3× larger catalogue-share weight offset (T_wG −0.0135/−0.0130). Mass-conditioning empties 90% of overlap balls in 2D (348/385 iiib), destroying the offset → 1D/2D dichotomy mechanically explained.
- Completion-leg carrier columns (L_comp, B_num_wbh, g_frac) are **bit-identical across venues** — the two venues were never independent replications of the carrier.
- d_L confounder: matched-pair d_L imbalance SMD 0.114 raw ([VERIFIER] **0.220 on the log10 scale** — the imbalance is larger than D-1 stated; direction unchanged), above M-2's own 0.10 balance bar; d_L was never an M-2 matching covariate while the C-4 census selects overlap on a 2-σ d_L window. Sensitivity re-match (+d_L as third covariate): 2D residual falls 68%/65% to +0.00706/+0.00716 (still T_Lcomp-carried, cluster p 0.013/0.011); 1D collapses entirely (p 0.96/0.78).
- [VERIFIER] transcription slip in the D-1 report text: joint_r1 1D T_w1m is −0.001253 (JSON correct); no claim depended on it.

**D-2 (extended-covariate matching, reproduction anchor bitwise; all five rungs bitwise-reproduced by the verifier from fully independent covariate construction):**

- Rung trajectory 2D (iiib): m1 +0.02225 (cl-p 4.0e-3) → m2 (+log10 ball-2D count) +0.00395 (p 0.33) → m3 +0.00281 (p 0.51) → m4 +0.00066 (ratio 0.030). joint_r1: +0.02070 → +0.00385 (p 0.17) → +0.00082 → +0.00233 (ratio 0.112). **KILLED at every rung ≥ m2**; call rests on m2/m3 where balance passes (m4 fails the 0.10 SMD bar on radius — supporting only).
- Single-covariate panel: each of the four density covariates alone kills it (−77% to −94%, all p ≥ 0.098); each single geometry covariate leaves it alive (log10_dL +0.0077, cl-p ~4.5e-3; galactic-latitude matching *strengthens* it +24–30%).
- [VERIFIER — material, verifier reading wins] **The D-2 exclusivity claim ("collapse driven EXCLUSIVELY by density covariates; refutes the sky-position reading; overlap carries no information beyond ball density") is an over-claim.** The verifier's adversarial geometry-only rung m_geo (radius, SNR, log10_dL, log10_rel_dL_err, gal-lat, ecl-lat; **no density covariates**; balance clean at max|SMD| 0.0767) **also kills the residual at both venues** (−0.0018 p 0.45 / −0.0015 p 0.50). Root cause quantified: density and geometry covariates are collinear in this stratum (matching on ball count balances log10_dL SMD 0.64 → 0.004/0.057 uninstructed, and vice versa ≤ 0.094); log10_dL is the strongest single outcome predictor among controls (Spearman +0.84 vs −0.50..−0.65 for density). **Correct statement: the residual is absorbed by the correlated density-AND-d_L-geometry covariate bundle; matching cannot decide which family owns it.**
- 1D fragile joint_r1 signal (M-2 p 0.0414) retired: dies under every richer matching (verifier: −0.0012..−0.0036, p 0.12–0.87 at every rung ≥ m2).
- Balance-before fact (both scripts agree): at their much larger radii, overlap events are ball-**sparse** relative to what their radius would predict (raw ball-2D SMD −0.213 iiib / −0.068 joint_r1).

---

## 2. Per-hypothesis scoring (against the pre-stated blind signatures, claim file §5)

Signatures were declared before any D-1/D-2 number was seen; the append-only verdict discipline of the claim file is honored (this readout is a new file; the claim file is untouched).

### H-a — completion-leg difference: **MIXED — carrier CONFIRMED, ownership NOT**
- Pre-stated Refute-by (D-1 L_comp/g_frac read shows no positive cluster-robust residual): **not triggered** — T_Lcomp is strongly positive and cluster-robust at both venues (p 6.5e-4/1.0e-3), fraction > 1; T_cat NULL; matches the predicted "present in completion leg, absent from catalogue leg" pattern, and the 1D/2D dichotomy is explained exactly as H-a anticipated (g_frac/completion structure enters the with-BH channel).
- Pre-stated persistence signature (≤ ~30% attenuation in D-2): **fails decisively** — attenuation is 82–97% at balanced rungs, and the geometry-only [VERIFIER] rung kills it outright. Predicted carrier was a with-BH-channel object (g_frac/B_num_wbh); the actual g_frac term is negative and −5% of total.
- Score: H-a is right about *where* (the completion leg carries the arithmetic of the residual) and wrong about *what* (a stratum-differential completion-leg h-slope as an intrinsic mechanism). The completion leg is a single-event, catalogue-independent integral of the event's d_L posterior; its h-response is d_L-dependent by construction, so once the stratum's d_L/density composition is balanced, the "difference" disappears.

### H-b — g_frac–clustering correlation: **REFUTED as owner**
- Pre-stated signature (g_frac-column residual present AND ≥50% attenuation under density covariates): the attenuation half holds, but the carrier half fails with the **wrong sign**: g_frac raw chord matched diff −0.000879 and T_gfrac −0.0010/−0.0011 — g_frac *opposes* the +0.021 residual at −5% of total. A −0.001 nats/event channel cannot own a +0.021 residual.
- The density-coupling *intuition* behind H-b survives in transmuted form (density covariates do absorb the residual) but re-attaches to the completion-leg/d_L bundle, not to g_frac. [LIT-4]'s tempering (must claim a coupling, not "dense sky biases per se") turned out moot: the stratum is ball-*sparse* for its radius, not dense.
- Formal note: H-b's literal Refute-by disjunction ("g_frac read null OR no attenuation") is not satisfied (g_frac is non-null and attenuation occurred), but the measured sign and magnitude refute the mechanism as stated more strongly than the pre-registered criterion contemplated. Scored on substance.

### H-c — residual selection confounding: **SUPPORTED — surviving account (in modified form)**
- Pre-stated signature, clause 2 (combined residual shrinks toward zero under richer matching, ≥50% attenuation and/or cluster p ≥ 0.0455): **holds emphatically** — 82–97% attenuation at cleanly balanced rungs, all p null; D-1's independent d_L-sensitivity re-match cuts 68%/65%; the 1D fragile signal dies entirely; dose-response is smooth, sign-stable, and reproduced bitwise by the verifier.
- Pre-stated signature, clause 1 (no single component survives alone; diffuse small same-sign shifts): **fails** — the residual is component-coherent (T_Lcomp, p 5e-5, |fraction| > 1). So H-c survives *modified*: not "diffuse noise pattern" but "a coherent completion-leg pull whose stratum contrast is manufactured by covariate imbalance the M-2 matching did not control" — specifically d_L (SMD 0.114 raw / 0.220 log [VERIFIER], induced by C-4's own 2-σ d_L-window predicate) and the collinear ball-density family.
- [LIT-2] (Inconsistency 4, σ(d_L)-dependent low-H0 pull) supplied H-c's documented mechanism candidate; the data are consistent with the milder deterministic version: the completion-leg h-chord is a function of the event's d_L posterior, and the overlap stratum simply sits at systematically different (d_L, density) than its radius+SNR-matched controls.

### H-d — composition-weight h-dependence: **REFUTED as owner**
- Pre-stated signature (residual visible in composed pieces but neither bare leg; persists under D-2): fails on both counts. The weight channel T_wG is significant but **negative** (−24%/−30% — it partially *cancels* the residual rather than carrying it), the bare completion leg carries it, and nothing persists under D-2.
- Retained explanatory value (not ownership): the weight/share structure is precisely what makes the 1D channel null — S_A (catalogue share) is ~3–13× smaller for overlap events than controls, and the ~3× larger 1D T_wG offset cancels the identical completion-leg pull there. "Through the weights" explains the *dichotomy*, not the residual. No register-trigger-(b)/(f) escalation: no stratum-differential mixture-share mechanism claim survives.

### H-e — chance/multiplicity: **DISFAVORED — but the pre-declared decisive test was not run**
- Anti-chance evidence (all pre-stated as such): coherent single-component localization at both venues (T_Lcomp p 5e-5, sign-stable); smooth monotone-ish dose-response under covariate enrichment (iiib strictly monotone; joint_r1 approximately, all-null within noise), the opposite of the erratic movement H-e predicts; verifier's independent fresh-seed reproduction of every p on the same side of every threshold.
- **Honest gap:** the pre-registered DECISIVE discriminator — component-level sign-flip over the C-4 overlap-graph (1620-sky-pair) connected components, plus jackknife-over-components and re-matching seed stability — was **not executed** in D-1, D-2, or the adjudication. Cluster-robust inference throughout used M-2's 234 control-reuse clusters, which does not address overlap-*event* correlation through the shared-sky graph (effective N of independent pairs possibly ≪ 385). H-e therefore cannot be scored CLOSED; it is disfavored on the coherence/dose-response evidence only. This is a FREE read (minutes) and is queued below.
- Operational note: H-c and H-e converge on the same thread outcome ("no mechanism owner; residual dissolves"), so this loophole does not change the stage-2 recommendation's direction — it changes how decisively the chronicle can word the dissolution.

---

## 3. Localization statement (stage-1 finding)

The M-2 matched 2D overlap residual is **carried** by the completion (out-of-catalogue) leg — T_Lcomp, the per-event completion-likelihood chord weighted by the completion share (fraction +1.28/+1.33, cluster-robust p 6.5e-4/1.0e-3) — partially offset by the composition-weight channel (T_wG, −24%/−30%), with the catalogue-leg movement NULL (mostly structurally absent [VERIFIER]). The carrier columns are bit-identical across venues, which mechanically explains the venue-stability of the residual and demotes "replicates at both venues" from independent evidence to arithmetic.

The stratum contrast itself is **confounding-absorbable**: roughly two-thirds vanishes when linear d_L enters the matching (D-1 sensitivity), and the whole of it vanishes under either the density family (D-2 m2/m3) or the geometry-only family ([VERIFIER] m_geo) — two faces of one collinear covariate bundle. The overlap stratum is farther (d_L +0.13–0.19 Gpc vs matched controls), ball-sparse for its radius, and sits differently in (σ_dL, latitude) space; matched on any sufficient slice of that bundle, its low-h chord excess reproduces in controls and the "residual" is gone. D-1's surviving +0.007 under d_L-only enrichment does not survive the richer D-2/verifier matchings and should be read as incomplete-covariate residue, not fine-structure evidence.

**What this rules out:** a shared-galaxy/catalogue-path mechanism (T_cat null; carrier catalogue-independent), g_frac as carrier (wrong sign, −5%), the composition-weight channel as carrier (wrong sign, −24/−30%), the Eq. (31) cross-term (already NEGLECT-registered; nothing here re-implicates composition beyond what the register's conditional closure anticipated — no trigger (b)/(f) event), and the sky-position/selection-artifact reading *as distinct from* the density reading (they are not separable by matching in this stratum [VERIFIER]).

---

## 4. What stage 1 cannot decide

1. **Family attribution inside the bundle.** Whether d_L-geometry or ball-density "owns" the confounding is undecidable by matching — the covariates are collinear here (SMD cross-balancing both directions; Spearman with the chord: log10_dL +0.84, density −0.50..−0.65). Any ownership statement finer than "the bundle" requires a functional read, not more matching. [VERIFIER discrepancies 1–2 govern this point.]
2. **Artifact vs. benign physics vs. weight-coupled mechanism.** D-2's own call text mandates non-auto-interpretation: "KILLED" means *not attributable to overlap status beyond the bundle*, not *artifact confirmed*. The over-matching trap is live because the killer density covariates are themselves mechanism candidates (they feed the composition weights — the live clue's only non-annihilated pathway). Stage 1 cannot distinguish (a) residual selection artifact in the matched design, (b) deterministic d_L-dependence of the completion-leg h-response (expected physics) sampled unevenly by the C-4 census, (c) a genuine density→composition-weight coupling. Note that (b) and (c) predict different *functional* shapes of chord vs (d_L, σ_dL, w_pop), which is measurable.
3. **H-e closure.** The overlap-graph component-level sign-flip (the pre-declared decisive chance test) has not been run; effective-N deflation through the 1620-sky-pair graph remains untested.
4. **Tail structure.** The m1 effect was mean-dominated (medians +0.0037/+0.0070 vs means +0.0223/+0.0207); the residual and its collapse are tail-carried. Nothing at stage 1 characterizes which events constitute the tail.
5. **External validity of the exoneration scaffolding** is inherited, not re-proven: event_idx == CRB row index (flagged M-2 assumption), and the frozeng→postfix cross-run covariate join (sanctioned by the M-4 precedent).

---

## 5. Stage-2 recommendation

**Primary recommendation: close the owner-hunt toward a confounding chronicle verdict (H-c, modified form) — a success per ch13 ("full dissolution stops the cycle at stage 1 with a confounding/chance chronicle verdict") — but only after two FREE loophole-closing reads, and with one named pre-registrable instrument on offer if the author wants the (b)-vs-(c) discrimination of §4.2 before chronicling.**

**(A) FREE MEASURE-MORE reads first (no approval barrier beyond this readout; free-read rails; minutes each):**
1. **H-e decisive test, as pre-registered:** component-level sign-flip over the C-4 overlap-graph connected components at both venues, plus jackknife-over-components and re-matching seed stability. Survival at both venues closes H-e per the prereg; failure at either venue reopens the chance account and the chronicle wording changes materially. This was promised in the claim file and must not be silently dropped.
2. **Bundle-functional sanity read:** regression/decomposition of the per-event 2D completion-leg chord against (d_L, σ_dL) over all 1588 events (not just the stratum), to check whether the chord is the smooth deterministic function of the d_L posterior that reading (b) predicts. If yes, "confounding-absorbable" upgrades to "explained: d_L-dependent completion-leg h-response × stratum d_L composition" — a stronger, physics-grounded close.

**(B) Named pre-registrable stage-2 instrument (only if free read A2 leaves structure unexplained, or the author wants the density-vs-geometry discrimination on record):**
- **Name:** completion-weight functional read ("chord-vs-density-inside-the-weights"), the discriminator D-2's call text itself names.
- **Measurement sketch (bands deliberately NOT stated — bands get locked at prereg time per discipline):** for all 1588 events per venue, decompose the combined 2D chord into the exact LMDI terms (machinery already exists in `d1_component_decomposition.py`) and fit/compare the *functional dependence* of T_Lcomp and T_wG on ball w_pop and on (d_L, σ_dL) separately, using the collinearity structure across the full event population (where density and d_L decorrelate far better than inside the C-4 stratum) to identify which family the completion-leg h-response actually tracks; report partial associations with cluster-aware uncertainty. Outcome semantics to be pre-registered: which functional family (pure-d_L, pure-density, mixed) is favored, with a pre-stated equivalence band.
- This is a free-read-class instrument on existing emits (no cluster, no new likelihood evaluations), but as a *gated stage-2 measurement* it still requires prereg.

**(iii) applies only if free read A1 fails:** if the overlap-graph sign-flip does not survive, H-e becomes the surviving account and the thread closes as chance/effective-N — chronicle accordingly.

**Governance (explicit, per mandate):**
- **Stage 2 requires author approval.** Nothing in (A) or (B) runs before the author has read this readout and approved the route.
- **Per the W-PRE-18 lesson: any stage-2 prereg must be COMMITTED before its first gated execution.** No "prereg in working tree" execution; the prereg commit precedes the first gated run.

**Stage L:** the conditional R1 forward-citation pass (citations of arXiv:2212.08694 filtered on line-of-sight/overdensity/clustering bias) is **not warranted** on this outcome — the trigger was "a density/clustering-coupled *mechanism* left standing", and the stage-1 result is dissolution into a confounding bundle with no mechanism claim. It re-arms only if instrument (B) later favors a genuine density coupling. The standing L-b trigger (two consecutive MIXED/UNDETERMINED) is not tripped: stage 1 returned a DETERMINED localization + KILLED confounding check, both CONFIRMED. Queued for stage 6 regardless of route (from the claim file, unchanged by these results): dated addendum to LITERATURE_WARNINGS row H-e (stale direction note), and a proposed new row for §4.2 Inconsistency 4 (σ(d_L) mismodeling, currently unregistered). No register edits now (rails).

---

## 6. Provenance & file-hygiene notes

- Data: `results/run_20260804_postfix/{iiib,joint_r1}/diagnostics/event_likelihoods.csv` (65108 rows each) + `prepared_cramer_rao_bounds.csv` (1590 rows); ball sets `results/run_20260804_frozeng/<venue>/posteriors_with_bh_mass/h_0_73.json`; md5s recorded in the D-1/D-2/adjudication JSONs.
- RNG provenance: D-1 seeds 20260805/99, D-2 seed 20260807, verifier seed 424242; 20000 perms everywhere; all p-values agree across seeds within MC error and never cross a decision threshold ([VERIFIER] discrepancy 6).
- Rails: free reads only; no cluster jobs; no edits to any existing claim/prereg/ledger file; ledger and book untouched (author reads this readout first).
- **Post-run lint normalization (disclosed):** to pass the repository's pre-commit quality gate, the three analysis scripts were run through `ruff format` and given comment-only `# noqa: B023` annotations *after* their JSONs were produced. Semantic equivalence was proven by AST comparison (all three scripts AST-identical to the exact versions that produced `d1_results.json`, `d2_results.json`, `adjudication_results.json`); no re-run was needed and no number changed.
- This readout is consumed by the author before any stage-2 action; the verdict lines of the claim file (§5, append-only) remain for the author to append after reading, per the claim file's own discipline.
